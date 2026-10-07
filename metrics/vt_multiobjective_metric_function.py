"""
Metric function for NA6P parameter optimization.
"""

import logging
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).parent))
from metric_helpers import append_kinematics, cpu_seconds, record_truth_kinematics  # noqa: E402

nclusters_threshold = 4  # Minimum clusters for a track to be considered reconstructed
OBJECTIVE_DIRECTIONS = ("maximize", "minimize")
OBJECTIVE_NAMES = ("efficiency", "CPU time per reconstructed track (s)")
logger = logging.getLogger("metrics.vt_multiobjective_metric_function")


def _reco_kinematics(params: np.ndarray) -> dict[str, float] | None:
    """Estimate fitted p, pT, and y (using a pion mass for rapidity)."""
    tx, ty, q_over_pxz = map(float, params[2:5])
    if not np.isfinite([tx, ty, q_over_pxz]).all() or q_over_pxz == 0 or abs(tx) > 1:
        return None
    pxz = 1.0 / abs(q_over_pxz)
    px, py = tx * pxz, ty * pxz
    pz = np.sqrt(1.0 - tx * tx) * pxz
    pt = float(np.hypot(px, py))
    return {
        "p": float(np.hypot(pt, pz)),
        "pt": pt,
        "y": float(np.arcsinh(pz / np.hypot(pt, 0.13957039))),
    }


def _array_from_first_available_branch(tree: uproot.TTree, candidates: list[str]) -> np.ndarray:
    available_keys = set(tree.keys())
    for branch_name in candidates:
        if branch_name in available_keys:
            return tree[branch_name].array(library="np")

    preview = ", ".join(sorted(available_keys)[:12])
    raise KeyError(
        f"None of branches {candidates} found in tree '{tree.object_path}'. "
        f"Available keys (first 12): {preview}"
    )


def _label_track_id(raw_label: int) -> int | None:
    """Decode the MC track id from NA6PMCComposedLabel's raw value."""
    raw_label = int(raw_label)
    if raw_label == 0xFFFFFFFFFFFFFFFF or raw_label == 0xFFFFFFFFFFFFFFFE:
        return None  # unset/noise
    if raw_label & (1 << 63):
        return None  # fake label
    return raw_label & ((1 << 31) - 1)


def _cluster_truth_data(
    tree_path: Path,
    tree_name: str,
    truth_branch: str,
    detector_ids: np.ndarray,
    include_labels: bool = False,
) -> tuple[list[dict[int, set[int]]], list[list[tuple[int, ...]]]]:
    """Read cluster truth with ROOT's native streamer.

    uproot cannot deserialize the vector of NA6PMCComposedLabel in the custom
    NA6PMCTruthContainer. The small reader is compiled once and then reused.
    """
    source = Path(__file__).with_name("cluster_truth_reader.cc")
    install_root = os.environ.get("NA6PROOT_ROOT")
    if not install_root:
        raise RuntimeError("NA6PROOT_ROOT must point to the NA6P installation")
    install = Path(install_root)
    digest = hashlib.sha256(source.read_bytes() + os.fsencode(install)).hexdigest()[:16]
    binary = Path("/tmp") / f"na6p_cluster_truth_reader_{digest}"
    lock_path = binary.with_suffix(".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not binary.exists():
            root_flags = subprocess.check_output(["root-config", "--cflags", "--libs"], text=True).split()
            temporary_binary = binary.with_name(binary.name + f".{os.getpid()}.tmp")
            command = [
                "g++", "-std=c++17", "-O2", str(source),
                f"-I{install / 'include'}", f"-L{install / 'lib'}", "-lbaseLib",
                "-o", str(temporary_binary), *root_flags,
            ]
            subprocess.run(command, check=True)
            os.replace(temporary_binary, binary)
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    result = subprocess.run(
        [str(binary), str(tree_path), tree_name, truth_branch],
        check=True, capture_output=True, text=True,
    )
    lines = iter(result.stdout.splitlines())
    event_layers = []
    event_labels = []
    for event_index, line in enumerate(lines):
        nclusters = int(line)
        layers_by_particle: dict[int, set[int]] = {}
        cluster_labels = []
        for cluster_index in range(nclusters):
            fields = next(lines).split()
            nlabels = int(fields[0])
            labels = tuple(int(value) for value in fields[1:1 + nlabels])
            if include_labels:
                cluster_labels.append(labels)
            for raw_label in labels:
                particle_id = _label_track_id(raw_label)
                if particle_id is not None:
                    layers_by_particle.setdefault(particle_id, set()).add(
                        int(detector_ids[event_index][cluster_index])
                    )
        event_layers.append(layers_by_particle)
        if include_labels:
            event_labels.append(cluster_labels)
    return event_layers, event_labels


def _cluster_truth_layers(
    tree_path: Path,
    tree_name: str,
    truth_branch: str,
    detector_ids: np.ndarray,
) -> list[dict[int, set[int]]]:
    """Return the particle IDs and detector layers from cluster truth."""
    return _cluster_truth_data(tree_path, tree_name, truth_branch, detector_ids)[0]


def _mc_events(tree_path: Path, n_events: int):
    """Stream PDG, mother ID, and four-momentum from ROOT's TParticle branch."""
    source = Path(__file__).with_name("mc_particle_reader.cc")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    binary = Path("/tmp") / f"na6p_mc_particle_reader_{digest}"
    lock_path = binary.with_suffix(".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not binary.exists():
            root_flags = subprocess.check_output(
                ["root-config", "--cflags", "--libs"], text=True
            ).split()
            temporary_binary = binary.with_name(binary.name + f".{os.getpid()}.tmp")
            subprocess.run(
                ["g++", "-std=c++17", "-O2", str(source), "-o",
                 str(temporary_binary), *root_flags],
                check=True,
            )
            os.replace(temporary_binary, binary)
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    process = subprocess.Popen(
        [str(binary), str(tree_path), "mckine", str(n_events)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    for event_index in range(n_events):
        line = process.stdout.readline()
        if not line:
            _, stderr = process.communicate()
            raise RuntimeError(
                f"MC particle reader stopped at event {event_index}: {stderr.strip()}"
            )
        n_particles = int(line)
        pdg = np.empty(n_particles, dtype=np.int32)
        mothers = np.empty(n_particles, dtype=np.int32)
        momentum = np.empty((n_particles, 4), dtype=np.float64)
        for particle_index in range(n_particles):
            fields = process.stdout.readline().split()
            if len(fields) != 6:
                process.kill()
                raise RuntimeError(
                    f"Malformed MC particle data at event {event_index}, "
                    f"particle {particle_index}"
                )
            pdg[particle_index], mothers[particle_index] = map(int, fields[:2])
            momentum[particle_index] = [float(value) for value in fields[2:]]
        yield pdg, mothers, momentum

    process.stdout.close()
    stderr = process.stderr.read() if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"MC particle reader failed: {stderr.strip()}")


def _save_binned_plots(output_path: Path, denominator: dict[str, list[float]],
                       numerator: dict[str, list[float]], filename: str,
                       ylabel: str) -> None:
    """Save ROOT TH1D binomial fraction plots for p, pT, and y."""
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    axes = {
        "p": ("p (GeV/c)", 0.0, 30.0),
        "pt": ("p_{T} (GeV/c)", 0.0, 3.0),
        "y": ("y", 0.0, 7.0),
    }
    prefix = Path(filename).stem
    canvas = ROOT.TCanvas(f"c_{prefix}", ylabel, 1500, 500)
    canvas.Divide(3, 1)
    histograms = []  # Keep PyROOT proxies alive until the canvas is saved.

    for pad_index, (name, (xlabel, low, high)) in enumerate(axes.items(), 1):
        pad = canvas.cd(pad_index)
        pad.SetLeftMargin(0.15)
        pad.SetRightMargin(0.04)
        pad.SetBottomMargin(0.15)
        pad.SetTopMargin(0.08)
        pad.SetGridy()

        all_hist = ROOT.TH1D(f"{prefix}_{name}_all", "", 20, low, high)
        selected_hist = ROOT.TH1D(f"{prefix}_{name}_selected", "", 20, low, high)
        fraction = ROOT.TH1D(f"{prefix}_{name}_fraction", "", 20, low, high)
        for hist in (all_hist, selected_hist, fraction):
            hist.SetDirectory(0)
            hist.Sumw2()
        for value in denominator[name]:
            if np.isfinite(value):
                all_hist.Fill(float(value))
        for value in numerator[name]:
            if np.isfinite(value):
                selected_hist.Fill(float(value))
        fraction.Divide(selected_hist, all_hist, 1.0, 1.0, "B")
        fraction.SetTitle(f";{xlabel};{ylabel}")
        fraction.SetStats(False)
        fraction.SetMinimum(0.0)
        fraction.SetMaximum(1.05)
        fraction.SetLineColor(ROOT.kBlue + 1)
        fraction.SetMarkerColor(ROOT.kBlue + 1)
        fraction.SetMarkerStyle(20)
        fraction.SetMarkerSize(0.8)
        fraction.GetXaxis().SetTitleSize(0.055)
        fraction.GetYaxis().SetTitleSize(0.055)
        fraction.GetXaxis().SetLabelSize(0.045)
        fraction.GetYaxis().SetLabelSize(0.045)
        fraction.GetYaxis().SetTitleOffset(1.25)
        fraction.Draw("E1 P")
        histograms.extend((all_hist, selected_hist, fraction))

    canvas.SaveAs(str(output_path / filename))
    canvas.Close()


def _mean_bin_efficiency(output_path: Path, denominator: dict[str, list[float]],
                         numerator: dict[str, list[float]], filename: str,
                         n_pt_bins: int = 10, n_y_bins: int = 10) -> float:
    """Save the 2D efficiency (pt vs y) and return the unweighted mean over filled bins."""
    import ROOT

    pt_range, y_range = (0.0, 3.0), (0.0, 7.0)
    hist_range = [pt_range, y_range]
    bins = [n_pt_bins, n_y_bins]
    den, _, _ = np.histogram2d(denominator["pt"], denominator["y"], bins=bins, range=hist_range)
    num, _, _ = np.histogram2d(numerator["pt"], numerator["y"], bins=bins, range=hist_range)

    filled = den > 0  # bins without trackable particles carry no information
    eff = np.divide(num, den, out=np.zeros_like(den), where=filled)

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    prefix = Path(filename).stem
    hist = ROOT.TH2D(f"{prefix}_h2", ";p_{T} (GeV/c);y;Efficiency",
                     n_pt_bins, *pt_range, n_y_bins, *y_range)
    hist.SetDirectory(0)
    for i, j in zip(*np.nonzero(filled)):
        hist.SetBinContent(int(i) + 1, int(j) + 1, float(eff[i, j]))
    hist.SetMinimum(0.0)
    hist.SetMaximum(1.0)
    canvas = ROOT.TCanvas(f"c_{prefix}", "Efficiency", 800, 600)
    canvas.SetRightMargin(0.15)
    hist.Draw("COLZ")
    canvas.SaveAs(str(output_path / filename))
    canvas.Close()

    return float(eff[filled].mean()) if filled.any() else 0.0


def metric_function(output_dir: str) -> tuple[float, float]:
    """
    Calculate the VerTel bin-averaged efficiency and time per reconstructed track.
    Returns:
        (efficiency to maximize, seconds per reconstructed track to minimize)
    """
    output_path = Path(output_dir)

    # Open ROOT files
    with uproot.open(output_path / "TracksVerTel.root") as f_tracks, \
         uproot.open(output_path / "ClustersVerTel.root") as f_clusters:

        tracks = f_tracks["tracksVerTel"]
        clusters = f_clusters["clustersVerTel"]

        # reconstructed tracks
        n_clusters = tracks["VerTel/VerTel.mNClusters"].array(library="np")
        # Track truth is now stored separately from NA6PTrack.  Do not use
        # VerTel.mPID: that is the fitted particle hypothesis, not MC truth.
        track_labels = _array_from_first_available_branch(
            tracks,
            ["VerTelMCTruth.mLabel", "VerTelMCTruth/VerTelMCTruth.mLabel"],
        )
        track_params = tracks["VerTel/VerTel.mP[5]"].array(library="np")
        # cluster info
        detector_ids = clusters["VerTel/VerTel.mLayer"].array(library="np")
        cluster_layers = _cluster_truth_layers(
            output_path / "ClustersVerTel.root",
            "clustersVerTel",
            "VerTelMCTruth",
            detector_ids,
        )

    n_trackable = 0
    n_reconstructed = 0
    n_candidates = 0
    n_fake = 0
    n_events = len(n_clusters)
    trackable_kinematics = {name: [] for name in ("p", "pt", "y")}
    reconstructed_kinematics = {name: [] for name in ("p", "pt", "y")}
    candidate_kinematics = {name: [] for name in ("p", "pt", "y")}
    fake_kinematics = {name: [] for name in ("p", "pt", "y")}

    logger.info("Total events: %s", n_events)

    with uproot.open(output_path / "MCKine.root") as f_kine:
        if (len(track_labels) != n_events or len(track_params) != n_events
                or len(cluster_layers) != n_events or len(detector_ids) != n_events
                or f_kine["mckine"].num_entries < n_events):
            raise ValueError("VerTel tracks, clusters, and MC events are misaligned")

        for event_nclusters, event_track_labels, event_params, event_layers, particle_data in zip(
            n_clusters, track_labels, track_params, cluster_layers,
            _mc_events(output_path / "MCKine.root", n_events), strict=True,
        ):
            _, _, momentum = particle_data

            qualifying = [
                (raw_label, params)
                for raw_label, ncl, params in zip(
                    event_track_labels, event_nclusters, event_params, strict=True,
                ) if ncl >= nclusters_threshold
            ]
            reconstructed_truth_ids = {
                track_id for raw_label, _ in qualifying
                if (track_id := _label_track_id(raw_label)) is not None
            }
            n_candidates += len(qualifying)
            for raw_label, params in qualifying:
                is_fake = _label_track_id(raw_label) is None
                n_fake += is_fake
                values = _reco_kinematics(params)
                if values is not None:
                    append_kinematics(candidate_kinematics, values)
                    if is_fake:
                        append_kinematics(fake_kinematics, values)

            # Count distinct detector layers, since one particle can leave
            # multiple clusters in the same layer.
            trackable_ids = {
                particle_id for particle_id, layers in event_layers.items()
                if len(layers) >= nclusters_threshold
            }
            if any(particle_id >= len(momentum) for particle_id in trackable_ids):
                raise ValueError("VerTel cluster truth refers to a missing MC particle")
            matched_ids = reconstructed_truth_ids.intersection(trackable_ids)
            record_truth_kinematics(
                trackable_ids, matched_ids, momentum,
                trackable_kinematics, reconstructed_kinematics,
            )

            n_trackable += len(trackable_ids)
            n_reconstructed += len(matched_ids)

    _save_binned_plots(output_path, trackable_kinematics, reconstructed_kinematics,
                       "particle_efficiency_vs_p_pt_y.png", "Efficiency")
    _save_binned_plots(output_path, candidate_kinematics, fake_kinematics,
                       "fake_track_fraction_vs_p_pt_y.png", "Fake track fraction")
    efficiency_global = n_reconstructed / n_trackable if n_trackable > 0 else 0.0
    efficiency = _mean_bin_efficiency(
        output_path, trackable_kinematics, reconstructed_kinematics,
        "particle_efficiency_pt_vs_y.png",
    )

    time_seconds = cpu_seconds(output_path)
    time_per_track = time_seconds / n_reconstructed if n_reconstructed > 0 else 0.0

    time_per_event = time_seconds / n_events if n_events else 0.0

    logger.info(
        "Reconstructed: %s, Trackable: %s, Fake: %s, Candidates: %s",
        n_reconstructed,
        n_trackable,
        n_fake,
        n_candidates,
    )
    logger.info(
        "Efficiency (bin-averaged): %.4f, global: %.4f, Time/event: %.6fs, Time/track: %.6fs",
        efficiency,
        efficiency_global,
        time_per_event,
        time_per_track,
    )

    metrics_file = output_path / "metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Reconstructed: {n_reconstructed}, Trackable: {n_trackable}, "
                f"Fake: {n_fake}, Candidates: {n_candidates}\n")
        f.write(f"Efficiency: {efficiency:.4f}, Efficiency (global): {efficiency_global:.4f}, "
                f"Time/event: {time_per_event:.6f}s, "
                f"Time/track: {time_per_track:.6f}s\n")

    return efficiency, time_per_track
