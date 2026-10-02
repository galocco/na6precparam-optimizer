"""Metric function for NA6P parameter optimization."""

import logging
import fcntl
import hashlib
import os
import sys
from pathlib import Path
import shutil
import subprocess

sys.path.insert(0, str(Path(__file__).parent))
from vt_eff_time_metric import (  # noqa: E402
    _array_from_first_available_branch,
    _cluster_truth_data,
    _label_track_id,
)

import re
import numpy as np
import uproot

nclusters_threshold = 4
first_muonspec_layer = 5
accepted_mother_pdgs = {443, 223, 221, 333, 23}  # J/psi, omega, eta, phi, Z
logger = logging.getLogger("metrics.example_metric_function")


def _truth_key(raw_label: int) -> int | None:
    """Identify a particle including its event and source, ignoring the fake bit."""
    raw_label = int(raw_label)
    if raw_label in (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFE):
        return None
    return raw_label & ((1 << 58) - 1)


def _track_has_wrong_hit(
    raw_label: int,
    cluster_indices: np.ndarray,
    n_clusters: int,
    cluster_labels: list[tuple[int, ...]],
    delta_by_mother: dict[int, set[int]],
) -> bool:
    main_key = _truth_key(raw_label)
    if main_key is None:
        return True

    main_id = main_key & ((1 << 31) - 1)
    origin = main_key & ~((1 << 31) - 1)
    # A hit from the particle's own delta electron is valid. A delta electron
    # produced by another particle has a different mother and remains invalid.
    allowed = {main_key}
    allowed.update(origin | delta_id for delta_id in delta_by_mother.get(main_id, ()))

    associated = [int(index) for index in cluster_indices if index >= 0]
    if len(associated) != n_clusters:
        return True
    for index in associated:
        if index >= len(cluster_labels):
            return True
        labels = cluster_labels[index]
        if not labels or any(_truth_key(label) not in allowed for label in labels):
            return True
    return False


def _has_required_layers(
    cluster_indices: np.ndarray,
    n_clusters: int,
    detector_ids: np.ndarray,
    required_layers: set[int],
) -> bool:
    """Require a track cluster in each of the first MuonSpec layers."""
    associated = [int(index) for index in cluster_indices if index >= 0]
    if len(associated) != n_clusters or any(index >= len(detector_ids) for index in associated):
        return False
    layers = {int(detector_ids[index]) for index in associated}
    return required_layers.issubset(layers)


def _mc_events(tree_path: Path, n_events: int):
    """Stream PDG, mother ID, and four-momentum from ROOT's TParticle branch."""
    source = Path(__file__).with_name("mc_particle_reader.cc")
    root_config = shutil.which("root-config") or (
        "/home/galocco/alice/sw/ubuntu2404_x86-64/ROOT/"
        "v6-36-10-alice4-1/bin/root-config"
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    binary = Path("/tmp") / f"na6p_mc_particle_reader_{digest}"
    lock_path = binary.with_suffix(".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not binary.exists():
            root_flags = subprocess.check_output(
                [root_config, "--cflags", "--libs"], text=True
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


def _reco_kinematics(params: np.ndarray) -> dict[str, float] | None:
    """Compute a MuonSpec track's p, pT, and y from its fitted parameters."""
    tx, ty, q_over_pxz = map(float, params[2:5])
    if not np.isfinite([tx, ty, q_over_pxz]).all() or q_over_pxz == 0 or abs(tx) > 1:
        return None
    pxz = 1.0 / abs(q_over_pxz)  # muon charge has unit magnitude
    px, py = tx * pxz, ty * pxz
    pz = np.sqrt(1.0 - tx * tx) * pxz
    pt = float(np.hypot(px, py))
    return {
        "p": float(np.hypot(pt, pz)),
        "pt": pt,
        "y": float(np.arcsinh(pz / np.hypot(pt, 0.1056583755))),
    }


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
        "y": ("y", 1.0, 7.0),
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

    pt_range, y_range = (0.0, 3.0), (1.0, 7.0)
    hist_range = [pt_range, y_range]
    bins = [n_pt_bins, n_y_bins]
    den, _, _ = np.histogram2d(denominator["pt"], denominator["y"], bins=bins, range=hist_range)
    num, _, _ = np.histogram2d(numerator["pt"], numerator["y"], bins=bins, range=hist_range)

    filled = den > 0  # bins without trackable muons carry no information
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


def metric_function(output_dir: str) -> float:
    """Calculate MuonSpec efficiency minus the time penalty."""
    output_path = Path(output_dir)

    with uproot.open(output_path / "TracksMuonSpec.root") as f_tracks, \
         uproot.open(output_path / "ClustersMuonSpec.root") as f_clusters:
        tracks = f_tracks["tracksMuonSpec"]
        clusters = f_clusters["clustersMuonSpec"]
        n_clusters = tracks["MuonSpec/MuonSpec.mNClusters"].array(library="np")
        track_labels = _array_from_first_available_branch(
            tracks,
            ["MuonSpecMCTruth.mLabel", "MuonSpecMCTruth/MuonSpecMCTruth.mLabel"],
        )
        track_cluster_indices = tracks[
            "MuonSpec/MuonSpec.mClusterIndices[16]"
        ].array(library="np")
        track_params = tracks["MuonSpec/MuonSpec.mP[5]"].array(library="np")
        detector_ids = clusters["MuonSpec/MuonSpec.mLayer"].array(library="np")

    cluster_layers, cluster_labels = _cluster_truth_data(
        output_path / "ClustersMuonSpec.root",
        "clustersMuonSpec",
        "MuonSpecMCTruth",
        detector_ids,
        include_labels=True,
    )

    required_layers = set(range(
        first_muonspec_layer, first_muonspec_layer + nclusters_threshold
    ))
    n_trackable = n_reconstructed = n_candidates = n_fake = 0
    trackable_kinematics = {name: [] for name in ("p", "pt", "y")}
    reconstructed_kinematics = {name: [] for name in ("p", "pt", "y")}
    candidate_kinematics = {name: [] for name in ("p", "pt", "y")}
    fake_kinematics = {name: [] for name in ("p", "pt", "y")}
    n_events = len(n_clusters)
    logger.info("Total events: %s", n_events)

    with uproot.open(output_path / "MCKine.root") as f_kine:
        kine = f_kine["mckine"]
        if not (n_events == len(track_labels) == len(track_cluster_indices) == len(track_params)
                == len(cluster_layers) == len(cluster_labels) == len(detector_ids)) or kine.num_entries < n_events:
            raise ValueError("MuonSpec tracks, clusters, and MC events are misaligned")

        for event_nclusters, event_labels, event_indices, event_params, event_layers, event_cluster_labels, event_detector_ids, particle_data in zip(
            n_clusters, track_labels, track_cluster_indices, track_params, cluster_layers,
            cluster_labels, detector_ids,
            _mc_events(output_path / "MCKine.root", n_events),
            strict=True,
        ):
            qualifying = [
                (raw_label, indices, int(ncl), params)
                for raw_label, indices, ncl, params in zip(
                    event_labels, event_indices, event_nclusters, event_params, strict=True,
                )
                if ncl >= nclusters_threshold
                and _has_required_layers(indices, int(ncl), event_detector_ids, required_layers)
            ]
            main_ids = {
                key & ((1 << 31) - 1)
                for raw_label, _, _, _ in qualifying
                if (key := _truth_key(raw_label)) is not None
            }
            delta_by_mother: dict[int, set[int]] = {}
            pdg, mothers, momentum = particle_data
            eligible_muon_ids = {
                particle_id
                for particle_id, (particle_pdg, mother_id) in enumerate(zip(pdg, mothers))
                if abs(int(particle_pdg)) == 13
                and (mother_id < 0 or (
                    mother_id < len(pdg)
                    and abs(int(pdg[mother_id])) in accepted_mother_pdgs
                ))
            }
            if main_ids and len(pdg):
                delta_ids = np.flatnonzero(
                    (np.abs(pdg) == 11) & np.isin(mothers, list(main_ids))
                )
                for delta_id in delta_ids:
                    delta_by_mother.setdefault(int(mothers[delta_id]), set()).add(int(delta_id))

            reconstructed_truth_ids = set()
            for raw_label, indices, ncl, params in qualifying:
                n_candidates += 1
                is_fake = _track_has_wrong_hit(
                    raw_label, indices, ncl, event_cluster_labels, delta_by_mother,
                )
                values = _reco_kinematics(params)
                if values is not None:
                    for name, value in values.items():
                        candidate_kinematics[name].append(value)
                        if is_fake:
                            fake_kinematics[name].append(value)
                if is_fake:
                    n_fake += 1
                else:
                    reconstructed_truth_ids.add(
                        _label_track_id(int(raw_label) & ~(1 << 63))
                    )

            trackable_ids = {
                particle_id for particle_id, layers in event_layers.items()
                if particle_id in eligible_muon_ids and required_layers.issubset(layers)
            }
            matched_ids = reconstructed_truth_ids.intersection(trackable_ids)
            for particle_id in trackable_ids:
                px, py, pz, energy = momentum[particle_id]
                pt = float(np.hypot(px, py))
                values = {
                    "p": float(np.hypot(pt, pz)),
                    "pt": pt,
                    "y": float(np.arctanh(pz / energy)) if energy > abs(pz) else float("nan"),
                }
                if not all(np.isfinite(value) for value in values.values()):
                    continue
                for name, value in values.items():
                    trackable_kinematics[name].append(value)
                    if particle_id in matched_ids:
                        reconstructed_kinematics[name].append(value)
            n_trackable += len(trackable_ids)
            n_reconstructed += len(matched_ids)

    _save_binned_plots(output_path, trackable_kinematics, reconstructed_kinematics,
                       "muon_efficiency_vs_p_pt_y.png", "Efficiency")
    _save_binned_plots(output_path, candidate_kinematics, fake_kinematics,
                       "fake_track_fraction_vs_p_pt_y.png", "Fake track fraction")
    efficiency_global = n_reconstructed / n_trackable if n_trackable > 0 else 0.0
    efficiency = _mean_bin_efficiency(
        output_path, trackable_kinematics, reconstructed_kinematics,
        "muon_efficiency_pt_vs_y.png",
    )
    time_seconds = 0.0
    log_path = output_path / "stdout.log"
    if log_path.exists():
        match = re.search(r"CP time\s+([0-9.]+)", log_path.read_text())
        if match:
            time_seconds = float(match.group(1))
    time_per_track = time_seconds / n_reconstructed if n_reconstructed > 0 else 0.0

    logger.info(
        "Reconstructed: %s, Trackable: %s, Fake: %s, Candidates: %s",
        n_reconstructed, n_trackable, n_fake, n_candidates,
    )
    logger.info(
        "Efficiency (bin-averaged): %.4f, global: %.4f, Time/event: %.6fs, Time/track: %.6fs",
        efficiency, efficiency_global,
        time_seconds / n_events if n_events > 0 else 0.0, time_per_track,
    )
    metric = efficiency - time_per_track
    logger.info("Metric: %.4f", metric)

    with open(output_path / "metrics.txt", "w") as metrics_file:
        metrics_file.write(
            f"Reconstructed: {n_reconstructed}, Trackable: {n_trackable}, "
            f"Fake: {n_fake}, Candidates: {n_candidates}\n"
        )
        metrics_file.write(
            f"Efficiency: {efficiency:.4f}, Efficiency (global): {efficiency_global:.4f}, "
            f"Time/event: {time_seconds / n_events:.6f}s, "
            f"Time/track: {time_per_track:.6f}s\n"
        )
    return metric
