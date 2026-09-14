"""
Metric function for NA6P parameter optimization.
"""

import logging
import fcntl
import hashlib
import os
import re
from pathlib import Path
import shutil
import subprocess
import numpy as np
import uproot

nclusters_threshold = 4  # Minimum clusters for a track to be considered reconstructed
logger = logging.getLogger("metrics.vt_eff_time_metric")


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


def _cluster_truth_layers(
    tree_path: Path,
    tree_name: str,
    truth_branch: str,
    detector_ids: np.ndarray,
) -> list[dict[int, set[int]]]:
    """Read cluster truth with ROOT's native streamer.

    uproot cannot deserialize the vector of NA6PMCComposedLabel in the custom
    NA6PMCTruthContainer. The small reader is compiled once and then reused.
    """
    source = Path(__file__).with_name("vt_truth_reader.cc")
    install = Path("/home/galocco/NA6PRoot/install")
    root_config = shutil.which("root-config") or "/home/galocco/alice/sw/ubuntu2404_x86-64/ROOT/v6-36-10-alice4-1/bin/root-config"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    binary = Path("/tmp") / f"na6p_vt_truth_reader_{digest}"
    lock_path = binary.with_suffix(".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not binary.exists():
            root_flags = subprocess.check_output([root_config, "--cflags", "--libs"], text=True).split()
            tbb_lib = Path("/home/galocco/alice/sw/ubuntu2404_x86-64/TBB/v2022.3.0-5/lib")
            temporary_binary = binary.with_name(binary.name + f".{os.getpid()}.tmp")
            command = [
                "g++", "-std=c++17", "-O2", str(source),
                f"-I{install / 'include'}", f"-L{install / 'lib'}", f"-L{tbb_lib}",
                f"-Wl,-rpath,{install / 'lib'}", "-lbaseLib",
                "-o", str(temporary_binary), *root_flags,
            ]
            subprocess.run(command, check=True)
            os.replace(temporary_binary, binary)
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    environment = os.environ.copy()
    root_lib = Path(root_config).resolve().parent.parent / "lib"
    environment["LD_LIBRARY_PATH"] = ":".join(
        str(path) for path in (
            install / "lib", root_lib,
            Path("/home/galocco/alice/sw/ubuntu2404_x86-64/TBB/v2022.3.0-5/lib"),
            environment.get("LD_LIBRARY_PATH", ""),
        ) if path
    )
    result = subprocess.run(
        [str(binary), str(tree_path), tree_name, truth_branch],
        check=True, capture_output=True, text=True,
        env=environment,
    )
    lines = iter(result.stdout.splitlines())
    event_layers = []
    for event_index, line in enumerate(lines):
        nclusters = int(line)
        layers_by_particle: dict[int, set[int]] = {}
        for cluster_index in range(nclusters):
            fields = next(lines).split()
            nlabels = int(fields[0])
            for raw_label in fields[1:1 + nlabels]:
                particle_id = _label_track_id(int(raw_label))
                if particle_id is not None:
                    layers_by_particle.setdefault(particle_id, set()).add(
                        int(detector_ids[event_index][cluster_index])
                    )
        event_layers.append(layers_by_particle)
    return event_layers

def metric_function(output_dir: str) -> float:
    """
    Calculate reconstruction quality: efficiency - time_penalty
    Returns:
        Float to MAXIMIZE (higher = better)
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

    logger.info("Total events: %s", n_events)

    for i, (event_nclusters, event_track_labels, event_layers) in enumerate(
        zip(n_clusters, track_labels, cluster_layers)
    ):
        if i >= n_events:
            break

        qualifying_labels = [
            raw_label for raw_label, ncl in zip(event_track_labels, event_nclusters)
            if ncl >= nclusters_threshold
        ]
        reconstructed_truth_ids = {
            track_id for raw_label in qualifying_labels
            if (track_id := _label_track_id(raw_label)) is not None
        }
        n_candidates += len(qualifying_labels)
        n_fake += sum(
            _label_track_id(raw_label) is None for raw_label in qualifying_labels
        )

        # A particle is trackable when it left hits in enough distinct layers.
        # One particle can leave multiple clusters in the same layer, so count
        # layers rather than clusters.
        trackable_ids = {
            particle_id for particle_id, layers in event_layers.items()
            if len(layers) >= nclusters_threshold
        }

        n_trackable += len(trackable_ids)
        n_reconstructed += len(reconstructed_truth_ids.intersection(trackable_ids))

    # efficiency
    efficiency = n_reconstructed / n_trackable if n_trackable > 0 else 0.0

    # read timing
    time_seconds = 0.0
    log_path = output_path / "stdout.log"
    if log_path.exists():
        log = log_path.read_text()
        match = re.search(r"CP time\s+([0-9.]+)", log)
        if match:
            time_seconds = float(match.group(1))

    time_per_track = time_seconds / n_reconstructed if n_reconstructed > 0 else 0.0

    logger.info(
        "Reconstructed: %s, Trackable: %s, Fake: %s, Candidates: %s",
        n_reconstructed,
        n_trackable,
        n_fake,
        n_candidates,
    )
    logger.info(
        "Efficiency: %.4f, Time/event: %.6fs, Time/track: %.6fs",
        efficiency,
        time_seconds / n_events if n_events > 0 else 0.0,
        time_per_track,
    )

    metric = efficiency - time_per_track
    logger.info("Metric: %.4f", metric)

    metrics_file = output_path / "metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Reconstructed: {n_reconstructed}, Trackable: {n_trackable}, "
                f"Fake: {n_fake}, Candidates: {n_candidates}\n")
        f.write(f"Efficiency: {efficiency:.4f}, Time/event: {time_seconds / n_events:.6f}s, "
                f"Time/track: {time_per_track:.6f}s\n")

    return metric
