"""Metric function for NA6P parameter optimization."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vt_eff_time_metric import (  # noqa: E402
    _array_from_first_available_branch,
    _cluster_truth_layers,
    _label_track_id,
)

import re
import uproot

nclusters_threshold = 4
logger = logging.getLogger("metrics.example_metric_function")


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
        detector_ids = clusters["MuonSpec/MuonSpec.mLayer"].array(library="np")

    cluster_layers = _cluster_truth_layers(
        output_path / "ClustersMuonSpec.root",
        "clustersMuonSpec",
        "MuonSpecMCTruth",
        detector_ids,
    )

    n_trackable = n_reconstructed = n_candidates = n_fake = 0
    n_events = len(n_clusters)
    logger.info("Total events: %s", n_events)

    for event_nclusters, event_labels, event_layers in zip(
        n_clusters, track_labels, cluster_layers
    ):
        qualifying_labels = [
            raw_label for raw_label, ncl in zip(event_labels, event_nclusters)
            if ncl >= nclusters_threshold
        ]
        reconstructed_truth_ids = {
            track_id for raw_label in qualifying_labels
            if (track_id := _label_track_id(raw_label)) is not None
        }
        n_candidates += len(qualifying_labels)
        n_fake += sum(_label_track_id(label) is None for label in qualifying_labels)
        trackable_ids = {
            particle_id for particle_id, layers in event_layers.items()
            if len(layers) >= nclusters_threshold
        }
        n_trackable += len(trackable_ids)
        n_reconstructed += len(reconstructed_truth_ids.intersection(trackable_ids))

    efficiency = n_reconstructed / n_trackable if n_trackable > 0 else 0.0
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
        "Efficiency: %.4f, Time/event: %.6fs, Time/track: %.6fs",
        efficiency, time_seconds / n_events if n_events > 0 else 0.0, time_per_track,
    )
    metric = efficiency - time_per_track
    logger.info("Metric: %.4f", metric)

    with open(output_path / "metrics.txt", "w") as metrics_file:
        metrics_file.write(
            f"Reconstructed: {n_reconstructed}, Trackable: {n_trackable}, "
            f"Fake: {n_fake}, Candidates: {n_candidates}\n"
        )
        metrics_file.write(
            f"Efficiency: {efficiency:.4f}, "
            f"Time/event: {time_seconds / n_events:.6f}s, "
            f"Time/track: {time_per_track:.6f}s\n"
        )
    return metric
