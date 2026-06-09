"""
Metric function for NA6P parameter optimization.
"""

import logging
import re
from pathlib import Path
import numpy as np
import awkward as ak
import uproot

nclusters_threshold = 4  # Minimum clusters for a track to be considered reconstructed
logger = logging.getLogger("metrics.example_metric_function")

def metric_function(output_dir: str) -> float:
    """
    Calculate reconstruction quality: efficiency - time_penalty
    Returns:
        Float to MAXIMIZE (higher = better)
    """
    output_path = Path(output_dir)

    # Open ROOT files
    with uproot.open(output_path / "TracksMuonSpec.root") as f_tracks, \
         uproot.open(output_path / "ClustersMuonSpec.root") as f_clusters:

        tracks = f_tracks["tracksMuonSpec"]
        clusters = f_clusters["clustersMuonSpec"]

        # reconstructed tracks
        n_clusters = tracks["MuonSpec/MuonSpec.mNClusters"].array(library="ak")
        n_reconstructed = int(ak.sum(n_clusters >= nclusters_threshold))

        # cluster info
        track_ids = clusters["MuonSpec/MuonSpec.mParticleID"].array(library="ak")
        detector_ids = clusters["MuonSpec/MuonSpec.mLayer"].array(library="ak")

    n_trackable = 0
    n_events = len(track_ids)

    logger.info("Total events: %s", n_events)

    for event_tracks, event_detectors in zip(track_ids, detector_ids):
        # filter valid hits
        valid_mask = event_tracks >= 0
        valid_tracks = event_tracks[valid_mask]
        valid_detectors = event_detectors[valid_mask]
        if len(valid_tracks) == 0:
            continue

        # group by track id
        unique_tracks = np.unique(ak.to_numpy(valid_tracks))
        for track_id in unique_tracks:
            mask = valid_tracks == track_id
            hit_detectors = valid_detectors[mask]

            # build layer mask
            layer_mask = 0
            # use only the first nclusters_threshold layers to determine trackability
            for det_id in hit_detectors:
                if det_id - 5 >= nclusters_threshold:
                    continue
                layer = int(det_id) - 5  # 0-based (5-10 → 0-5)
                layer_mask |= (1 << layer)

            # full coverage
            if layer_mask >= (1 << nclusters_threshold) - 1:
                n_trackable += 1

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

    logger.info("Reconstructed: %s, Trackable: %s", n_reconstructed, n_trackable)
    logger.info(
        "Efficiency: %.4f, Time/event: %.6fs, Time/track: %.6fs",
        efficiency,
        time_seconds / n_events if n_events > 0 else 0.0,
        time_seconds / n_reconstructed if n_reconstructed > 0 else 0.0,
    )

    metric = efficiency - (time_seconds / n_reconstructed)
    logger.info("Metric: %.4f", metric)

    return metric