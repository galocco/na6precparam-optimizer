"""
Metric function for NA6P parameter optimization.

Follows the same trackability logic as plotTracks.C:
- Particle must have hits in all 5 layers (maskHits == 31)
"""

import re
from pathlib import Path
import numpy as np
import awkward as ak
import uproot


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
        n_reconstructed = int(ak.sum(n_clusters >= 6))

        # cluster info
        track_ids = clusters["MuonSpec/MuonSpec.mParticleID"].array(library="ak")
        detector_ids = clusters["MuonSpec/MuonSpec.mLayer"].array(library="ak")

    n_trackable = 0
    n_events = len(track_ids)

    print(f"Total events: {n_events}")

    # LOOP EVENTS (Awkward v2 safe usage)
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
            for det_id in hit_detectors:
                layer = int(det_id) - 5  # 0-based (5-10 → 0-5)
                layer_mask |= (1 << layer)

            # full coverage (6 layers)
            if layer_mask == 63:
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

    print(f"Reconstructed: {n_reconstructed}, Trackable: {n_trackable}")
    print(f"Efficiency: {efficiency:.4f}, Time/event: {time_seconds / n_events:.6f}s")

    metric = efficiency - (time_seconds / n_events) * 5
    print(f"Metric: {metric:.4f}")

    return metric