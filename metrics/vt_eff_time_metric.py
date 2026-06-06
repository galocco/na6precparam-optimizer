"""
Metric function for NA6P parameter optimization.
"""

import re
from pathlib import Path
import numpy as np
import awkward as ak
import uproot

nclusters_threshold = 4  # Minimum clusters for a track to be considered reconstructed

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
        n_clusters = tracks["VerTel/VerTel.mNClusters"].array(library="ak")
        n_partids = tracks["VerTel/VerTel.mParticleID"].array(library="ak")
        # cluster info
        track_ids = clusters["VerTel/VerTel.mParticleID"].array(library="ak")
        detector_ids = clusters["VerTel/VerTel.mLayer"].array(library="ak")

    n_trackable = 0
    n_reconstructed = 0
    n_reconstructed_candidates = 0
    n_fake = 0
    n_events = len(n_clusters)

    print(f"Total events: {n_events}")

    for i, (event_nclusters, event_partids, event_tracks, event_detectors) in enumerate(
        zip(n_clusters, n_partids, track_ids, detector_ids)
    ):
        if i >= n_events:
            break

        reconstructed_mask = (event_nclusters >= nclusters_threshold) & (
            event_partids >= 0
        )
        fake_mask = (event_nclusters >= nclusters_threshold) & (event_partids < 0)
        reconstructed_ids = ak.to_numpy(event_partids[reconstructed_mask])
        reconstructed_ids = np.unique(reconstructed_ids) if len(reconstructed_ids) else []
        reconstructed_truth_ids = set(map(int, reconstructed_ids))
        n_reconstructed_candidates += len(reconstructed_truth_ids)
        n_fake += int(ak.sum(fake_mask))

        # filter valid hits
        valid_mask = event_tracks >= 0
        valid_tracks = event_tracks[valid_mask]
        valid_detectors = event_detectors[valid_mask]
        if len(valid_tracks) == 0:
            continue

        # group by track id
        unique_tracks = np.unique(ak.to_numpy(valid_tracks))
        trackable_ids = set()
        for track_id in unique_tracks:
            mask = valid_tracks == track_id
            hit_detectors = valid_detectors[mask]

            # build layer mask
            layer_mask = 0
            # use only the first nclusters_threshold layers to determine trackability
            for det_id in hit_detectors:
                if det_id >= nclusters_threshold:
                    continue
                layer = int(det_id)
                layer_mask |= (1 << layer)

            # full coverage
            if layer_mask >= (1 << nclusters_threshold) - 1:
                trackable_ids.add(int(track_id))

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

    print(
        f"Reconstructed: {n_reconstructed}, Trackable: {n_trackable}, "
        f"Fake: {n_fake}, Candidates: {n_reconstructed_candidates}"
    )
    print(
        f"Efficiency: {efficiency:.4f}, Time/event: {time_seconds / n_events:.6f}s, "
        f"Time/track: {time_per_track:.6f}s"
    )

    metric = efficiency - time_per_track
    print(f"Metric: {metric:.4f}")

    metrics_file = output_path / "metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Reconstructed: {n_reconstructed}, Trackable: {n_trackable}, "
                f"Fake: {n_fake}, Candidates: {n_reconstructed_candidates}\n")
        f.write(f"Efficiency: {efficiency:.4f}, Time/event: {time_seconds / n_events:.6f}s, "
                f"Time/track: {time_per_track:.6f}s\n")

    return metric