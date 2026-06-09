"""
Metric function for NA6P parameter optimization.
"""

import logging
import re
from pathlib import Path
import numpy as np
import awkward as ak
import uproot

nclusters_threshold_vt = 4  # Minimum clusters for a track to be considered reconstructed
nclusters_threshold_ms = 4  # Minimum clusters for a track to be considered reconstructed
nclusters_threshold_mt = nclusters_threshold_vt + nclusters_threshold_ms # Minimum clusters for a track to be considered reconstructed
logger = logging.getLogger("metrics.vt_ms_matching_eff_metric")

def metric_function(output_dir: str) -> float:
    """
    Calculate reconstruction quality: efficiency - time_penalty
    Returns:
        Float to MAXIMIZE (higher = better)
    """
    output_path = Path(output_dir)

    # Open ROOT files
    with uproot.open(output_path / "TracksMatching.root") as f_tracks_mt, \
         uproot.open(output_path / "ClustersVerTel.root") as f_clusters_vt, \
         uproot.open(output_path / "ClustersMuonSpec.root") as f_clusters_ms:

        tracks_mt = f_tracks_mt["tracksMatching;1"]
        clusters_vt = f_clusters_vt["clustersVerTel"]
        clusters_ms = f_clusters_ms["clustersMuonSpec"]
        # reconstructed tracks
        n_clusters_mt = tracks_mt["Matching/Matching.mNClusters"].array(library="ak")
        n_partids_mt = tracks_mt["Matching/Matching.mParticleID"].array(library="ak")

        # cluster info
        track_ids_vt = clusters_vt["VerTel/VerTel.mParticleID"].array(library="ak")
        detector_ids_vt = clusters_vt["VerTel/VerTel.mLayer"].array(library="ak")
        track_ids_ms = clusters_ms["MuonSpec/MuonSpec.mParticleID"].array(library="ak")
        detector_ids_ms = clusters_ms["MuonSpec/MuonSpec.mLayer"].array(library="ak")

    n_trackable = 0
    n_reconstructed = 0
    n_reconstructed_candidates = 0
    n_fake = 0
    n_events = len(n_clusters_mt)

    logger.info("Total events: %s", n_events)

    for i, (event_nclusters_mt, event_partids_mt, event_tracks_vt, event_detectors_vt, event_tracks_ms, event_detectors_ms) in enumerate(
        zip(n_clusters_mt, n_partids_mt, track_ids_vt, detector_ids_vt, track_ids_ms, detector_ids_ms)
    ):
        if i >= n_events:
            break

        reconstructed_mask = (event_nclusters_mt >= nclusters_threshold_mt)
        fake_mask = (event_nclusters_mt >= nclusters_threshold_mt) & (event_partids_mt < 0)
        reconstructed_ids = ak.to_numpy(event_partids_mt[reconstructed_mask])
        reconstructed_ids = np.unique(reconstructed_ids) if len(reconstructed_ids) else []
        reconstructed_truth_ids = set(map(int, reconstructed_ids))
        n_reconstructed_candidates += len(reconstructed_truth_ids)
        n_fake += int(ak.sum(fake_mask))

        # filter valid hits VT
        valid_mask_vt = event_tracks_vt >= 0
        valid_tracks_vt = event_tracks_vt[valid_mask_vt]
        valid_detectors_vt = event_detectors_vt[valid_mask_vt]

        # filter valid hits MS
        valid_mask_ms = event_tracks_ms >= 0
        valid_tracks_ms = event_tracks_ms[valid_mask_ms]

        if len(valid_tracks_vt) == 0:
            continue

        # count VT hits per track
        unique_tracks = np.unique(ak.to_numpy(valid_tracks_vt))
        vt_hit_counts: dict[int, int] = {}
        for track_id in unique_tracks:
            mask = valid_tracks_vt == track_id
            hit_detectors = valid_detectors_vt[mask]
            # count only the first nclusters_threshold_vt layers
            layers_hit = set()
            for det_id in hit_detectors:
                if det_id < nclusters_threshold_vt:
                    layers_hit.add(int(det_id))
            vt_hit_counts[int(track_id)] = len(layers_hit)

        # count MS hits per track
        ms_hit_counts: dict[int, int] = {}
        if len(valid_tracks_ms) > 0:
            for track_id in np.unique(ak.to_numpy(valid_tracks_ms)):
                ms_hit_counts[int(track_id)] = int(ak.sum(valid_tracks_ms == track_id))

        # a track is reconstructable if it has enough hits in BOTH detectors
        trackable_ids = set()
        for track_id, vt_count in vt_hit_counts.items():
            ms_count = ms_hit_counts.get(track_id, 0)
            if vt_count >= nclusters_threshold_vt and ms_count >= nclusters_threshold_ms:
                trackable_ids.add(track_id)

        n_trackable += len(trackable_ids)
        n_reconstructed += len(reconstructed_truth_ids)

    # efficiency
    efficiency = n_reconstructed / n_trackable if n_trackable > 0 else 0.0


    logger.info(
        "Reconstructed: %s, Trackable: %s, Fake: %s, Candidates: %s",
        n_reconstructed,
        n_trackable,
        n_fake,
        n_reconstructed_candidates,
    )

    metric = efficiency
    logger.info("Metric: %.4f", metric)

    return metric