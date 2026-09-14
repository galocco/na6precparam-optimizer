"""Efficiency metric for matched VT/MS tracks."""

import logging
import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).parent))
from vt_eff_time_metric import (  # noqa: E402
    _array_from_first_available_branch,
    _cluster_truth_layers,
    _label_track_id,
)

nclusters_threshold_vt = 4
nclusters_threshold_ms = 4
nclusters_threshold_mt = nclusters_threshold_vt + nclusters_threshold_ms
logger = logging.getLogger("metrics.vt_ms_matching_eff_metric")


def metric_function(output_dir: str) -> float:
    """Return matching efficiency, with trackability based on truth layers."""
    output_path = Path(output_dir)

    with uproot.open(output_path / "TracksMatching.root") as f_tracks, \
         uproot.open(output_path / "ClustersVerTel.root") as f_clusters_vt, \
         uproot.open(output_path / "ClustersMuonSpec.root") as f_clusters_ms:
        tracks = f_tracks["tracksMatching"]
        clusters_vt = f_clusters_vt["clustersVerTel"]
        clusters_ms = f_clusters_ms["clustersMuonSpec"]

        n_clusters = tracks["Matching/Matching.mNClusters"].array(library="np")
        track_labels = _array_from_first_available_branch(
            tracks,
            ["MatchingMCTruth.mLabel", "MatchingMCTruth/MatchingMCTruth.mLabel"],
        )
        detector_ids_vt = clusters_vt["VerTel/VerTel.mLayer"].array(library="np")
        detector_ids_ms = clusters_ms["MuonSpec/MuonSpec.mLayer"].array(library="np")

    vt_layers = _cluster_truth_layers(
        output_path / "ClustersVerTel.root",
        "clustersVerTel",
        "VerTelMCTruth",
        detector_ids_vt,
    )
    ms_layers = _cluster_truth_layers(
        output_path / "ClustersMuonSpec.root",
        "clustersMuonSpec",
        "MuonSpecMCTruth",
        detector_ids_ms,
    )

    n_trackable = 0
    n_reconstructed = 0
    n_candidates = 0
    n_fake = 0
    n_events = len(n_clusters)

    for event_nclusters, event_labels, event_vt_layers, event_ms_layers in zip(
        n_clusters, track_labels, vt_layers, ms_layers
    ):
        qualifying_labels = [
            raw_label for raw_label, ncl in zip(event_labels, event_nclusters)
            if ncl >= nclusters_threshold_mt
        ]
        candidate_ids = {
            particle_id for raw_label in qualifying_labels
            if (particle_id := _label_track_id(raw_label)) is not None
        }
        n_candidates += len(qualifying_labels)
        n_fake += sum(_label_track_id(label) is None for label in qualifying_labels)

        vt_trackable = {
            particle_id for particle_id, layers in event_vt_layers.items()
            if len(layers) >= nclusters_threshold_vt
        }
        ms_trackable = {
            particle_id for particle_id, layers in event_ms_layers.items()
            if len(layers) >= nclusters_threshold_ms
        }
        trackable_ids = vt_trackable.intersection(ms_trackable)

        n_trackable += len(trackable_ids)
        n_reconstructed += len(candidate_ids.intersection(trackable_ids))

    efficiency = n_reconstructed / n_trackable if n_trackable > 0 else 0.0
    logger.info(
        "Reconstructed: %s, Trackable: %s, Fake: %s, Candidates: %s",
        n_reconstructed,
        n_trackable,
        n_fake,
        n_candidates,
    )
    logger.info("Efficiency: %.4f", efficiency)
    return efficiency
