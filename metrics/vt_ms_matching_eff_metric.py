"""Efficiency metric for matched VT/MS tracks."""

import logging
import sys
from pathlib import Path

import uproot

sys.path.insert(0, str(Path(__file__).parent))
from vt_multiobjective_metric_function import (  # noqa: E402
    _array_from_first_available_branch,
    _label_track_id,
)

nclusters_threshold_vt = 4
nclusters_threshold_ms = 4
nclusters_threshold_mt = nclusters_threshold_vt + nclusters_threshold_ms
OBJECTIVE_DIRECTIONS = ("maximize",)
OBJECTIVE_NAMES = ("matching efficiency",)
logger = logging.getLogger("metrics.vt_ms_matching_eff_metric")


def _track_ids(labels, cluster_counts, minimum_clusters: int) -> set[int]:
    """Return truth IDs with a reconstructed track above the cluster threshold."""
    return {
        particle_id
        for label, count in zip(labels, cluster_counts, strict=True)
        if count >= minimum_clusters
        if (particle_id := _label_track_id(label)) is not None
    }


def metric_function(output_dir: str) -> float:
    """Measure matching among particles with tracks in both detectors."""
    output_path = Path(output_dir)

    with uproot.open(output_path / "TracksMatching.root") as matching_file, \
         uproot.open(output_path / "TracksVerTel.root") as vt_file, \
         uproot.open(output_path / "TracksMuonSpec.root") as ms_file:
        matching_tracks = matching_file["tracksMatching"]
        vt_tracks = vt_file["tracksVerTel"]
        ms_tracks = ms_file["tracksMuonSpec"]

        matching_cluster_counts = matching_tracks[
            "Matching/Matching.mNClusters"
        ].array(library="np")
        matching_labels = _array_from_first_available_branch(
            matching_tracks,
            ["MatchingMCTruth.mLabel", "MatchingMCTruth/MatchingMCTruth.mLabel"],
        )
        vt_cluster_counts = vt_tracks["VerTel/VerTel.mNClusters"].array(library="np")
        vt_labels = _array_from_first_available_branch(
            vt_tracks, ["VerTelMCTruth.mLabel", "VerTelMCTruth/VerTelMCTruth.mLabel"],
        )
        ms_cluster_counts = ms_tracks["MuonSpec/MuonSpec.mNClusters"].array(library="np")
        ms_labels = _array_from_first_available_branch(
            ms_tracks, ["MuonSpecMCTruth.mLabel", "MuonSpecMCTruth/MuonSpecMCTruth.mLabel"],
        )

    n_matchable = 0
    n_reconstructed = 0
    n_candidates = 0
    n_fake = 0
    n_events = len(matching_cluster_counts)
    if not all(len(events) == n_events for events in (
        matching_labels, vt_cluster_counts, vt_labels, ms_cluster_counts, ms_labels,
    )):
        raise ValueError("Matching, VT, and MuonSpec track events are misaligned")

    for matching_counts, event_matching_labels, vt_counts, event_vt_labels, ms_counts, event_ms_labels in zip(
        matching_cluster_counts, matching_labels,
        vt_cluster_counts, vt_labels, ms_cluster_counts, ms_labels,
        strict=True,
    ):
        selected_labels = [
            raw_label for raw_label, ncl in zip(event_matching_labels, matching_counts, strict=True)
            if ncl >= nclusters_threshold_mt
        ]
        candidate_ids = _track_ids(
            event_matching_labels, matching_counts, nclusters_threshold_mt,
        )
        n_candidates += len(selected_labels)
        n_fake += sum(_label_track_id(label) is None for label in selected_labels)

        vt_track_ids = _track_ids(event_vt_labels, vt_counts, nclusters_threshold_vt)
        ms_track_ids = _track_ids(event_ms_labels, ms_counts, nclusters_threshold_ms)
        matchable_ids = vt_track_ids & ms_track_ids

        n_matchable += len(matchable_ids)
        n_reconstructed += len(candidate_ids & matchable_ids)

    efficiency = n_reconstructed / n_matchable if n_matchable > 0 else 0.0
    logger.info(
        "Matched: %s, Matchable: %s, Fake: %s, Candidates: %s",
        n_reconstructed,
        n_matchable,
        n_fake,
        n_candidates,
    )
    logger.info("Efficiency: %.4f", efficiency)
    return efficiency
