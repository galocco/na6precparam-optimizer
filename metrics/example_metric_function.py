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
    
    # Load data
    tracks = uproot.open(output_path / "TracksMuonSpec.root")["tracksMuonSpec"]
    clusters = uproot.open(output_path / "ClustersMuonSpec.root")["clustersMuonSpec"]
    
    # Count reconstructed tracks (with ≥5 hits like plotTracks.C)
    n_clusters = tracks["MuonSpec/MuonSpec.mNClusters"].array(library="ak")
    n_reconstructed = int(ak.sum(n_clusters >= 6))
    
    # Count trackable particles (maskHits == 31, meaning all 6 layers)
    track_ids = clusters["MuonSpec/MuonSpec.mParticleID"].array(library="ak")
    detector_ids = clusters["MuonSpec/MuonSpec.mLayer"].array(library="ak")
    
    n_trackable = 0
    n_events = len(track_ids)
    print(f"Total events: {n_events}")

    # Iterate through events
    for event_idx in range(n_events):
        event_tracks = ak.to_numpy(track_ids[event_idx])
        event_detectors = ak.to_numpy(detector_ids[event_idx])
        
        # Filter valid hits (trackID >= 0)
        valid_mask = event_tracks >= 0
        valid_tracks = event_tracks[valid_mask]
        valid_detectors = event_detectors[valid_mask]
        
        if len(valid_tracks) == 0:
            continue
        
        # Calculate layer masks for each particle
        unique_tracks = np.unique(valid_tracks)
        for track_id in unique_tracks:
            track_mask = valid_tracks == track_id
            hit_detectors = valid_detectors[track_mask]
            
            # Build layer mask: maskHits |= (1 << nLay)
            layer_mask = 0
            for det_id in hit_detectors:
                layer = int(det_id) - 5  # Convert to 0-based layer index (5-9 → 0-4)
                layer_mask |= (1 << layer)
            
            # Check if all 6 layers are hit (maskHits == 63 = 0b111111)
            if layer_mask == 63:
                n_trackable += 1
    
    # Calculate efficiency
    efficiency = n_reconstructed / n_trackable if n_trackable > 0 else 0.0
    
    # Extract execution time from log
    time_seconds = 0.0
    try:
        log = (output_path / "stdout.log").read_text()
        if match := re.search(r"CP time\s+([0-9.]+)", log):
            time_seconds = float(match.group(1))
    except FileNotFoundError:
        pass
    
    print(f"Reconstructed: {n_reconstructed}, Trackable: {n_trackable}")
    print(f"Efficiency: {efficiency:.4f}, Time per event: {time_seconds / n_events:.4f}s")
    metric = efficiency - time_seconds / n_events * 5  # Time penalty (weight=5)
    print(f"Metric: {metric:.4f}")
    
    return metric