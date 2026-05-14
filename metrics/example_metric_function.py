"""
Example of metric function for NA6P parameter optimization.

This module is loaded by optimize_reco_params.py to evaluate parameter sets.
Replace the implementation below with your actual reconstruction quality metric.

The metric_function should:
  - Take an output_dir (string) containing reconstruction results
  - Parse the output files (e.g., ROOT files, text logs)
  - Calculate a single float value representing quality
  - Return a value to MAXIMIZE

Common metrics:
  - Chi-squared from fits
  - Inverse of efficiency (1 - efficiency)
  - Resolution values
  - Composite scores (efficiency + resolution trade-offs)
"""

import uproot
from pathlib import Path


def metric_function(output_dir: str) -> float:
    """
    Calculate reconstruction quality metric.
    
    Args:
        output_dir: Path to directory containing na6prec output files
        
    Returns:
        Float value to MAXIMIZE
    """
    track_file = "TracksMuonSpec.root"
    output_path = Path(output_dir)

    track_path = output_path / track_file
    if not track_path.exists():
        raise FileNotFoundError(f"Track file not found: {track_path}")

    file = uproot.open(track_path)
    tree = file["tracksMuonSpec"]
    
    # Read mass data - this is a jagged array where each entry corresponds to one event
    # and contains an array of mass values (one per track in that event)
    mass_data = tree["MuonSpec.mMass"].array()
    
    # Count total number of tracks across ALL events
    # awkward.sum() flattens the jagged structure and counts all elements
    import awkward as ak
    n_tracks = ak.sum(ak.num(mass_data))
    
    print(f"Number of events: {len(mass_data)}")
    print(f"Total number of tracks across all events: {n_tracks}")

    # TODO: Replace with your actual metric calculation
    
    return float(n_tracks)