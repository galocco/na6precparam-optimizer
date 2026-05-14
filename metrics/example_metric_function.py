#!/usr/bin/env python3
"""
Example metric functions for NA6P optimization.
Replace these with your actual metric calculations.
"""

from pathlib import Path
import numpy as np


def metric_from_root_file(output_dir: str) -> float:
    """
    Example: Extract metric from ROOT file output.
    
    You'll need to install uproot: pip install uproot awkward
    """
    try:
        import uproot
    except ImportError:
        raise ImportError("Install uproot: pip install uproot awkward")
    
    output_dir = Path(output_dir)
    
    # Example: Find ROOT file in output directory
    root_files = list(output_dir.glob("*.root"))
    if not root_files:
        raise FileNotFoundError(f"No ROOT files found in {output_dir}")
    
    root_file = root_files[0]
    
    # Open ROOT file and extract metrics
    with uproot.open(root_file) as f:
        # Example: Get a tree and calculate efficiency
        # tree = f["YourTreeName"]
        # data = tree.arrays(["variable1", "variable2"], library="np")
        
        # Calculate your metric (e.g., 1/efficiency to minimize)
        # efficiency = calculate_efficiency(data)
        # metric = 1.0 / efficiency
        
        pass
    
    # Placeholder
    return 1.0


def metric_from_text_output(output_dir: str) -> float:
    """
    Example: Parse text output to extract metric.
    """
    output_dir = Path(output_dir)
    
    # Example: Parse log file
    log_files = list(output_dir.glob("*.log"))
    if log_files:
        with open(log_files[0], 'r') as f:
            content = f.read()
            
            # Parse metric from log
            # Example: search for "Chi2/NDF = X.XXX"
            # import re
            # match = re.search(r'Chi2/NDF\s*=\s*([0-9.]+)', content)
            # if match:
            #     return float(match.group(1))
            
            pass
    
    # Placeholder
    return 1.0


def combined_metric(output_dir: str) -> float:
    """
    Example: Combine multiple metrics with weights.
    """
    # Calculate individual metrics
    # efficiency = calculate_efficiency(output_dir)
    # resolution = calculate_resolution(output_dir)
    # purity = calculate_purity(output_dir)
    
    # Combine with weights (minimize this)
    # metric = (
    #     1.0 / efficiency +        # Want high efficiency
    #     resolution +               # Want low resolution
    #     (1.0 - purity)            # Want high purity
    # )
    
    # Placeholder
    return 1.0