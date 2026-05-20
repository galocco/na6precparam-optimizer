# NA6P Parameter Optimization Framework

A flexible Python framework for optimizing NA6P reconstruction parameters using Optuna, a state-of-the-art hyperparameter optimization library.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Adding Custom Metrics](#adding-custom-metrics)
- [Adding Parameters](#adding-parameters)
- [Advanced Usage](#advanced-usage)
- [Examples](#examples)

---

## Overview

This framework automates the process of finding optimal reconstruction parameters for NA6P by:

1. Running the simulation once (`na6psim`)
2. Testing different parameter combinations via reconstruction (`na6prec`)
3. Evaluating each combination using a custom metric
4. Using Bayesian optimization (Optuna) to efficiently explore the parameter space

---

## Installation

### Prerequisites

- Python 3.7+
- NA6P software environment (with `na6psim` and `na6prec` in PATH)
- `$NA6PROOT_ROOT` environment variable set

### Install Python Dependencies

```bash
pip install optuna matplotlib numpy
```

Optional (for ROOT file analysis):
```bash
pip install uproot awkward
```
---

## Quick Start

### 1. Prepare Your Configuration Files

Ensure you have:
- `na6pLayoutini` - Your layout configuration
- `na6pRecoParam.ini` - Your reconstruction parameter template

### 2. Define Your Metric

Create a `metric.py` file (or point to your custom metric module with `--metric-module`):

```python
# metric.py
def metric_function(output_dir: str) -> float:
    """Calculate reconstruction quality metric."""
    # Read output files from output_dir
    # Calculate metric (e.g., chi2, efficiency, resolution)
    # Return value to MAXIMIZE
    return metric_value
```

### 3. Define Parameters to Optimize

Create a `param_ranges.json` file in the project root, then point the script to it with `--param-ranges`:

```json
{
    "parameters": {
        "vertexerMaxDeltaThetaTracklet": {"min": 0.3, "max": 1.0, "type": "float"},
        "vertexerKDEBandwidth": {"min": 0.1, "max": 1.0, "type": "float"},
        "vtNIterationsTrackerCA": {"min": 1, "max": 4, "type": "int"},
        "vtMaxDeltaThetaTrackletsCA": {"min": 0.02, "max": 0.1, "type": "float", "iterations_param": "vtNIterationsTrackerCA"}
    }
}
```

Parameters with `iterations_param` are expanded automatically as indexed values like `vtMaxDeltaThetaTrackletsCA[0]`, `vtMaxDeltaThetaTrackletsCA[1]`, and so on, using the sampled count from `vtNIterationsTrackerCA`.

### 4. Run Optimization

```bash
./optimize_reco_params.py \
    --layout-ini na6pLayout_Jelle.ini \
    --reco-ini na6pRecoParam.ini \
    --metric-module metrics/example_metric_function.py \
    --param-ranges params/param_ranges.json \
    --n-trials 50 \
    --work-dir ./optimization_work
```

### 5. Check Results

After completion, you'll find:
- `optimization_work/best_reco_params.ini` - Best parameters found
- `optimization_work/optimization_history.png` - Progress visualization
- `optimization_work/param_importances.png` - Parameter importance plot
- `optimization_work/trial_N/` - Individual trial outputs

---

## How It Works

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                           │
│    - Load layout and reco parameter template                │
│    - Create working directory                               │
│    - Run na6psim ONCE (simulation)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. OPTIMIZATION LOOP (repeated n_trials times)              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ a. Optuna suggests new parameter values             │  │
│  └────────────┬─────────────────────────────────────────┘  │
│               ▼                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ b. Create modified INI file with new parameters      │  │
│  └────────────┬─────────────────────────────────────────┘  │
│               ▼                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ c. Run na6prec with new parameters                   │  │
│  └────────────┬─────────────────────────────────────────┘  │
│               ▼                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ d. Calculate metric from output                      │  │
│  └────────────┬─────────────────────────────────────────┘  │
│               ▼                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ e. Optuna updates optimization strategy              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. RESULTS                                                  │
│    - Best parameters identified                             │
│    - Best INI file generated                                │
│    - Visualization plots created                            │
└─────────────────────────────────────────────────────────────┘
```

### Optuna Optimization Algorithm

Optuna uses **Tree-structured Parzen Estimator (TPE)** by default, which:
- Builds a probabilistic model of parameter → metric relationship
- Balances exploration (trying new regions) and exploitation (refining promising regions)
- Becomes more efficient as it gathers more data
- Is much faster than grid search or random search

---

## Configuration

### Command-Line Arguments

```bash
./optimize_reco_params.py [OPTIONS]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--layout-ini` | Path to NA6P layout configuration INI file | **Required** |
| `--reco-ini` | Path to reconstruction parameter template INI file | **Required** |
| `--n-trials` | Number of optimization iterations (Optuna trials) | 50 |
| `--n-events` | Number of events per simulation and reconstruction run | 10000 |
| `--work-dir` | Working directory for trial outputs and logs | `./optimization_work` |
| `--study-name` | Optuna study identifier (used for resuming studies) | `na6p_optimization` |
| `--param-ranges` | Path to JSON file defining parameter search space | `params/param_ranges.json` |
| `--metric-module` | Python module containing `metric_function()` for optimization | `metrics/example_metric_function.py` |
| `--storage` | Database URL for Optuna (SQLite, PostgreSQL, etc.) | None (in-memory; lost after run) |

### Example Configurations

**Quick test run:**
```bash
./optimize_reco_params.py \
    --layout-ini layout.ini \
    --reco-ini reco.ini \
    --n-trials 10 \
    --n-events 5000
```

**Production run with persistence:**
```bash
./optimize_reco_params.py \
    --layout-ini layout.ini \
    --reco-ini reco.ini \
    --n-trials 200 \
    --storage sqlite:///optuna_study.db
```

**Resume interrupted optimization:**
```bash
# Same command as before - Optuna will load existing trials
./optimize_reco_params.py \
    --layout-ini layout.ini \
    --reco-ini reco.ini \
    --n-trials 200 \
    --storage sqlite:///optuna_study.db \
    --study-name my_study
```

---

## Adding Custom Metrics

The metric function is the heart of the optimization. It evaluates how "good" a parameter set is.

### Metric Function Template

```python
def your_metric_function(output_dir: str) -> float:
    """
    Calculate metric from reconstruction output.
    
    Args:
        output_dir: Directory containing na6prec output files
        
    Returns:
        Float value to MAXIMIZE (higher = better)
    """
    # Your implementation here
    return metric_value
```

### Example 1: Efficiency-Based Metric

```python
def efficiency_metric(output_dir: str) -> float:
    """Maximize reconstruction efficiency."""
    import uproot
    from pathlib import Path
    
    # Find ROOT file
    root_files = list(Path(output_dir).glob("*.root"))
    if not root_files:
        return 1e10  # Penalize failed runs
    
    with uproot.open(root_files[0]) as f:
        # Read reconstructed and true tracks
        tree = f["AnalysisTree"]
        n_true = len(tree["trueTracks"].array())
        n_reco = len(tree["recoTracks"].array())
        
        if n_true == 0:
            return 1e10
        
        efficiency = n_reco / n_true
        
        # Return efficiency directly (maximize efficiency)
        return efficiency
```

### Example 2: Resolution-Based Metric

```python
def resolution_metric(output_dir: str) -> float:
    """Maximize (negative) momentum resolution."""
    import uproot
    import numpy as np
    from pathlib import Path
    
    root_files = list(Path(output_dir).glob("*.root"))
    if not root_files:
        return 1e10
    
    with uproot.open(root_files[0]) as f:
        tree = f["AnalysisTree"]
        
        # Get true and reconstructed momenta
        p_true = tree["trueMomentum"].array()
        p_reco = tree["recoMomentum"].array()
        
        # Calculate relative resolution
        delta_p = np.abs(p_reco - p_true) / p_true
        resolution = np.mean(delta_p)
        
        return -resolution  # Maximize negative resolution == minimize resolution
```

### Example 3: Combined Metric

```python
def combined_metric(output_dir: str) -> float:
    """Combine multiple objectives with weights."""
    import uproot
    import numpy as np
    from pathlib import Path
    
    root_files = list(Path(output_dir).glob("*.root"))
    if not root_files:
        return 1e10
    
    with uproot.open(root_files[0]) as f:
        tree = f["AnalysisTree"]
        
        # Calculate efficiency
        n_true = len(tree["trueTracks"].array())
        n_reco = len(tree["recoTracks"].array())
        efficiency = n_reco / n_true if n_true > 0 else 0
        
        # Calculate resolution
        p_true = tree["trueMomentum"].array()
        p_reco = tree["recoMomentum"].array()
        resolution = np.mean(np.abs(p_reco - p_true) / p_true)
        
        # Calculate purity (fraction of correct matches)
        matches = tree["isCorrectMatch"].array()
        purity = np.mean(matches) if len(matches) > 0 else 0
        
        # Combine with weights
        metric = (
            2.0 * (1.0 / efficiency) +  # Weight: 2.0
            1.0 * resolution +            # Weight: 1.0
            1.0 * (1.0 - purity)         # Weight: 1.0
        )
        
        return metric
```

### Example 4: Parsing Text Output

```python
def chi2_from_log(output_dir: str) -> float:
    """Extract chi2 from log file."""
    import re
    from pathlib import Path
    
    log_files = list(Path(output_dir).glob("*.log"))
    if not log_files:
        return 1e10
    
    with open(log_files[0], 'r') as f:
        content = f.read()
    
    # Search for chi2 value
    match = re.search(r'Average Chi2/NDF:\s*([0-9.]+)', content)
    if match:
        chi2 = float(match.group(1))
        return chi2
    
    return 1e10  # Penalty if not found
```

### Tips for Writing Metrics

1. **Always return a float** - Optuna needs a single number
2. **Higher is better** - Optuna set to maximize
3. **Handle failures gracefully** - Return a large penalty value (e.g., `1e10`) if analysis fails
4. **Normalize different scales** - If combining metrics, scale them appropriately
5. **Consider adding constraints** - Return penalty if parameters violate physics constraints

---

## Adding Parameters

### Parameter Format

Parameters are defined in `params/param_ranges.json` under the `parameters` object:

```json
{
    "parameters": {
        "parameter_name": {"min": 0.1, "max": 1.0, "type": "float"}
    }
}
```

**Types:**
- `'float'` - Continuous decimal values
- `'int'` - Integer values
- `'log'` - Log-scale floats (for parameters spanning orders of magnitude)

For compatibility with the current script, encode these ranges in `params/param_ranges.json` using the JSON object format shown above.


## Examples

### Example 1: Quick Test Run

```bash
# Test with minimal trials
./optimize_reco_params.py \
    --layout-ini test_layout.ini \
    --reco-ini test_reco.ini \
    --n-trials 5 \
    --n-events 1000 \
    --work-dir ./quick_test
```

### Example 2: Optimize Vertexer Parameters

```json
{
    "vertexerMaxDeltaThetaTracklet": {"min": 0.3, "max": 1.0, "type": "float"},
    "vertexerMaxDeltaPhiTracklet": {"min": 0.01, "max": 0.1, "type": "float"},
    "vertexerMaxDeltaTanLamInOut": {"min": 0.5, "max": 2.0, "type": "float"},
    "vertexerMaxDCAxy": {"min": 0.1, "max": 0.5, "type": "float"},
    "vertexerKDEBandwidth": {"min": 0.1, "max": 2.0, "type": "float"},
    "vertexerZWindowWidth": {"min": 0.5, "max": 3.0, "type": "float"},
    "vertexerMinCountsInPeak": {"min": 1, "max": 10, "type": "int"}
}
```

```bash
./optimize_reco_params.py \
    --layout-ini layout.ini \
    --reco-ini reco.ini \
    --n-trials 100 \
    --storage sqlite:///vertexer_opt.db \
    --study-name vertexer_optimization
```

### Example 3: Optimize Tracker Parameters

```json
{
    "parameters": {
        "vtNIterationsTrackerCA": {"min": 1, "max": 4, "type": "int"},
        "vtMaxDeltaThetaTrackletsCA": {"min": 0.02, "max": 0.08, "type": "float", "iterations_param": "vtNIterationsTrackerCA"},
        "vtMaxDeltaPhiTrackletsCA": {"min": 0.05, "max": 0.15, "type": "float", "iterations_param": "vtNIterationsTrackerCA"},
        "vtMaxDeltaTanLCellsCA": {"min": 2, "max": 8, "type": "int", "iterations_param": "vtNIterationsTrackerCA"},
        "vtMaxDeltaPhiCellsCA": {"min": 0.2, "max": 0.6, "type": "float", "iterations_param": "vtNIterationsTrackerCA"}
    }
}
```

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Optuna Tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- [Hyperparameter Optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization)

