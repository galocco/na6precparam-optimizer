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

**Key Features:**
- Automatic parameter space exploration
- Persistent optimization (resume interrupted runs)
- Visualization of optimization progress
- Support for parallel trials
- Easy metric customization

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

### Download the Framework

```bash
# Clone or copy the optimization scripts
chmod +x optimize_reco_params.py
```

---

## Quick Start

### 1. Prepare Your Configuration Files

Ensure you have:
- `na6pLayout_Jelle.ini` - Your layout configuration
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

Common approaches:
- Parse ROOT files with `uproot` and calculate efficiency or resolution
- Parse reconstruction logs for chi-squared values
- Combine multiple metrics (efficiency + resolution trade-off)

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
| `--layout-ini` | Path to layout INI file | **Required** |
| `--reco-ini` | Path to reconstruction parameter template | **Required** |
| `--n-trials` | Number of optimization trials | 10 |
| `--n-events` | Number of simulation/reconstruction events | 100 |
| `--work-dir` | Working directory for trials | `./optimization_work` |
| `--study-name` | Optuna study name | `na6p_optimization` |
| `--param-ranges` | Parameter ranges config file | `params/param_ranges.json` |
| `--metric-module` | Python module defining `metric_function` | `metrics/example_metric_function.py` |
| `--storage` | Database URL for persistent storage | None (in-memory) |

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

### Example: Simple Parameters

```python
param_ranges = {
    # Vertexer parameters
    'vertexerMaxDeltaThetaTracklet': (0.3, 1.0, 'float'),
    'vertexerMaxDeltaPhiTracklet': (0.01, 0.1, 'float'),
    'vertexerMaxDCAxy': (0.1, 0.5, 'float'),
    
    # Integer parameters
    'vertexerPeakWidthBins': (1, 10, 'int'),
    'vertexerMinCountsInPeak': (1, 10, 'int'),
    
    # Log-scale parameters (e.g., bandwidth from 0.01 to 10)
    'vertexerKDEBandwidth': (0.01, 10.0, 'log'),
}
```

### Example: Array Parameters

INI files often have array parameters like `vtMaxDeltaThetaTrackletsCA[0]`:

```python
param_ranges = {
    # First iteration parameters (index 0)
    'vtMaxDeltaThetaTrackletsCA[0]': (0.02, 0.1, 'float'),
    'vtMaxDeltaPhiTrackletsCA[0]': (0.05, 0.2, 'float'),
    
    # Second iteration parameters (index 1)
    'vtMaxDeltaThetaTrackletsCA[1]': (0.05, 0.3, 'float'),
    'vtMaxDeltaPhiTrackletsCA[1]': (0.1, 0.5, 'float'),
    
    # Can optimize multiple indices
    'msMaxDeltaThetaTrackletsCA[0]': (0.03, 0.12, 'float'),
    'msMaxDeltaThetaTrackletsCA[1]': (0.05, 0.2, 'float'),
}
```

### Example: Optimizing Multiple Related Parameters

```python
param_ranges = {
    # Cell association parameters for VT (all iterations)
    'vtMaxDeltaTanLCellsCA[0]': (2, 8, 'int'),
    'vtMaxDeltaTanLCellsCA[1]': (5, 15, 'int'),
    'vtMaxDeltaTanLCellsCA[2]': (10, 30, 'int'),
    
    'vtMaxDeltaPhiCellsCA[0]': (0.2, 0.8, 'float'),
    'vtMaxDeltaPhiCellsCA[1]': (0.4, 1.0, 'float'),
    'vtMaxDeltaPhiCellsCA[2]': (0.8, 2.0, 'float'),
    
    # Chi2 cuts
    'vtMaxChi2ndfCellsCA[0]': (10, 200, 'float'),
    'vtMaxChi2ndfCellsCA[1]': (100, 1000, 'float'),
}
```

### Finding Good Parameter Ranges

**Strategy 1: Start wide, then narrow**
```python
# First optimization: wide ranges
param_ranges_v1 = {
    'parameterX': (0.01, 10.0, 'log'),  # Very wide
}

# After analyzing results, narrow down
param_ranges_v2 = {
    'parameterX': (0.5, 2.0, 'float'),  # Focused on promising region
}
```

**Strategy 2: Use current values ± margin**
```python
# If current value is 0.05
param_ranges = {
    'parameterX': (0.05 * 0.5, 0.05 * 2.0, 'float'),  # ±50%
}
```

**Strategy 3: Physics-based bounds**
```python
param_ranges = {
    'vertexerMaxDCAxy': (0.001, 1.0, 'float'),  # DCA can't be negative
    'vertexerZMin': (-50, 0, 'float'),           # Z must be negative
    'vertexerZMax': (0, 50, 'float'),            # Z must be positive
}
```

---

## Advanced Usage

### Parallel Optimization

Optuna supports parallel trials when using a database backend:

**Terminal 1:**
```bash
./optimize_reco_params.py \
    --layout-ini layout.ini \
    --reco-ini reco.ini \
    --n-trials 50 \
    --storage sqlite:///optuna.db \
    --study-name parallel_study \
    --work-dir ./work_1
```

**Terminal 2:**
```bash
./optimize_reco_params.py \
    --layout-ini layout.ini \
    --reco-ini reco.ini \
    --n-trials 50 \
    --storage sqlite:///optuna.db \
    --study-name parallel_study \
    --work-dir ./work_2
```

Both processes will coordinate through the database!

### Multi-Objective Optimization

Optimize multiple metrics simultaneously:

```python
def multi_objective(trial, param_ranges):
    # ... setup trial ...
    
    # Calculate multiple metrics
    efficiency = calculate_efficiency(trial_dir)
    resolution = calculate_resolution(trial_dir)
    
    # Return tuple of values to maximize
    return (1.0 / efficiency, resolution)

# Create multi-objective study
study = optuna.create_study(
    directions=['maximize', 'maximize'],  # Two objectives
    study_name="multi_obj"
)

study.optimize(multi_objective, n_trials=100)

# Get Pareto front
pareto_trials = study.best_trials
```

### Pruning Unpromising Trials

Save time by stopping bad trials early:

```python
def objective_with_pruning(trial, param_ranges):
    # ... setup ...
    
    # Run partial reconstruction
    partial_metric = run_partial_reconstruction(...)
    
    # Report intermediate value
    trial.report(partial_metric, step=0)
    
    # Check if trial should be pruned
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    # Continue with full reconstruction
    final_metric = run_full_reconstruction(...)
    return final_metric

# Use median pruner
study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
)
```

### Custom Sampling Strategy

Use different samplers:

```python
# Grid search (exhaustive but slow)
sampler = optuna.samplers.GridSampler({
    'param1': [0.1, 0.5, 1.0],
    'param2': [10, 50, 100]
})

# Random search (baseline)
sampler = optuna.samplers.RandomSampler()

# CMA-ES (good for continuous parameters)
sampler = optuna.samplers.CmaEsSampler()

study = optuna.create_study(sampler=sampler)
```

### Conditional Parameter Spaces

Optimize different parameters based on other choices:

```python
def conditional_objective(trial):
    # Choose reconstruction algorithm
    algo = trial.suggest_categorical('algorithm', ['KDE', 'PeakFinder'])
    
    if algo == 'KDE':
        # KDE-specific parameters
        bandwidth = trial.suggest_float('bandwidth', 0.1, 2.0)
        n_grid = trial.suggest_int('n_grid', 100, 1000)
    else:
        # PeakFinder-specific parameters
        peak_width = trial.suggest_int('peak_width', 1, 10)
        min_counts = trial.suggest_int('min_counts', 1, 20)
    
    # ... run with selected parameters ...
```

### Analyzing Optimization Results

```python
import optuna

# Load study
study = optuna.load_study(
    study_name="my_study",
    storage="sqlite:///optuna.db"
)

# Get best trial
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")

# Get all trials
df = study.trials_dataframe()
df.to_csv("all_trials.csv")

# Get parameter importances
importances = optuna.importance.get_param_importances(study)
for param, importance in importances.items():
    print(f"{param}: {importance:.4f}")

# Visualizations
import optuna.visualization as vis

# Parameter relationships
vis.plot_parallel_coordinate(study).show()
vis.plot_slice(study).show()
vis.plot_contour(study, params=['param1', 'param2']).show()

# Individual parameter effects
vis.plot_param_importances(study).show()
vis.plot_optimization_history(study).show()
```

---

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

### Example 4: Two-Stage Optimization

```bash
# Stage 1: Coarse search with wide ranges
./optimize_reco_params.py \
    --layout-ini na6pLayout_Jelle.ini \
    --reco-ini na6pRecoParam.ini \
    --n-trials 50 \
    --work-dir ./stage1

# Analyze results, identify promising region
# Edit param_ranges to narrow around best values

# Stage 2: Fine search with narrow ranges
./optimize_reco_params.py \
    --layout-ini na6pLayout_Jelle.ini \
    --reco-ini na6pRecoParam.ini \
    --n-trials 100 \
    --work-dir ./stage2
```

### Example 5: Custom Metric with Multiple Criteria

```python
def physics_metric(output_dir: str) -> float:
    """Optimize for physics performance."""
    import uproot
    import numpy as np
    from pathlib import Path
    
    root_file = list(Path(output_dir).glob("*.root"))[0]
    
    with uproot.open(root_file) as f:
        tree = f["RecoTree"]
        
        # Get arrays
        n_true = tree["nTrueTracks"].array()[0]
        n_reco = tree["nRecoTracks"].array()[0]
        chi2_values = tree["trackChi2"].array()
        mom_res = tree["momentumResolution"].array()
        
        # Calculate components
        efficiency = n_reco / n_true if n_true > 0 else 0
        avg_chi2 = np.mean(chi2_values) if len(chi2_values) > 0 else 1e10
        avg_resolution = np.mean(mom_res) if len(mom_res) > 0 else 1e10
        
        # Weighted combination
        # Want: high efficiency, low chi2, low resolution
        metric = (
            3.0 * (1.0 / max(efficiency, 0.01)) +  # High weight on efficiency
            1.0 * avg_chi2 / 100.0 +                 # Normalize chi2
            2.0 * avg_resolution                     # Medium weight on resolution
        )
        
        return metric

# Then pass to optimizer
optimizer = INIParameterOptimizer(
    ...,
    metric_function=physics_metric
)
```

---

## File Structure

After running optimization, your directory will look like:

```
project/
├── optimize_reco_params.py           # Main optimization script
├── metric_examples.py          # Example metric functions
├── README.md                   # This file
├── na6pLayout_Jelle.ini       # Your layout config
├── na6pRecoParam.ini          # Your reco param template
│
└── optimization_work/          # Working directory
    ├── best_reco_params.ini   # Best parameters found
    ├── optimization_history.png
    ├── param_importances.png
    │
    ├── trial_0/               # First trial
    │   ├── reco_params.ini
    │   └── [na6prec output files]
    │
    ├── trial_1/               # Second trial
    │   ├── reco_params.ini
    │   └── [na6prec output files]
    │
    └── ...
```

---

## Tips and Best Practices

### Starting an Optimization

1. **Start small**: Test with 5-10 trials first
2. **Check one parameter**: Optimize just one parameter to verify everything works
3. **Validate metric**: Ensure your metric function gives sensible values
4. **Use persistence**: Always use `--storage` for production runs

### During Optimization

1. **Monitor progress**: Watch the optimization history plot
2. **Check trial outputs**: Inspect `trial_N/` directories for failures
3. **Be patient**: Good optimization takes time (especially with many parameters)

### After Optimization

1. **Validate best params**: Run `na6prec` manually with best parameters
2. **Check stability**: Run optimization again - do you get similar results?
3. **Test on different data**: Verify parameters generalize to other datasets
4. **Document your findings**: Keep notes on what you tried and what worked

### Parameter Selection

1. **Don't optimize everything**: Focus on parameters that actually matter
2. **Understand dependencies**: Some parameters are correlated
3. **Use physics intuition**: Set sensible bounds based on detector geometry
4. **Check convergence**: Have you tried enough trials?

---

## Contributing

To extend this framework:

1. Add new metric functions in `metric_examples.py`
2. Add new parameter sets for different optimization scenarios
3. Add visualization functions for specific analyses
4. Share your configurations and results!

---

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Optuna Tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- [Hyperparameter Optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization)

