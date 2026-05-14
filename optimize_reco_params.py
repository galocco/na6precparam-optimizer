#!/usr/bin/env python3
"""
Parameter optimization script for NA6P reconstruction using Optuna.
"""

import argparse
import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import optuna
from optuna.trial import Trial
import re


class INIParameterOptimizer:
    """Optimizer for NA6P reconstruction parameters."""
    
    def __init__(
        self,
        layout_ini: str,
        reco_ini_template: str,
        metric_function: Callable[[str], float],
        n_sim_events: int = 50000,
        n_rec_events: int = 50000,
        work_dir: str = "./optimization_work"
    ):
        """
        Initialize the optimizer.
        
        Args:
            layout_ini: Path to layout ini file
            reco_ini_template: Path to template reco parameter ini file
            metric_function: Function that takes output directory and returns metric to MINIMIZE
            n_sim_events: Number of simulation events
            n_rec_events: Number of reconstruction events
            work_dir: Working directory for optimization trials
        """
        self.layout_ini = Path(layout_ini).resolve()
        self.reco_ini_template = Path(reco_ini_template).resolve()
        self.metric_function = metric_function
        self.n_sim_events = n_sim_events
        self.n_rec_events = n_rec_events
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        # Check if simulation needs to be run
        self.sim_done = False
        
        # Verify required files exist
        if not self.layout_ini.exists():
            raise FileNotFoundError(f"Layout INI not found: {self.layout_ini}")
        if not self.reco_ini_template.exists():
            raise FileNotFoundError(f"Reco INI template not found: {self.reco_ini_template}")
    
    def run_simulation(self):
        """Run na6psim once (only needs to be done once)."""
        if self.sim_done:
            return
        
        print("Running simulation (na6psim)...")
        cmd = [
            "na6psim",
            f"-n{self.n_sim_events}",
            "-g", f"$NA6PROOT_ROOT/share/test/genDimuonBgEvent.C+(1,\"Omega\")",
            "--load-ini", str(self.layout_ini)
        ]
        
        # Execute with shell to expand environment variables
        cmd_str = " ".join(cmd)
        result = subprocess.run(
            cmd_str,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self.work_dir
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Simulation failed:\n{result.stderr}")
        
        print("Simulation completed successfully")
        self.sim_done = True
    
    def update_ini_file(self, params: Dict[str, float], output_path: Path):
        """
        Update INI file with new parameter values.
        
        Args:
            params: Dictionary of parameter_name -> value
            output_path: Where to write the modified INI file
        """
        with open(self.reco_ini_template, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            modified = False
            for param_name, param_value in params.items():
                # Match parameter lines (handle both simple and array parameters)
                pattern = rf'^({re.escape(param_name)}(?:\[\d+\])?)\s*='
                if re.match(pattern, line.strip()):
                    # Extract the parameter name including array index if present
                    match = re.match(pattern, line.strip())
                    if match:
                        param_full = match.group(1)
                        updated_lines.append(f"{param_full}={param_value}\n")
                        modified = True
                        break
            
            if not modified:
                updated_lines.append(line)
        
        with open(output_path, 'w') as f:
            f.writelines(updated_lines)
    
    def run_reconstruction(self, reco_ini: Path, trial_dir: Path) -> float:
        """
        Run reconstruction with given parameters.
        
        Args:
            reco_ini: Path to reconstruction INI file
            trial_dir: Directory for this trial's output
            
        Returns:
            Metric value (to be minimized)
        """
        print(f"Running reconstruction in {trial_dir}...")
        
        cmd = [
            "na6prec",
            f"-l{self.n_rec_events}",
            "--load-recoparam", str(reco_ini),
            "--load-ini", str(self.layout_ini),
            "--doMatching", "false",
            "--doHitsToRecPoints", "true",
            "--doTrackletVertex", "false",
            "--doVTTracking", "false"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=trial_dir
        )
        
        if result.returncode != 0:
            print(f"Reconstruction failed:\n{result.stderr}")
            raise optuna.TrialPruned()
        
        # Calculate metric
        metric = self.metric_function(str(trial_dir))
        print(f"Metric value: {metric}")
        
        return metric
    
    def objective(
        self,
        trial: Trial,
        param_ranges: Dict[str, Tuple[float, float, str]]
    ) -> float:
        """
        Objective function for Optuna.
        
        Args:
            trial: Optuna trial object
            param_ranges: Dict of param_name -> (min, max, type)
                         where type is 'float', 'int', or 'log'
        
        Returns:
            Metric value to minimize
        """
        # Create trial directory
        trial_dir = self.work_dir / f"trial_{trial.number}"
        trial_dir.mkdir(exist_ok=True)
        
        # Suggest parameters
        params = {}
        for param_name, (min_val, max_val, param_type) in param_ranges.items():
            if param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, int(min_val), int(max_val))
            elif param_type == 'log':
                params[param_name] = trial.suggest_float(param_name, min_val, max_val, log=True)
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        # Create modified INI file
        trial_reco_ini = trial_dir / "reco_params.ini"
        self.update_ini_file(params, trial_reco_ini)
        
        # Run reconstruction and get metric
        metric = self.run_reconstruction(trial_reco_ini, trial_dir)
        
        return metric
    
    def optimize(
        self,
        param_ranges: Dict[str, Tuple[float, float, str]],
        n_trials: int = 100,
        study_name: str = "na6p_optimization",
        storage: str = None
    ) -> optuna.Study:
        """
        Run the optimization.
        
        Args:
            param_ranges: Dict of param_name -> (min, max, type)
            n_trials: Number of optimization trials
            study_name: Name for the Optuna study
            storage: Optional database URL for persistent storage
        
        Returns:
            Completed Optuna study
        """
        # Run simulation once
        self.run_simulation()
        
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            storage=storage,
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(
            lambda trial: self.objective(trial, param_ranges),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        return study


def example_metric_function(output_dir: str) -> float:
    """
    Example metric function - replace with your actual metric calculation.
    
    This should parse the output files in output_dir and return a single
    float value to be MINIMIZED.
    
    Args:
        output_dir: Directory containing reconstruction output
        
    Returns:
        Metric value (lower is better)
    """
    # Example: parse some output file and calculate chi2 or similar
    # This is just a placeholder - implement your actual metric here
    
    # For example, you might:
    # 1. Read ROOT files with uproot
    # 2. Calculate reconstruction efficiency
    # 3. Calculate resolution
    # 4. Return 1/efficiency or resolution as metric to minimize
    
    import random
    return random.random()  # Placeholder - replace with real metric!


def main():
    parser = argparse.ArgumentParser(
        description="Optimize NA6P reconstruction parameters using Optuna"
    )
    parser.add_argument(
        "--layout-ini",
        required=True,
        help="Path to layout INI file"
    )
    parser.add_argument(
        "--reco-ini",
        required=True,
        help="Path to reconstruction parameter INI file (template)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of optimization trials (default: 100)"
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=50000,
        help="Number of simulation events (default: 50000)"
    )
    parser.add_argument(
        "--n-rec",
        type=int,
        default=50000,
        help="Number of reconstruction events (default: 50000)"
    )
    parser.add_argument(
        "--work-dir",
        default="./optimization_work",
        help="Working directory (default: ./optimization_work)"
    )
    parser.add_argument(
        "--study-name",
        default="na6p_optimization",
        help="Optuna study name (default: na6p_optimization)"
    )
    parser.add_argument(
        "--storage",
        help="Optional Optuna storage URL (e.g., sqlite:///optuna.db)"
    )
    
    args = parser.parse_args()
    
    # Define parameter ranges to optimize
    # Format: param_name -> (min_value, max_value, type)
    # type can be 'float', 'int', or 'log' (for log-scale)
    param_ranges = {
        # Vertexer parameters
        'vertexerMaxDeltaThetaTracklet': (0.3, 1.0, 'float'),
        'vertexerMaxDeltaPhiTracklet': (0.01, 0.1, 'float'),
        'vertexerMaxDCAxy': (0.1, 0.5, 'float'),
        'vertexerKDEBandwidth': (0.1, 1.0, 'float'),
        
        # VT Tracker CA parameters (first iteration)
        'vtMaxDeltaThetaTrackletsCA[0]': (0.02, 0.1, 'float'),
        'vtMaxDeltaPhiTrackletsCA[0]': (0.05, 0.2, 'float'),
        
        # MS Tracker CA parameters (first iteration)
        'msMaxDeltaThetaTrackletsCA[0]': (0.03, 0.12, 'float'),
        'msMaxDeltaPhiTrackletsCA[0]': (0.05, 0.2, 'float'),
        
        # Add more parameters as needed
    }
    
    # Create optimizer
    optimizer = INIParameterOptimizer(
        layout_ini=args.layout_ini,
        reco_ini_template=args.reco_ini,
        metric_function=example_metric_function,  # Replace with your metric!
        n_sim_events=args.n_sim,
        n_rec_events=args.n_rec,
        work_dir=args.work_dir
    )
    
    # Run optimization
    study = optimizer.optimize(
        param_ranges=param_ranges,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage
    )
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best metric value: {study.best_value:.6f}")
    print("\nBest parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Save best parameters to file
    best_ini = Path(args.work_dir) / "best_reco_params.ini"
    optimizer.update_ini_file(study.best_params, best_ini)
    print(f"\nBest parameters saved to: {best_ini}")
    
    # Optionally: plot optimization history
    try:
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(Path(args.work_dir) / "optimization_history.png")
        print(f"Optimization history saved to: {args.work_dir}/optimization_history.png")
        
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(Path(args.work_dir) / "param_importances.png")
        print(f"Parameter importances saved to: {args.work_dir}/param_importances.png")
    except ImportError:
        print("\nInstall matplotlib for visualization: pip install matplotlib")


if __name__ == "__main__":
    main()