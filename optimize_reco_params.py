#!/usr/bin/env python3
"""
Parameter optimization script for NA6P reconstruction using Optuna.
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.fixes")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=".*pkg_resources.*"
)
warnings.filterwarnings("ignore", message=".*is experimental.*")

import argparse
import importlib.util
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict
import matplotlib.pyplot as plt

import optuna
from optuna.trial import Trial


logger = logging.getLogger(__name__)


def setup_logging(
    log_level: int = logging.INFO,
) -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def load_metric_function(module_path: str) -> Callable[[str], float]:
    """Dynamically load a metric function from a Python module."""
    path = Path(module_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Metric module not found: {path}")

    spec = importlib.util.spec_from_file_location("metric_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["metric_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "metric_function"):
        raise AttributeError(
            f"Module {path} must define a 'metric_function(output_dir: str) -> float' function"
        )

    func = getattr(module, "metric_function")
    if not callable(func):
        raise TypeError(f"'metric_function' in {path} must be callable")

    return func


def load_param_ranges(config_path: str) -> Dict[str, Any]:
    """Load optimization parameter configuration from a JSON or JSON5 file."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Parameter ranges file not found: {path}")

    if path.suffix.lower() == ".json5":
        try:
            import json5
        except ImportError as exc:
            raise ImportError(
                "Loading .json5 files requires the optional 'json5' package. "
                "Install it with: pip install json5"
            ) from exc

        with open(path, "r", encoding="utf-8") as file_handle:
            data = json5.load(file_handle)
    else:
        with open(path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

    if not isinstance(data, dict):
        raise ValueError(
            "Parameter ranges file must contain a JSON object at the top level"
        )

    parameters = data.get("parameters", data)
    if not isinstance(parameters, dict):
        raise ValueError("The 'parameters' section must be a JSON object")

    normalized: Dict[str, Any] = {
        "scalar": {},
        "iterated": {},
        "reco_options": {},
        "simulation_options": {},
    }
    for param_name, spec in parameters.items():
        normalized_spec = _normalize_param_spec(param_name, spec)
        if "iterations_param" in normalized_spec or "iterations" in normalized_spec:
            normalized["iterated"][str(param_name)] = normalized_spec
        else:
            normalized["scalar"][str(param_name)] = normalized_spec

    reco_options = data.get("reco_options", data.get("reconstruction_options", {}))
    if reco_options is None:
        reco_options = {}
    if not isinstance(reco_options, dict):
        raise ValueError(
            "The 'reco_options' (or 'reconstruction_options') section must be a JSON object"
        )
    normalized["reco_options"] = reco_options

    simulation_options = data.get("simulation_options", data.get("sim_options", {}))
    if simulation_options is None:
        simulation_options = {}
    if not isinstance(simulation_options, dict):
        raise ValueError(
            "The 'simulation_options' (or 'sim_options') section must be a JSON object"
        )
    normalized["simulation_options"] = simulation_options

    objectives = data.get("objective", ["maximize"])
    if isinstance(objectives, str):
        objectives = [objectives]
    normalized["objectives"] = objectives

    return normalized


def _normalize_param_spec(param_name: str, spec: Any) -> Dict[str, Any]:
    """Normalize a parameter spec from JSON/JSON5 into a dictionary."""
    if isinstance(spec, dict):
        if {"min", "max", "type"}.issubset(spec):
            normalized_spec = {
                "min": spec["min"],
                "max": spec["max"],
                "type": str(spec["type"]),
            }
            if "iterations_param" in spec:
                normalized_spec["iterations_param"] = str(spec["iterations_param"])
            if "iterations" in spec:
                normalized_spec["iterations"] = spec["iterations"]
            if "monotone_increasing" in spec:
                normalized_spec["monotone_increasing"] = bool(spec["monotone_increasing"])
            return normalized_spec

        raise ValueError(
            f"Parameter '{param_name}' must define 'min', 'max', and 'type'"
        )

    if isinstance(spec, (list, tuple)) and len(spec) == 3:
        min_value, max_value, param_type = spec
        return {
            "min": min_value,
            "max": max_value,
            "type": str(param_type),
        }

    raise ValueError(
        f"Parameter '{param_name}' must be either an object with min/max/type or a 3-item array"
    )


class INIParameterOptimizer:
    """Optimizer for NA6P reconstruction parameters."""

    def __init__(
        self,
        layout_ini: str,
        reco_ini_template: str,
        metric_function: Callable[[str], float],
        n_events: int = 50000,
        work_dir: str = "./optimization_work",
        reco_options: Dict[str, Any] = None,
        simulation_options: Dict[str, Any] = None,
        monotone_increasing: bool = False,
    ):
        self.layout_ini = Path(layout_ini).resolve()
        self.reco_ini_template = Path(reco_ini_template).resolve()
        self.metric_function = metric_function
        self.n_events = n_events
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.study_dir = self.work_dir
        self.sim_done = False
        self._reco_lock = threading.Lock()
        # NEW: global monotone_increasing flag (overridden per-param if set in config)
        self.monotone_increasing = monotone_increasing

        default_reco_options = {
            "doMatching": True,
            "doHitsToRecPoints": True,
            "doTrackletVertex": True,
            "doVTTracking": True,
            "doMSTracking": True
        }
        self.reco_options: Dict[str, Any] = dict(default_reco_options)
        if reco_options:
            for option_name, option_value in reco_options.items():
                normalized_option_name = str(option_name)
                if normalized_option_name.startswith("--"):
                    normalized_option_name = normalized_option_name[2:]
                self.reco_options[normalized_option_name] = option_value

        default_simulation_options = {
            "generator": '$NA6PROOT_ROOT/share/test/genDimuonBgEvent.C+(1,"Omega")',
            "hook": ""
        }
        self.simulation_options: Dict[str, Any] = dict(default_simulation_options)
        if simulation_options:
            for option_name, option_value in simulation_options.items():
                self.simulation_options[str(option_name)] = option_value

        if not self.layout_ini.exists():
            raise FileNotFoundError(f"Layout INI not found: {self.layout_ini}")
        if not self.reco_ini_template.exists():
            raise FileNotFoundError(
                f"Reco INI template not found: {self.reco_ini_template}"
            )

    @staticmethod
    def _format_cli_option_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _sanitize_study_name(study_name: str) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", study_name).strip("._")
        return safe_name or "na6p_optimization"

    def _create_versioned_study_dir(self, study_name: str) -> Path:
        safe_name = self._sanitize_study_name(study_name)
        study_root = self.work_dir / safe_name
        study_root.mkdir(parents=True, exist_ok=True)

        version = 1
        while (study_root / f"v{version:03d}").exists():
            version += 1

        study_dir = study_root / f"v{version:03d}"
        study_dir.mkdir(parents=True, exist_ok=False)
        return study_dir

    def _get_study_run_dir(self, study_name: str) -> Path:
        safe_name = self._sanitize_study_name(study_name)
        run_dir = self.work_dir / safe_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _enable_sqlite_wal(db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as connection:
            connection.execute("PRAGMA journal_mode=WAL;")
            connection.execute("PRAGMA synchronous=NORMAL;")
            connection.execute("PRAGMA busy_timeout=60000;")
            connection.commit()

    def _prepare_study_storage(
        self, study_name: str, storage: str | None
    ) -> tuple[str | None, str, bool]:
        if storage:
            normalized_storage = storage.strip()
            if normalized_storage.lower().startswith("sqlite:///"):
                db_path = Path(os.path.expanduser(normalized_storage[10:])).resolve()
                self._enable_sqlite_wal(db_path)

            self.study_dir = self._get_study_run_dir(study_name)

            return normalized_storage, study_name, True

        study_dir = self._create_versioned_study_dir(study_name)
        self.study_dir = study_dir
        db_path = study_dir / "optuna_study.db"
        self._enable_sqlite_wal(db_path)

        safe_name = self._sanitize_study_name(study_name)
        versioned_study_name = f"{safe_name}_{study_dir.name}"
        sqlite_storage = f"sqlite:///{db_path}"
        return sqlite_storage, versioned_study_name, False

    def run_simulation(self):
        """Run na6psim once (only needs to be done once)."""
        if self.sim_done:
            return

        logger.info("Running simulation (na6psim)...")
        generator = os.path.expandvars(
            str(self.simulation_options.get("generator", ""))
        )
        hook = os.path.expandvars(
            str(self.simulation_options.get("hook", ""))
        )
        staged_layout_ini_path = self._stage_layout_ini(
            config_dir=self.study_dir,
            input_dir=None,
            output_dir=self.study_dir,
            file_name="layout_simulation.ini",
        )
        cmd = [
            "na6psim",
            f"-n{self.n_events}",
            "-g",
            generator,
            *([f"-u{hook}"] if hook else []),
            "--load-ini",
            str(staged_layout_ini_path),
        ]
        logger.info("Executing command: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.study_dir,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Simulation failed:\n{result.stderr}")

        logger.info("Simulation completed successfully")
        self.sim_done = True

    def _suggest_param_value(
        self, trial: Trial, param_name: str, spec: Dict[str, Any]
    ) -> Any:
        min_value = spec["min"]
        max_value = spec["max"]
        param_type = str(spec["type"]).lower()

        if param_type == "float":
            return trial.suggest_float(param_name, float(min_value), float(max_value))
        if param_type == "int":
            return trial.suggest_int(param_name, int(min_value), int(max_value))
        if param_type == "log":
            return trial.suggest_float(
                param_name, float(min_value), float(max_value), log=True
            )

        raise ValueError(f"Unknown parameter type for '{param_name}': {param_type}")

    def _suggest_monotone_increasing_values(
        self,
        trial: Trial,
        param_name: str,
        spec: Dict[str, Any],
        iteration_count: int,
    ) -> Dict[str, Any]:
        """
        Sample `iteration_count` values for `param_name` that are guaranteed to be
        monotonically non-decreasing across iterations.

        Strategy: sample N percentages pct[i] in [0, 1] with FIXED bounds (so that
        all samplers, including CmaEsSampler, work without fallback), then reconstruct
        the actual values as:

            v[0] = min + pct[0] * (max - min)
            v[i] = v[i-1] + pct[i] * (max - v[i-1])   for i >= 1

        Each pct splits the remaining headroom above the previous value, so the
        sequence is always non-decreasing and stays within [min, max].

        The percentage variables are named  <param_name>_pct[i]  and never appear
        in the INI file; only the reconstructed  <param_name>[i]  values do.
        """
        min_val = float(spec["min"])
        max_val = float(spec["max"])
        param_type = str(spec["type"]).lower()

        if param_type not in ("float", "log", "int"):
            raise ValueError(f"Unknown type '{param_type}' for parameter '{param_name}'")

        result: Dict[str, Any] = {}
        prev = min_val

        for i in range(iteration_count):
            pct_name = f"{param_name}_pct[{i}]"
            pct = trial.suggest_float(pct_name, 0.0, 1.0)   # always [0,1] → fixed bounds
            new_val = prev + pct * (max_val - prev)

            if param_type == "int":
                new_val = round(new_val)

            result[f"{param_name}[{i}]"] = new_val
            prev = new_val

        return result

    def update_ini_file(self, params: Dict[str, Any], output_path: Path):
        """Update INI file with new parameter values."""
        with open(self.reco_ini_template, "r", encoding="utf-8") as file_handle:
            lines = file_handle.readlines()

        updated_lines = []
        for line in lines:
            stripped_line = line.strip()
            modified = False

            for param_name, param_value in params.items():
                pattern = rf"^({re.escape(param_name)}(?:\[\d+\])?)\s*="
                match = re.match(pattern, stripped_line)
                if match:
                    updated_lines.append(f"{match.group(1)}={param_value}\n")
                    modified = True
                    break

            if not modified:
                updated_lines.append(line)

        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.writelines(updated_lines)

    def _stage_layout_ini(
        self,
        config_dir: Path,
        input_dir: Path | None,
        output_dir: Path | None,
        file_name: str,
    ) -> Path:
        """Create a layout INI with explicit input/output directories when requested."""
        with open(self.layout_ini, "r", encoding="utf-8") as file_handle:
            lines = file_handle.readlines()

        updated_lines = []
        input_set = input_dir is None
        output_set = output_dir is None
        for line in lines:
            if re.match(r"^\s*input_dir\s*=", line):
                if input_dir is None:
                    updated_lines.append(line)
                else:
                    logger.warning(
                        "Warning: Overriding existing input_dir in layout INI "
                        f"for {config_dir.name} with {input_dir}"
                    )
                    updated_lines.append(f"input_dir={input_dir}\n")
                input_set = True
            elif re.match(r"^\s*output_dir\s*=", line):
                if output_dir is None:
                    updated_lines.append(line)
                else:
                    updated_lines.append(f"output_dir={output_dir}\n")
                output_set = True
            else:
                updated_lines.append(line)

        if not input_set:
            updated_lines.append(f"input_dir={input_dir}\n")
        if not output_set:
            updated_lines.append(f"output_dir={output_dir}\n")

        staged_layout_ini_path = (config_dir / file_name).resolve()
        with open(staged_layout_ini_path, "w", encoding="utf-8") as file_handle:
            file_handle.writelines(updated_lines)

        return staged_layout_ini_path

    def run_reconstruction(self, reco_ini: Path, trial_dir: Path) -> float:
        """Run reconstruction with given parameters."""
        logger.info("Running reconstruction in %s...", trial_dir)

        with self._reco_lock:
            for trial_file in trial_dir.glob("*.root"):
                trial_file.unlink()

            reco_ini_path = Path(reco_ini).resolve()
            staged_layout_ini_path = self._stage_layout_ini(
                config_dir=trial_dir.resolve(),
                input_dir=self.study_dir,
                output_dir=trial_dir.resolve(),
                file_name="layout_trial.ini",
            )

            cmd = [
                "na6prec",
                f"-l{self.n_events}",
                "--load-recoparam",
                str(reco_ini_path),
                "--load-ini",
                str(staged_layout_ini_path),
            ]

            for option_name, option_value in self.reco_options.items():
                cmd.extend(
                    [f"--{option_name}", self._format_cli_option_value(option_value)]
                )

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.study_dir,
            )

            if result.returncode != 0:
                if result.stdout:
                    logger.error("Reconstruction output:\n%s", result.stdout)
                    with open(str(trial_dir / "stdout.log"), "w") as f:
                        f.write(result.stdout)
                if result.stderr:
                    logger.error("Reconstruction failed:\n%s", result.stderr)
                else:
                    logger.error("Reconstruction failed with no stderr output.")
                raise optuna.TrialPruned()

            with open(str(trial_dir / "stdout.log"), "w") as f:
                f.write(result.stdout)

            # Copy simulation-only files (e.g. HitsVerTel.root, geometry.root)
            # that na6prec does not write but the metric may need.
            import shutil as _shutil
            for work_file in self.study_dir.glob("*.root"):
                trial_file = trial_dir / work_file.name
                if not trial_file.exists():
                    _shutil.copy2(work_file, trial_file)

            try:
                metric = self.metric_function(str(trial_dir))
            except Exception as error:
                logger.exception("Metric evaluation failed for %s: %s", trial_dir, error)
                raise optuna.TrialPruned()

            logger.info("Metric value: %s", metric)
            return metric

    def objective(self, trial: Trial, param_config: Dict[str, Any]) -> float:
        """Objective function for Optuna."""
        trial_dir = self.study_dir / f"trial_{trial.number}"
        trial_dir.mkdir(exist_ok=True)

        params: Dict[str, Any] = {}
        scalar_params = param_config.get("scalar", {})
        iterated_params = param_config.get("iterated", {})

        for param_name, spec in scalar_params.items():
            params[param_name] = self._suggest_param_value(trial, param_name, spec)

        for param_name, spec in iterated_params.items():
            if "iterations_param" in spec:
                iterations_param = spec["iterations_param"]
                if iterations_param not in params:
                    raise ValueError(
                        f"Iterated parameter '{param_name}' depends on missing count parameter '{iterations_param}'"
                    )
                iteration_count = int(params[iterations_param])
            elif "iterations" in spec:
                iteration_count = int(spec["iterations"])
            else:
                raise ValueError(
                    f"Iterated parameter '{param_name}' must define either 'iterations_param' or 'iterations'"
                )

            if iteration_count < 0:
                raise ValueError(
                    f"Iteration count for '{param_name}' must be non-negative"
                )

            use_monotone = spec.get("monotone_increasing", self.monotone_increasing)

            if use_monotone:
                indexed_params = self._suggest_monotone_increasing_values(
                    trial, param_name, spec, iteration_count
                )
                params.update(indexed_params)
            else:
                for index in range(iteration_count):
                    indexed_name = f"{param_name}[{index}]"
                    params[indexed_name] = self._suggest_param_value(
                        trial, indexed_name, spec
                    )

        trial_reco_ini = trial_dir / "reco_params.ini"
        self.update_ini_file(params, trial_reco_ini)

        # Store the fully reconstructed params (with param[i] keys, not _pct/_delta
        # internals) as a user attribute so we can recover them for the best trial
        # without having to re-run the sampling logic.
        import json as _json
        trial.set_user_attr("reconstructed_params", _json.dumps(
            {k: float(v) if not isinstance(v, int) else v for k, v in params.items()}
        ))

        try:
            metric = self.run_reconstruction(trial_reco_ini, trial_dir)
            return metric
        finally:
            pass

    def optimize(
        self,
        param_config: Dict[str, Any],
        n_trials: int = 100,
        study_name: str = "na6p_optimization",
        storage: str = None,
        n_jobs: int = 1,
        skip_simulation: bool = False,
        sampler_name: str = "tpe",
        directions: list[str] = None,
    ) -> optuna.Study:
        """Run the optimization.

        Args:
            sampler_name: Which Optuna sampler to use. Choices:
                - "tpe"  (default) — TPESampler, handles dynamic search spaces well.
                - "cmaes"          — CmaEsSampler, good for continuous spaces; requires
                                     monotone_increasing=True (fixed [0,1] bounds) to
                                     avoid independent-sampling fallback warnings.
                - "random"         — RandomSampler, useful as a baseline.
                - "nsga2"          — NSGAIISampler (multi-objective capable).
        """
        effective_storage, effective_study_name, load_if_exists = (
            self._prepare_study_storage(study_name, storage)
        )
        logger.info("Study directory: %s", self.study_dir)
        if effective_storage and effective_storage.lower().startswith("sqlite:///"):
            logger.info("Study storage: %s (WAL mode)", effective_storage)

        if not skip_simulation:
            self.run_simulation()

        sampler_name = sampler_name.lower()
        if sampler_name == "tpe":
            sampler = optuna.samplers.TPESampler(
                multivariate=True,
                group=True,
                seed=42,
                constant_liar=True,
            )
        elif sampler_name == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(
                seed=42,
                warn_independent_sampling=False,
                restart_strategy="ipop",
            )
        elif sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=42)
        elif sampler_name == "nsga2":
            sampler = optuna.samplers.NSGAIISampler(seed=42)
        else:
            raise ValueError(
                f"Unknown sampler '{sampler_name}'. Choose from: tpe, cmaes, random, nsga2"
            )

        is_multi = directions is not None and len(directions) > 1
        study = optuna.create_study(
            study_name=effective_study_name,
            direction=directions[0] if not is_multi else None,
            directions=directions if is_multi else None,
            storage=effective_storage,
            load_if_exists=load_if_exists,
            sampler=sampler,
        )

        study.optimize(
            lambda trial: self.objective(trial, param_config),
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=n_jobs,
        )

        return study


def example_metric_function(output_dir: str) -> float:
    """Example metric function - replace with your actual metric calculation."""
    import random
    return random.random()


def main():
    parser = argparse.ArgumentParser(
        description="Optimize NA6P reconstruction parameters using Optuna"
    )
    parser.add_argument("--layout-ini", "-l", required=True, help="Path to layout INI file")
    parser.add_argument(
        "--reco-ini", "-r", required=True,
        help="Path to reconstruction parameter INI file (template)",
    )
    parser.add_argument(
        "--n-trials", "-t", type=int, default=10,
        help="Number of optimization trials (default: 10)",
    )
    parser.add_argument(
        "--n-events", "-n", type=int, default=100,
        help="Number of simulation events (default: 100)",
    )
    parser.add_argument(
        "--work-dir", "-d", default="./optimization_work",
        help="Working directory (default: ./optimization_work)",
    )
    parser.add_argument(
        "--study-name", default="na6p_optimization",
        help="Optuna study name (default: na6p_optimization)",
    )
    parser.add_argument(
        "--param-ranges", "-p", default="params/param_ranges.json",
        help="Path to parameter ranges config file (default: params/param_ranges.json)",
    )
    parser.add_argument(
        "--metric-module", "-m", default="metrics/example_metric_function.py",
        help="Path to Python module defining 'metric_function' (default: metrics/example_metric_function.py)",
    )
    parser.add_argument(
        "--storage",
        help="Optional Optuna storage URL (e.g., sqlite:///optuna.db)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--skip-simulation", "-s", action="store_true",
        help="Skip the simulation step (use with pre-generated simulation data)",
    )
    parser.add_argument(
        "--n-jobs", "-j", type=int, default=1,
        help="Number of parallel jobs for optimization (default: 1)",
    )
    parser.add_argument(
        "--sampler",
        default="tpe",
        choices=["tpe", "cmaes", "random", "nsga2"],
        help=(
            "Optuna sampler to use (default: tpe). "
            "Use 'cmaes' together with --monotone-increasing for best results: "
            "the fixed [0,1] percentage encoding avoids dynamic-search-space warnings."
        ),
    )
    parser.add_argument(
        "--monotone-increasing",
        action="store_true",
        default=False,
        help=(
            "Force all iterated parameters to be monotonically non-decreasing "
            "across iterations. Can also be set per-parameter in the config file "
            "via 'monotone_increasing': true."
        ),
    )

    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level)

    param_config = load_param_ranges(args.param_ranges)
    objectives    = param_config.get("objectives", ["maximize"])

    metric_func = load_metric_function(args.metric_module)

    optimizer = INIParameterOptimizer(
        layout_ini=args.layout_ini,
        reco_ini_template=args.reco_ini,
        metric_function=metric_func,
        n_events=args.n_events,
        work_dir=args.work_dir,
        reco_options=param_config.get("reco_options", {}),
        simulation_options=param_config.get("simulation_options", {}),
        monotone_increasing=args.monotone_increasing,
    )

    study = optimizer.optimize(
        param_config=param_config,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        n_jobs=args.n_jobs,
        skip_simulation=args.skip_simulation,
        sampler_name=args.sampler,
        directions=objectives,
    )
    output_dir = optimizer.study_dir

    logger.info("\n%s", "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("%s", "=" * 80)

    is_multi = len(param_config.get("objectives", ["maximize"])) > 1
    obj_names = param_config.get("objective_names", [f"obj_{i}" for i in range(len(param_config.get("objectives", ["maximize"])))])

    if is_multi:
        pareto = study.best_trials
        if not pareto:
            logger.warning("No completed trials found. All trials were pruned or failed.")
            return
        logger.info("Pareto front: %s non-dominated solutions", len(pareto))
        for t in pareto:
            vals = ", ".join(f"{n}={v:.6f}" for n, v in zip(obj_names, t.values))
            logger.info("  Trial %s: %s", t.number, vals)
            reconstructed = json.loads(t.user_attrs.get("reconstructed_params", "{}"))
            best_ini = output_dir / f"best_reco_params_pareto_{t.number}.ini"
            optimizer.update_ini_file(reconstructed or t.params, best_ini)
            logger.info("    Saved to: %s", best_ini)
    else:
        if not study.best_trials:
            logger.warning("No completed trials found. All trials were pruned or failed.")
            return
        best_trial = study.best_trial
        logger.info("Best trial: %s", best_trial.number)
        logger.info("Best metric value: %.6f", study.best_value)
        reconstructed = json.loads(best_trial.user_attrs.get("reconstructed_params", "{}"))
        best_ini_params = reconstructed or study.best_params
        logger.info("Best parameters (reconstructed):")
        for param, value in best_ini_params.items():
            logger.info("  %s: %s", param, value)
        best_ini = output_dir / "best_reco_params.ini"
        optimizer.update_ini_file(best_ini_params, best_ini)
        logger.info("Best parameters saved to: %s", best_ini)

    if is_multi:
        if len(obj_names) == 2:
            ax = optuna.visualization.matplotlib.plot_pareto_front(
                study, target_names=obj_names
            )
            fig = ax.get_figure()
            fig.savefig(output_dir / "pareto_front.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Pareto front saved to: %s/pareto_front.png", output_dir)

        for i, obj_name in enumerate(obj_names):
            target_fn = lambda t, i=i: t.values[i]

            ax = optuna.visualization.matplotlib.plot_optimization_history(
                study, target=target_fn, target_name=obj_name
            )
            fig = ax.get_figure()
            fig.set_size_inches(12, 6)
            plt.tight_layout()
            fig.savefig(
                output_dir / f"optimization_history_{obj_name}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)

            ax = optuna.visualization.matplotlib.plot_param_importances(
                study, target=target_fn, target_name=obj_name
            )
            fig = ax.get_figure()
            fig.set_size_inches(12, max(6, len(study.best_trials[0].params) * 0.3))
            plt.tight_layout()
            fig.savefig(
                output_dir / f"param_importances_{obj_name}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)

        logger.info("Per-objective plots saved to: %s/", output_dir)

    else:
        ax = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig = ax.get_figure()
        fig.set_size_inches(12, 6)
        plt.tight_layout()
        fig.savefig(
            output_dir / "optimization_history.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        logger.info("Optimization history saved to: %s/optimization_history.png", output_dir)

        ax = optuna.visualization.matplotlib.plot_param_importances(study)
        fig = ax.get_figure()
        fig.set_size_inches(12, max(6, len(study.best_params) * 0.3))
        plt.tight_layout()
        fig.savefig(
            output_dir / "param_importances.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        logger.info("Parameter importances saved to: %s/param_importances.png", output_dir)


if __name__ == "__main__":
    main()