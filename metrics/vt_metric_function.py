"""Single-objective VerTel metric: efficiency minus CPU time per track."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vt_multiobjective_metric_function import metric_function as measure_vt  # noqa: E402


OBJECTIVE_DIRECTIONS = ("maximize",)
OBJECTIVE_NAMES = ("efficiency - time per track",)
logger = logging.getLogger("metrics.vt_metric_function")


def metric_function(output_dir: str) -> float:
    """Return the VT bin-averaged efficiency minus seconds per reconstructed track."""
    efficiency, time_per_track = measure_vt(output_dir)
    score = efficiency - time_per_track
    logger.info("Metric: %.4f", score)
    return score
