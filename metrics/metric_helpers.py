"""Small calculations shared by the reconstruction metrics."""

import re
from pathlib import Path

import numpy as np


def mc_kinematics(momentum: np.ndarray) -> dict[str, float] | None:
    """Return p, transverse momentum, and rapidity for a valid MC particle."""
    px, py, pz, energy = momentum
    pt = float(np.hypot(px, py))
    values = {
        "p": float(np.hypot(pt, pz)),
        "pt": pt,
        "y": float(np.arctanh(pz / energy)) if energy > abs(pz) else float("nan"),
    }
    return values if all(np.isfinite(value) for value in values.values()) else None


def append_kinematics(target: dict[str, list[float]], values: dict[str, float]) -> None:
    """Add one particle's p, pt, and y to histogram input lists."""
    for name, value in values.items():
        target[name].append(value)


def record_truth_kinematics(
    particle_ids: set[int],
    reconstructed_ids: set[int],
    momentum: np.ndarray,
    trackable: dict[str, list[float]],
    reconstructed: dict[str, list[float]],
) -> None:
    """Fill truth histograms for trackable and reconstructed particles."""
    for particle_id in particle_ids:
        values = mc_kinematics(momentum[particle_id])
        if values is None:
            continue
        append_kinematics(trackable, values)
        if particle_id in reconstructed_ids:
            append_kinematics(reconstructed, values)


def eligible_muon_ids(
    pdg: np.ndarray, mothers: np.ndarray, accepted_mother_pdgs: set[int]
) -> set[int]:
    """Find muons whose mother is accepted or absent."""
    return {
        particle_id
        for particle_id, (particle_pdg, mother_id) in enumerate(zip(pdg, mothers))
        if abs(int(particle_pdg)) == 13
        and (mother_id < 0 or (
            mother_id < len(pdg)
            and abs(int(pdg[mother_id])) in accepted_mother_pdgs
        ))
    }


def delta_electrons_by_mother(
    pdg: np.ndarray, mothers: np.ndarray, candidate_ids: set[int]
) -> dict[int, set[int]]:
    """Map each candidate to its own delta electron particle IDs."""
    result: dict[int, set[int]] = {}
    if candidate_ids and len(pdg):
        delta_ids = np.flatnonzero(
            (np.abs(pdg) == 11) & np.isin(mothers, list(candidate_ids))
        )
        for delta_id in delta_ids:
            result.setdefault(int(mothers[delta_id]), set()).add(int(delta_id))
    return result


def cpu_seconds(output_path: Path) -> float:
    """Read the reconstruction CPU time, defaulting to zero if unavailable."""
    log_path = output_path / "stdout.log"
    if not log_path.exists():
        return 0.0
    match = re.search(r"CP time\s+([0-9.]+)", log_path.read_text())
    return float(match.group(1)) if match else 0.0
