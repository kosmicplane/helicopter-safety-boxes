"""Discrete-time safety backtracking for virtual marker integration.

The continuous-time CBF/HJ filter produces a velocity command.  This module
performs a conservative, version-local Euler step check before moving the
on-screen virtual vehicle.  It is intentionally independent from camera and UI
code so the same behavior can be tested in live and synthetic workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .coordinates import GridFieldSampler, GridGeometry
from .hj_reachability import path_is_collision_free


@dataclass(frozen=True)
class DiscreteStepResult:
    """Result of one collision- and Poisson-checked integration attempt."""

    position_xy: np.ndarray
    accepted_dt_s: float
    backtracks: int
    accepted: bool
    reason: str | None


def backtracked_safe_step(
    *,
    position_xy: Iterable[float],
    velocity_xy: Iterable[float],
    nominal_dt_s: float,
    geometry: GridGeometry,
    inflated_occupancy: np.ndarray,
    poisson_sampler: GridFieldSampler,
    h_margin: float,
    tolerance: float = 1.0e-6,
    maximum_backtracks: int = 12,
    maximum_dt_s: float = 0.12,
) -> DiscreteStepResult:
    """Return the largest accepted Euler step obtained by halving ``dt``.

    A trial step is accepted only when the endpoint is inside the metric
    workspace, the complete segment is collision-free in the inflated occupancy,
    and the sampled Poisson value respects the configured buffer.  No unsafe
    fallback state is returned.
    """

    position = np.asarray(position_xy, dtype=float).reshape(2)
    velocity = np.asarray(velocity_xy, dtype=float).reshape(2)
    occupancy = np.asarray(inflated_occupancy, dtype=bool)
    if occupancy.shape != geometry.shape_yx:
        raise ValueError("inflated_occupancy shape must match GridGeometry.")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        return DiscreteStepResult(position.copy(), 0.0, 0, False, "non-finite position or velocity")
    dt = min(float(nominal_dt_s), float(maximum_dt_s))
    if not np.isfinite(dt) or dt <= 0.0:
        return DiscreteStepResult(position.copy(), 0.0, 0, False, "non-positive integration step")

    last_reason = "discrete-time safety check failed"
    for backtracks in range(int(maximum_backtracks) + 1):
        candidate = position + dt * velocity
        if not geometry.contains_xy(candidate):
            last_reason = "trial state outside workspace"
        elif not path_is_collision_free(
            np.asarray([position, candidate], dtype=float), occupancy, geometry
        ):
            last_reason = "trial segment intersects inflated occupancy"
        else:
            sample = poisson_sampler.sample(candidate)
            if not sample.valid or sample.h is None:
                last_reason = f"invalid trial Poisson sample: {sample.reason}"
            elif float(sample.h) < float(h_margin) - float(tolerance):
                last_reason = "trial state violates configured Poisson margin"
            else:
                return DiscreteStepResult(candidate, dt, backtracks, True, None)
        dt *= 0.5

    return DiscreteStepResult(position.copy(), 0.0, int(maximum_backtracks) + 1, False, last_reason)


__all__ = ["DiscreteStepResult", "backtracked_safe_step"]
