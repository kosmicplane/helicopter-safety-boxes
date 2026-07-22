"""High-level construction of CBF constraints.

The internal convention of this package is:

    A u >= b

where each row of ``A`` is one affine inequality in the decision variable ``u``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import CBFBoxConfig
from ..state import SystemState
from ..safety_data.sample import SafetySample
from .velocity_cbf import build_velocity_cbf_constraint
from .acceleration_hocbf import build_acceleration_hocbf_constraint


@dataclass
class Constraint:
    """Linear inequality of the form A u >= b."""

    A: np.ndarray
    b: np.ndarray
    name: str = "constraint"
    metadata: dict[str, Any] = field(default_factory=dict)

    def residual(self, u: np.ndarray) -> np.ndarray:
        """Return A u - b; feasible constraints have nonnegative residual."""
        return self.A @ np.asarray(u, dtype=float) - self.b


def build_constraints(config: CBFBoxConfig, state: SystemState, safety: SafetySample) -> list[Constraint]:
    """Build all CBF constraints for the requested mode.

    Control limits are handled separately as bounds by optimization solvers.
    """
    if config.mode == "velocity":
        return [build_velocity_cbf_constraint(safety, config.alpha)]
    if config.mode == "acceleration":
        if state.velocity is None:
            raise ValueError("Acceleration HOCBF mode requires state.velocity.")
        return [build_acceleration_hocbf_constraint(safety, state.velocity, config.alpha1, config.alpha2)]
    if config.mode == "backstepping":
        # Backstepping currently provides diagnostic values rather than a full QP
        # constraint because full derivatives of h_B require additional modeling.
        return [build_velocity_cbf_constraint(safety, config.alpha)]
    raise ValueError(f"Unsupported mode: {config.mode}")
