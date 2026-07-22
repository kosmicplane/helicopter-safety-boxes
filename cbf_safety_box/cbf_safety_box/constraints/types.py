"""Shared constraint dataclasses.

This file avoids circular imports between individual constraint builders and the
high-level dispatch logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Constraint:
    """Linear inequality constraint using the convention A u >= b."""

    A: np.ndarray
    b: np.ndarray
    name: str = "constraint"
    metadata: dict[str, Any] = field(default_factory=dict)

    def residual(self, u: np.ndarray) -> np.ndarray:
        """Return A u - b. Feasible constraints have nonnegative residuals."""
        return self.A @ np.asarray(u, dtype=float) - self.b
