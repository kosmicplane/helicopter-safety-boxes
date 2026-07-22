"""Common constraint dataclasses.

All CBF constraints in this package use the convention:

    A u >= b

where A is a row vector or matrix and b is a scalar/vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Constraint:
    """Linear inequality constraint with convention A u >= b."""

    A: np.ndarray
    b: np.ndarray
    name: str = "constraint"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.A = np.atleast_2d(np.asarray(self.A, dtype=float))
        self.b = np.atleast_1d(np.asarray(self.b, dtype=float))
        if self.A.shape[0] != self.b.size:
            raise ValueError("Constraint row count must match b size.")

    def residual(self, u: np.ndarray) -> np.ndarray:
        """Return A u - b. Feasible means all residuals >= 0."""
        u = np.asarray(u, dtype=float)
        return self.A @ u - self.b
