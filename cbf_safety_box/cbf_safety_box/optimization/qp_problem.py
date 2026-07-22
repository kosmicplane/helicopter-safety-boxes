"""Data model for the convex QPs solved by the CBF Safety Box."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class QPProblem:
    r"""Quadratic program in the internal convention ``A_ineq z >= b_ineq``.

    The objective is a weighted Euclidean projection,

    .. math::

        \min_z \tfrac12 (z-z_{nom})^T W (z-z_{nom}),

    where ``quadratic_weights`` stores the diagonal of :math:`W`.  Optional
    Euclidean norm bounds are included because acceleration commands are often
    bounded by ``||a||_2 <= a_max`` rather than component-wise clipping.
    """

    u_nom: np.ndarray
    A_ineq: np.ndarray
    b_ineq: np.ndarray
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    quadratic_weights: np.ndarray | None = None
    norm_bound_indices: list[list[int]] = field(default_factory=list)
    norm_bound_values: list[float] = field(default_factory=list)
    use_slack: bool = False
    slack_weight: float = 1.0e4
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize array shapes and reject malformed optimization problems."""
        self.u_nom = np.asarray(self.u_nom, dtype=float).reshape(-1)
        self.A_ineq = np.asarray(self.A_ineq, dtype=float)
        self.b_ineq = np.asarray(self.b_ineq, dtype=float).reshape(-1)

        if self.A_ineq.ndim == 1:
            self.A_ineq = self.A_ineq.reshape(1, -1)
        if self.A_ineq.shape[0] != self.b_ineq.size:
            raise ValueError("A_ineq and b_ineq row mismatch.")
        if self.A_ineq.shape[1] != self.u_nom.size:
            raise ValueError("A_ineq columns must match u_nom dimension.")

        if self.lower_bounds is not None:
            self.lower_bounds = np.asarray(self.lower_bounds, dtype=float).reshape(-1)
            if self.lower_bounds.size != self.u_nom.size:
                raise ValueError("lower_bounds must match the decision dimension.")
        if self.upper_bounds is not None:
            self.upper_bounds = np.asarray(self.upper_bounds, dtype=float).reshape(-1)
            if self.upper_bounds.size != self.u_nom.size:
                raise ValueError("upper_bounds must match the decision dimension.")

        if self.quadratic_weights is None:
            self.quadratic_weights = np.ones(self.u_nom.size, dtype=float)
        else:
            self.quadratic_weights = np.asarray(self.quadratic_weights, dtype=float).reshape(-1)
            if self.quadratic_weights.size != self.u_nom.size:
                raise ValueError("quadratic_weights must match the decision dimension.")
            if np.any(self.quadratic_weights <= 0.0):
                raise ValueError("quadratic_weights must be strictly positive.")

        if len(self.norm_bound_indices) != len(self.norm_bound_values):
            raise ValueError("Each norm-bound index group requires one bound value.")
        for indices, bound in zip(self.norm_bound_indices, self.norm_bound_values):
            if not indices:
                raise ValueError("A norm-bound index group cannot be empty.")
            if any(index < 0 or index >= self.u_nom.size for index in indices):
                raise ValueError("A norm-bound index is outside the decision vector.")
            if float(bound) <= 0.0:
                raise ValueError("Norm bounds must be strictly positive.")
