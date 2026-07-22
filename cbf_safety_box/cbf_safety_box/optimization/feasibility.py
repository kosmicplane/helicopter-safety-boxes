"""Explicit numerical feasibility checks for CBF-QP outputs."""

from __future__ import annotations

import numpy as np


def check_feasibility(
    u: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    lower_bounds=None,
    upper_bounds=None,
    tolerance: float = 1.0e-8,
    norm_bound_indices: list[list[int]] | None = None,
    norm_bound_values: list[float] | None = None,
) -> dict:
    """Check affine rows, component bounds, and Euclidean norm bounds."""
    u = np.asarray(u, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if A.ndim == 1:
        A = A.reshape(1, -1)

    residuals = A @ u - b
    affine_feasible = bool(np.all(residuals >= -tolerance))
    lower_feasible = True if lower_bounds is None else bool(np.all(u >= np.asarray(lower_bounds) - tolerance))
    upper_feasible = True if upper_bounds is None else bool(np.all(u <= np.asarray(upper_bounds) + tolerance))

    groups = norm_bound_indices or []
    bounds = norm_bound_values or []
    norm_residuals = np.array(
        [float(bound) - float(np.linalg.norm(u[np.asarray(indices, dtype=int)])) for indices, bound in zip(groups, bounds)],
        dtype=float,
    )
    norm_feasible = bool(np.all(norm_residuals >= -tolerance)) if norm_residuals.size else True

    return {
        "feasible": affine_feasible and lower_feasible and upper_feasible and norm_feasible,
        "cbf_feasible": affine_feasible,
        "lower_feasible": lower_feasible,
        "upper_feasible": upper_feasible,
        "norm_feasible": norm_feasible,
        "residuals": residuals,
        "norm_residuals": norm_residuals,
        "min_residual": float(np.min(residuals)) if residuals.size else None,
        "min_norm_residual": float(np.min(norm_residuals)) if norm_residuals.size else None,
    }
