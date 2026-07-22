"""Residual utilities for Poisson solvers."""

from __future__ import annotations

import numpy as np


def discrete_laplacian(h: np.ndarray, grid_spacing: tuple[float, ...]) -> np.ndarray:
    """Compute finite-difference Laplacian using NumPy gradients."""
    grads = np.gradient(h, *grid_spacing, edge_order=1)
    lap = np.zeros_like(h, dtype=float)
    for axis, grad_axis in enumerate(grads):
        lap += np.gradient(grad_axis, *grid_spacing, edge_order=1)[axis]
    return lap


def poisson_residual(h: np.ndarray, rhs: np.ndarray, solve_mask: np.ndarray, grid_spacing: tuple[float, ...]) -> dict:
    """Return residual norms for Δh-rhs in the solve region."""
    r = discrete_laplacian(h, grid_spacing) - rhs
    vals = r[solve_mask]
    if vals.size == 0:
        return {"max_abs": None, "l2": None}
    return {
        "max_abs": float(np.max(np.abs(vals))),
        "l2": float(np.linalg.norm(vals) / max(1, vals.size) ** 0.5),
    }
