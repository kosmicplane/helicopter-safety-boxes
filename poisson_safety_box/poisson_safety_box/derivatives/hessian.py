"""Finite-difference Hessian utilities."""

from __future__ import annotations

import numpy as np


def compute_hessian(h: np.ndarray, grid_spacing: tuple[float, ...]) -> np.ndarray:
    """Compute Hessian of h with shape (*grid_shape, dim, dim)."""
    dim = h.ndim
    grad = np.gradient(h, *grid_spacing, edge_order=1)
    H = np.zeros(h.shape + (dim, dim), dtype=float)
    for i in range(dim):
        second = np.gradient(grad[i], *grid_spacing, edge_order=1)
        for j in range(dim):
            H[..., i, j] = second[j]
    # Symmetrize to reduce numerical asymmetry.
    for i in range(dim):
        for j in range(i + 1, dim):
            val = 0.5 * (H[..., i, j] + H[..., j, i])
            H[..., i, j] = val
            H[..., j, i] = val
    return H
