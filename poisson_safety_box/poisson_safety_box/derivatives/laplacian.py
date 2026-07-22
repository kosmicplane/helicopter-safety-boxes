"""Finite-difference Laplacian utilities."""

from __future__ import annotations

import numpy as np

from .hessian import compute_hessian


def compute_laplacian(h: np.ndarray, grid_spacing: tuple[float, ...]) -> np.ndarray:
    """Compute Laplacian as trace of the finite-difference Hessian."""
    H = compute_hessian(h, grid_spacing)
    return np.trace(H, axis1=-2, axis2=-1)


def laplacian_from_hessian(hessian: np.ndarray) -> np.ndarray:
    """Return the trace of a Hessian field."""
    return np.trace(hessian, axis1=-2, axis2=-1)
