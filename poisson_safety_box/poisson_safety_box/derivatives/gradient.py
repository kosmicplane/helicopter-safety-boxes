"""Finite-difference gradient utilities."""

from __future__ import annotations

import numpy as np


def compute_gradient(h: np.ndarray, grid_spacing: tuple[float, ...]) -> np.ndarray:
    """Compute gradient of h with last dimension equal to grid dimension."""
    grads = np.gradient(h, *grid_spacing, edge_order=1)
    return np.stack(grads, axis=-1)
