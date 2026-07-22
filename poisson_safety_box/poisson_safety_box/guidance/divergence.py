"""Divergence computation for guidance vector fields."""

from __future__ import annotations

import numpy as np


def compute_divergence(vector_field: np.ndarray, grid_spacing: tuple[float, ...]) -> np.ndarray:
    """Compute divergence of a vector field stored with last dimension=dim."""
    dim = vector_field.shape[-1]
    if dim not in {2, 3}:
        raise ValueError("vector_field last dimension must be 2 or 3")
    div = np.zeros(vector_field.shape[:-1], dtype=float)
    for axis in range(dim):
        component = vector_field[..., axis]
        derivative = np.gradient(component, *grid_spacing, edge_order=1)[axis]
        div += derivative
    return div
