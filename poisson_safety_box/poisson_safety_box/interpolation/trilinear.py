"""Bilinear/trilinear interpolation for grid data."""

from __future__ import annotations

import numpy as np


def interpolate_grid(data: np.ndarray, point: np.ndarray, grid_spacing: tuple[float, ...], origin: tuple[float, ...] | None = None):
    """Interpolate scalar/vector/tensor grid data at a physical point.

    The grid axes are assumed to start at origin and use uniform spacing. This
    helper supports 2D and 3D arrays, with optional trailing data dimensions.
    """
    origin = tuple(0.0 for _ in grid_spacing) if origin is None else origin
    point = np.asarray(point, dtype=float)
    dim = len(grid_spacing)
    base_shape = data.shape[:dim]
    idx = (point[:dim] - np.asarray(origin)) / np.asarray(grid_spacing)
    lo = np.floor(idx).astype(int)
    frac = idx - lo
    if np.any(lo < 0) or np.any(lo + 1 >= np.asarray(base_shape)):
        return None, False

    out = 0.0
    for bits in np.ndindex(*(2,) * dim):
        weight = 1.0
        ind = []
        for ax, bit in enumerate(bits):
            if bit == 0:
                weight *= (1.0 - frac[ax]); ind.append(lo[ax])
            else:
                weight *= frac[ax]; ind.append(lo[ax] + 1)
        out = out + weight * data[tuple(ind)]
    return out, True
