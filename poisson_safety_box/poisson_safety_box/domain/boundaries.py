"""Boundary extraction for Dirichlet conditions."""

from __future__ import annotations

import numpy as np

from .connectivity import neighbor_offsets


def compute_boundary_mask(
    occupancy: np.ndarray,
    outer_boundary_as_dirichlet: bool = True,
) -> np.ndarray:
    """Compute a free-cell boundary mask from an occupancy matrix.

    A free cell is marked as a boundary cell if it touches an occupied cell via
    4-neighbor connectivity in 2D or 6-neighbor connectivity in 3D. Optionally,
    the outer domain boundary is also marked as Dirichlet.
    """
    occupied = np.asarray(occupancy, dtype=bool)
    free = ~occupied
    boundary = np.zeros_like(occupied, dtype=bool)
    shape = occupied.shape

    for offset in neighbor_offsets(occupied.ndim):
        src = []
        dst = []
        for axis, d in enumerate(offset):
            if d == -1:
                src.append(slice(1, None)); dst.append(slice(None, -1))
            elif d == 1:
                src.append(slice(None, -1)); dst.append(slice(1, None))
            else:
                src.append(slice(None)); dst.append(slice(None))
        src = tuple(src); dst = tuple(dst)
        boundary[src] |= free[src] & occupied[dst]

    if outer_boundary_as_dirichlet:
        if occupied.ndim == 2:
            boundary[0, :] |= free[0, :]
            boundary[-1, :] |= free[-1, :]
            boundary[:, 0] |= free[:, 0]
            boundary[:, -1] |= free[:, -1]
        else:
            boundary[0, :, :] |= free[0, :, :]
            boundary[-1, :, :] |= free[-1, :, :]
            boundary[:, 0, :] |= free[:, 0, :]
            boundary[:, -1, :] |= free[:, -1, :]
            boundary[:, :, 0] |= free[:, :, 0]
            boundary[:, :, -1] |= free[:, :, -1]

    return boundary


def compute_solve_mask(free_mask: np.ndarray, boundary_mask: np.ndarray) -> np.ndarray:
    """Return cells where h is unknown and Poisson is solved."""
    return np.asarray(free_mask, dtype=bool) & ~np.asarray(boundary_mask, dtype=bool)
