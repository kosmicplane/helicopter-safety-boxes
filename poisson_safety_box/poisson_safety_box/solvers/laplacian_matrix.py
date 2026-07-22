"""Sparse matrix assembly for finite-difference Poisson problems."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import sparse

from ..domain.connectivity import neighbor_offsets


def axis_coefficients(grid_spacing: tuple[float, ...]) -> tuple[float, ...]:
    """Return finite-difference coefficients 1/dx_i^2."""
    return tuple(1.0 / (float(s) ** 2) for s in grid_spacing)


def build_laplacian_system(
    rhs: np.ndarray,
    solve_mask: np.ndarray,
    boundary_values: np.ndarray,
    grid_spacing: tuple[float, ...],
):
    """Build the SPD system for -Δh = -f with Dirichlet boundaries.

    The continuous equation is Δh=f. We assemble the equivalent positive
    definite system -Δh=-f so that CG can be used.
    """
    rhs = np.asarray(rhs, dtype=float)
    solve_mask = np.asarray(solve_mask, dtype=bool)
    boundary_values = np.asarray(boundary_values, dtype=float)
    ndim = rhs.ndim
    coeffs = axis_coefficients(grid_spacing)

    index_map = -np.ones(rhs.shape, dtype=int)
    free_cells = np.argwhere(solve_mask)
    index_map[solve_mask] = np.arange(len(free_cells))

    rows, cols, data = [], [], []
    b = np.zeros(len(free_cells), dtype=float)

    for cell in free_cells:
        cell_t = tuple(cell)
        row = index_map[cell_t]
        center_coeff = 0.0
        b[row] = -rhs[cell_t]
        for off in neighbor_offsets(ndim):
            axis = next(i for i, d in enumerate(off) if d != 0)
            coeff = coeffs[axis]
            center_coeff += coeff
            nb = tuple(int(c + d) for c, d in zip(cell_t, off))
            if any(n < 0 or n >= rhs.shape[i] for i, n in enumerate(nb)):
                continue
            if solve_mask[nb]:
                rows.append(row); cols.append(index_map[nb]); data.append(-coeff)
            else:
                b[row] += coeff * boundary_values[nb]
        rows.append(row); cols.append(row); data.append(center_coeff)

    A = sparse.csr_matrix((data, (rows, cols)), shape=(len(free_cells), len(free_cells)))
    metadata = {
        "unknowns": int(len(free_cells)),
        "nonzeros": int(A.nnz),
        "ndim": int(ndim),
        "grid_spacing": tuple(float(s) for s in grid_spacing),
    }
    return A, b, index_map, free_cells, metadata


def scatter_solution(vector: np.ndarray, solve_mask: np.ndarray, boundary_values: np.ndarray) -> np.ndarray:
    """Scatter an unknown vector back into a full grid."""
    h = np.array(boundary_values, dtype=float)
    h[solve_mask] = vector
    return h
