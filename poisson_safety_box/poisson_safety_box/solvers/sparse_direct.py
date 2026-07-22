"""Sparse direct solver for Poisson equations."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.sparse.linalg import spsolve

from .laplacian_matrix import build_laplacian_system, scatter_solution
from .residuals import poisson_residual


def solve_poisson_sparse_direct(
    rhs: np.ndarray,
    solve_mask: np.ndarray,
    boundary_values: np.ndarray,
    grid_spacing: tuple[float, ...],
) -> tuple[np.ndarray, dict]:
    """Solve Δh=rhs using a sparse direct solver."""
    start = perf_counter()
    A, b, index_map, free_cells, meta = build_laplacian_system(rhs, solve_mask, boundary_values, grid_spacing)
    x = spsolve(A, b) if A.shape[0] > 0 else np.array([])
    h = scatter_solution(x, solve_mask, boundary_values)
    elapsed = perf_counter() - start
    info = {"solver": "sparse_direct", "elapsed_time": elapsed, **meta}
    info["residual"] = poisson_residual(h, rhs, solve_mask, grid_spacing)
    return h, info
