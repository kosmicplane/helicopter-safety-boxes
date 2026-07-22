"""Conjugate-gradient solver for the SPD -Laplacian system."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.sparse.linalg import cg

from .laplacian_matrix import build_laplacian_system, scatter_solution
from .residuals import poisson_residual


def solve_poisson_cg(
    rhs: np.ndarray,
    solve_mask: np.ndarray,
    boundary_values: np.ndarray,
    grid_spacing: tuple[float, ...],
    tolerance: float = 1e-6,
    max_iter: int = 2000,
) -> tuple[np.ndarray, dict]:
    """Solve Δh=rhs using conjugate gradient on -Δ."""
    start = perf_counter()
    A, b, index_map, free_cells, meta = build_laplacian_system(rhs, solve_mask, boundary_values, grid_spacing)
    if A.shape[0] == 0:
        x = np.array([]); info_code = 0
    else:
        try:
            x, info_code = cg(A, b, rtol=tolerance, atol=0.0, maxiter=max_iter)
        except TypeError:
            x, info_code = cg(A, b, tol=tolerance, maxiter=max_iter)
    h = scatter_solution(x, solve_mask, boundary_values)
    elapsed = perf_counter() - start
    info = {"solver": "conjugate_gradient", "elapsed_time": elapsed, "info_code": int(info_code), **meta}
    info["residual"] = poisson_residual(h, rhs, solve_mask, grid_spacing)
    return h, info
