"""Successive Over-Relaxation solver for 2D/3D Poisson equations.

This file implements a vectorized red-black SOR update. Red-black coloring is
useful because all cells of one color only touch cells of the opposite color;
therefore each half-sweep can be updated with NumPy array operations.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .residuals import poisson_residual


def _neighbor_values(arr: np.ndarray, axis: int, direction: int) -> np.ndarray:
    """Return neighbor values shifted along one axis without wraparound."""
    out = np.zeros_like(arr)
    src = [slice(None)] * arr.ndim
    dst = [slice(None)] * arr.ndim
    if direction == -1:
        src[axis] = slice(None, -1)
        dst[axis] = slice(1, None)
    elif direction == 1:
        src[axis] = slice(1, None)
        dst[axis] = slice(None, -1)
    else:
        raise ValueError("direction must be -1 or 1")
    out[tuple(dst)] = arr[tuple(src)]
    return out


def _neighbor_sum(h: np.ndarray, coeffs: tuple[float, ...]) -> np.ndarray:
    """Compute Σ coeff_axis*(h_plus + h_minus)."""
    total = np.zeros_like(h, dtype=float)
    for axis, coeff in enumerate(coeffs):
        total += coeff * _neighbor_values(h, axis, -1)
        total += coeff * _neighbor_values(h, axis, 1)
    return total


def _checkerboard_masks(shape: tuple[int, ...], solve_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return red/black masks restricted to solve_mask."""
    indices = np.indices(shape)
    parity = np.sum(indices, axis=0) % 2
    return solve_mask & (parity == 0), solve_mask & (parity == 1)


def solve_poisson_sor(
    rhs: np.ndarray,
    solve_mask: np.ndarray,
    boundary_values: np.ndarray,
    grid_spacing: tuple[float, ...],
    omega: float = 1.75,
    max_iter: int = 4000,
    tolerance: float = 1e-4,
    residual_check_interval: int = 25,
    warm_start: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Solve Δh=rhs using vectorized red-black SOR.

    The update is based on the finite-difference equation:

        Σ_axis c_axis (h_plus + h_minus - 2 h_center) = rhs

    where c_axis = 1/dx_axis². Solving for h_center gives:

        h_star = (Σ_axis c_axis (h_plus + h_minus) - rhs) / (2Σ c_axis)

    Then SOR applies:

        h_new = (1 - omega) h_old + omega h_star
    """
    if not (0.0 < omega < 2.0):
        raise ValueError("omega must be in (0, 2)")
    rhs = np.asarray(rhs, dtype=float)
    solve_mask = np.asarray(solve_mask, dtype=bool)
    h = np.array(boundary_values, dtype=float)
    if warm_start is not None:
        h[solve_mask] = np.asarray(warm_start, dtype=float)[solve_mask]
    else:
        h[solve_mask] = 0.0

    coeffs = tuple(1.0 / (float(s) ** 2) for s in grid_spacing)
    denom = 2.0 * sum(coeffs)
    red, black = _checkerboard_masks(rhs.shape, solve_mask)
    residual_history = []
    start = perf_counter()
    converged = False
    max_change = float("inf")

    for it in range(1, max_iter + 1):
        max_change = 0.0
        for color_mask in (red, black):
            ns = _neighbor_sum(h, coeffs)
            h_star = (ns - rhs) / denom
            old_vals = h[color_mask].copy()
            new_vals = (1.0 - omega) * old_vals + omega * h_star[color_mask]
            h[color_mask] = new_vals
            if old_vals.size:
                change = float(np.max(np.abs(new_vals - old_vals)))
                if change > max_change:
                    max_change = change
        if it == 1 or it % residual_check_interval == 0 or max_change < tolerance:
            residual_history.append(float(max_change))
        if max_change < tolerance:
            converged = True
            break

    elapsed = perf_counter() - start
    info = {
        "solver": "sor",
        "elapsed_time": elapsed,
        "iterations": int(it),
        "converged": bool(converged),
        "residual_history": residual_history,
        "max_change": float(max_change),
        "unknowns": int(np.count_nonzero(solve_mask)),
    }
    info["residual"] = poisson_residual(h, rhs, solve_mask, grid_spacing)
    return h, info
