"""Average-flux Poisson forcing."""

from __future__ import annotations

import numpy as np


def estimate_boundary_measure(boundary_mask: np.ndarray, grid_spacing: tuple[float, ...]) -> float:
    """Estimate boundary length in 2D or area in 3D.

    This simple estimate counts boundary cells and multiplies by a representative
    cell-face measure. It is robust enough for diagnostics and forcing scaling.
    """
    dim = boundary_mask.ndim
    if dim == 2:
        return float(np.count_nonzero(boundary_mask) * np.mean(grid_spacing))
    if dim == 3:
        dx, dy, dz = grid_spacing
        face_area = (dx * dy + dx * dz + dy * dz) / 3.0
        return float(np.count_nonzero(boundary_mask) * face_area)
    raise ValueError("Only 2D and 3D are supported")


def estimate_volume(solve_mask: np.ndarray, grid_spacing: tuple[float, ...]) -> float:
    """Estimate free volume/area used for average-flux scaling."""
    cell_volume = float(np.prod(grid_spacing))
    return float(np.count_nonzero(solve_mask) * cell_volume)


def build_average_flux_forcing(
    solve_mask: np.ndarray,
    boundary_mask: np.ndarray,
    grid_spacing: tuple[float, ...],
    b_bar: float = -1.0,
):
    """Build constant forcing f=b_bar*Area(boundary)/Vol(omega)."""
    from .base import ForcingResult

    if b_bar >= 0:
        raise ValueError("b_bar must be negative")
    boundary_measure = estimate_boundary_measure(boundary_mask, grid_spacing)
    volume = estimate_volume(solve_mask, grid_spacing)
    if volume <= 0:
        raise ValueError("solve region has zero volume")
    f_value = float(b_bar * boundary_measure / volume)
    f = np.zeros_like(solve_mask, dtype=float)
    f[solve_mask] = f_value
    diag = {
        "method": "average_flux",
        "b_bar": float(b_bar),
        "boundary_measure": boundary_measure,
        "volume": volume,
        "f_value": f_value,
    }
    return ForcingResult(f, diag)
