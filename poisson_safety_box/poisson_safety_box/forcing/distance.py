"""Distance-based Holder-continuous Poisson forcing."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from .base import ForcingResult


def build_distance_forcing(
    solve_mask: np.ndarray,
    boundary_mask: np.ndarray,
    grid_spacing: tuple[float, ...],
    alpha: float = 0.5,
) -> ForcingResult:
    """Build distance forcing f=-(dist/||dist||_inf)^alpha.

    The distance is measured from the solve region to all non-solve cells. This
    is useful for comparison but typically has lower smoothness than constant
    or guidance-based smooth forcing.
    """
    if not (0 < alpha < 1.5):
        raise ValueError("alpha should typically be in (0, 1) for Holder forcing")
    dist = distance_transform_edt(solve_mask, sampling=grid_spacing)
    max_dist = float(dist[solve_mask].max()) if np.any(solve_mask) else 0.0
    f = np.zeros_like(dist, dtype=float)
    if max_dist > 0:
        f[solve_mask] = -((dist[solve_mask] / max_dist) ** alpha)
    diag = {
        "method": "distance",
        "alpha": float(alpha),
        "max_distance": max_dist,
        "distance_map": dist,
    }
    return ForcingResult(f, diag)
