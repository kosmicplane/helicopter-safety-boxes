"""Guidance-vector-based smooth negative forcing."""

from __future__ import annotations

import numpy as np

from .base import ForcingResult
from ..guidance.divergence import compute_divergence


def build_guidance_forcing(
    solve_mask: np.ndarray,
    vector_field: np.ndarray,
    grid_spacing: tuple[float, ...],
    beta: float = 10.0,
    target_mean_abs: float | None = None,
) -> ForcingResult:
    """Build a smooth negative forcing from the divergence of guidance field."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    div_v = compute_divergence(vector_field, grid_spacing)
    f = -np.logaddexp(0.0, -beta * div_v) / beta
    f[solve_mask] = np.minimum(f[solve_mask], -1e-12)
    f[~solve_mask] = 0.0
    if target_mean_abs is not None and target_mean_abs > 0 and np.any(solve_mask):
        current = float(np.mean(np.abs(f[solve_mask])))
        if current > 1e-12:
            f *= float(target_mean_abs) / current
            f[~solve_mask] = 0.0
    diag = {
        "method": "guidance",
        "beta": float(beta),
        "target_mean_abs": target_mean_abs,
        "divergence": div_v,
    }
    return ForcingResult(f, diag)
