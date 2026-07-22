"""Constant negative Poisson forcing."""

from __future__ import annotations

import numpy as np

from .base import ForcingResult


def build_constant_forcing(solve_mask: np.ndarray, c: float = 1.0) -> ForcingResult:
    """Build f=-c in the Poisson solve region and f=0 elsewhere."""
    if c <= 0:
        raise ValueError("c must be positive because forcing is f=-c")
    f = np.zeros_like(solve_mask, dtype=float)
    f[solve_mask] = -float(c)
    diag = {
        "method": "constant",
        "c": float(c),
        "min": float(f[solve_mask].min()) if np.any(solve_mask) else None,
        "max": float(f[solve_mask].max()) if np.any(solve_mask) else None,
        "mean": float(f[solve_mask].mean()) if np.any(solve_mask) else None,
    }
    return ForcingResult(f, diag)
