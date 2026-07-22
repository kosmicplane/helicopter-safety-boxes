"""Control bound utilities."""
from __future__ import annotations

import numpy as np


def bounds_from_config(lower, upper, dim: int):
    """Convert optional list bounds into numpy arrays compatible with solvers."""
    lb = None if lower is None else np.asarray(lower, dtype=float)
    ub = None if upper is None else np.asarray(upper, dtype=float)
    if lb is not None and lb.shape != (dim,):
        raise ValueError("control_lower_bound dimension mismatch.")
    if ub is not None and ub.shape != (dim,):
        raise ValueError("control_upper_bound dimension mismatch.")
    if lb is not None and ub is not None and np.any(lb > ub):
        raise ValueError("control lower bound exceeds upper bound.")
    return lb, ub


def clip_to_bounds(u: np.ndarray, lb: np.ndarray | None, ub: np.ndarray | None) -> np.ndarray:
    """Clip a vector to optional component-wise lower/upper bounds."""
    out = np.asarray(u, dtype=float).copy()
    if lb is not None:
        out = np.maximum(out, lb)
    if ub is not None:
        out = np.minimum(out, ub)
    return out
