"""Validation helpers for safety data."""
from __future__ import annotations

import numpy as np


def require_matching_dimension(vector: np.ndarray, expected_dim: int, name: str) -> None:
    """Raise a clear error if a vector does not match the expected dimension."""
    v = np.asarray(vector)
    if v.shape != (expected_dim,):
        raise ValueError(f"{name} must have shape {(expected_dim,)}, got {v.shape}.")
