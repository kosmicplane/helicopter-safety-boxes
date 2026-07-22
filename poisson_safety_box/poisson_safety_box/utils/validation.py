"""Validation functions for occupancy and masks."""

from __future__ import annotations

import numpy as np


def require_bool_array(name: str, array: np.ndarray) -> np.ndarray:
    """Return a boolean copy of an input array and validate its dimension."""
    arr = np.asarray(array).astype(bool)
    if arr.ndim not in {2, 3}:
        raise ValueError(f"{name} must be 2D or 3D, got shape {arr.shape}")
    return arr


def require_nonempty_mask(name: str, mask: np.ndarray) -> None:
    """Raise an error if a boolean mask contains no true cells."""
    if not np.any(mask):
        raise ValueError(f"{name} is empty")


def normalize_spacing(spacing: tuple[float, ...], ndim: int) -> tuple[float, ...]:
    """Normalize spacing length to match the grid dimension."""
    if len(spacing) == ndim:
        return tuple(float(s) for s in spacing)
    if len(spacing) == 3 and ndim == 2:
        return (float(spacing[0]), float(spacing[1]))
    if len(spacing) == 2 and ndim == 3:
        raise ValueError("3D occupancy requires 3 grid spacing values")
    raise ValueError(f"Invalid spacing length {len(spacing)} for ndim={ndim}")
