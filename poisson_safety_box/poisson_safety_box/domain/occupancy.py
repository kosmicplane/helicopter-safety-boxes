"""Occupancy matrix validation and normalization."""

from __future__ import annotations

import numpy as np

from ..utils.validation import require_bool_array


def normalize_occupancy(occupancy: np.ndarray) -> np.ndarray:
    """Return a boolean occupancy matrix.

    True means occupied. False means free.
    """
    return require_bool_array("occupancy", occupancy)


def make_empty_occupancy(shape: tuple[int, ...]) -> np.ndarray:
    """Create an empty 2D or 3D occupancy grid."""
    if len(shape) not in {2, 3}:
        raise ValueError("shape must be length 2 or 3")
    return np.zeros(shape, dtype=bool)


def add_box_2d(occupancy: np.ndarray, start: tuple[int, int], stop: tuple[int, int]) -> None:
    """Mark a rectangular region as occupied in a 2D grid."""
    if occupancy.ndim != 2:
        raise ValueError("add_box_2d requires a 2D occupancy grid")
    occupancy[start[0]:stop[0], start[1]:stop[1]] = True


def add_box_3d(occupancy: np.ndarray, start: tuple[int, int, int], stop: tuple[int, int, int]) -> None:
    """Mark a cuboid region as occupied in a 3D grid."""
    if occupancy.ndim != 3:
        raise ValueError("add_box_3d requires a 3D occupancy grid")
    occupancy[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = True
