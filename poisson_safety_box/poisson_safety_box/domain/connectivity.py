"""Connectivity utilities for 2D and 3D grid masks."""

from __future__ import annotations

from typing import Iterable, Tuple


def neighbor_offsets(ndim: int) -> tuple[tuple[int, ...], ...]:
    """Return 4-neighbor offsets for 2D or 6-neighbor offsets for 3D."""
    if ndim == 2:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    if ndim == 3:
        return ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    raise ValueError("Only 2D and 3D grids are supported")


def inside(index: tuple[int, ...], shape: tuple[int, ...]) -> bool:
    """Return True if an index is inside a grid shape."""
    return all(0 <= i < n for i, n in zip(index, shape))


def iter_neighbors(index: tuple[int, ...], shape: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    """Yield valid neighbor indices for an index."""
    for off in neighbor_offsets(len(shape)):
        nb = tuple(i + di for i, di in zip(index, off))
        if inside(nb, shape):
            yield nb
