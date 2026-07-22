"""Mask extraction for occupied and free-space regions."""

from __future__ import annotations

import numpy as np

from .occupancy import normalize_occupancy


def compute_basic_masks(occupancy: np.ndarray) -> dict[str, np.ndarray]:
    """Compute occupied and free masks from occupancy."""
    occupied = normalize_occupancy(occupancy)
    free = ~occupied
    return {
        "occupied_mask": occupied,
        "free_mask": free,
        "omega_union_boundary_mask": free.copy(),
    }
