"""Boundary normal estimation from signed distance fields."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def signed_distance_from_occupancy(occupancy_mask: np.ndarray, grid_spacing: tuple[float, ...]) -> np.ndarray:
    """Compute a signed distance field, positive in free space."""
    occupied = np.asarray(occupancy_mask, dtype=bool)
    free = ~occupied
    dist_free = distance_transform_edt(free, sampling=grid_spacing)
    dist_occ = distance_transform_edt(occupied, sampling=grid_spacing)
    return dist_free - dist_occ


def normal_field_from_signed_distance(signed_distance: np.ndarray, grid_spacing: tuple[float, ...]) -> np.ndarray:
    """Compute normalized gradient of a signed distance field."""
    grads = np.gradient(signed_distance, *grid_spacing, edge_order=1)
    vec = np.stack(grads, axis=-1)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-12
    return vec / norm
