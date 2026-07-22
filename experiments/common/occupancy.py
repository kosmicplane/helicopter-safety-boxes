"""Metric occupancy construction, conservative inflation, and temporal filtering.

The boolean convention is intentionally strict throughout this repository:

``True``
    occupied / unsafe cell;
``False``
    free cell.

Only boolean arrays are sent to :class:`poisson_safety_box.PoissonSafetyBox`.
This module also keeps the uninflated perception product separate from the
configuration-space occupancy used for Poisson synthesis.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .coordinates import GridGeometry
from .segmentation import ensure_binary_mask


@dataclass(frozen=True)
class OccupancyMaps:
    """Uninflated and inflated occupancy grids with metric metadata.

    Parameters
    ----------
    occupancy:
        Boolean obstacle occupancy before robot-footprint inflation.
    inflated_occupancy:
        Boolean configuration-space occupancy.  Every ``True`` entry in
        ``occupancy`` must also be ``True`` here.
    grid_spacing_yx:
        Node spacing ``(dy, dx)`` in meters.
    inflation_radius_m:
        Total physical radius ``robot_radius + perception_margin``.
    metadata:
        Human-readable cell counts and configuration diagnostics.
    """

    occupancy: np.ndarray
    inflated_occupancy: np.ndarray
    grid_spacing_yx: tuple[float, float]
    inflation_radius_m: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        original = np.asarray(self.occupancy, dtype=bool)
        inflated = np.asarray(self.inflated_occupancy, dtype=bool)
        if original.ndim != 2:
            raise ValueError("Occupancy grids must be two-dimensional.")
        if inflated.shape != original.shape:
            raise ValueError("Original and inflated occupancy must have matching shapes.")
        if np.any(original & ~inflated):
            raise ValueError("Inflated occupancy must contain every original occupied cell.")
        dy, dx = float(self.grid_spacing_yx[0]), float(self.grid_spacing_yx[1])
        if dy <= 0.0 or dx <= 0.0:
            raise ValueError("Grid spacing must be positive.")
        if float(self.inflation_radius_m) < 0.0:
            raise ValueError("Inflation radius cannot be negative.")
        object.__setattr__(self, "occupancy", original)
        object.__setattr__(self, "inflated_occupancy", inflated)
        object.__setattr__(self, "grid_spacing_yx", (dy, dx))
        object.__setattr__(self, "inflation_radius_m", float(self.inflation_radius_m))

    @property
    def uninflated(self) -> np.ndarray:
        """Readable alias for the original perception occupancy."""

        return self.occupancy

    @property
    def inflated(self) -> np.ndarray:
        """Readable alias for the configuration-space occupancy."""

        return self.inflated_occupancy

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Readable alias for occupancy metadata."""

        return self.metadata

    @property
    def inflation_radius_cells_xy(self) -> tuple[int, int]:
        """Return the conservative discrete radii ``(radius_x, radius_y)``."""

        return (
            int(self.metadata.get("inflation_radius_x_cells", 0)),
            int(self.metadata.get("inflation_radius_y_cells", 0)),
        )


# Both names intentionally describe the same data contract.
OccupancyProducts = OccupancyMaps


def grid_spacing_from_workspace(
    workspace_size_m: tuple[float, float],
    grid_shape_yx: tuple[int, int],
) -> tuple[float, float]:
    """Compute node-centered spacing ``(dy, dx)``.

    The grid includes both workspace edges.  Therefore ``dx = W/(Nx-1)`` and
    ``dy = H/(Ny-1)`` rather than dividing by the number of cells.
    """

    width_m, height_m = float(workspace_size_m[0]), float(workspace_size_m[1])
    ny, nx = int(grid_shape_yx[0]), int(grid_shape_yx[1])
    if width_m <= 0.0 or height_m <= 0.0:
        raise ValueError("Workspace dimensions must be positive.")
    if ny < 2 or nx < 2:
        raise ValueError("Grid dimensions must each be at least two.")
    return height_m / float(ny - 1), width_m / float(nx - 1)


def _shape_from_geometry_or_tuple(
    geometry_or_shape: GridGeometry | tuple[int, int],
) -> tuple[int, int]:
    """Normalize a geometry object or explicit ``(Ny, Nx)`` shape."""

    if isinstance(geometry_or_shape, GridGeometry):
        return geometry_or_shape.shape_yx
    ny, nx = int(geometry_or_shape[0]), int(geometry_or_shape[1])
    return ny, nx


def mask_to_occupancy(
    mask: np.ndarray,
    geometry_or_shape: GridGeometry | tuple[int, int],
) -> np.ndarray:
    """Resize a binary obstacle mask into a boolean occupancy grid.

    Nearest-neighbor interpolation is mandatory: linear interpolation would
    create fractional boundary pixels and can alter obstacle topology.
    """

    binary = ensure_binary_mask(mask)
    ny, nx = _shape_from_geometry_or_tuple(geometry_or_shape)
    if ny < 2 or nx < 2:
        raise ValueError("Occupancy dimensions must each be at least two.")
    resized = cv2.resize(binary, (nx, ny), interpolation=cv2.INTER_NEAREST)
    return np.asarray(resized > 0, dtype=bool)


def _conservative_ellipse_kernel(radius_x_cells: int, radius_y_cells: int) -> np.ndarray:
    """Build an anisotropic ellipse that reaches both requested axis radii."""

    radius_x = max(0, int(radius_x_cells))
    radius_y = max(0, int(radius_y_cells))
    if radius_x == 0 and radius_y == 0:
        return np.ones((1, 1), dtype=np.uint8)

    yy, xx = np.mgrid[-radius_y : radius_y + 1, -radius_x : radius_x + 1]
    x_term = np.zeros_like(xx, dtype=float) if radius_x == 0 else (xx / float(radius_x)) ** 2
    y_term = np.zeros_like(yy, dtype=float) if radius_y == 0 else (yy / float(radius_y)) ** 2
    kernel = (x_term + y_term <= 1.0 + 1.0e-12).astype(np.uint8)

    # Explicitly preserve the requested conservative reach along both axes.
    kernel[radius_y, 0] = 1
    kernel[radius_y, -1] = 1
    kernel[0, radius_x] = 1
    kernel[-1, radius_x] = 1
    return kernel


def inflate_occupancy(
    occupancy: np.ndarray,
    geometry: GridGeometry,
    radius_m: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Inflate occupancy by a physical radius and report discrete radii.

    The conversion uses ``ceil(radius / spacing)`` so a partially intersected
    cell is conservatively treated as occupied.  The returned cell radii are in
    controller-friendly order ``(radius_x, radius_y)``.
    """

    occupied = np.asarray(occupancy, dtype=bool)
    if occupied.ndim != 2 or occupied.shape != geometry.shape_yx:
        raise ValueError("Occupancy shape must match GridGeometry.shape_yx.")
    radius = float(radius_m)
    if radius < 0.0:
        raise ValueError("Inflation radius cannot be negative.")

    radius_x = int(np.ceil(radius / geometry.dx - 1.0e-14)) if radius > 0.0 else 0
    radius_y = int(np.ceil(radius / geometry.dy - 1.0e-14)) if radius > 0.0 else 0
    if radius == 0.0 or not np.any(occupied):
        return occupied.copy(), (radius_x, radius_y)

    kernel = _conservative_ellipse_kernel(radius_x, radius_y)
    inflated = cv2.dilate(occupied.astype(np.uint8), kernel, iterations=1) > 0
    return np.asarray(inflated | occupied, dtype=bool), (radius_x, radius_y)


def inflate_occupancy_physical(
    occupancy: np.ndarray,
    radius_m: float,
    grid_spacing_yx: tuple[float, float],
) -> np.ndarray:
    """Inflate a grid when only its spacing is available."""

    occupied = np.asarray(occupancy, dtype=bool)
    geometry = GridGeometry(shape_yx=occupied.shape, spacing_yx=grid_spacing_yx)
    inflated, _radii = inflate_occupancy(occupied, geometry, radius_m)
    return inflated


def inflate_mask_physical(
    mask: np.ndarray,
    radius_m: float,
    workspace_size_m: tuple[float, float],
) -> np.ndarray:
    """Inflate a rectified pixel mask using metric anisotropic pixel spacing."""

    binary = ensure_binary_mask(mask) > 0
    height_px, width_px = binary.shape
    width_m, height_m = float(workspace_size_m[0]), float(workspace_size_m[1])
    geometry = GridGeometry(width_m=width_m, height_m=height_m, nx=width_px, ny=height_px)
    inflated, _radii = inflate_occupancy(binary, geometry, radius_m)
    return inflated.astype(np.uint8) * 255


def compute_occupancy_products(
    clean_mask: np.ndarray,
    geometry: GridGeometry,
    *,
    robot_radius_m: float,
    perception_margin_m: float,
) -> OccupancyProducts:
    """Build uninflated and configuration-space occupancy products."""

    robot_radius = float(robot_radius_m)
    margin = float(perception_margin_m)
    if robot_radius < 0.0 or margin < 0.0:
        raise ValueError("Robot radius and perception margin must be nonnegative.")

    total_radius = robot_radius + margin
    uninflated = mask_to_occupancy(clean_mask, geometry)
    inflated, (radius_x, radius_y) = inflate_occupancy(uninflated, geometry, total_radius)
    metadata = {
        "convention": "True means occupied; False means free",
        "occupied_cells_uninflated": int(np.count_nonzero(uninflated)),
        "occupied_cells_inflated": int(np.count_nonzero(inflated)),
        "occupied_fraction_uninflated": float(np.mean(uninflated)),
        "occupied_fraction_inflated": float(np.mean(inflated)),
        "inflation_radius_m": total_radius,
        "inflation_radius_x_cells": radius_x,
        "inflation_radius_y_cells": radius_y,
        # Stable compatibility names used by early notebooks and dashboards.
        "occupied_cells": int(np.count_nonzero(uninflated)),
        "inflated_occupied_cells": int(np.count_nonzero(inflated)),
        "occupied_fraction": float(np.mean(uninflated)),
        "inflated_occupied_fraction": float(np.mean(inflated)),
    }
    return OccupancyProducts(
        occupancy=uninflated,
        inflated_occupancy=inflated,
        grid_spacing_yx=geometry.spacing_yx,
        inflation_radius_m=total_radius,
        metadata=metadata,
    )


def build_occupancy_maps(
    clean_mask: np.ndarray,
    *,
    grid_shape_yx: tuple[int, int],
    workspace_size_m: tuple[float, float],
    robot_radius_m: float,
    perception_margin_m: float,
) -> OccupancyMaps:
    """Construct occupancy directly from workspace and grid dimensions."""

    width_m, height_m = float(workspace_size_m[0]), float(workspace_size_m[1])
    ny, nx = int(grid_shape_yx[0]), int(grid_shape_yx[1])
    geometry = GridGeometry(width_m=width_m, height_m=height_m, nx=nx, ny=ny)
    return compute_occupancy_products(
        clean_mask,
        geometry,
        robot_radius_m=robot_radius_m,
        perception_margin_m=perception_margin_m,
    )


def changed_fraction(previous: np.ndarray | None, current: np.ndarray) -> float:
    """Return the fraction of cells changed between two boolean grids."""

    current_bool = np.asarray(current, dtype=bool)
    if previous is None:
        return 1.0
    previous_bool = np.asarray(previous, dtype=bool)
    if previous_bool.shape != current_bool.shape:
        return 1.0
    return float(np.mean(previous_bool != current_bool))


def intersection_over_union(first: np.ndarray, second: np.ndarray) -> float:
    """Return binary intersection-over-union, including empty-empty = one."""

    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    if first_bool.shape != second_bool.shape:
        raise ValueError("IoU inputs must have the same shape.")
    union = int(np.count_nonzero(first_bool | second_bool))
    return 1.0 if union == 0 else float(np.count_nonzero(first_bool & second_bool) / union)


class TemporalOccupancyFilter:
    """Filter occupancy over time without adapting away stationary obstacles.

    Supported modes are:

    ``none``
        Pass each grid through unchanged.
    ``majority``
        Majority vote over a bounded recent window.
    ``ema``
        Exponential moving average followed by a threshold.
    ``hysteresis``
        Separate activation and deactivation frame counts.
    """

    def __init__(
        self,
        mode: str = "majority",
        *,
        window_size: int = 5,
        ema_alpha: float = 0.35,
        ema_threshold: float = 0.5,
        activation_frames: int = 2,
        deactivation_frames: int = 4,
    ) -> None:
        if mode not in {"none", "majority", "ema", "hysteresis"}:
            raise ValueError(f"Unsupported temporal filter mode: {mode!r}")
        self.mode = mode
        self.window_size = max(1, int(window_size))
        self.ema_alpha = float(ema_alpha)
        self.ema_threshold = float(ema_threshold)
        self.activation_frames = max(1, int(activation_frames))
        self.deactivation_frames = max(1, int(deactivation_frames))
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("EMA alpha must be in (0, 1].")
        if not 0.0 <= self.ema_threshold <= 1.0:
            raise ValueError("EMA threshold must be in [0, 1].")

        self._history: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._ema: np.ndarray | None = None
        self._state: np.ndarray | None = None
        self._on_count: np.ndarray | None = None
        self._off_count: np.ndarray | None = None

    def reset(self) -> None:
        """Clear every temporal state array."""

        self._history.clear()
        self._ema = None
        self._state = None
        self._on_count = None
        self._off_count = None

    def update(self, occupancy: np.ndarray) -> np.ndarray:
        """Filter one boolean occupancy grid and return a new array."""

        current = np.asarray(occupancy, dtype=bool)
        if current.ndim != 2:
            raise ValueError("Temporal occupancy filtering expects a 2D grid.")

        if self.mode == "none":
            return current.copy()

        if self.mode == "majority":
            self._history.append(current.copy())
            stacked = np.stack(tuple(self._history), axis=0)
            threshold = stacked.shape[0] // 2 + 1
            return np.sum(stacked, axis=0) >= threshold

        if self.mode == "ema":
            if self._ema is None or self._ema.shape != current.shape:
                self._ema = current.astype(float)
            else:
                self._ema = self.ema_alpha * current + (1.0 - self.ema_alpha) * self._ema
            return self._ema >= self.ema_threshold

        # Hysteresis mode.
        if self._state is None or self._state.shape != current.shape:
            self._state = current.copy()
            self._on_count = np.zeros(current.shape, dtype=np.int16)
            self._off_count = np.zeros(current.shape, dtype=np.int16)
            return self._state.copy()

        assert self._on_count is not None and self._off_count is not None
        self._on_count[current] += 1
        self._on_count[~current] = 0
        self._off_count[~current] += 1
        self._off_count[current] = 0
        self._state[self._on_count >= self.activation_frames] = True
        self._state[self._off_count >= self.deactivation_frames] = False
        return self._state.copy()


# Descriptive alias retained for compatibility with early experiment drafts.
build_occupancy_products = compute_occupancy_products


__all__ = [
    "OccupancyMaps",
    "OccupancyProducts",
    "TemporalOccupancyFilter",
    "build_occupancy_maps",
    "build_occupancy_products",
    "changed_fraction",
    "compute_occupancy_products",
    "grid_spacing_from_workspace",
    "inflate_mask_physical",
    "inflate_occupancy",
    "inflate_occupancy_physical",
    "intersection_over_union",
    "mask_to_occupancy",
]
