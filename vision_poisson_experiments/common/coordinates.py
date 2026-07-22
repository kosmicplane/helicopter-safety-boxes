"""Coordinate conventions and bilinear sampling for two-dimensional safety fields.

This module is intentionally the *only* place where array coordinates and physical
coordinates are exchanged.  OpenCV and NumPy index images as ``(row, column) =
(y, x)``, while the controller uses vectors in ``[x, y]`` order.  Centralizing the
conversion prevents silent transposition errors in gradients and Hessians.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from poisson_safety_box.interpolation import interpolate_grid


AXIS_PERMUTATION_2D = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)


@dataclass(frozen=True, init=False)
class GridGeometry:
    """Metric description of a node-centered row-major grid.

    Two construction styles are supported because they are useful in different
    layers of the project::

        GridGeometry(shape_yx=(Ny, Nx), spacing_yx=(dy, dx))
        GridGeometry(width_m=W, height_m=H, nx=Nx, ny=Ny)

    In both cases the grid includes both workspace edges, so ``dx = W/(Nx-1)``
    and ``dy = H/(Ny-1)``.  The physical frame has its origin at the rectified
    image's top-left corner, +x to the right, and +y downward.
    """

    shape_yx: tuple[int, int]
    spacing_yx: tuple[float, float]
    origin_yx: tuple[float, float]

    def __init__(
        self,
        shape_yx: tuple[int, int] | None = None,
        spacing_yx: tuple[float, float] | None = None,
        origin_yx: tuple[float, float] = (0.0, 0.0),
        *,
        width_m: float | None = None,
        height_m: float | None = None,
        nx: int | None = None,
        ny: int | None = None,
        workspace_width_m: float | None = None,
        workspace_height_m: float | None = None,
    ) -> None:
        """Validate either shape/spacing or metric-size/grid-count arguments."""

        width = width_m if width_m is not None else workspace_width_m
        height = height_m if height_m is not None else workspace_height_m

        metric_style_requested = any(value is not None for value in (width, height, nx, ny))
        array_style_requested = shape_yx is not None or spacing_yx is not None
        if metric_style_requested and array_style_requested:
            raise ValueError("Use either shape_yx/spacing_yx or width_m/height_m/nx/ny, not both.")

        if metric_style_requested:
            if None in (width, height, nx, ny):
                raise ValueError("Metric construction requires width_m, height_m, nx, and ny.")
            nx_value = int(nx)  # type: ignore[arg-type]
            ny_value = int(ny)  # type: ignore[arg-type]
            width_value = float(width)  # type: ignore[arg-type]
            height_value = float(height)  # type: ignore[arg-type]
            if nx_value < 2 or ny_value < 2:
                raise ValueError("Grid dimensions must each be at least two.")
            if width_value <= 0.0 or height_value <= 0.0:
                raise ValueError("Workspace width and height must be positive.")
            shape = (ny_value, nx_value)
            spacing = (height_value / (ny_value - 1), width_value / (nx_value - 1))
        else:
            if shape_yx is None or spacing_yx is None:
                raise ValueError("Array construction requires shape_yx and spacing_yx.")
            shape = (int(shape_yx[0]), int(shape_yx[1]))
            spacing = (float(spacing_yx[0]), float(spacing_yx[1]))
            if shape[0] < 2 or shape[1] < 2:
                raise ValueError("Grid dimensions must each be at least two.")
            if spacing[0] <= 0.0 or spacing[1] <= 0.0:
                raise ValueError("Grid spacing must be positive.")

        origin = (float(origin_yx[0]), float(origin_yx[1]))
        object.__setattr__(self, "shape_yx", shape)
        object.__setattr__(self, "spacing_yx", spacing)
        object.__setattr__(self, "origin_yx", origin)

    @property
    def ny(self) -> int:
        """Number of grid nodes along image rows / physical y."""

        return self.shape_yx[0]

    @property
    def nx(self) -> int:
        """Number of grid nodes along image columns / physical x."""

        return self.shape_yx[1]

    @property
    def dy(self) -> float:
        """Physical row spacing in meters."""

        return self.spacing_yx[0]

    @property
    def dx(self) -> float:
        """Physical column spacing in meters."""

        return self.spacing_yx[1]

    @property
    def extent_xy(self) -> tuple[float, float, float, float]:
        """Return ``(x_min, x_max, y_min, y_max)`` in meters."""

        y0, x0 = self.origin_yx
        return x0, x0 + (self.nx - 1) * self.dx, y0, y0 + (self.ny - 1) * self.dy

    @property
    def workspace_size_m(self) -> tuple[float, float]:
        """Return physical ``(width, height)`` represented by the grid."""

        x_min, x_max, y_min, y_max = self.extent_xy
        return x_max - x_min, y_max - y_min

    @property
    def width_m(self) -> float:
        """Workspace width in meters."""

        return self.workspace_size_m[0]

    @property
    def height_m(self) -> float:
        """Workspace height in meters."""

        return self.workspace_size_m[1]

    @property
    def workspace_width_m(self) -> float:
        """Backward-compatible alias for :attr:`width_m`."""

        return self.width_m

    @property
    def workspace_height_m(self) -> float:
        """Backward-compatible alias for :attr:`height_m`."""

        return self.height_m

    def xy_to_yx(self, point_xy: np.ndarray | tuple[float, float]) -> np.ndarray:
        """Convert a physical ``[x, y]`` point to Poisson ``[y, x]`` order."""

        return xy_to_yx(point_xy)

    def yx_to_xy(self, point_yx: np.ndarray | tuple[float, float]) -> np.ndarray:
        """Convert a Poisson ``[y, x]`` point to physical ``[x, y]`` order."""

        return yx_to_xy(point_yx)

    def xy_to_fractional_index_yx(self, point_xy: np.ndarray | tuple[float, float]) -> np.ndarray:
        """Return floating-point ``(row, column)`` indices for a physical point."""

        point_yx = self.xy_to_yx(point_xy)
        return (point_yx - np.asarray(self.origin_yx)) / np.asarray(self.spacing_yx)

    def nearest_index_yx(
        self,
        point_xy: np.ndarray | tuple[float, float],
        *,
        clip: bool = False,
    ) -> tuple[int, int]:
        """Return the nearest integer ``(row, column)`` for a physical point."""

        index = np.rint(self.xy_to_fractional_index_yx(point_xy)).astype(int)
        if clip:
            index = np.clip(index, [0, 0], np.asarray(self.shape_yx) - 1)
        return int(index[0]), int(index[1])

    def index_to_xy(self, row: int, column: int) -> np.ndarray:
        """Convert one integer array index into a physical ``[x, y]`` point."""

        y0, x0 = self.origin_yx
        return np.asarray([x0 + int(column) * self.dx, y0 + int(row) * self.dy], dtype=float)

    def contains_xy(
        self,
        point_xy: np.ndarray | tuple[float, float],
        *,
        tolerance: float = 1.0e-12,
    ) -> bool:
        """Return whether a point lies inside the closed physical grid extent."""

        x, y = np.asarray(point_xy, dtype=float).reshape(2)
        x_min, x_max, y_min, y_max = self.extent_xy
        return (
            x_min - tolerance <= x <= x_max + tolerance
            and y_min - tolerance <= y <= y_max + tolerance
        )

    def supports_bilinear_xy(self, point_xy: np.ndarray | tuple[float, float]) -> bool:
        """Return whether bilinear sampling can be evaluated at the point."""

        return self.contains_xy(point_xy)


def xy_to_yx(point_xy: np.ndarray | tuple[float, float]) -> np.ndarray:
    """Convert ``[x, y]`` to ``[y, x]`` without requiring a geometry object."""

    point = np.asarray(point_xy, dtype=float).reshape(2)
    return point[[1, 0]]


def yx_to_xy(point_yx: np.ndarray | tuple[float, float]) -> np.ndarray:
    """Convert ``[y, x]`` to ``[x, y]`` without requiring a geometry object."""

    point = np.asarray(point_yx, dtype=float).reshape(2)
    return point[[1, 0]]


# Descriptive alias retained for callers that prefer the longer name.
point_xy_to_yx = xy_to_yx


def gradient_yx_to_xy(gradient_yx: np.ndarray) -> np.ndarray:
    """Convert ``[dh/dy, dh/dx]`` to physical ``[dh/dx, dh/dy]``."""

    gradient = np.asarray(gradient_yx, dtype=float).reshape(2)
    return gradient[[1, 0]]


def gradient_xy_to_yx(gradient_xy: np.ndarray) -> np.ndarray:
    """Convert physical ``[dh/dx, dh/dy]`` to ``[dh/dy, dh/dx]``."""

    gradient = np.asarray(gradient_xy, dtype=float).reshape(2)
    return gradient[[1, 0]]


def hessian_yx_to_xy(hessian_yx: np.ndarray) -> np.ndarray:
    """Permute a 2D Hessian from array-axis order into physical-axis order."""

    hessian = np.asarray(hessian_yx, dtype=float).reshape(2, 2)
    return AXIS_PERMUTATION_2D @ hessian @ AXIS_PERMUTATION_2D.T


def hessian_xy_to_yx(hessian_xy: np.ndarray) -> np.ndarray:
    """Permute a 2D Hessian from physical-axis order into array-axis order."""

    hessian = np.asarray(hessian_xy, dtype=float).reshape(2, 2)
    return AXIS_PERMUTATION_2D.T @ hessian @ AXIS_PERMUTATION_2D


def _interpolate_inclusive(
    data: np.ndarray,
    point_yx: np.ndarray,
    geometry: GridGeometry,
) -> tuple[np.ndarray | float | None, bool]:
    """Use the Safety Box interpolator while supporting exact upper-grid edges."""

    value, valid = interpolate_grid(
        data,
        point_yx,
        geometry.spacing_yx,
        origin=geometry.origin_yx,
    )
    if valid:
        return value, True
    if not geometry.contains_xy(geometry.yx_to_xy(point_yx)):
        return None, False

    # The external interpolator rejects a point exactly on the final row/column
    # because the bilinear stencil would require ``lo + 1``.  A one-ULP inward
    # shift evaluates the continuous limiting value without changing interior
    # sampling behavior.
    y0, x0 = geometry.origin_yx
    maximum = np.asarray(
        [y0 + (geometry.ny - 1) * geometry.dy, x0 + (geometry.nx - 1) * geometry.dx],
        dtype=float,
    )
    adjusted = np.minimum(np.asarray(point_yx, dtype=float), np.nextafter(maximum, -np.inf))
    return interpolate_grid(data, adjusted, geometry.spacing_yx, origin=geometry.origin_yx)


@dataclass(frozen=True)
class FieldSample:
    """Bilinearly sampled Poisson values in physical ``(x, y)`` order."""

    valid: bool
    h: float | None = None
    gradient_xy: np.ndarray | None = None
    hessian_xy: np.ndarray | None = None
    laplacian: float | None = None
    point_xy: np.ndarray | None = None
    nearest_index_yx: tuple[int, int] | None = None
    reason: str = "ok"
    metadata: dict[str, Any] | None = None

    @property
    def grad_xy(self) -> np.ndarray | None:
        """Compact alias used by the CBF experiment and tests."""

        return self.gradient_xy


class GridFieldSampler:
    """Sample a Poisson result with explicit coordinate and derivative conversion.

    Preferred construction passes a ``PoissonBoxResult`` followed by geometry::

        GridFieldSampler(poisson_result, geometry)

    A field-array construction is also accepted for analytical tests::

        GridFieldSampler(geometry, h, grad_yx, hessian_yx, laplacian)
    """

    def __init__(self, *args: Any, reject_occupied: bool = True) -> None:
        if len(args) >= 2 and isinstance(args[0], GridGeometry):
            geometry = args[0]
            h = np.asarray(args[1], dtype=float)
            grad_h = np.asarray(args[2], dtype=float) if len(args) > 2 and args[2] is not None else None
            hessian_h = np.asarray(args[3], dtype=float) if len(args) > 3 and args[3] is not None else None
            laplacian_h = np.asarray(args[4], dtype=float) if len(args) > 4 and args[4] is not None else None
            if grad_h is None:
                raise ValueError("GridFieldSampler requires a gradient field.")
            result = SimpleNamespace(
                h=h,
                grad_h=grad_h,
                hessian_h=hessian_h,
                laplacian_h=laplacian_h,
                occupancy_mask=np.zeros(h.shape, dtype=bool),
                free_mask=np.ones(h.shape, dtype=bool),
            )
        elif len(args) >= 2 and isinstance(args[1], GridGeometry):
            result = args[0]
            geometry = args[1]
        else:
            raise TypeError(
                "Use GridFieldSampler(poisson_result, geometry) or "
                "GridFieldSampler(geometry, h, grad_yx, hessian_yx, laplacian)."
            )

        self.result = result
        self.geometry = geometry
        self.reject_occupied = bool(reject_occupied)
        if np.asarray(self.result.h).shape != geometry.shape_yx:
            raise ValueError(
                f"Poisson field shape {np.asarray(self.result.h).shape} does not match "
                f"geometry shape {geometry.shape_yx}."
            )
        if getattr(self.result, "grad_h", None) is None:
            raise ValueError("GridFieldSampler requires result.grad_h.")

    def occupancy_at_xy(self, point_xy: np.ndarray | tuple[float, float]) -> bool:
        """Return nearest-neighbor occupancy membership for a physical point."""

        if not self.geometry.contains_xy(point_xy):
            return True
        index = self.geometry.nearest_index_yx(point_xy, clip=True)
        occupancy = getattr(self.result, "occupancy_mask", np.zeros(self.geometry.shape_yx, dtype=bool))
        return bool(np.asarray(occupancy, dtype=bool)[index])

    def free_at_xy(self, point_xy: np.ndarray | tuple[float, float]) -> bool:
        """Return nearest-neighbor free-domain membership for a physical point."""

        if not self.geometry.contains_xy(point_xy):
            return False
        index = self.geometry.nearest_index_yx(point_xy, clip=True)
        free = getattr(self.result, "free_mask", np.ones(self.geometry.shape_yx, dtype=bool))
        return bool(np.asarray(free, dtype=bool)[index])

    def sample(self, point_xy: np.ndarray | tuple[float, float]) -> FieldSample:
        """Bilinearly interpolate ``h`` and its available derivatives."""

        point = np.asarray(point_xy, dtype=float).reshape(2)
        if not np.all(np.isfinite(point)):
            return FieldSample(False, point_xy=point, reason="nonfinite_point")
        if not self.geometry.supports_bilinear_xy(point):
            return FieldSample(False, point_xy=point, reason="outside_grid")
        nearest = self.geometry.nearest_index_yx(point, clip=True)
        if self.reject_occupied and self.occupancy_at_xy(point):
            return FieldSample(False, point_xy=point, nearest_index_yx=nearest, reason="occupied_cell")

        point_yx = self.geometry.xy_to_yx(point)
        h_value, h_valid = _interpolate_inclusive(np.asarray(self.result.h), point_yx, self.geometry)
        gradient_yx, gradient_valid = _interpolate_inclusive(
            np.asarray(self.result.grad_h), point_yx, self.geometry
        )
        if not h_valid or not gradient_valid or h_value is None or gradient_yx is None:
            return FieldSample(False, point_xy=point, nearest_index_yx=nearest, reason="interpolation_failed")

        gradient_xy = gradient_yx_to_xy(np.asarray(gradient_yx))
        hessian_xy = None
        hessian_field = getattr(self.result, "hessian_h", None)
        if hessian_field is not None:
            hessian_yx, valid = _interpolate_inclusive(np.asarray(hessian_field), point_yx, self.geometry)
            if not valid or hessian_yx is None:
                return FieldSample(False, point_xy=point, nearest_index_yx=nearest, reason="hessian_failed")
            hessian_xy = hessian_yx_to_xy(np.asarray(hessian_yx))

        laplacian = None
        laplacian_field = getattr(self.result, "laplacian_h", None)
        if laplacian_field is not None:
            value, valid = _interpolate_inclusive(np.asarray(laplacian_field), point_yx, self.geometry)
            if valid and value is not None:
                laplacian = float(value)

        h_scalar = float(h_value)
        if not np.isfinite(h_scalar) or not np.all(np.isfinite(gradient_xy)):
            return FieldSample(False, point_xy=point, nearest_index_yx=nearest, reason="nonfinite_field")
        if hessian_xy is not None and not np.all(np.isfinite(hessian_xy)):
            return FieldSample(False, point_xy=point, nearest_index_yx=nearest, reason="nonfinite_hessian")

        return FieldSample(
            valid=True,
            h=h_scalar,
            gradient_xy=gradient_xy,
            hessian_xy=hessian_xy,
            laplacian=laplacian,
            point_xy=point,
            nearest_index_yx=nearest,
            reason="ok",
            metadata={"point_yx": point_yx.tolist(), "sampling": "bilinear"},
        )

    def sample_xy(self, point_xy: np.ndarray | tuple[float, float]) -> FieldSample:
        """Readable alias for :meth:`sample`."""

        return self.sample(point_xy)

    def h_at_xy(self, point_xy: np.ndarray | tuple[float, float]) -> float:
        """Return only ``h``, raising a descriptive error when sampling fails."""

        sample = self.sample(point_xy)
        if not sample.valid or sample.h is None:
            raise ValueError(f"Cannot sample h at {np.asarray(point_xy)}: {sample.reason}")
        return sample.h

    def directional_derivative_check(
        self,
        point_xy: np.ndarray | tuple[float, float],
        direction_xy: np.ndarray | tuple[float, float],
        *,
        epsilon_m: float = 1.0e-4,
    ) -> dict[str, float]:
        """Compare the interpolated gradient with a centered finite difference."""

        point = np.asarray(point_xy, dtype=float).reshape(2)
        direction = np.asarray(direction_xy, dtype=float).reshape(2)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("Directional derivative check requires a nonzero direction.")
        direction /= norm
        center = self.sample(point)
        plus = self.sample(point + epsilon_m * direction)
        minus = self.sample(point - epsilon_m * direction)
        if not center.valid or center.gradient_xy is None or not plus.valid or not minus.valid:
            raise ValueError("Directional derivative check points must all be valid field samples.")
        assert plus.h is not None and minus.h is not None
        numerical = float((plus.h - minus.h) / (2.0 * epsilon_m))
        gradient_value = float(center.gradient_xy @ direction)
        return {
            "numerical": numerical,
            "gradient": gradient_value,
            "absolute_error": abs(numerical - gradient_value),
            "relative_error": abs(numerical - gradient_value) / max(1.0e-12, abs(numerical)),
        }


PoissonFieldSampler = GridFieldSampler
