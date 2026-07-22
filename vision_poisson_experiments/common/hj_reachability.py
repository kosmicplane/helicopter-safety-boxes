"""Planar Hamilton–Jacobi/Eikonal reachability over inflated occupancy grids.

For the reduced isotropic single-integrator

    p_dot = u,  ||u||_2 <= v_max,

the obstacle-aware reachability value for landing zone ``j`` is

    V_j(p, tau) = v_max * (-tau) - D_j(p),

where ``D_j`` is the shortest metric path through free grid cells to the complete
landing-zone target disk.  This is an exact HJ/Eikonal construction for the
reduced model; it is not a full multicopter or acceleration-level HJR solution.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from cbf_safety_box import rth_largest_pivot

from .coordinates import GridGeometry
from .mission_setup import MissionDefinition, metric_disk_mask


@dataclass(frozen=True)
class ReachabilityConfig:
    """Parameters for the reduced HJ/Eikonal model."""

    enabled: bool = True
    model: str = "planar_single_integrator"
    maximum_speed_mps: float = 0.50
    active_horizon_s: float = 20.0
    contingency_horizon_s: float = 10.0
    required_reachable: int = 2
    connectivity: int = 8
    recompute_on_occupancy_update: bool = True
    maximum_field_age_s: float = 0.75
    gradient_epsilon: float = 1.0e-9
    negative_value_plot_clip: float = -5.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ReachabilityConfig":
        """Construct and validate from YAML-compatible data."""

        config = cls(**dict(data or {}))
        config.validate()
        return config

    def validate(self) -> None:
        """Validate the reduced-model assumptions and numerical parameters."""

        if self.model != "planar_single_integrator":
            raise ValueError("Only reachability.model='planar_single_integrator' is implemented.")
        if not np.isfinite(self.maximum_speed_mps) or self.maximum_speed_mps <= 0.0:
            raise ValueError("maximum_speed_mps must be positive and finite.")
        if not np.isfinite(self.active_horizon_s) or self.active_horizon_s <= 0.0:
            raise ValueError("active_horizon_s must be positive and finite.")
        if not np.isfinite(self.contingency_horizon_s) or self.contingency_horizon_s <= 0.0:
            raise ValueError("contingency_horizon_s must be positive and finite.")
        if self.required_reachable < 1:
            raise ValueError("required_reachable must be positive.")
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8.")
        if self.maximum_field_age_s <= 0.0:
            raise ValueError("maximum_field_age_s must be positive.")
        if self.gradient_epsilon <= 0.0:
            raise ValueError("gradient_epsilon must be positive.")


@dataclass(frozen=True)
class ZoneReachabilityField:
    """Obstacle-aware distance, gradient, predecessor map, and target mask."""

    zone_identifier: int
    distance_m: np.ndarray
    gradient_xy: np.ndarray
    predecessor_yx: np.ndarray
    target_seed_mask: np.ndarray
    finite_mask: np.ndarray
    available: bool
    solve_time_s: float
    validation: dict[str, Any]

    def value_field(self, tau: float, maximum_speed_mps: float) -> np.ndarray:
        """Return ``V = v_max*(-tau)-D`` with invalid cells set to ``-inf``."""

        value = float(maximum_speed_mps) * (-float(tau)) - np.asarray(self.distance_m, dtype=float)
        output = np.full(value.shape, -np.inf, dtype=float)
        output[self.finite_mask] = value[self.finite_mask]
        return output


@dataclass(frozen=True)
class ReachabilityBundle:
    """All landing-zone distance fields synchronized to one occupancy version."""

    occupancy_version: int
    created_time_s: float
    geometry: GridGeometry
    inflated_occupancy: np.ndarray
    fields: dict[int, ZoneReachabilityField]
    total_solve_time_s: float

    def field(self, identifier: int) -> ZoneReachabilityField:
        """Return one zone field by identifier."""

        return self.fields[int(identifier)]

    def available_identifiers(self) -> list[int]:
        """Return zones with finite target seeds and at least one finite distance."""

        return [identifier for identifier, field in self.fields.items() if field.available]


@dataclass(frozen=True)
class FieldSample:
    """Bilinearly sampled scalar/gradient at one metric point."""

    valid: bool
    value: float
    gradient_xy: np.ndarray
    reason: str


@dataclass(frozen=True)
class ReachabilityStatus:
    """Live scalar values, pivot, and reachable-count certificate."""

    values: dict[int, float]
    distances_m: dict[int, float]
    reachable: dict[int, bool]
    reachable_count: int
    pivot: float
    critical_identifiers: tuple[int, ...]


def _neighbors(geometry: GridGeometry, connectivity: int) -> list[tuple[int, int, float]]:
    """Return metric neighbor offsets for the configured graph connectivity."""

    neighbors = [
        (-1, 0, geometry.dy),
        (1, 0, geometry.dy),
        (0, -1, geometry.dx),
        (0, 1, geometry.dx),
    ]
    if connectivity == 8:
        diagonal = float(np.hypot(geometry.dx, geometry.dy))
        neighbors.extend(
            [(-1, -1, diagonal), (-1, 1, diagonal), (1, -1, diagonal), (1, 1, diagonal)]
        )
    return neighbors


def geodesic_distance_to_target(
    inflated_occupancy: np.ndarray,
    target_seed_mask: np.ndarray,
    geometry: GridGeometry,
    *,
    connectivity: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute multi-source Dijkstra distance and parent links toward a target.

    ``predecessor_yx[row, column]`` stores the next grid index on a shortest path
    from that cell toward the target.  Target seed cells point to themselves.
    Unreachable cells store ``(-1, -1)``.
    """

    occupied = np.asarray(inflated_occupancy, dtype=bool)
    seeds = np.asarray(target_seed_mask, dtype=bool) & ~occupied
    if occupied.shape != geometry.shape_yx or seeds.shape != geometry.shape_yx:
        raise ValueError("Occupancy and target masks must match the metric grid.")
    distance = np.full(geometry.shape_yx, np.inf, dtype=float)
    predecessor = np.full((*geometry.shape_yx, 2), -1, dtype=np.int32)
    queue: list[tuple[float, int, int]] = []
    for row, column in zip(*np.nonzero(seeds)):
        distance[row, column] = 0.0
        predecessor[row, column] = (row, column)
        heapq.heappush(queue, (0.0, int(row), int(column)))
    if not queue:
        return distance, predecessor

    neighbors = _neighbors(geometry, connectivity)
    while queue:
        current_distance, row, column = heapq.heappop(queue)
        if current_distance > distance[row, column] + 1.0e-12:
            continue
        for delta_row, delta_column, edge_cost in neighbors:
            next_row = row + delta_row
            next_column = column + delta_column
            if not (0 <= next_row < geometry.ny and 0 <= next_column < geometry.nx):
                continue
            if occupied[next_row, next_column]:
                continue
            # For diagonal moves, prevent corner cutting between two occupied cells.
            if delta_row and delta_column:
                if occupied[row, next_column] or occupied[next_row, column]:
                    continue
            candidate = current_distance + edge_cost
            if candidate + 1.0e-12 < distance[next_row, next_column]:
                distance[next_row, next_column] = candidate
                predecessor[next_row, next_column] = (row, column)
                heapq.heappush(queue, (candidate, next_row, next_column))
    return distance, predecessor


def distance_gradient_xy(distance_m: np.ndarray, geometry: GridGeometry) -> np.ndarray:
    """Return finite-difference ``[dD/dx, dD/dy]`` with NaN on invalid cells."""

    distance = np.asarray(distance_m, dtype=float)
    finite = np.isfinite(distance)
    if not np.any(finite):
        return np.full((*distance.shape, 2), np.nan, dtype=float)
    # Replace infinities only for numerical differentiation; invalid cells are
    # masked again immediately afterward.
    finite_max = float(np.max(distance[finite]))
    filled = distance.copy()
    filled[~finite] = finite_max + max(geometry.dx, geometry.dy) * 4.0
    derivative_y, derivative_x = np.gradient(filled, geometry.dy, geometry.dx, edge_order=1)
    gradient = np.stack([derivative_x, derivative_y], axis=-1)
    gradient[~finite] = np.nan
    return gradient


def eikonal_validation(
    distance_m: np.ndarray,
    gradient_xy: np.ndarray,
    target_seed_mask: np.ndarray,
    inflated_occupancy: np.ndarray,
    geometry: GridGeometry,
) -> dict[str, Any]:
    """Report ``||grad D||`` away from seeds, obstacles, and grid boundaries."""

    finite = np.isfinite(distance_m)
    mask = finite & ~np.asarray(target_seed_mask, dtype=bool) & ~np.asarray(inflated_occupancy, dtype=bool)
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        boundary = np.zeros(mask.shape, dtype=bool)
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
        mask &= ~boundary
    # Avoid immediate obstacle/target neighborhoods where finite differences and
    # the viscosity solution are expected to be nonsmooth.
    neighbor_exclusion = np.asarray(target_seed_mask, dtype=bool) | np.asarray(inflated_occupancy, dtype=bool)
    padded = np.pad(neighbor_exclusion, 1, mode="constant", constant_values=True)
    dilated = np.zeros_like(mask)
    for dr in range(3):
        for dc in range(3):
            dilated |= padded[dr : dr + mask.shape[0], dc : dc + mask.shape[1]]
    mask &= ~dilated
    norms = np.linalg.norm(np.asarray(gradient_xy, dtype=float), axis=-1)
    valid_norms = norms[mask & np.isfinite(norms)]
    if valid_norms.size == 0:
        return {
            "sample_count": 0,
            "mean_gradient_norm": None,
            "median_gradient_norm": None,
            "mean_abs_error_from_one": None,
        }
    return {
        "sample_count": int(valid_norms.size),
        "mean_gradient_norm": float(np.mean(valid_norms)),
        "median_gradient_norm": float(np.median(valid_norms)),
        "mean_abs_error_from_one": float(np.mean(np.abs(valid_norms - 1.0))),
        "p95_abs_error_from_one": float(np.percentile(np.abs(valid_norms - 1.0), 95)),
        "grid_spacing_yx": list(geometry.spacing_yx),
    }


def build_reachability_bundle(
    inflated_occupancy: np.ndarray,
    *,
    mission: MissionDefinition,
    geometry: GridGeometry,
    config: ReachabilityConfig,
    occupancy_version: int,
    unavailable_identifiers: Iterable[int] = (),
) -> ReachabilityBundle:
    """Compute every landing-zone distance field for one occupancy version."""

    config.validate()
    start_time = perf_counter()
    unavailable = {int(identifier) for identifier in unavailable_identifiers}
    fields: dict[int, ZoneReachabilityField] = {}
    for zone in mission.landing_zones:
        solve_start = perf_counter()
        seed_mask = metric_disk_mask(geometry, zone.center_xy_m, zone.radius_m)
        available = zone.identifier not in unavailable and bool(np.any(seed_mask & ~np.asarray(inflated_occupancy, dtype=bool)))
        if available:
            distance, predecessor = geodesic_distance_to_target(
                inflated_occupancy,
                seed_mask,
                geometry,
                connectivity=config.connectivity,
            )
            finite = np.isfinite(distance)
            available = bool(np.any(finite) and np.any(seed_mask & finite))
        else:
            distance = np.full(geometry.shape_yx, np.inf, dtype=float)
            predecessor = np.full((*geometry.shape_yx, 2), -1, dtype=np.int32)
            finite = np.zeros(geometry.shape_yx, dtype=bool)
        gradient = distance_gradient_xy(distance, geometry)
        validation = eikonal_validation(distance, gradient, seed_mask, inflated_occupancy, geometry)
        validation.update(
            {
                "available": bool(available),
                "finite_cells": int(np.count_nonzero(finite)),
                "target_seed_cells": int(np.count_nonzero(seed_mask & ~np.asarray(inflated_occupancy, dtype=bool))),
            }
        )
        fields[zone.identifier] = ZoneReachabilityField(
            zone_identifier=zone.identifier,
            distance_m=distance,
            gradient_xy=gradient,
            predecessor_yx=predecessor,
            target_seed_mask=seed_mask,
            finite_mask=finite,
            available=available,
            solve_time_s=perf_counter() - solve_start,
            validation=validation,
        )
    return ReachabilityBundle(
        occupancy_version=int(occupancy_version),
        created_time_s=perf_counter(),
        geometry=geometry,
        inflated_occupancy=np.asarray(inflated_occupancy, dtype=bool).copy(),
        fields=fields,
        total_solve_time_s=perf_counter() - start_time,
    )


def _bilinear_scalar(field: np.ndarray, geometry: GridGeometry, point_xy: np.ndarray) -> tuple[float, bool]:
    """Bilinearly sample a scalar field while rejecting non-finite stencils."""

    if not geometry.contains_xy(point_xy):
        return float("nan"), False
    fractional = geometry.xy_to_fractional_index_yx(point_xy)
    row = float(np.clip(fractional[0], 0.0, geometry.ny - 1.0))
    column = float(np.clip(fractional[1], 0.0, geometry.nx - 1.0))
    row0 = min(int(np.floor(row)), geometry.ny - 2)
    column0 = min(int(np.floor(column)), geometry.nx - 2)
    row1, column1 = row0 + 1, column0 + 1
    tr, tc = row - row0, column - column0
    values = np.asarray(
        [field[row0, column0], field[row0, column1], field[row1, column0], field[row1, column1]],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        nearest_row, nearest_column = geometry.nearest_index_yx(point_xy, clip=True)
        nearest = float(field[nearest_row, nearest_column])
        return nearest, bool(np.isfinite(nearest))
    weights = np.asarray([(1.0 - tr) * (1.0 - tc), (1.0 - tr) * tc, tr * (1.0 - tc), tr * tc])
    return float(weights @ values), True


def sample_zone_field(
    field: ZoneReachabilityField,
    geometry: GridGeometry,
    point_xy: Iterable[float],
    *,
    tau: float,
    maximum_speed_mps: float,
) -> FieldSample:
    """Sample one HJ value and physical gradient at a metric point."""

    point = np.asarray(point_xy, dtype=float).reshape(2)
    distance, valid_distance = _bilinear_scalar(field.distance_m, geometry, point)
    if not valid_distance:
        return FieldSample(False, float("-inf"), np.full(2, np.nan), "distance is disconnected or out of domain")
    gradient_x, valid_x = _bilinear_scalar(field.gradient_xy[..., 0], geometry, point)
    gradient_y, valid_y = _bilinear_scalar(field.gradient_xy[..., 1], geometry, point)
    if not valid_x or not valid_y:
        return FieldSample(False, float("-inf"), np.full(2, np.nan), "distance gradient is invalid")
    value = float(maximum_speed_mps) * (-float(tau)) - distance
    # V = budget - D, therefore grad V = -grad D.
    return FieldSample(True, value, -np.asarray([gradient_x, gradient_y], dtype=float), "ok")


def compute_reachability_status(
    bundle: ReachabilityBundle,
    *,
    point_xy: Iterable[float],
    tau: float,
    maximum_speed_mps: float,
    required_reachable: int,
    available_identifiers: Iterable[int] | None = None,
) -> ReachabilityStatus:
    """Compute scalar values, reachable count, and the r-th-largest pivot."""

    available = set(bundle.fields) if available_identifiers is None else {int(identifier) for identifier in available_identifiers}
    values: dict[int, float] = {}
    distances: dict[int, float] = {}
    reachable: dict[int, bool] = {}
    finite_values: list[tuple[int, float]] = []
    for identifier, field in bundle.fields.items():
        if identifier not in available or not field.available:
            values[identifier] = float("-inf")
            distances[identifier] = float("inf")
            reachable[identifier] = False
            continue
        sample = sample_zone_field(
            field,
            bundle.geometry,
            point_xy,
            tau=tau,
            maximum_speed_mps=maximum_speed_mps,
        )
        distance, valid_distance = _bilinear_scalar(field.distance_m, bundle.geometry, np.asarray(point_xy, dtype=float))
        values[identifier] = sample.value
        distances[identifier] = distance if valid_distance else float("inf")
        reachable[identifier] = bool(sample.valid and sample.value >= 0.0)
        if sample.valid:
            finite_values.append((identifier, sample.value))
    if len(finite_values) < int(required_reachable):
        pivot = float("-inf")
        critical: tuple[int, ...] = tuple()
    else:
        pivot = rth_largest_pivot((value for _, value in finite_values), int(required_reachable))
        tolerance = 1.0e-8
        critical = tuple(identifier for identifier, value in finite_values if abs(value - pivot) <= tolerance)
    return ReachabilityStatus(
        values=values,
        distances_m=distances,
        reachable=reachable,
        reachable_count=int(sum(reachable.values())),
        pivot=float(pivot),
        critical_identifiers=critical,
    )


def pivot_and_reachable_count_fields(
    bundle: ReachabilityBundle,
    *,
    tau: float,
    maximum_speed_mps: float,
    required_reachable: int,
    available_identifiers: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Return spatial HJ fields, r-th-largest pivot, and reachable-count matrix."""

    available = set(bundle.fields) if available_identifiers is None else {int(identifier) for identifier in available_identifiers}
    values: dict[int, np.ndarray] = {}
    stack: list[np.ndarray] = []
    for identifier, field in bundle.fields.items():
        value = field.value_field(tau, maximum_speed_mps)
        if identifier not in available or not field.available:
            value = np.full(value.shape, -np.inf, dtype=float)
        values[identifier] = value
        stack.append(value)
    value_stack = np.stack(stack, axis=0)
    finite_stack = np.where(np.isfinite(value_stack), value_stack, -np.inf)
    count = np.sum(finite_stack >= 0.0, axis=0).astype(np.int16)
    if len(available) < required_reachable:
        pivot = np.full(bundle.geometry.shape_yx, -np.inf, dtype=float)
    else:
        sorted_values = np.sort(finite_stack, axis=0)
        pivot = sorted_values[-int(required_reachable)]
    return pivot, count, values


def extract_path_from_predecessor(
    field: ZoneReachabilityField,
    geometry: GridGeometry,
    start_xy: Iterable[float],
    *,
    maximum_steps: int | None = None,
) -> np.ndarray:
    """Trace a shortest grid path from a metric point into the target seed."""

    start = np.asarray(start_xy, dtype=float).reshape(2)
    if not geometry.contains_xy(start):
        return np.empty((0, 2), dtype=float)
    row, column = geometry.nearest_index_yx(start, clip=True)
    if not field.finite_mask[row, column]:
        return np.empty((0, 2), dtype=float)
    path_indices: list[tuple[int, int]] = [(row, column)]
    visited = {(row, column)}
    limit = int(maximum_steps or (geometry.nx * geometry.ny + 1))
    for _ in range(limit):
        if field.target_seed_mask[row, column]:
            break
        next_row, next_column = (int(value) for value in field.predecessor_yx[row, column])
        if next_row < 0 or next_column < 0:
            return np.empty((0, 2), dtype=float)
        if (next_row, next_column) == (row, column):
            break
        if (next_row, next_column) in visited:
            return np.empty((0, 2), dtype=float)
        visited.add((next_row, next_column))
        path_indices.append((next_row, next_column))
        row, column = next_row, next_column
    if not field.target_seed_mask[row, column]:
        return np.empty((0, 2), dtype=float)
    return np.asarray([geometry.index_to_xy(row, column) for row, column in path_indices], dtype=float)


def segment_is_collision_free(
    start_xy: Iterable[float],
    end_xy: Iterable[float],
    inflated_occupancy: np.ndarray,
    geometry: GridGeometry,
    *,
    sampling_step_m: float | None = None,
) -> bool:
    """Check a metric line segment against inflated occupancy."""

    start = np.asarray(start_xy, dtype=float).reshape(2)
    end = np.asarray(end_xy, dtype=float).reshape(2)
    length = float(np.linalg.norm(end - start))
    step = float(sampling_step_m or 0.4 * min(geometry.dx, geometry.dy))
    sample_count = max(2, int(np.ceil(length / max(step, 1.0e-9))) + 1)
    for fraction in np.linspace(0.0, 1.0, sample_count):
        point = (1.0 - fraction) * start + fraction * end
        if not geometry.contains_xy(point):
            return False
        row, column = geometry.nearest_index_yx(point, clip=True)
        if bool(np.asarray(inflated_occupancy, dtype=bool)[row, column]):
            return False
    return True


def simplify_path_line_of_sight(
    path_xy: np.ndarray,
    inflated_occupancy: np.ndarray,
    geometry: GridGeometry,
) -> np.ndarray:
    """Greedily simplify a path while collision-checking every retained segment."""

    path = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    if path.shape[0] <= 2:
        return path.copy()
    simplified = [path[0]]
    anchor = 0
    while anchor < path.shape[0] - 1:
        furthest = anchor + 1
        for candidate in range(anchor + 2, path.shape[0]):
            if segment_is_collision_free(path[anchor], path[candidate], inflated_occupancy, geometry):
                furthest = candidate
            else:
                break
        simplified.append(path[furthest])
        anchor = furthest
    return np.asarray(simplified, dtype=float)


def path_is_collision_free(path_xy: np.ndarray, inflated_occupancy: np.ndarray, geometry: GridGeometry) -> bool:
    """Return whether every segment in a metric path is safe."""

    path = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    if path.shape[0] == 0:
        return False
    return all(
        segment_is_collision_free(path[index], path[index + 1], inflated_occupancy, geometry)
        for index in range(path.shape[0] - 1)
    )


__all__ = [
    "FieldSample",
    "ReachabilityBundle",
    "ReachabilityConfig",
    "ReachabilityStatus",
    "ZoneReachabilityField",
    "build_reachability_bundle",
    "compute_reachability_status",
    "distance_gradient_xy",
    "eikonal_validation",
    "extract_path_from_predecessor",
    "geodesic_distance_to_target",
    "path_is_collision_free",
    "pivot_and_reachable_count_fields",
    "sample_zone_field",
    "segment_is_collision_free",
    "simplify_path_line_of_sight",
]
