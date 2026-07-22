"""Obstacle-aware path extraction and pure-pursuit velocity guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .hj_reachability import (
    ReachabilityBundle,
    extract_path_from_predecessor,
    path_is_collision_free,
    simplify_path_line_of_sight,
)


@dataclass(frozen=True)
class PlannerConfig:
    """Path simplification and nominal velocity guidance parameters."""

    lookahead_distance_m: float = 0.25
    goal_gain: float = 1.0
    maximum_nominal_speed_mps: float = 0.45
    target_tolerance_m: float = 0.08
    path_simplification: bool = True
    minimum_path_replan_interval_s: float = 0.10

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlannerConfig":
        config = cls(**dict(data or {}))
        config.validate()
        return config

    def validate(self) -> None:
        for name, value in (
            ("lookahead_distance_m", self.lookahead_distance_m),
            ("goal_gain", self.goal_gain),
            ("maximum_nominal_speed_mps", self.maximum_nominal_speed_mps),
            ("target_tolerance_m", self.target_tolerance_m),
            ("minimum_path_replan_interval_s", self.minimum_path_replan_interval_s),
        ):
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"planner.{name} must be positive and finite.")


@dataclass(frozen=True)
class PlannedPath:
    """Metric path synchronized to an occupancy version and target."""

    points_xy: np.ndarray
    active_zone_identifier: int
    occupancy_version: int
    valid: bool
    reason: str


def construct_active_path(
    bundle: ReachabilityBundle,
    *,
    position_xy: Iterable[float],
    active_zone_identifier: int,
    config: PlannerConfig,
) -> PlannedPath:
    """Trace and optionally simplify a shortest path to the active target disk."""

    field = bundle.fields[int(active_zone_identifier)]
    raw_path = extract_path_from_predecessor(field, bundle.geometry, position_xy)
    if raw_path.shape[0] == 0:
        return PlannedPath(
            points_xy=raw_path,
            active_zone_identifier=int(active_zone_identifier),
            occupancy_version=bundle.occupancy_version,
            valid=False,
            reason="no finite predecessor path to active landing zone",
        )
    path = (
        simplify_path_line_of_sight(raw_path, bundle.inflated_occupancy, bundle.geometry)
        if config.path_simplification
        else raw_path
    )
    valid = path_is_collision_free(path, bundle.inflated_occupancy, bundle.geometry)
    return PlannedPath(
        points_xy=path,
        active_zone_identifier=int(active_zone_identifier),
        occupancy_version=bundle.occupancy_version,
        valid=valid,
        reason="ok" if valid else "path contains an inflated-occupancy collision",
    )


def pure_pursuit_velocity(
    position_xy: Iterable[float],
    path_xy: np.ndarray,
    *,
    config: PlannerConfig,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return nominal velocity, look-ahead target, and path-end completion flag."""

    position = np.asarray(position_xy, dtype=float).reshape(2)
    path = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    if path.shape[0] == 0:
        return np.zeros(2, dtype=float), position.copy(), False
    final_distance = float(np.linalg.norm(path[-1] - position))
    if final_distance <= config.target_tolerance_m:
        return np.zeros(2, dtype=float), path[-1].copy(), True

    # Start at the closest path point and advance until the accumulated distance
    # reaches the configured look-ahead radius.
    closest_index = int(np.argmin(np.linalg.norm(path - position[None, :], axis=1)))
    target_index = closest_index
    accumulated = 0.0
    previous = position
    for index in range(closest_index, path.shape[0]):
        accumulated += float(np.linalg.norm(path[index] - previous))
        target_index = index
        previous = path[index]
        if accumulated >= config.lookahead_distance_m:
            break
    target = path[target_index]
    nominal = config.goal_gain * (target - position)
    norm = float(np.linalg.norm(nominal))
    if norm > config.maximum_nominal_speed_mps and norm > 0.0:
        nominal *= config.maximum_nominal_speed_mps / norm
    return nominal, target.copy(), False


__all__ = ["PlannedPath", "PlannerConfig", "construct_active_path", "pure_pursuit_velocity"]
