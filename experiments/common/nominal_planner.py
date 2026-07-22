"""Obstacle-aware nominal path planning for experiment orchestration.

The planner is intentionally *not* a safety certificate.  It supplies a
performance-oriented nominal acceleration to the multi-certificate filter;
Poisson-HOCBF, CLF, and contingency constraints remain the only formal
certificate layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import product
from typing import Mapping

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True, slots=True)
class NominalPath:
    """One deterministic grid path in metric coordinates."""

    points: np.ndarray
    target_id: str
    valid: bool
    expanded_nodes: int
    reason: str


class ObstacleAwareNominalPlanner:
    """A* + lookahead PD nominal controller for 2-D or 3-D occupancy grids.

    Parameters are deliberately separated from certificate parameters.  The
    planner may improve progress and avoid local deadlocks, but disabling it
    does not alter the mathematical definitions of the safety boxes.
    """

    def __init__(
        self,
        *,
        occupancy: np.ndarray,
        spacing: tuple[float, ...],
        targets: Mapping[str, np.ndarray],
        position_gain: float = 0.65,
        velocity_gain: float = 1.0,
        maximum_nominal_acceleration: float = 1.2,
        lookahead_distance_m: float = 1.5,
        clearance_weight: float = 2.0,
        minimum_clearance_cells: float = 1.0,
    ) -> None:
        self.occupancy = np.asarray(occupancy, dtype=bool)
        self.dimension = self.occupancy.ndim
        if self.dimension not in {2, 3}:
            raise ValueError("ObstacleAwareNominalPlanner supports 2-D or 3-D grids.")
        self.spacing = np.asarray(spacing, dtype=float).reshape(self.dimension)
        self.targets = {
            str(key): np.asarray(value, dtype=float).reshape(self.dimension)
            for key, value in targets.items()
        }
        self.position_gain = float(position_gain)
        self.velocity_gain = float(velocity_gain)
        self.maximum_nominal_acceleration = float(maximum_nominal_acceleration)
        self.lookahead_distance_m = float(lookahead_distance_m)
        self.clearance_weight = float(clearance_weight)
        self.minimum_clearance_cells = float(minimum_clearance_cells)
        if np.any(self.spacing <= 0.0):
            raise ValueError("Grid spacing must be positive.")
        if min(
            self.position_gain,
            self.velocity_gain,
            self.maximum_nominal_acceleration,
            self.lookahead_distance_m,
        ) <= 0.0:
            raise ValueError("Planner gains and limits must be positive.")
        self._cached_path: NominalPath | None = None

    def control(self, state: np.ndarray, target_id: str) -> np.ndarray:
        """Return a bounded acceleration toward a clearance-aware lookahead."""

        x = np.asarray(state, dtype=float).reshape(2 * self.dimension)
        target_id = str(target_id)
        if target_id not in self.targets:
            raise KeyError(f"Unknown target {target_id!r}.")
        if self._cached_path is None or self._cached_path.target_id != target_id:
            self._cached_path = self.plan(x[: self.dimension], self.targets[target_id], target_id)
        elif self._cached_path.points.size:
            # Replan if the vehicle drifts far from the current path.
            distance = float(
                np.min(np.linalg.norm(self._cached_path.points - x[: self.dimension], axis=1))
            )
            if distance > 2.5 * self.lookahead_distance_m:
                self._cached_path = self.plan(x[: self.dimension], self.targets[target_id], target_id)
        if not self._cached_path.valid:
            # Documented fallback: direct PD. The safety filter remains active.
            reference = self.targets[target_id]
        else:
            reference = self.lookahead(x[: self.dimension], self._cached_path.points)
        acceleration = (
            self.position_gain * (reference - x[: self.dimension])
            - self.velocity_gain * x[self.dimension :]
        )
        norm = float(np.linalg.norm(acceleration))
        if norm > self.maximum_nominal_acceleration:
            acceleration *= self.maximum_nominal_acceleration / norm
        return acceleration

    def plan(self, start: np.ndarray, goal: np.ndarray, target_id: str) -> NominalPath:
        """Plan a deterministic A* path with a physical-clearance penalty."""

        start_index = self._nearest_index(start)
        goal_index = self._nearest_index(goal)
        if self.occupancy[start_index] or self.occupancy[goal_index]:
            return NominalPath(np.empty((0, self.dimension)), target_id, False, 0, "start or goal occupied")

        clearance = distance_transform_edt(~self.occupancy, sampling=tuple(self.spacing))
        minimum_spacing = float(np.min(self.spacing))
        offsets = [offset for offset in product((-1, 0, 1), repeat=self.dimension) if any(offset)]
        queue: list[tuple[float, float, tuple[int, ...]]] = [(0.0, 0.0, start_index)]
        cost: dict[tuple[int, ...], float] = {start_index: 0.0}
        parent: dict[tuple[int, ...], tuple[int, ...]] = {}
        goal_array = np.asarray(goal_index, dtype=float)
        expanded = 0
        reached = False
        while queue:
            _, current_cost, current = heappop(queue)
            if current_cost > cost.get(current, float("inf")) + 1.0e-12:
                continue
            expanded += 1
            if current == goal_index:
                reached = True
                break
            for offset in offsets:
                neighbor = tuple(current[axis] + offset[axis] for axis in range(self.dimension))
                if not all(0 <= neighbor[axis] < self.occupancy.shape[axis] for axis in range(self.dimension)):
                    continue
                if self.occupancy[neighbor]:
                    continue
                clearance_cells = float(clearance[neighbor] / max(minimum_spacing, 1.0e-12))
                if clearance_cells < self.minimum_clearance_cells:
                    continue
                # Disallow diagonal corner cutting: every voxel in the local
                # axis-aligned cube between the two nodes must be free.
                ranges = [
                    range(min(current[a], neighbor[a]), max(current[a], neighbor[a]) + 1)
                    for a in range(self.dimension)
                ]
                if any(self.occupancy[index] for index in product(*ranges)):
                    continue
                step_length = float(np.linalg.norm(np.asarray(offset) * self.spacing))
                local_clearance = max(float(clearance[neighbor]), 0.25 * minimum_spacing)
                candidate = current_cost + step_length * (
                    1.0 + self.clearance_weight / (local_clearance * local_clearance)
                )
                if candidate + 1.0e-12 < cost.get(neighbor, float("inf")):
                    cost[neighbor] = candidate
                    parent[neighbor] = current
                    heuristic = float(
                        np.linalg.norm((np.asarray(neighbor, dtype=float) - goal_array) * self.spacing)
                    )
                    heappush(queue, (candidate + heuristic, candidate, neighbor))
        if not reached:
            return NominalPath(np.empty((0, self.dimension)), target_id, False, expanded, "no free A* path")
        indices = [goal_index]
        while indices[-1] != start_index:
            indices.append(parent[indices[-1]])
        indices.reverse()
        points = np.asarray(indices, dtype=float) * self.spacing
        points[0] = np.asarray(start, dtype=float)
        points[-1] = np.asarray(goal, dtype=float)
        # Deterministic light decimation keeps the clearance-aware shape.
        stride = max(1, int(round(0.45 / max(float(np.min(self.spacing)), 1.0e-12))))
        points = points[::stride]
        if not np.allclose(points[-1], goal):
            points = np.vstack([points, goal])
        return NominalPath(points, target_id, True, expanded, "ok")

    def lookahead(self, position: np.ndarray, path: np.ndarray) -> np.ndarray:
        """Return the first path point at least the configured arc distance away."""

        path = np.asarray(path, dtype=float).reshape(-1, self.dimension)
        if path.size == 0:
            return np.asarray(position, dtype=float).copy()
        position = np.asarray(position, dtype=float).reshape(self.dimension)
        closest = int(np.argmin(np.linalg.norm(path - position[None, :], axis=1)))
        accumulated = 0.0
        previous = position
        for index in range(closest, path.shape[0]):
            accumulated += float(np.linalg.norm(path[index] - previous))
            previous = path[index]
            if accumulated >= self.lookahead_distance_m:
                return path[index].copy()
        return path[-1].copy()

    def _nearest_index(self, point: np.ndarray) -> tuple[int, ...]:
        index = np.rint(np.asarray(point, dtype=float) / self.spacing).astype(int)
        index = np.clip(index, 0, np.asarray(self.occupancy.shape) - 1)
        return tuple(int(value) for value in index)
