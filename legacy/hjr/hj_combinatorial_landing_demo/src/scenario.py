"""3-D Mars-analog scenario, occupancy construction, and coordinate helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LandingZone:
    """One candidate landing target in the reduced position state space."""

    name: str
    center: np.ndarray
    radius: float
    science_score: float


@dataclass
class World:
    """Grid world shared by Poisson synthesis and HJ reachability."""

    size: np.ndarray
    shape: tuple[int, int, int]
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    spacing: tuple[float, float, float]
    mesh: tuple[np.ndarray, np.ndarray, np.ndarray]
    occupancy_initial: np.ndarray
    occupancy_updated: np.ndarray
    start: np.ndarray
    science_waypoint: np.ndarray
    landing_zones: list[LandingZone]
    visible_primitives: list[dict[str, Any]]
    hidden_hazard: dict[str, Any]

    def world_to_index(self, point: np.ndarray) -> tuple[int, int, int]:
        """Map a metric position to the nearest valid grid index."""
        out = []
        for axis, coordinate in zip(self.axes, np.asarray(point, dtype=float)):
            idx = int(np.argmin(np.abs(axis - coordinate)))
            out.append(max(0, min(len(axis) - 1, idx)))
        return tuple(out)  # type: ignore[return-value]

    def index_to_world(self, index: tuple[int, int, int]) -> np.ndarray:
        return np.array([self.axes[d][index[d]] for d in range(3)], dtype=float)


def _add_sphere(mask: np.ndarray, mesh, center, radius: float) -> None:
    X, Y, Z = mesh
    c = np.asarray(center, dtype=float)
    mask[(X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 <= radius**2] = True


def _add_box(mask: np.ndarray, mesh, center, size) -> None:
    X, Y, Z = mesh
    c = np.asarray(center, dtype=float)
    s = np.asarray(size, dtype=float)
    mask[(np.abs(X - c[0]) <= s[0] / 2) & (np.abs(Y - c[1]) <= s[1] / 2) & (np.abs(Z - c[2]) <= s[2] / 2)] = True


def _add_cylinder(mask: np.ndarray, mesh, center, radius: float, height: float) -> None:
    X, Y, Z = mesh
    c = np.asarray(center, dtype=float)
    mask[((X - c[0]) ** 2 + (Y - c[1]) ** 2 <= radius**2) & (np.abs(Z - c[2]) <= height / 2)] = True


def build_world(config: dict) -> World:
    """Build an obstacle-rich 3-D world and a hidden landing-zone hazard."""
    world_cfg = config["world"]
    size = np.asarray(world_cfg["size_m"], dtype=float)
    shape = tuple(int(x) for x in world_cfg["grid_shape"])
    axes = tuple(np.linspace(0.0, size[d], shape[d]) for d in range(3))
    spacing = tuple(float(axis[1] - axis[0]) for axis in axes)
    mesh = np.meshgrid(*axes, indexing="ij")
    occupancy = np.zeros(shape, dtype=bool)

    # Obstacles are selected to force a nontrivial, fully three-dimensional path.
    primitives: list[dict[str, Any]] = [
        {"kind": "box", "name": "ridge-west", "center": [5.0, 7.8, 3.1], "size": [0.85, 9.0, 5.0]},
        {"kind": "box", "name": "ridge-east", "center": [10.0, 5.1, 2.5], "size": [2.1, 2.0, 4.3]},
        {"kind": "sphere", "name": "aerial-rock-1", "center": [7.1, 3.7, 6.2], "radius": 1.20},
        {"kind": "sphere", "name": "aerial-rock-2", "center": [9.1, 9.7, 5.2], "radius": 1.15},
        {"kind": "sphere", "name": "aerial-rock-3", "center": [13.0, 7.9, 3.6], "radius": 1.00},
        {"kind": "cylinder", "name": "tower-1", "center": [12.0, 11.0, 2.8], "radius": 0.55, "height": 5.6},
        {"kind": "cylinder", "name": "tower-2", "center": [14.8, 6.8, 2.25], "radius": 0.55, "height": 4.5},
        {"kind": "box", "name": "landing-clutter", "center": [15.1, 13.6, 1.1], "size": [1.2, 1.2, 2.2]},
    ]
    for item in primitives:
        if item["kind"] == "sphere":
            _add_sphere(occupancy, mesh, item["center"], item["radius"])
        elif item["kind"] == "box":
            _add_box(occupancy, mesh, item["center"], item["size"])
        elif item["kind"] == "cylinder":
            _add_cylinder(occupancy, mesh, item["center"], item["radius"], item["height"])

    landing_zones = [
        LandingZone(
            name=str(item["name"]),
            center=np.asarray(item["center"], dtype=float),
            radius=float(item["radius"]),
            science_score=float(item["science_score"]),
        )
        for item in config["landing_zones"]
    ]

    # The first landing zone appears free in the prior map.  The hazard is added
    # only after the vehicle reaches the survey region and obtains better data.
    hidden_hazard = {
        "kind": "sphere",
        "name": "newly-detected-LZ1-hazard",
        "center": landing_zones[0].center + np.array([0.0, 0.0, 0.18]),
        "radius": 0.82,
    }
    occupancy_updated = occupancy.copy()
    _add_sphere(occupancy_updated, mesh, hidden_hazard["center"], hidden_hazard["radius"])

    # Inflate all obstacles by one voxel in each axis to represent vehicle size
    # and occupancy uncertainty.  The hidden hazard is inflated consistently.
    from scipy.ndimage import binary_dilation

    structure = np.ones((3, 3, 3), dtype=bool)
    occupancy = binary_dilation(occupancy, structure=structure, iterations=1)
    occupancy_updated = binary_dilation(occupancy_updated, structure=structure, iterations=1)

    return World(
        size=size,
        shape=shape,
        axes=axes,  # type: ignore[arg-type]
        spacing=spacing,  # type: ignore[arg-type]
        mesh=mesh,  # type: ignore[arg-type]
        occupancy_initial=occupancy,
        occupancy_updated=occupancy_updated,
        start=np.asarray(world_cfg["start"], dtype=float),
        science_waypoint=np.asarray(world_cfg["science_waypoint"], dtype=float),
        landing_zones=landing_zones,
        visible_primitives=primitives,
        hidden_hazard=hidden_hazard,
    )
