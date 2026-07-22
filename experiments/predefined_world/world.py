"""Deterministic 3-D Mars-analog obstacle world for the flagship experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from safety_box_core import EquilibriumTarget


@dataclass(frozen=True, slots=True)
class Obstacle:
    name: str
    kind: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PredefinedWorld:
    occupancy: np.ndarray
    spacing: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    start_state: np.ndarray
    targets: tuple[EquilibriumTarget, ...]
    obstacles: tuple[Obstacle, ...]
    inflation_m: float


def _box_mask(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    inflation: float,
) -> np.ndarray:
    lo = np.asarray(minimum, dtype=float) - inflation
    hi = np.asarray(maximum, dtype=float) + inflation
    return (
        (X >= lo[0])
        & (X <= hi[0])
        & (Y >= lo[1])
        & (Y <= hi[1])
        & (Z >= max(0.0, lo[2]))
        & (Z <= hi[2])
    )


def _cylinder_mask(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    center: tuple[float, float],
    radius: float,
    z_range: tuple[float, float],
    inflation: float,
) -> np.ndarray:
    radial = (X - center[0]) ** 2 + (Y - center[1]) ** 2
    return (
        radial <= (radius + inflation) ** 2
    ) & (Z >= max(0.0, z_range[0] - inflation)) & (Z <= z_range[1] + inflation)


def _ellipsoid_mask(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    inflation: float,
) -> np.ndarray:
    c = np.asarray(center, dtype=float)
    r = np.asarray(radii, dtype=float) + inflation
    value = (
        ((X - c[0]) / r[0]) ** 2
        + ((Y - c[1]) / r[1]) ** 2
        + ((Z - c[2]) / r[2]) ** 2
    )
    return value <= 1.0


def _annular_mask(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    center: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    z_range: tuple[float, float],
    inflation: float,
) -> np.ndarray:
    radial = (X - center[0]) ** 2 + (Y - center[1]) ** 2
    inner = max(0.0, inner_radius - 0.35 * inflation)
    outer = outer_radius + inflation
    return (
        (radial >= inner**2)
        & (radial <= outer**2)
        & (Z >= z_range[0])
        & (Z <= z_range[1] + inflation)
    )


def obstacle_library() -> tuple[Obstacle, ...]:
    """Return the fixed obstacle set used by the paper-oriented experiment."""

    return (
        Obstacle(
            "west_tower",
            "box",
            {"minimum": (3.6, 2.5, 0.0), "maximum": (4.6, 4.0, 7.2)},
        ),
        Obstacle(
            "north_tower",
            "cylinder",
            {"center": (7.7, 9.3), "radius": 0.95, "z_range": (0.0, 6.8)},
        ),
        Obstacle(
            "gate_south_post",
            "box",
            {"minimum": (9.7, 2.4, 0.0), "maximum": (10.7, 3.8, 6.0)},
        ),
        Obstacle(
            "gate_north_post",
            "box",
            {"minimum": (9.7, 6.0, 0.0), "maximum": (10.7, 7.4, 6.0)},
        ),
        Obstacle(
            "gate_beam",
            "box",
            {"minimum": (9.7, 2.4, 5.0), "maximum": (10.7, 7.4, 6.1)},
        ),
        Obstacle(
            "suspended_slab",
            "box",
            {"minimum": (12.2, 7.0, 3.6), "maximum": (14.2, 9.3, 5.4)},
        ),
        Obstacle(
            "aerial_boulder",
            "ellipsoid",
            {"center": (6.7, 11.3, 6.0), "radii": (1.35, 1.05, 1.0)},
        ),
        Obstacle(
            "central_spire",
            "cylinder",
            {"center": (12.0, 4.8), "radius": 0.72, "z_range": (0.0, 7.8)},
        ),
        Obstacle(
            "crater_rim",
            "annular_cylinder",
            {
                "center": (14.9, 7.1),
                "inner_radius": 0.95,
                "outer_radius": 1.55,
                "z_range": (0.0, 1.15),
            },
        ),
        Obstacle(
            "north_terrain_rock",
            "ellipsoid",
            {"center": (10.8, 11.0, 1.25), "radii": (1.15, 0.85, 1.2)},
        ),
        Obstacle(
            "low_ridge",
            "box",
            {"minimum": (5.4, 6.2, 0.0), "maximum": (7.3, 7.0, 1.45)},
        ),
    )


def build_world(config: Mapping[str, Any]) -> PredefinedWorld:
    """Rasterize the configured world and validate start and landing zones."""

    world_config = config["experiments"]["predefined_world"]["world"]
    extent = tuple(float(value) for value in world_config["extent_m"])
    shape = tuple(int(value) for value in world_config["grid_shape"])
    inflation = float(world_config.get("obstacle_inflation_m", 0.20))
    axes = tuple(np.linspace(0.0, extent[index], shape[index]) for index in range(3))
    spacing = tuple(float(axis[1] - axis[0]) for axis in axes)
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    occupancy = np.zeros(shape, dtype=bool)
    obstacles = obstacle_library()
    for obstacle in obstacles:
        parameters = obstacle.parameters
        if obstacle.kind == "box":
            occupancy |= _box_mask(
                X,
                Y,
                Z,
                parameters["minimum"],
                parameters["maximum"],
                inflation,
            )
        elif obstacle.kind == "cylinder":
            occupancy |= _cylinder_mask(
                X,
                Y,
                Z,
                parameters["center"],
                parameters["radius"],
                parameters["z_range"],
                inflation,
            )
        elif obstacle.kind == "ellipsoid":
            occupancy |= _ellipsoid_mask(
                X,
                Y,
                Z,
                parameters["center"],
                parameters["radii"],
                inflation,
            )
        elif obstacle.kind == "annular_cylinder":
            occupancy |= _annular_mask(
                X,
                Y,
                Z,
                parameters["center"],
                parameters["inner_radius"],
                parameters["outer_radius"],
                parameters["z_range"],
                inflation,
            )
        else:
            raise ValueError(f"Unsupported obstacle kind {obstacle.kind!r}.")

    # The computational boundary is Dirichlet and is represented as occupied.
    occupancy[0, :, :] = True
    occupancy[-1, :, :] = True
    occupancy[:, 0, :] = True
    occupancy[:, -1, :] = True
    occupancy[:, :, 0] = True
    occupancy[:, :, -1] = True

    start_state = np.asarray(world_config["start_state"], dtype=float)
    targets: list[EquilibriumTarget] = []
    landing_radius = float(world_config.get("landing_radius_m", 0.60))
    for index, position in enumerate(world_config["landing_zones"]):
        point = np.asarray(position, dtype=float)
        targets.append(
            EquilibriumTarget(
                identifier=f"LZ{index}",
                x_star=np.concatenate([point, np.zeros(3)]),
                u_star=np.zeros(3),
                metadata={
                    "position_m": point.tolist(),
                    "radius_m": landing_radius,
                },
            )
        )

    world = PredefinedWorld(
        occupancy=occupancy,
        spacing=spacing,
        extent_m=extent,
        axes=axes,
        start_state=start_state,
        targets=tuple(targets),
        obstacles=obstacles,
        inflation_m=inflation,
    )
    if point_is_occupied(world, start_state[:3]):
        raise RuntimeError("The start state lies in occupied space.")
    for target in targets:
        if point_is_occupied(world, target.x_star[:3]):
            raise RuntimeError(f"Landing target {target.identifier} lies in occupied space.")
    return world


def point_is_occupied(world: PredefinedWorld, point: np.ndarray) -> bool:
    """Query the nearest occupancy voxel, treating out-of-bounds points as occupied."""

    coordinate = np.asarray(point, dtype=float)
    index = np.rint(coordinate / np.asarray(world.spacing)).astype(int)
    if np.any(index < 0) or np.any(index >= np.asarray(world.occupancy.shape)):
        return True
    return bool(world.occupancy[tuple(index)])
