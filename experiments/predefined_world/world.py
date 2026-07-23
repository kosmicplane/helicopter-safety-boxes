"""Deterministic 3-D Mars-analog worlds for controlled landing studies.

The predefined-world experiment intentionally keeps geometry outside the
controller implementation.  A YAML world definition is converted into one
canonical occupancy grid, an analytic obstacle list for visualization, a
metric clearance field, and target equilibria.  The same geometry therefore
feeds the nominal planner, Poisson safety synthesis, collision checking, and
paper figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

from safety_box_core import EquilibriumTarget

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Obstacle:
    """One analytic obstacle used for rasterization and visualization."""

    name: str
    kind: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PredefinedWorld:
    """Canonical representation shared by all predefined-world components."""

    occupancy: np.ndarray
    clearance_m: np.ndarray
    spacing: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    start_state: np.ndarray
    targets: tuple[EquilibriumTarget, ...]
    obstacles: tuple[Obstacle, ...]
    inflation_m: float
    name: str
    summary: str
    source_file: str | None


def _float_tuple(values: Sequence[Any], length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values.")
    return result


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
        & (Z >= max(0.0, z_range[0] - inflation))
        & (Z <= z_range[1] + inflation)
    )


def _inline_default_obstacles() -> tuple[Obstacle, ...]:
    """Legacy fallback retained for backward compatibility."""

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


def _load_world_definition(config: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    world_config = dict(config["experiments"]["predefined_world"]["world"])
    world_file = world_config.get("world_file")
    if world_file is None:
        return world_config, None

    path = Path(str(world_file))
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Predefined-world file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError(f"World definition must be a mapping: {path}")

    # Inline values remain useful for experiment-specific overrides.
    merged = dict(loaded)
    for key, value in world_config.items():
        if key != "world_file":
            merged[key] = value
    return merged, str(path.relative_to(REPOSITORY_ROOT))


def _parse_obstacle(record: Mapping[str, Any], index: int) -> Obstacle:
    if not isinstance(record, Mapping):
        raise TypeError(f"Obstacle entry {index} must be a mapping.")
    name = str(record.get("name", f"obstacle_{index:02d}"))
    kind = str(record["type"])
    if kind == "box":
        parameters = {
            "minimum": _float_tuple(record["minimum"], 3, f"{name}.minimum"),
            "maximum": _float_tuple(record["maximum"], 3, f"{name}.maximum"),
        }
    elif kind == "cylinder":
        parameters = {
            "center": _float_tuple(record["center"], 2, f"{name}.center"),
            "radius": float(record["radius"]),
            "z_range": _float_tuple(record["z_range"], 2, f"{name}.z_range"),
        }
    elif kind == "ellipsoid":
        parameters = {
            "center": _float_tuple(record["center"], 3, f"{name}.center"),
            "radii": _float_tuple(record["radii"], 3, f"{name}.radii"),
        }
    elif kind == "annular_cylinder":
        parameters = {
            "center": _float_tuple(record["center"], 2, f"{name}.center"),
            "inner_radius": float(record["inner_radius"]),
            "outer_radius": float(record["outer_radius"]),
            "z_range": _float_tuple(record["z_range"], 2, f"{name}.z_range"),
        }
    else:
        raise ValueError(f"Unsupported obstacle type {kind!r} for {name!r}.")
    return Obstacle(name=name, kind=kind, parameters=parameters)


def _parse_targets(
    records: Sequence[Any],
    landing_radius: float,
) -> tuple[EquilibriumTarget, ...]:
    targets: list[EquilibriumTarget] = []
    for index, record in enumerate(records):
        if isinstance(record, Mapping):
            identifier = str(record.get("id", f"LZ{index}"))
            point = np.asarray(record["position_m"], dtype=float)
            label = str(record.get("label", identifier))
        else:
            identifier = f"LZ{index}"
            point = np.asarray(record, dtype=float)
            label = identifier
        if point.shape != (3,):
            raise ValueError(f"Landing target {identifier!r} must contain [x, y, z].")
        targets.append(
            EquilibriumTarget(
                identifier=identifier,
                x_star=np.concatenate([point, np.zeros(3)]),
                u_star=np.zeros(3),
                metadata={
                    "position_m": point.tolist(),
                    "radius_m": float(landing_radius),
                    "label": label,
                },
            )
        )
    if not targets:
        raise ValueError("At least one landing zone is required.")
    identifiers = [target.identifier for target in targets]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Landing-zone identifiers must be unique.")
    return tuple(targets)


def build_world(config: Mapping[str, Any]) -> PredefinedWorld:
    """Rasterize the configured world and validate all mission-critical points."""

    world_config, source_file = _load_world_definition(config)
    extent = _float_tuple(world_config["extent_m"], 3, "extent_m")
    shape = tuple(int(value) for value in world_config["grid_shape"])
    if len(shape) != 3 or min(shape) < 4:
        raise ValueError("grid_shape must contain three values of at least four cells.")
    inflation = float(world_config.get("obstacle_inflation_m", 0.20))
    axes = tuple(np.linspace(0.0, extent[index], shape[index]) for index in range(3))
    spacing = tuple(float(axis[1] - axis[0]) for axis in axes)
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    occupancy = np.zeros(shape, dtype=bool)

    obstacle_records = world_config.get("obstacles")
    obstacles = (
        tuple(_parse_obstacle(record, index) for index, record in enumerate(obstacle_records))
        if obstacle_records is not None
        else _inline_default_obstacles()
    )
    for obstacle in obstacles:
        parameters = obstacle.parameters
        if obstacle.kind == "box":
            occupancy |= _box_mask(
                X, Y, Z, parameters["minimum"], parameters["maximum"], inflation
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
                X, Y, Z, parameters["center"], parameters["radii"], inflation
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

    # The computational boundary carries the homogeneous Dirichlet condition.
    occupancy[0, :, :] = True
    occupancy[-1, :, :] = True
    occupancy[:, 0, :] = True
    occupancy[:, -1, :] = True
    occupancy[:, :, 0] = True
    occupancy[:, :, -1] = True

    clearance_m = distance_transform_edt(~occupancy, sampling=spacing)
    start_state = np.asarray(world_config["start_state"], dtype=float)
    if start_state.shape != (6,):
        raise ValueError("start_state must be [x, y, z, vx, vy, vz].")
    landing_radius = float(world_config.get("landing_radius_m", 0.60))
    targets = _parse_targets(world_config["landing_zones"], landing_radius)

    world = PredefinedWorld(
        occupancy=occupancy,
        clearance_m=clearance_m,
        spacing=spacing,
        extent_m=extent,
        axes=axes,
        start_state=start_state,
        targets=targets,
        obstacles=obstacles,
        inflation_m=inflation,
        name=str(world_config.get("name", "predefined_world")),
        summary=str(world_config.get("summary", "")),
        source_file=source_file,
    )
    if point_is_occupied(world, start_state[:3]):
        raise RuntimeError("The start state lies in occupied space.")
    for target in targets:
        if point_is_occupied(world, target.x_star[:3]):
            raise RuntimeError(f"Landing target {target.identifier} lies in occupied space.")
    return world


def _nearest_index(world: PredefinedWorld, point: np.ndarray) -> tuple[int, int, int]:
    coordinate = np.asarray(point, dtype=float)
    index = np.rint(coordinate / np.asarray(world.spacing)).astype(int)
    index = np.clip(index, 0, np.asarray(world.occupancy.shape) - 1)
    return tuple(int(value) for value in index)


def point_is_occupied(world: PredefinedWorld, point: np.ndarray) -> bool:
    """Query the nearest occupancy voxel, treating out-of-bounds points as occupied."""

    coordinate = np.asarray(point, dtype=float)
    if np.any(coordinate < 0.0) or np.any(coordinate > np.asarray(world.extent_m)):
        return True
    return bool(world.occupancy[_nearest_index(world, coordinate)])


def point_clearance_m(world: PredefinedWorld, point: np.ndarray) -> float:
    """Return nearest-voxel obstacle clearance in metres."""

    coordinate = np.asarray(point, dtype=float)
    if np.any(coordinate < 0.0) or np.any(coordinate > np.asarray(world.extent_m)):
        return 0.0
    return float(world.clearance_m[_nearest_index(world, coordinate)])


def segment_collision_fraction(
    world: PredefinedWorld,
    start: np.ndarray,
    goal: np.ndarray,
    *,
    samples: int = 500,
) -> float:
    """Return the fraction of a straight segment lying in occupied voxels."""

    points = np.linspace(np.asarray(start, dtype=float), np.asarray(goal, dtype=float), samples)
    occupied = np.fromiter((point_is_occupied(world, point) for point in points), dtype=bool)
    return float(np.mean(occupied))
