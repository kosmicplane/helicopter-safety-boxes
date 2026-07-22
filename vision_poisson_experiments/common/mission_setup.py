"""Interactive and file-based mission definition for the live contingency demo.

The workspace calibration establishes a metric top-down plane.  This module adds
one virtual start position and a set of metric circular landing zones on that
plane.  Users select only disk centers; a shared radius in meters is validated
against the robot footprint, the workspace boundary, other zones, and optional
occupancy grids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .calibration import CalibrationData
from .coordinates import GridGeometry


@dataclass(frozen=True)
class MissionSetupConfig:
    """Validated user-facing mission selection parameters."""

    mode: str = "interactive"
    file: str = "calibration/live_mission.json"
    minimum_landing_zones: int = 2
    maximum_landing_zones: int = 8
    default_landing_zone_radius_m: float = 0.30
    minimum_landing_zone_radius_m: float = 0.25
    maximum_landing_zone_radius_m: float = 0.42
    radius_adjustment_step_m: float = 0.025
    minimum_touchdown_margin_m: float = 0.05
    minimum_zone_edge_separation_m: float = 0.12
    minimum_start_zone_separation_m: float = 0.20
    minimum_boundary_clearance_m: float = 0.08
    default_required_reachable: int = 2
    save_selection: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MissionSetupConfig":
        """Construct and validate a configuration from YAML-compatible data."""

        config = cls(**dict(data or {}))
        config.validate()
        return config

    def validate(self) -> None:
        """Reject invalid bounds before the interactive UI starts."""

        if self.mode not in {"interactive", "load_file"}:
            raise ValueError("mission_setup.mode must be 'interactive' or 'load_file'.")
        if self.minimum_landing_zones < 1:
            raise ValueError("minimum_landing_zones must be at least one.")
        if self.maximum_landing_zones < self.minimum_landing_zones:
            raise ValueError("maximum_landing_zones must not be smaller than the minimum.")
        if not 1 <= self.default_required_reachable <= self.maximum_landing_zones:
            raise ValueError("default_required_reachable is outside the configured zone count range.")
        values = [
            self.default_landing_zone_radius_m,
            self.minimum_landing_zone_radius_m,
            self.maximum_landing_zone_radius_m,
            self.radius_adjustment_step_m,
            self.minimum_touchdown_margin_m,
            self.minimum_zone_edge_separation_m,
            self.minimum_start_zone_separation_m,
            self.minimum_boundary_clearance_m,
        ]
        if not all(np.isfinite(value) and float(value) >= 0.0 for value in values):
            raise ValueError("Mission setup metric parameters must be finite and nonnegative.")
        if self.minimum_landing_zone_radius_m <= 0.0:
            raise ValueError("minimum_landing_zone_radius_m must be positive.")
        if self.maximum_landing_zone_radius_m < self.minimum_landing_zone_radius_m:
            raise ValueError("maximum_landing_zone_radius_m must exceed the minimum.")
        if not (
            self.minimum_landing_zone_radius_m
            <= self.default_landing_zone_radius_m
            <= self.maximum_landing_zone_radius_m
        ):
            raise ValueError("The default landing-zone radius is outside its configured bounds.")
        if self.radius_adjustment_step_m <= 0.0:
            raise ValueError("radius_adjustment_step_m must be positive.")


@dataclass(frozen=True)
class LandingZone:
    """One metric circular landing-zone target."""

    identifier: int
    center_xy_m: np.ndarray
    radius_m: float
    source_center_px: np.ndarray | None = None
    priority: float = 1.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center_xy_m, dtype=float).reshape(2)
        if not np.all(np.isfinite(center)):
            raise ValueError("Landing-zone center must be finite.")
        if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("Landing-zone radius must be positive and finite.")
        pixel = None if self.source_center_px is None else np.asarray(self.source_center_px, dtype=float).reshape(2)
        object.__setattr__(self, "center_xy_m", center)
        object.__setattr__(self, "source_center_px", pixel)

    @property
    def name(self) -> str:
        """Return a stable human-readable label."""

        return f"LZ-{self.identifier}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "identifier": int(self.identifier),
            "name": self.name,
            "center_xy_m": self.center_xy_m.tolist(),
            "radius_m": float(self.radius_m),
            "source_center_px": None if self.source_center_px is None else self.source_center_px.tolist(),
            "priority": float(self.priority),
        }


@dataclass(frozen=True)
class MissionDefinition:
    """Serializable start, targets, active target, and r-out-of-p requirement."""

    start_xy_m: np.ndarray
    landing_zones: tuple[LandingZone, ...]
    active_zone_identifier: int
    required_reachable: int
    workspace_size_m: tuple[float, float]
    calibration_hash: str
    source_start_px: np.ndarray | None = None
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        start = np.asarray(self.start_xy_m, dtype=float).reshape(2)
        source = None if self.source_start_px is None else np.asarray(self.source_start_px, dtype=float).reshape(2)
        if not np.all(np.isfinite(start)):
            raise ValueError("Mission start must be finite.")
        if not self.landing_zones:
            raise ValueError("At least one landing zone is required.")
        identifiers = [zone.identifier for zone in self.landing_zones]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Landing-zone identifiers must be unique.")
        if self.active_zone_identifier not in identifiers:
            raise ValueError("The active landing-zone identifier is not present in the mission.")
        if not 1 <= int(self.required_reachable) <= len(self.landing_zones):
            raise ValueError("required_reachable must lie between one and the number of landing zones.")
        object.__setattr__(self, "start_xy_m", start)
        object.__setattr__(self, "source_start_px", source)
        object.__setattr__(self, "landing_zones", tuple(self.landing_zones))

    @property
    def p(self) -> int:
        """Return the number of configured landing zones."""

        return len(self.landing_zones)

    @property
    def common_radius_m(self) -> float:
        """Return the common target radius, validating consistency."""

        radii = np.asarray([zone.radius_m for zone in self.landing_zones], dtype=float)
        if not np.allclose(radii, radii[0]):
            raise ValueError("The live workflow expects one common landing-zone radius.")
        return float(radii[0])

    def zone_by_identifier(self, identifier: int) -> LandingZone:
        """Return one target by stable identifier."""

        for zone in self.landing_zones:
            if zone.identifier == int(identifier):
                return zone
        raise KeyError(f"Unknown landing-zone identifier: {identifier}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "start_xy_m": self.start_xy_m.tolist(),
            "source_start_px": None if self.source_start_px is None else self.source_start_px.tolist(),
            "landing_zones": [zone.to_dict() for zone in self.landing_zones],
            "active_zone_identifier": int(self.active_zone_identifier),
            "required_reachable": int(self.required_reachable),
            "p": self.p,
            "workspace_size_m": list(self.workspace_size_m),
            "calibration_hash": str(self.calibration_hash),
            "created_utc": str(self.created_utc),
        }

    def save(self, path: str | Path) -> None:
        """Save the mission as JSON."""

        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MissionDefinition":
        """Construct a mission from serialized data."""

        zones = tuple(
            LandingZone(
                identifier=int(zone_data.get("identifier", index + 1)),
                center_xy_m=np.asarray(zone_data["center_xy_m"], dtype=float),
                radius_m=float(zone_data["radius_m"]),
                source_center_px=(
                    None
                    if zone_data.get("source_center_px") is None
                    else np.asarray(zone_data["source_center_px"], dtype=float)
                ),
                priority=float(zone_data.get("priority", 1.0)),
            )
            for index, zone_data in enumerate(data["landing_zones"])
        )
        return cls(
            start_xy_m=np.asarray(data["start_xy_m"], dtype=float),
            landing_zones=zones,
            active_zone_identifier=int(data["active_zone_identifier"]),
            required_reachable=int(data["required_reachable"]),
            workspace_size_m=(float(data["workspace_size_m"][0]), float(data["workspace_size_m"][1])),
            calibration_hash=str(data.get("calibration_hash", "unknown")),
            source_start_px=(
                None if data.get("source_start_px") is None else np.asarray(data["source_start_px"], dtype=float)
            ),
            created_utc=str(data.get("created_utc", datetime.now(timezone.utc).isoformat())),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MissionDefinition":
        """Load a mission JSON file."""

        input_path = Path(path).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Mission file does not exist: {input_path}")
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Mission JSON must contain an object.")
        return cls.from_dict(data)


@dataclass(frozen=True)
class MissionValidationResult:
    """Validation outcome and per-zone reasons."""

    valid: bool
    reasons: tuple[str, ...]
    zone_reasons: dict[int, tuple[str, ...]]


def calibration_identifier(calibration: CalibrationData) -> str:
    """Return a stable hash for the metric camera-to-workspace mapping."""

    payload = json.dumps(calibration.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def rectified_pixel_to_metric(
    point_px: Iterable[float],
    *,
    image_shape_yx: tuple[int, int],
    workspace_size_m: tuple[float, float],
) -> np.ndarray:
    """Convert a rectified image pixel ``[x_px, y_px]`` to metric ``[x, y]``."""

    point = np.asarray(point_px, dtype=float).reshape(2)
    height_px, width_px = int(image_shape_yx[0]), int(image_shape_yx[1])
    if width_px < 2 or height_px < 2:
        raise ValueError("Rectified image dimensions must be at least two pixels.")
    width_m, height_m = float(workspace_size_m[0]), float(workspace_size_m[1])
    return np.asarray(
        [
            point[0] / float(width_px - 1) * width_m,
            point[1] / float(height_px - 1) * height_m,
        ],
        dtype=float,
    )


def metric_to_rectified_pixel(
    point_xy_m: Iterable[float],
    *,
    image_shape_yx: tuple[int, int],
    workspace_size_m: tuple[float, float],
) -> np.ndarray:
    """Convert metric ``[x, y]`` to floating-point rectified pixel coordinates."""

    point = np.asarray(point_xy_m, dtype=float).reshape(2)
    height_px, width_px = int(image_shape_yx[0]), int(image_shape_yx[1])
    width_m, height_m = float(workspace_size_m[0]), float(workspace_size_m[1])
    return np.asarray(
        [
            point[0] / width_m * float(width_px - 1),
            point[1] / height_m * float(height_px - 1),
        ],
        dtype=float,
    )


def metric_disk_pixel_radii(
    radius_m: float,
    *,
    image_shape_yx: tuple[int, int],
    workspace_size_m: tuple[float, float],
) -> tuple[int, int]:
    """Return ``(radius_x_px, radius_y_px)`` for a physical metric disk."""

    height_px, width_px = int(image_shape_yx[0]), int(image_shape_yx[1])
    width_m, height_m = float(workspace_size_m[0]), float(workspace_size_m[1])
    return (
        max(1, int(round(float(radius_m) / width_m * float(width_px - 1)))),
        max(1, int(round(float(radius_m) / height_m * float(height_px - 1)))),
    )


def metric_disk_mask(geometry: GridGeometry, center_xy_m: Iterable[float], radius_m: float) -> np.ndarray:
    """Rasterize a physical disk on a node-centered metric grid."""

    center = np.asarray(center_xy_m, dtype=float).reshape(2)
    x_values = geometry.extent_xy[0] + np.arange(geometry.nx, dtype=float) * geometry.dx
    y_values = geometry.extent_xy[2] + np.arange(geometry.ny, dtype=float) * geometry.dy
    xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
    return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= float(radius_m) ** 2 + 1.0e-12


def validate_mission(
    mission: MissionDefinition,
    *,
    setup_config: MissionSetupConfig,
    geometry: GridGeometry,
    robot_radius_m: float,
    perception_margin_m: float,
    raw_occupancy: np.ndarray | None = None,
    inflated_occupancy: np.ndarray | None = None,
) -> MissionValidationResult:
    """Validate geometry, footprint clearance, target overlap, and occupancy."""

    setup_config.validate()
    reasons: list[str] = []
    zone_reasons: dict[int, list[str]] = {zone.identifier: [] for zone in mission.landing_zones}
    if not setup_config.minimum_landing_zones <= mission.p <= setup_config.maximum_landing_zones:
        reasons.append(
            f"Landing-zone count {mission.p} is outside "
            f"[{setup_config.minimum_landing_zones}, {setup_config.maximum_landing_zones}]."
        )
    if not 1 <= mission.required_reachable <= mission.p:
        reasons.append("The r-out-of-p requirement is inconsistent with the landing-zone count.")

    minimum_radius = max(
        setup_config.minimum_landing_zone_radius_m,
        float(robot_radius_m) + float(perception_margin_m) + setup_config.minimum_touchdown_margin_m,
    )
    x_min, x_max, y_min, y_max = geometry.extent_xy
    for zone in mission.landing_zones:
        local = zone_reasons[zone.identifier]
        if zone.radius_m < minimum_radius - 1.0e-12:
            local.append(f"radius {zone.radius_m:.3f} m is below the required {minimum_radius:.3f} m")
        if zone.radius_m > setup_config.maximum_landing_zone_radius_m + 1.0e-12:
            local.append("radius exceeds the configured maximum")
        clearance = zone.radius_m + setup_config.minimum_boundary_clearance_m
        if not (
            zone.center_xy_m[0] - clearance >= x_min
            and zone.center_xy_m[0] + clearance <= x_max
            and zone.center_xy_m[1] - clearance >= y_min
            and zone.center_xy_m[1] + clearance <= y_max
        ):
            local.append("complete disk violates workspace boundary clearance")
        disk = metric_disk_mask(geometry, zone.center_xy_m, zone.radius_m)
        if int(np.count_nonzero(disk)) < 1:
            local.append("disk contains no grid cells")
        if raw_occupancy is not None and np.any(np.asarray(raw_occupancy, dtype=bool)[disk]):
            local.append("disk intersects raw occupancy")
        if inflated_occupancy is not None and np.any(np.asarray(inflated_occupancy, dtype=bool)[disk]):
            local.append("disk intersects inflated occupancy")

    for first_index, first in enumerate(mission.landing_zones):
        for second in mission.landing_zones[first_index + 1 :]:
            separation = float(np.linalg.norm(first.center_xy_m - second.center_xy_m))
            required = first.radius_m + second.radius_m + setup_config.minimum_zone_edge_separation_m
            if separation < required - 1.0e-12:
                message = f"overlaps {second.name} after edge-separation margin"
                zone_reasons[first.identifier].append(message)
                zone_reasons[second.identifier].append(
                    f"overlaps {first.name} after edge-separation margin"
                )

    for zone in mission.landing_zones:
        minimum_start_distance = zone.radius_m + setup_config.minimum_start_zone_separation_m
        if float(np.linalg.norm(mission.start_xy_m - zone.center_xy_m)) < minimum_start_distance:
            reasons.append(f"START is too close to {zone.name}.")

    if not geometry.contains_xy(mission.start_xy_m):
        reasons.append("START lies outside the metric workspace.")
    else:
        start_row, start_column = geometry.nearest_index_yx(mission.start_xy_m, clip=True)
        if inflated_occupancy is not None and bool(np.asarray(inflated_occupancy, dtype=bool)[start_row, start_column]):
            reasons.append("START lies inside inflated occupancy.")

    for identifier, local_reasons in zone_reasons.items():
        reasons.extend(f"LZ-{identifier}: {reason}" for reason in local_reasons)
    return MissionValidationResult(
        valid=not reasons,
        reasons=tuple(reasons),
        zone_reasons={identifier: tuple(local) for identifier, local in zone_reasons.items()},
    )


def draw_mission_overlay(
    image_bgr: np.ndarray,
    mission: MissionDefinition,
    *,
    validation: MissionValidationResult | None = None,
    active_zone_identifier: int | None = None,
    zone_states: dict[int, str] | None = None,
) -> np.ndarray:
    """Draw metric start/landing disks, active target, and validation state."""

    canvas = image_bgr.copy()
    image_shape = canvas.shape[:2]
    active = mission.active_zone_identifier if active_zone_identifier is None else int(active_zone_identifier)
    zone_states = zone_states or {}
    start_px = metric_to_rectified_pixel(
        mission.start_xy_m,
        image_shape_yx=image_shape,
        workspace_size_m=mission.workspace_size_m,
    ).astype(int)
    cv2.drawMarker(canvas, tuple(start_px), (0, 165, 255), cv2.MARKER_STAR, 20, 2, cv2.LINE_AA)
    cv2.putText(canvas, "START", (start_px[0] + 8, start_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

    palette = [(60, 60, 240), (80, 190, 80), (240, 150, 40), (180, 80, 220), (40, 210, 210), (210, 130, 40)]
    for index, zone in enumerate(mission.landing_zones):
        center_px = metric_to_rectified_pixel(
            zone.center_xy_m,
            image_shape_yx=image_shape,
            workspace_size_m=mission.workspace_size_m,
        ).astype(int)
        radii = metric_disk_pixel_radii(
            zone.radius_m,
            image_shape_yx=image_shape,
            workspace_size_m=mission.workspace_size_m,
        )
        state = zone_states.get(zone.identifier, "AVAILABLE")
        local_invalid = bool(validation and validation.zone_reasons.get(zone.identifier))
        if local_invalid or state in {"BLOCKED", "REJECTED"}:
            color = (0, 0, 255)
        elif state == "UNREACHABLE":
            color = (128, 128, 128)
        elif zone.identifier == active:
            color = (0, 255, 255)
        else:
            color = palette[index % len(palette)]
        cv2.ellipse(canvas, tuple(center_px), radii, 0.0, 0.0, 360.0, color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{zone.name}{' ACTIVE' if zone.identifier == active else ''}",
            (center_px[0] + 7, center_px[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )
        if state in {"BLOCKED", "REJECTED"}:
            cv2.line(canvas, (center_px[0] - radii[0], center_px[1] - radii[1]), (center_px[0] + radii[0], center_px[1] + radii[1]), (0, 0, 255), 2)
            cv2.line(canvas, (center_px[0] - radii[0], center_px[1] + radii[1]), (center_px[0] + radii[0], center_px[1] - radii[1]), (0, 0, 255), 2)

    cv2.putText(
        canvas,
        f"p={mission.p}, r={mission.required_reachable}, common radius={mission.common_radius_m:.3f} m",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if validation and validation.reasons:
        for row, reason in enumerate(validation.reasons[:4]):
            cv2.putText(canvas, reason, (10, 50 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)
    return canvas


def interactive_mission_setup(
    rectified_bgr: np.ndarray,
    *,
    calibration: CalibrationData,
    geometry: GridGeometry,
    setup_config: MissionSetupConfig,
    robot_radius_m: float,
    perception_margin_m: float,
    raw_occupancy: np.ndarray | None = None,
    inflated_occupancy: np.ndarray | None = None,
    window_name: str = "Mission setup",
) -> MissionDefinition:
    """Collect a start, landing-zone centers, active target, radius, and r value."""

    setup_config.validate()
    start_px: np.ndarray | None = None
    zone_pixels: list[np.ndarray] = []
    radius = float(setup_config.default_landing_zone_radius_m)
    active_index = 0
    required = int(setup_config.default_required_reachable)
    history: list[tuple[str, np.ndarray]] = []

    def callback(event: int, x: int, y: int, _flags: int, _parameter: object) -> None:
        nonlocal start_px, active_index, required
        point = np.asarray([float(x), float(y)], dtype=float)
        if event == cv2.EVENT_LBUTTONDOWN:
            if start_px is None:
                start_px = point
                history.append(("start", point.copy()))
            elif len(zone_pixels) < setup_config.maximum_landing_zones:
                zone_pixels.append(point)
                history.append(("zone", point.copy()))
                active_index %= max(1, len(zone_pixels))
                required = min(max(1, required), len(zone_pixels))
        elif event == cv2.EVENT_RBUTTONDOWN and zone_pixels:
            distances = [float(np.linalg.norm(candidate - point)) for candidate in zone_pixels]
            removed = int(np.argmin(distances))
            zone_pixels.pop(removed)
            active_index %= max(1, len(zone_pixels))
            required = min(max(1, required), max(1, len(zone_pixels)))

    def build_current() -> MissionDefinition | None:
        if start_px is None or not zone_pixels:
            return None
        start_metric = rectified_pixel_to_metric(
            start_px,
            image_shape_yx=rectified_bgr.shape[:2],
            workspace_size_m=calibration.workspace_size_m,
        )
        zones = tuple(
            LandingZone(
                identifier=index + 1,
                center_xy_m=rectified_pixel_to_metric(
                    pixel,
                    image_shape_yx=rectified_bgr.shape[:2],
                    workspace_size_m=calibration.workspace_size_m,
                ),
                radius_m=radius,
                source_center_px=pixel,
            )
            for index, pixel in enumerate(zone_pixels)
        )
        return MissionDefinition(
            start_xy_m=start_metric,
            source_start_px=start_px,
            landing_zones=zones,
            active_zone_identifier=zones[active_index].identifier,
            required_reachable=min(max(1, required), len(zones)),
            workspace_size_m=calibration.workspace_size_m,
            calibration_hash=calibration_identifier(calibration),
        )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, callback)
    try:
        while True:
            mission = build_current()
            validation = (
                None
                if mission is None
                else validate_mission(
                    mission,
                    setup_config=setup_config,
                    geometry=geometry,
                    robot_radius_m=robot_radius_m,
                    perception_margin_m=perception_margin_m,
                    raw_occupancy=raw_occupancy,
                    inflated_occupancy=inflated_occupancy,
                )
            )
            canvas = rectified_bgr.copy() if mission is None else draw_mission_overlay(rectified_bgr, mission, validation=validation)
            instruction = (
                "Left click START" if start_px is None else
                "Left click landing-zone centers; Enter confirms a valid mission"
            )
            cv2.putText(
                canvas,
                instruction,
                (10, canvas.shape[0] - 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Right click remove | Backspace undo | a active | [ ] radius | - + r | r reset | Esc cancel",
                (10, canvas.shape[0] - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13) and mission is not None and validation is not None and validation.valid:
                return mission
            if key in (8, 127):
                if zone_pixels:
                    zone_pixels.pop()
                elif start_px is not None:
                    start_px = None
                active_index %= max(1, len(zone_pixels))
                required = min(max(1, required), max(1, len(zone_pixels)))
            elif key == ord("a") and zone_pixels:
                active_index = (active_index + 1) % len(zone_pixels)
            elif key == ord("["):
                radius = max(setup_config.minimum_landing_zone_radius_m, radius - setup_config.radius_adjustment_step_m)
            elif key == ord("]"):
                radius = min(setup_config.maximum_landing_zone_radius_m, radius + setup_config.radius_adjustment_step_m)
            elif key in (ord("-"), ord("_")):
                required = max(1, required - 1)
            elif key in (ord("+"), ord("=")):
                required = min(max(1, len(zone_pixels)), required + 1)
            elif key == ord("r"):
                start_px = None
                zone_pixels.clear()
                active_index = 0
                required = setup_config.default_required_reachable
                radius = setup_config.default_landing_zone_radius_m
                history.clear()
            elif key == 27:
                raise RuntimeError("Mission setup cancelled by user.")
    finally:
        cv2.destroyWindow(window_name)


def load_or_select_mission(
    *,
    rectified_bgr: np.ndarray,
    calibration: CalibrationData,
    geometry: GridGeometry,
    setup_config: MissionSetupConfig,
    robot_radius_m: float,
    perception_margin_m: float,
    base_directory: str | Path,
    headless: bool,
    raw_occupancy: np.ndarray | None = None,
    inflated_occupancy: np.ndarray | None = None,
) -> MissionDefinition:
    """Load a repeatable mission or run the interactive selection UI."""

    if setup_config.mode == "load_file":
        path = (Path(base_directory) / setup_config.file).expanduser().resolve()
        mission = MissionDefinition.load(path)
    else:
        if headless:
            raise RuntimeError("Interactive mission setup is unavailable in headless mode.")
        mission = interactive_mission_setup(
            rectified_bgr,
            calibration=calibration,
            geometry=geometry,
            setup_config=setup_config,
            robot_radius_m=robot_radius_m,
            perception_margin_m=perception_margin_m,
            raw_occupancy=raw_occupancy,
            inflated_occupancy=inflated_occupancy,
        )

    validation = validate_mission(
        mission,
        setup_config=setup_config,
        geometry=geometry,
        robot_radius_m=robot_radius_m,
        perception_margin_m=perception_margin_m,
        raw_occupancy=raw_occupancy,
        inflated_occupancy=inflated_occupancy,
    )
    if not validation.valid:
        raise ValueError("Mission validation failed: " + "; ".join(validation.reasons))
    if mission.workspace_size_m != calibration.workspace_size_m:
        raise ValueError("Mission workspace dimensions do not match the current calibration.")
    return mission


__all__ = [
    "LandingZone",
    "MissionDefinition",
    "MissionSetupConfig",
    "MissionValidationResult",
    "calibration_identifier",
    "draw_mission_overlay",
    "interactive_mission_setup",
    "load_or_select_mission",
    "metric_disk_mask",
    "metric_disk_pixel_radii",
    "metric_to_rectified_pixel",
    "rectified_pixel_to_metric",
    "validate_mission",
]
