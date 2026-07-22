"""Landing-zone validation, hysteresis, latching, and certified target switching."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from .coordinates import GridGeometry
from .mission_setup import LandingZone, MissionDefinition, metric_disk_mask


class ZoneState(str, Enum):
    """Discrete state assigned to every configured landing zone."""

    AVAILABLE = "AVAILABLE"
    ACTIVE = "ACTIVE"
    BLOCKED = "BLOCKED"
    UNREACHABLE = "UNREACHABLE"
    REJECTED = "REJECTED"
    REACHED = "REACHED"


@dataclass(frozen=True)
class LandingZoneValidationConfig:
    """Thresholds and temporal hysteresis for perception-based zone blocking."""

    raw_occupied_fraction_threshold: float = 0.02
    inflated_occupied_fraction_threshold: float = 0.05
    minimum_clearance_m: float = 0.05
    minimum_valid_seed_cells: int = 3
    blocked_activation_frames: int = 2
    clear_deactivation_frames: int = 6
    latch_rejected_zones: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LandingZoneValidationConfig":
        """Construct and validate configuration data."""

        config = cls(**dict(data or {}))
        config.validate()
        return config

    def validate(self) -> None:
        """Validate fractions, counts, and physical distances."""

        for name, value in (
            ("raw_occupied_fraction_threshold", self.raw_occupied_fraction_threshold),
            ("inflated_occupied_fraction_threshold", self.inflated_occupied_fraction_threshold),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if not np.isfinite(self.minimum_clearance_m) or self.minimum_clearance_m < 0.0:
            raise ValueError("minimum_clearance_m must be finite and nonnegative.")
        if self.minimum_valid_seed_cells < 1:
            raise ValueError("minimum_valid_seed_cells must be positive.")
        if self.blocked_activation_frames < 1 or self.clear_deactivation_frames < 1:
            raise ValueError("Zone hysteresis frame counts must be positive.")


@dataclass(frozen=True)
class ZoneAssessment:
    """One frame's geometric/perceptual assessment for a target disk."""

    zone_identifier: int
    raw_occupied_fraction: float
    inflated_occupied_fraction: float
    minimum_clearance_m: float
    valid_seed_cells: int
    connected_to_free_space: bool
    blocked_now: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return {
            "zone_identifier": int(self.zone_identifier),
            "raw_occupied_fraction": float(self.raw_occupied_fraction),
            "inflated_occupied_fraction": float(self.inflated_occupied_fraction),
            "minimum_clearance_m": float(self.minimum_clearance_m),
            "valid_seed_cells": int(self.valid_seed_cells),
            "connected_to_free_space": bool(self.connected_to_free_space),
            "blocked_now": bool(self.blocked_now),
            "reasons": list(self.reasons),
        }


@dataclass
class ZoneRuntimeState:
    """Mutable latching and reachability state for one landing zone."""

    identifier: int
    state: ZoneState = ZoneState.AVAILABLE
    blocked_frames: int = 0
    clear_frames: int = 0
    reachable: bool = False
    reachability_value: float = float("-inf")
    geodesic_distance_m: float = float("inf")
    clearance_m: float = 0.0
    assessment: ZoneAssessment | None = None
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly runtime record."""

        return {
            "identifier": int(self.identifier),
            "state": self.state.value,
            "blocked_frames": int(self.blocked_frames),
            "clear_frames": int(self.clear_frames),
            "reachable": bool(self.reachable),
            "reachability_value": None if not np.isfinite(self.reachability_value) else float(self.reachability_value),
            "geodesic_distance_m": None if not np.isfinite(self.geodesic_distance_m) else float(self.geodesic_distance_m),
            "clearance_m": float(self.clearance_m),
            "assessment": None if self.assessment is None else self.assessment.to_dict(),
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class TargetSwitchEvent:
    """Audit record for one certified target switch."""

    time_s: float
    previous_identifier: int
    new_identifier: int
    reason: str
    occupancy_version: int | None
    previous_value: float | None
    new_value: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "time_s": float(self.time_s),
            "previous_identifier": int(self.previous_identifier),
            "new_identifier": int(self.new_identifier),
            "reason": str(self.reason),
            "occupancy_version": self.occupancy_version,
            "previous_value": self.previous_value,
            "new_value": float(self.new_value),
        }


def assess_landing_zone(
    zone: LandingZone,
    *,
    raw_occupancy: np.ndarray,
    inflated_occupancy: np.ndarray,
    geometry: GridGeometry,
    config: LandingZoneValidationConfig,
) -> ZoneAssessment:
    """Evaluate occupancy fractions, clearance, and target-seed connectivity."""

    raw = np.asarray(raw_occupancy, dtype=bool)
    inflated = np.asarray(inflated_occupancy, dtype=bool)
    if raw.shape != geometry.shape_yx or inflated.shape != geometry.shape_yx:
        raise ValueError("Occupancy shapes must match the metric grid.")
    disk = metric_disk_mask(geometry, zone.center_xy_m, zone.radius_m)
    disk_cells = int(np.count_nonzero(disk))
    if disk_cells == 0:
        return ZoneAssessment(
            zone_identifier=zone.identifier,
            raw_occupied_fraction=1.0,
            inflated_occupied_fraction=1.0,
            minimum_clearance_m=0.0,
            valid_seed_cells=0,
            connected_to_free_space=False,
            blocked_now=True,
            reasons=("target disk contains no grid cells",),
        )

    raw_fraction = float(np.mean(raw[disk]))
    inflated_fraction = float(np.mean(inflated[disk]))
    valid_seed_mask = disk & ~inflated
    valid_seed_cells = int(np.count_nonzero(valid_seed_mask))
    free = ~inflated
    clearance = distance_transform_edt(free, sampling=geometry.spacing_yx)
    minimum_clearance = float(np.min(clearance[disk])) if np.any(disk) else 0.0

    # Connectivity to non-target free space is a local geometric sanity check.
    # Full start-to-target connectivity is established by the geodesic/HJR field.
    dilated = np.zeros_like(disk)
    rows, columns = np.nonzero(valid_seed_mask)
    for row, column in zip(rows, columns):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = row + dr, column + dc
                if 0 <= nr < geometry.ny and 0 <= nc < geometry.nx:
                    dilated[nr, nc] = True
    connected = bool(np.any(dilated & free & ~disk)) or valid_seed_cells > 1

    reasons: list[str] = []
    if raw_fraction > config.raw_occupied_fraction_threshold:
        reasons.append("raw occupied fraction exceeds threshold")
    if inflated_fraction > config.inflated_occupied_fraction_threshold:
        reasons.append("inflated occupied fraction exceeds threshold")
    if minimum_clearance < config.minimum_clearance_m:
        reasons.append("minimum obstacle clearance is too small")
    if valid_seed_cells < config.minimum_valid_seed_cells:
        reasons.append("too few free target seed cells")
    if not connected:
        reasons.append("target disk is not connected to surrounding free space")

    return ZoneAssessment(
        zone_identifier=zone.identifier,
        raw_occupied_fraction=raw_fraction,
        inflated_occupied_fraction=inflated_fraction,
        minimum_clearance_m=minimum_clearance,
        valid_seed_cells=valid_seed_cells,
        connected_to_free_space=connected,
        blocked_now=bool(reasons),
        reasons=tuple(reasons),
    )


def assess_all_landing_zones(
    mission: MissionDefinition,
    *,
    raw_occupancy: np.ndarray,
    inflated_occupancy: np.ndarray,
    geometry: GridGeometry,
    config: LandingZoneValidationConfig,
) -> dict[int, ZoneAssessment]:
    """Assess every configured target on the same occupancy snapshot."""

    return {
        zone.identifier: assess_landing_zone(
            zone,
            raw_occupancy=raw_occupancy,
            inflated_occupancy=inflated_occupancy,
            geometry=geometry,
            config=config,
        )
        for zone in mission.landing_zones
    }


class TargetManager:
    """Maintain latching zone states and choose only certified alternatives."""

    def __init__(
        self,
        mission: MissionDefinition,
        config: LandingZoneValidationConfig,
    ) -> None:
        self.mission = mission
        self.config = config
        self.active_identifier = int(mission.active_zone_identifier)
        self.states = {
            zone.identifier: ZoneRuntimeState(
                identifier=zone.identifier,
                state=(ZoneState.ACTIVE if zone.identifier == self.active_identifier else ZoneState.AVAILABLE),
            )
            for zone in mission.landing_zones
        }
        self.switch_events: list[TargetSwitchEvent] = []
        self.hold_reason: str | None = None

    def reset_rejections(self) -> None:
        """Manually clear latches without changing the active target."""

        for state in self.states.values():
            if state.state in {ZoneState.REJECTED, ZoneState.BLOCKED, ZoneState.UNREACHABLE}:
                state.state = ZoneState.ACTIVE if state.identifier == self.active_identifier else ZoneState.AVAILABLE
                state.rejection_reason = None
                state.blocked_frames = 0
                state.clear_frames = 0
        self.hold_reason = None

    def update_assessments(self, assessments: dict[int, ZoneAssessment]) -> list[int]:
        """Apply frame-based blocking hysteresis and return newly rejected IDs."""

        newly_rejected: list[int] = []
        for identifier, assessment in assessments.items():
            runtime = self.states[identifier]
            runtime.assessment = assessment
            runtime.clearance_m = assessment.minimum_clearance_m
            if runtime.state in {ZoneState.REJECTED, ZoneState.REACHED} and self.config.latch_rejected_zones:
                continue
            if assessment.blocked_now:
                runtime.blocked_frames += 1
                runtime.clear_frames = 0
                if runtime.blocked_frames >= self.config.blocked_activation_frames:
                    if self.config.latch_rejected_zones:
                        runtime.state = ZoneState.REJECTED
                        runtime.rejection_reason = "; ".join(assessment.reasons) or "landing zone blocked"
                        newly_rejected.append(identifier)
                    else:
                        runtime.state = ZoneState.BLOCKED
            else:
                runtime.clear_frames += 1
                runtime.blocked_frames = 0
                if runtime.clear_frames >= self.config.clear_deactivation_frames:
                    runtime.state = ZoneState.ACTIVE if identifier == self.active_identifier else ZoneState.AVAILABLE
                    runtime.rejection_reason = None
        return newly_rejected

    def update_reachability(
        self,
        *,
        values: dict[int, float],
        distances: dict[int, float],
    ) -> None:
        """Update scalar HJ status while preserving rejected/reached latches."""

        for identifier, runtime in self.states.items():
            value = float(values.get(identifier, float("-inf")))
            distance = float(distances.get(identifier, float("inf")))
            runtime.reachability_value = value
            runtime.geodesic_distance_m = distance
            runtime.reachable = bool(np.isfinite(value) and value >= 0.0 and np.isfinite(distance))
            if runtime.state in {ZoneState.REJECTED, ZoneState.REACHED, ZoneState.BLOCKED}:
                continue
            if runtime.reachable:
                runtime.state = ZoneState.ACTIVE if identifier == self.active_identifier else ZoneState.AVAILABLE
            else:
                runtime.state = ZoneState.UNREACHABLE

    def available_identifiers(self) -> list[int]:
        """Return zones that are not blocked, rejected, or reached."""

        return [
            identifier
            for identifier, runtime in self.states.items()
            if runtime.state not in {ZoneState.BLOCKED, ZoneState.REJECTED, ZoneState.REACHED}
        ]

    def reachable_identifiers(self) -> list[int]:
        """Return currently certified candidate zones."""

        return [
            identifier
            for identifier, runtime in self.states.items()
            if runtime.reachable and runtime.state not in {ZoneState.BLOCKED, ZoneState.REJECTED, ZoneState.REACHED}
        ]

    def contingency_requirement_satisfied(self) -> bool:
        """Return whether at least r certified targets remain."""

        return len(self.reachable_identifiers()) >= self.mission.required_reachable

    def reject(self, identifier: int, reason: str) -> None:
        """Latch one zone as rejected."""

        runtime = self.states[int(identifier)]
        runtime.state = ZoneState.REJECTED
        runtime.rejection_reason = str(reason)
        runtime.reachable = False

    def choose_certified_alternative(
        self,
        *,
        exclude_identifier: int | None = None,
    ) -> int | None:
        """Rank reachable alternatives by margin, distance, clearance, and priority."""

        candidates: list[tuple[tuple[float, float, float, float], int]] = []
        for identifier in self.reachable_identifiers():
            if exclude_identifier is not None and identifier == int(exclude_identifier):
                continue
            runtime = self.states[identifier]
            zone = self.mission.zone_by_identifier(identifier)
            score = (
                float(runtime.reachability_value),
                -float(runtime.geodesic_distance_m),
                float(runtime.clearance_m),
                float(zone.priority),
            )
            candidates.append((score, identifier))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return int(candidates[0][1])

    def switch_to(
        self,
        new_identifier: int,
        *,
        time_s: float,
        reason: str,
        occupancy_version: int | None,
    ) -> TargetSwitchEvent:
        """Switch only to a currently reachable non-rejected target."""

        new_identifier = int(new_identifier)
        if new_identifier not in self.reachable_identifiers():
            raise ValueError(f"LZ-{new_identifier} is not a certified reachable target.")
        previous = self.active_identifier
        if previous == new_identifier:
            raise ValueError("The new target is already active.")
        previous_value = self.states[previous].reachability_value
        if self.states[previous].state not in {ZoneState.REJECTED, ZoneState.BLOCKED, ZoneState.REACHED}:
            self.states[previous].state = ZoneState.AVAILABLE if self.states[previous].reachable else ZoneState.UNREACHABLE
        self.active_identifier = new_identifier
        self.states[new_identifier].state = ZoneState.ACTIVE
        event = TargetSwitchEvent(
            time_s=float(time_s),
            previous_identifier=previous,
            new_identifier=new_identifier,
            reason=str(reason),
            occupancy_version=occupancy_version,
            previous_value=(None if not np.isfinite(previous_value) else float(previous_value)),
            new_value=float(self.states[new_identifier].reachability_value),
        )
        self.switch_events.append(event)
        return event

    def cycle_manual_target(self, *, time_s: float, occupancy_version: int | None) -> TargetSwitchEvent | None:
        """Cycle through certified targets without selecting an unsafe one."""

        certified = sorted(self.reachable_identifiers())
        if len(certified) <= 1:
            return None
        current_position = certified.index(self.active_identifier) if self.active_identifier in certified else -1
        new_identifier = certified[(current_position + 1) % len(certified)]
        return self.switch_to(
            new_identifier,
            time_s=time_s,
            reason="manual certified target cycle",
            occupancy_version=occupancy_version,
        )

    def mark_reached(self, identifier: int) -> None:
        """Mark a target reached and clear the active mission objective."""

        self.states[int(identifier)].state = ZoneState.REACHED

    def state_labels(self) -> dict[int, str]:
        """Return compact state labels for overlays."""

        return {identifier: runtime.state.value for identifier, runtime in self.states.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return the complete target-manager state."""

        return {
            "active_identifier": int(self.active_identifier),
            "required_reachable": int(self.mission.required_reachable),
            "hold_reason": self.hold_reason,
            "states": {str(identifier): runtime.to_dict() for identifier, runtime in self.states.items()},
            "switch_events": [event.to_dict() for event in self.switch_events],
        }


__all__ = [
    "LandingZoneValidationConfig",
    "TargetManager",
    "TargetSwitchEvent",
    "ZoneAssessment",
    "ZoneRuntimeState",
    "ZoneState",
    "assess_all_landing_zones",
    "assess_landing_zone",
]
