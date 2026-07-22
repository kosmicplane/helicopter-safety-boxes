"""Landing-zone blocking hysteresis and switching tests."""

from __future__ import annotations

import numpy as np

from common.coordinates import GridGeometry
from common.mission_setup import LandingZone, MissionDefinition
from common.target_manager import (
    LandingZoneValidationConfig,
    TargetManager,
    ZoneState,
    assess_all_landing_zones,
)


def mission() -> MissionDefinition:
    return MissionDefinition(
        start_xy_m=np.array([0.5, 0.5]),
        landing_zones=(
            LandingZone(1, np.array([2.0, 0.7]), 0.25),
            LandingZone(2, np.array([2.0, 2.0]), 0.25),
            LandingZone(3, np.array([3.2, 1.5]), 0.25),
        ),
        active_zone_identifier=1,
        required_reachable=2,
        workspace_size_m=(4.0, 3.0),
        calibration_hash="test",
    )


def config() -> LandingZoneValidationConfig:
    return LandingZoneValidationConfig(
        raw_occupied_fraction_threshold=0.01,
        inflated_occupied_fraction_threshold=0.01,
        minimum_clearance_m=0.0,
        minimum_valid_seed_cells=1,
        blocked_activation_frames=2,
        clear_deactivation_frames=2,
        latch_rejected_zones=True,
    )


def test_single_noisy_frame_does_not_reject() -> None:
    geometry = GridGeometry(width_m=4, height_m=3, nx=41, ny=31)
    mgr = TargetManager(mission(), config())
    raw = np.zeros(geometry.shape_yx, dtype=bool)
    raw[7, 20] = True
    assessments = assess_all_landing_zones(mission(), raw_occupancy=raw, inflated_occupancy=raw, geometry=geometry, config=config())
    mgr.update_assessments(assessments)
    assert mgr.states[1].state != ZoneState.REJECTED


def test_second_blocked_frame_latches_rejection() -> None:
    geometry = GridGeometry(width_m=4, height_m=3, nx=41, ny=31)
    mgr = TargetManager(mission(), config())
    raw = np.zeros(geometry.shape_yx, dtype=bool)
    raw[7, 20] = True
    assessments = assess_all_landing_zones(mission(), raw_occupancy=raw, inflated_occupancy=raw, geometry=geometry, config=config())
    mgr.update_assessments(assessments)
    newly = mgr.update_assessments(assessments)
    assert 1 in newly
    assert mgr.states[1].state == ZoneState.REJECTED


def test_rejected_zone_does_not_clear_automatically() -> None:
    geometry = GridGeometry(width_m=4, height_m=3, nx=41, ny=31)
    mgr = TargetManager(mission(), config())
    blocked = np.zeros(geometry.shape_yx, dtype=bool)
    blocked[7, 20] = True
    blocked_assessment = assess_all_landing_zones(mission(), raw_occupancy=blocked, inflated_occupancy=blocked, geometry=geometry, config=config())
    mgr.update_assessments(blocked_assessment)
    mgr.update_assessments(blocked_assessment)
    clear = np.zeros_like(blocked)
    clear_assessment = assess_all_landing_zones(mission(), raw_occupancy=clear, inflated_occupancy=clear, geometry=geometry, config=config())
    for _ in range(5):
        mgr.update_assessments(clear_assessment)
    assert mgr.states[1].state == ZoneState.REJECTED


def test_switch_selects_reachable_alternative() -> None:
    mgr = TargetManager(mission(), config())
    mgr.update_reachability(
        values={1: -1.0, 2: 2.0, 3: 3.0},
        distances={1: 5.0, 2: 3.0, 3: 4.0},
    )
    mgr.reject(1, "blocked")
    alternative = mgr.choose_certified_alternative(exclude_identifier=1)
    assert alternative == 3  # larger reachability margin has first priority
    event = mgr.switch_to(alternative, time_s=1.0, reason="test", occupancy_version=4)
    assert event.new_identifier == 3
    assert mgr.active_identifier == 3


def test_manual_cycle_never_selects_unreachable_zone() -> None:
    mgr = TargetManager(mission(), config())
    mgr.update_reachability(
        values={1: 2.0, 2: -1.0, 3: 1.0},
        distances={1: 1.0, 2: 3.0, 3: 2.0},
    )
    event = mgr.cycle_manual_target(time_s=0.0, occupancy_version=1)
    assert event is not None
    assert event.new_identifier == 3
