"""Mission-selection geometry and validation tests."""

from __future__ import annotations

import numpy as np

from common.coordinates import GridGeometry
from common.mission_setup import (
    LandingZone,
    MissionDefinition,
    MissionSetupConfig,
    metric_disk_mask,
    metric_disk_pixel_radii,
    metric_to_rectified_pixel,
    rectified_pixel_to_metric,
    validate_mission,
)


def make_mission(*, start=(0.5, 0.5), centers=((3.0, 1.0), (3.0, 2.5)), radius=0.3, r=2):
    return MissionDefinition(
        start_xy_m=np.asarray(start, dtype=float),
        landing_zones=tuple(
            LandingZone(identifier=index + 1, center_xy_m=np.asarray(center), radius_m=radius)
            for index, center in enumerate(centers)
        ),
        active_zone_identifier=1,
        required_reachable=r,
        workspace_size_m=(4.0, 3.0),
        calibration_hash="test",
    )


def setup_config() -> MissionSetupConfig:
    return MissionSetupConfig(
        minimum_landing_zones=2,
        maximum_landing_zones=6,
        default_landing_zone_radius_m=0.3,
        minimum_landing_zone_radius_m=0.2,
        maximum_landing_zone_radius_m=0.5,
        minimum_touchdown_margin_m=0.02,
        minimum_zone_edge_separation_m=0.1,
        minimum_start_zone_separation_m=0.1,
        minimum_boundary_clearance_m=0.05,
        default_required_reachable=2,
    )


def test_pixel_metric_round_trip() -> None:
    pixel = np.array([160.0, 120.0])
    metric = rectified_pixel_to_metric(pixel, image_shape_yx=(241, 321), workspace_size_m=(4.0, 3.0))
    np.testing.assert_allclose(metric, [2.0, 1.5])
    np.testing.assert_allclose(
        metric_to_rectified_pixel(metric, image_shape_yx=(241, 321), workspace_size_m=(4.0, 3.0)),
        pixel,
    )


def test_metric_circle_uses_unequal_pixel_radii() -> None:
    radius_x, radius_y = metric_disk_pixel_radii(
        0.5,
        image_shape_yx=(301, 801),
        workspace_size_m=(8.0, 3.0),
    )
    assert radius_x == 50
    assert radius_y == 50
    radius_x2, radius_y2 = metric_disk_pixel_radii(
        0.5,
        image_shape_yx=(301, 401),
        workspace_size_m=(8.0, 3.0),
    )
    assert radius_x2 == 25
    assert radius_y2 == 50


def test_metric_disk_mask_contains_center_and_respects_radius() -> None:
    geometry = GridGeometry(width_m=4.0, height_m=3.0, nx=41, ny=31)
    mask = metric_disk_mask(geometry, [2.0, 1.5], 0.3)
    assert mask[15, 20]
    assert not mask[15, 25]


def test_valid_mission_passes() -> None:
    geometry = GridGeometry(width_m=4.0, height_m=3.0, nx=41, ny=31)
    result = validate_mission(
        make_mission(),
        setup_config=setup_config(),
        geometry=geometry,
        robot_radius_m=0.1,
        perception_margin_m=0.05,
        raw_occupancy=np.zeros(geometry.shape_yx, dtype=bool),
        inflated_occupancy=np.zeros(geometry.shape_yx, dtype=bool),
    )
    assert result.valid, result.reasons


def test_radius_lower_bound_rejected() -> None:
    geometry = GridGeometry(width_m=4.0, height_m=3.0, nx=41, ny=31)
    result = validate_mission(
        make_mission(radius=0.1),
        setup_config=setup_config(),
        geometry=geometry,
        robot_radius_m=0.1,
        perception_margin_m=0.05,
    )
    assert not result.valid
    assert any("below" in reason for reason in result.reasons)


def test_boundary_clearance_rejected() -> None:
    geometry = GridGeometry(width_m=4.0, height_m=3.0, nx=41, ny=31)
    mission = make_mission(centers=((0.2, 1.0), (3.0, 2.5)))
    result = validate_mission(
        mission,
        setup_config=setup_config(),
        geometry=geometry,
        robot_radius_m=0.1,
        perception_margin_m=0.05,
    )
    assert not result.valid
    assert any("boundary" in reason for reason in result.reasons)


def test_overlap_rejected() -> None:
    geometry = GridGeometry(width_m=4.0, height_m=3.0, nx=41, ny=31)
    result = validate_mission(
        make_mission(centers=((2.0, 1.5), (2.4, 1.5))),
        setup_config=setup_config(),
        geometry=geometry,
        robot_radius_m=0.1,
        perception_margin_m=0.05,
    )
    assert not result.valid
    assert any("overlap" in reason for reason in result.reasons)


def test_start_inside_inflated_obstacle_rejected() -> None:
    geometry = GridGeometry(width_m=4.0, height_m=3.0, nx=41, ny=31)
    occupied = np.zeros(geometry.shape_yx, dtype=bool)
    occupied[5, 5] = True
    result = validate_mission(
        make_mission(start=(0.5, 0.5)),
        setup_config=setup_config(),
        geometry=geometry,
        robot_radius_m=0.1,
        perception_margin_m=0.05,
        inflated_occupancy=occupied,
    )
    assert not result.valid
    assert any("START" in reason for reason in result.reasons)


def test_zone_in_occupancy_rejected() -> None:
    geometry = GridGeometry(width_m=4.0, height_m=3.0, nx=41, ny=31)
    occupied = np.zeros(geometry.shape_yx, dtype=bool)
    occupied[10, 30] = True
    result = validate_mission(
        make_mission(),
        setup_config=setup_config(),
        geometry=geometry,
        robot_radius_m=0.1,
        perception_margin_m=0.05,
        raw_occupancy=occupied,
    )
    assert not result.valid
    assert any("raw occupancy" in reason for reason in result.reasons)


def test_mission_save_load_round_trip(tmp_path) -> None:
    mission = make_mission()
    path = tmp_path / "mission.json"
    mission.save(path)
    loaded = MissionDefinition.load(path)
    np.testing.assert_allclose(loaded.start_xy_m, mission.start_xy_m)
    assert loaded.required_reachable == 2
    assert loaded.p == 2
