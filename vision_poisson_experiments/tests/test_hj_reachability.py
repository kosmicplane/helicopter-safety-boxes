"""Reduced HJ/Eikonal reachability and obstacle-aware path tests."""

from __future__ import annotations

import numpy as np

from common.coordinates import GridGeometry
from common.hj_reachability import (
    ReachabilityConfig,
    build_reachability_bundle,
    compute_reachability_status,
    extract_path_from_predecessor,
    geodesic_distance_to_target,
    path_is_collision_free,
    pivot_and_reachable_count_fields,
    sample_zone_field,
    simplify_path_line_of_sight,
)
from common.mission_setup import LandingZone, MissionDefinition, metric_disk_mask


def mission() -> MissionDefinition:
    return MissionDefinition(
        start_xy_m=np.array([0.3, 1.5]),
        landing_zones=(
            LandingZone(1, np.array([3.7, 0.5]), 0.2),
            LandingZone(2, np.array([3.7, 2.5]), 0.2),
            LandingZone(3, np.array([3.4, 1.5]), 0.2),
        ),
        active_zone_identifier=1,
        required_reachable=2,
        workspace_size_m=(4.0, 3.0),
        calibration_hash="test",
    )


def config() -> ReachabilityConfig:
    return ReachabilityConfig(
        maximum_speed_mps=1.0,
        active_horizon_s=10.0,
        contingency_horizon_s=6.0,
        required_reachable=2,
        connectivity=8,
    )


def test_target_disk_is_zero_distance_seed() -> None:
    geometry = GridGeometry(width_m=4, height_m=3, nx=41, ny=31)
    occupied = np.zeros(geometry.shape_yx, dtype=bool)
    seed = metric_disk_mask(geometry, [3.5, 1.5], 0.25)
    distance, predecessor = geodesic_distance_to_target(occupied, seed, geometry)
    assert np.allclose(distance[seed], 0.0)
    rows, columns = np.nonzero(seed)
    assert np.all(predecessor[rows, columns, 0] == rows)
    assert np.all(predecessor[rows, columns, 1] == columns)


def test_geodesic_path_goes_around_wall() -> None:
    geometry = GridGeometry(width_m=4, height_m=3, nx=41, ny=31)
    occupied = np.zeros(geometry.shape_yx, dtype=bool)
    occupied[0:25, 20] = True  # wall with a gap at the bottom
    seed = metric_disk_mask(geometry, [3.7, 0.5], 0.2)
    distance, predecessor = geodesic_distance_to_target(occupied, seed, geometry)
    from common.hj_reachability import ZoneReachabilityField, distance_gradient_xy
    finite = np.isfinite(distance)
    field = ZoneReachabilityField(1, distance, distance_gradient_xy(distance, geometry), predecessor, seed, finite, True, 0.0, {})
    path = extract_path_from_predecessor(field, geometry, [0.3, 0.5])
    assert path.shape[0] > 2
    assert path_is_collision_free(path, occupied, geometry)
    assert np.max(path[:, 1]) > 2.4  # path must use the lower gap in +y image coordinates


def test_disconnected_target_returns_infinite_distance() -> None:
    geometry = GridGeometry(width_m=4, height_m=3, nx=41, ny=31)
    occupied = np.zeros(geometry.shape_yx, dtype=bool)
    occupied[:, 20] = True
    seed = metric_disk_mask(geometry, [3.5, 1.5], 0.2)
    distance, _ = geodesic_distance_to_target(occupied, seed, geometry)
    row, col = geometry.nearest_index_yx([0.5, 1.5])
    assert not np.isfinite(distance[row, col])


def test_hj_sign_convention_and_sample() -> None:
    geometry = GridGeometry(width_m=4, height_m=3, nx=41, ny=31)
    bundle = build_reachability_bundle(
        np.zeros(geometry.shape_yx, dtype=bool),
        mission=mission(),
        geometry=geometry,
        config=config(),
        occupancy_version=1,
    )
    near = sample_zone_field(bundle.fields[3], geometry, [3.2, 1.5], tau=-1.0, maximum_speed_mps=1.0)
    far = sample_zone_field(bundle.fields[3], geometry, [0.3, 1.5], tau=-1.0, maximum_speed_mps=1.0)
    assert near.valid and near.value >= 0.0
    assert far.valid and far.value < 0.0


def test_eikonal_gradient_validation_is_reasonable() -> None:
    geometry = GridGeometry(width_m=4, height_m=3, nx=81, ny=61)
    bundle = build_reachability_bundle(
        np.zeros(geometry.shape_yx, dtype=bool),
        mission=mission(),
        geometry=geometry,
        config=config(),
        occupancy_version=1,
    )
    validation = bundle.fields[1].validation
    assert validation["sample_count"] > 100
    assert 0.6 < validation["mean_gradient_norm"] < 1.4


def test_rth_pivot_and_reachable_count_fields() -> None:
    geometry = GridGeometry(width_m=4, height_m=3, nx=41, ny=31)
    bundle = build_reachability_bundle(
        np.zeros(geometry.shape_yx, dtype=bool),
        mission=mission(),
        geometry=geometry,
        config=config(),
        occupancy_version=1,
    )
    pivot, count, values = pivot_and_reachable_count_fields(
        bundle,
        tau=-6.0,
        maximum_speed_mps=1.0,
        required_reachable=2,
    )
    assert pivot.shape == geometry.shape_yx
    assert count.shape == geometry.shape_yx
    assert set(values) == {1, 2, 3}
    assert np.max(count) == 3


def test_live_status_reports_two_or_more_targets() -> None:
    geometry = GridGeometry(width_m=4, height_m=3, nx=41, ny=31)
    bundle = build_reachability_bundle(
        np.zeros(geometry.shape_yx, dtype=bool),
        mission=mission(),
        geometry=geometry,
        config=config(),
        occupancy_version=1,
    )
    status = compute_reachability_status(
        bundle,
        point_xy=[2.5, 1.5],
        tau=-6.0,
        maximum_speed_mps=1.0,
        required_reachable=2,
    )
    assert status.reachable_count >= 2
    assert status.pivot >= 0.0


def test_path_simplification_remains_collision_free() -> None:
    geometry = GridGeometry(width_m=4, height_m=3, nx=41, ny=31)
    occupied = np.zeros(geometry.shape_yx, dtype=bool)
    occupied[8:23, 18:23] = True
    bundle = build_reachability_bundle(
        occupied,
        mission=mission(),
        geometry=geometry,
        config=config(),
        occupancy_version=1,
    )
    raw = extract_path_from_predecessor(bundle.fields[2], geometry, mission().start_xy_m)
    simplified = simplify_path_line_of_sight(raw, occupied, geometry)
    assert simplified.shape[0] <= raw.shape[0]
    assert path_is_collision_free(simplified, occupied, geometry)
