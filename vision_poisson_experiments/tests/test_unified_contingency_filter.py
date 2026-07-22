"""Unified Poisson-CBF and HJ constraint integration through cbf_safety_box."""

from __future__ import annotations

import numpy as np

from common.coordinates import FieldSample, GridGeometry
from common.hj_reachability import ReachabilityConfig, build_reachability_bundle
from common.mission_setup import LandingZone, MissionDefinition
from common.unified_contingency_filter import (
    ContingencyFilterConfig,
    PoissonCBFConfig,
    UnifiedContingencyFilter,
)


def make_bundle():
    geometry = GridGeometry(width_m=4, height_m=3, nx=41, ny=31)
    mission = MissionDefinition(
        start_xy_m=np.array([1.5, 1.5]),
        landing_zones=(
            LandingZone(1, np.array([3.4, 0.8]), 0.25),
            LandingZone(2, np.array([3.4, 2.2]), 0.25),
            LandingZone(3, np.array([2.8, 1.5]), 0.25),
        ),
        active_zone_identifier=1,
        required_reachable=2,
        workspace_size_m=(4.0, 3.0),
        calibration_hash="test",
    )
    cfg = ReachabilityConfig(
        maximum_speed_mps=1.0,
        active_horizon_s=10.0,
        contingency_horizon_s=6.0,
        required_reachable=2,
    )
    bundle = build_reachability_bundle(
        np.zeros(geometry.shape_yx, dtype=bool),
        mission=mission,
        geometry=geometry,
        config=cfg,
        occupancy_version=4,
    )
    return geometry, mission, bundle


def make_filter(required=2):
    return UnifiedContingencyFilter(
        poisson_config=PoissonCBFConfig(alpha=2.0, h_margin=0.05),
        filter_config=ContingencyFilterConfig(solver="scipy", tolerance=1.0e-8),
        maximum_speed_mps=1.0,
        required_reachable=required,
    )


def safe_poisson_sample():
    return FieldSample(
        valid=True,
        h=1.0,
        gradient_xy=np.array([1.0, 0.0]),
        hessian_xy=None,
        laplacian=None,
        point_xy=np.array([1.5, 1.5]),
        reason="ok",
    )


def test_unified_filter_calls_cbf_box_and_returns_feasible_decision() -> None:
    _geometry, mission, bundle = make_bundle()
    result = make_filter().filter(
        position_xy=mission.start_xy_m,
        nominal_velocity_xy=[0.2, 0.0],
        poisson_sample=safe_poisson_sample(),
        reachability_bundle=bundle,
        active_identifier=1,
        available_identifiers=[1, 2, 3],
        tau_active=-10.0,
        tau_active_dot=1.0,
        tau_contingency=-6.0,
    )
    assert result.success, result.to_dict()
    assert result.box_result is not None
    assert result.solver_status == "optimal"
    assert np.linalg.norm(result.safe_velocity_xy) <= 1.0 + 1.0e-8
    assert min(result.residuals.values()) >= -1.0e-7


def test_poisson_constraint_has_zero_auxiliary_columns() -> None:
    _geometry, mission, bundle = make_bundle()
    result = make_filter().filter(
        position_xy=mission.start_xy_m,
        nominal_velocity_xy=[0.2, 0.0],
        poisson_sample=safe_poisson_sample(),
        reachability_bundle=bundle,
        active_identifier=1,
        available_identifiers=[1, 2, 3],
        tau_active=-10.0,
        tau_active_dot=1.0,
        tau_contingency=-6.0,
    )
    assert result.box_result is not None
    names = result.box_result.diagnostics["constraints"]
    index = names.index("poisson_velocity_cbf")
    row = result.box_result.constraint_matrix[index]
    np.testing.assert_allclose(row[2:], [0.0, 0.0])


def test_fewer_than_r_available_targets_causes_hold() -> None:
    _geometry, mission, bundle = make_bundle()
    result = make_filter(required=2).filter(
        position_xy=mission.start_xy_m,
        nominal_velocity_xy=[0.2, 0.0],
        poisson_sample=safe_poisson_sample(),
        reachability_bundle=bundle,
        active_identifier=1,
        available_identifiers=[1],
        tau_active=-10.0,
        tau_active_dot=1.0,
        tau_contingency=-6.0,
    )
    assert not result.success
    np.testing.assert_allclose(result.safe_velocity_xy, [0.0, 0.0])
    assert "available" in result.hold_reason


def test_invalid_poisson_sample_causes_hold() -> None:
    _geometry, mission, bundle = make_bundle()
    invalid = FieldSample(valid=False, reason="occupied_cell")
    result = make_filter().filter(
        position_xy=mission.start_xy_m,
        nominal_velocity_xy=[0.2, 0.0],
        poisson_sample=invalid,
        reachability_bundle=bundle,
        active_identifier=1,
        available_identifiers=[1, 2, 3],
        tau_active=-10.0,
        tau_active_dot=1.0,
        tau_contingency=-6.0,
    )
    assert not result.success
    assert "Poisson" in result.hold_reason


def test_negative_pivot_causes_hold_without_fallback() -> None:
    _geometry, mission, bundle = make_bundle()
    result = make_filter().filter(
        position_xy=[0.1, 1.5],
        nominal_velocity_xy=[0.2, 0.0],
        poisson_sample=safe_poisson_sample(),
        reachability_bundle=bundle,
        active_identifier=1,
        available_identifiers=[1, 2, 3],
        tau_active=-0.2,
        tau_active_dot=1.0,
        tau_contingency=-0.2,
    )
    assert not result.success
    assert result.reachable_count < 2
    np.testing.assert_allclose(result.safe_velocity_xy, [0.0, 0.0])
