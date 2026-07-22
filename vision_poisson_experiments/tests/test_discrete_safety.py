"""Discrete virtual-marker safety checks and step backtracking."""

from __future__ import annotations

import numpy as np

from common.coordinates import FieldSample, GridGeometry
from common.discrete_safety import backtracked_safe_step


class LinearSampler:
    """Minimal sampler whose safety value decreases with x."""

    def sample(self, point_xy):
        point = np.asarray(point_xy, dtype=float)
        return FieldSample(
            valid=True,
            h=float(0.2 - point[0]),
            gradient_xy=np.array([-1.0, 0.0]),
            point_xy=point,
            reason="ok",
        )


def test_safe_discrete_step_is_accepted_without_backtracking() -> None:
    geometry = GridGeometry(width_m=1.0, height_m=1.0, nx=21, ny=21)
    result = backtracked_safe_step(
        position_xy=[0.05, 0.5],
        velocity_xy=[0.1, 0.0],
        nominal_dt_s=0.1,
        geometry=geometry,
        inflated_occupancy=np.zeros(geometry.shape_yx, dtype=bool),
        poisson_sampler=LinearSampler(),
        h_margin=0.0,
    )
    assert result.accepted
    assert result.backtracks == 0
    np.testing.assert_allclose(result.position_xy, [0.06, 0.5])


def test_step_is_halved_until_poisson_margin_is_satisfied() -> None:
    geometry = GridGeometry(width_m=1.0, height_m=1.0, nx=21, ny=21)
    result = backtracked_safe_step(
        position_xy=[0.05, 0.5],
        velocity_xy=[1.0, 0.0],
        nominal_dt_s=0.2,
        maximum_dt_s=0.2,
        geometry=geometry,
        inflated_occupancy=np.zeros(geometry.shape_yx, dtype=bool),
        poisson_sampler=LinearSampler(),
        h_margin=0.05,
    )
    assert result.accepted
    assert result.backtracks >= 1
    assert result.position_xy[0] <= 0.15 + 1.0e-8


def test_collision_has_no_unsafe_fallback() -> None:
    geometry = GridGeometry(width_m=1.0, height_m=1.0, nx=21, ny=21)
    occupancy = np.zeros(geometry.shape_yx, dtype=bool)
    occupancy[:, 10] = True
    result = backtracked_safe_step(
        position_xy=[0.45, 0.5],
        velocity_xy=[1.0, 0.0],
        nominal_dt_s=0.2,
        maximum_dt_s=0.2,
        maximum_backtracks=2,
        geometry=geometry,
        inflated_occupancy=occupancy,
        poisson_sampler=LinearSampler(),
        h_margin=-1.0,
    )
    assert not result.accepted
    np.testing.assert_allclose(result.position_xy, [0.45, 0.5])
    assert result.accepted_dt_s == 0.0
