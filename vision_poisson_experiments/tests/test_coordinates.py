"""Coordinate and interpolation tests guard the most dangerous silent failure."""

from __future__ import annotations

import numpy as np

from common.coordinates import (
    GridFieldSampler,
    GridGeometry,
    gradient_yx_to_xy,
    hessian_yx_to_xy,
    xy_to_yx,
    yx_to_xy,
)


def test_xy_yx_round_trip_and_derivative_permutations() -> None:
    point_xy = np.array([1.25, 2.75])
    assert np.allclose(xy_to_yx(point_xy), [2.75, 1.25])
    assert np.allclose(yx_to_xy(xy_to_yx(point_xy)), point_xy)

    gradient_yx = np.array([3.0, 7.0])
    assert np.allclose(gradient_yx_to_xy(gradient_yx), [7.0, 3.0])

    hessian_yx = np.array([[11.0, 13.0], [13.0, 17.0]])
    expected_xy = np.array([[17.0, 13.0], [13.0, 11.0]])
    assert np.allclose(hessian_yx_to_xy(hessian_yx), expected_xy)


def test_bilinear_sample_matches_directional_derivative() -> None:
    geometry = GridGeometry(width_m=4.0, height_m=3.0, nx=41, ny=31)
    x = np.linspace(0.0, geometry.width_m, geometry.nx)
    y = np.linspace(0.0, geometry.height_m, geometry.ny)
    xx, yy = np.meshgrid(x, y)

    # Affine h makes both bilinear interpolation and its gradient exact.
    h = 1.0 + 2.5 * xx - 1.75 * yy
    grad_yx = np.empty(h.shape + (2,), dtype=float)
    grad_yx[..., 0] = -1.75  # dh/dy
    grad_yx[..., 1] = 2.5    # dh/dx
    hessian_yx = np.zeros(h.shape + (2, 2), dtype=float)
    sampler = GridFieldSampler(geometry, h, grad_yx, hessian_yx)

    point = np.array([1.37, 1.18])
    direction = np.array([0.6, -0.8])
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-4
    center = sampler.sample_xy(point)
    plus = sampler.sample_xy(point + epsilon * direction)
    minus = sampler.sample_xy(point - epsilon * direction)
    assert center.valid and plus.valid and minus.valid

    numeric = (plus.h - minus.h) / (2.0 * epsilon)
    gradient_prediction = float(center.grad_xy @ direction)
    assert np.isclose(numeric, gradient_prediction, rtol=1.0e-7, atol=1.0e-8)
    assert np.allclose(center.grad_xy, [2.5, -1.75])
    assert np.allclose(center.hessian_xy, np.zeros((2, 2)))
