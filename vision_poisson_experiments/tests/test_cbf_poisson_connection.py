"""Verify the CBF consumes numerical Poisson samples and enforces its residual."""

from __future__ import annotations

import numpy as np

from common.coordinates import GridFieldSampler, GridGeometry
from common.package_bootstrap import ensure_safety_boxes_importable
from common.poisson_runner import run_poisson_once


def test_real_poisson_gradient_filters_command_toward_boundary() -> None:
    ensure_safety_boxes_importable()
    from cbf_safety_box import CBFBox, CBFBoxConfig, SafetySample, SystemState

    geometry = GridGeometry(width_m=6.0, height_m=4.0, nx=61, ny=41)
    occupancy = np.zeros(geometry.shape_yx, dtype=bool)
    occupancy[10:31, 28:34] = True
    record = run_poisson_once(
        occupancy,
        spacing_yx=geometry.spacing_yx,
        settings={"constant": {"c": 1.0}, "compute_hessian": True, "validation_residual_tolerance": 1.0e-7},
        forcing_method="constant",
        solver="sparse_direct",
    )
    result = record.result
    sampler = GridFieldSampler(geometry, result.h, result.grad_h, result.hessian_h, result.laplacian_h)

    # Pick a low-h solve node with a nonzero gradient, then command strongly in
    # the direction of decreasing h. This guarantees nominal CBF violation.
    candidates = np.argwhere(result.solve_mask & (np.linalg.norm(result.grad_h, axis=-1) > 1.0e-5))
    h_values = result.h[tuple(candidates.T)]
    row, col = candidates[int(np.argmin(h_values))]
    point_xy = geometry.index_to_xy(int(row), int(col))
    sample = sampler.sample_xy(point_xy)
    assert sample.valid
    u_nom = -20.0 * sample.grad_xy

    alpha = 3.0
    nominal_residual = float(sample.grad_xy @ u_nom + alpha * sample.h)
    assert nominal_residual < 0.0
    box = CBFBox(CBFBoxConfig(mode="velocity", solver="closed_form", alpha=alpha))
    filtered = box.filter_control(
        SystemState(position=point_xy),
        SafetySample(h=sample.h, grad_h=sample.grad_xy, hessian_h=sample.hessian_xy),
        u_nom,
    )
    residual = float(sample.grad_xy @ filtered.u_safe + alpha * sample.h)
    assert filtered.was_filtered
    assert filtered.solver_status == "optimal"
    assert residual >= -1.0e-8
    assert np.linalg.norm(filtered.u_safe - u_nom) > 0.0


def test_poisson_cbf_trajectory_avoids_synthetic_obstacles() -> None:
    """A nominal collision is corrected by the sampled Poisson CBF trajectory."""

    import cv2

    from common.cbf_demo import CBFSimulationConfig, run_cbf_comparison
    from common.occupancy import compute_occupancy_products

    geometry = GridGeometry(width_m=6.0, height_m=4.5, nx=96, ny=72)
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(mask, (285, 90), (350, 350), 255, -1)
    cv2.circle(mask, (455, 145), 46, 255, -1)
    polygon = np.asarray([[105, 305], [190, 275], [215, 380], [130, 400]], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)

    occupancy = compute_occupancy_products(
        mask,
        geometry,
        robot_radius_m=0.12,
        perception_margin_m=0.08,
    )
    record = run_poisson_once(
        occupancy.inflated,
        spacing_yx=geometry.spacing_yx,
        settings={
            "constant": {"c": 1.0},
            "compute_hessian": True,
            "compute_laplacian_check": True,
            "validation_residual_tolerance": 1.0e-7,
        },
        forcing_method="constant",
        solver="sparse_direct",
    )
    comparison = run_cbf_comparison(
        record.result,
        grid_spacing_yx=geometry.spacing_yx,
        start_xy=[0.4, 3.8],
        goal_xy=[5.6, 0.5],
        config=CBFSimulationConfig(
            alpha=0.2,
            solver="closed_form",
            goal_gain=1.1,
            dt_s=0.02,
            maximum_steps=1300,
            maximum_speed_mps=0.8,
            goal_tolerance_m=0.1,
            enforce_control_bounds=False,
        ),
    )

    assert comparison.nominal.collided
    assert comparison.nominal.status == "collision_with_inflated_occupancy"
    assert comparison.safe.reached_goal
    assert not comparison.safe.collided
    assert comparison.safe.status == "goal_reached"
    residuals = [
        row["explicit_cbf_residual"]
        for row in comparison.safe.rows
        if np.isfinite(row["explicit_cbf_residual"])
    ]
    assert residuals
    assert min(residuals) >= -1.0e-7
