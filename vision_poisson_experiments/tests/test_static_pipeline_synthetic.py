"""Headless synthetic smoke test for the complete static pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common.poisson_runner import central_stencil_residual
from common.static_pipeline import run_static_experiment


def _synthetic_image(path: Path) -> None:
    image = np.full((180, 270, 3), 240, dtype=np.uint8)
    cv2.rectangle(image, (105, 35), (145, 125), (35, 35, 35), -1)
    cv2.circle(image, (205, 130), 22, (45, 45, 45), -1)
    assert cv2.imwrite(str(path), image)


def test_static_pipeline_outputs_and_poisson_consistency(tmp_path: Path) -> None:
    image_path = tmp_path / "workspace.png"
    _synthetic_image(image_path)
    output = tmp_path / "run"
    config = {
        "workspace": {
            "width_m": 4.5,
            "height_m": 3.0,
            "rectified_width_px": 270,
            "rectified_height_px": 180,
            "grid_nx": 36,
            "grid_ny": 24,
        },
        "calibration": {"mode": "assume_top_down"},
        "segmentation": {
            "mode": "hsv",
            "hsv_lower": [0, 0, 0],
            "hsv_upper": [180, 255, 100],
            "blur_kernel": 1,
            "open_kernel": 0,
            "close_kernel": 0,
            "min_component_area_px": 10,
            "fill_holes": True,
        },
        "occupancy": {"robot_radius_m": 0.06, "perception_margin_m": 0.03},
        "poisson": {
            "forcing_methods": ["constant"],
            "cbf_forcing_method": "constant",
            "solver": "sparse_direct",
            "compare_solvers": False,
            "compute_hessian": True,
            "compute_laplacian_check": True,
            "constant": {"c": 1.0},
            "validation_residual_tolerance": 1.0e-7,
        },
        "visualization": {"dpi": 60},
    }
    report = run_static_experiment(
        image_path=image_path,
        config=config,
        output_directory=output,
        assume_top_down=True,
        headless=True,
        run_cbf=False,
    )
    record = report.poisson.selected
    result = record.result
    assert record.validation.valid
    assert np.max(np.abs(result.h[result.boundary_mask])) <= 1.0e-10
    _residual, stats = central_stencil_residual(
        result.h,
        result.forcing,
        result.solve_mask,
        report.geometry.spacing_yx,
    )
    assert stats["max_abs"] is not None
    assert stats["max_abs"] < 1.0e-8
    assert (output / "perception_and_occupancy.npz").is_file()
    assert (output / "poisson" / "constant" / "result.npz").is_file()
    assert (output / "poisson" / "constant" / "figures" / "31_method_dashboard.png").is_file()
    assert (output / "experiment_summary.json").is_file()
    assert plt.get_fignums() == []
