"""Headless video test for asynchronous latest-map-only execution."""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common.live_pipeline import LivePoissonPipeline


def _write_video(path: Path) -> None:
    width, height = 180, 120
    background = np.full((height, width, 3), 235, dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height))
    assert writer.isOpened()
    try:
        for index in range(24):
            frame = background.copy()
            if index >= 4:
                cv2.rectangle(frame, (72, 25), (105, 88), (35, 35, 35), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_live_file_produces_metrics_and_bounded_queue(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    _write_video(video)
    output = tmp_path / "live_output"
    config = {
        "workspace": {
            "width_m": 4.5,
            "height_m": 3.0,
            "rectified_width_px": 180,
            "rectified_height_px": 120,
            "grid_nx": 30,
            "grid_ny": 20,
        },
        "calibration": {"mode": "assume_top_down"},
        "segmentation": {
            "mode": "background_reference",
            "live_modes": ["background_reference", "hsv"],
            "background_threshold": 20,
            "background_blur_kernel": 3,
            "blur_kernel": 1,
            "open_kernel": 0,
            "close_kernel": 3,
            "min_component_area_px": 8,
            "fill_holes": True,
            "hsv_lower": [0, 0, 0],
            "hsv_upper": [180, 255, 100],
        },
        "occupancy": {"robot_radius_m": 0.05, "perception_margin_m": 0.03},
        "temporal_filter": {"method": "ema", "ema_alpha": 0.7, "ema_threshold": 0.45},
        "poisson": {
            "forcing_method": "constant",
            "live_forcing_methods": ["constant"],
            "solver": "conjugate_gradient",
            "changed_fraction_threshold": 0.001,
            "laplacian_check_every_n_solves": 2,
            "compute_hessian": False,
            "constant": {"c": 1.0},
            "conjugate_gradient": {"tolerance": 1.0e-7, "max_iter": 800},
            "validation_residual_tolerance": 1.0e-5,
        },
        "camera_motion": {"check_interval_frames": 1000},
        "recording": {"record_dashboard_video": False, "snapshot_interval_frames": 0},
        "dashboard": {"panel_size": [120, 80], "stale_field_threshold_ms": 1000.0},
        "stream": {"reconnect_attempts": 1, "reconnect_delay_s": 0.01},
        "cbf_demo": {"enabled": False},
    }

    tracemalloc.start()
    report = LivePoissonPipeline(
        source=str(video),
        config=config,
        output_directory=output,
        headless=True,
        max_frames=24,
    ).run()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert report.frames_processed >= 20
    assert report.metrics_path.is_file()
    assert report.summary_path.is_file()
    summary = json.loads(report.summary_path.read_text())
    assert summary["worker"]["worker_queue_max_observed"] <= 1
    assert summary["worker"]["accepted_solves"] >= 1
    assert (output / "last_valid_field.npz").is_file()
    assert peak < 100 * 1024 * 1024
    assert plt.get_fignums() == []
