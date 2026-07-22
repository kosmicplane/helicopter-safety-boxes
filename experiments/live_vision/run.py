#!/usr/bin/env python3
"""Run the online/video perception-to-Poisson-to-CLF landing experiment."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import textwrap
from time import monotonic, sleep
from typing import Any

import cv2
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from safety_box_core import BoxStatus
from experiments.common.calibration import assume_top_down_calibration, rectify_image
from experiments.common.cli import (
    add_common_arguments,
    load_config_from_arguments,
    make_output_directory,
    save_run_metadata,
)
from experiments.common.controller import LandingController
from experiments.common.coordinates import GridGeometry
from experiments.common.occupancy import (
    TemporalOccupancyFilter,
    build_occupancy_maps,
    changed_fraction,
)
from experiments.common.plotting import (
    configure_exports,
    plot_clf_phase_portraits,
    plot_clf_roa_projections,
    plot_contingency_maps,
    plot_live_summary,
    plot_poisson_planes,
)
from experiments.common.poisson_field import compute_poisson_field
from experiments.common.segmentation import segment_image
from experiments.live_vision.dashboard import render_dashboard
from experiments.live_vision.worker import FieldSnapshot, LatestPoissonWorker
from experiments.static_image.pipeline import PlanarWorld, build_targets


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Process a camera, stream, or video with live Poisson-HOCBF and CLF/ROA filtering."
    )
    add_common_arguments(result)
    result.add_argument("--source", help="Video path, URL, or integer camera index.")
    result.add_argument("--max-frames", type=int, help="Maximum number of processed frames.")
    result.add_argument("--display", action="store_true", help="Show the dashboard while processing.")
    result.add_argument("--no-failure", action="store_true", help="Disable the configured target failure.")
    return result


def parse_source(value: str) -> str | int:
    stripped = value.strip()
    return int(stripped) if stripped.isdigit() and len(stripped) <= 2 else stripped


def controller_from_config(config: dict, targets: tuple, output: Path) -> LandingController:
    simulation = config["experiments"]["live_vision"]["simulation"]
    return LandingController(
        dimension=2,
        targets=targets,
        box_config=config["boxes"],
        filter_config=config["filter"],
        artifact_directory=output / "clf_artifacts",
        maximum_acceleration=float(simulation["maximum_acceleration_mps2"]),
        maximum_speed_component=float(simulation["maximum_speed_component_mps"]),
        nominal_position_gain=float(simulation["nominal_position_gain"]),
        nominal_velocity_gain=float(simulation["nominal_velocity_gain"]),
    )


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _is_live_source(source: str | int) -> bool:
    """Return True for camera indices and network streams."""

    if isinstance(source, int):
        return True
    value = str(source).strip().lower()
    return value.startswith(("http://", "https://", "rtsp://", "rtmp://"))


def _fit_image_to_region(
    image: np.ndarray,
    *,
    width: int,
    height: int,
    background_color: tuple[int, int, int] = (18, 18, 18),
) -> np.ndarray:
    """Letterbox an image into a fixed region without changing its aspect ratio."""

    if width <= 0 or height <= 0:
        raise ValueError("Display region dimensions must be positive.")
    source_height, source_width = image.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Cannot display an empty image.")

    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )

    canvas = np.full((height, width, 3), background_color, dtype=np.uint8)
    x_offset = (width - resized_width) // 2
    y_offset = (height - resized_height) // 2
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return canvas


def _render_setup_canvas(
    image: np.ndarray,
    *,
    status: str,
    occupied_fraction: float,
    background_ready: bool,
    canvas_size_px: tuple[int, int] = (1280, 720),
    panel_width_px: int = 390,
) -> np.ndarray:
    """Render a camera panel and a separate, readable operator panel."""

    canvas_width, canvas_height = (int(v) for v in canvas_size_px)
    panel_width = min(max(int(panel_width_px), 320), canvas_width - 320)
    camera_width = canvas_width - panel_width

    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)
    camera_panel = _fit_image_to_region(
        image,
        width=camera_width,
        height=canvas_height,
    )
    canvas[:, :camera_width] = camera_panel

    cv2.rectangle(
        canvas,
        (camera_width, 0),
        (canvas_width - 1, canvas_height - 1),
        (248, 248, 248),
        -1,
    )
    cv2.line(
        canvas,
        (camera_width, 0),
        (camera_width, canvas_height),
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )

    left = camera_width + 26
    right = canvas_width - 24
    text_width_px = max(220, right - left)

    cv2.putText(
        canvas,
        "LIVE VISION SETUP",
        (left, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (28, 28, 28),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Interactive calibration",
        (left, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )

    y = 132
    commands = [
        ("B", "Capture empty background"),
        ("SPACE", "Start after obstacles appear"),
        ("R", "Reset background"),
        ("Q / ESC", "Quit"),
    ]
    for key, description in commands:
        cv2.rectangle(canvas, (left, y - 23), (left + 76, y + 8), (32, 32, 32), -1)
        cv2.putText(
            canvas,
            key,
            (left + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (250, 250, 250),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            description,
            (left + 92, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
        y += 48

    y += 12
    cv2.line(canvas, (left, y), (right, y), (205, 205, 205), 1, cv2.LINE_AA)
    y += 38

    state_label = "BACKGROUND READY" if background_ready else "WAITING FOR BACKGROUND"
    state_color = (46, 130, 76) if background_ready else (42, 96, 180)
    cv2.putText(
        canvas,
        state_label,
        (left, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        state_color,
        2,
        cv2.LINE_AA,
    )
    y += 38

    # Wrap status according to the actual panel width rather than camera resolution.
    approximate_character_width = 9
    wrap_width = max(24, text_width_px // approximate_character_width)
    for line in textwrap.wrap(status, width=wrap_width):
        cv2.putText(
            canvas,
            line,
            (left, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )
        y += 27

    y += 24
    cv2.putText(
        canvas,
        "DETECTED OCCUPANCY",
        (left, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (95, 95, 95),
        1,
        cv2.LINE_AA,
    )
    y += 35
    cv2.putText(
        canvas,
        f"{100.0 * occupied_fraction:.3f}%",
        (left, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.86,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    footer = "Red overlay = detected obstacle mask"
    cv2.putText(
        canvas,
        footer,
        (left, canvas_height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (100, 100, 100),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _interactive_background_setup(
    *,
    capture: cv2.VideoCapture,
    first_frame: np.ndarray,
    calibration: Any,
    segmentation_config: dict[str, Any],
    output_data_directory: Path,
    minimum_start_occupied_fraction: float,
    setup_canvas_size_px: tuple[int, int] = (1280, 720),
) -> tuple[np.ndarray, np.ndarray]:
    """Capture the empty scene in the UI, then wait for obstacles.

    Controls
    --------
    B
        Capture or recapture the current rectified frame as the empty background.
    SPACE
        Start only after a background exists and obstacles are detected.
    R
        Clear the captured background and return to the empty-scene step.
    Q / ESC
        Abort.
    """

    window_name = "Live vision setup"
    window_flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    if hasattr(cv2, "WINDOW_GUI_NORMAL"):
        window_flags |= cv2.WINDOW_GUI_NORMAL
    cv2.namedWindow(window_name, window_flags)
    cv2.resizeWindow(window_name, *setup_canvas_size_px)
    background: np.ndarray | None = None
    raw_frame = first_frame
    status = "Keep the workspace empty and press B to capture the background."

    while True:
        rectified = rectify_image(raw_frame, calibration)
        occupied_fraction = 0.0
        preview = rectified.copy()

        if background is not None:
            segmentation = segment_image(
                rectified,
                segmentation_config,
                base_directory=REPOSITORY_ROOT,
                background_reference=background,
                allow_interactive=False,
            )
            mask = segmentation.clean_mask > 0
            occupied_fraction = float(np.mean(mask))
            if np.any(mask):
                tint = np.zeros_like(preview)
                tint[..., 2] = 255
                preview[mask] = cv2.addWeighted(
                    preview[mask], 0.35, tint[mask], 0.65, 0.0
                )
            status = (
                f"Obstacle mask: {100.0 * occupied_fraction:.3f}% occupied. "
                "Place obstacles, then press SPACE."
            )

        setup_canvas = _render_setup_canvas(
            preview,
            status=status,
            occupied_fraction=occupied_fraction,
            background_ready=background is not None,
            canvas_size_px=setup_canvas_size_px,
        )
        cv2.imshow(window_name, setup_canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("b"):
            background = rectified.copy()
            output_data_directory.mkdir(parents=True, exist_ok=True)
            saved_path = output_data_directory / "background_reference.png"
            if not cv2.imwrite(str(saved_path), background):
                raise RuntimeError(f"Could not save captured background to {saved_path}")
            status = "Background captured. Place obstacles and press SPACE."
        elif key == ord("r"):
            background = None
            status = "Background cleared. Keep the workspace empty and press B."
        elif key == 32:  # SPACE
            if background is None:
                status = "Capture the empty background first by pressing B."
            elif occupied_fraction < minimum_start_occupied_fraction:
                status = (
                    "No obstacles are detected yet. Place obstacles or lower the "
                    "segmentation threshold, then press SPACE again."
                )
            else:
                cv2.destroyWindow(window_name)
                return background, raw_frame
        elif key in {27, ord("q")}:
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt("Live-vision setup cancelled by the operator.")

        success, next_frame = capture.read()
        if not success or next_frame is None:
            cv2.destroyWindow(window_name)
            raise RuntimeError("The live source stopped during background setup.")
        raw_frame = next_frame


def run(arguments: argparse.Namespace) -> Path:
    config, _ = load_config_from_arguments(arguments)
    configure_exports(
        pdf=bool(config.get("visualization", {}).get("save_pdf", True)),
        svg=bool(config.get("visualization", {}).get("save_svg", True)),
    )
    output = make_output_directory(config=config, mode="live_vision", explicit=arguments.output)
    save_run_metadata(config=config, output=output, mode="live_vision", command=sys.argv)
    figures = output / "figures"
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)

    experiment = config["experiments"]["live_vision"]
    source_value = arguments.source or str(experiment["source"])
    source = parse_source(source_value)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video/camera source: {source_value}")
    success, first = capture.read()
    if not success or first is None:
        capture.release()
        raise RuntimeError("The source did not provide an initial frame.")

    workspace = tuple(float(value) for value in experiment["workspace_size_m"])
    output_size = tuple(int(value) for value in experiment["output_size_px"])
    calibration = assume_top_down_calibration(
        first.shape,
        output_size_px=output_size,
        workspace_size_m=workspace,
    )
    interactive_setup = (
        bool(experiment.get("capture_background_interactively", True))
        and bool(arguments.display)
        and _is_live_source(source)
    )
    if interactive_setup:
        background, first = _interactive_background_setup(
            capture=capture,
            first_frame=first,
            calibration=calibration,
            segmentation_config=dict(experiment["segmentation"]),
            output_data_directory=data,
            minimum_start_occupied_fraction=float(
                experiment.get("minimum_start_occupied_fraction", 0.0005)
            ),
            setup_canvas_size_px=tuple(
                int(value)
                for value in experiment.get("setup_canvas_size_px", [1280, 720])
            ),
        )
    else:
        background_path = (REPOSITORY_ROOT / str(experiment["background_file"])).resolve()
        background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if background is None:
            capture.release()
            raise FileNotFoundError(
                f"Could not read background image: {background_path}. "
                "For a camera or network stream, run with --display and capture "
                "the background interactively with B."
            )
        background = rectify_image(background, calibration)

    field_size = tuple(
        int(value) for value in experiment.get("poisson_grid_shape_px", output_size)
    )
    field_width_px, field_height_px = field_size
    geometry = GridGeometry(
        width_m=workspace[0],
        height_m=workspace[1],
        nx=field_width_px,
        ny=field_height_px,
    )
    targets = build_targets(experiment)
    world = PlanarWorld(extent_m=workspace, targets=targets)
    controller = controller_from_config(config, targets, output)
    target_positions = {target.identifier: target.x_star[:2] for target in targets}
    simulation = experiment["simulation"]
    active_target = str(simulation["initial_target"])
    availability = {target.identifier: True for target in targets}
    state = np.asarray(experiment["start_state"], dtype=float).reshape(4)

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = source_fps if np.isfinite(source_fps) and source_fps > 1.0 else float(config["visualization"]["live_video_fps"])
    dt_s = 1.0 / fps
    maximum_frames = int(arguments.max_frames or experiment["maximum_frames"])
    output_video = output / "live_dashboard.mp4"
    canvas_size = tuple(int(value) for value in config["visualization"]["live_dashboard_size_px"])
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        canvas_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_video}")

    temporal = TemporalOccupancyFilter(mode="majority", window_size=3)
    worker = LatestPoissonWorker(
        spacing_xy=(geometry.dx, geometry.dy),
        poisson_config=config["boxes"]["poisson"],
    )
    previous_submitted: np.ndarray | None = None
    field_version = 0
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    path: list[np.ndarray] = [state[:2].copy()]
    history_length = int(config["visualization"]["live_history_length"])
    histories = {
        key: deque(maxlen=history_length)
        for key in (
            "poisson_h",
            "hocbf_residual",
            "active_h_roa",
            "contingency_pivot",
            "intervention_norm",
            "omega",
        )
    }
    failed = False
    switched = False
    last_dashboard: np.ndarray | None = None
    last_rectified = rectify_image(first, calibration)
    last_occupancy = np.zeros((field_height_px, field_width_px), dtype=bool)
    last_valid_snapshot: FieldSnapshot | None = None

    try:
        frame = first
        for frame_index in range(maximum_frames):
            if frame_index > 0:
                success, frame = capture.read()
                if not success or frame is None:
                    break
            rectified = rectify_image(frame, calibration)
            last_rectified = rectified
            segmentation = segment_image(
                rectified,
                dict(experiment["segmentation"]),
                base_directory=REPOSITORY_ROOT,
                background_reference=background,
                allow_interactive=False,
            )
            occupancy_maps = build_occupancy_maps(
                segmentation.clean_mask,
                grid_shape_yx=(field_height_px, field_width_px),
                workspace_size_m=workspace,
                robot_radius_m=float(experiment["robot_radius_m"]),
                perception_margin_m=float(experiment["perception_margin_m"]),
            )
            occupancy_yx = temporal.update(occupancy_maps.inflated_occupancy)
            last_occupancy = occupancy_yx
            occupancy_xy = occupancy_yx.T
            occupancy_is_empty = not bool(np.any(occupancy_xy))
            change = changed_fraction(previous_submitted, occupancy_xy)
            update_interval = int(experiment["poisson_update_interval_frames"])
            if not occupancy_is_empty and (
                previous_submitted is None
                or frame_index % update_interval == 0
                or change >= float(experiment["occupancy_change_trigger"])
            ):
                field_version += 1
                worker.submit(field_version, occupancy_xy)
                previous_submitted = occupancy_xy.copy()

            worker_snapshot = worker.latest()
            if worker_snapshot is not None and worker_snapshot.valid:
                last_valid_snapshot = worker_snapshot
            # Make the first dashboard immediately usable and retain the last
            # valid field while a newer latest-only solve is in progress.
            if last_valid_snapshot is None and not occupancy_is_empty:
                initial_field = compute_poisson_field(
                    occupancy_xy,
                    spacing=(geometry.dx, geometry.dy),
                    config=config["boxes"]["poisson"],
                )
                last_valid_snapshot = FieldSnapshot(
                    version=0,
                    created_monotonic_s=monotonic(),
                    field=initial_field,
                    occupancy_xy=occupancy_xy.copy(),
                    error=None,
                )
            snapshot = worker_snapshot if worker_snapshot is not None else last_valid_snapshot
            now = monotonic()
            usable_snapshot = last_valid_snapshot
            field_age = (
                float("inf")
                if usable_snapshot is None
                else now - usable_snapshot.created_monotonic_s
            )
            valid_field = (
                usable_snapshot is not None
                and usable_snapshot.valid
                and field_age <= float(experiment["maximum_field_age_s"])
            )
            field = None if not valid_field else usable_snapshot.field

            if (
                not arguments.no_failure
                and not failed
                and frame_index >= int(simulation["failure_frame"])
            ):
                failed_target = str(simulation["failed_target"])
                availability[failed_target] = False
                failed = True
                events.append(
                    {
                        "frame": frame_index,
                        "time_s": frame_index * dt_s,
                        "event": "target_unavailable",
                        "target_id": failed_target,
                        "reason": "configured online target failure",
                    }
                )

            sample = None if field is None else field.sample(state[:2])
            control_step = controller.step(
                state=state,
                time_s=frame_index * dt_s,
                version=0 if usable_snapshot is None else int(usable_snapshot.version),
                active_target=active_target,
                availability=availability,
                safety_sample=sample,
                dt_s=dt_s,
            )
            if control_step.target_switched:
                switched = True
                events.append(
                    {
                        "frame": frame_index,
                        "time_s": frame_index * dt_s,
                        "event": "active_target_switched",
                        "target_id": control_step.active_target,
                        "reason": "maximum certified ROA margin",
                    }
                )
            active_target = control_step.active_target
            acceleration = (
                np.asarray(control_step.safe_control, dtype=float)
                if control_step.status is BoxStatus.READY
                else np.zeros(2)
            )
            position = state[:2]
            velocity = state[2:]
            next_position = position + velocity * dt_s + 0.5 * acceleration * dt_s**2
            next_velocity = velocity + acceleration * dt_s
            if geometry.contains_xy(next_position):
                row, column = geometry.nearest_index_yx(next_position, clip=True)
                if occupancy_yx[row, column]:
                    next_position = position.copy()
                    next_velocity = np.zeros_like(velocity)
            else:
                next_position = np.clip(next_position, [0.0, 0.0], np.asarray(workspace))
                next_velocity = np.zeros_like(velocity)
            state = np.concatenate([next_position, next_velocity])
            path.append(state[:2].copy())

            active_evaluation = control_step.evaluations.get(active_target)
            contingency = control_step.contingency
            filter_result = control_step.filter_result
            row_data: dict[str, Any] = {
                "frame": frame_index,
                "time_s": frame_index * dt_s,
                "x": float(position[0]),
                "y": float(position[1]),
                "vx": float(velocity[0]),
                "vy": float(velocity[1]),
                "a_nom_x": float(control_step.nominal_control[0]),
                "a_nom_y": float(control_step.nominal_control[1]),
                "a_safe_x": float(acceleration[0]),
                "a_safe_y": float(acceleration[1]),
                "active_target": active_target,
                "status": control_step.status.value,
                "poisson_h": np.nan if sample is None else float(sample.h),
                "hocbf_residual": _safe_float(control_step.hocbf_residual),
                "active_V": np.nan if active_evaluation is None else float(active_evaluation.V),
                "active_h_roa": np.nan if active_evaluation is None else float(active_evaluation.h_roa),
                "active_clf_residual": _safe_float(control_step.clf_residual),
                "contingency_pivot": np.nan if contingency is None else float(contingency.pivot),
                "certified_count": 0 if contingency is None else int(contingency.certified_count),
                "omega": float(control_step.omega),
                "intervention_norm": float(control_step.intervention_norm),
                "solver_time_s": 0.0 if filter_result is None else float(filter_result.solve_time_s),
                "field_version": -1 if usable_snapshot is None else int(usable_snapshot.version),
                "field_age_s": field_age,
                "occupancy_change_fraction": change,
            }
            for target_id, evaluation in control_step.evaluations.items():
                row_data[f"V_{target_id}"] = float(evaluation.V)
                row_data[f"h_roa_{target_id}"] = float(evaluation.h_roa)
                row_data[f"available_{target_id}"] = int(availability.get(target_id, False))
            records.append(row_data)
            for key in histories:
                histories[key].append(_safe_float(row_data.get(key)))

            warning = control_step.message if control_step.status is not BoxStatus.READY else None
            if occupancy_is_empty:
                warning = (
                    "No obstacles are present in the current mask. Poisson updating is "
                    "paused until a nonempty occupancy map is detected."
                )
            box_status = {
                "Poisson": (
                    "hold" if occupancy_is_empty else (
                        "ready" if valid_field else ("updating" if usable_snapshot is None else "hold")
                    ),
                    "empty obstacle mask" if occupancy_is_empty else (
                        "" if valid_field else (
                            "waiting for first field"
                            if usable_snapshot is None
                            else ((worker_snapshot.error if worker_snapshot is not None else None) or "stale field")
                        )
                    ),
                ),
                "HOCBF": ("ready" if controller.cbf.enabled and sample is not None else "disabled" if not controller.cbf.enabled else "hold", "environmental safety"),
                "CLF": ("ready" if controller.clf.enabled else "disabled", "active landing equilibrium"),
                "Contingency": ("ready" if controller.contingency.enabled else "disabled", f"r={controller.contingency.config.required_certified}" if controller.contingency.enabled else ""),
                "Filter": (control_step.status.value, "verified affine residuals"),
            }
            dashboard_metrics = {
                "poisson_h": row_data["poisson_h"],
                "hocbf_residual": row_data["hocbf_residual"],
                "active_V": row_data["active_V"],
                "active_h_roa": row_data["active_h_roa"],
                "contingency_pivot": row_data["contingency_pivot"],
                "certified_count": float(row_data["certified_count"]),
                "intervention_norm": row_data["intervention_norm"],
                "filter_time_ms": 1.0e3 * row_data["solver_time_s"],
                "field_age_s": field_age,
            }
            last_dashboard = render_dashboard(
                rectified_bgr=rectified,
                occupancy_yx=occupancy_yx,
                workspace_size_m=workspace,
                targets=target_positions,
                availability=availability,
                active_target=active_target,
                path_xy=path,
                state=state,
                nominal_control=control_step.nominal_control,
                safe_control=acceleration,
                box_status=box_status,
                metrics=dashboard_metrics,
                histories=histories,
                canvas_size=canvas_size,
                warning=warning,
            )
            writer.write(last_dashboard)
            if arguments.display:
                cv2.imshow("Helicopter safety dashboard", last_dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break
    finally:
        worker.stop()
        capture.release()
        writer.release()
        if arguments.display:
            cv2.destroyAllWindows()

    metrics_frame = pd.DataFrame(records)
    events_frame = pd.DataFrame(events)
    metrics_frame.to_csv(data / "live_metrics.csv", index=False)
    events_frame.to_csv(data / "events.csv", index=False)
    if last_dashboard is not None:
        cv2.imwrite(str(output / "last_dashboard.png"), last_dashboard)
    cv2.imwrite(str(data / "last_rectified_frame.png"), last_rectified)
    cv2.imwrite(str(data / "last_occupancy.png"), last_occupancy.astype(np.uint8) * 255)
    summary = {
        "frames": int(len(metrics_frame)),
        "duration_s": float(len(metrics_frame) * dt_s),
        "source": source_value,
        "output_video": output_video.name,
        "target_failed": failed,
        "target_switched": switched,
        "final_target": active_target,
        "minimum_poisson_h": None if metrics_frame.empty else _safe_float(metrics_frame["poisson_h"].min()),
        "minimum_hocbf_residual": None if metrics_frame.empty else _safe_float(metrics_frame["hocbf_residual"].min()),
        "minimum_contingency_pivot": None if metrics_frame.empty else _safe_float(metrics_frame["contingency_pivot"].min()),
        "mean_filter_time_ms": None if metrics_frame.empty else _safe_float(1.0e3 * metrics_frame["solver_time_s"].mean()),
    }
    (data / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    dpi = int(config["visualization"]["dpi"])
    plot_live_summary(metrics_frame, directory=figures, dpi=dpi)
    plot_clf_roa_projections(controller=controller, world=world, directory=figures, dpi=dpi)
    plot_clf_phase_portraits(controller=controller, target_id=str(simulation["initial_target"]), directory=figures, dpi=dpi)
    plot_contingency_maps(controller=controller, world=world, directory=figures, dpi=dpi, grid_points=75 if arguments.quick else 120)
    final_snapshot = worker.latest()
    if final_snapshot is None or not final_snapshot.valid:
        final_snapshot = last_valid_snapshot
    if final_snapshot is not None and final_snapshot.valid:
        plot_poisson_planes(field=final_snapshot.field, directory=figures, dpi=dpi)
        final_snapshot.field.save(data, stem="last_poisson_field")
    print(f"OUTPUT_DIRECTORY={output}")
    return output


def main() -> int:
    run(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
