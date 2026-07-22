"""OpenCV dashboard with geometry isolated from numerical text and histories."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


_BACKGROUND = (250, 250, 250)
_PANEL = (255, 255, 255)
_TEXT = (32, 36, 43)
_MUTED = (105, 112, 122)
_GOOD = (38, 150, 82)
_WARN = (35, 145, 215)
_BAD = (52, 63, 210)
_BLUE = (196, 105, 32)


def _put(
    image: np.ndarray,
    text: str,
    point: tuple[int, int],
    *,
    scale: float = 0.52,
    color: tuple[int, int, int] = _TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _panel(canvas: np.ndarray, box: tuple[int, int, int, int], title: str) -> np.ndarray:
    x0, y0, x1, y1 = box
    cv2.rectangle(canvas, (x0, y0), (x1, y1), _PANEL, thickness=-1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (222, 225, 230), thickness=1)
    _put(canvas, title, (x0 + 14, y0 + 25), scale=0.58, thickness=2)
    return canvas[y0:y1, x0:x1]


def metric_to_pixel(point_xy: Sequence[float], workspace: Sequence[float], shape: Sequence[int]) -> tuple[int, int]:
    width_m, height_m = float(workspace[0]), float(workspace[1])
    height_px, width_px = int(shape[0]), int(shape[1])
    x = int(round(float(point_xy[0]) / width_m * (width_px - 1)))
    y = int(round(float(point_xy[1]) / height_m * (height_px - 1)))
    return x, y


def _draw_arrow(
    image: np.ndarray,
    origin_xy: Sequence[float],
    vector_xy: Sequence[float],
    workspace: Sequence[float],
    color: tuple[int, int, int],
    scale_m: float = 0.28,
) -> None:
    vector = np.asarray(vector_xy, dtype=float).reshape(2)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-10:
        return
    unit = vector / norm
    origin = np.asarray(origin_xy, dtype=float)
    destination = origin + scale_m * unit
    cv2.arrowedLine(
        image,
        metric_to_pixel(origin, workspace, image.shape[:2]),
        metric_to_pixel(destination, workspace, image.shape[:2]),
        color,
        2,
        cv2.LINE_AA,
        tipLength=0.25,
    )


def _plot_history(
    panel: np.ndarray,
    histories: Mapping[str, Sequence[float]],
    *,
    zero_line: bool = False,
) -> None:
    height, width = panel.shape[:2]
    # Reserve a dedicated title strip (drawn by ``_panel``) and a separate
    # legend strip before the plotting area.  This prevents labels from
    # overlapping the panel title in 16:9 live dashboards.
    left, top, right, bottom = 48, 66, width - 14, height - 28
    cv2.rectangle(panel, (left, top), (right, bottom), (232, 234, 238), 1)
    values = [np.asarray(series, dtype=float) for series in histories.values() if len(series)]
    if not values:
        return
    finite = np.concatenate([series[np.isfinite(series)] for series in values if np.any(np.isfinite(series))])
    if finite.size == 0:
        return
    low, high = float(np.min(finite)), float(np.max(finite))
    if zero_line:
        low, high = min(low, 0.0), max(high, 0.0)
    if abs(high - low) < 1.0e-10:
        low -= 1.0
        high += 1.0
    colors = [_BLUE, (60, 160, 70), (50, 120, 210), (170, 80, 180), (90, 90, 90)]
    for (name, series), color in zip(histories.items(), colors, strict=False):
        array = np.asarray(series, dtype=float)
        if array.size < 2:
            continue
        finite_mask = np.isfinite(array)
        if np.count_nonzero(finite_mask) < 2:
            continue
        x = np.linspace(left, right, array.size)[finite_mask]
        finite_values = array[finite_mask]
        y = bottom - (finite_values - low) / (high - low) * (bottom - top)
        points = np.column_stack([x, y]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel, [points], False, color, 2, cv2.LINE_AA)
    if zero_line and low <= 0.0 <= high:
        y0 = int(round(bottom - (0.0 - low) / (high - low) * (bottom - top)))
        cv2.line(panel, (left, y0), (right, y0), (130, 130, 130), 1, cv2.LINE_AA)
    _put(panel, f"{high:.3g}", (5, top + 5), scale=0.40, color=_MUTED)
    _put(panel, f"{low:.3g}", (5, bottom), scale=0.40, color=_MUTED)
    # Draw the legend on its own baseline below the panel title.  Spacing is
    # computed from the available panel width rather than using a fixed
    # increment, so labels remain separated when the dashboard is resized.
    entries = list(histories.items())
    legend_y = 48
    if entries:
        usable_width = max(1, right - left - 12)
        slot_width = max(96, usable_width // len(entries))
        for index, ((name, _), color) in enumerate(zip(entries, colors, strict=False)):
            legend_x = left + 6 + index * slot_width
            cv2.line(panel, (legend_x, legend_y - 4), (legend_x + 16, legend_y - 4), color, 2, cv2.LINE_AA)
            _put(panel, name, (legend_x + 20, legend_y), scale=0.39, color=_MUTED)


def render_dashboard(
    *,
    rectified_bgr: np.ndarray,
    occupancy_yx: np.ndarray,
    workspace_size_m: Sequence[float],
    targets: Mapping[str, np.ndarray],
    availability: Mapping[str, bool],
    active_target: str,
    path_xy: Sequence[np.ndarray],
    state: np.ndarray,
    nominal_control: np.ndarray,
    safe_control: np.ndarray,
    box_status: Mapping[str, tuple[str, str]],
    metrics: Mapping[str, float],
    histories: Mapping[str, deque[float]],
    canvas_size: Sequence[int],
    warning: str | None,
) -> np.ndarray:
    """Render one 16:9 dashboard frame without covering the camera view with text."""

    width, height = int(canvas_size[0]), int(canvas_size[1])
    canvas = np.full((height, width, 3), _BACKGROUND, dtype=np.uint8)
    margin = 18
    camera_box = (margin, 52, int(width * 0.64), int(height * 0.69))
    camera_panel = _panel(canvas, camera_box, "Rectified perception and filtered motion")
    view_y0 = 34
    view = cv2.resize(
        rectified_bgr,
        (camera_panel.shape[1] - 24, camera_panel.shape[0] - view_y0 - 14),
        interpolation=cv2.INTER_LINEAR,
    )
    occupancy = cv2.resize(
        np.asarray(occupancy_yx, dtype=np.uint8) * 255,
        (view.shape[1], view.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    red = np.zeros_like(view)
    red[..., 2] = occupancy
    view = cv2.addWeighted(view, 1.0, red, 0.28, 0.0)

    for identifier, point in targets.items():
        pixel = metric_to_pixel(point[:2], workspace_size_m, view.shape[:2])
        color = _GOOD if availability.get(identifier, False) else _BAD
        radius_px = max(5, int(round(0.18 / float(workspace_size_m[0]) * view.shape[1])))
        cv2.circle(view, pixel, radius_px, color, 2, cv2.LINE_AA)
        _put(view, identifier, (pixel[0] + 5, pixel[1] - 7), scale=0.43, color=color, thickness=2)
    if len(path_xy) >= 2:
        points = np.array(
            [metric_to_pixel(point, workspace_size_m, view.shape[:2]) for point in path_xy],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(view, [points], False, _BLUE, 2, cv2.LINE_AA)
    vehicle = metric_to_pixel(state[:2], workspace_size_m, view.shape[:2])
    cv2.circle(view, vehicle, 7, (20, 80, 235), -1, cv2.LINE_AA)
    _draw_arrow(view, state[:2], nominal_control, workspace_size_m, (90, 90, 90))
    _draw_arrow(view, state[:2], safe_control, workspace_size_m, _GOOD)
    camera_panel[view_y0:view_y0 + view.shape[0], 12:12 + view.shape[1]] = view

    status_box = (int(width * 0.65), 52, width - margin, int(height * 0.69))
    status = _panel(canvas, status_box, "Certificate-box health and numerical state")
    y = 55
    for name in ("Poisson", "HOCBF", "CLF", "Contingency", "Filter"):
        state_name, detail = box_status.get(name, ("unknown", ""))
        color = _GOOD if state_name == "ready" else (_WARN if state_name in {"disabled", "updating"} else _BAD)
        cv2.circle(status, (20, y - 4), 6, color, -1, cv2.LINE_AA)
        _put(status, f"{name}: {state_name}", (34, y), scale=0.50, thickness=2)
        if detail:
            _put(status, detail[:54], (34, y + 20), scale=0.40, color=_MUTED)
        y += 50
    _put(status, f"active target: {active_target}", (18, y + 5), scale=0.52, thickness=2)
    y += 35
    metric_lines = [
        ("h_P", metrics.get("poisson_h", np.nan)),
        ("HOCBF residual", metrics.get("hocbf_residual", np.nan)),
        ("active V", metrics.get("active_V", np.nan)),
        ("active h_ROA", metrics.get("active_h_roa", np.nan)),
        ("r-th pivot", metrics.get("contingency_pivot", np.nan)),
        ("certified zones", metrics.get("certified_count", np.nan)),
        ("intervention", metrics.get("intervention_norm", np.nan)),
        ("filter time [ms]", metrics.get("filter_time_ms", np.nan)),
        ("field age [s]", metrics.get("field_age_s", np.nan)),
    ]
    for label, value in metric_lines:
        text = "--" if not np.isfinite(value) else f"{value:.4g}"
        _put(status, label, (18, y), scale=0.44, color=_MUTED)
        _put(status, text, (status.shape[1] - 135, y), scale=0.45, thickness=2)
        y += 25

    chart_top = int(height * 0.71)
    chart_height = height - chart_top - margin
    chart_width = (width - 3 * margin) // 2
    left_chart = _panel(canvas, (margin, chart_top, margin + chart_width, height - margin), "Safety and stability histories")
    right_chart = _panel(canvas, (2 * margin + chart_width, chart_top, width - margin, height - margin), "Contingency and control histories")
    _plot_history(
        left_chart,
        {
            "h_P": histories.get("poisson_h", ()),
            "HOCBF": histories.get("hocbf_residual", ()),
            "h_ROA": histories.get("active_h_roa", ()),
        },
        zero_line=True,
    )
    _plot_history(
        right_chart,
        {
            "pivot": histories.get("contingency_pivot", ()),
            "control": histories.get("intervention_norm", ()),
            "omega": histories.get("omega", ()),
        },
        zero_line=True,
    )

    title = "Live Poisson-HOCBF, CLF, and contingency landing"
    _put(canvas, title, (margin, 34), scale=0.80, thickness=2)
    if warning:
        cv2.rectangle(canvas, (int(width * 0.37), 8), (width - margin, 42), _BAD, thickness=-1)
        _put(canvas, warning[:90], (int(width * 0.38), 32), scale=0.55, color=(255, 255, 255), thickness=2)
    return canvas
