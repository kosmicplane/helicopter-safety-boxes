"""OpenCV overlays and low-rate Matplotlib figures for live contingency planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .hj_reachability import ReachabilityBundle, pivot_and_reachable_count_fields
from .mission_setup import MissionDefinition, draw_mission_overlay, metric_to_rectified_pixel
from .target_manager import TargetManager
from .unified_contingency_filter import UnifiedFilterResult


def _polyline_pixels(
    path_xy: np.ndarray,
    *,
    image_shape_yx: tuple[int, int],
    workspace_size_m: tuple[float, float],
) -> np.ndarray:
    """Convert a metric path into an OpenCV polyline array."""

    path = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    if path.shape[0] == 0:
        return np.empty((0, 1, 2), dtype=np.int32)
    pixels = np.asarray(
        [
            metric_to_rectified_pixel(
                point,
                image_shape_yx=image_shape_yx,
                workspace_size_m=workspace_size_m,
            )
            for point in path
        ],
        dtype=np.int32,
    )
    return pixels.reshape(-1, 1, 2)


def draw_contingency_overlay(
    rectified_bgr: np.ndarray,
    *,
    mission: MissionDefinition,
    target_manager: TargetManager,
    position_xy: Iterable[float],
    path_xy: np.ndarray | None,
    nominal_velocity_xy: Iterable[float] | None,
    filter_result: UnifiedFilterResult | None,
    lookahead_xy: Iterable[float] | None = None,
    show_hjr: bool = True,
    show_path: bool = True,
    arrow_scale_px_per_mps: float = 150.0,
    occupancy_version: int | None = None,
    field_age_s: float | None = None,
) -> np.ndarray:
    """Draw mission geometry, path, vehicle, commands, and certificate values."""

    canvas = draw_mission_overlay(
        rectified_bgr,
        mission,
        active_zone_identifier=target_manager.active_identifier,
        zone_states=target_manager.state_labels(),
    )
    image_shape = canvas.shape[:2]
    if show_path and path_xy is not None:
        polyline = _polyline_pixels(path_xy, image_shape_yx=image_shape, workspace_size_m=mission.workspace_size_m)
        if polyline.shape[0] >= 2:
            cv2.polylines(canvas, [polyline], False, (255, 255, 0), 2, cv2.LINE_AA)
    position = np.asarray(position_xy, dtype=float).reshape(2)
    origin = metric_to_rectified_pixel(
        position,
        image_shape_yx=image_shape,
        workspace_size_m=mission.workspace_size_m,
    ).astype(int)
    cv2.circle(canvas, tuple(origin), 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(origin), 9, (0, 0, 0), 2, cv2.LINE_AA)

    if lookahead_xy is not None:
        lookahead = metric_to_rectified_pixel(
            np.asarray(lookahead_xy, dtype=float),
            image_shape_yx=image_shape,
            workspace_size_m=mission.workspace_size_m,
        ).astype(int)
        cv2.circle(canvas, tuple(lookahead), 4, (255, 255, 0), -1, cv2.LINE_AA)

    if nominal_velocity_xy is not None:
        nominal = np.asarray(nominal_velocity_xy, dtype=float).reshape(2)
        nominal_end = tuple(np.rint(origin + float(arrow_scale_px_per_mps) * nominal).astype(int))
        cv2.arrowedLine(canvas, tuple(origin), nominal_end, (0, 165, 255), 2, cv2.LINE_AA, tipLength=0.25)
    if filter_result is not None:
        safe = np.asarray(filter_result.safe_velocity_xy, dtype=float).reshape(2)
        safe_end = tuple(np.rint(origin + float(arrow_scale_px_per_mps) * safe).astype(int))
        color = (0, 255, 0) if filter_result.success else (0, 0, 255)
        cv2.arrowedLine(canvas, tuple(origin), safe_end, color, 2, cv2.LINE_AA, tipLength=0.25)

    rows: list[str] = [
        f"active=LZ-{target_manager.active_identifier}",
        f"reachable={len(target_manager.reachable_identifiers())}/{mission.required_reachable}",
    ]
    if occupancy_version is not None:
        rows.append(f"version={occupancy_version}")
    if field_age_s is not None and np.isfinite(field_age_s):
        rows.append(f"field age={1000.0 * field_age_s:.0f} ms")
    if filter_result is not None:
        rows.extend(
            [
                f"pivot={filter_result.pivot:.3f}" if np.isfinite(filter_result.pivot) else "pivot=-inf",
                f"h={filter_result.poisson_h_raw:.3f}" if filter_result.poisson_h_raw is not None else "h=invalid",
                f"solver={filter_result.solver_status}",
                f"omega=({filter_result.omega_active:.2e}, {filter_result.omega_contingency:.2e})",
            ]
        )
        if show_hjr:
            values_text = ", ".join(
                f"V{identifier}={value:.2f}" if np.isfinite(value) else f"V{identifier}=X"
                for identifier, value in sorted(filter_result.reachability_values.items())
            )
            rows.append(values_text)
        if filter_result.hold_reason:
            rows.append("CONTINGENCY REQUIREMENT LOST - HOLD")
            rows.append(filter_result.hold_reason[:80])

    panel_width = min(canvas.shape[1], 620)
    panel_height = min(canvas.shape[0], 24 + 22 * len(rows))
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0.0)
    for index, text in enumerate(rows):
        color = (0, 0, 255) if "HOLD" in text else (255, 255, 255)
        cv2.putText(canvas, text, (8, 20 + 21 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return canvas


def save_reachability_snapshot_figures(
    output_directory: str | Path,
    *,
    mission: MissionDefinition,
    bundle: ReachabilityBundle,
    active_identifier: int,
    tau_active: float,
    tau_contingency: float,
    maximum_speed_mps: float,
    required_reachable: int,
    available_identifiers: Iterable[int],
    path_xy: np.ndarray | None = None,
    dpi: int = 160,
) -> None:
    """Save geodesic, HJ, pivot, reachable-count, and path diagnostics."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    extent = bundle.geometry.extent_xy
    image_extent = [extent[0], extent[1], extent[3], extent[2]]
    identifiers = sorted(bundle.fields)
    columns = 2
    rows = int(np.ceil(len(identifiers) / columns))

    for kind, tau, prefix in (
        ("distance", tau_contingency, "geodesic_distance"),
        ("active", tau_active, "HJ_active"),
        ("contingency", tau_contingency, "HJ_contingency"),
    ):
        fig, axes = plt.subplots(rows, columns, figsize=(11, 4.5 * rows), squeeze=False, constrained_layout=True)
        for ax, identifier in zip(axes.ravel(), identifiers):
            field = bundle.fields[identifier]
            if kind == "distance":
                data = np.ma.masked_where(~field.finite_mask, field.distance_m)
                title = f"LZ-{identifier} geodesic D"
                label = "distance [m]"
            else:
                value = field.value_field(tau, maximum_speed_mps)
                data = np.ma.masked_where(~np.isfinite(value), value)
                title = f"LZ-{identifier} V at tau={tau:.2f} s"
                label = "V"
            image = ax.imshow(data, origin="upper", extent=image_extent, aspect="auto", cmap="coolwarm" if kind != "distance" else "viridis")
            if kind != "distance" and np.any(np.isfinite(np.asarray(data.filled(np.nan)))):
                try:
                    x = np.linspace(extent[0], extent[1], bundle.geometry.nx)
                    y = np.linspace(extent[2], extent[3], bundle.geometry.ny)
                    ax.contour(x, y, np.asarray(data.filled(np.nan)), levels=[0.0], colors="black", linewidths=1.2)
                except Exception:
                    pass
            zone = mission.zone_by_identifier(identifier)
            ax.plot(zone.center_xy_m[0], zone.center_xy_m[1], marker="D", color="white", markeredgecolor="black")
            ax.set_title(title + (" ACTIVE" if identifier == active_identifier else ""))
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            fig.colorbar(image, ax=ax, label=label)
        for ax in axes.ravel()[len(identifiers) :]:
            ax.set_axis_off()
        fig.savefig(output / f"{prefix}_all_zones.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    pivot, count, _values = pivot_and_reachable_count_fields(
        bundle,
        tau=tau_contingency,
        maximum_speed_mps=maximum_speed_mps,
        required_reachable=required_reachable,
        available_identifiers=available_identifiers,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    pivot_masked = np.ma.masked_where(~np.isfinite(pivot), pivot)
    image = axes[0].imshow(pivot_masked, origin="upper", extent=image_extent, aspect="auto", cmap="coolwarm")
    try:
        axes[0].contour(
            np.linspace(extent[0], extent[1], bundle.geometry.nx),
            np.linspace(extent[2], extent[3], bundle.geometry.ny),
            np.asarray(pivot_masked.filled(np.nan)),
            levels=[0.0],
            colors="black",
        )
    except Exception:
        pass
    axes[0].set_title(f"r={required_reachable} combinatorial pivot")
    fig.colorbar(image, ax=axes[0], label="pivot")
    count_image = axes[1].imshow(count, origin="upper", extent=image_extent, aspect="auto", cmap="viridis", vmin=0, vmax=mission.p)
    axes[1].set_title("Number of reachable landing zones")
    fig.colorbar(count_image, ax=axes[1], label="reachable count")
    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if path_xy is not None and np.asarray(path_xy).size:
            path = np.asarray(path_xy)
            ax.plot(path[:, 0], path[:, 1], color="cyan", linewidth=2.0)
    fig.savefig(output / "pivot_and_reachable_count.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


__all__ = ["draw_contingency_overlay", "save_reachability_snapshot_figures"]
