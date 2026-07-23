"""Scenario-level figures used in the paper's Approach and Results sections.

These figures are intentionally separate from the generic plotting utilities.
They combine multiple mathematical objects from one completed rollout and do
not recompute controller outputs, which preserves traceability to saved CSV and
JSON artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from experiments.common.plotting import (
    configure_academic_style,
    draw_world_obstacles,
    save_figure,
)


def _event_state(metrics: pd.DataFrame, time_s: float) -> np.ndarray | None:
    if metrics.empty:
        return None
    index = int(np.argmin(np.abs(metrics["time_s"].to_numpy(float) - float(time_s))))
    return metrics.iloc[index][["x", "y", "z"]].to_numpy(float)


def _target_radius(target: Any) -> float:
    metadata = getattr(target, "metadata", {})
    return float(metadata.get("radius_m", 0.6)) if hasattr(metadata, "get") else 0.6


def _plot_target_disks_xy(axis: Any, controller: Any) -> None:
    for identifier, target in controller.targets.items():
        point = np.asarray(target.x_star[:3], dtype=float)
        axis.add_patch(Circle((point[0], point[1]), _target_radius(target), fill=False, linewidth=1.4))
        axis.scatter(point[0], point[1], s=28)
        axis.annotate(identifier, point[:2], xytext=(4, 4), textcoords="offset points")


def _plot_internal_occupancy_xy(axis: Any, world: Any, alpha: float = 0.12) -> None:
    occupancy = np.asarray(world.occupancy, dtype=bool).copy()
    occupancy[0, :, :] = False
    occupancy[-1, :, :] = False
    occupancy[:, 0, :] = False
    occupancy[:, -1, :] = False
    occupancy[:, :, 0] = False
    occupancy[:, :, -1] = False
    projection = np.any(occupancy, axis=2)
    axis.contourf(world.axes[0], world.axes[1], projection.T.astype(float), levels=[0.5, 1.5], alpha=alpha)


def plot_methodology_overview(
    *,
    world: Any,
    field: Any,
    controller: Any,
    metrics: pd.DataFrame,
    direct_line: np.ndarray,
    nominal_path: np.ndarray,
    directory: str | Path,
    dpi: int,
) -> None:
    """Create a six-panel data-to-certificate-to-control methodology figure."""

    if metrics.empty:
        return
    configure_academic_style()
    figure = plt.figure(figsize=(17.0, 10.5))
    grid = figure.add_gridspec(2, 3)

    axis_world = figure.add_subplot(grid[0, 0], projection="3d")
    draw_world_obstacles(axis_world, world, alpha=0.18)
    axis_world.plot(direct_line[:, 0], direct_line[:, 1], direct_line[:, 2], linestyle="--", linewidth=1.2, label="straight line")
    axis_world.plot(nominal_path[:, 0], nominal_path[:, 1], nominal_path[:, 2], linestyle=":", linewidth=1.7, label="A* nominal path")
    axis_world.plot(metrics["x"], metrics["y"], metrics["z"], linewidth=2.1, label="filtered trajectory")
    axis_world.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="(a) Mission geometry and nominal behavior")
    axis_world.set_xlim(0.0, world.extent_m[0]); axis_world.set_ylim(0.0, world.extent_m[1]); axis_world.set_zlim(0.0, world.extent_m[2])
    axis_world.view_init(elev=25, azim=-62)
    axis_world.legend(fontsize=7)

    axis_occ = figure.add_subplot(grid[0, 1])
    z_index = int(np.clip(round(np.median(metrics["z"]) / field.spacing[2]), 1, field.h.shape[2] - 2))
    axis_occ.imshow(
        field.occupancy[:, :, z_index].T,
        origin="lower",
        extent=(0.0, world.extent_m[0], 0.0, world.extent_m[1]),
        cmap="gray_r",
        aspect="equal",
    )
    axis_occ.contour(
        world.axes[0], world.axes[1], field.boundary_mask[:, :, z_index].T.astype(float),
        levels=[0.5], linewidths=1.0,
    )
    axis_occ.set(xlabel="x [m]", ylabel="y [m]", title=f"(b) Occupancy and Dirichlet boundary, z={world.axes[2][z_index]:.1f} m")

    axis_h = figure.add_subplot(grid[0, 2])
    h_image = axis_h.contourf(world.axes[0], world.axes[1], field.h[:, :, z_index].T, levels=45)
    axis_h.plot(metrics["x"], metrics["y"], linewidth=1.5)
    axis_h.set(xlabel="x [m]", ylabel="y [m]", title="(c) Poisson safety function and local HOCBF data", aspect="equal")
    figure.colorbar(h_image, ax=axis_h, shrink=0.76, label="$h_P$")

    axis_roa = figure.add_subplot(grid[1, 0])
    _plot_internal_occupancy_xy(axis_roa, world, alpha=0.08)
    x = np.linspace(0.0, world.extent_m[0], 180)
    y = np.linspace(0.0, world.extent_m[1], 140)
    X, Y = np.meshgrid(x, y)
    for identifier, artifact in controller.clf.artifacts.items():
        P = artifact.P
        d = controller.dimension
        projection = P[:d, :d] - P[:d, d:] @ np.linalg.solve(P[d:, d:], P[d:, :d])
        center = artifact.target.x_star[:d]
        delta = np.stack([X - center[0], Y - center[1], np.zeros_like(X)], axis=-1)
        normalized = np.einsum("...i,ij,...j->...", delta, projection, delta) / artifact.c
        if np.nanmin(normalized) <= 1.0 <= np.nanmax(normalized):
            axis_roa.contour(x, y, normalized, levels=[1.0], linewidths=1.6)
        axis_roa.scatter(center[0], center[1], marker="*", s=65)
        axis_roa.annotate(identifier, center[:2], xytext=(4, 4), textcoords="offset points")
    axis_roa.set(xlabel="x [m]", ylabel="y [m]", title="(d) CLF equilibria and projected attraction regions", xlim=(0.0, world.extent_m[0]), ylim=(0.0, world.extent_m[1]), aspect="equal")

    axis_cert = figure.add_subplot(grid[1, 1])
    for identifier in controller.targets:
        column = f"h_roa_{identifier}"
        if column in metrics:
            axis_cert.plot(metrics["time_s"], metrics[column], label=identifier)
    axis_cert.plot(metrics["time_s"], metrics["contingency_pivot"], linewidth=2.4, label="$\\widetilde h_r$")
    axis_cert.axhline(0.0, linestyle="--", linewidth=1.0)
    axis_cert.set(xlabel="time [s]", ylabel="certificate value", title="(e) Local ROA certificates and r-th pivot")
    axis_cert.legend(ncol=2, fontsize=7)

    axis_control = figure.add_subplot(grid[1, 2])
    axis_control.plot(metrics["time_s"], metrics["poisson_h"], label="$h_P$")
    axis_control.plot(metrics["time_s"], metrics["hocbf_residual"], label="HOCBF residual")
    axis_control.plot(metrics["time_s"], metrics["active_clf_residual"], label="CLF residual")
    axis_control.plot(metrics["time_s"], metrics["intervention_norm"], label="$\\|a_{safe}-a_{nom}\\|$")
    axis_control.axhline(0.0, linestyle="--", linewidth=1.0)
    axis_control.set(xlabel="time [s]", title="(f) Unified filter certificates and intervention")
    axis_control.legend(fontsize=7)

    figure.suptitle("Methodology: occupancy geometry to Poisson-HOCBF, CLF/ROA contingency, and safe control")
    save_figure(figure, directory, "paper_methodology_overview", dpi=dpi, svg=False)


def plot_obstacle_avoidance_result(
    *,
    world: Any,
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    controller: Any,
    summary: Mapping[str, Any],
    direct_line: np.ndarray,
    nominal_path: np.ndarray,
    directory: str | Path,
    dpi: int,
) -> None:
    """Show that a colliding straight line is converted into a safe landing path."""

    if metrics.empty:
        return
    configure_academic_style()
    figure = plt.figure(figsize=(15.8, 10.2))
    grid = figure.add_gridspec(2, 2)
    axis_3d = figure.add_subplot(grid[0, 0], projection="3d")
    draw_world_obstacles(axis_3d, world, alpha=0.20)
    axis_3d.plot(direct_line[:, 0], direct_line[:, 1], direct_line[:, 2], linestyle="--", linewidth=1.2, label="unsafe straight line")
    axis_3d.plot(nominal_path[:, 0], nominal_path[:, 1], nominal_path[:, 2], linestyle=":", linewidth=1.7, label="clearance-aware nominal")
    axis_3d.plot(metrics["x"], metrics["y"], metrics["z"], linewidth=2.4, label="filtered trajectory")
    axis_3d.scatter(metrics["x"].iloc[0], metrics["y"].iloc[0], metrics["z"].iloc[0], s=55, label="start")
    axis_3d.scatter(metrics["x"].iloc[-1], metrics["y"].iloc[-1], metrics["z"].iloc[-1], s=90, marker="*" if summary.get("landed") else "s", label=summary.get("terminal_status", "terminal"))
    axis_3d.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="(a) Obstacle-rich 3-D approach")
    axis_3d.set_xlim(0.0, world.extent_m[0]); axis_3d.set_ylim(0.0, world.extent_m[1]); axis_3d.set_zlim(0.0, world.extent_m[2])
    axis_3d.view_init(elev=25, azim=-62)
    axis_3d.legend(fontsize=7)

    axis_xy = figure.add_subplot(grid[0, 1])
    _plot_internal_occupancy_xy(axis_xy, world, alpha=0.15)
    axis_xy.plot(direct_line[:, 0], direct_line[:, 1], linestyle="--", linewidth=1.2, label="unsafe straight line")
    axis_xy.plot(nominal_path[:, 0], nominal_path[:, 1], linestyle=":", linewidth=1.7, label="nominal path")
    axis_xy.plot(metrics["x"], metrics["y"], linewidth=2.4, label="safe trajectory")
    _plot_target_disks_xy(axis_xy, controller)
    axis_xy.set(xlabel="x [m]", ylabel="y [m]", title="(b) Top view: the direct mission line intersects terrain", xlim=(0.0, world.extent_m[0]), ylim=(0.0, world.extent_m[1]), aspect="equal")
    axis_xy.legend(fontsize=7)

    axis_clearance = figure.add_subplot(grid[1, 0])
    axis_clearance.plot(metrics["time_s"], metrics["obstacle_clearance_m"], label="obstacle clearance")
    axis_clearance.plot(metrics["time_s"], metrics["poisson_h"], label="$h_P$")
    axis_clearance.axhline(0.0, linestyle="--", linewidth=1.0)
    axis_clearance.set(xlabel="time [s]", title="(c) Geometric and functional safety margins")
    axis_clearance.legend()

    axis_control = figure.add_subplot(grid[1, 1])
    axis_control.plot(metrics["time_s"], metrics["distance_to_active_target_m"], label="distance to active landing zone")
    axis_control.plot(metrics["time_s"], metrics["altitude_m"], label="altitude")
    axis_control.plot(metrics["time_s"], metrics["intervention_norm"], label="control intervention")
    for _, event in events.iterrows():
        axis_control.axvline(float(event["time_s"]), linestyle=":", linewidth=0.9)
    axis_control.set(xlabel="time [s]", title="(d) Descent, target progress, and safety-filter action")
    axis_control.legend()

    figure.suptitle("Obstacle avoidance and terminal landing performance in the Mars-analog world")
    save_figure(figure, directory, "paper_obstacle_avoidance_and_landing", dpi=dpi, svg=False)


def plot_landing_terminal_zoom(
    *,
    world: Any,
    metrics: pd.DataFrame,
    controller: Any,
    summary: Mapping[str, Any],
    simulation_config: Mapping[str, Any],
    directory: str | Path,
    dpi: int,
) -> None:
    """Verify touchdown position and speed inside the selected landing region."""

    if metrics.empty:
        return
    configure_academic_style()
    target_id = str(summary["final_target"])
    target = controller.targets[target_id]
    center = np.asarray(target.x_star[:3], dtype=float)
    radius = _target_radius(target)
    recent_start = max(float(metrics["time_s"].iloc[-1]) - 8.0, 0.0)
    recent = metrics[metrics["time_s"] >= recent_start]

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    axis_xy = axes[0, 0]
    _plot_internal_occupancy_xy(axis_xy, world, alpha=0.10)
    axis_xy.add_patch(Circle((center[0], center[1]), radius, fill=False, linewidth=2.0, label="landing region"))
    axis_xy.plot(recent["x"], recent["y"], linewidth=2.2, label="final approach")
    axis_xy.scatter(center[0], center[1], marker="*", s=90, label=f"{target_id} equilibrium")
    axis_xy.scatter(metrics["x"].iloc[-1], metrics["y"].iloc[-1], marker="X", s=75, label="terminal state")
    margin = 2.5 * radius
    axis_xy.set(xlabel="x [m]", ylabel="y [m]", title="(a) Touchdown position inside landing disk", xlim=(center[0] - margin, center[0] + margin), ylim=(center[1] - margin, center[1] + margin), aspect="equal")
    axis_xy.legend(fontsize=7)

    terminal_time = float(summary.get("duration_s", metrics["time_s"].iloc[-1]))
    terminal_state = np.asarray(summary.get("final_state", metrics.iloc[-1][["x", "y", "z", "vx", "vy", "vz"]]), dtype=float)
    terminal_error = float(summary.get("final_target_error_m", np.linalg.norm(terminal_state[:3] - center)))
    terminal_speed = float(summary.get("final_speed_mps", np.linalg.norm(terminal_state[3:6])))

    axes[0, 1].plot(recent["time_s"], recent["altitude_m"], label="sampled altitude")
    axes[0, 1].scatter(terminal_time, terminal_state[2], marker="*", s=65, label="terminal state")
    axes[0, 1].axhline(center[2], linestyle="--", linewidth=1.0, label="landing elevation")
    axes[0, 1].set(xlabel="time [s]", ylabel="z [m]", title="(b) Terminal descent profile")
    axes[0, 1].legend()

    axes[1, 0].plot(recent["time_s"], recent["distance_to_active_target_m"], label="sampled position error")
    axes[1, 0].scatter(terminal_time, terminal_error, marker="*", s=65, label="terminal error")
    axes[1, 0].axhline(float(simulation_config["landing_position_tolerance_m"]), linestyle="--", linewidth=1.0, label="position tolerance")
    axes[1, 0].set(xlabel="time [s]", ylabel="distance [m]", title="(c) Landing-position condition")
    axes[1, 0].legend()

    axes[1, 1].plot(recent["time_s"], recent["speed"], label="sampled speed")
    axes[1, 1].scatter(terminal_time, terminal_speed, marker="*", s=65, label="terminal speed")
    axes[1, 1].axhline(float(simulation_config["landing_speed_tolerance_mps"]), linestyle="--", linewidth=1.0, label="speed tolerance")
    axes[1, 1].set(xlabel="time [s]", ylabel="speed [m/s]", title="(d) Landing-speed condition")
    axes[1, 1].legend()

    outcome = "successful touchdown" if summary.get("landed") else str(summary.get("terminal_status", "terminal state"))
    figure.suptitle(f"Terminal verification at {target_id}: {outcome}")
    save_figure(figure, directory, "paper_terminal_landing_verification", dpi=dpi)


def plot_contingency_timeline(
    *,
    world: Any,
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    controller: Any,
    summary: Mapping[str, Any],
    directory: str | Path,
    dpi: int,
) -> None:
    """Expose target loss, retargeting, ROA preservation, and terminal outcome."""

    if metrics.empty:
        return
    configure_academic_style()
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    axis_xy = axes[0, 0]
    _plot_internal_occupancy_xy(axis_xy, world, alpha=0.12)
    target_ids = list(dict.fromkeys(metrics["active_target"].astype(str).tolist()))
    cmap = plt.get_cmap("tab10")
    for index, target_id in enumerate(target_ids):
        selected = metrics[metrics["active_target"].astype(str) == target_id]
        axis_xy.plot(selected["x"], selected["y"], linewidth=2.3, color=cmap(index % 10), label=f"toward {target_id}")
    _plot_target_disks_xy(axis_xy, controller)
    for _, event in events.iterrows():
        point = _event_state(metrics, float(event["time_s"]))
        if point is None:
            continue
        marker = {"target_failed": "X", "active_target_switched": "D", "hold": "s", "landed": "*"}.get(str(event["event"]), "o")
        axis_xy.scatter(point[0], point[1], marker=marker, s=70)
    axis_xy.set(xlabel="x [m]", ylabel="y [m]", title="(a) Retargeting trajectory and mission events", xlim=(0.0, world.extent_m[0]), ylim=(0.0, world.extent_m[1]), aspect="equal")
    axis_xy.legend(fontsize=7)

    axis_roa = axes[0, 1]
    for identifier in controller.targets:
        column = f"h_roa_{identifier}"
        if column in metrics:
            axis_roa.plot(metrics["time_s"], metrics[column], label=identifier)
    axis_roa.plot(metrics["time_s"], metrics["contingency_pivot"], linewidth=2.5, label="$\\widetilde h_r$")
    axis_roa.axhline(0.0, linestyle="--", linewidth=1.0)
    axis_roa.set(xlabel="time [s]", title="(b) Local attraction margins and r-th pivot")
    axis_roa.legend(ncol=2, fontsize=7)

    axis_counts = axes[1, 0]
    axis_counts.step(metrics["time_s"], metrics["available_count"], where="post", label="available sites")
    axis_counts.step(metrics["time_s"], metrics["certified_count"], where="post", label="certified sites")
    required = int(controller.contingency.config.required_certified) if controller.contingency.enabled else 0
    axis_counts.axhline(required, linestyle="--", linewidth=1.2, label=f"required r={required}")
    axis_counts.set(xlabel="time [s]", ylabel="count", title="(c) Availability and contingency requirement")
    axis_counts.legend()

    axis_distance = axes[1, 1]
    for identifier in controller.targets:
        column = f"distance_{identifier}_m"
        if column in metrics:
            axis_distance.plot(metrics["time_s"], metrics[column], label=identifier)
    for _, event in events.iterrows():
        axis_distance.axvline(float(event["time_s"]), linestyle=":", linewidth=0.9)
    axis_distance.set(xlabel="time [s]", ylabel="distance [m]", title="(d) Distance to candidate landing equilibria")
    axis_distance.legend(ncol=2, fontsize=7)

    figure.suptitle(
        f"Contingency response: {summary.get('failure_count', 0)} failures, "
        f"{summary.get('switch_count', 0)} switches, terminal status={summary.get('terminal_status')}"
    )
    save_figure(figure, directory, "paper_contingency_timeline", dpi=dpi)


def plot_scenario_comparison(
    *,
    world: Any,
    results: Mapping[str, Any],
    directory: str | Path,
    dpi: int,
) -> None:
    """Compare baseline landing, successful diversion, and contingency exhaustion."""

    if not results:
        return
    configure_academic_style()
    names = [name for name in ("baseline", "single_failure", "sequential_failure") if name in results]
    figure, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 5.4), squeeze=False)
    for axis, name in zip(axes.ravel(), names, strict=True):
        result = results[name]
        metrics = result.metrics
        _plot_internal_occupancy_xy(axis, world, alpha=0.12)
        if not metrics.empty:
            target_ids = list(dict.fromkeys(metrics["active_target"].astype(str).tolist()))
            cmap = plt.get_cmap("tab10")
            for index, target_id in enumerate(target_ids):
                selected = metrics[metrics["active_target"].astype(str) == target_id]
                axis.plot(selected["x"], selected["y"], linewidth=2.0, color=cmap(index % 10), label=target_id)
            marker = "*" if result.summary.get("landed") else "s" if result.summary.get("terminal_status") == "hold" else "x"
            axis.scatter(metrics["x"].iloc[-1], metrics["y"].iloc[-1], marker=marker, s=70)
        for target in world.targets:
            point = np.asarray(target.x_star[:3], dtype=float)
            axis.add_patch(Circle((point[0], point[1]), _target_radius(target), fill=False, linewidth=1.0))
        axis.set(xlabel="x [m]", ylabel="y [m]", title=f"{name.replace('_', ' ')}\n{result.summary.get('terminal_status')}", xlim=(0.0, world.extent_m[0]), ylim=(0.0, world.extent_m[1]), aspect="equal")
        axis.legend(fontsize=7)
    figure.suptitle("Controlled scenario matrix: landing, certified diversion, and graceful HOLD")
    save_figure(figure, directory, "paper_scenario_comparison", dpi=dpi)
