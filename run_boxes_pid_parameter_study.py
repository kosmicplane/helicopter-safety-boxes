#!/usr/bin/env python3
"""
Parameter-study demo for poisson_safety_box + cbf_safety_box.

This script is designed for paper/report figures. It does not use ROS, PX4,
Gazebo, or any external simulator.

What it compares:
    1. Effect of CBF alpha on the safe trajectory and diagnostics.
    2. Effect of Poisson forcing method on the safety field, trajectory, and
       Poisson solve time.
    3. Optional full grid of forcing methods x alpha values.

Pipeline:
    fake 2D obstacles
        -> occupancy matrix
        -> Poisson safety field h, grad_h
        -> PID/P nominal velocity
        -> CBF velocity filter
        -> point-mass integration
        -> CSV metrics + paper-ready plots

Default output folder:
    outputs/parameter_study_pid_cbf

Usage from the Helicopter parent folder:
    python run_boxes_pid_parameter_study.py \
        --output-dir outputs/paper_parameter_study

Recommended paper command:
    python run_boxes_pid_parameter_study.py \
        --output-dir outputs/paper_parameter_study \
        --alphas 0.25,0.5,1,2,5,10 \
        --forcing-methods constant,distance,average_flux,guidance \
        --fixed-forcing guidance \
        --fixed-alpha 2.0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Force non-interactive plotting so the script saves figures without opening
# many windows.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

# Robust imports when both boxes live under the same parent folder.
sys.path.insert(0, str(ROOT / "poisson_safety_box"))
sys.path.insert(0, str(ROOT / "cbf_safety_box"))

from poisson_safety_box.api import PoissonSafetyBox
from poisson_safety_box.config import PoissonBoxConfig

from cbf_safety_box.api import CBFBox
from cbf_safety_box.config import CBFBoxConfig
from cbf_safety_box.state import SystemState
from cbf_safety_box.safety_data.sample import SafetySample


# ============================================================
# Data containers
# ============================================================

@dataclass
class World2D:
    """Container for the synthetic 2D occupancy world."""

    Lx: float
    Ly: float
    nx: int
    ny: int
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    dx: float
    dy: float
    occupancy: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    obstacle_specs: List[dict]


@dataclass
class CaseResult:
    """Result for one parameter-study simulation case."""

    forcing_method: str
    alpha: float
    solver: str
    poisson_solve_time_s: float
    simulation_time_s: float
    total_time_s: float
    steps: int
    success: bool
    collision: bool
    final_distance: float
    path_length: float
    filtered_fraction: float
    min_h: float
    min_cbf_residual: float
    mean_correction: float
    max_correction: float
    final_x: float
    final_y: float
    data_file: str


@dataclass
class TrajectoryData:
    """Raw arrays for one simulation case."""

    trajectory: np.ndarray
    h_history: np.ndarray
    residual_history: np.ndarray
    u_nom_history: np.ndarray
    u_safe_history: np.ndarray
    filtered_history: np.ndarray
    correction_history: np.ndarray
    success: bool
    collision: bool


# ============================================================
# Utility functions
# ============================================================

def parse_float_list(text: str) -> List[float]:
    """Parse comma-separated floats from a CLI argument."""
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def parse_str_list(text: str) -> List[str]:
    """Parse comma-separated strings from a CLI argument."""
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one string value.")
    return values


def add_circle_obstacle(occupancy: np.ndarray, X: np.ndarray, Y: np.ndarray,
                        center: Tuple[float, float], radius: float) -> None:
    """Mark a circular obstacle inside a 2D occupancy matrix."""
    cx, cy = center
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
    occupancy[mask] = True


def saturate_vector(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Saturate a vector by Euclidean norm."""
    n = float(np.linalg.norm(v))
    if n <= max_norm or n < 1e-12:
        return v
    return max_norm * v / n


def sample_nearest(result, p: np.ndarray, dx: float, dy: float) -> Tuple[float, np.ndarray, Tuple[int, int]]:
    """Sample h and grad_h from the Poisson result using nearest neighbor."""
    h_grid = result.h
    grad_grid = result.grad_h

    nx, ny = h_grid.shape[:2]

    i = int(np.clip(round(float(p[0]) / dx), 0, nx - 1))
    j = int(np.clip(round(float(p[1]) / dy), 0, ny - 1))

    h_value = float(h_grid[i, j])
    grad_h = np.asarray(grad_grid[i, j], dtype=float)

    return h_value, grad_h, (i, j)


def path_length(trajectory: np.ndarray) -> float:
    """Compute geometric path length from a trajectory."""
    if len(trajectory) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))


def safe_min(values: np.ndarray, default: float = float("nan")) -> float:
    """Return min for non-empty arrays."""
    return float(np.min(values)) if len(values) else default


def safe_mean(values: np.ndarray, default: float = float("nan")) -> float:
    """Return mean for non-empty arrays."""
    return float(np.mean(values)) if len(values) else default


# ============================================================
# World and solver
# ============================================================

def create_world(nx: int = 140, ny: int = 140) -> World2D:
    """
    Create a compact 2D world with obstacles close to the direct start-goal path.

    This world is intentionally simple enough to run quickly but nontrivial
    enough that different CBF/Poisson settings can change the trajectory.
    """
    Lx = 12.0
    Ly = 12.0

    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    X, Y = np.meshgrid(x, y, indexing="ij")
    occupancy = np.zeros((nx, ny), dtype=bool)

    obstacle_specs = [
        {"type": "circle", "center": (5.4, 5.7), "radius": 1.25},
        {"type": "circle", "center": (7.3, 7.1), "radius": 1.00},
        {"type": "circle", "center": (3.7, 7.4), "radius": 0.80},
        {"type": "circle", "center": (5.0, 2.0), "radius": 0.80},
        {"type": "circle", "center": (10.0, 10.0), "radius": 0.80},
    ]

    for spec in obstacle_specs:
        add_circle_obstacle(occupancy, X, Y, spec["center"], spec["radius"])

    # Outer world boundary as occupied/Dirichlet boundary.
    occupancy[0, :] = True
    occupancy[-1, :] = True
    occupancy[:, 0] = True
    occupancy[:, -1] = True

    return World2D(
        Lx=Lx,
        Ly=Ly,
        nx=nx,
        ny=ny,
        x=x,
        y=y,
        X=X,
        Y=Y,
        dx=dx,
        dy=dy,
        occupancy=occupancy,
        start=np.array([1.0, 1.0], dtype=float),
        goal=np.array([11.0, 10.5], dtype=float),
        obstacle_specs=obstacle_specs,
    )


def compute_poisson(world: World2D, forcing_method: str, solver: str,
                    save_plots_dir: Optional[Path] = None):
    """Compute Poisson safety result and measure solver time."""
    config = PoissonBoxConfig(
        grid_spacing=(world.dx, world.dy),
        forcing_method=forcing_method,
        solver=solver,
        outer_boundary_as_dirichlet=True,
        compute_gradient=True,
        compute_hessian=True,
        plot=False,
        save_outputs=False,
    )

    box = PoissonSafetyBox(config)

    t0 = time.perf_counter()
    result = box.compute(world.occupancy)
    solve_time = time.perf_counter() - t0

    if save_plots_dir is not None:
        save_plots_dir.mkdir(parents=True, exist_ok=True)
        try:
            result.plot_all(save_plots_dir, show=False, save=True)
            plt.close("all")
        except Exception as exc:
            print(f"[warning] Poisson plot_all failed for forcing={forcing_method}: {exc}")

    return result, solve_time


# ============================================================
# Simulation
# ============================================================

def run_simulation(world: World2D, poisson_result, alpha: float,
                   max_speed: float = 1.0, kp: float = 0.8, dt: float = 0.05,
                   max_steps: int = 900, goal_tolerance: float = 0.18) -> TrajectoryData:
    """
    Simulate a single-integrator point robot with PID/P velocity command
    filtered by a velocity-level CBF.
    """
    cbf_config = CBFBoxConfig(
        mode="velocity",
        solver="closed_form",
        alpha=float(alpha),
        control_lower_bound=[-max_speed, -max_speed],
        control_upper_bound=[max_speed, max_speed],
    )
    cbf_box = CBFBox(cbf_config)

    p = world.start.copy()

    trajectory = []
    h_history = []
    residual_history = []
    u_nom_history = []
    u_safe_history = []
    filtered_history = []
    correction_history = []

    success = False
    collision = False

    for k in range(max_steps):
        trajectory.append(p.copy())

        error = world.goal - p
        u_nom = saturate_vector(kp * error, max_speed)

        h_value, grad_h, ij = sample_nearest(poisson_result, p, world.dx, world.dy)

        state = SystemState(
            position=p.copy(),
            velocity=u_nom.copy(),
            time=k * dt,
        )

        safety = SafetySample(
            h=h_value,
            grad_h=grad_h,
            hessian_h=None,
        )

        result = cbf_box.filter_control(
            state=state,
            safety=safety,
            u_nom=u_nom,
        )

        u_safe = np.asarray(result.u_safe, dtype=float)
        p = p + dt * u_safe

        residual = float(getattr(result, "cbf_residual", np.nan))
        correction = float(np.linalg.norm(u_safe - u_nom))

        h_history.append(float(h_value))
        residual_history.append(residual)
        u_nom_history.append(u_nom.copy())
        u_safe_history.append(u_safe.copy())
        filtered_history.append(bool(result.was_filtered))
        correction_history.append(correction)

        if np.linalg.norm(world.goal - p) < goal_tolerance:
            trajectory.append(p.copy())
            success = True
            break

        i, j = ij
        if world.occupancy[i, j]:
            collision = True
            print(f"[warning] robot entered occupied cell at step={k}, ij={ij}.")
            break

    return TrajectoryData(
        trajectory=np.asarray(trajectory),
        h_history=np.asarray(h_history),
        residual_history=np.asarray(residual_history),
        u_nom_history=np.asarray(u_nom_history),
        u_safe_history=np.asarray(u_safe_history),
        filtered_history=np.asarray(filtered_history, dtype=bool),
        correction_history=np.asarray(correction_history),
        success=success,
        collision=collision,
    )


def save_case_data(output_dir: Path, forcing_method: str, alpha: float,
                   world: World2D, poisson_result, traj: TrajectoryData) -> Path:
    """Save raw arrays for one simulation case."""
    safe_forcing = forcing_method.replace("/", "_")
    alpha_token = f"{alpha:g}".replace(".", "p").replace("-", "m")
    path = output_dir / "raw" / f"case_forcing-{safe_forcing}_alpha-{alpha_token}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        path,
        forcing_method=forcing_method,
        alpha=float(alpha),
        trajectory=traj.trajectory,
        h_history=traj.h_history,
        residual_history=traj.residual_history,
        u_nom_history=traj.u_nom_history,
        u_safe_history=traj.u_safe_history,
        filtered_history=traj.filtered_history,
        correction_history=traj.correction_history,
        occupancy=world.occupancy,
        h=poisson_result.h,
        grad_h=poisson_result.grad_h,
        start=world.start,
        goal=world.goal,
        dx=world.dx,
        dy=world.dy,
    )
    return path


def summarize_case(forcing_method: str, alpha: float, solver: str,
                   poisson_solve_time_s: float, simulation_time_s: float,
                   data_file: Path, world: World2D, traj: TrajectoryData) -> CaseResult:
    """Convert raw trajectory data into paper/report metrics."""
    total_time = poisson_solve_time_s + simulation_time_s
    final = traj.trajectory[-1]
    correction = traj.correction_history
    residual = traj.residual_history

    return CaseResult(
        forcing_method=forcing_method,
        alpha=float(alpha),
        solver=solver,
        poisson_solve_time_s=float(poisson_solve_time_s),
        simulation_time_s=float(simulation_time_s),
        total_time_s=float(total_time),
        steps=int(len(traj.trajectory)),
        success=bool(traj.success),
        collision=bool(traj.collision),
        final_distance=float(np.linalg.norm(world.goal - final)),
        path_length=path_length(traj.trajectory),
        filtered_fraction=safe_mean(traj.filtered_history.astype(float), default=0.0),
        min_h=safe_min(traj.h_history),
        min_cbf_residual=safe_min(residual),
        mean_correction=safe_mean(correction, default=0.0),
        max_correction=safe_min(-correction, default=0.0) * -1.0 if len(correction) else 0.0,
        final_x=float(final[0]),
        final_y=float(final[1]),
        data_file=str(data_file),
    )


# ============================================================
# Plotting helpers
# ============================================================

def plot_world_background(ax, world: World2D, poisson_result, title: str,
                          show_h: bool = True) -> None:
    """Plot occupancy, optional Poisson h contours, start and goal."""
    if show_h:
        h_plot = np.where(world.occupancy, np.nan, poisson_result.h)
        im = ax.contourf(world.X, world.Y, h_plot, levels=40)
        plt.colorbar(im, ax=ax, label="Poisson safety h")

    ax.contour(world.X, world.Y, world.occupancy.astype(float), levels=[0.5], linewidths=1.2)
    ax.scatter(world.start[0], world.start[1], s=65, marker="o", label="Start")
    ax.scatter(world.goal[0], world.goal[1], s=80, marker="x", label="Goal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")


def plot_alpha_trajectories(output_dir: Path, world: World2D, poisson_result,
                            alpha_results: Dict[float, Tuple[TrajectoryData, CaseResult]],
                            fixed_forcing: str) -> None:
    """Plot trajectories for different alpha values over the same h field."""
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    plot_world_background(ax, world, poisson_result, f"Effect of CBF alpha, forcing={fixed_forcing}")

    for alpha, (traj, metrics) in sorted(alpha_results.items(), key=lambda item: item[0]):
        label = f"alpha={alpha:g}, filt={metrics.filtered_fraction:.2f}"
        ax.plot(traj.trajectory[:, 0], traj.trajectory[:, 1], linewidth=2.0, label=label)

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "paper_alpha_trajectories.png", dpi=220)
    plt.close(fig)


def plot_alpha_metrics(output_dir: Path, alpha_metrics: List[CaseResult]) -> None:
    """Plot scalar metrics vs alpha."""
    rows = sorted(alpha_metrics, key=lambda r: r.alpha)
    alpha = np.asarray([r.alpha for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    axes[0, 0].plot(alpha, [r.final_distance for r in rows], marker="o")
    axes[0, 0].set_ylabel("final distance [m]")
    axes[0, 0].set_xlabel("alpha")
    axes[0, 0].set_title("Goal convergence")

    axes[0, 1].plot(alpha, [r.min_h for r in rows], marker="o")
    axes[0, 1].axhline(0.0, linestyle="--")
    axes[0, 1].set_ylabel("min h")
    axes[0, 1].set_xlabel("alpha")
    axes[0, 1].set_title("Minimum safety value")

    axes[1, 0].plot(alpha, [r.filtered_fraction for r in rows], marker="o")
    axes[1, 0].set_ylabel("filtered fraction")
    axes[1, 0].set_xlabel("alpha")
    axes[1, 0].set_title("How often CBF acts")

    axes[1, 1].plot(alpha, [r.mean_correction for r in rows], marker="o", label="mean")
    axes[1, 1].plot(alpha, [r.max_correction for r in rows], marker="o", label="max")
    axes[1, 1].set_ylabel("||u_safe - u_nom||")
    axes[1, 1].set_xlabel("alpha")
    axes[1, 1].set_title("Command correction")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)

    fig.suptitle("CBF alpha sensitivity")
    fig.tight_layout()
    fig.savefig(output_dir / "paper_alpha_metrics.png", dpi=220)
    plt.close(fig)


def plot_forcing_trajectories(output_dir: Path, world: World2D,
                              forcing_results: Dict[str, Tuple[object, TrajectoryData, CaseResult]],
                              fixed_alpha: float) -> None:
    """Plot one subplot per forcing method showing field + trajectory."""
    n = len(forcing_results)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5.5 * rows), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (forcing, (poisson_result, traj, metrics)) in zip(axes.ravel(), forcing_results.items()):
        ax.axis("on")
        title = (
            f"{forcing}, alpha={fixed_alpha:g}\n"
            f"solve={metrics.poisson_solve_time_s:.3f}s, "
            f"filt={metrics.filtered_fraction:.2f}, "
            f"min h={metrics.min_h:.3g}"
        )
        plot_world_background(ax, world, poisson_result, title)
        ax.plot(traj.trajectory[:, 0], traj.trajectory[:, 1], linewidth=2.4, label="trajectory")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Effect of Poisson forcing method")
    fig.tight_layout()
    fig.savefig(output_dir / "paper_forcing_trajectories.png", dpi=220)
    plt.close(fig)


def plot_forcing_metrics(output_dir: Path, forcing_metrics: List[CaseResult]) -> None:
    """Plot forcing-method metrics including solution time."""
    rows = list(forcing_metrics)
    names = [r.forcing_method for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    axes[0, 0].bar(x, [r.poisson_solve_time_s for r in rows])
    axes[0, 0].set_title("Poisson solve time")
    axes[0, 0].set_ylabel("time [s]")

    axes[0, 1].bar(x, [r.final_distance for r in rows])
    axes[0, 1].set_title("Final distance to goal")
    axes[0, 1].set_ylabel("distance [m]")

    axes[1, 0].bar(x, [r.min_h for r in rows])
    axes[1, 0].axhline(0.0, linestyle="--")
    axes[1, 0].set_title("Minimum h along trajectory")
    axes[1, 0].set_ylabel("min h")

    axes[1, 1].bar(x, [r.filtered_fraction for r in rows])
    axes[1, 1].set_title("CBF filtered fraction")
    axes[1, 1].set_ylabel("fraction")

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Poisson forcing sensitivity")
    fig.tight_layout()
    fig.savefig(output_dir / "paper_forcing_metrics.png", dpi=220)
    plt.close(fig)


def plot_solution_time(output_dir: Path, metrics: List[CaseResult]) -> None:
    """Plot solution-time comparison."""
    rows = list(metrics)
    labels = [f"{r.forcing_method}\nα={r.alpha:g}" for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(rows)), 5))
    ax.bar(x - 0.2, [r.poisson_solve_time_s for r in rows], width=0.4, label="Poisson solve")
    ax.bar(x + 0.2, [r.simulation_time_s for r in rows], width=0.4, label="CBF rollout")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("time [s]")
    ax.set_title("Computation time by case")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "paper_solution_time.png", dpi=220)
    plt.close(fig)


def plot_dashboard(output_dir: Path, world: World2D, fixed_poisson_result,
                   alpha_results: Dict[float, Tuple[TrajectoryData, CaseResult]],
                   forcing_metrics: List[CaseResult]) -> None:
    """Create one all-in-one dashboard useful for quick report screenshots."""
    alpha_rows = sorted([m for _, m in alpha_results.values()], key=lambda r: r.alpha)
    alphas = [r.alpha for r in alpha_rows]

    fig = plt.figure(figsize=(15, 11))

    ax1 = fig.add_subplot(2, 2, 1)
    plot_world_background(ax1, world, fixed_poisson_result, "Alpha sweep trajectories")
    for alpha, (traj, metrics) in sorted(alpha_results.items(), key=lambda item: item[0]):
        ax1.plot(traj.trajectory[:, 0], traj.trajectory[:, 1], linewidth=1.8, label=f"α={alpha:g}")
    ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(alphas, [r.filtered_fraction for r in alpha_rows], marker="o", label="filtered fraction")
    ax2.plot(alphas, [r.mean_correction for r in alpha_rows], marker="o", label="mean correction")
    ax2.set_xlabel("alpha")
    ax2.set_title("CBF sensitivity")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    names = [r.forcing_method for r in forcing_metrics]
    x = np.arange(len(names))
    ax3.bar(x, [r.poisson_solve_time_s for r in forcing_metrics])
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=25, ha="right")
    ax3.set_ylabel("time [s]")
    ax3.set_title("Poisson solve time by forcing")
    ax3.grid(True, axis="y", alpha=0.25)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(x, [r.min_h for r in forcing_metrics], width=0.35, label="min h")
    ax4.bar(x + 0.35, [r.filtered_fraction for r in forcing_metrics], width=0.35, label="filtered fraction")
    ax4.set_xticks(x + 0.175)
    ax4.set_xticklabels(names, rotation=25, ha="right")
    ax4.set_title("Forcing safety/CBF metrics")
    ax4.grid(True, axis="y", alpha=0.25)
    ax4.legend()

    fig.suptitle("Poisson-CBF parameter study summary")
    fig.tight_layout()
    fig.savefig(output_dir / "paper_dashboard_summary.png", dpi=220)
    plt.close(fig)


# ============================================================
# Reporting
# ============================================================

def write_metrics_csv(path: Path, rows: Sequence[CaseResult]) -> None:
    """Write metrics table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown_summary(path: Path, rows: Sequence[CaseResult]) -> None:
    """Write a compact Markdown summary for the paper/report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        f.write("# Poisson-CBF Parameter Study Summary\n\n")
        f.write("This file was generated automatically by `run_boxes_pid_parameter_study.py`.\n\n")
        f.write("## Interpretation guide\n\n")
        f.write("- Smaller CBF alpha usually makes the velocity-CBF constraint more conservative.\n")
        f.write("- Larger alpha allows faster decrease of h and often modifies the nominal command less.\n")
        f.write("- The forcing method changes the shape of the Poisson safety field and can change both the path and the amount of CBF correction.\n")
        f.write("- Poisson solve time measures field-generation cost; CBF rollout time measures repeated filtering and simulation.\n\n")
        f.write("## Metrics\n\n")
        f.write("| forcing | alpha | Poisson solve [s] | rollout [s] | success | collision | final dist | min h | min residual | filtered frac | mean correction | max correction |\n")
        f.write("|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r.forcing_method} | {r.alpha:g} | {r.poisson_solve_time_s:.4f} | "
                f"{r.simulation_time_s:.4f} | {r.success} | {r.collision} | "
                f"{r.final_distance:.4f} | {r.min_h:.4g} | {r.min_cbf_residual:.4g} | "
                f"{r.filtered_fraction:.3f} | {r.mean_correction:.4f} | {r.max_correction:.4f} |\n"
            )


def _json_safe(value):
    """Convert argparse/config values to JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def save_run_config(path: Path, args: argparse.Namespace) -> None:
    """Save command-line arguments as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_args = {key: _json_safe(value) for key, value in vars(args).items()}
    with path.open("w") as f:
        json.dump(serializable_args, f, indent=2)


# ============================================================
# Main study
# ============================================================

def run_case(world: World2D, poisson_cache: Dict[str, Tuple[object, float]],
             forcing_method: str, alpha: float, solver: str, output_dir: Path,
             save_poisson_plots: bool = False) -> Tuple[TrajectoryData, CaseResult, object]:
    """Run one forcing/alpha case with Poisson caching per forcing method."""
    if forcing_method not in poisson_cache:
        plot_dir = output_dir / "poisson_plots" / forcing_method if save_poisson_plots else None
        poisson_result, poisson_time = compute_poisson(
            world=world,
            forcing_method=forcing_method,
            solver=solver,
            save_plots_dir=plot_dir,
        )
        poisson_cache[forcing_method] = (poisson_result, poisson_time)

    poisson_result, poisson_time = poisson_cache[forcing_method]

    t0 = time.perf_counter()
    traj = run_simulation(world=world, poisson_result=poisson_result, alpha=alpha)
    simulation_time = time.perf_counter() - t0

    data_file = save_case_data(output_dir, forcing_method, alpha, world, poisson_result, traj)

    metrics = summarize_case(
        forcing_method=forcing_method,
        alpha=alpha,
        solver=solver,
        poisson_solve_time_s=poisson_time,
        simulation_time_s=simulation_time,
        data_file=data_file,
        world=world,
        traj=traj,
    )

    return traj, metrics, poisson_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run alpha and Poisson-forcing parameter studies for the 2D Poisson-CBF demo."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/parameter_study_pid_cbf"),
        help="Output directory for figures, CSV files, and raw arrays.",
    )
    parser.add_argument(
        "--alphas",
        type=parse_float_list,
        default=parse_float_list("0.25,0.5,1,2,5,10"),
        help="Comma-separated alpha values for the CBF alpha sweep.",
    )
    parser.add_argument(
        "--forcing-methods",
        type=parse_str_list,
        default=parse_str_list("constant,distance,average_flux,guidance"),
        help="Comma-separated Poisson forcing methods.",
    )
    parser.add_argument(
        "--fixed-forcing",
        type=str,
        default="guidance",
        help="Forcing method used for the alpha sweep.",
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=2.0,
        help="Alpha value used for the forcing-method sweep.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="sparse_direct",
        choices=["sor", "sparse_direct", "conjugate_gradient"],
        help="Poisson solver to use.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=140,
        help="Grid size for both dimensions. Use smaller values for quick debugging.",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Also run every forcing method with every alpha value.",
    )
    parser.add_argument(
        "--save-poisson-plots",
        action="store_true",
        help="Save detailed Poisson plots for each forcing method if supported by the box.",
    )

    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(output_dir / "run_config.json", args)

    world = create_world(nx=args.grid_size, ny=args.grid_size)

    poisson_cache: Dict[str, Tuple[object, float]] = {}
    all_metrics: List[CaseResult] = []

    # --------------------------------------------------------
    # 1. Alpha sweep with fixed forcing
    # --------------------------------------------------------
    alpha_results: Dict[float, Tuple[TrajectoryData, CaseResult]] = {}

    print("\n=== Alpha sweep ===")
    for alpha in args.alphas:
        traj, metrics, poisson_result = run_case(
            world=world,
            poisson_cache=poisson_cache,
            forcing_method=args.fixed_forcing,
            alpha=alpha,
            solver=args.solver,
            output_dir=output_dir,
            save_poisson_plots=args.save_poisson_plots,
        )
        alpha_results[alpha] = (traj, metrics)
        all_metrics.append(metrics)
        print(
            f"alpha={alpha:g} | final={metrics.final_distance:.3f} | "
            f"min_h={metrics.min_h:.3g} | filt={metrics.filtered_fraction:.2f} | "
            f"solve={metrics.poisson_solve_time_s:.3f}s"
        )

    fixed_poisson_result = poisson_cache[args.fixed_forcing][0]
    plot_alpha_trajectories(output_dir, world, fixed_poisson_result, alpha_results, args.fixed_forcing)
    plot_alpha_metrics(output_dir, [m for _, m in alpha_results.values()])

    # --------------------------------------------------------
    # 2. Forcing-method sweep with fixed alpha
    # --------------------------------------------------------
    forcing_results: Dict[str, Tuple[object, TrajectoryData, CaseResult]] = {}
    forcing_metrics: List[CaseResult] = []

    print("\n=== Forcing sweep ===")
    for forcing_method in args.forcing_methods:
        traj, metrics, poisson_result = run_case(
            world=world,
            poisson_cache=poisson_cache,
            forcing_method=forcing_method,
            alpha=args.fixed_alpha,
            solver=args.solver,
            output_dir=output_dir,
            save_poisson_plots=args.save_poisson_plots,
        )
        forcing_results[forcing_method] = (poisson_result, traj, metrics)
        forcing_metrics.append(metrics)

        # Avoid duplicating same case already present in alpha sweep.
        duplicate = (
            forcing_method == args.fixed_forcing
            and any(abs(args.fixed_alpha - a) < 1e-12 for a in args.alphas)
        )
        if not duplicate:
            all_metrics.append(metrics)

        print(
            f"forcing={forcing_method} | final={metrics.final_distance:.3f} | "
            f"min_h={metrics.min_h:.3g} | filt={metrics.filtered_fraction:.2f} | "
            f"solve={metrics.poisson_solve_time_s:.3f}s"
        )

    plot_forcing_trajectories(output_dir, world, forcing_results, args.fixed_alpha)
    plot_forcing_metrics(output_dir, forcing_metrics)

    # --------------------------------------------------------
    # 3. Optional full grid: forcing x alpha
    # --------------------------------------------------------
    if args.full_grid:
        print("\n=== Full grid sweep ===")
        for forcing_method in args.forcing_methods:
            for alpha in args.alphas:
                already_done = any(
                    (r.forcing_method == forcing_method and abs(r.alpha - alpha) < 1e-12)
                    for r in all_metrics
                )
                if already_done:
                    continue

                traj, metrics, poisson_result = run_case(
                    world=world,
                    poisson_cache=poisson_cache,
                    forcing_method=forcing_method,
                    alpha=alpha,
                    solver=args.solver,
                    output_dir=output_dir,
                    save_poisson_plots=False,
                )
                all_metrics.append(metrics)
                print(
                    f"forcing={forcing_method}, alpha={alpha:g} | "
                    f"final={metrics.final_distance:.3f} | "
                    f"filt={metrics.filtered_fraction:.2f}"
                )

    # --------------------------------------------------------
    # 4. Write reports and final figures
    # --------------------------------------------------------
    write_metrics_csv(output_dir / "metrics_all_cases.csv", all_metrics)
    write_metrics_csv(output_dir / "metrics_alpha_sweep.csv", [m for _, m in alpha_results.values()])
    write_metrics_csv(output_dir / "metrics_forcing_sweep.csv", forcing_metrics)
    write_markdown_summary(output_dir / "PARAMETER_STUDY_SUMMARY.md", all_metrics)
    plot_solution_time(output_dir, all_metrics)
    plot_dashboard(output_dir, world, fixed_poisson_result, alpha_results, forcing_metrics)

    print("\nSaved parameter-study outputs to:", output_dir)
    print("Key paper figures:")
    print("  - paper_alpha_trajectories.png")
    print("  - paper_alpha_metrics.png")
    print("  - paper_forcing_trajectories.png")
    print("  - paper_forcing_metrics.png")
    print("  - paper_solution_time.png")
    print("  - paper_dashboard_summary.png")
    print("Tables:")
    print("  - metrics_all_cases.csv")
    print("  - metrics_alpha_sweep.csv")
    print("  - metrics_forcing_sweep.csv")
    print("  - PARAMETER_STUDY_SUMMARY.md")


if __name__ == "__main__":
    main()