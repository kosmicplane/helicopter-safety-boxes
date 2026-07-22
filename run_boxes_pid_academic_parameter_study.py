#!/usr/bin/env python3
"""
NASA/Caltech-style 3D Poisson-CBF parameter study.

This script is intentionally a standalone research script placed at the root of
`Helicopter/`, beside:

    poisson_safety_box/
    cbf_safety_box/

It generates publication-oriented figures for a Poisson-HOCBF autonomous landing
study, including:

    1. 3D obstacle world and occupancy/boundary visualization.
    2. Occupancy matrix slices and boundary/frontier slices.
    3. Poisson safety field h for multiple forcing functions.
    4. Forcing field / guidance vector / gradient field visualizations.
    5. 3D Poisson contour slices.
    6. Alpha sweep with all trajectories from conservative to aggressive.
    7. Representative h(t), HOCBF residual, and correction histories.
    8. Quantitative metrics and compute-time comparisons.
    9. Solver timing comparison.
    10. Integrated paper-grade dashboard.

Default architecture:
    fake 3D world -> occupancy -> Poisson h, grad_h, Hessian(h)
    -> nominal PD acceleration -> HOCBF filter -> double-integrator rollout.

No ROS, PX4, or Gazebo is required for this script.

Run example:
    python run_poisson_cbf_nasa_parameter_study.py \
        --output-dir outputs/nasa_parameter_study \
        --grid-shape 64,52,36 \
        --alphas 0.05,0.08,0.12,0.2,0.35,0.5,0.75,1,1.5,2,3,5,8 \
        --forcing-methods constant,distance,average_flux,guidance \
        --fixed-forcing guidance \
        --fixed-alpha 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "poisson_safety_box"))
sys.path.insert(0, str(ROOT / "cbf_safety_box"))

import numpy as np
import matplotlib

# Save-only backend: no windows, deterministic report generation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from poisson_safety_box.api import PoissonSafetyBox
from poisson_safety_box.config import PoissonBoxConfig
from cbf_safety_box.api import CBFBox
from cbf_safety_box.config import CBFBoxConfig
from cbf_safety_box.state import SystemState
from cbf_safety_box.safety_data.sample import SafetySample


# =============================================================================
# Publication style
# =============================================================================


def set_publication_style() -> None:
    """Configure Matplotlib for clean technical figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": False,
            "mathtext.default": "regular",
        }
    )


def save_figure(fig: plt.Figure, figures_dir: Path, name: str, save_pdf: bool = True) -> None:
    """Save a figure as PNG and optionally PDF."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", bbox_inches="tight")
    if save_pdf:
        fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Data containers
# =============================================================================


@dataclass
class WorldData:
    """World, obstacle, grid, and occupancy information."""

    Lx: float
    Ly: float
    Lz: float
    nx: int
    ny: int
    nz: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    dx: float
    dy: float
    dz: float
    occupancy: np.ndarray
    boundary_mask: np.ndarray
    spheres: List[Dict[str, Any]]
    boxes: List[Dict[str, Any]]
    cylinders: List[Dict[str, Any]]
    start: np.ndarray
    goal: np.ndarray


@dataclass
class RolloutData:
    """Single rollout data for a given alpha/forcing configuration."""

    label: str
    alpha: float
    alpha1: float
    alpha2: float
    forcing_method: str
    solver: str
    trajectory: np.ndarray
    nominal_trajectory: np.ndarray
    h_history: np.ndarray
    residual_history: np.ndarray
    a_nom_history: np.ndarray
    a_safe_history: np.ndarray
    v_history: np.ndarray
    filtered_history: np.ndarray
    cbf_solve_time_history: np.ndarray
    collision: bool
    reached_goal: bool
    steps: int
    dt: float
    final_distance: float
    path_length: float
    min_h: float
    min_residual: float
    filtered_fraction: float
    mean_correction: float
    max_correction: float
    mean_cbf_solve_ms: float
    rollout_wall_time_sec: float


@dataclass
class PoissonCase:
    """Poisson result plus timing metadata."""

    forcing_method: str
    solver: str
    result: Any
    wall_time_sec: float
    timing: Dict[str, float]
    status: str
    error: str = ""


# =============================================================================
# Geometry and occupancy construction
# =============================================================================


def add_sphere_obstacle(occupancy: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                        center: Sequence[float], radius: float) -> None:
    """Mark a spherical obstacle in a 3D occupancy grid."""
    cx, cy, cz = center
    mask = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= radius ** 2
    occupancy[mask] = True


def add_box_obstacle(occupancy: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                     center: Sequence[float], size: Sequence[float]) -> None:
    """Mark an axis-aligned box obstacle in a 3D occupancy grid."""
    cx, cy, cz = center
    sx, sy, sz = size
    mask = (
        (np.abs(X - cx) <= sx / 2.0)
        & (np.abs(Y - cy) <= sy / 2.0)
        & (np.abs(Z - cz) <= sz / 2.0)
    )
    occupancy[mask] = True


def add_cylinder_obstacle(occupancy: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                          center: Sequence[float], radius: float, height: float) -> None:
    """Mark a vertical cylinder obstacle in a 3D occupancy grid."""
    cx, cy, cz = center
    radial = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
    vertical = np.abs(Z - cz) <= height / 2.0
    occupancy[radial & vertical] = True


def shifted_or(mask: np.ndarray, axis: int, direction: int) -> np.ndarray:
    """Shift a boolean mask without wrap-around."""
    out = np.zeros_like(mask, dtype=bool)
    src = [slice(None)] * mask.ndim
    dst = [slice(None)] * mask.ndim
    if direction > 0:
        src[axis] = slice(0, -1)
        dst[axis] = slice(1, None)
    else:
        src[axis] = slice(1, None)
        dst[axis] = slice(0, -1)
    out[tuple(dst)] = mask[tuple(src)]
    return out


def compute_free_boundary_mask(occupancy: np.ndarray) -> np.ndarray:
    """Compute free cells adjacent to occupied cells: a practical boundary/frontier mask."""
    occ = occupancy.astype(bool)
    neighbor_occ = np.zeros_like(occ, dtype=bool)
    for axis in range(occ.ndim):
        neighbor_occ |= shifted_or(occ, axis, +1)
        neighbor_occ |= shifted_or(occ, axis, -1)
    boundary = (~occ) & neighbor_occ
    return boundary


def make_research_world(grid_shape: Tuple[int, int, int]) -> WorldData:
    """Create a demanding 3D landing world for academic visualization."""
    Lx, Ly, Lz = 18.0, 14.0, 10.0
    nx, ny, nz = grid_shape
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    z = np.linspace(0.0, Lz, nz)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    occupancy = np.zeros((nx, ny, nz), dtype=bool)

    # Obstacle set designed to stress the safety field: a descending corridor,
    # floating obstacle chain, wall/gate-like structures, and tower-like objects.
    spheres = [
        {"name": "floating_cluster_1", "center": (5.2, 4.8, 6.8), "radius": 1.30},
        {"name": "floating_cluster_2", "center": (7.4, 5.9, 5.6), "radius": 1.15},
        {"name": "floating_cluster_3", "center": (9.7, 7.0, 4.4), "radius": 1.10},
        {"name": "floating_cluster_4", "center": (12.2, 8.4, 3.0), "radius": 1.00},
        {"name": "descent_blocker", "center": (14.2, 9.8, 1.85), "radius": 0.85},
    ]
    boxes = [
        {"name": "gate_wall_lower", "center": (4.0, 7.0, 3.0), "size": (0.65, 8.0, 4.2)},
        {"name": "gate_wall_upper", "center": (4.0, 7.0, 8.7), "size": (0.65, 8.0, 1.8)},
        {"name": "diagonal_box_1", "center": (7.0, 8.7, 2.6), "size": (1.4, 2.2, 3.0)},
        {"name": "diagonal_box_2", "center": (10.5, 4.8, 2.2), "size": (2.0, 1.7, 2.8)},
        {"name": "landing_zone_partial_block", "center": (15.4, 11.2, 0.85), "size": (1.4, 1.4, 1.7)},
    ]
    cylinders = [
        {"name": "tower_1", "center": (8.5, 10.7, 2.8), "radius": 0.45, "height": 5.6},
        {"name": "tower_2", "center": (11.6, 6.0, 2.2), "radius": 0.50, "height": 4.4},
        {"name": "tower_3", "center": (13.8, 7.2, 1.8), "radius": 0.40, "height": 3.6},
    ]

    for item in spheres:
        add_sphere_obstacle(occupancy, X, Y, Z, item["center"], item["radius"])
    for item in boxes:
        add_box_obstacle(occupancy, X, Y, Z, item["center"], item["size"])
    for item in cylinders:
        add_cylinder_obstacle(occupancy, X, Y, Z, item["center"], item["radius"], item["height"])

    # Outer Dirichlet boundary.
    occupancy[0, :, :] = True
    occupancy[-1, :, :] = True
    occupancy[:, 0, :] = True
    occupancy[:, -1, :] = True
    occupancy[:, :, 0] = True
    occupancy[:, :, -1] = True

    boundary_mask = compute_free_boundary_mask(occupancy)

    start = np.array([1.2, 1.2, 8.5], dtype=float)
    goal = np.array([16.5, 12.0, 0.8], dtype=float)

    return WorldData(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        nx=nx,
        ny=ny,
        nz=nz,
        x=x,
        y=y,
        z=z,
        X=X,
        Y=Y,
        Z=Z,
        dx=dx,
        dy=dy,
        dz=dz,
        occupancy=occupancy,
        boundary_mask=boundary_mask,
        spheres=spheres,
        boxes=boxes,
        cylinders=cylinders,
        start=start,
        goal=goal,
    )


# =============================================================================
# Dynamics, sampling, and simulation
# =============================================================================


def saturate_vector(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Limit a vector by Euclidean norm."""
    n = float(np.linalg.norm(v))
    if n <= max_norm or n < 1e-12:
        return v
    return max_norm * v / n


def sample_nearest_3d(result: Any, p: np.ndarray, world: WorldData) -> Tuple[float, np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """Nearest-neighbor sample of h, grad_h, and Hessian(h)."""
    h_grid = result.h
    grad_grid = result.grad_h
    hessian_grid = result.hessian_h
    i = int(np.clip(round(p[0] / world.dx), 0, world.nx - 1))
    j = int(np.clip(round(p[1] / world.dy), 0, world.ny - 1))
    k = int(np.clip(round(p[2] / world.dz), 0, world.nz - 1))
    h_value = float(h_grid[i, j, k])
    grad_h = np.asarray(grad_grid[i, j, k], dtype=float)
    hessian_h = np.asarray(hessian_grid[i, j, k], dtype=float)
    return h_value, grad_h, hessian_h, (i, j, k)


def nominal_pd_acceleration(p: np.ndarray, v: np.ndarray, goal: np.ndarray,
                            kp: np.ndarray, kd: np.ndarray, max_acc: float) -> np.ndarray:
    """Nominal acceleration controller: a_nom = kp*(goal-p) - kd*v."""
    a = kp * (goal - p) - kd * v
    return saturate_vector(a, max_acc)


def compute_poisson_case(world: WorldData, forcing_method: str, solver: str,
                         sor_max_iter: int, sor_tolerance: float, sor_omega: float,
                         cg_max_iter: int, cg_tolerance: float) -> PoissonCase:
    """Compute one Poisson field and record timing."""
    cfg = PoissonBoxConfig(
        grid_spacing=(world.dx, world.dy, world.dz),
        forcing_method=forcing_method,
        solver=solver,
        outer_boundary_as_dirichlet=True,
        compute_gradient=True,
        compute_hessian=True,
        compute_laplacian_check=True,
        plot=False,
        save_outputs=False,
    )
    cfg.sor.max_iter = sor_max_iter
    cfg.sor.tolerance = sor_tolerance
    cfg.sor.omega = sor_omega
    cfg.conjugate_gradient.max_iter = cg_max_iter
    cfg.conjugate_gradient.tolerance = cg_tolerance

    t0 = time.perf_counter()
    try:
        result = PoissonSafetyBox(cfg).compute(world.occupancy)
        wall = time.perf_counter() - t0
        return PoissonCase(
            forcing_method=forcing_method,
            solver=solver,
            result=result,
            wall_time_sec=wall,
            timing=dict(getattr(result, "timing", {}) or {}),
            status="ok",
        )
    except Exception as exc:
        wall = time.perf_counter() - t0
        return PoissonCase(
            forcing_method=forcing_method,
            solver=solver,
            result=None,
            wall_time_sec=wall,
            timing={},
            status="failed",
            error=repr(exc),
        )


def rollout_acceleration_hocbf(world: WorldData, poisson_case: PoissonCase, alpha: float,
                               dt: float, max_steps: int, goal_tolerance: float,
                               max_acc: float, max_speed: float) -> RolloutData:
    """Run a double-integrator HOCBF rollout for one alpha."""
    if poisson_case.result is None:
        raise RuntimeError(f"Poisson case failed: {poisson_case.error}")

    # Scalar alpha is mapped to a HOCBF pair. alpha1 is slower, alpha2 is faster.
    alpha1 = max(1.0e-6, 0.45 * alpha)
    alpha2 = max(1.0e-6, alpha)

    cbf = CBFBox(
        CBFBoxConfig(
            mode="acceleration",
            solver="closed_form",
            alpha1=alpha1,
            alpha2=alpha2,
        )
    )

    kp = np.array([0.55, 0.55, 0.45], dtype=float)
    kd = np.array([1.05, 1.05, 0.90], dtype=float)

    p_safe = world.start.copy()
    v_safe = np.zeros(3, dtype=float)
    p_nom = world.start.copy()
    v_nom = np.zeros(3, dtype=float)

    traj_safe: List[np.ndarray] = []
    traj_nom: List[np.ndarray] = []
    h_hist: List[float] = []
    residual_hist: List[float] = []
    a_nom_hist: List[np.ndarray] = []
    a_safe_hist: List[np.ndarray] = []
    v_hist: List[np.ndarray] = []
    filtered_hist: List[bool] = []
    cbf_time_hist: List[float] = []

    collision = False
    reached_goal = False
    t0 = time.perf_counter()

    for step in range(max_steps):
        traj_safe.append(p_safe.copy())
        traj_nom.append(p_nom.copy())

        # Nominal-only rollout for baseline comparison.
        a_nom_only = nominal_pd_acceleration(p_nom, v_nom, world.goal, kp, kd, max_acc)
        v_nom = saturate_vector(v_nom + dt * a_nom_only, max_speed)
        p_nom = p_nom + dt * v_nom

        # Same nominal controller, then HOCBF filter.
        a_nom = nominal_pd_acceleration(p_safe, v_safe, world.goal, kp, kd, max_acc)
        h_value, grad_h, hessian_h, ijk = sample_nearest_3d(poisson_case.result, p_safe, world)

        state = SystemState(position=p_safe.copy(), velocity=v_safe.copy(), time=step * dt)
        safety = SafetySample(h=h_value, grad_h=grad_h, hessian_h=hessian_h)

        cbf_start = time.perf_counter()
        result = cbf.filter_control(state=state, safety=safety, u_nom=a_nom)
        cbf_elapsed = time.perf_counter() - cbf_start

        a_safe = np.asarray(result.u_safe, dtype=float)
        v_safe = saturate_vector(v_safe + dt * a_safe, max_speed)
        p_safe = p_safe + dt * v_safe

        residual = getattr(result, "hocbf_residual", None)
        if residual is None:
            residual = getattr(result, "cbf_residual", np.nan)

        h_hist.append(float(h_value))
        residual_hist.append(float(residual))
        a_nom_hist.append(a_nom.copy())
        a_safe_hist.append(a_safe.copy())
        v_hist.append(v_safe.copy())
        filtered_hist.append(bool(result.was_filtered))
        cbf_time_hist.append(float(cbf_elapsed))

        if np.linalg.norm(world.goal - p_safe) < goal_tolerance:
            reached_goal = True
            traj_safe.append(p_safe.copy())
            traj_nom.append(p_nom.copy())
            break

        i, j, k = ijk
        if bool(world.occupancy[i, j, k]):
            collision = True
            traj_safe.append(p_safe.copy())
            traj_nom.append(p_nom.copy())
            break

        # Avoid integrating outside the plotting/domain box.
        if not (0.0 <= p_safe[0] <= world.Lx and 0.0 <= p_safe[1] <= world.Ly and 0.0 <= p_safe[2] <= world.Lz):
            collision = True
            traj_safe.append(p_safe.copy())
            traj_nom.append(p_nom.copy())
            break

    rollout_wall = time.perf_counter() - t0

    traj = np.asarray(traj_safe)
    traj_n = np.asarray(traj_nom)
    h = np.asarray(h_hist, dtype=float)
    residual = np.asarray(residual_hist, dtype=float)
    a_nom_arr = np.asarray(a_nom_hist, dtype=float)
    a_safe_arr = np.asarray(a_safe_hist, dtype=float)
    v_arr = np.asarray(v_hist, dtype=float)
    filtered_arr = np.asarray(filtered_hist, dtype=bool)
    cbf_time_arr = np.asarray(cbf_time_hist, dtype=float)

    correction = np.linalg.norm(a_safe_arr - a_nom_arr, axis=1) if len(a_safe_arr) else np.zeros(0)
    if len(traj) >= 2:
        path_length = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    else:
        path_length = 0.0

    return RolloutData(
        label=f"alpha={alpha:g}",
        alpha=float(alpha),
        alpha1=float(alpha1),
        alpha2=float(alpha2),
        forcing_method=poisson_case.forcing_method,
        solver=poisson_case.solver,
        trajectory=traj,
        nominal_trajectory=traj_n,
        h_history=h,
        residual_history=residual,
        a_nom_history=a_nom_arr,
        a_safe_history=a_safe_arr,
        v_history=v_arr,
        filtered_history=filtered_arr,
        cbf_solve_time_history=cbf_time_arr,
        collision=collision,
        reached_goal=reached_goal,
        steps=int(len(traj)),
        dt=float(dt),
        final_distance=float(np.linalg.norm(world.goal - traj[-1])) if len(traj) else float("nan"),
        path_length=path_length,
        min_h=float(np.min(h)) if len(h) else float("nan"),
        min_residual=float(np.min(residual)) if len(residual) else float("nan"),
        filtered_fraction=float(np.mean(filtered_arr)) if len(filtered_arr) else 0.0,
        mean_correction=float(np.mean(correction)) if len(correction) else 0.0,
        max_correction=float(np.max(correction)) if len(correction) else 0.0,
        mean_cbf_solve_ms=float(1000.0 * np.mean(cbf_time_arr)) if len(cbf_time_arr) else 0.0,
        rollout_wall_time_sec=float(rollout_wall),
    )


# =============================================================================
# Drawing utilities
# =============================================================================


def draw_sphere(ax: Any, center: Sequence[float], radius: float, alpha: float = 0.16) -> None:
    """Draw a wireframe sphere."""
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 14)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, linewidth=0.45, alpha=alpha)


def draw_box(ax: Any, center: Sequence[float], size: Sequence[float], alpha: float = 0.45) -> None:
    """Draw a wireframe box."""
    cx, cy, cz = center
    sx, sy, sz = size
    x = [cx - sx / 2.0, cx + sx / 2.0]
    y = [cy - sy / 2.0, cy + sy / 2.0]
    z = [cz - sz / 2.0, cz + sz / 2.0]
    corners = np.array(
        [
            [x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[0], y[1], z[0]],
            [x[0], y[0], z[1]], [x[1], y[0], z[1]], [x[1], y[1], z[1]], [x[0], y[1], z[1]],
        ]
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        ax.plot([corners[a, 0], corners[b, 0]], [corners[a, 1], corners[b, 1]], [corners[a, 2], corners[b, 2]], linewidth=0.9, alpha=alpha)


def draw_cylinder(ax: Any, center: Sequence[float], radius: float, height: float, alpha: float = 0.3) -> None:
    """Draw a vertical cylinder wireframe."""
    cx, cy, cz = center
    theta = np.linspace(0, 2 * np.pi, 36)
    z_values = np.array([cz - height / 2.0, cz + height / 2.0])
    theta_grid, z_grid = np.meshgrid(theta, z_values)
    x_grid = cx + radius * np.cos(theta_grid)
    y_grid = cy + radius * np.sin(theta_grid)
    ax.plot_wireframe(x_grid, y_grid, z_grid, linewidth=0.45, alpha=alpha)


def draw_world_wireframe(ax: Any, world: WorldData, alpha: float = 0.35) -> None:
    """Draw all analytic obstacles as wireframes."""
    for s in world.spheres:
        draw_sphere(ax, s["center"], s["radius"], alpha=alpha * 0.55)
    for b in world.boxes:
        draw_box(ax, b["center"], b["size"], alpha=alpha)
    for c in world.cylinders:
        draw_cylinder(ax, c["center"], c["radius"], c["height"], alpha=alpha * 0.9)


def configure_3d_axes(ax: Any, world: WorldData, title: str) -> None:
    """Common 3D axis formatting."""
    ax.set_xlim(0, world.Lx)
    ax.set_ylim(0, world.Ly)
    ax.set_zlim(0, world.Lz)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    ax.set_box_aspect((world.Lx, world.Ly, world.Lz))
    ax.view_init(elev=22, azim=-57)


def sample_mask_points(mask: np.ndarray, world: WorldData, max_points: int, seed: int = 7) -> np.ndarray:
    """Return physical coordinates for a random subset of True mask cells."""
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return np.zeros((0, 3))
    rng = np.random.default_rng(seed)
    if len(idx) > max_points:
        idx = idx[rng.choice(len(idx), size=max_points, replace=False)]
    pts = np.column_stack((world.x[idx[:, 0]], world.y[idx[:, 1]], world.z[idx[:, 2]]))
    return pts


def slice_index(values: np.ndarray, target: float) -> int:
    """Index closest to target coordinate."""
    return int(np.argmin(np.abs(values - target)))


def finite_field_for_plot(field: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    """Mask occupied values as NaN for scalar field plotting."""
    arr = np.asarray(field, dtype=float).copy()
    arr[occupancy] = np.nan
    return arr


# =============================================================================
# Figures: world, occupancy, boundaries
# =============================================================================


def plot_world_occupancy_boundary_3d(world: WorldData, figures_dir: Path) -> None:
    """3D overview of obstacles, occupied voxels, boundary/frontier, start, and goal."""
    fig = plt.figure(figsize=(14, 10))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    draw_world_wireframe(ax, world, alpha=0.6)
    occ_pts = sample_mask_points(world.occupancy, world, max_points=1800, seed=1)
    if len(occ_pts):
        ax.scatter(occ_pts[:, 0], occ_pts[:, 1], occ_pts[:, 2], s=2.5, alpha=0.08, label="occupied voxels")
    ax.scatter(*world.start, s=80, marker="o", label="start")
    ax.scatter(*world.goal, s=90, marker="x", label="landing target")
    configure_3d_axes(ax, world, "Analytic obstacle world + sampled occupancy")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(2, 2, 2, projection="3d")
    draw_world_wireframe(ax, world, alpha=0.25)
    b_pts = sample_mask_points(world.boundary_mask, world, max_points=2500, seed=2)
    if len(b_pts):
        ax.scatter(b_pts[:, 0], b_pts[:, 1], b_pts[:, 2], s=3.0, alpha=0.12, label="free-space boundary/frontier")
    ax.scatter(*world.start, s=80, marker="o")
    ax.scatter(*world.goal, s=90, marker="x")
    configure_3d_axes(ax, world, "Boundary/frontier cells used by safety construction")
    ax.legend(loc="upper left")

    # Occupancy projections.
    ax = fig.add_subplot(2, 2, 3)
    occ_xy = np.max(world.occupancy, axis=2).T
    ax.imshow(occ_xy, origin="lower", extent=[0, world.Lx, 0, world.Ly], aspect="auto", cmap="gray_r")
    ax.scatter(world.start[0], world.start[1], s=55, marker="o", label="start")
    ax.scatter(world.goal[0], world.goal[1], s=65, marker="x", label="goal")
    ax.set_title("Occupancy projection: XY")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 4)
    occ_xz = np.max(world.occupancy, axis=1).T
    ax.imshow(occ_xz, origin="lower", extent=[0, world.Lx, 0, world.Lz], aspect="auto", cmap="gray_r")
    ax.scatter(world.start[0], world.start[2], s=55, marker="o", label="start")
    ax.scatter(world.goal[0], world.goal[2], s=65, marker="x", label="goal")
    ax.set_title("Occupancy projection: XZ")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.legend(loc="best")

    fig.suptitle("World construction: obstacles, occupancy matrix, and boundary/frontier", fontsize=15)
    save_figure(fig, figures_dir, "00_world_occupancy_boundary_3d")


def plot_occupancy_boundary_slices(world: WorldData, figures_dir: Path) -> None:
    """Show occupancy and boundary slices at multiple heights."""
    z_targets = [1.0, 3.0, 5.0, 7.5]
    fig, axes = plt.subplots(2, len(z_targets), figsize=(4.1 * len(z_targets), 7.2), sharex=True, sharey=True)

    for col, zt in enumerate(z_targets):
        k = slice_index(world.z, zt)
        occ = world.occupancy[:, :, k].T.astype(float)
        bnd = world.boundary_mask[:, :, k].T.astype(float)

        ax = axes[0, col]
        ax.imshow(occ, origin="lower", extent=[0, world.Lx, 0, world.Ly], cmap="gray_r", aspect="auto")
        ax.scatter(world.start[0], world.start[1], marker="o", s=35)
        ax.scatter(world.goal[0], world.goal[1], marker="x", s=45)
        ax.set_title(f"Occupancy slice z={world.z[k]:.2f} m")
        ax.set_xlabel("x [m]")
        if col == 0:
            ax.set_ylabel("y [m]")

        ax = axes[1, col]
        ax.imshow(bnd, origin="lower", extent=[0, world.Lx, 0, world.Ly], cmap="magma", aspect="auto")
        ax.scatter(world.start[0], world.start[1], marker="o", s=35)
        ax.scatter(world.goal[0], world.goal[1], marker="x", s=45)
        ax.set_title(f"Boundary/frontier slice z={world.z[k]:.2f} m")
        ax.set_xlabel("x [m]")
        if col == 0:
            ax.set_ylabel("y [m]")

    fig.suptitle("Occupancy matrix and boundary/frontier slices", fontsize=15)
    save_figure(fig, figures_dir, "01_occupancy_boundary_slices")


# =============================================================================
# Figures: Poisson h, forcing, guidance/gradient, vector fields
# =============================================================================


def normalize_vectors(U: np.ndarray, V: np.ndarray, eps: float = 1.0e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize vector components and return magnitude."""
    mag = np.sqrt(U ** 2 + V ** 2)
    return U / (mag + eps), V / (mag + eps), mag


def plot_forcing_h_and_vector_fields(world: WorldData, cases: Dict[str, PoissonCase], rollouts: Dict[str, RolloutData], figures_dir: Path) -> None:
    """Grid of h contours and vector fields for each forcing method."""
    methods = [m for m in cases if cases[m].status == "ok"]
    if not methods:
        return

    z_star = 4.8
    k = slice_index(world.z, z_star)
    skip = max(2, int(min(world.nx, world.ny) / 24))
    X2 = world.X[:, :, k]
    Y2 = world.Y[:, :, k]

    fig, axes = plt.subplots(2, len(methods), figsize=(4.7 * len(methods), 8.2), sharex=True, sharey=True)
    if len(methods) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, method in enumerate(methods):
        case = cases[method]
        result = case.result
        rollout = rollouts.get(method)
        h_plot = finite_field_for_plot(result.h, world.occupancy)
        h_slice = h_plot[:, :, k]
        occ_slice = world.occupancy[:, :, k]

        ax = axes[0, col]
        cs = ax.contourf(X2, Y2, h_slice, levels=36, cmap="viridis")
        ax.contour(X2, Y2, occ_slice.astype(float), levels=[0.5], colors="black", linewidths=1.0)
        if rollout is not None:
            ax.plot(rollout.trajectory[:, 0], rollout.trajectory[:, 1], linewidth=2.3, label="trajectory")
        ax.scatter(world.start[0], world.start[1], s=45, marker="o", label="start")
        ax.scatter(world.goal[0], world.goal[1], s=55, marker="x", label="goal")
        ax.set_title(f"h(x,y,z={world.z[k]:.1f}) — {method}")
        ax.set_xlabel("x [m]")
        if col == 0:
            ax.set_ylabel("y [m]")
        fig.colorbar(cs, ax=ax, shrink=0.75, label="h")

        ax = axes[1, col]
        if getattr(result, "guidance_vector", None) is not None and method == "guidance":
            vector = np.asarray(result.guidance_vector)
            U = vector[:, :, k, 0]
            V = vector[:, :, k, 1]
            vector_name = "guidance vector"
        else:
            grad = np.asarray(result.grad_h)
            U = grad[:, :, k, 0]
            V = grad[:, :, k, 1]
            vector_name = r"$\nabla h$ direction"
        Un, Vn, mag = normalize_vectors(U, V)
        mag_plot = np.where(occ_slice, np.nan, mag)
        im = ax.contourf(X2, Y2, mag_plot, levels=32, cmap="plasma")
        ax.contour(X2, Y2, occ_slice.astype(float), levels=[0.5], colors="black", linewidths=1.0)
        ax.quiver(
            X2[::skip, ::skip],
            Y2[::skip, ::skip],
            Un[::skip, ::skip],
            Vn[::skip, ::skip],
            angles="xy",
            scale_units="xy",
            scale=1.4,
            width=0.003,
            alpha=0.75,
        )
        ax.scatter(world.start[0], world.start[1], s=45, marker="o")
        ax.scatter(world.goal[0], world.goal[1], s=55, marker="x")
        ax.set_title(f"{vector_name} and magnitude — {method}")
        ax.set_xlabel("x [m]")
        if col == 0:
            ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, shrink=0.75, label="magnitude")

    fig.suptitle("Poisson safety field, obstacle boundary, and induced vector directions", fontsize=15)
    save_figure(fig, figures_dir, "06_forcing_h_and_vector_fields")


def plot_forcing_scalar_and_laplacian(world: WorldData, cases: Dict[str, PoissonCase], figures_dir: Path) -> None:
    """Plot forcing f and numerical Δh on the same slice for each forcing method."""
    methods = [m for m in cases if cases[m].status == "ok"]
    if not methods:
        return
    z_star = 4.8
    k = slice_index(world.z, z_star)
    X2, Y2 = world.X[:, :, k], world.Y[:, :, k]

    fig, axes = plt.subplots(2, len(methods), figsize=(4.7 * len(methods), 8.2), sharex=True, sharey=True)
    if len(methods) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, method in enumerate(methods):
        result = cases[method].result
        forcing = finite_field_for_plot(np.asarray(result.forcing), world.occupancy)
        lap = getattr(result, "laplacian_h", None)
        if lap is not None:
            lap = finite_field_for_plot(np.asarray(lap), world.occupancy)

        ax = axes[0, col]
        im = ax.contourf(X2, Y2, forcing[:, :, k], levels=32, cmap="coolwarm")
        ax.contour(X2, Y2, world.occupancy[:, :, k].astype(float), levels=[0.5], colors="black", linewidths=1.0)
        ax.set_title(f"Forcing f — {method}")
        ax.set_xlabel("x [m]")
        if col == 0:
            ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, shrink=0.75, label="f")

        ax = axes[1, col]
        if lap is not None:
            im = ax.contourf(X2, Y2, lap[:, :, k], levels=32, cmap="coolwarm")
            fig.colorbar(im, ax=ax, shrink=0.75, label=r"$\Delta h$")
        ax.contour(X2, Y2, world.occupancy[:, :, k].astype(float), levels=[0.5], colors="black", linewidths=1.0)
        ax.set_title(rf"Numerical $\Delta h$ — {method}")
        ax.set_xlabel("x [m]")
        if col == 0:
            ax.set_ylabel("y [m]")

    fig.suptitle("Poisson forcing and numerical Laplacian consistency", fontsize=15)
    save_figure(fig, figures_dir, "07_forcing_and_laplacian_slices")


def plot_poisson_3d_slices(world: WorldData, cases: Dict[str, PoissonCase], figures_dir: Path) -> None:
    """3D perspective view of h contours for each forcing method."""
    methods = [m for m in cases if cases[m].status == "ok"]
    if not methods:
        return

    fig = plt.figure(figsize=(5.4 * len(methods), 5.2))
    z_planes = [2.0, 4.8, 7.2]

    for col, method in enumerate(methods, start=1):
        ax = fig.add_subplot(1, len(methods), col, projection="3d")
        result = cases[method].result
        h_plot = finite_field_for_plot(result.h, world.occupancy)
        all_vals = h_plot[np.isfinite(h_plot)]
        levels = np.linspace(np.nanpercentile(all_vals, 5), np.nanpercentile(all_vals, 95), 18)

        for zp in z_planes:
            k = slice_index(world.z, zp)
            X2 = world.X[:, :, k]
            Y2 = world.Y[:, :, k]
            H2 = h_plot[:, :, k]
            ax.contourf(X2, Y2, H2, zdir="z", offset=world.z[k], levels=levels, cmap="viridis", alpha=0.72)
            ax.contour(X2, Y2, world.occupancy[:, :, k].astype(float), zdir="z", offset=world.z[k], levels=[0.5], colors="black", linewidths=0.6)

        draw_world_wireframe(ax, world, alpha=0.16)
        ax.scatter(*world.start, s=60, marker="o")
        ax.scatter(*world.goal, s=70, marker="x")
        configure_3d_axes(ax, world, f"3D Poisson h slices — {method}")

    fig.suptitle("3D perspective of Poisson safety field slices", fontsize=15)
    save_figure(fig, figures_dir, "08_forcing_poisson_3d_slices")


# =============================================================================
# Figures: alpha trajectories and metrics
# =============================================================================


def colormap_for_values(values: Sequence[float], cmap_name: str = "turbo") -> Tuple[Any, Normalize]:
    """Return colormap and log/linear normalization for alpha values."""
    arr = np.asarray(values, dtype=float)
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=float(np.min(arr)), vmax=float(np.max(arr)))
    return cmap, norm


def plot_alpha_all_trajectories(world: WorldData, alpha_runs: List[RolloutData], fixed_case: PoissonCase, figures_dir: Path) -> None:
    """Show every alpha trajectory in 3D and in projections."""
    if not alpha_runs:
        return
    alphas = [r.alpha for r in alpha_runs]
    cmap, norm = colormap_for_values(alphas)

    fig = plt.figure(figsize=(17, 12))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    draw_world_wireframe(ax3d, world, alpha=0.25)
    for r in alpha_runs:
        ax3d.plot(r.trajectory[:, 0], r.trajectory[:, 1], r.trajectory[:, 2], color=cmap(norm(r.alpha)), linewidth=2.0, label=f"{r.alpha:g}")
    ax3d.plot(alpha_runs[0].nominal_trajectory[:, 0], alpha_runs[0].nominal_trajectory[:, 1], alpha_runs[0].nominal_trajectory[:, 2], "--", color="0.25", linewidth=1.5, label="nominal")
    ax3d.scatter(*world.start, s=75, marker="o")
    ax3d.scatter(*world.goal, s=90, marker="x")
    configure_3d_axes(ax3d, world, "3D trajectories: conservative → aggressive alpha")

    projections = [
        (0, 1, "XY top view", "x [m]", "y [m]"),
        (0, 2, "XZ side view", "x [m]", "z [m]"),
        (1, 2, "YZ side view", "y [m]", "z [m]"),
    ]
    for idx, (a, b, title, xlabel, ylabel) in enumerate(projections, start=2):
        ax = fig.add_subplot(2, 2, idx)
        # Obstacle projection contour.
        if (a, b) == (0, 1):
            proj = np.max(world.occupancy, axis=2).T
            extent = [0, world.Lx, 0, world.Ly]
        elif (a, b) == (0, 2):
            proj = np.max(world.occupancy, axis=1).T
            extent = [0, world.Lx, 0, world.Lz]
        else:
            proj = np.max(world.occupancy, axis=0).T
            extent = [0, world.Ly, 0, world.Lz]
        ax.imshow(proj, origin="lower", extent=extent, aspect="auto", cmap="gray_r", alpha=0.20)
        ax.plot(alpha_runs[0].nominal_trajectory[:, a], alpha_runs[0].nominal_trajectory[:, b], "--", color="0.25", linewidth=1.3, label="nominal")
        for r in alpha_runs:
            ax.plot(r.trajectory[:, a], r.trajectory[:, b], color=cmap(norm(r.alpha)), linewidth=2.0)
        ax.scatter(world.start[a], world.start[b], s=55, marker="o")
        ax.scatter(world.goal[a], world.goal[b], s=65, marker="x")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axis("equal")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.72, pad=0.03)
    cbar.set_label(r"HOCBF alpha scale $\alpha$")

    fig.suptitle("Alpha sensitivity: all trajectories shown on the same obstacle world", fontsize=16)
    save_figure(fig, figures_dir, "02_alpha_all_trajectories")


def plot_alpha_quantitative_metrics(alpha_runs: List[RolloutData], figures_dir: Path) -> None:
    """Plot alpha versus safety/performance metrics."""
    if not alpha_runs:
        return
    alpha = np.asarray([r.alpha for r in alpha_runs], dtype=float)
    min_h = np.asarray([r.min_h for r in alpha_runs])
    filtered = np.asarray([r.filtered_fraction for r in alpha_runs])
    mean_corr = np.asarray([r.mean_correction for r in alpha_runs])
    max_corr = np.asarray([r.max_correction for r in alpha_runs])
    final_dist = np.asarray([r.final_distance for r in alpha_runs])
    solve_ms = np.asarray([r.mean_cbf_solve_ms for r in alpha_runs])
    path_length = np.asarray([r.path_length for r in alpha_runs])

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    metrics = [
        (min_h, "Minimum safety value", "min h"),
        (filtered, "Filtered fraction", "fraction"),
        (mean_corr, "Mean correction", r"mean $||a_{safe}-a_{nom}||$"),
        (max_corr, "Maximum correction", r"max $||a_{safe}-a_{nom}||$"),
        (final_dist, "Final distance to goal", "m"),
        (path_length, "Path length", "m"),
    ]
    for ax, (y, title, ylabel) in zip(axes.flat, metrics):
        ax.plot(alpha, y, marker="o")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel(r"HOCBF alpha scale $\alpha$")
        ax.set_ylabel(ylabel)

    # Add CBF timing as a small inset-like text into the last plot.
    axes.flat[-1].text(
        0.03,
        0.95,
        f"mean CBF step time: {np.nanmean(solve_ms):.3f} ms",
        transform=axes.flat[-1].transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.suptitle("Alpha sensitivity: safety, control effort, and terminal performance", fontsize=16)
    save_figure(fig, figures_dir, "03_alpha_quantitative_metrics")


def plot_alpha_representative_time_histories(alpha_runs: List[RolloutData], figures_dir: Path) -> None:
    """Show h(t), residual, and correction for conservative/medium/aggressive cases."""
    if not alpha_runs:
        return
    sorted_runs = sorted(alpha_runs, key=lambda r: r.alpha)
    selected = [sorted_runs[0], sorted_runs[len(sorted_runs) // 2], sorted_runs[-1]]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for r in selected:
        t = np.arange(len(r.h_history)) * r.dt
        correction = np.linalg.norm(r.a_safe_history - r.a_nom_history, axis=1) if len(r.a_safe_history) else np.zeros(0)
        axes[0].plot(t, r.h_history, label=rf"$\alpha={r.alpha:g}$")
        axes[1].plot(t, r.residual_history, label=rf"$\alpha={r.alpha:g}$")
        axes[2].plot(t, correction, label=rf"$\alpha={r.alpha:g}$")

    axes[0].axhline(0.0, linestyle="--", color="0.3", linewidth=1.0)
    axes[1].axhline(0.0, linestyle="--", color="0.3", linewidth=1.0)
    axes[0].set_ylabel("h(p)")
    axes[1].set_ylabel("HOCBF residual")
    axes[2].set_ylabel(r"$||a_{safe}-a_{nom}||$")
    axes[2].set_xlabel("time [s]")
    axes[0].set_title("Safety value over time")
    axes[1].set_title("HOCBF constraint residual over time")
    axes[2].set_title("Acceleration correction over time")
    for ax in axes:
        ax.legend(loc="best")
    fig.suptitle("Representative histories: conservative, nominal, and aggressive CBF settings", fontsize=15)
    save_figure(fig, figures_dir, "04_alpha_representative_time_histories")


# =============================================================================
# Figures: forcing trajectories, forcing metrics, solver timing, dashboard
# =============================================================================


def plot_forcing_all_trajectories(world: WorldData, forcing_runs: Dict[str, RolloutData], figures_dir: Path) -> None:
    """Overlay trajectories produced by different Poisson forcing functions."""
    methods = list(forcing_runs.keys())
    if not methods:
        return
    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(16, 10))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    draw_world_wireframe(ax3d, world, alpha=0.30)
    for i, method in enumerate(methods):
        r = forcing_runs[method]
        ax3d.plot(r.trajectory[:, 0], r.trajectory[:, 1], r.trajectory[:, 2], color=cmap(i), linewidth=2.4, label=method)
    first = forcing_runs[methods[0]]
    ax3d.plot(first.nominal_trajectory[:, 0], first.nominal_trajectory[:, 1], first.nominal_trajectory[:, 2], "--", color="0.25", linewidth=1.3, label="nominal")
    ax3d.scatter(*world.start, s=70, marker="o")
    ax3d.scatter(*world.goal, s=85, marker="x")
    configure_3d_axes(ax3d, world, "3D forcing-function trajectory comparison")
    ax3d.legend(loc="upper left")

    ax = fig.add_subplot(1, 2, 2)
    proj = np.max(world.occupancy, axis=2).T
    ax.imshow(proj, origin="lower", extent=[0, world.Lx, 0, world.Ly], aspect="auto", cmap="gray_r", alpha=0.20)
    ax.plot(first.nominal_trajectory[:, 0], first.nominal_trajectory[:, 1], "--", color="0.25", linewidth=1.3, label="nominal")
    for i, method in enumerate(methods):
        r = forcing_runs[method]
        ax.plot(r.trajectory[:, 0], r.trajectory[:, 1], color=cmap(i), linewidth=2.4, label=method)
    ax.scatter(world.start[0], world.start[1], s=60, marker="o")
    ax.scatter(world.goal[0], world.goal[1], s=70, marker="x")
    ax.set_title("XY projection")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.legend(loc="best")

    fig.suptitle("Forcing-function comparison: geometry of h changes the filtered trajectory", fontsize=16)
    save_figure(fig, figures_dir, "09_forcing_all_trajectories")


def plot_forcing_quantitative_metrics(cases: Dict[str, PoissonCase], forcing_runs: Dict[str, RolloutData], figures_dir: Path) -> None:
    """Bar/line comparisons for forcing method performance and timing."""
    methods = [m for m in forcing_runs.keys()]
    if not methods:
        return
    x = np.arange(len(methods))
    poisson_time = np.asarray([cases[m].wall_time_sec for m in methods])
    rollout_time = np.asarray([forcing_runs[m].rollout_wall_time_sec for m in methods])
    min_h = np.asarray([forcing_runs[m].min_h for m in methods])
    filtered = np.asarray([forcing_runs[m].filtered_fraction for m in methods])
    correction = np.asarray([forcing_runs[m].mean_correction for m in methods])
    final_dist = np.asarray([forcing_runs[m].final_distance for m in methods])

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    bars = [
        (poisson_time, "Poisson wall time", "s"),
        (rollout_time, "Rollout wall time", "s"),
        (min_h, "Minimum h", "h"),
        (filtered, "Filtered fraction", "fraction"),
        (correction, "Mean correction", r"mean $||a_s-a_n||$"),
        (final_dist, "Final distance", "m"),
    ]
    for ax, (vals, title, ylabel) in zip(axes.flat, bars):
        ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    fig.suptitle("Forcing-function quantitative comparison", fontsize=16)
    save_figure(fig, figures_dir, "10_forcing_quantitative_metrics")


def plot_solver_timing(solver_cases: List[PoissonCase], figures_dir: Path) -> None:
    """Plot solver timing comparison."""
    if not solver_cases:
        return
    labels = [c.solver for c in solver_cases]
    total = np.asarray([c.wall_time_sec for c in solver_cases], dtype=float)
    solve = np.asarray([float(c.timing.get("solve", np.nan)) for c in solver_cases], dtype=float)
    forcing = np.asarray([float(c.timing.get("forcing", np.nan)) for c in solver_cases], dtype=float)
    derivative = np.asarray([float(c.timing.get("derivatives", np.nan)) for c in solver_cases], dtype=float)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.22
    ax.bar(x - width, total, width, label="total wall")
    ax.bar(x, solve, width, label="solve stage")
    ax.bar(x + width, forcing + derivative, width, label="forcing + derivatives")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("time [s]")
    ax.set_title("Poisson solver timing comparison")
    ax.legend(loc="best")
    for i, c in enumerate(solver_cases):
        if c.status != "ok":
            ax.text(i, 0.02, "failed", rotation=90, ha="center", va="bottom")
    save_figure(fig, figures_dir, "11_solver_timing")


def plot_integrated_dashboard(world: WorldData, alpha_runs: List[RolloutData], cases: Dict[str, PoissonCase],
                              forcing_runs: Dict[str, RolloutData], figures_dir: Path) -> None:
    """One highly compressed dashboard for presentations/reports."""
    if not alpha_runs:
        return
    methods = [m for m in cases if cases[m].status == "ok"]
    fixed_result = cases[alpha_runs[0].forcing_method].result
    z_k = slice_index(world.z, 4.8)
    X2, Y2 = world.X[:, :, z_k], world.Y[:, :, z_k]
    h_plot = finite_field_for_plot(fixed_result.h, world.occupancy)
    alphas = [r.alpha for r in alpha_runs]
    cmap, norm = colormap_for_values(alphas)

    fig = plt.figure(figsize=(18, 11))

    ax = fig.add_subplot(2, 3, 1, projection="3d")
    draw_world_wireframe(ax, world, alpha=0.28)
    for r in alpha_runs:
        ax.plot(r.trajectory[:, 0], r.trajectory[:, 1], r.trajectory[:, 2], color=cmap(norm(r.alpha)), linewidth=1.8)
    ax.scatter(*world.start, s=70, marker="o")
    ax.scatter(*world.goal, s=85, marker="x")
    configure_3d_axes(ax, world, "Alpha sweep trajectories")

    ax = fig.add_subplot(2, 3, 2)
    cs = ax.contourf(X2, Y2, h_plot[:, :, z_k], levels=36, cmap="viridis")
    ax.contour(X2, Y2, world.occupancy[:, :, z_k].astype(float), levels=[0.5], colors="black", linewidths=1.0)
    ax.set_title(f"Poisson h slice, forcing={alpha_runs[0].forcing_method}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(cs, ax=ax, shrink=0.78, label="h")

    ax = fig.add_subplot(2, 3, 3)
    for r in [alpha_runs[0], alpha_runs[len(alpha_runs)//2], alpha_runs[-1]]:
        t = np.arange(len(r.h_history)) * r.dt
        ax.plot(t, r.h_history, label=rf"$\alpha={r.alpha:g}$")
    ax.axhline(0, linestyle="--", color="0.25", linewidth=1.0)
    ax.set_title("Representative h(t)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("h")
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 3, 4)
    alpha = np.asarray([r.alpha for r in alpha_runs])
    ax.plot(alpha, [r.filtered_fraction for r in alpha_runs], marker="o", label="filtered fraction")
    ax.plot(alpha, [r.mean_correction for r in alpha_runs], marker="s", label="mean correction")
    ax.set_xscale("log")
    ax.set_title("Alpha safety/control metrics")
    ax.set_xlabel(r"$\alpha$")
    ax.legend(loc="best")

    ax = fig.add_subplot(2, 3, 5)
    if forcing_runs:
        names = list(forcing_runs.keys())
        ax.bar(np.arange(len(names)), [cases[n].wall_time_sec for n in names])
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_title("Poisson compute time by forcing")
    ax.set_ylabel("time [s]")

    ax = fig.add_subplot(2, 3, 6)
    if forcing_runs:
        names = list(forcing_runs.keys())
        ax.bar(np.arange(len(names)), [forcing_runs[n].min_h for n in names])
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_title("Minimum h by forcing")
    ax.set_ylabel("min h")

    fig.suptitle("Poisson-HOCBF landing study: field construction, safety filtering, and timing", fontsize=17)
    save_figure(fig, figures_dir, "12_integrated_nasa_dashboard")


# =============================================================================
# I/O utilities
# =============================================================================


def safe_number(x: Any) -> Any:
    """Convert numpy scalar/array/path values for JSON."""
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): safe_number(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe_number(v) for v in x]
    return x


def write_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write list of dictionaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: safe_number(row.get(k, "")) for k in keys})


def rollout_to_row(r: RolloutData) -> Dict[str, Any]:
    """Compact metrics row from rollout."""
    return {
        "label": r.label,
        "forcing_method": r.forcing_method,
        "solver": r.solver,
        "alpha": r.alpha,
        "alpha1": r.alpha1,
        "alpha2": r.alpha2,
        "steps": r.steps,
        "reached_goal": r.reached_goal,
        "collision": r.collision,
        "final_distance": r.final_distance,
        "path_length": r.path_length,
        "min_h": r.min_h,
        "min_residual": r.min_residual,
        "filtered_fraction": r.filtered_fraction,
        "mean_correction": r.mean_correction,
        "max_correction": r.max_correction,
        "mean_cbf_solve_ms": r.mean_cbf_solve_ms,
        "rollout_wall_time_sec": r.rollout_wall_time_sec,
    }


def write_paper_figure_guide(path: Path) -> None:
    """Save a guide for using the generated figures in a report/paper."""
    text = """# Generated Figure Guide

Recommended figures for an interim report or paper:

1. `00_world_occupancy_boundary_3d.png`
   - Shows the constructed 3D obstacle world, occupancy matrix projection, and boundary/frontier cells.

2. `01_occupancy_boundary_slices.png`
   - Shows how the 3D occupancy matrix changes across altitude slices.

3. `02_alpha_all_trajectories.png`
   - Shows all trajectories for the alpha sweep, from conservative to aggressive HOCBF behavior.

4. `03_alpha_quantitative_metrics.png`
   - Quantifies safety, control effort, final distance, and path length as alpha changes.

5. `04_alpha_representative_time_histories.png`
   - Shows h(t), HOCBF residual, and correction magnitude for representative alpha values.

6. `06_forcing_h_and_vector_fields.png`
   - Compares Poisson safety fields and induced vector directions across forcing functions.

7. `07_forcing_and_laplacian_slices.png`
   - Shows the forcing field f and numerical Δh consistency.

8. `08_forcing_poisson_3d_slices.png`
   - Shows 3D Poisson contour slices for each forcing function.

9. `09_forcing_all_trajectories.png`
   - Shows how different forcing choices affect the final filtered trajectory.

10. `10_forcing_quantitative_metrics.png`
    - Compares Poisson time, rollout time, min h, filtered fraction, correction, and final distance.

11. `11_solver_timing.png`
    - Compares Poisson solver timing.

12. `12_integrated_nasa_dashboard.png`
    - One integrated dashboard for talks or report overview.

Suggested explanation:

- Alpha controls how aggressively the HOCBF permits the trajectory to follow the nominal controller.
- The forcing function changes the geometry of the global Poisson safety field h.
- The gradient of h determines the local CBF correction direction.
- Poisson solve time is a field-construction cost, while CBF solve time is an online control-loop cost.
"""
    path.write_text(text, encoding="utf-8")


# =============================================================================
# Main study
# =============================================================================


def parse_csv_floats(text: str) -> List[float]:
    """Parse comma-separated floats."""
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> List[str]:
    """Parse comma-separated strings."""
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_grid_shape(text: str) -> Tuple[int, int, int]:
    """Parse grid shape as nx,ny,nz."""
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("grid shape must be nx,ny,nz")
    if min(parts) < 8:
        raise argparse.ArgumentTypeError("grid dimensions must be at least 8")
    return tuple(parts)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NASA/Caltech-style 3D Poisson-HOCBF parameter study.")
    parser.add_argument("--output-dir", default="outputs/nasa_poisson_cbf_study", help="Output directory for figures and data.")
    parser.add_argument("--grid-shape", type=parse_grid_shape, default=(64, 52, 36), help="3D grid shape as nx,ny,nz.")
    parser.add_argument(
        "--alphas",
        default="0.05,0.08,0.12,0.2,0.35,0.5,0.75,1,1.5,2,3,5,8",
        help="Comma-separated alpha scale values for the HOCBF sweep.",
    )
    parser.add_argument(
        "--forcing-methods",
        default="constant,distance,average_flux,guidance",
        help="Comma-separated Poisson forcing methods.",
    )
    parser.add_argument("--fixed-forcing", default="guidance", help="Forcing method used for the alpha sweep.")
    parser.add_argument("--fixed-alpha", type=float, default=0.5, help="Alpha used for forcing comparison.")
    parser.add_argument("--solver", default="sor", choices=["sor", "sparse_direct", "conjugate_gradient"], help="Main Poisson solver.")
    parser.add_argument("--solver-sweep", default="sor,sparse_direct,conjugate_gradient", help="Solvers to compare.")
    parser.add_argument("--skip-solver-sweep", action="store_true", help="Skip solver timing comparison.")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation time step.")
    parser.add_argument("--max-steps", type=int, default=1100, help="Maximum rollout steps.")
    parser.add_argument("--goal-tolerance", type=float, default=0.25, help="Goal tolerance in meters.")
    parser.add_argument("--max-acc", type=float, default=1.3, help="Nominal acceleration saturation.")
    parser.add_argument("--max-speed", type=float, default=2.0, help="Velocity saturation.")
    parser.add_argument("--sor-max-iter", type=int, default=900, help="SOR maximum iterations.")
    parser.add_argument("--sor-tolerance", type=float, default=1.0e-3, help="SOR tolerance.")
    parser.add_argument("--sor-omega", type=float, default=1.75, help="SOR relaxation factor.")
    parser.add_argument("--cg-max-iter", type=int, default=2000, help="CG maximum iterations.")
    parser.add_argument("--cg-tolerance", type=float, default=1.0e-5, help="CG tolerance.")
    parser.add_argument("--no-pdf", action="store_true", help="Only save PNG figures, not PDF.")
    return parser.parse_args()


def main() -> None:
    set_publication_style()
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    global_save_pdf = not args.no_pdf

    # Rebind save_figure PDF behavior through a tiny wrapper.
    global save_figure
    _save_figure_original = save_figure

    def _save_figure_with_arg(fig: plt.Figure, figs: Path, name: str, save_pdf: bool = True) -> None:
        _save_figure_original(fig, figs, name, save_pdf=global_save_pdf and save_pdf)

    save_figure = _save_figure_with_arg  # type: ignore[assignment]

    alphas = parse_csv_floats(args.alphas)
    forcing_methods = parse_csv_strings(args.forcing_methods)
    solver_sweep = parse_csv_strings(args.solver_sweep)

    if args.fixed_forcing not in forcing_methods:
        forcing_methods.append(args.fixed_forcing)

    world = make_research_world(args.grid_shape)
    print("[1/8] Built research world")
    print("      occupancy fraction:", float(np.mean(world.occupancy)))
    print("      boundary fraction: ", float(np.mean(world.boundary_mask)))

    # Save raw occupancy/boundary.
    np.savez_compressed(
        data_dir / "world_occupancy_boundary.npz",
        occupancy=world.occupancy,
        boundary_mask=world.boundary_mask,
        x=world.x,
        y=world.y,
        z=world.z,
        start=world.start,
        goal=world.goal,
    )

    plot_world_occupancy_boundary_3d(world, figures_dir)
    plot_occupancy_boundary_slices(world, figures_dir)
    print("[2/8] Saved world/occupancy/boundary figures")

    # Compute Poisson fields for forcing comparison.
    cases: Dict[str, PoissonCase] = {}
    for method in forcing_methods:
        print(f"[3/8] Computing Poisson forcing={method}, solver={args.solver} ...")
        case = compute_poisson_case(
            world=world,
            forcing_method=method,
            solver=args.solver,
            sor_max_iter=args.sor_max_iter,
            sor_tolerance=args.sor_tolerance,
            sor_omega=args.sor_omega,
            cg_max_iter=args.cg_max_iter,
            cg_tolerance=args.cg_tolerance,
        )
        cases[method] = case
        print(f"      status={case.status}, time={case.wall_time_sec:.3f}s")
        if case.status != "ok":
            print(f"      error={case.error}")

    if cases[args.fixed_forcing].status != "ok":
        raise RuntimeError(f"Fixed forcing {args.fixed_forcing!r} failed: {cases[args.fixed_forcing].error}")

    # Alpha sweep reusing the fixed forcing field.
    fixed_case = cases[args.fixed_forcing]
    alpha_runs: List[RolloutData] = []
    for alpha in alphas:
        print(f"[4/8] Rolling out alpha={alpha:g} with forcing={args.fixed_forcing} ...")
        r = rollout_acceleration_hocbf(
            world=world,
            poisson_case=fixed_case,
            alpha=alpha,
            dt=args.dt,
            max_steps=args.max_steps,
            goal_tolerance=args.goal_tolerance,
            max_acc=args.max_acc,
            max_speed=args.max_speed,
        )
        alpha_runs.append(r)
        print(f"      filtered={r.filtered_fraction:.3f}, min_h={r.min_h:.3g}, final_dist={r.final_distance:.3f}")

    # Forcing sweep with fixed alpha.
    forcing_runs: Dict[str, RolloutData] = {}
    for method, case in cases.items():
        if case.status != "ok":
            continue
        print(f"[5/8] Rolling out forcing={method} with alpha={args.fixed_alpha:g} ...")
        r = rollout_acceleration_hocbf(
            world=world,
            poisson_case=case,
            alpha=args.fixed_alpha,
            dt=args.dt,
            max_steps=args.max_steps,
            goal_tolerance=args.goal_tolerance,
            max_acc=args.max_acc,
            max_speed=args.max_speed,
        )
        forcing_runs[method] = r
        print(f"      filtered={r.filtered_fraction:.3f}, min_h={r.min_h:.3g}, final_dist={r.final_distance:.3f}")

    print("[6/8] Generating paper-grade figures ...")
    plot_alpha_all_trajectories(world, alpha_runs, fixed_case, figures_dir)
    plot_alpha_quantitative_metrics(alpha_runs, figures_dir)
    plot_alpha_representative_time_histories(alpha_runs, figures_dir)
    plot_forcing_h_and_vector_fields(world, cases, forcing_runs, figures_dir)
    plot_forcing_scalar_and_laplacian(world, cases, figures_dir)
    plot_poisson_3d_slices(world, cases, figures_dir)
    plot_forcing_all_trajectories(world, forcing_runs, figures_dir)
    plot_forcing_quantitative_metrics(cases, forcing_runs, figures_dir)

    # Solver timing sweep.
    solver_cases: List[PoissonCase] = []
    if not args.skip_solver_sweep:
        for solver in solver_sweep:
            print(f"[7/8] Solver timing forcing={args.fixed_forcing}, solver={solver} ...")
            scase = compute_poisson_case(
                world=world,
                forcing_method=args.fixed_forcing,
                solver=solver,
                sor_max_iter=args.sor_max_iter,
                sor_tolerance=args.sor_tolerance,
                sor_omega=args.sor_omega,
                cg_max_iter=args.cg_max_iter,
                cg_tolerance=args.cg_tolerance,
            )
            solver_cases.append(scase)
            print(f"      status={scase.status}, time={scase.wall_time_sec:.3f}s")
        plot_solver_timing(solver_cases, figures_dir)

    plot_integrated_dashboard(world, alpha_runs, cases, forcing_runs, figures_dir)

    # Save data.
    print("[8/8] Saving metrics and data ...")
    write_metrics_csv(data_dir / "alpha_sweep_metrics.csv", [rollout_to_row(r) for r in alpha_runs])
    write_metrics_csv(data_dir / "forcing_sweep_metrics.csv", [rollout_to_row(r) for r in forcing_runs.values()])
    write_metrics_csv(
        data_dir / "poisson_cases_metrics.csv",
        [
            {
                "forcing_method": m,
                "solver": c.solver,
                "status": c.status,
                "error": c.error,
                "wall_time_sec": c.wall_time_sec,
                **{f"timing_{k}": v for k, v in c.timing.items()},
            }
            for m, c in cases.items()
        ],
    )
    write_metrics_csv(
        data_dir / "solver_timing_metrics.csv",
        [
            {
                "forcing_method": c.forcing_method,
                "solver": c.solver,
                "status": c.status,
                "error": c.error,
                "wall_time_sec": c.wall_time_sec,
                **{f"timing_{k}": v for k, v in c.timing.items()},
            }
            for c in solver_cases
        ],
    )

    # Save representative arrays for later custom plotting.
    np.savez_compressed(
        data_dir / "alpha_rollouts_compact.npz",
        alphas=np.asarray([r.alpha for r in alpha_runs]),
        # Variable-length arrays are stored as object arrays for convenience.
        trajectories=np.asarray([r.trajectory for r in alpha_runs], dtype=object),
        h_histories=np.asarray([r.h_history for r in alpha_runs], dtype=object),
        residual_histories=np.asarray([r.residual_history for r in alpha_runs], dtype=object),
    )

    metadata = {
        "args": safe_number(vars(args)),
        "world": {
            "bounds": [world.Lx, world.Ly, world.Lz],
            "grid_shape": [world.nx, world.ny, world.nz],
            "spacing": [world.dx, world.dy, world.dz],
            "occupancy_fraction": float(np.mean(world.occupancy)),
            "boundary_fraction": float(np.mean(world.boundary_mask)),
            "start": world.start.tolist(),
            "goal": world.goal.tolist(),
        },
        "figure_directory": str(figures_dir),
        "data_directory": str(data_dir),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_paper_figure_guide(output_dir / "PAPER_FIGURE_GUIDE.md")

    print("\nDONE")
    print("Figures:", figures_dir)
    print("Data:   ", data_dir)
    print("Recommended first image:", figures_dir / "12_integrated_nasa_dashboard.png")


if __name__ == "__main__":
    main()