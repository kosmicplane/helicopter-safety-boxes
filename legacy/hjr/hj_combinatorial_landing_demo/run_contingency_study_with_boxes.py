#!/usr/bin/env python3
"""Offline Poisson-CBF-HOCBF contingency-aware Mars-analog landing study.

This runtime orchestrates the uploaded ``poisson_safety_box`` and
``cbf_safety_box`` packages in a reduced-order research simulation.  It does not
reimplement the Poisson PDE or the CBF-QP internally.  A double-integrator aerial
vehicle pursues one landing zone while preserving
reachability-proxy certificates for at least r out of p candidate zones.  It combines:

* a 3-D occupancy world passed to ``PoissonSafetyBox.compute()``,
* acceleration-level HOCBF collision avoidance built and solved by ``CBFBox``,
* a paper-inspired r-out-of-p combinatorial CBF filter,
* an active-target CLF condition (softened only if bounded-input feasibility fails),
* smooth resource depletion and a primary-zone failure/switch event,
* solver, forcing-function, and HOCBF-gain studies,
* report-quality PNG/PDF figures and machine-readable logs.

The r-out-of-p preservation constraints follow the combinatorial CBF construction in
"Steering with Contingencies: Combinatorial Stabilization and Reach-Avoid Filters"
(Lishkova, Ong, Tonkens, Herbert, and Ames, 2026): the pivot is the r-th largest
certificate, and p smooth affine-in-control inequalities preserve its nonnegative
superlevel set without enumerating target combinations.

Important scope statement
-------------------------
The landing-zone certificates in this script are smooth CLF/geodesic/Poisson
reachability *proxies*. They are not Hamilton-Jacobi reach-avoid value functions and
are not a full flight-stack certificate. The simulation is offline, reduced-order,
not PX4 validated, and not rate-controller validated.

Python: 3.12+
Dependencies: numpy, scipy, matplotlib, pandas; scikit-image is optional but used for
higher-quality isosurfaces when available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings

# Small/medium sparse linear algebra is much faster and more reproducible with a
# controlled BLAS thread count. Override with POISSON_CONTINGENCY_NUM_THREADS.
_NUM_THREADS = os.environ.get("POISSON_CONTINGENCY_NUM_THREADS", "1")
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = _NUM_THREADS

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.linalg import solve_continuous_lyapunov
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import LinearOperator

# The deliverable is self-contained: the two uploaded safety boxes are placed
# beside this runtime.  Adding their package roots here allows the study to run
# directly without a prior ``pip install -e`` step, while still importing and
# executing the real packages rather than copying their algorithms.
_PROJECT_ROOT = Path(__file__).resolve().parent
for _package_root in (_PROJECT_ROOT / "poisson_safety_box", _PROJECT_ROOT / "cbf_safety_box"):
    if str(_package_root) not in sys.path:
        sys.path.insert(0, str(_package_root))

from poisson_safety_box import PoissonBoxConfig, PoissonBoxResult, PoissonSafetyBox
from cbf_safety_box import (
    AffineCertificate,
    CBFBox,
    CBFBoxConfig,
    SystemState,
    build_acceleration_hocbf_constraint,
    build_active_target_clf_constraint,
    build_combinatorial_contingency_constraints,
    lift_constraint_with_auxiliary,
)
from cbf_safety_box.safety_data.poisson_adapter import sample_from_arrays

try:
    from skimage.measure import marching_cubes

    HAVE_SKIMAGE = True
except Exception:  # pragma: no cover - optional dependency
    marching_cubes = None
    HAVE_SKIMAGE = False


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GridSpec:
    shape: tuple[int, int, int]
    bounds: tuple[float, float, float] = (18.0, 14.0, 10.0)

    @property
    def nx(self) -> int:
        return self.shape[0]

    @property
    def ny(self) -> int:
        return self.shape[1]

    @property
    def nz(self) -> int:
        return self.shape[2]

    @property
    def x(self) -> np.ndarray:
        return np.linspace(0.0, self.bounds[0], self.nx)

    @property
    def y(self) -> np.ndarray:
        return np.linspace(0.0, self.bounds[1], self.ny)

    @property
    def z(self) -> np.ndarray:
        return np.linspace(0.0, self.bounds[2], self.nz)

    @property
    def spacing(self) -> tuple[float, float, float]:
        return (
            self.bounds[0] / (self.nx - 1),
            self.bounds[1] / (self.ny - 1),
            self.bounds[2] / (self.nz - 1),
        )

    @property
    def world_diagonal_xy(self) -> float:
        return float(math.hypot(self.bounds[0], self.bounds[1]))


@dataclass(frozen=True)
class Obstacle:
    name: str
    kind: str
    params: dict[str, Any]
    category: str = "obstacle"


@dataclass(frozen=True)
class LandingZone:
    index: int
    name: str
    position: np.ndarray
    radius: float
    science_score: float
    terrain_quality: float


@dataclass
class WorldData:
    grid: GridSpec
    occupancy: np.ndarray
    free_mask: np.ndarray
    boundary_mask: np.ndarray
    unknown_mask: np.ndarray
    obstacles: list[Obstacle]
    landing_zones: list[LandingZone]
    start: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class PoissonResult:
    """Thin, data-only view of one real ``PoissonSafetyBox`` execution.

    ``box_result`` is the authoritative package output.  The normalized arrays
    below are linear rescalings used by the research experiment; no PDE, forcing,
    gradient, or Hessian is reconstructed in this runtime.
    """

    forcing_method: str
    solver: str
    box_result: PoissonBoxResult
    h_raw: np.ndarray
    h: np.ndarray
    h_cbf: np.ndarray
    grad_h: np.ndarray
    hessian_h: np.ndarray
    solve_time_s: float
    relative_residual: float
    solver_info: dict[str, Any]
    iterations: int
    normalization_scale: float

    @property
    def forcing(self) -> np.ndarray:
        """Expose the forcing array produced by ``poisson_safety_box``."""
        return self.box_result.forcing

    @property
    def grad(self) -> np.ndarray:
        """Compatibility alias used by the plotting functions."""
        return self.grad_h

    @property
    def hessian(self) -> np.ndarray:
        """Compatibility alias used by the plotting functions."""
        return self.hessian_h



@dataclass
class GeodesicField:
    distance: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray


@dataclass
class ReachabilityModel:
    zones: list[LandingZone]
    geodesics: list[GeodesicField]
    P: np.ndarray
    quad_scales: np.ndarray
    c0: np.ndarray
    base_margins: np.ndarray
    beta_time: float
    beta_energy: float
    block_drop: float
    block_transition_s: float
    w_geodesic: float
    w_poisson_risk: float
    w_velocity: float
    risk_trigger: float
    risk_temperature: float
    world_diagonal_xy: float
    max_speed: float
    initial_values: np.ndarray
    initial_margins: np.ndarray


@dataclass
class SimulationConfig:
    dt: float = 0.05
    max_steps: int = 850
    Kp: float = 0.95
    Kd: float = 1.35
    max_acc: float = 1.35
    max_speed: float = 2.0
    hocbf_alpha: float = 0.5
    contingency_gamma: float = 0.18
    rho_aux_gain: float = 0.18
    task_gamma: float = 0.025
    omega_cost: float = 0.10
    omega_max: float = 80.0
    r_contingency: int = 2
    active_zone: int = 0
    failure_time: float = 18.0
    blocked_zone: int = 0
    approach_altitude: float = 4.2
    descent_delay_after_failure: float = 1.8
    descent_delay_after_switch: float = 2.5
    descent_horizontal_radius: float = 2.4
    landing_position_tolerance: float = 0.62
    landing_speed_tolerance: float = 0.38
    switch_margin: float = -0.015
    min_switch_dwell_s: float = 1.5
    deterministic_seed: int = 7


@dataclass
class QPSolution:
    acceleration: np.ndarray
    omega: float
    success: bool
    min_residual: float
    emergency_slack: float
    task_relaxed: bool
    iterations: int
    residuals: dict[str, float]


@dataclass
class SimulationResult:
    forcing_method: str
    solver: str
    hocbf_alpha: float
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    nominal_acceleration: np.ndarray
    safe_acceleration: np.ndarray
    reference: np.ndarray
    active_zone: np.ndarray
    h_value: np.ndarray
    hocbf_residual: np.ndarray
    task_clf_residual: np.ndarray
    contingency_residual: np.ndarray
    rho: np.ndarray
    contingency_margin: np.ndarray
    reachable_count: np.ndarray
    omega: np.ndarray
    correction_norm: np.ndarray
    energy_used: np.ndarray
    emergency_slack: np.ndarray
    task_relaxed: np.ndarray
    switch_events: list[dict[str, Any]]
    blocked_events: list[dict[str, Any]]
    terminated_reason: str
    landed: bool
    collided: bool
    contingency_lost: bool
    final_zone: int
    metrics: dict[str, Any]


# -----------------------------------------------------------------------------
# Utilities and configuration
# -----------------------------------------------------------------------------


ZONE_COLORS = ["#b2182b", "#2166ac", "#1b7837", "#762a83", "#d6604d", "#4d4d4d"]
FORCING_COLORS = {
    "constant": "#4c78a8",
    "distance": "#f58518",
    "average_flux": "#54a24b",
    "guidance": "#b279a2",
}
SOLVER_LABELS = {
    "sparse_direct": "Sparse direct",
    "conjugate_gradient": "Conjugate gradient",
    "sor": "SOR",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.titlesize": 14.0,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def parse_csv_floats(text: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid comma-separated float list: {text}") from exc
    if not values:
        raise argparse.ArgumentTypeError("The list must contain at least one value.")
    return values


def parse_csv_strings(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("The list must contain at least one item.")
    return values


def parse_grid_shape(text: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(item.strip()) for item in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid grid shape: {text}") from exc
    if len(values) != 3 or min(values) < 8:
        raise argparse.ArgumentTypeError("Grid shape must be nx,ny,nz with each dimension >= 8.")
    return values  # type: ignore[return-value]


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json_value(v) for v in value]
    if isinstance(value, (bool, str, int, float)) or value is None:
        return value
    return str(value)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe_json_value(data), handle, indent=2, sort_keys=True)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int, save_pdf: bool) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi)
    if save_pdf:
        fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def unit_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return np.zeros_like(vector)
    return vector / norm


def clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm < 1e-12:
        return vector.copy()
    return vector * (maximum / norm)


def rth_largest(values: np.ndarray, r: int) -> float:
    if r < 1 or r > len(values):
        raise ValueError(f"r must be in [1, {len(values)}], got {r}.")
    return float(np.partition(values, len(values) - r)[len(values) - r])

def certificate_pivot_and_count(certificates: list[dict[str, Any]], r: int) -> tuple[float, int]:
    values = np.asarray([item["rho"] for item in certificates if item.get("available", True)], dtype=float)
    count = int(np.count_nonzero(values >= 0.0))
    if len(values) < r:
        return -math.inf, count
    return rth_largest(values, r), count


def smoothstep_and_derivative(t: float, start: float, duration: float) -> tuple[float, float]:
    if t <= start:
        return 0.0, 0.0
    if t >= start + duration:
        return 1.0, 0.0
    q = (t - start) / duration
    value = q * q * (3.0 - 2.0 * q)
    derivative = 6.0 * q * (1.0 - q) / duration
    return float(value), float(derivative)


def stable_softplus(x: float, temperature: float) -> tuple[float, float]:
    """Return tau*log(1+exp(x/tau)) and derivative with respect to x."""
    q = x / temperature
    value = temperature * float(np.logaddexp(0.0, q))
    if q >= 0.0:
        derivative = 1.0 / (1.0 + math.exp(-min(q, 60.0)))
    else:
        eq = math.exp(max(q, -60.0))
        derivative = eq / (1.0 + eq)
    return value, derivative


def sample_trilinear(field: np.ndarray, point: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Sample scalar/vector/tensor field with uniform-grid trilinear interpolation."""
    dx, dy, dz = grid.spacing
    fx = np.clip(point[0] / dx, 0.0, grid.nx - 1.000001)
    fy = np.clip(point[1] / dy, 0.0, grid.ny - 1.000001)
    fz = np.clip(point[2] / dz, 0.0, grid.nz - 1.000001)
    i0, j0, k0 = int(math.floor(fx)), int(math.floor(fy)), int(math.floor(fz))
    i1, j1, k1 = min(i0 + 1, grid.nx - 1), min(j0 + 1, grid.ny - 1), min(k0 + 1, grid.nz - 1)
    tx, ty, tz = fx - i0, fy - j0, fz - k0

    c000 = field[i0, j0, k0]
    c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]
    c110 = field[i1, j1, k0]
    c001 = field[i0, j0, k1]
    c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]
    c111 = field[i1, j1, k1]

    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return np.asarray(c0 * (1.0 - tz) + c1 * tz)


def sample_bilinear(field: np.ndarray, point_xy: np.ndarray, grid: GridSpec) -> float:
    dx, dy, _ = grid.spacing
    fx = np.clip(point_xy[0] / dx, 0.0, grid.nx - 1.000001)
    fy = np.clip(point_xy[1] / dy, 0.0, grid.ny - 1.000001)
    i0, j0 = int(math.floor(fx)), int(math.floor(fy))
    i1, j1 = min(i0 + 1, grid.nx - 1), min(j0 + 1, grid.ny - 1)
    tx, ty = fx - i0, fy - j0
    return float(
        (1.0 - tx) * (1.0 - ty) * field[i0, j0]
        + tx * (1.0 - ty) * field[i1, j0]
        + (1.0 - tx) * ty * field[i0, j1]
        + tx * ty * field[i1, j1]
    )


# -----------------------------------------------------------------------------
# World construction and occupancy rasterization
# -----------------------------------------------------------------------------


def landing_zone_templates() -> list[LandingZone]:
    entries = [
        ("LZ-0 Primary", (16.7, 12.2, 0.85), 0.90, 0.95, 0.86),
        ("LZ-1 South", (16.2, 2.4, 0.85), 0.86, 0.77, 0.90),
        ("LZ-2 North Backup", (13.6, 12.5, 0.85), 0.84, 0.86, 0.82),
        ("LZ-3 Mid-route", (8.8, 1.8, 0.85), 0.82, 0.63, 0.76),
        ("LZ-4 Central", (14.8, 7.1, 0.85), 0.82, 0.72, 0.79),
        ("LZ-5 Northwest", (6.2, 12.0, 0.85), 0.80, 0.80, 0.71),
    ]
    zones: list[LandingZone] = []
    for index, (name, position, radius, science, terrain) in enumerate(entries):
        zones.append(
            LandingZone(
                index=index,
                name=name,
                position=np.asarray(position, dtype=float),
                radius=float(radius),
                science_score=float(science),
                terrain_quality=float(terrain),
            )
        )
    return zones


def build_obstacles() -> list[Obstacle]:
    return [
        Obstacle("West tower", "box", {"min": (3.6, 2.5, 0.0), "max": (4.6, 4.0, 7.2)}, "tower"),
        Obstacle("North tower", "cylinder", {"center": (7.7, 9.3), "radius": 0.95, "z": (0.0, 6.8)}, "tower"),
        Obstacle("Gate south post", "box", {"min": (9.7, 2.4, 0.0), "max": (10.7, 3.8, 6.0)}, "gate"),
        Obstacle("Gate north post", "box", {"min": (9.7, 6.0, 0.0), "max": (10.7, 7.4, 6.0)}, "gate"),
        Obstacle("Gate beam", "box", {"min": (9.7, 2.4, 5.0), "max": (10.7, 7.4, 6.1)}, "gate"),
        Obstacle("Suspended slab", "box", {"min": (12.2, 7.0, 3.6), "max": (14.2, 9.3, 5.4)}, "aerial"),
        Obstacle(
            "Aerial boulder",
            "ellipsoid",
            {"center": (6.7, 11.3, 6.0), "radii": (1.35, 1.05, 1.0)},
            "aerial",
        ),
        Obstacle(
            "Central spire",
            "cylinder",
            {"center": (12.0, 4.8), "radius": 0.72, "z": (0.0, 7.8)},
            "tower",
        ),
        Obstacle(
            "Crater rim",
            "annular_cylinder",
            {"center": (14.9, 7.1), "r_inner": 0.95, "r_outer": 1.55, "z": (0.0, 1.15)},
            "terrain",
        ),
        Obstacle(
            "North terrain rock",
            "ellipsoid",
            {"center": (10.8, 11.0, 1.25), "radii": (1.15, 0.85, 1.2)},
            "terrain",
        ),
        Obstacle(
            "Low ridge",
            "box",
            {"min": (5.4, 6.2, 0.0), "max": (7.3, 7.0, 1.45)},
            "terrain",
        ),
    ]


def obstacle_mask(obstacle: Obstacle, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, inflation: float) -> np.ndarray:
    kind = obstacle.kind
    p = obstacle.params
    if kind == "box":
        lo = np.asarray(p["min"], dtype=float) - inflation
        hi = np.asarray(p["max"], dtype=float) + inflation
        return (
            (X >= lo[0])
            & (X <= hi[0])
            & (Y >= lo[1])
            & (Y <= hi[1])
            & (Z >= max(0.0, lo[2]))
            & (Z <= hi[2])
        )
    if kind == "cylinder":
        cx, cy = p["center"]
        radius = float(p["radius"]) + inflation
        z0, z1 = p["z"]
        return ((X - cx) ** 2 + (Y - cy) ** 2 <= radius**2) & (Z >= max(0.0, z0 - inflation)) & (
            Z <= z1 + inflation
        )
    if kind == "annular_cylinder":
        cx, cy = p["center"]
        r_in = max(0.0, float(p["r_inner"]) - 0.4 * inflation)
        r_out = float(p["r_outer"]) + inflation
        z0, z1 = p["z"]
        rr = (X - cx) ** 2 + (Y - cy) ** 2
        return (rr >= r_in**2) & (rr <= r_out**2) & (Z >= z0) & (Z <= z1 + inflation)
    if kind == "ellipsoid":
        center = np.asarray(p["center"], dtype=float)
        radii = np.asarray(p["radii"], dtype=float) + inflation
        q = ((X - center[0]) / radii[0]) ** 2 + ((Y - center[1]) / radii[1]) ** 2 + (
            (Z - center[2]) / radii[2]
        ) ** 2
        return q <= 1.0
    raise ValueError(f"Unsupported obstacle kind: {kind}")


def build_world(grid: GridSpec, num_landing_zones: int, seed: int) -> WorldData:
    if not 2 <= num_landing_zones <= len(landing_zone_templates()):
        raise ValueError(f"num_landing_zones must be between 2 and {len(landing_zone_templates())}.")
    rng = np.random.default_rng(seed)
    del rng  # deterministic analytic world; seed retained for future stochastic extensions.

    X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    occupancy = np.zeros(grid.shape, dtype=bool)
    obstacles = build_obstacles()
    inflation = 0.22
    for obstacle in obstacles:
        occupancy |= obstacle_mask(obstacle, X, Y, Z, inflation)

    # The outer computational walls are occupied. The ground is a valid landing
    # surface, but z=0 is still the Dirichlet boundary; landing zones sit above it.
    occupancy[0, :, :] = True
    occupancy[-1, :, :] = True
    occupancy[:, 0, :] = True
    occupancy[:, -1, :] = True
    occupancy[:, :, 0] = True
    occupancy[:, :, -1] = True

    free_mask = ~occupancy
    structure = ndimage.generate_binary_structure(3, 1)
    boundary_mask = free_mask & ndimage.binary_dilation(occupancy, structure=structure)
    unknown_mask = free_mask & ~boundary_mask

    zones = landing_zone_templates()[:num_landing_zones]
    start = np.array([1.0, 1.1, 8.6], dtype=float)

    def nearest_index(point: np.ndarray) -> tuple[int, int, int]:
        return (
            int(np.argmin(np.abs(grid.x - point[0]))),
            int(np.argmin(np.abs(grid.y - point[1]))),
            int(np.argmin(np.abs(grid.z - point[2]))),
        )

    if occupancy[nearest_index(start)]:
        raise RuntimeError("World construction error: start state lies in occupied space.")
    for zone in zones:
        if occupancy[nearest_index(zone.position)]:
            raise RuntimeError(f"World construction error: {zone.name} lies in occupied space.")

    diagnostics = {
        "grid_shape": grid.shape,
        "world_bounds_m": grid.bounds,
        "spacing_m": grid.spacing,
        "occupied_voxels": int(np.count_nonzero(occupancy)),
        "free_voxels": int(np.count_nonzero(free_mask)),
        "boundary_voxels": int(np.count_nonzero(boundary_mask)),
        "unknown_voxels": int(np.count_nonzero(unknown_mask)),
        "occupancy_fraction": float(np.mean(occupancy)),
        "inflation_m": inflation,
    }
    return WorldData(
        grid=grid,
        occupancy=occupancy,
        free_mask=free_mask,
        boundary_mask=boundary_mask,
        unknown_mask=unknown_mask,
        obstacles=obstacles,
        landing_zones=zones,
        start=start,
        diagnostics=diagnostics,
    )


# -----------------------------------------------------------------------------
# Poisson PDE construction and solvers
# -----------------------------------------------------------------------------


def solve_poisson(
    world: WorldData,
    forcing_method: str,
    solver: str,
    tolerance: float,
    max_iterations: int,
    guidance_alpha: float = 0.5,
) -> PoissonResult:
    """Run the real :mod:`poisson_safety_box` and normalize its output.

    The only computations performed after ``PoissonSafetyBox.compute`` are a
    scalar normalization and the corresponding linear scaling of derivatives.
    These operations preserve the field geometry and avoid duplicating any PDE
    assembly, forcing, solver, gradient, or Hessian implementation.
    """

    # Configure the uploaded Poisson box using the physical grid spacing and the
    # forcing/solver selected by the experiment command line.
    box_config = PoissonBoxConfig(
        grid_spacing=world.grid.spacing,
        forcing_method=forcing_method,
        solver=solver,
        compute_gradient=True,
        compute_hessian=True,
        compute_laplacian_check=True,
        plot=False,
        save_outputs=False,
    )
    box_config.conjugate_gradient.tolerance = float(tolerance)
    box_config.conjugate_gradient.max_iter = int(max_iterations)
    box_config.sor.tolerance = max(float(tolerance), 1.0e-7)
    box_config.sor.max_iter = int(max_iterations)
    box_config.guidance.target_mean_abs_scale = float(guidance_alpha)

    # This call is the mandatory occupancy -> h, grad_h, Hessian_h pipeline.
    box_result = PoissonSafetyBox(box_config).compute(world.occupancy)
    if box_result.grad_h is None or box_result.hessian_h is None:
        raise RuntimeError("PoissonSafetyBox did not return the required derivatives.")

    # Normalize by a robust interior percentile so CBF gains remain comparable
    # across forcing methods.  Because Poisson's equation is linear, h, grad h,
    # and Hessian h are all divided by the same scalar.
    interior_values = np.asarray(box_result.h[box_result.solve_mask], dtype=float)
    normalization_scale = max(float(np.percentile(interior_values, 99.0)), 1.0e-10)
    h_normalized = np.maximum(np.asarray(box_result.h, dtype=float), 0.0) / normalization_scale
    h_normalized[~box_result.free_mask] = 0.0
    grad_normalized = np.asarray(box_result.grad_h, dtype=float) / normalization_scale
    hessian_normalized = np.asarray(box_result.hessian_h, dtype=float) / normalization_scale

    # A small zero-level offset represents finite numerical/vehicle clearance.
    # The field derivatives are unchanged because the offset is constant.
    safety_level = 0.0030
    h_cbf = h_normalized - safety_level

    residual_data = box_result.solver_info.get("residual", {})
    residual_l2 = float(residual_data.get("l2") or 0.0)
    forcing_rms = float(np.linalg.norm(box_result.forcing[box_result.solve_mask]) / max(1, np.count_nonzero(box_result.solve_mask)) ** 0.5)
    relative_residual = residual_l2 / max(forcing_rms, 1.0e-12)

    return PoissonResult(
        forcing_method=forcing_method,
        solver=solver,
        box_result=box_result,
        h_raw=np.asarray(box_result.h, dtype=float),
        h=h_normalized,
        h_cbf=h_cbf,
        grad_h=grad_normalized,
        hessian_h=hessian_normalized,
        solve_time_s=float(box_result.timing.get("solve", box_result.solver_info.get("elapsed_time", 0.0))),
        relative_residual=float(relative_residual),
        solver_info=dict(box_result.solver_info),
        iterations=int(box_result.solver_info.get("iterations", 0)),
        normalization_scale=normalization_scale,
    )


# -----------------------------------------------------------------------------
# Obstacle-aware geodesic fields and reachability-proxy certificates
# -----------------------------------------------------------------------------


def build_navigation_graph(world: WorldData, poisson: PoissonResult, cruise_altitude: float) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    grid = world.grid
    k = int(np.argmin(np.abs(grid.z - cruise_altitude)))
    z_indices = np.where((grid.z >= 0.65) & (grid.z <= max(cruise_altitude + 1.0, 5.0)))[0]
    blocked = np.any(world.occupancy[:, :, z_indices], axis=2)
    # Open true landing cells even when the low terrain projection is conservative.
    for zone in world.landing_zones:
        ix = int(np.argmin(np.abs(grid.x - zone.position[0])))
        iy = int(np.argmin(np.abs(grid.y - zone.position[1])))
        blocked[max(0, ix - 1) : min(grid.nx, ix + 2), max(0, iy - 1) : min(grid.ny, iy + 2)] = False
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    h_slice = np.maximum(poisson.h[:, :, k], 0.0)
    risk = 1.0 + 0.65 / np.maximum(h_slice + 0.08, 0.08)
    risk = np.clip(risk, 1.0, 7.0)
    risk[blocked] = np.inf

    dx, dy, _ = grid.spacing
    directions = [
        (1, 0, dx),
        (-1, 0, dx),
        (0, 1, dy),
        (0, -1, dy),
        (1, 1, math.hypot(dx, dy)),
        (1, -1, math.hypot(dx, dy)),
        (-1, 1, math.hypot(dx, dy)),
        (-1, -1, math.hypot(dx, dy)),
    ]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(grid.nx):
        for j in range(grid.ny):
            if blocked[i, j]:
                continue
            node = i * grid.ny + j
            for di, dj, step in directions:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= grid.nx or nj < 0 or nj >= grid.ny or blocked[ni, nj]:
                    continue
                neighbor = ni * grid.ny + nj
                weight = step * 0.5 * (risk[i, j] + risk[ni, nj])
                rows.append(node)
                cols.append(neighbor)
                data.append(float(weight))
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(grid.nx * grid.ny, grid.nx * grid.ny))
    return graph, blocked, risk


def nearest_unblocked_source(zone: LandingZone, blocked: np.ndarray, grid: GridSpec) -> int:
    X2, Y2 = np.meshgrid(grid.x, grid.y, indexing="ij")
    metric = (X2 - zone.position[0]) ** 2 + (Y2 - zone.position[1]) ** 2
    metric = np.where(blocked, np.inf, metric)
    index = int(np.argmin(metric))
    if not np.isfinite(metric.ravel()[index]):
        raise RuntimeError(f"No traversable 2-D source cell found for {zone.name}.")
    return index


def compute_geodesic_fields(world: WorldData, poisson: PoissonResult, cruise_altitude: float) -> list[GeodesicField]:
    graph, blocked, _ = build_navigation_graph(world, poisson, cruise_altitude)
    fields: list[GeodesicField] = []
    for zone in world.landing_zones:
        source = nearest_unblocked_source(zone, blocked, world.grid)
        distances = dijkstra(graph, directed=False, indices=source).reshape(world.grid.nx, world.grid.ny)
        finite = np.isfinite(distances)
        fill_value = float(np.max(distances[finite]) * 1.35) if np.any(finite) else 10.0 * world.grid.world_diagonal_xy
        distances = np.where(finite, distances, fill_value)
        distances = ndimage.gaussian_filter(distances, sigma=0.85, mode="nearest")
        gx, gy = np.gradient(distances, world.grid.spacing[0], world.grid.spacing[1], edge_order=1)
        fields.append(GeodesicField(distance=distances, grad_x=gx, grad_y=gy))
    return fields


def build_lqr_lyapunov_matrix(Kp: float, Kd: float) -> tuple[np.ndarray, float]:
    I = np.eye(3)
    Z = np.zeros((3, 3))
    A = np.block([[Z, I], [Z, Z]])
    B = np.block([[Z], [I]])
    K = np.block([Kp * I, Kd * I])
    Acl = A - B @ K
    Q = np.diag([1.0, 1.0, 1.0, 0.45, 0.45, 0.45])
    P = solve_continuous_lyapunov(Acl.T, -Q)
    if np.min(np.linalg.eigvalsh(P)) <= 0.0:
        raise RuntimeError("Failed to construct a positive-definite Lyapunov matrix.")
    decay_rate = float(np.min(np.linalg.eigvalsh(Q)) / np.max(np.linalg.eigvalsh(P)))
    return P, decay_rate


def build_reachability_model(
    world: WorldData,
    poisson: PoissonResult,
    config: SimulationConfig,
) -> ReachabilityModel:
    geodesics = compute_geodesic_fields(world, poisson, config.approach_altitude)
    P, _ = build_lqr_lyapunov_matrix(config.Kp, config.Kd)
    p0 = world.start
    v0 = np.zeros(3)

    quad_scales = []
    preliminary = []
    for zone, geodesic in zip(world.landing_zones, geodesics):
        xerr = np.concatenate([p0 - zone.position, v0])
        q0 = float(xerr @ P @ xerr)
        scale = max(q0 / 0.78, 1e-6)
        quad_scales.append(scale)
        d0 = sample_bilinear(geodesic.distance, p0[:2], world.grid)
        preliminary.append(0.78 + 0.14 * (d0 / world.grid.world_diagonal_xy) ** 2)

    base_margin_library = np.array([0.62, 0.46, 0.53, 0.34, 0.42, 0.38], dtype=float)
    base_margins = base_margin_library[: len(world.landing_zones)].copy()
    initial_values = np.asarray(preliminary, dtype=float)
    c0 = initial_values + base_margins
    initial_margins = c0 - initial_values

    return ReachabilityModel(
        zones=world.landing_zones,
        geodesics=geodesics,
        P=P,
        quad_scales=np.asarray(quad_scales),
        c0=c0,
        base_margins=base_margins,
        beta_time=0.0060,
        beta_energy=0.0012,
        block_drop=2.6,
        block_transition_s=1.25,
        w_geodesic=0.14,
        w_poisson_risk=0.34,
        w_velocity=0.85,
        risk_trigger=0.095,
        risk_temperature=0.025,
        world_diagonal_xy=world.grid.world_diagonal_xy,
        max_speed=config.max_speed,
        initial_values=initial_values,
        initial_margins=initial_margins,
    )


def capacity_and_derivative(
    model: ReachabilityModel,
    zone_index: int,
    t: float,
    energy_used: float,
    previous_power: float,
    blocked_zone: int,
    failure_time: float,
) -> tuple[float, float, float]:
    block_value = 0.0
    block_derivative = 0.0
    if zone_index == blocked_zone and failure_time >= 0.0:
        s, ds = smoothstep_and_derivative(t, failure_time, model.block_transition_s)
        block_value = model.block_drop * s
        block_derivative = model.block_drop * ds
    capacity = model.c0[zone_index] - model.beta_time * t - model.beta_energy * energy_used - block_value
    derivative = -model.beta_time - model.beta_energy * previous_power - block_derivative
    return float(capacity), float(derivative), float(block_value)


def evaluate_reachability_certificate(
    world: WorldData,
    poisson: PoissonResult,
    model: ReachabilityModel,
    zone_index: int,
    position: np.ndarray,
    velocity: np.ndarray,
    t: float,
    energy_used: float,
    previous_power: float,
    blocked_zone: int,
    failure_time: float,
) -> dict[str, Any]:
    zone = model.zones[zone_index]
    e = position - zone.position
    xerr = np.concatenate([e, velocity])
    P = model.P
    scale = model.quad_scales[zone_index]
    quad = float(xerr @ P @ xerr / scale)
    grad_p_quad = 2.0 * (P[:3, :3] @ e + P[:3, 3:] @ velocity) / scale
    grad_v_quad = 2.0 * (P[3:, :3] @ e + P[3:, 3:] @ velocity) / scale

    geodesic = model.geodesics[zone_index]
    distance = sample_bilinear(geodesic.distance, position[:2], world.grid)
    grad_distance = np.array(
        [
            sample_bilinear(geodesic.grad_x, position[:2], world.grid),
            sample_bilinear(geodesic.grad_y, position[:2], world.grid),
            0.0,
        ]
    )
    geo = model.w_geodesic * (distance / model.world_diagonal_xy) ** 2
    grad_geo = (
        2.0 * model.w_geodesic * distance / (model.world_diagonal_xy**2) * grad_distance
    )

    h_value = float(sample_trilinear(poisson.h, position, world.grid))
    h_grad = np.asarray(sample_trilinear(poisson.grad, position, world.grid), dtype=float)
    soft, dsoft_dx = stable_softplus(model.risk_trigger - h_value, model.risk_temperature)
    risk = model.w_poisson_risk * soft**2
    # x = risk_trigger - h, so drisk/dp = -2*w*soft*dsoft/dx*grad(h).
    grad_risk = -2.0 * model.w_poisson_risk * soft * dsoft_dx * h_grad

    # A normalized kinetic term makes the proxy a practical relative-degree-one
    # certificate for the double-integrator model.  It represents stopping effort:
    # high speed consumes reachability budget and gives acceleration direct authority
    # over dot(rho_i), unlike a purely positional geodesic score.
    velocity_cost = model.w_velocity * float(velocity @ velocity) / (model.max_speed**2)
    grad_velocity = 2.0 * model.w_velocity * velocity / (model.max_speed**2)

    value = quad + geo + risk + velocity_cost
    grad_p = grad_p_quad + grad_geo + grad_risk
    grad_v = grad_v_quad + grad_velocity
    capacity, c_dot, block_value = capacity_and_derivative(
        model,
        zone_index,
        t,
        energy_used,
        previous_power,
        blocked_zone,
        failure_time,
    )
    rho_raw = capacity - value
    available = not (zone_index == blocked_zone and failure_time >= 0.0 and t >= failure_time)
    # A newly confirmed hazard removes that landing site from the candidate family.
    # Its displayed margin is made negative, while the combinatorial filter continues
    # to require r certificates among the still-valid alternatives.
    rho = float(rho_raw if available else min(rho_raw, -0.75))
    return {
        "value": float(value),
        "quad": float(quad),
        "geodesic": float(geo),
        "poisson_risk": float(risk),
        "velocity_cost": float(velocity_cost),
        "distance": float(distance),
        "grad_p": grad_p,
        "grad_v": grad_v,
        "capacity": float(capacity),
        "capacity_dot": float(c_dot),
        "block_value": float(block_value),
        "rho_raw": float(rho_raw),
        "rho": float(rho),
        "available": bool(available),
    }


def evaluate_all_certificates(
    world: WorldData,
    poisson: PoissonResult,
    model: ReachabilityModel,
    position: np.ndarray,
    velocity: np.ndarray,
    t: float,
    energy_used: float,
    previous_power: float,
    blocked_zone: int,
    failure_time: float,
) -> list[dict[str, Any]]:
    return [
        evaluate_reachability_certificate(
            world,
            poisson,
            model,
            index,
            position,
            velocity,
            t,
            energy_used,
            previous_power,
            blocked_zone,
            failure_time,
        )
        for index in range(len(model.zones))
    ]


# -----------------------------------------------------------------------------
# Small convex QP solver and safety/contingency filter
# -----------------------------------------------------------------------------


def task_clf_terms(
    position: np.ndarray,
    velocity: np.ndarray,
    reference: np.ndarray,
    P: np.ndarray,
    scale: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    error = np.concatenate([position - reference, velocity])
    value = float(error @ P @ error / scale)
    grad_p = 2.0 * (P[:3, :3] @ (position - reference) + P[:3, 3:] @ velocity) / scale
    grad_v = 2.0 * (P[3:, :3] @ (position - reference) + P[3:, 3:] @ velocity) / scale
    return value, grad_p, grad_v


def filter_acceleration(
    world: WorldData,
    poisson: PoissonResult,
    model: ReachabilityModel,
    config: SimulationConfig,
    position: np.ndarray,
    velocity: np.ndarray,
    reference: np.ndarray,
    active_zone: int,
    a_nom: np.ndarray,
    certificates: list[dict[str, Any]],
) -> tuple[QPSolution, dict[str, Any]]:
    """Build all safety/stability rows and solve them through ``CBFBox``.

    The runtime supplies state, references, and certificate values.  The Poisson
    adapter, HOCBF builder, combinatorial builders, and QP solver all belong to
    the two mandatory safety boxes.
    """

    # Convert the gridded Poisson-box output into the standard SafetySample used
    # by cbf_safety_box.  This adapter performs nearest-grid sampling and keeps
    # the package boundary explicit and testable.
    safety = sample_from_arrays(
        h_grid=poisson.h_cbf,
        grad_h_grid=poisson.grad_h,
        hessian_h_grid=poisson.hessian_h,
        laplacian_h_grid=poisson.box_result.laplacian_h,
        position=position,
        grid_spacing=world.grid.spacing,
    )
    state = SystemState(position=np.asarray(position, dtype=float), velocity=np.asarray(velocity, dtype=float))

    # The existing CBF-box HOCBF builder implements the relative-degree-two row
    # for p_dot=v, v_dot=a.  It is lifted with a zero omega coefficient because
    # environmental collision avoidance is not relaxed by the contingency term.
    environment_constraint = build_acceleration_hocbf_constraint(
        safety,
        state.velocity,
        config.hocbf_alpha,
        config.hocbf_alpha,
    )
    environment_constraint = lift_constraint_with_auxiliary(environment_constraint, 0.0)

    rho_values = np.asarray([item["rho"] for item in certificates], dtype=float)

    # Build the active-target CLF row inside cbf_safety_box.  The auxiliary
    # variable may relax this liveness objective only after its own reachability
    # certificate becomes negative; hard safety rows remain independent of it.
    initial_error = np.concatenate([world.start - reference, np.zeros(3)])
    task_scale = max(float(initial_error @ model.P @ initial_error), 1.0)
    V_task, grad_p_task, grad_v_task = task_clf_terms(position, velocity, reference, model.P, task_scale)
    task_constraint = build_active_target_clf_constraint(
        value=V_task,
        drift=float(grad_p_task @ velocity),
        control_gradient=grad_v_task,
        gamma=config.task_gamma,
        relaxation_coefficient=max(-float(rho_values[active_zone]), 0.0),
    )

    # Translate each landing-zone reachability proxy into its local affine
    # derivative model rho_dot = drift + control_gradient^T a.
    affine_certificates: list[AffineCertificate] = []
    for index, item in enumerate(certificates):
        affine_certificates.append(
            AffineCertificate(
                name=str(index),
                value=float(item["rho"]),
                drift=float(item["capacity_dot"] - np.asarray(item["grad_p"], dtype=float) @ velocity),
                control_gradient=-np.asarray(item["grad_v"], dtype=float),
                available=bool(item.get("available", True)),
            )
        )

    contingency_constraints, pivot = build_combinatorial_contingency_constraints(
        affine_certificates,
        r=config.r_contingency,
        gamma=config.contingency_gamma,
        auxiliary_gain=config.rho_aux_gain,
    )

    # Configure the CBF box for a multi-constraint acceleration QP.  The augmented
    # decision z=[a_x,a_y,a_z,omega] has a weighted projection objective and an
    # exact Euclidean acceleration limit ||a||_2 <= max_acc.
    cbf_config = CBFBoxConfig(
        mode="acceleration",
        solver="scipy",
        alpha1=config.hocbf_alpha,
        alpha2=config.hocbf_alpha,
        use_slack=False,
        slack_weight=1.0e6,
    )
    cbf_box = CBFBox(cbf_config)
    decision_nominal = np.concatenate([np.asarray(a_nom, dtype=float), [0.0]])
    lower_bounds = np.array([-np.inf, -np.inf, -np.inf, 0.0], dtype=float)
    upper_bounds = np.array([np.inf, np.inf, np.inf, config.omega_max], dtype=float)
    quadratic_weights = np.array([1.0, 1.0, 1.0, 2.0 * config.omega_cost], dtype=float)
    all_constraints = [environment_constraint, task_constraint, *contingency_constraints]

    result = cbf_box.filter_affine_constraints(
        decision_nominal=decision_nominal,
        constraints=all_constraints,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        quadratic_weights=quadratic_weights,
        norm_bound_indices=[[0, 1, 2]],
        norm_bound_values=[config.max_acc],
        use_slack=False,
    )

    feasibility = result.diagnostics["feasibility"]
    task_relaxed = False

    # If bounded control conflicts with the liveness CLF, retry without that one
    # soft objective.  Environment and contingency constraints remain untouched.
    if not bool(feasibility["feasible"]):
        result_without_task = cbf_box.filter_affine_constraints(
            decision_nominal=decision_nominal,
            constraints=[environment_constraint, *contingency_constraints],
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            quadratic_weights=quadratic_weights,
            norm_bound_indices=[[0, 1, 2]],
            norm_bound_values=[config.max_acc],
            use_slack=False,
        )
        if bool(result_without_task.diagnostics["feasibility"]["feasible"]):
            result = result_without_task
            feasibility = result.diagnostics["feasibility"]
            task_relaxed = True

    # A final emergency-slack solve is diagnostic rather than a formal guarantee.
    # Its magnitude is logged so the study never hides local infeasibility.
    if not bool(feasibility["feasible"]):
        emergency_result = cbf_box.filter_affine_constraints(
            decision_nominal=decision_nominal,
            constraints=[environment_constraint, *contingency_constraints],
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            quadratic_weights=quadratic_weights,
            norm_bound_indices=[[0, 1, 2]],
            norm_bound_values=[config.max_acc],
            use_slack=True,
        )
        result = emergency_result
        feasibility = result.diagnostics["feasibility"]
        task_relaxed = True

    decision = np.asarray(result.u_safe, dtype=float)
    acceleration = decision[:3]
    omega = float(np.clip(decision[3], 0.0, config.omega_max))
    emergency_slack = float(result.diagnostics.get("emergency_slack", 0.0))

    # Compute named residuals directly from the exact rows returned by the box.
    labels = list(result.diagnostics["constraints"])
    raw_residuals = np.asarray(result.constraint_matrix @ decision - result.constraint_vector, dtype=float)
    residual_map = {label: float(value) for label, value in zip(labels, raw_residuals)}
    hocbf_residual = float(environment_constraint.residual(decision)[0])
    task_residual = float(task_constraint.residual(decision)[0])
    contingency_residuals = np.array(
        [float(constraint.residual(decision)[0]) for constraint in contingency_constraints],
        dtype=float,
    )

    solution = QPSolution(
        acceleration=acceleration,
        omega=omega,
        success=bool(feasibility["feasible"]),
        min_residual=float(np.min(raw_residuals)) if raw_residuals.size else math.inf,
        emergency_slack=emergency_slack,
        task_relaxed=task_relaxed,
        iterations=int(result.diagnostics.get("iterations", 0)),
        residuals=residual_map,
    )
    diagnostics = {
        "h": float(safety.h),
        "grad_h": np.asarray(safety.grad_h, dtype=float),
        "hessian_h": np.asarray(safety.hessian_h, dtype=float),
        "hocbf_rhs": float(environment_constraint.b[0]),
        "hocbf_residual": hocbf_residual,
        "task_V": V_task,
        "task_residual": task_residual,
        "contingency_pivot": pivot,
        "contingency_residuals": contingency_residuals,
        "min_contingency_residual": float(np.min(contingency_residuals)) if contingency_residuals.size else math.inf,
        "cbf_box_solver_status": result.solver_status,
        "cbf_box_solve_time": result.solve_time,
        "cbf_box_feasibility": feasibility,
    }
    return solution, diagnostics


# -----------------------------------------------------------------------------
# Mission planner and simulation
# -----------------------------------------------------------------------------


def select_backup_zone(
    model: ReachabilityModel,
    certificates: list[dict[str, Any]],
    active_zone: int,
    blocked_zone: int,
    t: float,
    failure_time: float,
) -> int:
    scores = np.full(len(model.zones), -np.inf, dtype=float)
    for index, (zone, item) in enumerate(zip(model.zones, certificates)):
        blocked = index == blocked_zone and t >= failure_time >= 0.0
        if blocked or index == active_zone:
            continue
        rho = float(item["rho"])
        # Slightly negative candidates are retained as a recovery fallback, but the
        # score strongly favors currently certified zones.
        feasibility_bonus = 2.25 * rho
        science_bonus = 0.34 * zone.science_score + 0.16 * zone.terrain_quality
        travel_penalty = 0.028 * float(item["distance"])
        scores[index] = feasibility_bonus + science_bonus - travel_penalty
    if np.all(~np.isfinite(scores)):
        return active_zone
    return int(np.nanargmax(scores))


def route_waypoints_xy(world: WorldData, zone_index: int) -> np.ndarray:
    """Return a deterministic obstacle-aware mission corridor for one landing zone.

    The reachability certificates still use Poisson-weighted graph geodesics.  This
    higher-level reference governor intentionally uses a small, auditable waypoint
    library so that the nominal PD controller does not chatter on finite-difference
    geodesic gradients near grid saddles.  The corridors are part of the offline
    mission-planner prototype, not a claim of globally optimal planning.
    """
    zone = world.landing_zones[zone_index]
    start_xy = world.start[:2]

    # A northern corridor passes above the gate, north tower and suspended slab.
    north = np.array(
        [
            start_xy,
            [2.8, 5.0],
            [5.1, 7.65],
            [8.75, 7.75],
            [11.15, 10.10],
            [14.85, 11.25],
        ],
        dtype=float,
    )
    # A southern corridor passes below the gate and central spire.
    south = np.array(
        [
            start_xy,
            [2.8, 5.0],
            [5.4, 5.05],
            [8.55, 4.65],
            [9.15, 1.65],
            [11.25, 1.55],
            [14.25, 2.05],
        ],
        dtype=float,
    )

    if zone_index in (0, 2):
        corridor = north
    elif zone_index in (1, 3):
        corridor = south
    elif zone_index == 4:
        corridor = np.vstack([north[:-1], np.array([[13.9, 10.4], [15.0, 9.7]])])
    else:  # Northwest / any additional template near the north side.
        corridor = np.array(
            [
                start_xy,
                [2.8, 5.0],
                [5.1, 7.65],
                [5.25, 9.75],
                [6.2, 11.15],
            ],
            dtype=float,
        )

    if np.linalg.norm(corridor[-1] - zone.position[:2]) > 1e-9:
        corridor = np.vstack([corridor, zone.position[:2]])
    return corridor


def polyline_lookahead_xy(polyline: np.ndarray, position_xy: np.ndarray, lookahead: float) -> np.ndarray:
    """Project onto a polyline and return a point `lookahead` metres downstream."""
    if len(polyline) < 2:
        return polyline[-1].copy()
    segment_vectors = np.diff(polyline, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])

    best_distance = math.inf
    best_arclength = 0.0
    for index, (a, direction, length) in enumerate(zip(polyline[:-1], segment_vectors, segment_lengths)):
        if length < 1e-12:
            continue
        fraction = float(np.clip(np.dot(position_xy - a, direction) / (length * length), 0.0, 1.0))
        projection = a + fraction * direction
        distance = float(np.linalg.norm(position_xy - projection))
        # A tiny progress preference prevents switching backward at self-near portions.
        candidate_arclength = cumulative[index] + fraction * length
        score = distance - 1e-5 * candidate_arclength
        if score < best_distance:
            best_distance = score
            best_arclength = candidate_arclength

    target_arclength = min(best_arclength + max(lookahead, 0.0), cumulative[-1])
    segment_index = int(np.searchsorted(cumulative, target_arclength, side="right") - 1)
    segment_index = int(np.clip(segment_index, 0, len(segment_lengths) - 1))
    length = segment_lengths[segment_index]
    if length < 1e-12:
        return polyline[segment_index + 1].copy()
    fraction = (target_arclength - cumulative[segment_index]) / length
    return polyline[segment_index] + fraction * segment_vectors[segment_index]


def mission_reference(
    world: WorldData,
    model: ReachabilityModel,
    zone_index: int,
    position: np.ndarray,
    velocity: np.ndarray,
    t: float,
    last_switch_time: float,
    config: SimulationConfig,
) -> np.ndarray:
    """Return an obstacle-aware look-ahead reference toward the active landing zone.

    The low-level nominal law remains exactly
    ``a_nom = Kp * (reference - p) - Kd * v``.  Only the reference is governed.
    The reference follows a transparent mission corridor at a conservative cruise
    altitude, then transitions to the selected touchdown point after the failure and
    switch dwell gates have elapsed.
    """
    del model  # The certificates use geodesics; the mission reference uses an auditable corridor.
    zone = world.landing_zones[zone_index]
    horizontal_distance = float(np.linalg.norm(zone.position[:2] - position[:2]))
    route = route_waypoints_xy(world, zone_index)

    # Short look-ahead keeps nominal cruise speed below the 2 m/s hard limit and
    # materially improves bounded-input HOCBF feasibility near obstacle curvature.
    lookahead = float(np.clip(0.32 * horizontal_distance, 1.05, 1.65))
    waypoint_xy = polyline_lookahead_xy(route, position[:2], lookahead)
    if horizontal_distance <= 1.65:
        waypoint_xy = zone.position[:2].copy()

    # Convert the path direction into a bounded desired horizontal velocity, then
    # express that velocity through the PD position reference.  Algebraically,
    # p_ref-p=(Kd/Kp)v_des gives a_nom_xy=Kd(v_des-v), while retaining the exact
    # nominal-controller form requested for the study.  The speed profile tapers
    # near the landing zone to preserve HOCBF braking authority.
    path_direction = unit_vector(waypoint_xy - position[:2])
    desired_horizontal_speed = min(0.95, 0.46 * horizontal_distance)
    desired_horizontal_velocity = desired_horizontal_speed * path_direction
    waypoint_xy = position[:2] + (config.Kd / max(config.Kp, 1e-9)) * desired_horizontal_velocity

    after_failure_gate = t >= config.failure_time + config.descent_delay_after_failure
    after_switch_gate = t >= last_switch_time + config.descent_delay_after_switch
    horizontal_speed = float(np.linalg.norm(velocity[:2]))
    descend = (
        after_failure_gate
        and after_switch_gate
        and horizontal_distance <= config.descent_horizontal_radius
        and horizontal_speed <= 0.55
    )
    if descend:
        # Altitude reference governor: descend with a bounded look-ahead instead of
        # commanding the ground-level target in one jump.  This preserves braking
        # authority for the acceleration-level Poisson HOCBF near the ground boundary.
        z_ref = max(float(zone.position[2]), float(position[2] - 0.72))
    else:
        z_ref = config.approach_altitude
    return np.array([waypoint_xy[0], waypoint_xy[1], z_ref], dtype=float)

def occupancy_at_point(world: WorldData, point: np.ndarray) -> bool:
    if np.any(point < 0.0) or np.any(point > np.asarray(world.grid.bounds)):
        return True
    ix = int(np.argmin(np.abs(world.grid.x - point[0])))
    iy = int(np.argmin(np.abs(world.grid.y - point[1])))
    iz = int(np.argmin(np.abs(world.grid.z - point[2])))
    return bool(world.occupancy[ix, iy, iz])


def simulate(
    world: WorldData,
    poisson: PoissonResult,
    model: ReachabilityModel,
    config: SimulationConfig,
    require_failure_switch: bool = True,
) -> SimulationResult:
    n_zones = len(model.zones)
    p = world.start.copy()
    v = np.zeros(3, dtype=float)
    active = int(config.active_zone)
    last_switch_time = 0.0
    energy = 0.0
    previous_acceleration = np.zeros(3)
    switch_events: list[dict[str, Any]] = []
    blocked_events: list[dict[str, Any]] = []
    failure_recorded = False

    records: dict[str, list[Any]] = {
        "time": [],
        "position": [],
        "velocity": [],
        "a_nom": [],
        "a_safe": [],
        "reference": [],
        "active": [],
        "h": [],
        "hocbf_residual": [],
        "task_residual": [],
        "cont_residual": [],
        "rho": [],
        "cont_margin": [],
        "reachable_count": [],
        "omega": [],
        "correction": [],
        "energy": [],
        "emergency_slack": [],
        "task_relaxed": [],
    }

    collided = False
    contingency_lost = False
    landed = False
    terminated_reason = "maximum_steps"

    for step in range(config.max_steps):
        t = step * config.dt
        previous_power = float(previous_acceleration @ previous_acceleration)
        certificates = evaluate_all_certificates(
            world,
            poisson,
            model,
            p,
            v,
            t,
            energy,
            previous_power,
            config.blocked_zone,
            config.failure_time,
        )
        rho_values = np.asarray([item["rho"] for item in certificates])
        pivot, reachable_count = certificate_pivot_and_count(certificates, config.r_contingency)

        if (
            config.failure_time >= 0.0
            and t >= config.failure_time
            and not failure_recorded
            and 0 <= config.blocked_zone < n_zones
        ):
            blocked_events.append(
                {
                    "time": t,
                    "zone": config.blocked_zone,
                    "name": model.zones[config.blocked_zone].name,
                    "reason": "newly detected landing hazard",
                }
            )
            failure_recorded = True

        active_blocked = active == config.blocked_zone and config.failure_time >= 0.0 and t >= config.failure_time
        active_losing = rho_values[active] < config.switch_margin
        can_switch = (t - last_switch_time) >= config.min_switch_dwell_s
        if can_switch and (active_blocked or active_losing):
            old_active = active
            new_active = select_backup_zone(
                model,
                certificates,
                active_zone=active,
                blocked_zone=config.blocked_zone,
                t=t,
                failure_time=config.failure_time,
            )
            if new_active != old_active:
                active = new_active
                last_switch_time = t
                switch_events.append(
                    {
                        "time": t,
                        "from_zone": old_active,
                        "to_zone": new_active,
                        "reason": "blocked" if active_blocked else "reachability_margin",
                        "rho_before": rho_values.tolist(),
                    }
                )

        reference = mission_reference(world, model, active, p, v, t, last_switch_time, config)
        a_nom = config.Kp * (reference - p) - config.Kd * v
        a_nom = clip_norm(a_nom, config.max_acc)
        qp_solution, filter_diag = filter_acceleration(
            world,
            poisson,
            model,
            config,
            p,
            v,
            reference,
            active,
            a_nom,
            certificates,
        )
        a_safe = qp_solution.acceleration

        records["time"].append(t)
        records["position"].append(p.copy())
        records["velocity"].append(v.copy())
        records["a_nom"].append(a_nom.copy())
        records["a_safe"].append(a_safe.copy())
        records["reference"].append(reference.copy())
        records["active"].append(active)
        records["h"].append(filter_diag["h"])
        records["hocbf_residual"].append(filter_diag["hocbf_residual"])
        records["task_residual"].append(filter_diag["task_residual"])
        records["cont_residual"].append(filter_diag["min_contingency_residual"])
        records["rho"].append(rho_values.copy())
        records["cont_margin"].append(pivot)
        records["reachable_count"].append(reachable_count)
        records["omega"].append(qp_solution.omega)
        records["correction"].append(float(np.linalg.norm(a_safe - a_nom)))
        records["energy"].append(energy)
        records["emergency_slack"].append(qp_solution.emergency_slack)
        records["task_relaxed"].append(qp_solution.task_relaxed)

        if occupancy_at_point(world, p):
            collided = True
            terminated_reason = "collision"
            break
        if pivot < -0.035:
            contingency_lost = True
            # Continue briefly only if numerical recovery is plausible; a material loss
            # is a validation failure and should terminate the rollout.
            terminated_reason = "contingency_lost"
            break

        target = model.zones[active].position
        if (
            t >= config.failure_time + config.descent_delay_after_failure
            and np.linalg.norm(p - target) <= config.landing_position_tolerance
            and np.linalg.norm(v) <= config.landing_speed_tolerance
        ):
            landed = True
            terminated_reason = "landed"
            break

        p_next = p + config.dt * v
        v_next = clip_norm(v + config.dt * a_safe, config.max_speed)
        energy += config.dt * float(a_safe @ a_safe)
        previous_acceleration = a_safe.copy()
        p, v = p_next, v_next

    arrays = {key: np.asarray(value) for key, value in records.items()}
    if len(arrays["time"]) == 0:
        raise RuntimeError("Simulation generated no samples.")

    final_zone = int(arrays["active"][-1])
    final_target = model.zones[final_zone].position
    switched_due_to_block = any(event["reason"] == "blocked" for event in switch_events)
    maintained_contingency = bool(np.min(arrays["cont_margin"]) >= -0.035)
    max_emergency_slack = float(np.max(arrays["emergency_slack"]))
    metrics = {
        "duration_s": float(arrays["time"][-1]),
        "steps": int(len(arrays["time"])),
        "landed": bool(landed),
        "collided": bool(collided),
        "contingency_lost": bool(contingency_lost),
        "maintained_r_out_of_p": maintained_contingency,
        "minimum_h_cbf": float(np.min(arrays["h"])),
        "minimum_hocbf_residual": float(np.min(arrays["hocbf_residual"])),
        "minimum_contingency_margin": float(np.min(arrays["cont_margin"])),
        "minimum_contingency_constraint_residual": float(np.min(arrays["cont_residual"])),
        "minimum_reachable_count": int(np.min(arrays["reachable_count"])),
        "final_reachable_count": int(arrays["reachable_count"][-1]),
        "final_distance_to_active_zone_m": float(np.linalg.norm(arrays["position"][-1] - final_target)),
        "final_speed_mps": float(np.linalg.norm(arrays["velocity"][-1])),
        "final_zone": final_zone,
        "final_zone_name": model.zones[final_zone].name,
        "num_switches": len(switch_events),
        "blocked_switch_occurred": switched_due_to_block,
        "mean_control_correction": float(np.mean(arrays["correction"])),
        "max_control_correction": float(np.max(arrays["correction"])),
        "integrated_control_correction": float(np.sum(arrays["correction"]) * config.dt),
        "energy_proxy": float(arrays["energy"][-1]),
        "max_omega": float(np.max(arrays["omega"])),
        "max_emergency_slack": max_emergency_slack,
        "fraction_task_clf_relaxed": float(np.mean(arrays["task_relaxed"].astype(float))),
        "terminated_reason": terminated_reason,
        "failure_switch_requirement_met": bool((not require_failure_switch) or switched_due_to_block),
    }

    return SimulationResult(
        forcing_method=poisson.forcing_method,
        solver=poisson.solver,
        hocbf_alpha=config.hocbf_alpha,
        time=arrays["time"],
        position=arrays["position"],
        velocity=arrays["velocity"],
        nominal_acceleration=arrays["a_nom"],
        safe_acceleration=arrays["a_safe"],
        reference=arrays["reference"],
        active_zone=arrays["active"].astype(int),
        h_value=arrays["h"],
        hocbf_residual=arrays["hocbf_residual"],
        task_clf_residual=arrays["task_residual"],
        contingency_residual=arrays["cont_residual"],
        rho=arrays["rho"],
        contingency_margin=arrays["cont_margin"],
        reachable_count=arrays["reachable_count"].astype(int),
        omega=arrays["omega"],
        correction_norm=arrays["correction"],
        energy_used=arrays["energy"],
        emergency_slack=arrays["emergency_slack"],
        task_relaxed=arrays["task_relaxed"].astype(bool),
        switch_events=switch_events,
        blocked_events=blocked_events,
        terminated_reason=terminated_reason,
        landed=landed,
        collided=collided,
        contingency_lost=contingency_lost,
        final_zone=final_zone,
        metrics=metrics,
    )


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def add_cuboid(ax: Any, lo: Sequence[float], hi: Sequence[float], color: str, alpha: float = 0.28) -> None:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [0, 3, 7, 4]],
    ]
    collection = Poly3DCollection(faces, facecolor=color, edgecolor="0.35", linewidth=0.45, alpha=alpha)
    ax.add_collection3d(collection)


def add_cylinder(ax: Any, center: Sequence[float], radius: float, z_range: Sequence[float], color: str, alpha: float = 0.28) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 36)
    z = np.linspace(z_range[0], z_range[1], 2)
    T, ZZ = np.meshgrid(theta, z)
    XX = center[0] + radius * np.cos(T)
    YY = center[1] + radius * np.sin(T)
    ax.plot_surface(XX, YY, ZZ, color=color, alpha=alpha, linewidth=0.0, shade=True)


def draw_obstacles_3d(ax: Any, obstacles: list[Obstacle], alpha: float = 0.22) -> None:
    category_colors = {"tower": "#6b6b6b", "gate": "#9a6b39", "aerial": "#7b5ea7", "terrain": "#b55d3d"}
    for obstacle in obstacles:
        color = category_colors.get(obstacle.category, "#777777")
        if obstacle.kind == "box":
            add_cuboid(ax, obstacle.params["min"], obstacle.params["max"], color, alpha)
        elif obstacle.kind == "cylinder":
            add_cylinder(ax, obstacle.params["center"], obstacle.params["radius"], obstacle.params["z"], color, alpha)
        elif obstacle.kind == "annular_cylinder":
            add_cylinder(ax, obstacle.params["center"], obstacle.params["r_outer"], obstacle.params["z"], color, alpha * 0.75)
        elif obstacle.kind == "ellipsoid":
            center = np.asarray(obstacle.params["center"], dtype=float)
            radii = np.asarray(obstacle.params["radii"], dtype=float)
            u = np.linspace(0.0, 2.0 * np.pi, 28)
            v = np.linspace(0.0, np.pi, 16)
            xx = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
            yy = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
            zz = center[2] + radii[2] * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, linewidth=0.0)


def draw_landing_zones_3d(ax: Any, zones: list[LandingZone], blocked_zone: int, label: bool = True) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 80)
    for zone in zones:
        color = ZONE_COLORS[zone.index % len(ZONE_COLORS)]
        x = zone.position[0] + zone.radius * np.cos(theta)
        y = zone.position[1] + zone.radius * np.sin(theta)
        z = np.full_like(theta, zone.position[2])
        ax.plot(x, y, z, color=color, linewidth=2.1)
        ax.scatter(*zone.position, color=color, marker="o", s=30, edgecolor="white", linewidth=0.6)
        if zone.index == blocked_zone:
            ax.scatter(*zone.position, color="black", marker="x", s=80, linewidth=2.0)
        if label:
            ax.text(
                zone.position[0],
                zone.position[1],
                zone.position[2] + 0.35,
                f"LZ{zone.index}\nS={zone.science_score:.2f}",
                color=color,
                fontsize=7.5,
                ha="center",
            )


def format_3d_axis(ax: Any, grid: GridSpec, title: str | None = None) -> None:
    ax.set_xlim(0.0, grid.bounds[0])
    ax.set_ylim(0.0, grid.bounds[1])
    ax.set_zlim(0.0, grid.bounds[2])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=25.0, azim=-58.0)
    try:
        ax.set_box_aspect((grid.bounds[0], grid.bounds[1], 0.72 * grid.bounds[2]))
    except Exception:
        pass
    if title:
        ax.set_title(title)


def overlay_obstacles_xy(ax: plt.Axes, world: WorldData, z_value: float, alpha: float = 0.28) -> None:
    k = int(np.argmin(np.abs(world.grid.z - z_value)))
    mask = world.occupancy[:, :, k].T
    ax.contourf(world.grid.x, world.grid.y, mask, levels=[0.5, 1.5], colors=["#4d4d4d"], alpha=alpha)


def plot_workflow(output_dir: Path, dpi: int, save_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(15.2, 4.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    boxes = [
        (0.02, "3-D world +\nscience map", "geometry"),
        (0.16, "Occupancy $O$ +\nfrontier $\\partial\\Omega$", "map"),
        (0.31, "PoissonSafetyBox\n.compute(O)", "pde"),
        (0.46, "$h,\\nabla h,\\nabla^2 h$\nfield samples", "pde"),
        (0.60, "Landing certificates\n$\\rho_i=c_i-W_i$", "reach"),
        (0.75, "CBF box rows\nCLF + $r$-out-of-$p$", "reach"),
        (0.88, "CBFBox QP +\nsafe command", "control"),
    ]
    colors = {"geometry": "#f6d8c8", "map": "#f4e3b2", "pde": "#cfe8ef", "reach": "#d9ead3", "control": "#d9d2e9"}
    width = 0.105
    for x, text, category in boxes:
        patch = FancyBboxPatch(
            (x, 0.38),
            width,
            0.32,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor=colors[category],
            edgecolor="#333333",
            linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2.0, 0.54, text, ha="center", va="center", fontsize=10.2)
    for (x0, _, _), (x1, _, _) in zip(boxes[:-1], boxes[1:]):
        arrow = FancyArrowPatch(
            (x0 + width + 0.004, 0.54),
            (x1 - 0.004, 0.54),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color="#444444",
        )
        ax.add_patch(arrow)
    ax.text(
        0.5,
        0.18,
        r"QP: $\min_{a,\omega}\;\frac{1}{2}\|a-a_{nom}\|^2+c_\omega\omega^2$  subject to Poisson-HOCBF, active CLF, "
        r"and $p$ paper-inspired combinatorial CBF constraints",
        ha="center",
        va="center",
        fontsize=11.0,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#777777"},
    )
    fig.suptitle("Contingency-aware Poisson-CBF landing workflow (offline reduced-order prototype)", y=0.98)
    save_figure(fig, output_dir, "fig00_contingency_workflow", dpi, save_pdf)


def plot_world_3d(world: WorldData, blocked_zone: int, output_dir: Path, dpi: int, save_pdf: bool) -> None:
    fig = plt.figure(figsize=(12.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    draw_obstacles_3d(ax, world.obstacles, alpha=0.30)
    draw_landing_zones_3d(ax, world.landing_zones, blocked_zone, label=True)
    ax.scatter(*world.start, color="#e66101", marker="*", s=140, edgecolor="black", linewidth=0.6, label="Start")
    ax.legend(loc="upper left", frameon=True)
    format_3d_axis(ax, world.grid, "Mars-analog obstacle world and candidate landing zones")
    ax.text2D(
        0.02,
        0.02,
        "Primary site is deliberately invalidated during flight; science scores are planning metadata, not safety certificates.",
        transform=ax.transAxes,
        fontsize=9,
    )
    save_figure(fig, output_dir, "fig01_world_with_landing_zones_3d", dpi, save_pdf)


def plot_occupancy_boundary_slices(world: WorldData, output_dir: Path, dpi: int, save_pdf: bool) -> None:
    z_values = np.linspace(1.2, 7.8, 4)
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.5), sharex=True, sharey=True)
    for col, z_value in enumerate(z_values):
        k = int(np.argmin(np.abs(world.grid.z - z_value)))
        axes[0, col].imshow(
            world.occupancy[:, :, k].T,
            origin="lower",
            extent=[0, world.grid.bounds[0], 0, world.grid.bounds[1]],
            cmap=ListedColormap(["white", "#264653"]),
            interpolation="nearest",
            aspect="auto",
        )
        axes[1, col].imshow(
            world.boundary_mask[:, :, k].T,
            origin="lower",
            extent=[0, world.grid.bounds[0], 0, world.grid.bounds[1]],
            cmap=ListedColormap(["white", "#e76f51"]),
            interpolation="nearest",
            aspect="auto",
        )
        axes[0, col].set_title(f"Occupancy, z={world.grid.z[k]:.2f} m")
        axes[1, col].set_title(f"Dirichlet frontier, z={world.grid.z[k]:.2f} m")
        for row in range(2):
            axes[row, col].set_xlabel("x [m]")
            if col == 0:
                axes[row, col].set_ylabel("y [m]")
            axes[row, col].grid(False)
    fig.suptitle("Occupancy matrix and free-space frontier used by the Poisson boundary-value problem")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_figure(fig, output_dir, "fig02_occupancy_boundary_slices", dpi, save_pdf)


def plot_poisson_forcing_gradient(
    world: WorldData,
    poisson_results: dict[str, PoissonResult],
    methods: list[str],
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    k = int(np.argmin(np.abs(world.grid.z - 4.2)))
    fig, axes = plt.subplots(len(methods), 3, figsize=(14.8, 3.15 * len(methods)), squeeze=False)
    for row, method in enumerate(methods):
        result = poisson_results[method]
        forcing = result.forcing[:, :, k].T
        h_slice = result.h[:, :, k].T
        im0 = axes[row, 0].imshow(
            forcing,
            origin="lower",
            extent=[0, world.grid.bounds[0], 0, world.grid.bounds[1]],
            cmap="coolwarm",
            aspect="auto",
        )
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.045, pad=0.02, label="$f_P$")
        im1 = axes[row, 1].imshow(
            h_slice,
            origin="lower",
            extent=[0, world.grid.bounds[0], 0, world.grid.bounds[1]],
            cmap="viridis",
            aspect="auto",
        )
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.045, pad=0.02, label="$h$ (normalized)")
        axes[row, 2].imshow(
            h_slice,
            origin="lower",
            extent=[0, world.grid.bounds[0], 0, world.grid.bounds[1]],
            cmap="Greys",
            alpha=0.35,
            aspect="auto",
        )
        stride = max(2, min(world.grid.nx, world.grid.ny) // 15)
        gx = result.grad[::stride, ::stride, k, 0].T
        gy = result.grad[::stride, ::stride, k, 1].T
        axes[row, 2].quiver(
            world.grid.x[::stride],
            world.grid.y[::stride],
            gx,
            gy,
            angles="xy",
            scale_units="xy",
            scale=max(float(np.percentile(np.hypot(gx, gy), 90)) * 7.0, 0.05),
            width=0.003,
        )
        overlay_obstacles_xy(axes[row, 2], world, world.grid.z[k], alpha=0.22)
        axes[row, 0].set_ylabel(f"{method.replace('_', ' ').title()}\ny [m]")
        for col in range(3):
            axes[row, col].set_xlabel("x [m]")
            axes[row, col].grid(False)
        axes[row, 0].set_title("Forcing field")
        axes[row, 1].set_title("Poisson safety field")
        axes[row, 2].set_title("Safety gradient $\\nabla h$")
    fig.suptitle(f"Forcing, Poisson field, and gradient comparison at z={world.grid.z[k]:.2f} m")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    save_figure(fig, output_dir, "fig03_poisson_h_gradient_by_forcing", dpi, save_pdf)


def plot_poisson_isosurfaces(
    world: WorldData,
    poisson_results: dict[str, PoissonResult],
    methods: list[str],
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
    fast: bool,
) -> None:
    ncols = 2
    nrows = int(math.ceil(len(methods) / ncols))
    fig = plt.figure(figsize=(14.0, 6.1 * nrows))
    for plot_index, method in enumerate(methods, start=1):
        ax = fig.add_subplot(nrows, ncols, plot_index, projection="3d")
        result = poisson_results[method]
        values = result.h[world.unknown_mask]
        levels = [float(np.percentile(values, 40)), float(np.percentile(values, 72))]
        colors = [cm.viridis(0.35), cm.viridis(0.76)]
        if HAVE_SKIMAGE:
            for level, color in zip(levels, colors):
                try:
                    verts, faces, _, _ = marching_cubes(
                        result.h.astype(np.float32),
                        level=level,
                        spacing=world.grid.spacing,
                        step_size=2 if fast else 1,
                        allow_degenerate=False,
                    )
                    if len(faces) > (18000 if fast else 45000):
                        faces = faces[:: max(1, len(faces) // (18000 if fast else 45000))]
                    mesh = Poly3DCollection(verts[faces], alpha=0.18, facecolor=color, edgecolor="none")
                    ax.add_collection3d(mesh)
                except Exception:
                    pass
        else:  # pragma: no cover - optional fallback
            for level, color in zip(levels, colors):
                tolerance = 0.025 * max(level, 0.05)
                points = np.argwhere(np.abs(result.h - level) <= tolerance)
                if len(points) > 3500:
                    points = points[:: len(points) // 3500]
                coords = points * np.asarray(world.grid.spacing)
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1.0, color=color, alpha=0.20)
        draw_landing_zones_3d(ax, world.landing_zones, blocked_zone=-1, label=False)
        ax.scatter(*world.start, color="#e66101", marker="*", s=70)
        format_3d_axis(ax, world.grid, method.replace("_", " ").title())
    fig.suptitle("Three-dimensional Poisson safety-field isosurfaces")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    save_figure(fig, output_dir, "fig04_poisson_3d_isosurfaces", dpi, save_pdf)


def certificate_map(
    world: WorldData,
    poisson: PoissonResult,
    model: ReachabilityModel,
    zone_index: int,
    z_value: float,
    t: float,
    energy: float = 0.0,
) -> np.ndarray:
    rho_map = np.empty((world.grid.nx, world.grid.ny), dtype=float)
    velocity = np.zeros(3)
    for i, x in enumerate(world.grid.x):
        for j, y in enumerate(world.grid.y):
            item = evaluate_reachability_certificate(
                world,
                poisson,
                model,
                zone_index,
                np.array([x, y, z_value]),
                velocity,
                t,
                energy,
                0.0,
                blocked_zone=-1,
                failure_time=-1.0,
            )
            rho_map[i, j] = item["rho"]
    return rho_map


def plot_reachability_fields(
    world: WorldData,
    poisson: PoissonResult,
    model: ReachabilityModel,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> list[np.ndarray]:
    z_value = 4.2
    maps = [certificate_map(world, poisson, model, i, z_value, 0.0) for i in range(len(model.zones))]
    ncols = 2
    nrows = int(math.ceil(len(maps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 5.4 * nrows), squeeze=False)
    vmax = max(float(np.max(np.abs(item))) for item in maps)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    for index, rho_map in enumerate(maps):
        ax = axes[index // ncols, index % ncols]
        im = ax.imshow(
            rho_map.T,
            origin="lower",
            extent=[0, world.grid.bounds[0], 0, world.grid.bounds[1]],
            cmap="RdYlGn",
            norm=norm,
            aspect="auto",
        )
        ax.contour(world.grid.x, world.grid.y, rho_map.T, levels=[0.0], colors="black", linewidths=1.4)
        overlay_obstacles_xy(ax, world, z_value, alpha=0.26)
        zone = model.zones[index]
        ax.scatter(zone.position[0], zone.position[1], marker="*", s=140, color=ZONE_COLORS[index], edgecolor="black")
        ax.set_title(f"$\\rho_{index}(p,0,0)$ — {zone.name}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="reachability-proxy margin")
    for index in range(len(maps), nrows * ncols):
        axes[index // ncols, index % ncols].axis("off")
    fig.suptitle(f"Landing-zone CLF/geodesic/Poisson reachability-proxy fields at z={z_value:.1f} m")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    save_figure(fig, output_dir, "fig05_landing_zone_reachability_fields", dpi, save_pdf)
    return maps


def plot_contingency_margin_map(
    world: WorldData,
    rho_maps: list[np.ndarray],
    r: int,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    stack = np.stack(rho_maps, axis=-1)
    sorted_values = np.sort(stack, axis=-1)
    pivot = sorted_values[..., -r]
    count = np.sum(stack >= 0.0, axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))
    vmax = max(float(np.max(np.abs(pivot))), 0.05)
    im0 = axes[0].imshow(
        pivot.T,
        origin="lower",
        extent=[0, world.grid.bounds[0], 0, world.grid.bounds[1]],
        cmap="RdYlGn",
        norm=Normalize(vmin=-vmax, vmax=vmax),
        aspect="auto",
    )
    axes[0].contour(world.grid.x, world.grid.y, pivot.T, levels=[0.0], colors="black", linewidths=1.6)
    overlay_obstacles_xy(axes[0], world, 4.2, alpha=0.28)
    fig.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.02, label=f"{r}-th largest margin")
    axes[0].set_title(f"Continuous r-out-of-p pivot $\\tilde\\rho=\\max^{{({r})}}_i\\rho_i$")

    cmap = ListedColormap(["#7f0000", "#d73027", "#fee08b", "#91cf60", "#1a9850", "#006837"][: len(rho_maps) + 1])
    boundaries = np.arange(-0.5, len(rho_maps) + 1.5, 1.0)
    im1 = axes[1].imshow(
        count.T,
        origin="lower",
        extent=[0, world.grid.bounds[0], 0, world.grid.bounds[1]],
        cmap=cmap,
        norm=BoundaryNorm(boundaries, cmap.N),
        aspect="auto",
    )
    overlay_obstacles_xy(axes[1], world, 4.2, alpha=0.28)
    colorbar = fig.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.02, ticks=np.arange(len(rho_maps) + 1))
    colorbar.set_label("number of nonnegative landing-zone certificates")
    axes[1].set_title(f"Reachable-zone count; safe contingency region requires count $\\geq {r}$")
    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(False)
    fig.suptitle("Combinatorial contingency geometry at zero velocity and cruise altitude")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_figure(fig, output_dir, "fig06_contingency_margin_map", dpi, save_pdf)


def plot_trajectory_switching_3d(
    world: WorldData,
    simulation: SimulationResult,
    blocked_zone: int,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    fig = plt.figure(figsize=(13.0, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    draw_obstacles_3d(ax, world.obstacles, alpha=0.20)
    draw_landing_zones_3d(ax, world.landing_zones, blocked_zone, label=True)

    # Plot trajectory segments colored by active target.
    changes = np.where(np.diff(simulation.active_zone) != 0)[0] + 1
    boundaries = np.concatenate([[0], changes, [len(simulation.time)]])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        zone_index = int(simulation.active_zone[start])
        segment = simulation.position[start:end]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=ZONE_COLORS[zone_index % len(ZONE_COLORS)],
            linewidth=3.0,
            label=f"Target LZ{zone_index}" if start == 0 or zone_index not in simulation.active_zone[:start] else None,
        )
    ax.plot(
        simulation.reference[:, 0],
        simulation.reference[:, 1],
        simulation.reference[:, 2],
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
        label="Planner reference",
    )
    ax.scatter(*simulation.position[0], marker="*", s=130, color="#e66101", edgecolor="black", label="Start")
    for event in simulation.switch_events:
        index = int(np.argmin(np.abs(simulation.time - event["time"])))
        point = simulation.position[index]
        ax.scatter(*point, marker="D", s=85, color="#ffd92f", edgecolor="black", linewidth=0.8)
        ax.text(point[0], point[1], point[2] + 0.45, f"switch t={event['time']:.1f}s", fontsize=8)
    format_3d_axis(ax, world.grid, "Contingency-aware trajectory, failure detection, and safe diversion")
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(), loc="upper left", frameon=True)
    ax.text2D(
        0.02,
        0.02,
        f"Outcome: {simulation.terminated_reason}; final target LZ{simulation.final_zone}; "
        f"min contingency margin={simulation.metrics['minimum_contingency_margin']:.3f}",
        transform=ax.transAxes,
        fontsize=9.2,
    )
    save_figure(fig, output_dir, "fig07_contingency_trajectory_switching_3d", dpi, save_pdf)


def plot_time_histories(
    simulation: SimulationResult,
    r: int,
    failure_time: float,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15.0, 11.0), sharex=True)
    t = simulation.time
    axes[0, 0].plot(t, simulation.h_value, label="$h_{CBF}$")
    axes[0, 0].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 0].set_ylabel("Poisson barrier")
    axes[0, 0].set_title("Environment safety")
    twin = axes[0, 0].twinx()
    twin.plot(t, simulation.hocbf_residual, color="#d95f02", alpha=0.75, label="HOCBF residual")
    twin.axhline(0.0, color="#d95f02", linewidth=0.8, linestyle="--")
    twin.set_ylabel("HOCBF residual", color="#d95f02")

    for index in range(simulation.rho.shape[1]):
        axes[0, 1].plot(t, simulation.rho[:, index], color=ZONE_COLORS[index], label=f"$\\rho_{index}$")
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 1].set_ylabel("certificate margin")
    axes[0, 1].set_title("Landing-zone reachability proxies")
    axes[0, 1].legend(ncol=2)

    axes[1, 0].plot(t, simulation.contingency_margin, color="#1b7837", label="$\\tilde\\rho$")
    axes[1, 0].axhline(0.0, color="black", linewidth=1.0)
    axes[1, 0].fill_between(t, 0.0, simulation.contingency_margin, where=simulation.contingency_margin >= 0.0, alpha=0.18)
    axes[1, 0].set_ylabel(f"{r}-out-of-p pivot")
    axes[1, 0].set_title("Combinatorial contingency margin")

    axes[1, 1].step(t, simulation.reachable_count, where="post", color="#2166ac")
    axes[1, 1].axhline(r, color="#b2182b", linestyle="--", label=f"required r={r}")
    axes[1, 1].set_ylabel("reachable zones")
    axes[1, 1].set_ylim(-0.15, simulation.rho.shape[1] + 0.35)
    axes[1, 1].set_title("Reachable-zone count")
    axes[1, 1].legend()

    axes[2, 0].step(t, simulation.active_zone, where="post", color="#762a83")
    axes[2, 0].set_yticks(np.arange(simulation.rho.shape[1]))
    axes[2, 0].set_ylabel("active LZ index")
    axes[2, 0].set_title("Target selection and switch")

    axes[2, 1].plot(t, simulation.correction_norm, label="$\\|a_{safe}-a_{nom}\\|$")
    axes[2, 1].plot(t, simulation.omega, label="$\\omega$", alpha=0.75)
    axes[2, 1].plot(t, simulation.emergency_slack, label="emergency slack", alpha=0.8)
    axes[2, 1].set_ylabel("filter activity")
    axes[2, 1].set_title("QP intervention")
    axes[2, 1].legend()

    for ax in axes.flat:
        if failure_time >= 0.0:
            ax.axvline(failure_time, color="#b2182b", linestyle=":", linewidth=1.2)
        ax.set_xlabel("time [s]")
    fig.suptitle("Contingency-aware Poisson-HOCBF time histories")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    save_figure(fig, output_dir, "fig08_contingency_time_histories", dpi, save_pdf)


def plot_alpha_sweep(
    world: WorldData,
    alpha_results: list[SimulationResult],
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    metrics = pd.DataFrame([{"alpha": result.hocbf_alpha, **result.metrics} for result in alpha_results]).sort_values("alpha")
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.5))
    norm = LogNorm(vmin=max(metrics["alpha"].min(), 1e-3), vmax=metrics["alpha"].max())
    for result in alpha_results:
        color = cm.plasma(norm(result.hocbf_alpha))
        axes[0, 0].plot(result.position[:, 0], result.position[:, 1], color=color, alpha=0.85)
    overlay_obstacles_xy(axes[0, 0], world, 4.2, alpha=0.22)
    for zone in world.landing_zones:
        axes[0, 0].scatter(zone.position[0], zone.position[1], color=ZONE_COLORS[zone.index], s=28)
    axes[0, 0].set_title("XY paths across HOCBF gain $\\alpha$")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].grid(False)
    scalar = cm.ScalarMappable(norm=norm, cmap="plasma")
    fig.colorbar(scalar, ax=axes[0, 0], fraction=0.045, pad=0.02, label="$\\alpha$")

    axes[0, 1].semilogx(metrics["alpha"], metrics["minimum_h_cbf"], marker="o")
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 1].set_title("Minimum Poisson barrier")
    axes[0, 1].set_xlabel("$\\alpha$")
    axes[0, 1].set_ylabel("min $h_{CBF}$")

    axes[0, 2].semilogx(metrics["alpha"], metrics["minimum_contingency_margin"], marker="o", color="#1b7837")
    axes[0, 2].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 2].set_title("Minimum r-out-of-p margin")
    axes[0, 2].set_xlabel("$\\alpha$")
    axes[0, 2].set_ylabel("min $\\tilde\\rho$")

    axes[1, 0].semilogx(metrics["alpha"], metrics["final_distance_to_active_zone_m"], marker="o", color="#2166ac")
    axes[1, 0].set_title("Final distance to selected LZ")
    axes[1, 0].set_xlabel("$\\alpha$")
    axes[1, 0].set_ylabel("distance [m]")

    axes[1, 1].semilogx(metrics["alpha"], metrics["num_switches"], marker="o", color="#762a83")
    axes[1, 1].set_title("Number of target switches")
    axes[1, 1].set_xlabel("$\\alpha$")
    axes[1, 1].set_ylabel("switches")

    axes[1, 2].semilogx(metrics["alpha"], metrics["integrated_control_correction"], marker="o", color="#d95f02")
    axes[1, 2].set_title("Integrated safety-filter correction")
    axes[1, 2].set_xlabel("$\\alpha$")
    axes[1, 2].set_ylabel("$\\int\\|a_{safe}-a_{nom}\\|dt$")

    fig.suptitle("HOCBF gain sweep: safety, contingency, and task trade-offs")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_figure(fig, output_dir, "fig09_alpha_sweep_contingency", dpi, save_pdf)


def plot_forcing_comparison(
    world: WorldData,
    forcing_results: list[SimulationResult],
    poisson_results: dict[str, PoissonResult],
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    metrics = pd.DataFrame([{"forcing": result.forcing_method, **result.metrics} for result in forcing_results])
    ordered = [result.forcing_method for result in forcing_results]
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.5))
    for result in forcing_results:
        axes[0, 0].plot(
            result.position[:, 0],
            result.position[:, 1],
            label=result.forcing_method.replace("_", " "),
            color=FORCING_COLORS.get(result.forcing_method),
        )
    overlay_obstacles_xy(axes[0, 0], world, 4.2, alpha=0.22)
    axes[0, 0].set_title("Contingency-aware trajectories")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].legend()
    axes[0, 0].grid(False)

    x = np.arange(len(ordered))
    colors = [FORCING_COLORS.get(item, "#777777") for item in ordered]
    axes[0, 1].bar(x, metrics.set_index("forcing").loc[ordered, "minimum_h_cbf"], color=colors)
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 1].set_title("Minimum Poisson barrier")
    axes[0, 1].set_xticks(x, [item.replace("_", "\n") for item in ordered])

    axes[0, 2].bar(x, metrics.set_index("forcing").loc[ordered, "minimum_contingency_margin"], color=colors)
    axes[0, 2].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 2].set_title("Minimum contingency margin")
    axes[0, 2].set_xticks(x, [item.replace("_", "\n") for item in ordered])

    axes[1, 0].bar(x, metrics.set_index("forcing").loc[ordered, "final_zone"], color=colors)
    axes[1, 0].set_title("Final landing-zone index")
    axes[1, 0].set_xticks(x, [item.replace("_", "\n") for item in ordered])
    axes[1, 0].set_ylabel("LZ index")

    axes[1, 1].bar(x, [poisson_results[item].solve_time_s for item in ordered], color=colors)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Poisson solve time")
    axes[1, 1].set_xticks(x, [item.replace("_", "\n") for item in ordered])
    axes[1, 1].set_ylabel("time [s]")

    axes[1, 2].bar(x, metrics.set_index("forcing").loc[ordered, "integrated_control_correction"], color=colors)
    axes[1, 2].set_title("Integrated filter correction")
    axes[1, 2].set_xticks(x, [item.replace("_", "\n") for item in ordered])
    axes[1, 2].set_ylabel("correction integral")

    fig.suptitle("Poisson forcing-function comparison under the contingency-aware filter")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_figure(fig, output_dir, "fig10_forcing_contingency_comparison", dpi, save_pdf)


def annotate_heatmap(ax: plt.Axes, matrix: np.ndarray, fmt: str) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "—" if not np.isfinite(value) else format(value, fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")


def plot_solver_heatmaps(
    solver_metrics: pd.DataFrame,
    solvers: list[str],
    methods: list[str],
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))
    quantities = [
        ("solve_time_s", "Poisson wall time [s]", "viridis", ".2g", True),
        ("relative_residual", "Relative residual", "magma_r", ".1e", True),
        ("relative_field_error", "Relative field error vs direct", "cividis", ".1e", True),
        ("contingency_outcome_score", "Contingency outcome score", "RdYlGn", ".2f", False),
    ]
    for ax, (column, title, cmap, fmt, log_scale) in zip(axes.flat, quantities):
        matrix = np.full((len(solvers), len(methods)), np.nan)
        for i, solver in enumerate(solvers):
            for j, method in enumerate(methods):
                match = solver_metrics[(solver_metrics["solver"] == solver) & (solver_metrics["forcing"] == method)]
                if not match.empty:
                    matrix[i, j] = float(match.iloc[0][column])
        if log_scale:
            positive = matrix[np.isfinite(matrix) & (matrix > 0)]
            if positive.size:
                norm = LogNorm(vmin=max(float(np.min(positive)), 1e-14), vmax=max(float(np.max(positive)), float(np.min(positive)) * 1.01))
            else:
                norm = None
        else:
            norm = Normalize(vmin=0.0, vmax=1.0)
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(np.arange(len(methods)), [item.replace("_", "\n") for item in methods])
        ax.set_yticks(np.arange(len(solvers)), [SOLVER_LABELS.get(item, item) for item in solvers])
        ax.set_title(title)
        ax.grid(False)
        annotate_heatmap(ax, matrix, fmt)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Solver × forcing diagnostics and contingency outcome")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_figure(fig, output_dir, "fig11_solver_contingency_heatmaps", dpi, save_pdf)


def plot_dashboard(
    world: WorldData,
    primary: SimulationResult,
    alpha_df: pd.DataFrame,
    forcing_df: pd.DataFrame,
    solver_df: pd.DataFrame,
    r: int,
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    fig = plt.figure(figsize=(18.0, 12.0))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.0, 0.95], hspace=0.34, wspace=0.31)

    ax_world = fig.add_subplot(gs[0:2, 0:2], projection="3d")
    draw_obstacles_3d(ax_world, world.obstacles, alpha=0.15)
    draw_landing_zones_3d(ax_world, world.landing_zones, blocked_zone=-1, label=False)
    ax_world.plot(primary.position[:, 0], primary.position[:, 1], primary.position[:, 2], color="#2166ac", linewidth=2.8)
    for event in primary.switch_events:
        idx = int(np.argmin(np.abs(primary.time - event["time"])))
        ax_world.scatter(*primary.position[idx], marker="D", s=65, color="#ffd92f", edgecolor="black")
    format_3d_axis(ax_world, world.grid, "Primary contingency-aware landing run")

    ax_rho = fig.add_subplot(gs[0, 2:4])
    for index in range(primary.rho.shape[1]):
        ax_rho.plot(primary.time, primary.rho[:, index], color=ZONE_COLORS[index], label=f"LZ{index}")
    ax_rho.plot(primary.time, primary.contingency_margin, color="black", linewidth=2.5, label="$\\tilde\\rho$")
    ax_rho.axhline(0.0, color="black", linewidth=0.8)
    ax_rho.set_title("Individual reachability proxies and r-out-of-p pivot")
    ax_rho.set_xlabel("time [s]")
    ax_rho.set_ylabel("margin")
    ax_rho.legend(ncol=3)

    ax_count = fig.add_subplot(gs[1, 2])
    ax_count.step(primary.time, primary.reachable_count, where="post", color="#2166ac")
    ax_count.axhline(r, color="#b2182b", linestyle="--")
    ax_count.set_title("Reachable-zone count")
    ax_count.set_xlabel("time [s]")
    ax_count.set_ylabel("count")

    ax_h = fig.add_subplot(gs[1, 3])
    ax_h.plot(primary.time, primary.h_value, label="$h_{CBF}$")
    ax_h.plot(primary.time, primary.hocbf_residual, label="HOCBF residual")
    ax_h.axhline(0.0, color="black", linewidth=0.8)
    ax_h.set_title("Poisson-HOCBF safety")
    ax_h.set_xlabel("time [s]")
    ax_h.legend()

    ax_alpha = fig.add_subplot(gs[2, 0])
    alpha_sorted = alpha_df.sort_values("alpha")
    ax_alpha.semilogx(alpha_sorted["alpha"], alpha_sorted["minimum_contingency_margin"], marker="o")
    ax_alpha.axhline(0.0, color="black", linewidth=0.8)
    ax_alpha.set_title("Alpha sweep: min contingency margin")
    ax_alpha.set_xlabel("$\\alpha$")

    ax_forcing = fig.add_subplot(gs[2, 1])
    ax_forcing.bar(
        np.arange(len(forcing_df)),
        forcing_df["minimum_h_cbf"],
        color=[FORCING_COLORS.get(item, "#777777") for item in forcing_df["forcing"]],
    )
    ax_forcing.set_xticks(np.arange(len(forcing_df)), [item.replace("_", "\n") for item in forcing_df["forcing"]], fontsize=7)
    ax_forcing.axhline(0.0, color="black", linewidth=0.8)
    ax_forcing.set_title("Forcing comparison: min h")

    ax_solver = fig.add_subplot(gs[2, 2])
    pivot = solver_df.pivot(index="solver", columns="forcing", values="solve_time_s")
    image = ax_solver.imshow(np.log10(np.maximum(pivot.to_numpy(), 1e-12)), cmap="viridis", aspect="auto")
    ax_solver.set_xticks(np.arange(len(pivot.columns)), [item.replace("_", "\n") for item in pivot.columns], fontsize=7)
    ax_solver.set_yticks(np.arange(len(pivot.index)), [SOLVER_LABELS.get(item, item) for item in pivot.index], fontsize=7)
    ax_solver.set_title("$\\log_{10}$ Poisson solve time")
    ax_solver.grid(False)
    fig.colorbar(image, ax=ax_solver, fraction=0.05, pad=0.03)

    ax_text = fig.add_subplot(gs[2, 3])
    ax_text.axis("off")
    metrics = primary.metrics
    summary_lines = [
        "PRIMARY RUN",
        f"Outcome: {metrics['terminated_reason']}",
        f"Final target: LZ{metrics['final_zone']}",
        f"Blocked-site switch: {metrics['blocked_switch_occurred']}",
        f"Min h: {metrics['minimum_h_cbf']:.3e}",
        f"Min r-out-of-p margin: {metrics['minimum_contingency_margin']:.3f}",
        f"Min reachable count: {metrics['minimum_reachable_count']}",
        f"Filter correction: {metrics['integrated_control_correction']:.2f}",
        "",
        "Scope: offline, reduced-order,",
        "not PX4/rate-controller validated.",
    ]
    ax_text.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=10.2,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f7f7f7", "edgecolor": "#666666"},
    )

    fig.suptitle("Poisson-CBF contingency landing study — integrated dashboard", fontsize=16.0, y=0.985)
    save_figure(fig, output_dir, "fig12_integrated_dashboard", dpi, save_pdf)


# -----------------------------------------------------------------------------
# Data exports and explainer
# -----------------------------------------------------------------------------


def simulation_to_dataframe(simulation: SimulationResult) -> pd.DataFrame:
    data: dict[str, Any] = {
        "time_s": simulation.time,
        "x_m": simulation.position[:, 0],
        "y_m": simulation.position[:, 1],
        "z_m": simulation.position[:, 2],
        "vx_mps": simulation.velocity[:, 0],
        "vy_mps": simulation.velocity[:, 1],
        "vz_mps": simulation.velocity[:, 2],
        "a_nom_x_mps2": simulation.nominal_acceleration[:, 0],
        "a_nom_y_mps2": simulation.nominal_acceleration[:, 1],
        "a_nom_z_mps2": simulation.nominal_acceleration[:, 2],
        "a_safe_x_mps2": simulation.safe_acceleration[:, 0],
        "a_safe_y_mps2": simulation.safe_acceleration[:, 1],
        "a_safe_z_mps2": simulation.safe_acceleration[:, 2],
        "reference_x_m": simulation.reference[:, 0],
        "reference_y_m": simulation.reference[:, 1],
        "reference_z_m": simulation.reference[:, 2],
        "active_zone": simulation.active_zone,
        "poisson_h_cbf": simulation.h_value,
        "hocbf_residual": simulation.hocbf_residual,
        "task_clf_residual": simulation.task_clf_residual,
        "min_contingency_constraint_residual": simulation.contingency_residual,
        "contingency_margin": simulation.contingency_margin,
        "reachable_zone_count": simulation.reachable_count,
        "omega": simulation.omega,
        "control_correction_norm": simulation.correction_norm,
        "energy_proxy": simulation.energy_used,
        "emergency_slack": simulation.emergency_slack,
        "task_clf_relaxed": simulation.task_relaxed.astype(int),
    }
    for index in range(simulation.rho.shape[1]):
        data[f"rho_{index}"] = simulation.rho[:, index]
    return pd.DataFrame(data)


def save_trajectory_npz(path: Path, simulation: SimulationResult) -> None:
    np.savez_compressed(
        path,
        time=simulation.time,
        position=simulation.position,
        velocity=simulation.velocity,
        nominal_acceleration=simulation.nominal_acceleration,
        safe_acceleration=simulation.safe_acceleration,
        reference=simulation.reference,
        active_zone=simulation.active_zone,
        h_value=simulation.h_value,
        hocbf_residual=simulation.hocbf_residual,
        task_clf_residual=simulation.task_clf_residual,
        contingency_residual=simulation.contingency_residual,
        rho=simulation.rho,
        contingency_margin=simulation.contingency_margin,
        reachable_count=simulation.reachable_count,
        omega=simulation.omega,
        correction_norm=simulation.correction_norm,
        energy_used=simulation.energy_used,
        emergency_slack=simulation.emergency_slack,
        task_relaxed=simulation.task_relaxed,
    )


def write_explainer(
    output_dir: Path,
    world: WorldData,
    config: SimulationConfig,
    primary: SimulationResult,
    forcing_methods: list[str],
    solvers: list[str],
) -> None:
    figures = [
        ("fig00_contingency_workflow.png", "End-to-end geometry → PDE → certificate → combinatorial filter → HOCBF pipeline."),
        ("fig01_world_with_landing_zones_3d.png", "Analytic obstacle world, landing sites, blocked primary, and science/terrain metadata."),
        ("fig02_occupancy_boundary_slices.png", "Voxel occupancy and the free-cell frontier on which the Dirichlet value is zero."),
        ("fig03_poisson_h_gradient_by_forcing.png", "Forcing, normalized Poisson field, and gradient at a cruise-altitude slice."),
        ("fig04_poisson_3d_isosurfaces.png", "Three-dimensional level-set geometry for each forcing method."),
        ("fig05_landing_zone_reachability_fields.png", "Per-zone smooth CLF/geodesic/Poisson proxy margins at zero velocity."),
        ("fig06_contingency_margin_map.png", "The r-th largest certificate and the integer number of currently reachable zones."),
        ("fig07_contingency_trajectory_switching_3d.png", "Primary approach, failure detection, switch marker, diversion, and landing trajectory."),
        ("fig08_contingency_time_histories.png", "Poisson safety, HOCBF residual, all certificates, pivot, count, target index, and QP activity."),
        ("fig09_alpha_sweep_contingency.png", "HOCBF-gain sensitivity of paths, safety, contingency, switching, and intervention."),
        ("fig10_forcing_contingency_comparison.png", "Trajectory and performance differences induced by the Poisson forcing choice."),
        ("fig11_solver_contingency_heatmaps.png", "Solver × forcing timing, residual, field error, and closed-loop outcome score."),
        ("fig12_integrated_dashboard.png", "One-page weekly-presentation dashboard."),
    ]
    lines = [
        "# Poisson-CBF Contingency Landing Study",
        "",
        "## Scope and honesty statement",
        "",
        "This is an **offline, reduced-order Poisson-CBF-HOCBF simulation** and a contingency-aware landing prototype. It is **not yet PX4 validated**, **not yet rate-controller validated**, and does not model attitude, rotor/motor dynamics, state-estimation error, aerodynamic uncertainty, or flight-stack tracking error.",
        "",
        "The combinatorial filter structure is taken directly from the paper-inspired CBF construction. The per-zone quantities used here are smooth numerical **CLF/geodesic/Poisson reachability proxies**, not Hamilton-Jacobi backward reach-avoid value functions. Therefore, `rho_i >= 0` is an offline certificate convention, not a theorem that the full vehicle can land at zone i.",
        "",
        "## Mandatory modular architecture",
        "",
        "The executable imports the real `poisson_safety_box` and `cbf_safety_box` packages located beside the runtime. The runtime only generates the world, landing certificates, mission reference, logs, and figures. It does not implement a second Poisson solver or a second CBF-QP.",
        "",
        "```text",
        "occupancy -> PoissonSafetyBox.compute -> PoissonBoxResult",
        "PoissonBoxResult -> poisson_adapter -> SafetySample",
        "SafetySample -> acceleration HOCBF builder",
        "landing certificates -> combinatorial contingency builders",
        "all affine rows -> CBFBox.filter_affine_constraints -> [a_safe, omega]",
        "```",
        "",
        "## 1. Poisson boundary-value problem",
        "",
        "The occupancy grid is passed to the mandatory `PoissonSafetyBox.compute(occupancy)` API. The box constructs free space `Omega`, the Dirichlet frontier `partial Omega`, the forcing field, the sparse system, and solves",
        "",
        "```math",
        "\\Delta h(y)=f_P(y),\\quad y\\in\\Omega,\\qquad h(y)=0,\\quad y\\in\\partial\\Omega.",
        "```",
        "",
        "Obstacle geometry enters through the domain and Dirichlet boundary. The forcing shapes the interior curvature; it does not directly encode obstacle indicator terms.",
        "",
        "## 2. Discrete stencil",
        "",
        "The positive-definite sparse system represents `-Delta h = -f_P` so that negative forcing yields positive interior values:",
        "",
        "```math",
        "A h=b,\\qquad A_{ii}=2(\\Delta x^{-2}+\\Delta y^{-2}+\\Delta z^{-2}),",
        "```",
        "",
        "with neighbor coefficients `-Delta x^-2`, `-Delta y^-2`, and `-Delta z^-2`. Frontier and occupied values are fixed at zero.",
        "",
        "## 3. Acceleration-level HOCBF",
        "",
        "For `p_dot=v`, `v_dot=a`, the environmental constraint is",
        "",
        "```math",
        "\\nabla h(p)^T a + v^T\\nabla^2h(p)v +(\\alpha_1+\\alpha_2)\\nabla h(p)^Tv+\\alpha_1\\alpha_2h(p)\\ge 0.",
        "```",
        "",
        "The study uses `alpha_1=alpha_2=alpha`. This constraint is affine in acceleration and is included as a hard QP inequality unless an explicitly logged emergency slack is required.",
        "",
        "## 4. Landing-zone reachability proxy",
        "",
        "For target `L_i`, define `x_i=[p-L_i; v]` and",
        "",
        "```math",
        "W_i(p,v)=\\frac{x_i^T P x_i}{s_i}+w_g\\left(\\frac{d_i^{geo}(p_{xy})}{D_{xy}}\\right)^2+w_h\\phi(h(p))^2+w_v\\frac{\\|v\\|^2}{v_{max}^2},",
        "```",
        "",
        "where `P` solves a continuous-time Lyapunov equation for the nominal double-integrator PD closed loop, `d_i^geo` is a smoothed obstacle/Poisson-weighted Dijkstra distance, `phi` is a smooth low-Poisson-value risk penalty, and the normalized kinetic term represents stopping effort. The time/resource budget is",
        "",
        "```math",
        "c_i(t,E)=c_{i0}-\\beta_t t-\\beta_E E-B_i(t),\\qquad \\rho_i=c_i-W_i.",
        "```",
        "",
        "`B_i(t)` is a smooth capacity collapse for the newly blocked zone. The derivative `dot rho_i` is affine in `a` because `W_i` contains position-velocity cross terms through `P` and an explicit kinetic/stopping term. A confirmed blocked site is marked unavailable and no longer preserved, while it remains plotted as an unreachable member of the original candidate family.",
        "",
        "## 5. Exact r-out-of-p pivot and paper-inspired p-constraint filter",
        "",
        f"The required contingency is `r={config.r_contingency}` out of `p={len(world.landing_zones)}`. The nonsmooth diagnostic pivot is the r-th largest certificate:",
        "",
        "```math",
        "\\tilde\\rho(x)=\\max^{(r)}\\{\\rho_i(x)\\}_{i=1}^p.",
        "```",
        "",
        "Rather than differentiating this order statistic, `cbf_safety_box.constraints.combinatorial_contingency` builds the paper-inspired smooth p-constraint construction:",
        "",
        "```math",
        "\\dot\\rho_i\\ge-\\gamma_c\\rho_i-\\omega R(\\rho_i-\\tilde\\rho),\\qquad i=1,\\dots,p,",
        "```",
        "",
        "with `R(s)=k_R s^2` and `omega>=0`. The relaxation vanishes for pivot-critical certificates, which is the mechanism that preserves the r-out-of-p superlevel set without enumerating combinations.",
        "",
        "## 6. Combined filter",
        "",
        "The online reduced-order QP is",
        "",
        "```math",
        "\\min_{a,\\omega\\ge0}\\;\\frac12\\|a-a_{nom}\\|_2^2+c_\\omega\\omega^2",
        "```",
        "",
        "subject to the environment HOCBF, one active-target CLF inequality, all `p` combinatorial certificate inequalities, and acceleration limits. The complete augmented decision is solved by `CBFBox.filter_affine_constraints`; the runtime contains no independent QP solver. Thus, the paper's active-CLF-plus-`p` structure is preserved and the Poisson HOCBF is added as an independent hard safety row. The active-target CLF is dropped only when bounded-input feasibility would otherwise force an emergency violation of the hard environment/contingency constraints; that event is logged as `task_clf_relaxed`.",
        "",
        "## 7. Failure and switch logic",
        "",
        f"The primary site LZ{config.blocked_zone} is declared blocked at `t={config.failure_time:.1f} s`. The planner then chooses the highest-scoring unblocked backup among the current certificates, using reachability margin, science/terrain value, and geodesic effort. The primary run recorded {len(primary.switch_events)} switch event(s) and terminated with `{primary.terminated_reason}` at LZ{primary.final_zone}.",
        "",
        "## 8. Figure guide",
        "",
    ]
    for filename, description in figures:
        lines.append(f"- **{filename}** — {description}")
    lines += [
        "",
        "## 9. Numerical configuration",
        "",
        f"- Grid: `{world.grid.shape[0]} x {world.grid.shape[1]} x {world.grid.shape[2]}` over `{world.grid.bounds}` m.",
        f"- Forcing methods: `{', '.join(forcing_methods)}`.",
        f"- Solver sweep: `{', '.join(solvers)}`.",
        f"- Dynamics step: `{config.dt}` s; maximum steps: `{config.max_steps}`.",
        f"- Acceleration/speed limits: `{config.max_acc}` m/s^2 and `{config.max_speed}` m/s.",
        "",
        "## 10. What remains unvalidated",
        "",
        "The next validation stages are ROS 2/Gazebo/PX4 Offboard integration, explicit tracking-error robustness, Crazyflie/OptiTrack experiments, and later X500-scale tests. A full HJ implementation would replace `rho_i` with numerically computed reach-avoid value functions and would require a separate high-dimensional reachability study.",
        "",
    ]
    (output_dir / "CONTINGENCY_STUDY_EXPLAINER.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main study orchestration
# -----------------------------------------------------------------------------


def outcome_score(simulation: SimulationResult) -> float:
    score = 0.0
    if not simulation.collided:
        score += 0.20
    if simulation.metrics["maintained_r_out_of_p"]:
        score += 0.35
    if simulation.metrics["blocked_switch_occurred"]:
        score += 0.20
    if simulation.landed:
        score += 0.25
    return float(score)


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    configure_matplotlib()
    output_dir = Path(args.output_dir).resolve()
    ensure_output_dir(output_dir)

    grid = GridSpec(shape=args.grid_shape)
    world = build_world(grid, args.num_landing_zones, args.seed)
    if not 1 <= args.r_contingency <= len(world.landing_zones):
        raise ValueError(f"r-contingency must be between 1 and {len(world.landing_zones)}.")
    if not 0 <= args.active_zone < len(world.landing_zones):
        raise ValueError("active-zone is out of range.")
    if not 0 <= args.blocked_zone < len(world.landing_zones):
        raise ValueError("blocked-zone is out of range.")

    forcing_methods = [item.lower() for item in args.forcing_methods]
    valid_forcing = {"constant", "distance", "average_flux", "guidance"}
    if any(item not in valid_forcing for item in forcing_methods):
        raise ValueError(f"Forcing methods must be drawn from {sorted(valid_forcing)}.")
    if args.fixed_forcing not in forcing_methods:
        raise ValueError("fixed-forcing must be included in forcing-methods.")
    solvers = [item.lower() for item in args.solver_sweep_solvers]
    valid_solvers = {"sparse_direct", "conjugate_gradient", "sor"}
    if args.solver not in valid_solvers or any(item not in valid_solvers for item in solvers):
        raise ValueError(f"Solvers must be drawn from {sorted(valid_solvers)}.")

    max_steps = args.max_steps
    alphas = list(args.alphas)
    if args.fast:
        max_steps = min(max_steps, 700)
        if len(alphas) > 6:
            desired = np.array([0.08, 0.20, 0.50, 1.50, 5.00])
            selected = []
            for value in desired:
                selected.append(min(alphas, key=lambda x: abs(math.log(max(x, 1e-6)) - math.log(value))))
            alphas = sorted(set(selected))

    base_config = SimulationConfig(
        dt=args.dt,
        max_steps=max_steps,
        max_acc=args.max_acc,
        max_speed=args.max_speed,
        r_contingency=args.r_contingency,
        active_zone=args.active_zone,
        failure_time=args.failure_time,
        blocked_zone=args.blocked_zone,
        deterministic_seed=args.seed,
    )
    primary_alpha = min(alphas, key=lambda value: abs(value - 0.5))

    print(f"[1/8] Initializing the mandatory Poisson and CBF safety boxes for grid {grid.shape}...")
    assembly_time = 0.0  # Matrix assembly is performed internally by PoissonSafetyBox.compute().

    print(f"[2/8] Solving {len(forcing_methods)} forcing fields through PoissonSafetyBox ({args.solver})...")
    poisson_results: dict[str, PoissonResult] = {}
    for method in forcing_methods:
        result = solve_poisson(
            world,
            method,
            args.solver,
            tolerance=args.solver_tolerance,
            max_iterations=args.solver_max_iterations,
        )
        poisson_results[method] = result
        print(
            f"  {method:>12s}: {result.solve_time_s:8.3f} s, residual={result.relative_residual:.2e}, "
            f"info={result.solver_info}"
        )

    fixed_poisson = poisson_results[args.fixed_forcing]
    primary_config = SimulationConfig(**{**asdict(base_config), "hocbf_alpha": primary_alpha})
    reach_model = build_reachability_model(world, fixed_poisson, primary_config)
    initial_certificates = evaluate_all_certificates(
        world,
        fixed_poisson,
        reach_model,
        world.start,
        np.zeros(3),
        0.0,
        0.0,
        0.0,
        args.blocked_zone,
        args.failure_time,
    )
    initial_rho = np.asarray([item["rho"] for item in initial_certificates])
    initial_count = int(np.count_nonzero([item["rho"] >= 0.0 and item.get("available", True) for item in initial_certificates]))
    if initial_count < args.r_contingency:
        warnings.warn(
            f"Initial contingency infeasible: only {initial_count} zones reachable, required {args.r_contingency}.",
            RuntimeWarning,
        )

    print(f"[3/8] Running primary failure/switch simulation (alpha={primary_alpha:g})...")
    primary = simulate(world, fixed_poisson, reach_model, primary_config, require_failure_switch=True)
    print(
        f"  outcome={primary.terminated_reason}, final=LZ{primary.final_zone}, switches={len(primary.switch_events)}, "
        f"min pivot={primary.metrics['minimum_contingency_margin']:.3f}"
    )

    print(f"[4/8] Running HOCBF alpha sweep ({len(alphas)} cases)...")
    alpha_results: list[SimulationResult] = []
    for alpha in alphas:
        config = SimulationConfig(**{**asdict(base_config), "hocbf_alpha": float(alpha)})
        alpha_results.append(simulate(world, fixed_poisson, reach_model, config, require_failure_switch=True))
    alpha_df = pd.DataFrame([{"alpha": result.hocbf_alpha, **result.metrics} for result in alpha_results])
    alpha_df.to_csv(output_dir / "alpha_sweep_metrics.csv", index=False)

    print(f"[5/8] Running forcing comparison ({len(forcing_methods)} cases)...")
    forcing_simulations: list[SimulationResult] = []
    for method in forcing_methods:
        model = build_reachability_model(world, poisson_results[method], primary_config)
        forcing_simulations.append(simulate(world, poisson_results[method], model, primary_config, require_failure_switch=True))
    forcing_df = pd.DataFrame(
        [
            {
                "forcing": result.forcing_method,
                "solver": result.solver,
                "poisson_solve_time_s": poisson_results[result.forcing_method].solve_time_s,
                "poisson_relative_residual": poisson_results[result.forcing_method].relative_residual,
                **result.metrics,
            }
            for result in forcing_simulations
        ]
    )
    forcing_df.to_csv(output_dir / "forcing_comparison_metrics.csv", index=False)

    print(f"[6/8] Running solver × forcing study ({len(solvers) * len(forcing_methods)} PDE solves and rollouts)...")
    solver_rows: list[dict[str, Any]] = []
    direct_references: dict[str, PoissonResult] = {}
    for method in forcing_methods:
        if args.solver == "sparse_direct":
            direct_references[method] = poisson_results[method]
        else:
            direct_references[method] = solve_poisson(
                world,
                method,
                "sparse_direct",
                tolerance=args.solver_tolerance,
                max_iterations=args.solver_max_iterations,
            )

    for solver_name in solvers:
        for method in forcing_methods:
            if solver_name == args.solver:
                pde_result = poisson_results[method]
            elif solver_name == "sparse_direct":
                pde_result = direct_references[method]
            else:
                pde_result = solve_poisson(
                    world,
                    method,
                    solver_name,
                    tolerance=args.solver_tolerance,
                    max_iterations=args.solver_max_iterations,
                )
            reference = direct_references[method]
            numerator = np.linalg.norm(pde_result.h_raw[world.unknown_mask] - reference.h_raw[world.unknown_mask])
            denominator = max(np.linalg.norm(reference.h_raw[world.unknown_mask]), 1e-14)
            field_error = float(numerator / denominator)
            model = build_reachability_model(world, pde_result, primary_config)
            sim = simulate(world, pde_result, model, primary_config, require_failure_switch=True)
            solver_rows.append(
                {
                    "solver": solver_name,
                    "forcing": method,
                    "solve_time_s": pde_result.solve_time_s,
                    "relative_residual": pde_result.relative_residual,
                    "relative_field_error": field_error,
                    "iterations": pde_result.iterations,
                    "solver_info": pde_result.solver_info,
                    "contingency_outcome_score": outcome_score(sim),
                    "maintained_r_out_of_p": sim.metrics["maintained_r_out_of_p"],
                    "blocked_switch_occurred": sim.metrics["blocked_switch_occurred"],
                    "landed": sim.landed,
                    "collided": sim.collided,
                    "minimum_contingency_margin": sim.metrics["minimum_contingency_margin"],
                    "minimum_h_cbf": sim.metrics["minimum_h_cbf"],
                    "final_zone": sim.final_zone,
                }
            )
    solver_df = pd.DataFrame(solver_rows)
    solver_df.to_csv(output_dir / "solver_metrics.csv", index=False)

    print("[7/8] Saving figures and data products...")
    plot_workflow(output_dir, args.dpi, not args.no_pdf)
    plot_world_3d(world, args.blocked_zone, output_dir, args.dpi, not args.no_pdf)
    plot_occupancy_boundary_slices(world, output_dir, args.dpi, not args.no_pdf)
    plot_poisson_forcing_gradient(world, poisson_results, forcing_methods, output_dir, args.dpi, not args.no_pdf)
    plot_poisson_isosurfaces(world, poisson_results, forcing_methods, output_dir, args.dpi, not args.no_pdf, args.fast)
    rho_maps = plot_reachability_fields(world, fixed_poisson, reach_model, output_dir, args.dpi, not args.no_pdf)
    plot_contingency_margin_map(world, rho_maps, args.r_contingency, output_dir, args.dpi, not args.no_pdf)
    plot_trajectory_switching_3d(world, primary, args.blocked_zone, output_dir, args.dpi, not args.no_pdf)
    plot_time_histories(primary, args.r_contingency, args.failure_time, output_dir, args.dpi, not args.no_pdf)
    plot_alpha_sweep(world, alpha_results, output_dir, args.dpi, not args.no_pdf)
    plot_forcing_comparison(world, forcing_simulations, poisson_results, output_dir, args.dpi, not args.no_pdf)
    plot_solver_heatmaps(solver_df, solvers, forcing_methods, output_dir, args.dpi, not args.no_pdf)
    plot_dashboard(world, primary, alpha_df, forcing_df, solver_df, args.r_contingency, output_dir, args.dpi, not args.no_pdf)

    trajectory_df = simulation_to_dataframe(primary)
    trajectory_df.to_csv(output_dir / "contingency_metrics.csv", index=False)
    save_trajectory_npz(output_dir / "trajectory_data.npz", primary)

    validation = {
        "imports_successful": True,
        "poisson_safety_box_executed": isinstance(fixed_poisson.box_result, PoissonBoxResult),
        "cbf_safety_box_executed": bool(primary.safe_acceleration.shape[0] > 0 and np.all(np.isfinite(primary.hocbf_residual))),
        "occupancy_nonempty": bool(np.count_nonzero(world.occupancy) > 0),
        "free_space_nonempty": bool(np.count_nonzero(world.free_mask) > 0),
        "start_not_occupied": bool(not occupancy_at_point(world, world.start)),
        "at_least_one_landing_zone_poisson_feasible": bool(
            any(float(sample_trilinear(fixed_poisson.h_cbf, zone.position, world.grid)) >= -0.01 for zone in world.landing_zones)
        ),
        "initial_reachable_count": initial_count,
        "initial_r_out_of_p_feasible": bool(initial_count >= args.r_contingency),
        "blocked_primary_forced_switch": bool(primary.metrics["blocked_switch_occurred"]),
        "primary_maintained_r_out_of_p": bool(primary.metrics["maintained_r_out_of_p"]),
        "primary_no_collision": bool(not primary.collided),
        "primary_landed": bool(primary.landed),
        "required_png_figures_exist": bool(
            all((output_dir / f"fig{index:02d}_{name}.png").exists() for index, name in [
                (0, "contingency_workflow"),
                (1, "world_with_landing_zones_3d"),
                (2, "occupancy_boundary_slices"),
                (3, "poisson_h_gradient_by_forcing"),
                (4, "poisson_3d_isosurfaces"),
                (5, "landing_zone_reachability_fields"),
                (6, "contingency_margin_map"),
                (7, "contingency_trajectory_switching_3d"),
                (8, "contingency_time_histories"),
                (9, "alpha_sweep_contingency"),
                (10, "forcing_contingency_comparison"),
                (11, "solver_contingency_heatmaps"),
                (12, "integrated_dashboard"),
            ])
        ),
    }

    summary = {
        "study": "Poisson-CBF contingency-aware Mars-analog landing",
        "scope": {
            "offline": True,
            "reduced_order": True,
            "poisson_cbf_hocbf": True,
            "contingency_aware_landing_prototype": True,
            "px4_validated": False,
            "rate_controller_validated": False,
            "full_hj_reachability": False,
        },
        "paper_exact_components": {
            "rth_largest_pivot_definition": True,
            "p_smooth_combinatorial_cbf_constraint_structure": True,
            "single_nonnegative_auxiliary_omega": True,
            "active_target_clf_plus_p_contingency_constraints": True,
        },
        "offline_approximations": {
            "landing_certificate": "smooth quadratic-CLF + Poisson-weighted geodesic + local Poisson-risk + kinetic stopping proxy",
            "resource_shrinking": "affine time/energy budget with smooth blocked-site capacity collapse",
            "hj_value_functions": "not implemented",
            "certificate_regularization": "finite-difference and Gaussian-smoothed map derivatives",
        },
        "configuration": {
            "arguments": vars(args),
            "simulation": asdict(primary_config),
            "world": world.diagnostics,
            "poisson_matrix_assembly_time_s": assembly_time,
            "poisson_unknowns": int(fixed_poisson.box_result.solver_info.get("unknowns", np.count_nonzero(fixed_poisson.box_result.solve_mask))),
            "poisson_nnz": int(fixed_poisson.box_result.solver_info.get("nnz", 0)),
            "poisson_box_used": True,
            "cbf_box_used": True,
            "initial_rho": initial_rho,
        },
        "primary_metrics": primary.metrics,
        "switch_events": primary.switch_events,
        "blocked_events": primary.blocked_events,
        "validation": validation,
        "warnings": [],
    }
    if not validation["initial_r_out_of_p_feasible"]:
        summary["warnings"].append("Initial r-out-of-p contingency is infeasible.")
    if not validation["blocked_primary_forced_switch"]:
        summary["warnings"].append("The blocked primary did not produce the required switch event.")
    if not validation["primary_maintained_r_out_of_p"]:
        summary["warnings"].append("The primary rollout lost the r-out-of-p contingency margin.")
    if not validation["primary_no_collision"]:
        summary["warnings"].append("The primary rollout collided with occupied space.")
    if float(primary.metrics["max_emergency_slack"]) > 1e-4:
        summary["warnings"].append("The bounded-input QP required a nonzero emergency slack; guarantees are locally compromised.")

    save_json(output_dir / "metrics_summary.json", summary)
    write_explainer(output_dir, world, primary_config, primary, forcing_methods, solvers)
    required_data_products = [
        "metrics_summary.json",
        "trajectory_data.npz",
        "contingency_metrics.csv",
        "solver_metrics.csv",
        "alpha_sweep_metrics.csv",
        "forcing_comparison_metrics.csv",
        "CONTINGENCY_STUDY_EXPLAINER.md",
    ]
    validation["required_data_products_exist"] = bool(
        all((output_dir / filename).exists() for filename in required_data_products)
    )
    summary["validation"] = validation
    save_json(output_dir / "metrics_summary.json", summary)

    print("[8/8] Validation summary")
    for key, value in validation.items():
        print(f"  {key}: {value}")
    if summary["warnings"]:
        for warning in summary["warnings"]:
            print(f"  WARNING: {warning}")
    print(f"Study outputs written to: {output_dir}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Poisson-CBF contingency-aware multi-zone landing study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default="outputs/contingency_poisson_landing")
    parser.add_argument("--grid-shape", type=parse_grid_shape, default=(48, 38, 28))
    parser.add_argument("--r-contingency", type=int, default=2)
    parser.add_argument("--num-landing-zones", type=int, default=4)
    parser.add_argument("--active-zone", type=int, default=0)
    parser.add_argument("--failure-time", type=float, default=18.0)
    parser.add_argument("--blocked-zone", type=int, default=0)
    parser.add_argument(
        "--forcing-methods",
        type=parse_csv_strings,
        default=["constant", "distance", "average_flux", "guidance"],
    )
    parser.add_argument("--fixed-forcing", type=str, default="guidance")
    parser.add_argument(
        "--alphas",
        type=parse_csv_floats,
        default=[0.05, 0.08, 0.12, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0],
    )
    parser.add_argument("--solver", type=str, default="conjugate_gradient")
    parser.add_argument(
        "--solver-sweep-solvers",
        type=parse_csv_strings,
        default=["sparse_direct", "conjugate_gradient", "sor"],
    )
    parser.add_argument("--solver-tolerance", type=float, default=1e-9)
    parser.add_argument("--solver-max-iterations", type=int, default=7000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=850)
    parser.add_argument("--max-acc", type=float, default=1.35)
    parser.add_argument("--max-speed", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=190)
    parser.add_argument("--fast", action="store_true", help="Use a shorter alpha sweep and rollout horizon.")
    parser.add_argument("--no-pdf", action="store_true", help="Save PNG figures only.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    # argparse leaves default list objects as-is and parsed CLI values as lists; normalize.
    args.forcing_methods = list(args.forcing_methods)
    args.solver_sweep_solvers = list(args.solver_sweep_solvers)
    args.alphas = list(args.alphas)
    args.solver = args.solver.lower()
    args.fixed_forcing = args.fixed_forcing.lower()
    try:
        run_study(args)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if os.environ.get("POISSON_CONTINGENCY_DEBUG", "0") == "1":
            raise
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
