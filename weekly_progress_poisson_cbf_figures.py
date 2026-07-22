"""
Weekly Progress Figure Generator: 3D Poisson-CBF Landing Workflow
=================================================================

Self-contained research-figure generator for a weekly progress presentation.
It creates publication/presentation-grade figures showing:

1. 3D obstacle world and sampled occupancy.
2. Occupancy matrix slices and boundary/frontier extraction.
3. Forcing functions, Poisson safety field h, and gradient fields.
4. 3D Poisson isosurfaces / contour surfaces.
5. Alpha sweep trajectories from conservative to aggressive.
6. Forcing-function trajectory comparison.
7. HOCBF residual, h(t), correction effort, and compute-time plots.
8. Solver timing comparison.
9. A compact integrated dashboard.

No ROS, PX4, Gazebo, or project-specific modules are required.

Example:
    python weekly_progress_poisson_cbf_figures.py \
        --output-dir outputs/weekly_progress_figures \
        --grid-shape 48,38,28 \
        --alphas 0.05,0.08,0.12,0.2,0.35,0.5,0.75,1,1.5,2,3,5,8

Outputs:
    <output-dir>/figures/*.png
    <output-dir>/figures/*.pdf
    <output-dir>/data/*.csv
    <output-dir>/run_config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import sparse
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import cg, bicgstab, spsolve
from skimage.measure import marching_cubes


# =============================================================================
# Style
# =============================================================================


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 330,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "mathtext.default": "regular",
        }
    )


def save(fig: plt.Figure, out: Path, name: str, pdf: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", facecolor="white")
    if pdf:
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def safe_norm(v: np.ndarray, axis: int = -1, keepdims: bool = False, eps: float = 1e-12) -> np.ndarray:
    return np.sqrt(np.sum(v * v, axis=axis, keepdims=keepdims) + eps)


# =============================================================================
# Data containers
# =============================================================================


@dataclass
class World:
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
    boundary: np.ndarray
    spheres: List[dict]
    boxes: List[dict]
    cylinders: List[dict]
    start: np.ndarray
    goal: np.ndarray
    landing_zones: List[np.ndarray]


@dataclass
class PoissonResult:
    method: str
    solver: str
    h: np.ndarray
    f: np.ndarray
    grad: Tuple[np.ndarray, np.ndarray, np.ndarray]
    hessian: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    wall_time: float
    residual_norm: float
    unknowns: int


@dataclass
class Rollout:
    alpha: float
    forcing: str
    trajectory: np.ndarray
    nominal: np.ndarray
    h_hist: np.ndarray
    residual_hist: np.ndarray
    filtered_hist: np.ndarray
    correction_hist: np.ndarray
    solve_ms_hist: np.ndarray
    final_distance: float
    path_length: float
    min_h: float
    min_residual: float
    filtered_fraction: float
    mean_correction: float
    max_correction: float
    reached_goal: bool
    collision: bool


# =============================================================================
# Geometry and occupancy
# =============================================================================


def add_sphere(occ: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, center: Sequence[float], radius: float) -> None:
    cx, cy, cz = center
    occ[(X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= radius**2] = True


def add_box(occ: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, center: Sequence[float], size: Sequence[float]) -> None:
    cx, cy, cz = center
    sx, sy, sz = size
    occ[(np.abs(X - cx) <= sx / 2) & (np.abs(Y - cy) <= sy / 2) & (np.abs(Z - cz) <= sz / 2)] = True


def add_cylinder(occ: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, center: Sequence[float], radius: float, height: float) -> None:
    cx, cy, cz = center
    occ[((X - cx) ** 2 + (Y - cy) ** 2 <= radius**2) & (np.abs(Z - cz) <= height / 2)] = True


def make_world(shape: Tuple[int, int, int]) -> World:
    Lx, Ly, Lz = 18.0, 14.0, 10.0
    nx, ny, nz = shape
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    occ = np.zeros((nx, ny, nz), dtype=bool)

    spheres = [
        {"name": "aerial hazard A", "center": (5.2, 4.8, 6.8), "radius": 1.25},
        {"name": "aerial hazard B", "center": (7.4, 5.9, 5.6), "radius": 1.10},
        {"name": "aerial hazard C", "center": (9.7, 7.0, 4.4), "radius": 1.05},
        {"name": "aerial hazard D", "center": (12.2, 8.4, 3.0), "radius": 0.95},
        {"name": "landing hazard", "center": (14.4, 9.7, 1.6), "radius": 0.75},
    ]
    boxes = [
        {"name": "gate wall lower", "center": (3.9, 7.0, 3.0), "size": (0.65, 7.8, 4.2)},
        {"name": "gate wall upper", "center": (3.9, 7.0, 8.6), "size": (0.65, 7.8, 1.8)},
        {"name": "terrain block A", "center": (7.0, 8.7, 2.5), "size": (1.4, 2.1, 3.0)},
        {"name": "terrain block B", "center": (10.5, 4.8, 2.1), "size": (2.0, 1.7, 2.7)},
        {"name": "partial landing block", "center": (15.4, 11.2, 0.8), "size": (1.3, 1.3, 1.6)},
    ]
    cylinders = [
        {"name": "tower 1", "center": (8.5, 10.7, 2.8), "radius": 0.45, "height": 5.6},
        {"name": "tower 2", "center": (11.6, 6.0, 2.2), "radius": 0.50, "height": 4.4},
        {"name": "tower 3", "center": (13.8, 7.2, 1.8), "radius": 0.40, "height": 3.6},
    ]

    for s in spheres:
        add_sphere(occ, X, Y, Z, s["center"], s["radius"])
    for b in boxes:
        add_box(occ, X, Y, Z, b["center"], b["size"])
    for c in cylinders:
        add_cylinder(occ, X, Y, Z, c["center"], c["radius"], c["height"])

    # Ground is not occupied everywhere; keep the solver domain open but mark an outer computational boundary.
    dil = binary_dilation(occ, structure=np.ones((3, 3, 3), dtype=bool))
    boundary = dil & (~occ)
    boundary[0, :, :] = True
    boundary[-1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True

    start = np.array([1.0, 1.1, 8.6])
    goal = np.array([16.7, 12.2, 0.85])
    landing_zones = [goal, np.array([16.2, 2.4, 0.85]), np.array([13.6, 12.5, 0.85])]
    return World(Lx, Ly, Lz, nx, ny, nz, x, y, z, X, Y, Z, dx, dy, dz, occ, boundary, spheres, boxes, cylinders, start, goal, landing_zones)


# =============================================================================
# Forcing functions and Poisson solve
# =============================================================================


def normalized(a: np.ndarray) -> np.ndarray:
    amin = np.nanmin(a)
    amax = np.nanmax(a)
    if not np.isfinite(amin) or not np.isfinite(amax) or (amax - amin) < 1e-12:
        return np.zeros_like(a)
    return (a - amin) / (amax - amin)


def line_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    AB = B - A
    AP = P - A.reshape((1, 1, 1, 3))
    t = np.clip(np.sum(AP * AB.reshape((1, 1, 1, 3)), axis=-1) / (np.dot(AB, AB) + 1e-12), 0, 1)
    closest = A.reshape((1, 1, 1, 3)) + t[..., None] * AB.reshape((1, 1, 1, 3))
    return safe_norm(P - closest, axis=-1)


def compute_forcing(world: World, method: str) -> np.ndarray:
    free = ~world.occupancy
    dist = distance_transform_edt(free, sampling=(world.dx, world.dy, world.dz))
    dist_n = normalized(dist)
    P = np.stack([world.X, world.Y, world.Z], axis=-1)
    d_line = line_distance(P, world.start, world.goal)
    corridor = np.exp(-(d_line**2) / (2 * 3.0**2))
    descent = normalized(world.Lz - world.Z)

    if method == "constant":
        f = np.ones_like(world.X)
    elif method == "distance":
        f = 0.45 + 1.7 * dist_n
    elif method == "average_flux":
        # Stronger curvature near obstacle/frontier cells; useful for testing conservatism.
        f = 0.55 + 1.15 * np.exp(-dist / 1.25) + 0.35 * dist_n
    elif method == "guidance":
        # Raise the safety landscape along a descending, goal-directed corridor while preserving clearance.
        f = 0.35 + 1.65 * corridor * (0.25 + dist_n) + 0.35 * descent
    else:
        raise ValueError(f"Unknown forcing method: {method}")
    f[world.occupancy] = 0.0
    return f.astype(float)


def build_system(world: World, forcing: np.ndarray) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    dirichlet = world.occupancy | world.boundary
    unknown = ~dirichlet
    idx = -np.ones(world.occupancy.shape, dtype=np.int64)
    coords = np.argwhere(unknown)
    for n, (i, j, k) in enumerate(coords):
        idx[i, j, k] = n

    invdx2, invdy2, invdz2 = 1.0 / world.dx**2, 1.0 / world.dy**2, 1.0 / world.dz**2
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    rhs = np.zeros(len(coords), dtype=float)

    for n, (i, j, k) in enumerate(coords):
        diag = 2 * (invdx2 + invdy2 + invdz2)
        rhs[n] = forcing[i, j, k]
        for di, dj, dk, coeff in [
            (-1, 0, 0, invdx2),
            (1, 0, 0, invdx2),
            (0, -1, 0, invdy2),
            (0, 1, 0, invdy2),
            (0, 0, -1, invdz2),
            (0, 0, 1, invdz2),
        ]:
            ni, nj, nk = i + di, j + dj, k + dk
            q = idx[ni, nj, nk]
            if q >= 0:
                rows.append(n)
                cols.append(int(q))
                vals.append(-coeff)
            # Dirichlet value is zero, so no RHS contribution.
        rows.append(n)
        cols.append(n)
        vals.append(diag)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(coords), len(coords)))
    return A, rhs, idx


def solve_poisson(world: World, method: str, solver: str, rtol: float = 1e-5, maxiter: int = 2000) -> PoissonResult:
    f = compute_forcing(world, method)
    A, b, idx = build_system(world, f)
    start = time.perf_counter()
    if solver == "sparse_direct":
        sol = spsolve(A, b)
    elif solver == "conjugate_gradient":
        sol, info = cg(A, b, rtol=rtol, atol=0.0, maxiter=maxiter)
        if info != 0:
            print(f"[warning] CG info={info} for {method}")
    elif solver == "bicgstab":
        sol, info = bicgstab(A, b, rtol=rtol, atol=0.0, maxiter=maxiter)
        if info != 0:
            print(f"[warning] BiCGSTAB info={info} for {method}")
    else:
        raise ValueError(solver)
    wall = time.perf_counter() - start
    h = np.zeros_like(world.X, dtype=float)
    h[idx >= 0] = sol[idx[idx >= 0]]
    # Normalize for comparability across forcing methods.
    h = h / (np.nanmax(h) + 1e-12)
    h[world.occupancy | world.boundary] = 0.0
    residual = np.linalg.norm(A @ sol - b) / (np.linalg.norm(b) + 1e-12)

    hx, hy, hz = np.gradient(h, world.dx, world.dy, world.dz, edge_order=2)
    hxx = np.gradient(hx, world.dx, axis=0, edge_order=2)
    hyy = np.gradient(hy, world.dy, axis=1, edge_order=2)
    hzz = np.gradient(hz, world.dz, axis=2, edge_order=2)
    hxy = np.gradient(hx, world.dy, axis=1, edge_order=2)
    hxz = np.gradient(hx, world.dz, axis=2, edge_order=2)
    hyz = np.gradient(hy, world.dz, axis=2, edge_order=2)
    return PoissonResult(method, solver, h, f, (hx, hy, hz), (hxx, hyy, hzz, hxy, hxz, hyz), wall, residual, A.shape[0])


# =============================================================================
# Rollout and sampling
# =============================================================================


class Sampler:
    def __init__(self, world: World, result: PoissonResult):
        self.world = world
        grid = (world.x, world.y, world.z)
        opts = dict(bounds_error=False, fill_value=0.0)
        self.h = RegularGridInterpolator(grid, result.h, **opts)
        self.g = [RegularGridInterpolator(grid, a, **opts) for a in result.grad]
        self.H = [RegularGridInterpolator(grid, a, **opts) for a in result.hessian]

    def sample(self, p: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        p2 = np.array(p, dtype=float).reshape(1, 3)
        h = float(self.h(p2)[0])
        g = np.array([float(gi(p2)[0]) for gi in self.g])
        hxx, hyy, hzz, hxy, hxz, hyz = [float(hi(p2)[0]) for hi in self.H]
        H = np.array([[hxx, hxy, hxz], [hxy, hyy, hyz], [hxz, hyz, hzz]], dtype=float)
        return h, g, H


def nearest_occupied_collision(world: World, p: np.ndarray) -> bool:
    i = int(np.clip(np.searchsorted(world.x, p[0]), 0, world.nx - 1))
    j = int(np.clip(np.searchsorted(world.y, p[1]), 0, world.ny - 1))
    k = int(np.clip(np.searchsorted(world.z, p[2]), 0, world.nz - 1))
    return bool(world.occupancy[i, j, k])


def hocbf_filter(p: np.ndarray, v: np.ndarray, a_nom: np.ndarray, sampler: Sampler, alpha: float) -> Tuple[np.ndarray, float, float, bool, float]:
    h, grad, Hess = sampler.sample(p)
    alpha1 = alpha
    alpha2 = alpha
    hdot = float(grad @ v)
    curvature = float(v @ Hess @ v)
    # Constraint: grad_h^T a + v^T H v + (a1+a2) hdot + a1*a2 h >= 0
    A = grad
    b = -curvature - (alpha1 + alpha2) * hdot - alpha1 * alpha2 * h
    lhs_nom = float(A @ a_nom)
    residual_nom = lhs_nom - b
    if np.linalg.norm(A) < 1e-9 or residual_nom >= 0:
        return a_nom, h, residual_nom, False, 0.0
    correction = (b - lhs_nom) / (float(A @ A) + 1e-12) * A
    a_safe = a_nom + correction
    residual_safe = float(A @ a_safe - b)
    return a_safe, h, residual_safe, True, float(np.linalg.norm(correction))


def rollout(world: World, result: PoissonResult, alpha: float, dt: float, max_steps: int, max_acc: float, max_speed: float) -> Rollout:
    sampler = Sampler(world, result)
    p = world.start.copy()
    v = np.zeros(3)
    traj = [p.copy()]
    nom_traj = [p.copy()]
    h_hist = []
    res_hist = []
    filt_hist = []
    corr_hist = []
    ms_hist = []
    collision = False

    # Nominal comparison trajectory: same controller with no CBF projection.
    pn = world.start.copy()
    vn = np.zeros(3)

    for _ in range(max_steps):
        to_goal = world.goal - p
        a_nom = 0.95 * to_goal - 1.35 * v
        n = np.linalg.norm(a_nom)
        if n > max_acc:
            a_nom = a_nom / n * max_acc
        tic = time.perf_counter()
        a_safe, hval, residual, filtered, corr = hocbf_filter(p, v, a_nom, sampler, alpha)
        ms_hist.append(1000 * (time.perf_counter() - tic))
        an = np.linalg.norm(a_safe)
        if an > max_acc:
            a_safe = a_safe / an * max_acc
        v = v + dt * a_safe
        sp = np.linalg.norm(v)
        if sp > max_speed:
            v = v / sp * max_speed
        p = p + dt * v
        p = np.clip(p, [0, 0, 0], [world.Lx, world.Ly, world.Lz])

        a_nom_n = 0.95 * (world.goal - pn) - 1.35 * vn
        nn = np.linalg.norm(a_nom_n)
        if nn > max_acc:
            a_nom_n = a_nom_n / nn * max_acc
        vn = vn + dt * a_nom_n
        spn = np.linalg.norm(vn)
        if spn > max_speed:
            vn = vn / spn * max_speed
        pn = np.clip(pn + dt * vn, [0, 0, 0], [world.Lx, world.Ly, world.Lz])

        traj.append(p.copy())
        nom_traj.append(pn.copy())
        h_hist.append(hval)
        res_hist.append(residual)
        filt_hist.append(float(filtered))
        corr_hist.append(corr)
        if nearest_occupied_collision(world, p):
            collision = True
            break
        if np.linalg.norm(p - world.goal) < 0.28 and np.linalg.norm(v) < 0.35:
            break

    T = np.array(traj)
    NT = np.array(nom_traj)
    segs = np.linalg.norm(np.diff(T, axis=0), axis=1)
    final_dist = float(np.linalg.norm(T[-1] - world.goal))
    reached = final_dist < 0.35
    return Rollout(
        alpha=float(alpha),
        forcing=result.method,
        trajectory=T,
        nominal=NT,
        h_hist=np.array(h_hist),
        residual_hist=np.array(res_hist),
        filtered_hist=np.array(filt_hist),
        correction_hist=np.array(corr_hist),
        solve_ms_hist=np.array(ms_hist),
        final_distance=final_dist,
        path_length=float(np.sum(segs)),
        min_h=float(np.min(h_hist) if h_hist else np.nan),
        min_residual=float(np.min(res_hist) if res_hist else np.nan),
        filtered_fraction=float(np.mean(filt_hist) if filt_hist else 0.0),
        mean_correction=float(np.mean(corr_hist) if corr_hist else 0.0),
        max_correction=float(np.max(corr_hist) if corr_hist else 0.0),
        reached_goal=reached,
        collision=collision,
    )


# =============================================================================
# Plot helpers
# =============================================================================


def draw_obstacles(ax, world: World, alpha: float = 0.22) -> None:
    # Spheres
    u = np.linspace(0, 2 * np.pi, 30)
    vv = np.linspace(0, np.pi, 16)
    for s in world.spheres:
        cx, cy, cz = s["center"]
        r = s["radius"]
        xs = cx + r * np.outer(np.cos(u), np.sin(vv))
        ys = cy + r * np.outer(np.sin(u), np.sin(vv))
        zs = cz + r * np.outer(np.ones_like(u), np.cos(vv))
        ax.plot_surface(xs, ys, zs, color="#2978A0", alpha=alpha, linewidth=0, shade=True)
    # Cylinders
    theta = np.linspace(0, 2 * np.pi, 32)
    zline = np.linspace(-0.5, 0.5, 16)
    TH, ZZ = np.meshgrid(theta, zline)
    for c in world.cylinders:
        cx, cy, cz = c["center"]
        r = c["radius"]
        h = c["height"]
        xs = cx + r * np.cos(TH)
        ys = cy + r * np.sin(TH)
        zs = cz + h * ZZ
        ax.plot_surface(xs, ys, zs, color="#4D908E", alpha=alpha, linewidth=0, shade=True)
    # Boxes as wireframes
    for b in world.boxes:
        cx, cy, cz = b["center"]
        sx, sy, sz = b["size"]
        xs = [cx - sx / 2, cx + sx / 2]
        ys = [cy - sy / 2, cy + sy / 2]
        zs = [cz - sz / 2, cz + sz / 2]
        corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
        edges = [(0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7), (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7)]
        for i, j in edges:
            ax.plot(*zip(corners[i], corners[j]), color="#F3722C", alpha=0.8, linewidth=1.2)


def set_3d_axes(ax, world: World, title: str = "") -> None:
    ax.set_xlim(0, world.Lx)
    ax.set_ylim(0, world.Ly)
    ax.set_zlim(0, world.Lz)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=24, azim=-62)
    ax.set_title(title, pad=12, fontweight="bold")
    try:
        ax.set_box_aspect((world.Lx, world.Ly, world.Lz))
    except Exception:
        pass


def add_start_goal(ax, world: World) -> None:
    ax.scatter(*world.start, s=85, marker="o", color="#E76F51", edgecolor="white", linewidth=0.8, label="start")
    ax.scatter(*world.goal, s=125, marker="*", color="#2A9D8F", edgecolor="white", linewidth=0.8, label="selected landing zone")
    for i, lz in enumerate(world.landing_zones[1:], start=2):
        ax.scatter(*lz, s=75, marker="D", color="#8AB17D", edgecolor="white", linewidth=0.7, label=f"backup LZ {i}")


# =============================================================================
# Figures
# =============================================================================


def fig_workflow(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7.8))
    ax.axis("off")
    ax.set_title("Weekly Progress Storyboard: Poisson-CBF Landing Workflow", fontsize=18, fontweight="bold", pad=16)
    boxes = [
        (0.05, 0.65, "3D world", "obstacles, terrain, landing zones"),
        (0.25, 0.65, "Occupancy grid", r"$\mathcal{O}(x,y,z)$"),
        (0.45, 0.65, "Boundary/frontier", r"$\partial\Omega$, Dirichlet set"),
        (0.65, 0.65, "Poisson solve", r"$-\nabla^2 h=f$ in $\Omega$, $h=0$ on $\partial\Omega$"),
        (0.85, 0.65, "Safety sample", r"$h,\nabla h,\nabla^2 h$"),
        (0.25, 0.30, "Nominal control", r"$u_{nom}$ from planner/PID"),
        (0.50, 0.30, "CBF/HOCBF filter", r"$\min\|u-u_{nom}\|^2$ s.t. safety"),
        (0.75, 0.30, "Safe command", r"$u_{safe}$ / Offboard setpoint"),
    ]
    for x, y, title, sub in boxes:
        rect = plt.Rectangle((x - 0.085, y - 0.085), 0.17, 0.14, fc="#F7F9FB", ec="#1D3557", lw=1.3, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x, y + 0.018, title, ha="center", va="center", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color="#1D3557")
        ax.text(x, y - 0.035, sub, ha="center", va="center", transform=ax.transAxes, fontsize=8.7, color="#333333")
    arrows = [
        ((0.135, 0.65), (0.165, 0.65)), ((0.335, 0.65), (0.365, 0.65)), ((0.535, 0.65), (0.565, 0.65)), ((0.735, 0.65), (0.765, 0.65)),
        ((0.85, 0.565), (0.55, 0.385)), ((0.335, 0.30), (0.415, 0.30)), ((0.585, 0.30), (0.665, 0.30)),
    ]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.5, color="#1D3557"))
    ax.text(0.50, 0.49, r"Velocity CBF: $\nabla h(p)^T v \geq -\alpha h(p)$", ha="center", transform=ax.transAxes, fontsize=12, color="#1D3557")
    ax.text(0.50, 0.44, r"Acceleration HOCBF: $\nabla h^T a + v^T\nabla^2h\,v +(\alpha_1+\alpha_2)\nabla h^T v +\alpha_1\alpha_2 h \geq 0$", ha="center", transform=ax.transAxes, fontsize=12, color="#1D3557")
    ax.text(0.05, 0.08, "Presentation logic: show one graph immediately after each mathematical object is introduced.", transform=ax.transAxes, fontsize=11, color="#333333")
    save(fig, out, "fig01_workflow_storyboard")


def fig_world(world: World, out: Path) -> None:
    fig = plt.figure(figsize=(15, 7.8))
    ax1 = fig.add_subplot(121, projection="3d")
    draw_obstacles(ax1, world, alpha=0.28)
    add_start_goal(ax1, world)
    set_3d_axes(ax1, world, "Analytic 3D obstacle world")
    ax1.legend(loc="upper left")

    ax2 = fig.add_subplot(122, projection="3d")
    pts = np.argwhere(world.occupancy)
    # Subsample to avoid visual saturation.
    if len(pts) > 5000:
        pts = pts[np.linspace(0, len(pts) - 1, 5000).astype(int)]
    ax2.scatter(world.x[pts[:, 0]], world.y[pts[:, 1]], world.z[pts[:, 2]], s=4, alpha=0.35, color="#277DA1", label="occupied voxels")
    bpts = np.argwhere(world.boundary)
    if len(bpts) > 5000:
        bpts = bpts[np.linspace(0, len(bpts) - 1, 5000).astype(int)]
    ax2.scatter(world.x[bpts[:, 0]], world.y[bpts[:, 1]], world.z[bpts[:, 2]], s=2, alpha=0.18, color="#F8961E", label="boundary/frontier")
    add_start_goal(ax2, world)
    set_3d_axes(ax2, world, "Sampled occupancy and free-space frontier")
    ax2.legend(loc="upper left")
    fig.suptitle("World model → occupancy representation", fontsize=16, fontweight="bold")
    save(fig, out, "fig02_world_model_and_occupancy_3d")


def fig_slices(world: World, out: Path) -> None:
    z_indices = [int(world.nz * q) for q in [0.15, 0.35, 0.55, 0.75]]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), constrained_layout=True)
    occ_cmap = ListedColormap(["white", "#1F77B4"])
    b_cmap = ListedColormap(["white", "#E76F51"])
    for col, k in enumerate(z_indices):
        ax = axes[0, col]
        ax.imshow(world.occupancy[:, :, k].T, origin="lower", extent=[0, world.Lx, 0, world.Ly], cmap=occ_cmap, interpolation="nearest", alpha=0.95)
        ax.contour(world.X[:, :, k], world.Y[:, :, k], world.boundary[:, :, k].astype(float), levels=[0.5], colors="#E76F51", linewidths=1.0)
        ax.scatter(world.start[0], world.start[1], s=40, color="#F4A261", edgecolor="white")
        ax.scatter(world.goal[0], world.goal[1], s=70, marker="*", color="#2A9D8F", edgecolor="white")
        ax.set_title(f"Occupancy XY slice, z={world.z[k]:.1f} m")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax = axes[1, col]
        ax.imshow(world.boundary[:, :, k].T, origin="lower", extent=[0, world.Lx, 0, world.Ly], cmap=b_cmap, interpolation="nearest", alpha=0.95)
        ax.contour(world.X[:, :, k], world.Y[:, :, k], world.occupancy[:, :, k].astype(float), levels=[0.5], colors="#1F77B4", linewidths=0.8)
        ax.set_title(f"Boundary/frontier mask, z={world.z[k]:.1f} m")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
    fig.suptitle("Occupancy matrix and Dirichlet frontier used by the Poisson solve", fontsize=16, fontweight="bold")
    save(fig, out, "fig03_occupancy_matrix_and_boundary_slices")


def fig_forcing_h_gradient(world: World, results: Dict[str, PoissonResult], out: Path) -> None:
    methods = list(results.keys())
    k = int(world.nz * 0.48)
    fig, axes = plt.subplots(len(methods), 3, figsize=(15, 3.2 * len(methods)), constrained_layout=True)
    if len(methods) == 1:
        axes = axes[None, :]
    for r, method in enumerate(methods):
        res = results[method]
        f = res.f[:, :, k]
        h = res.h[:, :, k]
        hx, hy, _ = [g[:, :, k] for g in res.grad]
        ax = axes[r, 0]
        im = ax.imshow(f.T, origin="lower", extent=[0, world.Lx, 0, world.Ly], cmap="magma", interpolation="bilinear")
        ax.contour(world.X[:, :, k], world.Y[:, :, k], world.occupancy[:, :, k].astype(float), levels=[0.5], colors="white", linewidths=0.8)
        ax.set_title(f"Forcing f: {method}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax = axes[r, 1]
        im = ax.imshow(h.T, origin="lower", extent=[0, world.Lx, 0, world.Ly], cmap="viridis", interpolation="bilinear")
        ax.contour(world.X[:, :, k], world.Y[:, :, k], world.boundary[:, :, k].astype(float), levels=[0.5], colors="white", linewidths=0.7)
        ax.set_title(r"Poisson safety field $h(x,y,z)$")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax = axes[r, 2]
        ax.imshow(h.T, origin="lower", extent=[0, world.Lx, 0, world.Ly], cmap="viridis", interpolation="bilinear", alpha=0.85)
        skip = (slice(None, None, 4), slice(None, None, 4))
        mag = np.sqrt(hx**2 + hy**2)
        ax.quiver(world.X[:, :, k][skip], world.Y[:, :, k][skip], hx[skip], hy[skip], mag[skip], cmap="plasma", angles="xy", scale_units="xy", scale=1.2, width=0.003)
        ax.set_title(r"Gradient field $\nabla h$ on h contours")
        for axx in axes[r, :]:
            axx.scatter(world.start[0], world.start[1], s=35, color="#F4A261", edgecolor="white")
            axx.scatter(world.goal[0], world.goal[1], s=70, marker="*", color="#2A9D8F", edgecolor="white")
            axx.set_xlabel("x [m]"); axx.set_ylabel("y [m]")
            axx.set_aspect("equal")
    fig.suptitle("Forcing design → Poisson safety landscape → gradient direction used by the CBF", fontsize=16, fontweight="bold")
    save(fig, out, "fig04_forcing_poisson_h_gradient_fields")


def fig_laplacian_curvature(world: World, results: Dict[str, PoissonResult], out: Path) -> None:
    methods = list(results.keys())
    k = int(world.nz * 0.48)
    fig, axes = plt.subplots(2, len(methods), figsize=(4.1 * len(methods), 7.3), constrained_layout=True)
    if len(methods) == 1:
        axes = axes[:, None]
    for c, method in enumerate(methods):
        res = results[method]
        hxx, hyy, hzz, _, _, _ = res.hessian
        lap = hxx + hyy + hzz
        gradmag = np.sqrt(sum(g**2 for g in res.grad))
        ax = axes[0, c]
        im = ax.imshow(lap[:, :, k].T, origin="lower", extent=[0, world.Lx, 0, world.Ly], cmap="coolwarm", interpolation="bilinear")
        ax.set_title(f"Laplacian / curvature\n{method}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax = axes[1, c]
        im = ax.imshow(gradmag[:, :, k].T, origin="lower", extent=[0, world.Lx, 0, world.Ly], cmap="inferno", interpolation="bilinear")
        ax.set_title(r"Gradient magnitude $\|\nabla h\|$")
        fig.colorbar(im, ax=ax, fraction=0.046)
        for axx in axes[:, c]:
            axx.set_xlabel("x [m]"); axx.set_ylabel("y [m]")
            axx.set_aspect("equal")
    fig.suptitle("Derivative diagnostics: what the CBF/HOCBF actually sees", fontsize=16, fontweight="bold")
    save(fig, out, "fig05_laplacian_gradient_magnitude_comparison")


def fig_poisson_3d(world: World, results: Dict[str, PoissonResult], out: Path) -> None:
    methods = list(results.keys())
    fig = plt.figure(figsize=(4.3 * len(methods), 6.8))
    for i, method in enumerate(methods, start=1):
        ax = fig.add_subplot(1, len(methods), i, projection="3d")
        res = results[method]
        h = res.h.copy()
        # Smooth enough for marching cubes; ensure levels inside range.
        for level, color, alpha in [(0.18, "#90BE6D", 0.18), (0.38, "#43AA8B", 0.22), (0.62, "#577590", 0.24)]:
            try:
                verts, faces, _, _ = marching_cubes(h, level=level, spacing=(world.dx, world.dy, world.dz))
                mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidths=0.0)
                mesh.set_facecolor(color)
                ax.add_collection3d(mesh)
            except Exception:
                pass
        draw_obstacles(ax, world, alpha=0.18)
        add_start_goal(ax, world)
        set_3d_axes(ax, world, f"3D h isosurfaces: {method}")
    fig.suptitle("Three-dimensional Poisson safety field geometry", fontsize=16, fontweight="bold")
    save(fig, out, "fig06_poisson_3d_isosurfaces_by_forcing")


def fig_alpha_trajectories(world: World, rolls: List[Rollout], out: Path) -> None:
    alphas = np.array([r.alpha for r in rolls])
    norm = Normalize(vmin=alphas.min(), vmax=alphas.max())
    cmap = plt.get_cmap("turbo")
    fig = plt.figure(figsize=(16, 8.2))
    ax3 = fig.add_subplot(121, projection="3d")
    draw_obstacles(ax3, world, alpha=0.16)
    for r in rolls:
        color = cmap(norm(r.alpha))
        ax3.plot(r.trajectory[:, 0], r.trajectory[:, 1], r.trajectory[:, 2], color=color, linewidth=2.0, alpha=0.92)
    add_start_goal(ax3, world)
    set_3d_axes(ax3, world, "Alpha sweep: 3D trajectories")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax3, fraction=0.03, pad=0.05)
    cbar.set_label(r"CBF/HOCBF gain $\alpha$")

    ax2 = fig.add_subplot(122)
    for r in rolls:
        ax2.plot(r.trajectory[:, 0], r.trajectory[:, 2], color=cmap(norm(r.alpha)), linewidth=2.1, alpha=0.9, label=f"{r.alpha:g}")
    for b in world.boxes:
        cx, cy, cz = b["center"]; sx, sy, sz = b["size"]
        ax2.add_patch(plt.Rectangle((cx - sx / 2, cz - sz / 2), sx, sz, color="#F3722C", alpha=0.12))
    ax2.scatter(world.start[0], world.start[2], s=80, color="#E76F51", edgecolor="white", label="start")
    ax2.scatter(world.goal[0], world.goal[2], s=120, marker="*", color="#2A9D8F", edgecolor="white", label="landing")
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("z [m]")
    ax2.set_title("Vertical descent profile: conservative → aggressive")
    ax2.set_xlim(0, world.Lx); ax2.set_ylim(0, world.Lz)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:2], labels[:2], loc="upper right")
    fig.suptitle("CBF gain sensitivity: all alpha trajectories in a single comparison", fontsize=16, fontweight="bold")
    save(fig, out, "fig07_alpha_sweep_all_trajectories")


def fig_alpha_metrics(rolls: List[Rollout], out: Path) -> None:
    alpha = np.array([r.alpha for r in rolls])
    metrics = {
        "final distance [m]": np.array([r.final_distance for r in rolls]),
        "path length [m]": np.array([r.path_length for r in rolls]),
        "minimum h": np.array([r.min_h for r in rolls]),
        "filtered fraction": np.array([r.filtered_fraction for r in rolls]),
        "mean correction": np.array([r.mean_correction for r in rolls]),
        "mean CBF solve [ms]": np.array([np.mean(r.solve_ms_hist) if len(r.solve_ms_hist) else 0 for r in rolls]),
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (name, vals) in zip(axes.flat, metrics.items()):
        ax.plot(alpha, vals, marker="o", linewidth=2.2)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(name)
        ax.set_title(name)
        if "minimum h" in name:
            ax.axhline(0.0, color="black", linewidth=1.1, linestyle="--")
    fig.suptitle("Quantitative alpha sweep: conservatism, goal convergence, and intervention cost", fontsize=16, fontweight="bold")
    save(fig, out, "fig08_alpha_sweep_metrics")


def fig_time_histories(rolls: List[Rollout], out: Path) -> None:
    # Pick representative low, middle, high alphas.
    picks = [rolls[0], rolls[len(rolls) // 2], rolls[-1]]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False, constrained_layout=True)
    for r in picks:
        t = np.arange(len(r.h_hist)) * 0.05
        axes[0].plot(t, r.h_hist, label=fr"$\alpha={r.alpha:g}$")
        axes[1].plot(t, r.residual_hist, label=fr"$\alpha={r.alpha:g}$")
        axes[2].plot(t, r.correction_hist, label=fr"$\alpha={r.alpha:g}$")
    axes[0].set_ylabel("h(t)"); axes[0].set_title("Safety value along trajectory")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("HOCBF residual"); axes[1].set_title("Constraint residual: positive means satisfied")
    axes[2].set_ylabel(r"$\|u_{safe}-u_{nom}\|$"); axes[2].set_xlabel("time [s]"); axes[2].set_title("Safety filter intervention")
    for ax in axes:
        ax.legend(loc="best")
    fig.suptitle("Representative time histories used to explain the HOCBF inequality", fontsize=16, fontweight="bold")
    save(fig, out, "fig09_representative_hocbf_time_histories")


def fig_forcing_trajectories(world: World, rolls: Dict[str, Rollout], out: Path) -> None:
    colors = {"constant": "#577590", "distance": "#43AA8B", "average_flux": "#F8961E", "guidance": "#D62828"}
    fig = plt.figure(figsize=(15, 7.6))
    ax = fig.add_subplot(121, projection="3d")
    draw_obstacles(ax, world, alpha=0.17)
    for method, r in rolls.items():
        ax.plot(r.trajectory[:, 0], r.trajectory[:, 1], r.trajectory[:, 2], label=method, color=colors.get(method), linewidth=2.4)
    add_start_goal(ax, world)
    set_3d_axes(ax, world, "Forcing-function comparison: 3D paths")
    ax.legend(loc="upper left")

    ax2 = fig.add_subplot(122)
    for method, r in rolls.items():
        ax2.plot(r.trajectory[:, 0], r.trajectory[:, 1], label=method, color=colors.get(method), linewidth=2.5)
    ax2.scatter(world.start[0], world.start[1], s=80, color="#E76F51", edgecolor="white")
    ax2.scatter(world.goal[0], world.goal[1], s=120, marker="*", color="#2A9D8F", edgecolor="white")
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    ax2.set_xlim(0, world.Lx); ax2.set_ylim(0, world.Ly); ax2.set_aspect("equal")
    ax2.set_title("Top-view path deformation caused by forcing choice")
    ax2.legend()
    fig.suptitle("Forcing function changes the shape of h, therefore the CBF correction direction", fontsize=16, fontweight="bold")
    save(fig, out, "fig10_forcing_function_trajectory_comparison")


def fig_forcing_metrics(results: Dict[str, PoissonResult], rolls: Dict[str, Rollout], out: Path) -> None:
    methods = list(results.keys())
    x = np.arange(len(methods))
    data = {
        "Poisson solve time [s]": [results[m].wall_time for m in methods],
        "final distance [m]": [rolls[m].final_distance for m in methods],
        "min h": [rolls[m].min_h for m in methods],
        "filtered fraction": [rolls[m].filtered_fraction for m in methods],
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), constrained_layout=True)
    for ax, (name, vals) in zip(axes.flat, data.items()):
        ax.bar(x, vals, width=0.65)
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel(name); ax.set_title(name)
        if name == "min h":
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
    fig.suptitle("Forcing-function comparison: geometry, safety, and computational cost", fontsize=16, fontweight="bold")
    save(fig, out, "fig11_forcing_function_metrics")


def fig_solver_timing(cases: List[PoissonResult], out: Path) -> None:
    labels = [c.solver for c in cases]
    times = [c.wall_time for c in cases]
    residuals = [c.residual_norm for c in cases]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    axes[0].bar(labels, times)
    axes[0].set_ylabel("wall time [s]")
    axes[0].set_title("Poisson field construction time")
    axes[1].bar(labels, residuals)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("relative residual")
    axes[1].set_title("Linear-system residual")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Solver timing: offline field construction versus real-time safety filtering", fontsize=16, fontweight="bold")
    save(fig, out, "fig12_solver_timing_comparison")



# =============================================================================
# Extended solver-comparison figures
# =============================================================================


def parse_solver_list(s: str) -> List[str]:
    """Parse a comma-separated solver list."""
    allowed = {"sparse_direct", "conjugate_gradient", "bicgstab"}
    solvers = [p.strip() for p in s.split(",") if p.strip()]
    bad = [p for p in solvers if p not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"Unknown solvers: {bad}. Allowed: {sorted(allowed)}")
    return solvers


def parse_grid_shape_list(s: str) -> List[Tuple[int, int, int]]:
    """Parse '24,20,14;32,26,18;40,32,22' into a list of 3D shapes."""
    shapes: List[Tuple[int, int, int]] = []
    for block in s.split(";"):
        block = block.strip()
        if not block:
            continue
        parts = [int(p.strip()) for p in block.split(",")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("Each solver grid shape must be nx,ny,nz")
        shapes.append((parts[0], parts[1], parts[2]))
    if not shapes:
        raise argparse.ArgumentTypeError("At least one grid shape is required")
    return shapes


def _unknown_count(world: World) -> int:
    """Number of free non-Dirichlet unknowns in the Poisson linear system."""
    return int(np.sum(~(world.occupancy | world.boundary)))


def solve_poisson_safe(world: World, forcing: str, solver: str, max_direct_unknowns: int) -> Tuple[dict, PoissonResult | None]:
    """Solve Poisson robustly and return a metadata record plus optional result."""
    unknowns = _unknown_count(world)
    record = {
        "grid_shape": f"{world.nx}x{world.ny}x{world.nz}",
        "nx": world.nx,
        "ny": world.ny,
        "nz": world.nz,
        "unknowns": unknowns,
        "forcing": forcing,
        "solver": solver,
        "status": "ok",
        "wall_time": np.nan,
        "residual_norm": np.nan,
        "h_l2_to_reference": np.nan,
        "h_linf_to_reference": np.nan,
        "speedup_vs_direct": np.nan,
        "error": "",
    }
    if solver == "sparse_direct" and unknowns > max_direct_unknowns:
        record["status"] = "skipped_direct_too_large"
        record["error"] = f"unknowns={unknowns} > max_direct_unknowns={max_direct_unknowns}"
        return record, None
    try:
        result = solve_poisson(world, forcing, solver)
        record["wall_time"] = float(result.wall_time)
        record["residual_norm"] = float(result.residual_norm)
        record["unknowns"] = int(result.unknowns)
        return record, result
    except Exception as exc:  # pragma: no cover - diagnostic path
        record["status"] = "failed"
        record["error"] = repr(exc)
        return record, None


def run_solver_deep_sweep(
    grid_shapes: List[Tuple[int, int, int]],
    forcing_methods: List[str],
    solvers: List[str],
    max_direct_unknowns: int,
) -> Tuple[List[dict], Dict[Tuple[str, str, str], PoissonResult]]:
    """
    Run a solver sweep over grid resolution, forcing function, and linear solver.

    The goal is not to make sparse_direct look bad or good. The goal is to expose
    the computational structure of the Poisson step: direct factorization can be a
    reliable reference, while iterative methods are usually more realistic for
    repeated field construction and warm-started/local updates.
    """
    records: List[dict] = []
    results: Dict[Tuple[str, str, str], PoissonResult] = {}
    by_problem: Dict[Tuple[str, str], List[Tuple[dict, PoissonResult | None]]] = {}

    for shape in grid_shapes:
        world_s = make_world(shape)
        shape_key = f"{shape[0]}x{shape[1]}x{shape[2]}"
        for forcing in forcing_methods:
            problem_key = (shape_key, forcing)
            by_problem[problem_key] = []
            for solver in solvers:
                rec, res = solve_poisson_safe(world_s, forcing, solver, max_direct_unknowns)
                records.append(rec)
                by_problem[problem_key].append((rec, res))
                if res is not None:
                    results[(shape_key, forcing, solver)] = res

    # Compute solver-to-reference field discrepancies. Prefer sparse direct when available;
    # otherwise use the lowest residual successful solution as the reference.
    for problem_key, entries in by_problem.items():
        ok_entries = [(rec, res) for rec, res in entries if res is not None and rec["status"] == "ok"]
        if not ok_entries:
            continue
        direct = [(rec, res) for rec, res in ok_entries if rec["solver"] == "sparse_direct"]
        if direct:
            ref_rec, ref = direct[0]
        else:
            ref_rec, ref = sorted(ok_entries, key=lambda rr: rr[0].get("residual_norm", np.inf))[0]
        direct_time = float(ref_rec["wall_time"]) if ref_rec["solver"] == "sparse_direct" else np.nan
        denom = np.linalg.norm(ref.h.ravel()) + 1e-12
        for rec, res in ok_entries:
            err = np.asarray(res.h - ref.h)
            rec["h_l2_to_reference"] = float(np.linalg.norm(err.ravel()) / denom)
            rec["h_linf_to_reference"] = float(np.max(np.abs(err)))
            if np.isfinite(direct_time) and rec["wall_time"] > 0:
                rec["speedup_vs_direct"] = float(direct_time / rec["wall_time"])

    return records, results


def _solver_matrix(records: List[dict], grid_shape: str, value_key: str, solvers: List[str], forcings: List[str]) -> np.ndarray:
    M = np.full((len(solvers), len(forcings)), np.nan)
    for i, solver in enumerate(solvers):
        for j, forcing in enumerate(forcings):
            vals = [r[value_key] for r in records if r["grid_shape"] == grid_shape and r["solver"] == solver and r["forcing"] == forcing and r["status"] == "ok"]
            if vals:
                M[i, j] = float(vals[0])
    return M


def fig_solver_forcing_heatmaps(records: List[dict], solvers: List[str], forcings: List[str], out: Path) -> None:
    """Heatmaps of solver wall time and residual across forcing functions."""
    if not records:
        return
    ok = [r for r in records if r["status"] == "ok"]
    if not ok:
        return
    # Use the largest grid shape available among successful records.
    grid_shape = sorted(set(r["grid_shape"] for r in ok), key=lambda s: np.prod([int(x) for x in s.split("x")]))[-1]
    T = _solver_matrix(records, grid_shape, "wall_time", solvers, forcings)
    R = _solver_matrix(records, grid_shape, "residual_norm", solvers, forcings)
    E = _solver_matrix(records, grid_shape, "h_l2_to_reference", solvers, forcings)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.3), constrained_layout=True)
    matrices = [np.log10(T), np.log10(R), np.log10(E + 1e-16)]
    titles = ["log10 wall time [s]", "log10 relative residual", "log10 relative field error"]
    raw_matrices = [T, R, E]
    for panel_idx, (ax, M, Raw, title) in enumerate(zip(axes, matrices, raw_matrices, titles)):
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(forcings)))
        ax.set_xticklabels(forcings, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(solvers)))
        ax.set_yticklabels(solvers)
        ax.set_title(title)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    raw = Raw[i, j]
                    if panel_idx == 0:
                        label = f"{raw:.3g}s"
                    else:
                        label = f"{raw:.1e}"
                    ax.text(j, i, label, ha="center", va="center", color="white", fontsize=8, fontweight="bold")
                else:
                    ax.text(j, i, "skip", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Solver × forcing comparison at grid {grid_shape}: speed, residual, and field consistency", fontsize=16, fontweight="bold")
    save(fig, out, "fig14_solver_forcing_heatmaps")


def fig_solver_scaling(records: List[dict], solvers: List[str], fixed_forcing: str, out: Path) -> None:
    """Wall-time scaling with grid size and unknown count."""
    ok = [r for r in records if r["status"] == "ok" and r["forcing"] == fixed_forcing]
    if not ok:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), constrained_layout=True)
    for solver in solvers:
        vals = sorted([r for r in ok if r["solver"] == solver], key=lambda r: r["unknowns"])
        if not vals:
            continue
        unknowns = np.array([r["unknowns"] for r in vals], dtype=float)
        times = np.array([r["wall_time"] for r in vals], dtype=float)
        residuals = np.array([r["residual_norm"] for r in vals], dtype=float)
        axes[0].plot(unknowns, times, marker="o", label=solver)
        axes[1].plot(unknowns, residuals, marker="o", label=solver)
        axes[2].plot(unknowns, times / np.maximum(unknowns, 1), marker="o", label=solver)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Poisson unknowns"); axes[0].set_ylabel("wall time [s]"); axes[0].set_title("Scaling of field construction")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Poisson unknowns"); axes[1].set_ylabel("relative residual"); axes[1].set_title("Residual versus grid size")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("Poisson unknowns"); axes[2].set_ylabel("time / unknown [s]"); axes[2].set_title("Cost normalized by problem size")
    for ax in axes:
        ax.legend()
    fig.suptitle(f"Solver scaling study for forcing='{fixed_forcing}'", fontsize=16, fontweight="bold")
    save(fig, out, "fig15_solver_scaling_with_grid_size")


def fig_solver_accuracy_tradeoff(records: List[dict], solvers: List[str], forcings: List[str], out: Path) -> None:
    """Tradeoff plot: computational time versus field error/residual."""
    ok = [r for r in records if r["status"] == "ok"]
    if not ok:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), constrained_layout=True)
    markers = {"sparse_direct": "o", "conjugate_gradient": "s", "bicgstab": "^"}
    for solver in solvers:
        vals = [r for r in ok if r["solver"] == solver]
        if not vals:
            continue
        t = np.array([r["wall_time"] for r in vals], dtype=float)
        e = np.array([r["h_l2_to_reference"] for r in vals], dtype=float)
        res = np.array([r["residual_norm"] for r in vals], dtype=float)
        axes[0].scatter(t, e + 1e-16, s=58, marker=markers.get(solver, "o"), label=solver, alpha=0.85)
        axes[1].scatter(t, res, s=58, marker=markers.get(solver, "o"), label=solver, alpha=0.85)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("wall time [s]"); axes[0].set_ylabel("relative h error vs reference")
    axes[0].set_title("Accuracy–time tradeoff")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("wall time [s]"); axes[1].set_ylabel("linear-system residual")
    axes[1].set_title("Residual–time tradeoff")
    for ax in axes:
        ax.legend()
    fig.suptitle("Solver trade space: fast field construction must still preserve the safety landscape", fontsize=16, fontweight="bold")
    save(fig, out, "fig16_solver_accuracy_time_tradeoff")


def fig_solver_speedup_table(records: List[dict], solvers: List[str], forcings: List[str], out: Path) -> None:
    """Matrix-style summary of speedup relative to sparse direct when a direct reference exists."""
    ok = [r for r in records if r["status"] == "ok" and np.isfinite(r.get("speedup_vs_direct", np.nan))]
    if not ok:
        return
    grid_shape = sorted(set(r["grid_shape"] for r in ok), key=lambda s: np.prod([int(x) for x in s.split("x")]))[-1]
    M = _solver_matrix(records, grid_shape, "speedup_vs_direct", solvers, forcings)
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    im = ax.imshow(np.log10(M), aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(forcings))); ax.set_xticklabels(forcings, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(solvers))); ax.set_yticklabels(solvers)
    ax.set_title(f"Speedup relative to sparse direct at grid {grid_shape} (log10 scale)")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.1f}×", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="log10 speedup")
    fig.suptitle("Practical solver selection: direct solve as reference, iterative solves for repeatability", fontsize=15, fontweight="bold")
    save(fig, out, "fig17_solver_speedup_matrix")


def write_solver_deep_csv(data_dir: Path, records: List[dict]) -> None:
    """Save extended solver sweep diagnostics."""
    if not records:
        return
    keys = [
        "grid_shape", "nx", "ny", "nz", "unknowns", "forcing", "solver", "status",
        "wall_time", "residual_norm", "h_l2_to_reference", "h_linf_to_reference", "speedup_vs_direct", "error",
    ]
    with open(data_dir / "solver_deep_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in keys})


def fig_dashboard(world: World, results: Dict[str, PoissonResult], alpha_rolls: List[Rollout], forcing_rolls: Dict[str, Rollout], solver_cases: List[PoissonResult], out: Path) -> None:
    guidance = results.get("guidance", next(iter(results.values())))
    k = int(world.nz * 0.48)
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1, 1], hspace=0.38, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    draw_obstacles(ax, world, alpha=0.20)
    add_start_goal(ax, world)
    set_3d_axes(ax, world, "3D world")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(world.occupancy[:, :, k].T, origin="lower", extent=[0, world.Lx, 0, world.Ly], cmap=ListedColormap(["white", "#1F77B4"]), interpolation="nearest")
    ax.contour(world.X[:, :, k], world.Y[:, :, k], world.boundary[:, :, k].astype(float), levels=[0.5], colors="#E76F51", linewidths=1.0)
    ax.set_title("Occupancy + boundary"); ax.set_aspect("equal")

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(guidance.h[:, :, k].T, origin="lower", extent=[0, world.Lx, 0, world.Ly], cmap="viridis", interpolation="bilinear")
    ax.set_title("Poisson safety field h")
    ax.set_aspect("equal"); fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 0], projection="3d")
    draw_obstacles(ax, world, alpha=0.12)
    for r in alpha_rolls:
        ax.plot(r.trajectory[:, 0], r.trajectory[:, 1], r.trajectory[:, 2], linewidth=1.8, alpha=0.8)
    set_3d_axes(ax, world, "Alpha trajectories")

    ax = fig.add_subplot(gs[1, 1])
    al = np.array([r.alpha for r in alpha_rolls])
    ax.plot(al, [r.final_distance for r in alpha_rolls], marker="o", label="final distance")
    ax.plot(al, [r.filtered_fraction for r in alpha_rolls], marker="s", label="filtered fraction")
    ax.set_xscale("log"); ax.set_xlabel(r"$\alpha$"); ax.set_title("Alpha metrics"); ax.legend()

    ax = fig.add_subplot(gs[1, 2])
    methods = list(forcing_rolls.keys())
    ax.bar(methods, [forcing_rolls[m].min_h for m in methods])
    ax.tick_params(axis="x", rotation=20); ax.set_title("Forcing: minimum h")

    ax = fig.add_subplot(gs[2, 0])
    hx, hy, _ = [g[:, :, k] for g in guidance.grad]
    h = guidance.h[:, :, k]
    ax.imshow(h.T, origin="lower", extent=[0, world.Lx, 0, world.Ly], cmap="viridis", alpha=0.85)
    skip = (slice(None, None, 5), slice(None, None, 5))
    ax.quiver(world.X[:, :, k][skip], world.Y[:, :, k][skip], hx[skip], hy[skip], color="white", scale_units="xy", scale=1.0, width=0.003)
    ax.set_title(r"$\nabla h$ safety-gradient field"); ax.set_aspect("equal")

    ax = fig.add_subplot(gs[2, 1])
    rep = alpha_rolls[len(alpha_rolls) // 2]
    t = np.arange(len(rep.h_hist)) * 0.05
    ax.plot(t, rep.h_hist, label="h")
    ax2 = ax.twinx()
    ax2.plot(t, rep.correction_hist, color="#D62828", alpha=0.8, label="correction")
    ax.set_title("Representative safety trace"); ax.set_xlabel("time [s]"); ax.set_ylabel("h(t)"); ax2.set_ylabel("correction")

    ax = fig.add_subplot(gs[2, 2])
    ax.bar([c.solver for c in solver_cases], [c.wall_time for c in solver_cases])
    ax.tick_params(axis="x", rotation=20); ax.set_ylabel("s"); ax.set_title("Solver timing")

    fig.suptitle("Integrated weekly progress dashboard: workflow, field synthesis, safety filtering, and comparisons", fontsize=18, fontweight="bold")
    save(fig, out, "fig13_integrated_weekly_progress_dashboard")


# =============================================================================
# Reporting
# =============================================================================


def write_csvs(data_dir: Path, alpha_rolls: List[Rollout], forcing_rolls: Dict[str, Rollout], results: Dict[str, PoissonResult], solver_cases: List[PoissonResult]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "alpha_sweep_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "forcing", "final_distance", "path_length", "min_h", "min_residual", "filtered_fraction", "mean_correction", "max_correction", "reached_goal", "collision"])
        for r in alpha_rolls:
            w.writerow([r.alpha, r.forcing, r.final_distance, r.path_length, r.min_h, r.min_residual, r.filtered_fraction, r.mean_correction, r.max_correction, r.reached_goal, r.collision])
    with open(data_dir / "forcing_comparison_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["forcing", "poisson_time", "poisson_residual", "unknowns", "final_distance", "path_length", "min_h", "filtered_fraction", "mean_correction"])
        for m, r in forcing_rolls.items():
            pr = results[m]
            w.writerow([m, pr.wall_time, pr.residual_norm, pr.unknowns, r.final_distance, r.path_length, r.min_h, r.filtered_fraction, r.mean_correction])
    with open(data_dir / "solver_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solver", "forcing", "wall_time", "residual_norm", "unknowns"])
        for c in solver_cases:
            w.writerow([c.solver, c.method, c.wall_time, c.residual_norm, c.unknowns])


def parse_tuple(s: str) -> Tuple[int, int, int]:
    parts = [int(p.strip()) for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected nx,ny,nz")
    return tuple(parts)  # type: ignore


def parse_float_list(s: str) -> List[float]:
    return [float(p.strip()) for p in s.split(",") if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate weekly-progress figures for the Poisson-CBF landing workflow, including rich solver comparisons."
    )
    parser.add_argument("--output-dir", default="outputs/weekly_progress_poisson_cbf", help="Output directory.")
    parser.add_argument("--grid-shape", type=parse_tuple, default=(48, 38, 28), help="Main grid shape nx,ny,nz for presentation figures.")
    parser.add_argument("--alphas", type=parse_float_list, default=parse_float_list("0.03,0.05,0.08,0.12,0.2,0.35,0.5,0.75,1,1.5,2,3,5,8,12"))
    parser.add_argument("--forcing-methods", default="constant,distance,average_flux,guidance")
    parser.add_argument("--fixed-forcing", default="guidance")
    parser.add_argument("--fixed-alpha", type=float, default=0.5)
    parser.add_argument("--solver", default="conjugate_gradient", choices=["sparse_direct", "conjugate_gradient", "bicgstab"], help="Solver used for main field/trajectory figures.")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=850)
    parser.add_argument("--max-acc", type=float, default=1.35)
    parser.add_argument("--max-speed", type=float, default=2.0)
    parser.add_argument("--no-pdf", action="store_true")

    # Extended solver sweep options. These use smaller grids by default so sparse_direct
    # can be included without dominating weekly-meeting runtime.
    parser.add_argument("--solver-sweep-grid-shapes", type=parse_grid_shape_list,
                        default=parse_grid_shape_list("20,16,12;28,22,16;36,28,20"),
                        help="Semicolon-separated grid shapes for solver scaling, e.g. '20,16,12;28,22,16;36,28,20'.")
    parser.add_argument("--solver-sweep-solvers", type=parse_solver_list,
                        default=parse_solver_list("sparse_direct,conjugate_gradient,bicgstab"),
                        help="Comma-separated solvers for deep solver comparison.")
    parser.add_argument("--solver-sweep-forcings", default="constant,distance,average_flux,guidance",
                        help="Forcing methods used in the solver × forcing heatmaps.")
    parser.add_argument("--max-direct-unknowns", type=int, default=26000,
                        help="Skip sparse_direct above this number of unknowns to avoid very long direct factorizations.")
    parser.add_argument("--skip-deep-solver-sweep", action="store_true",
                        help="Skip extended solver heatmaps/scaling/tradeoff figures.")
    args = parser.parse_args()

    set_style()
    out = Path(args.output_dir)
    fig_dir = out / "figures"
    data_dir = out / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["solver_sweep_grid_shapes"] = [list(s) for s in args.solver_sweep_grid_shapes]
    with open(out / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("[1/8] Building main world")
    world = make_world(args.grid_shape)

    print("[2/8] Solving main Poisson fields")
    methods = [m.strip() for m in args.forcing_methods.split(",") if m.strip()]
    results: Dict[str, PoissonResult] = {}
    for m in methods:
        print(f"  - {m} ({args.solver})")
        results[m] = solve_poisson(world, m, args.solver)

    print("[3/8] Basic solver timing comparison on main world")
    solver_cases: List[PoissonResult] = []
    for s in ["sparse_direct", "conjugate_gradient", "bicgstab"]:
        unknowns = _unknown_count(world)
        if s == "sparse_direct" and unknowns > args.max_direct_unknowns:
            print(f"  - {s}: skipped on main grid; unknowns={unknowns} > {args.max_direct_unknowns}")
            continue
        print(f"  - {s}")
        try:
            solver_cases.append(solve_poisson(world, args.fixed_forcing, s))
        except Exception as exc:
            print(f"[warning] main solver timing failed for {s}: {exc!r}")

    print("[4/8] Rollouts")
    fixed = results[args.fixed_forcing]
    alpha_rolls = [rollout(world, fixed, a, args.dt, args.max_steps, args.max_acc, args.max_speed) for a in args.alphas]
    forcing_rolls = {m: rollout(world, results[m], args.fixed_alpha, args.dt, args.max_steps, args.max_acc, args.max_speed) for m in methods}

    solver_deep_records: List[dict] = []
    if not args.skip_deep_solver_sweep:
        print("[5/8] Deep solver sweep: grid scaling + solver × forcing comparison")
        deep_forcings = [m.strip() for m in args.solver_sweep_forcings.split(",") if m.strip()]
        solver_deep_records, _ = run_solver_deep_sweep(
            grid_shapes=args.solver_sweep_grid_shapes,
            forcing_methods=deep_forcings,
            solvers=args.solver_sweep_solvers,
            max_direct_unknowns=args.max_direct_unknowns,
        )
    else:
        print("[5/8] Deep solver sweep skipped")

    print("[6/8] Figures")
    fig_workflow(fig_dir)
    fig_world(world, fig_dir)
    fig_slices(world, fig_dir)
    fig_forcing_h_gradient(world, results, fig_dir)
    fig_laplacian_curvature(world, results, fig_dir)
    fig_poisson_3d(world, results, fig_dir)
    fig_alpha_trajectories(world, alpha_rolls, fig_dir)
    fig_alpha_metrics(alpha_rolls, fig_dir)
    fig_time_histories(alpha_rolls, fig_dir)
    fig_forcing_trajectories(world, forcing_rolls, fig_dir)
    fig_forcing_metrics(results, forcing_rolls, fig_dir)
    if solver_cases:
        fig_solver_timing(solver_cases, fig_dir)
    if solver_deep_records:
        deep_forcings = [m.strip() for m in args.solver_sweep_forcings.split(",") if m.strip()]
        fig_solver_forcing_heatmaps(solver_deep_records, args.solver_sweep_solvers, deep_forcings, fig_dir)
        fig_solver_scaling(solver_deep_records, args.solver_sweep_solvers, args.fixed_forcing, fig_dir)
        fig_solver_accuracy_tradeoff(solver_deep_records, args.solver_sweep_solvers, deep_forcings, fig_dir)
        fig_solver_speedup_table(solver_deep_records, args.solver_sweep_solvers, deep_forcings, fig_dir)
    fig_dashboard(world, results, alpha_rolls, forcing_rolls, solver_cases, fig_dir)

    print("[7/8] Tables")
    write_csvs(data_dir, alpha_rolls, forcing_rolls, results, solver_cases)
    write_solver_deep_csv(data_dir, solver_deep_records)

    print("[8/8] Done")
    print(f"Figures saved to: {fig_dir}")
    print(f"Data saved to:    {data_dir}")
    print("Recommended weekly-presentation figures:")
    for name in [
        "fig01_workflow_storyboard.png",
        "fig02_world_model_and_occupancy_3d.png",
        "fig03_occupancy_matrix_and_boundary_slices.png",
        "fig04_forcing_poisson_h_gradient_fields.png",
        "fig05_laplacian_gradient_magnitude_comparison.png",
        "fig06_poisson_3d_isosurfaces_by_forcing.png",
        "fig07_alpha_sweep_all_trajectories.png",
        "fig08_alpha_sweep_metrics.png",
        "fig09_representative_hocbf_time_histories.png",
        "fig10_forcing_function_trajectory_comparison.png",
        "fig11_forcing_function_metrics.png",
        "fig12_solver_timing_comparison.png",
        "fig14_solver_forcing_heatmaps.png",
        "fig15_solver_scaling_with_grid_size.png",
        "fig16_solver_accuracy_time_tradeoff.png",
        "fig17_solver_speedup_matrix.png",
        "fig13_integrated_weekly_progress_dashboard.png",
    ]:
        print(f"  - {fig_dir / name}")


if __name__ == "__main__":
    main()