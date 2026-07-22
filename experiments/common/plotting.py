"""Publication-grade figures shared by all experiment modes.

The figure vocabulary is deliberately consistent across online, image-based,
and predefined-world experiments. Every static figure is exported as high-DPI
PNG and vector PDF/SVG, and every axis states its physical coordinates or the
fixed variables used for a state-space slice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from .poisson_field import PoissonField

_EXPORT_PDF = True
_EXPORT_SVG = True


def configure_exports(*, pdf: bool = True, svg: bool = True) -> None:
    """Configure vector exports once per experiment run.

    Raster-heavy figures may still disable SVG locally to avoid enormous
    contour path files. The paper profile exports PNG and PDF by default;
    the smoke profile can disable vector formats for fast validation.
    """

    global _EXPORT_PDF, _EXPORT_SVG
    _EXPORT_PDF = bool(pdf)
    _EXPORT_SVG = bool(svg)



def configure_academic_style() -> None:
    """Apply one restrained style matching the existing high-resolution outputs."""

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 360,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "mathtext.default": "regular",
            "figure.constrained_layout.use": True,
        }
    )


def save_figure(
    figure: plt.Figure,
    directory: str | Path,
    name: str,
    *,
    dpi: int = 360,
    pdf: bool = True,
    svg: bool = True,
) -> None:
    """Export one figure without duplicating plotting logic."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{name}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    if pdf and _EXPORT_PDF:
        figure.savefig(output / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    if svg and _EXPORT_SVG:
        figure.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _draw_box(ax: Any, minimum: Sequence[float], maximum: Sequence[float], alpha: float = 0.20) -> None:
    lo = np.asarray(minimum, dtype=float)
    hi = np.asarray(maximum, dtype=float)
    vertices = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
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
    collection = Poly3DCollection(faces, alpha=alpha, linewidths=0.5)
    ax.add_collection3d(collection)


def _draw_cylinder(
    ax: Any,
    center: Sequence[float],
    radius: float,
    z_range: Sequence[float],
    alpha: float = 0.20,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 40)
    z = np.linspace(float(z_range[0]), float(z_range[1]), 10)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = float(center[0]) + float(radius) * np.cos(theta_grid)
    y_grid = float(center[1]) + float(radius) * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, linewidth=0)


def _draw_ellipsoid(
    ax: Any,
    center: Sequence[float],
    radii: Sequence[float],
    alpha: float = 0.20,
) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 36)
    v = np.linspace(0.0, np.pi, 20)
    c = np.asarray(center, dtype=float)
    r = np.asarray(radii, dtype=float)
    x = c[0] + r[0] * np.outer(np.cos(u), np.sin(v))
    y = c[1] + r[1] * np.outer(np.sin(u), np.sin(v))
    z = c[2] + r[2] * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=alpha, linewidth=0)


def draw_world_obstacles(ax: Any, world: Any, alpha: float = 0.20) -> None:
    """Draw analytic obstacles when metadata are available."""

    for obstacle in getattr(world, "obstacles", ()):
        data = obstacle.parameters
        if obstacle.kind == "box":
            _draw_box(ax, data["minimum"], data["maximum"], alpha)
        elif obstacle.kind == "cylinder":
            _draw_cylinder(ax, data["center"], data["radius"], data["z_range"], alpha)
        elif obstacle.kind == "ellipsoid":
            _draw_ellipsoid(ax, data["center"], data["radii"], alpha)
        elif obstacle.kind == "annular_cylinder":
            _draw_cylinder(ax, data["center"], data["outer_radius"], data["z_range"], alpha * 0.75)


def _target_positions(controller: Any) -> dict[str, np.ndarray]:
    return {
        identifier: target.x_star[: controller.dimension]
        for identifier, target in controller.targets.items()
    }


def plot_world_trajectory_3d(
    *,
    world: Any,
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    controller: Any,
    directory: str | Path,
    dpi: int,
) -> None:
    """Plot the obstacle world, target centers, and filtered 3-D trajectory."""

    configure_academic_style()
    figure = plt.figure(figsize=(11.5, 8.5))
    axis = figure.add_subplot(111, projection="3d")
    draw_world_obstacles(axis, world, alpha=0.22)
    if not metrics.empty:
        axis.plot(metrics["x"], metrics["y"], metrics["z"], label="safe trajectory")
        axis.scatter(metrics["x"].iloc[0], metrics["y"].iloc[0], metrics["z"].iloc[0], s=70, label="start")
        axis.scatter(metrics["x"].iloc[-1], metrics["y"].iloc[-1], metrics["z"].iloc[-1], s=70, marker="x", label="terminal state")
    for identifier, point in _target_positions(controller).items():
        axis.scatter(*point, s=55, marker="o")
        axis.text(point[0], point[1], point[2] + 0.25, identifier)
    if not events.empty:
        switches = events[events["event"] == "active_target_switched"]
        for _, event in switches.iterrows():
            if metrics.empty:
                continue
            index = int(np.argmin(np.abs(metrics["time_s"].to_numpy() - event["time_s"])))
            point = metrics.iloc[index][["x", "y", "z"]].to_numpy(float)
            axis.scatter(*point, s=90, marker="D", label="target switch")
    axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    axis.set_xlim(0.0, world.extent_m[0])
    axis.set_ylim(0.0, world.extent_m[1])
    axis.set_zlim(0.0, world.extent_m[2])
    axis.view_init(elev=27, azim=-63)
    axis.set_title("Predefined 3-D landing scenario and filtered trajectory")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    axis.legend(unique.values(), unique.keys(), loc="upper left")
    save_figure(figure, directory, "world_trajectory_3d", dpi=dpi)


def plot_trajectory_views(
    *,
    metrics: pd.DataFrame,
    controller: Any,
    world: Any,
    directory: str | Path,
    dpi: int,
) -> None:
    """Plot orthogonal trajectory projections with common physical scales."""

    configure_academic_style()
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    views = [("x", "y", "XY view"), ("x", "z", "XZ view"), ("y", "z", "YZ view")]
    for axis, (horizontal, vertical, title) in zip(axes, views, strict=True):
        if not metrics.empty:
            axis.plot(metrics[horizontal], metrics[vertical], label="safe trajectory")
            axis.scatter(metrics[horizontal].iloc[0], metrics[vertical].iloc[0], s=50, label="start")
            axis.scatter(metrics[horizontal].iloc[-1], metrics[vertical].iloc[-1], s=55, marker="x", label="terminal")
        for identifier, point in _target_positions(controller).items():
            coordinate = {"x": point[0], "y": point[1], "z": point[2]}
            axis.scatter(coordinate[horizontal], coordinate[vertical], s=38)
            axis.annotate(identifier, (coordinate[horizontal], coordinate[vertical]), xytext=(4, 4), textcoords="offset points")
        axis.set_xlabel(f"{horizontal} [m]")
        axis.set_ylabel(f"{vertical} [m]")
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")
    axes[0].legend(loc="best")
    figure.suptitle("Orthogonal views of the 3-D landing trajectory")
    save_figure(figure, directory, "trajectory_orthogonal_views", dpi=dpi)


def _axis_coordinates(size: int, spacing: float) -> np.ndarray:
    """Return physical coordinates for one uniform grid axis."""

    return np.arange(int(size), dtype=float) * float(spacing)


def _slice_indices(shape: Sequence[int], count: int = 4) -> list[int]:
    return [int(round(value)) for value in np.linspace(1, shape[-1] - 2, count)]


def plot_occupancy_boundary_slices(
    *,
    field: PoissonField,
    directory: str | Path,
    dpi: int,
) -> None:
    """Show occupancy and Dirichlet boundary masks on representative z slices."""

    if field.dimension != 3:
        return
    configure_academic_style()
    indices = _slice_indices(field.result.occupancy_mask.shape, 4)
    figure, axes = plt.subplots(2, len(indices), figsize=(15.5, 7.0), sharex=True, sharey=True)
    for column, index in enumerate(indices):
        occupancy = field.result.occupancy_mask[:, :, index].T
        boundary = field.result.boundary_mask[:, :, index].T
        extent = [
            0.0,
            (field.result.occupancy_mask.shape[0] - 1) * field.spacing[0],
            0.0,
            (field.result.occupancy_mask.shape[1] - 1) * field.spacing[1],
        ]
        axes[0, column].imshow(
            occupancy, origin="lower", interpolation="nearest", aspect="equal", extent=extent
        )
        axes[1, column].imshow(
            boundary, origin="lower", interpolation="nearest", aspect="equal", extent=extent
        )
        z_value = index * field.spacing[2]
        axes[0, column].set_title(f"occupancy, z={z_value:.2f} m")
        axes[1, column].set_title(f"Dirichlet boundary, z={z_value:.2f} m")
        axes[1, column].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[1, 0].set_ylabel("y [m]")
    figure.suptitle("Occupancy and Poisson boundary slices")
    save_figure(figure, directory, "occupancy_boundary_slices", dpi=dpi, svg=False)


def plot_poisson_planes(
    *,
    field: PoissonField,
    directory: str | Path,
    dpi: int,
) -> None:
    """Plot ``h`` and ``||Dh||`` in metric XY, XZ, and YZ planes."""

    configure_academic_style()
    h = field.h
    gradient_magnitude = np.linalg.norm(field.grad_h, axis=-1)
    coordinates = [
        _axis_coordinates(size, spacing)
        for size, spacing in zip(h.shape, field.spacing, strict=True)
    ]
    if field.dimension == 2:
        x, y = coordinates
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
        im0 = axes[0].contourf(x, y, h.T, levels=40)
        im1 = axes[1].contourf(x, y, gradient_magnitude.T, levels=40)
        axes[0].set_title("Poisson safety function $h_P(x,y)$")
        axes[1].set_title(r"Gradient magnitude $\|D h_P(x,y)\|$")
        for axis in axes:
            axis.set(xlabel="x [m]", ylabel="y [m]", aspect="equal")
        figure.colorbar(im0, ax=axes[0], shrink=0.85, label="$h_P$")
        figure.colorbar(im1, ax=axes[1], shrink=0.85, label=r"$\|D h_P\|$")
        save_figure(figure, directory, "poisson_field_planes", dpi=dpi, svg=False)
        return

    x, y, z = coordinates
    ix, iy, iz = (dimension // 2 for dimension in h.shape)
    planes = [
        (x, y, h[:, :, iz].T, gradient_magnitude[:, :, iz].T, "XY", z[iz], "x [m]", "y [m]"),
        (x, z, h[:, iy, :].T, gradient_magnitude[:, iy, :].T, "XZ", y[iy], "x [m]", "z [m]"),
        (y, z, h[ix, :, :].T, gradient_magnitude[ix, :, :].T, "YZ", x[ix], "y [m]", "z [m]"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.0), constrained_layout=True)
    for column, (horizontal, vertical, h_plane, grad_plane, label, fixed_value, xlabel, ylabel) in enumerate(planes):
        image_h = axes[0, column].contourf(horizontal, vertical, h_plane, levels=45)
        image_g = axes[1, column].contourf(horizontal, vertical, grad_plane, levels=45)
        axes[0, column].set_title(f"$h_P$ on {label}; fixed coordinate={fixed_value:.2f} m")
        axes[1, column].set_title(rf"$\|D h_P\|$ on {label}; fixed coordinate={fixed_value:.2f} m")
        axes[0, column].set(xlabel=xlabel, ylabel=ylabel, aspect="equal")
        axes[1, column].set(xlabel=xlabel, ylabel=ylabel, aspect="equal")
        figure.colorbar(image_h, ax=axes[0, column], shrink=0.80, label="$h_P$")
        figure.colorbar(image_g, ax=axes[1, column], shrink=0.80, label=r"$\|D h_P\|$")
    figure.suptitle("Poisson safety function and spatial derivative magnitude")
    save_figure(figure, directory, "poisson_field_planes", dpi=dpi, svg=False)

def plot_poisson_diagnostics(
    *,
    fields: Mapping[str, PoissonField],
    directory: str | Path,
    dpi: int,
) -> None:
    """Compare forcing, h, gradient magnitude, and numerical Laplacian by method."""

    configure_academic_style()
    methods = list(fields)
    if not methods:
        return
    figure, axes = plt.subplots(4, len(methods), figsize=(4.0 * len(methods), 13.5), squeeze=False)
    for column, method in enumerate(methods):
        field = fields[method]
        result = field.result
        if field.dimension == 3:
            index = result.h.shape[2] // 2
            forcing = result.forcing[:, :, index].T
            h = field.h[:, :, index].T
            gradient = np.linalg.norm(field.grad_h[:, :, index, :], axis=-1).T
            laplacian = result.laplacian_h[:, :, index].T
        else:
            forcing = result.forcing.T
            h = field.h.T
            gradient = np.linalg.norm(field.grad_h, axis=-1).T
            laplacian = result.laplacian_h.T
        panels = [forcing, h, gradient, laplacian]
        titles = ["forcing $f_P$", "safety $h_P$", r"gradient magnitude $\|D h_P\|$", r"reconstructed $\Delta_h h_P$"]
        x_coordinates = _axis_coordinates(forcing.shape[1], field.spacing[0])
        y_coordinates = _axis_coordinates(forcing.shape[0], field.spacing[1])
        for row, (panel, title) in enumerate(zip(panels, titles, strict=True)):
            image = axes[row, column].contourf(x_coordinates, y_coordinates, panel, levels=42)
            axes[row, column].set_title(f"{method}: {title}")
            axes[row, column].set(xlabel="x [m]", ylabel="y [m]", aspect="equal")
            figure.colorbar(image, ax=axes[row, column], shrink=0.75)
    figure.suptitle("Poisson forcing and field diagnostics by forcing method")
    save_figure(figure, directory, "poisson_forcing_field_diagnostics", dpi=dpi, svg=False)


def plot_poisson_isosurfaces(
    *,
    field: PoissonField,
    world: Any,
    directory: str | Path,
    dpi: int,
) -> None:
    """Render several volumetric h level sets for a 3-D field."""

    if field.dimension != 3:
        return
    configure_academic_style()
    figure = plt.figure(figsize=(12.5, 8.8))
    axis = figure.add_subplot(111, projection="3d")
    values = field.h
    positive = values[field.result.solve_mask]
    levels = np.quantile(positive, [0.25, 0.50, 0.75])
    for level in levels:
        try:
            vertices, faces, _, _ = marching_cubes(values, level=float(level), spacing=field.spacing)
        except ValueError:
            continue
        mesh = Poly3DCollection(vertices[faces], alpha=0.12, linewidths=0.15)
        axis.add_collection3d(mesh)
    draw_world_obstacles(axis, world, alpha=0.12)
    axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    axis.set_xlim(0.0, world.extent_m[0])
    axis.set_ylim(0.0, world.extent_m[1])
    axis.set_zlim(0.0, world.extent_m[2])
    axis.view_init(elev=25, azim=-58)
    axis.set_title("Three-dimensional Poisson safety-field level sets")
    save_figure(figure, directory, "poisson_isosurfaces_3d", dpi=dpi, svg=False)


def _position_projection_matrix(P: np.ndarray, dimension: int) -> np.ndarray:
    Ppp = P[:dimension, :dimension]
    Ppv = P[:dimension, dimension:]
    Pvv = P[dimension:, dimension:]
    return Ppp - Ppv @ np.linalg.solve(Pvv, Ppv.T)


def _ellipse_from_quadratic(matrix: np.ndarray, c: float) -> tuple[np.ndarray, float, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    widths = 2.0 * np.sqrt(c / np.maximum(eigenvalues, 1.0e-12))
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    return widths, angle, float(np.linalg.det(matrix))


def plot_clf_roa_projections(
    *,
    controller: Any,
    world: Any,
    directory: str | Path,
    dpi: int,
) -> None:
    """Plot honest position projections of the CLF-certified ROAs.

    A double-integrator CLF lives in position--velocity state space.  The
    displayed scalar field is therefore the Schur-complement position
    projection

    ``V_proj(p) = (p-p*)^T (Ppp-Ppv Pvv^{-1} Pvp) (p-p*)``.

    For 3-D systems the main panels are XY slices through ``z=z*`` of this
    position-projected 6-D ROA.  The actual certified boundary is the contour
    ``V_proj/c = 1``.  If that boundary lies outside the displayed workspace,
    the panel says so instead of drawing a misleading clipped ellipsoid.
    """

    if not controller.clf.enabled or controller.dimension not in {2, 3}:
        return
    configure_academic_style()
    dimension = controller.dimension
    extent = tuple(float(value) for value in world.extent_m)
    artifacts = list(controller.clf.artifacts.items())
    if not artifacts:
        return

    x = np.linspace(0.0, extent[0], 180)
    y = np.linspace(0.0, extent[1], 145)
    X, Y = np.meshgrid(x, y)
    fields: list[tuple[str, Any, np.ndarray, np.ndarray]] = []
    global_max = 0.0
    for identifier, artifact in artifacts:
        projection = _position_projection_matrix(artifact.P, dimension)
        center = artifact.target.x_star[:dimension]
        if dimension == 2:
            delta = np.stack([X - center[0], Y - center[1]], axis=-1)
            normalized = np.einsum("...i,ij,...j->...", delta, projection, delta) / artifact.c
        else:
            delta = np.stack(
                [X - center[0], Y - center[1], np.zeros_like(X)], axis=-1
            )
            normalized = np.einsum("...i,ij,...j->...", delta, projection, delta) / artifact.c
        global_max = max(global_max, float(np.nanmax(normalized)))
        fields.append((identifier, artifact, projection, normalized))

    rows = int(np.ceil(len(fields) / 2))
    figure, axes = plt.subplots(
        rows,
        2,
        figsize=(13.5, 5.2 * rows),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    color_max = max(global_max, 1.0e-3)
    contour_levels = np.linspace(0.0, color_max, 35)
    image = None
    for axis, (identifier, artifact, projection, normalized) in zip(
        axes.ravel(), fields, strict=False
    ):
        image = axis.contourf(
            x,
            y,
            normalized,
            levels=contour_levels,
            vmin=0.0,
            vmax=color_max,
        )
        candidate_levels = [level for level in (0.025, 0.05, 0.10, 0.20, 0.50, 1.0) if float(np.nanmin(normalized)) <= level <= float(np.nanmax(normalized))]
        if candidate_levels:
            contours = axis.contour(
                x,
                y,
                normalized,
                levels=candidate_levels,
                linewidths=1.0,
                colors="white",
                alpha=0.85,
            )
            axis.clabel(contours, inline=True, fontsize=8, fmt=lambda value: f"{value:.3g}")
        center = artifact.target.x_star[:2]
        axis.scatter(center[0], center[1], marker="*", s=95, edgecolor="black", linewidth=0.6, label="landing equilibrium")
        if float(np.nanmax(normalized)) < 1.0:
            boundary_note = "certified boundary outside displayed workspace"
        else:
            axis.contour(x, y, normalized, levels=[1.0], colors="black", linewidths=2.2)
            boundary_note = r"black contour: $V_{proj}/c=1$"
        slice_note = (
            "position projection of 4-D ROA"
            if dimension == 2
            else f"XY slice of position projection at z={artifact.target.x_star[2]:.2f} m"
        )
        axis.set(
            xlim=(0.0, extent[0]),
            ylim=(0.0, extent[1]),
            xlabel="x [m]",
            ylabel="y [m]",
            aspect="equal",
            title=f"{identifier}: {slice_note}\n{boundary_note}",
        )
        axis.legend(loc="upper left", fontsize=8)
    for axis in axes.ravel()[len(fields):]:
        axis.set_visible(False)
    if image is not None:
        figure.colorbar(
            image,
            ax=[axis for axis in axes.ravel() if axis.get_visible()],
            shrink=0.86,
            label=r"normalized projected CLF $V_{proj}/c$",
        )
    figure.suptitle(
        "CLF attraction geometry and landing-equilibrium centers\n"
        "Displayed fields are projections/slices of the full state-space sublevel sets"
    )
    save_figure(figure, directory, "clf_regions_of_attraction", dpi=dpi, svg=False)

    # A second quantitative figure reports the full principal semi-axis lengths
    # of each position projection, even when the boundary is outside the world.
    figure_axes, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    positions = np.arange(len(fields), dtype=float)
    width = 0.22 if dimension == 3 else 0.32
    for principal_index in range(dimension):
        values = []
        for _identifier, artifact, projection, _normalized in fields:
            eigenvalues = np.linalg.eigvalsh(projection)
            values.append(float(np.sqrt(artifact.c / max(eigenvalues[principal_index], 1.0e-12))))
        offset = (principal_index - (dimension - 1) / 2.0) * width
        axis.bar(
            positions + offset,
            values,
            width=width,
            label=f"principal semi-axis {principal_index + 1}",
        )
    axis.axhline(float(np.linalg.norm(extent)), linestyle="--", linewidth=1.2, label="workspace diagonal")
    axis.set_xticks(positions, [identifier for identifier, *_ in fields])
    axis.set(
        xlabel="landing target",
        ylabel="position-projection semi-axis length [m]",
        title="Principal semi-axes of the CLF position projections",
    )
    axis.legend(ncol=2)
    save_figure(figure_axes, directory, "clf_roa_principal_axes", dpi=dpi)

def plot_clf_phase_portraits(
    *,
    controller: Any,
    target_id: str,
    directory: str | Path,
    dpi: int,
) -> None:
    """Show CLF contours and closed-loop dynamics on each position-velocity axis."""

    if not controller.clf.enabled:
        return
    configure_academic_style()
    artifact = controller.clf.artifacts[target_id]
    dimension = controller.dimension
    figure, axes = plt.subplots(1, dimension, figsize=(5.0 * dimension, 4.8))
    if dimension == 1:
        axes = [axes]
    for axis_index, axis in enumerate(axes):
        e_values = np.linspace(-4.0, 4.0, 45)
        v_values = np.linspace(-3.0, 3.0, 45)
        E, VEL = np.meshgrid(e_values, v_values)
        state_errors = np.zeros((E.size, 2 * dimension), dtype=float)
        state_errors[:, axis_index] = E.ravel()
        state_errors[:, dimension + axis_index] = VEL.ravel()
        values = np.einsum("bi,ij,bj->b", state_errors, artifact.P, state_errors).reshape(E.shape)
        control = -(artifact.K @ state_errors.T).T
        dE = VEL
        dV = control[:, axis_index].reshape(E.shape)
        axis.contour(E, VEL, values, levels=14, linewidths=0.8)
        axis.contour(E, VEL, values, levels=[artifact.c], linewidths=2.0)
        stride = 3
        axis.quiver(
            E[::stride, ::stride],
            VEL[::stride, ::stride],
            dE[::stride, ::stride],
            dV[::stride, ::stride],
            angles="xy",
            scale_units="xy",
            scale=9.0,
            width=0.0025,
        )
        axis.scatter(0.0, 0.0, marker="*", s=75, label="equilibrium")
        axis.set(
            xlabel=f"position error e_{'xyz'[axis_index]} [m]",
            ylabel=f"velocity v_{'xyz'[axis_index]} [m/s]",
            title=f"{target_id}: {'xyz'[axis_index]}-axis phase slice",
        )
        axis.legend(loc="upper right")
    figure.suptitle("Lyapunov contours and closed-loop attraction fields")
    save_figure(figure, directory, "clf_phase_portraits", dpi=dpi)


def plot_contingency_maps(
    *,
    controller: Any,
    world: Any,
    fixed_z: float = 0.0,
    directory: str | Path,
    dpi: int,
    grid_points: int = 160,
) -> None:
    """Plot per-target ROA margins, the r-th pivot, and certified count.

    The map is evaluated on the zero-velocity position slice.  For a 3-D
    model, ``fixed_z`` is held constant; for a 2-D model the state slice is
    simply ``[x, y, 0, 0]``.
    """

    if controller.dimension not in {2, 3} or not controller.clf.enabled:
        return
    configure_academic_style()
    extent = tuple(float(value) for value in world.extent_m)
    x_axis = np.linspace(0.0, extent[0], grid_points)
    y_axis = np.linspace(0.0, extent[1], grid_points)
    X, Y = np.meshgrid(x_axis, y_axis, indexing="xy")
    target_ids = list(controller.clf.artifacts)
    # Evaluate the complete spatial slice in one vectorized operation.  Calling
    # the high-level evaluator once per pixel is mathematically redundant and
    # dominated the runtime of high-resolution figure generation.
    point_count = X.size
    if controller.dimension == 3:
        states = np.column_stack(
            [
                X.ravel(),
                Y.ravel(),
                np.full(point_count, fixed_z),
                np.zeros((point_count, 3)),
            ]
        )
    else:
        states = np.column_stack(
            [X.ravel(), Y.ravel(), np.zeros((point_count, 2))]
        )
    margins = np.empty((len(target_ids), point_count), dtype=float)
    for index, target_id in enumerate(target_ids):
        artifact = controller.clf.artifacts[target_id]
        errors = states - artifact.target.x_star[None, :]
        values = np.einsum(
            "bi,ij,bj->b", errors, artifact.P, errors, optimize=True
        )
        margins[index] = float(artifact.c) - values
    margins = margins.reshape(len(target_ids), grid_points, grid_points)
    r = controller.contingency.config.required_certified
    pivot_index = margins.shape[0] - r
    pivot = np.partition(margins, pivot_index, axis=0)[pivot_index]
    count = np.sum(margins >= 0.0, axis=0)
    critical = np.argmin(np.abs(margins - pivot[None, :, :]), axis=0)

    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))
    selected_ids = target_ids[:3]
    slice_label = "v=0" if controller.dimension == 2 else f"z={fixed_z:.2f} m, v=0"
    for axis, target_id in zip(axes[0], selected_ids, strict=False):
        index = target_ids.index(target_id)
        image = axis.contourf(X, Y, margins[index], levels=42)
        if np.min(margins[index]) <= 0.0 <= np.max(margins[index]):
            axis.contour(X, Y, margins[index], levels=[0.0], linewidths=2.0)
        axis.set_title(f"ROA certificate h_{target_id}(x,y), {slice_label}")
        figure.colorbar(image, ax=axis, shrink=0.78)
    image_pivot = axes[1, 0].contourf(X, Y, pivot, levels=42)
    if np.min(pivot) <= 0.0 <= np.max(pivot):
        axes[1, 0].contour(X, Y, pivot, levels=[0.0], linewidths=2.0)
    axes[1, 0].set_title(f"r-th largest ROA pivot, r={r}")
    figure.colorbar(image_pivot, ax=axes[1, 0], shrink=0.78)
    image_count = axes[1, 1].pcolormesh(X, Y, count, shading="auto")
    axes[1, 1].set_title("Number of certified landing alternatives")
    figure.colorbar(image_count, ax=axes[1, 1], shrink=0.78)
    image_critical = axes[1, 2].pcolormesh(X, Y, critical, shading="auto")
    axes[1, 2].set_title("Critical certificate index")
    figure.colorbar(image_critical, ax=axes[1, 2], shrink=0.78)
    for axis in axes.ravel():
        axis.set(xlabel="x [m]", ylabel="y [m]", aspect="equal")
        axis.set_xlim(0.0, extent[0])
        axis.set_ylim(0.0, extent[1])
    figure.suptitle("Combinatorial regions of attraction on a fixed state slice")
    save_figure(figure, directory, "contingency_roa_maps", dpi=dpi, svg=False)

def plot_time_histories(
    *,
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    controller: Any,
    directory: str | Path,
    dpi: int,
) -> None:
    """Plot synchronized safety, stability, contingency, and runtime histories."""

    if metrics.empty:
        return
    configure_academic_style()
    figure, axes = plt.subplots(4, 2, figsize=(15.5, 12.5), sharex=True)
    time = metrics["time_s"]
    axes[0, 0].plot(time, metrics["poisson_h"])
    axes[0, 0].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("Poisson safety value h_P(t)")
    axes[0, 1].plot(time, metrics["hocbf_residual"])
    axes[0, 1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("HOCBF constraint residual")
    axes[1, 0].plot(time, metrics["active_V"])
    axes[1, 0].set_title("Active-target Lyapunov function V(t)")
    axes[1, 1].plot(time, metrics["active_clf_residual"])
    axes[1, 1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1, 1].set_title("Active-target CLF residual")
    for identifier in controller.targets:
        column = f"h_roa_{identifier}"
        if column in metrics:
            axes[2, 0].plot(time, metrics[column], label=identifier)
    axes[2, 0].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[2, 0].set_title("Landing-zone ROA certificates")
    axes[2, 0].legend(ncol=2)
    axes[2, 1].plot(time, metrics["contingency_pivot"], label="r-th pivot")
    axes[2, 1].step(time, metrics["certified_count"], where="post", label="certified count")
    axes[2, 1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[2, 1].set_title("Contingency margin and certified alternatives")
    axes[2, 1].legend()
    axes[3, 0].plot(time, metrics["intervention_norm"], label="||a_safe-a_nom||")
    axes[3, 0].plot(time, metrics["omega"], label="omega")
    if "clf_slack" in metrics:
        axes[3, 0].plot(time, metrics["clf_slack"], label="CLF slack")
    axes[3, 0].set_title("Control intervention and contingency auxiliary")
    axes[3, 0].legend()
    axes[3, 1].plot(time, 1.0e3 * metrics["solver_time_s"])
    axes[3, 1].set_title("Multi-certificate filter time")
    axes[3, 1].set_ylabel("time [ms]")
    for axis in axes[-1]:
        axis.set_xlabel("time [s]")
    if not events.empty:
        for _, event in events.iterrows():
            for axis in axes.ravel():
                axis.axvline(float(event["time_s"]), linestyle=":", linewidth=0.9)
    figure.suptitle("Synchronized safety, stability, contingency, and solver histories")
    save_figure(figure, directory, "certificate_time_histories", dpi=dpi)


def plot_solver_comparison(
    records: Sequence[Mapping[str, Any]],
    *,
    directory: str | Path,
    dpi: int,
) -> None:
    """Plot time, algebraic residual, Laplacian error, and field error by solver."""

    if not records:
        return
    configure_academic_style()
    frame = pd.DataFrame(records)
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    labels = frame["solver"].tolist()
    x = np.arange(len(labels))
    axes[0, 0].bar(x - 0.18, frame["total_wall_time_s"], width=0.36, label="total wall")
    axes[0, 0].bar(x + 0.18, frame["solve_time_s"], width=0.36, label="solve stage")
    axes[0, 0].set_xticks(x, labels, rotation=18)
    axes[0, 0].set_ylabel("time [s]")
    axes[0, 0].set_title("Poisson solver timing")
    axes[0, 0].legend()
    residual_l2 = np.maximum(frame["algebraic_residual_l2"].to_numpy(float), 1.0e-16)
    residual_linf = np.maximum(frame["algebraic_residual_linf"].to_numpy(float), 1.0e-16)
    axes[0, 1].bar(x - 0.18, residual_l2, width=0.36, label="RMS")
    axes[0, 1].bar(x + 0.18, residual_linf, width=0.36, label="Linf")
    axes[0, 1].set_xticks(x, labels, rotation=18)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("||A h - b||")
    axes[0, 1].set_title("Exact assembled-system residual")
    axes[0, 1].legend()
    field_error = np.maximum(frame["relative_l2_field_error"].to_numpy(float), 1.0e-16)
    axes[1, 0].bar(labels, field_error)
    axes[1, 0].set_yscale("log")
    axes[1, 0].tick_params(axis="x", rotation=18)
    axes[1, 0].set_ylabel("relative L2 error")
    axes[1, 0].set_title("Field error to sparse-direct reference")
    axes[1, 1].bar(labels, frame["laplacian_l2_error"])
    axes[1, 1].tick_params(axis="x", rotation=18)
    axes[1, 1].set_ylabel("RMS(Delta_h h - f_P)")
    axes[1, 1].set_title("Reconstructed Laplacian consistency")
    figure.suptitle("Poisson solver accuracy and computational performance")
    save_figure(figure, directory, "poisson_solver_comparison", dpi=dpi)


def plot_parameter_sweep(
    records: Sequence[Mapping[str, Any]],
    *,
    parameter: str,
    title: str,
    directory: str | Path,
    name: str,
    dpi: int,
) -> None:
    """Plot the common six paper metrics for a one-dimensional parameter sweep."""

    if not records:
        return
    configure_academic_style()
    frame = pd.DataFrame(records).sort_values(parameter)
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))
    metrics = [
        ("minimum_poisson_h", "minimum h_P"),
        ("minimum_hocbf_residual", "minimum HOCBF residual"),
        ("minimum_contingency_pivot", "minimum r-th ROA pivot"),
        ("mean_intervention_norm", "mean control intervention"),
        ("duration_s", "landing duration [s]"),
        ("p95_solver_time_ms", "filter p95 time [ms]"),
    ]
    for axis, (column, label) in zip(axes.ravel(), metrics, strict=True):
        axis.plot(frame[parameter], frame[column], marker="o")
        axis.set(xlabel=parameter, ylabel=label, title=label)
        if np.all(frame[parameter] > 0.0):
            axis.set_xscale("log")
    figure.suptitle(title)
    save_figure(figure, directory, name, dpi=dpi)


def plot_forcing_comparison(
    *,
    fields: Mapping[str, PoissonField],
    summaries: Sequence[Mapping[str, Any]],
    trajectories: Mapping[str, pd.DataFrame],
    world: Any,
    directory: str | Path,
    dpi: int,
) -> None:
    """Compare forcing geometry, trajectory response, and quantitative metrics."""

    configure_academic_style()
    methods = list(fields)
    figure, axes = plt.subplots(2, len(methods), figsize=(4.2 * len(methods), 8.2), squeeze=False)
    for column, method in enumerate(methods):
        field = fields[method]
        if field.dimension == 3:
            z_index = field.h.shape[2] // 2
            h_slice = field.h[:, :, z_index].T
        else:
            h_slice = field.h.T
        x_coordinates = _axis_coordinates(h_slice.shape[1], field.spacing[0])
        y_coordinates = _axis_coordinates(h_slice.shape[0], field.spacing[1])
        image = axes[0, column].contourf(x_coordinates, y_coordinates, h_slice, levels=45)
        axes[0, column].set(xlabel="x [m]", ylabel="y [m]", title=f"{method}: $h_P$", aspect="equal")
        figure.colorbar(image, ax=axes[0, column], shrink=0.75, label="$h_P$")
        trajectory = trajectories.get(method)
        if trajectory is not None and not trajectory.empty:
            axes[1, column].plot(trajectory["x"], trajectory["y"])
            axes[1, column].scatter(trajectory["x"].iloc[0], trajectory["y"].iloc[0], s=35)
            axes[1, column].scatter(trajectory["x"].iloc[-1], trajectory["y"].iloc[-1], marker="x", s=45)
        axes[1, column].set(xlabel="x [m]", ylabel="y [m]", title=f"{method}: XY trajectory", aspect="equal")
    figure.suptitle("Forcing-function influence on the Poisson field and landing trajectory")
    save_figure(figure, directory, "forcing_field_trajectory_comparison", dpi=dpi, svg=False)

    if summaries:
        frame = pd.DataFrame(summaries)
        figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
        columns = [
            ("poisson_wall_time_s", "Poisson wall time [s]"),
            ("minimum_poisson_h", "minimum h_P"),
            ("mean_intervention_norm", "mean intervention"),
            ("minimum_contingency_pivot", "minimum ROA pivot"),
            ("duration_s", "landing duration [s]"),
            ("p95_solver_time_ms", "filter p95 [ms]"),
        ]
        for axis, (column, label) in zip(axes.ravel(), columns, strict=True):
            axis.bar(frame["forcing_method"], frame[column])
            axis.set_title(label)
            axis.tick_params(axis="x", rotation=18)
        figure.suptitle("Quantitative comparison of Poisson forcing methods")
        save_figure(figure, directory, "forcing_quantitative_comparison", dpi=dpi)


def plot_integrated_dashboard(
    *,
    world: Any,
    field: PoissonField,
    metrics: pd.DataFrame,
    controller: Any,
    solver_records: Sequence[Mapping[str, Any]],
    directory: str | Path,
    dpi: int,
) -> None:
    """Create one compact results dashboard modeled on the existing high-res output."""

    if metrics.empty:
        return
    configure_academic_style()
    figure = plt.figure(figsize=(16.0, 10.2))
    grid = figure.add_gridspec(2, 3)
    axis_3d = figure.add_subplot(grid[0, 0], projection="3d")
    draw_world_obstacles(axis_3d, world, alpha=0.16)
    axis_3d.plot(metrics["x"], metrics["y"], metrics["z"])
    axis_3d.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="3-D trajectory")
    axis_3d.view_init(elev=26, azim=-62)

    axis_h = figure.add_subplot(grid[0, 1])
    z_index = field.h.shape[2] // 2
    x_coordinates = _axis_coordinates(field.h.shape[0], field.spacing[0])
    y_coordinates = _axis_coordinates(field.h.shape[1], field.spacing[1])
    image = axis_h.contourf(x_coordinates, y_coordinates, field.h[:, :, z_index].T, levels=45)
    axis_h.set(
        xlabel="x [m]",
        ylabel="y [m]",
        title=f"Poisson $h_P$, {field.forcing_method}; z={z_index * field.spacing[2]:.2f} m",
        aspect="equal",
    )
    figure.colorbar(image, ax=axis_h, shrink=0.78, label="$h_P$")

    axis_history = figure.add_subplot(grid[0, 2])
    axis_history.plot(metrics["time_s"], metrics["poisson_h"], label="h_P")
    axis_history.plot(metrics["time_s"], metrics["active_h_roa"], label="active h_ROA")
    axis_history.plot(metrics["time_s"], metrics["contingency_pivot"], label="pivot")
    axis_history.axhline(0.0, linestyle="--", linewidth=1.0)
    axis_history.set(xlabel="time [s]", title="Certificate histories")
    axis_history.legend()

    axis_intervention = figure.add_subplot(grid[1, 0])
    axis_intervention.plot(metrics["time_s"], metrics["intervention_norm"], label="control correction")
    axis_intervention.plot(metrics["time_s"], metrics["omega"], label="omega")
    axis_intervention.set(xlabel="time [s]", title="Control and contingency effort")
    axis_intervention.legend()

    axis_solver = figure.add_subplot(grid[1, 1])
    if solver_records:
        frame = pd.DataFrame(solver_records)
        axis_solver.bar(frame["solver"], frame["total_wall_time_s"])
        axis_solver.tick_params(axis="x", rotation=18)
    axis_solver.set(title="Poisson solver wall time", ylabel="time [s]")

    axis_roa = figure.add_subplot(grid[1, 2])
    for identifier in controller.targets:
        column = f"h_roa_{identifier}"
        if column in metrics:
            axis_roa.plot(metrics["time_s"], metrics[column], label=identifier)
    axis_roa.axhline(0.0, linestyle="--", linewidth=1.0)
    axis_roa.set(xlabel="time [s]", title="Landing-zone attraction margins")
    axis_roa.legend(ncol=2)
    figure.suptitle("Poisson-HOCBF, CLF, and contingency landing study")
    save_figure(figure, directory, "integrated_research_dashboard", dpi=dpi, svg=False)


def plot_image_pipeline(
    *,
    image_bgr: np.ndarray,
    raw_mask: np.ndarray,
    occupancy: np.ndarray,
    field: PoissonField,
    metrics: pd.DataFrame,
    target_positions: Mapping[str, np.ndarray],
    workspace_size_m: Sequence[float],
    directory: str | Path,
    dpi: int,
) -> None:
    """Create the static-image perception, field, and control dashboard."""

    import cv2

    configure_academic_style()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    width_m, height_m = float(workspace_size_m[0]), float(workspace_size_m[1])
    extent = (0.0, width_m, height_m, 0.0)
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    axes[0, 0].imshow(rgb, extent=extent)
    axes[0, 0].set_title("Rectified input image")
    axes[0, 1].imshow(raw_mask, cmap="gray", extent=extent)
    axes[0, 1].set_title("Segmented obstacle mask")
    axes[0, 2].imshow(occupancy, cmap="gray", extent=extent)
    axes[0, 2].set_title("Inflated configuration-space occupancy")
    h_image = axes[1, 0].contourf(
        np.linspace(0.0, width_m, field.h.shape[0]),
        np.linspace(0.0, height_m, field.h.shape[1]),
        field.h.T,
        levels=45,
    )
    axes[1, 0].set_title("Poisson safety function h(x,y)")
    figure.colorbar(h_image, ax=axes[1, 0], shrink=0.78)
    grad_image = axes[1, 1].contourf(
        np.linspace(0.0, width_m, field.h.shape[0]),
        np.linspace(0.0, height_m, field.h.shape[1]),
        np.linalg.norm(field.grad_h, axis=-1).T,
        levels=45,
    )
    axes[1, 1].set_title("Poisson gradient magnitude ||Dh||")
    figure.colorbar(grad_image, ax=axes[1, 1], shrink=0.78)
    axes[1, 2].imshow(rgb, extent=extent, alpha=0.68)
    if not metrics.empty:
        axes[1, 2].plot(metrics["x"], metrics["y"], linewidth=2.2, label="safe trajectory")
        axes[1, 2].scatter(metrics["x"].iloc[0], metrics["y"].iloc[0], s=45, label="start")
        axes[1, 2].scatter(metrics["x"].iloc[-1], metrics["y"].iloc[-1], marker="x", s=55, label="terminal")
    for identifier, position in target_positions.items():
        axes[1, 2].scatter(position[0], position[1], s=38)
        axes[1, 2].annotate(identifier, position[:2], xytext=(4, 4), textcoords="offset points")
    axes[1, 2].set_title("Filtered trajectory and landing candidates")
    axes[1, 2].legend(loc="best")
    for axis in axes.ravel():
        axis.set(xlabel="x [m]", ylabel="y [m]")
        axis.set_xlim(0.0, width_m)
        axis.set_ylim(height_m, 0.0)
        axis.set_aspect("equal", adjustable="box")
    for axis in (axes[1, 0], axes[1, 1]):
        axis.set_ylim(0.0, height_m)
    figure.suptitle("Static-image perception-to-safety experiment")
    save_figure(figure, directory, "static_image_pipeline", dpi=dpi, svg=False)


def plot_static_dashboard(
    *,
    image_bgr: np.ndarray,
    field: PoissonField,
    metrics: pd.DataFrame,
    controller: Any,
    workspace_size_m: Sequence[float],
    solver_records: Sequence[Mapping[str, Any]],
    directory: str | Path,
    dpi: int,
) -> None:
    """Create a compact 2x3 dashboard matching the high-resolution study style."""

    if metrics.empty:
        return
    import cv2

    configure_academic_style()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    width_m, height_m = float(workspace_size_m[0]), float(workspace_size_m[1])
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 10.0))
    axes[0, 0].imshow(rgb, extent=(0.0, width_m, height_m, 0.0), alpha=0.65)
    axes[0, 0].plot(metrics["x"], metrics["y"], label="safe trajectory")
    axes[0, 0].scatter(metrics["x"].iloc[0], metrics["y"].iloc[0], s=45, label="start")
    for identifier, point in _target_positions(controller).items():
        axes[0, 0].scatter(point[0], point[1], s=35)
        axes[0, 0].annotate(identifier, point[:2], xytext=(3, 3), textcoords="offset points")
    axes[0, 0].set(title="Image-space landing trajectory", xlabel="x [m]", ylabel="y [m]")
    axes[0, 0].set_xlim(0.0, width_m)
    axes[0, 0].set_ylim(height_m, 0.0)
    axes[0, 0].legend(loc="best")

    x_axis = np.linspace(0.0, width_m, field.h.shape[0])
    y_axis = np.linspace(0.0, height_m, field.h.shape[1])
    image = axes[0, 1].contourf(x_axis, y_axis, field.h.T, levels=45)
    axes[0, 1].plot(metrics["x"], metrics["y"], linewidth=1.6)
    axes[0, 1].set(title=f"Poisson h, forcing={field.forcing_method}", xlabel="x [m]", ylabel="y [m]")
    axes[0, 1].set_aspect("equal", adjustable="box")
    figure.colorbar(image, ax=axes[0, 1], shrink=0.78)

    axes[0, 2].plot(metrics["time_s"], metrics["poisson_h"], label="h_P")
    axes[0, 2].plot(metrics["time_s"], metrics["active_h_roa"], label="active h_ROA")
    axes[0, 2].plot(metrics["time_s"], metrics["contingency_pivot"], label="pivot")
    axes[0, 2].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[0, 2].set(title="Certificate histories", xlabel="time [s]")
    axes[0, 2].legend()

    axes[1, 0].plot(metrics["time_s"], metrics["intervention_norm"], label="control correction")
    axes[1, 0].plot(metrics["time_s"], metrics["omega"], label="omega")
    axes[1, 0].set(title="Control and contingency effort", xlabel="time [s]")
    axes[1, 0].legend()

    if solver_records:
        frame = pd.DataFrame(solver_records)
        axes[1, 1].bar(frame["solver"], frame["total_wall_time_s"])
        axes[1, 1].tick_params(axis="x", rotation=18)
    axes[1, 1].set(title="Poisson solver wall time", ylabel="time [s]")

    for identifier in controller.targets:
        column = f"h_roa_{identifier}"
        if column in metrics:
            axes[1, 2].plot(metrics["time_s"], metrics[column], label=identifier)
    axes[1, 2].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1, 2].set(title="Landing-zone attraction margins", xlabel="time [s]")
    axes[1, 2].legend(ncol=2)
    figure.suptitle("Static-image Poisson-HOCBF, CLF, and contingency study")
    save_figure(figure, directory, "integrated_research_dashboard", dpi=dpi, svg=False)

def plot_live_summary(
    metrics: pd.DataFrame,
    *,
    directory: str | Path,
    dpi: int,
) -> None:
    """Save publication-ready histories after a live/video experiment finishes."""

    if metrics.empty:
        return
    configure_academic_style()
    figure, axes = plt.subplots(3, 2, figsize=(13.5, 10.0), sharex=True)
    time = metrics["time_s"]
    pairs = [
        ("poisson_h", "Poisson h"),
        ("hocbf_residual", "HOCBF residual"),
        ("active_V", "Active V"),
        ("active_h_roa", "Active h_ROA"),
        ("contingency_pivot", "r-th ROA pivot"),
        ("intervention_norm", "Control intervention"),
    ]
    for axis, (column, title) in zip(axes.ravel(), pairs, strict=True):
        axis.plot(time, metrics[column])
        if "residual" in column or "h_roa" in column or "pivot" in column or column == "poisson_h":
            axis.axhline(0.0, linestyle="--", linewidth=1.0)
        axis.set_title(title)
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    figure.suptitle("Live vision experiment: real-time certificate histories")
    save_figure(figure, directory, "live_certificate_histories", dpi=dpi)


def plot_trajectory_family(
    trajectories: Mapping[float, pd.DataFrame],
    *,
    world: Any,
    parameter_label: str,
    title: str,
    directory: str | Path,
    name: str,
    dpi: int,
) -> None:
    """Plot one parameterized family of 3-D trajectories and its projections.

    The same scalar colormap is shared by all panels, so spatial changes can be
    attributed directly to the swept parameter rather than to arbitrary line
    styling. Empty or failed rollouts are skipped but remain present in the raw
    CSV summaries.
    """

    valid = {
        float(value): frame
        for value, frame in trajectories.items()
        if frame is not None and not frame.empty and {"x", "y"}.issubset(frame.columns)
    }
    if not valid:
        return
    configure_academic_style()
    values = np.asarray(sorted(valid), dtype=float)
    normalization = matplotlib.colors.LogNorm(values.min(), values.max()) if np.all(values > 0) else matplotlib.colors.Normalize(values.min(), values.max())
    cmap = plt.get_cmap("viridis")
    figure = plt.figure(figsize=(15.8, 10.0))
    grid = figure.add_gridspec(2, 2)
    ax3d = figure.add_subplot(grid[0, 0], projection="3d")
    ax_xy = figure.add_subplot(grid[0, 1])
    ax_xz = figure.add_subplot(grid[1, 0])
    ax_yz = figure.add_subplot(grid[1, 1])
    draw_world_obstacles(ax3d, world, alpha=0.08)
    for value in values:
        frame = valid[float(value)]
        color = cmap(normalization(value))
        if "z" in frame:
            ax3d.plot(frame["x"], frame["y"], frame["z"], color=color, linewidth=1.8)
            ax_xz.plot(frame["x"], frame["z"], color=color, linewidth=1.8)
            ax_yz.plot(frame["y"], frame["z"], color=color, linewidth=1.8)
        ax_xy.plot(frame["x"], frame["y"], color=color, linewidth=1.8)
    ax3d.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="3-D trajectory family")
    ax3d.view_init(elev=26, azim=-62)
    ax_xy.set(xlabel="x [m]", ylabel="y [m]", title="XY projection", aspect="equal")
    ax_xz.set(xlabel="x [m]", ylabel="z [m]", title="XZ projection")
    ax_yz.set(xlabel="y [m]", ylabel="z [m]", title="YZ projection")
    scalar = matplotlib.cm.ScalarMappable(norm=normalization, cmap=cmap)
    scalar.set_array(values)
    figure.colorbar(scalar, ax=[ax3d, ax_xy, ax_xz, ax_yz], shrink=0.78, label=parameter_label)
    figure.suptitle(title)
    save_figure(figure, directory, name, dpi=dpi)
