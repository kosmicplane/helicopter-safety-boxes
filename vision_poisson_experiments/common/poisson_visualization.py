"""Static scientific diagnostics and efficient OpenCV live dashboards."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib

# The experiment saves figures in containers and automated tests. Selecting the
# non-interactive backend here avoids hidden display dependencies.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .occupancy import OccupancyMaps
from .poisson_runner import PoissonRunRecord
from .segmentation import SegmentationResult


def _extent_for_downward_y(workspace_size_m: tuple[float, float]) -> list[float]:
    """Return an imshow extent whose physical y coordinate increases downward."""

    width_m, height_m = workspace_size_m
    return [0.0, float(width_m), float(height_m), 0.0]


def _grid_xy(shape_yx: tuple[int, int], workspace_size_m: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return physical X and Y mesh grids for a row-major field."""

    ny, nx = shape_yx
    width_m, height_m = workspace_size_m
    x = np.linspace(0.0, width_m, nx)
    y = np.linspace(0.0, height_m, ny)
    return np.meshgrid(x, y)


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    """Save and close one Matplotlib figure to prevent resource accumulation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.clear()
    plt.close(fig)
    # Matplotlib renderers can retain large RGB buffers after close. Explicit
    # collection keeps the all-method static experiment bounded in containers.
    gc.collect()


def _plot_image(
    image_bgr: np.ndarray,
    title: str,
    path: Path,
    *,
    workspace_size_m: tuple[float, float] | None = None,
    dpi: int = 180,
) -> None:
    """Save one BGR image with either pixel or metric axes."""

    fig, ax = plt.subplots(figsize=(8, 6))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if workspace_size_m is None:
        ax.imshow(image_rgb)
        ax.set_xlabel("pixel column")
        ax.set_ylabel("pixel row")
    else:
        ax.imshow(image_rgb, extent=_extent_for_downward_y(workspace_size_m), aspect="auto")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m], downward-positive")
    ax.set_title(title)
    _save_figure(fig, path, dpi)


def _plot_binary(
    mask: np.ndarray,
    title: str,
    path: Path,
    workspace_size_m: tuple[float, float],
    *,
    label: str = "True",
    dpi: int = 180,
) -> None:
    """Save a boolean or binary map with metric axes."""

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        np.asarray(mask, dtype=bool),
        origin="upper",
        extent=_extent_for_downward_y(workspace_size_m),
        interpolation="nearest",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(label)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m], downward-positive")
    _save_figure(fig, path, dpi)


def _plot_scalar(
    field: np.ndarray,
    title: str,
    path: Path,
    workspace_size_m: tuple[float, float],
    *,
    mask: np.ndarray | None = None,
    colorbar_label: str = "value",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 180,
) -> None:
    """Save a scalar field with optional invalid-domain masking."""

    values = np.asarray(field, dtype=float)
    shown = np.ma.masked_where(~np.asarray(mask, dtype=bool), values) if mask is not None else values
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        shown,
        origin="upper",
        extent=_extent_for_downward_y(workspace_size_m),
        interpolation="bilinear",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m], downward-positive")
    _save_figure(fig, path, dpi)


def save_static_input_figures(
    original_bgr: np.ndarray,
    rectified_bgr: np.ndarray,
    segmentation: SegmentationResult,
    occupancy_maps: OccupancyMaps,
    output_directory: str | Path,
    *,
    workspace_size_m: tuple[float, float],
    dpi: int = 180,
) -> None:
    """Save all perception and occupancy figures shared by forcing methods."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _plot_image(original_bgr, "Original camera image", output / "01_original_image.png", dpi=dpi)
    _plot_image(
        rectified_bgr,
        "Rectified top-down workspace",
        output / "02_rectified_image.png",
        workspace_size_m=workspace_size_m,
        dpi=dpi,
    )
    _plot_binary(
        segmentation.raw_mask > 0,
        "Raw obstacle mask",
        output / "03_raw_mask.png",
        workspace_size_m,
        label="occupied pixel",
        dpi=dpi,
    )
    _plot_binary(
        segmentation.clean_mask > 0,
        "Cleaned obstacle mask",
        output / "04_clean_mask.png",
        workspace_size_m,
        label="occupied pixel",
        dpi=dpi,
    )
    _plot_binary(
        occupancy_maps.occupancy,
        "Uninflated occupancy grid: True means occupied",
        output / "05_occupancy_uninflated.png",
        workspace_size_m,
        label="occupied cell",
        dpi=dpi,
    )
    _plot_binary(
        occupancy_maps.inflated_occupancy,
        f"Inflated occupancy grid: radius = {occupancy_maps.inflation_radius_m:.3f} m",
        output / "06_occupancy_inflated.png",
        workspace_size_m,
        label="occupied cell",
        dpi=dpi,
    )


def _plot_forcing_histogram(record: PoissonRunRecord, path: Path, dpi: int) -> None:
    """Save a histogram of forcing values inside the solve domain."""

    values = np.asarray(record.result.forcing)[record.result.solve_mask]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=min(60, max(10, int(np.sqrt(values.size)))))
    ax.set_title(f"Forcing histogram: {record.forcing_method}")
    ax.set_xlabel("forcing value")
    ax.set_ylabel("cell count")
    ax.grid(True, alpha=0.25)
    _save_figure(fig, path, dpi)


def _plot_surface(record: PoissonRunRecord, path: Path, workspace_size_m: tuple[float, float], dpi: int) -> None:
    """Save a 3D surface of h with occupied cells masked."""

    result = record.result
    x_grid, y_grid = _grid_xy(result.h.shape, workspace_size_m)
    surface = np.ma.masked_where(~result.free_mask, np.asarray(result.h, dtype=float))
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    plotted = ax.plot_surface(x_grid, y_grid, surface, cmap="viridis", linewidth=0.0, antialiased=True)
    fig.colorbar(plotted, ax=ax, shrink=0.65, pad=0.1, label="h")
    ax.set_title(f"Poisson safety function surface: {record.forcing_method}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m], downward-positive")
    ax.set_zlabel("h")
    ax.invert_yaxis()
    _save_figure(fig, path, dpi)


def save_solver_comparison(
    records: Mapping[str, PoissonRunRecord],
    output_directory: str | Path,
    *,
    workspace_size_m: tuple[float, float],
    dpi: int = 150,
) -> None:
    """Compare solver fields, exact residuals, and measured wall times.

    Every solver receives the same occupancy, forcing construction, and grid.
    The first row shows the resulting fields on a common color scale.  The
    second row shows absolute differences from ``sparse_direct`` when present,
    otherwise from the first record.  Separate bar charts report wall time and
    the independently assembled sparse-system residual.
    """

    if not records:
        return
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    names = list(records)
    reference_name = "sparse_direct" if "sparse_direct" in records else names[0]
    reference_h = np.asarray(records[reference_name].result.h, dtype=float)
    valid_values = np.concatenate(
        [np.asarray(record.result.h, dtype=float)[record.result.free_mask] for record in records.values()]
    )
    vmin = float(np.min(valid_values))
    vmax = float(np.max(valid_values))
    differences = [np.abs(np.asarray(records[name].result.h, dtype=float) - reference_h) for name in names]
    difference_max = max(float(np.max(values)) for values in differences)

    fig, axes = plt.subplots(2, len(names), figsize=(5.2 * len(names), 8.5), squeeze=False)
    for column, name in enumerate(names):
        record = records[name]
        field = np.ma.masked_where(~record.result.free_mask, np.asarray(record.result.h, dtype=float))
        image = axes[0, column].imshow(
            field,
            origin="upper",
            extent=_extent_for_downward_y(workspace_size_m),
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, column].set_title(f"{name}: h")
        axes[0, column].set_xlabel("x [m]")
        axes[0, column].set_ylabel("y [m]")
        diff_image = axes[1, column].imshow(
            differences[column],
            origin="upper",
            extent=_extent_for_downward_y(workspace_size_m),
            aspect="auto",
            vmin=0.0,
            vmax=difference_max if difference_max > 0.0 else 1.0,
        )
        axes[1, column].set_title(f"|h - h_{reference_name}|")
        axes[1, column].set_xlabel("x [m]")
        axes[1, column].set_ylabel("y [m]")
    fig.colorbar(image, ax=axes[0, :].tolist(), shrink=0.85, label="h")
    fig.colorbar(diff_image, ax=axes[1, :].tolist(), shrink=0.85, label="absolute difference")
    fig.suptitle("Poisson solver comparison on identical input", fontsize=14)
    _save_figure(fig, output / "01_solver_field_comparison.png", dpi)

    wall_times = [records[name].wall_time_s for name in names]
    residuals = [
        float(records[name].validation.get("residual_max_abs") or np.nan)
        for name in names
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, wall_times)
    ax.set_ylabel("wall time [s]")
    ax.set_title("Poisson solver wall-time comparison")
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, output / "02_solver_wall_time.png", dpi)

    fig, ax = plt.subplots(figsize=(8, 5))
    safe_residuals = np.maximum(np.asarray(residuals, dtype=float), np.finfo(float).tiny)
    ax.bar(names, safe_residuals)
    ax.set_yscale("log")
    ax.set_ylabel("max |A h - b|")
    ax.set_title("Exact assembled-system residual by solver")
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, output / "03_solver_exact_residual.png", dpi)


def save_live_poisson_surface(
    record: PoissonRunRecord,
    path: str | Path,
    *,
    workspace_size_m: tuple[float, float],
    dpi: int = 120,
) -> None:
    """Save one low-rate 3D Poisson snapshot for the live experiment.

    The live dashboard remains entirely OpenCV-based for responsiveness.  This
    helper is called only when a synchronized snapshot is requested, so the
    comparatively expensive Matplotlib surface never blocks every video frame.
    """

    _plot_surface(record, Path(path), workspace_size_m, int(dpi))


def _plot_contours(
    record: PoissonRunRecord,
    path: Path,
    workspace_size_m: tuple[float, float],
    dpi: int,
) -> None:
    """Save level sets of h and an explicit obstacle overlay."""

    result = record.result
    x_grid, y_grid = _grid_xy(result.h.shape, workspace_size_m)
    field = np.ma.masked_where(~result.free_mask, result.h)
    fig, ax = plt.subplots(figsize=(8, 6))
    contours = ax.contour(x_grid, y_grid, field, levels=18)
    ax.clabel(contours, inline=True, fontsize=7)
    ax.contourf(x_grid, y_grid, result.occupancy_mask.astype(float), levels=[0.5, 1.5], alpha=0.5)
    ax.set_xlim(0.0, workspace_size_m[0])
    ax.set_ylim(workspace_size_m[1], 0.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m], downward-positive")
    ax.set_title(f"Poisson h contours: {record.forcing_method}")
    _save_figure(fig, path, dpi)


def _plot_contour_overlay(
    record: PoissonRunRecord,
    rectified_bgr: np.ndarray,
    path: Path,
    workspace_size_m: tuple[float, float],
    dpi: int,
) -> None:
    """Overlay Poisson level sets on the rectified camera image."""

    result = record.result
    x_grid, y_grid = _grid_xy(result.h.shape, workspace_size_m)
    field = np.ma.masked_where(~result.free_mask, result.h)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(
        cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB),
        extent=_extent_for_downward_y(workspace_size_m),
        aspect="auto",
    )
    contours = ax.contour(x_grid, y_grid, field, levels=15, linewidths=1.0)
    ax.clabel(contours, inline=True, fontsize=6)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m], downward-positive")
    ax.set_title(f"Poisson h contours over rectified image: {record.forcing_method}")
    _save_figure(fig, path, dpi)


def _plot_gradient_quiver(
    record: PoissonRunRecord,
    path: Path,
    workspace_size_m: tuple[float, float],
    dpi: int,
) -> None:
    """Save a sub-sampled physical gradient vector field."""

    result = record.result
    if result.grad_h is None:
        return
    x_grid, y_grid = _grid_xy(result.h.shape, workspace_size_m)
    gradient = np.asarray(result.grad_h)
    grad_x = gradient[..., 1]
    grad_y = gradient[..., 0]
    stride = max(1, min(result.h.shape) // 24)
    valid = result.free_mask[::stride, ::stride]
    x_sample = x_grid[::stride, ::stride]
    y_sample = y_grid[::stride, ::stride]
    u_sample = np.where(valid, grad_x[::stride, ::stride], np.nan)
    v_sample = np.where(valid, grad_y[::stride, ::stride], np.nan)
    fig, ax = plt.subplots(figsize=(8, 6))
    background = ax.imshow(
        np.ma.masked_where(~result.free_mask, result.h),
        extent=_extent_for_downward_y(workspace_size_m),
        origin="upper",
        aspect="auto",
        alpha=0.65,
    )
    fig.colorbar(background, ax=ax, label="h")
    ax.quiver(x_sample, y_sample, u_sample, v_sample, angles="xy", scale_units="xy")
    ax.set_xlim(0.0, workspace_size_m[0])
    ax.set_ylim(workspace_size_m[1], 0.0)
    ax.set_title(f"Physical gradient field [dh/dx, dh/dy]: {record.forcing_method}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m], downward-positive")
    _save_figure(fig, path, dpi)


def _plot_residual_history(record: PoissonRunRecord, path: Path, dpi: int) -> None:
    """Save solver residual history when the selected backend exposes one."""

    history = record.result.solver_info.get("residual_history")
    if history is None or len(history) == 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Selected solver did not expose residual history.", ha="center", va="center")
        ax.set_axis_off()
    else:
        values = np.asarray(history, dtype=float)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(np.arange(values.size), np.maximum(values, np.finfo(float).tiny))
        ax.set_xlabel("reported residual sample")
        ax.set_ylabel("residual norm")
        ax.set_title(f"Solver residual history: {record.solver}")
        ax.grid(True, which="both", alpha=0.3)
    _save_figure(fig, path, dpi)


def _plot_timing(record: PoissonRunRecord, path: Path, dpi: int) -> None:
    """Save Poisson stage timings and independent wall time."""

    labels = list(record.result.timing.keys()) + ["total_wall"]
    values = [float(record.result.timing[key]) for key in record.result.timing] + [record.wall_time_s]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    ax.bar_label(bars, fmt="%.4f s", rotation=90, padding=3)
    ax.set_ylabel("time [s]")
    ax.set_title(f"Poisson computation timing: {record.forcing_method} / {record.solver}")
    ax.tick_params(axis="x", rotation=25)
    _save_figure(fig, path, dpi)


def _exact_residual_field(record: PoissonRunRecord) -> np.ndarray:
    """Return the residual of the exact sparse system used by the Safety Box.

    ``PoissonBoxResult.laplacian_h`` is a derivative diagnostic produced with
    numerical differentiation.  Near a Dirichlet frontier it need not match the
    finite-difference stencil used by the linear solver.  The independent
    validation layer reassembles the Safety Box system and stores its true
    algebraic residual, which is the quantity that should be shown in residual
    figures and used for solver acceptance.
    """

    residual = record.validation.get("residual_field")
    if residual is not None:
        return np.asarray(residual, dtype=float)
    result = record.result
    if result.laplacian_h is not None:
        return np.asarray(result.laplacian_h, dtype=float) - np.asarray(result.forcing, dtype=float)
    return np.full_like(np.asarray(result.h, dtype=float), np.nan)


def _plot_method_dashboard(
    record: PoissonRunRecord,
    path: Path,
    workspace_size_m: tuple[float, float],
    dpi: int,
) -> None:
    """Create one compact dashboard for rapid visual inspection."""

    result = record.result
    gradient = np.asarray(result.grad_h) if result.grad_h is not None else None
    gradient_norm = np.linalg.norm(gradient, axis=-1) if gradient is not None else np.zeros_like(result.h)
    residual = _exact_residual_field(record)
    panels = [
        (result.occupancy_mask, "occupancy", "binary", None),
        (result.boundary_mask, "Dirichlet boundary", "binary", None),
        (result.solve_mask, "solve domain", "binary", None),
        (result.forcing, "forcing", "scalar", result.solve_mask),
        (result.h, "h", "scalar", result.omega_union_boundary_mask),
        (gradient_norm, "gradient norm", "scalar", result.omega_union_boundary_mask),
        (
            result.laplacian_h if result.laplacian_h is not None else np.zeros_like(result.h),
            "Laplacian diagnostic",
            "scalar",
            result.omega_union_boundary_mask,
        ),
        (residual, "Exact assembled residual", "scalar", result.solve_mask),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for ax, (array, title, kind, panel_mask) in zip(axes.ravel(), panels):
        shown = np.asarray(array)
        if kind == "scalar" and panel_mask is not None:
            shown = np.ma.masked_where(~np.asarray(panel_mask, dtype=bool), shown)
        image = ax.imshow(
            shown,
            origin="upper",
            extent=_extent_for_downward_y(workspace_size_m),
            interpolation="nearest" if kind == "binary" else "bilinear",
            aspect="auto",
        )
        fig.colorbar(image, ax=ax, shrink=0.75)
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.suptitle(f"Poisson diagnostics: {record.forcing_method} / {record.solver}")
    _save_figure(fig, path, dpi)


def save_poisson_diagnostics(
    record: PoissonRunRecord,
    rectified_bgr: np.ndarray,
    output_directory: str | Path,
    *,
    workspace_size_m: tuple[float, float],
    dpi: int = 180,
) -> None:
    """Save the complete requested set of per-method Poisson diagnostics."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result = record.result
    mask = result.omega_union_boundary_mask

    _plot_binary(result.free_mask, "Free-space mask", output / "07_free_mask.png", workspace_size_m, dpi=dpi)
    _plot_binary(
        result.boundary_mask,
        "Dirichlet boundary mask",
        output / "08_boundary_mask.png",
        workspace_size_m,
        dpi=dpi,
    )
    _plot_binary(result.solve_mask, "Poisson solve mask", output / "09_solve_mask.png", workspace_size_m, dpi=dpi)
    _plot_binary(
        result.omega_union_boundary_mask,
        "Omega union boundary",
        output / "10_omega_union_boundary.png",
        workspace_size_m,
        dpi=dpi,
    )
    _plot_scalar(
        result.forcing,
        f"Poisson forcing: {record.forcing_method}",
        output / "11_forcing_map.png",
        workspace_size_m,
        mask=result.solve_mask,
        colorbar_label="f_P",
        dpi=dpi,
    )
    _plot_forcing_histogram(record, output / "12_forcing_histogram.png", dpi)
    _plot_scalar(
        result.h,
        f"Poisson safety function h: {record.forcing_method}",
        output / "13_h_heatmap.png",
        workspace_size_m,
        mask=mask,
        colorbar_label="h",
        dpi=dpi,
    )
    _plot_surface(record, output / "14_h_surface_3d.png", workspace_size_m, dpi)
    _plot_contours(record, output / "15_h_contours.png", workspace_size_m, dpi)
    _plot_contour_overlay(record, rectified_bgr, output / "16_h_contours_on_image.png", workspace_size_m, dpi)
    _plot_gradient_quiver(record, output / "17_gradient_quiver.png", workspace_size_m, dpi)

    if result.grad_h is not None:
        gradient_yx = np.asarray(result.grad_h)
        gradient_x = gradient_yx[..., 1]
        gradient_y = gradient_yx[..., 0]
        gradient_norm = np.linalg.norm(gradient_yx, axis=-1)
        _plot_scalar(
            gradient_norm,
            "Gradient norm",
            output / "18_gradient_norm.png",
            workspace_size_m,
            mask=mask,
            colorbar_label="norm(grad h)",
            dpi=dpi,
        )
        _plot_scalar(
            gradient_x,
            "Physical gradient component dh/dx",
            output / "19_gradient_x.png",
            workspace_size_m,
            mask=mask,
            colorbar_label="dh/dx",
            dpi=dpi,
        )
        _plot_scalar(
            gradient_y,
            "Physical gradient component dh/dy",
            output / "20_gradient_y.png",
            workspace_size_m,
            mask=mask,
            colorbar_label="dh/dy",
            dpi=dpi,
        )

    if result.hessian_h is not None:
        hessian_yx = np.asarray(result.hessian_h)
        h_xx = hessian_yx[..., 1, 1]
        h_xy = hessian_yx[..., 1, 0]
        h_yy = hessian_yx[..., 0, 0]
        trace = h_xx + h_yy
        determinant = h_xx * h_yy - h_xy * h_xy
        eigen_min = np.linalg.eigvalsh(hessian_yx)[..., 0]
        _plot_scalar(h_xx, "Hessian component h_xx", output / "21_hessian_xx.png", workspace_size_m, mask=mask, colorbar_label="h_xx", dpi=dpi)
        _plot_scalar(h_xy, "Hessian component h_xy", output / "22_hessian_xy.png", workspace_size_m, mask=mask, colorbar_label="h_xy", dpi=dpi)
        _plot_scalar(h_yy, "Hessian component h_yy", output / "23_hessian_yy.png", workspace_size_m, mask=mask, colorbar_label="h_yy", dpi=dpi)
        _plot_scalar(trace, "Hessian trace", output / "24_hessian_trace.png", workspace_size_m, mask=mask, colorbar_label="trace(H)", dpi=dpi)
        _plot_scalar(determinant, "Hessian determinant", output / "25_hessian_determinant.png", workspace_size_m, mask=mask, colorbar_label="det(H)", dpi=dpi)
        _plot_scalar(eigen_min, "Minimum Hessian eigenvalue", output / "26_hessian_eigenvalue_min.png", workspace_size_m, mask=mask, colorbar_label="lambda_min(H)", dpi=dpi)

    if result.laplacian_h is not None:
        laplacian = np.asarray(result.laplacian_h)
        _plot_scalar(laplacian, "Laplacian of h", output / "27_laplacian_h.png", workspace_size_m, mask=mask, colorbar_label="Delta h", dpi=dpi)

    # Plot the exact residual of the sparse linear system assembled by the
    # Safety Box.  This avoids presenting the separate ``numpy.gradient``
    # Laplacian diagnostic as though it were the solver's algebraic residual.
    residual = _exact_residual_field(record)
    _plot_scalar(
        residual,
        "Exact Poisson system residual",
        output / "28_poisson_residual.png",
        workspace_size_m,
        mask=result.solve_mask,
        colorbar_label="A h - b",
        dpi=dpi,
    )

    _plot_residual_history(record, output / "29_solver_residual_history.png", dpi)
    _plot_timing(record, output / "30_timing_summary.png", dpi)
    _plot_method_dashboard(record, output / "31_method_dashboard.png", workspace_size_m, dpi)


def save_forcing_comparison(
    records: Mapping[str, PoissonRunRecord],
    output_directory: str | Path,
    *,
    workspace_size_m: tuple[float, float],
    dpi: int = 180,
) -> None:
    """Save common-scale side-by-side h and forcing comparisons."""

    if not records:
        return
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    methods = list(records)
    h_values = np.concatenate([record.result.h[record.result.solve_mask] for record in records.values()])
    forcing_values = np.concatenate(
        [record.result.forcing[record.result.solve_mask] for record in records.values()]
    )
    h_limits = (float(np.min(h_values)), float(np.max(h_values)))
    forcing_limits = (float(np.min(forcing_values)), float(np.max(forcing_values)))

    columns = 2
    rows = int(np.ceil(len(methods) / columns))
    for field_name, limits, filename, label in (
        ("h", h_limits, "32_forcing_methods_h_comparison.png", "h"),
        ("forcing", forcing_limits, "33_forcing_methods_forcing_comparison.png", "f_P"),
    ):
        fig, axes = plt.subplots(rows, columns, figsize=(12, 5.5 * rows), squeeze=False, constrained_layout=True)
        last_image = None
        for ax, method in zip(axes.ravel(), methods):
            record = records[method]
            field = np.asarray(getattr(record.result, field_name))
            mask = record.result.solve_mask if field_name == "forcing" else record.result.omega_union_boundary_mask
            shown = np.ma.masked_where(~mask, field)
            last_image = ax.imshow(
                shown,
                origin="upper",
                extent=_extent_for_downward_y(workspace_size_m),
                aspect="auto",
                interpolation="bilinear",
                vmin=limits[0],
                vmax=limits[1],
            )
            ax.set_title(method)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
        for ax in axes.ravel()[len(methods):]:
            ax.set_axis_off()
        if last_image is not None:
            fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.82, label=label)
        fig.suptitle(f"Common-scale comparison of {label} across forcing methods")
        _save_figure(fig, output / filename, dpi)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = methods
    solve_times = [float(records[method].result.timing.get("solve", np.nan)) for method in methods]
    wall_times = [float(records[method].wall_time_s) for method in methods]
    index = np.arange(len(methods))
    width = 0.38
    ax.bar(index - width / 2, solve_times, width, label="Safety Box solve stage")
    ax.bar(index + width / 2, wall_times, width, label="total wall time")
    ax.set_xticks(index, labels)
    ax.set_ylabel("time [s]")
    ax.set_title("Forcing-method timing comparison")
    ax.legend()
    _save_figure(fig, output / "34_forcing_methods_timing_comparison.png", dpi)


def normalize_for_colormap(field: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Normalize a copy of a scalar field to uint8 for visualization only."""

    values = np.asarray(field, dtype=float)
    mask = np.isfinite(values) if valid_mask is None else np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    output = np.zeros(values.shape, dtype=np.uint8)
    if not np.any(mask):
        return output
    minimum = float(np.min(values[mask]))
    maximum = float(np.max(values[mask]))
    if maximum - minimum <= np.finfo(float).eps:
        output[mask] = 127
    else:
        output[mask] = np.clip(255.0 * (values[mask] - minimum) / (maximum - minimum), 0.0, 255.0).astype(np.uint8)
    return output


def colorize_scalar(field: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Colorize a scalar field without modifying the numerical field used by control."""

    normalized = normalize_for_colormap(field, valid_mask)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    if valid_mask is not None:
        colored[~np.asarray(valid_mask, dtype=bool)] = 0
    return colored


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, *, alpha: float = 0.45) -> np.ndarray:
    """Overlay occupied pixels in red on an image."""

    canvas = image_bgr.copy()
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != canvas.shape[:2]:
        binary = cv2.resize(binary.astype(np.uint8), (canvas.shape[1], canvas.shape[0]), cv2.INTER_NEAREST) > 0
    if not np.any(binary):
        return canvas
    overlay = np.zeros_like(canvas)
    overlay[..., 2] = 255
    blended = cv2.addWeighted(canvas[binary], 1.0 - alpha, overlay[binary], alpha, 0.0)
    if blended is not None:
        canvas[binary] = blended
    return canvas


def overlay_h_contours(image_bgr: np.ndarray, h: np.ndarray, valid_mask: np.ndarray, *, levels: int = 8) -> np.ndarray:
    """Draw approximate h level sets on a BGR image using OpenCV contours."""

    canvas = image_bgr.copy()
    normalized = normalize_for_colormap(h, valid_mask)
    resized = cv2.resize(normalized, (canvas.shape[1], canvas.shape[0]), cv2.INTER_LINEAR)
    valid_resized = cv2.resize(
        np.asarray(valid_mask, dtype=np.uint8),
        (canvas.shape[1], canvas.shape[0]),
        cv2.INTER_NEAREST,
    ) > 0
    for threshold in np.linspace(20, 235, max(2, int(levels))).astype(np.uint8):
        binary = ((resized >= threshold) & valid_resized).astype(np.uint8) * 255
        contours, _hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _fit_panel(image: np.ndarray, size: tuple[int, int], title: str) -> np.ndarray:
    """Resize an image into a dashboard panel and add a title band."""

    width, height = size
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    panel = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (width, 28), (0, 0, 0), -1)
    cv2.putText(panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def render_live_dashboard(
    *,
    original_bgr: np.ndarray,
    rectified_bgr: np.ndarray,
    obstacle_mask: np.ndarray,
    occupancy: np.ndarray,
    poisson_result: Any | None,
    metrics: Mapping[str, Any],
    warnings: list[str] | None = None,
    panel_size: tuple[int, int] = (480, 320),
) -> np.ndarray:
    """Compose a responsive six-panel OpenCV dashboard."""

    warnings = warnings or []
    mask_overlay = overlay_mask(rectified_bgr, obstacle_mask > 0)
    occupancy_image = (np.asarray(occupancy, dtype=np.uint8) * 255)
    occupancy_image = cv2.cvtColor(occupancy_image, cv2.COLOR_GRAY2BGR)

    if poisson_result is None:
        h_image = np.zeros_like(occupancy_image)
        gradient_image = np.zeros_like(occupancy_image)
        contour_overlay = rectified_bgr.copy()
        cv2.putText(h_image, "Waiting for first valid Poisson solve", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        h_image = colorize_scalar(poisson_result.h, poisson_result.omega_union_boundary_mask)
        contour_overlay = overlay_h_contours(
            rectified_bgr,
            poisson_result.h,
            poisson_result.omega_union_boundary_mask,
        )
        if poisson_result.grad_h is not None:
            gradient_norm = np.linalg.norm(np.asarray(poisson_result.grad_h), axis=-1)
            gradient_image = colorize_scalar(gradient_norm, poisson_result.free_mask)
        else:
            gradient_image = np.zeros_like(h_image)

    panels = [
        _fit_panel(original_bgr, panel_size, "Original source"),
        _fit_panel(rectified_bgr, panel_size, "Rectified workspace"),
        _fit_panel(mask_overlay, panel_size, "Obstacle segmentation"),
        _fit_panel(occupancy_image, panel_size, "Filtered and inflated occupancy"),
        _fit_panel(h_image, panel_size, "Poisson h visualization"),
        _fit_panel(contour_overlay, panel_size, "h contours on camera image"),
    ]
    first_row = np.hstack(panels[:3])
    second_row = np.hstack(panels[3:])
    dashboard = np.vstack([first_row, second_row])

    text_x = 10
    text_y = dashboard.shape[0] - 12
    metric_parts = [
        f"capture {float(metrics.get('capture_fps', 0.0)):.1f} FPS",
        f"display {float(metrics.get('display_fps', 0.0)):.1f} FPS",
        f"Poisson {float(metrics.get('poisson_updates_per_s', 0.0)):.2f} Hz",
        f"solve {1000.0 * float(metrics.get('last_solve_time_s', 0.0)):.1f} ms",
        f"field age {1000.0 * float(metrics.get('field_age_s', 0.0)):.1f} ms",
        f"residual {float(metrics.get('poisson_residual', np.nan)):.3e}",
        f"grid {metrics.get('grid_shape', 'unknown')}",
        f"forcing {metrics.get('forcing_method', 'unknown')}",
        f"solver {metrics.get('solver', 'unknown')}",
    ]
    cv2.rectangle(dashboard, (0, dashboard.shape[0] - 35), (dashboard.shape[1], dashboard.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        dashboard,
        " | ".join(metric_parts),
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for warning_index, warning in enumerate(warnings[:3]):
        cv2.putText(
            dashboard,
            warning,
            (10, 55 + 28 * warning_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return dashboard
