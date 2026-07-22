"""Validated execution helpers for the external Poisson Safety Box.

The PDE discretization and solvers remain entirely inside ``poisson_safety_box``.
This module translates experiment dictionaries into the Safety Box dataclass,
performs independent consistency checks, and persists reproducible artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np

from poisson_safety_box import PoissonBoxConfig, PoissonBoxResult, PoissonSafetyBox
from poisson_safety_box.solvers.laplacian_matrix import build_laplacian_system

from .io_utils import save_json, save_yaml


FORCING_METHODS = ("constant", "distance", "average_flux", "guidance")
POISSON_SOLVERS = ("sor", "sparse_direct", "conjugate_gradient")


class ValidationSummary(dict[str, Any]):
    """Dictionary-compatible validation report with an explicit ``valid`` flag."""

    @property
    def valid(self) -> bool:
        """Return whether all configured numerical checks passed."""

        return bool(self.get("valid", False))


@dataclass(frozen=True)
class PoissonRunRecord:
    """One Poisson solve, its exact configuration, and independent validation."""

    forcing_method: str
    solver: str
    config: PoissonBoxConfig
    result: PoissonBoxResult
    wall_time_s: float
    validation: ValidationSummary


@dataclass(frozen=True)
class PoissonRunBundle:
    """Collection of forcing-method records with one selected control field."""

    records: dict[str, PoissonRunRecord]
    selected_method: str

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("A PoissonRunBundle requires at least one solve record.")
        if self.selected_method not in self.records:
            raise ValueError(
                f"Selected forcing method {self.selected_method!r} is not present in records."
            )

    @property
    def selected(self) -> PoissonRunRecord:
        """Return the record used by the CBF demonstration."""

        return self.records[self.selected_method]


def _merge_dataclass_attributes(target: Any, values: Mapping[str, Any]) -> None:
    """Recursively apply known dictionary keys to a dataclass-like object."""

    for key, value in values.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(value, Mapping) and hasattr(current, "__dataclass_fields__"):
            _merge_dataclass_attributes(current, value)
        else:
            setattr(target, key, value)


def build_poisson_config(
    config_data: Mapping[str, Any],
    *,
    grid_spacing_yx: tuple[float, float],
    forcing_method: str | None = None,
    solver: str | None = None,
    live_mode: bool = False,
) -> PoissonBoxConfig:
    """Translate an experiment mapping into ``PoissonBoxConfig``.

    Unknown experiment-only keys, such as validation tolerances or solver lists,
    are deliberately ignored rather than forwarded into the external package.
    """

    cfg = PoissonBoxConfig()
    cfg.grid_spacing = tuple(float(value) for value in grid_spacing_yx)
    cfg.forcing_method = str(
        forcing_method
        or config_data.get("forcing_method")
        or config_data.get("primary_forcing_method")
        or "constant"
    )
    cfg.solver = str(solver or config_data.get("solver", "sparse_direct"))
    cfg.boundary_value = float(config_data.get("boundary_value", 0.0))
    cfg.outer_boundary_as_dirichlet = bool(config_data.get("outer_boundary_as_dirichlet", True))
    cfg.compute_gradient = bool(config_data.get("compute_gradient", True))
    cfg.compute_hessian = bool(config_data.get("compute_hessian", not live_mode))
    cfg.compute_laplacian_check = bool(config_data.get("compute_laplacian_check", not live_mode))
    cfg.plot = False
    cfg.save_outputs = False

    for nested_name in (
        "constant",
        "distance",
        "average_flux",
        "guidance",
        "sor",
        "conjugate_gradient",
    ):
        nested_values = config_data.get(nested_name, {}) or {}
        if not isinstance(nested_values, Mapping):
            raise ValueError(f"Poisson section {nested_name!r} must be a mapping.")
        _merge_dataclass_attributes(getattr(cfg, nested_name), nested_values)

    cfg.validate()
    return cfg


def validate_poisson_input(occupancy: np.ndarray) -> dict[str, Any]:
    """Validate the strict ``True=occupied`` two-dimensional convention."""

    array = np.asarray(occupancy)
    if array.ndim != 2:
        raise ValueError(f"The experiments require a 2D occupancy grid, received {array.shape}.")
    if array.dtype != bool:
        raise TypeError("Occupancy must be boolean: True means occupied and False means free.")
    if array.size == 0:
        raise ValueError("Occupancy grid is empty.")
    occupied_cells = int(np.count_nonzero(array))
    free_cells = int(array.size - occupied_cells)
    if free_cells == 0:
        raise ValueError("Occupancy is fully occupied; no Poisson domain remains.")
    return {
        "shape_yx": list(array.shape),
        "occupied_cells": occupied_cells,
        "free_cells": free_cells,
        "occupied_fraction": float(occupied_cells / array.size),
        "fully_free": occupied_cells == 0,
    }


def central_stencil_residual(
    h: np.ndarray,
    forcing: np.ndarray,
    solve_mask: np.ndarray,
    spacing_yx: tuple[float, float],
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    """Evaluate ``Delta h - f`` using the same centered 5-point stencil.

    Values outside ``solve_mask`` are stored as ``NaN``.  The function assumes
    the mask supplied by the Safety Box excludes outer rows and columns, but it
    still guards the array edges to make diagnostics robust to malformed inputs.
    """

    h_array = np.asarray(h, dtype=float)
    forcing_array = np.asarray(forcing, dtype=float)
    mask = np.asarray(solve_mask, dtype=bool)
    if h_array.ndim != 2 or forcing_array.shape != h_array.shape or mask.shape != h_array.shape:
        raise ValueError("h, forcing, and solve_mask must be matching two-dimensional arrays.")
    dy, dx = (float(spacing_yx[0]), float(spacing_yx[1]))
    if dy <= 0.0 or dx <= 0.0:
        raise ValueError("Grid spacing must be positive.")

    residual = np.full(h_array.shape, np.nan, dtype=float)
    interior = mask.copy()
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    rows, cols = np.where(interior)
    if rows.size:
        laplacian = (
            (h_array[rows + 1, cols] - 2.0 * h_array[rows, cols] + h_array[rows - 1, cols]) / (dy * dy)
            + (h_array[rows, cols + 1] - 2.0 * h_array[rows, cols] + h_array[rows, cols - 1]) / (dx * dx)
        )
        residual[rows, cols] = laplacian - forcing_array[rows, cols]
        values = residual[rows, cols]
        stats: dict[str, float | int | None] = {
            "count": int(values.size),
            "max_abs": float(np.max(np.abs(values))),
            "mean_abs": float(np.mean(np.abs(values))),
            "rms": float(np.linalg.norm(values) / np.sqrt(values.size)),
        }
    else:
        stats = {"count": 0, "max_abs": None, "mean_abs": None, "rms": None}
    return residual, stats


def assembled_system_residual(
    h: np.ndarray,
    forcing: np.ndarray,
    solve_mask: np.ndarray,
    spacing_yx: tuple[float, float],
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    """Evaluate the residual of the exact sparse system assembled by the Safety Box.

    This is the load-bearing numerical check.  The external package solves the
    positive-definite system ``-Delta h = -f``.  Reassembling that same system
    avoids confusing the validation with the separate ``numpy.gradient``
    Laplacian diagnostic stored by the Safety Box.
    """

    h_array = np.asarray(h, dtype=float)
    forcing_array = np.asarray(forcing, dtype=float)
    mask = np.asarray(solve_mask, dtype=bool)
    if h_array.ndim != 2 or forcing_array.shape != h_array.shape or mask.shape != h_array.shape:
        raise ValueError("h, forcing, and solve_mask must be matching two-dimensional arrays.")
    if not np.any(mask):
        return np.full(h_array.shape, np.nan), {
            "count": 0,
            "max_abs": None,
            "mean_abs": None,
            "rms": None,
        }

    # The solved field already contains the effective Dirichlet value on every
    # non-solve node.  Passing h as boundary_values therefore also supports a
    # nonzero configured boundary value without adding controller assumptions.
    matrix, right_hand_side, _index_map, free_cells, _metadata = build_laplacian_system(
        forcing_array,
        mask,
        h_array,
        tuple(float(value) for value in spacing_yx),
    )
    unknown_vector = np.asarray([h_array[tuple(cell)] for cell in free_cells], dtype=float)
    residual_vector = np.asarray(matrix @ unknown_vector - right_hand_side, dtype=float)
    residual_field = np.full(h_array.shape, np.nan, dtype=float)
    for cell, value in zip(free_cells, residual_vector, strict=True):
        residual_field[tuple(cell)] = value
    absolute = np.abs(residual_vector)
    return residual_field, {
        "count": int(residual_vector.size),
        "max_abs": float(np.max(absolute)),
        "mean_abs": float(np.mean(absolute)),
        "rms": float(np.linalg.norm(residual_vector) / np.sqrt(residual_vector.size)),
    }


def validate_poisson_result(
    result: PoissonBoxResult,
    *,
    spacing_yx: tuple[float, float],
    boundary_tolerance: float = 1.0e-8,
    residual_tolerance: float = 1.0e-5,
) -> ValidationSummary:
    """Check shapes, finite values, boundary data, and discrete PDE residual."""

    h = np.asarray(result.h, dtype=float)
    solve_mask = np.asarray(result.solve_mask, dtype=bool)
    boundary_mask = np.asarray(result.boundary_mask, dtype=bool)
    if h.shape != solve_mask.shape or h.shape != boundary_mask.shape:
        raise ValueError("Poisson result masks do not match the h field shape.")
    if not np.any(solve_mask):
        raise ValueError("Poisson result contains an empty solve mask.")

    finite_h = bool(np.all(np.isfinite(h[solve_mask])))
    finite_gradient = result.grad_h is None or bool(
        np.all(np.isfinite(np.asarray(result.grad_h)[solve_mask]))
    )
    finite_hessian = result.hessian_h is None or bool(
        np.all(np.isfinite(np.asarray(result.hessian_h)[solve_mask]))
    )
    boundary_abs_max = float(np.max(np.abs(h[boundary_mask]))) if np.any(boundary_mask) else 0.0
    stencil_residual_field, stencil_residual_stats = central_stencil_residual(
        h,
        np.asarray(result.forcing, dtype=float),
        solve_mask,
        spacing_yx,
    )
    assembled_residual_field, assembled_residual_stats = assembled_system_residual(
        h,
        np.asarray(result.forcing, dtype=float),
        solve_mask,
        spacing_yx,
    )
    residual_max = assembled_residual_stats["max_abs"]
    residual_ok = residual_max is not None and float(residual_max) <= float(residual_tolerance)
    boundary_ok = boundary_abs_max <= float(boundary_tolerance)
    solver_status = str(result.solver_info.get("status", result.solver_info.get("converged", "unknown")))
    solver_ok = bool(result.solver_info.get("converged", True)) and "fail" not in solver_status.lower()
    valid = finite_h and finite_gradient and finite_hessian and boundary_ok and residual_ok and solver_ok

    return ValidationSummary(
        valid=valid,
        finite_h=finite_h,
        finite_gradient=finite_gradient,
        finite_hessian=finite_hessian,
        boundary_ok=boundary_ok,
        residual_ok=residual_ok,
        solver_ok=solver_ok,
        h_min_solve=float(np.min(h[solve_mask])),
        h_max_solve=float(np.max(h[solve_mask])),
        h_mean_solve=float(np.mean(h[solve_mask])),
        boundary_abs_max=boundary_abs_max,
        residual_max_abs=residual_max,
        residual_l2=assembled_residual_stats["rms"],
        residual_count=assembled_residual_stats["count"],
        central_stencil_residual_max_abs=stencil_residual_stats["max_abs"],
        central_stencil_residual_l2=stencil_residual_stats["rms"],
        solve_cells=int(np.count_nonzero(solve_mask)),
        boundary_cells=int(np.count_nonzero(boundary_mask)),
        solver_status=solver_status,
        tolerances={
            "boundary": float(boundary_tolerance),
            "residual": float(residual_tolerance),
        },
        residual_field=assembled_residual_field,
        central_stencil_residual_field=stencil_residual_field,
    )


def run_poisson(
    occupancy: np.ndarray,
    *,
    grid_spacing_yx: tuple[float, float],
    poisson_config: Mapping[str, Any],
    forcing_method: str | None = None,
    solver: str | None = None,
    live_mode: bool = False,
) -> PoissonRunRecord:
    """Execute one solve through ``PoissonSafetyBox`` and validate it."""

    input_validation = validate_poisson_input(occupancy)
    effective_config = build_poisson_config(
        poisson_config,
        grid_spacing_yx=grid_spacing_yx,
        forcing_method=forcing_method,
        solver=solver,
        live_mode=live_mode,
    )
    start = perf_counter()
    result = PoissonSafetyBox(effective_config).compute(np.asarray(occupancy, dtype=bool))
    wall_time = perf_counter() - start
    result_validation = validate_poisson_result(
        result,
        spacing_yx=grid_spacing_yx,
        boundary_tolerance=float(poisson_config.get("validation_boundary_tolerance", 1.0e-8)),
        residual_tolerance=float(poisson_config.get("validation_residual_tolerance", 1.0e-5)),
    )
    validation = ValidationSummary(result_validation)
    validation["input"] = input_validation
    validation["result"] = {
        key: value
        for key, value in result_validation.items()
        if key not in {"residual_field", "central_stencil_residual_field"}
    }
    return PoissonRunRecord(
        forcing_method=effective_config.forcing_method,
        solver=effective_config.solver,
        config=effective_config,
        result=result,
        wall_time_s=wall_time,
        validation=validation,
    )


def run_poisson_once(
    occupancy: np.ndarray,
    *,
    spacing_yx: tuple[float, float],
    settings: Mapping[str, Any],
    forcing_method: str,
    solver: str,
) -> PoissonRunRecord:
    """Compatibility wrapper with the compact signature used in unit tests."""

    return run_poisson(
        occupancy,
        grid_spacing_yx=spacing_yx,
        poisson_config=settings,
        forcing_method=forcing_method,
        solver=solver,
        live_mode=False,
    )


def run_forcing_comparison(
    occupancy: np.ndarray,
    *,
    grid_spacing_yx: tuple[float, float],
    poisson_config: Mapping[str, Any],
    forcing_methods: Iterable[str] = FORCING_METHODS,
    solver: str | None = None,
) -> dict[str, PoissonRunRecord]:
    """Run the same inflated occupancy through multiple forcing methods."""

    records: dict[str, PoissonRunRecord] = {}
    for method in forcing_methods:
        method_text = str(method)
        if method_text not in FORCING_METHODS:
            raise ValueError(f"Unsupported forcing method: {method_text!r}")
        records[method_text] = run_poisson(
            occupancy,
            grid_spacing_yx=grid_spacing_yx,
            poisson_config=poisson_config,
            forcing_method=method_text,
            solver=solver,
            live_mode=False,
        )
    return records


def run_poisson_methods(
    occupancy: np.ndarray,
    *,
    spacing_yx: tuple[float, float],
    settings: Mapping[str, Any],
    forcing_methods: Iterable[str],
    selected_method: str,
    solver: str,
) -> PoissonRunBundle:
    """Run and bundle multiple forcing constructions for the static experiment."""

    records = run_forcing_comparison(
        occupancy,
        grid_spacing_yx=spacing_yx,
        poisson_config=settings,
        forcing_methods=forcing_methods,
        solver=solver,
    )
    selected = str(selected_method)
    if selected not in records:
        raise ValueError(
            f"Selected method {selected!r} was not included in forcing_methods={list(records)}."
        )
    return PoissonRunBundle(records=records, selected_method=selected)


def compare_poisson_solvers(
    occupancy: np.ndarray,
    *,
    spacing_yx: tuple[float, float],
    settings: Mapping[str, Any],
    forcing_method: str,
    solvers: Iterable[str] = POISSON_SOLVERS,
) -> dict[str, PoissonRunRecord]:
    """Compatibility wrapper for an identical-input solver benchmark."""

    return benchmark_poisson_solvers(
        occupancy,
        grid_spacing_yx=spacing_yx,
        poisson_config=settings,
        forcing_method=forcing_method,
        solvers=solvers,
    )


def benchmark_poisson_solvers(
    occupancy: np.ndarray,
    *,
    grid_spacing_yx: tuple[float, float],
    poisson_config: Mapping[str, Any],
    forcing_method: str,
    solvers: Iterable[str] = POISSON_SOLVERS,
) -> dict[str, PoissonRunRecord]:
    """Benchmark supported solvers on identical occupancy and forcing."""

    records: dict[str, PoissonRunRecord] = {}
    for solver in solvers:
        solver_text = str(solver)
        if solver_text not in POISSON_SOLVERS:
            raise ValueError(f"Unsupported Poisson solver: {solver_text!r}")
        records[solver_text] = run_poisson(
            occupancy,
            grid_spacing_yx=grid_spacing_yx,
            poisson_config=poisson_config,
            forcing_method=forcing_method,
            solver=solver_text,
            live_mode=False,
        )
    return records


def save_poisson_record(record: PoissonRunRecord, output_directory: str | Path) -> None:
    """Persist arrays, scalar diagnostics, and the exact effective configuration."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    record.result.save_npz(output / "result.npz")
    record.result.save_summary_json(output / "safety_box_summary.json")
    save_yaml(output / "effective_poisson_config.yaml", record.config.to_dict())
    residual_arrays = {
        name: np.asarray(record.validation[name], dtype=float)
        for name in ("residual_field", "central_stencil_residual_field")
        if name in record.validation
    }
    if residual_arrays:
        np.savez_compressed(output / "validation_residuals.npz", **residual_arrays)
    validation_without_array = {
        key: value
        for key, value in record.validation.items()
        if key not in {"residual_field", "central_stencil_residual_field"}
    }
    save_json(
        output / "run_summary.json",
        {
            "forcing_method": record.forcing_method,
            "solver": record.solver,
            "wall_time_s": record.wall_time_s,
            "timing": record.result.timing,
            "solver_info": record.result.solver_info,
            "diagnostics": record.result.diagnostics,
            "validation": validation_without_array,
        },
    )


__all__ = [
    "FORCING_METHODS",
    "POISSON_SOLVERS",
    "PoissonRunBundle",
    "PoissonRunRecord",
    "ValidationSummary",
    "assembled_system_residual",
    "benchmark_poisson_solvers",
    "build_poisson_config",
    "central_stencil_residual",
    "compare_poisson_solvers",
    "run_forcing_comparison",
    "run_poisson",
    "run_poisson_methods",
    "run_poisson_once",
    "save_poisson_record",
    "validate_poisson_input",
    "validate_poisson_result",
]
