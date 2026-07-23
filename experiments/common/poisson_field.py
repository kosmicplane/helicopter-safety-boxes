"""Poisson field construction, sampling, and solver-comparison utilities.

The module adapts :mod:`poisson_safety_box` to the three experiment modes.
It never creates control constraints; the CBF safety box owns that task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping
import json

import numpy as np

from cbf_safety_box import SafetySample
from poisson_safety_box import PoissonBoxConfig, PoissonSafetyBox
from poisson_safety_box.interpolation.trilinear import interpolate_grid
from poisson_safety_box.solvers.laplacian_matrix import build_laplacian_system


@dataclass(frozen=True, slots=True)
class PoissonField:
    """Uniform-grid Poisson result with local interpolation in metric coordinates."""

    result: Any
    spacing: tuple[float, ...]
    control_scale: float
    forcing_method: str
    solver: str

    @property
    def dimension(self) -> int:
        return len(self.spacing)

    @property
    def h(self) -> np.ndarray:
        return np.asarray(self.result.h, dtype=float) / self.control_scale

    @property
    def grad_h(self) -> np.ndarray:
        return np.asarray(self.result.grad_h, dtype=float) / self.control_scale

    @property
    def hessian_h(self) -> np.ndarray:
        return np.asarray(self.result.hessian_h, dtype=float) / self.control_scale

    @property
    def raw_h(self) -> np.ndarray:
        return np.asarray(self.result.h, dtype=float)

    @property
    def occupancy(self) -> np.ndarray:
        """Boolean occupancy mask used to construct the field."""

        return np.asarray(self.result.occupancy_mask, dtype=bool)

    @property
    def boundary_mask(self) -> np.ndarray:
        """Dirichlet frontier extracted from the occupancy representation."""

        return np.asarray(self.result.boundary_mask, dtype=bool)

    @property
    def forcing(self) -> np.ndarray:
        """Configured Poisson forcing field."""

        return np.asarray(self.result.forcing, dtype=float)

    def sample(self, point: np.ndarray, *, partial_h_t: float = 0.0) -> SafetySample | None:
        """Interpolate ``h``, ``Dh``, and ``D²h`` at one physical point."""

        coordinate = np.asarray(point, dtype=float).reshape(-1)
        if coordinate.size != self.dimension:
            raise ValueError(
                f"Expected a {self.dimension}-D point, received {coordinate.size} entries."
            )
        h, ok_h = interpolate_grid(self.h, coordinate, self.spacing)
        grad, ok_grad = interpolate_grid(self.grad_h, coordinate, self.spacing)
        hessian, ok_hessian = interpolate_grid(
            self.hessian_h,
            coordinate,
            self.spacing,
        )
        if not (ok_h and ok_grad and ok_hessian):
            return None
        if not (
            np.isfinite(h)
            and np.all(np.isfinite(grad))
            and np.all(np.isfinite(hessian))
        ):
            return None
        return SafetySample(
            h=float(h),
            grad_h=np.asarray(grad, dtype=float),
            hessian_h=np.asarray(hessian, dtype=float),
            partial_h_t=float(partial_h_t),
            metadata={
                "forcing_method": self.forcing_method,
                "solver": self.solver,
                "control_scale": self.control_scale,
            },
        )

    def save(self, directory: str | Path, *, stem: str = "poisson_field") -> None:
        """Persist field arrays and scalar metadata."""

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        self.result.save_npz(output / f"{stem}.npz")
        summary_path = output / f"{stem}_summary.json"
        self.result.save_summary_json(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "forcing_method": self.forcing_method,
                "solver": self.solver,
                "control_scale": self.control_scale,
                "grid_spacing": list(self.spacing),
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def poisson_config_from_mapping(
    data: Mapping[str, Any],
    *,
    spacing: Iterable[float],
    forcing_method: str | None = None,
    solver: str | None = None,
) -> PoissonBoxConfig:
    """Construct one validated Poisson configuration from the central YAML mapping."""

    config = PoissonBoxConfig()
    config.grid_spacing = tuple(float(value) for value in spacing)
    config.forcing_method = str(forcing_method or data.get("forcing_method", "constant"))
    config.solver = str(solver or data.get("solver", "conjugate_gradient"))
    config.outer_boundary_as_dirichlet = bool(
        data.get("outer_boundary_as_dirichlet", True)
    )
    config.compute_gradient = True
    config.compute_hessian = True
    config.compute_laplacian_check = bool(data.get("compute_laplacian_check", True))

    config.constant.c = float(data.get("constant", {}).get("c", 1.0))
    config.distance.alpha = float(data.get("distance", {}).get("alpha", 0.5))
    config.average_flux.b_bar = float(
        data.get("average_flux", {}).get("b_bar", -1.0)
    )
    guidance = data.get("guidance", {})
    config.guidance.beta = float(guidance.get("beta", 8.0))
    config.guidance.base_flux_strength = float(
        guidance.get("base_flux_strength", 0.5)
    )
    config.guidance.target_mean_abs_scale = float(
        guidance.get("target_mean_abs_scale", 0.35)
    )
    config.guidance.nonuniform_axis = guidance.get("nonuniform_axis")
    config.guidance.nonuniform_gain = float(guidance.get("nonuniform_gain", 0.0))

    cg = data.get("conjugate_gradient", {})
    config.conjugate_gradient.tolerance = float(cg.get("tolerance", 1.0e-7))
    config.conjugate_gradient.max_iter = int(cg.get("max_iter", 3000))
    sor = data.get("sor", {})
    config.sor.omega = float(sor.get("omega", 1.75))
    config.sor.max_iter = int(sor.get("max_iter", 5000))
    config.sor.tolerance = float(sor.get("tolerance", 1.0e-5))
    config.sor.residual_check_interval = int(sor.get("residual_check_interval", 25))
    config.validate()
    return config


def compute_poisson_field(
    occupancy: np.ndarray,
    *,
    spacing: Iterable[float],
    config: Mapping[str, Any],
    forcing_method: str | None = None,
    solver: str | None = None,
) -> PoissonField:
    """Solve the Poisson problem and return an interpolation-ready field."""

    box_config = poisson_config_from_mapping(
        config,
        spacing=spacing,
        forcing_method=forcing_method,
        solver=solver,
    )
    result = PoissonSafetyBox(box_config).compute(np.asarray(occupancy, dtype=bool))
    solve_values = np.asarray(result.h, dtype=float)[result.solve_mask]
    if solve_values.size == 0:
        raise RuntimeError("The Poisson solve mask is empty.")
    normalize = bool(config.get("normalize_for_control", True))
    scale = float(np.max(np.abs(solve_values))) if normalize else 1.0
    if not np.isfinite(scale) or scale <= 1.0e-12:
        raise RuntimeError("The Poisson field has no finite positive control scale.")
    return PoissonField(
        result=result,
        spacing=tuple(float(value) for value in spacing),
        control_scale=scale,
        forcing_method=box_config.forcing_method,
        solver=box_config.solver,
    )


def compare_poisson_solvers(
    occupancy: np.ndarray,
    *,
    spacing: Iterable[float],
    config: Mapping[str, Any],
    solvers: Iterable[str],
    forcing_method: str,
) -> tuple[list[dict[str, Any]], dict[str, PoissonField]]:
    """Compare wall time, algebraic residual, and field error to a direct reference."""

    solver_names = list(dict.fromkeys(str(name) for name in solvers))
    if "sparse_direct" not in solver_names:
        solver_names.insert(0, "sparse_direct")

    fields: dict[str, PoissonField] = {}
    records: list[dict[str, Any]] = []
    for name in solver_names:
        started = perf_counter()
        field = compute_poisson_field(
            occupancy,
            spacing=spacing,
            config=config,
            forcing_method=forcing_method,
            solver=name,
        )
        total = perf_counter() - started
        fields[name] = field
        solver_info = field.result.solver_info
        # Reassemble the exact linear system used by the sparse solvers and
        # evaluate A h - b for every backend, including SOR. This is distinct
        # from the reconstructed Laplacian diagnostic below.
        boundary_values = np.where(
            field.result.solve_mask,
            0.0,
            np.asarray(field.result.h, dtype=float),
        )
        A_exact, b_exact, _index_map, _free_cells, _meta = build_laplacian_system(
            np.asarray(field.result.forcing, dtype=float),
            np.asarray(field.result.solve_mask, dtype=bool),
            boundary_values,
            tuple(float(value) for value in field.spacing),
        )
        unknown_vector = np.asarray(field.result.h, dtype=float)[field.result.solve_mask]
        assembled_residual = np.asarray(A_exact @ unknown_vector - b_exact, dtype=float)
        algebraic_l2 = float(
            np.linalg.norm(assembled_residual)
            / max(1, assembled_residual.size) ** 0.5
        )
        algebraic_linf = float(np.max(np.abs(assembled_residual)))
        laplacian_error = field.result.diagnostics.get("laplacian_check", {}).get(
            "l2_delta_h_minus_f",
            np.nan,
        )
        records.append(
            {
                "solver": name,
                "total_wall_time_s": float(total),
                "domain_time_s": float(field.result.timing.get("domain", 0.0)),
                "forcing_time_s": float(field.result.timing.get("forcing", 0.0)),
                "solve_time_s": float(field.result.timing.get("solve", 0.0)),
                "derivative_time_s": float(field.result.timing.get("derivatives", 0.0)),
                "algebraic_residual": algebraic_l2,
                "algebraic_residual_l2": algebraic_l2,
                "algebraic_residual_linf": algebraic_linf,
                "laplacian_l2_error": float(laplacian_error),
                "iterations": int(solver_info.get("iterations", 0)),
                "status": str(
                    solver_info.get(
                        "status",
                        "converged" if solver_info.get("converged", solver_info.get("info_code", 0) == 0) else "warning",
                    )
                ),
            }
        )

    reference = fields["sparse_direct"]
    reference_h = reference.raw_h
    solve_mask = reference.result.solve_mask
    denominator = max(float(np.linalg.norm(reference_h[solve_mask])), 1.0e-12)
    for record in records:
        candidate = fields[record["solver"]].raw_h
        delta = candidate[solve_mask] - reference_h[solve_mask]
        record["relative_l2_field_error"] = float(np.linalg.norm(delta) / denominator)
        record["linf_field_error"] = float(np.max(np.abs(delta)))
    return records, fields


def compare_forcing_methods(
    occupancy: np.ndarray,
    *,
    spacing: Iterable[float],
    config: Mapping[str, Any],
    forcing_methods: Iterable[str],
    solver: str,
) -> dict[str, PoissonField]:
    """Compute one Poisson field per forcing method on the same geometry."""

    return {
        method: compute_poisson_field(
            occupancy,
            spacing=spacing,
            config=config,
            forcing_method=method,
            solver=solver,
        )
        for method in dict.fromkeys(str(value) for value in forcing_methods)
    }
