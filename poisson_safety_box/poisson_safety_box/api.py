"""High-level API for the Poisson safety box."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from .config import PoissonBoxConfig
from .result import PoissonBoxResult
from .domain.occupancy import normalize_occupancy
from .domain.masks import compute_basic_masks
from .domain.boundaries import compute_boundary_mask, compute_solve_mask
from .forcing.constant import build_constant_forcing
from .forcing.distance import build_distance_forcing
from .forcing.average_flux import build_average_flux_forcing
from .forcing.guidance import build_guidance_forcing
from .guidance.vector_field import build_guidance_vector_field
from .solvers.sor import solve_poisson_sor
from .solvers.sparse_direct import solve_poisson_sparse_direct
from .solvers.conjugate_gradient import solve_poisson_cg
from .derivatives.gradient import compute_gradient
from .derivatives.hessian import compute_hessian
from .derivatives.laplacian import compute_laplacian, laplacian_from_hessian
from .utils.timing import timed
from .utils.validation import normalize_spacing, require_nonempty_mask


class PoissonSafetyBox:
    """Reusable library class that computes Poisson safety functions from occupancy.

    This class is the only object most downstream modules need. It accepts an
    occupancy matrix and returns a :class:`PoissonBoxResult` containing h,
    derivatives, masks, forcing fields, and diagnostics.
    """

    def __init__(self, config: PoissonBoxConfig):
        self.config = config
        self.config.validate()

    def compute(self, occupancy: np.ndarray) -> PoissonBoxResult:
        """Compute h and optional derivatives from an occupancy matrix."""
        timing: Dict[str, float] = {}
        diagnostics: Dict[str, Any] = {}
        guidance_vector = None
        guidance_divergence = None

        with timed("domain", timing):
            occupancy_mask = normalize_occupancy(occupancy)
            spacing = normalize_spacing(self.config.grid_spacing, occupancy_mask.ndim)
            masks = compute_basic_masks(occupancy_mask)
            free_mask = masks["free_mask"]
            boundary_mask = compute_boundary_mask(occupancy_mask, self.config.outer_boundary_as_dirichlet)
            solve_mask = compute_solve_mask(free_mask, boundary_mask)
            omega_union_boundary_mask = free_mask.copy()
            require_nonempty_mask("solve_mask", solve_mask)
            boundary_values = np.zeros_like(occupancy_mask, dtype=float)
            boundary_values[boundary_mask] = self.config.boundary_value

        with timed("forcing", timing):
            forcing_result = self._build_forcing(occupancy_mask, solve_mask, boundary_mask, spacing, boundary_values)
            forcing = forcing_result.forcing
            diagnostics["forcing"] = _strip_large_arrays(forcing_result.diagnostics)
            if "guidance_vector" in forcing_result.diagnostics:
                guidance_vector = forcing_result.diagnostics["guidance_vector"]
            if "guidance_divergence" in forcing_result.diagnostics:
                guidance_divergence = forcing_result.diagnostics["guidance_divergence"]

        with timed("solve", timing):
            h, solver_info = self._solve(forcing, solve_mask, boundary_values, spacing)

        grad_h = None
        hessian_h = None
        laplacian_h = None
        with timed("derivatives", timing):
            if self.config.compute_gradient:
                grad_h = compute_gradient(h, spacing)
            if self.config.compute_hessian:
                hessian_h = compute_hessian(h, spacing)
            if self.config.compute_laplacian_check:
                if hessian_h is not None:
                    laplacian_h = laplacian_from_hessian(hessian_h)
                else:
                    laplacian_h = compute_laplacian(h, spacing)
                residual = laplacian_h - forcing
                diagnostics["laplacian_check"] = {
                    "max_abs_delta_h_minus_f": float(np.max(np.abs(residual[solve_mask]))),
                    "l2_delta_h_minus_f": float(np.linalg.norm(residual[solve_mask]) / max(1, np.count_nonzero(solve_mask)) ** 0.5),
                }

        return PoissonBoxResult(
            h=h,
            grad_h=grad_h,
            hessian_h=hessian_h,
            laplacian_h=laplacian_h,
            occupancy_mask=occupancy_mask,
            free_mask=free_mask,
            solve_mask=solve_mask,
            boundary_mask=boundary_mask,
            omega_union_boundary_mask=omega_union_boundary_mask,
            forcing=forcing,
            guidance_vector=guidance_vector,
            guidance_divergence=guidance_divergence,
            solver_info=solver_info,
            timing=timing,
            diagnostics=diagnostics,
        )

    def _solver_options(self) -> Dict[str, Any]:
        """Return solver-specific options from the config."""
        if self.config.solver == "sor":
            return {
                "omega": self.config.sor.omega,
                "max_iter": self.config.sor.max_iter,
                "tolerance": self.config.sor.tolerance,
                "residual_check_interval": self.config.sor.residual_check_interval,
            }
        if self.config.solver == "conjugate_gradient":
            return {
                "tolerance": self.config.conjugate_gradient.tolerance,
                "max_iter": self.config.conjugate_gradient.max_iter,
            }
        return {}

    def _solve(self, rhs: np.ndarray, solve_mask: np.ndarray, boundary_values: np.ndarray, spacing: tuple[float, ...]):
        """Dispatch to the selected Poisson solver."""
        if self.config.solver == "sor":
            return solve_poisson_sor(rhs, solve_mask, boundary_values, spacing, **self._solver_options())
        if self.config.solver == "sparse_direct":
            return solve_poisson_sparse_direct(rhs, solve_mask, boundary_values, spacing)
        if self.config.solver == "conjugate_gradient":
            return solve_poisson_cg(rhs, solve_mask, boundary_values, spacing, **self._solver_options())
        raise ValueError(f"Unsupported solver: {self.config.solver}")

    def _build_forcing(self, occupancy_mask, solve_mask, boundary_mask, spacing, boundary_values):
        """Dispatch to the selected forcing builder."""
        method = self.config.forcing_method
        if method == "constant":
            return build_constant_forcing(solve_mask, self.config.constant.c)
        if method == "distance":
            return build_distance_forcing(solve_mask, boundary_mask, spacing, self.config.distance.alpha)
        if method == "average_flux":
            return build_average_flux_forcing(solve_mask, boundary_mask, spacing, self.config.average_flux.b_bar)
        if method == "guidance":
            gcfg = self.config.guidance
            vector_field, guidance_info = build_guidance_vector_field(
                occupancy_mask=occupancy_mask,
                solve_mask=solve_mask,
                boundary_mask=boundary_mask,
                grid_spacing=spacing,
                solver=self.config.solver,
                solver_options=self._solver_options(),
                base_flux_strength=gcfg.base_flux_strength,
                nonuniform_axis=gcfg.nonuniform_axis,
                nonuniform_gain=gcfg.nonuniform_gain,
            )
            # Use constant forcing as reference scale.
            reference = build_constant_forcing(solve_mask, self.config.constant.c).forcing
            target = gcfg.target_mean_abs_scale * float(np.mean(np.abs(reference[solve_mask])))
            fr = build_guidance_forcing(solve_mask, vector_field, spacing, beta=gcfg.beta, target_mean_abs=target)
            fr.diagnostics.update(_strip_large_arrays(guidance_info))
            fr.diagnostics["guidance_vector"] = vector_field
            fr.diagnostics["guidance_divergence"] = fr.diagnostics["divergence"]
            return fr
        raise ValueError(f"Unsupported forcing method: {method}")


def _strip_large_arrays(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove large arrays from nested diagnostics intended for summaries."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        elif isinstance(v, list):
            out[k] = [_strip_large_arrays(x) if isinstance(x, dict) else x for x in v]
        elif isinstance(v, dict):
            out[k] = _strip_large_arrays(v)
        else:
            out[k] = v
    return out
