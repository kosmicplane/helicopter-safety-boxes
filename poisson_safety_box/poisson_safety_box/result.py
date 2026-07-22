"""Result object returned by the Poisson safety box."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import json
import numpy as np


@dataclass
class PoissonBoxResult:
    """Container with the safety function and all diagnostics.

    A future CBF package should consume h, grad_h, hessian_h, and optionally the
    interpolation utilities. This object is intentionally data-oriented.
    """

    h: np.ndarray
    grad_h: Optional[np.ndarray]
    hessian_h: Optional[np.ndarray]
    laplacian_h: Optional[np.ndarray]

    occupancy_mask: np.ndarray
    free_mask: np.ndarray
    solve_mask: np.ndarray
    boundary_mask: np.ndarray
    omega_union_boundary_mask: np.ndarray

    forcing: np.ndarray
    guidance_vector: Optional[np.ndarray] = None
    guidance_divergence: Optional[np.ndarray] = None

    solver_info: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def save_npz(self, path: str | Path) -> None:
        """Save arrays to a compressed NumPy archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            h=self.h,
            grad_h=self.grad_h if self.grad_h is not None else np.array([]),
            hessian_h=self.hessian_h if self.hessian_h is not None else np.array([]),
            laplacian_h=self.laplacian_h if self.laplacian_h is not None else np.array([]),
            occupancy_mask=self.occupancy_mask,
            free_mask=self.free_mask,
            solve_mask=self.solve_mask,
            boundary_mask=self.boundary_mask,
            omega_union_boundary_mask=self.omega_union_boundary_mask,
            forcing=self.forcing,
            guidance_vector=self.guidance_vector if self.guidance_vector is not None else np.array([]),
            guidance_divergence=self.guidance_divergence if self.guidance_divergence is not None else np.array([]),
        )

    def save_summary_json(self, path: str | Path) -> None:
        """Save scalar diagnostics to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "shape": list(self.h.shape),
            "h_min": float(np.nanmin(self.h[self.solve_mask])) if np.any(self.solve_mask) else None,
            "h_max": float(np.nanmax(self.h[self.solve_mask])) if np.any(self.solve_mask) else None,
            "h_mean": float(np.nanmean(self.h[self.solve_mask])) if np.any(self.solve_mask) else None,
            "forcing_min": float(np.nanmin(self.forcing[self.solve_mask])) if np.any(self.solve_mask) else None,
            "forcing_max": float(np.nanmax(self.forcing[self.solve_mask])) if np.any(self.solve_mask) else None,
            "free_cells": int(np.count_nonzero(self.free_mask)),
            "solve_cells": int(np.count_nonzero(self.solve_mask)),
            "boundary_cells": int(np.count_nonzero(self.boundary_mask)),
            "solver_info": _to_jsonable(self.solver_info),
            "timing": _to_jsonable(self.timing),
            "diagnostics": _to_jsonable(self.diagnostics),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def plot_all(self, output_dir: str | Path, show: bool = False, dpi: int = 180) -> None:
        """Save a standard set of diagnostic plots.

        Plotting is imported lazily so that the computational core can be used
        without importing Matplotlib.
        """
        from .visualization.occupancy_plots import plot_occupancy
        from .visualization.mask_plots import plot_masks
        from .visualization.forcing_plots import plot_forcing
        from .visualization.poisson_plots import plot_poisson_h
        from .visualization.diagnostics_plots import plot_residual_history, plot_timing_summary
        from .visualization.guidance_plots import plot_guidance

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_occupancy(self.occupancy_mask, output_dir, show=show, dpi=dpi)
        plot_masks(self.free_mask, self.boundary_mask, self.omega_union_boundary_mask, output_dir, show=show, dpi=dpi)
        plot_forcing(self.forcing, self.solve_mask, output_dir, show=show, dpi=dpi)
        plot_poisson_h(self.h, self.solve_mask, self.grad_h, output_dir, show=show, dpi=dpi)
        if self.guidance_vector is not None:
            plot_guidance(self.guidance_vector, self.guidance_divergence, self.solve_mask, output_dir, show=show, dpi=dpi)
        if "residual_history" in self.solver_info:
            plot_residual_history(self.solver_info["residual_history"], output_dir, show=show, dpi=dpi)
        plot_timing_summary(self.timing, output_dir, show=show, dpi=dpi)


def _to_jsonable(x: Any) -> Any:
    """Convert NumPy scalars/arrays into JSON-friendly objects."""
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        if x.size <= 20:
            return x.tolist()
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x
