"""Poisson Safety Box integration and smooth field interpolation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class PoissonFieldSampler:
    """Trilinear sampler for h, grad(h), Hessian(h), and Laplacian(h)."""

    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    result: object

    def __post_init__(self) -> None:
        options = dict(bounds_error=False, fill_value=0.0)
        self._h = RegularGridInterpolator(self.axes, self.result.h, **options)
        self._grad = [RegularGridInterpolator(self.axes, self.result.grad_h[..., d], **options) for d in range(3)]
        self._hessian = [
            [RegularGridInterpolator(self.axes, self.result.hessian_h[..., i, j], **options) for j in range(3)]
            for i in range(3)
        ]
        self._lap = RegularGridInterpolator(self.axes, self.result.laplacian_h, **options)

    def sample(self, position: np.ndarray) -> dict:
        point = np.asarray(position, dtype=float).reshape(1, 3)
        h = float(self._h(point)[0])
        grad = np.array([float(interp(point)[0]) for interp in self._grad], dtype=float)
        hessian = np.array(
            [[float(self._hessian[i][j](point)[0]) for j in range(3)] for i in range(3)],
            dtype=float,
        )
        laplacian = float(self._lap(point)[0])
        return {"h": h, "grad_h": grad, "hessian_h": hessian, "laplacian_h": laplacian}


def compute_poisson(box_classes: dict, occupancy: np.ndarray, spacing: tuple[float, ...], config: dict):
    """Compute the Poisson field exclusively through ``poisson_safety_box``."""
    PoissonBoxConfig = box_classes["PoissonBoxConfig"]
    PoissonSafetyBox = box_classes["PoissonSafetyBox"]
    pcfg = config["poisson"]
    cfg = PoissonBoxConfig(
        grid_spacing=tuple(spacing),
        forcing_method=str(pcfg["forcing_method"]),
        solver=str(pcfg["solver"]),
        compute_gradient=True,
        compute_hessian=True,
        compute_laplacian_check=True,
    )
    cfg.distance.alpha = float(pcfg.get("distance_alpha", 0.55))
    cfg.conjugate_gradient.tolerance = float(pcfg.get("cg_tolerance", 1e-6))
    cfg.conjugate_gradient.max_iter = int(pcfg.get("cg_max_iter", 2500))
    return PoissonSafetyBox(cfg).compute(occupancy)
