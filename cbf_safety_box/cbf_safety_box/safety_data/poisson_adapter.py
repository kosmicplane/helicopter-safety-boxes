"""Duck-typed adapters from a Poisson safety-function box to SafetySample.

This module intentionally avoids importing ``poisson_safety_box`` as a hard
dependency.  It works with objects that expose compatible arrays or interpolation
methods.
"""
from __future__ import annotations

import numpy as np

from .sample import SafetySample


def _nearest_index(position: np.ndarray, grid_spacing: tuple[float, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    idx = tuple(int(round(float(position[i]) / float(grid_spacing[i]))) for i in range(len(shape)))
    clipped = tuple(max(0, min(shape[i] - 1, idx[i])) for i in range(len(shape)))
    return clipped


def sample_from_arrays(h_grid: np.ndarray, grad_h_grid: np.ndarray, position: np.ndarray, grid_spacing: tuple[float, ...], hessian_h_grid: np.ndarray | None = None, laplacian_h_grid: np.ndarray | None = None) -> SafetySample:
    """Create a SafetySample from grid arrays using nearest-neighbor sampling.

    This is deliberately simple.  A production pipeline can pass a higher-quality
    interpolator callable instead.
    """
    h_grid = np.asarray(h_grid)
    grad_h_grid = np.asarray(grad_h_grid)
    position = np.asarray(position, dtype=float)
    idx = _nearest_index(position, tuple(grid_spacing), h_grid.shape)
    h = float(h_grid[idx])
    grad = np.asarray(grad_h_grid[idx], dtype=float)
    H = None if hessian_h_grid is None else np.asarray(hessian_h_grid[idx], dtype=float)
    lap = None if laplacian_h_grid is None else float(np.asarray(laplacian_h_grid)[idx])
    return SafetySample(h=h, grad_h=grad, hessian_h=H, laplacian_h=lap, metadata={"index": idx, "sampling": "nearest"})


def sample_from_poisson_result(poisson_result, position: np.ndarray, grid_spacing: tuple[float, ...] | None = None) -> SafetySample:
    """Create a SafetySample from a duck-typed Poisson result object.

    Accepted object patterns:
    - object with a callable ``sample(position)`` returning SafetySample-like data
    - object with arrays ``h``, ``grad_h``, optionally ``hessian_h`` and
      ``laplacian_h`` plus user-provided grid_spacing
    """
    if hasattr(poisson_result, "sample") and callable(poisson_result.sample):
        out = poisson_result.sample(position)
        if isinstance(out, SafetySample):
            return out
        if isinstance(out, dict):
            return SafetySample(**out)
        raise TypeError("poisson_result.sample(position) must return SafetySample or dict.")
    if not hasattr(poisson_result, "h") or not hasattr(poisson_result, "grad_h"):
        raise ValueError("poisson_result must expose h and grad_h arrays or a sample(position) method.")
    if grid_spacing is None:
        if hasattr(poisson_result, "grid_spacing"):
            grid_spacing = tuple(poisson_result.grid_spacing)
        else:
            raise ValueError("grid_spacing is required when sampling arrays from poisson_result.")
    return sample_from_arrays(
        h_grid=poisson_result.h,
        grad_h_grid=poisson_result.grad_h,
        position=position,
        grid_spacing=grid_spacing,
        hessian_h_grid=getattr(poisson_result, "hessian_h", None),
        laplacian_h_grid=getattr(poisson_result, "laplacian_h", None),
    )

# Compatibility helper used by tests and simple user adapters.
def sample_from_interpolator(interpolator, position):
    """Create a SafetySample from a user-provided interpolator callable.

    The callable must return either a SafetySample or a dictionary containing
    at least ``h`` and ``grad_h``.
    """
    out = interpolator(position)
    if isinstance(out, SafetySample):
        return out
    if isinstance(out, dict):
        return SafetySample(**out)
    raise TypeError("Interpolator must return SafetySample or dict.")


def sample_from_interpolator(interpolator, position: np.ndarray) -> SafetySample:
    """Create a SafetySample from a user-provided interpolation callable.

    The callable may return a SafetySample, a dictionary, or a tuple
    ``(h, grad_h, hessian_h)``. This function is kept small so the Poisson box
    and CBF box remain independently installable.
    """
    out = interpolator(np.asarray(position, dtype=float))
    if isinstance(out, SafetySample):
        return out
    if isinstance(out, dict):
        return SafetySample(**out)
    if isinstance(out, tuple) and len(out) >= 2:
        h = out[0]
        grad_h = out[1]
        H = out[2] if len(out) >= 3 else None
        return SafetySample(h=h, grad_h=grad_h, hessian_h=H)
    raise TypeError("Interpolator must return SafetySample, dict, or tuple(h, grad_h, hessian_h).")
