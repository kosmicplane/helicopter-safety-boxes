from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _ensure(output_dir):
    p=Path(output_dir); p.mkdir(parents=True, exist_ok=True); return p


def _middle_slices(arr):
    if arr.ndim == 2:
        return [("xy", arr)]
    i,j,k = [s//2 for s in arr.shape[:3]]
    return [("xy", arr[:, :, k]), ("xz", arr[:, j, :]), ("yz", arr[i, :, :])]

"""Safety function plotting functions."""


def plot_poisson_h(h, solve_mask, grad_h=None, output_dir='outputs', show=False, save=True, dpi=180):
    """Plot h slices and optional 2D contours."""
    output_dir = _ensure(output_dir)
    hp = np.where(solve_mask, h, np.nan)
    for name, sl in _middle_slices(hp):
        fig, ax = plt.subplots(figsize=(5,4)); im=ax.imshow(sl.T, origin='lower')
        ax.set_title(f'Poisson safety function h {name}'); fig.colorbar(im, ax=ax); fig.tight_layout()
        if save: fig.savefig(output_dir / f'poisson_h_{name}.png', dpi=dpi)
        if show: plt.show()
        plt.close(fig)
    if h.ndim == 2:
        fig, ax = plt.subplots(figsize=(5,4)); cs=ax.contour(h.T, levels=15); ax.clabel(cs, fontsize=7)
        ax.set_title('h contours'); fig.tight_layout()
        if save: fig.savefig(output_dir / 'poisson_h_contours.png', dpi=dpi)
        if show: plt.show()
        plt.close(fig)
