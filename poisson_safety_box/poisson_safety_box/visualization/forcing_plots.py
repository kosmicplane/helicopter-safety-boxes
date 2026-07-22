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

"""Forcing plotting functions."""


def plot_forcing(forcing, solve_mask, output_dir, show=False, save=True, dpi=180):
    """Plot forcing slices and histogram."""
    output_dir = _ensure(output_dir)
    f = np.where(solve_mask, forcing, np.nan)
    for name, sl in _middle_slices(f):
        fig, ax = plt.subplots(figsize=(5,4)); im=ax.imshow(sl.T, origin='lower')
        ax.set_title(f'Poisson forcing {name}'); fig.colorbar(im, ax=ax); fig.tight_layout()
        if save: fig.savefig(output_dir / f'forcing_{name}.png', dpi=dpi)
        if show: plt.show()
        plt.close(fig)
    vals = forcing[solve_mask]
    if vals.size:
        fig, ax = plt.subplots(figsize=(5,4)); ax.hist(vals.ravel(), bins=40)
        ax.set_title('Forcing histogram'); fig.tight_layout()
        if save: fig.savefig(output_dir / 'forcing_histogram.png', dpi=dpi)
        if show: plt.show()
        plt.close(fig)
