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

"""Guidance field plotting functions."""


def plot_guidance(vector_field, divergence, solve_mask, output_dir, show=False, save=True, dpi=180):
    """Plot guidance vector slices and divergence slices."""
    output_dir = _ensure(output_dir)
    dim = vector_field.shape[-1]
    if dim == 2:
        sl_vec = vector_field; mask = solve_mask
        fig, ax = plt.subplots(figsize=(6,5)); step=max(1, sl_vec.shape[0]//25)
        X,Y=np.meshgrid(np.arange(sl_vec.shape[1]), np.arange(sl_vec.shape[0]))
        ax.quiver(X[::step,::step], Y[::step,::step], sl_vec[::step,::step,1], sl_vec[::step,::step,0])
        ax.set_title('Guidance vector field'); fig.tight_layout()
        if save: fig.savefig(output_dir/'guidance_quiver.png', dpi=dpi)
        if show: plt.show()
        plt.close(fig)
    else:
        k = vector_field.shape[2]//2
        sl_vec = vector_field[:, :, k, :]; step=max(1, sl_vec.shape[0]//25)
        X,Y=np.meshgrid(np.arange(sl_vec.shape[1]), np.arange(sl_vec.shape[0]))
        fig, ax = plt.subplots(figsize=(6,5))
        ax.quiver(X[::step,::step], Y[::step,::step], sl_vec[::step,::step,1], sl_vec[::step,::step,0])
        ax.set_title('Guidance XY quiver'); fig.tight_layout()
        if save: fig.savefig(output_dir/'guidance_xy_quiver.png', dpi=dpi)
        if show: plt.show()
        plt.close(fig)
    if divergence is not None:
        div = np.where(solve_mask, divergence, np.nan)
        for name, sl in _middle_slices(div):
            fig, ax = plt.subplots(figsize=(5,4)); im=ax.imshow(sl.T, origin='lower')
            ax.set_title(f'Guidance divergence {name}'); fig.colorbar(im, ax=ax); fig.tight_layout()
            if save: fig.savefig(output_dir / f'guidance_divergence_{name}.png', dpi=dpi)
            if show: plt.show()
            plt.close(fig)
