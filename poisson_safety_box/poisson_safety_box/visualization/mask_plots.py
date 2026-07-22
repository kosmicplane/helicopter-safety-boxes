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

"""Mask plotting functions."""


def plot_masks(free_mask, boundary_mask, omega_union_boundary_mask, output_dir, show=False, save=True, dpi=180):
    """Plot free, boundary, and union masks."""
    output_dir = _ensure(output_dir)
    masks = {'free': free_mask, 'boundary': boundary_mask, 'omega_union_boundary': omega_union_boundary_mask}
    for label, mask in masks.items():
        for name, sl in _middle_slices(mask.astype(float)):
            fig, ax = plt.subplots(figsize=(5,4)); ax.imshow(sl.T, origin='lower')
            ax.set_title(f'{label} mask {name}'); fig.tight_layout()
            if save: fig.savefig(output_dir / f'{label}_mask_{name}.png', dpi=dpi)
            if show: plt.show()
            plt.close(fig)
