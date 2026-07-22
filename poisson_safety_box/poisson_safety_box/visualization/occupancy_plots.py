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

"""Occupancy plotting functions."""


def plot_occupancy(occupancy, output_dir, show=False, save=True, dpi=180):
    """Plot occupancy slices and an optional point cloud."""
    output_dir = _ensure(output_dir)
    for name, sl in _middle_slices(occupancy.astype(float)):
        fig, ax = plt.subplots(figsize=(5,4))
        ax.imshow(sl.T, origin='lower', cmap='gray_r')
        ax.set_title(f'Occupancy slice {name}')
        fig.tight_layout()
        if save: fig.savefig(output_dir / f'occupancy_{name}.png', dpi=dpi)
        if show: plt.show()
        plt.close(fig)
    if occupancy.ndim == 3:
        pts = np.argwhere(occupancy)
        if len(pts) > 5000:
            pts = pts[np.linspace(0, len(pts)-1, 5000).astype(int)]
        fig = plt.figure(figsize=(6,5)); ax = fig.add_subplot(111, projection='3d')
        if len(pts): ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=2, alpha=0.3)
        ax.set_title('3D occupied point cloud')
        if save: fig.savefig(output_dir / 'occupancy_3d.png', dpi=dpi)
        if show: plt.show()
        plt.close(fig)
