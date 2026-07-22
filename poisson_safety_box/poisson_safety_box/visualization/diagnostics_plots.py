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

"""Diagnostic plotting functions."""


def plot_residual_history(residual_history, output_dir, show=False, save=True, dpi=180):
    """Plot SOR residual/max-change history."""
    output_dir = _ensure(output_dir)
    if not residual_history:
        return
    fig, ax = plt.subplots(figsize=(5,4)); ax.semilogy(residual_history)
    ax.set_title('SOR residual/max-change history'); ax.set_xlabel('check'); ax.set_ylabel('value')
    fig.tight_layout()
    if save: fig.savefig(output_dir / 'solver_residual_history.png', dpi=dpi)
    if show: plt.show()
    plt.close(fig)


def plot_timing_summary(timing, output_dir, show=False, save=True, dpi=180):
    """Plot timing summary as a bar chart."""
    output_dir = _ensure(output_dir)
    if not timing:
        return
    keys=list(timing.keys()); vals=[timing[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7,4)); ax.bar(keys, vals); ax.set_ylabel('seconds')
    ax.set_title('Timing summary'); ax.tick_params(axis='x', rotation=45); fig.tight_layout()
    if save: fig.savefig(output_dir / 'timing_summary.png', dpi=dpi)
    if show: plt.show()
    plt.close(fig)
