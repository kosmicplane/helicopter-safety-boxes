"""Residual and safety-margin plots."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_residuals(t, residual, h=None, output_dir="outputs", filename="residuals.png", show=False, save=True, dpi=180):
    """Plot CBF residual and optionally h over time."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, residual, label="CBF residual")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    if h is not None:
        ax.plot(t, h, label="h")
    ax.set_xlabel("time")
    ax.set_title("Safety residuals")
    ax.grid(True); ax.legend()
    if save: fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)
