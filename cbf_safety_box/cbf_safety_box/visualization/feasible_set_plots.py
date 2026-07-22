"""2D feasible half-space visualization."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_halfspace_2d(a, b, u_nom, u_safe, output_dir="outputs", filename="feasible_halfspace.png", show=False, save=True, dpi=180):
    """Plot the 2D half-space a^T u >= b and the nominal/safe commands."""
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.size != 2:
        raise ValueError("plot_halfspace_2d requires a 2D constraint vector.")
    u_nom = np.asarray(u_nom, dtype=float)
    u_safe = np.asarray(u_safe, dtype=float)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(-3, 3, 300)
    ys = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = a[0] * X + a[1] * Y - float(b)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(X, Y, Z >= 0, levels=[-0.5, 0.5, 1.5], alpha=0.25)
    ax.contour(X, Y, Z, levels=[0.0], linewidths=2)
    ax.scatter([u_nom[0]], [u_nom[1]], s=80, marker="x", label="u_nom")
    ax.scatter([u_safe[0]], [u_safe[1]], s=80, marker="o", label="u_safe")
    ax.arrow(u_nom[0], u_nom[1], u_safe[0]-u_nom[0], u_safe[1]-u_nom[1], head_width=0.06, length_includes_head=True)
    ax.set_xlabel("u[0]")
    ax.set_ylabel("u[1]")
    ax.set_title("CBF feasible half-space: a^T u >= b")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    if save:
        fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

# Backward-compatible alias used by examples.
def plot_feasible_halfspace_2d(a, b, u_nom, u_safe, output_dir="outputs", filename="feasible_halfspace.png", show=False, save=True, dpi=180):
    """Alias for plot_halfspace_2d."""
    return plot_halfspace_2d(a, b, u_nom, u_safe, output_dir=output_dir, filename=filename, show=show, save=save, dpi=dpi)
