"""Optional trajectory plotting for user-provided trajectories."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory_2d(positions, output_dir="outputs", filename="trajectory_2d.png", show=False, save=True, dpi=180):
    """Plot a 2D projection of a provided trajectory."""
    p = np.asarray(positions)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(p[:, 0], p[:, 1], linewidth=2)
    ax.scatter([p[0, 0]], [p[0, 1]], label="start")
    ax.scatter([p[-1, 0]], [p[-1, 1]], label="end")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(True); ax.axis("equal"); ax.legend()
    if save: fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)
