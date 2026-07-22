"""Command history plotting."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_commands(t, u_nom, u_safe, output_dir="outputs", filename="commands.png", show=False, save=True, dpi=180):
    """Plot u_nom and u_safe component histories."""
    t = np.asarray(t)
    u_nom = np.asarray(u_nom)
    u_safe = np.asarray(u_safe)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(u_nom.shape[1]):
        ax.plot(t, u_nom[:, i], "--", label=f"u_nom[{i}]")
        ax.plot(t, u_safe[:, i], label=f"u_safe[{i}]")
    ax.set_xlabel("time")
    ax.set_ylabel("command")
    ax.set_title("Nominal vs safe command")
    ax.grid(True); ax.legend()
    if save: fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)
