"""QP timing plots."""
from pathlib import Path
import matplotlib.pyplot as plt


def plot_solve_time(t, solve_time, output_dir="outputs", filename="qp_solve_time.png", show=False, save=True, dpi=180):
    """Plot QP solve time over time."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, solve_time)
    ax.set_xlabel("time"); ax.set_ylabel("solve time [s]"); ax.grid(True)
    if save: fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)
