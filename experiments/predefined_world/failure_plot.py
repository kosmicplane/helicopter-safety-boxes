"""Paper-oriented visualization of sequential landing-zone failures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from experiments.common.plotting import configure_academic_style, save_figure


def plot_sequential_failure_response(
    *,
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    controller: Any,
    directory: str | Path,
    dpi: int,
) -> None:
    """Plot state, target switching, certificates, and control response."""

    if metrics.empty:
        return

    configure_academic_style()
    time = metrics["time_s"].to_numpy(float)
    target_order = list(controller.targets)
    target_to_index = {
        identifier: index for index, identifier in enumerate(target_order)
    }
    active_indices = metrics["active_target"].map(target_to_index).to_numpy(float)

    figure, axes = plt.subplots(4, 1, figsize=(15.5, 12.2), sharex=True)

    for coordinate in ("x", "y", "z"):
        if coordinate in metrics:
            axes[0].plot(time, metrics[coordinate], label=coordinate)
    axes[0].set_ylabel("position [m]")
    axes[0].set_title("Vehicle position during sequential landing-zone failures")
    axes[0].legend(ncol=3)

    axes[1].step(time, active_indices, where="post", label="active target")
    axes[1].set_yticks(
        np.arange(len(target_order), dtype=float),
        labels=target_order,
    )
    axes[1].set_ylabel("active target")
    axes[1].set_title("Target selection and remaining contingency")
    count_axis = axes[1].twinx()
    if "available_count" in metrics:
        count_axis.step(
            time,
            metrics["available_count"],
            where="post",
            linestyle="--",
            label="available",
        )
    if "certified_count" in metrics:
        count_axis.step(
            time,
            metrics["certified_count"],
            where="post",
            linestyle=":",
            label="certified",
        )
    count_axis.set_ylabel("zone count")
    handles_left, labels_left = axes[1].get_legend_handles_labels()
    handles_right, labels_right = count_axis.get_legend_handles_labels()
    axes[1].legend(
        handles_left + handles_right,
        labels_left + labels_right,
        ncol=3,
        loc="best",
    )

    if "active_h_roa" in metrics:
        axes[2].plot(time, metrics["active_h_roa"], label="active h_ROA")
    if "contingency_pivot" in metrics:
        axes[2].plot(
            time,
            metrics["contingency_pivot"],
            linestyle="--",
            label="r-th pivot",
        )
    if "poisson_h" in metrics:
        axes[2].plot(time, metrics["poisson_h"], linestyle=":", label="h_P")
    axes[2].axhline(0.0, linewidth=1.0, linestyle="-.")
    axes[2].set_ylabel("certificate value")
    axes[2].set_title("Safety, stability, and contingency certificates")
    axes[2].legend(ncol=3)

    if "intervention_norm" in metrics:
        axes[3].plot(
            time,
            metrics["intervention_norm"],
            label="control intervention",
        )
    if "omega" in metrics:
        axes[3].plot(time, metrics["omega"], linestyle="--", label="omega")
    if "clf_slack" in metrics:
        axes[3].plot(
            time,
            metrics["clf_slack"],
            linestyle=":",
            label="CLF slack",
        )
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("control quantity")
    axes[3].set_title("Safety-filter response")
    axes[3].legend(ncol=3)

    event_styles = {
        "target_failed": "--",
        "active_target_switched": ":",
        "hold": "-.",
        "landed": "-.",
    }
    used_labels: set[str] = set()
    if not events.empty:
        for _, event in events.iterrows():
            event_type = str(event["event"])
            if event_type not in event_styles:
                continue
            event_time = float(event["time_s"])
            target_id = str(event["target_id"])
            label = event_type.replace("_", " ")
            legend_label = label if label not in used_labels else None
            used_labels.add(label)
            for axis in axes:
                axis.axvline(
                    event_time,
                    linestyle=event_styles[event_type],
                    linewidth=1.0,
                    label=legend_label if axis is axes[0] else None,
                )
            axes[0].annotate(
                f"{label}: {target_id}",
                xy=(event_time, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -4),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    axes[0].legend(unique.values(), unique.keys(), ncol=4, loc="best")
    figure.suptitle(
        "Sequential contingency response: repeated landing-zone loss and safe hold"
    )
    save_figure(
        figure,
        directory,
        "sequential_landing_zone_failure_response",
        dpi=dpi,
    )
