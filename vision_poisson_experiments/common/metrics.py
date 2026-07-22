"""Thread-safe timing, rate, and summary metrics for both experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np

from .io_utils import save_json, write_csv


class RateMeter:
    """Estimate a recent event rate from a bounded timestamp window."""

    def __init__(self, window_seconds: float = 2.0) -> None:
        self.window_seconds = max(0.1, float(window_seconds))
        self._timestamps: deque[float] = deque()

    def tick(self, timestamp: float | None = None) -> float:
        """Record an event and return the current events-per-second estimate."""

        now = perf_counter() if timestamp is None else float(timestamp)
        self._timestamps.append(now)
        cutoff = now - self.window_seconds
        while len(self._timestamps) > 1 and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        return self.rate

    @property
    def rate(self) -> float:
        """Return the current events-per-second estimate."""

        if len(self._timestamps) < 2:
            return 0.0
        duration = self._timestamps[-1] - self._timestamps[0]
        return 0.0 if duration <= 0.0 else float((len(self._timestamps) - 1) / duration)


def distribution_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    """Return count, central tendency, and selected latency percentiles."""

    array = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


@dataclass(frozen=True)
class StageTimer:
    """Context manager that stores elapsed time in a mutable mapping."""

    target: dict[str, float]
    key: str

    def __enter__(self) -> "StageTimer":
        object.__setattr__(self, "_start", perf_counter())
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        self.target[self.key] = perf_counter() - getattr(self, "_start")


class MetricsRecorder:
    """Collect frame-level and solve-level rows from multiple threads."""

    def __init__(self) -> None:
        self._frame_rows: list[dict[str, Any]] = []
        self._solve_rows: list[dict[str, Any]] = []
        self._warnings: list[dict[str, Any]] = []
        self._lock = Lock()

    def add_frame(self, row: Mapping[str, Any]) -> None:
        """Append one frame-level metrics row."""

        with self._lock:
            self._frame_rows.append(dict(row))

    def add_solve(self, row: Mapping[str, Any]) -> None:
        """Append one Poisson-solve metrics row."""

        with self._lock:
            self._solve_rows.append(dict(row))

    def add_warning(self, message: str, **context: Any) -> None:
        """Record a timestamped warning with structured context."""

        with self._lock:
            self._warnings.append(
                {
                    "time_monotonic_s": perf_counter(),
                    "message": str(message),
                    **context,
                }
            )

    @property
    def frame_rows(self) -> list[dict[str, Any]]:
        """Return a defensive copy of frame rows."""

        with self._lock:
            return [dict(row) for row in self._frame_rows]

    @property
    def solve_rows(self) -> list[dict[str, Any]]:
        """Return a defensive copy of solve rows."""

        with self._lock:
            return [dict(row) for row in self._solve_rows]

    @property
    def warnings(self) -> list[dict[str, Any]]:
        """Return a defensive copy of warning rows."""

        with self._lock:
            return [dict(row) for row in self._warnings]

    def build_summary(self) -> dict[str, Any]:
        """Aggregate the load-bearing performance metrics without inventing rates."""

        frames = self.frame_rows
        solves = self.solve_rows
        pipeline_latency = [float(row["pipeline_latency_s"]) for row in frames if row.get("pipeline_latency_s") is not None]
        segmentation_latency = [
            float(row["segmentation_latency_s"])
            for row in frames
            if row.get("segmentation_latency_s") is not None
        ]
        occupancy_latency = [
            float(row["occupancy_preprocessing_latency_s"])
            for row in frames
            if row.get("occupancy_preprocessing_latency_s") is not None
        ]
        field_age = [float(row["field_age_s"]) for row in frames if row.get("field_age_s") is not None]
        solve_latency = [float(row["solve_wall_time_s"]) for row in solves if row.get("solve_wall_time_s") is not None]
        field_update_latency = [
            float(row["pipeline_from_submit_s"])
            for row in solves
            if row.get("pipeline_from_submit_s") is not None
        ]
        capture_fps = [float(row["capture_fps"]) for row in frames if row.get("capture_fps") is not None]
        display_fps = [float(row["display_fps"]) for row in frames if row.get("display_fps") is not None]
        poisson_rate = [
            float(row["poisson_updates_per_s"])
            for row in frames
            if row.get("poisson_updates_per_s") is not None
        ]
        return {
            "frame_count": len(frames),
            "solve_count": len(solves),
            "warning_count": len(self.warnings),
            "pipeline_latency_s": distribution_summary(pipeline_latency),
            "segmentation_latency_s": distribution_summary(segmentation_latency),
            "occupancy_preprocessing_latency_s": distribution_summary(occupancy_latency),
            "solve_latency_s": distribution_summary(solve_latency),
            "field_update_latency_s": distribution_summary(field_update_latency),
            "field_age_s": distribution_summary(field_age),
            "capture_fps": distribution_summary(capture_fps),
            "display_fps": distribution_summary(display_fps),
            "poisson_updates_per_s": distribution_summary(poisson_rate),
            "discarded_frames": int(sum(int(row.get("dropped_capture_frames", 0)) for row in frames[-1:])),
            "discarded_queued_tasks": int(sum(int(row.get("discarded_queued_tasks", 0)) for row in frames[-1:])),
            "discarded_obsolete_solves": int(sum(int(row.get("discarded_obsolete_solves", 0)) for row in frames[-1:])),
        }

    def save(self, output_directory: str | Path) -> dict[str, Any]:
        """Write frame metrics, solve metrics, warnings, and a summary JSON."""

        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        frames = self.frame_rows
        solves = self.solve_rows
        warnings = self.warnings
        write_csv(output / "metrics.csv", frames)
        write_csv(output / "solve_metrics.csv", solves)
        write_csv(output / "warnings.csv", warnings)
        summary = self.build_summary()
        save_json(output / "summary.json", summary)
        return summary
