"""Latest-only asynchronous Poisson field worker for live perception.

The worker intentionally retains at most one pending occupancy grid.  Slow PDE
solves therefore cannot create an unbounded queue or force the controller to
process stale geometry in chronological order.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any, Mapping

import numpy as np

from experiments.common.poisson_field import PoissonField, compute_poisson_field


@dataclass(frozen=True, slots=True)
class FieldSnapshot:
    """Versioned output of one completed Poisson solve."""

    version: int
    created_monotonic_s: float
    field: PoissonField | None
    occupancy_xy: np.ndarray
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.field is not None and self.error is None


class LatestPoissonWorker:
    """Compute Poisson fields without blocking video acquisition."""

    def __init__(
        self,
        *,
        spacing_xy: tuple[float, float],
        poisson_config: Mapping[str, Any],
    ) -> None:
        self._spacing = tuple(float(value) for value in spacing_xy)
        self._config = dict(poisson_config)
        self._queue: Queue[tuple[int, np.ndarray] | None] = Queue(maxsize=1)
        self._stop = Event()
        self._lock = Lock()
        self._latest: FieldSnapshot | None = None
        self._thread = Thread(target=self._run, name="poisson-field-worker", daemon=True)
        self._thread.start()

    def submit(self, version: int, occupancy_xy: np.ndarray) -> None:
        """Submit one grid, replacing any older grid that has not started."""

        item = (int(version), np.asarray(occupancy_xy, dtype=bool).copy())
        try:
            self._queue.put_nowait(item)
            return
        except Full:
            pass
        try:
            self._queue.get_nowait()
        except Empty:
            pass
        self._queue.put_nowait(item)

    def latest(self) -> FieldSnapshot | None:
        with self._lock:
            return self._latest

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            self._queue.put_nowait(None)
        self._thread.join(timeout=timeout_s)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except Empty:
                continue
            if item is None:
                break
            version, occupancy = item
            try:
                field = compute_poisson_field(
                    occupancy,
                    spacing=self._spacing,
                    config=self._config,
                )
                snapshot = FieldSnapshot(
                    version=version,
                    created_monotonic_s=monotonic(),
                    field=field,
                    occupancy_xy=occupancy,
                )
            except Exception as exc:  # worker boundary; propagated as typed snapshot
                snapshot = FieldSnapshot(
                    version=version,
                    created_monotonic_s=monotonic(),
                    field=None,
                    occupancy_xy=occupancy,
                    error=f"{type(exc).__name__}: {exc}",
                )
            with self._lock:
                if self._latest is None or snapshot.version >= self._latest.version:
                    self._latest = snapshot
