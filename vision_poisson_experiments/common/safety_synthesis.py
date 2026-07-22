"""Versioned latest-only worker for synchronized Poisson and HJ synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock, Thread
from time import perf_counter
from typing import Any

import numpy as np

from .coordinates import GridGeometry
from .hj_reachability import ReachabilityBundle, ReachabilityConfig, build_reachability_bundle
from .metrics import MetricsRecorder
from .mission_setup import MissionDefinition
from .poisson_runner import PoissonRunRecord, run_poisson
from .target_manager import (
    LandingZoneValidationConfig,
    ZoneAssessment,
    assess_all_landing_zones,
)


@dataclass(frozen=True)
class SafetySynthesisTask:
    """Immutable synchronized input for one Poisson/HJR computation."""

    version_id: int
    submitted_time_s: float
    raw_occupancy: np.ndarray
    filtered_occupancy: np.ndarray
    inflated_occupancy: np.ndarray
    forcing_method: str
    solver: str
    compute_laplacian_check: bool
    unavailable_identifiers: tuple[int, ...] = ()


@dataclass(frozen=True)
class SafetySynthesisSnapshot:
    """Synchronized local safety and target-reachability fields."""

    version_id: int
    submitted_time_s: float
    completed_time_s: float
    poisson_record: PoissonRunRecord
    reachability: ReachabilityBundle
    zone_assessments: dict[int, ZoneAssessment]
    raw_occupancy: np.ndarray
    filtered_occupancy: np.ndarray
    inflated_occupancy: np.ndarray

    @property
    def record(self) -> PoissonRunRecord:
        """Compatibility alias used by the original live pipeline."""

        return self.poisson_record

    @property
    def age_s(self) -> float:
        """Return age since the input occupancy was submitted."""

        return max(0.0, perf_counter() - self.submitted_time_s)

    @property
    def poisson_solve_time_s(self) -> float:
        """Return Poisson field construction time."""

        return float(self.poisson_record.wall_time_s)

    @property
    def hjr_solve_time_s(self) -> float:
        """Return total distance-field construction time."""

        return float(self.reachability.total_solve_time_s)


class SafetySynthesisWorker:
    """Compute only the newest synchronized Poisson/HJR task in one thread."""

    def __init__(
        self,
        *,
        geometry: GridGeometry,
        poisson_config: dict[str, Any],
        reachability_config: ReachabilityConfig,
        mission: MissionDefinition,
        zone_validation_config: LandingZoneValidationConfig,
        metrics: MetricsRecorder,
    ) -> None:
        self.geometry = geometry
        self.poisson_config = dict(poisson_config)
        self.reachability_config = reachability_config
        self.mission = mission
        self.zone_validation_config = zone_validation_config
        self.metrics = metrics
        self._queue: Queue[SafetySynthesisTask] = Queue(maxsize=1)
        self.discarded_items = 0
        self.maximum_observed_size = 0
        self._latest_submitted_version = -1
        self._latest_snapshot: SafetySynthesisSnapshot | None = None
        self._discarded_obsolete_solves = 0
        self._failed_solves = 0
        self._invalid_solves = 0
        self._stop_requested = False
        self._lock = Lock()
        self._thread = Thread(target=self._run, name="safety-synthesis-worker", daemon=True)
        self._thread.start()

    @property
    def queue(self) -> "SafetySynthesisWorker":
        """Compatibility proxy exposing queue counters and qsize()."""

        return self

    def qsize(self) -> int:
        """Return the bounded queue size."""

        return self._queue.qsize()

    @property
    def discarded_obsolete_solves(self) -> int:
        with self._lock:
            return self._discarded_obsolete_solves

    @property
    def failed_solves(self) -> int:
        with self._lock:
            return self._failed_solves

    @property
    def invalid_solves(self) -> int:
        with self._lock:
            return self._invalid_solves

    def submit(self, task: SafetySynthesisTask) -> None:
        """Submit the newest task, replacing any older task without blocking."""

        with self._lock:
            self._latest_submitted_version = max(self._latest_submitted_version, int(task.version_id))
        try:
            self._queue.put_nowait(task)
        except Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self.discarded_items += 1
            except Empty:
                pass
            self._queue.put_nowait(task)
        self.maximum_observed_size = max(self.maximum_observed_size, self._queue.qsize())

    def latest_snapshot(self) -> SafetySynthesisSnapshot | None:
        """Return the newest accepted synchronized snapshot."""

        with self._lock:
            return self._latest_snapshot

    def empty(self) -> bool:
        return self._queue.empty()

    def _run(self) -> None:
        while not self._stop_requested or not self._queue.empty():
            try:
                task = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                local_config = dict(self.poisson_config)
                local_config["compute_laplacian_check"] = bool(task.compute_laplacian_check)
                poisson_start = perf_counter()
                poisson_record = run_poisson(
                    np.asarray(task.inflated_occupancy, dtype=bool),
                    grid_spacing_yx=self.geometry.spacing_yx,
                    poisson_config=local_config,
                    forcing_method=task.forcing_method,
                    solver=task.solver,
                    live_mode=True,
                )
                poisson_wall = perf_counter() - poisson_start
                validation = poisson_record.validation.get("result", {})
                numerically_valid = bool(poisson_record.validation.valid)
                if not numerically_valid:
                    with self._lock:
                        self._invalid_solves += 1
                    self.metrics.add_warning(
                        "safety_synthesis_poisson_rejected",
                        version_id=task.version_id,
                        residual_max_abs=validation.get("residual_max_abs"),
                    )
                    continue

                assessments = assess_all_landing_zones(
                    self.mission,
                    raw_occupancy=np.asarray(task.raw_occupancy, dtype=bool),
                    inflated_occupancy=np.asarray(task.inflated_occupancy, dtype=bool),
                    geometry=self.geometry,
                    config=self.zone_validation_config,
                )
                bundle = build_reachability_bundle(
                    np.asarray(task.inflated_occupancy, dtype=bool),
                    mission=self.mission,
                    geometry=self.geometry,
                    config=self.reachability_config,
                    occupancy_version=task.version_id,
                    unavailable_identifiers=task.unavailable_identifiers,
                )
                completed = perf_counter()
                snapshot = SafetySynthesisSnapshot(
                    version_id=task.version_id,
                    submitted_time_s=task.submitted_time_s,
                    completed_time_s=completed,
                    poisson_record=poisson_record,
                    reachability=bundle,
                    zone_assessments=assessments,
                    raw_occupancy=np.asarray(task.raw_occupancy, dtype=bool).copy(),
                    filtered_occupancy=np.asarray(task.filtered_occupancy, dtype=bool).copy(),
                    inflated_occupancy=np.asarray(task.inflated_occupancy, dtype=bool).copy(),
                )
                with self._lock:
                    obsolete = int(task.version_id) < self._latest_submitted_version
                    if obsolete:
                        self._discarded_obsolete_solves += 1
                    else:
                        self._latest_snapshot = snapshot
                self.metrics.add_solve(
                    {
                        "solve_kind": "synchronized_poisson_hjr",
                        "version_id": task.version_id,
                        "submitted_time_s": task.submitted_time_s,
                        "completed_time_s": completed,
                        "solve_wall_time_s": float(poisson_record.wall_time_s + bundle.total_solve_time_s),
                        "poisson_solve_wall_time_s": float(poisson_record.wall_time_s),
                        "poisson_wrapper_wall_time_s": float(poisson_wall),
                        "hjr_solve_wall_time_s": float(bundle.total_solve_time_s),
                        "pipeline_from_submit_s": completed - task.submitted_time_s,
                        "forcing_method": task.forcing_method,
                        "solver": task.solver,
                        "accepted": bool(numerically_valid and not obsolete),
                        "numerically_valid": numerically_valid,
                        "obsolete": obsolete,
                        "status": validation.get("solver_status"),
                        "available_hjr_targets": len(bundle.available_identifiers()),
                    }
                )
            except Exception as error:
                with self._lock:
                    self._failed_solves += 1
                self.metrics.add_warning(
                    "safety_synthesis_worker_failure",
                    version_id=task.version_id,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            finally:
                self._queue.task_done()

    def close(self, *, timeout_s: float = 15.0) -> None:
        """Request shutdown and wait for the active/latest task."""

        self._stop_requested = True
        self._thread.join(timeout=max(0.1, float(timeout_s)))
        if self._thread.is_alive():
            self.metrics.add_warning("safety_synthesis_worker_join_timeout", timeout_s=timeout_s)


__all__ = ["SafetySynthesisSnapshot", "SafetySynthesisTask", "SafetySynthesisWorker"]
