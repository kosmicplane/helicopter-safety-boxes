"""Version synchronization and bounded latest-only safety worker tests."""

from __future__ import annotations

from time import perf_counter, sleep

import numpy as np

from common.coordinates import GridGeometry
from common.hj_reachability import ReachabilityConfig
from common.metrics import MetricsRecorder
from common.mission_setup import LandingZone, MissionDefinition
from common.safety_synthesis import SafetySynthesisTask, SafetySynthesisWorker
from common.target_manager import LandingZoneValidationConfig


def _mission() -> MissionDefinition:
    return MissionDefinition(
        start_xy_m=np.array([0.4, 0.4]),
        landing_zones=(
            LandingZone(1, np.array([2.6, 0.6]), 0.25),
            LandingZone(2, np.array([2.6, 1.8]), 0.25),
        ),
        active_zone_identifier=1,
        required_reachable=1,
        workspace_size_m=(3.0, 2.4),
        calibration_hash="worker-test",
    )


def _worker() -> tuple[SafetySynthesisWorker, GridGeometry, MetricsRecorder]:
    geometry = GridGeometry(width_m=3.0, height_m=2.4, nx=31, ny=25)
    metrics = MetricsRecorder()
    worker = SafetySynthesisWorker(
        geometry=geometry,
        poisson_config={
            "boundary_value": 0.0,
            "outer_boundary_as_dirichlet": True,
            "compute_gradient": True,
            "compute_hessian": False,
            "compute_laplacian_check": False,
            "validation_boundary_tolerance": 1.0e-8,
            "validation_residual_tolerance": 1.0e-5,
            "constant": {"c": 1.0},
            "conjugate_gradient": {"tolerance": 1.0e-7, "max_iter": 600},
        },
        reachability_config=ReachabilityConfig(
            maximum_speed_mps=0.5,
            active_horizon_s=10.0,
            contingency_horizon_s=6.0,
            required_reachable=1,
        ),
        mission=_mission(),
        zone_validation_config=LandingZoneValidationConfig(
            minimum_clearance_m=0.0,
            minimum_valid_seed_cells=1,
        ),
        metrics=metrics,
    )
    return worker, geometry, metrics


def _task(version: int, geometry: GridGeometry, occupancy: np.ndarray | None = None) -> SafetySynthesisTask:
    occ = np.zeros(geometry.shape_yx, dtype=bool) if occupancy is None else occupancy
    return SafetySynthesisTask(
        version_id=version,
        submitted_time_s=perf_counter(),
        raw_occupancy=occ.copy(),
        filtered_occupancy=occ.copy(),
        inflated_occupancy=occ.copy(),
        forcing_method="constant",
        solver="conjugate_gradient",
        compute_laplacian_check=False,
    )


def test_worker_accepts_only_latest_version_and_keeps_queue_bounded() -> None:
    worker, geometry, metrics = _worker()
    try:
        for version in range(1, 7):
            worker.submit(_task(version, geometry))
            assert worker.qsize() <= 1
        deadline = perf_counter() + 10.0
        snapshot = None
        while perf_counter() < deadline:
            snapshot = worker.latest_snapshot()
            if snapshot is not None and snapshot.version_id == 6:
                break
            sleep(0.02)
        assert snapshot is not None
        assert snapshot.version_id == 6
        assert snapshot.reachability.occupancy_version == 6
        assert worker.maximum_observed_size <= 1
        assert any(row.get("version_id") == 6 and row.get("accepted") for row in metrics.solve_rows)
    finally:
        worker.close()
