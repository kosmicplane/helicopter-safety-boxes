"""Live vision-to-Poisson-CBF and HJ contingency-planning pipeline.

This optional pipeline preserves the original camera/segmentation architecture but
adds interactive mission selection, synchronized geodesic HJR fields, certified
r-out-of-p target switching, path following, and one unified CBF Safety Box
projection.  It controls only a virtual planar marker.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .calibration import CalibrationData, MotionEstimate, assume_top_down_calibration, interactive_calibration, rectify_image
from .contingency_planner import PlannedPath, PlannerConfig, construct_active_path, pure_pursuit_velocity
from .contingency_visualization import draw_contingency_overlay, save_reachability_snapshot_figures
from .coordinates import GridFieldSampler, GridGeometry
from .discrete_safety import backtracked_safe_step
from .hj_reachability import (
    ReachabilityConfig,
    compute_reachability_status,
    pivot_and_reachable_count_fields,
    sample_zone_field,
)
from .io_utils import save_json, save_yaml, write_csv
from .live_pipeline import LiveExperimentReport, LivePoissonPipeline, VideoSource
from .mission_setup import (
    MissionDefinition,
    MissionSetupConfig,
    load_or_select_mission,
    validate_mission,
)
from .occupancy import changed_fraction, inflate_occupancy_physical, mask_to_occupancy
from .poisson_visualization import render_live_dashboard, save_live_poisson_surface
from .safety_synthesis import SafetySynthesisSnapshot, SafetySynthesisTask, SafetySynthesisWorker
from .segmentation import segment_image
from .target_manager import (
    LandingZoneValidationConfig,
    TargetManager,
    ZoneState,
    assess_all_landing_zones,
)
from .unified_contingency_filter import (
    ContingencyFilterConfig,
    PoissonCBFConfig,
    UnifiedContingencyFilter,
    UnifiedFilterResult,
)


class LiveContingencyPipeline(LivePoissonPipeline):
    """Live virtual-vehicle contingency workflow enabled by ``reachability.enabled``."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.mission_setup_config = MissionSetupConfig.from_dict(self.config.get("mission_setup", {}))
        self.zone_validation_config = LandingZoneValidationConfig.from_dict(
            self.config.get("landing_zone_validation", {})
        )
        self.reachability_config = ReachabilityConfig.from_dict(self.config.get("reachability", {}))
        self.planner_config = PlannerConfig.from_dict(self.config.get("planner", {}))
        self.poisson_cbf_config = PoissonCBFConfig.from_dict(self.config.get("poisson_cbf", {}))
        self.contingency_filter_config = ContingencyFilterConfig.from_dict(
            self.config.get("contingency_filter", {})
        )
        if not self.reachability_config.enabled:
            raise ValueError("LiveContingencyPipeline requires reachability.enabled=true.")
        self.geometry = GridGeometry(shape_yx=self.grid_shape_yx, spacing_yx=self.grid_spacing_yx)
        self.mission: MissionDefinition | None = None
        self.target_manager: TargetManager | None = None
        self.unified_filter: UnifiedContingencyFilter | None = None
        self.virtual_position_xy: np.ndarray | None = None
        self.tau_active = -self.reachability_config.active_horizon_s
        self.tau_contingency = -self.reachability_config.contingency_horizon_s
        self._controller_last_time_s: float | None = None
        self._current_path: PlannedPath | None = None
        self._current_lookahead_xy: np.ndarray | None = None
        self._last_filter_result: UnifiedFilterResult | None = None
        self._last_nominal_velocity = np.zeros(2, dtype=float)
        self._hold_reason: str | None = None
        self._show_hjr = True
        self._show_path = True
        self._mission_redefinition_requested = False
        self._manual_target_cycle_requested = False
        self._manual_rejection_reset_requested = False
        self._controller_enabled = bool(self.config.get("contingency_demo", {}).get("enabled", True))
        self._controller_history: list[dict[str, Any]] = []
        self._switch_event_count_seen = 0
        self._latest_accepted_safety_version = -1
        self._last_snapshot_for_figures: SafetySynthesisSnapshot | None = None

        # Interactive runs remain in SETUP HOLD after workspace and mission
        # selection. The live camera continues updating, but occupancy,
        # Poisson, HJR, planning, and virtual motion do not begin until SPACE.
        self._experiment_started = bool(self.headless)

    def _build_initial_occupancy(self, rectified: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """Segment one rectified frame and build raw, filtered, and inflated grids."""

        segmentation = segment_image(
            rectified,
            self._current_segmentation_config(),
            base_directory=self.config_directory,
            background_reference=self._background_reference,
            allow_interactive=not self.headless,
        )
        instantaneous = mask_to_occupancy(segmentation.clean_mask, self.grid_shape_yx)
        filtered = instantaneous.copy()
        radius = float(self.geometry_config.get("robot_radius_m", 0.0)) + float(
            self.geometry_config.get("perception_margin_m", 0.0)
        )
        inflated = inflate_occupancy_physical(filtered, radius, self.grid_spacing_yx)
        return instantaneous, filtered, inflated, segmentation

    def _initialize_mission(
        self,
        *,
        rectified: np.ndarray,
        calibration: CalibrationData,
        raw_occupancy: np.ndarray,
        inflated_occupancy: np.ndarray,
    ) -> None:
        """Load/select a mission and reset every mission-dependent runtime state."""

        self.mission = load_or_select_mission(
            rectified_bgr=rectified,
            calibration=calibration,
            geometry=self.geometry,
            setup_config=self.mission_setup_config,
            robot_radius_m=float(self.geometry_config.get("robot_radius_m", 0.0)),
            perception_margin_m=float(self.geometry_config.get("perception_margin_m", 0.0)),
            base_directory=self.config_directory,
            headless=self.headless,
            raw_occupancy=raw_occupancy,
            inflated_occupancy=inflated_occupancy,
        )
        self.mission.save(self.output_directory / "mission_setup.json")
        configured_path = (self.config_directory / self.mission_setup_config.file).expanduser().resolve()
        if self.mission_setup_config.save_selection and self.mission_setup_config.mode == "interactive":
            self.mission.save(configured_path)
        self.target_manager = TargetManager(self.mission, self.zone_validation_config)
        self.virtual_position_xy = self.mission.start_xy_m.copy()
        self.tau_active = -self.reachability_config.active_horizon_s
        self.tau_contingency = -self.reachability_config.contingency_horizon_s
        self._controller_last_time_s = perf_counter()
        self._current_path = None
        self._last_filter_result = None
        self._hold_reason = None
        self._last_submitted_occupancy = None
        self._last_submit_time_s = -np.inf
        self.unified_filter = UnifiedContingencyFilter(
            poisson_config=self.poisson_cbf_config,
            filter_config=self.contingency_filter_config,
            maximum_speed_mps=self.reachability_config.maximum_speed_mps,
            required_reachable=self.mission.required_reachable,
        )

    def _submit_safety_if_needed(
        self,
        *,
        raw_occupancy: np.ndarray,
        filtered_occupancy: np.ndarray,
        inflated_occupancy: np.ndarray,
        current_time_s: float,
        worker: SafetySynthesisWorker,
    ) -> tuple[bool, float]:
        """Submit a synchronized task when geometry changes or refresh is due."""

        fraction = changed_fraction(self._last_submitted_occupancy, inflated_occupancy)
        due_to_change = fraction >= self.minimum_changed_fraction
        due_to_time = current_time_s - self._last_submit_time_s >= self.maximum_submit_interval_s
        if not due_to_change and not due_to_time:
            return False, fraction
        self._version_counter += 1
        diagnostic = self.laplacian_every_n_solves > 0 and self._version_counter % self.laplacian_every_n_solves == 0
        rejected = tuple(
            identifier
            for identifier, state in (self.target_manager.states.items() if self.target_manager else [])
            if state.state == ZoneState.REJECTED
        )
        worker.submit(
            SafetySynthesisTask(
                version_id=self._version_counter,
                submitted_time_s=current_time_s,
                raw_occupancy=np.asarray(raw_occupancy, dtype=bool).copy(),
                filtered_occupancy=np.asarray(filtered_occupancy, dtype=bool).copy(),
                inflated_occupancy=np.asarray(inflated_occupancy, dtype=bool).copy(),
                forcing_method=self._current_forcing_method(),
                solver=self.poisson_solver,
                compute_laplacian_check=diagnostic,
                unavailable_identifiers=rejected,
            )
        )
        self._last_submitted_occupancy = np.asarray(inflated_occupancy, dtype=bool).copy()
        self._last_submit_time_s = current_time_s
        return True, fraction

    def _backtracked_step(
        self,
        *,
        position: np.ndarray,
        safe_velocity: np.ndarray,
        nominal_dt: float,
        snapshot: SafetySynthesisSnapshot,
    ) -> tuple[np.ndarray, float, int, str | None]:
        """Apply a collision- and Poisson-checked discrete marker step."""

        sampler = GridFieldSampler(snapshot.poisson_record.result, self.geometry)
        result = backtracked_safe_step(
            position_xy=position,
            velocity_xy=safe_velocity,
            nominal_dt_s=nominal_dt,
            geometry=self.geometry,
            inflated_occupancy=snapshot.inflated_occupancy,
            poisson_sampler=sampler,
            h_margin=self.poisson_cbf_config.h_margin,
            tolerance=1.0e-6,
            maximum_backtracks=12,
            maximum_dt_s=0.12,
        )
        return result.position_xy, result.accepted_dt_s, result.backtracks, result.reason

    def _controller_step(
        self,
        *,
        snapshot: SafetySynthesisSnapshot | None,
        camera_moved: bool,
        field_age_s: float,
    ) -> dict[str, Any]:
        """Update target logic, path, unified filter, and virtual marker once."""

        assert self.mission is not None and self.target_manager is not None
        assert self.virtual_position_xy is not None and self.unified_filter is not None
        now = perf_counter()
        if self._controller_last_time_s is None:
            self._controller_last_time_s = now
        elapsed = max(0.0, min(0.2, now - self._controller_last_time_s))
        self._controller_last_time_s = now
        if not self._controller_enabled:
            self._hold_reason = "virtual contingency controller disabled"
            return {"contingency_status": "disabled"}
        if camera_moved:
            self._hold_reason = "camera moved; metric map invalid"
            return {"contingency_status": "hold", "hold_reason": self._hold_reason}
        if snapshot is None:
            self._hold_reason = "waiting for synchronized Poisson/HJR snapshot"
            return {"contingency_status": "hold", "hold_reason": self._hold_reason}
        maximum_age = min(self.reachability_config.maximum_field_age_s, self.poisson_cbf_config.maximum_field_age_s)
        if field_age_s > maximum_age:
            self._hold_reason = f"synchronized safety snapshot is stale ({field_age_s:.3f} s)"
            return {"contingency_status": "hold", "hold_reason": self._hold_reason}
        if snapshot.version_id != snapshot.reachability.occupancy_version:
            self._hold_reason = "Poisson/HJR occupancy versions do not match"
            return {"contingency_status": "hold", "hold_reason": self._hold_reason}

        # Use temporally filtered/inflated occupancy assessments every accepted map.
        self.target_manager.update_assessments(snapshot.zone_assessments)
        available = self.target_manager.available_identifiers()
        status = compute_reachability_status(
            snapshot.reachability,
            point_xy=self.virtual_position_xy,
            tau=self.tau_contingency,
            maximum_speed_mps=self.reachability_config.maximum_speed_mps,
            required_reachable=self.mission.required_reachable,
            available_identifiers=available,
        )
        self.target_manager.update_reachability(values=status.values, distances=status.distances_m)
        available = self.target_manager.available_identifiers()

        if status.reachable_count < self.mission.required_reachable or not np.isfinite(status.pivot) or status.pivot < 0.0:
            self._hold_reason = (
                f"CONTINGENCY REQUIREMENT LOST: reachable={status.reachable_count}, "
                f"required={self.mission.required_reachable}, pivot={status.pivot}"
            )
            self.target_manager.hold_reason = self._hold_reason
            return {
                "contingency_status": "hold",
                "hold_reason": self._hold_reason,
                "reachable_count": status.reachable_count,
                "pivot": status.pivot,
            }

        active_runtime = self.target_manager.states[self.target_manager.active_identifier]
        switch_reason: str | None = None
        if active_runtime.state in {ZoneState.REJECTED, ZoneState.BLOCKED}:
            switch_reason = active_runtime.rejection_reason or "active landing zone blocked"
        else:
            active_field = snapshot.reachability.fields[self.target_manager.active_identifier]
            active_hj_sample = sample_zone_field(
                active_field,
                snapshot.reachability.geometry,
                self.virtual_position_xy,
                tau=self.tau_active,
                maximum_speed_mps=self.reachability_config.maximum_speed_mps,
            )
            if not active_hj_sample.valid or active_hj_sample.value < 0.0:
                self.target_manager.reject(
                    self.target_manager.active_identifier,
                    "active target is not reachable within the active horizon",
                )
                switch_reason = "active target became HJ-unreachable"

        if self._manual_rejection_reset_requested:
            self.target_manager.reset_rejections()
            self._manual_rejection_reset_requested = False
            available = self.target_manager.available_identifiers()
        if self._manual_target_cycle_requested:
            event = self.target_manager.cycle_manual_target(
                time_s=now,
                occupancy_version=snapshot.version_id,
            )
            self._manual_target_cycle_requested = False
            if event is not None:
                self.tau_active = self.tau_contingency
                self._current_path = None

        if switch_reason is not None:
            alternative = self.target_manager.choose_certified_alternative(
                exclude_identifier=self.target_manager.active_identifier
            )
            if alternative is None:
                self._hold_reason = "active target failed and no certified alternative exists"
                return {
                    "contingency_status": "hold",
                    "hold_reason": self._hold_reason,
                    "reachable_count": status.reachable_count,
                    "pivot": status.pivot,
                }
            self.target_manager.switch_to(
                alternative,
                time_s=now,
                reason=switch_reason,
                occupancy_version=snapshot.version_id,
            )
            self.tau_active = self.tau_contingency
            self._current_path = None
            available = self.target_manager.available_identifiers()
            status = compute_reachability_status(
                snapshot.reachability,
                point_xy=self.virtual_position_xy,
                tau=self.tau_contingency,
                maximum_speed_mps=self.reachability_config.maximum_speed_mps,
                required_reachable=self.mission.required_reachable,
                available_identifiers=available,
            )
            self.target_manager.update_reachability(values=status.values, distances=status.distances_m)

        self._current_path = construct_active_path(
            snapshot.reachability,
            position_xy=self.virtual_position_xy,
            active_zone_identifier=self.target_manager.active_identifier,
            config=self.planner_config,
        )
        if not self._current_path.valid:
            self._hold_reason = self._current_path.reason
            return {"contingency_status": "hold", "hold_reason": self._hold_reason}
        nominal, lookahead, reached = pure_pursuit_velocity(
            self.virtual_position_xy,
            self._current_path.points_xy,
            config=self.planner_config,
        )
        self._last_nominal_velocity = nominal
        self._current_lookahead_xy = lookahead
        active_zone = self.mission.zone_by_identifier(self.target_manager.active_identifier)
        if reached or float(np.linalg.norm(self.virtual_position_xy - active_zone.center_xy_m)) <= active_zone.radius_m:
            self.target_manager.mark_reached(active_zone.identifier)
            self._hold_reason = "active landing-zone disk reached"
            return {"contingency_status": "reached", "active_target": active_zone.identifier}

        sampler = GridFieldSampler(snapshot.poisson_record.result, self.geometry)
        poisson_sample = sampler.sample(self.virtual_position_xy)
        result = self.unified_filter.filter(
            position_xy=self.virtual_position_xy,
            nominal_velocity_xy=nominal,
            poisson_sample=poisson_sample,
            reachability_bundle=snapshot.reachability,
            active_identifier=self.target_manager.active_identifier,
            available_identifiers=available,
            tau_active=self.tau_active,
            tau_active_dot=1.0,
            tau_contingency=self.tau_contingency,
            tau_contingency_dot=0.0,
        )
        self._last_filter_result = result
        if not result.success:
            self._hold_reason = result.hold_reason
            return {
                "contingency_status": "hold",
                "hold_reason": self._hold_reason,
                **result.to_dict(),
            }

        candidate, accepted_dt, backtracks, rejected_reason = self._backtracked_step(
            position=self.virtual_position_xy,
            safe_velocity=result.safe_velocity_xy,
            nominal_dt=elapsed,
            snapshot=snapshot,
        )
        if rejected_reason is not None:
            self._hold_reason = "discrete-time safety check failed: " + rejected_reason
            return {
                "contingency_status": "hold",
                "hold_reason": self._hold_reason,
                "backtracks": backtracks,
                **result.to_dict(),
            }
        self.virtual_position_xy = candidate
        self.tau_active = min(0.0, self.tau_active + accepted_dt)
        self._hold_reason = None
        self.target_manager.hold_reason = None
        row = {
            "time_s": now,
            "occupancy_version": snapshot.version_id,
            "position_x_m": float(self.virtual_position_xy[0]),
            "position_y_m": float(self.virtual_position_xy[1]),
            "active_target": self.target_manager.active_identifier,
            "tau_active": float(self.tau_active),
            "tau_contingency": float(self.tau_contingency),
            "accepted_dt_s": float(accepted_dt),
            "backtracks": int(backtracks),
            "nominal_u_x": float(nominal[0]),
            "nominal_u_y": float(nominal[1]),
            "safe_u_x": float(result.safe_velocity_xy[0]),
            "safe_u_y": float(result.safe_velocity_xy[1]),
            "intervention_norm": float(np.linalg.norm(result.safe_velocity_xy - nominal)),
            **result.to_dict(),
        }
        self._controller_history.append(row)
        return {"contingency_status": "optimal", **row}

    def _draw_live_overlay(
        self,
        rectified: np.ndarray,
        snapshot: SafetySynthesisSnapshot | None,
        field_age_s: float,
    ) -> np.ndarray:
        """Draw mission, path, marker, and command/certificate diagnostics."""

        assert self.mission is not None and self.target_manager is not None and self.virtual_position_xy is not None
        return draw_contingency_overlay(
            rectified,
            mission=self.mission,
            target_manager=self.target_manager,
            position_xy=self.virtual_position_xy,
            path_xy=(None if self._current_path is None else self._current_path.points_xy),
            nominal_velocity_xy=self._last_nominal_velocity,
            filter_result=self._last_filter_result,
            lookahead_xy=self._current_lookahead_xy,
            show_hjr=self._show_hjr,
            show_path=self._show_path,
            arrow_scale_px_per_mps=float(self.config.get("contingency_demo", {}).get("arrow_scale_px_per_mps", 150.0)),
            occupancy_version=None if snapshot is None else snapshot.version_id,
            field_age_s=field_age_s,
        )

    def _handle_contingency_key(self, key: int, rectified: np.ndarray) -> bool:
        """Apply original and contingency-specific keyboard controls."""

        if key == 32:  # SPACE
            if not self._experiment_started:
                self._experiment_started = True
                self.temporal_filter.reset()
                self._last_submitted_occupancy = None
                self._controller_last_time_s = None
                self._hold_reason = None
                print(
                    "[LIVE] Experiment started. "
                    "Occupancy, Poisson, HJR, planning, and virtual control are enabled.",
                    flush=True,
                )
            return False

        if key == ord("l"):
            self._mission_redefinition_requested = True
            return False
        if key == ord("h"):
            self._show_hjr = not self._show_hjr
            return False
        if key == ord("g"):
            self._show_path = not self._show_path
            return False
        if key == ord("a"):
            self._manual_target_cycle_requested = True
            return False
        if key == ord("x"):
            self._manual_rejection_reset_requested = True
            return False
        if key == ord("c"):
            self._controller_enabled = not self._controller_enabled
            return False
        return super()._handle_key(key, rectified)

    def _save_contingency_snapshot(
        self,
        *,
        directory: Path,
        snapshot: SafetySynthesisSnapshot,
    ) -> None:
        """Persist all synchronized HJR, path, target, and controller artifacts."""

        assert self.mission is not None and self.target_manager is not None
        directory.mkdir(parents=True, exist_ok=True)
        self.mission.save(directory / "mission_setup.json")
        save_json(directory / "landing_zone_states.json", self.target_manager.to_dict())
        np.save(directory / "raw_occupancy.npy", snapshot.raw_occupancy)
        np.save(directory / "inflated_occupancy.npy", snapshot.inflated_occupancy)
        np.save(directory / "poisson_h.npy", snapshot.poisson_record.result.h)
        np.save(directory / "poisson_grad_h.npy", snapshot.poisson_record.result.grad_h)
        target_masks: dict[str, np.ndarray] = {}
        predecessors: dict[str, np.ndarray] = {}
        available = self.target_manager.available_identifiers()
        for identifier, field in snapshot.reachability.fields.items():
            np.save(directory / f"geodesic_distance_LZ_{identifier}.npy", field.distance_m)
            np.save(
                directory / f"HJ_active_LZ_{identifier}.npy",
                field.value_field(self.tau_active, self.reachability_config.maximum_speed_mps),
            )
            np.save(
                directory / f"HJ_contingency_LZ_{identifier}.npy",
                field.value_field(self.tau_contingency, self.reachability_config.maximum_speed_mps),
            )
            target_masks[f"LZ_{identifier}"] = field.target_seed_mask
            predecessors[f"LZ_{identifier}"] = field.predecessor_yx
        np.savez_compressed(directory / "target_seed_masks.npz", **target_masks)
        np.savez_compressed(directory / "predecessor_maps.npz", **predecessors)
        pivot_field, reachable_count_field, _contingency_values = pivot_and_reachable_count_fields(
            snapshot.reachability,
            tau=self.tau_contingency,
            maximum_speed_mps=self.reachability_config.maximum_speed_mps,
            required_reachable=self.mission.required_reachable,
            available_identifiers=available,
        )
        np.save(directory / "combinatorial_pivot.npy", pivot_field)
        np.save(directory / "reachable_count_matrix.npy", reachable_count_field)
        if self._current_path is not None:
            np.savetxt(directory / "active_path.csv", self._current_path.points_xy, delimiter=",", header="x_m,y_m", comments="")
        else:
            np.savetxt(directory / "active_path.csv", np.empty((0, 2)), delimiter=",", header="x_m,y_m", comments="")
        save_json(
            directory / "virtual_vehicle_state.json",
            {
                "position_xy_m": self.virtual_position_xy.tolist() if self.virtual_position_xy is not None else None,
                "tau_active": self.tau_active,
                "tau_contingency": self.tau_contingency,
                "active_target": self.target_manager.active_identifier,
                "occupancy_version": snapshot.version_id,
            },
        )
        write_csv(directory / "target_switch_events.csv", [event.to_dict() for event in self.target_manager.switch_events])
        write_csv(directory / "controller_metrics.csv", self._controller_history)
        write_csv(
            directory / "solve_timing.csv",
            [
                {
                    "occupancy_version": snapshot.version_id,
                    "poisson_solve_time_s": snapshot.poisson_solve_time_s,
                    "hjr_solve_time_s": snapshot.hjr_solve_time_s,
                    "snapshot_age_s": snapshot.age_s,
                    "optimizer_solve_time_s": (
                        None if self._last_filter_result is None else self._last_filter_result.solve_time_s
                    ),
                }
            ],
        )
        if self._last_filter_result and self._last_filter_result.box_result:
            result = self._last_filter_result.box_result
            np.save(directory / "constraint_matrix.npy", result.constraint_matrix)
            np.save(directory / "constraint_vector.npy", result.constraint_vector)
            write_csv(
                directory / "residuals.csv",
                [
                    {"constraint": name, "residual": residual}
                    for name, residual in self._last_filter_result.residuals.items()
                ],
            )
        save_json(
            directory / "synchronized_snapshot.json",
            {
                "occupancy_version": snapshot.version_id,
                "poisson_solve_time_s": snapshot.poisson_solve_time_s,
                "hjr_solve_time_s": snapshot.hjr_solve_time_s,
                "snapshot_age_s": snapshot.age_s,
            },
        )
        save_json(
            directory / "HJR_validation.json",
            {
                str(identifier): field.validation
                for identifier, field in snapshot.reachability.fields.items()
            },
        )
        save_reachability_snapshot_figures(
            directory / "hjr_figures",
            mission=self.mission,
            bundle=snapshot.reachability,
            active_identifier=self.target_manager.active_identifier,
            tau_active=self.tau_active,
            tau_contingency=self.tau_contingency,
            maximum_speed_mps=self.reachability_config.maximum_speed_mps,
            required_reachable=self.mission.required_reachable,
            available_identifiers=available,
            path_xy=(None if self._current_path is None else self._current_path.points_xy),
            dpi=int(self.output_config.get("snapshot_3d_dpi", 110)),
        )

    def run(self) -> LiveExperimentReport:
        """Execute live capture, synchronized safety synthesis, and virtual control."""

        source_cfg = self.live_config.get("source", {})
        video = VideoSource(
            self.source_value,
            reconnection_attempts=int(source_cfg.get("reconnection_attempts", 2)),
            reconnection_delay_s=float(source_cfg.get("reconnection_delay_s", 0.2)),
        )
        worker: SafetySynthesisWorker | None = None
        calibration: CalibrationData | None = None
        motion_detector = None
        frame_index = 0
        stop_requested = False
        last_frame: np.ndarray | None = None
        last_segmentation = None
        last_instantaneous = np.zeros(self.grid_shape_yx, dtype=bool)
        last_filtered = np.zeros(self.grid_shape_yx, dtype=bool)
        last_inflated = np.zeros(self.grid_shape_yx, dtype=bool)

        try:
            ok, first_frame, _timestamp = video.read()
            if not ok or first_frame is None:
                raise RuntimeError("The video source opened but did not provide an initial frame.")
            calibration = self._initial_calibration(first_frame)
            calibration.save(self.output_directory / "calibration.json")
            first_rectified = rectify_image(first_frame, calibration)
            self._background_reference = self._load_initial_background(first_rectified)
            if self._background_reference is not None:
                cv2.imwrite(str(self.output_directory / "background_reference.png"), self._background_reference)
            motion_detector = self._configure_motion_detector(first_rectified)
            initial_raw, initial_filtered, initial_inflated, initial_segmentation = self._build_initial_occupancy(first_rectified)
            self._initialize_mission(
                rectified=first_rectified,
                calibration=calibration,
                raw_occupancy=initial_filtered,
                inflated_occupancy=initial_inflated,
            )
            assert self.mission is not None
            worker = SafetySynthesisWorker(
                geometry=self.geometry,
                poisson_config=self.poisson_config,
                reachability_config=self.reachability_config,
                mission=self.mission,
                zone_validation_config=self.zone_validation_config,
                metrics=self.metrics,
            )
            pending_frame: np.ndarray | None = first_frame

            while not stop_requested:
                frame_start = perf_counter()
                if self.maximum_frames is not None and frame_index >= self.maximum_frames:
                    break
                if pending_frame is not None:
                    frame = pending_frame
                    pending_frame = None
                    capture_latency = 0.0
                elif self.state.paused and last_frame is not None:
                    frame = last_frame.copy()
                    capture_latency = 0.0
                else:
                    capture_start = perf_counter()
                    ok, frame, _capture_timestamp = video.read()
                    capture_latency = perf_counter() - capture_start
                    if not ok or frame is None:
                        break
                last_frame = frame.copy()
                frame_index += 1
                capture_fps = self.capture_rate.tick()

                recalibrated = False
                if self.state.recalibration_requested:
                    calibration = (
                        assume_top_down_calibration(
                            frame.shape,
                            output_size_px=self.rectified_size_px,
                            workspace_size_m=self.workspace_size_m,
                        )
                        if self.headless
                        else interactive_calibration(
                            frame,
                            output_size_px=self.rectified_size_px,
                            workspace_size_m=self.workspace_size_m,
                        )
                    )
                    calibration.save(self.output_directory / "calibration.json")
                    self.state.recalibration_requested = False
                    recalibrated = True
                    self.temporal_filter.reset()
                    self._last_submitted_occupancy = None
                    self._mission_redefinition_requested = not self.headless

                assert calibration is not None
                rectification_start = perf_counter()
                rectified = rectify_image(frame, calibration)
                rectification_latency = perf_counter() - rectification_start
                if recalibrated and motion_detector is not None:
                    motion_detector.set_reference(rectified)

                # Keep the camera live after mission selection while allowing
                # the operator to place obstacles. No safety computation or
                # virtual motion occurs until SPACE is pressed.
                if not self._experiment_started and not self.headless:
                    setup_preview = rectified.copy()

                    overlay = setup_preview.copy()
                    cv2.rectangle(
                        overlay,
                        (0, 0),
                        (setup_preview.shape[1], 105),
                        (0, 0, 0),
                        thickness=-1,
                    )
                    cv2.addWeighted(
                        overlay,
                        0.65,
                        setup_preview,
                        0.35,
                        0.0,
                        setup_preview,
                    )

                    cv2.putText(
                        setup_preview,
                        "SETUP HOLD - PLACE OBSTACLES",
                        (20, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.82,
                        (0, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        setup_preview,
                        "Press SPACE to start | B: capture empty background | Q: quit",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.56,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        setup_preview,
                        "Poisson, HJR, path planning, and marker motion are currently disabled.",
                        (20, 96),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.46,
                        (190, 190, 190),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.imshow(
                        "Phone Stream Poisson-CBF + HJ Contingency",
                        setup_preview,
                    )
                    key = cv2.waitKey(1) & 0xFF
                    stop_requested = self._handle_contingency_key(
                        key,
                        rectified,
                    )
                    continue

                motion_estimate: MotionEstimate | None = None
                camera_moved = False
                if motion_detector is not None:
                    motion_estimate = (
                        MotionEstimate(True, False, 0.0, 0.0, 1.0, 0, "reference_reset_after_recalibration")
                        if recalibrated
                        else motion_detector.estimate(rectified)
                    )
                    camera_moved = bool(motion_estimate.valid and motion_estimate.moved)

                segmentation_start = perf_counter()
                segmentation = segment_image(
                    rectified,
                    self._current_segmentation_config(),
                    base_directory=self.config_directory,
                    background_reference=self._background_reference,
                    allow_interactive=not self.headless,
                )
                segmentation_latency = perf_counter() - segmentation_start
                last_segmentation = segmentation

                occupancy_start = perf_counter()
                instantaneous = mask_to_occupancy(segmentation.clean_mask, self.grid_shape_yx)
                filtered = self.temporal_filter.update(instantaneous)
                inflation_radius = float(self.geometry_config.get("robot_radius_m", 0.0)) + float(
                    self.geometry_config.get("perception_margin_m", 0.0)
                )
                inflated = inflate_occupancy_physical(filtered, inflation_radius, self.grid_spacing_yx)
                occupancy_latency = perf_counter() - occupancy_start
                last_instantaneous, last_filtered, last_inflated = instantaneous, filtered, inflated

                if self._mission_redefinition_requested:
                    self._initialize_mission(
                        rectified=rectified,
                        calibration=calibration,
                        raw_occupancy=filtered,
                        inflated_occupancy=inflated,
                    )
                    assert self.mission is not None
                    worker.close(timeout_s=float(self.live_config.get("worker_shutdown_timeout_s", 15.0)))
                    worker = SafetySynthesisWorker(
                        geometry=self.geometry,
                        poisson_config=self.poisson_config,
                        reachability_config=self.reachability_config,
                        mission=self.mission,
                        zone_validation_config=self.zone_validation_config,
                        metrics=self.metrics,
                    )
                    self._mission_redefinition_requested = False
                    self._experiment_started = bool(self.headless)
                    self.temporal_filter.reset()
                    self._last_submitted_occupancy = None
                    self._controller_last_time_s = None

                # Fast per-frame hysteresis uses the temporally filtered occupancy.
                assert self.target_manager is not None and self.mission is not None
                assessments = assess_all_landing_zones(
                    self.mission,
                    raw_occupancy=filtered,
                    inflated_occupancy=inflated,
                    geometry=self.geometry,
                    config=self.zone_validation_config,
                )
                self.target_manager.update_assessments(assessments)

                submitted = False
                occupancy_change = changed_fraction(self._last_submitted_occupancy, inflated)
                if not camera_moved:
                    submitted, occupancy_change = self._submit_safety_if_needed(
                        raw_occupancy=filtered,
                        filtered_occupancy=filtered,
                        inflated_occupancy=inflated,
                        current_time_s=perf_counter(),
                        worker=worker,
                    )
                snapshot = worker.latest_snapshot()
                new_snapshot = self._accept_new_snapshot(snapshot)
                if new_snapshot and snapshot is not None:
                    self._latest_accepted_safety_version = snapshot.version_id
                    self._last_snapshot_for_figures = snapshot
                    if self.snapshot_every_n_solves and snapshot.version_id % self.snapshot_every_n_solves == 0:
                        self.state.snapshot_requested = True
                field_age_s = snapshot.age_s if snapshot is not None else np.inf
                field_stale = snapshot is not None and field_age_s > min(
                    self.reachability_config.maximum_field_age_s,
                    self.poisson_cbf_config.maximum_field_age_s,
                )
                controller_metrics = self._controller_step(
                    snapshot=snapshot,
                    camera_moved=camera_moved,
                    field_age_s=field_age_s,
                )
                annotated = self._draw_live_overlay(rectified, snapshot, field_age_s)
                poisson_result = snapshot.poisson_record.result if snapshot is not None else None
                validation = snapshot.poisson_record.validation.get("result", {}) if snapshot is not None else {}
                dashboard_metrics = {
                    "capture_fps": capture_fps,
                    "display_fps": self.display_rate.rate,
                    "poisson_updates_per_s": self.poisson_rate.rate,
                    "last_solve_time_s": snapshot.poisson_solve_time_s if snapshot is not None else 0.0,
                    "field_age_s": field_age_s if np.isfinite(field_age_s) else 0.0,
                    "poisson_residual": validation.get("residual_max_abs", np.nan),
                    "grid_shape": f"{self.grid_shape_yx[1]}x{self.grid_shape_yx[0]}",
                    "forcing_method": self._current_forcing_method(),
                    "solver": self.poisson_solver,
                    **controller_metrics,
                }
                warnings: list[str] = []
                if camera_moved:
                    warnings.append("CAMERA MOVED - MAP NOT METRIC - HOLD")
                if field_stale:
                    warnings.append(f"SYNCHRONIZED SAFETY SNAPSHOT STALE: {1000.0 * field_age_s:.0f} ms")
                if snapshot is None:
                    warnings.append("WAITING FOR FIRST SYNCHRONIZED POISSON/HJR SNAPSHOT")
                if self._hold_reason:
                    warnings.append(f"HOLD: {self._hold_reason}")
                dashboard = render_live_dashboard(
                    original_bgr=frame,
                    rectified_bgr=annotated,
                    obstacle_mask=segmentation.clean_mask,
                    occupancy=inflated,
                    poisson_result=poisson_result,
                    metrics=dashboard_metrics,
                    warnings=warnings,
                    panel_size=tuple(self.output_config.get("dashboard_panel_size", [420, 280])),
                )
                display_fps = self.display_rate.tick()
                self._last_dashboard = dashboard
                self._open_dashboard_writer(dashboard, video.nominal_fps)
                if self._video_writer is not None:
                    self._video_writer.write(dashboard)

                frame_row = {
                    "frame_index": frame_index,
                    "frame_time_s": frame_start,
                    "capture_latency_s": capture_latency,
                    "rectification_latency_s": rectification_latency,
                    "segmentation_latency_s": segmentation_latency,
                    "occupancy_preprocessing_latency_s": occupancy_latency,
                    "pipeline_latency_s": perf_counter() - frame_start,
                    "capture_fps": capture_fps,
                    "display_fps": display_fps,
                    "poisson_updates_per_s": self.poisson_rate.rate,
                    "poisson_version": snapshot.version_id if snapshot is not None else None,
                    "field_age_s": field_age_s if np.isfinite(field_age_s) else None,
                    "last_solve_time_s": snapshot.poisson_solve_time_s if snapshot is not None else None,
                    "hjr_solve_time_s": snapshot.hjr_solve_time_s if snapshot is not None else None,
                    "poisson_residual": validation.get("residual_max_abs"),
                    "forcing_method": self._current_forcing_method(),
                    "solver": self.poisson_solver,
                    "grid_nx": self.grid_shape_yx[1],
                    "grid_ny": self.grid_shape_yx[0],
                    "occupied_fraction_instantaneous": float(np.mean(instantaneous)),
                    "occupied_fraction_filtered": float(np.mean(filtered)),
                    "occupied_fraction_inflated": float(np.mean(inflated)),
                    "changed_fraction": occupancy_change,
                    "submitted_poisson_task": submitted,
                    "camera_moved": camera_moved,
                    "camera_translation_px": motion_estimate.translation_px if motion_estimate is not None else None,
                    "camera_rotation_deg": motion_estimate.rotation_deg if motion_estimate is not None else None,
                    "field_stale": field_stale,
                    "discarded_queued_tasks": worker.discarded_items,
                    "discarded_obsolete_solves": worker.discarded_obsolete_solves,
                    "queue_size": worker.qsize(),
                    "queue_maximum_observed_size": worker.maximum_observed_size,
                    **controller_metrics,
                }
                self.metrics.add_frame(frame_row)

                if self.state.snapshot_requested and snapshot is not None:
                    directory = self.output_directory / "snapshots" / f"snapshot_{self._snapshot_counter:04d}"
                    self._snapshot_counter += 1
                    directory.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(directory / "frame.png"), frame)
                    cv2.imwrite(str(directory / "rectified.png"), rectified)
                    cv2.imwrite(str(directory / "dashboard.png"), dashboard)
                    cv2.imwrite(str(directory / "raw_mask.png"), segmentation.raw_mask)
                    cv2.imwrite(str(directory / "clean_mask.png"), segmentation.clean_mask)
                    snapshot.poisson_record.result.save_npz(directory / "poisson_result.npz")
                    snapshot.poisson_record.result.save_summary_json(directory / "poisson_summary.json")
                    if bool(self.output_config.get("save_3d_snapshots", True)):
                        save_live_poisson_surface(
                            snapshot.poisson_record,
                            directory / "poisson_surface_3d.png",
                            workspace_size_m=self.workspace_size_m,
                            dpi=int(self.output_config.get("snapshot_3d_dpi", 110)),
                        )
                    self._save_contingency_snapshot(directory=directory, snapshot=snapshot)
                    self.state.snapshot_requested = False

                if not self.headless:
                    cv2.imshow("Phone Stream Poisson-CBF + HJ Contingency", dashboard)
                    key = cv2.waitKey(1) & 0xFF
                    stop_requested = self._handle_contingency_key(key, rectified)

        finally:
            if worker is not None:
                worker.close(timeout_s=float(self.live_config.get("worker_shutdown_timeout_s", 15.0)))
            video.release()
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
            if not self.headless:
                cv2.destroyAllWindows()
            summary = self.metrics.save(self.output_directory)
            if self.target_manager is not None:
                save_json(self.output_directory / "landing_zone_states.json", self.target_manager.to_dict())
                write_csv(
                    self.output_directory / "target_switch_events.csv",
                    [event.to_dict() for event in self.target_manager.switch_events],
                )
            write_csv(self.output_directory / "controller_metrics.csv", self._controller_history)
            latest = worker.latest_snapshot() if worker is not None else None
            if latest is not None:
                latest.poisson_record.result.save_npz(self.output_directory / "last_valid_field.npz")
                latest.poisson_record.result.save_summary_json(self.output_directory / "last_valid_field_summary.json")
            worker_summary = {
                "worker_queue_max_observed": worker.maximum_observed_size if worker is not None else 0,
                "discarded_queued_tasks": worker.discarded_items if worker is not None else 0,
                "discarded_obsolete_solves": worker.discarded_obsolete_solves if worker is not None else 0,
                "failed_solves": worker.failed_solves if worker is not None else 0,
                "invalid_solves": worker.invalid_solves if worker is not None else 0,
            }
            summary["worker"] = worker_summary
            summary["contingency"] = {
                "enabled": True,
                "mission": None if self.mission is None else self.mission.to_dict(),
                "final_position_xy_m": None if self.virtual_position_xy is None else self.virtual_position_xy.tolist(),
                "final_active_target": None if self.target_manager is None else self.target_manager.active_identifier,
                "switch_count": 0 if self.target_manager is None else len(self.target_manager.switch_events),
                "hold_reason": self._hold_reason,
                "controller_update_count": len(self._controller_history),
            }
            save_json(self.output_directory / "summary.json", summary)
            save_yaml(self.output_directory / "effective_config.yaml", self.config)
            save_json(
                self.output_directory / "runtime_state.json",
                {
                    "source": self.source_value,
                    "headless": self.headless,
                    "maximum_frames": self.maximum_frames,
                    "metrics_summary": summary,
                    "queue_maximum_observed_size": worker_summary["worker_queue_max_observed"],
                    "discarded_queued_tasks": worker_summary["discarded_queued_tasks"],
                    "discarded_obsolete_solves": worker_summary["discarded_obsolete_solves"],
                    "reconnections": video.reconnections,
                },
            )

        if worker is None:
            raise RuntimeError("The contingency pipeline terminated before worker initialization.")
        return LiveExperimentReport(
            output_directory=self.output_directory,
            frames_processed=frame_index,
            metrics_path=self.output_directory / "metrics.csv",
            summary_path=self.output_directory / "summary.json",
            runtime_state_path=self.output_directory / "runtime_state.json",
            queue_maximum_observed_size=worker.maximum_observed_size,
            discarded_queued_tasks=worker.discarded_items,
            discarded_obsolete_solves=worker.discarded_obsolete_solves,
            reconnections=video.reconnections,
        )


__all__ = ["LiveContingencyPipeline"]
