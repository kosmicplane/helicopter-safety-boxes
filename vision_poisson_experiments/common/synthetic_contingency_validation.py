"""Headless deterministic validation scenarios for the live contingency stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .contingency_planner import PlannerConfig, construct_active_path, pure_pursuit_velocity
from .coordinates import GridFieldSampler, GridGeometry
from .discrete_safety import backtracked_safe_step
from .hj_reachability import (
    ReachabilityConfig,
    build_reachability_bundle,
    compute_reachability_status,
    sample_zone_field,
)
from .io_utils import save_json, write_csv
from .mission_setup import LandingZone, MissionDefinition, metric_disk_mask
from .occupancy import inflate_occupancy_physical
from .poisson_runner import run_poisson
from .target_manager import LandingZoneValidationConfig, TargetManager, ZoneState, assess_all_landing_zones
from .unified_contingency_filter import (
    ContingencyFilterConfig,
    PoissonCBFConfig,
    UnifiedContingencyFilter,
)


@dataclass(frozen=True)
class ScenarioResult:
    """Summary and time history for one deterministic validation case."""

    name: str
    summary: dict[str, Any]
    history: list[dict[str, Any]]
    trajectory: np.ndarray
    final_occupancy: np.ndarray


def _mission() -> MissionDefinition:
    return MissionDefinition(
        start_xy_m=np.array([0.55, 2.25]),
        landing_zones=(
            LandingZone(1, np.array([5.35, 0.95]), 0.25, priority=1.0),
            LandingZone(2, np.array([5.35, 3.55]), 0.25, priority=0.95),
            LandingZone(3, np.array([4.45, 2.25]), 0.25, priority=0.90),
            LandingZone(4, np.array([2.00, 4.00]), 0.25, priority=0.85),
        ),
        active_zone_identifier=1,
        required_reachable=2,
        workspace_size_m=(6.0, 4.5),
        calibration_hash="synthetic-validation",
    )


def _base_occupancy(geometry: GridGeometry) -> np.ndarray:
    occupancy = np.zeros(geometry.shape_yx, dtype=bool)
    # Two obstacles form a corridor but leave upper and lower routes available.
    x_values = np.arange(geometry.nx) * geometry.dx
    y_values = np.arange(geometry.ny) * geometry.dy
    xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
    occupancy |= (xx >= 2.45) & (xx <= 2.85) & (yy >= 1.45) & (yy <= 3.25)
    occupancy |= (xx >= 3.60) & (xx <= 3.90) & (yy >= 0.0) & (yy <= 0.55)
    return occupancy


def _disk_obstacle(geometry: GridGeometry, center: np.ndarray, radius: float) -> np.ndarray:
    return metric_disk_mask(geometry, center, radius)


def _poisson_record(occupancy: np.ndarray, geometry: GridGeometry):
    config = {
        "boundary_value": 0.0,
        "outer_boundary_as_dirichlet": True,
        "compute_gradient": True,
        "compute_hessian": False,
        "compute_laplacian_check": False,
        "validation_boundary_tolerance": 1.0e-8,
        "validation_residual_tolerance": 1.0e-5,
        "constant": {"c": 1.0},
        "conjugate_gradient": {"tolerance": 1.0e-8, "max_iter": 1600},
    }
    return run_poisson(
        occupancy,
        grid_spacing_yx=geometry.spacing_yx,
        poisson_config=config,
        forcing_method="constant",
        solver="conjugate_gradient",
        live_mode=True,
    )


def run_scenario(name: str, output_directory: str | Path) -> ScenarioResult:
    """Run one of five required synthetic validation scenarios."""

    if name not in {"clear_active", "active_zone_blocked", "corridor_blocked", "contingency_lost", "camera_moved"}:
        raise ValueError(f"Unknown scenario: {name}")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    geometry = GridGeometry(width_m=6.0, height_m=4.5, nx=81, ny=61)
    mission = _mission()
    raw = _base_occupancy(geometry)
    inflated = inflate_occupancy_physical(raw, 0.12, geometry.spacing_yx)
    reach_cfg = ReachabilityConfig(
        maximum_speed_mps=0.60,
        active_horizon_s=22.0,
        contingency_horizon_s=12.0,
        required_reachable=2,
        connectivity=8,
        maximum_field_age_s=2.0,
    )
    planner_cfg = PlannerConfig(
        lookahead_distance_m=0.30,
        goal_gain=1.2,
        maximum_nominal_speed_mps=0.52,
        target_tolerance_m=0.10,
        path_simplification=True,
    )
    zone_cfg = LandingZoneValidationConfig(
        raw_occupied_fraction_threshold=0.02,
        inflated_occupied_fraction_threshold=0.05,
        minimum_clearance_m=0.03,
        minimum_valid_seed_cells=3,
        blocked_activation_frames=2,
        clear_deactivation_frames=6,
        latch_rejected_zones=True,
    )
    target_manager = TargetManager(mission, zone_cfg)
    filter_box = UnifiedContingencyFilter(
        poisson_config=PoissonCBFConfig(alpha=2.0, h_margin=0.010, maximum_field_age_s=2.0),
        filter_config=ContingencyFilterConfig(
            alpha_active=1.0,
            alpha_contingency=0.55,
            rho_gain=2.0,
            relaxation_weight_active=35.0,
            relaxation_weight_contingency=35.0,
            maximum_relaxation=50.0,
            solver="scipy",
            tolerance=1.0e-8,
        ),
        maximum_speed_mps=reach_cfg.maximum_speed_mps,
        required_reachable=mission.required_reachable,
    )

    occupancy_version = 1
    poisson = _poisson_record(inflated, geometry)
    bundle = build_reachability_bundle(
        inflated,
        mission=mission,
        geometry=geometry,
        config=reach_cfg,
        occupancy_version=occupancy_version,
    )
    position = mission.start_xy_m.copy()
    tau_active = -reach_cfg.active_horizon_s
    tau_contingency = -reach_cfg.contingency_horizon_s
    dt = 0.10
    trajectory = [position.copy()]
    history: list[dict[str, Any]] = []
    camera_invalid = False
    switch_count_before = 0
    reached = False
    collision = False
    hold = False
    hold_reason: str | None = None
    scenario_event_done = False
    minimum_raw_h = np.inf
    minimum_effective_h = np.inf
    minimum_poisson_residual = np.inf
    minimum_active_residual = np.inf
    minimum_contingency_residual = np.inf
    minimum_pivot = np.inf
    minimum_reachable = mission.p
    maximum_omega_active = 0.0
    maximum_omega_contingency = 0.0
    optimizer_successes = 0
    optimizer_attempts = 0
    optimizer_times: list[float] = []
    poisson_times = [float(poisson.wall_time_s)]
    hjr_times = [float(bundle.total_solve_time_s)]

    for step in range(360):
        time_s = step * dt
        if not scenario_event_done and time_s >= 3.0:
            if name == "active_zone_blocked":
                raw |= _disk_obstacle(geometry, mission.zone_by_identifier(1).center_xy_m, 0.34)
            elif name == "corridor_blocked":
                x_values = np.arange(geometry.nx) * geometry.dx
                y_values = np.arange(geometry.ny) * geometry.dy
                xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
                raw |= (xx >= 3.00) & (xx <= 3.35) & (yy >= 1.2) & (yy <= 2.7)
            elif name == "contingency_lost":
                # Block three target disks so fewer than r=2 candidates remain.
                for identifier in (1, 2, 3):
                    raw |= _disk_obstacle(geometry, mission.zone_by_identifier(identifier).center_xy_m, 0.35)
            elif name == "camera_moved":
                camera_invalid = True
            scenario_event_done = True
            if name != "camera_moved":
                inflated = inflate_occupancy_physical(raw, 0.12, geometry.spacing_yx)
                occupancy_version += 1
                poisson = _poisson_record(inflated, geometry)
                poisson_times.append(float(poisson.wall_time_s))
                rejected = [
                    identifier
                    for identifier, runtime in target_manager.states.items()
                    if runtime.state == ZoneState.REJECTED
                ]
                bundle = build_reachability_bundle(
                    inflated,
                    mission=mission,
                    geometry=geometry,
                    config=reach_cfg,
                    occupancy_version=occupancy_version,
                    unavailable_identifiers=rejected,
                )
                hjr_times.append(float(bundle.total_solve_time_s))

        assessments = assess_all_landing_zones(
            mission,
            raw_occupancy=raw,
            inflated_occupancy=inflated,
            geometry=geometry,
            config=zone_cfg,
        )
        target_manager.update_assessments(assessments)
        # Call a second time after the synthetic map update to emulate two
        # persistent filtered frames without tying the test to camera frame rate.
        if scenario_event_done and abs(time_s - 3.0) < 0.5 * dt and name in {"active_zone_blocked", "contingency_lost"}:
            target_manager.update_assessments(assessments)

        available = target_manager.available_identifiers()
        status = compute_reachability_status(
            bundle,
            point_xy=position,
            tau=tau_contingency,
            maximum_speed_mps=reach_cfg.maximum_speed_mps,
            required_reachable=mission.required_reachable,
            available_identifiers=available,
        )
        target_manager.update_reachability(values=status.values, distances=status.distances_m)
        minimum_pivot = min(minimum_pivot, status.pivot)
        minimum_reachable = min(minimum_reachable, status.reachable_count)

        if status.reachable_count < mission.required_reachable or status.pivot < 0.0:
            hold = True
            hold_reason = "CONTINGENCY REQUIREMENT LOST"
            break

        active_state = target_manager.states[target_manager.active_identifier]
        active_field = bundle.fields[target_manager.active_identifier]
        active_sample = sample_zone_field(
            active_field,
            geometry,
            position,
            tau=tau_active,
            maximum_speed_mps=reach_cfg.maximum_speed_mps,
        )
        active_failed = (
            active_state.state in {ZoneState.REJECTED, ZoneState.BLOCKED}
            or not active_sample.valid
            or active_sample.value < 0.0
        )
        if active_failed:
            if active_state.state != ZoneState.REJECTED:
                target_manager.reject(target_manager.active_identifier, "synthetic active target invalid")
            alternative = target_manager.choose_certified_alternative(exclude_identifier=target_manager.active_identifier)
            if alternative is None:
                hold = True
                hold_reason = "no certified alternative target"
                break
            target_manager.switch_to(
                alternative,
                time_s=time_s,
                reason="synthetic validation diversion",
                occupancy_version=occupancy_version,
            )
            tau_active = tau_contingency

        if camera_invalid:
            hold = True
            hold_reason = "camera moved; metric map invalid"
            break
        path = construct_active_path(
            bundle,
            position_xy=position,
            active_zone_identifier=target_manager.active_identifier,
            config=planner_cfg,
        )
        if not path.valid:
            hold = True
            hold_reason = path.reason
            break
        nominal, _lookahead, reached_path = pure_pursuit_velocity(position, path.points_xy, config=planner_cfg)
        active_zone = mission.zone_by_identifier(target_manager.active_identifier)
        if reached_path or float(np.linalg.norm(position - active_zone.center_xy_m)) <= active_zone.radius_m:
            reached = True
            target_manager.mark_reached(active_zone.identifier)
            break

        sampler = GridFieldSampler(poisson.result, geometry)
        sample = sampler.sample(position)
        optimizer_attempts += 1
        result = filter_box.filter(
            position_xy=position,
            nominal_velocity_xy=nominal,
            poisson_sample=sample,
            reachability_bundle=bundle,
            active_identifier=target_manager.active_identifier,
            available_identifiers=target_manager.available_identifiers(),
            tau_active=tau_active,
            tau_active_dot=1.0,
            tau_contingency=tau_contingency,
            tau_contingency_dot=0.0,
        )
        optimizer_times.append(result.solve_time_s)
        if not result.success:
            hold = True
            hold_reason = result.hold_reason
            break
        optimizer_successes += 1
        maximum_omega_active = max(maximum_omega_active, result.omega_active)
        maximum_omega_contingency = max(maximum_omega_contingency, result.omega_contingency)
        minimum_raw_h = min(minimum_raw_h, float(result.poisson_h_raw))
        minimum_effective_h = min(minimum_effective_h, float(result.poisson_h_effective))
        minimum_poisson_residual = min(minimum_poisson_residual, result.residuals.get("poisson_velocity_cbf", np.inf))
        active_names = [name for name in result.residuals if name.startswith("active_hj")]
        contingency_names = [name for name in result.residuals if name.startswith("contingency_")]
        if active_names:
            minimum_active_residual = min(minimum_active_residual, min(result.residuals[name] for name in active_names))
        if contingency_names:
            minimum_contingency_residual = min(
                minimum_contingency_residual,
                min(result.residuals[name] for name in contingency_names),
            )

        step_result = backtracked_safe_step(
            position_xy=position,
            velocity_xy=result.safe_velocity_xy,
            nominal_dt_s=dt,
            geometry=geometry,
            inflated_occupancy=inflated,
            poisson_sampler=sampler,
            h_margin=filter_box.poisson_config.h_margin,
            tolerance=1.0e-6,
            maximum_backtracks=14,
            maximum_dt_s=dt,
        )
        if not step_result.accepted:
            hold = True
            hold_reason = "discrete-time safety check failed: " + str(step_result.reason)
            break
        position = step_result.position_xy
        tau_active = min(0.0, tau_active + step_result.accepted_dt_s)
        trajectory.append(position.copy())
        history.append(
            {
                "time_s": time_s,
                "occupancy_version": occupancy_version,
                "position_x_m": float(position[0]),
                "position_y_m": float(position[1]),
                "active_target": target_manager.active_identifier,
                "reachable_count": result.reachable_count,
                "pivot": result.pivot,
                "poisson_h": result.poisson_h_raw,
                "poisson_effective_h": result.poisson_h_effective,
                "poisson_residual": result.residuals.get("poisson_velocity_cbf"),
                "omega_active": result.omega_active,
                "omega_contingency": result.omega_contingency,
                "optimizer_time_s": result.solve_time_s,
                "accepted_dt_s": step_result.accepted_dt_s,
                "backtracks": step_result.backtracks,
                "nominal_u_x": nominal[0],
                "nominal_u_y": nominal[1],
                "safe_u_x": result.safe_velocity_xy[0],
                "safe_u_y": result.safe_velocity_xy[1],
            }
        )

    final_active = target_manager.active_identifier
    summary = {
        "scenario": name,
        "target_reached": bool(reached),
        "final_target": int(final_active),
        "switches": len(target_manager.switch_events),
        "collision": bool(collision),
        "hold": bool(hold),
        "hold_reason": hold_reason,
        "final_position_xy_m": position.tolist(),
        "minimum_raw_poisson_h": None if not np.isfinite(minimum_raw_h) else float(minimum_raw_h),
        "minimum_effective_h": None if not np.isfinite(minimum_effective_h) else float(minimum_effective_h),
        "minimum_poisson_cbf_residual": None if not np.isfinite(minimum_poisson_residual) else float(minimum_poisson_residual),
        "minimum_active_hj_residual": None if not np.isfinite(minimum_active_residual) else float(minimum_active_residual),
        "minimum_contingency_residual": None if not np.isfinite(minimum_contingency_residual) else float(minimum_contingency_residual),
        "minimum_pivot": None if not np.isfinite(minimum_pivot) else float(minimum_pivot),
        "minimum_reachable_count": int(minimum_reachable),
        "maximum_omega_active": float(maximum_omega_active),
        "maximum_omega_contingency": float(maximum_omega_contingency),
        "poisson_solve_time_s": {
            "mean": float(np.mean(poisson_times)),
            "maximum": float(np.max(poisson_times)),
            "count": len(poisson_times),
        },
        "hjr_solve_time_s": {
            "mean": float(np.mean(hjr_times)),
            "maximum": float(np.max(hjr_times)),
            "count": len(hjr_times),
        },
        "optimizer_success_fraction": float(optimizer_successes / max(1, optimizer_attempts)),
        "optimizer_solve_time_s": {
            "mean": float(np.mean(optimizer_times)) if optimizer_times else None,
            "maximum": float(np.max(optimizer_times)) if optimizer_times else None,
            "count": len(optimizer_times),
        },
        "switch_events": [event.to_dict() for event in target_manager.switch_events],
    }
    save_json(output / "summary.json", summary)
    write_csv(output / "history.csv", history)
    write_csv(output / "target_switch_events.csv", [event.to_dict() for event in target_manager.switch_events])

    # Publication-style compact validation plot.
    trajectory_array = np.asarray(trajectory)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axes[0].imshow(inflated, origin="upper", extent=[0, 6.0, 4.5, 0], cmap="gray_r", alpha=0.65)
    axes[0].plot(trajectory_array[:, 0], trajectory_array[:, 1], linewidth=2.4, label="virtual trajectory")
    axes[0].plot(mission.start_xy_m[0], mission.start_xy_m[1], marker="o", label="START")
    for zone in mission.landing_zones:
        circle = plt.Circle(zone.center_xy_m, zone.radius_m, fill=False, linewidth=2.0)
        axes[0].add_patch(circle)
        axes[0].text(zone.center_xy_m[0], zone.center_xy_m[1], zone.name)
    axes[0].set_title(f"{name}: map, targets, and trajectory")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].legend(loc="best")
    if history:
        times = [row["time_s"] for row in history]
        axes[1].plot(times, [row["pivot"] for row in history], label="r-out-of-p pivot")
        axes[1].plot(times, [row["reachable_count"] for row in history], label="reachable count")
        axes[1].axhline(mission.required_reachable, linestyle="--", label="required r")
        axes[1].axhline(0.0, linestyle=":", color="black")
    axes[1].set_title("Contingency certificate history")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(loc="best")
    fig.suptitle(f"Synthetic contingency validation: {name}")
    fig.savefig(output / "validation_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return ScenarioResult(name, summary, history, trajectory_array, inflated)


def run_all_scenarios(output_directory: str | Path) -> dict[str, Any]:
    """Run all five required scenarios and save a combined report."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for name in ("clear_active", "active_zone_blocked", "corridor_blocked", "contingency_lost", "camera_moved"):
        result = run_scenario(name, output / name)
        summaries[name] = result.summary
    save_json(output / "all_scenarios_summary.json", summaries)
    return summaries


__all__ = ["ScenarioResult", "run_all_scenarios", "run_scenario"]
