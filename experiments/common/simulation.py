"""Deterministic double-integrator simulation shared by image and world studies.

The simulator is intentionally independent of geometry construction and plotting.
It advances the reduced-order state, evaluates scheduled landing-site failures,
logs every certificate and optimization diagnostic, and classifies the terminal
outcome as ``landed``, ``hold``, or ``timeout``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import json

import numpy as np
import pandas as pd

from safety_box_core import BoxStatus

from .controller import LandingController
from .poisson_field import PoissonField


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Complete machine-readable result of one rollout."""

    metrics: pd.DataFrame
    events: pd.DataFrame
    summary: dict[str, Any]
    final_state: np.ndarray
    controller: LandingController


def _normalized_failure_schedule(
    schedule: Sequence[Mapping[str, Any]] | None,
    *,
    target_failure_time_s: float | None,
    failed_target_id: str | None,
) -> list[dict[str, Any]]:
    """Normalize new and legacy target-failure configurations."""

    if schedule is None:
        if target_failure_time_s is None or failed_target_id is None:
            return []
        schedule = [
            {
                "target": str(failed_target_id),
                "time_s": float(target_failure_time_s),
                "reason": "scheduled experiment event",
            }
        ]

    normalized: list[dict[str, Any]] = []
    for index, event in enumerate(schedule):
        if not isinstance(event, Mapping):
            raise TypeError(f"Failure event {index} must be a mapping.")
        target = str(event.get("target", "active"))
        record = {
            "target": target,
            "reason": str(event.get("reason", "scheduled landing-zone failure")),
        }
        for key in (
            "time_s",
            "earliest_time_s",
            "minimum_delay_s",
            "trigger_distance_m",
        ):
            if key in event and event[key] is not None:
                record[key] = float(event[key])
        if not any(key in record for key in ("time_s", "earliest_time_s", "trigger_distance_m")):
            raise ValueError(
                f"Failure event {index} must define time_s, earliest_time_s, "
                "or trigger_distance_m."
            )
        normalized.append(record)
    return normalized


def _failure_event_ready(
    event: Mapping[str, Any],
    *,
    time_s: float,
    last_failure_time_s: float | None,
    position: np.ndarray,
    active_target: str,
    controller: LandingController,
) -> tuple[bool, str, float]:
    """Evaluate one event without mutating target availability."""

    requested = str(event.get("target", "active"))
    target_id = active_target if requested.lower() in {"active", "current"} else requested
    if target_id not in controller.targets:
        raise KeyError(f"Unknown landing-zone identifier in failure schedule: {target_id!r}")

    earliest = float(event.get("time_s", event.get("earliest_time_s", 0.0)))
    if last_failure_time_s is not None and "minimum_delay_s" in event:
        earliest = max(earliest, last_failure_time_s + float(event["minimum_delay_s"]))
    if time_s + 1.0e-12 < earliest:
        distance = float(
            np.linalg.norm(position - controller.targets[target_id].x_star[: controller.dimension])
        )
        return False, target_id, distance

    distance = float(
        np.linalg.norm(position - controller.targets[target_id].x_star[: controller.dimension])
    )
    threshold = event.get("trigger_distance_m")
    if threshold is not None and distance > float(threshold):
        return False, target_id, distance
    return True, target_id, distance


def run_simulation(
    *,
    controller: LandingController,
    field: PoissonField,
    start_state: np.ndarray,
    initial_target: str,
    output_directory: str | Path,
    dt_s: float,
    maximum_steps: int,
    landing_position_tolerance: float,
    landing_speed_tolerance: float,
    collision_check: Callable[[np.ndarray], bool],
    target_failure_time_s: float | None = None,
    failed_target_id: str | None = None,
    failure_schedule: Sequence[Mapping[str, Any]] | None = None,
    availability: Mapping[str, bool] | None = None,
    variant: str = "full",
    nominal_control_provider: Callable[[np.ndarray, str], np.ndarray] | None = None,
    clearance_query: Callable[[np.ndarray], float] | None = None,
    stop_on_landing: bool = True,
) -> SimulationResult:
    """Run a reproducible landing rollout and save machine-readable logs.

    Failure events may be time-triggered, proximity-triggered, or both.  A
    target value of ``active`` invalidates whichever landing zone is currently
    pursued.  At most one scheduled event is processed per control step so the
    controller can retarget before the next event is evaluated.
    """

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    state = np.asarray(start_state, dtype=float).copy()
    dimension = controller.dimension
    if state.shape != (2 * dimension,):
        raise ValueError("start_state dimension does not match the controller.")
    if initial_target not in controller.targets:
        raise KeyError(f"Unknown initial target {initial_target!r}.")

    availability_map = {
        target_id: bool(target.available)
        for target_id, target in controller.targets.items()
    }
    if availability is not None:
        availability_map.update({key: bool(value) for key, value in availability.items()})

    schedule = _normalized_failure_schedule(
        failure_schedule,
        target_failure_time_s=target_failure_time_s,
        failed_target_id=failed_target_id,
    )
    next_failure_event = 0
    last_failure_time_s: float | None = None
    failed_targets: list[str] = []

    active_target = initial_target
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    switched = False
    switch_count = 0
    landed = False
    hold_reason: str | None = None
    collision_guard_backtracks = 0
    cumulative_path_length = 0.0
    previous_position = state[:dimension].copy()

    for step_index in range(int(maximum_steps)):
        time_s = step_index * float(dt_s)
        position_before_control = state[:dimension]

        if next_failure_event < len(schedule):
            event = schedule[next_failure_event]
            ready, target_to_fail, target_distance = _failure_event_ready(
                event,
                time_s=time_s,
                last_failure_time_s=last_failure_time_s,
                position=position_before_control,
                active_target=active_target,
                controller=controller,
            )
            if ready:
                if availability_map.get(target_to_fail, False):
                    availability_map[target_to_fail] = False
                    failed_targets.append(target_to_fail)
                    last_failure_time_s = time_s
                    events.append(
                        {
                            "time_s": time_s,
                            "event": "target_failed",
                            "target_id": target_to_fail,
                            "reason": str(event["reason"]),
                            "distance_to_target_m": target_distance,
                            "schedule_index": next_failure_event,
                        }
                    )
                else:
                    events.append(
                        {
                            "time_s": time_s,
                            "event": "target_failure_skipped",
                            "target_id": target_to_fail,
                            "reason": "target was already unavailable",
                            "distance_to_target_m": target_distance,
                            "schedule_index": next_failure_event,
                        }
                    )
                next_failure_event += 1

        sample = field.sample(state[:dimension])
        control_step = controller.step(
            state=state,
            time_s=time_s,
            version=step_index,
            active_target=active_target,
            availability=availability_map,
            safety_sample=sample,
            dt_s=float(dt_s),
            nominal_control_provider=nominal_control_provider,
        )
        if control_step.target_switched:
            previous_target = active_target
            active_target = control_step.active_target
            switched = True
            switch_count += 1
            events.append(
                {
                    "time_s": time_s,
                    "event": "active_target_switched",
                    "target_id": active_target,
                    "reason": "previous target unavailable; maximum certified ROA margin",
                    "previous_target_id": previous_target,
                }
            )
        else:
            active_target = control_step.active_target

        if control_step.status is not BoxStatus.READY:
            hold_reason = control_step.message or control_step.status.value
            events.append(
                {
                    "time_s": time_s,
                    "event": "hold",
                    "target_id": active_target,
                    "reason": hold_reason,
                }
            )
            break

        acceleration = np.asarray(control_step.safe_control, dtype=float)
        position = state[:dimension]
        velocity = state[dimension:]
        local_dt = float(dt_s)
        integration_fraction = 1.0
        next_position = position + velocity * local_dt + 0.5 * acceleration * local_dt**2
        next_velocity = velocity + acceleration * local_dt

        while collision_check(next_position) and integration_fraction > 1.0 / 64.0:
            integration_fraction *= 0.5
            local_dt = float(dt_s) * integration_fraction
            next_position = position + velocity * local_dt + 0.5 * acceleration * local_dt**2
            next_velocity = velocity + acceleration * local_dt
        if collision_check(next_position):
            hold_reason = "sampled-data collision guard could not find a free integration step"
            events.append(
                {
                    "time_s": time_s,
                    "event": "hold",
                    "target_id": active_target,
                    "reason": hold_reason,
                }
            )
            break
        if integration_fraction < 1.0:
            collision_guard_backtracks += 1

        active_evaluation = control_step.evaluations.get(active_target)
        contingency = control_step.contingency
        filter_result = control_step.filter_result
        target_position = controller.targets[active_target].x_star[:dimension]
        distance_to_active = float(np.linalg.norm(position - target_position))
        clearance = np.nan if clearance_query is None else float(clearance_query(position))
        row: dict[str, Any] = {
            "step": step_index,
            "time_s": time_s,
            "active_target": active_target,
            "status": control_step.status.value,
            "poisson_h": np.nan if sample is None else float(sample.h),
            "poisson_gradient_norm": (
                np.nan if sample is None else float(np.linalg.norm(sample.grad_h))
            ),
            "hocbf_residual": (
                np.nan if control_step.hocbf_residual is None else control_step.hocbf_residual
            ),
            "active_V": np.nan if active_evaluation is None else active_evaluation.V,
            "active_h_roa": np.nan if active_evaluation is None else active_evaluation.h_roa,
            "active_clf_residual": (
                np.nan if control_step.clf_residual is None else control_step.clf_residual
            ),
            "contingency_pivot": np.nan if contingency is None else contingency.pivot,
            "certified_count": 0 if contingency is None else contingency.certified_count,
            "available_count": (
                sum(availability_map.values())
                if contingency is None
                else contingency.available_count
            ),
            "omega": control_step.omega,
            "clf_slack": control_step.clf_slack,
            "intervention_norm": control_step.intervention_norm,
            "speed": float(np.linalg.norm(velocity)),
            "altitude_m": float(position[2]) if dimension == 3 else np.nan,
            "distance_to_active_target_m": distance_to_active,
            "obstacle_clearance_m": clearance,
            "path_length_m": cumulative_path_length,
            "solver_status": "not_run" if filter_result is None else filter_result.solver_status,
            "solver_time_s": 0.0 if filter_result is None else filter_result.solve_time_s,
            "minimum_constraint_residual": (
                np.nan
                if filter_result is None or filter_result.residuals.size == 0
                else float(np.min(filter_result.residuals))
            ),
            "integration_fraction": integration_fraction,
        }
        for axis in range(dimension):
            label = "xyz"[axis]
            row[label] = float(position[axis])
            row[f"v{label}"] = float(velocity[axis])
            row[f"a_nom_{label}"] = float(control_step.nominal_control[axis])
            row[f"a_safe_{label}"] = float(acceleration[axis])
        for target_id, evaluation in control_step.evaluations.items():
            row[f"V_{target_id}"] = float(evaluation.V)
            row[f"h_roa_{target_id}"] = float(evaluation.h_roa)
            row[f"available_{target_id}"] = int(availability_map.get(target_id, False))
            row[f"distance_{target_id}_m"] = float(
                np.linalg.norm(position - controller.targets[target_id].x_star[:dimension])
            )
        records.append(row)

        state = np.concatenate([next_position, next_velocity])
        cumulative_path_length += float(np.linalg.norm(next_position - previous_position))
        previous_position = next_position.copy()
        target = controller.targets[active_target]
        position_error = np.linalg.norm(state[:dimension] - target.x_star[:dimension])
        speed = np.linalg.norm(state[dimension:])
        if (
            position_error <= float(landing_position_tolerance)
            and speed <= float(landing_speed_tolerance)
        ):
            landed = True
            events.append(
                {
                    "time_s": time_s + local_dt,
                    "event": "landed",
                    "target_id": active_target,
                    "reason": "position and speed tolerances satisfied",
                    "position_error_m": float(position_error),
                    "speed_mps": float(speed),
                }
            )
            if stop_on_landing:
                break

    metrics = pd.DataFrame(records)
    event_frame = pd.DataFrame(events)
    if event_frame.empty:
        event_frame = pd.DataFrame(columns=["time_s", "event", "target_id", "reason"])

    def scalar_metric(column: str, operation: str) -> float | None:
        if metrics.empty or column not in metrics:
            return None
        values = metrics[column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            return None
        return float(getattr(values, operation)())

    terminal_status = "landed" if landed else "hold" if hold_reason is not None else "timeout"
    final_target_position = controller.targets[active_target].x_star[:dimension]
    final_target_error = float(np.linalg.norm(state[:dimension] - final_target_position))
    final_speed = float(np.linalg.norm(state[dimension:]))
    summary = {
        "variant": variant,
        "terminal_status": terminal_status,
        "landed": landed,
        "hold_reason": hold_reason,
        "initial_target": initial_target,
        "final_target": active_target,
        "target_failed": bool(failed_targets),
        "failed_targets": failed_targets,
        "failure_count": len(failed_targets),
        "target_switched": switched,
        "switch_count": switch_count,
        "scheduled_failure_count": len(schedule),
        "processed_failure_events": next_failure_event,
        "remaining_available_targets": [
            identifier for identifier, is_available in availability_map.items() if is_available
        ],
        "steps": len(metrics),
        "duration_s": len(metrics) * float(dt_s),
        "path_length_m": float(cumulative_path_length),
        "final_target_error_m": final_target_error,
        "final_speed_mps": final_speed,
        "minimum_obstacle_clearance_m": scalar_metric("obstacle_clearance_m", "min"),
        "minimum_poisson_h": scalar_metric("poisson_h", "min"),
        "minimum_hocbf_residual": scalar_metric("hocbf_residual", "min"),
        "minimum_active_clf_residual": scalar_metric("active_clf_residual", "min"),
        "minimum_contingency_pivot": scalar_metric("contingency_pivot", "min"),
        "mean_intervention_norm": scalar_metric("intervention_norm", "mean"),
        "maximum_intervention_norm": scalar_metric("intervention_norm", "max"),
        "maximum_speed_mps": scalar_metric("speed", "max"),
        "mean_solver_time_ms": (
            None if metrics.empty else float(metrics["solver_time_s"].mean() * 1.0e3)
        ),
        "p95_solver_time_ms": (
            None if metrics.empty else float(metrics["solver_time_s"].quantile(0.95) * 1.0e3)
        ),
        "collision_guard_backtracks": collision_guard_backtracks,
        "final_state": state.tolist(),
        "forcing_method": field.forcing_method,
        "poisson_solver": field.solver,
        "poisson_control_scale": field.control_scale,
    }

    metrics.to_csv(output / f"metrics_{variant}.csv", index=False)
    event_frame.to_csv(output / f"events_{variant}.csv", index=False)
    (output / f"summary_{variant}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return SimulationResult(
        metrics=metrics,
        events=event_frame,
        summary=summary,
        final_state=state,
        controller=controller,
    )
