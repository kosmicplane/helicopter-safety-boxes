"""Deterministic double-integrator simulation shared by image and world studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import json

import numpy as np
import pandas as pd

from safety_box_core import BoxStatus

from .controller import LandingController
from .poisson_field import PoissonField


@dataclass(frozen=True, slots=True)
class SimulationResult:
    metrics: pd.DataFrame
    events: pd.DataFrame
    summary: dict[str, Any]
    final_state: np.ndarray
    controller: LandingController


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
    availability: Mapping[str, bool] | None = None,
    variant: str = "full",
    nominal_control_provider: Callable[[np.ndarray, str], np.ndarray] | None = None,
) -> SimulationResult:
    """Run a reproducible landing rollout and save machine-readable logs."""

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

    active_target = initial_target
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    failed = False
    switched = False
    landed = False
    hold_reason: str | None = None
    collision_guard_backtracks = 0

    for step_index in range(int(maximum_steps)):
        time_s = step_index * float(dt_s)
        if (
            target_failure_time_s is not None
            and failed_target_id is not None
            and not failed
            and time_s >= float(target_failure_time_s)
        ):
            availability_map[failed_target_id] = False
            failed = True
            events.append(
                {
                    "time_s": time_s,
                    "event": "target_failed",
                    "target_id": failed_target_id,
                    "reason": "scheduled experiment event",
                }
            )

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
            active_target = control_step.active_target
            switched = True
            events.append(
                {
                    "time_s": time_s,
                    "event": "active_target_switched",
                    "target_id": active_target,
                    "reason": "previous target unavailable; maximum certified ROA margin",
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
            next_position = (
                position
                + velocity * local_dt
                + 0.5 * acceleration * local_dt**2
            )
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
                np.nan
                if control_step.hocbf_residual is None
                else control_step.hocbf_residual
            ),
            "active_V": np.nan if active_evaluation is None else active_evaluation.V,
            "active_h_roa": (
                np.nan if active_evaluation is None else active_evaluation.h_roa
            ),
            "active_clf_residual": (
                np.nan
                if control_step.clf_residual is None
                else control_step.clf_residual
            ),
            "contingency_pivot": (
                np.nan if contingency is None else contingency.pivot
            ),
            "certified_count": (
                0 if contingency is None else contingency.certified_count
            ),
            "available_count": (
                sum(availability_map.values())
                if contingency is None
                else contingency.available_count
            ),
            "omega": control_step.omega,
            "clf_slack": control_step.clf_slack,
            "intervention_norm": control_step.intervention_norm,
            "speed": float(np.linalg.norm(velocity)),
            "solver_status": (
                "not_run" if filter_result is None else filter_result.solver_status
            ),
            "solver_time_s": (
                0.0 if filter_result is None else filter_result.solve_time_s
            ),
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
        records.append(row)

        state = np.concatenate([next_position, next_velocity])
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
                }
            )
            break

    metrics = pd.DataFrame(records)
    event_frame = pd.DataFrame(
        events,
        columns=["time_s", "event", "target_id", "reason"],
    )

    def scalar_metric(column: str, operation: str) -> float | None:
        if metrics.empty or column not in metrics:
            return None
        values = metrics[column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            return None
        return float(getattr(values, operation)())

    summary = {
        "variant": variant,
        "landed": landed,
        "hold_reason": hold_reason,
        "initial_target": initial_target,
        "final_target": active_target,
        "target_failed": failed,
        "target_switched": switched,
        "steps": len(metrics),
        "duration_s": len(metrics) * float(dt_s),
        "minimum_poisson_h": scalar_metric("poisson_h", "min"),
        "minimum_hocbf_residual": scalar_metric("hocbf_residual", "min"),
        "minimum_active_clf_residual": scalar_metric(
            "active_clf_residual",
            "min",
        ),
        "minimum_contingency_pivot": scalar_metric("contingency_pivot", "min"),
        "mean_intervention_norm": scalar_metric("intervention_norm", "mean"),
        "maximum_intervention_norm": scalar_metric("intervention_norm", "max"),
        "mean_solver_time_ms": (
            None
            if metrics.empty
            else float(metrics["solver_time_s"].mean() * 1.0e3)
        ),
        "p95_solver_time_ms": (
            None
            if metrics.empty
            else float(metrics["solver_time_s"].quantile(0.95) * 1.0e3)
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
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return SimulationResult(
        metrics=metrics,
        events=event_frame,
        summary=summary,
        final_state=state,
        controller=controller,
    )
