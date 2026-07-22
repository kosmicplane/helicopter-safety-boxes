"""Single-integrator CBF demonstration connected to the sampled Poisson field."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from cbf_safety_box import CBFBox, CBFBoxConfig, SafetySample, SystemState
from poisson_safety_box.interpolation import interpolate_grid

from .coordinates import GridFieldSampler, GridGeometry
from .io_utils import save_json, write_csv


@dataclass(frozen=True)
class CBFSimulationConfig:
    """Parameters for nominal and CBF-filtered single-integrator simulations."""

    alpha: float = 3.0
    solver: str = "scipy"
    goal_gain: float = 1.0
    dt_s: float = 0.03
    maximum_steps: int = 1000
    maximum_speed_mps: float = 1.0
    goal_tolerance_m: float = 0.08
    residual_tolerance: float = 1.0e-7
    h_tolerance: float = 1.0e-6
    enforce_control_bounds: bool = True
    component_bound_mode: str = "euclidean_conservative"
    maximum_step_backtracks: int = 14
    minimum_integration_dt_s: float = 1.0e-6

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CBFSimulationConfig":
        """Build a validated configuration from a YAML mapping."""

        cfg = cls(**(data or {}))
        if cfg.alpha <= 0.0 or cfg.goal_gain <= 0.0:
            raise ValueError("CBF alpha and goal gain must be positive.")
        if cfg.dt_s <= 0.0 or cfg.maximum_steps <= 0:
            raise ValueError("Simulation dt and maximum_steps must be positive.")
        if cfg.maximum_speed_mps <= 0.0 or cfg.goal_tolerance_m <= 0.0:
            raise ValueError("Maximum speed and goal tolerance must be positive.")
        if cfg.solver not in {"closed_form", "scipy", "cvxpy"}:
            raise ValueError(f"Unsupported CBF-QP solver: {cfg.solver!r}")
        if cfg.component_bound_mode not in {"euclidean_conservative", "per_axis"}:
            raise ValueError("component_bound_mode must be 'euclidean_conservative' or 'per_axis'.")
        if cfg.maximum_step_backtracks < 0:
            raise ValueError("maximum_step_backtracks cannot be negative.")
        if cfg.minimum_integration_dt_s <= 0.0:
            raise ValueError("minimum_integration_dt_s must be positive.")
        return cfg


@dataclass(frozen=True)
class TrajectoryResult:
    """Recorded trajectory rows and terminal status for one controller."""

    controller: str
    rows: list[dict[str, Any]]
    status: str
    reached_goal: bool
    collided: bool

    @property
    def positions(self) -> np.ndarray:
        """Return all recorded positions as an `(N, 2)` array."""

        return np.asarray([[row["x_m"], row["y_m"]] for row in self.rows], dtype=float)


@dataclass(frozen=True)
class CBFComparisonResult:
    """Nominal and filtered trajectories for one Poisson field."""

    nominal: TrajectoryResult
    safe: TrajectoryResult
    start_xy: np.ndarray
    goal_xy: np.ndarray
    configuration: CBFSimulationConfig


def saturate_vector_norm(vector: np.ndarray, maximum_norm: float) -> np.ndarray:
    """Return a copy whose Euclidean norm does not exceed ``maximum_norm``."""

    value = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= maximum_norm or norm <= np.finfo(float).eps:
        return value.copy()
    return value * (float(maximum_norm) / norm)


def _distance_field_to_occupied(occupancy: np.ndarray, spacing_yx: tuple[float, float]) -> np.ndarray:
    """Compute metric clearance to the nearest occupied cell for diagnostics only."""

    occupied = np.asarray(occupancy, dtype=bool)
    if not np.any(occupied):
        return np.full(occupied.shape, np.inf, dtype=float)
    return ndimage.distance_transform_edt(~occupied, sampling=spacing_yx)


def _sample_clearance(
    clearance_grid: np.ndarray,
    point_xy: np.ndarray,
    geometry: GridGeometry,
) -> float:
    """Bilinearly sample the diagnostic obstacle-clearance field."""

    value, valid = interpolate_grid(
        clearance_grid,
        geometry.xy_to_yx(point_xy),
        geometry.spacing_yx,
        origin=geometry.origin_yx,
    )
    if valid and value is not None:
        return float(value)
    index = geometry.nearest_index_yx(point_xy, clip=True)
    return float(clearance_grid[index])


def _validate_endpoint(name: str, point_xy: np.ndarray, sampler: GridFieldSampler) -> None:
    """Reject endpoints outside the free and interpolable Poisson domain."""

    if not sampler.geometry.contains_xy(point_xy):
        raise ValueError(f"{name} point {point_xy.tolist()} lies outside the physical workspace.")
    sample = sampler.sample(point_xy)
    if not sample.valid:
        raise ValueError(f"{name} point {point_xy.tolist()} is invalid: {sample.reason}.")
    if sampler.occupancy_at_xy(point_xy):
        raise ValueError(f"{name} point {point_xy.tolist()} lies in inflated occupancy.")


def _nominal_trajectory(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    sampler: GridFieldSampler,
    config: CBFSimulationConfig,
    clearance_grid: np.ndarray,
) -> TrajectoryResult:
    """Simulate the saturated goal-seeking command without a safety filter."""

    position = start_xy.astype(float).copy()
    rows: list[dict[str, Any]] = []
    status = "maximum_steps"
    reached_goal = False
    collided = False
    for step in range(config.maximum_steps + 1):
        distance_to_goal = float(np.linalg.norm(goal_xy - position))
        field_sample = sampler.sample(position)
        rows.append(
            {
                "step": step,
                "time_s": step * config.dt_s,
                "x_m": float(position[0]),
                "y_m": float(position[1]),
                "distance_to_goal_m": distance_to_goal,
                "h": field_sample.h if field_sample.valid else np.nan,
                "clearance_to_inflated_occupancy_m": _sample_clearance(
                    clearance_grid,
                    position,
                    sampler.geometry,
                ),
                "u_nom_x": 0.0,
                "u_nom_y": 0.0,
                "u_nom_norm": 0.0,
                "collision": bool(sampler.occupancy_at_xy(position)),
            }
        )
        if distance_to_goal <= config.goal_tolerance_m:
            status = "goal_reached"
            reached_goal = True
            break
        if sampler.occupancy_at_xy(position):
            status = "collision_with_inflated_occupancy"
            collided = True
            break
        if not field_sample.valid:
            status = f"invalid_field:{field_sample.reason}"
            break

        command = saturate_vector_norm(config.goal_gain * (goal_xy - position), config.maximum_speed_mps)
        rows[-1].update(
            {
                "u_nom_x": float(command[0]),
                "u_nom_y": float(command[1]),
                "u_nom_norm": float(np.linalg.norm(command)),
            }
        )
        next_position = position + config.dt_s * command
        if not sampler.geometry.contains_xy(next_position):
            status = "left_workspace"
            break
        position = next_position
    return TrajectoryResult("nominal", rows, status, reached_goal, collided)


def _cbf_box_configuration(config: CBFSimulationConfig, dimension: int) -> CBFBoxConfig:
    """Build a Safety Box configuration without unsafe post-solve clipping."""

    solver = config.solver
    lower = None
    upper = None
    if config.enforce_control_bounds:
        if solver == "closed_form":
            # The external closed-form solver clips after projection. That can
            # invalidate the CBF half-space, so strict bounds require a QP backend.
            solver = "scipy"
        component_limit = config.maximum_speed_mps
        if config.component_bound_mode == "euclidean_conservative":
            component_limit /= np.sqrt(float(dimension))
        lower = [-component_limit] * dimension
        upper = [component_limit] * dimension
    return CBFBoxConfig(
        mode="velocity",
        solver=solver,
        alpha=config.alpha,
        control_lower_bound=lower,
        control_upper_bound=upper,
        use_slack=False,
    )


def _safe_trajectory(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    sampler: GridFieldSampler,
    config: CBFSimulationConfig,
    clearance_grid: np.ndarray,
) -> TrajectoryResult:
    """Simulate a CBF-QP command using actual sampled Poisson values."""

    position = start_xy.astype(float).copy()
    box = CBFBox(_cbf_box_configuration(config, position.size))
    rows: list[dict[str, Any]] = []
    status = "maximum_steps"
    reached_goal = False
    collided = False

    for step in range(config.maximum_steps + 1):
        distance_to_goal = float(np.linalg.norm(goal_xy - position))
        sampled = sampler.sample(position)
        base_row: dict[str, Any] = {
            "step": step,
            "time_s": step * config.dt_s,
            "x_m": float(position[0]),
            "y_m": float(position[1]),
            "distance_to_goal_m": distance_to_goal,
            "clearance_to_inflated_occupancy_m": _sample_clearance(
                clearance_grid,
                position,
                sampler.geometry,
            ),
            "h": sampled.h if sampled.valid else np.nan,
            "gradient_x": sampled.gradient_xy[0] if sampled.valid and sampled.gradient_xy is not None else np.nan,
            "gradient_y": sampled.gradient_xy[1] if sampled.valid and sampled.gradient_xy is not None else np.nan,
            "u_nom_x": 0.0,
            "u_nom_y": 0.0,
            "u_safe_x": 0.0,
            "u_safe_y": 0.0,
            "u_nom_norm": 0.0,
            "u_safe_norm": 0.0,
            "intervention_norm": 0.0,
            "cbf_residual": np.nan,
            "explicit_cbf_residual": np.nan,
            "solver_status": "not_called",
            "solve_time_s": 0.0,
            "was_filtered": False,
            "integration_dt_s": 0.0,
            "integration_backtracks": 0,
            "collision": bool(sampler.occupancy_at_xy(position)),
        }
        rows.append(base_row)

        if distance_to_goal <= config.goal_tolerance_m:
            status = "goal_reached"
            reached_goal = True
            break
        if not sampled.valid or sampled.h is None or sampled.gradient_xy is None:
            status = f"invalid_field:{sampled.reason}"
            break
        if sampler.occupancy_at_xy(position):
            status = "collision_with_inflated_occupancy"
            collided = True
            break

        nominal = saturate_vector_norm(
            config.goal_gain * (goal_xy - position),
            config.maximum_speed_mps,
        )
        safety = SafetySample(
            h=sampled.h,
            grad_h=sampled.gradient_xy,
            hessian_h=sampled.hessian_xy,
            laplacian_h=sampled.laplacian,
            metadata={"sampling": "bilinear", "point_xy": position.tolist()},
        )
        state = SystemState(position=position, time=step * config.dt_s)
        filtered = box.filter_control(state, safety, nominal)
        safe_command = np.asarray(filtered.u_safe, dtype=float)
        explicit_residual = float(sampled.gradient_xy @ safe_command + config.alpha * sampled.h)
        solver_failed = not (
            filtered.solver_status.startswith("optimal")
            or filtered.solver_status in {"degenerate_feasible"}
        )
        rows[-1].update(
            {
                "u_nom_x": float(nominal[0]),
                "u_nom_y": float(nominal[1]),
                "u_safe_x": float(safe_command[0]),
                "u_safe_y": float(safe_command[1]),
                "u_nom_norm": float(np.linalg.norm(nominal)),
                "u_safe_norm": float(np.linalg.norm(safe_command)),
                "intervention_norm": float(np.linalg.norm(safe_command - nominal)),
                "cbf_residual": filtered.cbf_residual,
                "explicit_cbf_residual": explicit_residual,
                "solver_status": filtered.solver_status,
                "solve_time_s": filtered.solve_time,
                "was_filtered": filtered.was_filtered,
            }
        )
        if solver_failed:
            status = f"qp_failure:{filtered.solver_status}"
            break
        if explicit_residual < -config.residual_tolerance:
            status = "cbf_residual_violation"
            break

        # A continuous-time CBF inequality does not automatically make an
        # explicit Euler step safe at arbitrary dt.  Backtrack the integration
        # step until the *actual next state* remains in the interpolable free
        # domain with nonnegative h.  Rejected trial states are never applied.
        trial_dt = config.dt_s
        accepted_next: np.ndarray | None = None
        accepted_sample = None
        last_rejection = "unknown"
        used_backtracks = 0
        for backtrack in range(config.maximum_step_backtracks + 1):
            candidate = position + trial_dt * safe_command
            if not sampler.geometry.contains_xy(candidate):
                last_rejection = "outside_workspace"
            elif sampler.occupancy_at_xy(candidate):
                last_rejection = "occupied_cell"
            else:
                candidate_sample = sampler.sample(candidate)
                if not candidate_sample.valid:
                    last_rejection = f"invalid_field:{candidate_sample.reason}"
                elif candidate_sample.h is not None and candidate_sample.h < -config.h_tolerance:
                    last_rejection = "negative_h"
                else:
                    accepted_next = candidate
                    accepted_sample = candidate_sample
                    used_backtracks = backtrack
                    break
            trial_dt *= 0.5
            if trial_dt < config.minimum_integration_dt_s:
                break

        if accepted_next is None or accepted_sample is None:
            status = f"no_safe_discrete_step:{last_rejection}"
            break
        rows[-1]["integration_dt_s"] = float(trial_dt)
        rows[-1]["integration_backtracks"] = int(used_backtracks)
        position = accepted_next

    return TrajectoryResult("cbf", rows, status, reached_goal, collided)


def run_cbf_comparison(
    poisson_result: Any,
    *,
    grid_spacing_yx: tuple[float, float],
    start_xy: Iterable[float],
    goal_xy: Iterable[float],
    config: CBFSimulationConfig,
) -> CBFComparisonResult:
    """Run nominal and CBF trajectories over the same Poisson safety field."""

    geometry = GridGeometry(np.asarray(poisson_result.h).shape, grid_spacing_yx)
    sampler = GridFieldSampler(poisson_result, geometry, reject_occupied=True)
    start = np.asarray(list(start_xy), dtype=float).reshape(2)
    goal = np.asarray(list(goal_xy), dtype=float).reshape(2)
    _validate_endpoint("Start", start, sampler)
    _validate_endpoint("Goal", goal, sampler)
    clearance = _distance_field_to_occupied(poisson_result.occupancy_mask, grid_spacing_yx)
    nominal = _nominal_trajectory(start, goal, sampler, config, clearance)
    safe = _safe_trajectory(start, goal, sampler, config, clearance)
    return CBFComparisonResult(nominal, safe, start, goal, config)


def select_start_goal_interactive(
    rectified_bgr: np.ndarray,
    workspace_size_m: tuple[float, float],
    *,
    window_name: str = "Select start and goal",
) -> tuple[np.ndarray, np.ndarray]:
    """Select start and goal by two clicks on a rectified image."""

    clicks: list[tuple[int, int]] = []

    def mouse_callback(event: int, x: int, y: int, _flags: int, _parameter: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    try:
        while True:
            canvas = rectified_bgr.copy()
            for index, click in enumerate(clicks):
                color = (0, 255, 0) if index == 0 else (255, 0, 255)
                label = "start" if index == 0 else "goal"
                cv2.circle(canvas, click, 7, color, -1)
                cv2.putText(canvas, label, (click[0] + 8, click[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            instruction = "Click start" if len(clicks) == 0 else "Click goal" if len(clicks) == 1 else "Press Enter"
            cv2.putText(canvas, instruction + " | r: reset | Esc: cancel", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13) and len(clicks) == 2:
                width_m, height_m = workspace_size_m
                height_px, width_px = rectified_bgr.shape[:2]
                points = []
                for x_px, y_px in clicks:
                    points.append(
                        np.asarray(
                            [
                                width_m * x_px / max(1, width_px - 1),
                                height_m * y_px / max(1, height_px - 1),
                            ],
                            dtype=float,
                        )
                    )
                return points[0], points[1]
            if key == ord("r"):
                clicks.clear()
            elif key == 27:
                raise RuntimeError("Start/goal selection was cancelled by the user.")
    finally:
        cv2.destroyWindow(window_name)


def _plot_trajectory_background(
    ax: plt.Axes,
    background: np.ndarray,
    workspace_size_m: tuple[float, float],
    nominal_positions: np.ndarray,
    safe_positions: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    background_kind: str,
) -> None:
    """Draw both trajectories over one image, occupancy, or scalar field."""

    extent = [0.0, workspace_size_m[0], workspace_size_m[1], 0.0]
    if background_kind == "image":
        ax.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB), extent=extent, aspect="auto")
    else:
        ax.imshow(background, extent=extent, origin="upper", aspect="auto")
    if nominal_positions.size:
        ax.plot(nominal_positions[:, 0], nominal_positions[:, 1], "--", label="nominal")
    if safe_positions.size:
        ax.plot(safe_positions[:, 0], safe_positions[:, 1], "-", label="CBF filtered")
    ax.scatter([start_xy[0]], [start_xy[1]], marker="o", s=60, label="start")
    ax.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=100, label="goal")
    ax.set_xlim(0.0, workspace_size_m[0])
    ax.set_ylim(workspace_size_m[1], 0.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m], downward-positive")
    ax.legend(loc="best")


def save_cbf_comparison(
    comparison: CBFComparisonResult,
    *,
    poisson_result: Any,
    rectified_bgr: np.ndarray,
    workspace_size_m: tuple[float, float],
    output_directory: str | Path,
    dpi: int = 180,
) -> None:
    """Save CSV, NPZ, summary, and all trajectory/control diagnostic plots."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "nominal_trajectory.csv", comparison.nominal.rows)
    write_csv(output / "cbf_trajectory.csv", comparison.safe.rows)
    np.savez_compressed(
        output / "trajectories.npz",
        nominal_positions=comparison.nominal.positions,
        safe_positions=comparison.safe.positions,
        start_xy=comparison.start_xy,
        goal_xy=comparison.goal_xy,
    )
    safe_residuals = np.asarray(
        [row["explicit_cbf_residual"] for row in comparison.safe.rows if np.isfinite(row["explicit_cbf_residual"])],
        dtype=float,
    )
    safe_h = np.asarray([row["h"] for row in comparison.safe.rows if np.isfinite(row["h"])], dtype=float)
    save_json(
        output / "simulation_summary.json",
        {
            "nominal_status": comparison.nominal.status,
            "safe_status": comparison.safe.status,
            "nominal_reached_goal": comparison.nominal.reached_goal,
            "safe_reached_goal": comparison.safe.reached_goal,
            "nominal_collided": comparison.nominal.collided,
            "safe_collided": comparison.safe.collided,
            "minimum_explicit_cbf_residual": float(np.min(safe_residuals)) if safe_residuals.size else None,
            "minimum_h": float(np.min(safe_h)) if safe_h.size else None,
            "configuration": comparison.configuration.__dict__,
            "start_xy": comparison.start_xy,
            "goal_xy": comparison.goal_xy,
        },
    )

    nominal_positions = comparison.nominal.positions
    safe_positions = comparison.safe.positions
    backgrounds = [
        (rectified_bgr, "Trajectory over rectified image", "01_trajectory_on_image.png", "image"),
        (poisson_result.occupancy_mask.astype(float), "Trajectory over inflated occupancy", "02_trajectory_on_occupancy.png", "scalar"),
        (np.ma.masked_where(~poisson_result.free_mask, poisson_result.h), "Trajectory over Poisson h", "03_trajectory_on_h.png", "scalar"),
    ]
    for background, title, filename, kind in backgrounds:
        fig, ax = plt.subplots(figsize=(9, 7))
        _plot_trajectory_background(
            ax,
            background,
            workspace_size_m,
            nominal_positions,
            safe_positions,
            comparison.start_xy,
            comparison.goal_xy,
            background_kind=kind,
        )
        ax.set_title(title)
        fig.savefig(output / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    safe_rows = comparison.safe.rows
    time_values = np.asarray([row["time_s"] for row in safe_rows], dtype=float)

    def plot_series(filename: str, title: str, ylabel: str, series: list[tuple[str, np.ndarray]]) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, values in series:
            ax.plot(time_values[: values.size], values, label=label)
        ax.set_title(title)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if len(series) > 1:
            ax.legend()
        fig.savefig(output / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    plot_series(
        "04_h_over_time.png",
        "Poisson safety value along the CBF trajectory",
        "h",
        [("h", np.asarray([row["h"] for row in safe_rows], dtype=float))],
    )
    plot_series(
        "05_cbf_residual.png",
        "Velocity CBF residual",
        "grad(h)^T u_safe + alpha h",
        [("explicit residual", np.asarray([row["explicit_cbf_residual"] for row in safe_rows], dtype=float))],
    )
    plot_series(
        "06_control_components_x.png",
        "Control x components",
        "x velocity [m/s]",
        [
            ("u_nom_x", np.asarray([row["u_nom_x"] for row in safe_rows], dtype=float)),
            ("u_safe_x", np.asarray([row["u_safe_x"] for row in safe_rows], dtype=float)),
        ],
    )
    plot_series(
        "07_control_components_y.png",
        "Control y components",
        "y velocity [m/s]",
        [
            ("u_nom_y", np.asarray([row["u_nom_y"] for row in safe_rows], dtype=float)),
            ("u_safe_y", np.asarray([row["u_safe_y"] for row in safe_rows], dtype=float)),
        ],
    )
    plot_series(
        "08_control_norms.png",
        "Nominal and safe control norms",
        "speed [m/s]",
        [
            ("norm(u_nom)", np.asarray([row["u_nom_norm"] for row in safe_rows], dtype=float)),
            ("norm(u_safe)", np.asarray([row["u_safe_norm"] for row in safe_rows], dtype=float)),
        ],
    )
    plot_series(
        "09_intervention_norm.png",
        "CBF intervention magnitude",
        "norm(u_safe - u_nom) [m/s]",
        [("intervention", np.asarray([row["intervention_norm"] for row in safe_rows], dtype=float))],
    )
    plot_series(
        "10_obstacle_clearance_diagnostic.png",
        "Geometric clearance to inflated occupancy (diagnostic only)",
        "clearance [m]",
        [
            (
                "clearance",
                np.asarray([row["clearance_to_inflated_occupancy_m"] for row in safe_rows], dtype=float),
            )
        ],
    )
    plot_series(
        "11_qp_solve_time.png",
        "CBF-QP solve time",
        "solve time [s]",
        [("QP solve", np.asarray([row["solve_time_s"] for row in safe_rows], dtype=float))],
    )
