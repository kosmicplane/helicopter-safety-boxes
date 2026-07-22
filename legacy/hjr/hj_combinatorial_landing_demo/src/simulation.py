"""Mission simulation connecting both safety boxes to the HJ contingency QP."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np

from .combinatorial_filter import solve_unified_qp
from .hj_reachability import compute_reachability_fields
from .poisson_pipeline import PoissonFieldSampler, compute_poisson
from .scenario import World


@dataclass
class SimulationArtifacts:
    world: World
    poisson_before: object
    poisson_after: object
    reach_before: list
    reach_after: list
    log: dict[str, np.ndarray]
    summary: dict


@dataclass
class _LogBuffer:
    rows: dict[str, list] = field(default_factory=dict)

    def append(self, **items) -> None:
        for key, value in items.items():
            self.rows.setdefault(key, []).append(value)

    def arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for key, values in self.rows.items():
            try:
                out[key] = np.asarray(values)
            except ValueError:
                out[key] = np.asarray(values, dtype=object)
        return out


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector if norm <= maximum or norm < 1e-12 else vector * (maximum / norm)


def _nominal_velocity(position: np.ndarray, waypoint: np.ndarray, gain: float, speed: float) -> np.ndarray:
    return _clip_norm(gain * (waypoint - position), speed)


def _zone_score(position: np.ndarray, zone, reach_sample: dict) -> float:
    """Higher is better; unreachable or rejected zones are excluded elsewhere."""
    distance = float(reach_sample["distance"])
    return 2.5 * zone.science_score + 0.08 * float(reach_sample["V"]) - 0.08 * distance


def run_simulation(world: World, config: dict, boxes: dict, output_dir: str | Path) -> SimulationArtifacts:
    """Execute the complete 4-zone, r=2 contingency landing demonstration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    control_cfg = config["control"]
    mission_cfg = config["mission"]
    vmax = float(control_cfg["max_speed_mps"])
    r = int(mission_cfg["required_reachable"])

    print("[simulation] Computing initial Poisson safety field with Poisson Safety Box...")
    poisson_before = compute_poisson(boxes, world.occupancy_initial, world.spacing, config)
    sampler_before = PoissonFieldSampler(world.axes, poisson_before)
    print("[simulation] Computing four initial HJ/geodesic reach-avoid fields...")
    reach_before = compute_reachability_fields(world, world.occupancy_initial, vmax)

    # Precompute the post-discovery products.  The mission switches to these data
    # at the specified discovery time, mimicking an onboard map update.
    print("[simulation] Computing updated Poisson field after hidden obstacle discovery...")
    poisson_after = compute_poisson(boxes, world.occupancy_updated, world.spacing, config)
    sampler_after = PoissonFieldSampler(world.axes, poisson_after)
    print("[simulation] Recomputing reachability after rejecting LZ-1...")
    reach_after = compute_reachability_fields(world, world.occupancy_updated, vmax, rejected_indices={0})

    CBFBox = boxes["CBFBox"]
    CBFBoxConfig = boxes["CBFBoxConfig"]
    SafetySample = boxes["SafetySample"]
    SystemState = boxes["SystemState"]
    cbf = CBFBox(
        CBFBoxConfig(
            mode="velocity",
            solver="closed_form",
            alpha=float(control_cfg["poisson_cbf_alpha"]),
            control_lower_bound=[-vmax, -vmax, -vmax],
            control_upper_bound=[vmax, vmax, vmax],
        )
    )

    dt = float(mission_cfg["dt_s"])
    max_steps = int(np.ceil(float(mission_cfg["max_time_s"]) / dt))
    discovery_time = float(mission_cfg["discovery_time_s"])
    contingency_horizon = float(mission_cfg["contingency_horizon_s"])
    active_deadline = float(mission_cfg["active_horizon_s"])
    active_index = int(mission_cfg["active_target_initial"])
    rejected: set[int] = set()
    position = world.start.copy()
    phase = "CRUISE_TO_B"
    discovery_done = False
    switched = False
    collision = False
    landed = False
    log = _LogBuffer()

    for step in range(max_steps + 1):
        t = step * dt
        sampler = sampler_after if discovery_done else sampler_before
        reach_fields = reach_after if discovery_done else reach_before
        occupancy = world.occupancy_updated if discovery_done else world.occupancy_initial

        # The active horizon shrinks toward zero.  The contingency horizon is
        # fixed, matching the paper's airplane example with dot(tau_2)=0.
        tau1 = min(-1e-3, t - active_deadline)
        tau2 = -contingency_horizon
        tau1_dot = 1.0
        tau2_dot = 0.0

        # Trigger the perception update before constructing this step's command.
        if (not discovery_done) and t >= discovery_time:
            discovery_done = True
            rejected.add(0)
            sampler = sampler_after
            reach_fields = reach_after
            occupancy = world.occupancy_updated
            # Select the best certified contingency and reset tau_1 <- tau_2 as
            # described by the paper at a target-switching instant.
            candidate_samples = [field.sample(position, tau2) for field in reach_fields]
            candidates = [
                i for i, sample in enumerate(candidate_samples)
                if i not in rejected and sample["V"] >= 0.0
            ]
            if not candidates:
                candidates = [i for i in range(len(reach_fields)) if i not in rejected]
            active_index = max(
                candidates,
                key=lambda i: _zone_score(position, world.landing_zones[i], candidate_samples[i]),
            )
            active_deadline = t + contingency_horizon
            tau1 = -contingency_horizon
            switched = True
            phase = "DIVERT_AFTER_DISCOVERY"

        # Mission planner: first fly through a science-observation waypoint, then
        # approach a point above the active LZ, and finally descend.
        active_zone = world.landing_zones[active_index]
        distance_to_science = float(np.linalg.norm(position - world.science_waypoint))
        horizontal_distance = float(np.linalg.norm(position[:2] - active_zone.center[:2]))
        if phase == "CRUISE_TO_B" and distance_to_science <= float(mission_cfg["survey_tolerance_m"]):
            phase = "PROVISIONAL_APPROACH"
        if phase == "CRUISE_TO_B":
            waypoint = world.science_waypoint
        elif horizontal_distance > 1.1 or position[2] > 3.2:
            waypoint = active_zone.center.copy()
            waypoint[2] = 3.0
        else:
            waypoint = active_zone.center
            phase = "FINAL_DESCENT"

        u_nom = _nominal_velocity(
            position,
            waypoint,
            gain=float(control_cfg["nominal_gain"]),
            speed=float(control_cfg["nominal_speed_mps"]),
        )

        # Sample Poisson safety and ask the actual CBF Safety Box to build and
        # solve its velocity CBF.  We reuse the returned affine inequality in the
        # joint HJ+Poisson QP rather than writing a duplicate CBF formula.
        ps = sampler.sample(position)
        # The Poisson PDE is solved only in free space and equals zero on the
        # Dirichlet frontier.  A positive h-margin creates a practical buffer
        # against finite grid spacing and discrete-time integration.
        poisson_margin = float(control_cfg.get("poisson_h_margin", 0.0))
        ps_for_cbf = dict(ps)
        ps_for_cbf["h"] = float(ps["h"] - poisson_margin)
        safety_sample = SafetySample(**ps_for_cbf)
        cbf_result = cbf.filter_control(
            SystemState(position=position, time=t),
            safety_sample,
            u_nom,
        )

        active_sample = reach_fields[active_index].sample(position, tau1)
        contingency_samples = [field.sample(position, tau2) for field in reach_fields]
        # Add a discrete-time one-step condition on top of the continuous CBF
        # inequality returned by the box.  This compensates for the finite grid
        # and Euler integration used in this offline example:
        #     h(p + dt*u) ≈ h(p) + dt*grad(h)^T u >= h_margin.
        # It is a strengthening, not a replacement, of the CBF-box constraint.
        discrete_rhs = np.array([(poisson_margin - float(ps["h"])) / dt], dtype=float)
        joint_poisson_b = np.maximum(np.asarray(cbf_result.constraint_vector, dtype=float), discrete_rhs)
        qp = solve_unified_qp(
            u_nom=u_nom,
            poisson_A=cbf_result.constraint_matrix,
            poisson_b=joint_poisson_b,
            active_sample=active_sample,
            contingency_samples=contingency_samples,
            required_reachable=r,
            tau1_dot=tau1_dot,
            tau2_dot=tau2_dot,
            max_speed=vmax,
            alpha_active=float(control_cfg["active_hj_alpha"]),
            alpha_contingency=float(control_cfg["contingency_hj_alpha"]),
            rho_gain=float(control_cfg["rho_gain"]),
            omega_weight=float(control_cfg["omega_weight"]),
            max_omega=float(control_cfg["max_omega"]),
        )
        u_safe = qp.u_safe

        # Record before integrating so all arrays share the same time/state pair.
        zone_values = np.array([sample["V"] for sample in contingency_samples], dtype=float)
        log.append(
            time=t,
            position=position.copy(),
            u_nom=u_nom.copy(),
            u_cbf_only=cbf_result.u_safe.copy(),
            u_safe=u_safe.copy(),
            active_index=active_index,
            phase=phase,
            discovered=int(discovery_done),
            tau1=tau1,
            tau2=tau2,
            poisson_h=ps["h"],
            poisson_h_margin_value=ps_for_cbf["h"],
            poisson_grad_norm=float(np.linalg.norm(ps["grad_h"])),
            poisson_residual=qp.poisson_residual,
            active_value=active_sample["V"],
            active_hj_residual=qp.active_hj_residual,
            zone_values=zone_values,
            pivot_value=qp.pivot_value,
            reachable_count=qp.reachable_count,
            contingency_residuals=qp.contingency_residuals.copy(),
            omega_active=qp.omega_active,
            omega_contingency=qp.omega_contingency,
            qp_success=int(qp.success),
            qp_solve_ms=1000.0 * qp.solve_time_s,
            correction_norm=float(np.linalg.norm(u_safe - u_nom)),
        )

        # Stop after logging a terminal state.
        if float(np.linalg.norm(position - active_zone.center)) <= float(mission_cfg["landing_tolerance_m"]):
            landed = True
            break

        next_position = position + dt * u_safe
        next_position = np.minimum(np.maximum(next_position, np.zeros(3)), world.size)
        idx = world.world_to_index(next_position)
        if occupancy[idx]:
            collision = True
            break
        position = next_position

    arrays = log.arrays()
    final_values = arrays["zone_values"][-1] if len(arrays["zone_values"]) else np.full(4, np.nan)
    summary = {
        "landed": bool(landed),
        "collision": bool(collision),
        "switched_target": bool(switched),
        "final_active_zone": world.landing_zones[int(arrays["active_index"][-1])].name,
        "final_position": arrays["position"][-1].tolist(),
        "simulation_time_s": float(arrays["time"][-1]),
        "required_reachable": r,
        "minimum_reachable_count": int(np.min(arrays["reachable_count"])),
        "minimum_pivot_value": float(np.min(arrays["pivot_value"])),
        "minimum_poisson_h": float(np.min(arrays["poisson_h"])),
        "minimum_poisson_residual": float(np.min(arrays["poisson_residual"])),
        "minimum_active_hj_residual": float(np.min(arrays["active_hj_residual"])),
        "qp_success_fraction": float(np.mean(arrays["qp_success"])),
        "mean_qp_solve_ms": float(np.mean(arrays["qp_solve_ms"])),
        "final_zone_values": np.asarray(final_values).tolist(),
        "model_note": (
            "HJ reachability is exact for the reduced single-integrator model p_dot=u with "
            "isotropic speed bound and static grid obstacles. It is not a full 6-DOF PX4 model."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return SimulationArtifacts(world, poisson_before, poisson_after, reach_before, reach_after, arrays, summary)
