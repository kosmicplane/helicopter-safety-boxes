"""Reusable CLF/CBF/contingency controller shared by all experiment modes.

The controller owns no world or camera logic. It accepts a state, one local
Poisson safety sample, target availability, and a nominal command. This keeps
perception, field synthesis, certificates, and optimization independently
replaceable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from safety_box_core import (
    AffineConstraint,
    BoxStatus,
    ConstraintBundle,
    ConstraintHardness,
    DecisionLayout,
    EquilibriumTarget,
    FilterResult,
    StateSnapshot,
)
from cbf_safety_box import CBFBox, CBFBoxConfig, SafetySample
from clf_safety_box import CLFBox, CLFBoxConfig, DoubleIntegratorModel
from contingency_safety_box import (
    ContingencyBox,
    ContingencyBoxConfig,
    ContingencyEvaluation,
    maximum_margin,
)
from safety_filter_box import MultiCertificateFilter, SafetyFilterConfig


@dataclass(frozen=True, slots=True)
class ControllerStep:
    """All numerical objects produced by one control update."""

    safe_control: np.ndarray
    nominal_control: np.ndarray
    active_target: str
    target_switched: bool
    status: BoxStatus
    message: str
    omega: float
    clf_slack: float
    evaluations: Mapping[str, Any]
    contingency: ContingencyEvaluation | None
    filter_result: FilterResult | None
    hocbf_residual: float | None
    clf_residual: float | None
    intervention_norm: float


class LandingController:
    """Unified reduced-order landing controller with independently enabled boxes."""

    def __init__(
        self,
        *,
        dimension: int,
        targets: Sequence[EquilibriumTarget],
        box_config: Mapping[str, Any],
        filter_config: Mapping[str, Any],
        artifact_directory: str | Path,
        maximum_acceleration: float,
        maximum_speed_component: float,
        nominal_position_gain: float = 0.8,
        nominal_velocity_gain: float = 1.2,
    ) -> None:
        if dimension not in {2, 3}:
            raise ValueError("LandingController supports 2-D or 3-D double integrators.")
        self.dimension = int(dimension)
        self.model = DoubleIntegratorModel(self.dimension)
        self.maximum_acceleration = float(maximum_acceleration)
        self.maximum_speed_component = float(maximum_speed_component)
        self.nominal_position_gain = float(nominal_position_gain)
        self.nominal_velocity_gain = float(nominal_velocity_gain)
        if self.maximum_acceleration <= 0.0 or self.maximum_speed_component <= 0.0:
            raise ValueError("Control and speed limits must be positive.")

        target_tuple = tuple(targets)
        if not target_tuple:
            raise ValueError("At least one equilibrium target is required.")
        self.targets = {target.identifier: target for target in target_tuple}

        clf_cfg = dict(box_config["clf"])
        lower = tuple([-self.maximum_acceleration] * self.dimension)
        upper = tuple([self.maximum_acceleration] * self.dimension)
        clf_cfg["control_lower"] = lower
        clf_cfg["control_upper"] = upper
        self.clf = CLFBox(CLFBoxConfig.from_dict(clf_cfg), self.model)
        self.clf.prepare(target_tuple, artifact_directory)

        self.cbf = CBFBox(CBFBoxConfig.from_dict(box_config["cbf"]))
        self.contingency = ContingencyBox(
            ContingencyBoxConfig.from_dict(box_config["contingency"])
        )

        layout_sizes: dict[str, int] = {"control": self.dimension}
        if self.contingency.enabled:
            layout_sizes["omega_contingency"] = 1
        self.layout = DecisionLayout.from_sizes(**layout_sizes)
        filter_cfg = SafetyFilterConfig.from_dict(filter_config)
        if self.clf.enabled and filter_cfg.clf_slack_enabled:
            layout_sizes["delta_clf"] = 1
            self.layout = DecisionLayout.from_sizes(**layout_sizes)
        self.filter = MultiCertificateFilter(filter_cfg, self.layout)

    def evaluate(self, state: np.ndarray, availability: Mapping[str, bool]) -> Mapping[str, Any]:
        """Evaluate every CLF/ROA certificate without solving the filter."""

        return self.clf.evaluate_many(np.asarray(state, dtype=float), availability=availability)

    def nominal_control(self, state: np.ndarray, active_target: str) -> np.ndarray:
        """Return the active-target LQR command or a documented PD fallback."""

        state_array = np.asarray(state, dtype=float).reshape(2 * self.dimension)
        if self.clf.enabled and active_target in self.clf.artifacts:
            artifact = self.clf.artifacts[active_target]
            return -artifact.K @ (state_array - artifact.target.x_star)
        target = self.targets[active_target]
        position_error = target.x_star[: self.dimension] - state_array[: self.dimension]
        velocity_error = target.x_star[self.dimension :] - state_array[self.dimension :]
        return (
            self.nominal_position_gain * position_error
            + self.nominal_velocity_gain * velocity_error
        )

    def step(
        self,
        *,
        state: np.ndarray,
        time_s: float,
        version: int,
        active_target: str,
        availability: Mapping[str, bool],
        safety_sample: SafetySample | None,
        dt_s: float,
        nominal_control_provider: Callable[[np.ndarray, str], np.ndarray] | None = None,
    ) -> ControllerStep:
        """Compute one verified multi-certificate acceleration command."""

        state_array = np.asarray(state, dtype=float).reshape(2 * self.dimension)
        snapshot = StateSnapshot(state_array, float(time_s), int(version))
        evaluations = self.clf.evaluate_many(snapshot, availability=availability)
        switched = False

        if not availability.get(active_target, False):
            certificates = {
                identifier: evaluation.roa_certificate
                for identifier, evaluation in evaluations.items()
            }
            try:
                active_target = maximum_margin(certificates)
            except ValueError as exc:
                return self._hold(
                    state_array,
                    active_target,
                    str(exc),
                    evaluations,
                    None,
                )
            switched = True

        nominal = (
            np.asarray(nominal_control_provider(state_array, active_target), dtype=float)
            if nominal_control_provider is not None
            else self.nominal_control(state_array, active_target)
        ).reshape(self.dimension)
        evaluations = self.clf.evaluate_many(
            snapshot,
            nominal_control=nominal,
            availability=availability,
        )
        active_evaluation = evaluations.get(active_target)
        if self.clf.enabled and active_evaluation is None:
            return self._hold(
                state_array,
                active_target,
                "Active CLF evaluation is unavailable.",
                evaluations,
                None,
            )

        contingency_evaluation: ContingencyEvaluation | None = None
        if self.contingency.enabled:
            contingency_evaluation = self.contingency.evaluate_certificates(
                evaluation.roa_certificate for evaluation in evaluations.values()
            )
            if (
                self.contingency.config.hold_when_lost
                and not contingency_evaluation.satisfied
            ):
                return self._hold(
                    state_array,
                    active_target,
                    "The configured r-out-of-p attraction-region requirement is not satisfied.",
                    evaluations,
                    contingency_evaluation,
                )

        bundles: list[ConstraintBundle] = []
        if self.cbf.enabled:
            if safety_sample is None:
                return self._hold(
                    state_array,
                    active_target,
                    "No valid Poisson field sample is available.",
                    evaluations,
                    contingency_evaluation,
                )
            bundles.append(
                self.cbf.evaluate(
                    snapshot,
                    {"sample": safety_sample, "layout": self.layout},
                )
            )

        if self.clf.enabled and active_evaluation is not None:
            relaxation_block = None
            relaxation_coefficient = 0.0
            if self.contingency.enabled:
                relaxation_block = "omega_contingency"
                relaxation_coefficient = (
                    self.contingency.active_clf_relaxation_coefficient(
                        active_evaluation.roa_certificate
                    )
                )
            bundles.append(
                ConstraintBundle(
                    constraints=(
                        self.clf.active_target_constraint(
                            evaluation=active_evaluation,
                            layout=self.layout,
                            relaxation_block=relaxation_block,
                            relaxation_coefficient=relaxation_coefficient,
                            slack_block=(
                                "delta_clf"
                                if self.filter.config.clf_slack_enabled
                                and self.layout.has_block("delta_clf")
                                else None
                            ),
                        ),
                    ),
                    source_box=self.clf.name,
                )
            )

        if self.contingency.enabled and contingency_evaluation is not None:
            bundles.append(
                self.contingency.build_constraints(
                    evaluation=contingency_evaluation,
                    layout=self.layout,
                )
            )

        bundles.append(self._sampled_velocity_bounds(state_array, dt_s))

        nominal_decision = np.zeros(self.layout.dimension, dtype=float)
        nominal_decision[self.layout.block_slice("control")] = nominal
        lower = np.full(self.layout.dimension, -np.inf, dtype=float)
        upper = np.full(self.layout.dimension, np.inf, dtype=float)
        lower[self.layout.block_slice("control")] = -self.maximum_acceleration
        upper[self.layout.block_slice("control")] = self.maximum_acceleration
        weights = np.ones(self.layout.dimension, dtype=float)
        if self.contingency.enabled:
            omega_index = self.layout.scalar_index("omega_contingency")
            lower[omega_index] = 0.0
            upper[omega_index] = 250.0
            weights[omega_index] = self.filter.config.omega_weight
        if self.layout.has_block("delta_clf"):
            delta_index = self.layout.scalar_index("delta_clf")
            lower[delta_index] = 0.0
            upper[delta_index] = self.filter.config.clf_slack_max
            weights[delta_index] = self.filter.config.clf_slack_weight

        result = self.filter.solve(
            nominal_decision=nominal_decision,
            bundles=bundles,
            lower_bounds=lower,
            upper_bounds=upper,
            weights=weights,
        )
        if result.status is not BoxStatus.READY:
            return ControllerStep(
                safe_control=np.zeros(self.dimension, dtype=float),
                nominal_control=nominal,
                active_target=active_target,
                target_switched=switched,
                status=result.status,
                message=result.message,
                omega=0.0,
                clf_slack=0.0,
                evaluations=evaluations,
                contingency=contingency_evaluation,
                filter_result=result,
                hocbf_residual=None,
                clf_residual=None,
                intervention_norm=float(np.linalg.norm(nominal)),
            )

        safe = np.asarray(result.decision[self.layout.block_slice("control")])
        omega = (
            float(result.decision[self.layout.scalar_index("omega_contingency")])
            if self.contingency.enabled
            else 0.0
        )
        clf_slack = (
            float(result.decision[self.layout.scalar_index("delta_clf")])
            if self.layout.has_block("delta_clf")
            else 0.0
        )
        hocbf_residual = self._hocbf_residual(
            state_array,
            safe,
            safety_sample,
        )
        clf_residual = self._clf_residual(
            active_evaluation,
            safe,
            omega,
            clf_slack,
        )
        return ControllerStep(
            safe_control=safe,
            nominal_control=nominal,
            active_target=active_target,
            target_switched=switched,
            status=BoxStatus.READY,
            message="",
            omega=omega,
            clf_slack=clf_slack,
            evaluations=evaluations,
            contingency=contingency_evaluation,
            filter_result=result,
            hocbf_residual=hocbf_residual,
            clf_residual=clf_residual,
            intervention_norm=float(np.linalg.norm(safe - nominal)),
        )

    def _sampled_velocity_bounds(self, state: np.ndarray, dt_s: float) -> ConstraintBundle:
        rows: list[AffineConstraint] = []
        velocity = state[self.dimension :]
        control_slice = self.layout.block_slice("control")
        for axis, value in enumerate(velocity):
            upper_row = np.zeros(self.layout.dimension, dtype=float)
            lower_row = np.zeros(self.layout.dimension, dtype=float)
            upper_row[control_slice.start + axis] = -1.0
            lower_row[control_slice.start + axis] = 1.0
            rows.append(
                AffineConstraint(
                    A=upper_row.reshape(1, -1),
                    b=np.array(
                        [-(self.maximum_speed_component - value) / dt_s],
                        dtype=float,
                    ),
                    name=f"velocity_upper_{axis}",
                    source_box="dynamic_limits",
                    equation_id="sampled_velocity_bound",
                    hardness=ConstraintHardness.HARD,
                    priority=1,
                )
            )
            rows.append(
                AffineConstraint(
                    A=lower_row.reshape(1, -1),
                    b=np.array(
                        [(-self.maximum_speed_component - value) / dt_s],
                        dtype=float,
                    ),
                    name=f"velocity_lower_{axis}",
                    source_box="dynamic_limits",
                    equation_id="sampled_velocity_bound",
                    hardness=ConstraintHardness.HARD,
                    priority=1,
                )
            )
        return ConstraintBundle(tuple(rows), source_box="dynamic_limits")

    def _hocbf_residual(
        self,
        state: np.ndarray,
        safe_control: np.ndarray,
        sample: SafetySample | None,
    ) -> float | None:
        if not self.cbf.enabled or sample is None:
            return None
        if self.cbf.config.mode == "velocity_cbf":
            return float(
                sample.grad_h @ safe_control
                + self.cbf.config.velocity_gamma
                * (sample.h - self.cbf.config.h_margin)
                + sample.partial_h_t
            )
        velocity = state[self.dimension :]
        return float(
            sample.grad_h @ safe_control
            + velocity @ sample.hessian_h @ velocity
            + (self.cbf.config.gamma1 + self.cbf.config.gamma2)
            * sample.grad_h
            @ velocity
            + self.cbf.config.gamma1
            * self.cbf.config.gamma2
            * (sample.h - self.cbf.config.h_margin)
            + sample.partial_h_t
        )

    @staticmethod
    def _clf_residual(
        active_evaluation: Any | None,
        safe_control: np.ndarray,
        omega: float,
        clf_slack: float = 0.0,
    ) -> float | None:
        if active_evaluation is None:
            return None
        derivative = float(
            active_evaluation.LfV + active_evaluation.LgV @ safe_control
        )
        relaxation = omega * max(0.0, -active_evaluation.h_roa)
        return float(-active_evaluation.alpha_V - derivative + relaxation + clf_slack)

    def _hold(
        self,
        state: np.ndarray,
        active_target: str,
        message: str,
        evaluations: Mapping[str, Any],
        contingency: ContingencyEvaluation | None,
    ) -> ControllerStep:
        nominal = (
            self.nominal_control(state, active_target)
            if active_target in self.targets
            else np.zeros(self.dimension)
        )
        return ControllerStep(
            safe_control=np.zeros(self.dimension),
            nominal_control=nominal,
            active_target=active_target,
            target_switched=False,
            status=BoxStatus.HOLD,
            message=message,
            omega=0.0,
            clf_slack=0.0,
            evaluations=evaluations,
            contingency=contingency,
            filter_result=None,
            hocbf_residual=None,
            clf_residual=None,
            intervention_norm=float(np.linalg.norm(nominal)),
        )
