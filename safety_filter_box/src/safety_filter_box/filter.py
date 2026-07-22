r"""Unified optimization box for independently constructed certificate bundles.r"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import linprog

from safety_box_core import (
    AffineConstraint,
    BoxStatus,
    ConstraintBundle,
    DecisionLayout,
    FilterResult,
)

from .solvers import QPData, RawSolverResult, make_solver


@dataclass(frozen=True, slots=True)
class SafetyFilterConfig:
    r"""Numerical settings for the multi-certificate projection.r"""

    enabled: bool = True
    solver: str = "scipy_slsqp"
    tolerance: float = 1.0e-8
    max_iterations: int = 200
    residual_tolerance: float = 1.0e-5
    active_tolerance: float = 1.0e-5
    omega_weight: float = 30.0
    clf_slack_enabled: bool = True
    clf_slack_weight: float = 1.0e5
    clf_slack_max: float = 1.0e4

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0 or self.residual_tolerance <= 0.0:
            raise ValueError("Solver and residual tolerances must be positive.")
        if self.active_tolerance < 0.0:
            raise ValueError("active_tolerance must be nonnegative.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if self.omega_weight <= 0.0:
            raise ValueError("omega_weight must be positive.")
        if self.clf_slack_weight <= 0.0 or self.clf_slack_max <= 0.0:
            raise ValueError("CLF slack weight and maximum must be positive.")

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object] | None,
    ) -> "SafetyFilterConfig":
        return cls(**dict(data or {}))


class MultiCertificateFilter:
    r"""Solve one control projection from multiple modular constraint bundles.r"""

    name = "safety_filter_box"

    def __init__(
        self,
        config: SafetyFilterConfig,
        layout: DecisionLayout,
    ) -> None:
        self.config = config
        self.layout = layout
        self.enabled = bool(config.enabled)
        self.solver = make_solver(
            config.solver,
            tolerance=config.tolerance,
            max_iterations=config.max_iterations,
        )

    def solve(
        self,
        *,
        nominal_decision: np.ndarray,
        bundles: Iterable[ConstraintBundle],
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        weights: np.ndarray,
    ) -> FilterResult:
        r"""Solve and independently verify the augmented decision.

        All certificate rows use ``A z >= b``.  The optimization objective is

        .. math::

            \tfrac12(z-z_{\mathrm{nom}})^\top
            \operatorname{diag}(w)(z-z_{\mathrm{nom}}).

        Solver status is never trusted by itself: affine and bound residuals are
        recomputed explicitly.  When the local QP solver reports failure, a
        linear feasibility problem distinguishes true infeasibility from a
        numerical warning.
        r"""

        nominal = np.asarray(nominal_decision, dtype=float).reshape(
            self.layout.dimension
        )
        lower = np.asarray(lower_bounds, dtype=float).reshape(self.layout.dimension)
        upper = np.asarray(upper_bounds, dtype=float).reshape(self.layout.dimension)
        quadratic_weights = np.asarray(weights, dtype=float).reshape(
            self.layout.dimension
        )
        if np.any(lower >= upper):
            raise ValueError("Each lower decision bound must be below its upper bound.")
        if np.any(quadratic_weights <= 0.0):
            raise ValueError("All objective weights must be strictly positive.")

        bundle_tuple = tuple(bundles)
        blocking_statuses = {
            BoxStatus.HOLD,
            BoxStatus.INVALID,
            BoxStatus.ERROR,
            BoxStatus.INFEASIBLE,
        }
        blocking = [
            bundle for bundle in bundle_tuple if bundle.status in blocking_statuses
        ]
        if blocking:
            return FilterResult(
                decision=np.zeros_like(nominal),
                nominal_decision=nominal,
                status=BoxStatus.HOLD,
                solver_status="not_run",
                solve_time_s=0.0,
                residuals=np.array([], dtype=float),
                constraint_names=(),
                message="; ".join(
                    f"{bundle.source_box}: {bundle.message}" for bundle in blocking
                ),
                diagnostics={
                    "blocking_bundles": [bundle.source_box for bundle in blocking]
                },
            )

        constraints = [
            constraint
            for bundle in bundle_tuple
            for constraint in bundle.constraints
            if constraint.enabled
        ]
        constraints.sort(
            key=lambda constraint: (
                constraint.priority,
                constraint.source_box,
                constraint.name,
            )
        )

        if not self.enabled or not constraints:
            return FilterResult(
                decision=np.clip(nominal, lower, upper),
                nominal_decision=nominal,
                status=BoxStatus.DISABLED if not self.enabled else BoxStatus.READY,
                solver_status="passthrough",
                solve_time_s=0.0,
                residuals=np.array([], dtype=float),
                constraint_names=(),
                message="filter disabled" if not self.enabled else "no enabled constraints",
            )

        if any(
            constraint.decision_dimension != self.layout.dimension
            for constraint in constraints
        ):
            raise ValueError("At least one constraint has the wrong decision dimension.")

        A = np.vstack([constraint.A for constraint in constraints])
        b = np.concatenate([constraint.b for constraint in constraints])
        raw = self.solver.solve(
            QPData(
                nominal=nominal,
                weights=quadratic_weights,
                A=A,
                b=b,
                lower=lower,
                upper=upper,
            )
        )
        residual = A @ raw.x - b
        minimum_bound_residual = min(
            float(np.min(raw.x - lower)),
            float(np.min(upper - raw.x)),
        )
        explicit_feasible = bool(
            np.all(np.isfinite(raw.x))
            and float(np.min(residual)) >= -self.config.residual_tolerance
            and minimum_bound_residual >= -self.config.residual_tolerance
        )

        if not explicit_feasible:
            raw, residual, minimum_bound_residual, explicit_feasible = (
                self._try_feasible_fallback(
                    raw=raw,
                    A=A,
                    b=b,
                    lower=lower,
                    upper=upper,
                )
            )

        constraint_names, active_constraints = self._constraint_names(
            constraints,
            residual,
        )
        status = BoxStatus.READY if explicit_feasible else BoxStatus.INFEASIBLE
        solver_status = (
            raw.status
            if raw.success
            else ("optimal_inaccurate" if explicit_feasible else raw.status)
        )
        returned_decision = raw.x if explicit_feasible else np.zeros_like(nominal)

        return FilterResult(
            decision=returned_decision,
            nominal_decision=nominal,
            status=status,
            solver_status=solver_status,
            solve_time_s=raw.solve_time_s,
            residuals=residual,
            constraint_names=constraint_names,
            active_constraints=active_constraints,
            message=raw.message,
            diagnostics={
                "iterations": raw.iterations,
                "minimum_constraint_residual": float(np.min(residual)),
                "minimum_bound_residual": minimum_bound_residual,
                "intervention_norm": float(np.linalg.norm(raw.x - nominal)),
                "constraint_count": int(A.shape[0]),
                "decision_dimension": int(A.shape[1]),
            },
        )

    def _try_feasible_fallback(
        self,
        *,
        raw: RawSolverResult,
        A: np.ndarray,
        b: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[RawSolverResult, np.ndarray, float, bool]:
        r"""Use an LP to distinguish numerical failure from infeasibility.r"""

        feasibility_result = linprog(
            np.zeros(self.layout.dimension, dtype=float),
            A_ub=-A,
            b_ub=-b,
            bounds=list(zip(lower, upper, strict=True)),
            method="highs",
        )
        if not feasibility_result.success:
            residual = A @ raw.x - b
            bound_residual = min(
                float(np.min(raw.x - lower)),
                float(np.min(upper - raw.x)),
            )
            return raw, residual, bound_residual, False

        candidate = np.asarray(feasibility_result.x, dtype=float)
        candidate_residual = A @ candidate - b
        bound_residual = min(
            float(np.min(candidate - lower)),
            float(np.min(upper - candidate)),
        )
        feasible = bool(
            float(np.min(candidate_residual)) >= -self.config.residual_tolerance
            and bound_residual >= -self.config.residual_tolerance
        )
        if not feasible:
            return raw, candidate_residual, bound_residual, False

        fallback = RawSolverResult(
            x=candidate,
            status="feasible_fallback",
            success=True,
            solve_time_s=raw.solve_time_s,
            iterations=raw.iterations,
            message=(
                "The primary QP backend reported a warning; a separate linear "
                "feasibility solve produced an explicitly verified decision."
            ),
        )
        return fallback, candidate_residual, bound_residual, True

    def _constraint_names(
        self,
        constraints: list[AffineConstraint],
        residual: np.ndarray,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        names: list[str] = []
        active: list[str] = []
        cursor = 0
        for constraint in constraints:
            for row_index in range(constraint.rows):
                name = (
                    constraint.name
                    if constraint.rows == 1
                    else f"{constraint.name}[{row_index}]"
                )
                names.append(name)
                if abs(float(residual[cursor])) <= self.config.active_tolerance:
                    active.append(name)
                cursor += 1
        return tuple(names), tuple(active)
