"""Unified Poisson-CBF and combinatorial HJ safety filtering via cbf_safety_box."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from cbf_safety_box import (
    AffineCertificate,
    CBFBox,
    CBFBoxConfig,
    CBFBoxResult,
    SafetySample,
    build_active_target_reachability_constraint,
    build_combinatorial_contingency_constraints,
    lift_constraint_to_decision,
)
from cbf_safety_box.constraints.velocity_cbf import build_velocity_cbf_constraint

from .coordinates import FieldSample as PoissonFieldSample
from .hj_reachability import ReachabilityBundle, sample_zone_field


@dataclass(frozen=True)
class PoissonCBFConfig:
    """Configuration for the hard local Poisson velocity-CBF row."""

    enabled: bool = True
    alpha: float = 2.0
    h_margin: float = 0.05
    maximum_field_age_s: float = 0.50
    invalid_sample_action: str = "hold"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PoissonCBFConfig":
        config = cls(**dict(data or {}))
        if config.alpha <= 0.0:
            raise ValueError("poisson_cbf.alpha must be positive.")
        if config.h_margin < 0.0:
            raise ValueError("poisson_cbf.h_margin must be nonnegative.")
        if config.invalid_sample_action != "hold":
            raise ValueError("Only poisson_cbf.invalid_sample_action='hold' is supported.")
        return config


@dataclass(frozen=True)
class ContingencyFilterConfig:
    """Gains, auxiliary weights, solver, and failure behavior."""

    alpha_active: float = 1.0
    alpha_contingency: float = 0.55
    rho_gain: float = 2.0
    relaxation_weight_active: float = 35.0
    relaxation_weight_contingency: float = 35.0
    maximum_relaxation: float = 50.0
    solver: str = "scipy"
    tolerance: float = 1.0e-8
    failure_action: str = "hold"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ContingencyFilterConfig":
        config = cls(**dict(data or {}))
        if config.alpha_active <= 0.0 or config.alpha_contingency <= 0.0:
            raise ValueError("HJ alpha gains must be positive.")
        if config.rho_gain < 0.0:
            raise ValueError("rho_gain must be nonnegative.")
        if config.relaxation_weight_active <= 0.0 or config.relaxation_weight_contingency <= 0.0:
            raise ValueError("Relaxation weights must be positive.")
        if config.maximum_relaxation <= 0.0:
            raise ValueError("maximum_relaxation must be positive.")
        if config.solver not in {"scipy", "cvxpy"}:
            raise ValueError("contingency_filter.solver must be scipy or cvxpy.")
        if config.tolerance <= 0.0:
            raise ValueError("contingency_filter.tolerance must be positive.")
        if config.failure_action != "hold":
            raise ValueError("Only failure_action='hold' is supported.")
        return config


@dataclass(frozen=True)
class UnifiedFilterResult:
    """Safe planar command, certificate samples, residuals, and HOLD diagnostics."""

    success: bool
    safe_velocity_xy: np.ndarray
    nominal_velocity_xy: np.ndarray
    omega_active: float
    omega_contingency: float
    solver_status: str
    solve_time_s: float
    hold_reason: str | None
    poisson_h_raw: float | None
    poisson_h_effective: float | None
    reachability_values: dict[int, float]
    pivot: float
    reachable_count: int
    constraint_names: tuple[str, ...]
    residuals: dict[str, float]
    box_result: CBFBoxResult | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""

        return {
            "success": bool(self.success),
            "safe_velocity_xy": self.safe_velocity_xy.tolist(),
            "nominal_velocity_xy": self.nominal_velocity_xy.tolist(),
            "omega_active": float(self.omega_active),
            "omega_contingency": float(self.omega_contingency),
            "solver_status": str(self.solver_status),
            "solve_time_s": float(self.solve_time_s),
            "hold_reason": self.hold_reason,
            "poisson_h_raw": self.poisson_h_raw,
            "poisson_h_effective": self.poisson_h_effective,
            "reachability_values": {
                str(identifier): (None if not np.isfinite(value) else float(value))
                for identifier, value in self.reachability_values.items()
            },
            "pivot": None if not np.isfinite(self.pivot) else float(self.pivot),
            "reachable_count": int(self.reachable_count),
            "constraint_names": list(self.constraint_names),
            "residuals": {name: float(value) for name, value in self.residuals.items()},
        }


def _hold(
    nominal_velocity_xy: np.ndarray,
    reason: str,
    *,
    values: dict[int, float] | None = None,
    pivot: float = float("-inf"),
    reachable_count: int = 0,
    poisson_h_raw: float | None = None,
    poisson_h_effective: float | None = None,
) -> UnifiedFilterResult:
    """Construct a deterministic zero-command HOLD result."""

    nominal = np.asarray(nominal_velocity_xy, dtype=float).reshape(2)
    return UnifiedFilterResult(
        success=False,
        safe_velocity_xy=np.zeros(2, dtype=float),
        nominal_velocity_xy=nominal,
        omega_active=0.0,
        omega_contingency=0.0,
        solver_status="hold",
        solve_time_s=0.0,
        hold_reason=str(reason),
        poisson_h_raw=poisson_h_raw,
        poisson_h_effective=poisson_h_effective,
        reachability_values=dict(values or {}),
        pivot=float(pivot),
        reachable_count=int(reachable_count),
        constraint_names=tuple(),
        residuals={},
        box_result=None,
    )


class UnifiedContingencyFilter:
    """Build all rows and solve one augmented decision inside the CBF Safety Box."""

    def __init__(
        self,
        *,
        poisson_config: PoissonCBFConfig,
        filter_config: ContingencyFilterConfig,
        maximum_speed_mps: float,
        required_reachable: int,
    ) -> None:
        self.poisson_config = poisson_config
        self.filter_config = filter_config
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.required_reachable = int(required_reachable)
        if self.maximum_speed_mps <= 0.0:
            raise ValueError("maximum_speed_mps must be positive.")
        self.box = CBFBox(
            CBFBoxConfig(
                mode="velocity",
                solver=filter_config.solver,
                use_slack=False,
                alpha=poisson_config.alpha,
            )
        )

    def filter(
        self,
        *,
        position_xy: Iterable[float],
        nominal_velocity_xy: Iterable[float],
        poisson_sample: PoissonFieldSample | None,
        reachability_bundle: ReachabilityBundle,
        active_identifier: int,
        available_identifiers: Iterable[int],
        tau_active: float,
        tau_active_dot: float,
        tau_contingency: float,
        tau_contingency_dot: float = 0.0,
    ) -> UnifiedFilterResult:
        """Solve the hard Poisson row and paper-inspired HJ rows simultaneously."""

        position = np.asarray(position_xy, dtype=float).reshape(2)
        nominal = np.asarray(nominal_velocity_xy, dtype=float).reshape(2)
        available = [int(identifier) for identifier in available_identifiers]
        if len(available) < self.required_reachable:
            return _hold(nominal, f"only {len(available)} available zones remain; r={self.required_reachable}")
        if int(active_identifier) not in available:
            return _hold(nominal, f"active LZ-{active_identifier} is unavailable")
        if poisson_sample is None or not poisson_sample.valid or poisson_sample.h is None or poisson_sample.gradient_xy is None:
            reason = "invalid Poisson field sample" if poisson_sample is None else f"invalid Poisson sample: {poisson_sample.reason}"
            return _hold(nominal, reason)

        raw_h = float(poisson_sample.h)
        effective_h = raw_h - self.poisson_config.h_margin
        safety = SafetySample(
            h=effective_h,
            grad_h=np.asarray(poisson_sample.gradient_xy, dtype=float),
            hessian_h=(
                None if poisson_sample.hessian_xy is None else np.asarray(poisson_sample.hessian_xy, dtype=float)
            ),
            laplacian_h=poisson_sample.laplacian,
            metadata={
                "source": "poisson_safety_box",
                "raw_h": raw_h,
                "h_margin": self.poisson_config.h_margin,
                "occupancy_version": reachability_bundle.occupancy_version,
            },
        )
        poisson_row = build_velocity_cbf_constraint(safety, self.poisson_config.alpha)
        poisson_row = lift_constraint_to_decision(
            poisson_row,
            decision_dimension=4,
            source_to_target_columns=[0, 1],
            name="poisson_velocity_cbf",
        )

        active_field = reachability_bundle.fields[int(active_identifier)]
        active_sample = sample_zone_field(
            active_field,
            reachability_bundle.geometry,
            position,
            tau=tau_active,
            maximum_speed_mps=self.maximum_speed_mps,
        )
        if not active_sample.valid:
            return _hold(
                nominal,
                f"active LZ-{active_identifier} has no valid HJ sample: {active_sample.reason}",
                poisson_h_raw=raw_h,
                poisson_h_effective=effective_h,
            )
        active_drift = -self.maximum_speed_mps * float(tau_active_dot)
        active_row = build_active_target_reachability_constraint(
            value=active_sample.value,
            drift=active_drift,
            control_gradient=active_sample.gradient_xy,
            alpha=self.filter_config.alpha_active,
            active_auxiliary_index=2,
            decision_dimension=4,
            name=f"active_hj_LZ_{active_identifier}",
        )

        certificates: list[AffineCertificate] = []
        values: dict[int, float] = {}
        distances: dict[int, float] = {}
        reachable_count = 0
        for identifier, field in reachability_bundle.fields.items():
            is_available = identifier in available and field.available
            sample = sample_zone_field(
                field,
                reachability_bundle.geometry,
                position,
                tau=tau_contingency,
                maximum_speed_mps=self.maximum_speed_mps,
            )
            values[identifier] = sample.value if sample.valid and is_available else float("-inf")
            if sample.valid and is_available:
                distance = self.maximum_speed_mps * (-float(tau_contingency)) - sample.value
                distances[identifier] = float(distance)
                reachable_count += int(sample.value >= 0.0)
                certificates.append(
                    AffineCertificate(
                        name=f"LZ-{identifier}",
                        value=sample.value,
                        drift=-self.maximum_speed_mps * float(tau_contingency_dot),
                        control_gradient=sample.gradient_xy,
                        available=True,
                    )
                )
            else:
                distances[identifier] = float("inf")

        if len(certificates) < self.required_reachable:
            return _hold(
                nominal,
                f"only {len(certificates)} finite HJ certificates remain; r={self.required_reachable}",
                values=values,
                reachable_count=reachable_count,
                poisson_h_raw=raw_h,
                poisson_h_effective=effective_h,
            )
        contingency_rows, pivot = build_combinatorial_contingency_constraints(
            certificates,
            r=self.required_reachable,
            gamma=self.filter_config.alpha_contingency,
            auxiliary_gain=self.filter_config.rho_gain,
        )
        if not np.isfinite(pivot) or pivot < 0.0 or reachable_count < self.required_reachable:
            return _hold(
                nominal,
                "r-out-of-p reachability certificate is negative",
                values=values,
                pivot=pivot,
                reachable_count=reachable_count,
                poisson_h_raw=raw_h,
                poisson_h_effective=effective_h,
            )
        lifted_contingency = [
            lift_constraint_to_decision(
                row,
                decision_dimension=4,
                source_to_target_columns=[0, 1, 3],
            )
            for row in contingency_rows
        ]

        decision_nominal = np.asarray([nominal[0], nominal[1], 0.0, 0.0], dtype=float)
        constraints = [poisson_row, active_row, *lifted_contingency]
        result = self.box.filter_affine_constraints(
            decision_nominal=decision_nominal,
            constraints=constraints,
            lower_bounds=np.asarray([-np.inf, -np.inf, 0.0, 0.0]),
            upper_bounds=np.asarray(
                [
                    np.inf,
                    np.inf,
                    self.filter_config.maximum_relaxation,
                    self.filter_config.maximum_relaxation,
                ]
            ),
            quadratic_weights=np.asarray(
                [
                    1.0,
                    1.0,
                    self.filter_config.relaxation_weight_active,
                    self.filter_config.relaxation_weight_contingency,
                ]
            ),
            norm_bound_indices=[[0, 1]],
            norm_bound_values=[self.maximum_speed_mps],
            use_slack=False,
        )
        decision = np.asarray(result.u_safe, dtype=float).reshape(4)
        names = tuple(result.diagnostics.get("constraints", []))
        residual_vector = np.asarray(result.constraint_matrix @ decision - result.constraint_vector, dtype=float)
        residuals = {name: float(value) for name, value in zip(names, residual_vector)}
        feasibility = result.diagnostics.get("feasibility", {})
        explicit_feasible = bool(feasibility.get("feasible", False))
        norm_residuals = np.asarray(feasibility.get("norm_residuals", []), dtype=float)
        minimum_norm = float(np.min(norm_residuals)) if norm_residuals.size else np.inf
        successful = (
            result.solver_status == "optimal"
            and explicit_feasible
            and np.all(np.isfinite(decision))
            and (not residual_vector.size or float(np.min(residual_vector)) >= -10.0 * self.filter_config.tolerance)
            and minimum_norm >= -10.0 * self.filter_config.tolerance
        )
        if not successful:
            return UnifiedFilterResult(
                success=False,
                safe_velocity_xy=np.zeros(2, dtype=float),
                nominal_velocity_xy=nominal,
                omega_active=0.0,
                omega_contingency=0.0,
                solver_status=result.solver_status,
                solve_time_s=float(result.solve_time),
                hold_reason="unified CBF/HJ optimization failed explicit feasibility checks",
                poisson_h_raw=raw_h,
                poisson_h_effective=effective_h,
                reachability_values=values,
                pivot=float(pivot),
                reachable_count=int(reachable_count),
                constraint_names=names,
                residuals=residuals,
                box_result=result,
            )
        return UnifiedFilterResult(
            success=True,
            safe_velocity_xy=decision[:2].copy(),
            nominal_velocity_xy=nominal,
            omega_active=float(decision[2]),
            omega_contingency=float(decision[3]),
            solver_status=result.solver_status,
            solve_time_s=float(result.solve_time),
            hold_reason=None,
            poisson_h_raw=raw_h,
            poisson_h_effective=effective_h,
            reachability_values=values,
            pivot=float(pivot),
            reachable_count=int(reachable_count),
            constraint_names=names,
            residuals=residuals,
            box_result=result,
        )


__all__ = [
    "ContingencyFilterConfig",
    "PoissonCBFConfig",
    "UnifiedContingencyFilter",
    "UnifiedFilterResult",
]
