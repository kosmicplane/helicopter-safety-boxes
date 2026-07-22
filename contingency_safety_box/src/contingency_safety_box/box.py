r"""Independent ``r``-out-of-``p`` combinatorial contingency safety box.

For CLF-derived certificates

.. math::

    h_j(x)=c_j-V_j(x),

and the ``r``-th-largest pivot ``\widetilde h_r``, the paper-exact rows are

.. math::

    \dot h_j(x,u)
    \ge
    -\alpha_c(h_j(x))
    -\omega\rho(h_j(x)-\widetilde h_r(x)),

using one shared nonnegative auxiliary variable ``omega``.  This module does
not construct CLFs; it operates on the generic certificate contract from
``safety_box_core``.
r"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from safety_box_core import (
    AffineConstraint,
    BoxStatus,
    CertificateEvaluation,
    ConstraintBundle,
    ConstraintHardness,
    DecisionLayout,
)


@dataclass(frozen=True, slots=True)
class ContingencyBoxConfig:
    r"""Configuration for the combinatorial certificate construction.r"""

    enabled: bool = True
    required_certified: int = 2
    alpha_gain: float = 0.18
    rho_gain: float = 1.0
    pivot_tolerance: float = 1.0e-8
    hold_when_lost: bool = True

    def __post_init__(self) -> None:
        if self.required_certified < 1:
            raise ValueError("required_certified must be at least one.")
        if self.alpha_gain <= 0.0 or self.rho_gain <= 0.0:
            raise ValueError("alpha_gain and rho_gain must be positive.")
        if self.pivot_tolerance < 0.0:
            raise ValueError("pivot_tolerance must be nonnegative.")

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object] | None,
    ) -> "ContingencyBoxConfig":
        return cls(**dict(data or {}))


@dataclass(frozen=True, slots=True)
class ContingencyEvaluation:
    r"""Order-statistic and availability diagnostics for one state.r"""

    pivot: float
    certified_count: int
    available_count: int
    critical_ids: tuple[str, ...]
    sorted_ids: tuple[str, ...]
    sorted_values: tuple[float, ...]
    satisfied: bool
    certificates: tuple[CertificateEvaluation, ...]


def rth_largest(values: Iterable[float], r: int) -> float:
    r"""Return the ``r``-th largest value in expected linear time.r"""

    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0:
        raise ValueError("At least one certificate is required.")
    if not 1 <= int(r) <= array.size:
        raise ValueError(f"r must lie in [1, {array.size}].")
    partition_index = array.size - int(r)
    return float(np.partition(array, partition_index)[partition_index])


class ContingencyBox:
    r"""Evaluate and enforce a combinatorial set of candidate certificates.r"""

    name = "contingency_safety_box"

    def __init__(self, config: ContingencyBoxConfig) -> None:
        self.config = config
        self.enabled = bool(config.enabled)

    def evaluate_certificates(
        self,
        certificates: Iterable[CertificateEvaluation],
    ) -> ContingencyEvaluation:
        r"""Compute the pivot, certified count, and critical certificate set.r"""

        available = tuple(certificate for certificate in certificates if certificate.available)
        certified_count = sum(certificate.value >= 0.0 for certificate in available)

        if len(available) < self.config.required_certified:
            return ContingencyEvaluation(
                pivot=float("-inf"),
                certified_count=int(certified_count),
                available_count=len(available),
                critical_ids=(),
                sorted_ids=tuple(certificate.identifier for certificate in available),
                sorted_values=tuple(float(certificate.value) for certificate in available),
                satisfied=False,
                certificates=available,
            )

        pivot = rth_largest(
            (certificate.value for certificate in available),
            self.config.required_certified,
        )
        ordered = sorted(
            available,
            key=lambda certificate: (-certificate.value, certificate.identifier),
        )
        critical = tuple(
            certificate.identifier
            for certificate in available
            if abs(certificate.value - pivot) <= self.config.pivot_tolerance
        )
        return ContingencyEvaluation(
            pivot=pivot,
            certified_count=int(certified_count),
            available_count=len(available),
            critical_ids=critical,
            sorted_ids=tuple(certificate.identifier for certificate in ordered),
            sorted_values=tuple(float(certificate.value) for certificate in ordered),
            satisfied=bool(
                pivot >= 0.0
                and certified_count >= self.config.required_certified
            ),
            certificates=available,
        )

    @staticmethod
    def active_clf_relaxation_coefficient(
        certificate: CertificateEvaluation,
    ) -> float:
        r"""Return the theorem-specified ``ReLU(-h_active)`` coefficient.r"""

        return float(max(0.0, -certificate.value))

    def build_constraints(
        self,
        *,
        evaluation: ContingencyEvaluation,
        layout: DecisionLayout,
        control_block: str = "control",
        omega_block: str = "omega_contingency",
    ) -> ConstraintBundle:
        r"""Build all smooth combinatorial rows using one shared ``omega``.r"""

        if not self.enabled:
            return ConstraintBundle.disabled(self.name)
        if evaluation.available_count < self.config.required_certified:
            return ConstraintBundle.hold(
                self.name,
                (
                    f"only {evaluation.available_count} certificates available; "
                    f"r={self.config.required_certified}"
                ),
                evaluation=evaluation,
            )
        if self.config.hold_when_lost and not evaluation.satisfied:
            return ConstraintBundle.hold(
                self.name,
                "r-out-of-p CLF region-of-attraction certificate is negative",
                evaluation=evaluation,
            )

        omega_index = layout.scalar_index(omega_block)
        rows: list[AffineConstraint] = []
        for certificate in evaluation.certificates:
            row = layout.lift_row(certificate.control_gradient, control_block)
            delta = float(certificate.value - evaluation.pivot)
            rho = self.config.rho_gain * delta * delta
            row[omega_index] = rho

            # dot(h) = drift + control_gradient @ u, hence
            # control_gradient @ u + rho*omega >= -alpha*h - drift.
            bound = -self.config.alpha_gain * certificate.value - certificate.drift
            rows.append(
                AffineConstraint(
                    A=row.reshape(1, -1),
                    b=np.array([bound], dtype=float),
                    name=f"contingency_{certificate.identifier}",
                    source_box=self.name,
                    equation_id="CBF-combinatorial-ROA",
                    hardness=ConstraintHardness.HARD,
                    slack_group=None,
                    priority=20,
                    enabled=True,
                    metadata={
                        "target_id": certificate.identifier,
                        "h_roa": float(certificate.value),
                        "pivot": float(evaluation.pivot),
                        "delta_from_pivot": delta,
                        "rho": rho,
                        "critical": certificate.identifier in evaluation.critical_ids,
                    },
                )
            )
        return ConstraintBundle(
            constraints=tuple(rows),
            status=BoxStatus.READY,
            source_box=self.name,
            diagnostics={"evaluation": evaluation},
        )


def maximum_margin(
    certificates: Mapping[str, CertificateEvaluation],
) -> str:
    r"""Select the available certified target with maximum ROA margin.r"""

    valid = [
        certificate
        for certificate in certificates.values()
        if certificate.available and certificate.value >= 0.0
    ]
    if not valid:
        raise ValueError("No available certified target exists.")
    return max(valid, key=lambda certificate: (certificate.value, certificate.identifier)).identifier
