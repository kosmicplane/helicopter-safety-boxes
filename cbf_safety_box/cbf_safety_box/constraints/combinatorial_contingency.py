r"""Affine CLF and r-out-of-p contingency constraints for the CBF Safety Box.

The standard CBF box already constructs the environmental CBF/HOCBF row from a
``SafetySample``.  This module adds the *additional* affine rows required by the
contingency-aware landing study.  Keeping these builders inside the CBF package
is important: the runtime should describe the mission and certificates, while
this package owns the mathematical conversion from certificates to QP rows.

All constraints use the package-wide convention

    A z >= b,

where the augmented decision is

    z = [u_1, ..., u_m, omega].

The first ``m`` entries are the physical control command and ``omega`` is the
single nonnegative auxiliary variable used by the combinatorial construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .builders import Constraint


@dataclass(frozen=True)
class AffineCertificate:
    r"""Local affine model of one landing-zone certificate.

    ``value`` is the current certificate value :math:`\rho_i`.  Along the
    reduced-order dynamics, its derivative is represented as

    .. math::

        \dot\rho_i = d_i + g_i^T u,

    where ``drift`` stores :math:`d_i` and ``control_gradient`` stores
    :math:`g_i`.  ``available=False`` removes a target after a confirmed hazard.
    """

    name: str
    value: float
    drift: float
    control_gradient: np.ndarray
    available: bool = True

    def __post_init__(self) -> None:
        """Normalize and validate numerical fields at construction time."""
        gradient = np.asarray(self.control_gradient, dtype=float).reshape(-1)
        if gradient.size == 0 or not np.all(np.isfinite(gradient)):
            raise ValueError("control_gradient must be a finite nonempty vector.")
        if not np.isfinite(self.value) or not np.isfinite(self.drift):
            raise ValueError("Certificate value and drift must be finite.")
        object.__setattr__(self, "control_gradient", gradient)


def rth_largest_pivot(values: Iterable[float], r: int) -> float:
    """Return the r-th largest value using one-based ``r`` indexing.

    The function is intentionally small and deterministic because this pivot is
    the scalar that defines the r-out-of-p superlevel set.
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("At least one certificate value is required.")
    if not 1 <= int(r) <= array.size:
        raise ValueError(f"r must lie in [1, {array.size}], received {r}.")
    return float(np.sort(array)[-int(r)])


def lift_constraint_with_auxiliary(
    constraint: Constraint,
    auxiliary_coefficient: float = 0.0,
) -> Constraint:
    """Append one auxiliary-variable column to an existing affine constraint.

    This backward-compatible helper preserves its original behavior, including
    support for a nonzero coefficient.  New code that needs arbitrary column
    placement should use :func:`lift_constraint_to_decision`.
    """
    matrix = np.asarray(constraint.A, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    coefficient = float(auxiliary_coefficient)
    column = np.full((matrix.shape[0], 1), coefficient, dtype=float)
    return Constraint(
        A=np.hstack([matrix, column]),
        b=np.asarray(constraint.b, dtype=float).copy(),
        name=constraint.name,
        metadata={**constraint.metadata, "auxiliary_coefficient": coefficient},
    )


def build_active_target_clf_constraint(
    *,
    value: float,
    drift: float,
    control_gradient: np.ndarray,
    gamma: float,
    relaxation_coefficient: float = 0.0,
    name: str = "active_target_clf",
) -> Constraint:
    r"""Build a relaxed CLF row for the active landing reference.

    The desired CLF condition is

    .. math::

        d_V + g_V^T u \le -\gamma V + c_\omega\omega.

    Rearranging it into the package convention yields

    .. math::

        -g_V^T u + c_\omega\omega \ge \gamma V + d_V.
    """
    gradient = np.asarray(control_gradient, dtype=float).reshape(-1)
    row = np.concatenate([-gradient, [float(relaxation_coefficient)]])
    bound = float(gamma) * float(value) + float(drift)
    return Constraint(
        A=row.reshape(1, -1),
        b=np.array([bound], dtype=float),
        name=name,
        metadata={
            "type": "clf",
            "value": float(value),
            "drift": float(drift),
            "gamma": float(gamma),
            "relaxation_coefficient": float(relaxation_coefficient),
        },
    )


def build_combinatorial_contingency_constraints(
    certificates: Iterable[AffineCertificate],
    *,
    r: int,
    gamma: float,
    auxiliary_gain: float,
) -> tuple[list[Constraint], float]:
    r"""Build the ``p`` smooth combinatorial contingency rows.

    For every available certificate, the implemented condition is

    .. math::

        \dot\rho_i \ge -\gamma\rho_i
        - \omega k_R(\rho_i-\widetilde\rho)^2,

    where :math:`\widetilde\rho` is the r-th largest available certificate.  With
    :math:`\dot\rho_i=d_i+g_i^Tu`, the affine QP row becomes

    .. math::

        g_i^Tu + k_R(\rho_i-\widetilde\rho)^2\omega
        \ge -\gamma\rho_i-d_i.

    Unavailable targets are omitted rather than represented by fake constraints.
    The caller must separately verify that at least ``r`` candidates remain.
    """
    available = [certificate for certificate in certificates if certificate.available]
    if len(available) < int(r):
        raise ValueError(
            f"Only {len(available)} landing certificates are available, but r={r} is required."
        )

    pivot = rth_largest_pivot((certificate.value for certificate in available), r)
    constraints: list[Constraint] = []

    for certificate in available:
        delta = float(certificate.value - pivot)
        auxiliary_coefficient = float(auxiliary_gain) * delta * delta
        row = np.concatenate([certificate.control_gradient, [auxiliary_coefficient]])
        bound = -float(gamma) * float(certificate.value) - float(certificate.drift)
        constraints.append(
            Constraint(
                A=row.reshape(1, -1),
                b=np.array([bound], dtype=float),
                name=f"contingency_{certificate.name}",
                metadata={
                    "type": "combinatorial_contingency",
                    "rho": float(certificate.value),
                    "pivot": pivot,
                    "delta_from_pivot": delta,
                    "gamma": float(gamma),
                    "auxiliary_gain": float(auxiliary_gain),
                    "auxiliary_coefficient": auxiliary_coefficient,
                    "drift": float(certificate.drift),
                },
            )
        )

    return constraints, pivot



def lift_constraint_to_decision(
    constraint: Constraint,
    *,
    decision_dimension: int,
    source_to_target_columns: Iterable[int],
    name: str | None = None,
) -> Constraint:
    """Embed an affine row in a larger augmented decision vector.

    Parameters
    ----------
    constraint:
        Source inequality ``A x >= b``.
    decision_dimension:
        Number of columns in the destination decision vector.
    source_to_target_columns:
        Destination column for each source decision column.  For example,
        ``[0, 1]`` lifts a planar CBF row into ``[u_x, u_y, omega_1, omega_2]``;
        ``[0, 1, 3]`` lifts ``[u_x, u_y, omega_2]`` into the same decision.
    name:
        Optional replacement constraint name.

    The numerical row is copied without changing its mathematical meaning.  All
    unassigned destination columns are exactly zero.
    """
    matrix = np.asarray(constraint.A, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    mapping = [int(index) for index in source_to_target_columns]
    if len(mapping) != matrix.shape[1]:
        raise ValueError(
            "source_to_target_columns must contain one destination index for each source column."
        )
    if int(decision_dimension) <= 0:
        raise ValueError("decision_dimension must be positive.")
    if len(set(mapping)) != len(mapping):
        raise ValueError("Destination column indices must be unique.")
    if any(index < 0 or index >= int(decision_dimension) for index in mapping):
        raise ValueError("A destination column index is outside the augmented decision.")

    lifted = np.zeros((matrix.shape[0], int(decision_dimension)), dtype=float)
    for source_column, target_column in enumerate(mapping):
        lifted[:, target_column] = matrix[:, source_column]
    return Constraint(
        A=lifted,
        b=np.asarray(constraint.b, dtype=float).copy(),
        name=constraint.name if name is None else str(name),
        metadata={
            **constraint.metadata,
            "lifted_from_dimension": int(matrix.shape[1]),
            "lifted_to_dimension": int(decision_dimension),
            "source_to_target_columns": mapping,
        },
    )


def build_active_target_reachability_constraint(
    *,
    value: float,
    drift: float,
    control_gradient: np.ndarray,
    alpha: float,
    active_auxiliary_index: int | None = None,
    decision_dimension: int | None = None,
    name: str = "active_target_reachability",
) -> Constraint:
    r"""Build the paper-inspired active-target reachability row.

    The certificate derivative is represented as

    .. math::

        \dot V = d_V + g_V^T u.

    The implemented condition is

    .. math::

        \dot V \ge -\alpha V
        - \omega_a\,\operatorname{ReLU}(-\alpha V).

    With the package convention ``A z >= b`` this becomes

    .. math::

        g_V^T u
        + \operatorname{ReLU}(-\alpha V)\omega_a
        \ge -\alpha V-d_V.

    When ``active_auxiliary_index`` is ``None`` the hard row contains only the
    physical control.  Otherwise ``decision_dimension`` must identify the full
    augmented decision and the auxiliary coefficient is inserted at the selected
    column.  The first ``len(control_gradient)`` columns are the physical control.
    """
    gradient = np.asarray(control_gradient, dtype=float).reshape(-1)
    if gradient.size == 0 or not np.all(np.isfinite(gradient)):
        raise ValueError("control_gradient must be a finite nonempty vector.")
    if not np.isfinite(value) or not np.isfinite(drift):
        raise ValueError("value and drift must be finite.")
    if not np.isfinite(alpha) or float(alpha) <= 0.0:
        raise ValueError("alpha must be a positive finite scalar.")

    if active_auxiliary_index is None:
        dimension = gradient.size if decision_dimension is None else int(decision_dimension)
        if dimension < gradient.size:
            raise ValueError("decision_dimension cannot be smaller than the control dimension.")
        row = np.zeros(dimension, dtype=float)
        row[: gradient.size] = gradient
        auxiliary_coefficient = 0.0
    else:
        if decision_dimension is None:
            raise ValueError("decision_dimension is required when an auxiliary index is used.")
        dimension = int(decision_dimension)
        auxiliary_index = int(active_auxiliary_index)
        if dimension < gradient.size + 1:
            raise ValueError("The augmented decision must include the physical control and auxiliary variable.")
        if auxiliary_index < gradient.size or auxiliary_index >= dimension:
            raise ValueError("active_auxiliary_index must point to a non-control decision column.")
        row = np.zeros(dimension, dtype=float)
        row[: gradient.size] = gradient
        auxiliary_coefficient = max(0.0, -float(alpha) * float(value))
        row[auxiliary_index] = auxiliary_coefficient

    bound = -float(alpha) * float(value) - float(drift)
    return Constraint(
        A=row.reshape(1, -1),
        b=np.asarray([bound], dtype=float),
        name=str(name),
        metadata={
            "type": "active_target_reachability",
            "value": float(value),
            "drift": float(drift),
            "alpha": float(alpha),
            "auxiliary_coefficient": float(auxiliary_coefficient),
            "active_auxiliary_index": active_auxiliary_index,
            "control_dimension": int(gradient.size),
            "decision_dimension": int(row.size),
        },
    )
