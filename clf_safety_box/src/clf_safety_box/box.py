r"""Modular CLF construction, batch evaluation, and active-target constraints.

The CLF box owns stability certificates only.  It does not read occupancy
maps, solve Poisson equations, select failed targets, or compose
``r``-out-of-``p`` requirements.  Those responsibilities belong to separate
boxes and are connected through :class:`safety_box_core.CertificateEvaluation`.
r"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from safety_box_core import (
    AffineConstraint,
    CertificateEvaluation,
    ConstraintHardness,
    DecisionLayout,
    EquilibriumTarget,
    StateSnapshot,
)

from .alpha import AlphaFunction, alpha_from_config
from .quadratic import QuadraticCLFArtifact, construct_quadratic_clf


@dataclass(frozen=True, slots=True)
class CLFBoxConfig:
    r"""Configuration for local quadratic CLF construction.r"""

    enabled: bool = True
    lqr_q_position: float = 0.05
    lqr_q_velocity: float = 0.10
    lqr_r: float = 20.0
    lyapunov_q: float = 1.0
    roa_fraction: float = 0.95
    manual_c: float | None = None
    alpha: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "linear", "gain": 0.035}
    )
    control_lower: tuple[float, ...] = (-4.0, -4.0, -4.0)
    control_upper: tuple[float, ...] = (4.0, 4.0, 4.0)

    def __post_init__(self) -> None:
        if min(
            self.lqr_q_position,
            self.lqr_q_velocity,
            self.lqr_r,
            self.lyapunov_q,
        ) <= 0.0:
            raise ValueError("LQR and Lyapunov weights must be positive.")
        if not 0.0 < self.roa_fraction <= 1.0:
            raise ValueError("roa_fraction must lie in (0, 1].")
        if self.manual_c is not None and self.manual_c <= 0.0:
            raise ValueError("manual_c must be positive when supplied.")
        object.__setattr__(self, "alpha", MappingProxyType(dict(self.alpha)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "CLFBoxConfig":
        return cls(**dict(data or {}))


@dataclass(frozen=True, slots=True)
class CLFEvaluation:
    r"""CLF and ROA quantities evaluated for one candidate target.r"""

    target_id: str
    V: float
    grad_V: np.ndarray
    LfV: float
    LgV: np.ndarray
    alpha_V: float
    h_roa: float
    inside_roa: bool
    roa_certificate: CertificateEvaluation
    clf_residual_nominal: float | None = None

    def __post_init__(self) -> None:
        gradient = np.array(self.grad_V, dtype=float, copy=True)
        control_gradient = np.array(self.LgV, dtype=float, copy=True)
        gradient.setflags(write=False)
        control_gradient.setflags(write=False)
        object.__setattr__(self, "grad_V", gradient)
        object.__setattr__(self, "LgV", control_gradient)


class CLFBox:
    r"""Construct and evaluate one local CLF for each landing equilibrium.r"""

    name = "clf_safety_box"

    def __init__(self, config: CLFBoxConfig, model: object) -> None:
        self.config = config
        self.model = model
        self.enabled = bool(config.enabled)
        self.alpha: AlphaFunction = alpha_from_config(dict(config.alpha))
        self._artifacts: dict[str, QuadraticCLFArtifact] = {}
        self._order: tuple[str, ...] = ()

    @property
    def artifacts(self) -> Mapping[str, QuadraticCLFArtifact]:
        r"""Read-only-by-convention mapping of target IDs to CLF artifacts.r"""

        return self._artifacts

    def prepare(
        self,
        targets: Iterable[EquilibriumTarget],
        artifact_dir: str | Path | None = None,
    ) -> None:
        r"""Construct and optionally persist all target-specific CLFs.r"""

        if not self.enabled:
            self._artifacts = {}
            self._order = ()
            return

        control_dimension = int(self.model.control_dimension)
        state_dimension = int(self.model.state_dimension)
        if state_dimension < control_dimension:
            raise ValueError("state dimension cannot be smaller than control dimension.")

        q_lqr = np.diag(
            [self.config.lqr_q_position] * control_dimension
            + [self.config.lqr_q_velocity]
            * (state_dimension - control_dimension)
        )
        r_lqr = np.eye(control_dimension, dtype=float) * self.config.lqr_r
        q_lyapunov = np.eye(state_dimension, dtype=float) * self.config.lyapunov_q

        lower = np.asarray(self.config.control_lower, dtype=float)
        upper = np.asarray(self.config.control_upper, dtype=float)
        if lower.size == 1:
            lower = np.repeat(lower, control_dimension)
        if upper.size == 1:
            upper = np.repeat(upper, control_dimension)
        if lower.size != control_dimension or upper.size != control_dimension:
            raise ValueError("Control-bound dimensions do not match the model.")
        if np.any(lower >= upper):
            raise ValueError("Each lower control bound must be below its upper bound.")

        artifacts: dict[str, QuadraticCLFArtifact] = {}
        for target in targets:
            if target.identifier in artifacts:
                raise ValueError(f"Duplicate target identifier {target.identifier!r}.")
            if target.x_star.size != state_dimension:
                raise ValueError(
                    f"Target {target.identifier!r} state dimension does not match the model."
                )
            if target.u_star.size != control_dimension:
                raise ValueError(
                    f"Target {target.identifier!r} control dimension does not match the model."
                )

            artifact = construct_quadratic_clf(
                model=self.model,
                target=target,
                lqr_q=q_lqr,
                lqr_r=r_lqr,
                q_lyapunov=q_lyapunov,
                control_lower=lower,
                control_upper=upper,
                roa_fraction=self.config.roa_fraction,
                manual_c=self.config.manual_c,
            )
            artifacts[target.identifier] = artifact
            if artifact_dir is not None:
                artifact.save(artifact_dir)

        if not artifacts:
            raise ValueError("At least one landing target is required.")
        self._artifacts = artifacts
        self._order = tuple(artifacts)

    def evaluate_many(
        self,
        state: StateSnapshot | np.ndarray,
        *,
        nominal_control: np.ndarray | None = None,
        availability: Mapping[str, bool] | None = None,
    ) -> dict[str, CLFEvaluation]:
        r"""Evaluate all target CLFs in one vectorized batch.

        For each target ``j``:

        .. math::

            V_j=e_j^\top P_je_j,\qquad \nabla V_j=2P_je_j,

        and the ROA certificate is ``h_j = c_j - V_j``.
        r"""

        if not self.enabled:
            return {}
        if not self._artifacts:
            raise RuntimeError("CLFBox.prepare must be called before evaluation.")

        x = (
            state.x
            if isinstance(state, StateSnapshot)
            else np.asarray(state, dtype=float).reshape(-1)
        )
        if x.size != int(self.model.state_dimension):
            raise ValueError("State dimension does not match the configured model.")

        artifacts = [self._artifacts[target_id] for target_id in self._order]
        stars = np.stack([artifact.target.x_star for artifact in artifacts])
        matrices_P = np.stack([artifact.P for artifact in artifacts])
        errors = x[None, :] - stars

        values_V = np.einsum(
            "bi,bij,bj->b",
            errors,
            matrices_P,
            errors,
            optimize=True,
        )
        gradients = 2.0 * np.einsum(
            "bij,bj->bi",
            matrices_P,
            errors,
            optimize=True,
        )
        drift = np.asarray(self.model.f(x), dtype=float)
        actuation = np.asarray(self.model.g(x), dtype=float)
        values_LfV = gradients @ drift
        values_LgV = gradients @ actuation
        values_alpha = np.asarray(self.alpha(values_V), dtype=float)

        nominal = None
        if nominal_control is not None:
            nominal = np.asarray(nominal_control, dtype=float).reshape(
                int(self.model.control_dimension)
            )

        evaluations: dict[str, CLFEvaluation] = {}
        for index, (target_id, artifact) in enumerate(
            zip(self._order, artifacts, strict=True)
        ):
            is_available = (
                artifact.target.available
                if availability is None
                else bool(availability.get(target_id, artifact.target.available))
            )
            h_roa = float(artifact.c - values_V[index])

            # Since h_j = c_j - V_j:
            # dot(h_j) = -LfV_j - LgV_j u.
            certificate = CertificateEvaluation(
                identifier=target_id,
                value=h_roa,
                drift=float(-values_LfV[index]),
                control_gradient=-np.asarray(values_LgV[index], dtype=float),
                available=is_available,
                source="clf_roa",
                metadata={
                    "V": float(values_V[index]),
                    "c": float(artifact.c),
                    "certification_method": artifact.certification_method,
                },
            )
            residual_nominal = None
            if nominal is not None:
                residual_nominal = float(
                    -values_alpha[index]
                    - (values_LfV[index] + values_LgV[index] @ nominal)
                )

            evaluations[target_id] = CLFEvaluation(
                target_id=target_id,
                V=float(values_V[index]),
                grad_V=np.asarray(gradients[index], dtype=float),
                LfV=float(values_LfV[index]),
                LgV=np.asarray(values_LgV[index], dtype=float),
                alpha_V=float(values_alpha[index]),
                h_roa=h_roa,
                inside_roa=h_roa >= 0.0,
                roa_certificate=certificate,
                clf_residual_nominal=residual_nominal,
            )
        return evaluations

    def active_target_constraint(
        self,
        *,
        evaluation: CLFEvaluation,
        layout: DecisionLayout,
        control_block: str = "control",
        relaxation_block: str | None = None,
        relaxation_coefficient: float = 0.0,
        slack_block: str | None = None,
    ) -> AffineConstraint:
        r"""Build the active-target CLF row in canonical ``A z >= b`` form.

        The enforced inequality is

        .. math::

            \dot V_j \le -\alpha_j(V_j)
            + \omega\,\operatorname{ReLU}(-h_j) + \delta_{\mathrm{clf}}.

        Rearranging yields

        .. math::

            -L_gV_j u
            + \operatorname{ReLU}(-h_j)\omega + \delta_{\mathrm{clf}}
            \ge L_fV_j + \alpha_j(V_j).
        r"""

        if relaxation_coefficient < 0.0:
            raise ValueError("relaxation_coefficient must be nonnegative.")
        row = layout.lift_row(-evaluation.LgV, control_block)
        if relaxation_block is not None:
            row[layout.scalar_index(relaxation_block)] = float(
                relaxation_coefficient
            )
        if slack_block is not None:
            row[layout.scalar_index(slack_block)] = 1.0
        return AffineConstraint(
            A=row.reshape(1, -1),
            b=np.array([evaluation.LfV + evaluation.alpha_V], dtype=float),
            name=f"clf_active_{evaluation.target_id}",
            source_box=self.name,
            equation_id="CLF-active",
            hardness=ConstraintHardness.HARD,
            slack_group=None,
            priority=10,
            enabled=True,
            metadata={
                "target_id": evaluation.target_id,
                "V": evaluation.V,
                "h_roa": evaluation.h_roa,
                "relaxation_coefficient": float(relaxation_coefficient),
                "clf_slack_block": slack_block,
            },
        )
