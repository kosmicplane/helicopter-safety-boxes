"""Unit-level verification of the added CLF, contingency, and filter boxes."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from safety_box_core import (
    AffineConstraint,
    CertificateEvaluation,
    ConstraintBundle,
    DecisionLayout,
    EquilibriumTarget,
    StateSnapshot,
    load_experiment_config,
)
from clf_safety_box import CLFBox, CLFBoxConfig, DoubleIntegratorModel
from contingency_safety_box import ContingencyBox, ContingencyBoxConfig, rth_largest
from safety_filter_box import MultiCertificateFilter, SafetyFilterConfig
from cbf_safety_box import CBFBox, CBFBoxConfig, SafetySample, SystemState


def test_central_configuration_and_named_layout() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs" / "experiment.yaml", profile="smoke")
    assert config["boxes"]["clf"]["enabled"] is True
    layout = DecisionLayout.from_sizes(control=3, omega_contingency=1, delta_clf=1)
    assert layout.dimension == 5
    assert layout.has_block("delta_clf")
    assert np.allclose(layout.lift_row(np.array([1.0, 2.0, 3.0]), "control"), [1, 2, 3, 0, 0])


def _prepared_clf_box() -> tuple[CLFBox, EquilibriumTarget]:
    model = DoubleIntegratorModel(2)
    target = EquilibriumTarget("LZ0", np.array([2.0, 1.0, 0.0, 0.0]), np.zeros(2))
    config = CLFBoxConfig.from_dict(
        {
            "lqr_q_position": 1.0,
            "lqr_q_velocity": 0.5,
            "lqr_r": 10.0,
            "lyapunov_q": 1.0,
            "roa_fraction": 0.9,
            "control_lower": (-2.0, -2.0),
            "control_upper": (2.0, 2.0),
            "alpha": {"type": "linear", "gain": 0.05},
        }
    )
    box = CLFBox(config, model)
    box.prepare([target])
    return box, target


def test_quadratic_clf_construction_gradient_and_lyapunov_residual() -> None:
    box, target = _prepared_clf_box()
    artifact = box.artifacts[target.identifier]
    residual = artifact.A.copy()
    closed_loop = artifact.A - artifact.B @ artifact.K
    residual = closed_loop.T @ artifact.P + artifact.P @ closed_loop + artifact.Q_lyapunov
    assert np.linalg.norm(residual, ord="fro") < 1.0e-9
    assert np.min(np.linalg.eigvalsh(artifact.P)) > 0.0
    assert artifact.c > 0.0

    state = np.array([0.5, 0.2, 0.1, -0.3])
    evaluation = box.evaluate_many(StateSnapshot(state))[target.identifier]
    epsilon = 1.0e-6
    finite_difference = np.empty_like(state)
    for index in range(state.size):
        plus = state.copy(); plus[index] += epsilon
        minus = state.copy(); minus[index] -= epsilon
        vp = box.evaluate_many(plus)[target.identifier].V
        vm = box.evaluate_many(minus)[target.identifier].V
        finite_difference[index] = (vp - vm) / (2.0 * epsilon)
    assert np.allclose(evaluation.grad_V, finite_difference, rtol=2.0e-5, atol=2.0e-6)
    assert np.isclose(evaluation.h_roa, artifact.c - evaluation.V)


def test_clf_row_has_shared_omega_and_standard_clf_slack() -> None:
    box, target = _prepared_clf_box()
    evaluation = box.evaluate_many(np.array([0.4, 0.5, 0.1, -0.2]))[target.identifier]
    layout = DecisionLayout.from_sizes(control=2, omega_contingency=1, delta_clf=1)
    row = box.active_target_constraint(
        evaluation=evaluation,
        layout=layout,
        relaxation_block="omega_contingency",
        relaxation_coefficient=max(0.0, -evaluation.h_roa),
        slack_block="delta_clf",
    )
    assert row.A.shape == (1, 4)
    assert row.A[0, layout.scalar_index("delta_clf")] == 1.0
    assert row.A[0, layout.scalar_index("omega_contingency")] >= 0.0


def test_contingency_pivot_and_paper_structured_rows() -> None:
    certificates = tuple(
        CertificateEvaluation(
            identifier=f"LZ{index}",
            value=value,
            drift=0.1 * index,
            control_gradient=np.array([1.0 + index, -0.5]),
            available=True,
            source="clf_roa",
        )
        for index, value in enumerate([2.0, -0.2, 1.0, 0.5])
    )
    assert rth_largest([2.0, -0.2, 1.0, 0.5], 2) == 1.0
    box = ContingencyBox(ContingencyBoxConfig(required_certified=2, alpha_gain=0.2))
    evaluation = box.evaluate_certificates(certificates)
    assert evaluation.pivot == 1.0
    assert evaluation.certified_count == 3
    assert evaluation.satisfied
    assert evaluation.critical_ids == ("LZ2",)

    layout = DecisionLayout.from_sizes(control=2, omega_contingency=1)
    bundle = box.build_constraints(evaluation=evaluation, layout=layout)
    assert len(bundle.constraints) == 4
    critical = next(row for row in bundle.constraints if row.metadata["critical"])
    assert critical.A[0, layout.scalar_index("omega_contingency")] == 0.0
    assert all(row.A.shape == (1, 3) for row in bundle.constraints)


def test_hildreth_and_slsqp_produce_verified_projection() -> None:
    layout = DecisionLayout.from_sizes(control=2)
    constraint = AffineConstraint(
        A=np.array([[1.0, 0.0]]),
        b=np.array([1.0]),
        name="x_lower",
        source_box="test",
        equation_id="projection",
    )
    bundle = ConstraintBundle((constraint,), source_box="test")
    for solver in ("hildreth", "scipy_slsqp"):
        filter_box = MultiCertificateFilter(
            SafetyFilterConfig(solver=solver, residual_tolerance=1.0e-6),
            layout,
        )
        result = filter_box.solve(
            nominal_decision=np.array([-2.0, 0.25]),
            bundles=[bundle],
            lower_bounds=np.array([-4.0, -4.0]),
            upper_bounds=np.array([4.0, 4.0]),
            weights=np.ones(2),
        )
        assert result.status.value == "ready"
        assert np.min(result.residuals) >= -1.0e-6
        assert np.allclose(result.decision, [1.0, 0.25], atol=2.0e-5)


def test_full_cbf_box_retains_original_api_and_shared_constraint_adapter() -> None:
    config = CBFBoxConfig.from_dict(
        {
            "mode": "acceleration_hocbf",
            "gamma1": 1.2,
            "gamma2": 1.1,
            "h_margin": 0.01,
        }
    )
    box = CBFBox(config)
    sample = SafetySample(
        h=0.4,
        grad_h=np.array([1.0, 0.0]),
        hessian_h=np.eye(2) * 0.1,
        partial_h_t=0.0,
    )
    # Existing standalone API remains available.
    original = box.filter_control(
        state=SystemState(position=np.array([0.0, 0.0]), velocity=np.array([-0.2, 0.0])),
        safety=sample,
        u_nom=np.array([-1.0, 0.0]),
    )
    assert original.u_safe.shape == (2,)

    # New provider adapter emits the shared A z >= b contract.
    layout = DecisionLayout.from_sizes(control=2, omega_contingency=1, delta_clf=1)
    row = box.build_constraint(
        state=np.array([0.0, 0.0, -0.2, 0.0]),
        sample=sample,
        layout=layout,
    )
    assert row.A.shape == (1, 4)
    assert row.source_box == "cbf_safety_box"
