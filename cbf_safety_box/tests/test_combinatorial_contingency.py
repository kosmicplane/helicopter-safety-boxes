"""Tests for the contingency-aware extension of the CBF Safety Box."""

import numpy as np

from cbf_safety_box import (
    AffineCertificate,
    CBFBox,
    CBFBoxConfig,
    Constraint,
    build_combinatorial_contingency_constraints,
)


def test_r_out_of_p_constraints_are_solved_by_cbf_box() -> None:
    """The augmented CBF box should preserve a two-out-of-three certificate set."""
    certificates = [
        AffineCertificate("0", 0.4, -0.02, np.array([1.0, 0.0, 0.0])),
        AffineCertificate("1", 0.3, -0.01, np.array([0.0, 1.0, 0.0])),
        AffineCertificate("2", -0.2, 0.00, np.array([0.0, 0.0, 1.0])),
    ]
    contingency, pivot = build_combinatorial_contingency_constraints(
        certificates,
        r=2,
        gamma=0.2,
        auxiliary_gain=0.1,
    )
    assert np.isclose(pivot, 0.3)

    # Add one simple environmental row so the test exercises a heterogeneous
    # multi-row QP with the augmented decision z=[a_x,a_y,a_z,omega].
    environment = Constraint(
        A=np.array([[1.0, 0.0, 0.0, 0.0]]),
        b=np.array([-0.5]),
        name="environment",
    )
    box = CBFBox(CBFBoxConfig(mode="acceleration", solver="scipy"))
    result = box.filter_affine_constraints(
        decision_nominal=np.zeros(4),
        constraints=[environment, *contingency],
        lower_bounds=np.array([-np.inf, -np.inf, -np.inf, 0.0]),
        upper_bounds=np.array([np.inf, np.inf, np.inf, 10.0]),
        quadratic_weights=np.array([1.0, 1.0, 1.0, 0.2]),
        norm_bound_indices=[[0, 1, 2]],
        norm_bound_values=[1.0],
    )
    assert result.diagnostics["feasibility"]["feasible"]
    assert result.u_safe.shape == (4,)
    assert result.u_safe[3] >= -1.0e-9
