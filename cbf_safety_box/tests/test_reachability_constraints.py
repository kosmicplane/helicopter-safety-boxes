"""Tests for reachability-specific affine rows and generic decision lifting."""

from __future__ import annotations

import numpy as np
import pytest

from cbf_safety_box import (
    Constraint,
    build_active_target_reachability_constraint,
    lift_constraint_to_decision,
)


def test_active_reachability_row_has_correct_sign_and_auxiliary() -> None:
    constraint = build_active_target_reachability_constraint(
        value=-0.5,
        drift=-1.2,
        control_gradient=np.array([2.0, -3.0]),
        alpha=0.8,
        active_auxiliary_index=2,
        decision_dimension=4,
    )
    # ReLU(-alpha V) = ReLU(0.4) = 0.4.
    np.testing.assert_allclose(constraint.A, [[2.0, -3.0, 0.4, 0.0]])
    # -alpha V - drift = 0.4 + 1.2 = 1.6.
    np.testing.assert_allclose(constraint.b, [1.6])


def test_active_reachability_auxiliary_inactive_inside_set() -> None:
    constraint = build_active_target_reachability_constraint(
        value=2.0,
        drift=-0.5,
        control_gradient=np.array([1.0, 0.0]),
        alpha=1.0,
        active_auxiliary_index=2,
        decision_dimension=4,
    )
    np.testing.assert_allclose(constraint.A, [[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(constraint.b, [-1.5])


def test_lift_constraint_to_arbitrary_augmented_columns() -> None:
    source = Constraint(
        A=np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 5.0]]),
        b=np.array([4.0, 6.0]),
        name="source",
    )
    lifted = lift_constraint_to_decision(
        source,
        decision_dimension=5,
        source_to_target_columns=[0, 2, 4],
    )
    np.testing.assert_allclose(
        lifted.A,
        [[1.0, 0.0, 2.0, 0.0, 3.0], [-1.0, 0.0, 0.0, 0.0, 5.0]],
    )
    np.testing.assert_allclose(lifted.b, source.b)


def test_lift_rejects_duplicate_destination_columns() -> None:
    source = Constraint(A=np.eye(2), b=np.zeros(2))
    with pytest.raises(ValueError, match="unique"):
        lift_constraint_to_decision(
            source,
            decision_dimension=3,
            source_to_target_columns=[0, 0],
        )
