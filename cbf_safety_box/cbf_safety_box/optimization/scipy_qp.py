"""SciPy/SLSQP backend for the small convex QPs used by the CBF box."""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, minimize

from .qp_problem import QPProblem


def solve_qp_scipy(problem: QPProblem, tolerance: float = 1.0e-9) -> dict:
    """Solve a weighted safety-filter QP with affine and norm constraints.

    SLSQP accepts nonlinear inequality functions that are nonnegative when
    feasible.  Therefore the affine CBF rows are passed directly as ``A z-b``
    and each Euclidean norm bound is represented as ``limit²-||z_I||²``.
    """
    start = time.perf_counter()
    nominal = problem.u_nom
    weights = np.asarray(problem.quadratic_weights, dtype=float)
    decision_dimension = nominal.size

    # When enabled, one nonnegative emergency slack is appended to the decision.
    # The same slack relaxes every affine row; this mode is only used as a final
    # diagnostic fallback after the hard QP and task-relaxed QP have failed.
    if problem.use_slack:
        x0 = np.concatenate([nominal, np.array([0.0])])

        def objective(x: np.ndarray) -> float:
            delta = x[:decision_dimension] - nominal
            slack = float(x[decision_dimension])
            return 0.5 * float(np.sum(weights * delta * delta)) + problem.slack_weight * slack * slack

        def jacobian(x: np.ndarray) -> np.ndarray:
            gradient = np.zeros(decision_dimension + 1, dtype=float)
            gradient[:decision_dimension] = weights * (x[:decision_dimension] - nominal)
            gradient[decision_dimension] = 2.0 * problem.slack_weight * x[decision_dimension]
            return gradient

        def affine_residual(x: np.ndarray) -> np.ndarray:
            return problem.A_ineq @ x[:decision_dimension] + x[decision_dimension] - problem.b_ineq

        lower = np.full(decision_dimension + 1, -np.inf, dtype=float)
        upper = np.full(decision_dimension + 1, np.inf, dtype=float)
        if problem.lower_bounds is not None:
            lower[:decision_dimension] = problem.lower_bounds
        if problem.upper_bounds is not None:
            upper[:decision_dimension] = problem.upper_bounds
        lower[decision_dimension] = 0.0
    else:
        x0 = nominal.copy()

        def objective(x: np.ndarray) -> float:
            delta = x - nominal
            return 0.5 * float(np.sum(weights * delta * delta))

        def jacobian(x: np.ndarray) -> np.ndarray:
            return weights * (x - nominal)

        def affine_residual(x: np.ndarray) -> np.ndarray:
            return problem.A_ineq @ x - problem.b_ineq

        lower = np.full(decision_dimension, -np.inf, dtype=float)
        upper = np.full(decision_dimension, np.inf, dtype=float)
        if problem.lower_bounds is not None:
            lower[:] = problem.lower_bounds
        if problem.upper_bounds is not None:
            upper[:] = problem.upper_bounds

    constraints: list[dict] = [{"type": "ineq", "fun": affine_residual}]

    # Add one smooth convex norm constraint for every requested decision subset.
    # SLSQP sees the equivalent concave inequality limit²-||z_I||² >= 0.
    for indices, bound in zip(problem.norm_bound_indices, problem.norm_bound_values):
        index_array = np.asarray(indices, dtype=int)
        limit_squared = float(bound) ** 2

        def norm_residual(x: np.ndarray, idx=index_array, limit_sq=limit_squared) -> float:
            decision = x[:decision_dimension]
            return float(limit_sq - decision[idx] @ decision[idx])

        constraints.append({"type": "ineq", "fun": norm_residual})

    result = minimize(
        objective,
        x0,
        jac=jacobian,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        method="SLSQP",
        options={"ftol": tolerance, "maxiter": 180, "disp": False},
    )

    full_solution = np.asarray(result.x, dtype=float)
    decision = full_solution[:decision_dimension]
    affine_residuals = problem.A_ineq @ decision - problem.b_ineq
    norm_residuals = np.array(
        [
            float(bound) - float(np.linalg.norm(decision[np.asarray(indices, dtype=int)]))
            for indices, bound in zip(problem.norm_bound_indices, problem.norm_bound_values)
        ],
        dtype=float,
    )
    slack = float(full_solution[decision_dimension]) if problem.use_slack else 0.0

    # A solver may report failure even when the returned point is numerically
    # feasible.  Record both the native status and explicit residuals.
    minimum_affine = float(np.min(affine_residuals)) if affine_residuals.size else np.inf
    minimum_norm = float(np.min(norm_residuals)) if norm_residuals.size else np.inf
    feasible = minimum_affine + slack >= -10.0 * tolerance and minimum_norm >= -10.0 * tolerance
    status = "optimal" if result.success and feasible else f"scipy_failed:{result.message}"

    return {
        "u_safe": decision,
        "status": status,
        "was_filtered": bool(np.linalg.norm(decision - nominal) > 1.0e-8),
        "residual": minimum_affine,
        "residuals": affine_residuals,
        "norm_residuals": norm_residuals,
        "slack": slack,
        "solve_time": time.perf_counter() - start,
        "iterations": int(getattr(result, "nit", 0)),
        "active_constraints": [int(index) for index, value in enumerate(affine_residuals) if abs(value + slack) <= 1.0e-5],
        "raw_result": result,
    }
