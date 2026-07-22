"""Optional CVXPY backend for the CBF Safety Box."""

from __future__ import annotations

import time

import numpy as np

from .qp_problem import QPProblem


def solve_qp_cvxpy(problem: QPProblem) -> dict:
    """Solve the weighted convex QP with CVXPY when it is installed."""
    try:
        import cvxpy as cp
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("cvxpy is not installed. Use solver='scipy'.") from exc

    start = time.perf_counter()
    dimension = problem.u_nom.size
    decision = cp.Variable(dimension)
    weight_sqrt = np.sqrt(np.asarray(problem.quadratic_weights, dtype=float))

    constraints = [problem.A_ineq @ decision >= problem.b_ineq]
    if problem.lower_bounds is not None:
        constraints.append(decision >= problem.lower_bounds)
    if problem.upper_bounds is not None:
        constraints.append(decision <= problem.upper_bounds)
    for indices, bound in zip(problem.norm_bound_indices, problem.norm_bound_values):
        constraints.append(cp.norm(decision[np.asarray(indices, dtype=int)], 2) <= float(bound))

    slack_variable = None
    if problem.use_slack:
        # Replace the original affine row with its relaxed counterpart.
        constraints = constraints[1:]
        slack_variable = cp.Variable(nonneg=True)
        constraints.insert(0, problem.A_ineq @ decision + slack_variable >= problem.b_ineq)

    objective_expression = 0.5 * cp.sum_squares(cp.multiply(weight_sqrt, decision - problem.u_nom))
    if slack_variable is not None:
        objective_expression += problem.slack_weight * cp.square(slack_variable)

    optimization = cp.Problem(cp.Minimize(objective_expression), constraints)
    optimization.solve(solver=cp.CLARABEL, verbose=False)
    if decision.value is None:
        raise RuntimeError("CVXPY failed to return a solution.")

    solution = np.asarray(decision.value, dtype=float).reshape(-1)
    residuals = problem.A_ineq @ solution - problem.b_ineq
    return {
        "u_safe": solution,
        "status": str(optimization.status),
        "was_filtered": bool(np.linalg.norm(solution - problem.u_nom) > 1.0e-8),
        "residual": float(np.min(residuals)) if residuals.size else None,
        "residuals": residuals,
        "norm_residuals": np.array(
            [float(bound) - float(np.linalg.norm(solution[np.asarray(indices, dtype=int)]))
             for indices, bound in zip(problem.norm_bound_indices, problem.norm_bound_values)],
            dtype=float,
        ),
        "slack": float(slack_variable.value) if slack_variable is not None and slack_variable.value is not None else 0.0,
        "solve_time": time.perf_counter() - start,
        "active_constraints": [int(index) for index, value in enumerate(residuals) if abs(value) <= 1.0e-5],
    }
