"""QP solver backends used by the CBF Safety Box."""
from .qp_problem import QPProblem
from .closed_form import solve_halfspace_projection
from .scipy_qp import solve_qp_scipy
from .feasibility import check_feasibility


def solve_closed_form(problem: QPProblem):
    """Compatibility wrapper returning ``(u, info)`` for a QPProblem.

    This wrapper is intended for single half-space problems.  It calls the
    closed-form projection backend.
    """
    if problem.A_ineq.shape[0] != 1:
        raise ValueError("solve_closed_form supports exactly one inequality.")
    raw = solve_halfspace_projection(
        problem.u_nom,
        problem.A_ineq[0],
        problem.b_ineq[0],
        lower_bounds=problem.lower_bounds,
        upper_bounds=problem.upper_bounds,
    )
    info = dict(raw)
    info["feasible"] = raw.get("residual", -1.0) >= -1e-8
    return raw["u_safe"], info


def solve_scipy_qp(problem: QPProblem):
    """Compatibility wrapper returning ``(u, info)`` for SciPy SLSQP."""
    raw = solve_qp_scipy(problem)
    info = dict(raw)
    info["feasible"] = raw.get("residual", -1.0) >= -1e-6
    return raw["u_safe"], info

__all__ = [
    "QPProblem",
    "solve_halfspace_projection",
    "solve_qp_scipy",
    "check_feasibility",
    "solve_closed_form",
    "solve_scipy_qp",
]
