"""Small deterministic active-set helper for simple CBF-QPs.

This is not a full industrial active-set implementation.  It handles the common
single-halfspace case exactly and delegates multi-constraint problems to the
SciPy backend.  Keeping this file functional avoids a dead structural module
while preserving a clear upgrade path.
"""
from __future__ import annotations
from .closed_form import project_to_halfspace
from .scipy_qp import solve_scipy_qp


def solve_active_set(problem):
    """Solve a small QP problem using the simplest applicable active-set logic."""
    if problem.A_ineq is not None and len(problem.A_ineq) == 1:
        return project_to_halfspace(problem)
    return solve_scipy_qp(problem)
