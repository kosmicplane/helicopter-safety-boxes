import numpy as np
from cbf_safety_box.optimization import QPProblem, solve_closed_form, solve_scipy_qp


def test_closed_form_projection():
    p = QPProblem(u_nom=np.array([-1.0, 0.0]), A_ineq=np.array([[1.0,0.0]]), b_ineq=np.array([0.0]))
    u, info = solve_closed_form(p)
    assert u[0] >= -1e-8
    assert info["feasible"]


def test_scipy_projection():
    p = QPProblem(u_nom=np.array([-1.0, 0.0]), A_ineq=np.array([[1.0,0.0]]), b_ineq=np.array([0.0]), lower_bounds=np.array([-2,-2]), upper_bounds=np.array([2,2]))
    u, info = solve_scipy_qp(p)
    assert u[0] >= -1e-6
    assert info["feasible"]
