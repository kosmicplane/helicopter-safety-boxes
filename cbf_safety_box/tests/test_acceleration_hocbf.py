import numpy as np
from cbf_safety_box.constraints.acceleration_hocbf import build_acceleration_hocbf_constraint
from cbf_safety_box import SystemState, SafetySample, CBFBox, CBFBoxConfig


def test_hocbf_rhs_shape():
    safety = SafetySample(h=0.2, grad_h=np.array([1.0, 0.0]), hessian_h=np.eye(2))
    state = SystemState(position=np.zeros(2), velocity=np.array([-1.0, 0.0]))
    c = build_acceleration_hocbf_constraint(safety, state, 2.0, 3.0)
    assert c.A.shape == (1,2)
    assert c.b.shape == (1,)


def test_acceleration_solver_feasible():
    cfg = CBFBoxConfig(mode="acceleration", solver="scipy", alpha1=2.0, alpha2=2.0, control_lower_bound=[-5,-5], control_upper_bound=[5,5])
    safety = SafetySample(h=0.2, grad_h=np.array([1.0, 0.0]), hessian_h=np.eye(2)*0.1)
    state = SystemState(position=np.zeros(2), velocity=np.array([-1.0, 0.0]))
    r = CBFBox(cfg).filter_control(state, safety, np.array([-2.0, 0.0]))
    assert r.cbf_residual >= -1e-6
