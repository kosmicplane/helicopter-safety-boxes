import numpy as np
from cbf_safety_box import CBFBox, CBFBoxConfig, SystemState, SafetySample


def test_velocity_feasible_not_modified():
    box = CBFBox(CBFBoxConfig(mode="velocity", solver="closed_form", alpha=2.0))
    state = SystemState(position=np.zeros(2))
    safety = SafetySample(h=1.0, grad_h=np.array([1.0, 0.0]))
    u = np.array([0.0, 0.5])
    r = box.filter_control(state, safety, u)
    assert np.allclose(r.u_safe, u)
    assert r.cbf_residual >= -1e-8


def test_velocity_infeasible_modified():
    box = CBFBox(CBFBoxConfig(mode="velocity", solver="closed_form", alpha=1.0))
    state = SystemState(position=np.zeros(2))
    safety = SafetySample(h=0.1, grad_h=np.array([1.0, 0.0]))
    u = np.array([-1.0, 0.0])
    r = box.filter_control(state, safety, u)
    assert r.was_filtered
    assert r.cbf_residual >= -1e-8
