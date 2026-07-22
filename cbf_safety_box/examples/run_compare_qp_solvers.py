"""Compare QP solver backends for one velocity CBF instance."""

import numpy as np
from cbf_safety_box import CBFBox, CBFBoxConfig, SystemState, SafetySample

state = SystemState(position=np.array([0.0, 0.0]))
safety = SafetySample(h=0.1, grad_h=np.array([1.0, 0.5]))
u_nom = np.array([-1.0, -1.0])
for solver in ["closed_form", "scipy", "cvxpy"]:
    try:
        cfg = CBFBoxConfig(mode="velocity", solver=solver, alpha=2.0, control_lower_bound=[-2,-2], control_upper_bound=[2,2])
        result = CBFBox(cfg).filter_control(state, safety, u_nom)
        print(solver, result.to_dict())
    except Exception as e:
        print(solver, "not available or failed:", e)
