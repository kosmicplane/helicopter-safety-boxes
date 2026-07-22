"""Example showing how CBFBox connects to Poisson-like safety data in 3D."""

import numpy as np
from cbf_safety_box import CBFBox, CBFBoxConfig, SystemState, SafetySample

safety = SafetySample(h=0.3, grad_h=np.array([0.2, -0.1, 1.0]), hessian_h=np.eye(3) * 0.02)
state = SystemState(position=np.array([2.0, 1.0, 4.0]), velocity=np.array([0.3, 0.0, -1.2]))
a_nom = np.array([0.0, 0.0, -2.0])
cfg = CBFBoxConfig(mode="acceleration", solver="sor", alpha1=2.0, alpha2=2.0, control_lower_bound=[-3,-3,-3], control_upper_bound=[3,3,3])
result = CBFBox(cfg).filter_control(state, safety, a_nom)
print(result.to_dict())
