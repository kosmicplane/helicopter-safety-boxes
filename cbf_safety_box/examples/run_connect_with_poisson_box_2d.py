"""Example showing how CBFBox connects to Poisson-like safety data in 2D.

This example avoids a hard dependency on poisson_safety_box by constructing an
analytic safety sample. Replace the SafetySample construction with an
interpolated Poisson result in your larger project.
"""

import numpy as np
from cbf_safety_box import CBFBox, CBFBoxConfig, SystemState, SafetySample

# Suppose Poisson box provided h(p), grad_h(p), Hessian(p) here.
safety = SafetySample(h=0.25, grad_h=np.array([0.8, 0.3]), hessian_h=np.eye(2) * 0.05)
state = SystemState(position=np.array([2.0, 1.0]))
u_nom = np.array([-1.0, -0.8])
result = CBFBox(CBFBoxConfig(mode="velocity", solver="closed_form", alpha=2.5)).filter_control(state, safety, u_nom)
print("u_nom:", result.u_nom)
print("u_safe:", result.u_safe)
print("residual:", result.cbf_residual)
