"""Construct and evaluate a 2-D double-integrator landing CLF."""
from __future__ import annotations

import numpy as np
from safety_box_core import EquilibriumTarget, StateSnapshot
from clf_safety_box import CLFBox, CLFBoxConfig, DoubleIntegratorModel

model = DoubleIntegratorModel(spatial_dimension=2)
config = CLFBoxConfig(
    control_lower=(-2.0, -2.0),
    control_upper=(2.0, 2.0),
    roa_fraction=0.9,
)
box = CLFBox(config, model)
target = EquilibriumTarget(
    identifier="landing_zone",
    x_star=np.array([4.0, 3.0, 0.0, 0.0]),
    u_star=np.zeros(2),
)
artifacts = box.prepare((target,))
state = StateSnapshot(np.array([1.0, 1.0, 0.2, -0.1]))
result = box.evaluate_many(state)[0]
print("closed-loop eigenvalues:", artifacts[0].closed_loop_eigenvalues)
print("V:", result.V)
print("ROA margin c-V:", result.roa_margin)
print("inside ROA:", result.inside_roa)
