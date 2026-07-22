"""Basic velocity-CBF example.

This example creates a half-space CBF constraint that the nominal command
violates and shows how the CBF box projects it to safety.
"""

from pathlib import Path
import numpy as np
from cbf_safety_box import CBFBox, CBFBoxConfig, SystemState, SafetySample
from cbf_safety_box.visualization.feasible_set_plots import plot_feasible_halfspace_2d

out = Path("outputs/velocity_basic")
out.mkdir(parents=True, exist_ok=True)

config = CBFBoxConfig(mode="velocity", solver="closed_form", alpha=2.0)
state = SystemState(position=np.array([0.0, 0.0]))
safety = SafetySample(h=0.1, grad_h=np.array([1.0, 0.0]))
u_nom = np.array([-1.0, 0.4])

box = CBFBox(config)
result = box.filter_control(state, safety, u_nom)
print(result.to_dict())
result.save_json(out / "result.json")
plot_feasible_halfspace_2d(result.constraint_matrix[0], result.constraint_vector[0], result.u_nom, result.u_safe, out)
