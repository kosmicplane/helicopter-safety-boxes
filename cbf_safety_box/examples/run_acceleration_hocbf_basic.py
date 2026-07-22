"""Basic acceleration-HOCBF example."""

from pathlib import Path
import numpy as np
from cbf_safety_box import CBFBox, CBFBoxConfig, SystemState, SafetySample

out = Path("outputs/acceleration_basic")
out.mkdir(parents=True, exist_ok=True)

config = CBFBoxConfig(mode="acceleration", solver="scipy", alpha1=2.0, alpha2=2.0, control_lower_bound=[-5, -5], control_upper_bound=[5, 5])
state = SystemState(position=np.array([0.0, 0.0]), velocity=np.array([-0.6, 0.2]))
safety = SafetySample(h=0.15, grad_h=np.array([1.0, 0.0]), hessian_h=np.eye(2) * 0.1)
a_nom = np.array([-2.0, 0.0])

box = CBFBox(config)
result = box.filter_control(state, safety, a_nom)
print(result.to_dict())
result.save_json(out / "result.json")
