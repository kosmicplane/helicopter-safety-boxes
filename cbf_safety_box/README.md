# cbf_safety_box

`cbf_safety_box` is a portable Python library that converts local safety data
from a Poisson safety-function generator into safe control commands using
Control Barrier Function quadratic programs.

It **does not** compute Poisson fields, occupancy maps, or robot simulations.
It receives `h`, `grad_h`, and optionally `hessian_h` from another module, such
as `poisson_safety_box`, and returns `u_safe`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python examples/run_velocity_cbf_basic.py
python examples/run_acceleration_hocbf_basic.py
pytest -q
```

## Minimal API

```python
import numpy as np
from cbf_safety_box import CBFBox, CBFBoxConfig, SystemState, SafetySample

config = CBFBoxConfig(mode="velocity", solver="closed_form", alpha=3.0)
state = SystemState(position=np.array([0.0, 0.0]))
safety = SafetySample(h=0.2, grad_h=np.array([1.0, 0.0]))
u_nom = np.array([-1.0, 0.3])

result = CBFBox(config).filter_control(state, safety, u_nom)
print(result.u_safe)
```

## Augmented Poisson-CBF and HJ contingency projection

The package can also solve one augmented decision containing planar velocity and paper-inspired HJ auxiliary variables. `build_active_target_reachability_constraint` preserves a positive reachability certificate, `build_combinatorial_contingency_constraints` constructs r-out-of-p rows, and `lift_constraint_to_decision` embeds rows in a shared decision vector. The final solve is performed through `CBFBox.filter_affine_constraints` with optional Euclidean norm bounds.

These helpers do not compute HJ value functions. A caller supplies certificate values, drifts, and control gradients. The separate `vision_poisson_experiments` integration computes reduced single-integrator HJ/Eikonal fields and uses this package only for constraint construction and the unified online projection.
