# poisson_safety_box

`poisson_safety_box` is a portable Python library that converts an occupancy
matrix into a Poisson safety function `h` and its derivatives.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run examples

```bash
python examples/run_basic_2d.py
python examples/run_basic_3d.py
python examples/run_compare_forcing.py
python examples/run_compare_solvers.py
```

## Public API

```python
import numpy as np
from poisson_safety_box import PoissonSafetyBox, PoissonBoxConfig

occupancy = np.zeros((80, 60, 40), dtype=bool)
occupancy[30:40, 20:30, 10:20] = True

config = PoissonBoxConfig(
    grid_spacing=(0.25, 0.25, 0.25),
    forcing_method="guidance",
    solver="sor",
    compute_gradient=True,
    compute_hessian=True,
)

result = PoissonSafetyBox(config).compute(occupancy)
h = result.h
grad_h = result.grad_h
hessian_h = result.hessian_h
```
