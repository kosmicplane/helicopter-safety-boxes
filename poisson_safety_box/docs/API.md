# API

Use:

```python
from poisson_safety_box import PoissonSafetyBox, PoissonBoxConfig
result = PoissonSafetyBox(config).compute(occupancy)
```

Important result fields:

- `result.h`
- `result.grad_h`
- `result.hessian_h`
- `result.laplacian_h`
- `result.forcing`
- `result.free_mask`
- `result.boundary_mask`
- `result.solve_mask`
- `result.solver_info`
- `result.timing`

Saving:

```python
result.save_npz('result.npz')
result.save_summary_json('summary.json')
result.plot_all('figures')
```
