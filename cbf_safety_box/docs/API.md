# API

Main objects:

- `CBFBoxConfig`
- `SystemState`
- `SafetySample`
- `CBFBox`
- `CBFBoxResult`

Call:

```python
result = CBFBox(config).filter_control(state, safety, u_nom)
```

The result contains `u_safe`, `u_nom`, residuals, solver status, timing, and diagnostics.

## Reachability and augmented-decision helpers

`build_active_target_reachability_constraint(...)` builds the affine row

\[
\dot V \ge -\alpha V-\omega_a\operatorname{ReLU}(-\alpha V)
\]

using the package convention `A z >= b`. It is intentionally distinct from the
CLF builder because a positive reachability certificate must be kept nonnegative,
whereas a CLF is driven downward.

`lift_constraint_to_decision(...)` embeds an existing affine row in an arbitrary
augmented decision vector without changing its bound or nonzero coefficients.
This is used by the live contingency experiment to combine planar Poisson CBF,
active HJ, and combinatorial HJ rows in one optimization.
