# clf_safety_box

Reusable construction and evaluation of local Control Lyapunov Functions and CLF-certified attraction regions.

## Mathematical model

For a controlled equilibrium `j`, define

\[
e_j=x-x_j^\star,\qquad u=-K_je_j,
\]

and

\[
A_{cl,j}=A_j-B_jK_j.
\]

Given \(Q_j\succ0\), the package solves

\[
A_{cl,j}^TP_j+P_jA_{cl,j}=-Q_j
\]

and constructs

\[
V_j(x)=e_j^TP_je_j.
\]

The local attraction certificate is

\[
h_j^{ROA}(x)=c_j-V_j(x).
\]

## Capabilities

- generic control-affine model protocol;
- 2-D and 3-D single- and double-integrator models;
- continuous-time LQR construction or user-supplied gains;
- Lyapunov equation solve;
- closed-loop eigenvalue, symmetry, positive-definiteness, condition-number, and residual checks;
- vectorized multi-target `V`, gradient, Lie derivative, and ROA evaluation;
- linear, polynomial, and regularized finite-time class-K functions;
- analytic input-compatible ellipsoid threshold;
- active-target CLF affine row;
- persisted JSON and NPZ artifacts per target.

## Minimal use

```python
import numpy as np
from safety_box_core import DecisionLayout, EquilibriumTarget, StateSnapshot
from clf_safety_box import CLFBox, CLFBoxConfig, DoubleIntegratorModel

model = DoubleIntegratorModel(spatial_dimension=3)
box = CLFBox(
    CLFBoxConfig(
        control_lower=(-4.0, -4.0, -4.0),
        control_upper=(4.0, 4.0, 4.0),
    ),
    model,
)
targets = (
    EquilibriumTarget("LZ0", np.array([5., 3., 0.8, 0., 0., 0.]), np.zeros(3)),
)
box.prepare(targets)
evaluation = box.evaluate_many(StateSnapshot(np.array([1., 1., 4., 0., 0., 0.])))
layout = DecisionLayout.from_sizes(control=3, omega_contingency=1, delta_clf=1)
row = box.active_target_constraint(
    state=StateSnapshot(np.array([1., 1., 4., 0., 0., 0.])),
    target_id="LZ0",
    layout=layout,
)
```

See `examples/double_integrator_clf.py` and `docs/EQUATION_TO_CODE_MAP.md`.

## Scientific scope

A persisted CLF sublevel set is a certified inner approximation only under the dynamics, gain, input-bound, and local-validity assumptions recorded in its artifact. It is not an HJR backward reachable set.
