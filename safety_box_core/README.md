# safety_box_core

Shared contracts and configuration utilities for independently reusable safety boxes.

## Core convention

Every affine certificate row uses one sign convention:

\[
A z \ge b.
\]

The augmented decision `z` is described by named blocks instead of manually managed column numbers.

```python
from safety_box_core import DecisionLayout

layout = DecisionLayout.from_sizes(
    control=3,
    omega_contingency=1,
    delta_clf=1,
)
```

## Main contracts

- `AffineConstraint`: immutable affine rows with source, equation ID, hardness, priority, and metadata.
- `ConstraintBundle`: output of one enabled or disabled box.
- `DecisionLayout`: named slices for augmented decisions.
- `StateSnapshot`: versioned state and time.
- `EquilibriumTarget`: reusable target/equilibrium contract.
- `CertificateEvaluation`: local derivative representation `dot h = drift + control_gradient @ u`.
- `FilterResult`: verified decision, residuals, status, timing, and diagnostics.

## Configuration

`load_experiment_config()` reads the single central YAML, applies one named profile, applies dotted command-line overrides, validates dependencies, and produces a deterministic configuration hash.

See `examples/contracts.py` and the root `docs/ARCHITECTURE.md`.
