# Contingency HJR API Extension

Added backward-compatible helpers:

- `build_active_target_reachability_constraint`: correctly signed positive-certificate HJ/CBF row with an active auxiliary variable;
- `lift_constraint_to_decision`: generic affine-row embedding into augmented decision vectors.

The existing CLF builder remains unchanged and must not be used as a substitute for the reachability builder. Existing Poisson-CBF, HOCBF, solver, and combinatorial APIs remain compatible.
