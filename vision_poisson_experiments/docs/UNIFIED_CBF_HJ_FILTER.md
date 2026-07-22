# Unified Poisson-CBF and HJ Contingency Filter

## Authority boundary

`poisson_safety_box` computes the numerical Poisson field. `cbf_safety_box` builds the Poisson CBF row, builds the combinatorial contingency rows, and solves the one augmented online projection. The vision repository does not contain a duplicate optimizer or a second Poisson-CBF implementation.

## Decision

\[
z=[u_x,u_y,\omega_a,\omega_c]^T.
\]

The nominal decision is `[u_nom_x, u_nom_y, 0, 0]`. Positive weights penalize both paper-inspired auxiliary variables.

## Poisson row

With `h_eff = h_P - h_margin`, the CBF Safety Box builds

\[
\nabla h_P^T u\ge-\alpha_P h_{eff}.
\]

The row is lifted to four decision columns with zero auxiliary coefficients, so neither HJ relaxation can weaken local collision safety.

## Active target row

For the active target,

\[
\dot V_a=\nabla V_a^T u-v_{\max}\dot\tau_a.
\]

The reachability-specific builder enforces

\[
\dot V_a\ge-\alpha_aV_a-\omega_a\operatorname{ReLU}(-\alpha_aV_a).
\]

This builder is distinct from a CLF-decrease builder; their sign semantics are not interchangeable.

## Contingency rows

For each available target certificate, the CBF Box combinatorial builder enforces the paper-inspired r-out-of-p rows using a shared `omega_c` and

\[
\rho(s)=k_\rho s^2.
\]

The helper also computes the r-th-largest pivot.

## Solver call

All rows, auxiliary bounds, and the Euclidean velocity norm are passed to `CBFBox.filter_affine_constraints`. Because of the norm constraint, the numerical problem is best described as a convex norm-constrained quadratic projection or QCQP-like problem rather than a strictly linear-constraint QP.

## Acceptance

A result is used only when the solver status is acceptable, all returned values are finite, every affine residual is above negative tolerance, and the norm constraint is satisfied. Failure produces `[0,0]` and HOLD. Global emergency slack is disabled by default; only `omega_active` and `omega_contingency` are planned relaxations.
