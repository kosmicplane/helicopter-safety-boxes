# safety_filter_box

Unified minimum-intervention optimization over independently generated certificate bundles.

## Optimization

For nominal augmented decision \(z_{nom}\), diagonal weights \(W\succ0\), and affine rows \(Az\ge b\), the box solves

\[
\min_z\ \frac12(z-z_{nom})^TW(z-z_{nom})
\]

subject to all enabled hard and explicitly modeled soft constraints and decision bounds.

## Backends

- `hildreth`: warm-started deterministic dual-coordinate projection solver for small dense affine QPs;
- `scipy_slsqp`: portable reference backend.

Both backends are followed by solver-independent checks of:

- every affine residual;
- every decision bound;
- active-constraint identity;
- finite numerical values.

The filter never repairs an infeasible solution by post-solve clipping.

## Priority structure used by the landing experiments

1. acceleration/input limits — hard;
2. Poisson CBF/HOCBF — hard;
3. combinatorial ROA rows — hard;
4. active-target CLF — standard explicit `delta_clf` relaxation when enabled;
5. nominal performance — objective only.

The CLF relaxation is heavily penalized and logged. It does not relax environmental or contingency rows.

See `examples/affine_projection.py` and `docs/EQUATION_TO_CODE_MAP.md`.
