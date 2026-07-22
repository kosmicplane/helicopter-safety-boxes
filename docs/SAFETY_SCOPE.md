# Safety and Guarantee Scope

## Supported by the implemented reduced-order formulation

Subject to the stated smoothness, feasibility, continuous-time, and model assumptions, the implementation evaluates:

- the Poisson-HOCBF inequality for single- or double-integrator reduced-order dynamics;
- the active-target CLF decrease condition, including any explicitly logged CLF relaxation;
- membership in CLF sublevel regions of attraction;
- the `r`-out-of-`p` combinatorial certificate rows;
- affine-row and decision-bound residuals of the optimization result.

## Numerically demonstrated by the included experiments

- collision-free sampled rollouts in the tested worlds;
- successful landing under the configured tolerances;
- target invalidation and diversion to a certified alternative;
- Poisson solver residuals and field comparisons;
- filter and field-computation timing on the machine that generated the result;
- online processing of the included video and versioned field updates.

## Not established by this workspace alone

- full-order multirotor safety;
- guaranteed tracking of filtered acceleration or velocity by PX4;
- intersample safety for arbitrary control periods;
- robustness to unmodeled estimator, actuation, calibration, or communication error;
- hardware-flight guarantees;
- equivalence to Martian aerodynamics or gravity;
- a formal finite-horizon reach-avoid guarantee.

## Required path to full-order claims

A full-order claim requires an experimentally or analytically justified tracking-error bound between the reduced-order safe command and the actual vehicle behavior. The environmental certificate should then be robustified against this bound, estimator uncertainty, delay, and sampled-data effects. Until that step is completed, all formal statements must be scoped to the reduced-order model.
