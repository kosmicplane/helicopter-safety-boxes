# Original Workspace Preservation

This release deliberately preserves the complete original safety-box environment supplied in `Helicopter.zip`.

## Preserved full packages

### `cbf_safety_box`

The original package remains a general reusable CBF/HOCBF library. It still contains:

- velocity-level CBFs;
- acceleration-level HOCBFs;
- backstepping constraints;
- control-limit constraints;
- original contingency and reachability-related historical modules;
- single- and double-integrator models;
- closed-form, active-set, SciPy, and CVXPY optimization code;
- feasibility checks;
- diagnostics and timing utilities;
- trajectory, residual, feasible-set, command, and QP visualization tools;
- examples, configuration files, tests, and documentation.

The new multi-certificate architecture adds a backwards-compatible adapter that exports constraints through the canonical `A z >= b` contract. It does not replace the full package with a minimal implementation.

### `poisson_safety_box`

The original package remains intact as the reusable occupancy-to-safety-field implementation, including:

- domain and boundary construction;
- constant, distance, average-flux, and guidance forcing;
- sparse direct, conjugate-gradient, and SOR solvers;
- field derivatives and interpolation;
- numerical diagnostics;
- original examples, tests, documentation, and outputs.

### `vision_poisson_experiments`

The original static-image and phone/video perception experiments remain available with their documentation, sample outputs, scripts, reports, and tests. Historical HJR experiments remain reproducible as legacy material.

## Added packages

The following additions are independent and reusable:

- `safety_box_core`: canonical immutable contracts;
- `clf_safety_box`: CLF synthesis, evaluation, and ROA artifacts;
- `contingency_safety_box`: generic `r`-out-of-`p` certificate composition;
- `safety_filter_box`: verified affine multi-certificate optimization;
- `experiments`: unified predefined-world, static-image, and live-video studies.

## Active methodology

The new paper experiments use CLF-certified regions of attraction:

```math
V_j(x)=(x-x_j^\star)^\top P_j(x-x_j^\star),
\qquad h_j^{\mathrm{ROA}}(x)=c_j-V_j(x),
```

and the `r`-th-largest pivot. HJR is not imported by these entry points. Preserving HJR source files does not make them part of the active methodology.

## Release-build preservation check

The final release builder compared every meaningful file in the original archive against the release tree. Build caches (`__pycache__`, `.pytest_cache`, `*.egg-info`, and compiled bytecode) were intentionally excluded. The result was:

```text
meaningful original files missing: 0
historical HJR files relocated under legacy/hjr: 51
```
