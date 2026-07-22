# Architecture

## Design objective

The workspace is organized as a collection of reusable safety boxes connected by immutable data contracts. Each box owns one mathematical responsibility and can be enabled, disabled, tested, and reused independently. The three experiment modes share the same certificate and optimization path; only the source of geometry changes.

```text
predefined world | static image | video/camera
                    |
                    v
             occupancy representation
                    |
                    v
            poisson_safety_box
              h_P, D h_P, D^2 h_P
                    |
                    v
             cbf_safety_box
          environmental CBF/HOCBF row
                    |
nominal planner -->+-----------------------------+
                    |                             |
                    v                             v
              clf_safety_box            contingency_safety_box
             V_j, P_j, K_j, c_j       h_j=c_j-V_j, r-out-of-p
                    |                             |
                    +--------------+--------------+
                                   v
                         safety_filter_box
                    verified minimum intervention
                                   |
                                   v
                               u_safe
```

## Package responsibilities

### `poisson_safety_box`

The complete original Poisson package is retained. It constructs the free domain and Dirichlet frontier from occupancy, builds configurable forcing fields, solves the discrete Poisson system, evaluates algebraic residuals, computes derivatives, and exposes local safety samples.

### `cbf_safety_box`

The complete original CBF/HOCBF package is retained, including its models, constraints, optimization backends, diagnostics, visualization, examples, and tests. A backwards-compatible adapter was added so it can emit the shared affine constraint contract

\[
A z \ge b.
\]

The adapter does not replace the original API.

### `clf_safety_box`

This package constructs target-specific local quadratic CLFs. For each equilibrium, it computes a stabilizing feedback gain, solves the continuous Lyapunov equation, verifies the closed-loop eigenvalues and Lyapunov residual, derives an input-compatible sublevel threshold, evaluates all targets in a vectorized batch, and emits an active-target CLF row.

### `contingency_safety_box`

This package is independent of CLF synthesis. It consumes generic differentiable certificates, computes the `r`-th-largest pivot, identifies critical certificates, evaluates the certified-alternative count, and emits the smooth combinatorial rows. The default landing application supplies CLF-derived ROA certificates.

### `safety_filter_box`

This package solves the unified minimum-intervention optimization. It consumes affine rows from any enabled box, maintains named decision blocks, checks every residual independently of solver status, and returns typed `READY`, `HOLD`, `INFEASIBLE`, or `ERROR` results. Hildreth dual-coordinate and SciPy SLSQP backends are available.

### `safety_box_core`

This package defines the canonical immutable contracts: `AffineConstraint`, `ConstraintBundle`, `DecisionLayout`, `StateSnapshot`, `EquilibriumTarget`, `CertificateEvaluation`, and `FilterResult`. It also owns the central configuration loader and validation.

## Experiment modes

All three modes call the same CLF, contingency, HOCBF, and filter APIs.

1. `experiments/predefined_world`: an analytically defined 3-D obstacle world for controlled validation and paper ablations.
2. `experiments/static_image`: one image or mask is converted to metric occupancy and evaluated offline.
3. `experiments/live_vision`: a video, camera, or OpenCV-compatible stream updates occupancy and Poisson fields asynchronously while the controller and dashboard run online.

## Nominal planning and formal certificates

The A*–PD planner is a nominal behavior generator. It uses occupancy and clearance to avoid obvious dead ends and supplies `u_nominal`. It is not called a reachability or safety certificate. Formal environmental safety is supplied by the Poisson-HOCBF row; convergence is supplied by the CLF row; contingency is supplied by the combinatorial ROA rows.

## Active versus legacy functionality

The complete original vision workspace is preserved because it supports other applications. Its historical HJR modules remain available for reproducibility, but the new entry points under `experiments/` do not import or execute them. The active landing methodology uses CLF sublevel sets and combinatorial stabilization.
