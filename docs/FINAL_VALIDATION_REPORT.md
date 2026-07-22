# Final Validation Report

**Release date:** 2026-07-22  
**Workspace:** `Helicopter`  
**Active methodology:** Poisson safety functions + CBF/HOCBF + CLF-certified regions of attraction + `r`-out-of-`p` contingency

## Preservation policy

The release was constructed from the original `Helicopter.zip`, not from a reduced reimplementation. The complete reusable source trees for `poisson_safety_box`, `cbf_safety_box`, and `vision_poisson_experiments` remain present. The CBF package retains its original models, constraints, optimizers, diagnostics, visualizations, examples, documentation, and tests. Backwards-compatible adapters were added; original public functionality was not replaced by a minimal box.

The new CLF/ROA runtime is separated into independently installable packages:

- `safety_box_core`
- `clf_safety_box`
- `contingency_safety_box`
- `safety_filter_box`

Historical HJR code is preserved for reproducibility, but the new CLF/ROA experiment entry points do not import it.

## Commands executed

```bash
source scripts/workspace_env.sh
./run_checks.sh

python experiments/predefined_world/run.py \
  --profile smoke \
  --output outputs/reference_results/predefined_world

python experiments/predefined_world/run.py \
  --profile smoke \
  --compare \
  --output outputs/reference_results/predefined_comparison

python experiments/predefined_world/run_sweeps.py \
  --profile smoke \
  --output outputs/reference_results/parameter_sweeps

python experiments/static_image/run.py \
  --profile smoke \
  --image experiments/static_image/input/example_scene.png \
  --output outputs/reference_results/static_image

python experiments/static_image/run.py \
  --profile smoke \
  --image experiments/static_image/input/example_scene.png \
  --compare \
  --output outputs/reference_results/static_comparison

python experiments/live_vision/run.py \
  --profile smoke \
  --source experiments/live_vision/assets/example_stream.avi \
  --output outputs/reference_results/live_vision
```

## Automated checks

| Check | Result |
|---|---:|
| Active workspace, original CBF, and original Poisson tests | **26 passed** |
| Preserved original vision tests used by the active regression command | **42 passed** |
| Python compilation | Passed |
| Active CLF runtime import scan | **42 Python files checked; passed** |
| HJR imports in new CLF/ROA entry points | None found |
| Post-solve affine residual checks | Enabled and tested |
| ZIP integrity | Checked during release packaging |

The historical five-scenario HJR end-to-end test module remains available as an opt-in legacy regression. It is excluded from the default active validation because HJR is no longer the selected research methodology and that historical suite has very long teardown/runtime behavior in the current Python environment.

## Predefined 3-D reference result

Source: `outputs/reference_results/predefined_world/data/summary_full_failure.json`

| Metric | Result |
|---|---:|
| Landed | `true` |
| Initial target | `LZ0` |
| Final target after failure | `LZ2` |
| Target failure detected | `true` |
| Certified target switch | `true` |
| Steps | 362 |
| Simulated duration | 21.72 s |
| Minimum Poisson safety value | 0.0258716 |
| Minimum HOCBF residual | 3.55e-11 |
| Minimum active CLF residual | 8.02e-08 |
| Minimum `r`-out-of-`p` pivot | 1880.64 |
| Mean control intervention | 0.0173053 |
| Mean filter time | 0.139 ms |
| Filter p95 | 0.600 ms |
| Collision-guard backtracks | 0 |

## Static-image reference result

Source: `outputs/reference_results/static_image/data/summary_static_image_failure.json`

| Metric | Result |
|---|---:|
| Landed | `true` |
| Initial target | `LZ0` |
| Final target after failure | `LZ3` |
| Target failure detected | `true` |
| Certified target switch | `true` |
| Steps | 504 |
| Simulated duration | 22.68 s |
| Minimum HOCBF residual | -8.85e-06 |
| Minimum active CLF residual | 1.38e-08 |
| Minimum contingency pivot | 1538.49 |
| Mean filter time | 0.334 ms |
| Filter p95 | 0.566 ms |

The static-image HOCBF residual is within the experiment's configured numerical feasibility tolerance of `1e-5`.

## Live-video reference result

Source: `outputs/reference_results/live_vision/data/summary.json`

| Metric | Result |
|---|---:|
| Frames processed | 60 |
| Video duration | 3.0 s |
| Target failure detected | `true` |
| Certified target switch | `true` |
| Final target | `LZ2` |
| Minimum Poisson safety value | 0.110692 |
| Minimum HOCBF residual | 0.238178 |
| Minimum contingency pivot | 1538.49 |
| Mean filter time | 0.0860 ms |

The final dashboard was visually checked after reserving independent title, legend, and plotting bands. Numerical text does not overlap the camera image or plot titles.

## Poisson solver reference comparison

Source: `outputs/reference_results/predefined_comparison/data/poisson_solver_comparison.csv`

| Solver | Total wall time [s] | Algebraic RMS residual | Relative field error vs. sparse direct | Status |
|---|---:|---:|---:|---|
| Sparse direct | 0.0659 | 2.82e-15 | 0 | converged |
| Conjugate gradient | 0.0532 | 9.56e-08 | 3.45e-08 | converged |
| SOR | 0.0178 | 6.48e-06 | 2.22e-06 | converged |

The comparison distinguishes the exact assembled-system residual `A h - b` from the reconstructed finite-difference Laplacian diagnostic.

## Figure review

The following reference figures were generated and visually inspected:

- 3-D world trajectory;
- XY, XZ, and YZ trajectory projections;
- occupancy and Dirichlet-boundary slices;
- Poisson field slices in all coordinate planes;
- three-dimensional Poisson isosurfaces;
- forcing-field and numerical-Laplacian diagnostics;
- forcing-method trajectory and metric comparisons;
- Poisson-solver timing, residual, and field-error comparison;
- CLF region-of-attraction projections and principal axes;
- Lyapunov phase portraits and vector fields;
- `r`-out-of-`p` contingency maps;
- synchronized certificate histories;
- integrated research dashboard;
- live video dashboard and final live certificate figures.

## Scope of the result

These checks establish that the reduced-order, offline, static-image, and video pathways execute reproducibly in the tested environment. They do not establish a full-order PX4, hardware, tracking-error, or Martian-aerodynamics safety theorem. Formal and empirical claims are separated in `docs/SAFETY_SCOPE.md`.
