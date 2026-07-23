# File Guide

This guide identifies the files a researcher normally edits, runs, or inspects.

## Configuration and worlds

| File | Use |
|---|---|
| `configs/experiment.yaml` | gains, solvers, bounds, targets, schedules, profiles, sweeps |
| `configs/worlds/mars_analog_landing.yaml` | obstacle geometry for the flagship 3-D experiment |

## Predefined-world experiment

| File | Use |
|---|---|
| `experiments/predefined_world/world.py` | loads and rasterizes analytic obstacles |
| `experiments/predefined_world/scenarios.py` | resolves baseline, single-failure, and sequential-failure event schedules |
| `experiments/predefined_world/run.py` | executes one named landing scenario |
| `experiments/predefined_world/run_sweeps.py` | HOCBF, CLF, and ROA sweeps on a fixed scenario |
| `experiments/predefined_world/run_paper_suite.py` | controlled scenario matrix, sweeps, and cross-scenario figure index |
| `experiments/predefined_world/paper_figures.py` | claim-specific Approach and Results figures |

## Shared experiment runtime

| File | Use |
|---|---|
| `experiments/common/cli.py` | configuration profiles, overrides, metadata |
| `experiments/common/controller.py` | evaluates boxes and assembles one control step |
| `experiments/common/nominal_planner.py` | A* route and lookahead nominal acceleration |
| `experiments/common/poisson_field.py` | solves and samples Poisson fields |
| `experiments/common/simulation.py` | failure events, integration, logging, terminal classification |
| `experiments/common/plotting.py` | reusable diagnostics and sweeps |

## Safety boxes

| File | Use |
|---|---|
| `safety_box_core/src/safety_box_core/types.py` | canonical immutable contracts |
| `safety_box_core/src/safety_box_core/config.py` | central configuration validation |
| `clf_safety_box/src/clf_safety_box/quadratic.py` | LQR, Lyapunov equation, ROA threshold |
| `clf_safety_box/src/clf_safety_box/box.py` | vectorized CLF/ROA evaluation |
| `contingency_safety_box/src/contingency_safety_box/box.py` | pivot, critical set, combinatorial rows |
| `safety_filter_box/src/safety_filter_box/filter.py` | unified problem and verification |
| `safety_filter_box/src/safety_filter_box/solvers.py` | optimization backends |
| `cbf_safety_box/cbf_safety_box/api.py` | shared HOCBF adapter |
| `poisson_safety_box/poisson_safety_box/solver.py` | Poisson solve backends |

## Vision modes

| File | Use |
|---|---|
| `experiments/static_image/run.py` | offline image experiment |
| `experiments/static_image/pipeline.py` | image-to-occupancy processing |
| `experiments/live_vision/run.py` | video/camera/IP-stream entry point |
| `experiments/live_vision/worker.py` | asynchronous Poisson worker |
| `experiments/live_vision/dashboard.py` | live panels and histories |

## Reproducibility scripts

| File | Use |
|---|---|
| `scripts/run_paper_experiments.sh` | complete paper experiment suite |
| `scripts/run_smoke_suite.sh` | fast end-to-end validation |
| `scripts/verify_clf_runtime.py` | confirms active new entry points do not import HJR |
| `run_checks.sh` | tests, imports, and static compilation |

## Generated data

Never edit generated files to change an experiment. Re-run from the configuration instead.

```text
outputs/<run>/effective_config.yaml
outputs/<run>/run_metadata.json
outputs/<run>/data/*.csv
outputs/<run>/data/*.json
outputs/<run>/clf_artifacts/*
outputs/<run>/figures/*
```
