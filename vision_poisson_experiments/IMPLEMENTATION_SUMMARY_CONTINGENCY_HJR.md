# File-by-File Contingency HJR Implementation Summary

## New vision modules

- `common/mission_setup.py` — metric START/landing-zone data model, validation, interactive selection, JSON persistence, and overlay drawing.
- `common/target_manager.py` — zone states, perception hysteresis, latching, reachability status, certified switching, and event logging.
- `common/hj_reachability.py` — metric Dijkstra fields, HJ/Eikonal values, gradients, pivot/count fields, predecessor paths, and collision checks.
- `common/contingency_planner.py` — active shortest-path construction, line-of-sight simplification, and pure-pursuit nominal velocity.
- `common/unified_contingency_filter.py` — one augmented Poisson-CBF/HJ projection delegated to `cbf_safety_box`.
- `common/safety_synthesis.py` — latest-only synchronized Poisson/HJR worker with occupancy versions.
- `common/discrete_safety.py` — finite-step collision and Poisson-margin backtracking.
- `common/contingency_visualization.py` — live overlays and HJR snapshot figures.
- `common/contingency_live_pipeline.py` — optional live contingency orchestration and synchronized outputs.
- `common/synthetic_contingency_validation.py` — five deterministic end-to-end scenarios.
- `scripts/run_contingency_synthetic_validation.py` — validation CLI.

## Modified vision files

- `02_phone_stream_poisson_realtime/run_experiment.py` — selects the contingency pipeline only when reachability is enabled.
- `config.yaml` — interactive mission, HJR, planner, Poisson-CBF, and unified-filter defaults.
- `config_synthetic.yaml` — preserves the original baseline with reachability disabled.
- `config_contingency_synthetic.yaml` — repeatable headless local-video configuration.
- `README.md`, `IMPLEMENTATION_REPORT.md`, `docs/ARCHITECTURE.md`, and `docs/CONFIGURATION.md` — architecture, commands, scope, and limitations.
- Snapshot saving — geodesic/HJ arrays, pivot, reachable count, path, residual, timing, and version metadata.

## New and extended tests

- mission geometry and persistence;
- landing-zone hysteresis and certified switching;
- geodesic/HJ values, Eikonal diagnostics, pivot/count, and safe paths;
- unified CBF Safety Box integration;
- discrete-time safety backtracking;
- synchronized latest-only worker;
- local-video live contingency pipeline;
- five deterministic end-to-end scenarios.

## CBF Safety Box extension

- `build_active_target_reachability_constraint` — correctly signed active positive-certificate row.
- `lift_constraint_to_decision` — generic affine-row embedding for augmented decisions.
- exports, tests, API documentation, and changelog were updated.

## Poisson Safety Box

No numerical Poisson implementation was duplicated or replaced. The supplied package remains authoritative and unchanged in behavior.
