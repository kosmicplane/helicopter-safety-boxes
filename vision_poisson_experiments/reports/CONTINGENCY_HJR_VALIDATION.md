# Contingency HJR Validation Report

## Implemented model

The deterministic validation uses the reduced planar model

\[
\dot p=u,\qquad \|u\|_2\le v_{\max},\qquad
V_j(p,\tau)=v_{\max}(-\tau)-D_j(p).
\]

`D_j` is an 8-connected metric geodesic distance over inflated occupancy. The Poisson-CBF row, active HJ row, combinatorial rows, velocity norm, and two auxiliary variables are solved in one call to `CBFBox.filter_affine_constraints`.

## Automated suites

| Repository | Result |
|---|---:|
| `poisson_safety_box` | **6 passed** |
| `cbf_safety_box` | **14 passed** |
| `vision_poisson_experiments` | **47 passed** |
| **Total** | **67 passed, 0 failed** |

## Five required scenarios

| Scenario | Reached | Final target | Switches | HOLD | Collision | Min reachable | Min pivot | Optimizer success [%] |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| clear_active | yes | LZ-1 | 0 | no | no | 4 | 2.742 | 100.000 |
| active_zone_blocked | yes | LZ-4 | 1 | no | no | 3 | 2.742 | 100.000 |
| corridor_blocked | yes | LZ-1 | 0 | no | no | 4 | 2.742 | 100.000 |
| contingency_lost | no | LZ-1 | 0 | CONTINGENCY REQUIREMENT LOST | no | 1 | n/a | 100.000 |
| camera_moved | no | LZ-1 | 0 | camera moved; metric map invalid | no | 4 | 2.742 | 100.000 |

### Expected outcomes

- `clear_active`: original target reached without switching.
- `active_zone_blocked`: LZ-1 is persistently blocked, rejected, and replaced by a certified alternative while at least two targets remain reachable.
- `corridor_blocked`: the path is recomputed around a new corridor obstacle without changing the target.
- `contingency_lost`: fewer than `r=2` targets remain, so the command becomes zero and no fallback target is selected.
- `camera_moved`: metric validity is lost and the marker enters HOLD.

Each scenario directory contains `summary.json`, `history.csv`, `target_switch_events.csv`, and `validation_dashboard.png`.

## Live local-video integration

The packaged 60-frame local-video run is stored under `sample_outputs/live_contingency_demo`.
It processed the live segmentation/occupancy stream with the synchronized latest-only worker and produced:

- queue maximum: 1;
- worker failures: 0;
- invalid solves: 0;
- synchronized Poisson/HJR snapshot outputs;
- one certified target switch;
- final target disk reached by the virtual marker;
- saved Poisson, HJ, geodesic, pivot, reachable-count, path, residual, timing, and dashboard artifacts.

## Scope

These results validate the software architecture and the reduced single-integrator certificate. They do not certify a full multicopter, touchdown dynamics, physical camera uncertainty, PX4 tracking, wind, or hardware behavior.
