# Validation Report

This report records commands and results actually executed in the delivered sibling workspace. The static CBF
experiment is a reduced-order numerical validation. The live pipeline is a sequence of quasi-static Poisson solves,
not a formal moving-obstacle certificate.

## Workspace layout

```text
Docker/Workspace/
├── poisson_safety_box/
├── cbf_safety_box/
└── vision_poisson_experiments/
```

The layout checker and editable-install import smoke test both completed successfully.

## Automated tests

| Test suite | Result |
|---|---:|
| Original `poisson_safety_box` tests | **6 passed** |
| `cbf_safety_box` tests | **14 passed** |
| `vision_poisson_experiments` tests | **47 passed** |
| **Total** | **67 passed, 0 failed** |

## Static image experiment

```bash
python 01_static_image_poisson_cbf/run_experiment.py \
  --image examples/assets/static_scene.png \
  --config 01_static_image_poisson_cbf/config_synthetic.yaml \
  --output sample_outputs/static_demo \
  --assume-top-down --headless --verbose
```

Wall time: **0:36.38**; maximum resident set: **1,003,252 KiB**;
artifacts: **173 files** (15.61 MiB).

| Forcing | Valid | Wall time [s] | exact max `|A h-b|` | boundary max `|h|` |
|---|---:|---:|---:|---:|
| `constant` | `true` | 0.048940 | 2.416e-13 | 0.000e+00 |
| `distance` | `true` | 0.043857 | 1.958e-13 | 0.000e+00 |
| `average_flux` | `true` | 0.043119 | 6.875e-13 | 0.000e+00 |
| `guidance` | `true` | 0.139564 | 6.814e-14 | 0.000e+00 |

```text
nominal controller: collision_with_inflated_occupancy
CBF-filtered controller: goal_reached
minimum sampled h: 0.0356621430003
minimum explicit CBF residual: -2.06432093641e-16
```

## Poisson solver comparison

| Solver | Valid | Wall time [s] | exact max `|A h-b|` | boundary max `|h|` |
|---|---:|---:|---:|---:|
| `sparse_direct` | `true` | 0.042141 | 2.416e-13 | 0.000e+00 |
| `conjugate_gradient` | `true` | 0.046019 | 3.881e-07 | 0.000e+00 |
| `sor` | `true` | 0.076355 | 5.445e-07 | 0.000e+00 |

## Live video experiment

```bash
python 02_phone_stream_poisson_realtime/run_experiment.py \
  --source examples/assets/live_scene.avi \
  --config 02_phone_stream_poisson_realtime/config_synthetic.yaml \
  --output sample_outputs/live_demo \
  --headless --max-frames 60 --verbose
```

Wall time: **0:08.42**; maximum resident set: **1,047,596 KiB**;
artifacts: **22 files** (0.63 MiB).

| Live metric | Mean | Median | p95 | Maximum |
|---|---:|---:|---:|---:|
| Display FPS | 15.469 | 16.359 | 19.760 | 27.004 |
| Poisson updates/s | 14.621 | 14.831 | 16.335 | 16.517 |
| Poisson solve latency [ms] | 36.388 | 30.527 | 37.823 | 315.473 |
| Field update latency [ms] | 62.330 | 54.035 | 69.575 | 422.086 |
| Frame pipeline latency [ms] | 71.868 | 60.399 | 90.141 | 672.160 |
| Field age [ms] | 75.495 | 61.537 | 94.029 | 429.634 |

```text
frames: 60
accepted solves: 59
queue maximum observed: 1
discarded queued tasks: 0
discarded obsolete solves: 0
failed solves: 0
invalid solves: 0
warnings: 0
```

## Final logs

```text
reports/validation/logs/poisson_safety_box_pytest.txt
reports/validation/logs/cbf_safety_box_pytest.txt
reports/validation/logs/vision_poisson_experiments_pytest.txt
reports/validation/logs/editable_install_check.txt
reports/validation/logs/static_demo_stdout.txt
reports/validation/logs/static_demo_stderr.txt
reports/validation/logs/static_demo_time.txt
reports/validation/logs/live_demo_stdout.txt
reports/validation/logs/live_demo_stderr.txt
reports/validation/logs/live_demo_time.txt
```

See `IMPLEMENTATION_REPORT.md` for architecture, limitations, and the ROS 2/Gazebo/PX4 roadmap.

## Contingency HJR extension

The optional contingency mode was validated with five deterministic scenarios and a 60-frame local-video run. See `CONTINGENCY_HJR_VALIDATION.md` and `reports/contingency_validation/`. The local-video worker retained a maximum queue size of one, had no worker failures, and saved synchronized Poisson/HJR fields with matching occupancy versions.
