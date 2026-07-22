# Implementation Report

## Delivered sibling workspace

The implementation is placed in the requested sibling layout:

```text
Docker/Workspace/
├── poisson_safety_box/             # authoritative Poisson PDE package
├── cbf_safety_box/                 # authoritative CBF/HOCBF-QP package
├── validation_logs/                # final command transcripts
└── vision_poisson_experiments/     # delivered experiment repository
```

`vision_poisson_experiments` imports the two Safety Boxes as sibling dependencies. It does not duplicate their
Poisson discretization, forcing implementation, linear solvers, CBF/HOCBF equations, or QP solvers. Python source
comments, docstrings, identifiers, log messages, and YAML comments in the delivered repository are written in English.

## Implemented experiments

### `01_static_image_poisson_cbf`

The static experiment implements the full data chain:

```text
2D image
→ planar calibration / homography
→ obstacle segmentation
→ Boolean occupancy (`True = occupied`)
→ physical configuration-space inflation
→ Ω and ∂Ω
→ Poisson forcing and solve
→ h, ∇h, ∇²h, Δh, exact residual validation
→ bilinear physical-coordinate SafetySample
→ nominal versus CBF-filtered single-integrator simulation
```

Supported calibration and segmentation modes include top-down identity mapping, four-point interactive homography,
loaded calibration, mask file, manually drawn polygons, HSV segmentation, and frozen-background differencing. The
static run compares all four Safety Box forcing methods and all three Poisson solvers, saves the underlying matrices,
and renders the requested scientific diagnostics.

### `02_phone_stream_poisson_realtime`

The live experiment accepts a webcam index, local video, HTTP/MJPEG URL, or RTSP URL through `cv2.VideoCapture`.
It uses a responsive capture/perception loop and one independent Poisson worker. The queue has capacity one and
implements a latest-item-only policy; every task and result has a monotonic version identifier, so stale work cannot
replace a newer valid field.

The dashboard reports display rate and Poisson update rate separately. It records segmentation, occupancy, solve,
field-update, total pipeline, and field-age latency; applies temporal occupancy filtering; optionally detects global
camera motion; preserves the last valid field with an explicit staleness age; and can save synchronized 2D and low-rate
3D snapshots. The optional virtual CBF diagnostic never commands hardware.

## Main modules

| Module | Responsibility |
|---|---|
| `common/calibration.py` | Four-point homography, reusable calibration, rectification, fixed-camera motion diagnostics. |
| `common/segmentation.py` | Mask-file, polygon, HSV, frozen-background segmentation, cleanup, and manual correction. |
| `common/occupancy.py` | Strict Boolean occupancy, metric inflation, temporal filters, and occupancy-change metrics. |
| `common/coordinates.py` | Centralized `(row, col)=(y,x)` ↔ physical `(x,y)` conversion and bilinear field sampling. |
| `common/poisson_runner.py` | Safety Box configuration, execution, exact assembled-system validation, comparison, serialization. |
| `common/poisson_visualization.py` | Full static diagnostics, solver/forcing comparisons, live dashboard, and 3D snapshots. |
| `common/cbf_demo.py` | Numerical Poisson-to-CBF connection and nominal/filtered trajectory comparison. |
| `common/static_pipeline.py` | End-to-end static orchestration and reproducible artifact generation. |
| `common/live_pipeline.py` | Capture, perception, size-one queue, worker, validity/staleness policy, UI, snapshots, metrics. |
| `common/metrics.py` | Thread-safe measurements, rates, percentiles, CSV, and JSON summaries. |
| `common/external_boxes.py` | Discovery of the two sibling Safety Box repositories. |
| `common/io_utils.py` | Paths, YAML/JSON/CSV helpers, timestamps, and logging. |

## Coordinate and numerical integrity

The central adapter enforces:

```text
physical position [x, y]       → interpolation position [y, x]
raw gradient [dh/dy, dh/dx]    → physical gradient [dh/dx, dh/dy]
H_xy                           = P @ H_yx @ P.T
```

Obstacle masks use nearest-neighbor resizing to preserve binarity. `h`, gradient, and Hessian samples use bilinear
interpolation. The CBF receives raw numerical field values; color normalization is confined to display functions.
The load-bearing PDE validation reconstructs the exact sparse system assembled by the Poisson Safety Box and checks
`A h - b`, rather than treating the separate `numpy.gradient` Laplacian diagnostic as the algebraic solver residual.

## Executed automated validation

| Test suite | Result |
|---|---:|
| Original `poisson_safety_box` tests | **6 passed** |
| Original `cbf_safety_box` tests | **9 passed** |
| `vision_poisson_experiments` integration tests | **9 passed** |
| **Total** | **24 passed, 0 failed** |

The editable-install smoke test also completed successfully for all three sibling repositories. The new tests cover
coordinate and Hessian permutations, directional derivatives, image-to-occupancy alignment, physical inflation,
Poisson boundary/residual consistency, a real numerical Poisson-to-CBF connection, nominal collision versus CBF
avoidance, headless live-video processing, size-one queue behavior, output creation, clean shutdown, and a bounded-
memory smoke check.

## Executed static demonstration

```bash
python 01_static_image_poisson_cbf/run_experiment.py \
  --image examples/assets/static_scene.png \
  --config 01_static_image_poisson_cbf/config_synthetic.yaml \
  --output sample_outputs/static_demo \
  --assume-top-down --headless --verbose
```

Measured wall time: **0:36.38**. Maximum resident set: **1,003,252 KiB**.
The run generated **173 files** (15.61 MiB).

### Occupancy

| Quantity | Value |
|---|---:|
| Workspace | `6.0 m × 4.5 m` |
| Poisson grid | `96 × 72` (`nx × ny`) |
| Uninflated occupied cells | `749` |
| Inflated occupied cells | `1634` |
| Physical inflation radius | `0.2 m` |
| Inflation in grid cells | `4 x-cells`, `4 y-cells` |

### Poisson forcing comparison

| Forcing | Valid | Wall time [s] | exact max `|A h-b|` | max boundary `|h|` |
|---|---:|---:|---:|---:|
| `constant` | `true` | 0.048940 | 2.416e-13 | 0.000e+00 |
| `distance` | `true` | 0.043857 | 1.958e-13 | 0.000e+00 |
| `average_flux` | `true` | 0.043119 | 6.875e-13 | 0.000e+00 |
| `guidance` | `true` | 0.139564 | 6.814e-14 | 0.000e+00 |

All four fields and requested derivatives were finite, all Dirichlet boundaries were zero to numerical precision,
and all exact residuals passed the configured tolerance.

### CBF outcome

| Quantity | Result |
|---|---:|
| Start | `[0.4, 3.8]` m |
| Goal | `[5.6, 0.5]` m |
| Nominal status | `collision_with_inflated_occupancy` |
| CBF-filtered status | `goal_reached` |
| Nominal collision | `True` |
| CBF collision | `False` |
| CBF reached goal | `True` |
| Minimum sampled `h` | `0.0356621430003` |
| Minimum explicit CBF residual | `-2.06432093641e-16` |

The nominal straight-line controller entered the inflated occupancy. The CBF-filtered trajectory reached the goal
without an occupied-state transition. The logged residual `grad_h.T @ u_safe + alpha*h` remained nonnegative up to
floating-point tolerance.

### Identical-input solver comparison

| Solver | Valid | Wall time [s] | exact max `|A h-b|` | max boundary `|h|` |
|---|---:|---:|---:|---:|
| `sparse_direct` | `true` | 0.042141 | 2.416e-13 | 0.000e+00 |
| `conjugate_gradient` | `true` | 0.046019 | 3.881e-07 | 0.000e+00 |
| `sor` | `true` | 0.076355 | 5.445e-07 | 0.000e+00 |

These are measurements for the delivered synthetic grid in this container, not universal solver benchmarks.

## Executed live video demonstration

```bash
python 02_phone_stream_poisson_realtime/run_experiment.py \
  --source examples/assets/live_scene.avi \
  --config 02_phone_stream_poisson_realtime/config_synthetic.yaml \
  --output sample_outputs/live_demo \
  --headless --max-frames 60 --verbose
```

Measured wall time: **0:08.42**. Maximum resident set: **1,047,596 KiB**.
The run generated **22 files** (0.63 MiB), including one synchronized 3D snapshot.

| Live metric | Mean | Median | p95 | Maximum |
|---|---:|---:|---:|---:|
| Display FPS | 15.469 | 16.359 | 19.760 | 27.004 |
| Poisson updates/s | 14.621 | 14.831 | 16.335 | 16.517 |
| Poisson solve latency [ms] | 36.388 | 30.527 | 37.823 | 315.473 |
| Field update latency [ms] | 62.330 | 54.035 | 69.575 | 422.086 |
| Frame pipeline latency [ms] | 71.868 | 60.399 | 90.141 | 672.160 |
| Field age [ms] | 75.495 | 61.537 | 94.029 | 429.634 |

```text
frames processed: 60
accepted solves: 59
maximum queue size observed: 1
discarded queued tasks: 0
discarded obsolete solves: 0
failed solves: 0
invalid solves: 0
warnings: 0
```

This validates the file-source and asynchronous runtime paths. A physical phone URL, webcam device, interactive GUI,
and hardware command path were not available in the container and are not represented as hardware-tested.

## Installation

From `Docker/Workspace/vision_poisson_experiments`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ../poisson_safety_box
pip install -e ../cbf_safety_box
pip install -e ".[dev]"
make verify-layout
make test
```

Synthetic demonstrations:

```bash
make generate-demo
make static-demo
make live-demo-file
```

## Known limitations and safety scope

1. A photograph is metric only under a physically justified top-down assumption or a valid planar homography.
2. Classical segmentation requires scene-specific tuning and conservative uncertainty margins.
3. The static CBF result is reduced-order numerical evidence under the stated field, interpolation, integration,
   solver, and single-integrator assumptions. It is not a certificate for unmodeled hardware dynamics.
4. A live result is a sequence of quasi-static Poisson fields. The estimated `partial_h_t` is diagnostic only.
   Formal moving-obstacle guarantees require a time-dependent barrier condition, moving-boundary modeling, latency
   and uncertainty bounds, and a validated tracking model.
5. Camera movement invalidates the metric map; the pipeline warns, stops accepting new metric solves, retains the
   last valid field with staleness, and disables the optional virtual CBF until recalibration.
6. ROS 2, Gazebo, PX4, Crazyflie, OptiTrack, and real flight hardware are intentionally outside this delivery.

## Integration roadmap

The next layer should wrap the tested data contracts rather than modify their mathematical core:

1. ROS 2 image subscriber → calibrated image and Boolean occupancy message;
2. Poisson worker node → versioned `h`, gradient, Hessian, validity, and timestamp;
3. CBF node → nominal command plus `SafetySample` → safe command;
4. PX4 Offboard adapter → validated setpoint publication and watchdog;
5. Gazebo/PX4 SITL regression scenarios → latency, tracking-error, and safety metrics;
6. OptiTrack/Crazyflie and later X500 experiments → calibrated uncertainty and tracking bounds.

## Reproducibility artifacts

- `reports/TEST_REPORT.md`: concise executed-validation report;
- `reports/validation/`: final JSON summaries and command transcripts;
- `sample_outputs/static_demo/`: complete static output;
- `sample_outputs/live_demo/live_20260715T162248_649Z/`: complete live output;
- `REPOSITORY_TREE.txt`: generated repository inventory.

# Live Contingency HJR Extension

The repository now contains an optional `LiveContingencyPipeline` selected by `reachability.enabled=true`. It preserves the existing fixed-camera perception stack and adds metric mission selection, landing-zone validation, planar single-integrator HJ/Eikonal reachability, obstacle-aware paths, r-out-of-p target preservation, certified switching, and a single unified projection through `cbf_safety_box`.

The implementation is deliberately limited to an on-screen virtual velocity model. It does not command hardware and does not claim full multicopter reachability. See the new documents in `docs/` and `reports/contingency_validation/` for equations, synchronization rules, failure behavior, and deterministic validation results.
