# Vision → Occupancy → Poisson → CBF Experiments

This repository implements two reproducible perception-to-safety experiments around the existing
`poisson_safety_box` and `cbf_safety_box` packages. It is intended to be placed beside both Safety Boxes:

```text
Docker/Workspace/
├── poisson_safety_box/
├── cbf_safety_box/
└── vision_poisson_experiments/
```

The Poisson PDE discretization, forcing construction, numerical solvers, CBF/HOCBF equations, and QP
solvers remain inside the two Safety Boxes. This repository provides camera calibration, classical
segmentation, occupancy construction, physical obstacle inflation, coordinate adaptation, experiment
orchestration, validation, visualization, and logging.

## Experiments

### 1. Static photograph → full Poisson diagnostics → CBF simulation

`01_static_image_poisson_cbf/` accepts one 2D photograph or top-down image. It can:

- calibrate four workspace corners and compute a homography;
- segment obstacles with a supplied mask, manually drawn polygons, HSV thresholds, or a frozen background;
- build uninflated and physically inflated Boolean occupancy matrices (`True = occupied`);
- compare `constant`, `distance`, `average_flux`, and `guidance` Poisson forcing methods;
- compare `sparse_direct`, `conjugate_gradient`, and `sor` Poisson solvers;
- save `h`, `grad_h`, `hessian_h`, `laplacian_h`, masks, forcing, exact assembled-system residuals, and timing;
- produce the complete static figure set, including a 3D surface, contours, gradient/Hessian maps, residuals,
  and method comparisons;
- run nominal and CBF-filtered single-integrator trajectories using bilinear samples from the numerical
  Poisson field.

### 2. Phone/webcam/video stream → asynchronous live Poisson dashboard

`02_phone_stream_poisson_realtime/` accepts a webcam index, local file, HTTP/MJPEG URL, or RTSP URL through
`cv2.VideoCapture`. It uses:

- a capture/perception loop that remains independent from the PDE solver;
- a single Poisson worker and a size-one latest-item queue;
- temporal occupancy filtering with majority vote, EMA, or hysteresis;
- optional ORB/RANSAC global-camera-motion detection;
- exact measurement of capture/display rates, segmentation and pipeline latency, Poisson latency/update rate,
  field age, queue replacement, obsolete solves, and solver failures;
- an OpenCV dashboard plus synchronized NPZ/PNG snapshots and optional low-rate 3D Poisson surfaces;
- an optional virtual CBF diagnostic that never sends commands to hardware.


### 3. Interactive live Poisson-CBF + HJ landing contingencies

When `reachability.enabled: true`, the live entrypoint uses `LiveContingencyPipeline`. After the existing
four-corner calibration, the user selects a metric START point, landing-zone centers, a common physical
landing radius, an active target, and an `r`-out-of-`p` requirement. The runtime then provides:

- obstacle-aware geodesic paths to complete landing-zone disks;
- exact HJ/Eikonal values for the reduced planar model `p_dot = u`, `||u|| <= v_max`;
- a live `r`-th-largest pivot guaranteeing at least `r` reachable targets;
- temporally persistent landing-zone rejection and certified target switching;
- one synchronized latest-only Poisson/HJR worker;
- one unified projection solved by `cbf_safety_box`;
- hard HOLD behavior for camera movement, stale maps, lost contingencies, or solver failure.

Poisson and HJR consume the same occupancy map in parallel. Poisson provides a smooth local collision
certificate; HJR provides finite-horizon target reachability. The demonstration moves only an on-screen
virtual marker.

## End-to-end chain

```text
image or video
  → planar calibration / homographic rectification
  → classical obstacle segmentation
  → Boolean occupancy O (True = occupied)
  → physical obstacle inflation
  → free domain Ω and Dirichlet boundary ∂Ω
  → forcing f_P
  → Poisson safety function h, ∇h, ∇²h, and diagnostics
  → bilinear SafetySample in physical (x, y) coordinates
  → CBF-QP safety filter
```


For contingency mode, the map-dependent branches are synchronized as:

```text
                         ┌──> Poisson Safety Box ──> Poisson CBF row ──┐
image -> occupancy ------┤                                             ├──> CBFBox unified projection
                         └──> HJ/Eikonal fields ──> target/path rows ──┘
```

## Coordinate convention

OpenCV and NumPy index arrays as `(row, column) = (y, x)`. Robot states and commands use `[x, y]`. The
rectified metric frame uses the image top-left as its origin, +x to the right, and +y downward. All axis
permutations are centralized in `common/coordinates.py`:

```text
p_xy                         → p_yx
[∂h/∂y, ∂h/∂x]               → [∂h/∂x, ∂h/∂y]
H_yx                         → P H_yx Pᵀ
```

The CBF receives unnormalized `h`, `grad_h`, and `hessian_h`. Normalization is used only for color display.

## Installation

From `Docker/Workspace/vision_poisson_experiments`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ../poisson_safety_box
pip install -e ../cbf_safety_box
pip install -e ".[dev]"
```

Equivalent automated setup:

```bash
make install
```

The included sibling copy of `poisson_safety_box` contains only the minimal packaging metadata required for
an editable install; its numerical API and implementation are unchanged.

## Reproducible validation

```bash
make verify-layout
make generate-demo
make test
make static-smoke   # fast, lower-resolution all-method validation
make static-demo    # full synthetic diagnostic run
make live-demo-file
```

`make test` executes the original Poisson Safety Box tests, the original CBF Safety Box tests, and this
repository's integration tests. The completed validation commands, numerical residuals, CBF outcome, and measured
live-pipeline rates are recorded in `reports/TEST_REPORT.md`.

Direct static demo:

```bash
python 01_static_image_poisson_cbf/run_experiment.py \
  --image examples/assets/static_scene.png \
  --config 01_static_image_poisson_cbf/config_synthetic.yaml \
  --output sample_outputs/static_demo \
  --assume-top-down --headless
```

Direct live-file demo:

```bash
python 02_phone_stream_poisson_realtime/run_experiment.py \
  --source examples/assets/live_scene.avi \
  --config 02_phone_stream_poisson_realtime/config_synthetic.yaml \
  --output sample_outputs/live_demo \
  --headless --max-frames 60
```


Headless deterministic contingency validation:

```bash
python scripts/run_contingency_synthetic_validation.py \
  --output reports/contingency_validation
```

Headless local-video contingency run:

```bash
python 02_phone_stream_poisson_realtime/run_experiment.py \
  --source examples/assets/live_scene.avi \
  --config 02_phone_stream_poisson_realtime/config_contingency_synthetic.yaml \
  --output sample_outputs/live_contingency_demo \
  --headless --max-frames 60
```

Phone stream example:

```bash
python 02_phone_stream_poisson_realtime/run_experiment.py \
  --source "http://PHONE_IP:PORT/video" \
  --config 02_phone_stream_poisson_realtime/config.yaml
```

A webcam can be selected with `--source 0`.

## Photograph calibration

A photograph represents a metric 2D map only when the camera is top-down or the planar workspace is rectified.
Use `--assume-top-down` only when that assumption is physically valid. Otherwise run with `--interactive`, set
`calibration.mode: interactive`, and click:

1. top-left;
2. top-right;
3. bottom-right;
4. bottom-left.

The source points, homography, output size, and physical workspace size are stored in `calibration.json`.

## Segmentation modes

The common segmentation interface supports:

- `mask_file`: load a user-provided binary mask;
- `manual_polygon`: add or erase obstacle polygons with the mouse;
- `hsv`: use configurable HSV bounds and optional interactive tuning;
- `background_reference`: compare against a frozen empty-workspace image.

The live baseline never adapts the background continuously. A stationary obstacle therefore cannot disappear merely
because it has remained in view. Morphological opening/closing, component filtering, hole filling, inversion, and
manual correction are configurable.

## Static CBF demonstration

The reduced-order model is `p_dot = u`. The nominal command is a saturated go-to-goal velocity. At each integration
step the program bilinearly samples the actual Poisson arrays, converts derivatives from `(y, x)` to physical `(x, y)`,
and calls `CBFBox` with

```text
∇h(p)ᵀ u ≥ -α h(p).
```

The closed-form CBF solver is used only for the unbounded single-halfspace problem. When component bounds are enabled,
the SciPy QP backend is used so post-solution clipping cannot silently violate the CBF condition. CSV logs include the
explicit CBF residual, solver status, sampled `h`, control intervention, integration backtracking, and a separate
geometric-clearance diagnostic.

## Live controls

```text
q  quit                     s  save synchronized snapshot
r  recalibrate              l  redefine START / landing zones
b  freeze a new background  p  pause/resume
f  cycle forcing method     m  cycle segmentation
d  toggle diagnostics       c  toggle virtual controller
h  toggle HJR overlays      g  toggle active path
a  cycle certified target   x  manually clear rejection latches
```

Video FPS and Poisson update rate are reported independently. A responsive 30 FPS video display is never mislabeled as
a 30 Hz Poisson solver.

## Safety scope and limitations

The static CBF experiment provides numerical evidence only under its stated single-integrator model, field interpolation,
integration step, perception map, and solver tolerances. It does not certify unmodeled hardware dynamics.

Each live result is a quasi-static field `h_t` computed from one occupancy snapshot. The finite-difference estimate of
`partial_h_t` is diagnostic only. A formal moving-obstacle guarantee would require an explicit time-dependent barrier
condition, moving-boundary modeling, latency and uncertainty bounds, and a validated tracking model. A colored Poisson
heatmap alone is not a safety guarantee.


The contingency HJR is exact only for a planar isotropic single integrator. It does not certify
multicopter acceleration, attitude, touchdown velocity, battery, wind, or hardware tracking. See
`docs/HJR_EIKONAL_MODEL.md` and `docs/FAILURE_AND_HOLD_BEHAVIOR.md`.

## Outputs

Static runs save effective YAML, calibration, raw/clean masks, occupancy arrays, one Poisson directory per forcing method,
solver comparisons, scientific figures, CBF CSV/NPZ logs, and JSON summaries.

Live runs save effective YAML, calibration/background references, frame and solve CSVs, percentile summaries, warning
logs, the latest valid field and exact validation, optional annotated video, and synchronized 2D/3D snapshots.

See `IMPLEMENTATION_REPORT.md`, `docs/ARCHITECTURE.md`, `docs/CONTINGENCY_HJR_LIVE.md`,
`docs/HJR_EIKONAL_MODEL.md`, `docs/UNIFIED_CBF_HJ_FILTER.md`, and the validation reports for details.
