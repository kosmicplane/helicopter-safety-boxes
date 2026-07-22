# Software Architecture

## Design objective

The repository separates perception, numerical safety synthesis, control filtering, visualization, and
runtime orchestration. The separation is not cosmetic: each stage owns a precise data contract, can be
tested independently, and emits diagnostics that localize failures.

The two external Safety Boxes remain the authoritative implementations of Poisson synthesis and CBF
filtering. This repository never substitutes analytical barriers or a second PDE implementation.

## Workspace layout

```text
Docker/Workspace/
├── poisson_safety_box/            external PDE package
├── cbf_safety_box/                external CBF/QP package
└── vision_poisson_experiments/    perception and experiment package
```

`common/external_boxes.py` resolves the sibling paths. Environment variables
`POISSON_SAFETY_BOX_ROOT` and `CBF_SAFETY_BOX_ROOT` can override the default layout for CI or a custom
Docker mount.

## Static data path

```text
photograph
  -> CalibrationData
  -> rectified BGR image
  -> SegmentationResult(raw_mask, clean_mask)
  -> OccupancyMaps(uninflated, inflated)
  -> PoissonRunBundle(method -> PoissonRunRecord)
  -> GridFieldSampler
  -> SafetySample
  -> CBFComparisonResult(nominal, safe)
```

### Calibration

`calibration.py` validates the ordered quadrilateral, builds a perspective transform, saves reusable JSON,
and rectifies the source image. A top-down identity-like mapping must be requested explicitly.

### Segmentation

`segmentation.py` implements four interchangeable strategies behind one interface:

- exact binary mask file;
- manual polygons;
- HSV thresholding, optionally tuned with trackbars;
- frozen-background differencing.

Every mode enters the same deterministic cleanup chain: blur, threshold, opening, closing, connected
component filtering, optional hole filling, optional inversion, and optional manual brush correction.

### Occupancy

`occupancy.py` enforces `True = occupied`. Binary masks are resized with nearest-neighbor interpolation.
Robot radius and perception margin are converted from meters to anisotropic grid radii with `ceil`, then
applied with a conservative elliptical structuring element. Both the perception occupancy and the inflated
configuration-space occupancy are retained.

### Poisson orchestration

`poisson_runner.py` constructs `PoissonBoxConfig` and calls `PoissonSafetyBox.compute`. It validates finite
arrays, boundary values, solver state, and the exact residual of the sparse linear system assembled by the
Safety Box. A centered five-point stencil is retained as a secondary diagnostic. It stores the raw Safety Box
result without normalizing the field.

### Coordinate adapter

`coordinates.py` is the only location that changes `(y, x)` array order into `(x, y)` control order. It
performs bilinear scalar, gradient, Hessian, and Laplacian interpolation and returns explicit validity
reasons for out-of-domain or occupied queries.

### CBF demonstration

`cbf_demo.py` builds a saturated go-to-goal command and sends sampled Poisson values to `CBFBox`. It logs
solver status, CBF residual, raw `h`, controls, intervention norm, clearance diagnostics, and accepted Euler
step size. A bounded backtracking guard rejects candidate discrete states that would enter inflated
occupancy; it does not replace the CBF or alter the Poisson field.

## Live runtime

The live pipeline uses three logical stages:

1. **Capture/UI thread** — obtains the newest decoded frame and keeps the dashboard responsive.
2. **Perception stage** — rectifies, segments, temporally filters, and inflates the occupancy map.
3. **Poisson worker** — solves only the newest submitted occupancy task.

`LatestItemQueue` has capacity one. A new task replaces an older waiting task. Each task carries a monotonic
`version_id`; a result that finishes after a newer version has become current is counted and discarded.

The main loop never waits for a Poisson solve. It renders only the latest numerically validated result and
reports its age from occupancy submission time, including solve delay. Invalid or obsolete results are counted
and cannot replace the last valid field. This makes display FPS and Poisson update frequency independent
quantities.

## Camera movement and map validity

`GlobalMotionDetector` compares the rectified stream to a fixed ORB reference with robust affine fitting.
When movement exceeds configured translation or rotation limits:

- no new Poisson task is submitted;
- the dashboard displays `CAMERA MOVED - MAP NOT METRIC`;
- the last valid field remains visible with its age;
- the virtual CBF diagnostic is disabled;
- recalibration resets the metric transform and the motion reference.

The detector is diagnostic rather than an automatic stabilizer. It never hides a changed camera geometry.

## Failure handling

All entry points produce actionable errors for missing files, empty frames, invalid homographies, invalid
masks, empty Poisson domains, nonfinite fields, nonconverged solvers, invalid endpoints, stale fields, and QP
failures. The live worker rejects an invalid or nonconverged field without replacing the last valid result and
records the event to the structured warning CSV.

## Output model

Each run is self-describing. It stores the effective configuration, calibration, numerical arrays, JSON
summaries, CSV time series, logs, and figures. File names are deterministic inside a timestamped run folder,
which makes comparisons and regression testing straightforward.

## Extension boundary

No module imports ROS 2, Gazebo, PX4, or a phone-specific application. Future integrations should adapt
state, occupancy, and command messages at the repository boundary while preserving the tested internal data
contracts.

## Contingency live runtime

When `reachability.enabled=true`, the entrypoint instantiates `LiveContingencyPipeline`. The original
perception stages remain unchanged, but the old Poisson-only worker is replaced by a versioned
`SafetySynthesisWorker` with queue capacity one. One accepted snapshot contains the Poisson result, every
geodesic/HJ field, landing-zone assessments, and one occupancy version.

```text
                         ┌──> poisson_safety_box ──> Poisson CBF row ──┐
raw/filtered/inflated O ─┤                                             ├──> cbf_safety_box unified solve
                         └──> HJ/Eikonal bundle ──> HJ rows / path ───┘
```

`mission_setup.py` owns metric START and landing-zone geometry. `target_manager.py` owns latching and
discrete switching. `hj_reachability.py` owns the single-integrator distance/value construction.
`contingency_planner.py` extracts collision-free paths and nominal velocity.
`unified_contingency_filter.py` delegates all online constraint construction and optimization to
`cbf_safety_box`. `discrete_safety.py` guards the finite Euler step.

Poisson never feeds the HJ equation. Both branches consume one synchronized map. If a result is stale,
version-mismatched, infeasible, or invalidated by camera motion, the virtual command is exactly zero.
