# Configuration Guide

Both experiments use YAML. Command-line arguments override only explicitly supplied values, and every run
stores the resulting effective configuration.

## Workspace

```yaml
workspace:
  width_m: 6.0
  height_m: 4.5
  grid:
    nx: 96
    ny: 72
  rectified_image:
    width_px: 640
    height_px: 480
```

The Poisson grid is node-centered, so `dx = width_m / (nx - 1)` and
`dy = height_m / (ny - 1)`.

## Calibration

```yaml
calibration:
  mode: assume_top_down   # assume_top_down, interactive, load_file
  file: calibration/phone_calibration.json
```

`assume_top_down` is valid only when the source image already represents a planar top view. Interactive mode requires four points in this exact order: **top-left, top-right, bottom-right, bottom-left**.
The quadrilateral is validated before a homography is accepted.

## Segmentation

```yaml
segmentation:
  mode: background_reference  # mask_file, manual_polygon, hsv, background_reference
  mask_file: path/to/mask.png
  reference_file: path/to/empty_workspace.png
  hsv:
    lower: [0, 0, 0]
    upper: [179, 255, 100]
    invert: false
    tune_interactively: false
  background:
    difference_threshold: 25
    difference_blur_kernel: 5
    use_color_norm: false
  cleanup:
    blur_kernel: 3
    threshold: 127
    open_kernel: 3
    opening_iterations: 1
    close_kernel: 5
    closing_iterations: 1
    minimum_component_area_px: 40
    fill_holes: true
    invert: false
  manual_correction:
    enabled: false
    brush_radius_px: 12
```

All kernel sizes are converted to odd positive sizes. A zero kernel disables that operation.

## Occupancy and geometry

```yaml
occupancy:
  robot_radius_m: 0.12
  perception_margin_m: 0.08
```

The historical section name `geometry` is also accepted. When both exist, `occupancy` keys override
matching `geometry` keys while unspecified legacy values are retained.

## Poisson

```yaml
poisson:
  forcing_methods: [constant, distance, average_flux, guidance]
  primary_forcing_method: constant
  solver: sparse_direct
  compare_solvers: false
  solver_comparison: [sparse_direct, conjugate_gradient, sor]
  boundary_value: 0.0
  outer_boundary_as_dirichlet: true
  compute_gradient: true
  compute_hessian: true
  compute_laplacian_check: true
  validation_residual_tolerance: 1.0e-5
```

Poisson solvers are `sparse_direct`, `conjugate_gradient`, and `sor`. These names are never used for the
CBF-QP.

## Static CBF simulation

```yaml
cbf:
  enabled: true
  start_xy_m: [0.4, 3.8]
  goal_xy_m: [5.6, 0.5]
  alpha: 0.2
  solver: closed_form      # closed_form, scipy, cvxpy
  goal_gain: 1.1
  dt_s: 0.02
  maximum_steps: 1300
  maximum_speed_mps: 0.8
  goal_tolerance_m: 0.1
  enforce_control_bounds: false
  component_bound_mode: euclidean_conservative
```

Use `scipy` or `cvxpy` when hard component bounds are required. The unbounded one-halfspace velocity CBF
can use `closed_form`.

## Live runtime

```yaml
live:
  capture_first_frame_as_background: true
  segmentation_modes: [background_reference, hsv]
  forcing_methods: [constant, distance]
  minimum_changed_fraction: 0.002
  maximum_submit_interval_s: 0.5
  laplacian_every_n_solves: 10
  maximum_field_age_s: 0.75
  worker_shutdown_timeout_s: 15.0
  temporal_filter:
    mode: majority          # majority, ema, hysteresis
    window_size: 5
    ema_alpha: 0.35
    ema_threshold: 0.5
    activation_frames: 2
    deactivation_frames: 4
```

The queue capacity is fixed at one by design and is not configurable.

## Camera movement

```yaml
camera_motion:
  enabled: true
  translation_threshold_px: 4.0
  rotation_threshold_deg: 1.5
  minimum_matches: 20
  maximum_features: 1200
```

## Output

```yaml
output:
  root: outputs
  figure_dpi: 135
  dashboard_panel_size: [420, 280]
  record_dashboard_video: true
  dashboard_video_fps: 15.0
  snapshot_every_n_solves: 10
  save_3d_snapshots: true
  snapshot_3d_dpi: 110
```

## Contingency and HJR sections

The optional live contingency mode uses five additional sections:

- `mission_setup`: interactive/load-file mission geometry, radius bounds, and required `r`;
- `landing_zone_validation`: occupancy fractions, clearance, seed count, and temporal hysteresis;
- `reachability`: single-integrator speed and active/contingency horizons;
- `planner`: look-ahead path-following parameters;
- `poisson_cbf`: effective Poisson margin and field age;
- `contingency_filter`: active/contingency gains, auxiliary penalties, norm-constrained solver, and HOLD policy.

Set `reachability.enabled: false` to retain the original live pipeline. In headless mode use
`mission_setup.mode: load_file`; interactive mission selection requires an OpenCV window. All physical
lengths are in meters and all horizons are in seconds.

The packaged `config_contingency_synthetic.yaml` is intended for deterministic local-video validation.
The regular `config.yaml` uses interactive workspace and mission setup.
