# Experiments

## 1. Predefined 3-D world

This is the primary controlled experiment for the first paper results. It tests a 3-D double integrator in an obstacle-rich world with four landing equilibria, Poisson-HOCBF collision avoidance, active-target CLF convergence, an `r=2` contingency requirement, and a programmed target failure.

### Smoke validation

```bash
python experiments/predefined_world/run.py \
  --profile smoke \
  --output outputs/runs/predefined_smoke
```

### Paper-resolution experiment and comparisons

```bash
python experiments/predefined_world/run.py \
  --compare \
  --output outputs/paper/predefined_world
```

The `--compare` option evaluates every configured Poisson forcing method and Poisson solver against the same geometry. Solver comparisons separate wall time, solve time, iterations, assembled-system residual, reconstructed-Laplacian error, and field error relative to the sparse-direct reference.

### Gain and ROA sweeps

```bash
python experiments/predefined_world/run_sweeps.py \
  --output outputs/paper/parameter_sweeps
```

The script compares HOCBF gain scales, CLF decay gains, and ROA threshold fractions. It exports raw CSV data, metric panels, and 3-D plus orthogonal trajectory families.

## 2. Static-image experiment

```bash
python experiments/static_image/run.py \
  --image experiments/static_image/input/example_scene.png \
  --compare \
  --output outputs/paper/static_image
```

The image is rectified, segmented or paired with a supplied mask, inflated in metric units, converted to occupancy, and used to synthesize a Poisson field. The same controller and certificate stack used in the 3-D world is then applied in 2-D.

## 3. Live vision

### Included video

```bash
python experiments/live_vision/run.py \
  --source experiments/live_vision/assets/example_stream.avi \
  --display \
  --output outputs/paper/live_vision
```

### USB camera

```bash
python experiments/live_vision/run.py \
  --source 0 \
  --display \
  --output outputs/live_camera
```

The live mode uses a latest-only asynchronous Poisson worker. The control loop retains the newest valid versioned field while a replacement is being computed and reports field age, occupancy version, dropped tasks, filter status, and certificate histories.

## Reproducibility

Every run saves:

- the effective merged configuration;
- a configuration hash and run metadata;
- CLF matrices and ROA thresholds;
- raw CSV/JSON/NPZ data;
- all figures in a dedicated directory.

Use a timestamped output directory to retain independent runs:

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
python experiments/predefined_world/run.py \
  --output "outputs/runs/${RUN_ID}"
```

## Recommended first-results matrix

| Study | Independent variable | Required metrics |
|---|---|---|
| HOCBF sensitivity | \(\gamma_1,\gamma_2\) | minimum \(h_P\), HOCBF residual, intervention, path length, landing error |
| CLF sensitivity | \(\alpha_V\) gain | CLF residual, CLF slack, settling time, intervention, terminal error |
| ROA sensitivity | \(c_j\) fraction | projected volume, initial certified count, minimum pivot, landing success |
| Forcing comparison | constant, distance, average flux, guidance | field geometry, minimum \(h_P\), intervention, solve time, trajectory |
| Poisson solver comparison | sparse direct, CG, SOR | time, iterations, \(\|Ah-b\|\), \(\|\Delta_hh-f_P\|\), field error |
| Architecture ablation | nominal, HOCBF, CLF, HOCBF+CLF, full contingency | safety, convergence, contingency, feasibility, time |
