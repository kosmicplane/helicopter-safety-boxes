# Experiments

## 1. Complete predefined-world paper suite

The recommended entry point is:

```bash
bash scripts/run_paper_experiments.sh
```

It executes four controlled studies:

```text
01_baseline              obstacle-rich successful primary landing
02_single_failure        certified diversion and successful contingency landing
03_sequential_failure    repeated site failures ending in HOLD
04_parameter_sweeps      no-failure HOCBF, CLF, and ROA studies
```

It also generates:

```text
00_cross_scenario_figures/paper_scenario_comparison.*
paper_scenario_summary.csv
paper_scenario_summary.json
PAPER_FIGURE_INDEX.md
```

A fast structural check is:

```bash
bash scripts/run_paper_experiments.sh \
  --profile smoke \
  --skip-comparisons \
  --skip-sweeps
```

## 2. Individual predefined-world scenarios

### Baseline landing

```bash
python experiments/predefined_world/run.py \
  --scenario baseline \
  --compare \
  --output outputs/paper/01_baseline
```

### Single failure and diversion

```bash
python experiments/predefined_world/run.py \
  --scenario single_failure \
  --output outputs/paper/02_single_failure
```

### Sequential failures and HOLD

```bash
python experiments/predefined_world/run.py \
  --scenario sequential_failure \
  --output outputs/paper/03_sequential_failure
```

### Parameter sweeps

```bash
python experiments/predefined_world/run_sweeps.py \
  --scenario baseline \
  --output outputs/paper/04_parameter_sweeps
```

The sweep scenario is fixed to the no-failure baseline by default so that target-failure events do not confound HOCBF, CLF, ROA, forcing, or solver comparisons.

## 3. Static-image experiment

```bash
python experiments/static_image/run.py \
  --image experiments/static_image/input/example_scene.png \
  --compare \
  --output outputs/paper/static_image
```

The image is rectified, segmented or paired with a supplied mask, inflated in metric units, converted to occupancy, and used to synthesize a Poisson field. The same certificate stack is then evaluated in 2-D.

## 4. Live vision

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

### IP stream

```bash
python experiments/live_vision/run.py \
  --source "http://CAMERA_IP:PORT/stream.mjpg" \
  --display \
  --output outputs/live_ip_camera
```

Interactive setup uses `B` to capture the empty background, `SPACE` to start, `R` to reset, and `Q` or `Esc` to stop and save.

## 5. Reproducibility

Every run saves:

- effective merged configuration;
- run command and metadata;
- CLF matrices and ROA thresholds;
- raw CSV, JSON, and NPZ data;
- publication figures;
- solver residuals and timing.

Use a timestamped output directory for independent runs:

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
python experiments/predefined_world/run_paper_suite.py \
  --output "outputs/paper/mars_analog_suite_${RUN_ID}"
```
