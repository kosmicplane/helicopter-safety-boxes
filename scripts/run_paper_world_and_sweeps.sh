#!/usr/bin/env bash
set -euo pipefail

ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"

cd "$ROOT"
source .venv_boxes/bin/activate

export PYTHONPATH="$PWD/cbf_safety_box:$PWD/poisson_safety_box:$PWD/safety_box_core/src:$PWD/clf_safety_box/src:$PWD/contingency_safety_box/src:$PWD/safety_filter_box/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="outputs/paper/enlarged_world_${RUN_ID}"

mkdir -p "$OUTPUT_ROOT"

echo
echo "1/4 Baseline successful landing and method comparisons"
echo

python experiments/predefined_world/run.py \
  --no-failure \
  --compare \
  --output "$OUTPUT_ROOT/01_baseline_success"

echo
echo "2/4 Single landing-zone failure and contingency landing"
echo

python experiments/predefined_world/run.py \
  --set 'experiments.predefined_world.simulation.failure_schedule=[{time_s: 10.0, target: active, reason: Primary landing zone became unavailable}]' \
  --output "$OUTPUT_ROOT/02_single_failure_contingency"

echo
echo "3/4 Sequential landing-zone failure stress test"
echo

python experiments/predefined_world/run.py \
  --output "$OUTPUT_ROOT/03_sequential_failure_stress_test"

echo
echo "4/4 Fair parameter sweeps without target failures"
echo

python experiments/predefined_world/run_sweeps.py \
  --no-failure \
  --output "$OUTPUT_ROOT/04_parameter_sweeps"

echo
echo "Paper outputs:"
echo "$OUTPUT_ROOT"
echo

find "$OUTPUT_ROOT" \
  -type f \
  \( \
    -name "*.png" \
    -o -name "*.pdf" \
    -o -name "*.svg" \
    -o -name "*.csv" \
    -o -name "*.json" \
  \) \
  | sort

xdg-open "$OUTPUT_ROOT" >/dev/null 2>&1 || true
