#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f .venv_boxes/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv_boxes/bin/activate
  elif [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
fi

export PYTHONPATH="$ROOT/cbf_safety_box:$ROOT/poisson_safety_box:$ROOT/safety_box_core/src:$ROOT/clf_safety_box/src:$ROOT/contingency_safety_box/src:$ROOT/safety_filter_box/src:$ROOT/vision_poisson_experiments:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
from cbf_safety_box import CBFBox, CBFBoxConfig, SafetySample
from experiments.common.simulation import resolve_failure_schedule
print("Local safety-box imports verified.")
PY

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="outputs/paper/multi_failure_${RUN_ID}"
WORLD_OUTPUT="$OUTPUT_ROOT/predefined_world"
SWEEP_OUTPUT="$OUTPUT_ROOT/parameter_sweeps"

mkdir -p "$OUTPUT_ROOT"

python experiments/predefined_world/run.py \
  --compare \
  --output "$WORLD_OUTPUT"

python experiments/predefined_world/run_sweeps.py \
  --output "$SWEEP_OUTPUT"

printf '\nPaper experiment outputs:\n  World:  %s\n  Sweeps: %s\n' \
  "$WORLD_OUTPUT" "$SWEEP_OUTPUT"

find "$OUTPUT_ROOT" \
  -type f \
  \( -name '*.png' -o -name '*.pdf' -o -name '*.svg' -o -name '*.csv' -o -name '*.json' \) \
  | sort
