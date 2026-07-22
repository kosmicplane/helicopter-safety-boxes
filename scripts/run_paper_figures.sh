#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/workspace_env.sh"
if [[ -d .venv ]]; then source .venv/bin/activate; fi

OUT="${1:-outputs/paper}"
mkdir -p "$OUT"

python experiments/predefined_world/run.py \
  --compare \
  --output "$OUT/predefined_world"

python experiments/predefined_world/run_sweeps.py \
  --output "$OUT/parameter_sweeps"

python experiments/static_image/run.py \
  --image experiments/static_image/input/example_scene.png \
  --compare \
  --output "$OUT/static_image"

python experiments/live_vision/run.py \
  --source experiments/live_vision/assets/example_stream.avi \
  --output "$OUT/live_vision"

printf '\nPaper figure suite completed. Results: %s\n' "$OUT"
