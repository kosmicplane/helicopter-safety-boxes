#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/workspace_env.sh"
if [[ -d .venv ]]; then source .venv/bin/activate; fi

OUT="${1:-outputs/runs/smoke_suite}"
mkdir -p "$OUT"

python experiments/predefined_world/run.py \
  --profile smoke \
  --output "$OUT/predefined_world"

python experiments/static_image/run.py \
  --profile smoke \
  --image experiments/static_image/input/example_scene.png \
  --output "$OUT/static_image"

python experiments/live_vision/run.py \
  --profile smoke \
  --source experiments/live_vision/assets/example_stream.avi \
  --output "$OUT/live_vision"

printf '\nSmoke suite completed. Results: %s\n' "$OUT"
