#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv_boxes/bin/activate ]]; then
  source .venv_boxes/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
source "$ROOT/scripts/workspace_env.sh"

OUT="${1:-outputs/paper/full_multimode_suite}"
mkdir -p "$OUT"

# Controlled paper claims and parameter sweeps on the canonical Mars-analog world.
python experiments/predefined_world/run_paper_suite.py \
  --output "$OUT/predefined_world"

# Repeatable perception-to-safety evidence from one fixed image.
python experiments/static_image/run.py \
  --image experiments/static_image/input/example_scene.png \
  --compare \
  --output "$OUT/static_image"

# Reproducible online pipeline using the included video source.
python experiments/live_vision/run.py \
  --source experiments/live_vision/assets/example_stream.avi \
  --output "$OUT/live_vision"

printf '\nFull multi-mode paper figure suite completed. Results: %s\n' "$OUT"
