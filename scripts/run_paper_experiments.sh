#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv_boxes/bin/activate ]]; then
  source .venv_boxes/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="$PWD/cbf_safety_box:$PWD/poisson_safety_box:$PWD/safety_box_core/src:$PWD/clf_safety_box/src:$PWD/contingency_safety_box/src:$PWD/safety_filter_box/src:$PWD/vision_poisson_experiments:$PWD${PYTHONPATH:+:$PYTHONPATH}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
python experiments/predefined_world/run_paper_suite.py \
  --output "outputs/paper/mars_analog_suite_${RUN_ID}" \
  "$@"
