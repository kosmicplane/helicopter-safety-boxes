#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
for package in \
  safety_box_core \
  safety_filter_box \
  clf_safety_box \
  contingency_safety_box \
  poisson_safety_box \
  cbf_safety_box \
  vision_poisson_experiments; do
  python -m pip install -e "$package"
done
printf '\nInstallation complete. Activate with:\n  source %s/.venv/bin/activate\n' "$ROOT"
