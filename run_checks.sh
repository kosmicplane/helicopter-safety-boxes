#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -d .venv ]]; then source .venv/bin/activate; fi

python -m compileall -q \
  safety_box_core/src safety_filter_box/src clf_safety_box/src \
  contingency_safety_box/src cbf_safety_box/cbf_safety_box \
  poisson_safety_box/poisson_safety_box experiments

# Active CLF/ROA architecture plus the complete original CBF and Poisson suites.
python -m pytest

# Preserve and regression-test the complete original vision workspace. The
# historical five-scenario HJR end-to-end suite is inactive in the new CLF
# methodology and can be computationally long, so it is opt-in.
if [[ "${RUN_LEGACY_HJR_TESTS:-0}" == "1" ]]; then
  python scripts/run_vision_regression.py --include-legacy-hjr
else
  python scripts/run_vision_regression.py
fi

python scripts/verify_clf_runtime.py
printf '\nAll active workspace checks passed.\n'
if [[ "${RUN_LEGACY_HJR_TESTS:-0}" != "1" ]]; then
  printf 'Historical HJR scenario tests were preserved but skipped. Run with RUN_LEGACY_HJR_TESTS=1 to include them.\n'
fi
