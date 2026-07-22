#!/usr/bin/env bash
# Full study command matching the requested 48 x 38 x 28 experiment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python run_contingency_study_with_boxes.py \
  --output-dir outputs/full_contingency_boxes \
  --grid-shape 48,38,28 \
  --r-contingency 2 \
  --num-landing-zones 4 \
  --failure-time 18.0 \
  --forcing-methods constant,distance,average_flux,guidance \
  --fixed-forcing guidance \
  --alphas 0.05,0.08,0.12,0.2,0.35,0.5,0.75,1,1.5,2,3,5,8,12 \
  --solver conjugate_gradient \
  --solver-sweep-solvers sparse_direct,conjugate_gradient,sor
