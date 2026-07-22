#!/usr/bin/env bash
# Source-tree import path for convenience. Editable installation remains the
# recommended user workflow, but experiment scripts can also run directly from
# a clean checkout.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:$ROOT/safety_box_core/src:$ROOT/safety_filter_box/src:$ROOT/clf_safety_box/src:$ROOT/contingency_safety_box/src:$ROOT/poisson_safety_box:$ROOT/cbf_safety_box:$ROOT/vision_poisson_experiments${PYTHONPATH:+:$PYTHONPATH}"
