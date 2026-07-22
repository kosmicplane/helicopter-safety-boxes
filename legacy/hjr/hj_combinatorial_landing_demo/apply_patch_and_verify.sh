#!/usr/bin/env bash
# Install the corrected local boxes into the current Helicopter project.
# Run this script from inside the extracted patch directory.
set -euo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-$PWD}"

# Prevent accidental installation into the patch directory itself when the user
# intended to patch another workspace.
if [[ "$TARGET_DIR" == "$PATCH_DIR" ]]; then
  echo "Usage: ./apply_patch_and_verify.sh /path/to/Helicopter"
  exit 2
fi

mkdir -p "$TARGET_DIR"

# Preserve the current box once.  This makes rollback straightforward.
if [[ -d "$TARGET_DIR/cbf_safety_box" && ! -d "$TARGET_DIR/cbf_safety_box_before_contingency_patch" ]]; then
  cp -a "$TARGET_DIR/cbf_safety_box" "$TARGET_DIR/cbf_safety_box_before_contingency_patch"
fi

# Overlay complete source trees.  The contingency extension changes more than
# __init__.py, including the generic multi-row QP API and norm constraints.
rsync -a --delete --exclude='outputs/' "$PATCH_DIR/cbf_safety_box/" "$TARGET_DIR/cbf_safety_box/"
rsync -a --delete --exclude='outputs/' "$PATCH_DIR/poisson_safety_box/" "$TARGET_DIR/poisson_safety_box/"
cp "$PATCH_DIR/run_contingency_study_with_boxes.py" "$TARGET_DIR/"
cp "$PATCH_DIR/verify_box_integration.py" "$TARGET_DIR/"

# Remove stale bytecode so Python cannot display confusing old tracebacks.
find "$TARGET_DIR/cbf_safety_box" "$TARGET_DIR/poisson_safety_box" \
  -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "$TARGET_DIR/cbf_safety_box" "$TARGET_DIR/poisson_safety_box" \
  -type f -name '*.pyc' -delete 2>/dev/null || true

cd "$TARGET_DIR"
python verify_box_integration.py

echo "Patch installed successfully in: $TARGET_DIR"
