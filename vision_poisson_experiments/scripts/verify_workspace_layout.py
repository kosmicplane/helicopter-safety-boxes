#!/usr/bin/env python3
"""Verify that both sibling Safety Boxes are discoverable from this repository."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.safety_box_paths import ensure_safety_boxes_importable


if __name__ == "__main__":
    repositories = ensure_safety_boxes_importable()
    for label, repository in repositories.as_dict().items():
        print(f"{label}: {repository}")
