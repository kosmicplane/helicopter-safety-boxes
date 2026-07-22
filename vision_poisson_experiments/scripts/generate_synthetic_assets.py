#!/usr/bin/env python3
"""Generate deterministic static-image and live-video fixtures."""

from __future__ import annotations

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.synthetic import generate_all_assets


def main() -> int:
    """Generate all assets under ``examples/assets``."""

    output = REPOSITORY_ROOT / "examples" / "assets"
    assets = generate_all_assets(output)
    for name, path in sorted(assets.items()):
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
