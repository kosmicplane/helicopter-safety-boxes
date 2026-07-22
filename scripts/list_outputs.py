#!/usr/bin/env python3
"""Print a compact inventory of generated figures and data files."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="outputs")
    args = parser.parse_args()
    root = Path(args.path)
    if not root.exists():
        raise SystemExit(f"Output path does not exist: {root}")
    for suffix in (".png", ".pdf", ".svg", ".csv", ".json", ".npz", ".mp4"):
        files = sorted(root.rglob(f"*{suffix}"))
        if files:
            print(f"\n{suffix[1:].upper()} ({len(files)})")
            for path in files:
                print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
