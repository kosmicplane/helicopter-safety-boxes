#!/usr/bin/env python3
"""Run the preserved vision regression suite and force a clean process exit.

Some historical worker tests intentionally create background executors. Under
certain Python/OpenCV combinations those threads can delay interpreter teardown
after pytest has already reported the final result. Running pytest in this
small subprocess and using ``os._exit`` preserves the test status without
allowing teardown behavior to block the workspace validation command.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-legacy-hjr", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "vision_poisson_experiments"
    pytest_args = ["-q", str(root / "tests")]
    if not args.include_legacy_hjr:
        pytest_args.append(
            f"--ignore={root / 'tests' / 'test_synthetic_contingency_validation.py'}"
        )
    return int(pytest.main(pytest_args))


if __name__ == "__main__":
    code = main()
    os._exit(code)
