#!/usr/bin/env python3
"""Run all deterministic Poisson-CBF + HJ contingency validation scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.synthetic_contingency_validation import run_all_scenarios


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/contingency_validation")
    args = parser.parse_args()
    summaries = run_all_scenarios(Path(args.output))
    for name, summary in summaries.items():
        print(
            f"{name}: reached={summary['target_reached']} hold={summary['hold']} "
            f"final=LZ-{summary['final_target']} switches={summary['switches']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
