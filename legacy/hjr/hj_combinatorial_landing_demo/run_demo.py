#!/usr/bin/env python3
"""Run the complete four-zone HJ + Poisson-CBF landing demonstration.

This runner uses both user-provided packages as libraries:

* poisson_safety_box: occupancy -> h, grad(h), Hessian(h), Laplacian(h)
* cbf_safety_box:     Poisson safety sample -> affine velocity-CBF constraint

The paper's r-out-of-p Hamilton-Jacobi constraints are then appended to the same
online QP.  Four landing zones are defined and r=2 must remain reachable.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.box_loader import load_safety_boxes
from src.plotting import generate_all_figures
from src.scenario import build_world
from src.simulation import run_simulation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "four_zone_demo"))
    parser.add_argument("--poisson-box", default=str(PROJECT_ROOT.parent / "poisson_safety_box"))
    parser.add_argument("--cbf-box", default=str(PROJECT_ROOT.parent / "cbf_safety_box"))
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    boxes = load_safety_boxes(args.poisson_box, args.cbf_box)
    world = build_world(config)
    artifacts = run_simulation(world, config, boxes, args.output_dir)
    generate_all_figures(artifacts, args.output_dir, config.get("plots", {}))

    print("\n=== DEMO COMPLETE ===")
    for key, value in artifacts.summary.items():
        print(f"{key}: {value}")
    print(f"Figures: {Path(args.output_dir) / 'figures'}")
    print(f"Matrices/data: {Path(args.output_dir) / 'data'}")
    return 0 if artifacts.summary["landed"] and not artifacts.summary["collision"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
