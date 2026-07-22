#!/usr/bin/env python3
"""Verify that the runtime sees the extended Poisson and CBF safety boxes.

This script intentionally performs more than a superficial import test:
1. It reports the exact package files imported by Python.
2. It verifies the contingency symbols added to ``cbf_safety_box``.
3. It executes a small Poisson solve through ``PoissonSafetyBox.compute``.
4. It solves a small r-out-of-p QP through ``CBFBox.filter_affine_constraints``.

Run it from the project root with:

    python verify_box_integration.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


# Add both local package roots before importing.  This mirrors the main runtime
# and guarantees that the checked files are the boxes beside this script.
ROOT = Path(__file__).resolve().parent
for package_root in (ROOT / "poisson_safety_box", ROOT / "cbf_safety_box"):
    sys.path.insert(0, str(package_root))

import poisson_safety_box as poisson_package
import cbf_safety_box as cbf_package
from poisson_safety_box import PoissonBoxConfig, PoissonSafetyBox
from cbf_safety_box import (
    AffineCertificate,
    CBFBox,
    CBFBoxConfig,
    Constraint,
    build_combinatorial_contingency_constraints,
)


def verify_public_api() -> None:
    """Check the public symbols required by the contingency runtime."""
    required_symbols = (
        "AffineCertificate",
        "build_active_target_clf_constraint",
        "build_combinatorial_contingency_constraints",
        "lift_constraint_with_auxiliary",
    )
    missing = [name for name in required_symbols if not hasattr(cbf_package, name)]
    if missing:
        raise RuntimeError(
            "The local cbf_safety_box is still the old baseline version. "
            f"Missing symbols: {missing}. Extract the patch with overwrite enabled."
        )
    if not hasattr(CBFBox, "filter_affine_constraints"):
        raise RuntimeError("CBFBox.filter_affine_constraints is missing from the installed box.")


def verify_poisson_box_execution() -> None:
    """Execute a tiny, fast Poisson solve using the actual Poisson box."""
    occupancy = np.zeros((14, 12), dtype=bool)
    occupancy[5:9, 5:8] = True
    config = PoissonBoxConfig(
        grid_spacing=(0.2, 0.2),
        forcing_method="constant",
        solver="sparse_direct",
        plot=False,
    )
    result = PoissonSafetyBox(config).compute(occupancy)
    if result.h.shape != occupancy.shape or not np.all(np.isfinite(result.h)):
        raise RuntimeError("PoissonSafetyBox returned an invalid field.")


def verify_cbf_box_execution() -> None:
    """Execute a small 2-out-of-3 contingency QP through the actual CBF box."""
    certificates = [
        AffineCertificate("LZ0", 0.40, -0.02, np.array([1.0, 0.0, 0.0])),
        AffineCertificate("LZ1", 0.30, -0.01, np.array([0.0, 1.0, 0.0])),
        AffineCertificate("LZ2", -0.20, 0.00, np.array([0.0, 0.0, 1.0])),
    ]
    contingency_rows, pivot = build_combinatorial_contingency_constraints(
        certificates,
        r=2,
        gamma=0.2,
        auxiliary_gain=0.1,
    )
    if not np.isclose(pivot, 0.30):
        raise RuntimeError(f"Unexpected r-th-largest pivot: {pivot}")

    # The augmented decision is z = [a_x, a_y, a_z, omega].
    environment_row = Constraint(
        A=np.array([[1.0, 0.0, 0.0, 0.0]]),
        b=np.array([-0.5]),
        name="environment_smoke_test",
    )
    box = CBFBox(CBFBoxConfig(mode="acceleration", solver="scipy"))
    result = box.filter_affine_constraints(
        decision_nominal=np.zeros(4),
        constraints=[environment_row, *contingency_rows],
        lower_bounds=np.array([-np.inf, -np.inf, -np.inf, 0.0]),
        upper_bounds=np.array([np.inf, np.inf, np.inf, 10.0]),
        quadratic_weights=np.array([1.0, 1.0, 1.0, 0.2]),
        norm_bound_indices=[[0, 1, 2]],
        norm_bound_values=[1.0],
    )
    if not bool(result.diagnostics["feasibility"]["feasible"]):
        raise RuntimeError("The CBF contingency QP was not feasible in the smoke test.")


def main() -> None:
    """Run all checks and print an auditable import report."""
    print("Poisson package:", Path(poisson_package.__file__).resolve())
    print("CBF package:    ", Path(cbf_package.__file__).resolve())
    verify_public_api()
    verify_poisson_box_execution()
    verify_cbf_box_execution()
    print("PASS: PoissonSafetyBox and the extended CBFBox are correctly installed and executed.")


if __name__ == "__main__":
    main()
