"""Import helpers for the two externally supplied safety boxes.

The demo is intended to live beside these folders:

    workspace/
      poisson_safety_box/
      cbf_safety_box/
      hj_combinatorial_landing_demo/

The command-line runner can also receive explicit paths.  No source code is
copied from either safety box into this project.
"""
from __future__ import annotations

import sys
from pathlib import Path


def load_safety_boxes(poisson_box_path: str | Path, cbf_box_path: str | Path):
    """Add both package roots to ``sys.path`` and return their public classes."""
    poisson_root = Path(poisson_box_path).expanduser().resolve()
    cbf_root = Path(cbf_box_path).expanduser().resolve()
    if not (poisson_root / "poisson_safety_box" / "api.py").exists():
        raise FileNotFoundError(
            f"Poisson Safety Box was not found at {poisson_root}. "
            "Expected poisson_safety_box/api.py below that directory."
        )
    if not (cbf_root / "cbf_safety_box" / "api.py").exists():
        raise FileNotFoundError(
            f"CBF Safety Box was not found at {cbf_root}. "
            "Expected cbf_safety_box/api.py below that directory."
        )
    for root in (poisson_root, cbf_root):
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    from poisson_safety_box import PoissonBoxConfig, PoissonSafetyBox
    from cbf_safety_box import CBFBox, CBFBoxConfig, SafetySample, SystemState

    return {
        "PoissonBoxConfig": PoissonBoxConfig,
        "PoissonSafetyBox": PoissonSafetyBox,
        "CBFBox": CBFBox,
        "CBFBoxConfig": CBFBoxConfig,
        "SafetySample": SafetySample,
        "SystemState": SystemState,
    }
