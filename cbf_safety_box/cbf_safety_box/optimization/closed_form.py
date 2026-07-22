"""Closed-form projection onto a single CBF half-space."""
from __future__ import annotations

import time
import numpy as np

from ..constraints.control_limits import clip_to_bounds


def solve_halfspace_projection(u_nom: np.ndarray, a: np.ndarray, b: float, lower_bounds: np.ndarray | None = None, upper_bounds: np.ndarray | None = None, tolerance: float = 1.0e-9) -> dict:
    """Solve ``min 1/2||u-u_nom||^2`` subject to ``a^T u >= b``.

    The unconstrained projection has a closed form.  Optional clipping is
    provided for convenience, but clipping can break CBF feasibility; the
    returned diagnostics explicitly report the final residual.
    """
    start = time.perf_counter()
    u_nom = np.asarray(u_nom, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)
    norm_sq = float(a @ a)
    if norm_sq < tolerance:
        residual = float(a @ u_nom - b)
        status = "degenerate_feasible" if residual >= -tolerance else "degenerate_infeasible"
        return {
            "u_safe": clip_to_bounds(u_nom, lower_bounds, upper_bounds),
            "status": status,
            "was_filtered": False,
            "residual": residual,
            "solve_time": time.perf_counter() - start,
            "active_constraints": [] if residual >= -tolerance else [0],
        }
    residual_nom = float(a @ u_nom - b)
    if residual_nom >= -tolerance:
        u = u_nom.copy()
        was_filtered = False
    else:
        u = u_nom + ((float(b) - float(a @ u_nom)) / norm_sq) * a
        was_filtered = True
    if lower_bounds is not None or upper_bounds is not None:
        u = clip_to_bounds(u, lower_bounds, upper_bounds)
    residual = float(a @ u - b)
    status = "optimal" if residual >= -tolerance else "bound_clipped_may_be_infeasible"
    return {
        "u_safe": u,
        "status": status,
        "was_filtered": bool(was_filtered or np.linalg.norm(u - u_nom) > tolerance),
        "residual": residual,
        "solve_time": time.perf_counter() - start,
        "active_constraints": [0] if abs(residual) <= 1.0e-6 or was_filtered else [],
    }
