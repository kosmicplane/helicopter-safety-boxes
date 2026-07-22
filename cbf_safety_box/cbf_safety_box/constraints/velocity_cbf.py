"""Velocity-level CBF constraint for p_dot = v."""
from __future__ import annotations

import numpy as np

from ..safety_data.sample import SafetySample


def build_velocity_cbf_constraint(safety: SafetySample, alpha: float):
    """Build ``grad_h^T v >= -alpha h`` as ``A u >= b``.

    Parameters
    ----------
    safety:
        Scalar safety value and gradient sampled at the current position.
    alpha:
        Positive class-K gain for the linear CBF condition.
    """
    from .builders import Constraint

    grad = np.asarray(safety.grad_h, dtype=float).reshape(1, -1)
    b = np.array([-float(alpha) * float(safety.h)], dtype=float)
    return Constraint(A=grad, b=b, name="velocity_cbf", metadata={"alpha": alpha, "h": float(safety.h)})
