"""Acceleration-level relative-degree-2 HOCBF constraint."""
from __future__ import annotations

import numpy as np

from ..safety_data.sample import SafetySample


def build_acceleration_hocbf_constraint(safety: SafetySample, velocity, alpha1: float, alpha2: float):
    """Build the HOCBF inequality for p_dot = v, v_dot = a.

    The constraint is returned in the internal form ``A a >= b`` where:

        A = grad_h^T
        b = -v^T H v - (alpha1 + alpha2) grad_h^T v - alpha1 alpha2 h
    """
    from .builders import Constraint

    if safety.hessian_h is None:
        raise ValueError("Acceleration HOCBF requires safety.hessian_h.")
    v_in = velocity.velocity if hasattr(velocity, "velocity") else velocity
    v = np.asarray(v_in, dtype=float)
    grad = np.asarray(safety.grad_h, dtype=float)
    H = np.asarray(safety.hessian_h, dtype=float)
    if H.shape != (grad.size, grad.size):
        raise ValueError("safety.hessian_h has incompatible shape.")
    h_dot = float(grad @ v)
    curvature = float(v.T @ H @ v)
    rhs = -curvature - (alpha1 + alpha2) * h_dot - alpha1 * alpha2 * float(safety.h)
    metadata = {
        "alpha1": float(alpha1),
        "alpha2": float(alpha2),
        "h": float(safety.h),
        "h_dot": h_dot,
        "curvature": curvature,
        "rhs": float(rhs),
        "psi1": h_dot + alpha1 * float(safety.h),
    }
    return Constraint(A=grad.reshape(1, -1), b=np.array([rhs], dtype=float), name="acceleration_hocbf", metadata=metadata)
