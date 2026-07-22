"""Experimental backstepping-style relative-degree-2 safety helper."""
from __future__ import annotations

import numpy as np


def _get_grad(obj) -> np.ndarray:
    """Accept either a gradient vector or a SafetySample-like object."""
    return np.asarray(obj.grad_h if hasattr(obj, "grad_h") else obj, dtype=float)


def auxiliary_k1(grad_h, k1_type: str = "gradient_ascent", gain: float = 1.0, nominal_velocity: np.ndarray | None = None) -> np.ndarray:
    """Compute the auxiliary safe velocity field k1(p).

    ``grad_h`` can be a raw gradient vector or a SafetySample-like object.
    """
    grad = _get_grad(grad_h)
    if k1_type == "gradient_ascent":
        norm = np.linalg.norm(grad) + 1.0e-12
        return gain * grad / norm
    if k1_type == "nominal_tracking":
        if nominal_velocity is None:
            raise ValueError("nominal_tracking k1 requires nominal_velocity.")
        v = np.asarray(nominal_velocity, dtype=float)
        norm = np.linalg.norm(v)
        if norm > gain > 0:
            return gain * v / norm
        return v
    if k1_type == "zero":
        return np.zeros_like(grad)
    raise ValueError(f"Unsupported k1_type: {k1_type}")


def compute_backstepping_value(h, velocity: np.ndarray, mu: float = 1.0, k1=None, grad_h=None, k1_type: str = "gradient_ascent", k1_gain: float = 1.0, nominal_velocity: np.ndarray | None = None) -> dict:
    """Compute the backstepping candidate h_B and diagnostics.

    Compatible call patterns:
    - ``compute_backstepping_value(h_float, velocity, mu, grad_h=grad)``
    - ``compute_backstepping_value(safety_sample, velocity, mu, k1)``
    """
    if mu <= 0:
        raise ValueError("mu must be positive.")
    if hasattr(h, "h"):
        h_value = float(h.h)
        grad = np.asarray(h.grad_h, dtype=float)
    else:
        h_value = float(h)
        if grad_h is None and k1 is None:
            raise ValueError("grad_h or k1 must be provided when h is a scalar.")
        grad = None if grad_h is None else np.asarray(grad_h, dtype=float)
    v = np.asarray(velocity, dtype=float)
    if k1 is None:
        k1 = auxiliary_k1(grad, k1_type=k1_type, gain=k1_gain, nominal_velocity=nominal_velocity)
    else:
        k1 = np.asarray(k1, dtype=float)
    error = v - k1
    h_B = h_value - 0.5 / float(mu) * float(error @ error)
    return {
        "h_B": h_B,
        "k1": k1,
        "velocity_error": error,
        "velocity_error_norm": float(np.linalg.norm(error)),
        "mu": float(mu),
        "k1_type": k1_type,
    }
