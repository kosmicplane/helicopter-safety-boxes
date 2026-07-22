"""Metric computations for CBF-QP results."""
from __future__ import annotations

import numpy as np


def command_metrics(u_nom: np.ndarray, u_safe: np.ndarray) -> dict:
    """Return norms and correction magnitude for a safety-filter result."""
    u_nom = np.asarray(u_nom, dtype=float)
    u_safe = np.asarray(u_safe, dtype=float)
    return {
        "norm_u_nom": float(np.linalg.norm(u_nom)),
        "norm_u_safe": float(np.linalg.norm(u_safe)),
        "correction_norm": float(np.linalg.norm(u_safe - u_nom)),
    }


def hocbf_metrics(h: float, grad_h: np.ndarray, velocity: np.ndarray, hessian_h: np.ndarray, alpha1: float) -> dict:
    """Compute h_dot and psi_1 for HOCBF diagnostics."""
    grad = np.asarray(grad_h, dtype=float)
    v = np.asarray(velocity, dtype=float)
    H = np.asarray(hessian_h, dtype=float)
    h_dot = float(grad @ v)
    return {
        "h_dot": h_dot,
        "psi1": h_dot + alpha1 * float(h),
        "curvature": float(v.T @ H @ v),
    }
