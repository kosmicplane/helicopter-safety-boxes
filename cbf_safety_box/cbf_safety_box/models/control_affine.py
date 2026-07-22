"""Future-facing minimal structure for control-affine systems.

A full nonlinear control-affine CBF implementation would use:

    x_dot = f(x) + g(x)u
    L_f h + L_g h u >= -gamma(h)

This box currently focuses on reduced-order single/double integrator models.
"""
from dataclasses import dataclass
from typing import Callable

@dataclass
class ControlAffineModel:
    f: Callable
    g: Callable
