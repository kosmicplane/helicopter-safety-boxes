r"""Reduced-order control-affine models used by the CLF safety box.

The models in this module intentionally expose only the mathematical contract
required by :class:`clf_safety_box.CLFBox`:

``f(x, t)``
    Drift vector in ``x_dot = f(x, t) + g(x, t) u``.
``g(x, t)``
    Input distribution matrix.
``linearize(x_star, u_star, t)``
    Local matrices ``(A, B)`` about a controlled equilibrium.

Keeping the model interface independent from Poisson fields, image processing,
and target-contingency logic is a deliberate architectural boundary.
r"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DoubleIntegratorModel:
    r"""Cartesian double-integrator model.

    State and input are ordered as

    .. math::

        x = [p^\top, v^\top]^\top,\qquad u = a,

    with dynamics

    .. math::

        \dot p = v,\qquad \dot v = a.
    r"""

    spatial_dimension: int = 3

    def __post_init__(self) -> None:
        if self.spatial_dimension < 1:
            raise ValueError("spatial_dimension must be positive.")

    @property
    def state_dimension(self) -> int:
        return 2 * self.spatial_dimension

    @property
    def control_dimension(self) -> int:
        return self.spatial_dimension

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Return the constant state and input matrices ``(A, B)``.r"""

        d = self.spatial_dimension
        zeros = np.zeros((d, d), dtype=float)
        identity = np.eye(d, dtype=float)
        A = np.block([[zeros, identity], [zeros, zeros]])
        B = np.vstack([zeros, identity])
        return A, B

    def f(self, x: np.ndarray, time_s: float = 0.0) -> np.ndarray:
        r"""Evaluate the drift ``f(x) = [v, 0]``.r"""

        del time_s
        state = np.asarray(x, dtype=float).reshape(self.state_dimension)
        d = self.spatial_dimension
        return np.concatenate([state[d:], np.zeros(d, dtype=float)])

    def g(self, x: np.ndarray, time_s: float = 0.0) -> np.ndarray:
        r"""Evaluate the constant input matrix ``g(x) = B``.r"""

        del x, time_s
        return self.matrices()[1]

    def linearize(
        self,
        x_star: np.ndarray,
        u_star: np.ndarray,
        time_s: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Return the exact linear representation about any equilibrium.r"""

        del x_star, u_star, time_s
        return self.matrices()


@dataclass(frozen=True, slots=True)
class SingleIntegratorModel:
    r"""Cartesian single-integrator model ``x_dot = u``.r"""

    spatial_dimension: int = 2

    def __post_init__(self) -> None:
        if self.spatial_dimension < 1:
            raise ValueError("spatial_dimension must be positive.")

    @property
    def state_dimension(self) -> int:
        return self.spatial_dimension

    @property
    def control_dimension(self) -> int:
        return self.spatial_dimension

    def f(self, x: np.ndarray, time_s: float = 0.0) -> np.ndarray:
        del time_s
        state = np.asarray(x, dtype=float).reshape(self.state_dimension)
        return np.zeros_like(state)

    def g(self, x: np.ndarray, time_s: float = 0.0) -> np.ndarray:
        del x, time_s
        return np.eye(self.spatial_dimension, dtype=float)

    def linearize(
        self,
        x_star: np.ndarray,
        u_star: np.ndarray,
        time_s: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        del x_star, u_star, time_s
        return (
            np.zeros((self.spatial_dimension, self.spatial_dimension), dtype=float),
            np.eye(self.spatial_dimension, dtype=float),
        )
