r"""Configurable class-:math:`\mathcal K` CLF decrease functions.

Every implementation maps nonnegative Lyapunov values to nonnegative desired
rates.  The default linear choice corresponds to

.. math::

    \alpha_V(V) = c_V V.
r"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class AlphaFunction(Protocol):
    r"""Callable protocol shared by all CLF decrease functions.r"""

    name: str

    def __call__(self, value: ArrayLike) -> NDArray[np.float64]: ...


@dataclass(frozen=True, slots=True)
class LinearAlpha:
    r"""Linear decrease function ``alpha(V) = gain * V``.r"""

    gain: float = 0.04
    name: str = "linear"

    def __post_init__(self) -> None:
        if self.gain <= 0.0:
            raise ValueError("gain must be positive.")

    def __call__(self, value: ArrayLike) -> NDArray[np.float64]:
        return self.gain * np.asarray(value, dtype=float)


@dataclass(frozen=True, slots=True)
class PolynomialAlpha:
    r"""Polynomial decrease function ``c1 V + c2 V^2``.r"""

    linear_gain: float = 0.025
    quadratic_gain: float = 1.0e-4
    name: str = "polynomial"

    def __post_init__(self) -> None:
        if self.linear_gain <= 0.0 or self.quadratic_gain < 0.0:
            raise ValueError("linear_gain must be positive and quadratic_gain nonnegative.")

    def __call__(self, value: ArrayLike) -> NDArray[np.float64]:
        v = np.asarray(value, dtype=float)
        return self.linear_gain * v + self.quadratic_gain * v * v


@dataclass(frozen=True, slots=True)
class RegularizedFiniteTimeAlpha:
    r"""Smooth finite-time-inspired decrease family.

    The regularization makes the expression finite and continuously evaluable
    at ``V = 0``.  It is included for controlled ablation studies; the linear
    family remains the default.
    r"""

    gain: float = 0.08
    exponent: float = 0.75
    epsilon: float = 1.0e-6
    name: str = "regularized_finite_time"

    def __post_init__(self) -> None:
        if self.gain <= 0.0:
            raise ValueError("gain must be positive.")
        if not 0.0 < self.exponent < 1.0:
            raise ValueError("exponent must lie strictly between zero and one.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

    def __call__(self, value: ArrayLike) -> NDArray[np.float64]:
        v = np.maximum(np.asarray(value, dtype=float), 0.0)
        return self.gain * (
            (v + self.epsilon) ** self.exponent
            - self.epsilon**self.exponent
        )


_ALPHA_REGISTRY: dict[str, type[LinearAlpha] | type[PolynomialAlpha] | type[RegularizedFiniteTimeAlpha]] = {
    "linear": LinearAlpha,
    "polynomial": PolynomialAlpha,
    "regularized_finite_time": RegularizedFiniteTimeAlpha,
}


def alpha_from_config(config: dict[str, object] | None) -> AlphaFunction:
    r"""Construct a decrease function from the central experiment config.r"""

    cfg = dict(config or {})
    kind = str(cfg.pop("type", "linear"))
    try:
        constructor = _ALPHA_REGISTRY[kind]
    except KeyError as exc:
        supported = ", ".join(sorted(_ALPHA_REGISTRY))
        raise ValueError(f"Unsupported alpha function {kind!r}; choose {supported}.") from exc
    return constructor(**cfg)  # type: ignore[arg-type]
