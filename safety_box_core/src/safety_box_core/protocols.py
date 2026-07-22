"""Structural protocols shared by independently installable safety boxes."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from .types import ConstraintBundle, StateSnapshot


@runtime_checkable
class ControlAffineModel(Protocol):
    """Model contract for ``x_dot = f(x,t) + g(x,t)u``."""

    state_dimension: int
    control_dimension: int

    def f(self, x: np.ndarray, time_s: float = 0.0) -> np.ndarray: ...

    def g(self, x: np.ndarray, time_s: float = 0.0) -> np.ndarray: ...

    def linearize(
        self,
        x_star: np.ndarray,
        u_star: np.ndarray,
        time_s: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]: ...


@runtime_checkable
class ConstraintProvider(Protocol):
    """Box contract for modules that contribute affine constraints."""

    enabled: bool
    name: str

    def prepare(self, context: Any = None) -> None: ...

    def evaluate(
        self,
        state: StateSnapshot,
        context: Any = None,
    ) -> ConstraintBundle: ...
