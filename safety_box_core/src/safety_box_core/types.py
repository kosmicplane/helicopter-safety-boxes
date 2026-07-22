"""Canonical immutable contracts shared by all safety boxes.

All affine constraints use exactly one convention:

    A z >= b

where ``z`` is an augmented decision vector described by :class:`DecisionLayout`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np


class BoxStatus(str, Enum):
    READY = "ready"
    DISABLED = "disabled"
    HOLD = "hold"
    INVALID = "invalid"
    INFEASIBLE = "infeasible"
    ERROR = "error"


class ConstraintHardness(str, Enum):
    HARD = "hard"
    SOFT = "soft"


def _immutable_array(value: np.ndarray | Iterable[float], *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected {ndim} dimensions, received {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Array contains NaN or infinity.")
    output = np.array(array, dtype=float, copy=True)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class AffineConstraint:
    A: np.ndarray
    b: np.ndarray
    name: str
    source_box: str
    equation_id: str
    hardness: ConstraintHardness = ConstraintHardness.HARD
    slack_group: str | None = None
    priority: int = 100
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.atleast_2d(np.asarray(self.A, dtype=float))
        vector = np.atleast_1d(np.asarray(self.b, dtype=float)).reshape(-1)
        if matrix.shape[0] != vector.size:
            raise ValueError("Constraint row count must equal b length.")
        if matrix.shape[1] < 1:
            raise ValueError("Constraint needs at least one decision column.")
        object.__setattr__(self, "A", _immutable_array(matrix, ndim=2))
        object.__setattr__(self, "b", _immutable_array(vector, ndim=1))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def rows(self) -> int:
        return int(self.A.shape[0])

    @property
    def decision_dimension(self) -> int:
        return int(self.A.shape[1])

    def residual(self, decision: np.ndarray) -> np.ndarray:
        return self.A @ np.asarray(decision, dtype=float).reshape(-1) - self.b


@dataclass(frozen=True, slots=True)
class ConstraintBundle:
    constraints: tuple[AffineConstraint, ...] = ()
    status: BoxStatus = BoxStatus.READY
    source_box: str = "unknown"
    message: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @classmethod
    def disabled(cls, source_box: str) -> "ConstraintBundle":
        return cls(status=BoxStatus.DISABLED, source_box=source_box, message="box disabled")

    @classmethod
    def hold(cls, source_box: str, message: str, **diagnostics: Any) -> "ConstraintBundle":
        return cls(status=BoxStatus.HOLD, source_box=source_box, message=message, diagnostics=diagnostics)


@dataclass(frozen=True, slots=True)
class DecisionLayout:
    blocks: Mapping[str, tuple[int, int]]

    def __post_init__(self) -> None:
        blocks = {str(k): (int(v[0]), int(v[1])) for k, v in self.blocks.items()}
        occupied: set[int] = set()
        for name, (start, stop) in blocks.items():
            if start < 0 or stop <= start:
                raise ValueError(f"Invalid decision block {name}: {(start, stop)}")
            indices = set(range(start, stop))
            if occupied & indices:
                raise ValueError(f"Decision block {name!r} overlaps another block.")
            occupied |= indices
        if occupied and occupied != set(range(max(occupied) + 1)):
            raise ValueError("Decision layout must be contiguous from zero.")
        object.__setattr__(self, "blocks", MappingProxyType(blocks))

    @classmethod
    def from_sizes(cls, **sizes: int) -> "DecisionLayout":
        cursor = 0
        blocks: dict[str, tuple[int, int]] = {}
        for name, size in sizes.items():
            size = int(size)
            if size <= 0:
                raise ValueError(f"Decision block {name!r} must have positive size.")
            blocks[name] = (cursor, cursor + size)
            cursor += size
        return cls(blocks)

    @property
    def dimension(self) -> int:
        return max((stop for _, stop in self.blocks.values()), default=0)

    def block_slice(self, name: str) -> slice:
        start, stop = self.blocks[name]
        return slice(start, stop)

    def has_block(self, name: str) -> bool:
        """Return whether a named block is present in the augmented decision."""

        return str(name) in self.blocks

    def scalar_index(self, name: str) -> int:
        start, stop = self.blocks[name]
        if stop - start != 1:
            raise ValueError(f"Decision block {name!r} is not scalar.")
        return start

    def lift_row(self, coefficients: np.ndarray, block: str) -> np.ndarray:
        coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
        result = np.zeros(self.dimension, dtype=float)
        target = self.block_slice(block)
        if target.stop - target.start != coefficients.size:
            raise ValueError(f"Coefficient size does not match block {block!r}.")
        result[target] = coefficients
        return result


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    x: np.ndarray
    time_s: float = 0.0
    version: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _immutable_array(self.x, ndim=1))
        if not np.isfinite(self.time_s):
            raise ValueError("time_s must be finite.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class EquilibriumTarget:
    identifier: str
    x_star: np.ndarray
    u_star: np.ndarray
    available: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_star", _immutable_array(self.x_star, ndim=1))
        object.__setattr__(self, "u_star", _immutable_array(self.u_star, ndim=1))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class CertificateEvaluation:
    """Local affine certificate derivative ``dot h = drift + grad_u @ u``."""

    identifier: str
    value: float
    drift: float
    control_gradient: np.ndarray
    available: bool = True
    source: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.value) or not np.isfinite(self.drift):
            raise ValueError("Certificate value and drift must be finite.")
        object.__setattr__(self, "control_gradient", _immutable_array(self.control_gradient, ndim=1))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class FilterResult:
    decision: np.ndarray
    nominal_decision: np.ndarray
    status: BoxStatus
    solver_status: str
    solve_time_s: float
    residuals: np.ndarray
    constraint_names: tuple[str, ...]
    active_constraints: tuple[str, ...] = ()
    message: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", _immutable_array(self.decision, ndim=1))
        object.__setattr__(self, "nominal_decision", _immutable_array(self.nominal_decision, ndim=1))
        residuals = np.asarray(self.residuals, dtype=float).reshape(-1)
        residuals = np.array(residuals, copy=True)
        residuals.setflags(write=False)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "constraint_names", tuple(self.constraint_names))
        object.__setattr__(self, "active_constraints", tuple(self.active_constraints))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
