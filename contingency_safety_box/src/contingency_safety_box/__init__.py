"""Public API for combinatorial attraction-region contingency."""

from .box import (
    ContingencyBox,
    ContingencyBoxConfig,
    ContingencyEvaluation,
    maximum_margin,
    rth_largest,
)

__all__ = [
    "ContingencyBox",
    "ContingencyBoxConfig",
    "ContingencyEvaluation",
    "maximum_margin",
    "rth_largest",
]
