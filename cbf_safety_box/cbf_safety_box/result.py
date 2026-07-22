"""Result dataclass returned by the CBF Safety Box."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np


def _to_jsonable(value: Any) -> Any:
    """Convert numpy-heavy objects into JSON-friendly values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


@dataclass
class CBFBoxResult:
    """Output of a CBF safety-filter computation."""

    u_safe: np.ndarray
    u_nom: np.ndarray
    was_filtered: bool
    cbf_residual: float | None = None
    hocbf_residual: float | None = None
    constraint_matrix: np.ndarray | None = None
    constraint_vector: np.ndarray | None = None
    solver_status: str = "unknown"
    solve_time: float = 0.0
    active_constraints: list[int] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return _to_jsonable({
            "u_safe": self.u_safe,
            "u_nom": self.u_nom,
            "was_filtered": self.was_filtered,
            "cbf_residual": self.cbf_residual,
            "hocbf_residual": self.hocbf_residual,
            "constraint_matrix": self.constraint_matrix,
            "constraint_vector": self.constraint_vector,
            "solver_status": self.solver_status,
            "solve_time": self.solve_time,
            "active_constraints": self.active_constraints,
            "diagnostics": self.diagnostics,
        })

    def save_json(self, path: str | Path) -> None:
        """Save the result as a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
