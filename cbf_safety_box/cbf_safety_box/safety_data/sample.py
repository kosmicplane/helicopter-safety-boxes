"""Dataclass representing h, grad_h, Hessian h at one state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SafetySample:
    """Safety-function values sampled at a specific point.

    This object is the expected bridge from a safety-function generator, such as
    ``poisson_safety_box``, into this CBF box.
    """

    h: float
    grad_h: np.ndarray
    hessian_h: np.ndarray | None = None
    laplacian_h: float | None = None
    partial_h_t: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.h = float(self.h)
        self.grad_h = np.asarray(self.grad_h, dtype=float).reshape(-1)
        if not np.isfinite(self.h):
            raise ValueError("SafetySample.h must be finite.")
        if self.grad_h.size == 0 or not np.all(np.isfinite(self.grad_h)):
            raise ValueError("SafetySample.grad_h must be a finite nonempty vector.")
        if self.hessian_h is not None:
            self.hessian_h = np.asarray(self.hessian_h, dtype=float)
            dim = self.grad_h.size
            if self.hessian_h.shape != (dim, dim):
                raise ValueError(f"SafetySample.hessian_h must have shape {(dim, dim)}.")
            if not np.all(np.isfinite(self.hessian_h)):
                raise ValueError("SafetySample.hessian_h contains NaN or inf.")
        if self.laplacian_h is not None:
            self.laplacian_h = float(self.laplacian_h)
        if self.partial_h_t is not None:
            self.partial_h_t = float(self.partial_h_t)

    @property
    def dimension(self) -> int:
        """Return the spatial/control dimension implied by grad_h."""
        return int(self.grad_h.size)
