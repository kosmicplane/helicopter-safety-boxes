"""State dataclasses used by the CBF Safety Box."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SystemState:
    """Minimal system state required by the CBF filters.

    The box intentionally avoids assuming a full robot model.  For a velocity CBF,
    only ``position`` is conceptually needed.  For an acceleration HOCBF and the
    backstepping helper, ``velocity`` is also required.
    """

    position: np.ndarray
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=float)
        if self.acceleration is not None:
            self.acceleration = np.asarray(self.acceleration, dtype=float)
        if self.position.ndim != 1:
            raise ValueError("SystemState.position must be a 1D vector.")
        if self.velocity is not None and self.velocity.shape != self.position.shape:
            raise ValueError("SystemState.velocity must match position dimension.")
        if self.acceleration is not None and self.acceleration.shape != self.position.shape:
            raise ValueError("SystemState.acceleration must match position dimension.")
