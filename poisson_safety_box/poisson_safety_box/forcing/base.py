"""Base data structures for forcing functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class ForcingResult:
    """Output from a forcing builder."""

    forcing: np.ndarray
    diagnostics: Dict[str, Any] = field(default_factory=dict)
