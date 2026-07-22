"""Small math utilities."""
from __future__ import annotations

import numpy as np


def safe_norm(x: np.ndarray, eps: float = 1.0e-12) -> float:
    """Return ||x|| with an epsilon floor for safe division."""
    return float(np.linalg.norm(x) + eps)
