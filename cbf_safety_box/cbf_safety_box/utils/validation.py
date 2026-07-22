"""Validation helpers."""
import numpy as np


def require_finite_array(x, name: str) -> None:
    """Ensure an array contains only finite values."""
    arr = np.asarray(x)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf.")
