"""Input/output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: Any) -> None:
    """Save JSON data with indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file."""
    return yaml.safe_load(Path(path).read_text()) or {}


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Save arrays to a compressed NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
