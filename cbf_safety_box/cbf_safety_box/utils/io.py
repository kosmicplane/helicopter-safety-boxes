"""I/O helpers."""
from pathlib import Path
import json


def save_json(data, path):
    """Save a JSON file with parent directory creation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
