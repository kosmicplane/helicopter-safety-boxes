"""File, configuration, logging, and serialization helpers."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


LOGGER_NAME = "vision_poisson_experiments"


def utc_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp with millisecond resolution."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")[:-4] + "Z"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject non-mapping top-level documents."""

    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at {yaml_path}, received {type(data).__name__}.")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def resolve_path(value: str | Path | None, *, base_directory: str | Path) -> Path | None:
    """Resolve an optional path relative to a specified base directory."""

    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base_directory) / path
    return path.resolve()


def make_run_directory(root: str | Path, *, prefix: str = "run") -> Path:
    """Create and return a unique timestamped output directory."""

    output_root = Path(root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_directory = output_root / f"{prefix}_{utc_timestamp()}"
    suffix = 1
    while run_directory.exists():
        run_directory = output_root / f"{prefix}_{utc_timestamp()}_{suffix:02d}"
        suffix += 1
    run_directory.mkdir(parents=True)
    return run_directory


def _jsonable(value: Any) -> Any:
    """Convert common scientific Python values into JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path: str | Path, data: Any) -> None:
    """Write JSON using deterministic indentation and scientific-type conversion."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def save_yaml(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write a YAML mapping after converting nonstandard scientific values."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(_jsonable(data), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    """Write mappings to a CSV file, creating a header from the first row if needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in materialized:
            for key in row:
                if key not in seen:
                    ordered.append(key)
                    seen.add(key)
        fieldnames = ordered
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _jsonable(value) for key, value in row.items()})


def setup_logging(output_directory: str | Path | None = None, *, verbose: bool = False) -> logging.Logger:
    """Configure one console handler and an optional run-local file handler."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if output_directory is not None:
        log_path = Path(output_directory) / "experiment.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_logger(component: str | None = None) -> logging.Logger:
    """Return the project logger or one of its named children."""

    return logging.getLogger(LOGGER_NAME if component is None else f"{LOGGER_NAME}.{component}")


def parse_float_pair(text: str | None) -> tuple[float, float] | None:
    """Parse a comma-separated ``x,y`` pair from a CLI argument."""

    if text is None:
        return None
    values = [piece.strip() for piece in text.split(",")]
    if len(values) != 2:
        raise ValueError(f"Expected two comma-separated values, received: {text!r}")
    return float(values[0]), float(values[1])
