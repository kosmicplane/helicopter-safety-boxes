"""Single-file configuration and reproducibility helpers for the workspace."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


def deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def apply_dotted_override(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError("Override must have form dotted.key=value.")
    key, raw_value = expression.split("=", 1)
    if not key or key.startswith(".") or key.endswith("."):
        raise ValueError(f"Invalid dotted override key {key!r}.")
    cursor = config
    parts = key.split(".")
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot descend through non-mapping key {part!r}.")
        cursor = child
    cursor[parts[-1]] = yaml.safe_load(raw_value)


def load_experiment_config(
    path: str | Path,
    *,
    profile: str | None = None,
    overrides: Iterable[str] = (),
    validate: bool = True,
) -> dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError("The root configuration must be a mapping.")
    profiles = data.pop("profiles", {})
    if profile:
        if profile not in profiles:
            raise KeyError(f"Unknown profile {profile!r}.")
        data = deep_merge(data, profiles[profile])
    for override in overrides:
        apply_dotted_override(data, override)
    if validate:
        validate_experiment_config(data)
    return data


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    for section in ("runtime", "boxes", "filter", "visualization", "experiments"):
        if section not in config:
            raise ValueError(f"Missing configuration section {section!r}.")
    boxes = config["boxes"]
    for name in ("poisson", "cbf", "clf", "contingency"):
        if name not in boxes or "enabled" not in boxes[name]:
            raise ValueError(f"boxes.{name}.enabled must be explicit.")
    if boxes["contingency"]["enabled"] and not boxes["clf"]["enabled"]:
        raise ValueError("The contingency box requires CLF-derived ROA certificates.")
    for mode in ("predefined_world", "static_image", "live_vision"):
        if mode not in config["experiments"]:
            raise ValueError(f"experiments.{mode} is required.")
    required = int(boxes["contingency"].get("required_certified", 1))
    for mode in ("predefined_world", "static_image", "live_vision"):
        zones = config["experiments"][mode].get("landing_zones")
        if zones is None and mode == "predefined_world":
            zones = config["experiments"][mode]["world"]["landing_zones"]
        if required > len(zones):
            raise ValueError(f"r={required} exceeds the landing-zone count in {mode}.")


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def save_effective_config(config: Mapping[str, Any], path: str | Path) -> str:
    digest = config_hash(config)
    payload = deepcopy(dict(config))
    payload.setdefault("runtime_metadata", {})["config_sha256"] = digest
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return digest
