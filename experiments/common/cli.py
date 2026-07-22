"""Shared command-line and run-directory helpers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json

from safety_box_core import load_experiment_config, save_effective_config


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Central experiment YAML.",
    )
    parser.add_argument("--profile", help="Optional profile stored in the central YAML.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Dotted override, for example boxes.clf.alpha.gain=0.04.",
    )
    parser.add_argument("--output", help="Explicit output directory.")
    parser.add_argument("--quick", action="store_true", help="Use the smoke profile unless another profile is supplied.")


def load_config_from_arguments(arguments: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    profile = arguments.profile or ("smoke" if arguments.quick else None)
    config_path = Path(arguments.config).expanduser().resolve()
    config = load_experiment_config(
        config_path,
        profile=profile,
        overrides=arguments.overrides,
    )
    return config, config_path


def make_output_directory(
    *,
    config: dict[str, Any],
    mode: str,
    explicit: str | None,
) -> Path:
    if explicit:
        output = Path(explicit).expanduser().resolve()
    else:
        root = Path(config["runtime"].get("output_root", "outputs"))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (root / mode / timestamp).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_run_metadata(
    *,
    config: dict[str, Any],
    output: Path,
    mode: str,
    command: Iterable[str] | None = None,
) -> str:
    digest = save_effective_config(config, output / "effective_config.yaml")
    metadata = {
        "mode": mode,
        "config_sha256": digest,
        "command": list(command or ()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return digest
