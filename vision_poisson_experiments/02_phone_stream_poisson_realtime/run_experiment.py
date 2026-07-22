#!/usr/bin/env python3
"""Run the asynchronous phone/video-to-Poisson experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import traceback
from urllib.parse import urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.io_utils import load_yaml, make_run_directory, resolve_path, save_json, setup_logging
from common.live_pipeline import LivePoissonPipeline, parse_video_source
from common.contingency_live_pipeline import LiveContingencyPipeline


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""

    parser = argparse.ArgumentParser(
        description="Convert a webcam, phone stream, or video file into an asynchronously updated Poisson field."
    )
    parser.add_argument("--source", required=True, help="Webcam index, file path, HTTP/MJPEG URL, or RTSP URL.")
    parser.add_argument("--config", required=True, help="Path to the YAML experiment configuration.")
    parser.add_argument("--output", help="Optional output-root override.")
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV windows for testing or servers.")
    parser.add_argument("--max-frames", type=int, help="Stop after this many processed frames.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def _resolve_source(source_text: str, config_directory: Path) -> str | int:
    """Resolve local file sources while preserving URLs and webcam indices."""

    parsed = parse_video_source(source_text)
    if isinstance(parsed, int):
        return parsed
    scheme = urlparse(parsed).scheme.lower()
    if scheme in {"http", "https", "rtsp", "rtmp", "udp", "tcp"}:
        return parsed
    candidate = Path(parsed).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    relative_to_config = (config_directory / candidate).resolve()
    if relative_to_config.is_file():
        return str(relative_to_config)
    # Preserve non-file strings because OpenCV may support another backend-specific URI.
    return parsed


def run(args: argparse.Namespace) -> Path:
    """Execute the configured live pipeline and return its output directory."""

    config_path = Path(args.config).expanduser().resolve()
    config_directory = config_path.parent
    config = load_yaml(config_path)
    configured_root = resolve_path(config.get("output", {}).get("root", "outputs"), base_directory=config_directory)
    output_root = Path(args.output).expanduser().resolve() if args.output else configured_root
    assert output_root is not None
    output_directory = make_run_directory(output_root, prefix="live")
    logger = setup_logging(output_directory, verbose=args.verbose)
    source = _resolve_source(args.source, config_directory)
    logger.info("Live experiment output: %s", output_directory)
    logger.info("Opening source: %r", source)

    pipeline_class = (
        LiveContingencyPipeline
        if bool(config.get("reachability", {}).get("enabled", False))
        else LivePoissonPipeline
    )
    pipeline = pipeline_class(
        source=source,
        config=config,
        config_directory=config_directory,
        output_directory=output_directory,
        headless=args.headless,
        maximum_frames=args.max_frames,
    )
    report = pipeline.run()
    save_json(output_directory / "entrypoint_summary.json", asdict(report))
    logger.info("Live experiment completed: %s", report)
    return output_directory


def main() -> int:
    """CLI entry point with graceful interruption and optional traceback."""

    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        output = run(args)
        print(f"OUTPUT_DIRECTORY={output}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
