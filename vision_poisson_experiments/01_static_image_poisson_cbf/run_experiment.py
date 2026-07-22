#!/usr/bin/env python3
"""Command-line entry point for the static photograph Poisson + CBF experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
from typing import Any

# Allow direct execution from the repository root without requiring installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.io_utils import load_yaml, make_run_directory, parse_float_pair, resolve_path, setup_logging
from common.package_bootstrap import ensure_safety_boxes_importable
from common.static_pipeline import run_static_experiment


def build_argument_parser() -> argparse.ArgumentParser:
    """Create a CLI whose overrides mirror the experiment's main physical choices."""

    parser = argparse.ArgumentParser(
        description=(
            "Rectify a 2D photograph, build boolean occupancy, synthesize Poisson "
            "safety fields and derivatives, save scientific diagnostics, and run "
            "an optional numerical Poisson-driven CBF-QP demonstration."
        )
    )
    parser.add_argument("--image", required=True, help="Input photograph or already top-down image.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="YAML experiment configuration.",
    )
    parser.add_argument("--mask", help="Override segmentation with a binary obstacle-mask file.")
    parser.add_argument("--background", help="Override the frozen empty-workspace reference image.")
    parser.add_argument(
        "--segmentation-mode",
        choices=["mask_file", "manual_polygon", "hsv", "background_reference"],
        help="Override segmentation.mode while preserving the remaining YAML parameters.",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["assume_top_down", "interactive", "load_file"],
        help="Override calibration.mode. The explicit --assume-top-down flag takes precedence.",
    )
    parser.add_argument("--output", help="Write directly to this run directory instead of creating a timestamped one.")
    parser.add_argument(
        "--assume-top-down",
        action="store_true",
        help="Explicitly map the full image rectangle to the metric workspace.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable calibration, manual masks, correction, and start/goal windows.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable every interactive OpenCV window.")
    parser.add_argument("--no-cbf", action="store_true", help="Skip the static CBF simulation.")
    parser.add_argument("--start", help="Override CBF start as comma-separated x,y meters.")
    parser.add_argument("--goal", help="Override CBF goal as comma-separated x,y meters.")
    parser.add_argument(
        "--forcing",
        action="append",
        choices=["constant", "distance", "average_flux", "guidance"],
        help="Forcing method to run; repeat the option to compare multiple methods.",
    )
    parser.add_argument(
        "--selected-forcing",
        choices=["constant", "distance", "average_flux", "guidance"],
        help=(
            "Poisson field consumed by the CBF demonstration. When omitted and --forcing "
            "is used, the first requested forcing method is selected automatically."
        ),
    )
    parser.add_argument("--solver", choices=["sor", "sparse_direct", "conjugate_gradient"])
    parser.add_argument(
        "--compare-solvers",
        dest="compare_solvers",
        action="store_true",
        help="Run the configured Poisson solver comparison for the selected field.",
    )
    parser.add_argument(
        "--no-compare-solvers",
        dest="compare_solvers",
        action="store_false",
        help="Skip the optional Poisson solver comparison.",
    )
    parser.set_defaults(compare_solvers=None)

    # Frequently tuned scalar overrides.  Values omitted here remain controlled by YAML.
    parser.add_argument("--workspace-width", type=float, help="Workspace width in meters.")
    parser.add_argument("--workspace-height", type=float, help="Workspace height in meters.")
    parser.add_argument("--nx", type=int, help="Number of Poisson nodes along x.")
    parser.add_argument("--ny", type=int, help="Number of Poisson nodes along y.")
    parser.add_argument("--robot-radius", type=float, help="Robot footprint radius in meters.")
    parser.add_argument("--perception-margin", type=float, help="Additional perception margin in meters.")
    parser.add_argument("--cbf-alpha", type=float, help="Velocity-CBF class-K gain.")
    parser.add_argument("--dt", type=float, help="CBF simulation integration step in seconds.")
    parser.add_argument("--max-speed", type=float, help="CBF simulation maximum speed in m/s.")
    parser.add_argument("--max-steps", type=int, help="CBF simulation maximum integration steps.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging and tracebacks.")
    return parser


def _apply_cli_overrides(config: dict[str, Any], arguments: argparse.Namespace) -> dict[str, Any]:
    """Apply only explicit CLI values while preserving the YAML as the source of truth."""

    workspace = config.setdefault("workspace", {})
    if arguments.workspace_width is not None:
        workspace["width_m"] = arguments.workspace_width
    if arguments.workspace_height is not None:
        workspace["height_m"] = arguments.workspace_height
    grid = workspace.setdefault("grid", {})
    if arguments.nx is not None:
        grid["nx"] = arguments.nx
    if arguments.ny is not None:
        grid["ny"] = arguments.ny

    # Both "occupancy" and "geometry" are accepted by the shared pipeline.
    # Do not create an empty ``occupancy`` section when no CLI override was
    # provided: doing so would shadow valid radius values stored in ``geometry``.
    if arguments.robot_radius is not None or arguments.perception_margin is not None:
        occupancy = config.setdefault("occupancy", dict(config.get("geometry", {})))
        if arguments.robot_radius is not None:
            occupancy["robot_radius_m"] = arguments.robot_radius
        if arguments.perception_margin is not None:
            occupancy["perception_margin_m"] = arguments.perception_margin

    calibration = config.setdefault("calibration", {})
    if arguments.calibration_mode:
        calibration["mode"] = arguments.calibration_mode

    segmentation = config.setdefault("segmentation", {})
    if arguments.segmentation_mode:
        segmentation["mode"] = arguments.segmentation_mode
    if arguments.mask:
        segmentation["mode"] = "mask_file"
        segmentation["mask_file"] = str(Path(arguments.mask).expanduser().resolve())
    if arguments.background:
        segmentation["reference_file"] = str(Path(arguments.background).expanduser().resolve())

    if arguments.compare_solvers is not None:
        config.setdefault("poisson", {})["compare_solvers"] = bool(arguments.compare_solvers)

    cbf = config.setdefault("cbf", {})
    if arguments.cbf_alpha is not None:
        cbf["alpha"] = arguments.cbf_alpha
    if arguments.dt is not None:
        cbf["dt_s"] = arguments.dt
    if arguments.max_speed is not None:
        cbf["maximum_speed_mps"] = arguments.max_speed
    if arguments.max_steps is not None:
        cbf["maximum_steps"] = arguments.max_steps
    return config


def run(arguments: argparse.Namespace) -> Path:
    """Resolve files, execute the static pipeline, and return the run directory."""

    ensure_safety_boxes_importable()
    config_path = Path(arguments.config).expanduser().resolve()
    config = _apply_cli_overrides(load_yaml(config_path), arguments)

    if arguments.output:
        output = Path(arguments.output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
    else:
        configured_root = resolve_path(
            config.get("output", {}).get("root", "outputs"),
            base_directory=config_path.parent,
        )
        assert configured_root is not None
        output = make_run_directory(configured_root, prefix="static")

    logger = setup_logging(output, verbose=arguments.verbose)
    logger.info("Static experiment output: %s", output)

    # Interactive behavior must be requested explicitly.  This keeps the default
    # command deterministic in Docker and continuous-integration environments.
    headless = bool(arguments.headless or not arguments.interactive)
    report = run_static_experiment(
        image_path=arguments.image,
        config=config,
        output_directory=output,
        base_directory=config_path.parent,
        assume_top_down=arguments.assume_top_down,
        headless=headless,
        run_cbf=not arguments.no_cbf,
        start_override=parse_float_pair(arguments.start),
        goal_override=parse_float_pair(arguments.goal),
        forcing_methods_override=arguments.forcing,
        solver_override=arguments.solver,
        selected_forcing_override=(
            arguments.selected_forcing
            or (arguments.forcing[0] if arguments.forcing else None)
        ),
    )
    logger.info("Selected forcing method: %s", report.poisson.selected_method)
    logger.info("Generated %d forcing-method records.", len(report.poisson.records))
    if report.cbf is not None:
        logger.info("Nominal trajectory status: %s", report.cbf.nominal.status)
        logger.info("CBF trajectory status: %s", report.cbf.safe.status)
    return report.output_directory


def main() -> int:
    """CLI entry point with clear failures and optional full tracebacks."""

    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        output = run(arguments)
        print(f"OUTPUT_DIRECTORY={output}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if arguments.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
