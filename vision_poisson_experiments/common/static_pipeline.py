"""End-to-end static photograph experiment.

The orchestration is deliberately explicit: every intermediate data product is
saved, each Poisson method is validated independently, and the CBF demonstration
consumes numerical field samples rather than analytical or hard-coded barriers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .calibration import (
    CalibrationData,
    assume_top_down_calibration,
    draw_calibration_points,
    interactive_calibration,
    rectify_image,
)
from .cbf_demo import (
    CBFComparisonResult,
    CBFSimulationConfig,
    run_cbf_comparison,
    save_cbf_comparison,
    select_start_goal_interactive,
)
from .coordinates import GridFieldSampler, GridGeometry
from .io_utils import save_json, save_yaml
from .occupancy import OccupancyProducts, compute_occupancy_products
from .poisson_runner import (
    PoissonRunBundle,
    PoissonRunRecord,
    compare_poisson_solvers,
    run_poisson_methods,
    save_poisson_record,
)
from .poisson_visualization import (
    save_forcing_comparison,
    save_poisson_diagnostics,
    save_solver_comparison,
    save_static_input_figures,
)
from .segmentation import SegmentationResult, segment_image


@dataclass(frozen=True)
class StaticExperimentReport:
    """Programmatic result of a complete static experiment."""

    output_directory: Path
    geometry: GridGeometry
    calibration: CalibrationData
    segmentation: SegmentationResult
    occupancy: OccupancyProducts
    poisson: PoissonRunBundle
    solver_comparison: dict[str, PoissonRunRecord]
    cbf: CBFComparisonResult | None
    summary_path: Path


def _workspace_geometry(config: Mapping[str, Any]) -> tuple[GridGeometry, tuple[int, int]]:
    """Read canonical or legacy workspace/grid configuration."""

    workspace = dict(config.get("workspace", {}))
    width_m = float(workspace.get("width_m", config.get("workspace_width_m", 4.0)))
    height_m = float(workspace.get("height_m", config.get("workspace_height_m", 3.0)))

    nested_grid = dict(workspace.get("grid", {}))
    top_grid = dict(config.get("grid", {}))
    nx = int(
        workspace.get(
            "grid_nx",
            nested_grid.get("nx", top_grid.get("nx", config.get("Nx", 80))),
        )
    )
    ny = int(
        workspace.get(
            "grid_ny",
            nested_grid.get("ny", top_grid.get("ny", config.get("Ny", 60))),
        )
    )
    geometry = GridGeometry(width_m=width_m, height_m=height_m, nx=nx, ny=ny)

    rectified = dict(workspace.get("rectified_image", {}))
    calibration = dict(config.get("calibration", {}))
    width_px = int(
        workspace.get(
            "rectified_width_px",
            rectified.get("width_px", calibration.get("output_width_px", 640)),
        )
    )
    height_px = int(
        workspace.get(
            "rectified_height_px",
            rectified.get("height_px", calibration.get("output_height_px", 480)),
        )
    )
    if width_px < 2 or height_px < 2:
        raise ValueError("Rectified image dimensions must each be at least two pixels.")
    return geometry, (width_px, height_px)


def _normalized_segmentation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize concise and fully nested segmentation YAML forms."""

    source = dict(config.get("segmentation", {}))
    normalized = dict(source)
    mode = str(source.get("mode", "hsv"))
    normalized["mode"] = mode

    # Flat test-friendly HSV fields.
    hsv = dict(source.get("hsv", {}))
    if "hsv_lower" in source:
        hsv["lower"] = source["hsv_lower"]
    if "hsv_upper" in source:
        hsv["upper"] = source["hsv_upper"]
    normalized["hsv"] = hsv

    # Flat background-reference fields.
    background = dict(source.get("background", source.get("background_reference", {})))
    if "background_threshold" in source:
        background["difference_threshold"] = source["background_threshold"]
    if "background_blur_kernel" in source:
        background["difference_blur_kernel"] = source["background_blur_kernel"]
    normalized["background"] = background

    cleanup = dict(source.get("cleanup", {}))
    aliases = {
        "blur_kernel": "blur_kernel",
        "threshold": "threshold",
        "open_kernel": "open_kernel",
        "close_kernel": "close_kernel",
        "min_component_area_px": "minimum_component_area_px",
        "minimum_component_area_px": "minimum_component_area_px",
        "fill_holes": "fill_holes",
        "invert": "invert",
    }
    for old_key, new_key in aliases.items():
        if old_key in source:
            cleanup[new_key] = source[old_key]
    # Translate a legacy single morphology kernel when explicit kernels are absent.
    if "morphology_kernel" in cleanup:
        kernel = cleanup.pop("morphology_kernel")
        cleanup.setdefault("open_kernel", kernel)
        cleanup.setdefault("close_kernel", kernel)
    normalized["cleanup"] = cleanup

    if "mask_path" in source and "mask_file" not in normalized:
        normalized["mask_file"] = source["mask_path"]
    if "background_path" in source and "reference_file" not in normalized:
        normalized["reference_file"] = source["background_path"]
    if isinstance(source.get("manual_correction"), bool):
        normalized["manual_correction"] = {"enabled": bool(source["manual_correction"])}
    return normalized


def _resolve_calibration(
    image: np.ndarray,
    config: Mapping[str, Any],
    geometry: GridGeometry,
    rectified_size_px: tuple[int, int],
    *,
    base_directory: Path,
    assume_top_down: bool,
    headless: bool,
) -> CalibrationData:
    """Load, construct, or interactively select a planar homography."""

    calibration_cfg = dict(config.get("calibration", {}))
    mode = str(calibration_cfg.get("mode", "assume_top_down")).lower()
    if mode in {"top_down", "identity"}:
        mode = "assume_top_down"
    if assume_top_down:
        mode = "assume_top_down"
    if mode == "assume_top_down":
        return assume_top_down_calibration(
            image.shape,
            output_size_px=rectified_size_px,
            workspace_size_m=geometry.workspace_size_m,
        )
    if mode in {"load", "load_file", "file"}:
        file_value = calibration_cfg.get("file")
        if not file_value:
            raise ValueError("Calibration mode 'load_file' requires calibration.file.")
        path = Path(file_value)
        if not path.is_absolute():
            path = base_directory / path
        return CalibrationData.load(path)
    if mode in {"interactive", "four_point"}:
        if headless:
            raise RuntimeError("Interactive four-point calibration is unavailable in headless mode.")
        return interactive_calibration(
            image,
            output_size_px=rectified_size_px,
            workspace_size_m=geometry.workspace_size_m,
        )
    raise ValueError(f"Unsupported calibration mode: {mode!r}")


def _cbf_configuration(data: Mapping[str, Any]) -> CBFSimulationConfig:
    """Translate canonical and legacy CBF fields into the simulation dataclass."""

    source = dict(data)
    translated = {
        "alpha": source.get("alpha", 3.0),
        "solver": source.get("solver", "scipy"),
        "goal_gain": source.get("goal_gain", 1.0),
        "dt_s": source.get("dt_s", source.get("dt", 0.03)),
        "maximum_steps": source.get("maximum_steps", source.get("max_steps", 1000)),
        "maximum_speed_mps": source.get("maximum_speed_mps", source.get("maximum_speed", 1.0)),
        "goal_tolerance_m": source.get("goal_tolerance_m", source.get("goal_tolerance", 0.08)),
        "residual_tolerance": source.get("residual_tolerance", 1.0e-7),
        "h_tolerance": source.get("h_tolerance", 1.0e-6),
        "enforce_control_bounds": source.get(
            "enforce_control_bounds",
            source.get("use_control_bounds", True),
        ),
        "component_bound_mode": source.get("component_bound_mode", "euclidean_conservative"),
        "maximum_step_backtracks": source.get("maximum_step_backtracks", 14),
        "minimum_integration_dt_s": source.get("minimum_integration_dt_s", 1.0e-6),
    }
    return CBFSimulationConfig.from_dict(translated)


def _start_goal(
    cbf_data: Mapping[str, Any],
    rectified: np.ndarray,
    geometry: GridGeometry,
    *,
    headless: bool,
    start_override: Iterable[float] | None,
    goal_override: Iterable[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve start and goal from CLI, interactive clicks, or YAML."""

    if start_override is not None or goal_override is not None:
        if start_override is None or goal_override is None:
            raise ValueError("Start and goal CLI overrides must be provided together.")
        return np.asarray(tuple(start_override), dtype=float), np.asarray(tuple(goal_override), dtype=float)
    interactive = bool(
        cbf_data.get("interactive_start_goal", cbf_data.get("select_start_goal_interactively", False))
    )
    if interactive:
        if headless:
            raise RuntimeError("Interactive start/goal selection is unavailable in headless mode.")
        return select_start_goal_interactive(rectified, geometry.workspace_size_m)
    start = cbf_data.get("start_xy_m", cbf_data.get("start_xy", [0.4, 0.4]))
    goal = cbf_data.get(
        "goal_xy_m",
        cbf_data.get("goal_xy", [geometry.width_m - 0.4, geometry.height_m - 0.4]),
    )
    return np.asarray(start, dtype=float).reshape(2), np.asarray(goal, dtype=float).reshape(2)


def run_static_experiment(
    *,
    image_path: str | Path,
    config: Mapping[str, Any],
    output_directory: str | Path,
    assume_top_down: bool = False,
    headless: bool = True,
    run_cbf: bool = True,
    base_directory: str | Path | None = None,
    start_override: Iterable[float] | None = None,
    goal_override: Iterable[float] | None = None,
    forcing_methods_override: Iterable[str] | None = None,
    solver_override: str | None = None,
    selected_forcing_override: str | None = None,
) -> StaticExperimentReport:
    """Run the complete image → occupancy → Poisson → CBF experiment."""

    source = Path(image_path).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = Path(base_directory).expanduser().resolve() if base_directory else source.parent
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {source}")

    geometry, rectified_size = _workspace_geometry(config)
    calibration = _resolve_calibration(
        image,
        config,
        geometry,
        rectified_size,
        base_directory=base,
        assume_top_down=assume_top_down,
        headless=headless,
    )
    calibration.save(output / "calibration.json")
    reusable_value = dict(config.get("calibration", {})).get("save_reusable_file")
    if reusable_value:
        reusable_path = Path(str(reusable_value)).expanduser()
        if not reusable_path.is_absolute():
            reusable_path = base / reusable_path
        calibration.save(reusable_path.resolve())
    calibration_overlay = draw_calibration_points(image, calibration.source_points_px)
    cv2.imwrite(str(output / "calibration_points.png"), calibration_overlay)
    rectified = rectify_image(image, calibration)
    cv2.imwrite(str(output / "rectified_image.png"), rectified)

    segmentation_cfg = _normalized_segmentation_config(config)
    segmentation = segment_image(
        rectified,
        segmentation_cfg,
        base_directory=base,
        allow_interactive=not headless,
    )
    cv2.imwrite(str(output / "raw_mask.png"), segmentation.raw_mask)
    cv2.imwrite(str(output / "clean_mask.png"), segmentation.clean_mask)

    # ``geometry`` is the historical name and ``occupancy`` is the canonical
    # experiment name.  Merge both so a partial occupancy override cannot erase
    # a robot radius or perception margin defined in the legacy section.
    occupancy_cfg = {
        **dict(config.get("geometry", {})),
        **dict(config.get("occupancy", {})),
    }
    occupancy = compute_occupancy_products(
        segmentation.clean_mask,
        geometry,
        robot_radius_m=float(occupancy_cfg.get("robot_radius_m", 0.0)),
        perception_margin_m=float(occupancy_cfg.get("perception_margin_m", 0.0)),
    )
    np.savez_compressed(
        output / "perception_and_occupancy.npz",
        raw_mask=segmentation.raw_mask,
        clean_mask=segmentation.clean_mask,
        occupancy_uninflated=occupancy.uninflated,
        occupancy_inflated=occupancy.inflated,
        grid_spacing_yx=np.asarray(geometry.spacing_yx),
        workspace_size_m=np.asarray(geometry.workspace_size_m),
    )

    plot_cfg = dict(config.get("visualization", config.get("plots", config.get("output", {}))))
    dpi = int(plot_cfg.get("dpi", plot_cfg.get("figure_dpi", 150)))
    save_static_input_figures(
        image,
        rectified,
        segmentation,
        occupancy,
        output / "preprocessing_figures",
        workspace_size_m=geometry.workspace_size_m,
        dpi=dpi,
    )

    poisson_cfg = dict(config.get("poisson", {}))
    methods = list(
        forcing_methods_override
        if forcing_methods_override is not None
        else poisson_cfg.get("forcing_methods", [poisson_cfg.get("forcing_method", "constant")])
    )
    solver = str(solver_override or poisson_cfg.get("solver", "sparse_direct"))
    selected_method = str(
        selected_forcing_override
        or poisson_cfg.get("cbf_forcing_method")
        or poisson_cfg.get("primary_forcing_method")
        or poisson_cfg.get("primary_method")
        or methods[0]
    )
    poisson = run_poisson_methods(
        occupancy.inflated,
        spacing_yx=geometry.spacing_yx,
        settings=poisson_cfg,
        forcing_methods=methods,
        selected_method=selected_method,
        solver=solver,
    )

    # Complete every numerical computation before rendering the large static
    # diagnostic suite.  Matplotlib can temporarily retain substantial RGB
    # buffers, so keeping the CBF and solver computations ahead of the plotting
    # phase makes runtime and memory use predictable in constrained containers.
    solver_records: dict[str, PoissonRunRecord] = {}
    compare_setting = poisson_cfg.get("compare_solvers", False)
    if bool(compare_setting):
        solvers = (
            list(compare_setting)
            if isinstance(compare_setting, (list, tuple))
            else poisson_cfg.get(
                "solver_comparison",
                ["sparse_direct", "conjugate_gradient", "sor"],
            )
        )
        solver_records = compare_poisson_solvers(
            occupancy.inflated,
            spacing_yx=geometry.spacing_yx,
            settings=poisson_cfg,
            forcing_method=poisson.selected_method,
            solvers=solvers,
        )

    cbf_result: CBFComparisonResult | None = None
    cbf_cfg = dict(config.get("cbf", {}))
    if run_cbf and bool(cbf_cfg.get("enabled", True)):
        start_xy, goal_xy = _start_goal(
            cbf_cfg,
            rectified,
            geometry,
            headless=headless,
            start_override=start_override,
            goal_override=goal_override,
        )
        cbf_result = run_cbf_comparison(
            poisson.selected.result,
            grid_spacing_yx=geometry.spacing_yx,
            start_xy=start_xy,
            goal_xy=goal_xy,
            config=_cbf_configuration(cbf_cfg),
        )

        # Save the CBF comparison before the much larger all-method figure set.
        # This ordering also makes it explicit that the controller consumes raw
        # numerical h and gradient values, never visualization-normalized data.
        cbf_output = output / "cbf_simulation"
        save_cbf_comparison(
            cbf_result,
            poisson_result=poisson.selected.result,
            rectified_bgr=rectified,
            workspace_size_m=geometry.workspace_size_m,
            output_directory=cbf_output,
            dpi=dpi,
        )
        sampler = GridFieldSampler(poisson.selected.result, geometry)
        direction = goal_xy - start_xy
        try:
            derivative_check = sampler.directional_derivative_check(
                start_xy,
                direction,
                epsilon_m=0.1 * min(geometry.spacing_yx),
            )
            save_json(cbf_output / "directional_derivative_check.json", derivative_check)
        except ValueError as error:
            save_json(
                cbf_output / "directional_derivative_check.json",
                {"valid": False, "reason": str(error)},
            )

    # Persist numerical arrays and render scientific diagnostics only after all
    # controller and solver computations have completed successfully.
    for method, record in poisson.records.items():
        method_directory = output / "poisson" / method
        save_poisson_record(record, method_directory)
        save_poisson_diagnostics(
            record,
            rectified,
            method_directory / "figures",
            workspace_size_m=geometry.workspace_size_m,
            dpi=dpi,
        )
    save_forcing_comparison(
        poisson.records,
        output / "poisson" / "forcing_comparison",
        workspace_size_m=geometry.workspace_size_m,
        dpi=dpi,
    )

    if solver_records:
        for solver_name, record in solver_records.items():
            save_poisson_record(record, output / "solver_comparison" / solver_name)
        save_json(
            output / "solver_comparison" / "summary.json",
            {
                name: {
                    "wall_time_s": record.wall_time_s,
                    "valid": record.validation.valid,
                    "exact_residual_max_abs": record.validation.get("residual_max_abs"),
                    "exact_residual_rms": record.validation.get("residual_l2"),
                    "boundary_abs_max": record.validation.get("boundary_abs_max"),
                    "solver_info": record.result.solver_info,
                }
                for name, record in solver_records.items()
            },
        )
        save_solver_comparison(
            solver_records,
            output / "solver_comparison",
            workspace_size_m=geometry.workspace_size_m,
            dpi=dpi,
        )

    effective_config = dict(config)
    effective_config["runtime"] = {
        "image_path": str(source),
        "output_directory": str(output),
        "headless": bool(headless),
        "forcing_methods": methods,
        "selected_forcing_method": poisson.selected_method,
        "solver": solver,
    }
    save_yaml(output / "effective_config.yaml", effective_config)
    summary = {
        "input_image": str(source),
        "output_directory": str(output),
        "workspace": {
            "width_m": geometry.width_m,
            "height_m": geometry.height_m,
            "grid_nx": geometry.nx,
            "grid_ny": geometry.ny,
            "spacing_yx": geometry.spacing_yx,
        },
        "segmentation": segmentation.metadata,
        "occupancy": occupancy.diagnostics,
        "poisson": {
            "selected_method": poisson.selected_method,
            "records": {
                method: {
                    "solver": record.solver,
                    "wall_time_s": record.wall_time_s,
                    "valid": record.validation.valid,
                    "validation": {
                        key: value
                        for key, value in record.validation.items()
                        if key not in {"residual_field", "central_stencil_residual_field"}
                    },
                    "timing": record.result.timing,
                    "solver_info": record.result.solver_info,
                }
                for method, record in poisson.records.items()
            },
        },
        "solver_comparison": {
            name: {
                "wall_time_s": record.wall_time_s,
                "valid": record.validation.valid,
                "exact_residual_max_abs": record.validation.get("residual_max_abs"),
                "exact_residual_rms": record.validation.get("residual_l2"),
                "boundary_abs_max": record.validation.get("boundary_abs_max"),
            }
            for name, record in solver_records.items()
        },
        "cbf": None
        if cbf_result is None
        else {
            "nominal_status": cbf_result.nominal.status,
            "safe_status": cbf_result.safe.status,
            "nominal_collided": cbf_result.nominal.collided,
            "safe_collided": cbf_result.safe.collided,
            "safe_reached_goal": cbf_result.safe.reached_goal,
        },
        "safety_scope": {
            "field": "static Poisson safety function",
            "controller_model": "single integrator" if cbf_result is not None else None,
            "dynamic_obstacle_guarantee": False,
        },
    }
    summary_path = output / "experiment_summary.json"
    save_json(summary_path, summary)
    plt.close("all")
    return StaticExperimentReport(
        output_directory=output,
        geometry=geometry,
        calibration=calibration,
        segmentation=segmentation,
        occupancy=occupancy,
        poisson=poisson,
        solver_comparison=solver_records,
        cbf=cbf_result,
        summary_path=summary_path,
    )


__all__ = ["StaticExperimentReport", "run_static_experiment"]
