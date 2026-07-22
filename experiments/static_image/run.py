#!/usr/bin/env python3
"""Run the reproducible static-image CLF/Poisson landing experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.common.cli import (
    add_common_arguments,
    load_config_from_arguments,
    make_output_directory,
    save_run_metadata,
)
from experiments.common.controller import LandingController
from experiments.common.plotting import (
    configure_exports,
    plot_clf_phase_portraits,
    plot_clf_roa_projections,
    plot_contingency_maps,
    plot_forcing_comparison,
    plot_image_pipeline,
    plot_live_summary,
    plot_poisson_diagnostics,
    plot_poisson_planes,
    plot_solver_comparison,
    plot_static_dashboard,
    plot_time_histories,
)
from experiments.common.poisson_field import compare_forcing_methods, compare_poisson_solvers
from experiments.common.simulation import run_simulation
from experiments.static_image.pipeline import (
    PlanarWorld,
    build_static_image_products,
    point_is_occupied,
    save_perception_products,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build a Poisson safety field from one image and simulate CLF/ROA landing."
    )
    add_common_arguments(result)
    result.add_argument(
        "--image",
        default="experiments/static_image/input/example_scene.png",
        help="Input image. The configured mask_file is used by the default reproducible example.",
    )
    result.add_argument("--compare", action="store_true", help="Compare forcing methods and Poisson solvers.")
    result.add_argument("--no-failure", action="store_true", help="Disable planned landing-zone failure.")
    return result


def make_controller(config: dict, targets: tuple, output: Path) -> LandingController:
    simulation = config["experiments"]["static_image"]["simulation"]
    return LandingController(
        dimension=2,
        targets=targets,
        box_config=config["boxes"],
        filter_config=config["filter"],
        artifact_directory=output / "clf_artifacts",
        maximum_acceleration=float(simulation["maximum_acceleration_mps2"]),
        maximum_speed_component=float(simulation["maximum_speed_component_mps"]),
        nominal_position_gain=float(simulation["nominal_position_gain"]),
        nominal_velocity_gain=float(simulation["nominal_velocity_gain"]),
    )


def run(arguments: argparse.Namespace) -> Path:
    config, _ = load_config_from_arguments(arguments)
    configure_exports(
        pdf=bool(config.get("visualization", {}).get("save_pdf", True)),
        svg=bool(config.get("visualization", {}).get("save_svg", True)),
    )
    output = make_output_directory(config=config, mode="static_image", explicit=arguments.output)
    save_run_metadata(config=config, output=output, mode="static_image", command=sys.argv)
    figures = output / "figures"
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)

    experiment = config["experiments"]["static_image"]
    products = build_static_image_products(
        image_path=arguments.image,
        experiment_config=experiment,
        poisson_config=config["boxes"]["poisson"],
        repository_root=REPOSITORY_ROOT,
    )
    save_perception_products(products, data)
    world = PlanarWorld(
        extent_m=tuple(float(value) for value in experiment["workspace_size_m"]),
        targets=products.targets,
    )
    controller = make_controller(config, products.targets, output)
    simulation = experiment["simulation"]
    failure_time = None if arguments.no_failure else float(simulation["failure_time_s"])
    failed_target = None if arguments.no_failure else str(simulation["failed_target"])
    result = run_simulation(
        controller=controller,
        field=products.poisson_field,
        start_state=np.asarray(experiment["start_state"], dtype=float),
        initial_target=str(simulation["initial_target"]),
        output_directory=data,
        dt_s=float(simulation["dt_s"]),
        maximum_steps=int(simulation["maximum_steps"]),
        landing_position_tolerance=float(simulation["landing_position_tolerance_m"]),
        landing_speed_tolerance=float(simulation["landing_speed_tolerance_mps"]),
        collision_check=lambda point: point_is_occupied(products, point),
        target_failure_time_s=failure_time,
        failed_target_id=failed_target,
        variant="static_image_failure" if failure_time is not None else "static_image",
    )

    dpi = int(config["visualization"]["dpi"])
    target_positions = {target.identifier: target.x_star[:2] for target in products.targets}
    plot_image_pipeline(
        image_bgr=products.rectified_image_bgr,
        raw_mask=products.segmentation.clean_mask,
        occupancy=products.occupancy.inflated_occupancy,
        field=products.poisson_field,
        metrics=result.metrics,
        target_positions=target_positions,
        workspace_size_m=experiment["workspace_size_m"],
        directory=figures,
        dpi=dpi,
    )
    plot_poisson_planes(field=products.poisson_field, directory=figures, dpi=dpi)
    plot_clf_roa_projections(controller=controller, world=world, directory=figures, dpi=dpi)
    plot_clf_phase_portraits(
        controller=controller,
        target_id=str(simulation["initial_target"]),
        directory=figures,
        dpi=dpi,
    )
    plot_contingency_maps(
        controller=controller,
        world=world,
        directory=figures,
        dpi=dpi,
        grid_points=80 if arguments.quick else 130,
    )
    plot_time_histories(
        metrics=result.metrics,
        events=result.events,
        controller=controller,
        directory=figures,
        dpi=dpi,
    )

    solver_records: list[dict] = []
    forcing_summaries: list[dict] = []
    forcing_trajectories: dict[str, pd.DataFrame] = {}
    forcing_fields = {products.poisson_field.forcing_method: products.poisson_field}
    occupancy_xy = products.occupancy.inflated_occupancy.T
    spacing_xy = (products.geometry.dx, products.geometry.dy)
    if arguments.compare:
        sweep = config["experiments"]["predefined_world"]["sweeps"]
        solver_records, _ = compare_poisson_solvers(
            occupancy_xy,
            spacing=spacing_xy,
            config=config["boxes"]["poisson"],
            solvers=sweep["poisson_solvers"],
            forcing_method=str(sweep["solver_forcing"]),
        )
        pd.DataFrame(solver_records).to_csv(data / "poisson_solver_comparison.csv", index=False)
        (data / "poisson_solver_comparison.json").write_text(json.dumps(solver_records, indent=2), encoding="utf-8")
        plot_solver_comparison(solver_records, directory=figures, dpi=dpi)

        forcing_fields = compare_forcing_methods(
            occupancy_xy,
            spacing=spacing_xy,
            config=config["boxes"]["poisson"],
            forcing_methods=sweep["forcing_methods"],
            solver=str(config["boxes"]["poisson"]["solver"]),
        )
        for method, field in forcing_fields.items():
            local_output = data / "forcing" / method
            local_controller = make_controller(config, products.targets, local_output)
            local_result = run_simulation(
                controller=local_controller,
                field=field,
                start_state=np.asarray(experiment["start_state"], dtype=float),
                initial_target=str(simulation["initial_target"]),
                output_directory=local_output,
                dt_s=float(simulation["dt_s"]),
                maximum_steps=int(simulation["maximum_steps"]),
                landing_position_tolerance=float(simulation["landing_position_tolerance_m"]),
                landing_speed_tolerance=float(simulation["landing_speed_tolerance_mps"]),
                collision_check=lambda point: point_is_occupied(products, point),
                target_failure_time_s=failure_time,
                failed_target_id=failed_target,
                variant=f"forcing_{method}",
            )
            forcing_trajectories[method] = local_result.metrics
            forcing_summaries.append(
                {
                    **local_result.summary,
                    "forcing_method": method,
                    "poisson_wall_time_s": float(sum(field.result.timing.values())),
                }
            )
        pd.DataFrame(forcing_summaries).to_csv(data / "forcing_comparison.csv", index=False)
        plot_poisson_diagnostics(fields=forcing_fields, directory=figures, dpi=dpi)
        plot_forcing_comparison(
            fields=forcing_fields,
            summaries=forcing_summaries,
            trajectories=forcing_trajectories,
            world=world,
            directory=figures,
            dpi=dpi,
        )

    plot_static_dashboard(
        image_bgr=products.rectified_image_bgr,
        field=products.poisson_field,
        metrics=result.metrics,
        controller=controller,
        workspace_size_m=experiment["workspace_size_m"],
        solver_records=solver_records,
        directory=figures,
        dpi=dpi,
    )
    plot_live_summary(result.metrics, directory=figures, dpi=dpi)
    print(f"OUTPUT_DIRECTORY={output}")
    return output


def main() -> int:
    run(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
