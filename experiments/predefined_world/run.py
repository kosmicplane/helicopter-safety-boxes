#!/usr/bin/env python3
"""Run the predefined 3-D landing experiment and generate paper-ready figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from safety_box_core import EquilibriumTarget
from experiments.common.cli import (
    add_common_arguments,
    load_config_from_arguments,
    make_output_directory,
    save_run_metadata,
)
from experiments.common.controller import LandingController
from experiments.common.poisson_field import (
    compare_forcing_methods,
    compare_poisson_solvers,
    compute_poisson_field,
)
from experiments.common.simulation import run_simulation
from experiments.common.nominal_planner import ObstacleAwareNominalPlanner
from experiments.common.plotting import (
    configure_exports,
    plot_clf_phase_portraits,
    plot_clf_roa_projections,
    plot_contingency_maps,
    plot_forcing_comparison,
    plot_integrated_dashboard,
    plot_occupancy_boundary_slices,
    plot_poisson_diagnostics,
    plot_poisson_isosurfaces,
    plot_poisson_planes,
    plot_solver_comparison,
    plot_time_histories,
    plot_trajectory_views,
    plot_world_trajectory_3d,
)
from experiments.predefined_world.world import build_world, point_is_occupied


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run the predefined 3-D Poisson-HOCBF, CLF, and contingency landing study."
    )
    add_common_arguments(result)
    result.add_argument(
        "--compare",
        action="store_true",
        help="Also compare all configured forcing methods and Poisson solvers.",
    )
    result.add_argument(
        "--no-failure",
        action="store_true",
        help="Disable the planned landing-zone failure.",
    )
    return result


def _controller(
    config: dict,
    world: object,
    output: Path,
) -> LandingController:
    simulation = config["experiments"]["predefined_world"]["simulation"]
    return LandingController(
        dimension=3,
        targets=world.targets,
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
    output = make_output_directory(
        config=config,
        mode="predefined_world",
        explicit=arguments.output,
    )
    save_run_metadata(config=config, output=output, mode="predefined_world", command=sys.argv)
    figures = output / "figures"
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)

    world = build_world(config)
    poisson_config = config["boxes"]["poisson"]
    field = compute_poisson_field(
        world.occupancy,
        spacing=world.spacing,
        config=poisson_config,
    )
    field.save(data)
    controller = _controller(config, world, output)
    simulation_config = config["experiments"]["predefined_world"]["simulation"]
    planner_config = config.get("planner", {})
    nominal_planner = ObstacleAwareNominalPlanner(
        occupancy=world.occupancy,
        spacing=world.spacing,
        targets={target.identifier: target.x_star[:3] for target in world.targets},
        position_gain=float(planner_config.get("position_gain", 0.65)),
        velocity_gain=float(planner_config.get("velocity_gain", 1.0)),
        maximum_nominal_acceleration=float(
            planner_config.get("maximum_nominal_acceleration", 1.2)
        ),
        lookahead_distance_m=float(planner_config.get("lookahead_distance_m", 1.5)),
        clearance_weight=float(planner_config.get("clearance_weight", 2.0)),
        minimum_clearance_cells=float(planner_config.get("minimum_clearance_cells", 1.0)),
    )
    failure_time = None if arguments.no_failure else float(simulation_config["failure_time_s"])
    failed_target = None if arguments.no_failure else str(simulation_config["failed_target"])
    result = run_simulation(
        controller=controller,
        field=field,
        start_state=world.start_state,
        initial_target=str(simulation_config["initial_target"]),
        output_directory=data,
        dt_s=float(simulation_config["dt_s"]),
        maximum_steps=int(simulation_config["maximum_steps"]),
        landing_position_tolerance=float(simulation_config["landing_position_tolerance_m"]),
        landing_speed_tolerance=float(simulation_config["landing_speed_tolerance_mps"]),
        collision_check=lambda point: point_is_occupied(world, point),
        target_failure_time_s=failure_time,
        failed_target_id=failed_target,
        variant="full_failure" if failure_time is not None else "full",
        nominal_control_provider=nominal_planner.control,
    )

    dpi = int(config["visualization"]["dpi"])
    plot_world_trajectory_3d(
        world=world,
        metrics=result.metrics,
        events=result.events,
        controller=controller,
        directory=figures,
        dpi=dpi,
    )
    plot_trajectory_views(
        metrics=result.metrics,
        controller=controller,
        world=world,
        directory=figures,
        dpi=dpi,
    )
    plot_occupancy_boundary_slices(field=field, directory=figures, dpi=dpi)
    plot_poisson_planes(field=field, directory=figures, dpi=dpi)
    plot_poisson_isosurfaces(field=field, world=world, directory=figures, dpi=dpi)
    plot_clf_roa_projections(
        controller=controller,
        world=world,
        directory=figures,
        dpi=dpi,
    )
    plot_clf_phase_portraits(
        controller=controller,
        target_id=str(simulation_config["initial_target"]),
        directory=figures,
        dpi=dpi,
    )
    fixed_z = float(world.targets[0].x_star[2])
    plot_contingency_maps(
        controller=controller,
        world=world,
        fixed_z=fixed_z,
        directory=figures,
        dpi=dpi,
        grid_points=100 if arguments.quick else 150,
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
    forcing_fields: dict[str, object] = {field.forcing_method: field}
    if arguments.compare:
        sweeps = config["experiments"]["predefined_world"]["sweeps"]
        solver_records, solver_fields = compare_poisson_solvers(
            world.occupancy,
            spacing=world.spacing,
            config=poisson_config,
            solvers=sweeps["poisson_solvers"],
            forcing_method=str(sweeps["solver_forcing"]),
        )
        pd.DataFrame(solver_records).to_csv(data / "poisson_solver_comparison.csv", index=False)
        (data / "poisson_solver_comparison.json").write_text(
            json.dumps(solver_records, indent=2),
            encoding="utf-8",
        )
        plot_solver_comparison(solver_records, directory=figures, dpi=dpi)

        forcing_fields = compare_forcing_methods(
            world.occupancy,
            spacing=world.spacing,
            config=poisson_config,
            forcing_methods=sweeps["forcing_methods"],
            solver=str(poisson_config["solver"]),
        )
        for method, forcing_field in forcing_fields.items():
            method_output = data / "forcing" / method
            local_controller = _controller(config, world, method_output)
            local_planner = ObstacleAwareNominalPlanner(
                occupancy=world.occupancy,
                spacing=world.spacing,
                targets={target.identifier: target.x_star[:3] for target in world.targets},
                position_gain=float(planner_config.get("position_gain", 0.65)),
                velocity_gain=float(planner_config.get("velocity_gain", 1.0)),
                maximum_nominal_acceleration=float(planner_config.get("maximum_nominal_acceleration", 1.2)),
                lookahead_distance_m=float(planner_config.get("lookahead_distance_m", 1.5)),
                clearance_weight=float(planner_config.get("clearance_weight", 2.0)),
                minimum_clearance_cells=float(planner_config.get("minimum_clearance_cells", 1.0)),
            )
            local_result = run_simulation(
                controller=local_controller,
                field=forcing_field,
                start_state=world.start_state,
                initial_target=str(simulation_config["initial_target"]),
                output_directory=method_output,
                dt_s=float(simulation_config["dt_s"]),
                maximum_steps=int(simulation_config["maximum_steps"]),
                landing_position_tolerance=float(simulation_config["landing_position_tolerance_m"]),
                landing_speed_tolerance=float(simulation_config["landing_speed_tolerance_mps"]),
                collision_check=lambda point: point_is_occupied(world, point),
                target_failure_time_s=failure_time,
                failed_target_id=failed_target,
                variant=f"forcing_{method}",
                nominal_control_provider=local_planner.control,
            )
            forcing_trajectories[method] = local_result.metrics
            forcing_summaries.append(
                {
                    **local_result.summary,
                    "forcing_method": method,
                    "poisson_wall_time_s": float(sum(forcing_field.result.timing.values())),
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

    plot_integrated_dashboard(
        world=world,
        field=field,
        metrics=result.metrics,
        controller=controller,
        solver_records=solver_records,
        directory=figures,
        dpi=dpi,
    )
    print(f"OUTPUT_DIRECTORY={output}")
    return output


def main() -> int:
    arguments = parser().parse_args()
    run(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
