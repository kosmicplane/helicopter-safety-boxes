#!/usr/bin/env python3
"""Run one controlled 3-D landing scenario and generate paper-ready artifacts."""

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
from experiments.common.simulation import SimulationResult, run_simulation
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
from experiments.predefined_world.paper_figures import (
    plot_contingency_timeline,
    plot_landing_terminal_zoom,
    plot_methodology_overview,
    plot_obstacle_avoidance_result,
)
from experiments.predefined_world.scenarios import resolve_scenario
from experiments.predefined_world.world import (
    build_world,
    point_clearance_m,
    point_is_occupied,
    segment_collision_fraction,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run a predefined Mars-analog 3-D Poisson-HOCBF, CLF, and "
            "combinatorial-contingency landing scenario."
        )
    )
    add_common_arguments(result)
    result.add_argument(
        "--compare",
        action="store_true",
        help="Also compare configured Poisson forcing methods and solvers.",
    )
    result.add_argument(
        "--scenario",
        default=None,
        help="Scenario name from experiments.predefined_world.simulation.scenarios.",
    )
    result.add_argument(
        "--no-failure",
        action="store_true",
        help="Alias for --scenario baseline.",
    )
    return result


def _controller(config: dict, world: object, output: Path) -> LandingController:
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


def _planner(config: dict, world: object) -> ObstacleAwareNominalPlanner:
    planner_config = config.get("planner", {})
    return ObstacleAwareNominalPlanner(
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


def _mission_references(
    *,
    world: object,
    planner: ObstacleAwareNominalPlanner,
    initial_target: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    target = next(item for item in world.targets if item.identifier == initial_target)
    start = np.asarray(world.start_state[:3], dtype=float)
    goal = np.asarray(target.x_star[:3], dtype=float)
    direct_line = np.linspace(start, goal, 500)
    nominal = planner.plan(start, goal, initial_target)
    if not nominal.valid:
        raise RuntimeError(f"Nominal A* path is invalid: {nominal.reason}")
    metadata = {
        "initial_target": initial_target,
        "straight_line_collision_fraction": segment_collision_fraction(
            world, start, goal, samples=500
        ),
        "straight_line_is_collision_free": bool(
            segment_collision_fraction(world, start, goal, samples=500) == 0.0
        ),
        "nominal_path_points": int(nominal.points.shape[0]),
        "nominal_expanded_nodes": int(nominal.expanded_nodes),
        "nominal_path_reason": nominal.reason,
    }
    return direct_line, nominal.points, metadata


def _run_rollout(
    *,
    config: dict,
    world: object,
    field: object,
    output: Path,
    scenario_name: str,
    failure_schedule: tuple[dict, ...] | tuple,
    variant: str,
) -> tuple[SimulationResult, LandingController, ObstacleAwareNominalPlanner]:
    simulation = config["experiments"]["predefined_world"]["simulation"]
    controller = _controller(config, world, output)
    planner = _planner(config, world)
    result = run_simulation(
        controller=controller,
        field=field,
        start_state=world.start_state,
        initial_target=str(simulation["initial_target"]),
        output_directory=output,
        dt_s=float(simulation["dt_s"]),
        maximum_steps=int(simulation["maximum_steps"]),
        landing_position_tolerance=float(simulation["landing_position_tolerance_m"]),
        landing_speed_tolerance=float(simulation["landing_speed_tolerance_mps"]),
        collision_check=lambda point: point_is_occupied(world, point),
        clearance_query=lambda point: point_clearance_m(world, point),
        failure_schedule=failure_schedule,
        variant=variant,
        nominal_control_provider=planner.control,
    )
    result.summary["scenario"] = scenario_name
    return result, controller, planner


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
    scenario = resolve_scenario(
        config,
        "baseline" if arguments.no_failure else arguments.scenario,
    )
    save_run_metadata(
        config=config,
        output=output,
        mode=f"predefined_world:{scenario.name}",
        command=sys.argv,
    )
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
    simulation_config = config["experiments"]["predefined_world"]["simulation"]
    result, controller, nominal_planner = _run_rollout(
        config=config,
        world=world,
        field=field,
        output=data,
        scenario_name=scenario.name,
        failure_schedule=scenario.failure_schedule,
        variant=scenario.name,
    )
    direct_line, nominal_path, world_metadata = _mission_references(
        world=world,
        planner=nominal_planner,
        initial_target=str(simulation_config["initial_target"]),
    )
    world_metadata.update(
        {
            "world_name": world.name,
            "world_summary": world.summary,
            "world_file": world.source_file,
            "extent_m": list(world.extent_m),
            "grid_shape": list(world.occupancy.shape),
            "spacing_m": list(world.spacing),
            "obstacle_count": len(world.obstacles),
            "scenario": scenario.name,
            "scenario_description": scenario.description,
            "failure_schedule": list(scenario.failure_schedule),
        }
    )
    (data / "world_and_scenario_summary.json").write_text(
        json.dumps(world_metadata, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        data / "mission_reference_paths.npz",
        direct_line=direct_line,
        nominal_path=nominal_path,
    )

    dpi = int(config["visualization"]["dpi"])
    plot_world_trajectory_3d(
        world=world,
        metrics=result.metrics,
        events=result.events,
        summary=result.summary,
        controller=controller,
        nominal_path=nominal_path,
        direct_line=direct_line,
        directory=figures,
        dpi=dpi,
    )
    plot_trajectory_views(
        metrics=result.metrics,
        summary=result.summary,
        controller=controller,
        world=world,
        nominal_path=nominal_path,
        direct_line=direct_line,
        directory=figures,
        dpi=dpi,
    )
    plot_occupancy_boundary_slices(field=field, directory=figures, dpi=dpi)
    plot_poisson_planes(field=field, directory=figures, dpi=dpi)
    plot_poisson_isosurfaces(field=field, world=world, directory=figures, dpi=dpi)
    plot_clf_roa_projections(controller=controller, world=world, directory=figures, dpi=dpi)
    plot_clf_phase_portraits(
        controller=controller,
        target_id=str(simulation_config["initial_target"]),
        directory=figures,
        dpi=dpi,
    )
    plot_contingency_maps(
        controller=controller,
        world=world,
        fixed_z=float(world.targets[0].x_star[2]),
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
    plot_methodology_overview(
        world=world,
        field=field,
        controller=controller,
        metrics=result.metrics,
        direct_line=direct_line,
        nominal_path=nominal_path,
        directory=figures,
        dpi=dpi,
    )
    plot_obstacle_avoidance_result(
        world=world,
        metrics=result.metrics,
        events=result.events,
        controller=controller,
        summary=result.summary,
        direct_line=direct_line,
        nominal_path=nominal_path,
        directory=figures,
        dpi=dpi,
    )
    plot_contingency_timeline(
        world=world,
        metrics=result.metrics,
        events=result.events,
        controller=controller,
        summary=result.summary,
        directory=figures,
        dpi=dpi,
    )
    if result.summary.get("landed"):
        plot_landing_terminal_zoom(
            world=world,
            metrics=result.metrics,
            controller=controller,
            summary=result.summary,
            simulation_config=simulation_config,
            directory=figures,
            dpi=dpi,
        )

    solver_records: list[dict] = []
    forcing_summaries: list[dict] = []
    forcing_trajectories: dict[str, pd.DataFrame] = {}
    forcing_fields: dict[str, object] = {field.forcing_method: field}
    if arguments.compare:
        sweeps = config["experiments"]["predefined_world"]["sweeps"]
        solver_records, _ = compare_poisson_solvers(
            world.occupancy,
            spacing=world.spacing,
            config=poisson_config,
            solvers=sweeps["poisson_solvers"],
            forcing_method=str(sweeps["solver_forcing"]),
        )
        pd.DataFrame(solver_records).to_csv(
            data / "poisson_solver_comparison.csv", index=False
        )
        (data / "poisson_solver_comparison.json").write_text(
            json.dumps(solver_records, indent=2), encoding="utf-8"
        )
        plot_solver_comparison(solver_records, directory=figures, dpi=dpi)

        forcing_fields = compare_forcing_methods(
            world.occupancy,
            spacing=world.spacing,
            config=poisson_config,
            forcing_methods=sweeps["forcing_methods"],
            solver=str(poisson_config["solver"]),
        )
        baseline = resolve_scenario(config, "baseline")
        for method, forcing_field in forcing_fields.items():
            method_output = data / "forcing" / method
            local_result, _, _ = _run_rollout(
                config=config,
                world=world,
                field=forcing_field,
                output=method_output,
                scenario_name="baseline",
                failure_schedule=baseline.failure_schedule,
                variant=f"forcing_{method}",
            )
            forcing_trajectories[method] = local_result.metrics
            forcing_summaries.append(
                {
                    **local_result.summary,
                    "forcing_method": method,
                    "poisson_wall_time_s": float(sum(forcing_field.result.timing.values())),
                }
            )
        pd.DataFrame(forcing_summaries).to_csv(
            data / "forcing_comparison.csv", index=False
        )
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
    print(f"SCENARIO={scenario.name}")
    print(f"TERMINAL_STATUS={result.summary['terminal_status']}")
    print(f"OUTPUT_DIRECTORY={output}")
    return output


def main() -> int:
    run(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
