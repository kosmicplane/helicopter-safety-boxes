#!/usr/bin/env python3
"""Run paper-oriented parameter sweeps for the predefined 3-D world."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.common.plotting import configure_exports
from experiments.common.cli import add_common_arguments, load_config_from_arguments, make_output_directory, save_run_metadata
from experiments.common.controller import LandingController
from experiments.common.poisson_field import compute_poisson_field
from experiments.common.simulation import run_simulation
from experiments.common.nominal_planner import ObstacleAwareNominalPlanner
from experiments.common.plotting import plot_parameter_sweep, plot_trajectory_family
from experiments.predefined_world.world import build_world, point_is_occupied


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run HOCBF and CLF alpha sweeps.")
    add_common_arguments(result)
    return result


def _run_case(config: dict, world: object, field: object, output: Path, label: str) -> tuple[dict, pd.DataFrame]:
    simulation = config["experiments"]["predefined_world"]["simulation"]
    controller = LandingController(
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
    planner_config = config.get("planner", {})
    nominal_planner = ObstacleAwareNominalPlanner(
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
        target_failure_time_s=float(simulation["failure_time_s"]),
        failed_target_id=str(simulation["failed_target"]),
        variant=label,
        nominal_control_provider=nominal_planner.control,
    )
    return result.summary, result.metrics


def run(arguments: argparse.Namespace) -> Path:
    config, _ = load_config_from_arguments(arguments)
    configure_exports(
        pdf=bool(config.get("visualization", {}).get("save_pdf", True)),
        svg=bool(config.get("visualization", {}).get("save_svg", True)),
    )
    output = make_output_directory(config=config, mode="predefined_world_sweeps", explicit=arguments.output)
    save_run_metadata(config=config, output=output, mode="predefined_world_sweeps", command=sys.argv)
    figures = output / "figures"
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    world = build_world(config)
    field = compute_poisson_field(world.occupancy, spacing=world.spacing, config=config["boxes"]["poisson"])
    sweeps = config["experiments"]["predefined_world"]["sweeps"]

    hocbf_records = []
    hocbf_trajectories: dict[float, pd.DataFrame] = {}
    base_gamma1 = float(config["boxes"]["cbf"]["gamma1"])
    base_gamma2 = float(config["boxes"]["cbf"]["gamma2"])
    for scale in sweeps["hocbf_alpha_scales"]:
        case = deepcopy(config)
        case["boxes"]["cbf"]["gamma1"] = base_gamma1 * float(scale)
        case["boxes"]["cbf"]["gamma2"] = base_gamma2 * float(scale)
        summary, metrics = _run_case(case, world, field, data / f"hocbf_alpha_{float(scale):.4g}", f"hocbf_alpha_{float(scale):.4g}")
        hocbf_records.append({"alpha_scale": float(scale), **summary})
        hocbf_trajectories[float(scale)] = metrics
    pd.DataFrame(hocbf_records).to_csv(data / "hocbf_alpha_sweep.csv", index=False)
    plot_parameter_sweep(
        hocbf_records,
        parameter="alpha_scale",
        title="HOCBF gain sensitivity: safety, intervention, contingency, and timing",
        directory=figures,
        name="hocbf_alpha_sensitivity",
        dpi=int(config["visualization"]["dpi"]),
    )

    plot_trajectory_family(
        hocbf_trajectories,
        world=world,
        parameter_label="HOCBF alpha scale",
        title="HOCBF gain sensitivity: three-dimensional landing trajectories",
        directory=figures,
        name="hocbf_alpha_trajectory_family",
        dpi=int(config["visualization"]["dpi"]),
    )

    clf_records = []
    clf_trajectories: dict[float, pd.DataFrame] = {}
    for gain in sweeps["clf_alpha_gains"]:
        case = deepcopy(config)
        case["boxes"]["clf"]["alpha"] = {"type": "linear", "gain": float(gain)}
        summary, metrics = _run_case(case, world, field, data / f"clf_alpha_{float(gain):.4g}", f"clf_alpha_{float(gain):.4g}")
        clf_records.append({"alpha_gain": float(gain), **summary})
        clf_trajectories[float(gain)] = metrics
    pd.DataFrame(clf_records).to_csv(data / "clf_alpha_sweep.csv", index=False)
    plot_parameter_sweep(
        clf_records,
        parameter="alpha_gain",
        title="CLF decrease-rate sensitivity: convergence, intervention, and contingency",
        directory=figures,
        name="clf_alpha_sensitivity",
        dpi=int(config["visualization"]["dpi"]),
    )
    plot_trajectory_family(
        clf_trajectories,
        world=world,
        parameter_label="CLF alpha gain",
        title="CLF decrease-rate sensitivity: three-dimensional landing trajectories",
        directory=figures,
        name="clf_alpha_trajectory_family",
        dpi=int(config["visualization"]["dpi"]),
    )

    roa_records = []
    for fraction in sweeps.get("roa_fractions", []):
        case = deepcopy(config)
        case["boxes"]["clf"]["roa_fraction"] = float(fraction)
        summary, _ = _run_case(
            case, world, field,
            data / f"roa_fraction_{float(fraction):.4g}",
            f"roa_fraction_{float(fraction):.4g}",
        )
        roa_records.append({"roa_fraction": float(fraction), **summary})
    if roa_records:
        pd.DataFrame(roa_records).to_csv(data / "roa_fraction_sweep.csv", index=False)
        plot_parameter_sweep(
            roa_records,
            parameter="roa_fraction",
            title="Region-of-attraction scaling: contingency margin and landing performance",
            directory=figures,
            name="roa_fraction_sensitivity",
            dpi=int(config["visualization"]["dpi"]),
        )

    print(f"OUTPUT_DIRECTORY={output}")
    return output


def main() -> int:
    run(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
