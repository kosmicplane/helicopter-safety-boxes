#!/usr/bin/env python3
"""Generate the controlled figure set recommended for the paper.

The suite separates three scientifically distinct claims:

1. obstacle-rich successful landing without target loss;
2. successful certified diversion after one landing-site rejection;
3. sequential contingency exhaustion ending in a deliberate HOLD.

Parameter sweeps are run on the baseline scenario to isolate the parameter
being studied from mission-event changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.common.plotting import configure_exports
from experiments.common.cli import add_common_arguments, load_config_from_arguments, make_output_directory, save_run_metadata
from experiments.predefined_world.paper_figures import plot_scenario_comparison
from experiments.predefined_world.world import build_world


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate the complete predefined-world paper figure suite.")
    add_common_arguments(result)
    result.add_argument("--skip-sweeps", action="store_true", help="Skip HOCBF, CLF, and ROA sweeps.")
    result.add_argument("--skip-comparisons", action="store_true", help="Skip forcing and solver comparisons in the baseline run.")
    return result


def _base_command(arguments: argparse.Namespace) -> list[str]:
    command = [sys.executable]
    if arguments.config:
        common = ["--config", str(arguments.config)]
    else:
        common = []
    if arguments.profile:
        common += ["--profile", str(arguments.profile)]
    if arguments.quick:
        common += ["--quick"]
    for override in arguments.overrides:
        common += ["--set", str(override)]
    return command, common


def _load_result(directory: Path, scenario: str) -> SimpleNamespace:
    data = directory / "data"
    metrics = pd.read_csv(data / f"metrics_{scenario}.csv")
    events_path = data / f"events_{scenario}.csv"
    events = pd.read_csv(events_path) if events_path.is_file() else pd.DataFrame()
    summary = json.loads((data / f"summary_{scenario}.json").read_text(encoding="utf-8"))
    return SimpleNamespace(metrics=metrics, events=events, summary=summary)


def run(arguments: argparse.Namespace) -> Path:
    config, _ = load_config_from_arguments(arguments)
    configure_exports(
        pdf=bool(config.get("visualization", {}).get("save_pdf", True)),
        svg=bool(config.get("visualization", {}).get("save_svg", True)),
    )
    output = make_output_directory(config=config, mode="predefined_world_paper_suite", explicit=arguments.output)
    save_run_metadata(config=config, output=output, mode="predefined_world_paper_suite", command=sys.argv)
    python_prefix, common = _base_command(arguments)

    scenario_directories: dict[str, Path] = {}
    for index, scenario in enumerate(("baseline", "single_failure", "sequential_failure"), start=1):
        destination = output / f"{index:02d}_{scenario}"
        command = python_prefix + [
            str(REPOSITORY_ROOT / "experiments/predefined_world/run.py"),
            *common,
            "--scenario",
            scenario,
            "--output",
            str(destination),
        ]
        if scenario == "baseline" and not arguments.skip_comparisons:
            command.append("--compare")
        print(f"[{index}/4] Running {scenario}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        scenario_directories[scenario] = destination

    if not arguments.skip_sweeps:
        sweep_output = output / "04_parameter_sweeps"
        command = python_prefix + [
            str(REPOSITORY_ROOT / "experiments/predefined_world/run_sweeps.py"),
            *common,
            "--scenario",
            "baseline",
            "--output",
            str(sweep_output),
        ]
        print(f"[4/4] Running parameter sweeps: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)

    results = {name: _load_result(directory, name) for name, directory in scenario_directories.items()}
    world = build_world(config)
    figure_directory = output / "00_cross_scenario_figures"
    plot_scenario_comparison(
        world=world,
        results=results,
        directory=figure_directory,
        dpi=int(config["visualization"]["dpi"]),
    )

    rows = []
    for name, result in results.items():
        rows.append({"scenario": name, **result.summary})
    pd.DataFrame(rows).to_csv(output / "paper_scenario_summary.csv", index=False)
    (output / "paper_scenario_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    index_lines = [
        "# Paper Figure Index",
        "",
        "The suite separates methodological figures from outcome-specific result figures.",
        "",
        "## Approach and methodology",
        "",
        "- `01_baseline/figures/paper_methodology_overview.*`: full perception/geometry-to-control chain.",
        "- `01_baseline/figures/occupancy_boundary_slices.*`: occupancy and Dirichlet frontier.",
        "- `01_baseline/figures/poisson_field_planes.*`: spatial Poisson safety function.",
        "- `01_baseline/figures/clf_regions_of_attraction.*`: CLF equilibria and projected ROAs.",
        "- `01_baseline/figures/contingency_roa_maps.*`: local ROA certificates, pivot, and certified count.",
        "",
        "## Main results",
        "",
        "- `01_baseline/figures/paper_obstacle_avoidance_and_landing.*`: obstacle avoidance and successful primary landing.",
        "- `01_baseline/figures/paper_terminal_landing_verification.*`: touchdown position and speed verification.",
        "- `02_single_failure/figures/paper_contingency_timeline.*`: successful diversion after one site rejection.",
        "- `02_single_failure/figures/paper_terminal_landing_verification.*`: successful contingency landing.",
        "- `03_sequential_failure/figures/paper_contingency_timeline.*`: repeated failures and graceful HOLD.",
        "- `00_cross_scenario_figures/paper_scenario_comparison.*`: baseline, diversion, and HOLD comparison.",
        "- `04_parameter_sweeps/figures/hocbf_alpha_trajectory_family.*`: HOCBF alpha family on one fixed world.",
        "- `04_parameter_sweeps/figures/hocbf_alpha_sensitivity.*`: quantitative HOCBF sensitivity.",
        "- `01_baseline/figures/poisson_solver_comparison.*`: solver timing and numerical error.",
        "- `01_baseline/figures/forcing_field_trajectory_comparison.*`: forcing influence on field and trajectory.",
        "",
        "## Supplementary material",
        "",
        "All remaining figures are useful for appendices, supplementary material, or internal diagnostics.",
    ]
    (output / "PAPER_FIGURE_INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"OUTPUT_DIRECTORY={output}")
    return output


def main() -> int:
    run(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
