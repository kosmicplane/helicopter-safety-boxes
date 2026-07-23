"""Scenario definitions for controlled paper experiments.

A scenario changes only mission events (for example, landing-zone failures).
World geometry, controller equations, and numerical solvers remain identical,
which keeps comparisons scientifically interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    """Resolved mission-event configuration for one rollout."""

    name: str
    description: str
    failure_schedule: tuple[Mapping[str, Any], ...]


def available_scenarios(config: Mapping[str, Any]) -> tuple[str, ...]:
    simulation = config["experiments"]["predefined_world"]["simulation"]
    records = simulation.get("scenarios", {})
    return tuple(str(name) for name in records)


def resolve_scenario(
    config: Mapping[str, Any],
    name: str | None,
) -> ScenarioDefinition:
    """Resolve a named scenario while preserving legacy single-failure configs."""

    simulation = config["experiments"]["predefined_world"]["simulation"]
    scenario_name = str(name or simulation.get("default_scenario", "baseline"))
    scenarios = simulation.get("scenarios", {})
    if scenario_name in scenarios:
        record = scenarios[scenario_name] or {}
        schedule = tuple(record.get("failure_schedule", ()))
        return ScenarioDefinition(
            name=scenario_name,
            description=str(record.get("description", scenario_name)),
            failure_schedule=schedule,
        )

    # Backward compatibility with the original one-failure experiment.
    if scenario_name == "legacy_single_failure":
        if "failure_time_s" not in simulation or "failed_target" not in simulation:
            raise KeyError("Legacy single-failure parameters are not configured.")
        return ScenarioDefinition(
            name=scenario_name,
            description="Legacy fixed-time landing-zone failure.",
            failure_schedule=(
                {
                    "time_s": float(simulation["failure_time_s"]),
                    "target": str(simulation["failed_target"]),
                    "reason": "legacy scheduled target failure",
                },
            ),
        )

    choices = ", ".join(available_scenarios(config))
    raise KeyError(f"Unknown predefined-world scenario {scenario_name!r}. Available: {choices}")
