"""End-to-end deterministic validation of all required contingency scenarios."""

from __future__ import annotations

import pytest

from common.synthetic_contingency_validation import run_all_scenarios


@pytest.fixture(scope="module")
def summaries(tmp_path_factory):
    output = tmp_path_factory.mktemp("contingency_validation")
    return run_all_scenarios(output)


def test_clear_active_reaches_original_zone(summaries) -> None:
    result = summaries["clear_active"]
    assert result["target_reached"]
    assert result["final_target"] == 1
    assert result["switches"] == 0
    assert not result["collision"]
    assert result["minimum_reachable_count"] >= 2


def test_active_zone_blocked_diverts_to_certified_alternative(summaries) -> None:
    result = summaries["active_zone_blocked"]
    assert result["target_reached"]
    assert result["final_target"] != 1
    assert result["switches"] == 1
    assert not result["hold"]
    assert result["minimum_reachable_count"] >= 2
    assert result["minimum_pivot"] >= 0.0


def test_corridor_block_replans_without_target_switch(summaries) -> None:
    result = summaries["corridor_blocked"]
    assert result["target_reached"]
    assert result["final_target"] == 1
    assert result["switches"] == 0
    assert not result["collision"]


def test_contingency_loss_holds_without_uncertified_fallback(summaries) -> None:
    result = summaries["contingency_lost"]
    assert result["hold"]
    assert result["hold_reason"] == "CONTINGENCY REQUIREMENT LOST"
    assert result["switches"] == 0
    assert result["minimum_reachable_count"] < 2
    assert not result["target_reached"]


def test_camera_movement_invalidates_metric_motion(summaries) -> None:
    result = summaries["camera_moved"]
    assert result["hold"]
    assert "camera moved" in result["hold_reason"]
    assert result["switches"] == 0
    assert not result["collision"]
