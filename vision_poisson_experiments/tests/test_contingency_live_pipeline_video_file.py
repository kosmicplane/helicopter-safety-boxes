"""Headless local-video integration test for the contingency live pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from common.contingency_live_pipeline import LiveContingencyPipeline
from common.io_utils import load_yaml


def test_contingency_live_file_uses_synchronized_bounded_worker(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "02_phone_stream_poisson_realtime" / "config_contingency_synthetic.yaml"
    config = load_yaml(config_path)
    source = root / "examples" / "assets" / "live_scene.avi"
    output = tmp_path / "contingency_live"

    report = LiveContingencyPipeline(
        source=str(source),
        config=config,
        config_directory=config_path.parent,
        output_directory=output,
        headless=True,
        max_frames=16,
    ).run()

    assert report.frames_processed == 16
    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary["worker"]["worker_queue_max_observed"] <= 1
    assert summary["worker"]["failed_solves"] == 0
    assert summary["worker"]["invalid_solves"] == 0
    assert summary["contingency"]["enabled"]
    assert summary["contingency"]["controller_update_count"] >= 1
    assert (output / "mission_setup.json").is_file()
    assert (output / "landing_zone_states.json").is_file()
    assert (output / "last_valid_field.npz").is_file()
