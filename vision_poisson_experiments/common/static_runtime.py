"""Backward-compatible import aliases for the canonical static pipeline.

New code should import :func:`common.static_pipeline.run_static_experiment`.
This module remains intentionally tiny so there is only one orchestration
implementation to maintain and test.
"""

from .static_pipeline import StaticExperimentReport, run_static_experiment

__all__ = ["StaticExperimentReport", "run_static_experiment"]
