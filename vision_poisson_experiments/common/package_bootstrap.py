"""Compatibility helper for making sibling Safety Box packages importable."""

from __future__ import annotations

from .external_boxes import ExternalBoxPaths, bootstrap_external_boxes


def ensure_safety_boxes_importable() -> ExternalBoxPaths:
    """Discover sibling repositories, update ``sys.path``, and return paths."""

    return bootstrap_external_boxes()


__all__ = ["ensure_safety_boxes_importable"]
