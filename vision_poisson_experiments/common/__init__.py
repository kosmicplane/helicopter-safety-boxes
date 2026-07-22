"""Reusable components for the static-image and live-stream experiments.

Every module is deliberately small enough to be tested independently.  The
Poisson and CBF mathematics remain inside their respective sibling Safety Box
packages; this package coordinates perception, sampling, diagnostics, and
experiment execution.
"""

from .external_boxes import bootstrap_external_boxes

# Importing ``common`` makes the sibling libraries available to downstream
# modules, while all path-discovery logic remains centralized and testable.
EXTERNAL_BOX_PATHS = bootstrap_external_boxes()

__all__ = ["EXTERNAL_BOX_PATHS", "bootstrap_external_boxes"]
