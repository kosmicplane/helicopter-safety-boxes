"""Discovery and import bootstrapping for the sibling Safety Box packages.

The project is intentionally installed next to ``poisson_safety_box`` and
``cbf_safety_box``.  This module keeps that workspace-level assumption in one
place and provides actionable errors when the expected layout is not present.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys


@dataclass(frozen=True)
class ExternalBoxPaths:
    """Resolved roots of the two external Safety Box repositories."""

    poisson_root: Path
    cbf_root: Path

    def as_dict(self) -> dict[str, str]:
        """Return paths in a JSON-friendly representation."""

        return {
            "poisson_root": str(self.poisson_root),
            "cbf_root": str(self.cbf_root),
        }


def repository_root() -> Path:
    """Return the root directory of ``vision_poisson_experiments``."""

    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    """Return the directory that contains all three sibling repositories."""

    return repository_root().parent


def _resolve_box_root(environment_variable: str, sibling_name: str, package_name: str) -> Path:
    """Resolve one Safety Box root from an environment override or sibling path."""

    override = os.environ.get(environment_variable)
    candidate = Path(override).expanduser().resolve() if override else workspace_root() / sibling_name
    package_dir = candidate / package_name
    if not package_dir.is_dir():
        raise FileNotFoundError(
            f"Could not locate {package_name!r}. Expected package directory at {package_dir}. "
            f"Place {sibling_name!r} next to {repository_root().name!r}, or set "
            f"the {environment_variable} environment variable."
        )
    return candidate.resolve()


def discover_external_boxes() -> ExternalBoxPaths:
    """Discover both sibling repositories without modifying ``sys.path``."""

    return ExternalBoxPaths(
        poisson_root=_resolve_box_root(
            "POISSON_SAFETY_BOX_ROOT",
            "poisson_safety_box",
            "poisson_safety_box",
        ),
        cbf_root=_resolve_box_root(
            "CBF_SAFETY_BOX_ROOT",
            "cbf_safety_box",
            "cbf_safety_box",
        ),
    )


def bootstrap_external_boxes() -> ExternalBoxPaths:
    """Add the two repository roots to ``sys.path`` and return their paths.

    Inserting repository roots, rather than package directories, reproduces the
    behavior of an editable installation while keeping the user's requested
    sibling layout intact.  Existing entries are not duplicated.
    """

    paths = discover_external_boxes()
    for root in (paths.poisson_root, paths.cbf_root):
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
    return paths
