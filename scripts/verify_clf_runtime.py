#!/usr/bin/env python3
"""Verify that the new CLF/ROA experiment runtime does not import HJR code.

The complete original vision package is intentionally preserved, including its
historical HJR applications. This check is therefore limited to the active new
packages and entry points rather than deleting reusable legacy functionality.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOTS = (
    ROOT / "safety_box_core",
    ROOT / "safety_filter_box",
    ROOT / "clf_safety_box",
    ROOT / "contingency_safety_box",
    ROOT / "experiments",
)
FORBIDDEN_IMPORT_FRAGMENTS = (
    "hj_reachability",
    "hjr",
    "eikonal",
    "contingency_live_pipeline",
    "unified_contingency_filter",
)


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def main() -> int:
    failures: list[str] = []
    checked = 0
    for package_root in ACTIVE_ROOTS:
        for path in package_root.rglob("*.py"):
            if any(part in {"__pycache__", "legacy"} for part in path.parts):
                continue
            checked += 1
            try:
                modules = imported_modules(path)
            except SyntaxError as exc:
                failures.append(f"syntax error: {path.relative_to(ROOT)}: {exc}")
                continue
            for module in modules:
                lowered = module.lower()
                if any(token in lowered for token in FORBIDDEN_IMPORT_FRAGMENTS):
                    failures.append(
                        f"forbidden active import: {path.relative_to(ROOT)} -> {module}"
                    )
    if failures:
        print("CLF runtime verification failed:")
        print("\n".join(f"  - {item}" for item in failures))
        return 1
    print(f"CLF runtime verification passed: {checked} Python files checked.")
    print("Historical HJR code remains preserved but is not imported by the new runtime.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
