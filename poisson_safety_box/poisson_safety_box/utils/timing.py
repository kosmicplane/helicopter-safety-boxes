"""Small timing helpers used across the Poisson safety box."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Dict, Iterator


@contextmanager
def timed(name: str, timing: Dict[str, float]) -> Iterator[None]:
    """Measure elapsed time and store it in a dictionary."""
    start = perf_counter()
    try:
        yield
    finally:
        timing[name] = perf_counter() - start
