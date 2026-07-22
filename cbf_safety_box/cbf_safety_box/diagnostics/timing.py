"""Small timing helper."""
from __future__ import annotations

from contextlib import contextmanager
import time

@contextmanager
def timed():
    """Yield a function that returns elapsed time in seconds."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
