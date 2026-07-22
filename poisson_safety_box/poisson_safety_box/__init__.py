"""Public package interface for poisson_safety_box.

This library converts an occupancy matrix into a Poisson safety function h and
its derivatives. It intentionally does not implement CBF control or robot
simulation; those belong in separate boxes.
"""

from .config import PoissonBoxConfig
from .api import PoissonSafetyBox
from .result import PoissonBoxResult

__all__ = ["PoissonSafetyBox", "PoissonBoxConfig", "PoissonBoxResult"]
