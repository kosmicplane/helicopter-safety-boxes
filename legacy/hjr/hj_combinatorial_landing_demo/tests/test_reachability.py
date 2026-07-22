"""Small unit tests for the reduced HJ value and r-th order statistic."""
import numpy as np
from src.hj_reachability import rth_largest


def test_rth_largest():
    assert rth_largest(np.array([0.7, -0.2, 0.4, 0.1]), 2) == 0.4
    assert rth_largest(np.array([0.7, -0.2, 0.4, 0.1]), 3) == 0.1
