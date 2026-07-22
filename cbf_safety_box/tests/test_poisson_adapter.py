import numpy as np
from cbf_safety_box.safety_data.poisson_adapter import sample_from_interpolator


def test_interpolator_adapter():
    def interp(p):
        return {"h": 1.0, "grad_h": np.ones(2), "hessian_h": np.eye(2)}
    s = sample_from_interpolator(interp, np.zeros(2))
    assert s.h == 1.0
    assert s.grad_h.shape == (2,)
