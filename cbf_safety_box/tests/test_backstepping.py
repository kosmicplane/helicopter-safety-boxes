import numpy as np
from cbf_safety_box.safety_data import SafetySample
from cbf_safety_box.constraints.backstepping import auxiliary_k1, compute_backstepping_value


def test_backstepping_value():
    safety = SafetySample(h=1.0, grad_h=np.array([1.0, 0.0]))
    k1 = auxiliary_k1(safety, gain=1.0)
    d = compute_backstepping_value(safety, np.array([1.0, 0.0]), 1.0, k1)
    assert d["h_B"] > 0.9
