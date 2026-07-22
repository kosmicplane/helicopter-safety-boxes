import numpy as np
from poisson_safety_box.forcing.constant import build_constant_forcing
from poisson_safety_box.forcing.distance import build_distance_forcing

def test_constant_negative():
    mask=np.ones((5,5), dtype=bool)
    f=build_constant_forcing(mask, 2.0).forcing
    assert np.all(f[mask] < 0)

def test_distance_nonpositive():
    mask=np.ones((10,10), dtype=bool); mask[0,:]=False
    b=~mask
    f=build_distance_forcing(mask,b,(1.0,1.0),0.5).forcing
    assert np.all(f[mask] <= 0)
