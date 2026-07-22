import numpy as np
from poisson_safety_box.derivatives.gradient import compute_gradient
from poisson_safety_box.derivatives.hessian import compute_hessian

def test_derivative_shapes():
    h=np.random.rand(8,9,7)
    g=compute_gradient(h,(1,1,1))
    H=compute_hessian(h,(1,1,1))
    assert g.shape == h.shape + (3,)
    assert H.shape == h.shape + (3,3)
