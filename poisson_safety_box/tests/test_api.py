import numpy as np
from poisson_safety_box import PoissonSafetyBox, PoissonBoxConfig

def test_api_result_fields():
    occ=np.zeros((20,20), dtype=bool); occ[8:12,8:12]=True
    cfg=PoissonBoxConfig(grid_spacing=(1,1), forcing_method='constant', solver='sor')
    cfg.sor.max_iter=200; cfg.sor.tolerance=1e-3
    res=PoissonSafetyBox(cfg).compute(occ)
    assert res.h is not None
    assert res.grad_h is not None
    assert res.hessian_h is not None
