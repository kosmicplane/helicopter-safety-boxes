import numpy as np
from poisson_safety_box import PoissonSafetyBox, PoissonBoxConfig

def test_sor_returns_positive_h():
    occ=np.zeros((30,30), dtype=bool)
    occ[10:15,10:15]=True
    cfg=PoissonBoxConfig(grid_spacing=(1.0,1.0), forcing_method='constant', solver='sor')
    cfg.sor.max_iter=500; cfg.sor.tolerance=1e-3
    res=PoissonSafetyBox(cfg).compute(occ)
    assert res.h[res.solve_mask].max() > 0
