#!/usr/bin/env python3
"""Compare SOR, sparse direct, and CG on a small 2D problem."""
from pathlib import Path
import sys, json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poisson_safety_box import PoissonSafetyBox, PoissonBoxConfig

out=Path('outputs/compare_solvers'); out.mkdir(parents=True, exist_ok=True)
N=70
Y,X=np.mgrid[0:N,0:N]
occ=np.zeros((N,N), dtype=bool)
occ[(X-35)**2+(Y-35)**2<12**2]=True
summary=[]
for solver in ['sor','sparse_direct','conjugate_gradient']:
    cfg=PoissonBoxConfig(grid_spacing=(0.1,0.1), forcing_method='constant', solver=solver)
    cfg.sor.max_iter=1500; cfg.sor.tolerance=1e-5
    res=PoissonSafetyBox(cfg).compute(occ)
    res.plot_all(out / solver / 'figures')
    summary.append({'solver':solver, 'solver_info':res.solver_info, 'timing':res.timing})
(out/'summary.json').write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
