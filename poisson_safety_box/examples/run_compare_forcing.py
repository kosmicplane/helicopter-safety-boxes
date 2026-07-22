#!/usr/bin/env python3
"""Compare all forcing methods on one 2D grid."""
from pathlib import Path
import sys, json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poisson_safety_box import PoissonSafetyBox, PoissonBoxConfig

out = Path('outputs/compare_forcing')
out.mkdir(parents=True, exist_ok=True)
N=80
Y, X = np.mgrid[0:N, 0:N]
occ=np.zeros((N,N), dtype=bool)
occ[(X-25)**2+(Y-40)**2<8**2]=True
occ[(X-55)**2+(Y-38)**2<10**2]=True
occ[50:60, 10:25]=True
summary=[]
for method in ['constant','distance','average_flux','guidance']:
    cfg=PoissonBoxConfig(grid_spacing=(0.1,0.1), forcing_method=method, solver='sor')
    cfg.sor.max_iter=1200; cfg.sor.tolerance=1e-4
    res=PoissonSafetyBox(cfg).compute(occ)
    res.plot_all(out / method / 'figures')
    res.save_summary_json(out / method / 'summary.json')
    summary.append({'method':method, 'hmax':float(res.h[res.solve_mask].max()), 'time':res.timing, 'residual':res.solver_info.get('residual')})
(out/'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
