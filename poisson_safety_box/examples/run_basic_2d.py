#!/usr/bin/env python3
"""Basic 2D example for poisson_safety_box."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poisson_safety_box import PoissonSafetyBox, PoissonBoxConfig

out = Path('outputs/basic_2d')
out.mkdir(parents=True, exist_ok=True)

N = 60
Y, X = np.mgrid[0:N, 0:N]
occ = np.zeros((N, N), dtype=bool)
occ[(X-45)**2 + (Y-45)**2 < 12**2] = True
occ[20:30, 10:30] = True

for method in ['constant', 'distance', 'average_flux', 'guidance']:
    cfg = PoissonBoxConfig(grid_spacing=(0.1, 0.1), forcing_method=method, solver='sor', plot=True)
    cfg.sor.max_iter = 600
    cfg.sor.tolerance = 1e-4
    result = PoissonSafetyBox(cfg).compute(occ)
    method_dir = out / method
    result.save_npz(method_dir / 'result.npz')
    result.save_summary_json(method_dir / 'summary.json')
    result.plot_all(method_dir / 'figures')
    print(method, result.solver_info.get('residual'))
