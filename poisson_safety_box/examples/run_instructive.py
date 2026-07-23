#!/usr/bin/env python3
"""Basic 3D example for poisson_safety_box.

Creates a compact 3D occupancy grid, computes a Poisson safety function using
Guidance forcing, and saves h/gradient/Hessian diagnostics and plots.
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poisson_safety_box import PoissonSafetyBox, PoissonBoxConfig

out = Path('outputs/basic_3d')
out.mkdir(parents=True, exist_ok=True)

shape = (34, 28, 22)
X, Y, Z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
occ = np.zeros(shape, dtype=bool)
occ[(X-17)**2 + (Y-14)**2 + (Z-11)**2 < 4**2] = True
occ[10:18, 8:15, 5:20] = True
occ[22:27, 17:23, 2:10] = True
occ[5:10, 5:10, 15:20] = True

cfg = PoissonBoxConfig(
    grid_spacing=(0.25, 0.25, 0.25),
    forcing_method='guidance',
    solver='sor',
    plot=True,
)
cfg.sor.max_iter = 400
cfg.sor.tolerance = 2e-4
cfg.guidance.target_mean_abs_scale = 0.25

result = PoissonSafetyBox(cfg).compute(occ)
#get the h, gradient, and Hessian at the center of the grid
center = (shape[0]//2, shape[1]//2, shape[2]//2)
print(result)