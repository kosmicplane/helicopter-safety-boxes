import numpy as np
from poisson_safety_box.domain.boundaries import compute_boundary_mask, compute_solve_mask

def test_boundary_extraction_2d():
    occ = np.zeros((10,10), dtype=bool); occ[4:6,4:6]=True
    b = compute_boundary_mask(occ, True)
    assert b.any()
    solve = compute_solve_mask(~occ, b)
    assert solve.any()
