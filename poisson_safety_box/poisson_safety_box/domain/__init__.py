"""Domain and mask utilities for the Poisson safety box."""
from .occupancy import normalize_occupancy, make_empty_occupancy, add_box_2d, add_box_3d
from .masks import compute_basic_masks
from .boundaries import compute_boundary_mask, compute_solve_mask
