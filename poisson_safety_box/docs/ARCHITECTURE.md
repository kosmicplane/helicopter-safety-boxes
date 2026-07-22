# Architecture

This package is a single modular box. It accepts an occupancy matrix and returns
Poisson safety data.

- `domain/`: occupancy validation, Ω, ∂Ω, solve masks.
- `forcing/`: forcing fields.
- `guidance/`: guidance vector fields and divergence.
- `solvers/`: SOR, sparse direct, conjugate gradient.
- `derivatives/`: gradient, Hessian, Laplacian.
- `interpolation/`: bilinear/trilinear interpolation.
- `visualization/`: optional plots.
- `api.py`: public high-level API.
