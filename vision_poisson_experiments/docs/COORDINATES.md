# Coordinate and Array Conventions

The camera and Poisson arrays use row-major indexing. The control state uses the conventional
`[x, y]` ordering. This repository intentionally uses a downward-positive physical `y` axis so that a
rectified top-down image maps to physical coordinates without a hidden reflection.

For a grid with shape `(Ny, Nx)` and physical size `(height, width)`:

```text
dy = height / (Ny - 1)
dx = width  / (Nx - 1)
```

A physical point is converted before interpolation as follows:

```python
p_xy = np.array([x, y])
p_yx = p_xy[[1, 0]]
```

The Poisson package computes derivatives in array-axis order:

```text
grad_yx = [dh/dy, dh/dx]
```

The control package expects physical state order:

```python
grad_xy = grad_yx[[1, 0]]
```

For the Hessian, define the permutation matrix:

```python
P = np.array([[0.0, 1.0],
              [1.0, 0.0]])
H_xy = P @ H_yx @ P.T
```

All field sampling used by the CBF simulation is bilinear. Nearest-neighbor sampling is used only for
boolean occupancy membership tests.
