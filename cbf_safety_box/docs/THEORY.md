# Theory

This box assumes a safety function `h` is already available. Typically another
box computes it from Poisson's equation. This package only turns local samples
of `h` into control constraints.

## Velocity CBF

For `p_dot = v`:

```text
grad(h)^T v >= -alpha h
```

The QP is:

```text
min 1/2 ||v - v_nom||^2
s.t. grad(h)^T v >= -alpha h
```

## Acceleration HOCBF

For `p_dot = v`, `v_dot = a`:

```text
grad(h)^T a >= -v^T H v -(alpha1+alpha2)grad(h)^T v - alpha1 alpha2 h
```

where `H` is the Hessian of `h`.

## Backstepping helper

The package includes an experimental helper:

```text
h_B = h - 1/(2 mu)||v-k1||^2
```

This is diagnostic and not a full symbolic backstepping controller.
