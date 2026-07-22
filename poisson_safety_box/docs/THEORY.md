# Theory

Given an occupancy matrix, free space is Ω and the obstacle boundary is ∂Ω. The
safety function is computed as the solution of

```math
Δh = f_P \quad \text{in } Ω, \qquad h=0 \quad \text{on } ∂Ω.
```

Obstacles are encoded by the domain and the boundary condition, not by the
forcing function. The forcing function shapes the surface h inside free space.

A future CBF module can evaluate h and ∇h at a robot position p and build a
constraint such as

```math
∇h(p)^T v ≥ -α h(p).
```

This package only computes h and derivatives; it does not solve CBF-QPs.
