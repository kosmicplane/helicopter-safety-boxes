# Reduced HJ/Eikonal Reachability Model

## Model and sign convention

The live demonstrator uses

\[
\dot p=u,\qquad \|u\|_2\le v_{\max}.
\]

For landing-zone disk `j`, let `D_j(p)` be the shortest free-space path length to the disk. The value function is

\[
V_j(p,\tau)=v_{\max}(-\tau)-D_j(p),\qquad \tau\le0.
\]

The convention is

\[
V_j\ge0 \iff D_j\le v_{\max}(-\tau),
\]

so nonnegative values are reachable.

## Why it is Hamilton-Jacobi

Away from target seeds, obstacles, boundaries, and nonsmooth medial axes, the geodesic distance satisfies the Eikonal equation

\[
\|\nabla D_j\|_2=1.
\]

Since `grad(V_j) = -grad(D_j)` and `partial_tau(V_j)=-v_max`, the Hamiltonian of the isotropic single integrator is

\[
H(\nabla V)=\max_{\|u\|\le v_{\max}}\nabla V^T u
=v_{\max}\|\nabla V\|,
\]

and therefore `partial_tau V + H = 0` where the distance is smooth.

## Numerical construction

- target seed: the complete free landing disk;
- graph: 8-connected grid;
- horizontal cost: `dx`;
- vertical cost: `dy`;
- diagonal cost: `sqrt(dx^2+dy^2)`;
- diagonal corner cutting through occupied cells: prohibited;
- disconnected cells: infinite distance;
- path: predecessor tracing followed by collision-checked line-of-sight simplification.

Finite differences provide gradients. The code stores Eikonal diagnostics and does not claim differentiability at path-switching or medial-axis locations.

## Horizons

The active horizon evolves as

\[
\tau_a(0)=-T_a,\qquad \dot\tau_a=1.
\]

The contingency horizon is held fixed:

\[
\tau_c=-T_c,\qquad \dot\tau_c=0.
\]

After a certified switch, `tau_active <- tau_contingency`.

## r-out-of-p pivot

For `p` values and required `r`, the pivot is the r-th largest value:

\[
\widetilde h=\max^{(r)}\{V_1,\ldots,V_p\}.
\]

`h_tilde >= 0` is equivalent to at least `r` nonnegative target values.

## Limitation

This model does not represent braking or touchdown velocity. The next dynamics extension is `x=[p,v]`, `p_dot=v`, `v_dot=a`, with acceleration limits and an acceleration-level CBF/HOCBF.
