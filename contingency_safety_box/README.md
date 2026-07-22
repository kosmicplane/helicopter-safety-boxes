# contingency_safety_box

Independent `r`-out-of-`p` composition of differentiable certificates. The package does not construct CLFs or Poisson fields; it consumes generic local certificate values and derivatives.

## Landing application

The default landing application provides

\[
h_j(x)=c_j-V_j(x)
\]

from `clf_safety_box`. The pivot is

\[
\widetilde h_r(x)=\max^{(r)}\{h_1(x),\ldots,h_p(x)\}.
\]

At least `r` certificates are nonnegative exactly when \(\widetilde h_r\ge0\).

The smooth combinatorial rows use one shared nonnegative auxiliary decision:

\[
\dot h_j\ge-\alpha_c(h_j)-\omega\rho(h_j-\widetilde h_r).
\]

At a critical certificate, \(h_j=\widetilde h_r\), the relaxation term vanishes.

## Capabilities

- availability-aware pivot and count;
- deterministic sorted margins;
- critical-certificate identification with configurable tolerance;
- `READY` or `HOLD` status;
- smooth affine row construction;
- maximum-margin target-selection policy;
- generic certificate input for future applications.

See `examples/r_out_of_p.py`.
