# Equation-to-Code Map

This document provides traceability from the equations used in the paper to their implementation, tests, and logged diagnostics.

| Equation or object | Meaning | Implementation | Primary test | Logged or plotted quantity |
|---|---|---|---|---|
| \(\dot x=f(x)+g(x)u\) | Generic control-affine model | `safety_box_core/src/safety_box_core/protocols.py`; `clf_safety_box/src/clf_safety_box/models.py` | `tests/test_new_safety_boxes.py` | state and control histories |
| \(A z\ge b\) | Canonical affine row convention | `safety_box_core/src/safety_box_core/types.py::AffineConstraint` | `tests/test_new_safety_boxes.py` | per-row residual \(Az-b\) |
| \(\Delta h_P=f_P\), \(h_P|_{\partial\Omega}=0\) | Poisson Dirichlet problem | `poisson_safety_box/poisson_safety_box/solver.py`, `matrix.py`, `forcing.py`; orchestration in `experiments/common/poisson_field.py` | original Poisson tests | assembled residual, Laplacian error, solve time |
| \(\dot p=v,\ \dot v=a\) | Double-integrator reduced model | `clf_safety_box/src/clf_safety_box/models.py::DoubleIntegratorModel` | `tests/test_new_safety_boxes.py` | position, velocity, acceleration |
| \(D h_P a\ge -v^T D^2h_Pv-(\gamma_1+\gamma_2)D h_Pv-\gamma_1\gamma_2h_P\) | Relative-degree-two environmental HOCBF | original: `cbf_safety_box/cbf_safety_box/constraints/acceleration_hocbf.py`; adapter: `cbf_safety_box/cbf_safety_box/api.py::build_constraint` | original CBF tests and `tests/test_new_safety_boxes.py` | HOCBF residual, \(h_P\), curvature term |
| \(A_{cl}=A-BK\) | Stabilizing local feedback convention | `clf_safety_box/src/clf_safety_box/quadratic.py::construct_quadratic_clf` | `tests/test_new_safety_boxes.py` | closed-loop eigenvalues |
| \(A_{cl}^TP+PA_{cl}=-Q\) | Continuous Lyapunov equation | `clf_safety_box/src/clf_safety_box/quadratic.py` | `tests/test_new_safety_boxes.py` | Lyapunov residual and condition number |
| \(V_j=e_j^TP_je_j\) | Target-specific quadratic CLF | `clf_safety_box/src/clf_safety_box/box.py::evaluate_many` | `tests/test_new_safety_boxes.py` | all \(V_j(t)\), active CLF residual |
| \(\nabla V_j=2P_je_j\) | Analytic CLF gradient | `clf_safety_box/src/clf_safety_box/box.py::evaluate_many` | finite-difference gradient test | gradient and Lie derivatives |
| \(\dot V_j\le-\alpha_V(V_j)+\delta\) | Active-target CLF row | `clf_safety_box/src/clf_safety_box/box.py::active_target_constraint` | `tests/test_new_safety_boxes.py` | CLF residual and `delta_clf` |
| \(\max_{e^TPe\le c}|k_i e|=\sqrt{c\,k_iP^{-1}k_i^T}\) | Analytic input-compatible ROA threshold | `clf_safety_box/src/clf_safety_box/quadratic.py::analytic_input_feasible_c` | `tests/test_new_safety_boxes.py` | target artifact `c` and bound margins |
| \(h_j^{ROA}=c_j-V_j\) | Barrier representation of a CLF sublevel set | `clf_safety_box/src/clf_safety_box/box.py::evaluate_many` | `tests/test_new_safety_boxes.py` | all ROA margins |
| \(\widetilde h_r=\max^{(r)}\{h_j\}\) | `r`-out-of-`p` pivot | `contingency_safety_box/src/contingency_safety_box/box.py::rth_largest` | `tests/test_new_safety_boxes.py` | pivot and certified count |
| \(\dot h_j\ge-\alpha_c(h_j)-\omega\rho(h_j-\widetilde h_r)\) | Smooth combinatorial certificate row | `contingency_safety_box/src/contingency_safety_box/box.py::build_constraints` | `tests/test_new_safety_boxes.py` | all combinatorial residuals and \(\omega\) |
| \(\min_z\frac12(z-z_{nom})^TW(z-z_{nom})\) | Minimum-intervention unified filter | `safety_filter_box/src/safety_filter_box/filter.py` | `tests/test_new_safety_boxes.py` | intervention norm and solve time |
| Hildreth dual coordinate updates | Deterministic projection-QP backend | `safety_filter_box/src/safety_filter_box/solvers.py::HildrethQPSolver` | `tests/test_new_safety_boxes.py` | iterations, p50/p95 time, residual |
| \(S(x)=\frac{\int_{\mathcal F(x)}Q(p)w(p)dA}{\int_{\mathcal F(x)}w(p)dA}\) | Future TA-MPPI nominal viewpoint score | interface only; TA-MPPI is not reimplemented in this repository | not applicable | future planner metric |

## Sign conventions

- Safe spatial states satisfy \(h_P\ge0\).
- CLF attraction regions satisfy \(h_j^{ROA}=c_j-V_j\ge0\).
- Every shared affine row is represented as \(A z-b\ge0\).
- The feedback convention is \(u=-Ke\), hence \(A_{cl}=A-BK\).
- `delta_clf` is a standard, explicitly logged CLF relaxation; it never relaxes environmental HOCBF or combinatorial rows.
