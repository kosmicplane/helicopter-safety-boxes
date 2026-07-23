# Theory References and Software Role

This document records the primary sources that define the mathematical responsibilities of the active safety boxes. The repository does not redistribute the source papers.

## Control Lyapunov functions

- A. D. Ames and P. Tabuada, *Lecture 24: Constructing Control Lyapunov Functions*.
  - Used for feedback-linearization/linearization-based CLF construction, Lyapunov equations, and CLF-QP synthesis.
  - Software: `clf_safety_box/`.

## Control barrier functions and higher relative degree

- A. D. Ames and P. Tabuada, *Lecture 25: Control Barrier Functions*.
- A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, “Control Barrier Functions: Theory and Applications,” ECC, 2019.
  - Used for zero-superlevel safe sets, forward invariance, CBF-QP filtering, relative degree, and HOCBF construction.
  - Software: `cbf_safety_box/`.

## Unified stability and safety optimization

- A. D. Ames and P. Tabuada, *Lecture 26: Unifying Safety and Stability*.
  - Used for the joint CLF-CBF optimization structure, explicit stability relaxation, and hard safety prioritization.
  - Software: `safety_filter_box/` and `experiments/common/controller.py`.

## Poisson safety synthesis

- G. Bahati, R. M. Bena, and A. D. Ames, “Dynamic Safety in Complex Environments: Synthesizing Safety Filters with Poisson’s Equation,” 2025.
- E. Yamaguchi, R. M. Bena, G. Bahati, and A. D. Ames, “Layered Safety: Enhancing Autonomous Collision Avoidance via Multistage CBF Safety Filters,” 2026.
  - Used for occupancy-to-Dirichlet construction, Poisson safety functions, derivative computation, and separation between geometry synthesis and dynamics-dependent filtering.
  - Software: `poisson_safety_box/` and `experiments/common/poisson_field.py`.

## CLF regions of attraction and contingency

- Y. Lishkova, P. Ong, S. Tonkens, S. Herbert, and A. D. Ames, “Steering with Contingencies: Combinatorial Stabilization and Reach-Avoid Filters,” 2026.
  - The active repository uses the CLF-based combinatorial stabilization branch.
  - Each local certificate is `h_j = c_j - V_j`; the pivot is the `r`-th largest certificate.
  - HJR remains archived only under `legacy/hjr/`.
  - Software: `contingency_safety_box/`.

## Nominal perception-aware planning context

- J. Todd, P. Roque, A. A. Johnson, and J. W. Burdick, “Terrain-Aware Perceptual Planning for Aerial Vehicles in Martian Environments,” 2026.
  - Used to motivate the future nominal perception-aware planning layer.
  - The current predefined-world planner is deterministic A* plus lookahead PD and is not presented as TA-MPPI.

## Equation traceability

See `docs/EQUATION_TO_CODE_MAP.md` for the equation, implementation, test, and logged diagnostic associated with each active certificate.
