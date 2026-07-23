# Modular Safety Boxes for Mars-Analog Helicopter Landing

This repository is a research workspace for **perception-driven, multi-certificate aerial landing**.  It preserves the original Poisson, CBF/HOCBF, and vision modules and extends them with reusable Control Lyapunov Function (CLF), region-of-attraction (ROA), contingency, and unified optimization boxes.

The active paper pipeline is

```text
environment or perception data
        ↓
occupancy and Dirichlet boundary
        ↓
Poisson safety function hP, DhP, D²hP
        ↓
nominal planner or external controller → u_nom
        ↓
Poisson-HOCBF + active-target CLF + r-out-of-p ROA constraints
        ↓
minimum-intervention multi-certificate filter
        ↓
u_safe
```

The repository is designed around two requirements:

1. **Scientific traceability.** Every control claim is associated with a mathematical object, a source file, a test, and a logged residual.
2. **Software modularity.** Each safety box owns one responsibility, exposes a stable input/output contract, and can be reused outside the helicopter experiments.

The current paper experiments use reduced-order single- and double-integrator models. They do not claim full-order PX4, hardware, estimator, or Martian-aerodynamics guarantees. The exact scope is documented in [`docs/SAFETY_SCOPE.md`](docs/SAFETY_SCOPE.md).

---

## 1. Scientific question

The central problem is not only collision avoidance. A Mars-analog aerial vehicle must:

- traverse a cluttered three-dimensional environment;
- avoid walls, cliffs, boulders, spires, suspended obstacles, and crater rims;
- descend to a controlled landing equilibrium;
- preserve alternative landing sites while approaching the preferred site;
- retarget when a landing site becomes unavailable;
- stop honestly in a certified `HOLD` state when the configured contingency guarantee can no longer be maintained.

The repository separates these requirements mathematically:

| Requirement | Mathematical object | Software owner |
|---|---|---|
| Environmental geometry | occupancy domain \(\Omega\), boundary \(\partial\Omega\) | `poisson_safety_box` |
| Smooth collision certificate | Poisson safety function \(h_P\) | `poisson_safety_box` |
| Dynamic obstacle avoidance | CBF/HOCBF inequality | `cbf_safety_box` |
| Landing convergence | target-specific CLF \(V_j\) | `clf_safety_box` |
| Local landing feasibility | CLF sublevel set \(\mathcal R_j(c_j)\) | `clf_safety_box` |
| Backup preservation | \(r\)-out-of-\(p\) ROA pivot \(\widetilde h_r\) | `contingency_safety_box` |
| Minimum intervention | constrained optimization | `safety_filter_box` |
| Shared contracts | immutable state, certificate, and constraint types | `safety_box_core` |

---

## 2. Mathematical formulation

### 2.1 Reduced-order dynamics

The paper experiments use the three-dimensional double integrator

\[
\dot p = v,\qquad \dot v = a,
\]

with state and control

\[
x = \begin{bmatrix}p\\v\end{bmatrix}\in\mathbb R^6,
\qquad
u = a\in\mathbb R^3.
\]

The generic package interfaces remain control-affine:

\[
\dot x=f(x)+g(x)u.
\]

The reduced model is intentionally separated from perception and environment code. A different control-affine model can be supplied to the CLF box without rewriting Poisson or contingency logic.

### 2.2 Poisson safety synthesis

Given an occupancy-derived free domain \(\Omega\subset\mathbb R^d\), the Poisson box solves the Dirichlet problem

\[
\begin{cases}
\Delta h_P(y)=f_P(y), & y\in\Omega,\\
h_P(y)=0, & y\in\partial\Omega,
\end{cases}
\]

where \(f_P<0\) is a configurable forcing function. The numerical outputs are

\[
h_P,\qquad D h_P,\qquad D^2h_P.
\]

The geometry is encoded by \(\Omega\) and \(\partial\Omega\); the forcing function changes the interior field and gradient distribution. Available forcing methods are:

```text
constant
distance
average_flux
guidance
```

Available Poisson solvers are:

```text
sparse_direct
conjugate_gradient
sor
```

The experiments distinguish:

- the exact assembled-system residual \(A_h\mathbf h-\mathbf b\);
- the reconstructed finite-difference diagnostic \(\Delta_hh_P-f_P\);
- field error relative to the sparse-direct reference;
- solve time, derivative time, and total wall time.

### 2.3 Environmental HOCBF

For the double integrator and a spatial Poisson function \(h_P(p)\),

\[
\dot h_P = D h_P(p)v,
\]

\[
\ddot h_P = D h_P(p)a + v^\top D^2h_P(p)v.
\]

With linear higher-order class-\(\mathcal K\) functions, the implemented relative-degree-two HOCBF row is

\[
D h_P(p)a
\geq
-v^\top D^2h_P(p)v
-(\gamma_1+\gamma_2)D h_P(p)v
-\gamma_1\gamma_2h_P(p).
\]

The Poisson box does not select the control. It supplies the differentiable safety quantities. The CBF box converts those quantities into one affine row for the optimization filter.

### 2.4 Landing CLFs

Each candidate landing site \(j\) is represented by a controlled equilibrium

\[
x_j^\star =
\begin{bmatrix}
p_j^\star\\0
\end{bmatrix}.
\]

With error \(e_j=x-x_j^\star\), a stabilizing local feedback is constructed using

\[
u=-K_je_j,
\qquad
A_{\mathrm{cl},j}=A-BK_j.
\]

For \(Q_j\succ0\), the CLF box solves

\[
A_{\mathrm{cl},j}^\top P_j+P_jA_{\mathrm{cl},j}=-Q_j
\]

and defines

\[
V_j(x)=e_j^\top P_je_j.
\]

The box verifies:

- that \(A-BK_j\) is Hurwitz;
- that \(P_j\succ0\);
- the Lyapunov-equation residual;
- matrix conditioning;
- compatibility of the selected sublevel threshold with configured input bounds.

The active-target CLF condition is

\[
L_fV_j(x)+L_gV_j(x)u
\leq
-\alpha_{V,j}(V_j(x))+\delta_{\mathrm{clf}}.
\]

The optional \(\delta_{\mathrm{clf}}\) is explicitly penalized and logged. It does not relax the environmental HOCBF or combinatorial contingency constraints.

### 2.5 Regions of attraction

A certified local inner approximation of the region of attraction is represented by

\[
\mathcal R_j(c_j)=\{x:V_j(x)\leq c_j\}.
\]

The corresponding zero-superlevel certificate is

\[
h_j^{\mathrm{ROA}}(x)=c_j-V_j(x).
\]

Therefore,

\[
h_j^{\mathrm{ROA}}(x)\geq0
\iff
x\in\mathcal R_j(c_j).
\]

The full ROA is a six-dimensional state-space object. Any ellipse or ellipsoid shown in a spatial figure is explicitly a **projection** or **slice**, not the complete ROA.

### 2.6 Combinatorial contingency

For \(p\) landing sites and a required count \(r\), define the pivot

\[
\widetilde h_r(x)
=
\max^{(r)}
\left\{
h_1^{\mathrm{ROA}}(x),\ldots,h_p^{\mathrm{ROA}}(x)
\right\},
\]

where \(\max^{(r)}\) denotes the \(r\)-th largest value. Then

\[
\widetilde h_r(x)\geq0
\]

means that at least \(r\) landing-site ROA certificates are nonnegative.

The implemented smooth combinatorial rows are

\[
\dot h_j^{\mathrm{ROA}}(x,u)
\geq
-\alpha_c\!\left(h_j^{\mathrm{ROA}}(x)\right)
-\omega\rho\!\left(
 h_j^{\mathrm{ROA}}(x)-\widetilde h_r(x)
\right),
\]

with one shared nonnegative auxiliary variable \(\omega\). The contingency box also reports:

- available landing sites;
- certified landing sites;
- the pivot value;
- critical certificate identities;
- target invalidations;
- certified retarget events;
- the reason for entering `HOLD`.

### 2.7 Unified filter

All new packages share the affine convention

\[
Az\geq b.
\]

For the double-integrator contingency experiment, the augmented decision contains

\[
z=\begin{bmatrix}a\\\omega\\\delta_{\mathrm{clf}}\end{bmatrix}
\]

when all optional blocks are enabled. The filter solves a minimum-intervention problem of the form

\[
\min_z
\frac12(z-z_{\mathrm{nom}})^\top W(z-z_{\mathrm{nom}})
\]

subject to:

1. acceleration bounds;
2. hard environmental HOCBF rows;
3. active-target CLF row;
4. hard combinatorial ROA rows;
5. \(\omega\geq0\) and configured slack bounds.

The solver result is accepted only after independently checking all residuals. The runtime does not clip an infeasible solution after optimization.

---

## 3. Macrosystem architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ Geometry source                                                          │
│ predefined 3-D world | static image | video/camera/stream               │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ Occupancy representation                                                 │
│ metric inflation → free domain Ω → Dirichlet boundary ∂Ω                │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ poisson_safety_box                                                       │
│ forcing fP → solve Ah=b → validate residual → hP, DhP, D²hP             │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ↓ local SafetySample
          ┌──────────────────────┼─────────────────────────┐
          ↓                      ↓                         ↓
┌──────────────────┐   ┌────────────────────┐   ┌──────────────────────────┐
│ cbf_safety_box   │   │ clf_safety_box     │   │ contingency_safety_box   │
│ HOCBF row        │   │ Vj, Pj, Kj, cj     │   │ hj=cj−Vj, r-out-of-p      │
└────────┬─────────┘   └─────────┬──────────┘   └─────────────┬────────────┘
         │                       │                            │
         └───────────────────────┴────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ safety_filter_box                                                        │
│ nominal decision + affine bundles → verified minimum-intervention solve │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ↓
                              u_safe
```

The nominal planner remains separate from the formal certificate layers. In the predefined world it uses deterministic A* with a physical-clearance penalty and a lookahead PD controller. It improves mission progress but is not described as a safety or reachability certificate.

---

## 4. Safety-box responsibilities

### 4.1 `poisson_safety_box/`

The complete original Poisson package is retained.

**Input**

```text
occupancy mask
physical grid spacing
forcing configuration
solver configuration
```

**Output**

```text
free-space and boundary masks
forcing field
Poisson solution hP
gradient DhP
Hessian D²hP
algebraic residuals
Laplacian diagnostics
timing and iteration metadata
```

**Key implementation files**

| File | Responsibility |
|---|---|
| `poisson_safety_box/poisson_safety_box/matrix.py` | assemble the sparse finite-difference operator |
| `poisson_safety_box/poisson_safety_box/forcing.py` | construct constant, distance, average-flux, and guidance forcing |
| `poisson_safety_box/poisson_safety_box/solver.py` | direct, CG, and SOR solve backends |
| `poisson_safety_box/poisson_safety_box/derivatives.py` | gradient, Hessian, and Laplacian diagnostics |
| `experiments/common/poisson_field.py` | experiment-facing orchestration and local sampling |

### 4.2 `cbf_safety_box/`

The complete original CBF package is retained. New experiment code uses its adapter API without removing original models, constraints, solvers, examples, or tests.

**Input**

```text
StateSnapshot
SafetySample(hP, DhP, D²hP, optional ∂thP)
DecisionLayout
```

**Output**

```text
ConstraintBundle containing one or more affine CBF/HOCBF rows
```

**Key implementation files**

| File | Responsibility |
|---|---|
| `cbf_safety_box/cbf_safety_box/api.py` | reusable box adapter and shared affine-row output |
| `cbf_safety_box/cbf_safety_box/constraints/velocity_cbf.py` | relative-degree-one CBF |
| `cbf_safety_box/cbf_safety_box/constraints/acceleration_hocbf.py` | relative-degree-two HOCBF |
| `cbf_safety_box/cbf_safety_box/optimization/` | preserved original optimization backends |

### 4.3 `clf_safety_box/`

This package owns landing convergence and local attraction regions.

**Input**

```text
ControlAffineModel
EquilibriumTarget objects
LQR and Lyapunov weights
input bounds
CLF alpha configuration
```

**Output**

```text
one QuadraticCLFArtifact per target
Vj, ∇Vj, LfVj, LgVj
cj and hjROA=cj−Vj
active-target affine CLF row
```

**Key implementation files**

| File | Responsibility |
|---|---|
| `clf_safety_box/src/clf_safety_box/models.py` | generic model interfaces and single/double integrators |
| `clf_safety_box/src/clf_safety_box/quadratic.py` | LQR gain, Lyapunov equation, ROA threshold |
| `clf_safety_box/src/clf_safety_box/alpha.py` | configurable CLF decrease functions |
| `clf_safety_box/src/clf_safety_box/box.py` | vectorized multi-target evaluation and active CLF row |

### 4.4 `contingency_safety_box/`

This package does not construct CLFs. It composes generic differentiable certificates and is therefore reusable with future certificate families.

**Input**

```text
CertificateEvaluation objects
availability map
required count r
```

**Output**

```text
r-th-largest pivot
critical set
available and certified counts
combinatorial affine rows
READY or HOLD status
```

**Key implementation files**

| File | Responsibility |
|---|---|
| `contingency_safety_box/src/contingency_safety_box/box.py` | pivot, count, critical set, and combinatorial rows |
| `contingency_safety_box/src/contingency_safety_box/policies.py` | certified target-selection policies |

### 4.5 `safety_filter_box/`

This package solves the unified optimization and verifies the result.

**Input**

```text
nominal decision
DecisionLayout
ConstraintBundle objects from enabled boxes
variable bounds
```

**Output**

```text
FilterResult
safe control
solver status and timing
constraint residuals
active constraints
```

**Key implementation files**

| File | Responsibility |
|---|---|
| `safety_filter_box/src/safety_filter_box/filter.py` | bundle assembly, objective, verification, typed result |
| `safety_filter_box/src/safety_filter_box/solvers.py` | Hildreth and SLSQP backends |

### 4.6 `safety_box_core/`

This package is the shared contract layer.

**Key types**

```text
AffineConstraint
ConstraintBundle
DecisionLayout
StateSnapshot
EquilibriumTarget
CertificateEvaluation
FilterResult
BoxStatus
```

No experiment box should invent a competing constraint dataclass.

---

## 5. Repository map

```text
Helicopter/
├── configs/
│   ├── experiment.yaml              # single user-facing experiment configuration
│   └── worlds/
│       └── mars_analog_landing.yaml         # reproducible Mars-analog obstacle geometry
│
├── experiments/
│   ├── common/
│   │   ├── cli.py                   # profiles, dotted overrides, output metadata
│   │   ├── controller.py            # combines independent boxes at one control step
│   │   ├── nominal_planner.py       # A* + lookahead PD nominal behavior
│   │   ├── poisson_field.py         # experiment-facing Poisson adapter
│   │   ├── simulation.py            # deterministic double-integrator rollout and events
│   │   ├── plotting.py              # generic publication-quality diagnostics
│   │   └── segmentation.py          # image/video occupancy processing
│   │
│   ├── predefined_world/
│   │   ├── world.py                 # load/rasterize the analytic 3-D world
│   │   ├── scenarios.py             # baseline, single-failure, and sequential-failure missions
│   │   ├── paper_figures.py         # claim-specific Approach/Results figures
│   │   ├── run.py                   # execute one named scenario
│   │   ├── run_sweeps.py            # HOCBF, CLF, and ROA parameter sweeps
│   │   └── run_paper_suite.py        # complete controlled paper experiment matrix
│   │
│   ├── static_image/
│   │   ├── pipeline.py              # image → occupancy → Poisson → rollout
│   │   └── run.py                   # static-image entry point
│   │
│   └── live_vision/
│       ├── run.py                   # stream/camera runtime
│       ├── worker.py                # latest-only asynchronous Poisson worker
│       └── dashboard.py             # live status and bounded histories
│
├── poisson_safety_box/              # complete original Poisson package
├── cbf_safety_box/                  # complete original CBF/HOCBF package
├── vision_poisson_experiments/      # complete original vision experiments
├── safety_box_core/                 # canonical data contracts
├── clf_safety_box/                  # CLF and ROA construction
├── contingency_safety_box/          # r-out-of-p contingency
├── safety_filter_box/               # unified optimizer
├── scripts/                         # reproducibility and verification commands
├── tests/                           # new shared-box tests
├── docs/                            # theory, architecture, figures, and safety scope
├── outputs/                         # generated artifacts and preserved reference outputs
└── legacy/hjr/                      # archived HJR code, inactive in the CLF experiments
```

A more detailed file-by-file guide is provided in [`docs/FILE_GUIDE.md`](docs/FILE_GUIDE.md).

---

## 6. Three experiment modes

All three modes use the same CLF, HOCBF, contingency, and optimization APIs. Only the source of geometry changes.

### 6.1 Predefined Mars-analog world

```text
configs/worlds/mars_analog_landing.yaml
        ↓
analytic obstacle rasterization
        ↓
3-D occupancy and Poisson field
        ↓
double-integrator landing experiment
```

The default world is **32 m × 24 m × 14 m** and contains:

- three tall escarpments that force a north-side corridor;
- elevated boulders in the corridor;
- a descent spire;
- a crater rim around the primary site;
- low ridges and secondary rock fields;
- four candidate landing sites.

The direct start-to-primary-site line intersects occupied geometry. The A* nominal path and the certificate filter must therefore produce a nontrivial route and terminal descent.

Run the complete paper suite:

```bash
bash scripts/run_paper_experiments.sh
```

The suite executes the three scientifically distinct scenarios and then runs all parameter sweeps on the fixed no-failure baseline. It writes a cross-scenario comparison and an index that maps each generated figure to the claim it supports. A fast end-to-end check is:

```bash
bash scripts/run_paper_experiments.sh \
  --profile smoke \
  --skip-comparisons \
  --skip-sweeps
```

Or run individual claims:

```bash
# Obstacle-demanding successful landing
python experiments/predefined_world/run.py \
  --scenario baseline \
  --compare \
  --output outputs/paper/01_baseline

# One site failure, certified retargeting, successful contingency landing
python experiments/predefined_world/run.py \
  --scenario single_failure \
  --output outputs/paper/02_single_failure

# Repeated failures, then certified HOLD when r can no longer be maintained
python experiments/predefined_world/run.py \
  --scenario sequential_failure \
  --output outputs/paper/03_sequential_failure

# Fair parameter sweeps on the no-failure baseline
python experiments/predefined_world/run_sweeps.py \
  --scenario baseline \
  --output outputs/paper/04_parameter_sweeps
```

### 6.2 Static-image experiment

```text
one image or supplied binary mask
        ↓
rectification and cleanup
        ↓
metric occupancy and inflation
        ↓
2-D Poisson field and filtered landing rollout
```

```bash
python experiments/static_image/run.py \
  --profile smoke \
  --image experiments/static_image/input/example_scene.png \
  --compare \
  --output outputs/paper/static_image
```

This mode is useful for repeatable perception-to-safety figures without a live stream.

### 6.3 Live vision

```text
video | USB camera | OpenCV-compatible IP stream
        ↓
interactive background capture
        ↓
occupancy updates
        ↓
latest-only asynchronous Poisson worker
        ↓
real-time HOCBF + CLF + contingency filter
```

Example with a file:

```bash
python experiments/live_vision/run.py \
  --source experiments/live_vision/assets/example_stream.avi \
  --display \
  --output outputs/paper/live_vision
```

Example with an IP stream:

```bash
python experiments/live_vision/run.py \
  --source "http://CAMERA_IP:PORT/stream.mjpg" \
  --display \
  --output outputs/paper/live_ip_camera
```

Interactive controls:

```text
B       capture or recapture the empty background
SPACE   begin the experiment after obstacles are placed
R       reset setup
Q/ESC   stop and save data/figures
```

The current live mode uses real camera geometry with a virtual reduced-order vehicle. It does not yet send commands to PX4.

---

## 7. Paper experiments and claim separation

The repository deliberately separates three mission claims.

### Experiment A — successful obstacle-demanding landing

```text
no target failures
A* nominal route through the canyon
Poisson-HOCBF obstacle avoidance
CLF convergence to LZ0
verified touchdown position and speed
```

This experiment supports the claim that the integrated architecture can avoid nontrivial 3-D geometry and reach a landing equilibrium.

### Experiment B — one failure and contingency landing

```text
approach LZ0
LZ0 becomes unavailable
select a currently available and certified alternative
retarget
land at the alternative site
```

This experiment supports the claim that a target failure can be handled without abandoning environmental safety or the remaining ROA requirement.

### Experiment C — sequential contingency depletion

```text
active site fails
certified retarget
new active site fails
certified retarget
remaining certified count falls below r
enter HOLD
```

This experiment is a stress test. Its terminal point is labelled `HOLD`, never `touchdown`.

### Parameter studies

Sweeps use the same no-failure world so the effect of one parameter is not confounded by different event sequences.

- HOCBF \(\gamma_1,\gamma_2\) scale;
- CLF decrease gain;
- ROA fraction;
- Poisson forcing method;
- Poisson solver.

---

## 8. Figure methodology

The paper-oriented plotting module is

```text
experiments/predefined_world/paper_figures.py
```

It produces claim-specific figures:

| Figure | Intended paper section | Scientific purpose |
|---|---|---|
| `paper_methodology_overview` | Approach | geometry → occupancy → forcing → Poisson → CLF ROAs → filtered trajectory |
| `paper_obstacle_avoidance_and_landing` | Results | direct unsafe reference, A* nominal path, safe trajectory, obstacles, certificates |
| `paper_terminal_landing_verification` | Results | verifies actual touchdown position and speed conditions |
| `paper_contingency_timeline` | Results | target failures, retargeting, per-site ROA margins, pivot, counts, intervention |
| `hocbf_alpha_trajectory_family` | Results/Appendix | geometric effect of HOCBF gain on all trajectories |
| `hocbf_alpha_sensitivity` | Results | safety–intervention–performance trade-off |
| `clf_alpha_trajectory_family` | Results/Appendix | geometric effect of CLF decrease rate |
| `poisson_solver_comparison` | Results | timing, exact residual, Laplacian diagnostic, field error |
| `forcing_field_trajectory_comparison` | Results | forcing-dependent field and trajectory geometry |
| `poisson_field_planes` | Approach/Appendix | \(h_P\) and \(\|Dh_P\|\) on XY/XZ/YZ planes |
| `poisson_isosurfaces_3d` | Approach/Appendix | volumetric safety landscape |
| `clf_regions_of_attraction` | Approach | projected attraction regions and landing equilibria |
| `clf_phase_portraits` | Appendix | closed-loop Lyapunov vector fields and convergence |

The recommended main-paper selection and appendix allocation are documented in [`docs/PAPER_FIGURE_PLAN.md`](docs/PAPER_FIGURE_PLAN.md).

All paper figures are exported as high-DPI PNG and, where practical, PDF/SVG. Raster-heavy contour and isosurface figures intentionally omit SVG because extremely dense vector paths are not useful in publication workflows.

---

## 9. Output structure

Each run is self-describing:

```text
run_directory/
├── effective_config.yaml
├── run_metadata.json
├── clf_artifacts/
│   ├── LZ0.json / LZ0.npz
│   ├── LZ1.json / LZ1.npz
│   └── ...
├── data/
│   ├── metrics_*.csv
│   ├── events_*.csv
│   ├── summary_*.json
│   ├── poisson_field.npz
│   ├── poisson_summary.json
│   └── initial_nominal_path.csv
└── figures/
    ├── paper_methodology_overview.png
    ├── paper_obstacle_avoidance_and_landing.png
    ├── paper_terminal_landing_verification.png      # successful runs only
    ├── paper_contingency_timeline.png  # failure runs only
    └── ...
```

The terminal status is one of:

```text
landed   position and speed tolerances were satisfied
hold     the runtime intentionally stopped because a required guarantee was lost
timeout  the maximum simulation horizon was reached
```

A plotting function must use this status rather than labelling the final state generically as a landing.

---

## 10. Central configuration

New experiments use one user-facing configuration:

```text
configs/experiment.yaml
```

The physical obstacle world is stored separately as reproducible experiment data:

```text
configs/worlds/mars_analog_landing.yaml
```

The controller configuration remains centralized; obstacle geometry is separated because it is scenario data, not a control gain.

Examples:

```bash
# Select another initial landing zone
python experiments/predefined_world/run.py \
  --scenario baseline \
  --set experiments.predefined_world.simulation.initial_target=LZ2

# Require three certified alternatives
python experiments/predefined_world/run.py \
  --scenario baseline \
  --set boxes.contingency.required_certified=3

# Compare a different Poisson forcing method
python experiments/predefined_world/run.py \
  --scenario baseline \
  --set boxes.poisson.forcing_method=constant

# Change HOCBF gains
python experiments/predefined_world/run.py \
  --scenario baseline \
  --set boxes.cbf.gamma1=1.5 \
  --set boxes.cbf.gamma2=1.5

# Change CLF decrease rate
python experiments/predefined_world/run.py \
  --scenario baseline \
  --set boxes.clf.alpha.gain=0.045
```

### Adding obstacles

Add or modify records in:

```text
configs/worlds/mars_analog_landing.yaml
```

Supported analytic geometry:

```text
box
cylinder
ellipsoid
annular_cylinder
```

Each obstacle is rasterized using the configured physical inflation margin. The runtime verifies that the initial state and landing equilibria are not occupied.

### Changing landing sites

Landing positions are listed in order in `configs/experiment.yaml`:

```yaml
landing_zones:
  - id: LZ0
    position_m: [29.0, 21.0, 1.2]
    label: primary crater floor
  - id: LZ1
    position_m: [29.0, 4.0, 1.2]
    label: southern crater floor
  - id: LZ2
    position_m: [16.5, 20.5, 1.2]
    label: northwestern contingency basin
  - id: LZ3
    position_m: [18.5, 3.8, 1.2]
    label: western contingency pad
```

Changing a site reconstructs its equilibrium, LQR gain, Lyapunov matrix, CLF, and ROA artifact at startup.

---

## 11. Installation and verification

```bash
unzip Helicopter_Paper_Release.zip
cd Helicopter
chmod +x install.sh run_checks.sh scripts/*.sh
./install.sh
source .venv/bin/activate
./run_checks.sh
```

Manual editable installation:

```bash
python -m pip install -r requirements.txt
python -m pip install -e safety_box_core
python -m pip install -e safety_filter_box
python -m pip install -e clf_safety_box
python -m pip install -e contingency_safety_box
python -m pip install -e poisson_safety_box
python -m pip install -e cbf_safety_box
python -m pip install -e vision_poisson_experiments
```

Run tests:

```bash
python -m pytest -q
(cd poisson_safety_box && python -m pytest -q)
(cd cbf_safety_box && python -m pytest -q)
python scripts/verify_clf_runtime.py
```

---

## 12. Reproducibility policy

Every experiment records:

- the effective merged configuration;
- a command and metadata record;
- raw time histories;
- event logs;
- CLF matrices and certification metadata;
- Poisson field arrays and solver diagnostics;
- high-resolution figures.

Fair comparisons use:

- a fixed random seed;
- the same world and initial state;
- the same failure schedule, or no failures for parameter sweeps;
- identical terminal conditions;
- independently checked solver residuals.

Generated outputs should not be interpreted only from plots. The associated CSV and JSON files are the source of quantitative tables in the paper.

---

## 13. Guarantee and interpretation boundaries

### Supported within the stated reduced-order model and assumptions

- satisfaction of the logged affine HOCBF row up to numerical tolerance;
- satisfaction of the logged CLF row, accounting for any explicit CLF slack;
- satisfaction of the logged combinatorial rows while the optimization remains feasible;
- membership in the configured CLF sublevel sets;
- preservation of at least \(r\) nonnegative available ROA certificates while the pivot remains nonnegative.

### Numerically observed in a finite-step simulation

- collision-free sampled trajectories;
- landing success;
- retargeting behavior;
- intervention magnitude;
- solver timing;
- response to configured target invalidations.

### Not established by this repository alone

- full-order multirotor forward invariance;
- PX4 tracking error bounds;
- safety under estimator latency or unmodelled actuator dynamics;
- hardware flight safety;
- equivalence to Martian gravity or aerodynamics;
- finite-horizon HJ reachability guarantees.

The active contingency method is CLF/ROA-based combinatorial stabilization. Archived HJR code remains under `legacy/hjr/` only for historical reproducibility.

---

## 14. Documentation

- [`docs/SCIENTIFIC_WORKFLOW.md`](docs/SCIENTIFIC_WORKFLOW.md) — detailed equation-by-equation methodology
- [`docs/MARS_ANALOG_SCENARIO_DESIGN.md`](docs/MARS_ANALOG_SCENARIO_DESIGN.md) — flagship world geometry and experimental roles
- [`docs/RELEASE_VALIDATION.md`](docs/RELEASE_VALIDATION.md) — automated checks and reference outcomes
- [`docs/THEORY_REFERENCES.md`](docs/THEORY_REFERENCES.md) — source-to-software responsibility map
- [`docs/PAPER_FIGURE_PLAN.md`](docs/PAPER_FIGURE_PLAN.md) — main-paper and appendix figure selection
- [`docs/FILE_GUIDE.md`](docs/FILE_GUIDE.md) — file-by-file responsibility map
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — package interfaces and data flow
- [`docs/EQUATION_TO_CODE_MAP.md`](docs/EQUATION_TO_CODE_MAP.md) — equation → implementation → test → metric
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — commands and expected outputs
- [`docs/PLOTTING_GUIDE.md`](docs/PLOTTING_GUIDE.md) — publication figure conventions
- [`docs/SAFETY_SCOPE.md`](docs/SAFETY_SCOPE.md) — precise guarantee boundaries
- [`docs/MIGRATION_FROM_HJR.md`](docs/MIGRATION_FROM_HJR.md) — methodological migration

Each package also retains its own README, examples, and tests.

---


## 15. Theoretical source map

The software organization mirrors the theoretical separation used in the source literature:

- CLF synthesis and local convergence follow the feedback-linearization and Lyapunov-equation construction used in the CLF lectures.
- CBF/HOCBF rows encode forward invariance and relative-degree extensions, while the unified QP treats safety as hard and stability through an explicit relaxation.
- Poisson safety synthesis converts occupancy geometry into a smooth function through a Dirichlet boundary-value problem before the dynamics-dependent CBF/HOCBF row is formed.
- Combinatorial stabilization constructs each landing-site certificate from a CLF sublevel set and preserves at least `r` out of `p` alternatives using the pivot function.
- Terrain-aware planning is treated as a nominal planning layer; it does not replace formal environmental or ROA certificates.

The equation-to-code correspondence is maintained in [`docs/EQUATION_TO_CODE_MAP.md`](docs/EQUATION_TO_CODE_MAP.md), and exact bibliography information is summarized in [`docs/THEORY_REFERENCES.md`](docs/THEORY_REFERENCES.md).

---

## 16. Citation

Repository metadata are provided in [`CITATION.cff`](CITATION.cff). For a paper or report, cite the specific theoretical sources for Poisson safety functions, CBF/HOCBF safety filtering, CLF construction, and combinatorial stabilization in addition to this software artifact.
