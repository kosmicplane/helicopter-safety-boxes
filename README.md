# Helicopter Safety-Box Research Workspace

A modular research workspace for perception-driven, multi-certificate aerial landing experiments. The active landing pipeline combines:

- **Poisson safety functions** for smooth environmental safety fields;
- **CBF/HOCBF filtering** for collision avoidance;
- **control Lyapunov functions (CLFs)** for convergence to landing equilibria;
- **CLF sublevel regions of attraction (ROAs)** for local landing feasibility;
- **`r`-out-of-`p` combinatorial contingency** for preserving backup landing sites;
- a verified, minimally invasive optimization filter that maps `u_nominal` to `u_safe`.

The complete original `poisson_safety_box`, `cbf_safety_box`, and `vision_poisson_experiments` packages are preserved. They were not replaced by reduced versions. New interoperable CLF, contingency, core-contract, and unified-filter packages were added, and the full CBF box received a backwards-compatible adapter to the shared `A z >= b` constraint contract.

## Architecture

```text
nominal planner / external controller
                 |
                 v
             u_nominal
                 |
    +------------+-------------+----------------+
    |                          |                |
Poisson field              CLF box       contingency box
h, Dh, D2h                 V_j, c_j       h_j=c_j-V_j
    |                          |                |
HOCBF environmental row       +------ r-out-of-p rows
    |                                           |
    +---------------- multi-certificate filter-+
                              |
                              v
                           u_safe
```

For the double-integrator experiments,

```math
\dot p=v, \qquad \dot v=a.
```

The environmental HOCBF row is

```math
D h_P(p)a \ge
-v^\top D^2h_P(p)v
-(\gamma_1+\gamma_2)D h_P(p)v
-\gamma_1\gamma_2 h_P(p).
```

For landing zone `j`, the quadratic CLF and ROA certificate are

```math
V_j(x)=(x-x_j^\star)^\top P_j(x-x_j^\star),
\qquad h_j^{\mathrm{ROA}}(x)=c_j-V_j(x).
```

The contingency pivot is the `r`-th largest ROA certificate,

```math
\widetilde h_r(x)=\max^{(r)}\{h_1^{\mathrm{ROA}}(x),\ldots,h_p^{\mathrm{ROA}}(x)\}.
```

## Repository layout

```text
Helicopter/
├── poisson_safety_box/          # complete original Poisson package
├── cbf_safety_box/              # complete original CBF/HOCBF package + adapter
├── vision_poisson_experiments/  # complete original image/video package
├── safety_box_core/             # canonical immutable contracts
├── clf_safety_box/              # CLF synthesis, evaluation, and ROA artifacts
├── contingency_safety_box/      # independent r-out-of-p composition
├── safety_filter_box/           # Hildreth and SLSQP verified QP backends
├── experiments/
│   ├── predefined_world/        # controlled 3-D paper experiment
│   ├── static_image/            # one-image offline experiment
│   ├── live_vision/             # video/camera online dashboard
│   └── common/                  # shared orchestration and plotting
├── configs/experiment.yaml      # single user-facing configuration
├── outputs/
│   ├── poisson_cbf_study_highres/  # preserved original high-resolution results
│   └── reference_results/          # verified CLF/ROA reference runs
├── docs/
├── tests/
└── legacy/hjr/                  # archived HJR demonstration; not used by new entry points
```

## Installation

Ubuntu or another Linux environment with Python 3.10 or newer is recommended.

```bash
unzip Helicopter_Safety_Boxes.zip
cd Helicopter
chmod +x install.sh run_checks.sh scripts/*.sh
./install.sh
source .venv/bin/activate
./run_checks.sh
```

Manual installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e safety_box_core
python -m pip install -e safety_filter_box
python -m pip install -e clf_safety_box
python -m pip install -e contingency_safety_box
python -m pip install -e poisson_safety_box
python -m pip install -e cbf_safety_box
python -m pip install -e vision_poisson_experiments
```

## Generate the first paper figures

### 1. Verified 3-D predefined-world experiment

Fast validation run:

```bash
python experiments/predefined_world/run.py \
  --profile smoke \
  --output outputs/runs/predefined_smoke
```

Full paper-resolution run with forcing and Poisson-solver comparisons:

```bash
python experiments/predefined_world/run.py \
  --compare \
  --output outputs/paper/predefined_world
```

This experiment generates:

- a 3-D obstacle world and filtered landing trajectory;
- XY, XZ, and YZ trajectory projections;
- occupancy and Dirichlet-boundary slices;
- Poisson `h` slices in all coordinate planes;
- 3-D Poisson isosurfaces;
- CLF ROA projections and landing-equilibrium centers;
- Lyapunov phase portraits and closed-loop vector fields;
- `r`-out-of-`p` contingency maps;
- synchronized HOCBF, CLF, ROA, pivot, intervention, slack, and timing histories;
- forcing-field and Poisson-solver comparisons;
- an integrated research dashboard.

### 2. HOCBF, CLF, and ROA parameter sweeps

```bash
python experiments/predefined_world/run_sweeps.py \
  --profile smoke \
  --output outputs/paper/parameter_sweeps
```

The script compares:

- HOCBF gain scales;
- CLF decrease gains;
- ROA scale fractions;
- safety margin, convergence, intervention, contingency, and filter time;
- 3-D and orthogonal trajectory families for each gain sweep.

### 3. Offline experiment from one image

```bash
python experiments/static_image/run.py \
  --profile smoke \
  --image experiments/static_image/input/example_scene.png \
  --compare \
  --output outputs/paper/static_image
```

This mode performs image rectification, segmentation, metric obstacle inflation, occupancy construction, Poisson synthesis, landing simulation, and the same CLF/ROA/contingency analysis.

### 4. Online/video experiment

```bash
python experiments/live_vision/run.py \
  --profile smoke \
  --source experiments/live_vision/assets/example_stream.avi \
  --output outputs/paper/live_vision
```

For a USB camera:

```bash
python experiments/live_vision/run.py \
  --source 0 \
  --display \
  --output outputs/live_camera
```

The dashboard keeps numerical text outside the camera panel and displays the current Poisson, HOCBF, CLF, contingency, and filter status together with bounded real-time histories.

### One-command paper suite

```bash
./scripts/run_paper_figures.sh
```

A faster end-to-end verification suite is:

```bash
./scripts/run_smoke_suite.sh
```

## Output organization

Every new run contains:

```text
run_directory/
├── effective_config.yaml
├── run_metadata.json
├── clf_artifacts/        # P, K, Q, c, eigenvalues, residuals
├── data/                 # CSV, JSON, NPZ, masks, fields
└── figures/              # high-DPI PNG and selected PDF/SVG exports
```

The default paper profile exports 360-DPI PNG and vector PDF/SVG when appropriate. Dense contour and isosurface figures omit SVG intentionally because millions of vector paths produce extremely large, slow files; they still export high-resolution PNG and vector PDF.

## Central configuration

All new experiments use only:

```text
configs/experiment.yaml
```

Examples:

```bash
# Change Poisson forcing
python experiments/predefined_world/run.py --quick \
  --set boxes.poisson.forcing_method=constant

# Change Poisson solver
python experiments/predefined_world/run.py --quick \
  --set boxes.poisson.solver=sparse_direct

# Change HOCBF gains
python experiments/predefined_world/run.py --quick \
  --set boxes.cbf.gamma1=1.6 \
  --set boxes.cbf.gamma2=1.6

# Change CLF decay rate
python experiments/predefined_world/run.py --quick \
  --set boxes.clf.alpha.gain=0.045

# Require three certified alternatives
python experiments/predefined_world/run.py --quick \
  --set boxes.contingency.required_certified=3

# Disable a box for an ablation
python experiments/predefined_world/run.py --quick \
  --set boxes.contingency.enabled=false
```

## Safety-box contracts

Each new constraint provider emits immutable `AffineConstraint` objects using exactly one convention:

```math
A z \ge b.
```

The augmented decision uses named blocks, such as `control`, `omega_contingency`, and `delta_clf`, so modules do not rely on brittle hard-coded column indices.

The filter treats:

- input limits and Poisson HOCBF rows as hard;
- combinatorial ROA rows as hard for the stated contingency condition;
- the active CLF as a prioritized convergence condition with an optional, heavily penalized standard CLF slack;
- nominal performance only in the objective.

The paper-structured shared `omega` coupling remains available. The additional `delta_clf` implements the standard CLF-CBF hierarchy when environmental safety and instantaneous CLF decrease are temporarily incompatible. It is logged explicitly and never relaxes environmental or contingency rows.

## Tests

```bash
./run_checks.sh
```

Individual suites:

```bash
python -m pytest
(cd vision_poisson_experiments && python -m pytest -q)
python scripts/verify_clf_runtime.py
```

## Scientific scope

The verified experiments establish numerical behavior for reduced-order single- and double-integrator models. They do not, by themselves, establish full-order PX4, hardware, estimator, or Martian-aerodynamics guarantees. See [`docs/SAFETY_SCOPE.md`](docs/SAFETY_SCOPE.md).

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/EQUATION_TO_CODE_MAP.md`](docs/EQUATION_TO_CODE_MAP.md)
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
- [`docs/PLOTTING_GUIDE.md`](docs/PLOTTING_GUIDE.md)
- [`docs/MIGRATION_FROM_HJR.md`](docs/MIGRATION_FROM_HJR.md)
- [`docs/SAFETY_SCOPE.md`](docs/SAFETY_SCOPE.md)
- each preserved original package retains its own README, examples, tests, and documentation.
