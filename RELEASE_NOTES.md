# Release Notes — Paper-Ready CLF/ROA Revision

## Full safety-box environment preserved

This release retains the complete original Poisson, CBF/HOCBF, and vision packages. It does not replace them with reduced implementations. The CLF, contingency, shared-contract, and unified-filter packages remain additive and reusable across applications.

## Corrected paper methodology

- HJR is inactive in the new paper runtime and retained only under `legacy/hjr/`.
- Landing feasibility is represented by target-specific CLFs and sublevel ROAs.
- Contingency is evaluated through `h_j^{ROA}=c_j-V_j` and an `r`-out-of-`p` pivot.
- Environmental Poisson-HOCBF safety remains independent and hard.
- Successful landing, successful single-failure diversion, and sequential-failure `HOLD` are separate experiments and are never conflated.

## Revised Mars-analog world

The canonical source is `configs/worlds/mars_analog_landing.yaml`. The 32 m × 24 m × 14 m world includes escarpments, a central mesa, an arch, spires, suspended obstacles, crater rims, low ridges, and four spatially separated landing sites. The direct start-to-primary-site reference intersects occupied geometry.

## Paper figure workflow

`bash scripts/run_paper_experiments.sh` generates:

1. baseline obstacle-rich landing;
2. single-failure contingency landing;
3. sequential-failure stress test ending in `HOLD`;
4. no-failure HOCBF, CLF, and ROA sweeps;
5. forcing and Poisson-solver comparisons;
6. cross-scenario figures and a paper figure index.

## Documentation revision

The root README now proceeds from the scientific question to equations, package contracts, file responsibilities, experiments, figure selection, reproducibility, and guarantee scope. New documents include:

- `docs/SCIENTIFIC_WORKFLOW.md`
- `docs/MARS_ANALOG_SCENARIO_DESIGN.md`
- `docs/PAPER_FIGURE_PLAN.md`
- `docs/FILE_GUIDE.md`
- `docs/THEORY_REFERENCES.md`
- `docs/RELEASE_VALIDATION.md`

## Validation

- Python compilation: passed
- active test suite: `26 passed`
- active HJR-import scan: passed over 45 Python files
- compact reference outputs included for baseline, one-failure landing, sequential-failure `HOLD`, and parameter sweeps
