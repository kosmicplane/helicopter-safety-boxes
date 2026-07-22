# Validation Status

This release preserves the complete original Poisson, CBF/HOCBF, and vision workspaces and adds independently installable CLF, contingency, shared-contract, and unified-filter packages.

## Verified in the release environment

- Python compilation of active packages and experiment entry points;
- root and original CBF/Poisson tests;
- original vision test suite;
- active-runtime separation check;
- predefined 3-D smoke experiment, including target failure and diversion;
- static-image smoke experiment;
- live-video smoke experiment;
- ZIP integrity check.

The exact command outputs, reference metrics, and figure review are recorded in `docs/FINAL_VALIDATION_REPORT.md`.

## Interpretation

Passing these checks establishes reproducibility of the software pathways in the tested environment. It does not upgrade the reduced-order numerical experiments to a full-order PX4 or hardware safety theorem. See `docs/SAFETY_SCOPE.md`.
