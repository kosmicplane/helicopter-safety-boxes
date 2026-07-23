# Release Validation

## Source tree

This release is based on the full preserved safety-box workspace. The original Poisson, CBF/HOCBF, and vision packages remain present; the CLF, contingency, shared-contract, and unified-filter packages are additive.

## Automated checks

Executed on the release source tree:

```text
python -m compileall ...       passed
python -m pytest               26 passed
python scripts/verify_clf_runtime.py
                               45 active Python files checked; no active HJR import
```

Historical HJR material remains isolated under `legacy/hjr/` and is not imported by the new predefined-world, static-image, or live-vision entry points.

## Included reference outcomes

Compact outputs are stored in:

```text
outputs/reference_results/paper_release/
```

### Baseline landing

```text
terminal status            landed
final target               LZ0
final position error       0.2280 m
final speed                0.3475 m/s
minimum obstacle clearance 0.8750 m
minimum Poisson h          0.07440
minimum HOCBF residual    -2.66e-11
minimum active CLF residual 7.26e-08
minimum contingency pivot  2116.48
filter time p95             0.339 ms
```

### Single-failure contingency landing

```text
terminal status            landed
failed target              LZ0
switches                   1
final target               LZ2
final position error       0.0998 m
final speed                0.3285 m/s
minimum obstacle clearance 0.8750 m
minimum Poisson h          0.07737
minimum HOCBF residual    -3.54e-11
minimum active CLF residual 1.63e-07
minimum contingency pivot  2993.81
filter time p95             0.355 ms
```

### Sequential-failure stress test

```text
terminal status            hold
failed targets             LZ0, LZ2, LZ3
switches                   2
remaining available site   LZ1
HOLD reason                configured r-out-of-p attraction-region requirement lost
minimum obstacle clearance 0.9143 m
minimum Poisson h          0.01714
minimum HOCBF residual    -2.12e-10
minimum active CLF residual -5.77e-06
minimum contingency pivot  2993.81 before depletion
filter time p95             0.430 ms
```

The stress test is not described as a landing. It demonstrates repeated certified retargeting and deliberate termination when the requested `r=2` contingency property can no longer be supported.

## Interpretation

The small negative residuals above are numerical solver/discretization quantities and must be interpreted against the configured residual tolerance. These reference runs establish reproducible reduced-order numerical behavior on the included machine/configuration; they do not establish full-order PX4 or hardware guarantees.

## Publication workflow

Regenerate final tables and figures on the target machine using:

```bash
bash scripts/run_paper_experiments.sh
```

The output includes the effective configuration, raw metrics, events, CLF artifacts, Poisson field arrays, paper figures, and a cross-scenario index.
