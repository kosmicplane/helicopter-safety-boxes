# Mars-Analog Scenario Design

## Purpose

The flagship predefined world is intentionally larger than the original proof-of-concept map and is designed to make both environmental safety and landing contingency visible. A valid run must not be a trivial high-altitude transfer followed by a vertical descent at the boundary. The straight start-to-primary-site reference intersects occupied geometry, while the feasible route requires lateral corridor selection, vertical clearance, and a final crater-rim descent.

## Canonical world

The source of truth is:

```text
configs/worlds/mars_analog_landing.yaml
```

The canonical dimensions are:

```text
32 m × 24 m × 14 m
```

The world contains four controlled landing equilibria:

| ID | Position [m] | Operational role |
|---|---:|---|
| `LZ0` | `[29.0, 21.0, 1.2]` | primary crater floor |
| `LZ1` | `[29.0, 4.0, 1.2]` | southern crater floor |
| `LZ2` | `[16.5, 20.5, 1.2]` | northwestern contingency basin |
| `LZ3` | `[18.5, 3.8, 1.2]` | western contingency pad |

## Geometric roles

| Geometry | Purpose in the validation |
|---|---|
| western escarpment | blocks the direct route from the initial condition |
| central mesa | prevents a trivial post-corridor straight descent |
| arch posts and beam | create coupled lateral/vertical clearance requirements |
| tall spires | produce close-proximity HOCBF intervention |
| suspended boulder and shelf | require vertical as well as lateral avoidance |
| crater rims | make the terminal approach a constrained landing rather than point tracking in free space |
| low ridges and rock fields | test final-approach clearance and alternative-site geometry |

## Controlled experiment matrix

### Baseline

No landing site fails. The system must avoid the obstacle field and satisfy both position and speed touchdown conditions at `LZ0`.

### Single-failure contingency

The active primary site is invalidated after hazard confirmation. The controller must select an available target whose CLF/ROA certificate remains valid, retarget, and complete a landing. In the included reference run, the final target is `LZ2`.

### Sequential-failure stress test

Successive active targets are invalidated. The system retargets while the configured `r`-out-of-`p` requirement remains supportable and enters `HOLD` after contingency is exhausted. This run is not described as a landing.

### Parameter studies

HOCBF, CLF, ROA, forcing, and solver comparisons use the fixed baseline mission without target failures. This prevents failure timing from confounding the independent variable.

## Reference outcomes included in the release

The repository contains compact validation outputs under:

```text
outputs/reference_results/paper_release/
```

The recorded outcomes are:

- baseline: successful touchdown at `LZ0`;
- single failure: one certified switch and touchdown at `LZ2`;
- sequential failure: three invalidations, repeated retargeting, and `HOLD` because the `r=2` requirement can no longer be maintained.

These reference artifacts demonstrate the intended behavior, but publication tables should be regenerated in the target computing environment and reported with the associated configuration and hardware information.
