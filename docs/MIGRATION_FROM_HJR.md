# Migration from HJR to CLF-Based Contingency

## Methodological decision

The active landing methodology uses the combinatorial-stabilization branch: local CLFs construct certified inner approximations of attraction regions, and a combinatorial certificate preserves at least `r` among `p` alternatives. The new experiment entry points do not use finite-horizon HJ/HJR value functions.

## Mapping

| Historical HJR object | Active CLF/ROA object |
|---|---|
| time-indexed reach-avoid value field | target-specific local CLF \(V_j(x)\) |
| HJR value zero-superlevel set | CLF sublevel set \(V_j\le c_j\) |
| `tau_active`, `tau_contingency` | not used |
| HJR value gradient | analytic \(\nabla V_j=2P_je_j\) |
| reach-avoid pivot | ROA pivot \(\max^{(r)}\{c_j-V_j\}\) |
| finite-horizon target row | active CLF decrease row |

## Preserved historical code

The complete original vision package is retained because it supports other applications and provides regression value. Historical HJR-specific demonstrations are also stored under `legacy/hjr/`. They are not imported by:

- `experiments/predefined_world`;
- `experiments/static_image`;
- `experiments/live_vision`;
- `clf_safety_box`;
- `contingency_safety_box`;
- `safety_filter_box`.

Run `python scripts/verify_clf_runtime.py` to verify this separation.

## Scientific distinction

A CLF sublevel set is a local infinite-horizon stabilization certificate under its stated model and feedback assumptions. It is not a shrinking finite-horizon backward reach-avoid set. Consequently, the active paper should use the terms `region of attraction`, `certified landing alternative`, and `combinatorial stabilization`, rather than claiming HJR reachability.
