# Scientific Workflow

This document describes the complete experimental methodology from environment definition to quantitative paper results.

## 1. Define one reproducible scenario

The predefined experiment separates controller parameters from obstacle geometry:

```text
configs/experiment.yaml
configs/worlds/mars_analog_landing.yaml
```

`experiment.yaml` defines dynamics, gains, bounds, landing sites, failure schedules, sweeps, and plotting settings. `mars_analog_landing.yaml` defines the physical obstacle world. This separation allows the same controller to be evaluated on another world without editing Python source.

The world loader validates:

- supported obstacle type;
- finite coordinates;
- unique names;
- obstacle bounds within the workspace;
- unoccupied start and target locations.

## 2. Rasterize geometry

`experiments/predefined_world/world.py` samples analytic boxes, cylinders, ellipsoids, and annular cylinders onto a metric grid. A physical inflation margin accounts for the vehicle radius and modelled perception/geometric margin.

The resulting occupancy tensor is

\[
O_{ijk}\in\{0,1\}.
\]

The outer computational boundary is included in the Dirichlet boundary when configured.

## 3. Construct the Poisson problem

From occupancy, the Poisson package constructs:

- free-domain mask \(\Omega\);
- boundary mask \(\partial\Omega\);
- sparse finite-difference operator \(A_h\);
- forcing vector \(\mathbf b\).

It solves

\[
A_h\mathbf h=\mathbf b
\]

and verifies

\[
r_A=A_h\mathbf h-\mathbf b.
\]

The solution is reshaped into \(h_P\), and spatial derivatives are computed on the physical grid.

## 4. Build landing-site CLFs

For every configured site:

1. create equilibrium \(x_j^\star=[p_j^{\star\top},0^\top]^\top\);
2. construct or load \(K_j\);
3. verify \(A-BK_j\) is Hurwitz;
4. solve the continuous Lyapunov equation;
5. verify \(P_j\succ0\) and the residual;
6. compute an input-compatible sublevel threshold \(c_j\);
7. save JSON/NPZ artifacts.

The output is a target-indexed set

\[
\{V_j,h_j^{\mathrm{ROA}},K_j,P_j,c_j\}_{j=1}^{p}.
\]

## 5. Plan a nominal path

The predefined-world planner uses A* over free voxels with a clearance penalty. A lookahead PD law converts the path into nominal acceleration.

This planner supports progress and produces interpretable reference paths. It is not used as the source of formal safety or contingency guarantees.

## 6. Evaluate certificates at each control step

At time \(t_k\):

1. sample \(h_P,Dh_P,D^2h_P\) at the current position;
2. evaluate every \(V_j\) and \(h_j^{\mathrm{ROA}}\);
3. apply the target availability mask;
4. compute \(\widetilde h_r\), critical IDs, and certified count;
5. build environmental HOCBF, active CLF, and combinatorial rows;
6. solve the unified minimum-intervention problem;
7. verify all affine residuals;
8. integrate the double integrator;
9. log states, certificates, events, and timing.

## 7. Process target failures

A failure event may name a fixed site or use `target: active`. The latter invalidates whichever site is currently pursued.

When the active target becomes unavailable, the default policy selects the available certificate with maximum ROA margin. A switch occurs only among available CLF-derived certificates.

If the required \(r\)-out-of-\(p\) set is not satisfied and `hold_when_lost` is enabled, the runtime returns `HOLD` before integrating another state.

## 8. Evaluate terminal conditions

A run is `landed` only if

\[
\|p-p_j^\star\|\leq\varepsilon_p,
\qquad
\|v\|\leq\varepsilon_v.
\]

Otherwise the terminal status is `hold` or `timeout`. This status controls figure titles and markers.

## 9. Produce paper outputs

Every run saves:

- effective configuration;
- raw trajectory and certificate histories;
- event log;
- Poisson arrays and solver metadata;
- CLF artifacts;
- publication figures;
- machine-readable summary.

Parameter sweeps are executed without target failures unless the research question is explicitly about failure timing. This isolates the parameter effect.

## 10. Interpret results

A successful nominal run establishes only the tested reduced-order result. A single-failure run shows use of one preserved alternative. A sequential-failure run establishes whether the runtime can detect contingency depletion and enter HOLD. None of these runs alone establishes full-order flight safety.
