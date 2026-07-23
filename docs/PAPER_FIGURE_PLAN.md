# Paper Figure Plan

This document defines a compact, nonredundant figure suite for the **Approach** and **Results** sections. Each figure answers one scientific question and maps to one reproducible script or data product.

## 1. Figure-selection principles

1. **Separate construction from performance.** Approach figures show how mathematical objects are created; Results figures show whether they work.
2. **Separate landing success from failure exhaustion.** A successful contingency landing and a terminal `HOLD` are different claims and must not share the same title or interpretation.
3. **Show intermediate evidence.** Occupancy, boundary, forcing, derivatives, residuals, and certificate values must support every trajectory claim.
4. **Avoid duplicate panels.** Full 3-D fields and all plane slices should not both occupy main-paper space unless they answer different questions.
5. **Use one fixed scenario for parameter sweeps.** Geometry, start, target, and failure schedule must be identical across the sweep.
6. **Label projections honestly.** A spatial ellipse or ellipsoid is a projection/slice of a higher-dimensional ROA, not the complete state-space region.

---

## 2. Approach-section figures

### Figure A — End-to-end multi-certificate architecture

**Question:** How does information move from perception/geometry to the final safe control input?

**Panels:**

1. image/predefined world;
2. occupancy and boundary;
3. Poisson solve;
4. local field sample;
5. HOCBF row;
6. per-target CLFs and ROA certificates;
7. `r`-out-of-`p` composition;
8. unified filter;
9. reduced-order vehicle.

**Equations:**

\[
\Delta h_P=f_P,\qquad h_P|_{\partial\Omega}=0,
\]

\[
V_j=e_j^\top P_je_j,\qquad h_j^{\mathrm{ROA}}=c_j-V_j,
\]

\[
\widetilde h_r=\max^{(r)}\{h_j^{\mathrm{ROA}}\},
\]

\[
u_{\rm safe}=\arg\min_u\|u-u_{\rm nom}\|^2
\quad\text{s.t. certificate rows.}
\]

**Source:** schematic generated specifically for the paper; do not use a runtime dashboard as the architecture figure.

---

### Figure B — Poisson safety-function construction

**Question:** How does occupancy geometry become a differentiable safety function?

**Recommended 2×3 layout:**

1. analytic or perceived obstacle geometry;
2. occupancy \(O\);
3. Dirichlet boundary \(\partial\Omega\);
4. forcing \(f_P\);
5. safety function \(h_P\);
6. gradient magnitude and representative vectors.

**Main claim:** the obstacle boundary is fixed while the forcing changes interior field geometry.

**Code/data:**

- `experiments/common/poisson_field.py`
- `plot_occupancy_boundary_slices`
- `plot_poisson_diagnostics`

---

### Figure C — Relative-degree-two safety row

**Question:** What information from the Poisson field enters the HOCBF?

**Panels:**

1. \(h_P\) slice;
2. \(\|D h_P\|\);
3. curvature term \(v^\top D^2h_Pv\);
4. HOCBF residual along one representative trajectory.

**Equation:**

\[
D h_Pa
+v^\top D^2h_Pv
+(\gamma_1+\gamma_2)D h_Pv
+\gamma_1\gamma_2h_P
\ge0.
\]

---

### Figure D — CLF and combinatorial ROA construction

**Question:** How are landing alternatives represented and combined?

**Recommended layout:**

1. projected CLF level sets for all targets;
2. one phase portrait with closed-loop vector field;
3. per-site \(h_j^{\mathrm{ROA}}\) maps on a declared state slice;
4. pivot \(\widetilde h_r\);
5. certified-count map;
6. critical-certificate identity map.

**Required caption language:** the ellipses/ellipsoids are projections or slices of the full state-space ROAs.

---

### Figure E — Unified optimization problem

**Question:** Which requirements are hard, which are relaxed, and what is optimized?

Use a compact mathematical block rather than another trajectory plot:

\[
\min_{a,\omega,\delta_V}
\frac12\|a-a_{\rm nom}\|_R^2
+c_\omega\omega^2+p_V\delta_V^2
\]

subject to

- acceleration/velocity limits;
- hard Poisson HOCBF;
- active CLF;
- `r`-out-of-`p` rows;
- \(\omega\ge0\), \(\delta_V\ge0\).

A side panel should state the priority hierarchy.

---

## 3. Results-section figures

### Figure 1 — Mars-analog obstacle-rich successful landing

**Question:** Can the full stack navigate a constrained descent corridor and complete touchdown?

**Scenario:** no target failure.

**Required panels:**

1. 3-D world with terrain, start, nominal reference path, filtered trajectory, and touchdown;
2. XY top view showing corridor traversal;
3. XZ view showing vertical avoidance/descent;
4. YZ view showing lateral separation from terrain.

**Required metrics in caption or table:**

- final target;
- final position error;
- final speed;
- minimum \(h_P\);
- minimum HOCBF residual;
- maximum intervention;
- solve-time p95.

**Terminal status:** `landed`.

---

### Figure 2 — Successful single-failure contingency landing

**Question:** Can the system reject the primary site, replan to a certified alternative, and still land?

**Scenario:** one state- or time-triggered invalidation of the active target.

**Required visual encoding:**

- trajectory segments colored by active target;
- failed site marked explicitly;
- target-switch marker;
- final touchdown marker;
- nominal reference before and after switch;
- available landing disks and unavailable-site symbol.

**Companion histories:**

1. all \(h_j^{\mathrm{ROA}}(t)\);
2. pivot \(\widetilde h_r(t)\);
3. certified and available counts;
4. active target as a categorical step signal;
5. Poisson/HOCBF and CLF residuals.

**Terminal status:** `landed`.

---

### Figure 3 — Sequential-failure stress test ending in HOLD

**Question:** Does the implementation stop honestly when the requested contingency guarantee is no longer supportable?

**Scenario:** successive invalidation of the active target until fewer than \(r\) certified alternatives remain.

**Required title:** `Sequential landing-zone failure stress test ending in certified HOLD`.

**Do not call this a landing trajectory.**

**Required panels:**

1. 3-D trajectory with each failure and switch;
2. available/certified count with horizontal `r` threshold;
3. pivot with zero boundary;
4. active-target signal;
5. HOLD event and terminal reason.

**Terminal status:** `hold`.

---

### Figure 4 — HOCBF alpha sensitivity

**Question:** How does the class-\(\mathcal K\) tuning change conservatism, safety margin, and terminal performance?

**Main-paper spatial panel:** show 5–6 representative trajectories on the same world.

- solid line: landed;
- dashed line: HOLD;
- dotted line: timeout;
- terminal marker indicates outcome.

**Quantitative panels:**

- minimum \(h_P\);
- minimum HOCBF residual;
- mean and maximum intervention;
- path length;
- landing time/rollout duration;
- terminal error;
- filter p95.

Do not interpret a run that timed out as a landing.

---

### Figure 5 — CLF decrease-rate and ROA scaling

**Question:** How do CLF convergence aggressiveness and attraction-region size affect touchdown and contingency?

Recommended split:

- CLF alpha gain versus convergence/intervention;
- ROA fraction versus initial certified count, minimum pivot, and feasibility;
- representative \(V(t)\) and \(h^{\mathrm{ROA}}(t)\) histories.

---

### Figure 6 — Forcing-function comparison

**Question:** How does the Poisson forcing change the field and the filtered trajectory under fixed geometry and controller parameters?

**Panels:**

1. forcing fields;
2. \(h_P\) slices;
3. gradient vectors/magnitude;
4. filtered trajectories;
5. minimum \(h_P\), intervention, landing time, and field-construction time.

---

### Figure 7 — Poisson solver accuracy–time trade-off

**Question:** Which numerical solver provides sufficient field accuracy within the mapping-rate budget?

**Required metrics:**

- total wall time;
- solve-stage time;
- iteration count;
- exact algebraic residual \(\|Ah-b\|/\|b\|\);
- relative field error versus sparse direct;
- reconstructed-Laplacian error;
- optional derivative error.

A solver should not be selected on timing alone.

---

### Figure 8 — Integrated evidence dashboard

Use once, near the end of Results or in supplementary material.

The dashboard should connect:

- trajectory;
- local field value;
- HOCBF residual;
- active CLF value/residual;
- pivot and certified count;
- intervention;
- solver timing.

It should summarize evidence, not replace the focused figures above.

---

## 4. Main paper versus appendix

### Main paper

- Architecture
- Poisson construction
- CLF/ROA/contingency construction
- Successful landing
- Successful single-failure landing
- Sequential-failure HOLD
- One compact alpha/forcing/solver comparison each

### Appendix or supplementary material

- all occupancy slices;
- all plane slices;
- dense 3-D isosurfaces;
- every alpha trajectory;
- complete solver residual histories;
- full phase portraits for every target;
- static-image and live-vision dashboards;
- additional ablations.

---

## 5. Experiment integrity checks

Before a figure is accepted for the paper, verify:

1. `terminal_status` matches the title and marker;
2. a reported landing satisfies both position and speed tolerances;
3. the minimum HOCBF residual is above the declared numerical tolerance;
4. the contingency pivot remains nonnegative until a planned HOLD;
5. the start and all landing sites are geometrically free;
6. the direct or nominal path genuinely intersects/approaches terrain closely enough to exercise the safety layer;
7. the figure uses physical units;
8. all parameter comparisons use the same world and initial condition;
9. raw CSV/JSON/NPZ data are saved with the figure;
10. the caption distinguishes theorem-supported reduced-order properties from numerical observations.
