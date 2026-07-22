# Plotting Guide

The plotting subsystem is shared by all three experiment modes and is implemented in `experiments/common/plotting.py`. The design follows the visual language of the preserved high-resolution outputs: white backgrounds, consistent typography, restrained annotations, synchronized axes, explicit units, and separate panels for geometry, certificates, and numerical diagnostics.

## Exports

The paper profile uses 360-DPI PNG plus PDF and SVG where appropriate. Dense contours and 3-D isosurfaces export PNG and PDF but intentionally omit SVG to avoid files containing millions of paths.

```yaml
visualization:
  dpi: 360
  save_pdf: true
  save_svg: true
```

## Figure families

### World and trajectories

- full 3-D trajectory with obstacle geometry and landing equilibria;
- XY, XZ, and YZ projections with identical trajectory time coloring;
- nominal versus filtered trajectories;
- parameter-family trajectories for HOCBF and CLF sweeps.

### Poisson field

- occupancy and Dirichlet-frontier slices;
- forcing, safety field, gradient magnitude, and reconstructed Laplacian;
- XY, XZ, and YZ field planes;
- volumetric isosurfaces;
- forcing-method comparison;
- solver timing, residual, and field-error comparison.

### Lyapunov and attraction regions

- contours of \(V_j\) with the \(V_j=c_j\) boundary;
- landing-equilibrium centers;
- projected ROA ellipses or ellipsoids;
- phase portraits and closed-loop vector fields;
- synchronized \(V_j(t)\), \(h_j^{ROA}(t)\), CLF residual, and CLF slack.

The spatial ROA plots use a Schur-complement projection of the full position–velocity quadratic form. They are labeled as projections and must not be interpreted as the full 4-D or 6-D attraction region.

### Contingency

- each local ROA certificate;
- `r`-th-largest pivot map;
- certified-alternative count map;
- critical-certificate identity map;
- target availability heatmap and event markers;
- all combinatorial residuals and shared \(\omega\).

### Integrated dashboards

The saved dashboard combines trajectory, safety field, certificate histories, intervention, contingency, and timing. The live dashboard places status text and history plots outside the camera panel to prevent occlusion.

## Adding a figure

1. Add a pure plotting function to `experiments/common/plotting.py`.
2. Accept already-computed data rather than recomputing the experiment.
3. Use `configure_academic_style()` and `save_figure()`.
4. Include units in axis labels.
5. Save raw data used by the figure.
6. Add the equation and plot to `docs/EQUATION_TO_CODE_MAP.md` when it supports a formal claim.
