# How to edit

- Change forcing method: `PoissonBoxConfig(forcing_method='guidance')`.
- Change solver: `PoissonBoxConfig(solver='sor')`.
- Change SOR gain: `config.sor.omega = 1.75`.
- Change guidance conservativeness:
  - `config.guidance.beta`
  - `config.guidance.base_flux_strength`
  - `config.guidance.target_mean_abs_scale`
- Change boundary behavior: `config.outer_boundary_as_dirichlet`.
