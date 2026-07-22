# Experiment 01 — Static Image to Poisson Field and CBF

This experiment converts a planar photograph into a metric occupancy grid, inflates obstacles by
`robot_radius + perception_margin`, synthesizes and compares Poisson safety fields, and runs a
single-integrator CBF safety-filter simulation using bilinearly interpolated numerical field values.

## Reproducible synthetic run

```bash
python 01_static_image_poisson_cbf/run_experiment.py \
  --image examples/assets/image.png \
  --config 01_static_image_poisson_cbf/config_synthetic.yaml \
  --output sample_outputs/static_demo \
  --assume-top-down --headless
```

## Real photograph

Set `calibration.mode: interactive`, run with `--interactive`, and click the four workspace corners in the order
specified in the root README. Use `--mask path/to/mask.png` for `mask_file` mode or
`--background empty_workspace.png` for `background_reference` mode.

Useful overrides:

```text
--start X,Y --goal X,Y
--solver sparse_direct
--forcing constant --forcing distance --forcing average_flux --forcing guidance
--selected-forcing guidance
--cbf-alpha 1.0 --dt 0.02 --max-speed 0.5 --max-steps 1000
--no-cbf
```

## Main outputs

- `preprocessing_figures/`: original image, rectification, masks, and occupancy;
- `poisson/<forcing>/`: arrays, exact configuration, validation, and full diagnostics;
- `poisson/forcing_comparison/`: same-map forcing comparisons;
- `solver_comparison/`: identical-input solver benchmark when enabled;
- `cbf_simulation/`: nominal/safe trajectories, signals, CSV, NPZ, and summary;
- `experiment_summary.json`: compact machine-readable result.

The clearance column in trajectory CSV files is a diagnostic distance transform. It is never substituted for the
Poisson safety function used by the CBF.
