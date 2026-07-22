# Live Poisson-CBF and HJ Contingency Workflow

## Scope

The live contingency mode extends the fixed-camera vision experiment with an on-screen virtual vehicle, metric landing-zone selection, obstacle-aware paths, finite-horizon landing-zone reachability, and certified target switching. It does **not** command a physical robot, PX4, ROS 2, Gazebo, motors, or actuators.

The implemented reduced model is

\[
\dot p=u,\qquad p=[x,y]^T,\qquad \|u\|_2\le v_{\max}.
\]

For this planar isotropic single integrator, every landing zone `j` has an obstacle-aware geodesic distance field `D_j`. Its HJ/Eikonal value is

\[
V_j(p,\tau)=v_{\max}(-\tau)-D_j(p).
\]

`V_j >= 0` means that the target disk is reachable within the remaining horizon under the reduced model. This is not full multicopter reachability and does not include acceleration, attitude, rates, aerodynamics, battery, wind, or touchdown dynamics.

## Runtime chain

```text
fixed camera/video
  -> workspace calibration
  -> interactive START and landing-zone mission setup
  -> rectification and segmentation
  -> temporal occupancy filtering
  -> physical configuration-space inflation
                       +----------------------------+
                       |                            |
                       v                            v
             Poisson Safety Box             HJ/Eikonal reachability
             h and grad(h)                   D_j, V_j, paths, pivot
                       |                            |
                       v                            v
              Poisson CBF row              active + contingency rows
                       +-------------+--------------+
                                     v
                       CBFBox.filter_affine_constraints
                                     v
                             safe virtual velocity
```

Poisson and HJR consume the same versioned occupancy map **in parallel**. Poisson is not an input to the HJ equation.

## Interactive flow

1. Click the workspace corners in the existing order: top-left, top-right, bottom-right, bottom-left.
2. In the rectified mission window, click START.
3. Click landing-zone centers. A common metric radius is drawn by the program.
4. Select the active target and required `r` value.
5. Confirm only after all metric and occupancy checks pass.
6. During the live run, temporally persistent occupancy over the active landing disk causes rejection and a switch to a certified reachable alternative.

## Backward compatibility

If `reachability.enabled` is false, `run_experiment.py` selects the original `LivePoissonPipeline`. Existing static and baseline live experiments remain available.

## Safety behavior

The virtual marker moves only when calibration is valid and a fresh synchronized snapshot exists. The snapshot must contain Poisson and HJR results with one identical occupancy version. A failed optimizer, stale result, camera movement, invalid Poisson sample, negative pivot, or fewer than `r` reachable zones produces a zero velocity and a visible HOLD state.
