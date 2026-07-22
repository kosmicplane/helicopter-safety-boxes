# Failure and HOLD Behavior

The virtual marker receives exactly zero velocity when any safety prerequisite fails.

## HOLD triggers

- invalid or moved-camera calibration;
- no synchronized Poisson/HJR snapshot;
- stale snapshot;
- mismatched occupancy versions;
- invalid or occupied Poisson sample;
- fewer than `r` available or reachable landing zones;
- negative or nonfinite r-out-of-p pivot;
- active target failure with no certified replacement;
- invalid or collision-containing path;
- unified optimizer failure or residual violation;
- failed discrete-time backtracking check.

## No unsafe fallback

HOLD never substitutes the nominal command, an old command, or an uncertified target. The last plotted field may remain visible for diagnosis, but stale fields cannot move the virtual marker.

## Discrete-time guard

The continuous-time filter is followed by a bounded Euler-step backtracking check. Candidate endpoints and full segments must remain inside the workspace, outside inflated occupancy, and above the configured Poisson margin. If all step sizes fail, the marker remains at its current position.

## Recovery

The live runtime resumes only after a new synchronized valid snapshot and feasible unified solution exist. Rejected landing zones remain latched unless the operator explicitly clears them.
