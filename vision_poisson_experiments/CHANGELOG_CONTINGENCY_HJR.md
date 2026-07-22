# Contingency HJR Extension Changelog

## Added

- interactive START, landing-zone, active-target, radius, and `r` mission setup;
- metric mission validation and JSON persistence;
- landing-zone blocking hysteresis and state manager;
- planar single-integrator HJ/Eikonal reachability;
- metric 8-connected geodesic fields, predecessor paths, and Eikonal diagnostics;
- r-out-of-p pivot and reachable-count maps;
- certified target switching and active-horizon reset;
- synchronized latest-only Poisson/HJR worker;
- obstacle-aware path extraction and pure-pursuit guidance;
- one unified Poisson-CBF + active-HJ + contingency-HJ projection through `cbf_safety_box`;
- hard HOLD behavior and discrete-time backtracking;
- live HJR/path/target overlays and synchronized HJR snapshots;
- five deterministic validation scenarios;
- new tests and documentation.

## Changed

- the live entrypoint selects `LiveContingencyPipeline` only when `reachability.enabled=true`;
- solve metrics include synchronized Poisson and HJR timings;
- the active target is evaluated with its active horizon, while r-out-of-p alternatives use the contingency horizon.

## Preserved

- original static experiments;
- original live Poisson worker behavior when reachability is disabled;
- authoritative external Poisson and CBF package implementations;
- webcam, local file, HTTP/MJPEG, and RTSP source handling.
