# Synchronization and Occupancy Versioning

## Motivation

Combining a Poisson field from one map with HJ fields from another invalidates both residuals and switching decisions. The live extension therefore synthesizes all map-dependent objects under one monotonic occupancy version.

## Worker

`SafetySynthesisWorker` has a queue capacity of one. A new task replaces an older waiting task. Each task contains raw, filtered, and inflated occupancy plus target availability. One accepted result contains:

- Poisson result;
- all geodesic/HJ fields;
- landing-zone assessments;
- identical occupancy version;
- solve and completion times.

A completed result is discarded if a newer version was submitted while it was solving.

## Controller gate

Movement requires:

- valid fixed-camera calibration;
- accepted synchronized snapshot;
- matching Poisson and HJR versions;
- acceptable field age;
- valid path for the same target and version;
- feasible unified filter.

## Metrics

The runtime logs queue replacements, obsolete solves, failures, invalid solves, Poisson time, HJR time, end-to-end field age, optimizer time, and the exact version used by each controller sample and snapshot.
