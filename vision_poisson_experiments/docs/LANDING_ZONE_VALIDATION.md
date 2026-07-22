# Landing-Zone Validation and State Machine

## States

Every landing zone has one state:

- `AVAILABLE`: visually valid candidate;
- `ACTIVE`: current target;
- `BLOCKED`: persistently unsafe when latching is disabled;
- `UNREACHABLE`: no nonnegative HJ certificate at the contingency horizon;
- `REJECTED`: latched removal after confirmed blocking or mission logic;
- `REACHED`: virtual marker entered the touchdown disk.

## Per-map assessment

For each metric disk, the runtime computes:

1. raw occupied fraction;
2. inflated occupied fraction;
3. minimum obstacle clearance;
4. count of free target-seed cells;
5. local connection to surrounding free space.

Start-to-target connectivity is established separately by the geodesic field.

## Hysteresis

A single noisy frame cannot reject a target. `blocked_activation_frames` consecutive blocked assessments are required. A rejected target remains latched when `latch_rejected_zones` is true. Manual reset is explicit; there is no automatic restoration from one clear frame.

## Target switching

When the active disk is blocked/rejected or its active-horizon HJ value is negative, the target manager considers only certified alternatives. Candidate ranking uses positive reachability margin first, then shorter geodesic distance, landing clearance, and optional priority. A switch resets the active horizon to the reserved contingency horizon.

If fewer than `r` zones are available or reachable, the system enters HOLD before attempting any switch. There is no uncertified fallback target.
