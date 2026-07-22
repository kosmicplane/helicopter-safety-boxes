# Mission Selection UI

## Purpose

After the existing four-corner workspace calibration, the mission UI defines one virtual start and a set of physically sized landing zones on the rectified metric image.

## Controls

| Input | Action |
|---|---|
| First left click | Set START |
| Later left clicks | Add landing-zone centers |
| Right click | Remove nearest landing zone |
| Backspace | Undo the latest selection |
| `a` | Cycle initial active target |
| `[` / `]` | Decrease / increase common radius |
| `-` / `+` | Decrease / increase required `r` |
| `r` | Reset mission selections |
| Enter | Confirm a valid mission |
| Escape | Cancel |

The user selects only centers. The displayed touchdown region is a metric disk. If x and y pixel scales differ, it appears as an ellipse in the image while remaining a circle in physical coordinates.

## Validation

A mission is rejected when any of the following holds:

- too few or too many landing zones;
- invalid `r`;
- a radius outside configured bounds;
- a radius smaller than robot radius plus perception and touchdown margins;
- a complete disk outside the workspace or boundary-clearance requirement;
- overlapping landing zones;
- START inside a landing disk or inflated obstacle;
- a landing disk intersecting initial raw or inflated occupancy;
- too few grid cells in a target seed.

Invalid selections are displayed in red with explicit reasons. Enter never accepts an invalid mission.

## Persistence

Interactive missions can be stored in JSON. Saved records include metric coordinates, source pixels, radius, active index, `p`, `r`, workspace dimensions, calibration identifier, and timestamp. `mission_setup.mode: load_file` makes headless and repeatable runs possible.
