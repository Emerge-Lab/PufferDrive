# PufferDrive Map Notes

## Map Storage

### RoadMapElement (polyline)

Every map feature (lane, road edge, paint line) is one `RoadMapElement`. Geometry is a polyline of
`segment_size` points in parallel flat arrays: `x[]`, `y[]`, `z[]`, `headings[]`.

A **segment** is the edge between consecutive points `i → i+1`. Loops over segments use
`j < segment_size - 1`. There are always `segment_size - 1` segments per element.

`headings[j]` = `atan2(y[j+1]-y[j], x[j+1]-x[j])`, precomputed so hot paths avoid trig.

### Road element types (datatypes.h)

| Range | Meaning |
|-------|---------|
| 0–9   | Drivable lanes (freeway, surface street, bike lane…) |
| 10–19 | Road lines (painted markings) |
| 20–29 | Road edges (curbs, medians — hard boundary) |
| 30+   | Misc (crosswalk, speed bump) |

Lanes (0–9) additionally store `entry_lanes[]` and `exit_lanes[]` — indices into the global
`road_elements[]` array — forming a directed connectivity graph.

### Segment lengths (measured from CARLA .bin files)

Polylines come from the external **123Drive `serialize.py`** binary creator, pre-sampled at ~10m
resolution. `drive.h` does no splitting; it reads points as-is from the binary.

| Map    | # segments | Median | Max   | >5m |
|--------|------------|--------|-------|-----|
| Town01 | 2,056      | 9.2m   | 10.0m | 68% |
| Town02 | 1,067      | 8.1m   | 10.0m | 57% |
| Town03 | 5,744      | 4.5m   | 10.0m | 45% |
| Town04 | 10,929     | 7.3m   | 10.0m | 80% |

The comment at line 125 of `drive.h` explains `MAX_ENTITIES_PER_CELL`:

```c
// Depends on resolution of data Formula: 3 * (2 + GRID_CELL_SIZE*sqrt(2)/resolution)
```

With `resolution=10m` and `GRID_CELL_SIZE=5m`: `3 * (2 + 5*1.414/10) = 8.1`.
`MAX_ENTITIES_PER_CELL = 30` is set well above this as a safety buffer.

---

## GridMap (init_grid_map)

Spatial hash built once at load time. World is divided into 5×5m cells (`GRID_CELL_SIZE`).
Each cell stores a list of `GridMapEntity { entity_idx, geometry_idx }` — which road element
and which segment falls in that cell.

### Build steps

1. **Bounding box** — scan all lane/edge points (Y grows upward: `top_left_y` = max Y).
2. **Grid dimensions** — `grid_cols = ceil(width/5)`, `grid_rows = ceil(height/5)`.
3. **First pass (count)** — for each segment, compute its representative point, find its cell,
   increment `cell_entities_count[cell]`. No data stored yet.
4. **Allocate** — `cells[cell]` gets an exact-size array from the count above.
5. **Second pass (fill)** — same loop, call `add_entity_to_grid` to write
   `(entity_idx, geometry_idx)` into each cell. Also mark cells with drivable lane segments.
6. **Compact drivable list** — `grid_index_drivable[]` stores only cells that contain at least
   one drivable segment. Used by `spawn_agent` for O(1) random drivable-position sampling.

### Current bug: striped spawn distribution

**Root cause:** `init_grid_map` uses the **midpoint** of each segment to assign it to a cell.
Segments are ~10m long; cells are 5m. The midpoint falls in one cell, but the segment physically
spans two adjacent cells. The adjacent cells get no record of this segment and are excluded from
`grid_index_drivable`.

Consequence: along any straight road, only every other ~5m cell is drivable. `spawn_agent` picks
uniformly from `grid_index_drivable`, producing the **striped spawn density** visible in the
spawn heatmaps.

```
Cell A (x: 0–5m)   Cell B (x: 5–10m)   Cell C (x: 10–15m)
│                  │                    │
│  seg0 start      │ ← midpoint (5m)    │  seg0 end (10m)
│  ────────────────┼────────────────────│
│  NOT in grid     │  registered here   │  NOT in grid
│                  │                    │
│  seg1 starts     │                    │  seg1 midpoint
```

### Fix: use ¼ and ¾ points instead of midpoint

In `init_grid_map`, replace the midpoint sample with two probe points at ¼ and ¾ along each
segment. If both fall in the same cell, register once; if they differ, register the segment in
both cells. This applies to both passes (count and fill).

```c
// Replace the single midpoint with two quarter-points
float q1_x = element->x[j] + 0.25f * (element->x[j+1] - element->x[j]);
float q1_y = element->y[j] + 0.25f * (element->y[j+1] - element->y[j]);
float q3_x = element->x[j] + 0.75f * (element->x[j+1] - element->x[j]);
float q3_y = element->y[j] + 0.75f * (element->y[j+1] - element->y[j]);

int grid_index_q1 = get_grid_index(env, q1_x, q1_y);
int grid_index_q3 = get_grid_index(env, q3_x, q3_y);

// Register in first cell (always)
cell_entities_count[grid_index_q1]++;          // or add_entity_to_grid(...)

// Register in second cell only if it differs
if (grid_index_q3 != grid_index_q1) {
    cell_entities_count[grid_index_q3]++;      // or add_entity_to_grid(...)
}
```

This guarantees every 5m portion of a segment is covered, at the cost of at most 2× grid entries
(safe because segments cap at 10m = 2 cells max).

Note: `MAX_ENTITIES_PER_CELL` may need to increase from 30 to 60 if segments can now appear in
two cells each. Recalculate: `3 * (2 + 5*1.414/10) * 2 ≈ 16.2` — still well under 30, so no
change needed.

---

## Agent Spawn (GIGAFLOW mode, spawn_agent)

1. Rejection loop (up to 30 attempts):
   - Pick random cell from `grid_index_drivable`
   - Pick random drivable segment in that cell
   - Read position + heading from `x[geometry_idx]`, `y[geometry_idx]`, `headings[geometry_idx]`
   - Reject if: OBB collision with existing agents, corners cross a road edge, sitting on red stop line
2. On success: set `sim_x/y/z/heading`, randomize vehicle size, set initial speed.
3. Generate route via random walk through `exit_lanes` graph (`generate_random_route`).
4. Interpolate `Path` waypoints at `min_waypoint_spacing` intervals (`build_path`).
5. Place N goal positions along path (`compute_goals`).
