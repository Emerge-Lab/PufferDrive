# ── Jump-lane helpers — Python ports of drive.h ────────────────────────────────
#
# build_path_w_jump injects a single lane-change into an agent's existing path:
#   1. Pick a random waypoint JUMP_MIN_AHEAD..JUMP_MAX_AHEAD steps ahead of spawn.
#   2. find_parallel_lane_candidate: nearest drivable lane within [1, 5.55 m]
#      lateral offset, <30° heading diff, not blocked by solid lines/road edges.
#   3. Re-interpolate the path tail from that lane onward (same logic as build_path).
#   4. Extend with a random-walk route from the target lane.

LANE_WIDTH       = 3.7
MAX_LATERAL_DIST = LANE_WIDTH * 1.5   # ~5.55 m — one adjacent lane
MIN_LATERAL_DIST = 1.0                # avoid same-lane false positives
MAX_HEADING_DIFF = np.pi / 6.0        # 30° angular freedom

# Road types that block a lateral crossing (mirrors find_parallel_lane_candidate)
SOLID_BLOCKING_TYPES = frozenset({
    12,   # ROAD_LINE_SOLID_SINGLE_WHITE
    13,   # ROAD_LINE_SOLID_DOUBLE_WHITE
    16,   # ROAD_LINE_SOLID_SINGLE_YELLOW
    17,   # ROAD_LINE_SOLID_DOUBLE_YELLOW
    18,   # ROAD_LINE_PASSING_DOUBLE_YELLOW
    20,   # ROAD_EDGE_UNKNOWN
    21,   # ROAD_EDGE_BOUNDARY
    22,   # ROAD_EDGE_MEDIAN
})


def _compute_heading_diff(h1: float, h2: float) -> float:
    """Normalize heading difference to (−π, π]. Mirrors compute_heading_diff."""
    d = h1 - h2
    while d >  np.pi: d -= 2 * np.pi
    while d < -np.pi: d += 2 * np.pi
    return d


def _find_closest_segment_on_lane(lane: dict, px: float, py: float):
    """
    Port of find_closest_segment_on_lane().
    Returns (seg_idx, signed_dist).
    signed_dist < 0 → point is LEFT of lane direction.
    """
    xyz    = lane['xyz']
    n      = len(xyz) - 1
    if n < 1:
        return 0, 1e9
    best_dsq = 1e18; best_i = 0; best_cross = 0.0
    for i in range(n):
        x0, y0 = float(xyz[i,   0]), float(xyz[i,   1])
        x1, y1 = float(xyz[i+1, 0]), float(xyz[i+1, 1])
        dx, dy  = x1 - x0, y1 - y0
        sl2     = dx*dx + dy*dy
        tx, ty  = px - x0, py - y0
        cross   = dx*ty - dy*tx
        if sl2 > 1e-6:
            t = (tx*dx + ty*dy) / sl2
            if   t <= 0.0: dsq = tx*tx + ty*ty
            elif t >= 1.0: dsq = (px-x1)**2 + (py-y1)**2
            else:          dsq = cross*cross / sl2
        else:
            dsq = tx*tx + ty*ty
        if dsq < best_dsq:
            best_dsq = dsq; best_i = i; best_cross = cross
    abs_d = np.sqrt(best_dsq)
    return best_i, (-abs_d if best_cross >= 0 else abs_d)


def _compute_multi_segment_alignment(lane: dict, center_seg: int) -> float:
    """
    Weighted average heading ±1 segment around center.
    Mirrors compute_multi_segment_alignment() — center segment has weight 2.
    """
    n     = max(0, len(lane['xyz']) - 2)
    start = max(0, center_seg - 1)
    end   = min(n, center_seg + 1)
    avg_h = 0.0; total_w = 0.0
    headings = lane['heading']
    for i in range(start, end + 1):
        if i >= len(headings): break
        h = float(headings[i])
        w = 2.0 if i == center_seg else 1.0
        if total_w == 0.0:
            avg_h = h
        else:
            avg_h += w * _compute_heading_diff(h, avg_h) / (total_w + w)
        total_w += w
    return avg_h


def _check_line_intersection(p1, p2, q1, q2) -> bool:
    """Segment-segment intersection test. Port of check_line_intersection()."""
    if (max(p1[0], p2[0]) < min(q1[0], q2[0]) or
        min(p1[0], p2[0]) > max(q1[0], q2[0]) or
        max(p1[1], p2[1]) < min(q1[1], q2[1]) or
        min(p1[1], p2[1]) > max(q1[1], q2[1])):
        return False
    dx1, dy1 = p2[0]-p1[0], p2[1]-p1[1]
    dx2, dy2 = q2[0]-q1[0], q2[1]-q1[1]
    cross = dx1*dy2 - dy1*dx2
    if cross == 0: return False
    dx3, dy3 = p1[0]-q1[0], p1[1]-q1[1]
    s = (dx1*dy3 - dy1*dx3) / cross
    t = (dx2*dy3 - dy2*dx3) / cross
    return 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0


def _find_parallel_lane_candidate(road_map: dict,
                                   jump_x: float, jump_y: float,
                                   jump_heading: float,
                                   current_lane_id: int,
                                   search_radius: float = 15.0):
    """
    Port of find_parallel_lane_candidate() — brute-force (no spatial grid).
    Returns the lane_id of the best drivable parallel neighbour, or None.

    Criteria (mirrors C code):
      * Lateral distance in [MIN_LATERAL_DIST, MAX_LATERAL_DIST]  (~1–5.55 m)
      * Heading alignment < 30°
      * No solid road line / road edge crosses the lateral gap segment
    Score = lateral_dist + heading_diff  (lower = better).
    """
    best_score = 1e9
    best_id    = None

    for lid, lane in road_map.items():
        if lid == current_lane_id: continue
        if lane['type'] not in DRIVABLE_TYPES: continue

        # Bounding-box pre-filter
        xs = lane['xyz'][:, 0]; ys = lane['xyz'][:, 1]
        if (xs.max() < jump_x - search_radius or xs.min() > jump_x + search_radius or
            ys.max() < jump_y - search_radius or ys.min() > jump_y + search_radius):
            continue

        seg_idx, signed_d = _find_closest_segment_on_lane(lane, jump_x, jump_y)
        abs_d = abs(signed_d)
        if abs_d < MIN_LATERAL_DIST or abs_d > MAX_LATERAL_DIST: continue

        cand_h = _compute_multi_segment_alignment(lane, seg_idx)
        hdiff  = abs(_compute_heading_diff(jump_heading, cand_h))
        if hdiff > MAX_HEADING_DIFF: continue

        # Project jump position onto candidate to find entry point
        n   = len(lane['xyz'])
        ns  = min(seg_idx + 1, n - 1)
        sx0, sy0 = float(lane['xyz'][seg_idx, 0]), float(lane['xyz'][seg_idx, 1])
        sx1, sy1 = float(lane['xyz'][ns, 0]),      float(lane['xyz'][ns, 1])
        sdx, sdy = sx1 - sx0, sy1 - sy0
        slen2    = sdx*sdx + sdy*sdy
        tt = (max(0.0, min(1.0, ((jump_x-sx0)*sdx + (jump_y-sy0)*sdy) / slen2))
              if slen2 > 1e-6 else 0.0)
        entry_x = sx0 + tt*sdx
        entry_y = sy0 + tt*sdy

        # Blocking check: solid lines + road edges cross the lateral gap?
        p1 = (jump_x, jump_y); p2 = (entry_x, entry_y)
        bx_lo = min(p1[0], p2[0]) - 1; bx_hi = max(p1[0], p2[0]) + 1
        by_lo = min(p1[1], p2[1]) - 1; by_hi = max(p1[1], p2[1]) + 1
        blocked = False
        for blk in road_map.values():
            if blk['type'] not in SOLID_BLOCKING_TYPES: continue
            bxs = blk['xyz'][:, 0]; bys = blk['xyz'][:, 1]
            if (bxs.max() < bx_lo or bxs.min() > bx_hi or
                bys.max() < by_lo or bys.min() > by_hi): continue
            for si in range(len(blk['xyz']) - 1):
                q1 = (float(blk['xyz'][si,   0]), float(blk['xyz'][si,   1]))
                q2 = (float(blk['xyz'][si+1, 0]), float(blk['xyz'][si+1, 1]))
                if _check_line_intersection(p1, p2, q1, q2):
                    blocked = True; break
            if blocked: break
        if blocked: continue

        score = abs_d + hdiff
        if score < best_score:
            best_score = score; best_id = lid

    return best_id


# ── build_path_tracked ────────────────────────────────────────────────────────

def build_path_tracked(road_map: dict, route: list,
                        waypoints_spacing: float = 20.0,
                        max_wp: int = 4096):
    """
    Like build_path() but also returns a per-waypoint lane-ID array.
    Returns (path_wp (N, 4) float32, lane_ids (N,) int64).
    Needed by build_path_w_jump so it knows which lane each waypoint belongs to.
    """
    wps = []; lids = []; wc = 0; px = py = pz = ps = None
    for lid in route:
        if lid not in road_map: continue
        xyz = road_map[lid]['xyz']
        for i in range(len(xyz)):
            cx, cy, cz = float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2])
            if wc == 0:
                wps.append((cx, cy, cz, 0.0)); lids.append(lid)
                px, py, pz, ps = cx, cy, cz, 0.0; wc += 1; continue
            dx, dy, dz = cx-px, cy-py, cz-pz
            sl = np.sqrt(dx*dx + dy*dy + dz*dz)
            if sl < 1e-6: px, py, pz = cx, cy, cz; continue
            cs = ps + sl; ts = wc * waypoints_spacing
            while ts < cs and wc < max_wp:
                t = (ts - ps) / sl
                wps.append((px+t*dx, py+t*dy, pz+t*dz, ts))
                lids.append(lid); wc += 1; ts = wc * waypoints_spacing
            px, py, pz, ps = cx, cy, cz, cs
    if not wps:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return np.array(wps, dtype=np.float32), np.array(lids, dtype=np.int64)


# ── build_path_w_jump ─────────────────────────────────────────────────────────

def build_path_w_jump(road_map: dict,
                       path_wp: np.ndarray, lane_ids: np.ndarray,
                       sx: float, sy: float, sh: float, rng,
                       waypoints_spacing: float = 20.0,
                       max_wp: int = 200):
    """
    Port of build_path_w_jump() from drive.h.

    Keeps waypoints 0..jump_wp-1 unchanged, then re-interpolates the tail
    from a parallel target lane.  The target is chosen 10–50 waypoints
    (200–1000 m at 20 m spacing) ahead of the spawn position, capped at
    75% of the total path length.

    Returns (new_path_wp, new_lane_ids, jump_wp_idx).
    jump_wp_idx is None (path returned unchanged) when:
      - path is too short  (<8 waypoints)
      - jump window is degenerate  (jump_lo >= jump_hi)
      - no valid parallel lane found at the jump position
    """
    JUMP_MIN_AHEAD = 10
    JUMP_MAX_AHEAD = 50

    if len(path_wp) < 8:
        return path_wp, lane_ids, None

    base_idx    = _get_base_wp_idx(path_wp, sx, sy, sh)
    jump_lo     = base_idx + JUMP_MIN_AHEAD
    jump_hi     = base_idx + JUMP_MAX_AHEAD
    max_allowed = (len(path_wp) * 3) // 4
    jump_hi     = min(jump_hi, max_allowed)
    if jump_lo >= jump_hi:
        return path_wp, lane_ids, None

    jump_wp = jump_lo + int(rng.integers(jump_hi - jump_lo))

    jump_x = float(path_wp[jump_wp, 0])
    jump_y = float(path_wp[jump_wp, 1])
    # Derive heading from path geometry at the jump waypoint
    if jump_wp > 0:
        ddx = float(path_wp[jump_wp, 0] - path_wp[jump_wp-1, 0])
        ddy = float(path_wp[jump_wp, 1] - path_wp[jump_wp-1, 1])
        jump_heading = float(np.arctan2(ddy, ddx))
    else:
        jump_heading = sh

    current_lane_id = int(lane_ids[jump_wp])
    target_id = _find_parallel_lane_candidate(
        road_map, jump_x, jump_y, jump_heading, current_lane_id)
    if target_id is None:
        return path_wp, lane_ids, None

    target_lane = road_map[target_id]
    entry_seg, _ = _find_closest_segment_on_lane(target_lane, jump_x, jump_y)

    # ── Keep prefix 0..jump_wp-1, rebuild tail from target_lane ──────────────
    new_wps  = [path_wp[i] for i in range(jump_wp)]
    new_lids = [int(lane_ids[i]) for i in range(jump_wp)]
    wp_count = jump_wp

    prev_x = float(path_wp[jump_wp-1, 0])
    prev_y = float(path_wp[jump_wp-1, 1])
    prev_z = float(path_wp[jump_wp-1, 2])
    prev_s = float(path_wp[jump_wp-1, 3])

    # Interpolate target lane from entry_seg onward (same pattern as build_path)
    target_xyz = target_lane['xyz']
    for i in range(entry_seg, len(target_xyz)):
        cx, cy, cz = float(target_xyz[i,0]), float(target_xyz[i,1]), float(target_xyz[i,2])
        dx, dy, dz  = cx-prev_x, cy-prev_y, cz-prev_z
        sl = np.sqrt(dx*dx + dy*dy + dz*dz)
        if sl < 1e-6: prev_x, prev_y, prev_z = cx, cy, cz; continue
        cs = prev_s + sl; ts = wp_count * waypoints_spacing
        while ts < cs and wp_count < max_wp:
            t = (ts - prev_s) / sl
            new_wps.append(np.array([prev_x+t*dx, prev_y+t*dy, prev_z+t*dz, ts],
                                     dtype=np.float32))
            new_lids.append(target_id); wp_count += 1; ts = wp_count * waypoints_spacing
        prev_x, prev_y, prev_z, prev_s = cx, cy, cz, cs

    # Extend with a random walk from target_lane (mirrors C extension logic)
    ext_dist  = waypoints_spacing * 3 * 20.0   # min_spacing * num_target_waypoints * 20
    ext_route = generate_random_route(road_map, target_id, ext_dist,
                                       jump_x, jump_y, rng=rng)
    for ri in range(1, len(ext_route)):
        if wp_count >= max_wp: break
        rl = road_map.get(ext_route[ri])
        if rl is None: continue
        for i in range(len(rl['xyz'])):
            cx, cy, cz = float(rl['xyz'][i,0]), float(rl['xyz'][i,1]), float(rl['xyz'][i,2])
            dx, dy, dz  = cx-prev_x, cy-prev_y, cz-prev_z
            sl = np.sqrt(dx*dx + dy*dy + dz*dz)
            if sl < 1e-6: prev_x, prev_y, prev_z = cx, cy, cz; continue
            cs = prev_s + sl; ts = wp_count * waypoints_spacing
            while ts < cs and wp_count < max_wp:
                t = (ts - prev_s) / sl
                new_wps.append(np.array([prev_x+t*dx, prev_y+t*dy, prev_z+t*dz, ts],
                                         dtype=np.float32))
                new_lids.append(ext_route[ri]); wp_count += 1; ts = wp_count * waypoints_spacing
            prev_x, prev_y, prev_z, prev_s = cx, cy, cz, cs

    if wp_count < 2:
        return path_wp, lane_ids, None

    return (np.array(new_wps, dtype=np.float32),
            np.array(new_lids, dtype=np.int64),
            jump_wp)


# ── Updated simulate_gigaflow_agent (overrides definition in previous cell) ───

def simulate_gigaflow_agent(road_map: dict, rng,
                              mode: str = 'static',
                              min_ws: float = 20.0, max_ws: float = 60.0,
                              num_goals: int = 3,
                              min_route_dist: float = _MIN_ROUTE_DIST,
                              max_attempts: int = 30):
    """
    Simulate one GIGAFLOW spawn episode.

    mode='static'  Straight route; goals at U[20, 60] m spacings (default).
    mode='jump'    Same as static, but path has a lane-change injected
                   10–50 waypoints ahead of spawn via build_path_w_jump().
                   The returned dict includes 'jump_fwd_idx' (index of the
                   lane-change waypoint in path_fwd, or None if no valid
                   parallel lane was found).
    """
    drivable = [(lid, l) for lid, l in road_map.items()
                if l['type'] in DRIVABLE_TYPES]
    for _ in range(max_attempts):
        lid, lane = drivable[int(rng.integers(len(drivable)))]
        seg = int(rng.integers(len(lane['xyz'])))
        sx  = float(lane['xyz'][seg, 0])
        sy  = float(lane['xyz'][seg, 1])
        sh  = float(lane['heading'][seg])

        route = generate_random_route(road_map, lid, min_route_dist, sx, sy, rng=rng)

        if mode == 'jump':
            path_wp, lane_ids = build_path_tracked(road_map, route)
        else:
            path_wp = build_path(road_map, route)

        if len(path_wp) < 5: continue

        base_idx    = _get_base_wp_idx(path_wp, sx, sy, sh)
        jump_wp_idx = None

        if mode == 'jump':
            path_wp, lane_ids, jump_wp_idx = build_path_w_jump(
                road_map, path_wp, lane_ids, sx, sy, sh, rng)

        path_fwd = path_wp[base_idx:]
        if len(path_fwd) < 3: continue

        goals, spacings = compute_goals_gigaflow(
            path_wp, sx, sy, sh,
            min_spacing=min_ws, max_spacing=max_ws,
            num_goals=num_goals, rng=rng)
        if not goals: continue

        result = dict(spawn_x=sx, spawn_y=sy, spawn_heading=sh,
                      spawn_lane=lid, route=route,
                      path_fwd=path_fwd, goals=goals, spacings=spacings)
        if mode == 'jump':
            result['jump_fwd_idx'] = (jump_wp_idx - base_idx
                                       if jump_wp_idx is not None else None)
        return result
    return None
