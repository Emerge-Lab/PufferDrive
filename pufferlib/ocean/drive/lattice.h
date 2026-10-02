#ifndef PUFFERLIB_OCEAN_DRIVE_LATTICE_H
#define PUFFERLIB_OCEAN_DRIVE_LATTICE_H

// Werling Frenet lattice for DYNAMICS_MODEL_SPLINE_WERLING: committed plans on a smoothed lane rail, jerk-integrated.

typedef struct {
    double value;
    double first;
    double second;
} LatticePlanPoint;

typedef struct {
    float s;
    float d;
    float heading_error;
    float curvature;
    float s_dot;
    float d_dot;
    float speed;
    int sample_idx;
} LatticeFrenet;

// ========================================
// Polynomials (double: plans are evaluated up to LATTICE_STOP_T_MAX_S)
// ========================================

static void lattice_quintic_coefs(double x0, double v0, double a0, double x1, double v1, double a1, double T, double c[6]) {
    double T2 = T * T;
    double T3 = T2 * T;
    double residual_pos = x1 - x0 - v0 * T - 0.5 * a0 * T2;
    double residual_vel = v1 - v0 - a0 * T;
    double residual_acc = a1 - a0;
    c[0] = x0;
    c[1] = v0;
    c[2] = 0.5 * a0;
    c[3] = (10.0 * residual_pos - 4.0 * residual_vel * T + 0.5 * residual_acc * T2) / T3;
    c[4] = (-15.0 * residual_pos + 7.0 * residual_vel * T - residual_acc * T2) / (T3 * T);
    c[5] = (6.0 * residual_pos - 3.0 * residual_vel * T + 0.5 * residual_acc * T2) / (T3 * T2);
}

static void lattice_quartic_speed_coefs(double x0, double v0, double a0, double v1, double T, double c[6]) {
    double residual_vel = v1 - v0 - a0 * T;
    double residual_acc = -a0;
    c[0] = x0;
    c[1] = v0;
    c[2] = 0.5 * a0;
    c[3] = residual_vel / (T * T) - residual_acc / (3.0 * T);
    c[4] = residual_acc / (4.0 * T * T) - residual_vel / (2.0 * T * T * T);
    c[5] = 0.0;
}

static void lattice_hold_coefs(double x0, double v0, double c[6]) {
    c[0] = x0;
    c[1] = v0;
    c[2] = 0.0;
    c[3] = 0.0;
    c[4] = 0.0;
    c[5] = 0.0;
}

static LatticePlanPoint lattice_poly_point(const double c[6], double u) {
    LatticePlanPoint p;
    p.value = c[0] + u * (c[1] + u * (c[2] + u * (c[3] + u * (c[4] + u * c[5]))));
    p.first = c[1] + u * (2.0 * c[2] + u * (3.0 * c[3] + u * (4.0 * c[4] + u * 5.0 * c[5])));
    p.second = 2.0 * c[2] + u * (6.0 * c[3] + u * (12.0 * c[4] + u * 20.0 * c[5]));
    return p;
}

static double lattice_poly_third(const double c[6], double u) {
    return 6.0 * c[3] + u * (24.0 * c[4] + u * 60.0 * c[5]);
}

static float lattice_wrap_angle(float angle) {
    while (angle > (float) M_PI) {
        angle -= 2.0f * (float) M_PI;
    }
    while (angle < -(float) M_PI) {
        angle += 2.0f * (float) M_PI;
    }
    return angle;
}

static int lattice_duration_steps(float duration_s, float dt, int *steps_out) {
    float ratio = duration_s / dt;
    long steps = lroundf(ratio);
    if (!(fabsf(ratio - (float) steps) <= LATTICE_DURATION_TOLERANCE * fmaxf(1.0f, ratio)) || steps < 1
        || steps > LATTICE_MAX_PLAN_STEPS) {
        return -1;
    }
    *steps_out = (int) steps;
    return 0;
}

// ========================================
// Menu layout
// ========================================

static int lattice_config_error(const char *message, float value) {
    fprintf(stderr, "[ERROR] lattice config: %s (got %g)\n", message, value);
    return -1;
}

static int lattice_values_ok(const float *values, int count, int max_count, float min_value, float max_value) {
    if (count < 1 || count > max_count) {
        return 0;
    }
    for (int value_idx = 0; value_idx < count; value_idx++) {
        if (!isfinite(values[value_idx]) || values[value_idx] < min_value || values[value_idx] > max_value) {
            return 0;
        }
    }
    return 1;
}

static int init_lattice_config(Drive *env) {
    struct LatticeConfig *cfg = &env->lattice;
    float backup_max_m = LATTICE_TRAIL_BEHIND_M - LATTICE_BACKUP_LANDING_M - LATTICE_REVERSE_EMERGENCY_MAX_M
        - 0.5f * LATTICE_SPAWN_MAX_LENGTH_M;
    if (!lattice_values_ok(cfg->lat_offsets_m, cfg->lat_offset_count, LATTICE_MAX_LAT_OFFSETS, -LANE_WIDTH, LANE_WIDTH)) {
        return lattice_config_error("lattice_lat_offsets_m count or range", (float) cfg->lat_offset_count);
    }
    int has_zero_offset = 0;
    for (int offset_idx = 0; offset_idx < cfg->lat_offset_count; offset_idx++) {
        has_zero_offset |= cfg->lat_offsets_m[offset_idx] == 0.0f;
        if (offset_idx > 0 && !(cfg->lat_offsets_m[offset_idx] > cfg->lat_offsets_m[offset_idx - 1])) {
            return lattice_config_error("lattice_lat_offsets_m must be strictly ascending", cfg->lat_offsets_m[offset_idx]);
        }
    }
    if (!has_zero_offset) {
        return lattice_config_error("lattice_lat_offsets_m must contain 0", 0.0f);
    }
    if (!lattice_values_ok(cfg->lat_durations_s, cfg->lat_duration_count, LATTICE_MAX_LAT_DURATIONS, env->dt, LATTICE_STOP_T_MAX_S)
        || !lattice_values_ok(cfg->low_speed_distances_m, cfg->lat_duration_count, LATTICE_MAX_LAT_DURATIONS, 1.0f, 100.0f)) {
        return lattice_config_error("lattice_lat_durations_s / lattice_low_speed_distances_m", (float) cfg->lat_duration_count);
    }
    if (!lattice_values_ok(cfg->lon_speeds_mps, cfg->lon_speed_count, LATTICE_MAX_LON_SPEEDS, 0.0f, env->base_max_speed_mps)
        || !lattice_values_ok(cfg->lon_durations_s, cfg->lon_duration_count, LATTICE_MAX_LON_DURATIONS, env->dt, LATTICE_STOP_T_MAX_S)) {
        return lattice_config_error("lattice_lon_speeds_mps / lattice_lon_durations_s", (float) cfg->lon_speed_count);
    }
    if (!lattice_values_ok(cfg->stop_distances_m, cfg->stop_distance_count, LATTICE_MAX_STOP_DISTANCES, 0.5f, 500.0f)) {
        return lattice_config_error("lattice_stop_distances_m", (float) cfg->stop_distance_count);
    }
    if (!lattice_values_ok(cfg->backup_distances_m, cfg->backup_distance_count, LATTICE_MAX_BACKUP_DISTANCES, LATTICE_MIN_BACKUP_DISTANCE_M, backup_max_m)) {
        return lattice_config_error("lattice_backup_distances_m must lie in (0, 13.37] m", (float) cfg->backup_distance_count);
    }
    if (!(cfg->low_speed_mps > 0.0f) || !(cfg->low_speed_mps < env->base_max_speed_mps)) {
        return lattice_config_error("lattice_low_speed_mps", cfg->low_speed_mps);
    }
    if (cfg->exit_mode != LATTICE_EXIT_MODE_POLICY && cfg->exit_mode != LATTICE_EXIT_MODE_GOAL) {
        return lattice_config_error("lattice_exit_mode", (float) cfg->exit_mode);
    }
    if (cfg->oncoming_overtake != 0 && cfg->oncoming_overtake != 1) {
        return lattice_config_error("lattice_oncoming_overtake", (float) cfg->oncoming_overtake);
    }
    if (cfg->turnaround != 0 && cfg->turnaround != 1) {
        return lattice_config_error("lattice_turnaround", (float) cfg->turnaround);
    }
    if (cfg->oncoming_overtake && !(cfg->lat_offsets_m[cfg->lat_offset_count - 1] < LATTICE_BORROW_MIN_D_M)) {
        return lattice_config_error("lattice_oncoming_overtake needs every lattice_lat_offsets_m below 1.25 m", cfg->lat_offsets_m[cfg->lat_offset_count - 1]);
    }
    if (lattice_duration_steps(cfg->decision_period_s, env->dt, &cfg->decision_period_steps) != 0) {
        return lattice_config_error("lattice_decision_period_s must be a whole number of dt", cfg->decision_period_s);
    }
    for (int duration_idx = 0; duration_idx < cfg->lat_duration_count; duration_idx++) {
        if (lattice_duration_steps(cfg->lat_durations_s[duration_idx], env->dt, &cfg->lat_duration_steps[duration_idx]) != 0) {
            return lattice_config_error("lattice_lat_durations_s must be whole numbers of dt", cfg->lat_durations_s[duration_idx]);
        }
    }
    for (int duration_idx = 0; duration_idx < cfg->lon_duration_count; duration_idx++) {
        if (lattice_duration_steps(cfg->lon_durations_s[duration_idx], env->dt, &cfg->lon_duration_steps[duration_idx]) != 0) {
            return lattice_config_error("lattice_lon_durations_s must be whole numbers of dt", cfg->lon_durations_s[duration_idx]);
        }
    }
    int stop_grid_steps = 0;
    int backup_grid_steps = 0;
    if (lattice_duration_steps(LATTICE_STOP_T_GRID_S, env->dt, &stop_grid_steps) != 0
        || lattice_duration_steps(LATTICE_BACKUP_T_GRID_S, env->dt, &backup_grid_steps) != 0) {
        return lattice_config_error("dt must divide the 1.2 s stop grid and the 0.3 s back-up grid", env->dt);
    }
    cfg->lat_cell_count = (cfg->lat_offset_count + 2 + cfg->oncoming_overtake) * cfg->lat_duration_count;
    cfg->lon_speed_cell_count = cfg->lon_speed_count * cfg->lon_duration_count;
    cfg->lon_stop_cell_base = cfg->lon_speed_cell_count;
    cfg->lon_stop_line_cell = cfg->lon_stop_cell_base + cfg->stop_distance_count;
    cfg->lon_emergency_cell = cfg->lon_stop_line_cell + 1;
    cfg->lon_backup_cell_base = cfg->lon_emergency_cell + 1;
    cfg->lon_turn_cell = cfg->turnaround ? cfg->lon_backup_cell_base + cfg->backup_distance_count : -1;
    cfg->lon_cell_count = cfg->lon_backup_cell_base + cfg->backup_distance_count + cfg->turnaround;
    cfg->nvec[LATTICE_FACTOR_LAT_GATE] = LATTICE_GATE_COUNT;
    cfg->nvec[LATTICE_FACTOR_LAT_CELL] = cfg->lat_cell_count;
    cfg->nvec[LATTICE_FACTOR_LON_GATE] = LATTICE_GATE_COUNT;
    cfg->nvec[LATTICE_FACTOR_LON_CELL] = cfg->lon_cell_count;
    cfg->nvec[LATTICE_FACTOR_EXIT] = LATTICE_EXIT_SLOTS;
    cfg->mask_feature_count = 0;
    for (int factor_idx = 0; factor_idx < LATTICE_ACTION_FACTORS; factor_idx++) {
        cfg->mask_feature_count += cfg->nvec[factor_idx];
    }
    return 0;
}

static int lattice_mask_offset(const struct LatticeConfig *cfg, int factor_idx) {
    int offset = 0;
    for (int previous_idx = 0; previous_idx < factor_idx; previous_idx++) {
        offset += cfg->nvec[previous_idx];
    }
    return offset;
}

// lateral cell = choice_count * duration_idx + choice; choice 0 = right lane, offsets, left lane, then oncoming lane
static int lattice_lat_choice_count(const struct LatticeConfig *cfg) {
    return cfg->lat_offset_count + 2 + cfg->oncoming_overtake;
}

static int lattice_lat_is_oncoming(const struct LatticeConfig *cfg, int cell) {
    return cell % lattice_lat_choice_count(cfg) == cfg->lat_offset_count + 2;
}

static int lattice_lat_lane_side(const struct LatticeConfig *cfg, int cell) {
    int choice = cell % lattice_lat_choice_count(cfg);
    if (choice == 0) {
        return -1;
    }
    return choice == cfg->lat_offset_count + 1 ? 1 : 0;
}

static float lattice_lat_cell_offset(const struct LatticeConfig *cfg, int cell) {
    int choice = cell % lattice_lat_choice_count(cfg);
    return cfg->lat_offsets_m[choice - 1];
}

static int lattice_lat_cell_duration_idx(const struct LatticeConfig *cfg, int cell) {
    return cell / lattice_lat_choice_count(cfg);
}

static int lattice_plan_feature_count(const struct LatticeConfig *cfg) {
    return LATTICE_PLAN_FEATURES + cfg->oncoming_overtake + LATTICE_TURN_PLAN_FEATURES * cfg->turnaround;
}

// ========================================
// Map-level lattice data (per env, built once after the map-cache branch)
// ========================================

static int lattice_map_error(Drive *env, const char *message, int lane_idx) {
    fprintf(stderr, "[ERROR] lattice map %s: %s (road element %d)\n", env->map_name, message, lane_idx);
    return -1;
}

static float lattice_segment_heading(const RoadMapElement *lane, int seg_idx) {
    return atan2f(lane->y[seg_idx + 1] - lane->y[seg_idx], lane->x[seg_idx + 1] - lane->x[seg_idx]);
}

static float lattice_lane_start_heading(const RoadMapElement *lane) {
    for (int seg_idx = 0; seg_idx < lane->segment_size - 1; seg_idx++) {
        float dx = lane->x[seg_idx + 1] - lane->x[seg_idx];
        float dy = lane->y[seg_idx + 1] - lane->y[seg_idx];
        if (dx * dx + dy * dy > LATTICE_MIN_SEGMENT_M * LATTICE_MIN_SEGMENT_M) {
            return atan2f(dy, dx);
        }
    }
    return 0.0f;
}

static float lattice_lane_end_heading(const RoadMapElement *lane) {
    for (int seg_idx = lane->segment_size - 2; seg_idx >= 0; seg_idx--) {
        float dx = lane->x[seg_idx + 1] - lane->x[seg_idx];
        float dy = lane->y[seg_idx + 1] - lane->y[seg_idx];
        if (dx * dx + dy * dy > LATTICE_MIN_SEGMENT_M * LATTICE_MIN_SEGMENT_M) {
            return atan2f(dy, dx);
        }
    }
    return 0.0f;
}

static int lattice_is_drivable_lane_idx(const Drive *env, int element_idx) {
    return element_idx >= 0 && element_idx < env->num_road_elements
        && is_drivable_road_lane(env->road_elements[element_idx].type);
}

// point, unit tangent and segment of a lane at arc position arc_m (clamped to the lane)
static void lattice_lane_point_at_arc(
    const Drive *env,
    int lane_idx,
    float arc_m,
    float *x_out,
    float *y_out,
    float *heading_out) {
    const RoadMapElement *lane = &env->road_elements[lane_idx];
    const struct LatticeLaneInfo *info = &env->lattice_lanes[lane_idx];
    const float *cum = &env->lattice_lane_cum_m[info->cum_offset];
    float clamped_arc = clip(arc_m, 0.0f, info->length_m);
    int seg_idx = 0;
    while (seg_idx < lane->segment_size - 2 && cum[seg_idx + 1] < clamped_arc) {
        seg_idx++;
    }
    float seg_len = cum[seg_idx + 1] - cum[seg_idx];
    float t = seg_len > LATTICE_MIN_SEGMENT_M ? (clamped_arc - cum[seg_idx]) / seg_len : 0.0f;
    *x_out = lane->x[seg_idx] + t * (lane->x[seg_idx + 1] - lane->x[seg_idx]);
    *y_out = lane->y[seg_idx] + t * (lane->y[seg_idx + 1] - lane->y[seg_idx]);
    *heading_out = seg_len > LATTICE_MIN_SEGMENT_M ? lattice_segment_heading(lane, seg_idx) : lattice_lane_start_heading(lane);
}

// heading after following the straightest successors for ahead_m from arc_m
static float lattice_heading_ahead(const Drive *env, int lane_idx, float arc_m, float ahead_m) {
    float remaining_m = arc_m + ahead_m;
    int current_lane = lane_idx;
    for (int hop_idx = 0; hop_idx < LATTICE_CHAIN_MAX_LANES; hop_idx++) {
        const struct LatticeLaneInfo *info = &env->lattice_lanes[current_lane];
        if (remaining_m <= info->length_m || info->exit_count == 0) {
            break;
        }
        remaining_m -= info->length_m;
        current_lane = info->exit_slots[0];
    }
    float x, y, heading;
    lattice_lane_point_at_arc(env, current_lane, remaining_m, &x, &y, &heading);
    return heading;
}

static int build_lattice_lane_geometry(Drive *env) {
    int total_points = 0;
    for (int element_idx = 0; element_idx < env->num_road_elements; element_idx++) {
        const RoadMapElement *element = &env->road_elements[element_idx];
        if (!is_drivable_road_lane(element->type)) {
            continue;
        }
        if (element->segment_size < 2) {
            return lattice_map_error(env, "drivable lane with fewer than 2 points", element_idx);
        }
        total_points += element->segment_size;
    }
    env->lattice_lane_cum_m = (float *) calloc(total_points > 0 ? total_points : 1, sizeof(float));
    if (env->lattice_lane_cum_m == NULL) {
        return lattice_map_error(env, "out of memory for lane lengths", -1);
    }
    int cum_offset = 0;
    for (int element_idx = 0; element_idx < env->num_road_elements; element_idx++) {
        const RoadMapElement *element = &env->road_elements[element_idx];
        struct LatticeLaneInfo *info = &env->lattice_lanes[element_idx];
        info->traffic_light_idx = -1;
        if (!is_drivable_road_lane(element->type)) {
            continue;
        }
        info->cum_offset = cum_offset;
        info->point_count = element->segment_size;
        float *cum = &env->lattice_lane_cum_m[cum_offset];
        cum[0] = 0.0f;
        for (int point_idx = 1; point_idx < element->segment_size; point_idx++) {
            float dx = element->x[point_idx] - element->x[point_idx - 1];
            float dy = element->y[point_idx] - element->y[point_idx - 1];
            if (!isfinite(dx) || !isfinite(dy)) {
                return lattice_map_error(env, "non-finite lane point", element_idx);
            }
            cum[point_idx] = cum[point_idx - 1] + sqrtf(dx * dx + dy * dy);
        }
        info->length_m = cum[element->segment_size - 1];
        if (!(info->length_m > LATTICE_MIN_SEGMENT_M)) {
            return lattice_map_error(env, "drivable lane of zero length", element_idx);
        }
        float max_curvature = 0.0f;
        for (int point_idx = 1; point_idx < element->segment_size - 1; point_idx++) {
            float len_in = cum[point_idx] - cum[point_idx - 1];
            float len_out = cum[point_idx + 1] - cum[point_idx];
            if (len_in <= LATTICE_MIN_SEGMENT_M || len_out <= LATTICE_MIN_SEGMENT_M) {
                continue;
            }
            float turn = fabsf(lattice_wrap_angle(lattice_segment_heading(element, point_idx) - lattice_segment_heading(element, point_idx - 1)));
            float tangent_m = 0.5f * fminf(len_in, len_out);
            max_curvature = fmaxf(max_curvature, tanf(0.5f * fminf(turn, LATTICE_FILLET_MAX_TURN_RAD)) / tangent_m);
        }
        info->max_curvature = max_curvature;
        cum_offset += element->segment_size;
    }
    return 0;
}

static int build_lattice_lane_links(Drive *env) {
    for (int element_idx = 0; element_idx < env->num_road_elements; element_idx++) {
        const RoadMapElement *element = &env->road_elements[element_idx];
        if (!is_road_lane(element->type)) {
            continue;
        }
        if (element->num_exits < 0 || element->num_entries < 0) {
            return lattice_map_error(env, "negative lane link count", element_idx);
        }
        for (int link_idx = 0; link_idx < element->num_exits; link_idx++) {
            int target = element->exit_lanes[link_idx];
            if (target != -1 && (target < 0 || target >= env->num_road_elements)) {
                return lattice_map_error(env, "exit link out of range", element_idx);
            }
        }
        for (int link_idx = 0; link_idx < element->num_entries; link_idx++) {
            int source = element->entry_lanes[link_idx];
            if (source != -1 && (source < 0 || source >= env->num_road_elements)) {
                return lattice_map_error(env, "entry link out of range", element_idx);
            }
        }
    }
    for (int element_idx = 0; element_idx < env->num_road_elements; element_idx++) {
        const RoadMapElement *element = &env->road_elements[element_idx];
        struct LatticeLaneInfo *info = &env->lattice_lanes[element_idx];
        if (!is_drivable_road_lane(element->type)) {
            continue;
        }
        int candidates[LATTICE_EXIT_SLOTS];
        float turns[LATTICE_EXIT_SLOTS];
        int candidate_count = 0;
        float end_heading = lattice_lane_end_heading(element);
        for (int link_idx = 0; link_idx < element->num_exits; link_idx++) {
            int target = element->exit_lanes[link_idx];
            if (!lattice_is_drivable_lane_idx(env, target)) {
                continue;
            }
            const RoadMapElement *exit_lane = &env->road_elements[target];
            float gap_x = exit_lane->x[0] - element->x[element->segment_size - 1];
            float gap_y = exit_lane->y[0] - element->y[element->segment_size - 1];
            if (gap_x * gap_x + gap_y * gap_y > LATTICE_JOIN_GAP_MAX_M * LATTICE_JOIN_GAP_MAX_M) {
                continue;
            }
            int duplicate = 0;
            for (int candidate_idx = 0; candidate_idx < candidate_count; candidate_idx++) {
                duplicate |= candidates[candidate_idx] == target;
            }
            if (duplicate) {
                continue;
            }
            if (candidate_count == LATTICE_EXIT_SLOTS) {
                return lattice_map_error(env, "split with more than LATTICE_EXIT_SLOTS drivable exits", element_idx);
            }
            candidates[candidate_count] = target;
            turns[candidate_count] = lattice_wrap_angle(lattice_lane_end_heading(exit_lane) - end_heading);
            candidate_count++;
        }
        for (int slot_idx = 0; slot_idx < LATTICE_EXIT_SLOTS; slot_idx++) {
            info->exit_slots[slot_idx] = -1;
            info->exit_turn_rad[slot_idx] = 0.0f;
        }
        info->exit_count = candidate_count;
        if (candidate_count == 0) {
            continue;
        }
        int straightest = 0;
        for (int candidate_idx = 1; candidate_idx < candidate_count; candidate_idx++) {
            float abs_turn = fabsf(turns[candidate_idx]);
            float best_abs_turn = fabsf(turns[straightest]);
            if (abs_turn < best_abs_turn || (abs_turn == best_abs_turn && candidates[candidate_idx] < candidates[straightest])) {
                straightest = candidate_idx;
            }
        }
        info->exit_slots[0] = candidates[straightest];
        info->exit_turn_rad[0] = turns[straightest];
        int filled = 1;
        int used[LATTICE_EXIT_SLOTS] = {0};
        used[straightest] = 1;
        for (int fill_idx = 1; fill_idx < candidate_count; fill_idx++) {
            int best = -1;
            for (int candidate_idx = 0; candidate_idx < candidate_count; candidate_idx++) {
                if (used[candidate_idx]) {
                    continue;
                }
                if (best == -1 || turns[candidate_idx] > turns[best]
                    || (turns[candidate_idx] == turns[best] && candidates[candidate_idx] < candidates[best])) {
                    best = candidate_idx;
                }
            }
            used[best] = 1;
            info->exit_slots[filled] = candidates[best];
            info->exit_turn_rad[filled] = turns[best];
            filled++;
        }
    }
    for (int element_idx = 0; element_idx < env->num_road_elements; element_idx++) {
        const struct LatticeLaneInfo *info = &env->lattice_lanes[element_idx];
        for (int slot_idx = 0; slot_idx < info->exit_count; slot_idx++) {
            struct LatticeLaneInfo *target_info = &env->lattice_lanes[info->exit_slots[slot_idx]];
            if (target_info->predecessor_count == LATTICE_MAX_PREDECESSORS) {
                return lattice_map_error(env, "lane with more than LATTICE_MAX_PREDECESSORS predecessors", info->exit_slots[slot_idx]);
            }
            target_info->predecessors[target_info->predecessor_count++] = element_idx;
        }
    }
    for (int element_idx = 0; element_idx < env->num_road_elements; element_idx++) {
        struct LatticeLaneInfo *info = &env->lattice_lanes[element_idx];
        if (info->predecessor_count < 2) {
            continue;
        }
        float start_heading = lattice_lane_start_heading(&env->road_elements[element_idx]);
        for (int sorted_idx = 1; sorted_idx < info->predecessor_count; sorted_idx++) {
            int lane = info->predecessors[sorted_idx];
            float turn = fabsf(lattice_wrap_angle(start_heading - lattice_lane_start_heading(&env->road_elements[lane])));
            int insert_idx = sorted_idx;
            while (insert_idx > 0) {
                int previous = info->predecessors[insert_idx - 1];
                float previous_turn = fabsf(lattice_wrap_angle(start_heading - lattice_lane_start_heading(&env->road_elements[previous])));
                if (previous_turn < turn || (previous_turn == turn && previous < lane)) {
                    break;
                }
                info->predecessors[insert_idx] = previous;
                insert_idx--;
            }
            info->predecessors[insert_idx] = lane;
        }
    }
    for (int traffic_idx = 0; traffic_idx < env->num_traffic_elements; traffic_idx++) {
        const TrafficControlElement *traffic = &env->traffic_elements[traffic_idx];
        if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            continue;
        }
        for (int lane_link_idx = 0; lane_link_idx < traffic->num_controlled_lanes; lane_link_idx++) {
            int lane = traffic->controlled_lanes[lane_link_idx];
            if (lane < 0 || lane >= env->num_road_elements) {
                return lattice_map_error(env, "traffic light controls a lane index out of range", lane);
            }
            if (env->lattice_lanes[lane].traffic_light_idx == -1) {
                env->lattice_lanes[lane].traffic_light_idx = traffic_idx;
            }
        }
    }
    for (int element_idx = 0; element_idx < env->num_road_elements; element_idx++) {
        struct LatticeLaneInfo *info = &env->lattice_lanes[element_idx];
        int connector = info->traffic_light_idx != -1;
        for (int pred_idx = 0; pred_idx < info->predecessor_count; pred_idx++) {
            connector |= env->lattice_lanes[info->predecessors[pred_idx]].exit_count >= 2;
        }
        for (int slot_idx = 0; slot_idx < info->exit_count; slot_idx++) {
            connector |= env->lattice_lanes[info->exit_slots[slot_idx]].predecessor_count >= 2;
        }
        info->is_connector = connector;
    }
    return 0;
}

// signed distance along the left normal from (px,py) to segment a->b; returns 0 when the line misses it
static int lattice_normal_line_hit(
    float px,
    float py,
    float normal_x,
    float normal_y,
    float ax,
    float ay,
    float bx,
    float by,
    float *lambda_out,
    float *mu_out) {
    float ex = bx - ax;
    float ey = by - ay;
    float denom = normal_x * ey - normal_y * ex;
    if (fabsf(denom) < LATTICE_GEOMETRY_EPS) {
        return 0;
    }
    float rx = ax - px;
    float ry = ay - py;
    float lambda = (rx * ey - ry * ex) / denom;
    float mu = (rx * normal_y - ry * normal_x) / denom;
    if (mu < 0.0f || mu > 1.0f) {
        return 0;
    }
    *lambda_out = lambda;
    *mu_out = mu;
    return 1;
}

static void compute_lattice_profile_sample(
    Drive *env,
    int lane_idx,
    float arc_m,
    GridMapEntity *entity_list,
    struct LatticeProfileSample *sample) {
    float px, py, heading;
    lattice_lane_point_at_arc(env, lane_idx, arc_m, &px, &py, &heading);
    float normal_x = -sinf(heading);
    float normal_y = cosf(heading);
    float lane_z = env->road_elements[lane_idx].z[0];
    sample->edge_left_m = LATTICE_EDGE_SEARCH_M;
    sample->edge_right_m = LATTICE_EDGE_SEARCH_M;
    float best_cos[2] = {-2.0f, -2.0f};
    for (int side = 0; side < 2; side++) {
        sample->neighbour_lane[side] = -1;
        sample->neighbour_offset_m[side] = 0.0f;
        sample->neighbour_arc_m[side] = 0.0f;
    }
    sample->oncoming_lane = -1;
    sample->oncoming_offset_m = 0.0f;
    sample->turn_lane = -1;
    sample->turn_arc_m = 0.0f;
    float turn_offset_m = 0.0f;
    int list_size = get_neighbors_entities(env, px, py, entity_list, ROAD_QUERY_ENTITY_COUNT, ROAD_OFFSETS, (int) (sizeof(ROAD_OFFSETS) / sizeof(ROAD_OFFSETS[0])));
    for (int entity_idx = 0; entity_idx < list_size; entity_idx++) {
        int element_idx = entity_list[entity_idx].entity_idx;
        int geometry_idx = entity_list[entity_idx].geometry_idx;
        if (element_idx < 0 || element_idx == lane_idx) {
            continue;
        }
        const RoadMapElement *element = &env->road_elements[element_idx];
        if (geometry_idx + 1 >= element->segment_size || fabsf(element->z[geometry_idx] - lane_z) > Z_BUFFER) {
            continue;
        }
        int is_edge = is_road_edge(element->type);
        if (!is_edge && !is_drivable_road_lane(element->type)) {
            continue;
        }
        float lambda, mu;
        if (!lattice_normal_line_hit(px, py, normal_x, normal_y, element->x[geometry_idx], element->y[geometry_idx],
                                     element->x[geometry_idx + 1], element->y[geometry_idx + 1], &lambda, &mu)) {
            continue;
        }
        int side = lambda > 0.0f ? 1 : 0;
        float distance_m = fabsf(lambda);
        if (is_edge) {
            float *edge = side ? &sample->edge_left_m : &sample->edge_right_m;
            *edge = fminf(*edge, distance_m);
            continue;
        }
        float heading_cos = cosf(lattice_segment_heading(element, geometry_idx) - heading);
        const float *cum = &env->lattice_lane_cum_m[env->lattice_lanes[element_idx].cum_offset];
        float hit_arc_m = cum[geometry_idx] + mu * (cum[geometry_idx + 1] - cum[geometry_idx]);
        if (side == 1 && heading_cos < -LATTICE_NEIGHBOUR_COS && distance_m <= LATTICE_EDGE_SEARCH_M
            && (sample->turn_lane < 0 || distance_m < turn_offset_m)) {
            sample->turn_lane = element_idx;
            sample->turn_arc_m = hit_arc_m;
            turn_offset_m = distance_m;
        }
        if (distance_m < LATTICE_NEIGHBOUR_MIN_M || distance_m > LATTICE_NEIGHBOUR_MAX_M) {
            continue;
        }
        if (side == 1 && heading_cos < -LATTICE_NEIGHBOUR_COS && (sample->oncoming_lane < 0 || distance_m < sample->oncoming_offset_m)) {
            sample->oncoming_lane = element_idx;
            sample->oncoming_offset_m = distance_m;
        }
        if (heading_cos <= LATTICE_NEIGHBOUR_COS) {
            continue;
        }
        float neighbour_arc_m = hit_arc_m;
        float ahead_cos = cosf(lattice_heading_ahead(env, element_idx, neighbour_arc_m, LATTICE_NEIGHBOUR_TIEBREAK_M)
                               - lattice_heading_ahead(env, lane_idx, arc_m, LATTICE_NEIGHBOUR_TIEBREAK_M));
        if (ahead_cos > best_cos[side]) {
            best_cos[side] = ahead_cos;
            sample->neighbour_lane[side] = element_idx;
            sample->neighbour_offset_m[side] = lambda;
            sample->neighbour_arc_m[side] = neighbour_arc_m;
        }
    }
    if (sample->neighbour_lane[0] >= 0 || sample->neighbour_lane[1] >= 0 || sample->edge_left_m < sample->oncoming_offset_m) {
        sample->oncoming_lane = -1;
        sample->oncoming_offset_m = 0.0f;
    }
    if (sample->edge_left_m < turn_offset_m) {
        sample->turn_lane = -1;
        sample->turn_arc_m = 0.0f;
    }
}

static int build_lattice_profiles(Drive *env) {
    int total_samples = 0;
    for (int element_idx = 0; element_idx < env->num_road_elements; element_idx++) {
        struct LatticeLaneInfo *info = &env->lattice_lanes[element_idx];
        if (!is_drivable_road_lane(env->road_elements[element_idx].type)) {
            continue;
        }
        int count = (int) floorf(info->length_m / LATTICE_PROFILE_SPACING_M) + 1;
        if (count > LATTICE_MAX_PROFILE_SAMPLES_PER_LANE) {
            return lattice_map_error(env, "lane too long for the profile capacity", element_idx);
        }
        info->profile_offset = total_samples;
        info->profile_count = count;
        total_samples += count;
    }
    env->lattice_profile_count = total_samples;
    env->lattice_profiles = (struct LatticeProfileSample *) calloc(total_samples > 0 ? total_samples : 1, sizeof(struct LatticeProfileSample));
    if (env->lattice_profiles == NULL) {
        return lattice_map_error(env, "out of memory for lane profiles", -1);
    }
    GridMapEntity *entity_list = (GridMapEntity *) malloc(ROAD_QUERY_ENTITY_COUNT * sizeof(GridMapEntity));
    if (entity_list == NULL) {
        return lattice_map_error(env, "out of memory for profile queries", -1);
    }
    for (int element_idx = 0; element_idx < env->num_road_elements; element_idx++) {
        const struct LatticeLaneInfo *info = &env->lattice_lanes[element_idx];
        for (int sample_idx = 0; sample_idx < info->profile_count; sample_idx++) {
            float arc_m = fminf(sample_idx * LATTICE_PROFILE_SPACING_M, info->length_m);
            compute_lattice_profile_sample(env, element_idx, arc_m, entity_list, &env->lattice_profiles[info->profile_offset + sample_idx]);
        }
    }
    free(entity_list);
    return 0;
}

static const struct LatticeProfileSample *lattice_profile_at(const Drive *env, int lane_idx, float arc_m) {
    const struct LatticeLaneInfo *info = &env->lattice_lanes[lane_idx];
    int sample_idx = (int) lroundf(arc_m / LATTICE_PROFILE_SPACING_M);
    sample_idx = sample_idx < 0 ? 0 : (sample_idx >= info->profile_count ? info->profile_count - 1 : sample_idx);
    return &env->lattice_profiles[info->profile_offset + sample_idx];
}

static int validate_lattice_map(Drive *env) {
    if (env->lane_graph.n_lanes <= 0 || env->lane_graph.lane_to_graph_idx == NULL) {
        return lattice_map_error(env, "spline_werling needs a lane graph (exit-to-goal distances)", -1);
    }
    int drivable_lanes = 0;
    for (int element_idx = 0; element_idx < env->num_road_elements; element_idx++) {
        drivable_lanes += is_drivable_road_lane(env->road_elements[element_idx].type);
    }
    if (drivable_lanes == 0) {
        return lattice_map_error(env, "no drivable lanes", -1);
    }
    return 0;
}

static void free_lattice(Drive *env) {
    free(env->lattice_lanes);
    free(env->lattice_profiles);
    free(env->lattice_agents);
    free(env->lattice_scratch_rail);
    free(env->lattice_build_scratch);
    free(env->lattice_lane_cum_m);
    env->lattice_lanes = NULL;
    env->lattice_profiles = NULL;
    env->lattice_agents = NULL;
    env->lattice_scratch_rail = NULL;
    env->lattice_build_scratch = NULL;
    env->lattice_lane_cum_m = NULL;
}

static int init_lattice_map(Drive *env) {
    if (init_lattice_config(env) != 0) {
        return -1;
    }
    env->lattice_lanes = (struct LatticeLaneInfo *) calloc(env->num_road_elements > 0 ? env->num_road_elements : 1, sizeof(struct LatticeLaneInfo));
    env->lattice_scratch_rail = (struct LatticeRail *) calloc(1, sizeof(struct LatticeRail));
    env->lattice_build_scratch = (struct LatticeBuildScratch *) calloc(1, sizeof(struct LatticeBuildScratch));
    if (env->lattice_lanes == NULL || env->lattice_scratch_rail == NULL || env->lattice_build_scratch == NULL) {
        return lattice_map_error(env, "out of memory for lattice map data", -1);
    }
    if (build_lattice_lane_geometry(env) != 0 || build_lattice_lane_links(env) != 0 || validate_lattice_map(env) != 0
        || build_lattice_profiles(env) != 0) {
        return -1;
    }
    return 0;
}

static int init_lattice_agents(Drive *env) {
    env->lattice_agents = (struct LatticeAgent *) calloc(env->active_agent_count > 0 ? env->active_agent_count : 1, sizeof(struct LatticeAgent));
    if (env->lattice_agents == NULL) {
        return lattice_map_error(env, "out of memory for lattice agents", -1);
    }
    return 0;
}

// ========================================
// Chain (ordered lanes behind and ahead of the car) and rail building
// ========================================

static float lattice_required_lookahead_m(const Drive *env) {
    float freeze_m = LATTICE_EXIT_FREEZE_M + LATTICE_LOOKAHEAD_FREEZE_PAD_M;
    float horizon_m = env->base_max_speed_mps * LATTICE_LOOKAHEAD_HORIZON_S + LATTICE_LOOKAHEAD_SPEED_PAD_M;
    return fmaxf(fmaxf(freeze_m, horizon_m), LATTICE_LOOKAHEAD_MIN_M);
}

static float lattice_chain_arc_before(const Drive *env, const struct LatticeRail *rail, int slot) {
    float arc_m = 0.0f;
    for (int lane_slot = 0; lane_slot < slot; lane_slot++) {
        arc_m += env->lattice_lanes[rail->lanes[lane_slot]].length_m;
    }
    return arc_m;
}

static float lattice_chain_total_arc(const Drive *env, const struct LatticeRail *rail) {
    return lattice_chain_arc_before(env, rail, rail->lane_count);
}

static void lattice_chain_drop_front(struct LatticeRail *rail, int drop_count) {
    for (int lane_slot = drop_count; lane_slot < rail->lane_count; lane_slot++) {
        rail->lanes[lane_slot - drop_count] = rail->lanes[lane_slot];
        rail->exit_decided[lane_slot - drop_count] = rail->exit_decided[lane_slot];
    }
    rail->lane_count -= drop_count;
}

// prepend straightest predecessors until arc_before(anchor) >= behind_m; returns lanes added
static int lattice_chain_extend_back(const Drive *env, struct LatticeRail *rail, float behind_m) {
    int added = 0;
    for (int hop_idx = 0; hop_idx < LATTICE_CHAIN_MAX_LANES && rail->lane_count < LATTICE_CHAIN_MAX_LANES; hop_idx++) {
        if (behind_m <= 0.0f) {
            break;
        }
        const struct LatticeLaneInfo *first_info = &env->lattice_lanes[rail->lanes[0]];
        if (first_info->predecessor_count == 0) {
            break;
        }
        int predecessor = first_info->predecessors[0];
        for (int lane_slot = rail->lane_count; lane_slot > 0; lane_slot--) {
            rail->lanes[lane_slot] = rail->lanes[lane_slot - 1];
            rail->exit_decided[lane_slot] = rail->exit_decided[lane_slot - 1];
        }
        rail->lanes[0] = predecessor;
        rail->exit_decided[0] = 1;
        rail->lane_count++;
        added++;
        behind_m -= env->lattice_lanes[predecessor].length_m;
    }
    return added;
}

// append exits (decided ones already in the chain, straightest otherwise) until ahead_m of chain after start_arc_m
static void lattice_chain_extend_forward(const Drive *env, struct LatticeRail *rail, float target_arc_m) {
    rail->chain_is_dead_end = 0;
    for (int hop_idx = 0; hop_idx < LATTICE_CHAIN_MAX_LANES; hop_idx++) {
        if (lattice_chain_total_arc(env, rail) >= target_arc_m) {
            return;
        }
        if (rail->lane_count == LATTICE_CHAIN_MAX_LANES) {
            return;
        }
        const struct LatticeLaneInfo *last_info = &env->lattice_lanes[rail->lanes[rail->lane_count - 1]];
        if (last_info->exit_count == 0) {
            rail->chain_is_dead_end = 1;
            return;
        }
        rail->exit_decided[rail->lane_count - 1] = last_info->exit_count == 1;
        int next_lane = last_info->exit_slots[0];
        rail->lanes[rail->lane_count] = next_lane;
        rail->exit_decided[rail->lane_count] = env->lattice_lanes[next_lane].exit_count <= 1;
        rail->lane_count++;
    }
}

static int lattice_collect_vertices(
    const Drive *env,
    const struct LatticeRail *rail,
    float window_begin_arc_m,
    float window_end_arc_m,
    struct LatticeVertex *vertices,
    float *vertex_chain_arc_m) {
    int vertex_count = 0;
    float lane_begin_arc_m = 0.0f;
    for (int lane_slot = 0; lane_slot < rail->lane_count; lane_slot++) {
        int lane_idx = rail->lanes[lane_slot];
        const RoadMapElement *lane = &env->road_elements[lane_idx];
        const struct LatticeLaneInfo *info = &env->lattice_lanes[lane_idx];
        const float *cum = &env->lattice_lane_cum_m[info->cum_offset];
        float lane_end_arc_m = lane_begin_arc_m + info->length_m;
        if (lane_end_arc_m < window_begin_arc_m || lane_begin_arc_m > window_end_arc_m) {
            lane_begin_arc_m = lane_end_arc_m;
            continue;
        }
        for (int point_idx = 0; point_idx < lane->segment_size; point_idx++) {
            float chain_arc_m = lane_begin_arc_m + cum[point_idx];
            int next_in_window = point_idx + 1 < lane->segment_size && lane_begin_arc_m + cum[point_idx + 1] >= window_begin_arc_m;
            if (chain_arc_m < window_begin_arc_m && !next_in_window && point_idx + 1 < lane->segment_size) {
                continue;
            }
            if (vertex_count > 0) {
                float dx = lane->x[point_idx] - vertices[vertex_count - 1].x;
                float dy = lane->y[point_idx] - vertices[vertex_count - 1].y;
                if (dx * dx + dy * dy < LATTICE_MIN_SEGMENT_M * LATTICE_MIN_SEGMENT_M) {
                    continue;
                }
            }
            if (vertex_count == LATTICE_MAX_RAIL_VERTICES) {
                return vertex_count;
            }
            vertices[vertex_count].x = lane->x[point_idx];
            vertices[vertex_count].y = lane->y[point_idx];
            vertices[vertex_count].lane_arc_m = cum[point_idx];
            vertices[vertex_count].chain_slot = lane_slot;
            vertex_chain_arc_m[vertex_count] = chain_arc_m;
            vertex_count++;
            if (chain_arc_m > window_end_arc_m) {
                return vertex_count;
            }
        }
        lane_begin_arc_m = lane_end_arc_m;
    }
    return vertex_count;
}

typedef struct {
    float x;
    float y;
    float heading;
    float remaining_m;
    int count;
} LatticeSampler;

static void lattice_emit_raw_sample(struct LatticeBuildScratch *scratch, LatticeSampler *sampler, float x, float y, int slot, float lane_arc_m) {
    if (sampler->count >= LATTICE_MAX_RAW_SAMPLES) {
        return;
    }
    scratch->raw_x[sampler->count] = x;
    scratch->raw_y[sampler->count] = y;
    scratch->raw_slot[sampler->count] = (unsigned char) slot;
    scratch->raw_lane_arc_m[sampler->count] = lane_arc_m;
    sampler->count++;
}

// straight piece from (x0,y0) of length_m along heading; samples every LATTICE_RAIL_SPACING_M of path length
static void lattice_sample_straight(
    struct LatticeBuildScratch *scratch,
    LatticeSampler *sampler,
    float x0,
    float y0,
    float heading,
    float length_m,
    int slot,
    float lane_arc0_m,
    float lane_arc_rate) {
    float cos_h = cosf(heading);
    float sin_h = sinf(heading);
    float travelled_m = sampler->remaining_m;
    while (travelled_m <= length_m && sampler->count < LATTICE_MAX_RAW_SAMPLES) {
        lattice_emit_raw_sample(scratch, sampler, x0 + travelled_m * cos_h, y0 + travelled_m * sin_h, slot, lane_arc0_m + lane_arc_rate * travelled_m);
        travelled_m += LATTICE_RAIL_SPACING_M;
    }
    sampler->remaining_m = travelled_m - length_m;
}

// the second half of a fillet arc lies past its vertex, on the next segment's lane (next_slot, arc from next_arc0_m)
static void lattice_sample_arc(
    struct LatticeBuildScratch *scratch,
    LatticeSampler *sampler,
    float x0,
    float y0,
    float heading0,
    float curvature,
    float length_m,
    int slot,
    float lane_arc0_m,
    int next_slot,
    float next_arc0_m) {
    float travelled_m = sampler->remaining_m;
    while (travelled_m <= length_m && sampler->count < LATTICE_MAX_RAW_SAMPLES) {
        float turn = curvature * travelled_m;
        float x = x0 + (sinf(heading0 + turn) - sinf(heading0)) / curvature;
        float y = y0 - (cosf(heading0 + turn) - cosf(heading0)) / curvature;
        int past_vertex = travelled_m > 0.5f * length_m;
        lattice_emit_raw_sample(scratch, sampler, x, y, past_vertex ? next_slot : slot,
                                past_vertex ? next_arc0_m + travelled_m - 0.5f * length_m : lane_arc0_m + travelled_m);
        travelled_m += LATTICE_RAIL_SPACING_M;
    }
    sampler->remaining_m = travelled_m - length_m;
}

// arc fillets at interior vertices (tangent length half the shorter neighbouring segment), sampled at 0.5 m
static int lattice_sample_fillet_path(const struct LatticeVertex *vertices, int vertex_count, struct LatticeBuildScratch *scratch) {
    LatticeSampler sampler = {vertices[0].x, vertices[0].y, 0.0f, 0.0f, 0};
    float cursor_x = vertices[0].x;
    float cursor_y = vertices[0].y;
    for (int vertex_idx = 1; vertex_idx < vertex_count; vertex_idx++) {
        const struct LatticeVertex *prev = &vertices[vertex_idx - 1];
        const struct LatticeVertex *curr = &vertices[vertex_idx];
        float in_x = curr->x - prev->x;
        float in_y = curr->y - prev->y;
        float in_len = sqrtf(in_x * in_x + in_y * in_y);
        float in_heading = atan2f(in_y, in_x);
        float tangent_m = 0.0f;
        float turn = 0.0f;
        float out_heading = in_heading;
        if (vertex_idx + 1 < vertex_count) {
            const struct LatticeVertex *next = &vertices[vertex_idx + 1];
            float out_x = next->x - curr->x;
            float out_y = next->y - curr->y;
            out_heading = atan2f(out_y, out_x);
            turn = lattice_wrap_angle(out_heading - in_heading);
            if (fabsf(turn) > LATTICE_FILLET_MIN_TURN_RAD && fabsf(turn) < LATTICE_FILLET_MAX_TURN_RAD) {
                tangent_m = 0.5f * fminf(in_len, sqrtf(out_x * out_x + out_y * out_y));
            }
        }
        float cursor_dx = cursor_x - prev->x;
        float cursor_dy = cursor_y - prev->y;
        float consumed_m = sqrtf(cursor_dx * cursor_dx + cursor_dy * cursor_dy);
        // a segment entering a new chain lane starts that lane at arc 0 at the join vertex
        float prev_lane_arc_m = curr->chain_slot == prev->chain_slot ? prev->lane_arc_m : fmaxf(0.0f, curr->lane_arc_m - in_len);
        float straight_m = fmaxf(0.0f, in_len - consumed_m - tangent_m);
        lattice_sample_straight(scratch, &sampler, cursor_x, cursor_y, in_heading, straight_m, curr->chain_slot, prev_lane_arc_m + consumed_m, 1.0f);
        if (tangent_m <= 0.0f) {
            cursor_x = curr->x;
            cursor_y = curr->y;
            continue;
        }
        cursor_x += straight_m * cosf(in_heading);
        cursor_y += straight_m * sinf(in_heading);
        float radius_m = tangent_m / tanf(0.5f * fabsf(turn));
        float curvature = (turn > 0.0f ? 1.0f : -1.0f) / radius_m;
        const struct LatticeVertex *next = &vertices[vertex_idx + 1];
        float next_arc0_m = next->chain_slot == curr->chain_slot ? curr->lane_arc_m : 0.0f;
        lattice_sample_arc(scratch, &sampler, cursor_x, cursor_y, in_heading, curvature, radius_m * fabsf(turn), curr->chain_slot,
                           prev_lane_arc_m + consumed_m + straight_m, next->chain_slot, next_arc0_m);
        cursor_x = curr->x + tangent_m * cosf(out_heading);
        cursor_y = curr->y + tangent_m * sinf(out_heading);
    }
    return sampler.count;
}

static void lattice_filter_positions(struct LatticeBuildScratch *scratch, int raw_count) {
    for (int sample_idx = 0; sample_idx < raw_count; sample_idx++) {
        int half = LATTICE_SMOOTH_HALF_WINDOW;
        half = sample_idx < half ? sample_idx : half;
        half = raw_count - 1 - sample_idx < half ? raw_count - 1 - sample_idx : half;
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        for (int window_idx = sample_idx - half; window_idx <= sample_idx + half; window_idx++) {
            sum_x += scratch->raw_x[window_idx];
            sum_y += scratch->raw_y[window_idx];
        }
        scratch->filtered_x[sample_idx] = sum_x / (2 * half + 1);
        scratch->filtered_y[sample_idx] = sum_y / (2 * half + 1);
    }
}

static void lattice_finish_rail_geometry(struct LatticeRail *rail) {
    int count = rail->sample_count;
    float chord[LATTICE_RAIL_SAMPLES];
    for (int sample_idx = 0; sample_idx + 1 < count; sample_idx++) {
        chord[sample_idx] = atan2f(rail->y[sample_idx + 1] - rail->y[sample_idx], rail->x[sample_idx + 1] - rail->x[sample_idx]);
    }
    chord[count - 1] = count >= 2 ? chord[count - 2] : rail->heading[0];
    for (int sample_idx = 0; sample_idx < count; sample_idx++) {
        float before = chord[sample_idx > 0 ? sample_idx - 1 : 0];
        float after = chord[sample_idx];
        rail->heading[sample_idx] = lattice_wrap_angle(before + 0.5f * lattice_wrap_angle(after - before));
        rail->curvature[sample_idx] = lattice_wrap_angle(after - before) / LATTICE_RAIL_SPACING_M;
    }
    if (count >= 2) {
        rail->curvature[0] = rail->curvature[1];
    }
}

// resample the filtered polyline at exactly LATTICE_RAIL_SPACING_M, starting at filtered sample first_idx
static void lattice_resample_rail(const Drive *env, struct LatticeRail *rail, struct LatticeBuildScratch *scratch, int filtered_count, int first_idx) {
    int count = 0;
    rail->x[0] = scratch->filtered_x[first_idx];
    rail->y[0] = scratch->filtered_y[first_idx];
    rail->chain_slot[0] = scratch->raw_slot[first_idx];
    rail->lane_arc_m[0] = scratch->raw_lane_arc_m[first_idx];
    count = 1;
    float carry_m = 0.0f;
    for (int seg_idx = first_idx; seg_idx + 1 < filtered_count && count < LATTICE_RAIL_SAMPLES; seg_idx++) {
        float seg_x = scratch->filtered_x[seg_idx + 1] - scratch->filtered_x[seg_idx];
        float seg_y = scratch->filtered_y[seg_idx + 1] - scratch->filtered_y[seg_idx];
        float seg_len = sqrtf(seg_x * seg_x + seg_y * seg_y);
        if (seg_len < LATTICE_GEOMETRY_EPS) {
            continue;
        }
        float position_m = LATTICE_RAIL_SPACING_M - carry_m;
        while (position_m <= seg_len && count < LATTICE_RAIL_SAMPLES) {
            float t = position_m / seg_len;
            rail->x[count] = scratch->filtered_x[seg_idx] + t * seg_x;
            rail->y[count] = scratch->filtered_y[seg_idx] + t * seg_y;
            int nearest = t < 0.5f ? seg_idx : seg_idx + 1;
            rail->chain_slot[count] = scratch->raw_slot[nearest];
            rail->lane_arc_m[count] = scratch->raw_lane_arc_m[nearest];
            count++;
            position_m += LATTICE_RAIL_SPACING_M;
        }
        carry_m = seg_len - (position_m - LATTICE_RAIL_SPACING_M);
    }
    rail->sample_count = count;
    lattice_finish_rail_geometry(rail);
    for (int sample_idx = 0; sample_idx < count; sample_idx++) {
        int lane_idx = rail->lanes[rail->chain_slot[sample_idx]];
        const struct LatticeProfileSample *profile = lattice_profile_at(env, lane_idx, rail->lane_arc_m[sample_idx]);
        rail->edge_left_m[sample_idx] = profile->edge_left_m;
        rail->edge_right_m[sample_idx] = profile->edge_right_m;
    }
}

static void lattice_rail_lane_bounds(const Drive *env, struct LatticeRail *rail) {
    for (int lane_slot = 0; lane_slot < rail->lane_count; lane_slot++) {
        rail->lane_start_s_m[lane_slot] = 1e9f;
        rail->lane_end_s_m[lane_slot] = -1e9f;
    }
    for (int sample_idx = 0; sample_idx < rail->sample_count; sample_idx++) {
        int lane_slot = rail->chain_slot[sample_idx];
        float s_m = rail->s_start_m + sample_idx * LATTICE_RAIL_SPACING_M;
        rail->lane_start_s_m[lane_slot] = fminf(rail->lane_start_s_m[lane_slot], s_m);
        rail->lane_end_s_m[lane_slot] = fmaxf(rail->lane_end_s_m[lane_slot], s_m);
    }
    int first_slot = rail->sample_count > 0 ? rail->chain_slot[0] : 0;
    for (int lane_slot = first_slot + 1; lane_slot < rail->lane_count; lane_slot++) {
        if (rail->lane_start_s_m[lane_slot] > rail->lane_end_s_m[lane_slot]) {
            rail->lane_start_s_m[lane_slot] = rail->lane_end_s_m[lane_slot - 1];
            rail->lane_end_s_m[lane_slot] = rail->lane_end_s_m[lane_slot - 1];
        }
    }
    float rail_end_s_m = rail->s_start_m + (rail->sample_count - 1) * LATTICE_RAIL_SPACING_M;
    int last_slot = rail->sample_count > 0 ? rail->chain_slot[rail->sample_count - 1] : 0;
    float next_start_s_m = rail_end_s_m;
    for (int lane_slot = last_slot; lane_slot < rail->lane_count; lane_slot++) {
        const struct LatticeLaneInfo *info = &env->lattice_lanes[rail->lanes[lane_slot]];
        if (lane_slot > last_slot) {
            rail->lane_start_s_m[lane_slot] = next_start_s_m;
            rail->lane_end_s_m[lane_slot] = next_start_s_m + info->length_m;
        } else {
            rail->lane_end_s_m[lane_slot] = rail_end_s_m + fmaxf(0.0f, info->length_m - rail->lane_arc_m[rail->sample_count - 1]);
        }
        next_start_s_m = rail->lane_end_s_m[lane_slot];
    }
}

// builds rail samples covering [anchor - behind_m, anchor + ahead_m] of chain arc; returns sample count (0 on failure)
static int build_lattice_rail(Drive *env, struct LatticeRail *rail, float anchor_chain_arc_m, float behind_m, float ahead_m) {
    struct LatticeBuildScratch *scratch = env->lattice_build_scratch;
    float begin_arc_m = fmaxf(0.0f, anchor_chain_arc_m - behind_m);
    float end_arc_m = fminf(lattice_chain_total_arc(env, rail), anchor_chain_arc_m + ahead_m);
    float vertex_chain_arc_m[LATTICE_MAX_RAIL_VERTICES];
    int vertex_count = lattice_collect_vertices(env, rail, begin_arc_m - LATTICE_BUILD_PAD_M, end_arc_m + LATTICE_BUILD_PAD_M, scratch->vertices, vertex_chain_arc_m);
    rail->is_straight_fallback = 0;
    if (vertex_count < 2) {
        rail->sample_count = 0;
        return 0;
    }
    int raw_count = lattice_sample_fillet_path(scratch->vertices, vertex_count, scratch);
    if (raw_count < 2) {
        rail->sample_count = 0;
        return 0;
    }
    lattice_filter_positions(scratch, raw_count);
    int first_idx = 0;
    float first_vertex_arc_m = vertex_chain_arc_m[0];
    float target_offset_m = begin_arc_m - first_vertex_arc_m;
    first_idx = (int) floorf(target_offset_m / LATTICE_RAIL_SPACING_M);
    first_idx = first_idx < 0 ? 0 : (first_idx > raw_count - 2 ? raw_count - 2 : first_idx);
    lattice_resample_rail(env, rail, scratch, raw_count, first_idx);
    rail->chain_start_arc_m = first_vertex_arc_m + first_idx * LATTICE_RAIL_SPACING_M;
    return rail->sample_count;
}

static void compute_lattice_envelope(const Agent *agent, struct LatticeRail *rail);

// straight reference along the car's heading for no-lane mode
static void build_lattice_straight_rail(struct LatticeRail *rail, const Agent *agent) {
    rail->lane_count = 0;
    rail->is_straight_fallback = 1;
    rail->chain_is_dead_end = 0;
    rail->chain_is_complete = 1;
    rail->sample_count = LATTICE_RAIL_SAMPLES;
    rail->s_start_m = 0.0f;
    float start_x = agent->sim_x - LATTICE_TRAIL_KEEP_M * agent->cos_heading;
    float start_y = agent->sim_y - LATTICE_TRAIL_KEEP_M * agent->sin_heading;
    for (int sample_idx = 0; sample_idx < LATTICE_RAIL_SAMPLES; sample_idx++) {
        rail->x[sample_idx] = start_x + sample_idx * LATTICE_RAIL_SPACING_M * agent->cos_heading;
        rail->y[sample_idx] = start_y + sample_idx * LATTICE_RAIL_SPACING_M * agent->sin_heading;
        rail->heading[sample_idx] = agent->sim_heading;
        rail->curvature[sample_idx] = 0.0f;
        rail->edge_left_m[sample_idx] = LATTICE_EDGE_SEARCH_M;
        rail->edge_right_m[sample_idx] = LATTICE_EDGE_SEARCH_M;
        rail->lane_arc_m[sample_idx] = 0.0f;
        rail->chain_slot[sample_idx] = 0;
    }
    compute_lattice_envelope(agent, rail);
}

// ========================================
// Rail queries and projection
// ========================================

static float lattice_rail_end_s(const struct LatticeRail *rail) {
    return rail->s_start_m + (rail->sample_count - 1) * LATTICE_RAIL_SPACING_M;
}

typedef struct {
    float x;
    float y;
    float heading;
    float curvature;
    float curvature_rate;
    float v_env;
    float edge_left_m;
    float edge_right_m;
    int sample_idx;
} LatticeRailPoint;

static LatticeRailPoint lattice_rail_at(const struct LatticeRail *rail, float s_m) {
    float position = (s_m - rail->s_start_m) / LATTICE_RAIL_SPACING_M;
    int sample_idx = (int) floorf(position);
    sample_idx = sample_idx < 0 ? 0 : (sample_idx > rail->sample_count - 2 ? rail->sample_count - 2 : sample_idx);
    float t = clip(position - sample_idx, 0.0f, 1.0f);
    int next_idx = sample_idx + 1;
    LatticeRailPoint point;
    point.x = rail->x[sample_idx] + t * (rail->x[next_idx] - rail->x[sample_idx]);
    point.y = rail->y[sample_idx] + t * (rail->y[next_idx] - rail->y[sample_idx]);
    point.heading = lattice_wrap_angle(rail->heading[sample_idx] + t * lattice_wrap_angle(rail->heading[next_idx] - rail->heading[sample_idx]));
    point.curvature = rail->curvature[sample_idx] + t * (rail->curvature[next_idx] - rail->curvature[sample_idx]);
    point.curvature_rate = (rail->curvature[next_idx] - rail->curvature[sample_idx]) / LATTICE_RAIL_SPACING_M;
    point.v_env = fminf(rail->v_env[sample_idx], rail->v_env[next_idx]);
    point.edge_left_m = fminf(rail->edge_left_m[sample_idx], rail->edge_left_m[next_idx]);
    point.edge_right_m = fminf(rail->edge_right_m[sample_idx], rail->edge_right_m[next_idx]);
    point.sample_idx = t < 0.5f ? sample_idx : next_idx;
    return point;
}

typedef struct {
    float curvature;
    float curvature_rate;
    float v_env;
    float edge_left_m;
    float edge_right_m;
} LatticeRailProfile;

// the quantities the cell checks need at rail s (no position / heading interpolation)
static inline LatticeRailProfile lattice_rail_profile_at(const struct LatticeRail *rail, float s_m) {
    float position = (s_m - rail->s_start_m) * (1.0f / LATTICE_RAIL_SPACING_M);
    int sample_idx = (int) position;
    sample_idx = position < 0.0f ? 0 : (sample_idx > rail->sample_count - 2 ? rail->sample_count - 2 : sample_idx);
    float t = position - sample_idx;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    int next_idx = sample_idx + 1;
    float curvature0 = rail->curvature[sample_idx];
    float curvature1 = rail->curvature[next_idx];
    LatticeRailProfile profile;
    profile.curvature = curvature0 + t * (curvature1 - curvature0);
    profile.curvature_rate = (curvature1 - curvature0) * (1.0f / LATTICE_RAIL_SPACING_M);
    profile.v_env = rail->v_env[sample_idx] < rail->v_env[next_idx] ? rail->v_env[sample_idx] : rail->v_env[next_idx];
    profile.edge_left_m = rail->edge_left_m[sample_idx] < rail->edge_left_m[next_idx] ? rail->edge_left_m[sample_idx] : rail->edge_left_m[next_idx];
    profile.edge_right_m = rail->edge_right_m[sample_idx] < rail->edge_right_m[next_idx] ? rail->edge_right_m[sample_idx] : rail->edge_right_m[next_idx];
    return profile;
}

// foot point on the rail polyline nearest (x, y), searched around hint (whole rail when hint < 0)
static LatticeFrenet lattice_project(const struct LatticeRail *rail, float x, float y, int hint) {
    int first = 0;
    int last = rail->sample_count - 2;
    if (hint >= 0) {
        first = hint - LATTICE_PROJECTION_WINDOW_SAMPLES < 0 ? 0 : hint - LATTICE_PROJECTION_WINDOW_SAMPLES;
        last = hint + LATTICE_PROJECTION_WINDOW_SAMPLES > rail->sample_count - 2 ? rail->sample_count - 2 : hint + LATTICE_PROJECTION_WINDOW_SAMPLES;
    }
    float best_dist_sq = 1e30f;
    int best_idx = first;
    float best_t = 0.0f;
    for (int seg_idx = first; seg_idx <= last; seg_idx++) {
        float seg_x = rail->x[seg_idx + 1] - rail->x[seg_idx];
        float seg_y = rail->y[seg_idx + 1] - rail->y[seg_idx];
        float rel_x = x - rail->x[seg_idx];
        float rel_y = y - rail->y[seg_idx];
        float seg_len_sq = seg_x * seg_x + seg_y * seg_y;
        float t = seg_len_sq > 1e-9f ? clip((rel_x * seg_x + rel_y * seg_y) / seg_len_sq, 0.0f, 1.0f) : 0.0f;
        float dx = rel_x - t * seg_x;
        float dy = rel_y - t * seg_y;
        float dist_sq = dx * dx + dy * dy;
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_idx = seg_idx;
            best_t = t;
        }
    }
    float seg_x = rail->x[best_idx + 1] - rail->x[best_idx];
    float seg_y = rail->y[best_idx + 1] - rail->y[best_idx];
    float seg_len = fmaxf(sqrtf(seg_x * seg_x + seg_y * seg_y), LATTICE_GEOMETRY_EPS);
    float rel_x = x - rail->x[best_idx];
    float rel_y = y - rail->y[best_idx];
    float along_m = (rel_x * seg_x + rel_y * seg_y) / seg_len;
    LatticeFrenet frenet;
    frenet.s = rail->s_start_m + best_idx * LATTICE_RAIL_SPACING_M + along_m * LATTICE_RAIL_SPACING_M / seg_len;
    frenet.d = (seg_x * rel_y - seg_y * rel_x) / seg_len;
    LatticeRailPoint point = lattice_rail_at(rail, frenet.s);
    frenet.curvature = point.curvature;
    frenet.heading_error = 0.0f;
    frenet.s_dot = 0.0f;
    frenet.d_dot = 0.0f;
    frenet.speed = 0.0f;
    frenet.sample_idx = best_t < 0.5f ? best_idx : best_idx + 1;
    return frenet;
}

static LatticeFrenet lattice_frenet_state(const struct LatticeRail *rail, const Agent *agent, int hint) {
    LatticeFrenet frenet = lattice_project(rail, agent->sim_x, agent->sim_y, hint);
    LatticeRailPoint point = lattice_rail_at(rail, frenet.s);
    frenet.heading_error = lattice_wrap_angle(agent->sim_heading - point.heading);
    frenet.speed = agent->sim_speed_signed;
    float frenet_factor = fmaxf(1.0f - point.curvature * frenet.d, LATTICE_MIN_FRENET_FACTOR);
    frenet.s_dot = frenet.speed * cosf(frenet.heading_error) / frenet_factor;
    frenet.d_dot = frenet.speed * sinf(frenet.heading_error);
    return frenet;
}

// ========================================
// Per-agent rail management
// ========================================

// curvature, curvature-rate and steering-rate speed limits (margin m), then a backward braking pass; no speed cap
static void compute_lattice_envelope(const Agent *agent, struct LatticeRail *rail) {
    float wheelbase = agent->wheelbase;
    float c_steer = agent->reward_coefs[REWARD_COEF_STEER];
    float margin = LATTICE_ENVELOPE_MARGIN;
    int count = rail->sample_count;
    // curvature averaged over one map segment: the fillet staircase of 8 m lane segments would double the rate
    float averaged_curvature[LATTICE_RAIL_SAMPLES];
    for (int sample_idx = 0; sample_idx < count; sample_idx++) {
        int half = LATTICE_ENVELOPE_CURVATURE_HALF_WINDOW;
        half = sample_idx < half ? sample_idx : half;
        half = count - 1 - sample_idx < half ? count - 1 - sample_idx : half;
        float sum = 0.0f;
        for (int window_idx = sample_idx - half; window_idx <= sample_idx + half; window_idx++) {
            sum += rail->curvature[window_idx];
        }
        averaged_curvature[sample_idx] = sum / (2 * half + 1);
    }
    for (int sample_idx = 0; sample_idx < count; sample_idx++) {
        float curvature = fabsf(averaged_curvature[sample_idx]);
        float curvature_rate = sample_idx + 1 < count
            ? fabsf(averaged_curvature[sample_idx + 1] - averaged_curvature[sample_idx]) / LATTICE_RAIL_SPACING_M : 0.0f;
        float limit = 1e9f;
        if (curvature > LATTICE_GEOMETRY_EPS) {
            limit = fminf(limit, sqrtf(ACCEL_LAT_LIMIT[1] * margin / curvature));
        }
        if (curvature_rate > LATTICE_GEOMETRY_EPS) {
            limit = fminf(limit, cbrtf(ACCEL_LAT_LIMIT[1] * margin * c_steer / curvature_rate));
        }
        float steer_rate_per_mps = curvature_rate * wheelbase / (1.0f + curvature * curvature * wheelbase * wheelbase);
        if (steer_rate_per_mps > LATTICE_GEOMETRY_EPS) {
            limit = fminf(limit, LATTICE_STEER_RATE_RPS * margin / steer_rate_per_mps);
        }
        rail->v_env[sample_idx] = limit;
    }
    if (rail->chain_is_dead_end && rail->chain_is_complete && count > 0) {
        rail->v_env[count - 1] = 0.0f;
    }
    for (int sample_idx = count - 2; sample_idx >= 0; sample_idx--) {
        float next = rail->v_env[sample_idx + 1];
        rail->v_env[sample_idx] = fminf(rail->v_env[sample_idx], sqrtf(next * next + 2.0f * LATTICE_ENVELOPE_BRAKE_MPS2 * LATTICE_RAIL_SPACING_M));
    }
}

static float lattice_car_chain_arc(const Drive *env, const struct LatticeRail *rail, int sample_idx) {
    return lattice_chain_arc_before(env, rail, rail->chain_slot[sample_idx]) + rail->lane_arc_m[sample_idx];
}

// fills samples over [car - behind_m, car + ahead_m] and sets s_start so the car keeps s_keep_m (fresh rails pass s_keep_m < 0)
static int lattice_sample_rail_span(Drive *env, struct LatticeRail *rail, const Agent *agent, float car_chain_arc_m, float s_keep_m, float behind_m, float ahead_m) {
    float drop_before_m = car_chain_arc_m - behind_m - LATTICE_BUILD_PAD_M;
    while (rail->lane_count > 1 && lattice_chain_arc_before(env, rail, 1) < drop_before_m) {
        float dropped_m = env->lattice_lanes[rail->lanes[0]].length_m;
        lattice_chain_drop_front(rail, 1);
        car_chain_arc_m -= dropped_m;
        drop_before_m -= dropped_m;
    }
    if (car_chain_arc_m < behind_m + LATTICE_BUILD_PAD_M) {
        float before_m = lattice_chain_total_arc(env, rail);
        lattice_chain_extend_back(env, rail, behind_m + LATTICE_BUILD_PAD_M - car_chain_arc_m);
        car_chain_arc_m += lattice_chain_total_arc(env, rail) - before_m;
    }
    float target_arc_m = car_chain_arc_m + fminf(ahead_m, lattice_required_lookahead_m(env) + LATTICE_LOOKAHEAD_HYSTERESIS_M) + LATTICE_BUILD_PAD_M;
    lattice_chain_extend_forward(env, rail, target_arc_m);
    float chain_total_m = lattice_chain_total_arc(env, rail);
    if (build_lattice_rail(env, rail, car_chain_arc_m, behind_m, ahead_m) < 2) {
        return -1;
    }
    float rail_span_m = (rail->sample_count - 1) * LATTICE_RAIL_SPACING_M;
    rail->chain_is_complete = rail->chain_is_dead_end && rail->chain_start_arc_m + rail_span_m >= chain_total_m - LATTICE_RAIL_SPACING_M;
    rail->s_start_m = 0.0f;
    int expected_idx = (int) lroundf((car_chain_arc_m - rail->chain_start_arc_m) / LATTICE_RAIL_SPACING_M);
    expected_idx = expected_idx < 0 ? 0 : (expected_idx > rail->sample_count - 1 ? rail->sample_count - 1 : expected_idx);
    LatticeFrenet frenet = lattice_project(rail, agent->sim_x, agent->sim_y, expected_idx);
    if (s_keep_m >= 0.0f) {
        rail->s_start_m = s_keep_m - frenet.s;
    }
    lattice_rail_lane_bounds(env, rail);
    compute_lattice_envelope(agent, rail);
    return frenet.sample_idx;
}

static int lattice_sample_rail_around(Drive *env, struct LatticeRail *rail, const Agent *agent, float car_chain_arc_m, float s_keep_m) {
    float ahead_m = (LATTICE_RAIL_SAMPLES - 1) * LATTICE_RAIL_SPACING_M - LATTICE_TRAIL_KEEP_M;
    return lattice_sample_rail_span(env, rail, agent, car_chain_arc_m, s_keep_m, LATTICE_TRAIL_KEEP_M, ahead_m);
}

// nearest drivable lane to (x, y) within LATTICE_BASE_LANE_RANGE_M whose heading has cos >= LATTICE_BASE_LANE_COS
static int lattice_find_base_lane_at(
    Drive *env,
    float x,
    float y,
    float z,
    float cos_h,
    float sin_h,
    int *lane_out,
    float *arc_out) {
    GridMapEntity entity_list[ROAD_QUERY_ENTITY_COUNT];
    int list_size = get_neighbors_entities(
        env,
        x,
        y,
        entity_list,
        ROAD_QUERY_ENTITY_COUNT,
        ROAD_OFFSETS,
        (int) (sizeof(ROAD_OFFSETS) / sizeof(ROAD_OFFSETS[0])));
    float best_dist_sq = LATTICE_BASE_LANE_RANGE_M * LATTICE_BASE_LANE_RANGE_M;
    int best_lane = -1;
    float best_arc_m = 0.0f;
    for (int entity_idx = 0; entity_idx < list_size; entity_idx++) {
        int element_idx = entity_list[entity_idx].entity_idx;
        int geometry_idx = entity_list[entity_idx].geometry_idx;
        if (!lattice_is_drivable_lane_idx(env, element_idx)) {
            continue;
        }
        const RoadMapElement *lane = &env->road_elements[element_idx];
        if (geometry_idx + 1 >= lane->segment_size || fabsf(lane->z[geometry_idx] - z) > Z_BUFFER) {
            continue;
        }
        float seg_x = lane->x[geometry_idx + 1] - lane->x[geometry_idx];
        float seg_y = lane->y[geometry_idx + 1] - lane->y[geometry_idx];
        float seg_len_sq = seg_x * seg_x + seg_y * seg_y;
        if (seg_len_sq < LATTICE_MIN_SEGMENT_M * LATTICE_MIN_SEGMENT_M) {
            continue;
        }
        float seg_len = sqrtf(seg_len_sq);
        if ((seg_x * cos_h + seg_y * sin_h) / seg_len <= LATTICE_BASE_LANE_COS) {
            continue;
        }
        float rel_x = x - lane->x[geometry_idx];
        float rel_y = y - lane->y[geometry_idx];
        float t = clip((rel_x * seg_x + rel_y * seg_y) / seg_len_sq, 0.0f, 1.0f);
        float dx = rel_x - t * seg_x;
        float dy = rel_y - t * seg_y;
        float dist_sq = dx * dx + dy * dy;
        if (dist_sq < best_dist_sq || (dist_sq == best_dist_sq && element_idx < best_lane)) {
            const float *cum = &env->lattice_lane_cum_m[env->lattice_lanes[element_idx].cum_offset];
            best_dist_sq = dist_sq;
            best_lane = element_idx;
            best_arc_m = cum[geometry_idx] + t * seg_len;
        }
    }
    *lane_out = best_lane;
    *arc_out = best_arc_m;
    return best_lane >= 0;
}

static int lattice_find_base_lane(Drive *env, const Agent *agent, int *lane_out, float *arc_out) {
    return lattice_find_base_lane_at(
        env,
        agent->sim_x,
        agent->sim_y,
        agent->sim_z,
        agent->cos_heading,
        agent->sin_heading,
        lane_out,
        arc_out);
}

// ========================================
// Turning around: rest-to-rest constant-curvature legs inside the road edges (geometry only)
// ========================================

static bool check_segment_intersects_aabb(float p0[2], float p1[2], float half_l, float half_w);

typedef struct {
    float x;
    float y;
    float heading;
} LatticePose;

// the integrator's exact motion for distance_m (signed) at constant curvature
static LatticePose lattice_arc_pose(LatticePose start, float curvature, float distance_m) {
    float theta = distance_m * curvature;
    float dx_local = distance_m;
    float dy_local = 0.0f;
    if (fabsf(curvature) >= 1e-5f && fabsf(theta) >= 1e-5f) {
        dx_local = sinf(theta) / curvature;
        dy_local = (1.0f - cosf(theta)) / curvature;
    }
    float cos_h = cosf(start.heading);
    float sin_h = sinf(start.heading);
    LatticePose pose
        = {start.x + dx_local * cos_h - dy_local * sin_h,
           start.y + dx_local * sin_h + dy_local * cos_h,
           lattice_wrap_angle(start.heading + theta)};
    return pose;
}

// also within the curvature one step of lateral jerk sets at rest, where the integrator's curvature is a_lat / 1
static float lattice_turn_curvature(const Drive *env, const Agent *agent) {
    float lock_curvature = tanf(STEERING_ANGLE_LIMIT) / agent->wheelbase;
    float rest_curvature = agent->reward_coefs[REWARD_COEF_STEER] * JERK_LAT[2] * env->dt;
    return LATTICE_TURN_CURVATURE_FRACTION * fminf(lock_curvature, rest_curvature);
}

// farthest the inflated box gets from the start pose over any turn-around on arcs of this curvature
static float lattice_turn_reach_m(const Agent *agent, float curvature_abs) {
    float radius_m = 1.0f / curvature_abs;
    float half_length_m = 0.5f * agent->sim_length + LATTICE_TURN_MARGIN_M;
    float half_width_m = 0.5f * agent->sim_width + LATTICE_TURN_MARGIN_M;
    return 2.0f * radius_m
        + sqrtf((radius_m + half_width_m) * (radius_m + half_width_m) + half_length_m * half_length_m);
}

// road-edge segments that the inflated box can reach while turning around on the widest of radius_count arcs
static int lattice_collect_turn_edges(
    Drive *env,
    const Agent *agent,
    int radius_count,
    struct LatticeTurnEdges *edges) {
    float widest_curvature = LATTICE_TURN_CURVATURE_SCALES[radius_count - 1] * lattice_turn_curvature(env, agent);
    float reach_m = lattice_turn_reach_m(agent, widest_curvature);
    int center_idx = get_grid_index(env, agent->sim_x, agent->sim_y);
    edges->count = 0;
    if (center_idx < 0) {
        return 0;
    }
    const struct GridMap *grid = env->grid_map;
    int cell_reach = (int) ceilf(reach_m / GRID_CELL_SIZE) + 1; // segments are binned by one point and run up to 10 m
    int center_col = center_idx % grid->grid_cols;
    int center_row = center_idx / grid->grid_cols;
    for (int row = center_row - cell_reach; row <= center_row + cell_reach; row++) {
        for (int col = center_col - cell_reach; col <= center_col + cell_reach; col++) {
            if (row < 0 || row >= grid->grid_rows || col < 0 || col >= grid->grid_cols) {
                continue;
            }
            int cell_idx = row * grid->grid_cols + col;
            for (int entity_idx = 0; entity_idx < grid->cell_entities_count[cell_idx]; entity_idx++) {
                const GridMapEntity *entity = &grid->cells[cell_idx][entity_idx];
                if (entity->entity_idx < 0) {
                    continue;
                }
                const RoadMapElement *element = &env->road_elements[entity->entity_idx];
                int geometry_idx = entity->geometry_idx;
                if (!is_road_edge(element->type) || geometry_idx + 1 >= element->segment_size
                    || fabsf(element->z[geometry_idx] - agent->sim_z) > Z_BUFFER + LATTICE_TURN_Z_SLACK_M) {
                    continue;
                }
                if (edges->count == LATTICE_TURN_MAX_EDGES) {
                    return 0;
                }
                edges->ax[edges->count] = element->x[geometry_idx];
                edges->ay[edges->count] = element->y[geometry_idx];
                edges->bx[edges->count] = element->x[geometry_idx + 1];
                edges->by[edges->count] = element->y[geometry_idx + 1];
                edges->count++;
            }
        }
    }
    return 1;
}

static int lattice_turn_pose_clear(
    const struct LatticeTurnEdges *edges,
    LatticePose pose,
    float half_length_m,
    float half_width_m) {
    float cos_h = cosf(pose.heading);
    float sin_h = sinf(pose.heading);
    for (int edge_idx = 0; edge_idx < edges->count; edge_idx++) {
        float a[2], b[2];
        project_point_to_local(edges->ax[edge_idx], edges->ay[edge_idx], pose.x, pose.y, cos_h, sin_h, &a[0], &a[1]);
        project_point_to_local(edges->bx[edge_idx], edges->by[edge_idx], pose.x, pose.y, cos_h, sin_h, &b[0], &b[1]);
        if (check_segment_intersects_aabb(a, b, half_length_m, half_width_m)) {
            return 0;
        }
    }
    return 1;
}

// the band of half_width_m around the segment (ax, ay)-(bx, by) touches no collected edge
static int lattice_turn_band_clear(
    const struct LatticeTurnEdges *edges,
    float ax,
    float ay,
    float bx,
    float by,
    float half_width_m) {
    LatticePose middle = {0.5f * (ax + bx), 0.5f * (ay + by), atan2f(by - ay, bx - ax)};
    return lattice_turn_pose_clear(edges, middle, 0.5f * hypotf(bx - ax, by - ay), half_width_m);
}

// rotation still to go in the turn's sense, in [-LATTICE_TURN_OVERSHOOT_RAD, 2 pi - LATTICE_TURN_OVERSHOOT_RAD)
static float lattice_turn_rotation_to_go(int sense, float target_heading, float heading) {
    float rotation_rad = sense * lattice_wrap_angle(target_heading - heading);
    return rotation_rad < -LATTICE_TURN_OVERSHOOT_RAD ? rotation_rad + 2.0f * (float) M_PI : rotation_rad;
}

// legs alternating gear, each rotating toward target_heading (sense +1 counter-clockwise) until the box nears an edge
static int lattice_simulate_turn(
    const struct LatticeTurnEdges *edges,
    LatticePose start,
    float target_heading,
    float curvature_abs,
    int sense,
    int first_gear,
    float half_length_m,
    float half_width_m,
    struct LatticeTurnPlan *plan) {
    float remaining_rad = lattice_turn_rotation_to_go(sense, target_heading, start.heading);
    if (remaining_rad <= LATTICE_TURN_HEADING_TOL_RAD) {
        return 0;
    }
    LatticePose pose = start;
    plan->leg_count = 0;
    plan->driven_m = 0.0f;
    for (int leg_idx = 0; leg_idx < LATTICE_TURN_MAX_LEGS; leg_idx++) {
        int gear = leg_idx % 2 == 0 ? first_gear : -first_gear;
        float curvature = sense * gear * curvature_abs;
        float needed_m = remaining_rad / curvature_abs;
        float travelled_m = 0.0f;
        int step_count = (int) ceilf(needed_m / LATTICE_TURN_STEP_M);
        for (int step_idx = 1; step_idx <= step_count; step_idx++) {
            float next_m = fminf(step_idx * LATTICE_TURN_STEP_M, needed_m);
            if (!lattice_turn_pose_clear(
                    edges,
                    lattice_arc_pose(pose, curvature, gear * next_m),
                    half_length_m,
                    half_width_m)) {
                break;
            }
            travelled_m = next_m;
        }
        if (travelled_m < LATTICE_TURN_MIN_LEG_M && travelled_m < needed_m) {
            return 0;
        }
        struct LatticeTurnLeg *leg = &plan->legs[plan->leg_count++];
        leg->gear = gear;
        leg->curvature = curvature;
        leg->length_m = travelled_m;
        plan->driven_m += travelled_m;
        pose = lattice_arc_pose(pose, curvature, gear * travelled_m);
        remaining_rad -= travelled_m * curvature_abs;
        if (remaining_rad <= LATTICE_TURN_HEADING_TOL_RAD) {
            plan->sense = sense;
            plan->end_x = pose.x;
            plan->end_y = pose.y;
            plan->end_heading = pose.heading;
            return 1;
        }
    }
    return 0;
}

// re-aims at the landing lane's heading on curves; 1 with the last pass landing within limits (any when committed)
static int aim_lattice_turn(
    Drive *env,
    const Agent *agent,
    const struct LatticeTurnEdges *edges,
    LatticePose start,
    int start_lane,
    float target_heading,
    float curvature_abs,
    int sense,
    int first_gear,
    float half_length_m,
    float half_width_m,
    int committed,
    struct LatticeTurnPlan *candidate) {
    float aim_heading = target_heading;
    int landed = 0;
    for (int pass_idx = 0; pass_idx < LATTICE_TURN_AIM_PASSES; pass_idx++) {
        struct LatticeTurnPlan trial;
        int landing_lane;
        float landing_arc_m;
        if (!lattice_simulate_turn(
                edges,
                start,
                aim_heading,
                curvature_abs,
                sense,
                first_gear,
                half_length_m,
                half_width_m,
                &trial)
            || !lattice_find_base_lane_at(
                env,
                trial.end_x,
                trial.end_y,
                agent->sim_z,
                cosf(trial.end_heading),
                sinf(trial.end_heading),
                &landing_lane,
                &landing_arc_m)
            || landing_lane == start_lane || env->lattice_lanes[landing_lane].is_connector) {
            return landed;
        }
        float lane_x, lane_y, lane_heading;
        lattice_lane_point_at_arc(env, landing_lane, landing_arc_m, &lane_x, &lane_y, &lane_heading);
        trial.target_heading = aim_heading;
        trial.landing_lane = landing_lane;
        trial.landing_arc_m = landing_arc_m;
        trial.landing_offset_m = hypotf(trial.end_x - lane_x, trial.end_y - lane_y);
        float heading_error_rad = fabsf(lattice_wrap_angle(lane_heading - trial.end_heading));
        float exit_x, exit_y, exit_heading;
        float exit_arc_m = fminf(landing_arc_m + LATTICE_TURN_EXIT_CHECK_M, env->lattice_lanes[landing_lane].length_m);
        lattice_lane_point_at_arc(env, landing_lane, exit_arc_m, &exit_x, &exit_y, &exit_heading);
        int merge_clear
            = lattice_turn_band_clear(edges, trial.end_x, trial.end_y, lane_x, lane_y, LATTICE_TURN_CORRIDOR_HALF_M)
            && lattice_turn_band_clear(edges, trial.end_x, trial.end_y, exit_x, exit_y, 0.5f * agent->sim_width);
        if (committed
            || (trial.landing_offset_m <= LATTICE_TURN_MAX_LANDING_M
                && heading_error_rad <= LATTICE_TURN_LAND_HEADING_TOL_RAD && merge_clear)) {
            *candidate = trial;
            landed = 1;
        }
        if (heading_error_rad <= LATTICE_TURN_ALIGN_TOL_RAD) {
            return landed;
        }
        aim_heading = lane_heading;
    }
    return landed;
}

// plan ranking: legs, an off-centre landing counting extra unless committed; ties to fewer legs, centred, shorter
static int lattice_turn_plan_better(
    const struct LatticeTurnPlan *candidate,
    const struct LatticeTurnPlan *best,
    int committed) {
    int off_centre_legs = committed ? 0 : LATTICE_TURN_OFF_CENTRE_LEGS;
    int centred = candidate->landing_offset_m <= LATTICE_TURN_LANDED_D_M;
    int best_centred = best->landing_offset_m <= LATTICE_TURN_LANDED_D_M;
    int cost = candidate->leg_count + off_centre_legs * !centred;
    int best_cost = best->leg_count + off_centre_legs * !best_centred;
    if (cost != best_cost) {
        return cost < best_cost;
    }
    if (candidate->leg_count != best->leg_count) {
        return candidate->leg_count < best->leg_count;
    }
    if (centred != best_centred) {
        return centred;
    }
    return candidate->driven_m < best->driven_m;
}

// the best-ranked turn-around onto another non-connector lane over the search's senses and arc radii
static int plan_lattice_turnaround(
    Drive *env,
    const Agent *agent,
    int start_lane,
    float target_heading,
    const struct LatticeTurnSearch *search,
    struct LatticeTurnPlan *plan) {
    struct LatticeTurnEdges *edges = &env->lattice_build_scratch->turn_edges;
    if (!lattice_collect_turn_edges(env, agent, search->radius_count, edges)) {
        return 0;
    }
    float half_length_m = 0.5f * agent->sim_length + search->sweep_margin_m;
    float half_width_m = 0.5f * agent->sim_width + search->sweep_margin_m;
    LatticePose start = {agent->sim_x, agent->sim_y, agent->sim_heading};
    if (!lattice_turn_pose_clear(
            edges,
            start,
            0.5f * agent->sim_length + search->start_margin_m,
            0.5f * agent->sim_width + search->start_margin_m)) {
        return 0;
    }
    float tight_curvature = lattice_turn_curvature(env, agent);
    int found = 0;
    for (int attempt_idx = 0; attempt_idx < 4 * search->radius_count; attempt_idx++) {
        int radius_idx = attempt_idx / 4;
        int sense = attempt_idx % 4 < 2 ? 1 : -1;
        int first_gear = attempt_idx % 2 == 0 ? 1 : -1;
        // a wider single arc only drives farther: nothing beats a centred one-leg plan of a tighter radius
        if (found && plan->leg_count == 1 && plan->landing_offset_m <= LATTICE_TURN_LANDED_D_M
            && radius_idx > plan->radius_idx) {
            break;
        }
        if (search->sense_only != 0 && sense != search->sense_only) {
            continue;
        }
        float curvature_abs = LATTICE_TURN_CURVATURE_SCALES[radius_idx] * tight_curvature;
        struct LatticeTurnPlan candidate;
        if (!aim_lattice_turn(
                env,
                agent,
                edges,
                start,
                start_lane,
                target_heading,
                curvature_abs,
                sense,
                first_gear,
                half_length_m,
                half_width_m,
                search->committed,
                &candidate)) {
            continue;
        }
        candidate.radius_idx = radius_idx;
        if (!found || lattice_turn_plan_better(&candidate, plan, search->committed)) {
            *plan = candidate;
            found = 1;
        }
    }
    return found;
}

// short rail on a neighbour lane covering one lane-change cell horizon, for the mask checks only
static int build_lattice_check_rail(Drive *env, struct LatticeRail *rail, const Agent *agent, int lane_idx, float arc_m) {
    rail->lane_count = 1;
    rail->lanes[0] = lane_idx;
    rail->exit_decided[0] = env->lattice_lanes[lane_idx].exit_count <= 1;
    rail->chain_is_dead_end = 0;
    rail->chain_is_complete = 0;
    float ahead_m = fabsf(agent->sim_speed) * LATTICE_KEEP_CHECK_S + LATTICE_CHECK_RAIL_PAD_M;
    ahead_m = fminf(ahead_m, (LATTICE_RAIL_SAMPLES - 1) * LATTICE_RAIL_SPACING_M - LATTICE_CHECK_RAIL_BEHIND_M);
    return lattice_sample_rail_span(env, rail, agent, arc_m, -1.0f, LATTICE_CHECK_RAIL_BEHIND_M, ahead_m);
}

// new rail on lane_idx at lane arc arc_m (spawn, rail change, lost); returns the car's sample or -1
static int build_lattice_fresh_rail(Drive *env, struct LatticeRail *rail, const Agent *agent, int lane_idx, float arc_m) {
    rail->lane_count = 1;
    rail->lanes[0] = lane_idx;
    rail->exit_decided[0] = env->lattice_lanes[lane_idx].exit_count <= 1;
    rail->chain_is_dead_end = 0;
    rail->chain_is_complete = 0;
    int sample_idx = lattice_sample_rail_around(env, rail, agent, arc_m, -1.0f);
    return sample_idx;
}

// regenerates the agent's rail from its chain, keeping the car's s (extension, trim, exit decision, back-up trail)
static void regenerate_lattice_rail_at(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, float car_chain_arc_m, float car_s_m) {
    int sample_idx = lattice_sample_rail_around(env, &lattice_agent->rail, agent, car_chain_arc_m, car_s_m);
    lattice_agent->projection_hint = sample_idx >= 0 ? sample_idx : 0;
    lattice_agent->counters.rail_regens += 1.0f;
}


static void lattice_neighbours_at(const Drive *env, struct LatticeAgent *lattice_agent) {
    const struct LatticeRail *rail = &lattice_agent->rail;
    for (int side = 0; side < 2; side++) {
        lattice_agent->neighbour_lane[side] = -1;
        lattice_agent->neighbour_offset_m[side] = 0.0f;
        lattice_agent->neighbour_arc_m[side] = 0.0f;
    }
    if (!lattice_agent->has_reference || rail->is_straight_fallback) {
        return;
    }
    int sample_idx = lattice_agent->projection_hint;
    int lane_idx = rail->lanes[rail->chain_slot[sample_idx]];
    const struct LatticeProfileSample *profile = lattice_profile_at(env, lane_idx, rail->lane_arc_m[sample_idx]);
    for (int side = 0; side < 2; side++) {
        lattice_agent->neighbour_lane[side] = profile->neighbour_lane[side];
        lattice_agent->neighbour_offset_m[side] = profile->neighbour_offset_m[side];
        lattice_agent->neighbour_arc_m[side] = profile->neighbour_arc_m[side];
    }
}

static int lattice_car_on_connector(const Drive *env, const struct LatticeAgent *lattice_agent) {
    const struct LatticeRail *rail = &lattice_agent->rail;
    if (!lattice_agent->has_reference || rail->is_straight_fallback) {
        return 1;
    }
    return env->lattice_lanes[rail->lanes[rail->chain_slot[lattice_agent->projection_hint]]].is_connector;
}

// offset (left, > 0) of the oncoming lane beside rail sample sample_idx; 0 when there is none or the feature is off
static float lattice_oncoming_offset_at(const Drive *env, const struct LatticeAgent *lattice_agent, int sample_idx) {
    const struct LatticeRail *rail = &lattice_agent->rail;
    if (!env->lattice.oncoming_overtake || !lattice_agent->has_reference || rail->is_straight_fallback) {
        return 0.0f;
    }
    const struct LatticeProfileSample *profile = lattice_profile_at(env, rail->lanes[rail->chain_slot[sample_idx]], rail->lane_arc_m[sample_idx]);
    return profile->oncoming_lane >= 0 ? profile->oncoming_offset_m : 0.0f;
}

// the car is in the oncoming lane: past LATTICE_BORROW_FRACTION of its offset, still facing along the rail
static int lattice_is_borrowing(const Drive *env, const struct LatticeAgent *lattice_agent, const LatticeFrenet *frenet) {
    float offset_m = lattice_oncoming_offset_at(env, lattice_agent, frenet->sample_idx);
    return offset_m > 0.0f && frenet->d > LATTICE_BORROW_FRACTION * offset_m && cosf(frenet->heading_error) > LATTICE_LOST_COS;
}

// the committed lateral plan ends in the oncoming lane beside the car
static int lattice_plan_in_oncoming(const Drive *env, const struct LatticeAgent *lattice_agent, const LatticeFrenet *frenet) {
    float offset_m = lattice_oncoming_offset_at(env, lattice_agent, frenet->sample_idx);
    return offset_m > 0.0f && lattice_agent->lat.target_d_m > LATTICE_BORROW_FRACTION * offset_m;
}

// from the car to window_m ahead the rail stays off connectors and keeps an oncoming lane beside it
static int lattice_oncoming_clear(const Drive *env, const struct LatticeAgent *lattice_agent, const LatticeFrenet *frenet, float window_m) {
    const struct LatticeRail *rail = &lattice_agent->rail;
    float end_s_m = frenet->s + window_m;
    if (end_s_m > lattice_rail_end_s(rail)) {
        return 0;
    }
    for (int lane_slot = rail->chain_slot[frenet->sample_idx]; lane_slot < rail->lane_count && rail->lane_start_s_m[lane_slot] <= end_s_m; lane_slot++) {
        if (env->lattice_lanes[rail->lanes[lane_slot]].is_connector) {
            return 0;
        }
    }
    for (float ahead_m = 0.0f; ahead_m <= window_m; ahead_m += LATTICE_PROFILE_SPACING_M) {
        if (!(lattice_oncoming_offset_at(env, lattice_agent, lattice_rail_at(rail, frenet->s + ahead_m).sample_idx) > 0.0f)) {
            return 0;
        }
    }
    return 1;
}

// lane-graph distance from the start of lane_idx to the start of goal_lane_idx; INFINITY when unreachable or unknown
static float lattice_goal_distance_m(const Drive *env, int lane_idx, int goal_lane_idx) {
    if (lane_idx < 0 || goal_lane_idx < 0 || goal_lane_idx >= env->num_road_elements || lane_idx >= env->num_road_elements) {
        return INFINITY;
    }
    int from_idx = env->lane_graph.lane_to_graph_idx[lane_idx];
    int to_idx = env->lane_graph.lane_to_graph_idx[goal_lane_idx];
    if (from_idx < 0 || to_idx < 0) {
        return INFINITY;
    }
    float distance_m = env->lane_graph.distances[from_idx * env->lane_graph.n_lanes + to_idx];
    return (!isfinite(distance_m) || distance_m < 0.0f) ? INFINITY : distance_m;
}

// driving distance along lanes from the car (projected at frenet) to goal_arc_m on goal_lane; INFINITY if unreachable
static float lattice_route_distance_m(const Drive *env, const struct LatticeAgent *lattice_agent, LatticeFrenet frenet, int goal_lane, float goal_arc_m) {
    const struct LatticeRail *rail = &lattice_agent->rail;
    if (!lattice_agent->has_reference || rail->is_straight_fallback) {
        return INFINITY;
    }
    int car_slot = rail->chain_slot[frenet.sample_idx];
    int car_lane = rail->lanes[car_slot];
    float car_arc_m = frenet.s - rail->lane_start_s_m[car_slot];
    if (car_lane == goal_lane && goal_arc_m >= car_arc_m) {
        return goal_arc_m - car_arc_m;
    }
    const struct LatticeLaneInfo *info = &env->lattice_lanes[car_lane];
    float best_exit_m = INFINITY;
    for (int slot_idx = 0; slot_idx < info->exit_count; slot_idx++) {
        best_exit_m = fminf(best_exit_m, lattice_goal_distance_m(env, info->exit_slots[slot_idx], goal_lane));
    }
    return rail->lane_end_s_m[car_slot] - frenet.s + best_exit_m + goal_arc_m;
}

static int lattice_agent_goal_lane(const Agent *agent) {
    if (agent->current_goal_idx < 0 || agent->current_goal_idx >= agent->goal_count) {
        return -1;
    }
    return agent->list_goal_lane[agent->current_goal_idx];
}

// nearest undecided split ahead of the car on the chain; -1 if none within max_distance_m
static int lattice_next_split(const Drive *env, const struct LatticeAgent *lattice_agent, float car_s_m, float max_distance_m) {
    const struct LatticeRail *rail = &lattice_agent->rail;
    if (!lattice_agent->has_reference || rail->is_straight_fallback) {
        return -1;
    }
    for (int lane_slot = rail->chain_slot[lattice_agent->projection_hint]; lane_slot < rail->lane_count; lane_slot++) {
        if (rail->lane_end_s_m[lane_slot] - car_s_m > max_distance_m) {
            return -1;
        }
        if (rail->lane_end_s_m[lane_slot] < car_s_m) {
            continue;
        }
        if (env->lattice_lanes[rail->lanes[lane_slot]].exit_count >= 2 && !rail->exit_decided[lane_slot]) {
            return lane_slot;
        }
    }
    return -1;
}

static int lattice_goal_exit_slot(const Drive *env, const Agent *agent, int split_lane_idx) {
    const struct LatticeLaneInfo *info = &env->lattice_lanes[split_lane_idx];
    int goal_lane = lattice_agent_goal_lane(agent);
    int best_slot = 0;
    float best_distance_m = 1e30f;
    for (int slot_idx = 0; slot_idx < info->exit_count; slot_idx++) {
        float distance_m = lattice_goal_distance_m(env, info->exit_slots[slot_idx], goal_lane);
        if (distance_m < best_distance_m) {
            best_distance_m = distance_m;
            best_slot = slot_idx;
        }
    }
    return best_slot;
}

// commits exit slot exit_slot at chain split lane_slot; rebuilds the chain beyond the split when it changes
static void apply_lattice_exit(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, int lane_slot, int exit_slot) {
    struct LatticeRail *rail = &lattice_agent->rail;
    const struct LatticeLaneInfo *info = &env->lattice_lanes[rail->lanes[lane_slot]];
    int exit_lane = info->exit_slots[exit_slot];
    rail->exit_decided[lane_slot] = 1;
    lattice_agent->counters.exit_decisions += 1.0f;
    lattice_agent->counters.exit_nonstraight += exit_slot != 0;
    if (lane_slot + 1 < rail->lane_count && rail->lanes[lane_slot + 1] == exit_lane) {
        return;
    }
    LatticeFrenet frenet = lattice_project(rail, agent->sim_x, agent->sim_y, lattice_agent->projection_hint);
    float car_chain_arc_m = lattice_car_chain_arc(env, rail, frenet.sample_idx);
    if (lane_slot + 1 >= LATTICE_CHAIN_MAX_LANES) {
        car_chain_arc_m -= env->lattice_lanes[rail->lanes[0]].length_m;
        lattice_chain_drop_front(rail, 1);
        lane_slot--;
    }
    rail->lane_count = lane_slot + 1;
    rail->lanes[rail->lane_count] = exit_lane;
    rail->exit_decided[rail->lane_count] = env->lattice_lanes[exit_lane].exit_count <= 1;
    rail->lane_count++;
    rail->chain_is_complete = 0;
    regenerate_lattice_rail_at(env, lattice_agent, agent, car_chain_arc_m, frenet.s);
}

// ========================================
// Plans: evaluation and construction
// ========================================

static double lattice_elapsed_s(const Drive *env, int start_step, int now_step) {
    return (double) (now_step - start_step) * (double) env->dt;
}

static LatticePlanPoint lattice_lon_eval(const struct LatticeLonPlan *plan, double u_s) {
    if (u_s <= plan->horizon_s) {
        return lattice_poly_point(plan->coefs, u_s < 0.0 ? 0.0 : u_s);
    }
    LatticePlanPoint end = lattice_poly_point(plan->coefs, plan->horizon_s);
    if (plan->kind == LATTICE_LON_KIND_SPEED) {
        end.value += end.first * (u_s - plan->horizon_s);
        end.second = 0.0;
        return end;
    }
    end.first = 0.0;
    end.second = 0.0;
    return end;
}

static LatticePlanPoint lattice_lat_eval_time(const struct LatticeLatPlan *plan, double u_s) {
    if (u_s >= plan->horizon) {
        LatticePlanPoint end = {plan->target_d_m, 0.0, 0.0};
        return end;
    }
    return lattice_poly_point(plan->coefs, u_s < 0.0 ? 0.0 : u_s);
}

// derivatives with respect to rail s (d' = dir * p'(u), d'' = p''(u))
static LatticePlanPoint lattice_lat_eval_dist(const struct LatticeLatPlan *plan, float s_m) {
    double u_m = plan->dir * (double) (s_m - plan->start_s_m);
    if (u_m >= plan->horizon) {
        LatticePlanPoint end = {plan->target_d_m, 0.0, 0.0};
        return end;
    }
    LatticePlanPoint point = lattice_poly_point(plan->coefs, u_m < 0.0 ? 0.0 : u_m);
    point.first *= plan->dir;
    return point;
}

// lateral plan value and time derivatives at (step, rail s, rail rate, rail acceleration)
static LatticePlanPoint lattice_lat_state(const Drive *env, const struct LatticeLatPlan *plan, int step, float s_m, float s_dot, float s_ddot) {
    if (plan->mode == LATTICE_LAT_MODE_TIME) {
        return lattice_lat_eval_time(plan, lattice_elapsed_s(env, plan->start_step, step));
    }
    LatticePlanPoint spatial = lattice_lat_eval_dist(plan, s_m);
    LatticePlanPoint timed = {spatial.value, spatial.first * s_dot, spatial.second * s_dot * s_dot + spatial.first * s_ddot};
    return timed;
}

static int lattice_lat_plan_ended(const Drive *env, const struct LatticeLatPlan *plan, int step, float s_m) {
    if (plan->mode == LATTICE_LAT_MODE_TIME) {
        return step >= plan->end_step;
    }
    return plan->dir * (s_m - plan->start_s_m) >= plan->horizon;
}

static void set_lattice_lat_time_plan(struct LatticeLatPlan *plan, int kind, float d0, float d_dot0, float d_ddot0, float target_d_m, int steps, float dt, int now_step) {
    plan->mode = LATTICE_LAT_MODE_TIME;
    plan->kind = kind;
    plan->horizon = steps * dt;
    plan->start_step = now_step;
    plan->end_step = now_step + steps;
    plan->start_s_m = 0.0f;
    plan->dir = 1;
    plan->target_d_m = target_d_m;
    plan->reindexed_low_speed = 0;
    plan->reindexed_stop = 0;
    lattice_quintic_coefs(d0, d_dot0, d_ddot0, target_d_m, 0.0, 0.0, plan->horizon, plan->coefs);
}

static void set_lattice_lat_dist_plan(struct LatticeLatPlan *plan, int kind, float d0, float d_prime0, float d_second0, float target_d_m, float distance_m, float start_s_m, int dir) {
    plan->mode = LATTICE_LAT_MODE_DIST;
    plan->kind = kind;
    plan->horizon = distance_m;
    plan->start_step = 0;
    plan->end_step = 0;
    plan->start_s_m = start_s_m;
    plan->dir = dir;
    plan->target_d_m = target_d_m;
    plan->reindexed_low_speed = 0;
    plan->reindexed_stop = 0;
    lattice_quintic_coefs(d0, dir * d_prime0, d_second0, target_d_m, 0.0, 0.0, distance_m, plan->coefs);
}

static void set_lattice_lat_hold(struct LatticeLatPlan *plan, float d_m, float start_s_m, int dir) {
    plan->mode = LATTICE_LAT_MODE_DIST;
    plan->kind = LATTICE_LAT_KIND_HOLD;
    plan->horizon = 0.0f;
    plan->start_step = 0;
    plan->end_step = 0;
    plan->start_s_m = start_s_m;
    plan->dir = dir;
    plan->target_d_m = d_m;
    plan->reindexed_low_speed = 1;
    plan->reindexed_stop = 1;
    lattice_hold_coefs(d_m, 0.0, plan->coefs);
}

static void set_lattice_lon_speed_plan(struct LatticeLonPlan *plan, double sigma0, float v0, float a0, float speed_mps, int steps, float dt, int now_step, int cell) {
    plan->kind = LATTICE_LON_KIND_SPEED;
    plan->horizon_s = steps * dt;
    plan->start_step = now_step;
    plan->end_step = now_step + steps;
    plan->cell = cell;
    plan->target_speed_mps = speed_mps;
    plan->target_sigma_m = 0.0;
    plan->release_latched = 0;
    plan->two_step_stage = 0;
    lattice_quartic_speed_coefs(sigma0, v0, a0, speed_mps, plan->horizon_s, plan->coefs);
}

static void set_lattice_lon_hold(struct LatticeLonPlan *plan, double sigma0, float v0, int now_step) {
    plan->kind = LATTICE_LON_KIND_SPEED;
    plan->horizon_s = 0.0f;
    plan->start_step = now_step;
    plan->end_step = now_step;
    plan->cell = -1;
    plan->target_speed_mps = v0;
    plan->target_sigma_m = sigma0;
    plan->release_latched = 0;
    plan->two_step_stage = 0;
    lattice_hold_coefs(sigma0, v0, plan->coefs);
}

// stop (distance > 0) or back-up (distance < 0) quintic in sigma, ending at rest
static void set_lattice_lon_stop_plan(struct LatticeLonPlan *plan, int kind, double sigma0, float v0, float a0, float distance_m, int steps, float dt, int now_step, int cell) {
    plan->kind = kind;
    plan->horizon_s = steps * dt;
    plan->start_step = now_step;
    plan->end_step = now_step + steps;
    plan->cell = cell;
    plan->target_speed_mps = 0.0f;
    plan->target_sigma_m = sigma0 + distance_m;
    plan->release_latched = 0;
    plan->two_step_stage = 0;
    lattice_quintic_coefs(sigma0, v0, a0, sigma0 + distance_m, 0.0, 0.0, plan->horizon_s, plan->coefs);
}

static void set_lattice_lon_emergency(struct LatticeLonPlan *plan, int now_step, int cell) {
    plan->kind = LATTICE_LON_KIND_EMERGENCY;
    plan->horizon_s = 0.0f;
    plan->start_step = now_step;
    plan->end_step = now_step;
    plan->cell = cell;
    plan->target_speed_mps = 0.0f;
    plan->target_sigma_m = 0.0;
    plan->release_latched = 0;
    plan->two_step_stage = 0;
    lattice_hold_coefs(0.0, 0.0, plan->coefs);
}

static int lattice_lon_is_stopping(const struct LatticeLonPlan *plan) {
    return plan->kind == LATTICE_LON_KIND_STOP || plan->kind == LATTICE_LON_KIND_STOP_LINE
        || plan->kind == LATTICE_LON_KIND_EMERGENCY;
}

static int lattice_lon_has_final_approach(const struct LatticeLonPlan *plan) {
    return plan->kind == LATTICE_LON_KIND_STOP || plan->kind == LATTICE_LON_KIND_STOP_LINE
        || plan->kind == LATTICE_LON_KIND_BACKUP || (plan->kind == LATTICE_LON_KIND_SPEED && plan->target_speed_mps == 0.0f && plan->horizon_s > 0.0f);
}

// ========================================
// Integrator twin (drive.h jerk branch, longitudinal part) and the EMERGENCY / release rules
// ========================================

typedef struct {
    float speed;
    float accel;
} LatticeLongState;

static LatticeLongState lattice_integrate_long(LatticeLongState state, float jerk, const Agent *agent, float dt, float speed_cap_mps) {
    float c_throttle = agent->reward_coefs[REWARD_COEF_THROTTLE];
    float c_acc = agent->reward_coefs[REWARD_COEF_ACC];
    float clipped_jerk = clip(jerk, JERK_LONG[0], JERK_LONG[3]);
    float accel_new = state.accel + c_throttle * clipped_jerk * dt;
    if (state.accel * accel_new < 0) {
        accel_new = 0.0f;
    } else {
        accel_new = clip(accel_new, ACCEL_LONG_LIMIT[0], ACCEL_LONG_LIMIT[1] * c_acc);
    }
    float speed_new = state.speed + 0.5f * (accel_new + state.accel) * dt;
    if (state.speed * speed_new < 0) {
        speed_new = 0.0f;
    } else {
        speed_new = clip(speed_new, MAX_BACKWARD_SPEED, speed_cap_mps);
    }
    LatticeLongState next = {speed_new, accel_new};
    return next;
}

static float lattice_speed_cap_mps(const Drive *env, const Agent *agent) {
    return env->base_max_speed_mps * agent->reward_coefs[REWARD_COEF_SPEED];
}

// gear +1: most negative jerk whose "then +4 until a >= 0" never reverses; gear -1 mirrored (never moves forward)
static float lattice_emergency_jerk(const Drive *env, const Agent *agent, LatticeLongState start, int gear, float *stop_distance_m) {
    float dt = env->dt;
    float speed_cap = lattice_speed_cap_mps(env, agent);
    int candidate_count = (int) lroundf((JERK_LONG[3] - JERK_LONG[0]) / LATTICE_EMERGENCY_JERK_STEP);
    for (int candidate_idx = 0; candidate_idx <= candidate_count; candidate_idx++) {
        float jerk = gear > 0 ? JERK_LONG[0] + LATTICE_EMERGENCY_JERK_STEP * candidate_idx
                              : JERK_LONG[3] - LATTICE_EMERGENCY_JERK_STEP * candidate_idx;
        float release_jerk = gear > 0 ? JERK_LONG[3] : JERK_LONG[0];
        LatticeLongState state = lattice_integrate_long(start, jerk, agent, dt, speed_cap);
        float distance_m = 0.5f * (state.speed + start.speed) * dt;
        int ok = gear * state.speed >= -LATTICE_REVERSAL_TOLERANCE_MPS;
        for (int release_idx = 0; ok && release_idx < LATTICE_EMERGENCY_MAX_RELEASE_STEPS && gear * state.accel < 0.0f; release_idx++) {
            LatticeLongState next = lattice_integrate_long(state, release_jerk, agent, dt, speed_cap);
            distance_m += 0.5f * (next.speed + state.speed) * dt;
            state = next;
            ok = gear * state.speed >= -LATTICE_REVERSAL_TOLERANCE_MPS;
        }
        if (ok) {
            LatticeLongState next = lattice_integrate_long(state, 0.0f, agent, dt, speed_cap);
            ok = gear * next.speed >= -LATTICE_REVERSAL_TOLERANCE_MPS;
        }
        if (ok) {
            if (stop_distance_m != NULL) {
                *stop_distance_m = fabsf(distance_m);
            }
            return jerk;
        }
    }
    if (stop_distance_m != NULL) {
        *stop_distance_m = 0.0f;
    }
    return gear > 0 ? JERK_LONG[3] : JERK_LONG[0];
}

// distance to rest under the EMERGENCY rule applied every step
static float lattice_emergency_stop_distance(const Drive *env, const Agent *agent, LatticeLongState start, int gear) {
    float distance_m = 0.0f;
    LatticeLongState state = start;
    float speed_cap = lattice_speed_cap_mps(env, agent);
    for (int step_idx = 0; step_idx < LATTICE_EMERGENCY_MAX_RELEASE_STEPS; step_idx++) {
        if (fabsf(state.speed) < LATTICE_STOPPED_SPEED_MPS && fabsf(state.accel) < LATTICE_STOPPED_ACCEL_MPS2) {
            break;
        }
        float jerk = lattice_emergency_jerk(env, agent, state, gear, NULL);
        LatticeLongState next = lattice_integrate_long(state, jerk, agent, env->dt, speed_cap);
        distance_m += 0.5f * (next.speed + state.speed) * env->dt;
        state = next;
    }
    return fabsf(distance_m);
}

// the exact two-step stop leaves float residuals of ~1e-8; anything below these tolerances is at rest
static int lattice_is_stopped_exactly(const Agent *agent) {
    return fabsf(agent->sim_speed_signed) < LATTICE_STOPPED_SPEED_MPS && fabsf(agent->accel_long) < LATTICE_STOPPED_ACCEL_MPS2;
}

// ========================================
// Tracking command: feedforward + clamped PD, mapped through the Frenet inverse into implied jerks
// ========================================

typedef struct {
    float jerk_long;
    float jerk_lat;
    float lat_error_m;
    float speed_error_mps;
} LatticeCommand;

static float lattice_longitudinal_jerk(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, int now_step, float *speed_error_out) {
    struct LatticeLonPlan *plan = &lattice_agent->lon;
    float dt = env->dt;
    float c_throttle = agent->reward_coefs[REWARD_COEF_THROTTLE];
    float speed = agent->sim_speed_signed;
    LatticeLongState now = {speed, agent->accel_long};
    *speed_error_out = 0.0f;
    if (plan->kind == LATTICE_LON_KIND_EMERGENCY) {
        return lattice_emergency_jerk(env, agent, now, lattice_agent->gear, NULL);
    }
    double u_s = lattice_elapsed_s(env, plan->start_step, now_step);
    LatticePlanPoint point = lattice_lon_eval(plan, u_s);
    LatticePlanPoint next = lattice_lon_eval(plan, u_s + dt);
    float sigma_error = clip((float) (point.value - lattice_agent->sigma_m), -LATTICE_E_SIGMA_MAX_M, LATTICE_E_SIGMA_MAX_M);
    float accel_cmd = (float) next.second + LATTICE_K_SPEED * ((float) point.first - speed) + LATTICE_K_SIGMA * sigma_error;
    float jerk_track = (accel_cmd - agent->accel_long) / (c_throttle * dt);
    *speed_error_out = (float) point.first - speed;
    if (!lattice_lon_has_final_approach(plan) || !(u_s > 0.5 * plan->horizon_s + 1e-9)) {
        return jerk_track;
    }
    int gear = plan->kind == LATTICE_LON_KIND_BACKUP ? -1 : 1;
    float jerk_release = lattice_emergency_jerk(env, agent, now, gear, NULL);
    if (!plan->release_latched && fabsf(speed) < LATTICE_RELEASE_SPEED_MPS) {
        plan->release_latched = 1;
    }
    if (!plan->release_latched) {
        return gear > 0 ? fmaxf(jerk_track, jerk_release) : fminf(jerk_track, jerk_release);
    }
    if (plan->two_step_stage == 1) {
        plan->two_step_stage = 2;
        return gear > 0 ? JERK_LONG[3] : JERK_LONG[0];
    }
    if (plan->two_step_stage == 0) {
        float travel_speed = gear * speed;
        float travel_accel = gear * agent->accel_long;
        if (travel_speed > LATTICE_TWO_STEP_EPS && travel_accel <= LATTICE_TWO_STEP_EPS) {
            float remaining = travel_speed + 0.5f * travel_accel * dt;
            float first_accel = -remaining / dt;
            float first_jerk = gear * (first_accel - travel_accel) / (c_throttle * dt);
            if (remaining >= 0.0f && -first_accel <= JERK_LONG[3] * c_throttle * dt && first_jerk >= JERK_LONG[0]
                && first_jerk <= JERK_LONG[3]) {
                plan->two_step_stage = 1;
                return first_jerk;
            }
        }
    }
    return jerk_release;
}

// |v| < low speed or reversing: path curvature of the planned offset with spatial PD, via the integrator's next speed
static float lattice_low_speed_lateral_accel(
    Drive *env,
    const struct LatticeAgent *lattice_agent,
    const Agent *agent,
    const LatticeFrenet *frenet,
    float jerk_long,
    int now_step) {
    const struct LatticeRail *rail = &lattice_agent->rail;
    const struct LatticeLatPlan *plan = &lattice_agent->lat;
    float dt = env->dt;
    float s_ahead = frenet->s + frenet->s_dot * dt;
    LatticeRailPoint ahead = lattice_rail_at(rail, s_ahead);
    int dist_mode = plan->mode == LATTICE_LAT_MODE_DIST;
    LatticePlanPoint target = dist_mode ? lattice_lat_eval_dist(plan, s_ahead)
                                        : lattice_lat_eval_time(plan, lattice_elapsed_s(env, plan->start_step, now_step + 1));
    float d_prime = dist_mode ? (float) target.first : 0.0f;
    float d_second = dist_mode ? (float) target.second : 0.0f;
    float frenet_factor = fmaxf(1.0f - ahead.curvature * (float) target.value, LATTICE_MIN_FRENET_FACTOR);
    float plan_heading = atan2f(d_prime, frenet_factor);
    float cos_plan = cosf(plan_heading);
    float path_curvature = ((d_second + (ahead.curvature_rate * (float) target.value + ahead.curvature * d_prime) * tanf(plan_heading)) * cos_plan * cos_plan / frenet_factor + ahead.curvature) * cos_plan / frenet_factor;
    LatticePlanPoint here = dist_mode ? lattice_lat_eval_dist(plan, frenet->s)
                                      : lattice_lat_eval_time(plan, lattice_elapsed_s(env, plan->start_step, now_step));
    LatticeRailPoint at_car = lattice_rail_at(rail, frenet->s);
    float heading_plan_now = atan2f(dist_mode ? (float) here.first : 0.0f, fmaxf(1.0f - at_car.curvature * (float) here.value, LATTICE_MIN_FRENET_FACTOR));
    float feedback_gain = fminf(1.0f, fabsf(frenet->speed) / LATTICE_FEEDBACK_FULL_SPEED_MPS);
    float curvature_cmd = path_curvature
        + feedback_gain * (LATTICE_K_D_SPATIAL * ((float) here.value - frenet->d) + lattice_agent->gear * LATTICE_K_HEADING_SPATIAL * (heading_plan_now - frenet->heading_error));
    LatticeLongState now_long = {agent->sim_speed_signed, agent->accel_long};
    LatticeLongState next_long = lattice_integrate_long(now_long, jerk_long, agent, dt, lattice_speed_cap_mps(env, agent));
    float effective_speed = fmaxf(fabsf(next_long.speed), 1.0f);
    return curvature_cmd * effective_speed * effective_speed;
}

// forward above the low speed: d'' command mapped at the predicted point with rail sdot from the predicted real speed
static float lattice_lateral_accel(
    Drive *env,
    const struct LatticeAgent *lattice_agent,
    const Agent *agent,
    const LatticeFrenet *frenet,
    float jerk_long,
    int now_step,
    float *lat_error_out) {
    const struct LatticeRail *rail = &lattice_agent->rail;
    const struct LatticeLatPlan *plan = &lattice_agent->lat;
    float dt = env->dt;
    double u_s = lattice_elapsed_s(env, lattice_agent->lon.start_step, now_step);
    float plan_accel = lattice_agent->lon.kind == LATTICE_LON_KIND_EMERGENCY ? agent->accel_long : (float) lattice_lon_eval(&lattice_agent->lon, u_s + dt).second;
    LatticePlanPoint here = lattice_lat_state(env, plan, now_step, frenet->s, frenet->s_dot, plan_accel);
    LatticePlanPoint ahead = lattice_lat_state(env, plan, now_step + 1, frenet->s + frenet->s_dot * dt, frenet->s_dot, plan_accel);
    *lat_error_out = (float) here.value - frenet->d;
    float error_d = clip((float) here.value - frenet->d, -LATTICE_E_D_MAX_M, LATTICE_E_D_MAX_M);
    float error_d_dot = clip((float) here.first - frenet->d_dot, -LATTICE_E_DD_MAX_MPS, LATTICE_E_DD_MAX_MPS);
    float feedback_gain = fminf(1.0f, fabsf(frenet->speed) / LATTICE_FEEDBACK_FULL_SPEED_MPS);
    float d_ddot_cmd = (float) ahead.second + feedback_gain * (LATTICE_KP_LAT * error_d + LATTICE_KD_LAT * error_d_dot);
    LatticeLongState now_long = {agent->sim_speed_signed, agent->accel_long};
    LatticeLongState next_long = lattice_integrate_long(now_long, jerk_long, agent, dt, lattice_speed_cap_mps(env, agent));
    float speed_next = frenet->speed + next_long.accel * dt;
    float d_dot_next = frenet->d_dot + d_ddot_cmd * dt;
    float d_next = frenet->d + frenet->d_dot * dt;
    float s_next = frenet->s + frenet->s_dot * dt;
    LatticeRailPoint ahead_rail = lattice_rail_at(rail, s_next);
    float factor_next = fmaxf(1.0f - ahead_rail.curvature * d_next, LATTICE_MIN_FRENET_FACTOR);
    float s_dot_next = (speed_next < 0.0f ? -1.0f : 1.0f) * sqrtf(fmaxf(0.0f, speed_next * speed_next - d_dot_next * d_dot_next)) / factor_next;
    float s_ddot_cmd = (s_dot_next - frenet->s_dot) / dt;
    float accel_tangent = s_ddot_cmd * factor_next - s_dot_next * (2.0f * ahead_rail.curvature * d_dot_next + ahead_rail.curvature_rate * s_dot_next * d_next);
    float accel_normal = ahead_rail.curvature * s_dot_next * s_dot_next * factor_next + d_ddot_cmd;
    float world_x = accel_tangent * cosf(ahead_rail.heading) - accel_normal * sinf(ahead_rail.heading);
    float world_y = accel_tangent * sinf(ahead_rail.heading) + accel_normal * cosf(ahead_rail.heading);
    return -world_x * agent->sin_heading + world_y * agent->cos_heading;
}

static LatticeCommand compute_lattice_command(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, const LatticeFrenet *frenet, int now_step) {
    LatticeCommand command = {0};
    float dt = env->dt;
    command.jerk_long = lattice_longitudinal_jerk(env, lattice_agent, agent, now_step, &command.speed_error_mps);
    float accel_lat_cmd;
    int low_speed = fabsf(frenet->speed) < env->lattice.low_speed_mps || lattice_agent->gear < 0;
    if (low_speed) {
        accel_lat_cmd = lattice_low_speed_lateral_accel(env, lattice_agent, agent, frenet, command.jerk_long, now_step);
        LatticePlanPoint here = lattice_lat_state(env, &lattice_agent->lat, now_step, frenet->s, frenet->s_dot, 0.0f);
        command.lat_error_m = (float) here.value - frenet->d;
    } else {
        accel_lat_cmd = lattice_lateral_accel(env, lattice_agent, agent, frenet, command.jerk_long, now_step, &command.lat_error_m);
    }
    command.jerk_lat = (accel_lat_cmd - agent->accel_lat) / (agent->reward_coefs[REWARD_COEF_STEER] * dt);
    return command;
}

// ========================================
// Plan initialisation, automatic re-plans and rail switches
// ========================================

static int lattice_is_policy_agent(const Agent *agent) {
    return agent->controller == CONTROLLER_POLICY && !agent->removed && agent->sim_valid;
}

static float lattice_curvature_limit(const Agent *agent) {
    return tanf(STEERING_ANGLE_LIMIT) / agent->wheelbase;
}

static LatticePlanPoint lattice_measured_lat(const Agent *agent, const LatticeFrenet *frenet) {
    float accel_normal = agent->accel_lat * cosf(frenet->heading_error) + agent->accel_long * sinf(frenet->heading_error);
    float factor = fmaxf(1.0f - frenet->curvature * frenet->d, LATTICE_MIN_FRENET_FACTOR);
    LatticePlanPoint state = {frenet->d, frenet->d_dot, accel_normal - frenet->curvature * frenet->s_dot * frenet->s_dot * factor};
    return state;
}

// Werling start: continue from the committed plan when it is tracked, else from the measured state (time derivatives)
static LatticePlanPoint lattice_lat_start(Drive *env, const struct LatticeAgent *lattice_agent, const Agent *agent, const LatticeFrenet *frenet, int now_step) {
    LatticePlanPoint planned = lattice_lat_state(env, &lattice_agent->lat, now_step, frenet->s, frenet->s_dot, agent->accel_long);
    if (fabs(planned.value - frenet->d) < LATTICE_WERLING_TOL_D_M && fabs(planned.first - frenet->d_dot) < LATTICE_WERLING_TOL_DD_MPS) {
        return planned;
    }
    return lattice_measured_lat(agent, frenet);
}

static void lattice_lon_start(Drive *env, const struct LatticeAgent *lattice_agent, const Agent *agent, int now_step, double *sigma0, float *v0, float *a0) {
    *sigma0 = lattice_agent->sigma_m;
    *v0 = agent->sim_speed_signed;
    *a0 = agent->accel_long;
    const struct LatticeLonPlan *plan = &lattice_agent->lon;
    if (plan->kind == LATTICE_LON_KIND_EMERGENCY) {
        return;
    }
    LatticePlanPoint planned = lattice_lon_eval(plan, lattice_elapsed_s(env, plan->start_step, now_step));
    if (fabs(planned.first - *v0) < LATTICE_WERLING_TOL_V_MPS && fabs(planned.value - *sigma0) < LATTICE_WERLING_TOL_SIGMA_M) {
        *sigma0 = planned.value;
        *v0 = (float) planned.first;
        *a0 = (float) planned.second;
    }
}

// shortest duration (0.3 s grid) whose transient to target keeps |d''| and |d'''| inside the tracking margin
static int lattice_replan_steps(const Drive *env, const Agent *agent, LatticePlanPoint start, float target_d_m) {
    float accel_limit = LATTICE_REPLAN_ACCEL_MARGIN * ACCEL_LAT_LIMIT[1];
    float jerk_limit = LATTICE_REPLAN_ACCEL_MARGIN * JERK_LAT[2] * agent->reward_coefs[REWARD_COEF_STEER];
    int grid_steps = (int) lroundf(LATTICE_REPLAN_T_STEP_S / env->dt);
    grid_steps = grid_steps < 1 ? 1 : grid_steps;
    int max_steps = (int) lroundf(LATTICE_REPLAN_T_MAX_S / env->dt);
    for (int steps = grid_steps; steps <= max_steps; steps += grid_steps) {
        double coefs[6];
        double horizon = steps * env->dt;
        lattice_quintic_coefs(start.value, start.first, start.second, target_d_m, 0.0, 0.0, horizon, coefs);
        int ok = 1;
        for (int sample_idx = 0; sample_idx <= LATTICE_STOP_PROFILE_SAMPLES && ok; sample_idx++) {
            double u = horizon * sample_idx / LATTICE_STOP_PROFILE_SAMPLES;
            ok = fabs(lattice_poly_point(coefs, u).second) <= accel_limit && fabs(lattice_poly_third(coefs, u)) <= jerk_limit;
        }
        if (ok) {
            return steps;
        }
    }
    return max_steps;
}

// distance over which a rest-to-rest lateral move of offset_m stays within the steering-curvature margin
static float lattice_offset_distance_m(const Agent *agent, float offset_m) {
    return sqrtf(LATTICE_LOW_SPEED_CURVATURE_PEAK * fabsf(offset_m) / (LATTICE_ENVELOPE_MARGIN * lattice_curvature_limit(agent)));
}

static float lattice_backup_remaining_m(const struct LatticeAgent *lattice_agent) {
    if (lattice_agent->lon.kind != LATTICE_LON_KIND_BACKUP) {
        return 0.0f;
    }
    return (float) fabs(lattice_agent->lon.target_sigma_m - lattice_agent->sigma_m);
}

static void replan_lattice_lateral(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, const LatticeFrenet *frenet, int now_step) {
    struct LatticeLatPlan *plan = &lattice_agent->lat;
    LatticePlanPoint measured = lattice_measured_lat(agent, frenet);
    int kind = plan->kind == LATTICE_LAT_KIND_HOLD ? LATTICE_LAT_KIND_OFFSET : plan->kind;
    lattice_agent->counters.auto_replans += 1.0f;
    if (plan->mode == LATTICE_LAT_MODE_TIME) {
        int steps = lattice_replan_steps(env, agent, measured, plan->target_d_m);
        set_lattice_lat_time_plan(plan, kind, frenet->d, frenet->d_dot, (float) measured.second, plan->target_d_m, steps, env->dt, now_step);
        return;
    }
    int dir = plan->dir;
    float distance_m = fmaxf(lattice_offset_distance_m(agent, plan->target_d_m - frenet->d), 2.0f * LATTICE_RAIL_SPACING_M);
    if (lattice_agent->gear < 0 && distance_m > lattice_backup_remaining_m(lattice_agent)) {
        float held_d = (float) lattice_lat_eval_dist(plan, frenet->s).value;
        int reindexed_stop = plan->reindexed_stop;
        set_lattice_lat_hold(plan, held_d, frenet->s, dir);
        plan->reindexed_stop = reindexed_stop;
        return;
    }
    float d_prime = fabsf(frenet->s_dot) > 0.1f ? frenet->d_dot / frenet->s_dot : 0.0f;
    set_lattice_lat_dist_plan(plan, kind, frenet->d, d_prime, 0.0f, plan->target_d_m, distance_m, frenet->s, dir);
    plan->reindexed_low_speed = 1;
}

// once per trigger: a stop/EMERGENCY becomes active -> natural d(s) continuation of the current plan state
static void reindex_lattice_lateral_for_stop(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, const LatticeFrenet *frenet, int now_step, float stopping_distance_m) {
    struct LatticeLatPlan *plan = &lattice_agent->lat;
    if (plan->reindexed_stop || fabsf(frenet->s_dot) < LATTICE_RELEASE_SPEED_MPS) {
        return;
    }
    LatticePlanPoint state = lattice_lat_state(env, plan, now_step, frenet->s, frenet->s_dot, agent->accel_long);
    float d_prime = (float) state.first / frenet->s_dot;
    float distance_m = fmaxf(LATTICE_EMERGENCY_MIN_DIST_M, stopping_distance_m);
    int dir = frenet->s_dot >= 0.0f ? 1 : -1;
    int kind = plan->kind;
    set_lattice_lat_dist_plan(plan, kind, (float) state.value, d_prime, 0.0f, (float) state.value + 0.5f * d_prime * dir * distance_m, distance_m, frenet->s, dir);
    plan->reindexed_stop = 1;
    plan->reindexed_low_speed = 1;
}

// once per trigger: speed drops below the low speed -> committed target as a d(s) plan
static void reindex_lattice_lateral_low_speed(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, const LatticeFrenet *frenet, int now_step) {
    struct LatticeLatPlan *plan = &lattice_agent->lat;
    if (plan->mode != LATTICE_LAT_MODE_TIME || plan->reindexed_low_speed) {
        return;
    }
    LatticePlanPoint state = lattice_lat_state(env, plan, now_step, frenet->s, frenet->s_dot, agent->accel_long);
    const struct LatticeConfig *cfg = &env->lattice;
    float distance_m = cfg->low_speed_distances_m[cfg->lat_duration_count - 1];
    for (int distance_idx = 0; distance_idx < cfg->lat_duration_count; distance_idx++) {
        float candidate_m = cfg->low_speed_distances_m[distance_idx];
        if (LATTICE_LOW_SPEED_CURVATURE_PEAK * fabsf(plan->target_d_m - (float) state.value) / (candidate_m * candidate_m) <= LATTICE_ENVELOPE_MARGIN * lattice_curvature_limit(agent)) {
            distance_m = candidate_m;
            break;
        }
    }
    float d_prime = fabsf(frenet->s_dot) > 0.1f ? (float) state.first / frenet->s_dot : 0.0f;
    int dir = frenet->s_dot >= 0.0f ? 1 : -1;
    set_lattice_lat_dist_plan(plan, plan->kind, (float) state.value, d_prime, 0.0f, plan->target_d_m, distance_m, frenet->s, dir);
    plan->reindexed_low_speed = 1;
}

// automatic plan updates of the move stage: phantom-braking end, tracking re-plan, low-speed re-index
static void update_lattice_plans_automatic(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, const LatticeFrenet *frenet, int now_step) {
    int phantom_active = agent->phantom_braking_counter > 0;
    if (lattice_agent->phantom_was_active && !phantom_active) {
        float speed = fabsf(agent->sim_speed_signed) < LATTICE_RELEASE_SPEED_MPS ? 0.0f : agent->sim_speed_signed;
        set_lattice_lon_hold(&lattice_agent->lon, lattice_agent->sigma_m, speed, now_step);
    }
    lattice_agent->phantom_was_active = phantom_active;
    if (!lattice_agent->has_reference) {
        return;
    }
    struct LatticeLatPlan *plan = &lattice_agent->lat;
    int stopping = lattice_lon_is_stopping(&lattice_agent->lon);
    if (fabsf(frenet->speed) < env->lattice.low_speed_mps && !stopping && lattice_agent->gear > 0) {
        reindex_lattice_lateral_low_speed(env, lattice_agent, agent, frenet, now_step);
    }
    LatticePlanPoint planned = lattice_lat_state(env, plan, now_step, frenet->s, frenet->s_dot, agent->accel_long);
    if (fabs(planned.value - frenet->d) > LATTICE_REPLAN_D_M || fabs(planned.first - frenet->d_dot) > LATTICE_REPLAN_DD_MPS) {
        replan_lattice_lateral(env, lattice_agent, agent, frenet, now_step);
    }
    if (lattice_agent->lane_change_active && lattice_lat_plan_ended(env, plan, now_step, frenet->s)) {
        lattice_agent->lane_change_active = 0;
    }
}

// re-expresses the lateral plan on a new rail offset by spacing_m (same physical plan)
static void carry_lattice_lat_plan(Drive *env, struct LatticeAgent *lattice_agent, LatticePlanPoint time_state, LatticePlanPoint spatial_state, float remaining, float spacing_m, float new_s_m, int now_step) {
    struct LatticeLatPlan *plan = &lattice_agent->lat;
    float target_d_m = plan->target_d_m - spacing_m;
    int kind = plan->kind;
    int reindexed_low_speed = plan->reindexed_low_speed;
    int reindexed_stop = plan->reindexed_stop;
    int dir = plan->dir;
    if (plan->mode == LATTICE_LAT_MODE_TIME && remaining >= 1.0f) {
        set_lattice_lat_time_plan(plan, kind, (float) time_state.value - spacing_m, (float) time_state.first, (float) time_state.second, target_d_m, (int) remaining, env->dt, now_step);
    } else if (plan->mode == LATTICE_LAT_MODE_DIST && remaining > LATTICE_RAIL_SPACING_M) {
        set_lattice_lat_dist_plan(plan, kind, (float) spatial_state.value - spacing_m, (float) spatial_state.first, (float) spatial_state.second, target_d_m, remaining, new_s_m, dir);
    } else {
        set_lattice_lat_hold(plan, target_d_m, new_s_m, dir);
    }
    plan->reindexed_low_speed = reindexed_low_speed;
    plan->reindexed_stop = reindexed_stop;
}

static void mark_lattice_rail_changed(struct LatticeAgent *lattice_agent) {
    lattice_agent->rail_changed_flag = 1;
    lattice_agent->late_exit_pending = 1;
}

// drift: the car is closer to the neighbour's rail; switch rails, keep the plans physically unchanged
static void switch_lattice_rail_drift(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, int side, int now_step) {
    LatticeFrenet old_frenet = lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint);
    struct LatticeLatPlan *plan = &lattice_agent->lat;
    LatticePlanPoint time_state = lattice_lat_state(env, plan, now_step, old_frenet.s, old_frenet.s_dot, agent->accel_long);
    LatticePlanPoint spatial_state = plan->mode == LATTICE_LAT_MODE_DIST ? lattice_lat_eval_dist(plan, old_frenet.s) : time_state;
    float remaining = plan->mode == LATTICE_LAT_MODE_TIME ? (float) (plan->end_step - now_step)
                                                          : plan->horizon - plan->dir * (old_frenet.s - plan->start_s_m);
    float spacing_m = lattice_agent->neighbour_offset_m[side];
    int sample_idx = build_lattice_fresh_rail(env, &lattice_agent->rail, agent, lattice_agent->neighbour_lane[side], lattice_agent->neighbour_arc_m[side]);
    if (sample_idx < 0) {
        build_lattice_straight_rail(&lattice_agent->rail, agent);
        lattice_agent->has_reference = 0;
        sample_idx = (int) lroundf(LATTICE_TRAIL_KEEP_M / LATTICE_RAIL_SPACING_M);
    }
    lattice_agent->projection_hint = sample_idx;
    LatticeFrenet new_frenet = lattice_frenet_state(&lattice_agent->rail, agent, sample_idx);
    carry_lattice_lat_plan(env, lattice_agent, time_state, spatial_state, remaining, spacing_m, new_frenet.s, now_step);
    mark_lattice_rail_changed(lattice_agent);
    lattice_agent->counters.drift_changes += 1.0f;
}

// rebuild from the geometric search (lost, no-lane recovery); the lateral plan holds the measured offset
static int rebuild_lattice_reference(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, int now_step) {
    int lane_idx;
    float arc_m;
    int found = lattice_find_base_lane(env, agent, &lane_idx, &arc_m);
    int sample_idx = found ? build_lattice_fresh_rail(env, &lattice_agent->rail, agent, lane_idx, arc_m) : -1;
    if (sample_idx < 0) {
        build_lattice_straight_rail(&lattice_agent->rail, agent);
        sample_idx = (int) lroundf(LATTICE_TRAIL_KEEP_M / LATTICE_RAIL_SPACING_M);
    }
    lattice_agent->has_reference = sample_idx >= 0 && !lattice_agent->rail.is_straight_fallback;
    lattice_agent->projection_hint = sample_idx;
    LatticeFrenet frenet = lattice_frenet_state(&lattice_agent->rail, agent, sample_idx);
    int dir = lattice_agent->gear;
    set_lattice_lat_hold(&lattice_agent->lat, lattice_agent->has_reference ? frenet.d : 0.0f, frenet.s, dir);
    lattice_agent->lane_change_active = 0;
    mark_lattice_rail_changed(lattice_agent);
    (void) now_step;
    return lattice_agent->has_reference;
}

static float lattice_backup_duration_s(const Drive *env, const Agent *agent, float distance_m) {
    float c_acc = fminf(agent->reward_coefs[REWARD_COEF_ACC], 1.0f);
    float c_throttle = fminf(agent->reward_coefs[REWARD_COEF_THROTTLE], 1.0f);
    int grid_steps = (int) lroundf(LATTICE_BACKUP_T_GRID_S / env->dt);
    int max_steps = (int) lroundf(LATTICE_BACKUP_T_MAX_S / env->dt);
    for (int steps = grid_steps; steps <= max_steps; steps += grid_steps) {
        double coefs[6];
        double horizon = steps * env->dt;
        lattice_quintic_coefs(0.0, 0.0, 0.0, -distance_m, 0.0, 0.0, horizon, coefs);
        // every step (not the sparse mask grid): a one-step plan would otherwise only check its end state
        int sample_count = steps < 2 ? 2 : steps;
        double previous_t = 0.0;
        double previous_accel = 0.0;
        int ok = 1;
        for (int sample_idx = 1; sample_idx <= sample_count && ok; sample_idx++) {
            double t = lround((double) sample_idx * steps / sample_count) * env->dt;
            LatticePlanPoint point = lattice_poly_point(coefs, t);
            double jerk = (point.second - previous_accel) / (t - previous_t);
            ok = point.first >= MAX_BACKWARD_SPEED - LATTICE_CHECK_EPS && point.first <= LATTICE_CHECK_EPS
                && point.second >= ACCEL_LONG_LIMIT[0] && point.second <= ACCEL_LONG_LIMIT[1] * c_acc + LATTICE_CHECK_EPS
                && jerk >= JERK_LONG[0] * c_throttle - LATTICE_CHECK_EPS && jerk <= JERK_LONG[3] * c_throttle + LATTICE_CHECK_EPS;
            previous_t = t;
            previous_accel = point.second;
        }
        if (ok) {
            return steps * env->dt;
        }
    }
    return max_steps * env->dt;
}

// fresh per-episode lattice state for one active slot (all c_reset branches, after compute_metrics)
static void reset_lattice_slot(Drive *env, int active_idx) {
    struct LatticeAgent *lattice_agent = &env->lattice_agents[active_idx];
    Agent *agent = &env->agents[env->active_agent_indices[active_idx]];
    memset(&lattice_agent->counters, 0, sizeof(lattice_agent->counters));
    memset(lattice_agent->mask, 0, sizeof(lattice_agent->mask));
    memset(lattice_agent->preview_world_xy, 0, sizeof(lattice_agent->preview_world_xy));
    lattice_agent->plan_change_rms_m = 0.0f;
    lattice_agent->route_progress.goal_idx = -1;
    lattice_agent->route_progress.previous_distance_m = INFINITY;
    memset(&lattice_agent->turn, 0, sizeof(lattice_agent->turn));
    lattice_agent->turn.goal_idx = -1;
    lattice_agent->counters.first_motion_step = -1.0f;
    lattice_agent->sigma_m = 0.0;
    lattice_agent->lane_change_active = 0;
    lattice_agent->rail_changed_flag = 1;
    lattice_agent->late_exit_pending = 1;
    lattice_agent->context_step = -1;
    lattice_agent->live_split_slot = -1;
    lattice_agent->phantom_was_active = 0;
    lattice_agent->has_reference = 0;
    lattice_agent->projection_hint = 0;
    lattice_agent->rail.sample_count = 0;
    lattice_agent->neighbour_check_lane[0] = -1;
    lattice_agent->neighbour_check_lane[1] = -1;
    float speed = agent->sim_speed_signed;
    lattice_agent->gear = speed < -LATTICE_RELEASE_SPEED_MPS ? -1 : 1;
    int now_step = env->timestep + 1;
    set_lattice_lon_hold(&lattice_agent->lon, 0.0, fabsf(speed) < LATTICE_RELEASE_SPEED_MPS ? 0.0f : speed, now_step);
    set_lattice_lat_hold(&lattice_agent->lat, 0.0f, 0.0f, lattice_agent->gear);
    if (!lattice_is_policy_agent(agent)) {
        return;
    }
    for (int backup_idx = 0; backup_idx < env->lattice.backup_distance_count; backup_idx++) {
        lattice_agent->backup_duration_s[backup_idx] = lattice_backup_duration_s(env, agent, env->lattice.backup_distances_m[backup_idx]);
    }
    lattice_agent->reverse_emergency_stop_m = lattice_emergency_stop_distance(env, agent, (LatticeLongState) {MAX_BACKWARD_SPEED, 0.0f}, -1);
    rebuild_lattice_reference(env, lattice_agent, agent, now_step);
}

// per step after the moves: follow the car on its rail, extend / trim / regrow the trail, clear ended lane changes
static void update_lattice_chain(Drive *env, int active_idx) {
    struct LatticeAgent *lattice_agent = &env->lattice_agents[active_idx];
    Agent *agent = &env->agents[env->active_agent_indices[active_idx]];
    if (!lattice_is_policy_agent(agent) || lattice_agent->rail.sample_count < 2) {
        return;
    }
    struct LatticeRail *rail = &lattice_agent->rail;
    LatticeFrenet frenet = lattice_project(rail, agent->sim_x, agent->sim_y, lattice_agent->projection_hint);
    lattice_agent->projection_hint = frenet.sample_idx;
    if (!lattice_agent->has_reference) {
        return;
    }
    int lookahead_short = lattice_rail_end_s(rail) - frenet.s < lattice_required_lookahead_m(env) && !rail->chain_is_complete;
    int trail_short = frenet.s - rail->s_start_m < LATTICE_TRAIL_BEHIND_M && fabsf(agent->sim_speed) < LATTICE_RELEASE_SPEED_MPS
        && rail->chain_start_arc_m < LATTICE_RAIL_SPACING_M && env->lattice_lanes[rail->lanes[0]].predecessor_count > 0
        && rail->lane_count < LATTICE_CHAIN_MAX_LANES;
    if (lookahead_short || trail_short) {
        regenerate_lattice_rail_at(env, lattice_agent, agent, lattice_car_chain_arc(env, rail, frenet.sample_idx), frenet.s);
    }
}

// ========================================
// Cell validity (plan pair checked on K = min(T/dt, LATTICE_MASK_SAMPLES) evenly spaced steps)
// ========================================

typedef struct {
    float sigma_m;
    float speed;
    float accel;
} LatticeLonSample;

typedef struct {
    const struct LatticeRail *rail;
    const struct LatticeLatPlan *lat;
    int lat_elapsed_steps;
    const struct LatticeLonPlan *lon;
    int lon_elapsed_steps;
    const LatticeLonSample *lon_profile;
    int horizon_steps;
    int reverse;
    float start_s_m;
    float end_speed_mps;
    int check_end_speed;
} LatticeCheck;

static LatticeLonSample lattice_lon_sample_at(const Drive *env, const LatticeCheck *check, int step) {
    if (check->lon_profile != NULL) {
        return check->lon_profile[step];
    }
    LatticePlanPoint point = lattice_lon_eval(check->lon, (double) (check->lon_elapsed_steps + step) * env->dt);
    LatticeLonSample sample = {(float) point.value, (float) point.first, (float) point.second};
    return sample;
}

// per-step longitudinal profile of the EMERGENCY rule from the car's state
static void lattice_emergency_profile(const Drive *env, const Agent *agent, int gear, double sigma_m, int step_count, LatticeLonSample *profile) {
    LatticeLongState state = {agent->sim_speed_signed, agent->accel_long};
    float speed_cap = lattice_speed_cap_mps(env, agent);
    profile[0].sigma_m = (float) sigma_m;
    profile[0].speed = state.speed;
    profile[0].accel = state.accel;
    for (int step = 1; step <= step_count; step++) {
        float jerk = lattice_emergency_jerk(env, agent, state, gear, NULL);
        LatticeLongState next = lattice_integrate_long(state, jerk, agent, env->dt, speed_cap);
        profile[step].sigma_m = profile[step - 1].sigma_m + 0.5f * (next.speed + state.speed) * env->dt;
        profile[step].speed = next.speed;
        profile[step].accel = next.accel;
        state = next;
    }
}

typedef struct {
    float s_m;
    float d_m;
    float speed;
    float s_dot;
    float accel_long;
    float accel_lat;
    float steer;
    float envelope_excess;
    float edge_excess;
    float frenet_deficit;
    float unfollowable_excess;
    int unfollowable;
} LatticeCheckState;

static LatticeCheckState lattice_check_state(const Drive *env, const Agent *agent, const LatticeCheck *check, int step, float s_m, LatticeLonSample lon) {
    LatticeRailProfile rail_point = lattice_rail_profile_at(check->rail, s_m);
    float curvature = rail_point.curvature;
    const struct LatticeLatPlan *lat = check->lat;
    float d_m, d_dot, d_ddot, path_curvature;
    float speed = lon.speed;
    if (lat->mode == LATTICE_LAT_MODE_TIME) {
        LatticePlanPoint point = lattice_lat_eval_time(lat, (double) (check->lat_elapsed_steps + step) * env->dt);
        d_m = (float) point.value;
        d_dot = (float) point.first;
        d_ddot = (float) point.second;
    } else {
        LatticePlanPoint spatial = lattice_lat_eval_dist(lat, s_m);
        float factor = fmaxf(1.0f - curvature * (float) spatial.value, LATTICE_MIN_FRENET_FACTOR);
        float s_dot = speed / sqrtf(factor * factor + (float) (spatial.first * spatial.first));
        d_m = (float) spatial.value;
        d_dot = (float) spatial.first * s_dot;
        d_ddot = (float) spatial.second * s_dot * s_dot + (float) spatial.first * lon.accel;
    }
    float factor = 1.0f - curvature * d_m;
    float safe_factor = fmaxf(factor, LATTICE_MIN_FRENET_FACTOR);
    float s_dot = (speed < 0.0f ? -1.0f : 1.0f) * sqrtf(fmaxf(0.0f, speed * speed - d_dot * d_dot)) / safe_factor;
    float along_rate = fmaxf(fabsf(s_dot) * safe_factor, LATTICE_MIN_RATE_MPS);
    float path_norm = sqrtf(along_rate * along_rate + d_dot * d_dot);
    float path_cos = along_rate / path_norm;
    float path_sin = d_dot / path_norm;
    float accel_normal = curvature * s_dot * s_dot * safe_factor + d_ddot;
    LatticeCheckState state;
    state.s_m = s_m;
    state.d_m = d_m;
    state.speed = speed;
    state.s_dot = s_dot;
    state.accel_long = lon.accel;
    state.accel_lat = accel_normal * path_cos - lon.accel * path_sin;
    if (lat->mode == LATTICE_LAT_MODE_DIST) {
        LatticePlanPoint spatial = lattice_lat_eval_dist(lat, s_m);
        float d_prime = (float) spatial.first;
        float cos_h = safe_factor / sqrtf(safe_factor * safe_factor + d_prime * d_prime);
        float tan_h = d_prime / safe_factor;
        path_curvature = (((float) spatial.second + (rail_point.curvature_rate * d_m + curvature * d_prime) * tan_h) * cos_h * cos_h / safe_factor + curvature) * cos_h / safe_factor;
    } else {
        float speed_floor = fmaxf(fabsf(speed), 1.0f);
        path_curvature = state.accel_lat / (speed_floor * speed_floor);
    }
    float curvature_limit = lattice_curvature_limit(agent);
    state.unfollowable = fabsf(curvature) > curvature_limit;
    state.steer = atanf(path_curvature * agent->wheelbase);
    state.unfollowable_excess = fabsf(path_curvature) - (fabsf(curvature) / safe_factor + LATTICE_UNFOLLOWABLE_TOLERANCE);
    state.envelope_excess = fabsf(s_dot) - rail_point.v_env - LATTICE_ENVELOPE_TOLERANCE_MPS;
    float half_width = 0.5f * agent->sim_width;
    float left_bound = fmaxf(rail_point.edge_left_m - half_width, 0.0f);
    float right_bound = fmaxf(rail_point.edge_right_m - half_width, 0.0f);
    state.edge_excess = fmaxf(d_m - left_bound, -right_bound - d_m);
    state.frenet_deficit = LATTICE_MIN_FRENET_FACTOR - factor;
    return state;
}

static inline int lattice_sample_step(int sample_idx, int horizon_steps, int sample_count) {
    return (2 * sample_idx * horizon_steps + sample_count) / (2 * sample_count);
}

// longitudinal-only part of lattice_check_ok on the same samples: a cheap exact pre-screen
static int lattice_check_longitudinal_ok(const Drive *env, const Agent *agent, const LatticeCheck *check, const LatticeLonSample *samples) {
    float dt = env->dt;
    float c_throttle = agent->reward_coefs[REWARD_COEF_THROTTLE];
    float c_acc = agent->reward_coefs[REWARD_COEF_ACC];
    float speed_cap = lattice_speed_cap_mps(env, agent);
    int horizon_steps = check->horizon_steps < 1 ? 1 : check->horizon_steps;
    int sample_count = horizon_steps < LATTICE_MASK_SAMPLES ? horizon_steps : LATTICE_MASK_SAMPLES;
    LatticeLonSample previous = samples[0];
    int previous_step = 0;
    for (int sample_idx = 1; sample_idx <= sample_count; sample_idx++) {
        int step = lattice_sample_step(sample_idx, horizon_steps, sample_count);
        float interval_s = (step - previous_step) * dt;
        LatticeLonSample lon = samples[sample_idx];
        float accel_rate = (lon.accel - previous.accel) / interval_s;
        int ok = lon.accel >= ACCEL_LONG_LIMIT[0] - LATTICE_CHECK_EPS && lon.accel <= ACCEL_LONG_LIMIT[1] * c_acc + LATTICE_CHECK_EPS
            && accel_rate >= JERK_LONG[0] * c_throttle - LATTICE_CHECK_EPS && accel_rate <= JERK_LONG[3] * c_throttle + LATTICE_CHECK_EPS
            && fabsf(lon.speed) <= speed_cap + LATTICE_CHECK_EPS;
        ok = ok && (check->reverse ? lon.speed <= LATTICE_CHECK_EPS && lon.speed >= MAX_BACKWARD_SPEED - LATTICE_CHECK_EPS : lon.speed >= -LATTICE_CHECK_EPS);
        if (!ok) {
            return 0;
        }
        previous = lon;
        previous_step = step;
    }
    return 1;
}

static int lattice_check_ok(const Drive *env, const Agent *agent, const LatticeCheck *check) {
    int check_horizon_steps = check->horizon_steps < 1 ? 1 : check->horizon_steps;
    int check_sample_count = check_horizon_steps < LATTICE_MASK_SAMPLES ? check_horizon_steps : LATTICE_MASK_SAMPLES;
    LatticeLonSample samples[LATTICE_MASK_SAMPLES + 1];
    samples[0] = lattice_lon_sample_at(env, check, 0);
    for (int sample_idx = 1; sample_idx <= check_sample_count; sample_idx++) {
        samples[sample_idx] = lattice_lon_sample_at(env, check, lattice_sample_step(sample_idx, check_horizon_steps, check_sample_count));
    }
    if (!lattice_check_longitudinal_ok(env, agent, check, samples)) {
        return 0;
    }
    float dt = env->dt;
    float c_steer = agent->reward_coefs[REWARD_COEF_STEER];
    float c_throttle = agent->reward_coefs[REWARD_COEF_THROTTLE];
    float c_acc = agent->reward_coefs[REWARD_COEF_ACC];
    float speed_cap = lattice_speed_cap_mps(env, agent);
    int horizon_steps = check->horizon_steps < 1 ? 1 : check->horizon_steps;
    int sample_count = horizon_steps < LATTICE_MASK_SAMPLES ? horizon_steps : LATTICE_MASK_SAMPLES;
    LatticeCheckState previous = lattice_check_state(env, agent, check, 0, check->start_s_m, samples[0]);
    float envelope_allow = fmaxf(0.0f, previous.envelope_excess);
    float edge_allow = fmaxf(0.0f, previous.edge_excess);
    float frenet_allow = fmaxf(0.0f, previous.frenet_deficit);
    float accel_lat_allow = fmaxf(ACCEL_LAT_LIMIT[1], fabsf(previous.accel_lat));
    int previous_step = 0;
    float s_m = check->start_s_m;
    float sigma_prev = samples[0].sigma_m;
    for (int sample_idx = 1; sample_idx <= sample_count; sample_idx++) {
        int step = lattice_sample_step(sample_idx, horizon_steps, sample_count);
        float interval_s = (step - previous_step) * dt;
        LatticeLonSample lon = samples[sample_idx];
        float rail_ratio = fabsf(previous.speed) > 1e-3f ? previous.s_dot / previous.speed : 1.0f;
        s_m += (lon.sigma_m - sigma_prev) * rail_ratio;
        sigma_prev = lon.sigma_m;
        LatticeCheckState state = lattice_check_state(env, agent, check, step, s_m, lon);
        float accel_lat_rate = fabsf(state.accel_lat - previous.accel_lat) / interval_s;
        float accel_long_rate = (state.accel_long - previous.accel_long) / interval_s;
        int ok = fabsf(state.accel_lat) <= accel_lat_allow + LATTICE_CHECK_EPS
            && state.accel_long >= ACCEL_LONG_LIMIT[0] - LATTICE_CHECK_EPS && state.accel_long <= ACCEL_LONG_LIMIT[1] * c_acc + LATTICE_CHECK_EPS
            && accel_lat_rate <= JERK_LAT[2] * c_steer + LATTICE_CHECK_EPS
            && accel_long_rate >= JERK_LONG[0] * c_throttle - LATTICE_CHECK_EPS && accel_long_rate <= JERK_LONG[3] * c_throttle + LATTICE_CHECK_EPS
            && fabsf(state.speed) <= speed_cap + LATTICE_CHECK_EPS
            && state.envelope_excess <= envelope_allow + LATTICE_CHECK_EPS
            && state.edge_excess <= edge_allow + LATTICE_CHECK_EPS
            && state.frenet_deficit <= frenet_allow + LATTICE_CHECK_EPS;
        if (check->reverse) {
            ok = ok && state.speed <= LATTICE_CHECK_EPS && state.speed >= MAX_BACKWARD_SPEED - LATTICE_CHECK_EPS;
        } else {
            ok = ok && state.speed >= -LATTICE_CHECK_EPS;
        }
        if (state.unfollowable) {
            ok = ok && state.unfollowable_excess <= LATTICE_CHECK_EPS;
        } else if (!previous.unfollowable) {
            ok = ok && fabsf(state.steer) <= STEERING_ANGLE_LIMIT + LATTICE_CHECK_EPS
                && fabsf(state.steer - previous.steer) <= LATTICE_STEER_RATE_RPS * interval_s + LATTICE_CHECK_EPS;
        }
        if (!ok) {
            return 0;
        }
        previous = state;
        previous_step = step;
    }
    if (check->check_end_speed) {
        LatticeRailProfile end_point = lattice_rail_profile_at(check->rail, s_m);
        return fabsf(check->end_speed_mps) <= end_point.v_env + LATTICE_ENVELOPE_TOLERANCE_MPS + envelope_allow;
    }
    return 1;
}

// ========================================
// Decision context: snapshot, cell construction, masks
// ========================================

typedef struct {
    LatticeFrenet frenet;
    int now_step;
    int stopped_exactly;
    int at_rest;
    int reversing;
    int stopping;
    int low_speed;
    LatticePlanPoint lat_start;
    LatticePlanPoint lat_start_spatial;
    double lon_sigma0;
    float lon_speed0;
    float lon_accel0;
    float backup_remaining_m;
} LatticeContext;

static int lattice_is_reversing(const struct LatticeAgent *lattice_agent, const Agent *agent) {
    if (agent->sim_speed_signed < -LATTICE_RELEASE_SPEED_MPS) {
        return 1;
    }
    return lattice_agent->gear < 0 && !lattice_is_stopped_exactly(agent);
}

static LatticeContext lattice_make_context(Drive *env, const struct LatticeAgent *lattice_agent, const Agent *agent, int now_step) {
    LatticeContext ctx;
    ctx.now_step = now_step;
    ctx.frenet = lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint);
    ctx.stopped_exactly = lattice_is_stopped_exactly(agent);
    const struct LatticeLonPlan *lon = &lattice_agent->lon;
    float planned_speed = lon->kind == LATTICE_LON_KIND_EMERGENCY ? 0.0f : (float) lattice_lon_eval(lon, lattice_elapsed_s(env, lon->start_step, now_step)).first;
    ctx.at_rest = fabsf(agent->sim_speed_signed) < LATTICE_RELEASE_SPEED_MPS && fabsf(planned_speed) < LATTICE_RELEASE_SPEED_MPS;
    ctx.reversing = lattice_is_reversing(lattice_agent, agent);
    ctx.stopping = lattice_lon_is_stopping(lon);
    ctx.low_speed = fabsf(agent->sim_speed_signed) < env->lattice.low_speed_mps;
    ctx.lat_start = lattice_lat_start(env, lattice_agent, agent, &ctx.frenet, now_step);
    if (lattice_agent->lat.mode == LATTICE_LAT_MODE_DIST && fabs(ctx.lat_start.value - lattice_lat_eval_dist(&lattice_agent->lat, ctx.frenet.s).value) < 1e-6) {
        ctx.lat_start_spatial = lattice_lat_eval_dist(&lattice_agent->lat, ctx.frenet.s);
    } else {
        float s_dot = ctx.frenet.s_dot;
        float d_prime = fabsf(s_dot) > LATTICE_FEEDBACK_FULL_SPEED_MPS ? (float) ctx.lat_start.first / s_dot : 0.0f;
        float d_second = fabsf(s_dot) > LATTICE_FEEDBACK_FULL_SPEED_MPS ? ((float) ctx.lat_start.second - d_prime * agent->accel_long) / (s_dot * s_dot) : 0.0f;
        LatticePlanPoint spatial = {ctx.lat_start.value, d_prime, d_second};
        ctx.lat_start_spatial = spatial;
    }
    lattice_lon_start(env, lattice_agent, agent, now_step, &ctx.lon_sigma0, &ctx.lon_speed0, &ctx.lon_accel0);
    ctx.backup_remaining_m = lattice_backup_remaining_m(lattice_agent);
    return ctx;
}

// steps until the longitudinal profile has covered distance_m (capped), used as the horizon of d(s) checks
static int lattice_steps_to_cover(const Drive *env, const LatticeCheck *check, float distance_m) {
    int max_steps = (int) lroundf(LATTICE_STOP_T_MAX_S / env->dt);
    float sigma0 = lattice_lon_sample_at(env, check, 0).sigma_m;
    for (int step = 1; step <= max_steps; step++) {
        if (fabsf(lattice_lon_sample_at(env, check, step).sigma_m - sigma0) >= distance_m) {
            return step;
        }
    }
    return max_steps;
}

// lateral candidate for a cell on `rail` (d_shift = offset of that rail from the agent's own); 0 when not buildable
// reverse_distance_m > 0 applies the reverse rules (last duration only, S = the remaining back-up distance, dir -1)
static int lattice_build_lat_cell(Drive *env, const struct LatticeAgent *lattice_agent, const Agent *agent, const LatticeContext *ctx, int cell, float s_on_rail, float d_shift, float reverse_distance_m, struct LatticeLatPlan *plan) {
    const struct LatticeConfig *cfg = &env->lattice;
    int duration_idx = lattice_lat_cell_duration_idx(cfg, cell);
    int lane_side = lattice_lat_lane_side(cfg, cell);
    int kind = lane_side != 0 ? LATTICE_LAT_KIND_LANE_CHANGE : LATTICE_LAT_KIND_OFFSET;
    float target_d_m = 0.0f;
    if (lattice_lat_is_oncoming(cfg, cell)) {
        assert(d_shift == 0.0f);
        target_d_m = lattice_oncoming_offset_at(env, lattice_agent, ctx->frenet.sample_idx);
        if (!(target_d_m > 0.0f) || reverse_distance_m > 0.0f) {
            return 0;
        }
    } else if (lane_side == 0) {
        target_d_m = lattice_lat_cell_offset(cfg, cell);
    }
    float d0 = (float) ctx->lat_start.value - d_shift;
    if (reverse_distance_m > 0.0f) {
        if (duration_idx != cfg->lat_duration_count - 1 || lane_side != 0 || reverse_distance_m < 2.0f * LATTICE_RAIL_SPACING_M) {
            return 0;
        }
        set_lattice_lat_dist_plan(plan, kind, d0, (float) ctx->lat_start_spatial.first, (float) ctx->lat_start_spatial.second, target_d_m, reverse_distance_m, s_on_rail, -1);
        return 1;
    }
    if (ctx->at_rest) {
        set_lattice_lat_dist_plan(plan, kind, d0, 0.0f, 0.0f, target_d_m, cfg->low_speed_distances_m[duration_idx], s_on_rail, 1);
        return 1;
    }
    if (ctx->low_speed || ctx->stopping) {
        set_lattice_lat_dist_plan(plan, kind, d0, (float) ctx->lat_start_spatial.first, (float) ctx->lat_start_spatial.second, target_d_m, cfg->low_speed_distances_m[duration_idx], s_on_rail, 1);
        plan->reindexed_low_speed = 1;
        plan->reindexed_stop = ctx->stopping;
        return 1;
    }
    set_lattice_lat_time_plan(plan, kind, d0, (float) ctx->lat_start.first, (float) ctx->lat_start.second, target_d_m, cfg->lat_duration_steps[duration_idx], env->dt, ctx->now_step);
    return 1;
}

// d(s) horizons (steps to cover a distance) memoised per longitudinal source within one context
typedef struct {
    int count;
    float distance_m[2 * LATTICE_MAX_LAT_DURATIONS + 2];
    int steps[2 * LATTICE_MAX_LAT_DURATIONS + 2];
} LatticeCoverCache;

static int lattice_cached_steps_to_cover(const Drive *env, const LatticeCheck *check, float distance_m, LatticeCoverCache *cache) {
    for (int entry_idx = 0; cache != NULL && entry_idx < cache->count; entry_idx++) {
        if (cache->distance_m[entry_idx] == distance_m) {
            return cache->steps[entry_idx];
        }
    }
    int steps = lattice_steps_to_cover(env, check, distance_m);
    if (cache != NULL && cache->count < 2 * LATTICE_MAX_LAT_DURATIONS + 2) {
        cache->distance_m[cache->count] = distance_m;
        cache->steps[cache->count] = steps;
        cache->count++;
    }
    return steps;
}

// lateral candidate checked against the held longitudinal plan (nominal launch at rest; emergency_profile if EMERGENCY)
static int lattice_lat_cell_ok(Drive *env, const struct LatticeAgent *lattice_agent, const Agent *agent, const LatticeContext *ctx, const struct LatticeRail *rail, float s_on_rail, const struct LatticeLatPlan *plan, const struct LatticeLonPlan *lon_override, const LatticeLonSample *emergency_profile, LatticeCoverCache *cover_cache) {
    LatticeCheck check = {0};
    check.rail = rail;
    check.lat = plan;
    check.lat_elapsed_steps = 0;
    check.start_s_m = s_on_rail;
    check.reverse = plan->mode == LATTICE_LAT_MODE_DIST && plan->dir < 0;
    struct LatticeLonPlan launch;
    const struct LatticeLonPlan *lon = lon_override != NULL ? lon_override : &lattice_agent->lon;
    int lon_elapsed = lon_override != NULL ? 0 : ctx->now_step - lattice_agent->lon.start_step;
    if (lon_override == NULL && ctx->at_rest && !check.reverse) {
        int launch_steps = (int) lroundf(LATTICE_NOMINAL_LAUNCH_T_S / env->dt);
        set_lattice_lon_speed_plan(&launch, lattice_agent->sigma_m, 0.0f, 0.0f, LATTICE_NOMINAL_LAUNCH_SPEED_MPS, launch_steps, env->dt, ctx->now_step, -1);
        lon = &launch;
        lon_elapsed = 0;
    }
    check.lon = lon;
    check.lon_elapsed_steps = lon_elapsed;
    if (lon->kind == LATTICE_LON_KIND_EMERGENCY) {
        check.lon_profile = emergency_profile;
    }
    if (plan->mode == LATTICE_LAT_MODE_TIME) {
        check.horizon_steps = plan->end_step - ctx->now_step;
    } else {
        check.horizon_steps = lattice_cached_steps_to_cover(env, &check, plan->horizon, cover_cache);
    }
    return lattice_check_ok(env, agent, &check);
}

// longitudinal candidate checked against the held lateral plan; stop cells pick their duration here
static int lattice_build_lon_cell(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, const LatticeContext *ctx, int cell, const struct LatticeLatPlan *lat_override, struct LatticeLonPlan *plan) {
    const struct LatticeConfig *cfg = &env->lattice;
    assert(cell != cfg->lon_turn_cell);
    LatticeCheck check = {0};
    check.rail = &lattice_agent->rail;
    check.lat = lat_override != NULL ? lat_override : &lattice_agent->lat;
    check.lat_elapsed_steps = lat_override != NULL ? 0 : ctx->now_step - lattice_agent->lat.start_step;
    check.start_s_m = ctx->frenet.s;
    check.lon = plan;
    check.lon_elapsed_steps = 0;
    if (cell == cfg->lon_emergency_cell) {
        set_lattice_lon_emergency(plan, ctx->now_step, cell);
        return 1;
    }
    if (cell < cfg->lon_speed_cell_count) {
        int speed_idx = cell % cfg->lon_speed_count;
        int duration_idx = cell / cfg->lon_speed_count;
        float speed_mps = cfg->lon_speeds_mps[speed_idx];
        set_lattice_lon_speed_plan(plan, ctx->lon_sigma0, ctx->lon_speed0, ctx->lon_accel0, speed_mps, cfg->lon_duration_steps[duration_idx], env->dt, ctx->now_step, cell);
        check.horizon_steps = cfg->lon_duration_steps[duration_idx];
        check.check_end_speed = 1;
        check.end_speed_mps = speed_mps;
        return lattice_check_ok(env, agent, &check);
    }
    if (cell >= cfg->lon_backup_cell_base) {
        int backup_idx = cell - cfg->lon_backup_cell_base;
        int steps = (int) lroundf(lattice_agent->backup_duration_s[backup_idx] / env->dt);
        set_lattice_lon_stop_plan(plan, LATTICE_LON_KIND_BACKUP, lattice_agent->sigma_m, 0.0f, 0.0f, -cfg->backup_distances_m[backup_idx], steps, env->dt, ctx->now_step, cell);
        check.horizon_steps = steps;
        check.reverse = 1;
        return lattice_check_ok(env, agent, &check);
    }
    int is_stop_line = cell == cfg->lon_stop_line_cell;
    float distance_m = is_stop_line ? lattice_agent->stop_line_distance_m : cfg->stop_distances_m[cell - cfg->lon_stop_cell_base];
    if (!(distance_m > LATTICE_RAIL_SPACING_M)) {
        return 0;
    }
    int kind = is_stop_line ? LATTICE_LON_KIND_STOP_LINE : LATTICE_LON_KIND_STOP;
    int grid_steps = (int) lroundf(LATTICE_STOP_T_GRID_S / env->dt);
    int max_steps = (int) lroundf(LATTICE_STOP_T_MAX_S / env->dt);
    int fallback_steps[LATTICE_STOP_FALLBACK_CANDIDATES];
    int fallback_count = 0;
    int comfortable_tried = 0;
    // shortest comfortable duration first; if its full check fails, the shortest feasible ones
    for (int steps = grid_steps; steps <= max_steps; steps += grid_steps) {
        double coefs[6];
        double horizon = steps * env->dt;
        lattice_quintic_coefs(ctx->lon_sigma0, ctx->lon_speed0, ctx->lon_accel0, ctx->lon_sigma0 + distance_m, 0.0, 0.0, horizon, coefs);
        int feasible = 1;
        int comfortable = 1;
        for (int sample_idx = 1; sample_idx <= LATTICE_STOP_PROFILE_SAMPLES && feasible; sample_idx++) {
            double u = horizon * sample_idx / LATTICE_STOP_PROFILE_SAMPLES;
            LatticePlanPoint point = lattice_poly_point(coefs, u);
            feasible = point.first >= -LATTICE_CHECK_EPS && point.second >= ACCEL_LONG_LIMIT[0] && point.second <= ACCEL_LONG_LIMIT[1] * agent->reward_coefs[REWARD_COEF_ACC];
            comfortable = comfortable && fabs(point.second) <= LATTICE_COMFORT_ACCEL_MPS2 && lattice_poly_third(coefs, u) >= LATTICE_COMFORT_BRAKE_JERK;
        }
        if (!feasible) {
            continue;
        }
        if (comfortable && !comfortable_tried) {
            comfortable_tried = 1;
            set_lattice_lon_stop_plan(plan, kind, ctx->lon_sigma0, ctx->lon_speed0, ctx->lon_accel0, distance_m, steps, env->dt, ctx->now_step, cell);
            check.horizon_steps = steps;
            if (lattice_check_ok(env, agent, &check)) {
                return 1;
            }
            continue;
        }
        if (fallback_count < LATTICE_STOP_FALLBACK_CANDIDATES) {
            fallback_steps[fallback_count++] = steps;
        }
        if (comfortable_tried && fallback_count == LATTICE_STOP_FALLBACK_CANDIDATES) {
            break;
        }
    }
    for (int candidate_idx = 0; candidate_idx < fallback_count; candidate_idx++) {
        set_lattice_lon_stop_plan(plan, kind, ctx->lon_sigma0, ctx->lon_speed0, ctx->lon_accel0, distance_m, fallback_steps[candidate_idx], env->dt, ctx->now_step, cell);
        check.horizon_steps = fallback_steps[candidate_idx];
        if (lattice_check_ok(env, agent, &check)) {
            return 1;
        }
    }
    return 0;
}

// the committed longitudinal plan (hold phase included) passes the checks over LATTICE_KEEP_CHECK_S
static int lattice_lon_keep_ok(Drive *env, const struct LatticeAgent *lattice_agent, const Agent *agent, const LatticeContext *ctx) {
    if (lattice_agent->lon.kind == LATTICE_LON_KIND_EMERGENCY) {
        return 1;
    }
    LatticeCheck check = {0};
    check.rail = &lattice_agent->rail;
    check.lat = &lattice_agent->lat;
    check.lat_elapsed_steps = ctx->now_step - lattice_agent->lat.start_step;
    check.lon = &lattice_agent->lon;
    check.lon_elapsed_steps = ctx->now_step - lattice_agent->lon.start_step;
    check.start_s_m = ctx->frenet.s;
    check.horizon_steps = (int) lroundf(LATTICE_KEEP_CHECK_S / env->dt);
    check.reverse = lattice_agent->lon.kind == LATTICE_LON_KIND_BACKUP;
    return lattice_check_ok(env, agent, &check);
}

// a light-controlled lane (or one entered from one) on the chain within extent_m behind car_s_m
static int lattice_light_lane_behind(
    const Drive *env,
    const struct LatticeAgent *lattice_agent,
    float car_s_m,
    float extent_m) {
    const struct LatticeRail *rail = &lattice_agent->rail;
    int car_slot = rail->chain_slot[lattice_agent->projection_hint];
    for (int lane_slot = car_slot; lane_slot >= 0; lane_slot--) {
        if (rail->lane_end_s_m[lane_slot] < car_s_m - extent_m) {
            break;
        }
        const struct LatticeLaneInfo *info = &env->lattice_lanes[rail->lanes[lane_slot]];
        if (info->traffic_light_idx != -1) {
            return 1;
        }
        for (int pred_idx = 0; pred_idx < info->predecessor_count && lane_slot > 0; pred_idx++) {
            if (env->lattice_lanes[info->predecessors[pred_idx]].traffic_light_idx != -1
                && rail->lane_start_s_m[lane_slot] > car_s_m - extent_m) {
                return 1;
            }
        }
    }
    return 0;
}

// back-up allowed: stopped exactly, trail long enough, and no light-controlled lane within the backward extent
static int lattice_backup_allowed(
    Drive *env,
    const struct LatticeAgent *lattice_agent,
    const Agent *agent,
    const LatticeContext *ctx,
    float distance_m) {
    if (!ctx->stopped_exactly || !lattice_agent->has_reference) {
        return 0;
    }
    const struct LatticeRail *rail = &lattice_agent->rail;
    float extent_m
        = distance_m + LATTICE_BACKUP_LANDING_M + lattice_agent->reverse_emergency_stop_m + 0.5f * agent->sim_length;
    if (ctx->frenet.s - rail->s_start_m < extent_m) {
        return 0;
    }
    return !lattice_light_lane_behind(env, lattice_agent, ctx->frenet.s, extent_m);
}

// distance from the car to the next light-controlled stop line on the chain, minus half length and margin
typedef struct {
    float distance_m; // -1 when no stop line lies ahead on the chain
    int traffic_idx;
} LatticeStopLine;

static LatticeStopLine lattice_next_stop_line(const Drive *env, const struct LatticeAgent *lattice_agent, const Agent *agent, float car_s_m) {
    const struct LatticeRail *rail = &lattice_agent->rail;
    LatticeStopLine none = {-1.0f, -1};
    if (!lattice_agent->has_reference) {
        return none;
    }
    for (int lane_slot = rail->chain_slot[lattice_agent->projection_hint]; lane_slot < rail->lane_count; lane_slot++) {
        int traffic_idx = env->lattice_lanes[rail->lanes[lane_slot]].traffic_light_idx;
        if (traffic_idx == -1 || rail->lane_start_s_m[lane_slot] > lattice_rail_end_s(rail)) {
            continue;
        }
        const TrafficControlElement *traffic = &env->traffic_elements[traffic_idx];
        float mid_x = 0.5f * (traffic->stop_line[0] + traffic->stop_line[3]);
        float mid_y = 0.5f * (traffic->stop_line[1] + traffic->stop_line[4]);
        int hint = (int) lroundf((rail->lane_start_s_m[lane_slot] - rail->s_start_m) / LATTICE_RAIL_SPACING_M);
        hint = hint < 0 ? 0 : (hint > rail->sample_count - 1 ? rail->sample_count - 1 : hint);
        LatticeFrenet line = lattice_project(rail, mid_x, mid_y, hint);
        float distance_m = line.s - car_s_m - 0.5f * agent->sim_length - LATTICE_STOP_LINE_MARGIN_M;
        if (line.s > car_s_m) {
            LatticeStopLine stop_line = {distance_m, traffic_idx};
            return stop_line;
        }
    }
    return none;
}

typedef struct {
    float distance_m; // -1 when no stop line is reported
    int light_state;
} LatticeReportedLight;

// the car's own next stop line and light state as its observation reports them
static LatticeReportedLight lattice_reported_light(const Drive *env, const struct LatticeAgent *lattice_agent, const Agent *agent, float car_s_m) {
    LatticeStopLine stop_line = lattice_next_stop_line(env, lattice_agent, agent, car_s_m);
    LatticeReportedLight reported = {-1.0f, TRAFFIC_CONTROL_STATE_UNKNOWN};
    if (stop_line.traffic_idx < 0 || (env->lattice.light_in_view && !traffic_control_in_view(env, agent, stop_line.traffic_idx))) {
        return reported;
    }
    reported.distance_m = stop_line.distance_m;
    const TrafficControlElement *light = &env->traffic_elements[stop_line.traffic_idx];
    if (env->timestep < light->state_size) {
        reported.light_state = light->states[env->timestep];
    }
    return reported;
}

// ========================================
// Turning around: execution on lane-less leg rails, the offer rule and the masks while turning
// ========================================

static float compute_lane_progress(
    RoadMapElement *lane,
    float pos_x,
    float pos_y,
    float cos_heading,
    float sin_heading,
    bool align_heading,
    float *out_dist_sq);

// the leg's arc through the car's pose along its heading, padded past both ends; returns the car's sample
static int build_lattice_turn_leg_rail(const Agent *agent, const struct LatticeTurnLeg *leg, struct LatticeRail *rail) {
    float lap_m = LATTICE_TURN_RAIL_LAP_FRACTION * 2.0f * (float) M_PI / fabsf(leg->curvature);
    float pad_m = fmaxf(fminf(LATTICE_TURN_RAIL_PAD_M, 0.5f * (lap_m - leg->length_m)), 2.0f * LATTICE_RAIL_SPACING_M);
    float behind_m = pad_m + (leg->gear < 0 ? leg->length_m : 0.0f);
    float ahead_m = pad_m + (leg->gear > 0 ? leg->length_m : 0.0f);
    int behind_samples = (int) ceilf(behind_m / LATTICE_RAIL_SPACING_M);
    int count = behind_samples + (int) ceilf(ahead_m / LATTICE_RAIL_SPACING_M) + 1;
    assert(count <= LATTICE_RAIL_SAMPLES);
    LatticePose car = {agent->sim_x, agent->sim_y, agent->sim_heading};
    rail->lane_count = 0;
    rail->is_straight_fallback = 1;
    rail->chain_is_dead_end = 0;
    rail->chain_is_complete = 1;
    rail->chain_start_arc_m = 0.0f;
    rail->sample_count = count;
    rail->s_start_m = 0.0f;
    for (int sample_idx = 0; sample_idx < count; sample_idx++) {
        LatticePose pose
            = lattice_arc_pose(car, leg->curvature, (sample_idx - behind_samples) * LATTICE_RAIL_SPACING_M);
        rail->x[sample_idx] = pose.x;
        rail->y[sample_idx] = pose.y;
        rail->heading[sample_idx] = pose.heading;
        rail->curvature[sample_idx] = leg->curvature;
        rail->edge_left_m[sample_idx] = LATTICE_EDGE_SEARCH_M;
        rail->edge_right_m[sample_idx] = LATTICE_EDGE_SEARCH_M;
        rail->lane_arc_m[sample_idx] = 0.0f;
        rail->chain_slot[sample_idx] = 0;
    }
    compute_lattice_envelope(agent, rail);
    return behind_samples;
}

// shortest 0.3 s-grid rest-to-rest move of distance_m (signed) inside the longitudinal limits and |v| <= speed_cap_mps
static int lattice_turn_leg_steps(const Drive *env, const Agent *agent, float distance_m, float speed_cap_mps) {
    float c_acc = fminf(agent->reward_coefs[REWARD_COEF_ACC], 1.0f);
    float c_throttle = fminf(agent->reward_coefs[REWARD_COEF_THROTTLE], 1.0f);
    int grid_steps = (int) lroundf(LATTICE_BACKUP_T_GRID_S / env->dt);
    int max_steps = (int) lroundf(LATTICE_TURN_T_MAX_S / env->dt);
    float direction = distance_m < 0.0f ? -1.0f : 1.0f;
    for (int steps = grid_steps; steps <= max_steps; steps += grid_steps) {
        double coefs[6];
        double horizon = steps * env->dt;
        lattice_quintic_coefs(0.0, 0.0, 0.0, distance_m, 0.0, 0.0, horizon, coefs);
        int ok = 1;
        for (int sample_idx = 0; sample_idx <= LATTICE_STOP_PROFILE_SAMPLES && ok; sample_idx++) {
            double u = horizon * sample_idx / LATTICE_STOP_PROFILE_SAMPLES;
            LatticePlanPoint point = lattice_poly_point(coefs, u);
            double jerk = lattice_poly_third(coefs, u);
            ok = direction * point.first >= -LATTICE_CHECK_EPS && fabs(point.first) <= speed_cap_mps + LATTICE_CHECK_EPS
                && point.second >= LATTICE_TURN_LIMIT_FRACTION * ACCEL_LONG_LIMIT[0]
                && point.second <= LATTICE_TURN_LIMIT_FRACTION * ACCEL_LONG_LIMIT[1] * c_acc
                && jerk >= LATTICE_TURN_LIMIT_FRACTION * JERK_LONG[0] * c_throttle
                && jerk <= LATTICE_TURN_LIMIT_FRACTION * JERK_LONG[3] * c_throttle;
        }
        if (ok) {
            return steps;
        }
    }
    return max_steps;
}

// first leg of turn->plan from rest: leg rail, d = 0 hold, wheel swung to the leg's lock, then the rest-to-rest move
static void start_lattice_turn_leg(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, int now_step) {
    const struct LatticeTurnLeg *leg = &lattice_agent->turn.plan.legs[0];
    int car_sample = build_lattice_turn_leg_rail(agent, leg, &lattice_agent->rail);
    lattice_agent->has_reference = 0;
    lattice_agent->projection_hint = car_sample;
    lattice_agent->live_split_slot = -1;
    lattice_agent->gear = leg->gear;
    lattice_agent->lane_change_active = 0;
    LatticeFrenet frenet = lattice_frenet_state(&lattice_agent->rail, agent, car_sample);
    set_lattice_lat_hold(&lattice_agent->lat, 0.0f, frenet.s, leg->gear);
    float steer_target = atanf(leg->curvature * agent->wheelbase);
    int dwell_steps = (int) ceilf(fabsf(steer_target - agent->steering_angle) / (LATTICE_STEER_RATE_RPS * env->dt)) + 1;
    float speed_cap = fminf(
        fminf(LATTICE_TURN_LEG_SPEED_MPS, sqrtf(LATTICE_ENVELOPE_MARGIN * ACCEL_LAT_LIMIT[1] / fabsf(leg->curvature))),
        lattice_speed_cap_mps(env, agent));
    float distance_m = leg->gear * leg->length_m;
    int steps = lattice_turn_leg_steps(env, agent, distance_m, speed_cap);
    int kind = leg->gear > 0 ? LATTICE_LON_KIND_STOP : LATTICE_LON_KIND_BACKUP;
    set_lattice_lon_stop_plan(
        &lattice_agent->lon,
        kind,
        lattice_agent->sigma_m,
        0.0f,
        0.0f,
        distance_m,
        steps,
        env->dt,
        now_step + dwell_steps,
        env->lattice.lon_turn_cell);
    mark_lattice_rail_changed(lattice_agent);
}

static float lattice_turn_remaining_rad(const struct LatticeTurnState *turn, const Agent *agent) {
    return lattice_turn_rotation_to_go(turn->plan.sense, turn->target_heading, agent->sim_heading);
}

// done or aborted: back on a lane rail from the base-lane search, plans held
static void finish_lattice_turn(
    Drive *env,
    struct LatticeAgent *lattice_agent,
    const Agent *agent,
    int now_step,
    int completed) {
    struct LatticeTurnState *turn = &lattice_agent->turn;
    turn->active = 0;
    turn->cache_valid = 0;
    int at_rest = lattice_is_stopped_exactly(agent);
    if (at_rest) {
        lattice_agent->gear = 1;
    }
    rebuild_lattice_reference(env, lattice_agent, agent, now_step);
    if (at_rest) {
        set_lattice_lon_hold(&lattice_agent->lon, lattice_agent->sigma_m, 0.0f, now_step);
    } else {
        set_lattice_lon_emergency(&lattice_agent->lon, now_step, env->lattice.lon_emergency_cell);
    }
    // the last leg can end off the landing lane's centre: merge onto it over distance once the car pulls away
    if (completed && at_rest && lattice_agent->has_reference) {
        const struct LatticeConfig *cfg = &env->lattice;
        LatticeFrenet frenet = lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint);
        float distance_m = fmaxf(
            lattice_offset_distance_m(agent, frenet.d),
            cfg->low_speed_distances_m[cfg->lat_duration_count - 1]);
        set_lattice_lat_dist_plan(
            &lattice_agent->lat,
            LATTICE_LAT_KIND_OFFSET,
            frenet.d,
            0.0f,
            0.0f,
            0.0f,
            distance_m,
            frenet.s,
            1);
        turn->landing = fabsf(frenet.d) > LATTICE_TURN_LANDED_D_M;
    }
    lattice_agent->counters.turn_completions += completed;
    lattice_agent->counters.turn_aborts += !completed;
}

// at rest mid-turn: finish once the heading is reached, else re-plan the rest from the actual pose
static void continue_lattice_turn(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, int now_step) {
    struct LatticeTurnState *turn = &lattice_agent->turn;
    if (lattice_turn_remaining_rad(turn, agent) <= LATTICE_TURN_HEADING_TOL_RAD) {
        finish_lattice_turn(env, lattice_agent, agent, now_step, 1);
        return;
    }
    // committed: narrower sweeps, then either rotation sense, before giving up
    const float sweep_margins_m[3] = {LATTICE_TURN_MARGIN_M, LATTICE_TURN_START_MARGIN_M, LATTICE_TURN_LAST_MARGIN_M};
    struct LatticeTurnPlan plan;
    int found = 0;
    for (int try_idx = 0; try_idx < 6 && !found; try_idx++) {
        int sense = try_idx < 3 ? turn->plan.sense : 0;
        const struct LatticeTurnSearch search
            = {sense, turn->radius_count, sweep_margins_m[try_idx % 3], LATTICE_TURN_REPLAN_START_MARGIN_M, 1};
        found = plan_lattice_turnaround(env, agent, turn->start_lane, turn->target_heading, &search, &plan);
    }
    if (!found) {
        finish_lattice_turn(
            env,
            lattice_agent,
            agent,
            now_step,
            lattice_turn_remaining_rad(turn, agent) <= LATTICE_TURN_SETTLE_HEADING_RAD);
        return;
    }
    turn->plan = plan;
    turn->target_heading = plan.target_heading;
    start_lattice_turn_leg(env, lattice_agent, agent, now_step);
}

static void start_lattice_turn(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, int now_step) {
    struct LatticeTurnState *turn = &lattice_agent->turn;
    assert(turn->cache_valid && turn->cache_ok);
    turn->active = 1;
    turn->plan = turn->cache_plan;
    turn->target_heading = turn->cache_plan.target_heading;
    turn->radius_count = turn->cache_plan.radius_idx + 1;
    turn->cache_valid = 0;
    lattice_agent->counters.turn_starts += 1.0f;
    start_lattice_turn_leg(env, lattice_agent, agent, now_step);
}

// move stage while turning: a leg that has run its plan and come to rest hands over; one that never rests aborts
static void advance_lattice_turn(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, int now_step) {
    if (now_step < lattice_agent->lon.end_step) {
        return;
    }
    if (lattice_is_stopped_exactly(agent)) {
        continue_lattice_turn(env, lattice_agent, agent, now_step);
    } else if (now_step > lattice_agent->lon.end_step + LATTICE_TURN_SETTLE_STEPS) {
        finish_lattice_turn(env, lattice_agent, agent, now_step, 0);
    }
}

// a light, stop or yield line of any direction within radius_m of the car: a turn would cross or land on it
static int lattice_stop_line_within(const Drive *env, const Agent *agent, float radius_m) {
    for (int traffic_idx = 0; traffic_idx < env->num_traffic_elements; traffic_idx++) {
        const TrafficControlElement *traffic = &env->traffic_elements[traffic_idx];
        if (traffic->type == TRAFFIC_CONTROL_TYPE_NONE || fabsf(traffic->stop_line[2] - agent->sim_z) > Z_BUFFER) {
            continue;
        }
        float seg_x = traffic->stop_line[3] - traffic->stop_line[0];
        float seg_y = traffic->stop_line[4] - traffic->stop_line[1];
        float rel_x = agent->sim_x - traffic->stop_line[0];
        float rel_y = agent->sim_y - traffic->stop_line[1];
        float seg_len_sq = seg_x * seg_x + seg_y * seg_y;
        float t = seg_len_sq > 1e-9f ? clip((rel_x * seg_x + rel_y * seg_y) / seg_len_sq, 0.0f, 1.0f) : 0.0f;
        float dx = rel_x - t * seg_x;
        float dy = rel_y - t * seg_y;
        if (dx * dx + dy * dy < radius_m * radius_m) {
            return 1;
        }
    }
    return 0;
}

// time the plan takes: per leg a lock-to-lock wheel swing at rest plus the rest-to-rest move
static float lattice_turn_duration_s(const Drive *env, const Agent *agent, const struct LatticeTurnPlan *plan) {
    float duration_s = 0.0f;
    for (int leg_idx = 0; leg_idx < plan->leg_count; leg_idx++) {
        const struct LatticeTurnLeg *leg = &plan->legs[leg_idx];
        float speed_cap = fminf(
            LATTICE_TURN_LEG_SPEED_MPS,
            sqrtf(LATTICE_ENVELOPE_MARGIN * ACCEL_LAT_LIMIT[1] / fabsf(leg->curvature)));
        duration_s += 2.0f * atanf(fabsf(leg->curvature) * agent->wheelbase) / LATTICE_STEER_RATE_RPS;
        duration_s += lattice_turn_leg_steps(env, agent, leg->gear * leg->length_m, speed_cap) * env->dt;
    }
    return duration_s;
}

// no other vehicle inside the turn's reach now, nor closing in fast enough to get there before the turn ends
static int lattice_turn_partners_clear(const Drive *env, const Agent *agent, float reach_m, float duration_s) {
    for (int agent_idx = 0; agent_idx < env->num_total_agents; agent_idx++) {
        const Agent *other = &env->agents[agent_idx];
        if (other == agent || other->removed || other->sim_x == INVALID_POSITION) {
            continue;
        }
        float rel_x = other->sim_x - agent->sim_x;
        float rel_y = other->sim_y - agent->sim_y;
        float distance_m = sqrtf(rel_x * rel_x + rel_y * rel_y);
        if (distance_m > LATTICE_TURN_PARTNER_HORIZON_M + reach_m || fabsf(other->sim_z - agent->sim_z) > Z_BUFFER) {
            continue;
        }
        float closing_mps
            = distance_m > 1e-3f ? fmaxf(0.0f, -(other->sim_vx * rel_x + other->sim_vy * rel_y) / distance_m) : 0.0f;
        float other_radius_m
            = 0.5f * sqrtf(other->sim_length * other->sim_length + other->sim_width * other->sim_width);
        if (distance_m - other_radius_m < reach_m + closing_mps * duration_s) {
            return 0;
        }
    }
    return 1;
}

// no stop line or light the turn could cross or land on, and no vehicle inside reach_m or closing in on it in time
static int lattice_turn_clear_within(
    const Drive *env,
    const struct LatticeAgent *lattice_agent,
    const Agent *agent,
    const LatticeFrenet *frenet,
    float reach_m,
    float duration_s) {
    LatticeStopLine stop_line = lattice_next_stop_line(env, lattice_agent, agent, frenet->s);
    if ((stop_line.traffic_idx >= 0 && stop_line.distance_m < reach_m)
        || lattice_light_lane_behind(env, lattice_agent, frenet->s, reach_m)
        || lattice_stop_line_within(env, agent, reach_m)) {
        return 0;
    }
    return lattice_turn_partners_clear(env, agent, reach_m, duration_s);
}

// all rules for turning around from the car's pose (rest not required) and the plan; 0 when not possible here now
static int lattice_turnaround_possible(
    Drive *env,
    struct LatticeAgent *lattice_agent,
    const Agent *agent,
    const LatticeFrenet *frenet) {
    struct LatticeTurnState *turn = &lattice_agent->turn;
    if (!lattice_agent->has_reference || lattice_agent->rail.is_straight_fallback || agent->is_phantom_braker
        || lattice_car_on_connector(env, lattice_agent) || lattice_plan_in_oncoming(env, lattice_agent, frenet)
        || lattice_is_borrowing(env, lattice_agent, frenet)) {
        return 0;
    }
    const struct LatticeRail *rail = &lattice_agent->rail;
    turn->start_lane = rail->lanes[rail->chain_slot[lattice_agent->projection_hint]];
    turn->target_heading = lattice_wrap_angle(lattice_rail_at(rail, frenet->s).heading + (float) M_PI);
    float tight_reach_m = lattice_turn_reach_m(agent, lattice_turn_curvature(env, agent));
    const struct LatticeTurnSearch search
        = {0, LATTICE_TURN_RADIUS_COUNT, LATTICE_TURN_MARGIN_M, LATTICE_TURN_START_MARGIN_M, 0};
    if (!lattice_turn_clear_within(env, lattice_agent, agent, frenet, tight_reach_m, 0.0f)
        || !plan_lattice_turnaround(env, agent, turn->start_lane, turn->target_heading, &search, &turn->cache_plan)) {
        return 0;
    }
    float reach_m = lattice_turn_reach_m(agent, fabsf(turn->cache_plan.legs[0].curvature));
    return lattice_turn_clear_within(
        env,
        lattice_agent,
        agent,
        frenet,
        reach_m,
        lattice_turn_duration_s(env, agent, &turn->cache_plan));
}

// offered from rest when possible; cached while the car does not move
static int lattice_turnaround_offered(
    Drive *env,
    struct LatticeAgent *lattice_agent,
    const Agent *agent,
    const LatticeContext *ctx) {
    struct LatticeTurnState *turn = &lattice_agent->turn;
    if (!ctx->stopped_exactly) {
        turn->cache_valid = 0;
        return 0;
    }
    if (!(turn->cache_valid && turn->cache_x == agent->sim_x && turn->cache_y == agent->sim_y
          && turn->cache_heading == agent->sim_heading)) {
        turn->cache_valid = 1;
        turn->cache_x = agent->sim_x;
        turn->cache_y = agent->sim_y;
        turn->cache_heading = agent->sim_heading;
        turn->cache_ok = lattice_turnaround_possible(env, lattice_agent, agent, &ctx->frenet);
    }
    turn->feasible = turn->cache_ok;
    turn->probe_step = ctx->now_step;
    return turn->cache_ok;
}

// driving distance from lane_idx at arc_m to goal_arc_m on goal_lane along lane links; INFINITY if unreachable
static float lattice_lane_route_m(const Drive *env, int lane_idx, float arc_m, int goal_lane, float goal_arc_m) {
    if (lane_idx == goal_lane && goal_arc_m >= arc_m) {
        return goal_arc_m - arc_m;
    }
    const struct LatticeLaneInfo *info = &env->lattice_lanes[lane_idx];
    float best_exit_m = INFINITY;
    for (int slot_idx = 0; slot_idx < info->exit_count; slot_idx++) {
        best_exit_m = fminf(best_exit_m, lattice_goal_distance_m(env, info->exit_slots[slot_idx], goal_lane));
    }
    return info->length_m - arc_m + best_exit_m + goal_arc_m;
}

// (route if the car turned around here - route ahead) / LATTICE_OBS_ROUTE_GAP_NORM_M in [-1, 1]; 1 = no way
static float lattice_turn_route_gap(
    Drive *env,
    struct LatticeAgent *lattice_agent,
    const Agent *agent,
    const LatticeFrenet *frenet) {
    const struct LatticeRail *rail = &lattice_agent->rail;
    int goal_lane = lattice_agent_goal_lane(agent);
    if (!lattice_agent->has_reference || rail->is_straight_fallback || goal_lane < 0) {
        return 1.0f;
    }
    int car_lane = rail->lanes[rail->chain_slot[frenet->sample_idx]];
    const struct LatticeProfileSample *profile
        = lattice_profile_at(env, car_lane, rail->lane_arc_m[frenet->sample_idx]);
    if (profile->turn_lane < 0) {
        return 1.0f;
    }
    struct LatticeTurnState *turn = &lattice_agent->turn;
    if (turn->goal_idx != agent->current_goal_idx || turn->goal_x != agent->current_goal_x
        || turn->goal_y != agent->current_goal_y) {
        turn->goal_idx = agent->current_goal_idx;
        turn->goal_x = agent->current_goal_x;
        turn->goal_y = agent->current_goal_y;
        turn->goal_arc_m = compute_lane_progress(
            &env->road_elements[goal_lane],
            agent->current_goal_x,
            agent->current_goal_y,
            1.0f,
            0.0f,
            false,
            NULL);
    }
    float ahead_m = lattice_route_distance_m(env, lattice_agent, *frenet, goal_lane, turn->goal_arc_m);
    float turned_m = (float) M_PI / lattice_turn_curvature(env, agent)
        + lattice_lane_route_m(env, profile->turn_lane, profile->turn_arc_m, goal_lane, turn->goal_arc_m);
    if (!isfinite(turned_m)) {
        return 1.0f;
    }
    if (!isfinite(ahead_m)) {
        return -1.0f;
    }
    return clip((turned_m - ahead_m) / LATTICE_OBS_ROUTE_GAP_NORM_M, -1.0f, 1.0f);
}

static void lattice_mask_index0_only(struct LatticeAgent *lattice_agent, const struct LatticeConfig *cfg) {
    memset(lattice_agent->mask, 0, sizeof(lattice_agent->mask));
    for (int factor_idx = 0; factor_idx < LATTICE_ACTION_FACTORS; factor_idx++) {
        lattice_agent->mask[lattice_mask_offset(cfg, factor_idx)] = 1;
    }
}

static int lattice_exit_slot_feasible(const Drive *env, const Agent *agent, int exit_lane, float distance_m) {
    float max_curvature = env->lattice_lanes[exit_lane].max_curvature;
    if (!(max_curvature > LATTICE_GEOMETRY_EPS)) {
        return 1;
    }
    float exit_speed = sqrtf(ACCEL_LAT_LIMIT[1] * LATTICE_ENVELOPE_MARGIN / max_curvature);
    float braking_m = fmaxf(0.0f, distance_m - 0.5f * agent->sim_length);
    return fabsf(agent->sim_speed_signed) <= sqrtf(exit_speed * exit_speed + 2.0f * LATTICE_ENVELOPE_BRAKE_MPS2 * braking_m);
}

// all per-factor masks for one decision, in one context
static void compute_lattice_masks(Drive *env, int active_idx, int now_step) {
    struct LatticeAgent *lattice_agent = &env->lattice_agents[active_idx];
    Agent *agent = &env->agents[env->active_agent_indices[active_idx]];
    const struct LatticeConfig *cfg = &env->lattice;
    unsigned char *mask = lattice_agent->mask;
    lattice_mask_index0_only(lattice_agent, cfg);
    LatticeContext ctx = lattice_make_context(env, lattice_agent, agent, now_step);
    int lat_gate = lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_GATE);
    int lat_cells = lattice_mask_offset(cfg, LATTICE_FACTOR_LAT_CELL);
    int lon_gate = lattice_mask_offset(cfg, LATTICE_FACTOR_LON_GATE);
    int lon_cells = lattice_mask_offset(cfg, LATTICE_FACTOR_LON_CELL);
    int exit_offset = lattice_mask_offset(cfg, LATTICE_FACTOR_EXIT);
    LatticeLonSample emergency_profile[LATTICE_MAX_PLAN_STEPS + 1];
    if (lattice_agent->lon.kind == LATTICE_LON_KIND_EMERGENCY) {
        lattice_emergency_profile(env, agent, lattice_agent->gear, lattice_agent->sigma_m, (int) lroundf(LATTICE_STOP_T_MAX_S / env->dt), emergency_profile);
    }
    LatticeCoverCache cover_cache = {0};
    mask[lat_cells] = 0;
    mask[lon_cells] = 0;
    int any_lat = 0;
    int exit_live = lattice_agent->live_split_slot >= 0;
    float reverse_distance_m = 0.0f;
    if (ctx.reversing) {
        LatticeLongState now_long = {agent->sim_speed_signed, agent->accel_long};
        reverse_distance_m = lattice_agent->lon.kind == LATTICE_LON_KIND_EMERGENCY ? lattice_emergency_stop_distance(env, agent, now_long, -1)
                                                                                  : ctx.backup_remaining_m;
        reverse_distance_m = fmaxf(reverse_distance_m, LATTICE_CHECK_EPS);
    }
    int must_return = 0;
    if (lattice_agent->has_reference) {
        int forward = !ctx.reversing && lattice_agent->gear > 0;
        int plan_in_oncoming = lattice_plan_in_oncoming(env, lattice_agent, &ctx.frenet);
        float speed_mps = fabsf(agent->sim_speed);
        float window_m = plan_in_oncoming ? fmaxf(LATTICE_ONCOMING_KEEP_MIN_M, LATTICE_ONCOMING_KEEP_S * speed_mps)
                                          : fmaxf(LATTICE_ONCOMING_START_MIN_M, LATTICE_ONCOMING_START_S * speed_mps);
        int oncoming_allowed = cfg->oncoming_overtake && forward && lattice_oncoming_clear(env, lattice_agent, &ctx.frenet, window_m);
        must_return = plan_in_oncoming && forward && !oncoming_allowed;
        for (int cell = 0; cell < cfg->lat_cell_count; cell++) {
            if (lattice_lat_lane_side(cfg, cell) != 0 || (lattice_lat_is_oncoming(cfg, cell) && !oncoming_allowed)) {
                continue;
            }
            struct LatticeLatPlan candidate;
            if (!lattice_build_lat_cell(env, lattice_agent, agent, &ctx, cell, ctx.frenet.s, 0.0f, reverse_distance_m, &candidate)) {
                continue;
            }
            int ok = lattice_lat_cell_ok(env, lattice_agent, agent, &ctx, &lattice_agent->rail, ctx.frenet.s, &candidate, NULL, emergency_profile, &cover_cache);
            mask[lat_cells + cell] = (unsigned char) ok;
            any_lat |= ok;
        }
        int lane_change_allowed = !exit_live && forward && !lattice_car_on_connector(env, lattice_agent) && !plan_in_oncoming
            && !lattice_is_borrowing(env, lattice_agent, &ctx.frenet);
        for (int side_idx = 0; side_idx < 2 && lane_change_allowed; side_idx++) {
            int neighbour_lane = lattice_agent->neighbour_lane[side_idx];
            if (neighbour_lane < 0) {
                continue;
            }
            struct LatticeRail *scratch_rail = &lattice_agent->neighbour_check_rail[side_idx];
            int neighbour_hint = lattice_agent->neighbour_check_hint[side_idx];
            float needed_ahead_m = fabsf(agent->sim_speed) * LATTICE_KEEP_CHECK_S + LATTICE_CHECK_RAIL_PAD_M - LATTICE_CHECK_RAIL_REUSE_SLACK_M;
            int reuse = lattice_agent->neighbour_check_lane[side_idx] == neighbour_lane && scratch_rail->sample_count >= 2;
            LatticeFrenet on_neighbour = {0};
            if (reuse) {
                on_neighbour = lattice_project(scratch_rail, agent->sim_x, agent->sim_y, neighbour_hint);
                reuse = on_neighbour.s - scratch_rail->s_start_m >= LATTICE_SMOOTH_HALF_WINDOW * LATTICE_RAIL_SPACING_M
                    && lattice_rail_end_s(scratch_rail) - on_neighbour.s >= fminf(needed_ahead_m, lattice_rail_end_s(scratch_rail) - scratch_rail->s_start_m - LATTICE_CHECK_RAIL_BEHIND_M)
                    && fabsf(on_neighbour.d) < LATTICE_NEIGHBOUR_MAX_M + LATTICE_DRIFT_MARGIN_M;
            }
            if (!reuse) {
                neighbour_hint = build_lattice_check_rail(env, scratch_rail, agent, neighbour_lane, lattice_agent->neighbour_arc_m[side_idx]);
                lattice_agent->neighbour_check_lane[side_idx] = neighbour_hint >= 0 ? neighbour_lane : -1;
                if (neighbour_hint < 0) {
                    continue;
                }
                on_neighbour = lattice_project(scratch_rail, agent->sim_x, agent->sim_y, neighbour_hint);
            }
            lattice_agent->neighbour_check_hint[side_idx] = on_neighbour.sample_idx;
            int wanted_side = side_idx == 1 ? 1 : -1;
            for (int cell = 0; cell < cfg->lat_cell_count; cell++) {
                if (lattice_lat_lane_side(cfg, cell) != wanted_side) {
                    continue;
                }
                struct LatticeLatPlan candidate;
                if (!lattice_build_lat_cell(env, lattice_agent, agent, &ctx, cell, on_neighbour.s, lattice_agent->neighbour_offset_m[side_idx], 0.0f, &candidate)) {
                    continue;
                }
                int ok = lattice_lat_cell_ok(env, lattice_agent, agent, &ctx, scratch_rail, on_neighbour.s, &candidate, NULL, emergency_profile, &cover_cache);
                mask[lat_cells + cell] = (unsigned char) ok;
                any_lat |= ok;
            }
        }
    }
    mask[lat_gate + LATTICE_GATE_KEEP] = (unsigned char) !(must_return && any_lat);
    mask[lat_gate + LATTICE_GATE_NEW] = (unsigned char) any_lat;
    if (!any_lat) {
        mask[lat_cells] = 1;
    }
    int any_lon = 0;
    lattice_agent->stop_line_distance_m = lattice_next_stop_line(env, lattice_agent, agent, ctx.frenet.s).distance_m;
    int forward_allowed = !ctx.reversing && (lattice_agent->gear > 0 || ctx.stopped_exactly);
    for (int cell = 0; cell < cfg->lon_cell_count; cell++) {
        if (cell == cfg->lon_turn_cell) {
            continue;
        }
        int is_backup = cell >= cfg->lon_backup_cell_base;
        int is_emergency = cell == cfg->lon_emergency_cell;
        if (!is_emergency && !is_backup && !forward_allowed) {
            continue;
        }
        if (is_backup && !lattice_backup_allowed(env, lattice_agent, agent, &ctx, cfg->backup_distances_m[cell - cfg->lon_backup_cell_base])) {
            continue;
        }
        struct LatticeLonPlan candidate = {0};
        int ok = lattice_build_lon_cell(env, lattice_agent, agent, &ctx, cell, NULL, &candidate);
        mask[lon_cells + cell] = (unsigned char) ok;
        lattice_agent->lon_cell_steps[cell] = (short) (candidate.end_step - candidate.start_step);
        any_lon |= ok;
    }
    if (cfg->turnaround && lattice_turnaround_offered(env, lattice_agent, agent, &ctx)) {
        mask[lon_cells + cfg->lon_turn_cell] = 1;
        any_lon = 1;
    }
    mask[lon_gate + LATTICE_GATE_KEEP] = (unsigned char) lattice_lon_keep_ok(env, lattice_agent, agent, &ctx);
    mask[lon_gate + LATTICE_GATE_NEW] = (unsigned char) any_lon;
    if (!any_lon) {
        mask[lon_cells] = 1;
    }
    if (exit_live && cfg->exit_mode == LATTICE_EXIT_MODE_POLICY) {
        const struct LatticeRail *rail = &lattice_agent->rail;
        int split_lane = rail->lanes[lattice_agent->live_split_slot];
        const struct LatticeLaneInfo *info = &env->lattice_lanes[split_lane];
        float distance_m = rail->lane_end_s_m[lattice_agent->live_split_slot] - ctx.frenet.s;
        for (int slot_idx = 1; slot_idx < info->exit_count; slot_idx++) {
            mask[exit_offset + slot_idx] = (unsigned char) lattice_exit_slot_feasible(env, agent, info->exit_slots[slot_idx], distance_m);
        }
    }
}

// ========================================
// Context stage (end of step, before observations) and decision decode (move stage)
// ========================================

static int lattice_is_context_step(const Drive *env) {
    return (env->timestep - env->episode_start_step) % env->lattice.decision_period_steps == 0;
}

static int lattice_is_decision_step(const Drive *env) {
    int period = env->lattice.decision_period_steps;
    return (env->timestep - env->episode_start_step) % period == 1 % period;
}

// lost / drift / no-lane recovery, exit bookkeeping and the masks for the next decision
static void run_lattice_context(Drive *env, int active_idx) {
    struct LatticeAgent *lattice_agent = &env->lattice_agents[active_idx];
    Agent *agent = &env->agents[env->active_agent_indices[active_idx]];
    const struct LatticeConfig *cfg = &env->lattice;
    int now_step = env->timestep + 1;
    lattice_agent->context_step = env->timestep;
    lattice_agent->live_split_slot = -1;
    if (lattice_agent->turn.active && agent->stopped) {
        finish_lattice_turn(env, lattice_agent, agent, now_step, 0);
    }
    if (!lattice_is_policy_agent(agent) || agent->stopped || lattice_agent->rail.sample_count < 2) {
        lattice_mask_index0_only(lattice_agent, cfg);
        return;
    }
    if (lattice_agent->turn.active) {
        lattice_mask_index0_only(lattice_agent, cfg);
        return;
    }
    if (!lattice_agent->has_reference) {
        rebuild_lattice_reference(env, lattice_agent, agent, now_step);
    }
    LatticeFrenet frenet = lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint);
    if (lattice_agent->has_reference && (fabsf(frenet.d) > LATTICE_LOST_OFFSET_M || cosf(frenet.heading_error) < LATTICE_LOST_COS)) {
        lattice_agent->counters.lost += 1.0f;
        if (lattice_agent->gear < 0) {
            set_lattice_lon_emergency(&lattice_agent->lon, now_step, cfg->lon_emergency_cell);
        }
        rebuild_lattice_reference(env, lattice_agent, agent, now_step);
        frenet = lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint);
    }
    lattice_neighbours_at(env, lattice_agent);
    int drift_allowed = lattice_agent->has_reference && lattice_agent->gear > 0 && !lattice_agent->lane_change_active
        && !lattice_car_on_connector(env, lattice_agent);
    int drift_side = frenet.d > 0.0f ? 1 : 0;
    if (drift_allowed && lattice_agent->neighbour_lane[drift_side] >= 0
        && fabsf(frenet.d - lattice_agent->neighbour_offset_m[drift_side]) + LATTICE_DRIFT_MARGIN_M <= fabsf(frenet.d)) {
        switch_lattice_rail_drift(env, lattice_agent, agent, drift_side, now_step);
        frenet = lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint);
        lattice_neighbours_at(env, lattice_agent);
    }
    if (lattice_agent->has_reference && lattice_agent->gear > 0 && !lattice_is_reversing(lattice_agent, agent)) {
        int split_slot = lattice_next_split(env, lattice_agent, frenet.s, LATTICE_EXIT_FREEZE_M);
        if (split_slot >= 0 && cfg->exit_mode == LATTICE_EXIT_MODE_GOAL) {
            apply_lattice_exit(env, lattice_agent, agent, split_slot, lattice_goal_exit_slot(env, agent, lattice_agent->rail.lanes[split_slot]));
        } else {
            lattice_agent->live_split_slot = split_slot;
        }
    }
    if (lattice_agent->live_split_slot < 0 && lattice_next_split(env, lattice_agent, frenet.s, LATTICE_EXIT_FREEZE_M) < 0) {
        lattice_agent->late_exit_pending = 0;
    }
    compute_lattice_masks(env, active_idx, now_step);
    struct LatticeTurnState *turn = &lattice_agent->turn;
    int probe_due = now_step - turn->probe_step >= LATTICE_TURN_PROBE_STEPS * cfg->decision_period_steps;
    if (cfg->turnaround && !lattice_is_stopped_exactly(agent) && probe_due) {
        turn->probe_step = now_step;
        frenet = lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint);
        turn->feasible = lattice_turn_route_gap(env, lattice_agent, agent, &frenet) < 0.0f
            && lattice_turnaround_possible(env, lattice_agent, agent, &frenet);
    }
}

static int lattice_mask_allows(const struct LatticeAgent *lattice_agent, const struct LatticeConfig *cfg, int factor_idx, int value) {
    return value >= 0 && value < cfg->nvec[factor_idx] && lattice_agent->mask[lattice_mask_offset(cfg, factor_idx) + value];
}

// candidate lateral cell re-checked against the new longitudinal plan; falls back to longer durations of the same choice
static int lattice_lateral_with_coupling(Drive *env, struct LatticeAgent *lattice_agent, const Agent *agent, const LatticeContext *ctx, int cell, const struct LatticeRail *rail, float s_on_rail, float d_shift, float reverse_distance_m, const struct LatticeLonPlan *new_lon, struct LatticeLatPlan *plan) {
    const struct LatticeConfig *cfg = &env->lattice;
    LatticeLonSample emergency_profile[LATTICE_MAX_PLAN_STEPS + 1];
    const struct LatticeLonPlan *held_lon = new_lon != NULL ? new_lon : &lattice_agent->lon;
    if (held_lon->kind == LATTICE_LON_KIND_EMERGENCY) {
        lattice_emergency_profile(env, agent, lattice_agent->gear, lattice_agent->sigma_m, (int) lroundf(LATTICE_STOP_T_MAX_S / env->dt), emergency_profile);
    }
    int choice_count = lattice_lat_choice_count(cfg);
    int first_duration = lattice_lat_cell_duration_idx(cfg, cell);
    for (int duration_idx = first_duration; duration_idx < cfg->lat_duration_count; duration_idx++) {
        int candidate_cell = duration_idx * choice_count + cell % choice_count;
        if (!lattice_build_lat_cell(env, lattice_agent, agent, ctx, candidate_cell, s_on_rail, d_shift, reverse_distance_m, plan)) {
            continue;
        }
        if (new_lon == NULL || lattice_lat_cell_ok(env, lattice_agent, agent, ctx, rail, s_on_rail, plan, new_lon, emergency_profile, NULL)) {
            return 1;
        }
        if (reverse_distance_m > 0.0f) {
            return 0;
        }
    }
    return 0;
}

static void apply_lattice_gear_change(struct LatticeAgent *lattice_agent, const LatticeContext *ctx, int new_gear) {
    if (new_gear == lattice_agent->gear) {
        return;
    }
    lattice_agent->gear = new_gear;
    lattice_agent->lane_change_active = 0;
    struct LatticeLatPlan *plan = &lattice_agent->lat;
    int plan_dir = plan->mode == LATTICE_LAT_MODE_DIST ? plan->dir : 1;
    if (plan_dir != new_gear) {
        set_lattice_lat_hold(plan, (float) ctx->lat_start.value, ctx->frenet.s, new_gear);
    }
}

static void apply_lattice_action(Drive *env, int active_idx, Agent *agent, int now_step) {
    struct LatticeAgent *lattice_agent = &env->lattice_agents[active_idx];
    const struct LatticeConfig *cfg = &env->lattice;
    const int *action = (const int *) env->actions + active_idx * LATTICE_ACTION_FACTORS;
    assert(lattice_agent->context_step == env->timestep - 1);
    int lat_gate = action[LATTICE_FACTOR_LAT_GATE];
    int lat_cell = action[LATTICE_FACTOR_LAT_CELL];
    int lon_gate = action[LATTICE_FACTOR_LON_GATE];
    int lon_cell = action[LATTICE_FACTOR_LON_CELL];
    int exit_slot = action[LATTICE_FACTOR_EXIT];
    int invalid = 0;
    if (!lattice_mask_allows(lattice_agent, cfg, LATTICE_FACTOR_LAT_GATE, lat_gate)
        || (lat_gate == LATTICE_GATE_NEW && !lattice_mask_allows(lattice_agent, cfg, LATTICE_FACTOR_LAT_CELL, lat_cell))) {
        invalid = 1;
        lat_gate = LATTICE_GATE_KEEP;
    }
    if (!lattice_mask_allows(lattice_agent, cfg, LATTICE_FACTOR_LON_GATE, lon_gate)
        || (lon_gate == LATTICE_GATE_NEW && !lattice_mask_allows(lattice_agent, cfg, LATTICE_FACTOR_LON_CELL, lon_cell))) {
        invalid = 1;
        int keep_valid = lattice_agent->mask[lattice_mask_offset(cfg, LATTICE_FACTOR_LON_GATE) + LATTICE_GATE_KEEP];
        lon_gate = keep_valid ? LATTICE_GATE_KEEP : LATTICE_GATE_NEW;
        lon_cell = cfg->lon_emergency_cell;
    }
    if (!lattice_mask_allows(lattice_agent, cfg, LATTICE_FACTOR_EXIT, exit_slot)) {
        invalid |= exit_slot != 0;
        exit_slot = 0;
    }
    lattice_agent->counters.invalid_actions += invalid;
    lattice_agent->counters.decisions += 1.0f;
    if (!lattice_is_policy_agent(agent) || agent->stopped || lattice_agent->rail.sample_count < 2) {
        return;
    }
    if (lattice_agent->turn.active) {
        return;
    }
    if (lattice_agent->live_split_slot >= 0) {
        int split_slot = lattice_agent->live_split_slot;
        float split_distance_m = lattice_agent->rail.lane_end_s_m[split_slot] - lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint).s;
        lattice_agent->counters.late_exit_decisions += lattice_agent->late_exit_pending && split_distance_m < LATTICE_EXIT_FREEZE_M - LATTICE_LOOKAHEAD_FREEZE_PAD_M;
        apply_lattice_exit(env, lattice_agent, agent, split_slot, exit_slot);
        lattice_agent->live_split_slot = -1;
    }
    LatticeContext ctx = lattice_make_context(env, lattice_agent, agent, now_step);
    const struct LatticeLonPlan *new_lon = NULL;
    if (lon_gate == LATTICE_GATE_NEW && lon_cell == cfg->lon_turn_cell) {
        start_lattice_turn(env, lattice_agent, agent, now_step);
        return;
    }
    if (lon_gate == LATTICE_GATE_NEW) {
        struct LatticeLonPlan candidate = {0};
        lattice_build_lon_cell(env, lattice_agent, agent, &ctx, lon_cell, NULL, &candidate);
        struct LatticeLonPlan *committed = &lattice_agent->lon;
        int same_plan = candidate.kind == committed->kind && candidate.cell == committed->cell && candidate.end_step == committed->end_step
            && candidate.kind != LATTICE_LON_KIND_EMERGENCY;
        if (!same_plan) {
            int new_gear = candidate.kind == LATTICE_LON_KIND_BACKUP ? -1 : (candidate.kind == LATTICE_LON_KIND_EMERGENCY ? lattice_agent->gear : 1);
            *committed = candidate;
            new_lon = committed;
            lattice_agent->counters.lon_new += 1.0f;
            lattice_agent->counters.backups += candidate.kind == LATTICE_LON_KIND_BACKUP;
            apply_lattice_gear_change(lattice_agent, &ctx, new_gear);
            if (lattice_agent->gear > 0 && lattice_lon_is_stopping(committed)) {
                LatticeLongState now_long = {agent->sim_speed_signed, agent->accel_long};
                float stopping_m = committed->kind == LATTICE_LON_KIND_EMERGENCY ? lattice_emergency_stop_distance(env, agent, now_long, 1)
                                                                                 : (float) (committed->target_sigma_m - lattice_agent->sigma_m);
                reindex_lattice_lateral_for_stop(env, lattice_agent, agent, &ctx.frenet, now_step, stopping_m);
            }
            ctx = lattice_make_context(env, lattice_agent, agent, now_step);
        }
    }
    if (lat_gate != LATTICE_GATE_NEW || !lattice_agent->has_reference) {
        return;
    }
    int lane_side = lattice_lat_lane_side(cfg, lat_cell);
    float reverse_distance_m = 0.0f;
    if (lattice_agent->lon.kind == LATTICE_LON_KIND_BACKUP && lattice_agent->gear < 0 && new_lon != NULL) {
        reverse_distance_m = (float) fabs(lattice_agent->lon.target_sigma_m - lattice_agent->sigma_m);
    } else if (ctx.reversing) {
        LatticeLongState now_long = {agent->sim_speed_signed, agent->accel_long};
        reverse_distance_m = fmaxf(lattice_agent->lon.kind == LATTICE_LON_KIND_EMERGENCY ? lattice_emergency_stop_distance(env, agent, now_long, -1) : ctx.backup_remaining_m, LATTICE_CHECK_EPS);
    }
    struct LatticeLatPlan candidate;
    if (lane_side != 0) {
        int side_idx = lane_side > 0 ? 1 : 0;
        struct LatticeRail *scratch_rail = env->lattice_scratch_rail;
        int neighbour_hint = build_lattice_fresh_rail(env, scratch_rail, agent, lattice_agent->neighbour_lane[side_idx], lattice_agent->neighbour_arc_m[side_idx]);
        if (neighbour_hint < 0) {
            lattice_agent->counters.decode_rejects += 1.0f;
            return;
        }
        LatticeFrenet on_neighbour = lattice_project(scratch_rail, agent->sim_x, agent->sim_y, neighbour_hint);
        if (!lattice_lateral_with_coupling(env, lattice_agent, agent, &ctx, lat_cell, scratch_rail, on_neighbour.s, lattice_agent->neighbour_offset_m[side_idx], 0.0f, new_lon, &candidate)) {
            lattice_agent->counters.decode_rejects += 1.0f;
            return;
        }
        lattice_agent->rail = *scratch_rail;
        lattice_agent->projection_hint = on_neighbour.sample_idx;
        lattice_agent->lat = candidate;
        lattice_agent->lane_change_active = 1;
        mark_lattice_rail_changed(lattice_agent);
        lattice_agent->counters.chosen_changes += 1.0f;
        lattice_agent->counters.lat_new += 1.0f;
        return;
    }
    if (!lattice_lateral_with_coupling(env, lattice_agent, agent, &ctx, lat_cell, &lattice_agent->rail, ctx.frenet.s, 0.0f, reverse_distance_m, new_lon, &candidate)) {
        lattice_agent->counters.decode_rejects += 1.0f;
        return;
    }
    struct LatticeLatPlan *committed = &lattice_agent->lat;
    int same_plan = candidate.mode == LATTICE_LAT_MODE_TIME && committed->mode == LATTICE_LAT_MODE_TIME && candidate.target_d_m == committed->target_d_m
        && candidate.end_step == committed->end_step;
    if (!same_plan) {
        lattice_agent->counters.oncoming_starts += lattice_lat_is_oncoming(cfg, lat_cell) && !lattice_plan_in_oncoming(env, lattice_agent, &ctx.frenet);
        *committed = candidate;
        lattice_agent->counters.lat_new += 1.0f;
    }
}

typedef struct {
    float x[LATTICE_PREVIEW_SUBSTEPS + 1];
    float y[LATTICE_PREVIEW_SUBSTEPS + 1];
    float speed[LATTICE_PREVIEW_SUBSTEPS + 1];
} LatticePreview;

static LatticePreview compute_lattice_preview(Drive *env, const struct LatticeAgent *lattice_agent, const Agent *agent, int now_step);

// RMS world distance between the path observed before this decision and the newly committed one
static float lattice_plan_change_rms_m(Drive *env, const struct LatticeAgent *lattice_agent, const Agent *agent, int now_step) {
    LatticePreview preview = compute_lattice_preview(env, lattice_agent, agent, now_step);
    float sum_m2 = 0.0f;
    for (int sample_idx = 1; sample_idx <= LATTICE_CONSISTENCY_SAMPLES; sample_idx++) {
        int substep_idx = sample_idx * LATTICE_PREVIEW_RENDER_STRIDE;
        float dx = preview.x[substep_idx] - lattice_agent->preview_world_xy[sample_idx][0];
        float dy = preview.y[substep_idx] - lattice_agent->preview_world_xy[sample_idx][1];
        sum_m2 += dx * dx + dy * dy;
    }
    return sqrtf(sum_m2 / LATTICE_CONSISTENCY_SAMPLES);
}

// move stage: decode on decision steps, automatic plan updates, then the tracking command
static LatticeCommand move_lattice_agent(Drive *env, int active_idx, Agent *agent) {
    struct LatticeAgent *lattice_agent = &env->lattice_agents[active_idx];
    int now_step = env->timestep;
    LatticeCommand command = {0};
    if (lattice_agent->rail.sample_count < 2) {
        return command;
    }
    if (lattice_is_decision_step(env)) {
        lattice_agent->rail_changed_flag = 0;
        float plans_committed = lattice_agent->counters.lat_new + lattice_agent->counters.lon_new;
        apply_lattice_action(env, active_idx, agent, now_step);
        if (lattice_agent->counters.lat_new + lattice_agent->counters.lon_new > plans_committed) {
            lattice_agent->plan_change_rms_m = lattice_plan_change_rms_m(env, lattice_agent, agent, now_step);
            lattice_agent->counters.plan_change_rms_m += lattice_agent->plan_change_rms_m;
        }
    }
    if (lattice_agent->turn.active) {
        advance_lattice_turn(env, lattice_agent, agent, now_step);
    }
    LatticeFrenet frenet = lattice_frenet_state(&lattice_agent->rail, agent, lattice_agent->projection_hint);
    lattice_agent->projection_hint = frenet.sample_idx;
    lattice_agent->turn.landing &= lattice_agent->has_reference && fabsf(frenet.d) > LATTICE_TURN_LANDED_D_M;
    update_lattice_plans_automatic(env, lattice_agent, agent, &frenet, now_step);
    command = compute_lattice_command(env, lattice_agent, agent, &frenet, now_step);
    struct LatticeCounters *counters = &lattice_agent->counters;
    counters->steps += 1.0f;
    counters->tracking_error_m += fabsf(command.lat_error_m);
    counters->speed_error_mps += fabsf(command.speed_error_mps);
    counters->no_reference_steps += !lattice_agent->has_reference && !lattice_agent->turn.active;
    counters->turn_steps += lattice_agent->turn.active;
    counters->dist_mode_steps += lattice_agent->lat.mode == LATTICE_LAT_MODE_DIST;
    counters->oncoming_steps += lattice_is_borrowing(env, lattice_agent, &frenet);
    counters->emergency_steps += lattice_agent->lon.kind == LATTICE_LON_KIND_EMERGENCY;
    counters->unfollowable_steps += fabsf(frenet.curvature) > lattice_curvature_limit(agent);
    counters->jerk_clip_long += command.jerk_long < JERK_LONG[0] || command.jerk_long > JERK_LONG[3];
    counters->jerk_clip_lat += command.jerk_lat < JERK_LAT[0] || command.jerk_lat > JERK_LAT[2];
    return command;
}

// distance driven, from the integrator's own trapezoid (drive.h jerk branch)
static void update_lattice_odometer(Drive *env, int active_idx, float speed_before, float speed_after, int steer_rate_saturated) {
    struct LatticeAgent *lattice_agent = &env->lattice_agents[active_idx];
    struct LatticeCounters *counters = &lattice_agent->counters;
    float step_distance_m = 0.5f * (speed_after + speed_before) * env->dt;
    lattice_agent->sigma_m += step_distance_m;
    counters->steer_rate_saturated += steer_rate_saturated;
    if (fabsf(speed_after) > LATTICE_RELEASE_SPEED_MPS) {
        counters->moving_steps += 1.0f;
        if (counters->first_motion_step < 0.0f) {
            counters->first_motion_step = (float) (env->timestep - env->episode_start_step);
        }
    }
    if (step_distance_m < 0.0f) {
        counters->backup_m -= step_distance_m;
    }
}

// ========================================
// Preview of the committed plans (observation + rendered predicted path) and observations
// ========================================

static LatticePreview compute_lattice_preview(Drive *env, const struct LatticeAgent *lattice_agent, const Agent *agent, int now_step) {
    LatticePreview preview;
    const struct LatticeRail *rail = &lattice_agent->rail;
    LatticeFrenet frenet = lattice_frenet_state(rail, agent, lattice_agent->projection_hint);
    int emergency = lattice_agent->lon.kind == LATTICE_LON_KIND_EMERGENCY;
    int steps_per_substep = (int) lroundf(LATTICE_PREVIEW_SUBSTEP_S / env->dt);
    steps_per_substep = steps_per_substep < 1 ? 1 : steps_per_substep;
    float substep_s = steps_per_substep * env->dt;
    LatticeLongState emergency_state = {agent->sim_speed_signed, agent->accel_long};
    float speed_cap = lattice_speed_cap_mps(env, agent);
    double sigma_now = emergency ? lattice_agent->sigma_m : lattice_lon_eval(&lattice_agent->lon, lattice_elapsed_s(env, lattice_agent->lon.start_step, now_step)).value;
    double sigma_prev = sigma_now;
    float s_m = frenet.s;
    float speed_prev = agent->sim_speed_signed;
    float factor_prev = 1.0f;
    float d_prime_prev = 0.0f;
    for (int substep_idx = 0; substep_idx <= LATTICE_PREVIEW_SUBSTEPS; substep_idx++) {
        double sigma_m;
        float speed;
        if (emergency) {
            sigma_m = sigma_prev;
            speed = emergency_state.speed;
            if (substep_idx > 0) {
                for (int step_idx = 0; step_idx < steps_per_substep; step_idx++) {
                    float jerk = lattice_emergency_jerk(env, agent, emergency_state, lattice_agent->gear, NULL);
                    LatticeLongState next = lattice_integrate_long(emergency_state, jerk, agent, env->dt, speed_cap);
                    sigma_m += 0.5 * (next.speed + emergency_state.speed) * env->dt;
                    emergency_state = next;
                }
                speed = emergency_state.speed;
            }
        } else {
            LatticePlanPoint lon = lattice_lon_eval(&lattice_agent->lon, lattice_elapsed_s(env, lattice_agent->lon.start_step, now_step) + substep_idx * substep_s);
            sigma_m = lon.value;
            speed = (float) lon.first;
        }
        if (substep_idx > 0) {
            float rail_ratio = 1.0f / sqrtf(factor_prev * factor_prev + d_prime_prev * d_prime_prev);
            s_m += (float) (sigma_m - sigma_prev) * rail_ratio;
        }
        LatticeRailPoint point = lattice_rail_at(rail, s_m);
        float d_m;
        float d_prime;
        if (lattice_agent->lat.mode == LATTICE_LAT_MODE_TIME) {
            LatticePlanPoint lat = lattice_lat_eval_time(&lattice_agent->lat, lattice_elapsed_s(env, lattice_agent->lat.start_step, now_step) + substep_idx * substep_s);
            d_m = (float) lat.value;
            float s_dot = fmaxf(fabsf(speed), LATTICE_MIN_RATE_MPS);
            d_prime = (float) lat.first / s_dot;
        } else {
            LatticePlanPoint lat = lattice_lat_eval_dist(&lattice_agent->lat, s_m);
            d_m = (float) lat.value;
            d_prime = (float) lat.first;
        }
        float factor = fmaxf(1.0f - point.curvature * d_m, LATTICE_MIN_FRENET_FACTOR);
        factor_prev = factor;
        d_prime_prev = d_prime;
        preview.x[substep_idx] = point.x - d_m * sinf(point.heading);
        preview.y[substep_idx] = point.y + d_m * cosf(point.heading);
        preview.speed[substep_idx] = speed;
        sigma_prev = sigma_m;
        speed_prev = speed;
    }
    (void) speed_prev;
    return preview;
}

static void lattice_store_render_path(struct LatticeAgent *lattice_agent, const LatticePreview *preview) {
    for (int sample_idx = 0; sample_idx < AGENT_F32_PATH_SAMPLES; sample_idx++) {
        int substep_idx = sample_idx * LATTICE_PREVIEW_RENDER_STRIDE;
        substep_idx = substep_idx > LATTICE_PREVIEW_SUBSTEPS ? LATTICE_PREVIEW_SUBSTEPS : substep_idx;
        lattice_agent->preview_world_xy[sample_idx][0] = preview->x[substep_idx];
        lattice_agent->preview_world_xy[sample_idx][1] = preview->y[substep_idx];
    }
}

static float lattice_plan_curvature_ahead(const struct LatticeRail *rail, float s_m, float d_m) {
    LatticeRailPoint point = lattice_rail_at(rail, s_m);
    return point.curvature / fmaxf(1.0f - point.curvature * d_m, LATTICE_MIN_FRENET_FACTOR);
}

// plan block (lattice_plan_feature_count floats) written right after the ego block
static int write_lattice_plan_obs(Drive *env, int active_idx, float *obs, int obs_idx) {
    int start_idx = obs_idx;
    int feature_count = lattice_plan_feature_count(&env->lattice);
    struct LatticeAgent *lattice_agent = &env->lattice_agents[active_idx];
    Agent *agent = &env->agents[env->active_agent_indices[active_idx]];
    if (!lattice_is_policy_agent(agent) || lattice_agent->rail.sample_count < 2) {
        return start_idx + feature_count;
    }
    const struct LatticeRail *rail = &lattice_agent->rail;
    const struct LatticeLatPlan *lat = &lattice_agent->lat;
    const struct LatticeLonPlan *lon = &lattice_agent->lon;
    int now_step = env->timestep + 1;
    LatticeFrenet frenet = lattice_frenet_state(rail, agent, lattice_agent->projection_hint);
    LatticePlanPoint lat_now = lattice_lat_state(env, lat, now_step, frenet.s, frenet.s_dot, agent->accel_long);
    float lat_remaining = lat->mode == LATTICE_LAT_MODE_TIME ? fmaxf(0.0f, (lat->end_step - now_step) * env->dt) / LATTICE_OBS_TIME_NORM_S
                                                             : fmaxf(0.0f, lat->horizon - lat->dir * (frenet.s - lat->start_s_m)) / LATTICE_OBS_DIST_NORM_M;
    obs[obs_idx++] = (lat->target_d_m - frenet.d) / LANE_WIDTH;
    obs[obs_idx++] = lat_remaining;
    obs[obs_idx++] = (float) (lat->mode == LATTICE_LAT_MODE_DIST);
    obs[obs_idx++] = frenet.d / LANE_WIDTH;
    obs[obs_idx++] = sinf(frenet.heading_error);
    obs[obs_idx++] = ((float) lat_now.value - frenet.d) / LANE_WIDTH;
    int emergency = lon->kind == LATTICE_LON_KIND_EMERGENCY;
    LatticePlanPoint lon_now = emergency ? (LatticePlanPoint) {lattice_agent->sigma_m, 0.0, 0.0} : lattice_lon_eval(lon, lattice_elapsed_s(env, lon->start_step, now_step));
    int stop_kind = lon->kind == LATTICE_LON_KIND_STOP || lon->kind == LATTICE_LON_KIND_STOP_LINE || lon->kind == LATTICE_LON_KIND_BACKUP;
    obs[obs_idx++] = (lon->kind == LATTICE_LON_KIND_SPEED ? lon->target_speed_mps : 0.0f) / LATTICE_OBS_SPEED_NORM_MPS;
    obs[obs_idx++] = fmaxf(0.0f, (lon->end_step - now_step) * env->dt) / LATTICE_OBS_TIME_NORM_S;
    obs[obs_idx++] = (float) (lon->kind == LATTICE_LON_KIND_STOP || lon->kind == LATTICE_LON_KIND_STOP_LINE);
    obs[obs_idx++] = (float) emergency;
    obs[obs_idx++] = stop_kind ? (float) (lon->target_sigma_m - lattice_agent->sigma_m) / LATTICE_OBS_DIST_NORM_M : 0.0f;
    obs[obs_idx++] = emergency ? 0.0f : ((float) lon_now.first - agent->sim_speed_signed) / LATTICE_OBS_SPEED_NORM_MPS;
    obs[obs_idx++] = (float) lattice_agent->gear;
    float plan_d = (float) lat_now.value;
    obs[obs_idx++] = LATTICE_OBS_CURVATURE_SCALE_M * lattice_plan_curvature_ahead(rail, frenet.s + LATTICE_OBS_ROAD_AHEAD_NEAR_M, plan_d);
    obs[obs_idx++] = LATTICE_OBS_CURVATURE_SCALE_M * lattice_plan_curvature_ahead(rail, frenet.s + LATTICE_OBS_ROAD_AHEAD_MID_M, plan_d);
    obs[obs_idx++] = LATTICE_OBS_CURVATURE_SCALE_M * lattice_plan_curvature_ahead(rail, frenet.s + LATTICE_OBS_ROAD_AHEAD_FAR_M, plan_d);
    float curvature_limit = lattice_curvature_limit(agent);
    float margin = curvature_limit;
    float v_env_min = 1e9f;
    for (float ahead_m = 0.0f; ahead_m <= LATTICE_OBS_ENVELOPE_WINDOW_M; ahead_m += LATTICE_RAIL_SPACING_M) {
        LatticeRailPoint point = lattice_rail_at(rail, frenet.s + ahead_m);
        if (ahead_m <= LATTICE_OBS_MARGIN_WINDOW_M) {
            margin = fminf(margin, curvature_limit - fabsf(point.curvature));
        }
        v_env_min = fminf(v_env_min, point.v_env);
    }
    obs[obs_idx++] = LATTICE_OBS_CURVATURE_SCALE_M * margin;
    obs[obs_idx++] = fminf(lattice_rail_at(rail, frenet.s).v_env, lattice_speed_cap_mps(env, agent)) / LATTICE_OBS_SPEED_NORM_MPS;
    obs[obs_idx++] = fminf(v_env_min, lattice_speed_cap_mps(env, agent)) / LATTICE_OBS_SPEED_NORM_MPS;
    int goal_lane = lattice_agent_goal_lane(agent);
    int split_slot = -1;
    int car_slot = rail->chain_slot[lattice_agent->projection_hint];
    for (int lane_slot = car_slot; lattice_agent->has_reference && lane_slot < rail->lane_count && split_slot < 0; lane_slot++) {
        if (env->lattice_lanes[rail->lanes[lane_slot]].exit_count >= 2 && rail->lane_end_s_m[lane_slot] >= frenet.s) {
            split_slot = lane_slot;
        }
    }
    float split_distance_m = split_slot >= 0 ? rail->lane_end_s_m[split_slot] - frenet.s : LATTICE_EXIT_FREEZE_M;
    obs[obs_idx++] = fminf(split_distance_m, LATTICE_EXIT_FREEZE_M) / LATTICE_EXIT_FREEZE_M;
    float committed_turn = 0.0f;
    if (split_slot >= 0 && split_slot + 1 < rail->lane_count) {
        const struct LatticeLaneInfo *info = &env->lattice_lanes[rail->lanes[split_slot]];
        for (int slot_idx = 0; slot_idx < info->exit_count; slot_idx++) {
            if (info->exit_slots[slot_idx] == rail->lanes[split_slot + 1]) {
                committed_turn = info->exit_turn_rad[slot_idx];
            }
        }
    }
    obs[obs_idx++] = committed_turn / (float) M_PI;
    obs[obs_idx++] = (float) (lattice_agent->live_split_slot >= 0);
    const struct LatticeLaneInfo *split_info = split_slot >= 0 ? &env->lattice_lanes[rail->lanes[split_slot]] : NULL;
    int exit_count = split_info != NULL ? split_info->exit_count : 0;
    float exit_goal_distance_m[LATTICE_EXIT_SLOTS];
    float best_exit_goal_distance_m = INFINITY;
    for (int slot_idx = 0; slot_idx < exit_count; slot_idx++) {
        exit_goal_distance_m[slot_idx] = lattice_goal_distance_m(env, split_info->exit_slots[slot_idx], goal_lane);
        best_exit_goal_distance_m = fminf(best_exit_goal_distance_m, exit_goal_distance_m[slot_idx]);
    }
    // per exit: extra route length over the best exit, 1 when unreachable or absent
    for (int slot_idx = 0; slot_idx < LATTICE_EXIT_SLOTS; slot_idx++) {
        int exists = slot_idx < exit_count;
        float route_gap = 1.0f;
        if (exists && isfinite(exit_goal_distance_m[slot_idx])) {
            route_gap = fminf((exit_goal_distance_m[slot_idx] - best_exit_goal_distance_m) / LATTICE_OBS_ROUTE_GAP_NORM_M, 1.0f);
        }
        obs[obs_idx++] = (float) exists;
        obs[obs_idx++] = exists ? split_info->exit_turn_rad[slot_idx] / (float) M_PI : 0.0f;
        obs[obs_idx++] = route_gap;
    }
    int car_lane = lattice_agent->has_reference
        ? rail->lanes[car_slot]
        : (lattice_agent->turn.active ? lattice_agent->turn.plan.landing_lane : -1);
    float car_goal_distance_m = lattice_goal_distance_m(env, car_lane, goal_lane);
    obs[obs_idx++] = isfinite(car_goal_distance_m)
        ? fminf(log1pf(car_goal_distance_m / LATTICE_OBS_ROUTE_LOG_SCALE_M) / log1pf(LATTICE_OBS_ROUTE_LOG_MAX_M / LATTICE_OBS_ROUTE_LOG_SCALE_M), 1.0f)
        : 1.0f;
    LatticeReportedLight light = lattice_reported_light(env, lattice_agent, agent, frenet.s);
    obs[obs_idx++] = light.distance_m >= 0.0f ? fminf(light.distance_m, LATTICE_OBS_STOP_LINE_NORM_M) / LATTICE_OBS_STOP_LINE_NORM_M : 1.0f;
    obs[obs_idx++] = (float) (light.light_state == TRAFFIC_CONTROL_STATE_RED);
    obs[obs_idx++] = (float) (light.light_state == TRAFFIC_CONTROL_STATE_YELLOW);
    obs[obs_idx++] = (float) lattice_agent->rail_changed_flag;
    LatticePreview preview = compute_lattice_preview(env, lattice_agent, agent, now_step);
    lattice_store_render_path(lattice_agent, &preview);
    for (int point_idx = 1; point_idx <= LATTICE_PREVIEW_POINTS; point_idx++) {
        int substep_idx = point_idx * LATTICE_PREVIEW_OBS_STRIDE;
        float local_x, local_y;
        project_point_to_ego_frame(agent, preview.x[substep_idx], preview.y[substep_idx], &local_x, &local_y);
        obs[obs_idx++] = local_x / LATTICE_OBS_PREVIEW_NORM_M;
        obs[obs_idx++] = local_y / LATTICE_OBS_PREVIEW_NORM_M;
        obs[obs_idx++] = preview.speed[substep_idx] / LATTICE_OBS_SPEED_NORM_MPS;
    }
    if (env->lattice.oncoming_overtake) {
        obs[obs_idx++] = (float) lattice_is_borrowing(env, lattice_agent, &frenet);
    }
    if (env->lattice.turnaround) {
        const struct LatticeTurnState *turn = &lattice_agent->turn;
        float route_gap
            = turn->active || !turn->feasible ? 1.0f : lattice_turn_route_gap(env, lattice_agent, agent, &frenet);
        obs[obs_idx++] = (float) turn->active;
        obs[obs_idx++] = turn->active ? clip(lattice_turn_remaining_rad(turn, agent) / (float) M_PI, 0.0f, 1.0f) : 0.0f;
        obs[obs_idx++] = turn->active ? 0.0f : (route_gap < 0.0f ? route_gap : 1.0f);
    }
    assert(obs_idx == start_idx + feature_count);
    return obs_idx;
}

static int write_lattice_mask_obs(Drive *env, int active_idx, float *obs, int obs_idx) {
    const struct LatticeAgent *lattice_agent = &env->lattice_agents[active_idx];
    int context_now = lattice_agent->context_step == env->timestep && lattice_is_context_step(env);
    const struct LatticeConfig *cfg = &env->lattice;
    for (int feature_idx = 0; feature_idx < cfg->mask_feature_count; feature_idx++) {
        obs[obs_idx + feature_idx] = context_now ? (float) lattice_agent->mask[feature_idx] : 0.0f;
    }
    if (!context_now) {
        for (int factor_idx = 0; factor_idx < LATTICE_ACTION_FACTORS; factor_idx++) {
            obs[obs_idx + lattice_mask_offset(cfg, factor_idx)] = 1.0f;
        }
    }
    return obs_idx + cfg->mask_feature_count;
}

// a policy car borrowing the oncoming lane is scored in its own lane's direction, centred on the borrowed lane
static void lattice_borrowed_lane(const Drive *env, int active_idx, const Agent *agent, int *lane_idx, float *signed_lane_distance_m, float *lane_heading) {
    const struct LatticeAgent *lattice_agent = &env->lattice_agents[active_idx];
    const struct LatticeRail *rail = &lattice_agent->rail;
    if (!lattice_is_policy_agent(agent) || rail->sample_count < 2) {
        return;
    }
    LatticeFrenet frenet = lattice_frenet_state(rail, agent, lattice_agent->projection_hint);
    if (!lattice_is_borrowing(env, lattice_agent, &frenet)) {
        return;
    }
    *lane_idx = rail->lanes[rail->chain_slot[frenet.sample_idx]];
    *signed_lane_distance_m = lattice_oncoming_offset_at(env, lattice_agent, frenet.sample_idx) - frenet.d; // left negative
    *lane_heading = lattice_wrap_angle(agent->sim_heading - frenet.heading_error);
}

// after a turn-around the nearest lane can be the old one, now faced against: the landing lane until the car is in it
static void lattice_landing_lane(
    const Drive *env,
    int active_idx,
    const Agent *agent,
    int *lane_idx,
    float *signed_lane_distance_m,
    float *lane_heading) {
    const struct LatticeAgent *lattice_agent = &env->lattice_agents[active_idx];
    if (!lattice_agent->turn.landing) {
        return;
    }
    const struct LatticeRail *rail = &lattice_agent->rail;
    LatticeFrenet frenet = lattice_frenet_state(rail, agent, lattice_agent->projection_hint);
    *lane_idx = rail->lanes[rail->chain_slot[frenet.sample_idx]];
    *signed_lane_distance_m = -frenet.d; // left negative
    *lane_heading = lattice_wrap_angle(agent->sim_heading - frenet.heading_error);
}

static void add_lattice_log(Drive *env, int active_idx, Log *episode_log) {
    const struct LatticeCounters *counters = &env->lattice_agents[active_idx].counters;
    float decisions = fmaxf(counters->decisions, 1.0f);
    float steps = fmaxf(counters->steps, 1.0f);
    episode_log->lattice_lat_new_rate += counters->lat_new / decisions;
    episode_log->lattice_lon_new_rate += counters->lon_new / decisions;
    episode_log->lattice_invalid_action_rate += counters->invalid_actions / decisions;
    episode_log->lattice_decode_reject_rate += counters->decode_rejects / decisions;
    episode_log->lattice_auto_replan_rate += counters->auto_replans / decisions;
    episode_log->lattice_no_reference_rate += counters->no_reference_steps / steps;
    episode_log->lattice_lost_rate += counters->lost / decisions;
    episode_log->lattice_chosen_change_rate += counters->chosen_changes / decisions;
    episode_log->lattice_drift_change_rate += counters->drift_changes / decisions;
    episode_log->lattice_tracking_error_m += counters->tracking_error_m / steps;
    episode_log->lattice_speed_error_mps += counters->speed_error_mps / steps;
    episode_log->lattice_jerk_clip_long_rate += counters->jerk_clip_long / steps;
    episode_log->lattice_jerk_clip_lat_rate += counters->jerk_clip_lat / steps;
    episode_log->lattice_steer_rate_saturation_rate += counters->steer_rate_saturated / steps;
    episode_log->lattice_dist_mode_rate += counters->dist_mode_steps / steps;
    episode_log->lattice_emergency_rate += counters->emergency_steps / steps;
    episode_log->lattice_unfollowable_rate += counters->unfollowable_steps / steps;
    episode_log->lattice_exit_decisions += counters->exit_decisions;
    episode_log->lattice_exit_nonstraight_rate += counters->exit_nonstraight / fmaxf(counters->exit_decisions, 1.0f);
    episode_log->lattice_late_exit_decisions += counters->late_exit_decisions;
    episode_log->lattice_backup_rate += counters->backups / decisions;
    episode_log->lattice_backup_m += counters->backup_m;
    episode_log->lattice_moving_fraction += counters->moving_steps / steps;
    float first_motion_steps = counters->first_motion_step >= 0.0f ? counters->first_motion_step : counters->steps;
    episode_log->lattice_time_to_first_motion_s += first_motion_steps * env->dt;
    episode_log->lattice_rail_regen_rate += counters->rail_regens / steps;
    episode_log->lattice_plan_change_rms_m += counters->plan_change_rms_m / decisions;
    episode_log->lattice_oncoming_rate += counters->oncoming_steps / steps;
    episode_log->lattice_oncoming_starts += counters->oncoming_starts;
    episode_log->lattice_turn_rate += counters->turn_steps / steps;
    episode_log->lattice_turn_starts += counters->turn_starts;
    episode_log->lattice_turn_completions += counters->turn_completions;
    episode_log->lattice_turn_aborts += counters->turn_aborts;
}

// all active slots: follow the rails every step, recompute context and masks on context steps
static void update_lattice_before_observations(Drive *env) {
    int context_step = lattice_is_context_step(env);
    for (int active_idx = 0; active_idx < env->active_agent_count; active_idx++) {
        update_lattice_chain(env, active_idx);
        if (context_step) {
            run_lattice_context(env, active_idx);
        }
    }
}

// all c_reset branches: fresh lattice state and the first context (timestep == episode start is a context step)
static void reset_lattice_state(Drive *env) {
    env->episode_start_step = env->timestep;
    for (int active_idx = 0; active_idx < env->active_agent_count; active_idx++) {
        reset_lattice_slot(env, active_idx);
    }
    update_lattice_before_observations(env);
}

#endif
