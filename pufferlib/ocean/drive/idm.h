#ifndef PUFFERLIB_OCEAN_DRIVE_IDM_H
#define PUFFERLIB_OCEAN_DRIVE_IDM_H

#define IDM_MINIMUM_LEAD_DISTANCE 0.1f
#define IDM_MIN_SPACING 2.0f
#define IDM_SAFE_TIME_HEADWAY 2.0f
#define NUM_ACCELERATION_VALUES ((int)(sizeof(ACCELERATION_VALUES) / sizeof(ACCELERATION_VALUES[0])))
#define IDM_MAX_ACCEL ACCELERATION_VALUES[NUM_ACCELERATION_VALUES - 1]
// Max braking is harmonized to 5 m/s^2 across all planners (IDM variants, PDM, jerk policy)
// so baselines share the same braking authority; decoupled from the discrete action table.
#define IDM_MAX_DECEL 5.0f
#define IDM_DELTA 4.0f
#define IDM_LOOKAHEAD_TIME 5.0f
#define IDM_MIN_LOOKAHEAD 20.0f
#define IDM_MAX_LOOKAHEAD 120.0f
#define IDM_BBOX_MARGIN 0.05f
#define IDM_COLLISION_BBOX_MARGIN 0.05f
#define IDM_DEFAULT_DESIRED_SPEED 15.0f
#define IDM_ROUTE_SAMPLE_DS 1.0f
#define IDM_MAX_CANDIDATES 256
#define IDM_LATERAL_SNAP_THRESHOLD 0.05f
#define IDM_MAX_LATERAL_STEP 0.05f
#define IDM_LATERAL_STEP_RATIO 0.2f
#define IDM_HEADING_SNAP_THRESHOLD 0.05f
#define IDM_MAX_HEADING_STEP 0.05f
#define IDM_HEADING_STEP_RATIO 0.1f
#define IDM_WALL_LEADER_IDX -2

#define NUPLAN_IDM_MIN_SPACING 1.0f
#define NUPLAN_IDM_SAFE_TIME_HEADWAY 1.5f
#define NUPLAN_IDM_MAX_ACCEL 1.0f
#define NUPLAN_IDM_MAX_DECEL 5.0f // harmonized max braking (see IDM_MAX_DECEL)
#define NUPLAN_IDM_LOOKAHEAD_TIME 5.0f
#define NUPLAN_IDM_MIN_LOOKAHEAD 20.0f
#define NUPLAN_IDM_MAX_LOOKAHEAD 40.0f
#define NUPLAN_IDM_DEFAULT_DESIRED_SPEED 10.0f
#define NUPLAN_IDM_MAX_CANDIDATES 64

typedef struct {
    int has_leader;
    int leader_agent_idx;
    int is_traffic_light;
    float gap;
    float leader_speed;
} IDMLeader;

typedef struct {
    int valid;
    int route_idx;
    int lane_idx;
    int segment_idx;
    float t;
    float dist_sq;
} IDMLaneProjection;

static inline IDMLeader idm_no_leader(void) {
    IDMLeader leader = {0};
    leader.leader_agent_idx = -1;
    leader.gap = INFINITY;
    return leader;
}

static inline void idm_update_best_leader(IDMLeader *best, int leader_agent_idx, int is_traffic_light, float gap,
                                          float leader_speed) {
    if (gap < 0.0f) {
        gap = IDM_MINIMUM_LEAD_DISTANCE;
    }
    if (gap >= best->gap) {
        return;
    }

    best->has_leader = 1;
    best->leader_agent_idx = leader_agent_idx;
    best->is_traffic_light = is_traffic_light;
    best->gap = fmaxf(gap, IDM_MINIMUM_LEAD_DISTANCE);
    best->leader_speed = fmaxf(0.0f, leader_speed);
}

static inline void idm_point_to_ego_frame(const Agent *ego, float x, float y, float *out_x, float *out_y) {
    float dx = x - ego->sim_x;
    float dy = y - ego->sim_y;
    *out_x = dx * ego->cos_heading + dy * ego->sin_heading;
    *out_y = -dx * ego->sin_heading + dy * ego->cos_heading;
}

static void idm_consider_agent_leader(Drive *env, int ego_idx, int other_idx, float corridor_start, float corridor_end,
                                      float corridor_half_width, IDMLeader *best) {
    if (other_idx == ego_idx) {
        return;
    }

    Agent *ego = &env->agents[ego_idx];
    Agent *other = &env->agents[other_idx];
    if (other->removed || other->sim_x == INVALID_POSITION || other->sim_valid == 0) {
        return;
    }

    if (!check_z_collision_possibility(ego, other)) {
        return;
    }

    float half_length = 0.5f * other->sim_length + IDM_BBOX_MARGIN;
    float half_width = 0.5f * other->sim_width + IDM_BBOX_MARGIN;
    float min_x = INFINITY;
    float max_x = -INFINITY;
    float min_y = INFINITY;
    float max_y = -INFINITY;

    for (int i = 0; i < 4; i++) {
        float corner_x = other->sim_x + offsets[i][0] * half_length * other->cos_heading -
                         offsets[i][1] * half_width * other->sin_heading;
        float corner_y = other->sim_y + offsets[i][0] * half_length * other->sin_heading +
                         offsets[i][1] * half_width * other->cos_heading;
        float rel_x;
        float rel_y;
        idm_point_to_ego_frame(ego, corner_x, corner_y, &rel_x, &rel_y);
        min_x = fminf(min_x, rel_x);
        max_x = fmaxf(max_x, rel_x);
        min_y = fminf(min_y, rel_y);
        max_y = fmaxf(max_y, rel_y);
    }

    if (max_x < corridor_start || min_x > corridor_end) {
        return;
    }
    if (max_y < -corridor_half_width || min_y > corridor_half_width) {
        return;
    }

    float gap = min_x - corridor_start;
    float leader_speed = other->sim_vx * ego->cos_heading + other->sim_vy * ego->sin_heading;
    idm_update_best_leader(best, other_idx, 0, gap, leader_speed);
}

static int idm_traffic_light_controls_lane(TrafficControlElement *traffic, int lane_idx) {
    if (lane_idx == -1 || traffic->num_controlled_lanes <= 0) {
        return 0;
    }
    for (int i = 0; i < traffic->num_controlled_lanes; i++) {
        if (traffic->controlled_lanes[i] == lane_idx) {
            return 1;
        }
    }
    return 0;
}

static inline int idm_is_stop_light_obstacle_state(int state) {
    return state == TRAFFIC_CONTROL_STATE_RED || state == TRAFFIC_CONTROL_STATE_YELLOW;
}

static void idm_consider_red_light_leader(Drive *env, int ego_idx, float corridor_start, float corridor_end,
                                          float corridor_half_width, IDMLeader *best) {
    Agent *ego = &env->agents[ego_idx];
    if (ego->current_lane_idx == -1) {
        return;
    }

    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *traffic = &env->traffic_elements[i];
        if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            continue;
        }
        if (!idm_traffic_light_controls_lane(traffic, ego->current_lane_idx)) {
            continue;
        }
        if (env->timestep < 0 || env->timestep >= traffic->state_length || traffic->states == NULL) {
            continue;
        }
        if (!idm_is_stop_light_obstacle_state(traffic->states[env->timestep])) {
            continue;
        }

        float x1;
        float y1;
        float x2;
        float y2;
        idm_point_to_ego_frame(ego, traffic->stop_line[0], traffic->stop_line[1], &x1, &y1);
        idm_point_to_ego_frame(ego, traffic->stop_line[3], traffic->stop_line[4], &x2, &y2);

        float min_x = fminf(x1, x2);
        float max_x = fmaxf(x1, x2);
        float min_y = fminf(y1, y2);
        float max_y = fmaxf(y1, y2);
        if (max_x < corridor_start || min_x > corridor_end) {
            continue;
        }
        if (max_y < -corridor_half_width || min_y > corridor_half_width) {
            continue;
        }

        float stop_x = 0.5f * (x1 + x2);
        float gap = stop_x - corridor_start;
        idm_update_best_leader(best, -1, 1, gap, 0.0f);
    }
}

static IDMLeader idm_find_leader_by_corridor(Drive *env, int ego_idx) {
    Agent *ego = &env->agents[ego_idx];
    IDMLeader best = idm_no_leader();

    float speed = fmaxf(0.0f, ego->sim_speed_signed);
    float lookahead = clip(speed * IDM_LOOKAHEAD_TIME, IDM_MIN_LOOKAHEAD, IDM_MAX_LOOKAHEAD);
    float corridor_start = 0.5f * ego->sim_length + IDM_BBOX_MARGIN;
    float corridor_end = corridor_start + lookahead;
    float corridor_half_width = 0.5f * ego->sim_width + IDM_BBOX_MARGIN;

    for (int i = 0; i < env->num_agents; i++) {
        int other_idx = -1;
        if (i < env->active_agent_count) {
            other_idx = env->active_agent_indices[i];
        } else {
            other_idx = env->static_agent_indices[i - env->active_agent_count];
        }
        if (other_idx == -1) {
            continue;
        }
        idm_consider_agent_leader(env, ego_idx, other_idx, corridor_start, corridor_end, corridor_half_width, &best);
    }

    idm_consider_red_light_leader(env, ego_idx, corridor_start, corridor_end, corridor_half_width, &best);
    return best;
}

static IDMLaneProjection idm_project_to_route_lanes(Drive *env, Agent *agent);
static float idm_lane_segment_length(RoadMapElement *lane, int seg_idx);

static inline void idm_agent_corners(const Agent *agent, float corners[4][2]) {
    float half_length = 0.5f * agent->sim_length;
    float half_width = 0.5f * agent->sim_width;
    for (int i = 0; i < 4; i++) {
        corners[i][0] = agent->sim_x + offsets[i][0] * half_length * agent->cos_heading -
                        offsets[i][1] * half_width * agent->sin_heading;
        corners[i][1] = agent->sim_y + offsets[i][0] * half_length * agent->sin_heading +
                        offsets[i][1] * half_width * agent->cos_heading;
    }
}

static int idm_collect_route_candidates(Drive *env, int ego_idx, float lookahead, int *candidates, int max_candidates) {
    Agent *ego = &env->agents[ego_idx];
    int count = 0;

    for (int i = 0; i < env->num_agents && count < max_candidates; i++) {
        int other_idx = -1;
        if (i < env->active_agent_count) {
            other_idx = env->active_agent_indices[i];
        } else {
            other_idx = env->static_agent_indices[i - env->active_agent_count];
        }
        if (other_idx == -1 || other_idx == ego_idx) {
            continue;
        }

        Agent *other = &env->agents[other_idx];
        if (other->removed || other->sim_x == INVALID_POSITION || other->sim_valid == 0) {
            continue;
        }
        float dx = other->sim_x - ego->sim_x;
        float dy = other->sim_y - ego->sim_y;
        float max_dist = lookahead + 0.5f * ego->sim_length + 0.5f * other->sim_length + 5.0f + 2.0f * IDM_BBOX_MARGIN;
        if (dx * dx + dy * dy > max_dist * max_dist) {
            continue;
        }

        candidates[count++] = other_idx;
    }

    return count;
}

static inline Agent idm_make_sample_agent(const Agent *ego, float x, float y, float z, float heading) {
    Agent sample = *ego;
    sample.sim_x = x;
    sample.sim_y = y;
    sample.sim_z = z;
    sample.sim_heading = normalize_heading(heading);
    sample.cos_heading = cosf(sample.sim_heading);
    sample.sin_heading = sinf(sample.sim_heading);
    sample.sim_length = ego->sim_length + 2.0f * IDM_BBOX_MARGIN;
    sample.sim_width = ego->sim_width + 2.0f * IDM_BBOX_MARGIN;
    sample.removed = 0;
    sample.sim_valid = 1;
    return sample;
}

static int idm_sample_hits_agent(const Agent *sample, Agent *other) {
    if (!check_z_collision_possibility(sample, other)) {
        return 0;
    }

    float dx = other->sim_x - sample->sim_x;
    float dy = other->sim_y - sample->sim_y;
    float local_radius = 0.5f * sample->sim_length + 0.5f * other->sim_length + sample->sim_width + other->sim_width +
                         1.0f + 2.0f * IDM_BBOX_MARGIN;
    if (dx * dx + dy * dy > local_radius * local_radius) {
        return 0;
    }

    Agent other_expanded = *other;
    Agent sample_expanded = *sample;
    sample_expanded.sim_length = sample->sim_length + 2.0f * IDM_COLLISION_BBOX_MARGIN;
    sample_expanded.sim_width = sample->sim_width + 2.0f * IDM_COLLISION_BBOX_MARGIN;
    other_expanded.sim_length = other->sim_length + 2.0f * (IDM_BBOX_MARGIN + IDM_COLLISION_BBOX_MARGIN);
    other_expanded.sim_width = other->sim_width + 2.0f * (IDM_BBOX_MARGIN + IDM_COLLISION_BBOX_MARGIN);
    return check_obb_collision(&sample_expanded, &other_expanded);
}

static int idm_sample_hits_road_edge(Drive *env, const Agent *sample) {
    if (get_grid_index(env, sample->sim_x, sample->sim_y) == -1) {
        return 1;
    }

    float corners[4][2];
    idm_agent_corners(sample, corners);

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25];
    int list_size = get_neighbors_entities(env, sample->sim_x, sample->sim_y, entity_list, MAX_ENTITIES_PER_CELL * 25,
                                           collision_offsets, 25);
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_type != ENTITY_TYPE_ROAD_ELEMENT) {
            continue;
        }

        int entity_idx = entity_list[i].entity_idx;
        int geometry_idx = entity_list[i].geometry_idx;
        if (entity_idx < 0 || entity_idx >= env->num_road_elements) {
            continue;
        }
        RoadMapElement *element = &env->road_elements[entity_idx];
        if (!is_road_edge(element->type)) {
            continue;
        }
        if (geometry_idx < 0 || geometry_idx >= element->segment_length - 1) {
            continue;
        }

        float abs_dz = fabsf(element->z[geometry_idx] - sample->sim_z);
        if (abs_dz > Z_BUFFER) {
            continue;
        }

        float start[2] = {element->x[geometry_idx], element->y[geometry_idx]};
        float end[2] = {element->x[geometry_idx + 1], element->y[geometry_idx + 1]};
        for (int k = 0; k < 4; k++) {
            int next = (k + 1) % 4;
            if (check_line_intersection(corners[k], corners[next], start, end)) {
                return 1;
            }
        }
    }

    return 0;
}

static int idm_sample_hits_red_light(Drive *env, Agent *sample, int lane_idx) {
    float corners[4][2];
    idm_agent_corners(sample, corners);

    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *traffic = &env->traffic_elements[i];
        if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            continue;
        }
        if (!idm_traffic_light_controls_lane(traffic, lane_idx)) {
            continue;
        }
        if (env->timestep < 0 || env->timestep >= traffic->state_length || traffic->states == NULL) {
            continue;
        }
        if (!idm_is_stop_light_obstacle_state(traffic->states[env->timestep])) {
            continue;
        }

        float mid_x = 0.5f * (traffic->stop_line[0] + traffic->stop_line[3]);
        float mid_y = 0.5f * (traffic->stop_line[1] + traffic->stop_line[4]);
        float dx = sample->sim_x - mid_x;
        float dy = sample->sim_y - mid_y;
        if (dx * dx + dy * dy > TRAFFIC_LIGHT_DISTANCE_THRESHOLD * TRAFFIC_LIGHT_DISTANCE_THRESHOLD) {
            continue;
        }

        float heading_diff = compute_heading_diff(sample->sim_heading, traffic->heading);
        if (fabsf(heading_diff) > RED_LIGHT_HEADING_THRESHOLD) {
            continue;
        }

        float sl_dx = traffic->stop_line[3] - traffic->stop_line[0];
        float sl_dy = traffic->stop_line[4] - traffic->stop_line[1];
        float ext = (STOP_LINE_EXTENSION_FACTOR - 1.0f) * 0.5f;
        float ext_p1[2] = {traffic->stop_line[0] - ext * sl_dx, traffic->stop_line[1] - ext * sl_dy};
        float ext_p2[2] = {traffic->stop_line[3] + ext * sl_dx, traffic->stop_line[4] + ext * sl_dy};

        for (int k = 0; k < 4; k++) {
            if (k == 2) {
                continue;
            }
            int next = (k + 1) % 4;
            if (check_line_intersection(corners[k], corners[next], ext_p1, ext_p2)) {
                return 1;
            }
        }
    }

    return 0;
}

static inline void idm_update_sample_agent_pose(Agent *sample, RoadMapElement *lane, int seg_idx, float t) {
    sample->sim_x = lane->x[seg_idx] + t * (lane->x[seg_idx + 1] - lane->x[seg_idx]);
    sample->sim_y = lane->y[seg_idx] + t * (lane->y[seg_idx + 1] - lane->y[seg_idx]);
    sample->sim_z = lane->z[seg_idx] + t * (lane->z[seg_idx + 1] - lane->z[seg_idx]);
    sample->sim_heading = normalize_heading(lane->headings[seg_idx]);
    sample->cos_heading = cosf(sample->sim_heading);
    sample->sin_heading = sinf(sample->sim_heading);
}

static inline void idm_limit_pose_toward_target(Drive *env, const Agent *reference, float target_x, float target_y,
                                                float target_z, float target_heading, float speed, float duration,
                                                float *out_x, float *out_y, float *out_z, float *out_heading) {
    float forward_x = cosf(target_heading);
    float forward_y = sinf(target_heading);
    float lateral_x = -forward_y;
    float lateral_y = forward_x;

    float dx = target_x - reference->sim_x;
    float dy = target_y - reference->sim_y;
    float forward_step = dx * forward_x + dy * forward_y;
    float lateral_step = dx * lateral_x + dy * lateral_y;

    float positive_speed = fmaxf(0.0f, speed);
    float max_lateral_step =
        fminf(IDM_MAX_LATERAL_STEP * duration / env->dt, positive_speed * duration * IDM_LATERAL_STEP_RATIO);
    if (fabsf(lateral_step) > IDM_LATERAL_SNAP_THRESHOLD && fabsf(lateral_step) > max_lateral_step) {
        lateral_step = clip(lateral_step, -max_lateral_step, max_lateral_step);
        *out_x = reference->sim_x + forward_step * forward_x + lateral_step * lateral_x;
        *out_y = reference->sim_y + forward_step * forward_y + lateral_step * lateral_y;
    } else {
        *out_x = target_x;
        *out_y = target_y;
    }
    *out_z = target_z;

    float heading_delta = compute_heading_diff(target_heading, reference->sim_heading);
    float max_heading_step =
        fminf(IDM_MAX_HEADING_STEP * duration / env->dt, positive_speed * duration * IDM_HEADING_STEP_RATIO);
    if (fabsf(heading_delta) > IDM_HEADING_SNAP_THRESHOLD && fabsf(heading_delta) > max_heading_step) {
        heading_delta = clip(heading_delta, -max_heading_step, max_heading_step);
        *out_heading = normalize_heading(reference->sim_heading + heading_delta);
    } else {
        *out_heading = target_heading;
    }
}

static inline void idm_snap_sample_to_rail(Agent *limited_sample, const Agent *rail_sample) {
    limited_sample->sim_x = rail_sample->sim_x;
    limited_sample->sim_y = rail_sample->sim_y;
    limited_sample->sim_z = rail_sample->sim_z;
    limited_sample->sim_heading = rail_sample->sim_heading;
    limited_sample->cos_heading = rail_sample->cos_heading;
    limited_sample->sin_heading = rail_sample->sin_heading;
}

static inline int idm_limited_sample_reached_rail(const Agent *limited_sample, const Agent *rail_sample) {
    float dx = limited_sample->sim_x - rail_sample->sim_x;
    float dy = limited_sample->sim_y - rail_sample->sim_y;
    float dz = limited_sample->sim_z - rail_sample->sim_z;
    return dx * dx + dy * dy + dz * dz <= IDM_LATERAL_SNAP_THRESHOLD * IDM_LATERAL_SNAP_THRESHOLD;
}

static inline void idm_propagate_limited_route_sample(Drive *env, Agent *limited_sample, const Agent *rail_sample,
                                                      float ds, float speed, int *merged_to_path) {
    if (*merged_to_path) {
        idm_snap_sample_to_rail(limited_sample, rail_sample);
        return;
    }

    float duration = ds / fmaxf(speed, 1e-3f);
    idm_limit_pose_toward_target(env, limited_sample, rail_sample->sim_x, rail_sample->sim_y, rail_sample->sim_z,
                                 rail_sample->sim_heading, speed, duration, &limited_sample->sim_x,
                                 &limited_sample->sim_y, &limited_sample->sim_z, &limited_sample->sim_heading);
    limited_sample->cos_heading = cosf(limited_sample->sim_heading);
    limited_sample->sin_heading = sinf(limited_sample->sim_heading);

    if (idm_limited_sample_reached_rail(limited_sample, rail_sample)) {
        *merged_to_path = 1;
        idm_snap_sample_to_rail(limited_sample, rail_sample);
    }
}

static inline float nuplan_idm_projected_footprint_length(const Agent *agent) {
    return 0.5f * agent->sim_length + fmaxf(0.0f, agent->sim_speed_signed) * NUPLAN_IDM_SAFE_TIME_HEADWAY;
}

static int nuplan_idm_collect_route_candidates(Drive *env, int ego_idx, float lookahead, int *candidates,
                                               int max_candidates) {
    Agent *ego = &env->agents[ego_idx];
    int count = 0;

    for (int i = 0; i < env->num_agents && count < max_candidates; i++) {
        int other_idx = -1;
        if (i < env->active_agent_count) {
            other_idx = env->active_agent_indices[i];
        } else {
            other_idx = env->static_agent_indices[i - env->active_agent_count];
        }
        if (other_idx == -1 || other_idx == ego_idx) {
            continue;
        }

        Agent *other = &env->agents[other_idx];
        if (other->removed || other->sim_x == INVALID_POSITION || other->sim_valid == 0) {
            continue;
        }
        if (!check_z_collision_possibility(ego, other)) {
            continue;
        }

        float dx = other->sim_x - ego->sim_x;
        float dy = other->sim_y - ego->sim_y;
        float max_dist = lookahead + 0.5f * ego->sim_length + nuplan_idm_projected_footprint_length(other) + 5.0f +
                         2.0f * IDM_BBOX_MARGIN;
        if (dx * dx + dy * dy > max_dist * max_dist) {
            continue;
        }

        candidates[count++] = other_idx;
    }

    return count;
}

static int nuplan_idm_boxes_overlap(const Agent *sample, const Agent *other) {
    if (!check_z_collision_possibility(sample, other)) {
        return 0;
    }

    float dx = other->sim_x - sample->sim_x;
    float dy = other->sim_y - sample->sim_y;
    float local_radius = 0.5f * sample->sim_length + 0.5f * other->sim_length + sample->sim_width + other->sim_width +
                         1.0f + 2.0f * IDM_BBOX_MARGIN;
    if (dx * dx + dy * dy > local_radius * local_radius) {
        return 0;
    }

    Agent sample_expanded = *sample;
    Agent other_expanded = *other;
    other_expanded.sim_length = other->sim_length + 2.0f * IDM_BBOX_MARGIN;
    other_expanded.sim_width = other->sim_width + 2.0f * IDM_BBOX_MARGIN;
    return check_obb_collision(&sample_expanded, &other_expanded);
}

static int nuplan_idm_set_projected_agent_pose(Drive *env, Agent *agent, IDMLaneProjection projection, float distance) {
    int route_idx = projection.route_idx;
    int seg_idx = projection.segment_idx;
    float t = projection.t;

    while (route_idx < agent->route_length) {
        int lane_idx = agent->route[route_idx];
        if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
            return 0;
        }
        RoadMapElement *lane = &env->road_elements[lane_idx];
        if (lane->segment_length < 2) {
            return 0;
        }

        while (seg_idx < lane->segment_length - 1) {
            float seg_len = idm_lane_segment_length(lane, seg_idx);
            if (seg_len < 1e-6f) {
                seg_idx++;
                t = 0.0f;
                continue;
            }

            float remaining = (1.0f - t) * seg_len;
            if (distance <= remaining) {
                float next_t = t + distance / seg_len;
                idm_update_sample_agent_pose(agent, lane, seg_idx, clip(next_t, 0.0f, 1.0f));
                return 1;
            }

            distance -= remaining;
            seg_idx++;
            t = 0.0f;
        }

        route_idx++;
        seg_idx = 0;
        t = 0.0f;
    }

    return 0;
}

static int nuplan_idm_sample_hits_projected_agent(Drive *env, const Agent *sample, int other_idx) {
    Agent *other = &env->agents[other_idx];
    if (nuplan_idm_boxes_overlap(sample, other)) {
        return 1;
    }

    IDMLaneProjection projection = idm_project_to_route_lanes(env, other);
    if (!projection.valid) {
        return 0;
    }

    Agent projected = *other;
    float end_s = nuplan_idm_projected_footprint_length(other);
    for (float s = IDM_ROUTE_SAMPLE_DS; s <= end_s + 1e-4f; s += IDM_ROUTE_SAMPLE_DS) {
        projected = *other;
        if (!nuplan_idm_set_projected_agent_pose(env, &projected, projection, s)) {
            return 0;
        }
        if (nuplan_idm_boxes_overlap(sample, &projected)) {
            return 1;
        }
    }

    return 0;
}

static IDMLeader nuplan_idm_find_leader_by_route_boxes(Drive *env, int ego_idx) {
    Agent *ego = &env->agents[ego_idx];
    IDMLeader no_leader = idm_no_leader();
    IDMLaneProjection projection = idm_project_to_route_lanes(env, ego);
    if (!projection.valid) {
        return no_leader;
    }

    float speed = fmaxf(0.0f, ego->sim_speed_signed);
    float lookahead = clip(speed * NUPLAN_IDM_LOOKAHEAD_TIME, NUPLAN_IDM_MIN_LOOKAHEAD, NUPLAN_IDM_MAX_LOOKAHEAD);
    int candidates[NUPLAN_IDM_MAX_CANDIDATES];
    int num_candidates =
        nuplan_idm_collect_route_candidates(env, ego_idx, lookahead, candidates, NUPLAN_IDM_MAX_CANDIDATES);

    Agent sample = idm_make_sample_agent(ego, ego->sim_x, ego->sim_y, ego->sim_z, ego->sim_heading);
    Agent limited_sample = sample;
    int limited_merged_to_path = projection.dist_sq <= IDM_LATERAL_SNAP_THRESHOLD * IDM_LATERAL_SNAP_THRESHOLD;
    float next_sample_s = IDM_ROUTE_SAMPLE_DS;
    float prev_sample_s = 0.0f;
    float traveled_s = 0.0f;
    int route_idx = projection.route_idx;
    int seg_idx = projection.segment_idx;
    float t = projection.t;

    while (route_idx < ego->route_length && next_sample_s <= lookahead + 1e-4f) {
        int lane_idx = ego->route[route_idx];
        if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
            break;
        }
        RoadMapElement *lane = &env->road_elements[lane_idx];
        if (lane->segment_length < 2) {
            break;
        }

        while (seg_idx < lane->segment_length - 1 && next_sample_s <= lookahead + 1e-4f) {
            float seg_len = idm_lane_segment_length(lane, seg_idx);
            if (seg_len < 1e-6f) {
                seg_idx++;
                t = 0.0f;
                continue;
            }

            float remaining = (1.0f - t) * seg_len;
            if (traveled_s + remaining + 1e-4f < next_sample_s) {
                traveled_s += remaining;
                seg_idx++;
                t = 0.0f;
                continue;
            }

            float sample_t = t + (next_sample_s - traveled_s) / seg_len;
            sample_t = clip(sample_t, 0.0f, 1.0f);
            idm_update_sample_agent_pose(&sample, lane, seg_idx, sample_t);
            idm_propagate_limited_route_sample(env, &limited_sample, &sample, next_sample_s - prev_sample_s, speed,
                                               &limited_merged_to_path);

            if (idm_sample_hits_red_light(env, &sample, lane_idx) ||
                idm_sample_hits_red_light(env, &limited_sample, lane_idx)) {
                idm_update_best_leader(&no_leader, -1, 1, next_sample_s, 0.0f);
                return no_leader;
            }
            if (idm_sample_hits_road_edge(env, &sample) || idm_sample_hits_road_edge(env, &limited_sample)) {
                idm_update_best_leader(&no_leader, IDM_WALL_LEADER_IDX, 0, next_sample_s, 0.0f);
                return no_leader;
            }

            IDMLeader best_at_sample = idm_no_leader();
            for (int i = 0; i < num_candidates; i++) {
                int other_idx = candidates[i];
                Agent *other = &env->agents[other_idx];
                int rail_hit = nuplan_idm_sample_hits_projected_agent(env, &sample, other_idx);
                int limited_hit = !rail_hit && nuplan_idm_sample_hits_projected_agent(env, &limited_sample, other_idx);
                if (!rail_hit && !limited_hit) {
                    continue;
                }
                Agent *hit_sample = rail_hit ? &sample : &limited_sample;
                float leader_speed = other->sim_vx * hit_sample->cos_heading + other->sim_vy * hit_sample->sin_heading;
                idm_update_best_leader(&best_at_sample, other_idx, 0, next_sample_s, leader_speed);
            }
            if (best_at_sample.has_leader) {
                return best_at_sample;
            }

            next_sample_s += IDM_ROUTE_SAMPLE_DS;
            prev_sample_s = next_sample_s - IDM_ROUTE_SAMPLE_DS;
        }

        route_idx++;
        seg_idx = 0;
        t = 0.0f;
    }

    return no_leader;
}

static IDMLeader idm_find_leader_by_route_boxes(Drive *env, int ego_idx) {
    Agent *ego = &env->agents[ego_idx];
    IDMLeader no_leader = idm_no_leader();
    IDMLaneProjection projection = idm_project_to_route_lanes(env, ego);
    if (!projection.valid) {
        return no_leader;
    }

    float speed = fmaxf(0.0f, ego->sim_speed_signed);
    float lookahead = clip(speed * IDM_LOOKAHEAD_TIME, IDM_MIN_LOOKAHEAD, IDM_MAX_LOOKAHEAD);
    int candidates[IDM_MAX_CANDIDATES];
    int num_candidates = idm_collect_route_candidates(env, ego_idx, lookahead, candidates, IDM_MAX_CANDIDATES);

    Agent sample = idm_make_sample_agent(ego, ego->sim_x, ego->sim_y, ego->sim_z, ego->sim_heading);
    Agent limited_sample = sample;
    int limited_merged_to_path = projection.dist_sq <= IDM_LATERAL_SNAP_THRESHOLD * IDM_LATERAL_SNAP_THRESHOLD;
    float next_sample_s = IDM_ROUTE_SAMPLE_DS;
    float prev_sample_s = 0.0f;
    float traveled_s = 0.0f;
    int route_idx = projection.route_idx;
    int seg_idx = projection.segment_idx;
    float t = projection.t;

    while (route_idx < ego->route_length && next_sample_s <= lookahead + 1e-4f) {
        int lane_idx = ego->route[route_idx];
        if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
            break;
        }
        RoadMapElement *lane = &env->road_elements[lane_idx];
        if (lane->segment_length < 2) {
            break;
        }

        while (seg_idx < lane->segment_length - 1 && next_sample_s <= lookahead + 1e-4f) {
            float seg_len = idm_lane_segment_length(lane, seg_idx);
            if (seg_len < 1e-6f) {
                seg_idx++;
                t = 0.0f;
                continue;
            }

            float remaining = (1.0f - t) * seg_len;
            if (traveled_s + remaining + 1e-4f < next_sample_s) {
                traveled_s += remaining;
                seg_idx++;
                t = 0.0f;
                continue;
            }

            float sample_t = t + (next_sample_s - traveled_s) / seg_len;
            sample_t = clip(sample_t, 0.0f, 1.0f);
            idm_update_sample_agent_pose(&sample, lane, seg_idx, sample_t);
            idm_propagate_limited_route_sample(env, &limited_sample, &sample, next_sample_s - prev_sample_s, speed,
                                               &limited_merged_to_path);

            if (idm_sample_hits_red_light(env, &sample, lane_idx) ||
                idm_sample_hits_red_light(env, &limited_sample, lane_idx)) {
                idm_update_best_leader(&no_leader, -1, 1, next_sample_s, 0.0f);
                return no_leader;
            }
            if (idm_sample_hits_road_edge(env, &sample) || idm_sample_hits_road_edge(env, &limited_sample)) {
                idm_update_best_leader(&no_leader, IDM_WALL_LEADER_IDX, 0, next_sample_s, 0.0f);
                return no_leader;
            }

            IDMLeader best_at_sample = idm_no_leader();
            for (int i = 0; i < num_candidates; i++) {
                int other_idx = candidates[i];
                Agent *other = &env->agents[other_idx];
                int rail_hit = idm_sample_hits_agent(&sample, other);
                int limited_hit = !rail_hit && idm_sample_hits_agent(&limited_sample, other);
                if (!rail_hit && !limited_hit) {
                    continue;
                }
                Agent *hit_sample = rail_hit ? &sample : &limited_sample;
                float leader_speed = other->sim_vx * hit_sample->cos_heading + other->sim_vy * hit_sample->sin_heading;
                idm_update_best_leader(&best_at_sample, other_idx, 0, next_sample_s, leader_speed);
            }
            if (best_at_sample.has_leader) {
                return best_at_sample;
            }

            next_sample_s += IDM_ROUTE_SAMPLE_DS;
            prev_sample_s = next_sample_s - IDM_ROUTE_SAMPLE_DS;
        }

        route_idx++;
        seg_idx = 0;
        t = 0.0f;
    }

    return no_leader;
}

static inline float idm_lane_speed_limit(Drive *env, int lane_idx) {
    if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
        return 0.0f;
    }
    return env->road_elements[lane_idx].speed_limit;
}

static float idm_desired_speed(Drive *env, Agent *agent) {
    float desired_speed = idm_lane_speed_limit(env, agent->current_lane_idx);

    if (desired_speed <= 0.0f && agent->route != NULL && agent->route_length > 0) {
        int route_idx = agent->current_route_index;
        if (route_idx < 0) {
            route_idx = 0;
        } else if (route_idx >= agent->route_length) {
            route_idx = agent->route_length - 1;
        }
        desired_speed = idm_lane_speed_limit(env, agent->route[route_idx]);
    }

    if (desired_speed <= 0.0f) {
        desired_speed = IDM_DEFAULT_DESIRED_SPEED;
    }

    return clip(desired_speed, 1.0f, MAX_SPEED);
}

static float idm_compute_acceleration(Drive *env, Agent *agent, IDMLeader leader) {
    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    float desired_speed = idm_desired_speed(env, agent);
    float speed_ratio = current_speed / desired_speed;
    float free_road_term = powf(speed_ratio, IDM_DELTA);
    float leader_term = 0.0f;

    if (leader.has_leader) {
        float s_star = IDM_MIN_SPACING + fmaxf(0.0f, current_speed * IDM_SAFE_TIME_HEADWAY +
                                                         current_speed * (current_speed - leader.leader_speed) /
                                                             (2.0f * sqrtf(IDM_MAX_ACCEL * IDM_MAX_DECEL)));
        float lead_dist = fmaxf(leader.gap, IDM_MINIMUM_LEAD_DISTANCE);
        leader_term = (s_star / lead_dist) * (s_star / lead_dist);
    }

    return IDM_MAX_ACCEL * (1.0f - free_road_term - leader_term);
}

static float nuplan_idm_desired_speed(Drive *env, Agent *agent) {
    float desired_speed = idm_lane_speed_limit(env, agent->current_lane_idx);

    if (desired_speed <= 0.0f && agent->route != NULL && agent->route_length > 0) {
        int route_idx = agent->current_route_index;
        if (route_idx < 0) {
            route_idx = 0;
        } else if (route_idx >= agent->route_length) {
            route_idx = agent->route_length - 1;
        }
        desired_speed = idm_lane_speed_limit(env, agent->route[route_idx]);
    }

    if (desired_speed <= 0.0f) {
        desired_speed = NUPLAN_IDM_DEFAULT_DESIRED_SPEED;
    }

    desired_speed = fminf(desired_speed, NUPLAN_IDM_DEFAULT_DESIRED_SPEED);
    return clip(desired_speed, 1.0f, MAX_SPEED);
}

static float nuplan_idm_compute_acceleration(Drive *env, Agent *agent, IDMLeader leader) {
    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    float desired_speed = nuplan_idm_desired_speed(env, agent);
    float speed_ratio = current_speed / desired_speed;
    float free_road_term = powf(speed_ratio, IDM_DELTA);
    float leader_term = 0.0f;

    if (leader.has_leader) {
        float s_star =
            NUPLAN_IDM_MIN_SPACING + fmaxf(0.0f, current_speed * NUPLAN_IDM_SAFE_TIME_HEADWAY +
                                                     current_speed * (current_speed - leader.leader_speed) /
                                                         (2.0f * sqrtf(NUPLAN_IDM_MAX_ACCEL * NUPLAN_IDM_MAX_DECEL)));
        float lead_dist = fmaxf(leader.gap, IDM_MINIMUM_LEAD_DISTANCE);
        leader_term = (s_star / lead_dist) * (s_star / lead_dist);
    }

    return NUPLAN_IDM_MAX_ACCEL * (1.0f - free_road_term - leader_term);
}

static IDMLaneProjection idm_project_to_route_lanes(Drive *env, Agent *agent) {
    IDMLaneProjection best = {0};
    best.route_idx = 0;
    best.lane_idx = -1;
    best.segment_idx = 0;
    best.t = 0.0f;
    best.dist_sq = INFINITY;

    if (agent->route == NULL || agent->route_length <= 0) {
        return best;
    }

    int start_route = agent->current_route_index - 1;
    if (start_route < 0) {
        start_route = 0;
    }
    int end_route = agent->current_route_index + 4;
    if (end_route > agent->route_length) {
        end_route = agent->route_length;
    }

    for (int pass = 0; pass < 2; pass++) {
        for (int route_idx = start_route; route_idx < end_route; route_idx++) {
            int lane_idx = agent->route[route_idx];
            if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
                continue;
            }
            RoadMapElement *lane = &env->road_elements[lane_idx];
            if (lane->segment_length < 2) {
                continue;
            }
            for (int seg_idx = 0; seg_idx < lane->segment_length - 1; seg_idx++) {
                float dx = lane->x[seg_idx + 1] - lane->x[seg_idx];
                float dy = lane->y[seg_idx + 1] - lane->y[seg_idx];
                float dz = lane->z[seg_idx + 1] - lane->z[seg_idx];
                float seg_len_sq = dx * dx + dy * dy + dz * dz;
                if (seg_len_sq < 1e-6f) {
                    continue;
                }

                float ax = agent->sim_x - lane->x[seg_idx];
                float ay = agent->sim_y - lane->y[seg_idx];
                float az = agent->sim_z - lane->z[seg_idx];
                float t = (ax * dx + ay * dy + az * dz) / seg_len_sq;
                t = clip(t, 0.0f, 1.0f);

                float px = lane->x[seg_idx] + t * dx;
                float py = lane->y[seg_idx] + t * dy;
                float pz = lane->z[seg_idx] + t * dz;
                float err_x = agent->sim_x - px;
                float err_y = agent->sim_y - py;
                float err_z = agent->sim_z - pz;
                float dist_sq = err_x * err_x + err_y * err_y + err_z * err_z;

                if (dist_sq < best.dist_sq) {
                    best.valid = 1;
                    best.route_idx = route_idx;
                    best.lane_idx = lane_idx;
                    best.segment_idx = seg_idx;
                    best.t = t;
                    best.dist_sq = dist_sq;
                }
            }
        }

        if (best.valid) {
            break;
        }

        start_route = 0;
        end_route = agent->route_length;
    }

    return best;
}

static float idm_lane_segment_length(RoadMapElement *lane, int seg_idx) {
    float dx = lane->x[seg_idx + 1] - lane->x[seg_idx];
    float dy = lane->y[seg_idx + 1] - lane->y[seg_idx];
    float dz = lane->z[seg_idx + 1] - lane->z[seg_idx];
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

static int idm_set_pose_on_lane_segment(Drive *env, Agent *agent, int route_idx, int lane_idx, int seg_idx, float t,
                                        float *old_heading_out) {
    if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
        return 0;
    }
    RoadMapElement *lane = &env->road_elements[lane_idx];
    if (seg_idx < 0 || seg_idx >= lane->segment_length - 1) {
        return 0;
    }
    t = clip(t, 0.0f, 1.0f);

    if (old_heading_out != NULL) {
        *old_heading_out = agent->sim_heading;
    }

    agent->sim_x = lane->x[seg_idx] + t * (lane->x[seg_idx + 1] - lane->x[seg_idx]);
    agent->sim_y = lane->y[seg_idx] + t * (lane->y[seg_idx + 1] - lane->y[seg_idx]);
    agent->sim_z = lane->z[seg_idx] + t * (lane->z[seg_idx + 1] - lane->z[seg_idx]);
    agent->sim_heading = normalize_heading(lane->headings[seg_idx]);
    agent->cos_heading = cosf(agent->sim_heading);
    agent->sin_heading = sinf(agent->sim_heading);
    agent->current_route_index = route_idx;
    agent->current_lane_idx = lane_idx;
    agent->current_lane_geometry_idx = seg_idx;
    return 1;
}

static int idm_refresh_route_at_lane_end(Drive *env, int agent_idx, int lane_idx) {
    Agent *agent = &env->agents[agent_idx];
    if (lane_idx == -1) {
        lane_idx = agent->current_lane_idx;
    }
    if (lane_idx == -1) {
        return 0;
    }

    if (!compute_new_route(env, agent_idx, lane_idx)) {
        return 0;
    }
    return compute_goals(env, agent_idx);
}

static int idm_advance_along_route_lanes(Drive *env, int agent_idx, float distance, float *old_heading_out) {
    Agent *agent = &env->agents[agent_idx];
    if (distance <= 0.0f) {
        if (old_heading_out != NULL) {
            *old_heading_out = agent->sim_heading;
        }
        return 1;
    }

    for (int attempt = 0; attempt < 4; attempt++) {
        IDMLaneProjection projection = idm_project_to_route_lanes(env, agent);
        if (!projection.valid) {
            return 0;
        }

        int route_idx = projection.route_idx;
        int seg_idx = projection.segment_idx;
        float t = projection.t;
        int lane_idx = projection.lane_idx;

        while (route_idx < agent->route_length) {
            lane_idx = agent->route[route_idx];
            if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
                return 0;
            }
            RoadMapElement *lane = &env->road_elements[lane_idx];
            if (lane->segment_length < 2) {
                return 0;
            }

            while (seg_idx < lane->segment_length - 1) {
                float seg_len = idm_lane_segment_length(lane, seg_idx);
                if (seg_len < 1e-6f) {
                    seg_idx++;
                    t = 0.0f;
                    continue;
                }

                float remaining = (1.0f - t) * seg_len;
                if (distance <= remaining) {
                    float next_t = t + distance / seg_len;
                    return idm_set_pose_on_lane_segment(env, agent, route_idx, lane_idx, seg_idx, next_t,
                                                        old_heading_out);
                }

                distance -= remaining;
                seg_idx++;
                t = 0.0f;
            }

            route_idx++;
            seg_idx = 0;
            t = 0.0f;
        }

        if (!idm_refresh_route_at_lane_end(env, agent_idx, lane_idx)) {
            return 0;
        }
    }

    return 0;
}

static int idm_advance_along_route_lanes_limited(Drive *env, int agent_idx, float distance, float speed,
                                                 float *old_heading_out) {
    Agent *agent = &env->agents[agent_idx];
    IDMLaneProjection pre_projection = idm_project_to_route_lanes(env, agent);
    int merged_to_path =
        pre_projection.valid && pre_projection.dist_sq <= IDM_LATERAL_SNAP_THRESHOLD * IDM_LATERAL_SNAP_THRESHOLD;
    Agent reference = *agent;

    if (!idm_advance_along_route_lanes(env, agent_idx, distance, old_heading_out)) {
        return 0;
    }

    if (merged_to_path) {
        return 1;
    }

    float target_x = agent->sim_x;
    float target_y = agent->sim_y;
    float target_z = agent->sim_z;
    float target_heading = agent->sim_heading;

    idm_limit_pose_toward_target(env, &reference, target_x, target_y, target_z, target_heading, speed, env->dt,
                                 &agent->sim_x, &agent->sim_y, &agent->sim_z, &agent->sim_heading);
    agent->cos_heading = cosf(agent->sim_heading);
    agent->sin_heading = sinf(agent->sim_heading);
    return 1;
}

static void idm_move_with_leader(Drive *env, int agent_idx, IDMLeader leader) {
    Agent *agent = &env->agents[agent_idx];

    if (agent->removed) {
        invalidate_agent(agent);
        return;
    }

    if (agent->stopped || agent->sim_x == INVALID_POSITION) {
        agent->sim_vx = 0.0f;
        agent->sim_vy = 0.0f;
        agent->yaw_rate = 0.0f;
        agent->sim_speed = 0.0f;
        agent->sim_speed_signed = 0.0f;
        agent->a_long = 0.0f;
        agent->a_lat = 0.0f;
        agent->jerk_long = 0.0f;
        agent->jerk_lat = 0.0f;
        agent->steering_angle = 0.0f;
        return;
    }

    float old_a_long = agent->a_long;
    float accel = idm_compute_acceleration(env, agent, leader);
    accel = clip(accel, -IDM_MAX_DECEL, IDM_MAX_ACCEL);

    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    float new_speed = current_speed + accel * env->dt;
    if (new_speed < 0.0f) {
        new_speed = 0.0f;
    }
    accel = (new_speed - current_speed) / env->dt;

    float old_heading = agent->sim_heading;
    float distance = new_speed * env->dt;
    if (!idm_advance_along_route_lanes_limited(env, agent_idx, distance, new_speed, &old_heading)) {
        agent->stopped = 1;
        new_speed = 0.0f;
        accel = (new_speed - current_speed) / env->dt;
    }
    agent->sim_vx = new_speed * agent->cos_heading;
    agent->sim_vy = new_speed * agent->sin_heading;
    agent->yaw_rate = compute_heading_diff(agent->sim_heading, old_heading) / env->dt;
    agent->jerk_long = (accel - old_a_long) / env->dt;
    float new_a_lat = new_speed * agent->yaw_rate;
    agent->jerk_lat = (new_a_lat - agent->a_lat) / env->dt;
    agent->a_long = accel;
    agent->a_lat = new_a_lat;
    agent->steering_angle = 0.0f;
    update_agent_speed(agent);
}

static void nuplan_idm_move_with_leader(Drive *env, int agent_idx, IDMLeader leader) {
    Agent *agent = &env->agents[agent_idx];

    if (agent->removed) {
        invalidate_agent(agent);
        return;
    }

    if (agent->stopped || agent->sim_x == INVALID_POSITION) {
        agent->sim_vx = 0.0f;
        agent->sim_vy = 0.0f;
        agent->yaw_rate = 0.0f;
        agent->sim_speed = 0.0f;
        agent->sim_speed_signed = 0.0f;
        agent->a_long = 0.0f;
        agent->a_lat = 0.0f;
        agent->jerk_long = 0.0f;
        agent->jerk_lat = 0.0f;
        agent->steering_angle = 0.0f;
        return;
    }

    float old_a_long = agent->a_long;
    float accel = nuplan_idm_compute_acceleration(env, agent, leader);
    accel = clip(accel, -NUPLAN_IDM_MAX_DECEL, NUPLAN_IDM_MAX_ACCEL);

    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    float new_speed = current_speed + accel * env->dt;
    if (new_speed < 0.0f) {
        new_speed = 0.0f;
    }
    accel = (new_speed - current_speed) / env->dt;

    float old_heading = agent->sim_heading;
    float distance = new_speed * env->dt;
    if (!idm_advance_along_route_lanes_limited(env, agent_idx, distance, new_speed, &old_heading)) {
        agent->stopped = 1;
        new_speed = 0.0f;
        accel = (new_speed - current_speed) / env->dt;
    }
    agent->sim_vx = new_speed * agent->cos_heading;
    agent->sim_vy = new_speed * agent->sin_heading;
    agent->yaw_rate = compute_heading_diff(agent->sim_heading, old_heading) / env->dt;
    agent->jerk_long = (accel - old_a_long) / env->dt;
    float new_a_lat = new_speed * agent->yaw_rate;
    agent->jerk_lat = (new_a_lat - agent->a_lat) / env->dt;
    agent->a_long = accel;
    agent->a_lat = new_a_lat;
    agent->steering_angle = 0.0f;
    update_agent_speed(agent);
}

static void move_corridor_idm(Drive *env, int agent_idx) {
    IDMLeader leader = idm_find_leader_by_corridor(env, agent_idx);
    idm_move_with_leader(env, agent_idx, leader);
}

static void move_idm(Drive *env, int agent_idx) {
    IDMLeader leader = idm_find_leader_by_route_boxes(env, agent_idx);
    idm_move_with_leader(env, agent_idx, leader);
}

static void move_nuplan_idm(Drive *env, int agent_idx) {
    IDMLeader leader = nuplan_idm_find_leader_by_route_boxes(env, agent_idx);
    nuplan_idm_move_with_leader(env, agent_idx, leader);
}

#endif
