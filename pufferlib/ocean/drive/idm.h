#ifndef PUFFERLIB_OCEAN_DRIVE_IDM_H
#define PUFFERLIB_OCEAN_DRIVE_IDM_H

#define IDM_MINIMUM_LEAD_DISTANCE 0.1f
#define IDM_MIN_SPACING 1.0f
#define IDM_SAFE_TIME_HEADWAY 1.5f
#define IDM_MAX_ACCEL 1.0f
#define IDM_MAX_DECEL 3.0f
#define IDM_DELTA 4.0f
#define IDM_LOOKAHEAD_TIME 5.0f
#define IDM_MIN_LOOKAHEAD 20.0f
#define IDM_MAX_LOOKAHEAD 40.0f
#define IDM_BBOX_MARGIN 0.05f
#define IDM_DEFAULT_DESIRED_SPEED 10.0f
#define IDM_ROUTE_SAMPLE_DS 1.0f
#define IDM_MAX_CANDIDATES 64

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

static inline void idm_update_best_leader(
    IDMLeader *best,
    int leader_agent_idx,
    int is_traffic_light,
    float gap,
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

static inline int idm_check_z_overlap(const Agent *a, const Agent *b) {
    float a_bottom = a->sim_z;
    float a_top = a->sim_z + a->sim_height;
    float b_bottom = b->sim_z;
    float b_top = b->sim_z + b->sim_height;
    return !(a_top < b_bottom || b_top < a_bottom);
}

static inline float idm_projected_footprint_length(const Agent *agent) {
    return 0.5f * agent->sim_length + fmaxf(0.0f, agent->sim_speed_signed) * IDM_SAFE_TIME_HEADWAY;
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

static IDMLaneProjection idm_project_to_route_lanes(Drive *env, Agent *agent);
static float idm_lane_segment_size(RoadMapElement *lane, int seg_idx);

static int idm_collect_route_candidates(Drive *env, int ego_idx, float lookahead, int *candidates, int max_candidates) {
    Agent *ego = &env->agents[ego_idx];
    int count = 0;

    for (int i = 0; i < env->num_total_agents && count < max_candidates; i++) {
        int other_idx = i;
        if (other_idx == ego_idx) {
            continue;
        }

        Agent *other = &env->agents[other_idx];
        if (other->removed || other->sim_x == INVALID_POSITION || other->sim_valid == 0) {
            continue;
        }
        if (!idm_check_z_overlap(ego, other)) {
            continue;
        }

        float dx = other->sim_x - ego->sim_x;
        float dy = other->sim_y - ego->sim_y;
        float max_dist = lookahead + 0.5f * ego->sim_length + idm_projected_footprint_length(other) + 5.0f
            + 2.0f * IDM_BBOX_MARGIN;
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
    // Static probe: prev == cur so any swept geometry check degenerates to a static box test.
    sample.prev_x = x;
    sample.prev_y = y;
    sample.prev_cos_heading = sample.cos_heading;
    sample.prev_sin_heading = sample.sin_heading;
    sample.sim_length = ego->sim_length + 2.0f * IDM_BBOX_MARGIN;
    sample.sim_width = ego->sim_width + 2.0f * IDM_BBOX_MARGIN;
    sample.removed = 0;
    sample.sim_valid = 1;
    return sample;
}

static int idm_boxes_overlap(const Agent *sample, const Agent *other) {
    if (!idm_check_z_overlap(sample, other)) {
        return 0;
    }

    float dx = other->sim_x - sample->sim_x;
    float dy = other->sim_y - sample->sim_y;
    float local_radius = 0.5f * sample->sim_length + 0.5f * other->sim_length + sample->sim_width + other->sim_width
        + 1.0f + 2.0f * IDM_BBOX_MARGIN;
    if (dx * dx + dy * dy > local_radius * local_radius) {
        return 0;
    }

    Agent sample_expanded = *sample;
    Agent other_expanded = *other;
    other_expanded.sim_length = other->sim_length + 2.0f * IDM_BBOX_MARGIN;
    other_expanded.sim_width = other->sim_width + 2.0f * IDM_BBOX_MARGIN;
    return check_obb_collision(&sample_expanded, &other_expanded);
}
static int idm_sample_hits_red_light(Drive *env, Agent *sample, int lane_idx) {
    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *traffic = &env->traffic_elements[i];
        if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            continue;
        }
        if (!idm_traffic_light_controls_lane(traffic, lane_idx)) {
            continue;
        }
        if (env->timestep < 0 || env->timestep >= traffic->state_size || traffic->states == NULL) {
            continue;
        }
        if (!idm_is_stop_light_obstacle_state(traffic->states[env->timestep])) {
            continue;
        }

        float mid_x = 0.5f * (traffic->stop_line[0] + traffic->stop_line[3]);
        float mid_y = 0.5f * (traffic->stop_line[1] + traffic->stop_line[4]);
        float dx = sample->sim_x - mid_x;
        float dy = sample->sim_y - mid_y;
        if (dx * dx + dy * dy > STOP_LINE_DIST_SQ) {
            continue;
        }

        float heading_diff = compute_heading_diff(sample->sim_heading, traffic->heading);
        if (fabsf(heading_diff) > STOP_LINE_HEADING_THRESHOLD) {
            continue;
        }

        float sl_dx = traffic->stop_line[3] - traffic->stop_line[0];
        float sl_dy = traffic->stop_line[4] - traffic->stop_line[1];
        float ext = (STOP_LINE_EXTENSION_FACTOR - 1.0f) * 0.5f;
        float ext_p1[2] = {traffic->stop_line[0] - ext * sl_dx, traffic->stop_line[1] - ext * sl_dy};
        float ext_p2[2] = {traffic->stop_line[3] + ext * sl_dx, traffic->stop_line[4] + ext * sl_dy};

        if (check_segment_crosses_moving_box(ext_p1[0], ext_p1[1], ext_p2[0], ext_p2[1], sample)) {
            return 1;
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
    // Keep prev == cur so swept geometry checks stay static for the static probe.
    sample->prev_x = sample->sim_x;
    sample->prev_y = sample->sim_y;
    sample->prev_cos_heading = sample->cos_heading;
    sample->prev_sin_heading = sample->sin_heading;
}

static int idm_set_projected_agent_pose(Drive *env, Agent *agent, IDMLaneProjection projection, float distance) {
    int route_idx = projection.route_idx;
    int seg_idx = projection.segment_idx;
    float t = projection.t;

    while (route_idx < agent->route_length) {
        int lane_idx = agent->route[route_idx];
        if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
            return 0;
        }
        RoadMapElement *lane = &env->road_elements[lane_idx];
        if (lane->segment_size < 2) {
            return 0;
        }

        while (seg_idx < lane->segment_size - 1) {
            float seg_len = idm_lane_segment_size(lane, seg_idx);
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

static int idm_sample_hits_projected_agent(Drive *env, const Agent *sample, int other_idx) {
    Agent *other = &env->agents[other_idx];
    if (idm_boxes_overlap(sample, other)) {
        return 1;
    }

    IDMLaneProjection projection = idm_project_to_route_lanes(env, other);
    if (!projection.valid) {
        return 0;
    }

    Agent projected = *other;
    float end_s = idm_projected_footprint_length(other);
    for (float s = IDM_ROUTE_SAMPLE_DS; s <= end_s + 1e-4f; s += IDM_ROUTE_SAMPLE_DS) {
        projected = *other;
        if (!idm_set_projected_agent_pose(env, &projected, projection, s)) {
            return 0;
        }
        if (idm_boxes_overlap(sample, &projected)) {
            return 1;
        }
    }

    return 0;
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
    float next_sample_s = IDM_ROUTE_SAMPLE_DS;
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
        if (lane->segment_size < 2) {
            break;
        }

        while (seg_idx < lane->segment_size - 1 && next_sample_s <= lookahead + 1e-4f) {
            float seg_len = idm_lane_segment_size(lane, seg_idx);
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

            if (idm_sample_hits_red_light(env, &sample, lane_idx)) {
                idm_update_best_leader(&no_leader, -1, 1, next_sample_s, 0.0f);
                return no_leader;
            }

            IDMLeader best_at_sample = idm_no_leader();
            for (int i = 0; i < num_candidates; i++) {
                int other_idx = candidates[i];
                Agent *other = &env->agents[other_idx];
                if (!idm_sample_hits_projected_agent(env, &sample, other_idx)) {
                    continue;
                }
                float leader_speed = other->sim_vx * sample.cos_heading + other->sim_vy * sample.sin_heading;
                idm_update_best_leader(&best_at_sample, other_idx, 0, next_sample_s, leader_speed);
            }
            if (best_at_sample.has_leader) {
                return best_at_sample;
            }

            next_sample_s += IDM_ROUTE_SAMPLE_DS;
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
        int route_idx = agent->current_route_idx;
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

    return clip(desired_speed, 1.0f, env->base_max_speed_mps);
}

static float idm_compute_acceleration(Drive *env, Agent *agent, IDMLeader leader) {
    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    float desired_speed = idm_desired_speed(env, agent);
    float speed_ratio = current_speed / desired_speed;
    float free_road_term = powf(speed_ratio, IDM_DELTA);
    float leader_term = 0.0f;

    if (leader.has_leader) {
        float s_star = IDM_MIN_SPACING
            + fmaxf(0.0f,
                    current_speed * IDM_SAFE_TIME_HEADWAY
                        + current_speed * (current_speed - leader.leader_speed)
                            / (2.0f * sqrtf(IDM_MAX_ACCEL * IDM_MAX_DECEL)));
        float lead_dist = fmaxf(leader.gap, IDM_MINIMUM_LEAD_DISTANCE);
        leader_term = (s_star / lead_dist) * (s_star / lead_dist);
    }

    return IDM_MAX_ACCEL * (1.0f - free_road_term - leader_term);
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

    int start_route = agent->current_route_idx - 1;
    if (start_route < 0) {
        start_route = 0;
    }
    int end_route = agent->current_route_idx + 4;
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
            if (lane->segment_size < 2) {
                continue;
            }
            for (int seg_idx = 0; seg_idx < lane->segment_size - 1; seg_idx++) {
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

static float idm_lane_segment_size(RoadMapElement *lane, int seg_idx) {
    float dx = lane->x[seg_idx + 1] - lane->x[seg_idx];
    float dy = lane->y[seg_idx + 1] - lane->y[seg_idx];
    float dz = lane->z[seg_idx + 1] - lane->z[seg_idx];
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

static int idm_set_pose_on_lane_segment(
    Drive *env,
    Agent *agent,
    int route_idx,
    int lane_idx,
    int seg_idx,
    float t,
    float *old_heading_out) {
    if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
        return 0;
    }
    RoadMapElement *lane = &env->road_elements[lane_idx];
    if (seg_idx < 0 || seg_idx >= lane->segment_size - 1) {
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
    agent->current_route_idx = route_idx;
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

    if (!compute_new_route(env, agent, lane_idx)) {
        return 0;
    }
    generate_new_goals_from_route(env, agent);
    return 1;
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
            if (lane->segment_size < 2) {
                return 0;
            }

            while (seg_idx < lane->segment_size - 1) {
                float seg_len = idm_lane_segment_size(lane, seg_idx);
                if (seg_len < 1e-6f) {
                    seg_idx++;
                    t = 0.0f;
                    continue;
                }

                float remaining = (1.0f - t) * seg_len;
                if (distance <= remaining) {
                    float next_t = t + distance / seg_len;
                    return idm_set_pose_on_lane_segment(
                        env,
                        agent,
                        route_idx,
                        lane_idx,
                        seg_idx,
                        next_t,
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

static void idm_move_with_leader(Drive *env, int agent_idx, IDMLeader leader) {
    Agent *agent = &env->agents[agent_idx];
    copy_pose_to_prev(agent);

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
        agent->accel_long = 0.0f;
        agent->accel_lat = 0.0f;
        agent->jerk_long = 0.0f;
        agent->jerk_lat = 0.0f;
        agent->steering_angle = 0.0f;
        return;
    }

    float old_a_long = agent->accel_long;
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
    if (!idm_advance_along_route_lanes(env, agent_idx, distance, &old_heading)) {
        agent->stopped = 1;
        new_speed = 0.0f;
        accel = (new_speed - current_speed) / env->dt;
    }
    agent->sim_vx = new_speed * agent->cos_heading;
    agent->sim_vy = new_speed * agent->sin_heading;
    agent->yaw_rate = compute_heading_diff(agent->sim_heading, old_heading) / env->dt;
    agent->jerk_long = (accel - old_a_long) / env->dt;
    float new_a_lat = new_speed * agent->yaw_rate;
    agent->jerk_lat = (new_a_lat - agent->accel_lat) / env->dt;
    agent->accel_long = accel;
    agent->accel_lat = new_a_lat;
    agent->steering_angle = 0.0f;
    update_agent_speed(agent);
}

static void move_idm(Drive *env, int agent_idx) {
    IDMLeader leader = idm_find_leader_by_route_boxes(env, agent_idx);
    idm_move_with_leader(env, agent_idx, leader);
}

#endif
