#ifndef PUFFERLIB_OCEAN_DRIVE_IDM_H
#define PUFFERLIB_OCEAN_DRIVE_IDM_H

#define IDM_MINIMUM_LEAD_DISTANCE 0.1f
#define IDM_MIN_SPACING 2.0f
#define IDM_SAFE_TIME_HEADWAY 2.0f
#define IDM_MAX_ACCEL 2.0f
#define IDM_MAX_DECEL 4.0f
#define IDM_DELTA 4.0f
#define IDM_LOOKAHEAD_TIME 5.0f
#define IDM_MIN_LOOKAHEAD 20.0f
#define IDM_MAX_LOOKAHEAD 80.0f
#define IDM_BBOX_MARGIN 0.05f
#define IDM_DEFAULT_DESIRED_SPEED 15.0f

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
        if (traffic->states[env->timestep] != TRAFFIC_CONTROL_STATE_RED) {
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

static void move_idm(Drive *env, int agent_idx) {
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

    IDMLeader leader = idm_find_leader_by_corridor(env, agent_idx);
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
    agent->jerk_lat = (new_a_lat - agent->a_lat) / env->dt;
    agent->a_long = accel;
    agent->a_lat = new_a_lat;
    agent->steering_angle = 0.0f;
    update_agent_speed(agent);
}

#endif
