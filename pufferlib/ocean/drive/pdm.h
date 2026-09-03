#ifndef PUFFERLIB_OCEAN_DRIVE_PDM_H
#define PUFFERLIB_OCEAN_DRIVE_PDM_H

#define PDM_NUM_OFFSETS 3
#define PDM_NUM_SPEED_FRACTIONS 5
#define PDM_NUM_CANDIDATES (PDM_NUM_OFFSETS * PDM_NUM_SPEED_FRACTIONS)
#define PDM_DEFAULT_HORIZON 4.0f
#define PDM_DEFAULT_PLANNING_DT 0.5f
#define PDM_MIN_HORIZON 0.5f
#define PDM_MAX_HORIZON 8.0f
#define PDM_MIN_PLANNING_DT 0.1f
#define PDM_MAX_PLANNING_DT 1.0f
#define PDM_MAX_ROLLOUT_STEPS 81
#define PDM_DANGER_TTC 2.0f
#define PDM_DANGER_TTC_BUFFER 0.5f
#define PDM_SAFE_SPEED_TTC 5.0f
#define PDM_COLLISION_PENALTY 48.0f
#define PDM_SPEED_WEIGHT 3.0f
#define PDM_TTC_WEIGHT 10.0f
#define PDM_CENTER_BONUS 1.0f
#define PDM_ROUTE_EXTENSION_MARGIN 10.0f
#define PDM_STEERING_RATE_LIMIT 0.6f
#define PDM_PURSUIT_K 0.6f
#define PDM_PURSUIT_MIN 3.0f
#define PDM_PURSUIT_MAX 15.0f
#define PDM_MIN_SPACING 2.0f
#define PDM_SAFE_TIME_HEADWAY 2.0f
#define PDM_MAX_ACCEL ACCELERATION_VALUES[NUM_ACCELERATION_ACTIONS - 1]
#define PDM_MAX_DECEL 5.0f
#define PDM_URGENT_DECEL PDM_MAX_DECEL
#define PDM_DELTA 4.0f
#define PDM_LOOKAHEAD_TIME 5.0f
#define PDM_MIN_LOOKAHEAD 20.0f
#define PDM_MAX_LOOKAHEAD 120.0f
#define PDM_DEFAULT_DESIRED_SPEED 15.0f
#define PDM_MAX_AGENT_CANDIDATES 256
#define PDM_NUM_ROUTE_EXIT_CANDIDATES 8
#define PDM_ROUTE_PROJECTION_SEGMENT_RADIUS 8
#define PDM_AGENT_QUERY_MARGIN 5.0f
#define PDM_COLLISION_QUERY_MARGIN 10.0f
#define PDM_COLLISION_BBOX_MARGIN 0.05f
#define PDM_MIN_DESIRED_SPEED 1.0f
#define PDM_GEOMETRY_EPSILON 1e-6f
#define PDM_DISTANCE_EPSILON 1e-4f
#define PDM_SPEED_EPSILON 1e-3f

static const float PDM_OFFSETS[PDM_NUM_OFFSETS] = {0.0f, -1.0f, 1.0f};
static const float PDM_SPEED_FRACTIONS[PDM_NUM_SPEED_FRACTIONS] = {1.0f, 0.75f, 0.5f, 0.25f, 0.01f};

static inline float pdm_horizon_seconds(Drive *env) {
    float horizon = env->pdm_horizon_seconds > 0.0f ? env->pdm_horizon_seconds : PDM_DEFAULT_HORIZON;
    return clip(horizon, PDM_MIN_HORIZON, PDM_MAX_HORIZON);
}

static inline float pdm_speed_aware_ttc(float speed) {
    float stopping_ttc = speed / fmaxf(PDM_URGENT_DECEL, PDM_SPEED_EPSILON) + PDM_DANGER_TTC_BUFFER;
    return fmaxf(PDM_DANGER_TTC, stopping_ttc);
}

static inline float pdm_agent_horizon_seconds(Drive *env, const Agent *agent) {
    float speed = fmaxf(0.0f, agent->sim_speed_signed);
    return clip(fmaxf(pdm_horizon_seconds(env), pdm_speed_aware_ttc(speed)), PDM_MIN_HORIZON, PDM_MAX_HORIZON);
}

static inline float pdm_planning_dt_seconds(Drive *env) {
    float planning_dt = env->pdm_planning_dt_seconds > 0.0f ? env->pdm_planning_dt_seconds : PDM_DEFAULT_PLANNING_DT;
    planning_dt = clip(planning_dt, PDM_MIN_PLANNING_DT, PDM_MAX_PLANNING_DT);
    float min_dt_for_capacity = pdm_horizon_seconds(env) / (float) (PDM_MAX_ROLLOUT_STEPS - 1);
    return fmaxf(planning_dt, min_dt_for_capacity);
}

typedef struct {
    int valid;
    float t;
    float s;
    float x;
    float y;
    float z;
    float heading;
    float cos_heading;
    float sin_heading;
    float speed;
    float steering_angle;
    int lane_idx;
    int route_idx;
    int segment_idx;
    float segment_t;
    int center_lane_idx;
    float center_x;
    float center_y;
    float center_z;
    float center_heading;
} PDMRolloutStep;

typedef struct {
    int valid;
    int num_steps;
    PDMRolloutStep action_step;
    PDMRolloutStep steps[PDM_MAX_ROLLOUT_STEPS];
} PDMRollout;

typedef struct {
    float offset;
    float speed_fraction;
    float target_speed;
    float new_speed;
    float accel;
    float collision_ttc;
    float traffic_light_ttc;
    float offroad_ttc;
    float min_ttc;
    float score;
    int valid;
    PDMRollout rollout;
} PDMCandidateScore;

typedef struct {
    int length;
    int lanes[MAX_ROUTE_LENGTH];
    int source_route_indices[MAX_ROUTE_LENGTH];
} PDMPlanningRoute;

typedef struct {
    float x;
    float y;
    float z;
    float heading;
    float speed;
    float steering_angle;
} PDMTrackingState;

static IDMLaneProjection pdm_project_from_route_state(Drive *env, Agent *agent);

static float pdm_desired_speed(Drive *env, Agent *agent) {
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
        desired_speed = PDM_DEFAULT_DESIRED_SPEED;
    }

    return clip(desired_speed, PDM_MIN_DESIRED_SPEED, env->base_max_speed_mps);
}

static int pdm_collect_route_candidates(Drive *env, int ego_idx, float lookahead, int *candidates, int max_candidates) {
    Agent *ego = &env->agents[ego_idx];
    int count = 0;

    for (int i = 0; i < env->num_agents && count < max_candidates; i++) {
        int other_idx;
        if (i < env->active_agent_count) {
            other_idx = env->active_agent_indices[i];
        } else {
            other_idx = env->static_agent_indices[i - env->active_agent_count];
        }
        if (other_idx == ego_idx) {
            continue;
        }

        Agent *other = &env->agents[other_idx];
        if (other->removed || other->sim_x == INVALID_POSITION || other->sim_valid == 0) {
            continue;
        }
        float dx = other->sim_x - ego->sim_x;
        float dy = other->sim_y - ego->sim_y;
        float max_dist = lookahead + 0.5f * ego->sim_length + 0.5f * other->sim_length + PDM_AGENT_QUERY_MARGIN
            + 2.0f * IDM_BBOX_MARGIN;
        if (dx * dx + dy * dy > max_dist * max_dist) {
            continue;
        }

        candidates[count++] = other_idx;
    }

    return count;
}

static float pdm_compute_idm_acceleration(Drive *env, Agent *agent, float desired_speed, IDMLeader leader) {
    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    desired_speed = clip(desired_speed, PDM_MIN_DESIRED_SPEED, env->base_max_speed_mps);
    float speed_ratio = current_speed / desired_speed;
    float free_road_term = powf(speed_ratio, PDM_DELTA);
    float leader_term = 0.0f;

    if (leader.has_leader) {
        float s_star = PDM_MIN_SPACING
            + fmaxf(0.0f,
                    current_speed * PDM_SAFE_TIME_HEADWAY
                        + current_speed * (current_speed - leader.leader_speed)
                            / (2.0f * sqrtf(PDM_MAX_ACCEL * PDM_MAX_DECEL)));
        float lead_dist = fmaxf(leader.gap, IDM_MINIMUM_LEAD_DISTANCE);
        leader_term = (s_star / lead_dist) * (s_star / lead_dist);
    }

    return PDM_MAX_ACCEL * (1.0f - free_road_term - leader_term);
}

static float pdm_required_route_distance(Drive *env, const Agent *agent) {
    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    float max_next_speed = fminf(env->base_max_speed_mps, current_speed + PDM_MAX_ACCEL * env->dt);
    return pdm_agent_horizon_seconds(env, agent) * max_next_speed + PDM_ROUTE_EXTENSION_MARGIN;
}

static float pdm_remaining_route_distance(Drive *env, const PDMPlanningRoute *route, IDMLaneProjection projection) {
    if (!projection.valid || route->length <= 0) {
        return 0.0f;
    }

    float distance = 0.0f;
    for (int route_idx = projection.route_idx; route_idx < route->length; route_idx++) {
        int lane_idx = route->lanes[route_idx];
        if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
            break;
        }
        RoadMapElement *lane = &env->road_elements[lane_idx];
        if (lane->segment_size < 2) {
            break;
        }

        int start_seg = route_idx == projection.route_idx ? projection.segment_idx : 0;
        float start_t = route_idx == projection.route_idx ? projection.t : 0.0f;
        for (int seg_idx = start_seg; seg_idx < lane->segment_size - 1; seg_idx++) {
            float seg_len = idm_lane_segment_size(lane, seg_idx);
            if (seg_len < PDM_GEOMETRY_EPSILON) {
                continue;
            }
            float seg_fraction = seg_idx == start_seg ? 1.0f - clip(start_t, 0.0f, 1.0f) : 1.0f;
            distance += seg_fraction * seg_len;
        }
    }

    return distance;
}

static int pdm_route_contains_lane(const int *route, int route_length, int lane_idx) {
    for (int i = 0; i < route_length; i++) {
        if (route[i] == lane_idx) {
            return 1;
        }
    }
    return 0;
}

static float pdm_lane_end_distance_sq(const RoadMapElement *lane, float origin_x, float origin_y) {
    if (lane->segment_size <= 0) {
        return 0.0f;
    }
    int end_idx = lane->segment_size - 1;
    float dx = lane->x[end_idx] - origin_x;
    float dy = lane->y[end_idx] - origin_y;
    return dx * dx + dy * dy;
}

static int pdm_choose_route_extension_exit(
    Drive *env,
    RoadMapElement *current_lane,
    const int *route,
    int route_length,
    float origin_x,
    float origin_y,
    float *max_end_distance_sq) {
    int valid_exits[PDM_NUM_ROUTE_EXIT_CANDIDATES];
    float valid_exit_dist_sq[PDM_NUM_ROUTE_EXIT_CANDIDATES];
    int num_valid_exits = 0;
    int progressing_exits[PDM_NUM_ROUTE_EXIT_CANDIDATES];
    float progressing_dist_sq[PDM_NUM_ROUTE_EXIT_CANDIDATES];
    int num_progressing_exits = 0;

    for (int allow_revisit = 0; allow_revisit <= 1 && num_valid_exits == 0; allow_revisit++) {
        for (int exit_idx = 0; exit_idx < current_lane->num_exits && num_valid_exits < PDM_NUM_ROUTE_EXIT_CANDIDATES;
             exit_idx++) {
            int exit_lane_idx = current_lane->exit_lanes[exit_idx];
            if (exit_lane_idx < 0 || exit_lane_idx >= env->num_road_elements) {
                continue;
            }
            if (!allow_revisit && pdm_route_contains_lane(route, route_length, exit_lane_idx)) {
                continue;
            }

            float exit_end_distance_sq
                = pdm_lane_end_distance_sq(&env->road_elements[exit_lane_idx], origin_x, origin_y);
            valid_exits[num_valid_exits] = exit_lane_idx;
            valid_exit_dist_sq[num_valid_exits] = exit_end_distance_sq;
            num_valid_exits++;

            if (exit_end_distance_sq > *max_end_distance_sq) {
                progressing_exits[num_progressing_exits] = exit_lane_idx;
                progressing_dist_sq[num_progressing_exits] = exit_end_distance_sq;
                num_progressing_exits++;
            }
        }
    }

    if (num_valid_exits == 0) {
        return -1;
    }

    int chosen_exit_idx;
    float chosen_exit_dist_sq;
    if (num_progressing_exits > 0) {
        int chosen_idx = rng_below(&env->rng_state, num_progressing_exits);
        chosen_exit_idx = progressing_exits[chosen_idx];
        chosen_exit_dist_sq = progressing_dist_sq[chosen_idx];
    } else {
        int best_idx = 0;
        float best_dist_sq = valid_exit_dist_sq[0];
        for (int i = 1; i < num_valid_exits; i++) {
            if (valid_exit_dist_sq[i] > best_dist_sq) {
                best_dist_sq = valid_exit_dist_sq[i];
                best_idx = i;
            }
        }
        chosen_exit_idx = valid_exits[best_idx];
        chosen_exit_dist_sq = valid_exit_dist_sq[best_idx];
    }

    if (chosen_exit_dist_sq > *max_end_distance_sq) {
        *max_end_distance_sq = chosen_exit_dist_sq;
    }
    return chosen_exit_idx;
}

static int pdm_build_planning_route(Drive *env, Agent *agent, IDMLaneProjection *projection, PDMPlanningRoute *route) {
    *route = (PDMPlanningRoute) {0};
    if (agent->route == NULL || agent->route_length <= 0) {
        return 0;
    }

    if (!projection->valid) {
        return 0;
    }

    int dropped_prefix = projection->route_idx;
    if (dropped_prefix < 0 || dropped_prefix >= agent->route_length) {
        return 0;
    }

    for (int i = dropped_prefix; i < agent->route_length && route->length < MAX_ROUTE_LENGTH; i++) {
        route->lanes[route->length] = agent->route[i];
        route->source_route_indices[route->length] = i;
        route->length++;
    }

    if (route->length <= 0) {
        return 0;
    }

    projection->route_idx = 0;
    float required_distance = pdm_required_route_distance(env, agent);
    float remaining_distance = pdm_remaining_route_distance(env, route, *projection);
    if (remaining_distance >= required_distance || env->simulation_mode == SIMULATION_MODE_REPLAY) {
        return 1;
    }

    int current_lane_idx = route->lanes[route->length - 1];
    if (current_lane_idx < 0 || current_lane_idx >= env->num_road_elements) {
        return 0;
    }

    float max_end_distance_sq
        = pdm_lane_end_distance_sq(&env->road_elements[current_lane_idx], agent->sim_x, agent->sim_y);

    while (remaining_distance < required_distance && route->length < MAX_ROUTE_LENGTH) {
        RoadMapElement *current_lane = &env->road_elements[current_lane_idx];
        int next_lane_idx = pdm_choose_route_extension_exit(
            env,
            current_lane,
            route->lanes,
            route->length,
            agent->sim_x,
            agent->sim_y,
            &max_end_distance_sq);
        if (next_lane_idx == -1) {
            break;
        }

        route->lanes[route->length] = next_lane_idx;
        route->source_route_indices[route->length] = -1;
        route->length++;
        remaining_distance += env->road_elements[next_lane_idx].length;
        current_lane_idx = next_lane_idx;
    }

    return remaining_distance >= required_distance;
}

static IDMLaneProjection pdm_project_from_route_state(Drive *env, Agent *agent) {
    IDMLaneProjection best = {0};
    best.route_idx = 0;
    best.lane_idx = -1;
    best.segment_idx = 0;
    best.t = 0.0f;
    best.dist_sq = INFINITY;

    if (agent->route == NULL || agent->route_length <= 0) {
        return best;
    }

    int center_route_idx = agent->current_route_idx;
    if (center_route_idx < 0) {
        center_route_idx = 0;
    } else if (center_route_idx >= agent->route_length) {
        center_route_idx = agent->route_length - 1;
    }

    int center_seg_idx = agent->current_lane_geometry_idx;
    if (center_seg_idx < 0) {
        center_seg_idx = 0;
    }

    for (int route_idx = center_route_idx - 1; route_idx <= center_route_idx + 1; route_idx++) {
        if (route_idx < 0 || route_idx >= agent->route_length) {
            continue;
        }

        int lane_idx = agent->route[route_idx];
        if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
            continue;
        }
        RoadMapElement *lane = &env->road_elements[lane_idx];
        if (lane->segment_size < 2) {
            continue;
        }

        int start_seg = 0;
        int end_seg = lane->segment_size - 1;
        if (route_idx == center_route_idx) {
            start_seg = center_seg_idx - PDM_ROUTE_PROJECTION_SEGMENT_RADIUS;
            end_seg = center_seg_idx + PDM_ROUTE_PROJECTION_SEGMENT_RADIUS + 1;
            if (start_seg < 0) {
                start_seg = 0;
            }
            if (end_seg > lane->segment_size - 1) {
                end_seg = lane->segment_size - 1;
            }
        } else if (route_idx < center_route_idx) {
            int segment_window = PDM_ROUTE_PROJECTION_SEGMENT_RADIUS + 1;
            start_seg = lane->segment_size > segment_window ? lane->segment_size - segment_window : 0;
        } else {
            end_seg = lane->segment_size - 1 < PDM_ROUTE_PROJECTION_SEGMENT_RADIUS
                ? lane->segment_size - 1
                : PDM_ROUTE_PROJECTION_SEGMENT_RADIUS;
        }

        for (int seg_idx = start_seg; seg_idx < end_seg; seg_idx++) {
            float dx = lane->x[seg_idx + 1] - lane->x[seg_idx];
            float dy = lane->y[seg_idx + 1] - lane->y[seg_idx];
            float dz = lane->z[seg_idx + 1] - lane->z[seg_idx];
            float seg_len_sq = dx * dx + dy * dy + dz * dz;
            if (seg_len_sq < PDM_GEOMETRY_EPSILON) {
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

    if (!best.valid) {
        return idm_project_to_route_lanes(env, agent);
    }
    return best;
}

static int pdm_sample_offset_route_pose(
    Drive *env,
    const PDMPlanningRoute *route,
    IDMLaneProjection projection,
    float distance,
    float offset,
    PDMRolloutStep *out) {
    if (!projection.valid || route->length <= 0) {
        return 0;
    }

    float traveled_s = 0.0f;
    int route_idx = projection.route_idx;
    int seg_idx = projection.segment_idx;
    float t = projection.t;

    while (route_idx < route->length) {
        int lane_idx = route->lanes[route_idx];
        if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
            return 0;
        }
        RoadMapElement *lane = &env->road_elements[lane_idx];
        if (lane->segment_size < 2) {
            return 0;
        }

        while (seg_idx < lane->segment_size - 1) {
            float seg_len = idm_lane_segment_size(lane, seg_idx);
            if (seg_len < PDM_GEOMETRY_EPSILON) {
                seg_idx++;
                t = 0.0f;
                continue;
            }

            float remaining = (1.0f - t) * seg_len;
            if (traveled_s + remaining + PDM_DISTANCE_EPSILON < distance) {
                traveled_s += remaining;
                seg_idx++;
                t = 0.0f;
                continue;
            }

            float sample_t = t + (distance - traveled_s) / seg_len;
            sample_t = clip(sample_t, 0.0f, 1.0f);
            float heading = normalize_heading(lane->headings[seg_idx]);
            float cos_heading = cosf(heading);
            float sin_heading = sinf(heading);
            float center_x = lane->x[seg_idx] + sample_t * (lane->x[seg_idx + 1] - lane->x[seg_idx]);
            float center_y = lane->y[seg_idx] + sample_t * (lane->y[seg_idx + 1] - lane->y[seg_idx]);

            out->valid = 1;
            out->s = distance;
            out->x = center_x - sin_heading * offset;
            out->y = center_y + cos_heading * offset;
            out->z = lane->z[seg_idx] + sample_t * (lane->z[seg_idx + 1] - lane->z[seg_idx]);
            out->heading = heading;
            out->cos_heading = cos_heading;
            out->sin_heading = sin_heading;
            out->lane_idx = lane_idx;
            out->route_idx = route->source_route_indices[route_idx];
            out->segment_idx = seg_idx;
            out->segment_t = sample_t;
            out->center_lane_idx = lane_idx;
            out->center_x = center_x;
            out->center_y = center_y;
            out->center_z = out->z;
            out->center_heading = heading;
            return 1;
        }

        route_idx++;
        seg_idx = 0;
        t = 0.0f;
    }

    return 0;
}

static PDMTrackingState pdm_initial_tracking_state(const Agent *agent) {
    return (PDMTrackingState) {
        .x = agent->sim_x,
        .y = agent->sim_y,
        .z = agent->sim_z,
        .heading = agent->sim_heading,
        .speed = fmaxf(0.0f, agent->sim_speed_signed),
        .steering_angle = agent->steering_angle,
    };
}

static PDMRolloutStep pdm_tracking_state_to_step(PDMTrackingState state, PDMRolloutStep target, float t, float s) {
    PDMRolloutStep step = target;
    step.valid = 1;
    step.t = t;
    step.s = s;
    step.x = state.x;
    step.y = state.y;
    step.z = state.z;
    step.heading = normalize_heading(state.heading);
    step.cos_heading = cosf(step.heading);
    step.sin_heading = sinf(step.heading);
    step.speed = state.speed;
    step.steering_angle = state.steering_angle;
    return step;
}

// Sample a pure-pursuit aim point a speed-scaled lookahead ahead of progress distance s.
// Falls back to the progress point near the route end so it never fails when s itself is valid.
static int pdm_sample_pursuit_aim(
    Drive *env,
    const PDMPlanningRoute *route,
    IDMLaneProjection projection,
    float offset,
    float s,
    float speed,
    PDMRolloutStep *aim) {
    float ld = clip(PDM_PURSUIT_K * speed, PDM_PURSUIT_MIN, PDM_PURSUIT_MAX);
    if (pdm_sample_offset_route_pose(env, route, projection, s + ld, offset, aim)) {
        return 1;
    }
    return pdm_sample_offset_route_pose(env, route, projection, s, offset, aim);
}

static PDMTrackingState pdm_track_target_step(
    Drive *env,
    const Agent *agent,
    PDMTrackingState state,
    PDMRolloutStep aim,
    float target_speed,
    float dt) {
    (void) env;
    if (dt <= 0.0f) {
        return state;
    }

    float speed = fmaxf(0.0f, state.speed);
    float accel = (target_speed - speed) / dt;
    accel = clip(accel, -PDM_URGENT_DECEL, PDM_MAX_ACCEL);
    float new_speed = clip(speed + accel * dt, 0.0f, env->base_max_speed_mps);

    // Geometric pure-pursuit steering toward a lookahead aim point: curvature = 2 sin(alpha) / L_d.
    // Speed-stable, unlike "null all heading error within one dt" which is unstable at short lookahead.
    float dx = aim.x - state.x;
    float dy = aim.y - state.y;
    float ld = sqrtf(dx * dx + dy * dy);
    float steering = 0.0f;
    if (ld > PDM_SPEED_EPSILON) {
        float bearing = atan2f(dy, dx);
        float alpha = compute_heading_diff(bearing, state.heading);
        float curvature = 2.0f * sinf(alpha) / ld;
        steering = atanf(curvature * agent->wheelbase);
    }

    float max_steering = STEERING_VALUES[NUM_STEERING_ACTIONS - 1];
    float delta_steer
        = clip(steering - state.steering_angle, -PDM_STEERING_RATE_LIMIT * dt, PDM_STEERING_RATE_LIMIT * dt);
    steering = clip(state.steering_angle + delta_steer, -max_steering, max_steering);

    float beta = atanf(REAR_AXLE_RATIO * tanf(steering));
    float yaw_rate = 0.0f;
    if (new_speed > PDM_SPEED_EPSILON) {
        yaw_rate = (new_speed * cosf(beta) * tanf(steering)) / agent->wheelbase;
    }

    state.x += new_speed * cosf(state.heading + beta) * dt;
    state.y += new_speed * sinf(state.heading + beta) * dt;
    state.z = aim.z;
    state.heading = normalize_heading(state.heading + yaw_rate * dt);
    state.speed = new_speed;
    state.steering_angle = steering;
    return state;
}

// Speed/progress at time t under constant accel toward target_speed, then holding it.
// This lets braking candidates actually slow over the horizon instead of holding the
// one-step-ahead speed, so PDM can evaluate slowing down.
static void pdm_speed_profile(
    float current,
    float accel,
    float target,
    float t,
    float max_speed,
    float *v_out,
    float *s_out) {
    float t_reach = INFINITY;
    if (fabsf(accel) > PDM_GEOMETRY_EPSILON) {
        float tr = (target - current) / accel;
        if (tr > 0.0f) {
            t_reach = tr;
        }
    }

    float v;
    float s;
    if (t <= t_reach) {
        v = current + accel * t;
        s = current * t + 0.5f * accel * t * t;
    } else {
        s = current * t_reach + 0.5f * accel * t_reach * t_reach + target * (t - t_reach);
        v = target;
    }
    *v_out = clip(v, 0.0f, max_speed);
    *s_out = fmaxf(s, 0.0f);
}

static PDMRollout pdm_generate_rollout(
    Drive *env,
    Agent *agent,
    const PDMPlanningRoute *route,
    IDMLaneProjection projection,
    float offset,
    float current_speed,
    float accel,
    float target_speed) {
    PDMRollout rollout = {0};
    float new_speed = clip(current_speed + accel * env->dt, 0.0f, env->base_max_speed_mps);
    PDMRolloutStep start = {0};
    if (!pdm_sample_offset_route_pose(env, route, projection, 0.0f, offset, &start)) {
        return rollout;
    }
    rollout.valid = 1;

    // Applied first step is unchanged: one dt at new_speed.
    PDMTrackingState action_state = pdm_initial_tracking_state(agent);
    PDMRolloutStep action_target = {0};
    PDMRolloutStep action_aim = {0};
    if (!pdm_sample_offset_route_pose(env, route, projection, new_speed * env->dt, offset, &action_target)
        || !pdm_sample_pursuit_aim(env, route, projection, offset, new_speed * env->dt, new_speed, &action_aim)) {
        rollout.valid = 0;
        return rollout;
    }
    action_state = pdm_track_target_step(env, agent, action_state, action_aim, new_speed, env->dt);
    rollout.action_step = pdm_tracking_state_to_step(action_state, action_target, env->dt, new_speed * env->dt);

    float horizon = pdm_agent_horizon_seconds(env, agent);
    float planning_dt = pdm_planning_dt_seconds(env);
    PDMTrackingState state = pdm_initial_tracking_state(agent);
    float prev_t = 0.0f;

    for (int step = 0; step < PDM_MAX_ROLLOUT_STEPS; step++) {
        float t = step * planning_dt;
        if (t > horizon + PDM_DISTANCE_EPSILON) {
            break;
        }

        float v;
        float s;
        pdm_speed_profile(current_speed, accel, target_speed, t, env->base_max_speed_mps, &v, &s);

        PDMRolloutStep target_step = {0};
        if (!pdm_sample_offset_route_pose(env, route, projection, s, offset, &target_step)) {
            if (env->simulation_mode != SIMULATION_MODE_REPLAY) {
                rollout.valid = 0;
            }
            break;
        }

        if (step > 0) {
            PDMRolloutStep aim_step = {0};
            if (!pdm_sample_pursuit_aim(env, route, projection, offset, s, v, &aim_step)) {
                rollout.valid = 0;
                break;
            }
            state = pdm_track_target_step(env, agent, state, aim_step, v, t - prev_t);
        }
        PDMRolloutStep rollout_step = pdm_tracking_state_to_step(state, target_step, t, s);
        rollout.steps[rollout.num_steps++] = rollout_step;
        prev_t = t;
    }

    return rollout;
}

static int pdm_sample_hits_agent(const Agent *sample, Agent *other) {
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

    Agent other_expanded = *other;
    Agent sample_expanded = *sample;
    sample_expanded.sim_length = sample->sim_length + 2.0f * PDM_COLLISION_BBOX_MARGIN;
    sample_expanded.sim_width = sample->sim_width + 2.0f * PDM_COLLISION_BBOX_MARGIN;
    other_expanded.sim_length = other->sim_length + 2.0f * (IDM_BBOX_MARGIN + PDM_COLLISION_BBOX_MARGIN);
    other_expanded.sim_width = other->sim_width + 2.0f * (IDM_BBOX_MARGIN + PDM_COLLISION_BBOX_MARGIN);
    return check_obb_collision(&sample_expanded, &other_expanded);
}

static IDMLeader pdm_find_leader_by_offset_route_boxes(
    Drive *env,
    int agent_idx,
    const PDMPlanningRoute *route,
    IDMLaneProjection projection,
    float offset) {
    Agent *agent = &env->agents[agent_idx];
    IDMLeader no_leader = idm_no_leader();
    if (!projection.valid) {
        return no_leader;
    }

    float speed = fmaxf(0.0f, agent->sim_speed_signed);
    float lookahead = clip(speed * PDM_LOOKAHEAD_TIME, PDM_MIN_LOOKAHEAD, PDM_MAX_LOOKAHEAD);
    int candidates[PDM_MAX_AGENT_CANDIDATES];
    int num_candidates = pdm_collect_route_candidates(env, agent_idx, lookahead, candidates, PDM_MAX_AGENT_CANDIDATES);

    for (float sample_s = IDM_ROUTE_SAMPLE_DS; sample_s <= lookahead + PDM_DISTANCE_EPSILON;
         sample_s += IDM_ROUTE_SAMPLE_DS) {
        PDMRolloutStep sample_step = {0};
        if (!pdm_sample_offset_route_pose(env, route, projection, sample_s, offset, &sample_step)) {
            break;
        }

        Agent sample = idm_make_sample_agent(agent, sample_step.x, sample_step.y, sample_step.z, sample_step.heading);
        if (idm_sample_hits_red_light(env, &sample, sample_step.lane_idx)) {
            idm_update_best_leader(&no_leader, -1, 1, sample_s, 0.0f);
            return no_leader;
        }

        IDMLeader best_at_sample = idm_no_leader();
        for (int i = 0; i < num_candidates; i++) {
            int other_idx = candidates[i];
            Agent *other = &env->agents[other_idx];
            if (!pdm_sample_hits_agent(&sample, other)) {
                continue;
            }
            float leader_speed = other->sim_vx * sample.cos_heading + other->sim_vy * sample.sin_heading;
            idm_update_best_leader(&best_at_sample, other_idx, 0, sample_s, leader_speed);
        }
        if (best_at_sample.has_leader) {
            return best_at_sample;
        }
    }

    return no_leader;
}

static Agent pdm_predict_other_agent(const Agent *other, float t) {
    Agent predicted = *other;
    predicted.sim_x = other->sim_x + other->sim_vx * t;
    predicted.sim_y = other->sim_y + other->sim_vy * t;
    return predicted;
}

static int pdm_agent_is_ahead_of_rear_bumper(const Agent *ego_sample, const Agent *other) {
    float dx = other->sim_x - ego_sample->sim_x;
    float dy = other->sim_y - ego_sample->sim_y;
    float rel_x = dx * ego_sample->cos_heading + dy * ego_sample->sin_heading;
    float other_forward_projection
        = fabsf(other->cos_heading * ego_sample->cos_heading + other->sin_heading * ego_sample->sin_heading);
    float other_lateral_projection
        = fabsf(-other->sin_heading * ego_sample->cos_heading + other->cos_heading * ego_sample->sin_heading);
    float other_half_extent
        = 0.5f * other->sim_length * other_forward_projection + 0.5f * other->sim_width * other_lateral_projection;
    return rel_x + other_half_extent >= -0.5f * ego_sample->sim_length;
}

static float pdm_compute_collision_ttc(Drive *env, int agent_idx, PDMCandidateScore *candidate) {
    if (!candidate->valid || !candidate->rollout.valid) {
        return 0.0f;
    }

    Agent *agent = &env->agents[agent_idx];
    float horizon = pdm_agent_horizon_seconds(env, agent);
    float max_distance = candidate->new_speed * horizon + 0.5f * agent->sim_length + PDM_COLLISION_QUERY_MARGIN;
    int candidates[PDM_MAX_AGENT_CANDIDATES];
    int num_candidates
        = pdm_collect_route_candidates(env, agent_idx, max_distance, candidates, PDM_MAX_AGENT_CANDIDATES);
    int frontal_candidates[PDM_MAX_AGENT_CANDIDATES];
    int num_frontal_candidates = 0;
    for (int i = 0; i < num_candidates; i++) {
        Agent *other = &env->agents[candidates[i]];
        if (!pdm_agent_is_ahead_of_rear_bumper(agent, other)) {
            continue;
        }
        frontal_candidates[num_frontal_candidates++] = candidates[i];
    }

    for (int step_idx = 1; step_idx < candidate->rollout.num_steps; step_idx++) {
        PDMRolloutStep step = candidate->rollout.steps[step_idx];
        Agent sample = idm_make_sample_agent(agent, step.x, step.y, step.z, step.heading);

        for (int i = 0; i < num_frontal_candidates; i++) {
            Agent other = pdm_predict_other_agent(&env->agents[frontal_candidates[i]], step.t);
            if (pdm_sample_hits_agent(&sample, &other)) {
                return step.t;
            }
        }
    }

    return horizon;
}

static float pdm_compute_traffic_light_ttc(Drive *env, const Agent *agent, PDMCandidateScore *candidate) {
    if (!candidate->valid || !candidate->rollout.valid) {
        return 0.0f;
    }

    for (int step_idx = 1; step_idx < candidate->rollout.num_steps; step_idx++) {
        PDMRolloutStep step = candidate->rollout.steps[step_idx];
        Agent sample = idm_make_sample_agent(agent, step.x, step.y, step.z, step.heading);
        if (idm_sample_hits_red_light(env, &sample, step.lane_idx)) {
            return step.t;
        }
    }

    return pdm_agent_horizon_seconds(env, agent);
}

static int pdm_sample_is_offroad(Drive *env, const Agent *agent, PDMRolloutStep step) {
    if (get_grid_index(env, step.x, step.y) == -1) {
        return 1;
    }

    Agent sample = *agent;
    sample.sim_x = step.x;
    sample.sim_y = step.y;
    sample.sim_z = step.z;
    sample.cos_heading = step.cos_heading;
    sample.sin_heading = step.sin_heading;
    sample.prev_x = step.x;
    sample.prev_y = step.y;
    sample.prev_cos_heading = step.cos_heading;
    sample.prev_sin_heading = step.sin_heading;

    GridMapEntity entity_list[ROAD_QUERY_ENTITY_COUNT];
    int list_size = get_neighbors_entities(env, step.x, step.y, entity_list, ROAD_QUERY_ENTITY_COUNT, ROAD_OFFSETS, 25);
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_idx == -1) {
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
        if (geometry_idx < 0 || geometry_idx >= element->segment_size - 1) {
            continue;
        }

        float abs_dz = fabsf(element->z[geometry_idx] - step.z);
        if (abs_dz > Z_BUFFER) {
            continue;
        }

        if (check_segment_crosses_moving_box(
                element->x[geometry_idx],
                element->y[geometry_idx],
                element->x[geometry_idx + 1],
                element->y[geometry_idx + 1],
                &sample)) {
            return 1;
        }
    }

    return 0;
}

static float pdm_compute_offroad_ttc(Drive *env, int agent_idx, PDMCandidateScore *candidate) {
    if (!candidate->valid || !candidate->rollout.valid) {
        return 0.0f;
    }

    Agent *agent = &env->agents[agent_idx];
    for (int step_idx = 1; step_idx < candidate->rollout.num_steps; step_idx++) {
        PDMRolloutStep step = candidate->rollout.steps[step_idx];
        if (pdm_sample_is_offroad(env, agent, step)) {
            return step.t;
        }
    }

    return pdm_agent_horizon_seconds(env, agent);
}

static void pdm_score_candidate(Drive *env, int agent_idx, PDMCandidateScore *candidate, float speed_limit) {
    if (!candidate->valid) {
        candidate->score = -INFINITY;
        candidate->collision_ttc = 0.0f;
        candidate->traffic_light_ttc = 0.0f;
        candidate->offroad_ttc = 0.0f;
        candidate->min_ttc = 0.0f;
        return;
    }

    Agent *agent = &env->agents[agent_idx];
    float horizon = pdm_agent_horizon_seconds(env, agent);
    candidate->collision_ttc = pdm_compute_collision_ttc(env, agent_idx, candidate);
    candidate->traffic_light_ttc = pdm_compute_traffic_light_ttc(env, agent, candidate);
    candidate->offroad_ttc = INFINITY;
    candidate->min_ttc = fminf(candidate->collision_ttc, candidate->traffic_light_ttc);

    float ttc_score = clip(candidate->min_ttc / horizon, 0.0f, 1.0f);
    float speed_score = clip(candidate->new_speed / fmaxf(speed_limit, PDM_MIN_DESIRED_SPEED), 0.0f, 1.0f);
    if (candidate->min_ttc < horizon && candidate->min_ttc <= PDM_SAFE_SPEED_TTC) {
        speed_score = 0.0f;
    }

    float danger_ttc = fminf(pdm_speed_aware_ttc(fmaxf(0.0f, agent->sim_speed_signed)), horizon);
    float collision_penalty = (candidate->min_ttc < danger_ttc) ? PDM_COLLISION_PENALTY : 0.0f;
    float center_bonus = fabsf(candidate->offset) < PDM_DISTANCE_EPSILON ? PDM_CENTER_BONUS : 0.0f;
    candidate->score = -collision_penalty + PDM_SPEED_WEIGHT * speed_score + PDM_TTC_WEIGHT * ttc_score + center_bonus;
}

static int pdm_build_candidates(Drive *env, int agent_idx, PDMCandidateScore *candidates, int max_candidates) {
    Agent *agent = &env->agents[agent_idx];
    int count = 0;
    float speed_limit = pdm_desired_speed(env, agent);
    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    float horizon = pdm_agent_horizon_seconds(env, agent);
    IDMLaneProjection projection = pdm_project_from_route_state(env, agent);
    PDMPlanningRoute planning_route = {0};
    IDMLeader offset_leaders[PDM_NUM_OFFSETS];

    if (!projection.valid || !pdm_build_planning_route(env, agent, &projection, &planning_route)) {
        return 0;
    }

    for (int offset_idx = 0; offset_idx < PDM_NUM_OFFSETS; offset_idx++) {
        offset_leaders[offset_idx] = pdm_find_leader_by_offset_route_boxes(
            env,
            agent_idx,
            &planning_route,
            projection,
            PDM_OFFSETS[offset_idx]);
    }

    for (int offset_idx = 0; offset_idx < PDM_NUM_OFFSETS; offset_idx++) {
        for (int speed_idx = 0; speed_idx < PDM_NUM_SPEED_FRACTIONS; speed_idx++) {
            if (count >= max_candidates) {
                return count;
            }

            float speed_fraction = PDM_SPEED_FRACTIONS[speed_idx];
            float target_speed = speed_fraction * speed_limit;
            float accel = pdm_compute_idm_acceleration(env, agent, target_speed, offset_leaders[offset_idx]);
            accel = clip(accel, -PDM_MAX_DECEL, PDM_MAX_ACCEL);

            float new_speed = current_speed + accel * env->dt;
            if (new_speed < 0.0f) {
                new_speed = 0.0f;
            }
            accel = (new_speed - current_speed) / env->dt;

            PDMRollout rollout = pdm_generate_rollout(
                env,
                agent,
                &planning_route,
                projection,
                PDM_OFFSETS[offset_idx],
                current_speed,
                accel,
                target_speed);
            candidates[count++] = (PDMCandidateScore) {
                .offset = PDM_OFFSETS[offset_idx],
                .speed_fraction = speed_fraction,
                .target_speed = target_speed,
                .new_speed = new_speed,
                .accel = accel,
                .collision_ttc = horizon,
                .traffic_light_ttc = horizon,
                .offroad_ttc = INFINITY,
                .min_ttc = horizon,
                .score = -INFINITY,
                .valid = rollout.valid,
                .rollout = rollout,
            };
            pdm_score_candidate(env, agent_idx, &candidates[count - 1], speed_limit);
        }
    }

    return count;
}

static PDMCandidateScore pdm_select_best_candidate(PDMCandidateScore *candidates, int num_candidates) {
    PDMCandidateScore best = {0};
    best.valid = 0;
    best.score = -INFINITY;

    for (int i = 0; i < num_candidates; i++) {
        if (!candidates[i].valid) {
            continue;
        }
        if (!best.valid || candidates[i].score > best.score) {
            best = candidates[i];
        }
    }

    return best;
}

static PDMCandidateScore pdm_select_best_feasible_candidate(
    Drive *env,
    int agent_idx,
    PDMCandidateScore *candidates,
    int num_candidates) {
    int checked[PDM_NUM_CANDIDATES] = {0};
    Agent *agent = &env->agents[agent_idx];
    float horizon = pdm_agent_horizon_seconds(env, agent);
    PDMCandidateScore best_offroad = {0};
    best_offroad.valid = 0;
    best_offroad.score = -INFINITY;
    best_offroad.offroad_ttc = 0.0f;

    for (int iter = 0; iter < num_candidates; iter++) {
        int best_idx = -1;
        float best_score = -INFINITY;
        for (int i = 0; i < num_candidates; i++) {
            if (checked[i] || !candidates[i].valid) {
                continue;
            }
            if (best_idx == -1 || candidates[i].score > best_score) {
                best_idx = i;
                best_score = candidates[i].score;
            }
        }

        if (best_idx == -1) {
            break;
        }
        checked[best_idx] = 1;

        candidates[best_idx].offroad_ttc = pdm_compute_offroad_ttc(env, agent_idx, &candidates[best_idx]);
        candidates[best_idx].min_ttc = fminf(candidates[best_idx].min_ttc, candidates[best_idx].offroad_ttc);
        if (candidates[best_idx].offroad_ttc >= horizon) {
            return candidates[best_idx];
        }

        if (!best_offroad.valid || candidates[best_idx].offroad_ttc > best_offroad.offroad_ttc
            || (candidates[best_idx].offroad_ttc == best_offroad.offroad_ttc
                && candidates[best_idx].score > best_offroad.score)) {
            best_offroad = candidates[best_idx];
        }
    }

    return best_offroad.valid ? best_offroad : pdm_select_best_candidate(candidates, num_candidates);
}

static int pdm_build_urgent_action_step(
    Drive *env,
    int agent_idx,
    float offset,
    float urgent_speed,
    PDMRolloutStep *out) {
    Agent *agent = &env->agents[agent_idx];
    *out = (PDMRolloutStep) {0};

    IDMLaneProjection projection = pdm_project_from_route_state(env, agent);
    PDMPlanningRoute planning_route = {0};
    if (!projection.valid || !pdm_build_planning_route(env, agent, &projection, &planning_route)) {
        return 0;
    }

    PDMRolloutStep target = {0};
    PDMRolloutStep aim = {0};
    if (!pdm_sample_offset_route_pose(env, &planning_route, projection, urgent_speed * env->dt, offset, &target)
        || !pdm_sample_pursuit_aim(
            env,
            &planning_route,
            projection,
            offset,
            urgent_speed * env->dt,
            urgent_speed,
            &aim)) {
        return 0;
    }

    PDMTrackingState state = pdm_initial_tracking_state(agent);
    state = pdm_track_target_step(env, agent, state, aim, urgent_speed, env->dt);
    *out = pdm_tracking_state_to_step(state, target, env->dt, urgent_speed * env->dt);
    return out->valid;
}

static void pdm_apply_speed_along_route_or_heading(Drive *env, int agent_idx, float new_speed) {
    Agent *agent = &env->agents[agent_idx];
    float old_a_long = agent->accel_long;
    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    new_speed = fmaxf(0.0f, new_speed);
    float accel = (new_speed - current_speed) / env->dt;
    float old_heading = agent->sim_heading;
    float distance = new_speed * env->dt;

    if (!idm_advance_along_route_lanes(env, agent_idx, distance, &old_heading)) {
        agent->sim_x += distance * agent->cos_heading;
        agent->sim_y += distance * agent->sin_heading;
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

static void pdm_apply_urgent_brake_fallback(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    float new_speed = fmaxf(0.0f, current_speed - PDM_URGENT_DECEL * env->dt);
    pdm_apply_speed_along_route_or_heading(env, agent_idx, new_speed);
}

static void pdm_apply_tracked_step(Drive *env, int agent_idx, PDMCandidateScore candidate) {
    Agent *agent = &env->agents[agent_idx];
    float old_a_long = agent->accel_long;
    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    float new_speed = candidate.new_speed;
    if (new_speed < 0.0f) {
        new_speed = 0.0f;
    }
    float accel = (new_speed - current_speed) / env->dt;

    float old_heading = agent->sim_heading;
    if (candidate.rollout.action_step.valid) {
        PDMRolloutStep step = candidate.rollout.action_step;
        agent->sim_x = step.x;
        agent->sim_y = step.y;
        agent->sim_z = step.z;
        agent->sim_heading = normalize_heading(step.heading);
        agent->cos_heading = cosf(agent->sim_heading);
        agent->sin_heading = sinf(agent->sim_heading);
        new_speed = fmaxf(0.0f, step.speed);
        accel = (new_speed - current_speed) / env->dt;
        agent->steering_angle = step.steering_angle;
        if (step.route_idx >= 0 && step.route_idx < agent->route_length) {
            agent->current_route_idx = step.route_idx;
        }
        if (step.center_lane_idx >= 0) {
            agent->current_lane_idx = step.center_lane_idx;
        } else if (step.lane_idx >= 0) {
            agent->current_lane_idx = step.lane_idx;
        }
        if (step.segment_idx >= 0) {
            agent->current_lane_geometry_idx = step.segment_idx;
        }
    } else {
        pdm_apply_speed_along_route_or_heading(env, agent_idx, new_speed);
        return;
    }

    agent->sim_vx = new_speed * agent->cos_heading;
    agent->sim_vy = new_speed * agent->sin_heading;
    agent->yaw_rate = compute_heading_diff(agent->sim_heading, old_heading) / env->dt;
    agent->jerk_long = (accel - old_a_long) / env->dt;
    float new_a_lat = new_speed * agent->yaw_rate;
    agent->jerk_lat = (new_a_lat - agent->accel_lat) / env->dt;
    agent->accel_long = accel;
    agent->accel_lat = new_a_lat;
    update_agent_speed(agent);
}

static void move_pdm(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    copy_pose_to_prev(agent);

    if (agent->removed) {
        invalidate_agent(agent);
        return;
    }

    if (agent->stopped || agent->sim_x == INVALID_POSITION) {
        clear_agent_motion(agent);
        agent->steering_angle = 0.0f;
        return;
    }

    PDMCandidateScore candidates[PDM_NUM_CANDIDATES];
    int num_candidates = pdm_build_candidates(env, agent_idx, candidates, PDM_NUM_CANDIDATES);
    PDMCandidateScore best = pdm_select_best_feasible_candidate(env, agent_idx, candidates, num_candidates);
    if (!best.valid) {
        pdm_apply_urgent_brake_fallback(env, agent_idx);
        return;
    }

    float danger_ttc
        = fminf(pdm_speed_aware_ttc(fmaxf(0.0f, agent->sim_speed_signed)), pdm_agent_horizon_seconds(env, agent));
    if (best.min_ttc < danger_ttc) {
        float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
        float urgent_speed = fmaxf(0.0f, current_speed - PDM_URGENT_DECEL * env->dt);
        PDMRolloutStep urgent_step = {0};
        if (pdm_build_urgent_action_step(env, agent_idx, best.offset, urgent_speed, &urgent_step)) {
            best.new_speed = urgent_speed;
            best.accel = (urgent_speed - current_speed) / env->dt;
            best.rollout.action_step = urgent_step;
        } else {
            pdm_apply_urgent_brake_fallback(env, agent_idx);
            return;
        }
    }

    pdm_apply_tracked_step(env, agent_idx, best);
}

#endif
