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
#define PDM_URGENT_DECEL 8.0f
#define PDM_SAFE_SPEED_TTC 5.0f
#define PDM_COLLISION_PENALTY 48.0f
#define PDM_SPEED_WEIGHT 3.0f
#define PDM_TTC_WEIGHT 10.0f
#define PDM_CENTER_BONUS 1.0f
#define PDM_BEZIER_SAMPLES 32
#define PDM_ALIGNMENT_FACTOR 0.5f
#define PDM_CURVATURE_FACTOR 0.5f
#define PDM_MERGE_H_MIN 0.3f
#define PDM_MERGE_D_BASE 5.0f
#define PDM_MERGE_K_V 0.5f
#define PDM_ROUTE_EXTENSION_MARGIN 10.0f
#define PDM_STEERING_RATE_LIMIT 0.6f

static const float PDM_OFFSETS[PDM_NUM_OFFSETS] = {0.0f, -1.0f, 1.0f};
static const float PDM_SPEED_FRACTIONS[PDM_NUM_SPEED_FRACTIONS] = {1.0f, 0.8f, 0.6f, 0.4f, 0.2f};

static inline float pdm_horizon(Drive *env) {
    float horizon = env->pdm_horizon > 0.0f ? env->pdm_horizon : PDM_DEFAULT_HORIZON;
    return clip(horizon, PDM_MIN_HORIZON, PDM_MAX_HORIZON);
}

static inline float pdm_speed_aware_ttc(float speed) {
    float stopping_ttc = speed / fmaxf(PDM_URGENT_DECEL, 1e-3f) + PDM_DANGER_TTC_BUFFER;
    return fmaxf(PDM_DANGER_TTC, stopping_ttc);
}

static inline float pdm_agent_horizon(Drive *env, const Agent *agent) {
    float speed = agent != NULL ? fmaxf(0.0f, agent->sim_speed_signed) : 0.0f;
    return clip(fmaxf(pdm_horizon(env), pdm_speed_aware_ttc(speed)), PDM_MIN_HORIZON, PDM_MAX_HORIZON);
}

static inline float pdm_planning_dt(Drive *env) {
    float planning_dt = env->pdm_planning_dt > 0.0f ? env->pdm_planning_dt : PDM_DEFAULT_PLANNING_DT;
    planning_dt = clip(planning_dt, PDM_MIN_PLANNING_DT, PDM_MAX_PLANNING_DT);
    float min_dt_for_capacity = pdm_horizon(env) / (float)(PDM_MAX_ROLLOUT_STEPS - 1);
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
    int valid;
    int num_points;
    float x[PDM_BEZIER_SAMPLES];
    float y[PDM_BEZIER_SAMPLES];
    float z[PDM_BEZIER_SAMPLES];
    float heading[PDM_BEZIER_SAMPLES];
    float arc_lengths[PDM_BEZIER_SAMPLES];
    float merge_route_s;
    float bezier_length;
} PDMBezierPath;

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

static float pdm_compute_idm_acceleration(Agent *agent, float desired_speed, IDMLeader leader) {
    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    desired_speed = clip(desired_speed, 1.0f, MAX_SPEED);
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

static float pdm_required_route_distance(Drive *env, const Agent *agent) {
    float current_speed = agent != NULL ? fmaxf(0.0f, agent->sim_speed_signed) : 0.0f;
    float max_next_speed = fminf(MAX_SPEED, current_speed + IDM_MAX_ACCEL * env->dt);
    return pdm_agent_horizon(env, agent) * max_next_speed + PDM_ROUTE_EXTENSION_MARGIN;
}

static float pdm_remaining_route_distance(Drive *env, const PDMPlanningRoute *route, IDMLaneProjection projection) {
    if (!projection.valid || route == NULL || route->length <= 0) {
        return 0.0f;
    }

    float distance = 0.0f;
    for (int route_idx = projection.route_idx; route_idx < route->length; route_idx++) {
        int lane_idx = route->lanes[route_idx];
        if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
            break;
        }
        RoadMapElement *lane = &env->road_elements[lane_idx];
        if (lane->segment_length < 2) {
            break;
        }

        int start_seg = route_idx == projection.route_idx ? projection.segment_idx : 0;
        float start_t = route_idx == projection.route_idx ? projection.t : 0.0f;
        for (int seg_idx = start_seg; seg_idx < lane->segment_length - 1; seg_idx++) {
            float seg_len = idm_lane_segment_length(lane, seg_idx);
            if (seg_len < 1e-6f) {
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

static int pdm_choose_route_extension_exit(Drive *env, RoadMapElement *current_lane, const int *route, int route_length,
                                           float origin_x, float origin_y, float *max_end_distance_sq) {
    int valid_exits[8];
    float valid_exit_dist_sq[8];
    int num_valid_exits = 0;
    int progressing_exits[8];
    float progressing_dist_sq[8];
    int num_progressing_exits = 0;

    for (int allow_revisit = 0; allow_revisit <= 1 && num_valid_exits == 0; allow_revisit++) {
        for (int e = 0; e < current_lane->num_exits && num_valid_exits < 8; e++) {
            int exit_lane_idx = current_lane->exit_lanes[e];
            if (exit_lane_idx < 0 || exit_lane_idx >= env->num_road_elements) {
                continue;
            }
            if (!allow_revisit && pdm_route_contains_lane(route, route_length, exit_lane_idx)) {
                continue;
            }

            float exit_end_distance_sq =
                compute_lane_end_distance_sq(&env->road_elements[exit_lane_idx], origin_x, origin_y);
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
        int chosen_idx = rand() % num_progressing_exits;
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
    *route = (PDMPlanningRoute){0};
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
    if (remaining_distance >= required_distance) {
        return 1;
    }

    int current_lane_idx = route->lanes[route->length - 1];
    if (current_lane_idx < 0 || current_lane_idx >= env->num_road_elements) {
        return 0;
    }

    float max_end_distance_sq =
        compute_lane_end_distance_sq(&env->road_elements[current_lane_idx], agent->sim_x, agent->sim_y);

    while (remaining_distance < required_distance && route->length < MAX_ROUTE_LENGTH) {
        RoadMapElement *current_lane = &env->road_elements[current_lane_idx];
        int next_lane_idx = pdm_choose_route_extension_exit(env, current_lane, route->lanes, route->length,
                                                            agent->sim_x, agent->sim_y, &max_end_distance_sq);
        if (next_lane_idx == -1) {
            break;
        }

        route->lanes[route->length] = next_lane_idx;
        route->source_route_indices[route->length] = -1;
        route->length++;
        remaining_distance += compute_lane_length(&env->road_elements[next_lane_idx]);
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

    int center_route_idx = agent->current_route_index;
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
        if (lane->segment_length < 2) {
            continue;
        }

        int start_seg = 0;
        int end_seg = lane->segment_length - 1;
        if (route_idx == center_route_idx) {
            start_seg = center_seg_idx - 8;
            end_seg = center_seg_idx + 9;
            if (start_seg < 0) {
                start_seg = 0;
            }
            if (end_seg > lane->segment_length - 1) {
                end_seg = lane->segment_length - 1;
            }
        } else if (route_idx < center_route_idx) {
            start_seg = lane->segment_length > 9 ? lane->segment_length - 9 : 0;
        } else {
            end_seg = lane->segment_length - 1 < 8 ? lane->segment_length - 1 : 8;
        }

        for (int seg_idx = start_seg; seg_idx < end_seg; seg_idx++) {
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

    if (!best.valid) {
        return idm_project_to_route_lanes(env, agent);
    }
    return best;
}

static int pdm_sample_offset_route_pose(Drive *env, const PDMPlanningRoute *route, IDMLaneProjection projection,
                                        float distance, float offset, PDMRolloutStep *out) {
    if (!projection.valid || route == NULL || route->length <= 0) {
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
            if (traveled_s + remaining + 1e-4f < distance) {
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

static float pdm_route_max_curvature(Drive *env, const PDMPlanningRoute *route, IDMLaneProjection projection,
                                     float distance) {
    if (!projection.valid || route == NULL || route->length <= 0 || distance <= 0.0f) {
        return 0.0f;
    }

    float traveled_s = 0.0f;
    float max_curvature = 0.0f;
    int route_idx = projection.route_idx;
    int seg_idx = projection.segment_idx;
    float prev_heading = 0.0f;
    int has_prev_heading = 0;

    while (route_idx < route->length && traveled_s < distance) {
        int lane_idx = route->lanes[route_idx];
        if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
            break;
        }
        RoadMapElement *lane = &env->road_elements[lane_idx];
        if (lane->segment_length < 2) {
            break;
        }

        while (seg_idx < lane->segment_length - 1 && traveled_s < distance) {
            float seg_len = idm_lane_segment_length(lane, seg_idx);
            float heading = normalize_heading(lane->headings[seg_idx]);
            if (has_prev_heading && seg_len > 1e-6f) {
                float curvature = fabsf(compute_heading_diff(heading, prev_heading)) / seg_len;
                max_curvature = fmaxf(max_curvature, curvature);
            }
            prev_heading = heading;
            has_prev_heading = 1;
            traveled_s += fmaxf(seg_len, 0.0f);
            seg_idx++;
        }

        route_idx++;
        seg_idx = 0;
    }

    return max_curvature;
}

static void pdm_bezier_point(float alpha, float p0x, float p0y, float p0z, float p1x, float p1y, float p1z, float p2x,
                             float p2y, float p2z, float p3x, float p3y, float p3z, float *x, float *y, float *z) {
    float beta = 1.0f - alpha;
    float b0 = beta * beta * beta;
    float b1 = 3.0f * beta * beta * alpha;
    float b2 = 3.0f * beta * alpha * alpha;
    float b3 = alpha * alpha * alpha;
    *x = b0 * p0x + b1 * p1x + b2 * p2x + b3 * p3x;
    *y = b0 * p0y + b1 * p1y + b2 * p2y + b3 * p3y;
    *z = b0 * p0z + b1 * p1z + b2 * p2z + b3 * p3z;
}

static int pdm_build_smooth_path(Drive *env, Agent *agent, const PDMPlanningRoute *route, IDMLaneProjection projection,
                                 float offset, float speed, PDMBezierPath *path) {
    *path = (PDMBezierPath){0};
    if (!projection.valid) {
        return 0;
    }

    float speed_for_merge = fmaxf(speed, 0.1f);
    PDMRolloutStep target_start = {0};
    PDMRolloutStep target_ahead = {0};
    if (!pdm_sample_offset_route_pose(env, route, projection, 0.0f, offset, &target_start) ||
        !pdm_sample_offset_route_pose(env, route, projection, 1.0f, offset, &target_ahead)) {
        return 0;
    }

    float target_dx = target_ahead.x - target_start.x;
    float target_dy = target_ahead.y - target_start.y;
    float target_norm = sqrtf(target_dx * target_dx + target_dy * target_dy);
    float target_tx = target_start.cos_heading;
    float target_ty = target_start.sin_heading;
    if (target_norm > 1e-6f) {
        target_tx = target_dx / target_norm;
        target_ty = target_dy / target_norm;
    }

    float vel_norm = sqrtf(agent->sim_vx * agent->sim_vx + agent->sim_vy * agent->sim_vy);
    float start_tx = agent->cos_heading;
    float start_ty = agent->sin_heading;
    if (vel_norm > 0.5f) {
        start_tx = agent->sim_vx / vel_norm;
        start_ty = agent->sim_vy / vel_norm;
    }

    float alignment = clip(start_tx * target_tx + start_ty * target_ty, -1.0f, 1.0f);
    float alignment_scaling = 1.0f - PDM_ALIGNMENT_FACTOR * alignment;
    float horizon = pdm_horizon(env);
    float curvature_lookahead = clip(speed_for_merge * horizon, 5.0f, IDM_MAX_LOOKAHEAD);
    float max_curvature = pdm_route_max_curvature(env, route, projection, curvature_lookahead);
    float d_merge =
        (PDM_MERGE_D_BASE + PDM_MERGE_K_V * speed - PDM_CURVATURE_FACTOR * max_curvature) * alignment_scaling;
    float feasible_merge = 0.0f;

    float lateral_error = sqrtf((target_start.x - agent->sim_x) * (target_start.x - agent->sim_x) +
                                (target_start.y - agent->sim_y) * (target_start.y - agent->sim_y));
    if (lateral_error > 0.05f && PDM_STEERING_RATE_LIMIT > 1e-6f && agent->wheelbase > 1e-6f) {
        float speed_for_feasibility = fmaxf(speed_for_merge, 1.0f);
        feasible_merge =
            cbrtf(6.0f * lateral_error * speed_for_feasibility * agent->wheelbase / PDM_STEERING_RATE_LIMIT);
        d_merge = fmaxf(d_merge, feasible_merge);
    }

    float h_merge = d_merge / speed_for_merge;
    h_merge = clip(h_merge, PDM_MERGE_H_MIN, horizon);
    d_merge = fmaxf(h_merge * speed_for_merge, 0.0f);
    d_merge = fmaxf(d_merge, feasible_merge);

    PDMRolloutStep merge_step = {0};
    if (!pdm_sample_offset_route_pose(env, route, projection, d_merge, offset, &merge_step)) {
        return 0;
    }

    float p0x = agent->sim_x;
    float p0y = agent->sim_y;
    float p0z = agent->sim_z;
    float p1x = p0x + start_tx * d_merge / 3.0f;
    float p1y = p0y + start_ty * d_merge / 3.0f;
    float p1z = p0z;
    float p3x = merge_step.x;
    float p3y = merge_step.y;
    float p3z = merge_step.z;
    float p2x = p3x - merge_step.cos_heading * d_merge / 3.0f;
    float p2y = p3y - merge_step.sin_heading * d_merge / 3.0f;
    float p2z = p3z;

    path->valid = 1;
    path->num_points = PDM_BEZIER_SAMPLES;
    path->merge_route_s = d_merge;
    for (int i = 0; i < PDM_BEZIER_SAMPLES; i++) {
        float alpha = (float)i / (float)(PDM_BEZIER_SAMPLES - 1);
        pdm_bezier_point(alpha, p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, &path->x[i], &path->y[i],
                         &path->z[i]);
        if (i == 0) {
            path->arc_lengths[i] = 0.0f;
        } else {
            float dx = path->x[i] - path->x[i - 1];
            float dy = path->y[i] - path->y[i - 1];
            float dz = path->z[i] - path->z[i - 1];
            path->arc_lengths[i] = path->arc_lengths[i - 1] + sqrtf(dx * dx + dy * dy + dz * dz);
        }
    }
    path->bezier_length = path->arc_lengths[PDM_BEZIER_SAMPLES - 1];

    for (int i = 0; i < PDM_BEZIER_SAMPLES; i++) {
        if (i + 1 < PDM_BEZIER_SAMPLES) {
            path->heading[i] = atan2f(path->y[i + 1] - path->y[i], path->x[i + 1] - path->x[i]);
        } else if (i > 0) {
            path->heading[i] = path->heading[i - 1];
        } else {
            path->heading[i] = agent->sim_heading;
        }
    }
    path->heading[PDM_BEZIER_SAMPLES - 1] = merge_step.heading;
    return 1;
}

static int pdm_sample_smooth_path(Drive *env, const PDMPlanningRoute *route, IDMLaneProjection projection, float offset,
                                  PDMBezierPath *path, float distance, PDMRolloutStep *out) {
    if (!path->valid) {
        return 0;
    }

    if (distance <= path->bezier_length + 1e-4f) {
        for (int i = 1; i < path->num_points; i++) {
            if (path->arc_lengths[i] + 1e-4f < distance) {
                continue;
            }

            float seg_len = path->arc_lengths[i] - path->arc_lengths[i - 1];
            float t = seg_len > 1e-6f ? (distance - path->arc_lengths[i - 1]) / seg_len : 0.0f;
            t = clip(t, 0.0f, 1.0f);
            float heading = path->heading[i - 1];
            float center_distance =
                path->bezier_length > 1e-6f ? distance * path->merge_route_s / path->bezier_length : 0.0f;
            PDMRolloutStep center_step = {0};
            if (!pdm_sample_offset_route_pose(env, route, projection, center_distance, 0.0f, &center_step)) {
                return 0;
            }
            out->valid = 1;
            out->s = distance;
            out->x = path->x[i - 1] + t * (path->x[i] - path->x[i - 1]);
            out->y = path->y[i - 1] + t * (path->y[i] - path->y[i - 1]);
            out->z = path->z[i - 1] + t * (path->z[i] - path->z[i - 1]);
            out->heading = normalize_heading(heading);
            out->cos_heading = cosf(out->heading);
            out->sin_heading = sinf(out->heading);
            out->lane_idx = center_step.lane_idx;
            out->route_idx = center_step.route_idx;
            out->segment_idx = center_step.segment_idx;
            out->segment_t = center_step.segment_t;
            out->center_lane_idx = center_step.center_lane_idx;
            out->center_x = center_step.center_x;
            out->center_y = center_step.center_y;
            out->center_z = center_step.center_z;
            out->center_heading = center_step.center_heading;
            return 1;
        }
    }

    float route_distance = path->merge_route_s + fmaxf(0.0f, distance - path->bezier_length);
    return pdm_sample_offset_route_pose(env, route, projection, route_distance, offset, out);
}

static PDMTrackingState pdm_initial_tracking_state(const Agent *agent) {
    return (PDMTrackingState){
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

static PDMTrackingState pdm_track_target_step(Drive *env, const Agent *agent, PDMTrackingState state,
                                              PDMRolloutStep target, float target_speed, float dt) {
    if (dt <= 0.0f) {
        return state;
    }

    float speed = fmaxf(0.0f, state.speed);
    float accel = (target_speed - speed) / dt;
    accel = clip(accel, -PDM_URGENT_DECEL, IDM_MAX_ACCEL);
    float new_speed = clip(speed + accel * dt, 0.0f, MAX_SPEED);

    float dx_to_target = target.x - state.x;
    float dy_to_target = target.y - state.y;
    float target_distance = sqrtf(dx_to_target * dx_to_target + dy_to_target * dy_to_target);
    float target_heading = state.heading;
    if (target_distance > 1e-4f) {
        target_heading = atan2f(dy_to_target, dx_to_target);
    }

    float heading_error = compute_heading_diff(target_heading, state.heading);
    float desired_yaw_rate = heading_error / dt;
    float steering = 0.0f;
    if (new_speed > 1e-3f) {
        steering = atanf(desired_yaw_rate * agent->wheelbase / new_speed);
    }

    float max_steering = STEERING_VALUES[8];
    float delta_steer =
        clip(steering - state.steering_angle, -PDM_STEERING_RATE_LIMIT * dt, PDM_STEERING_RATE_LIMIT * dt);
    steering = clip(state.steering_angle + delta_steer, -max_steering, max_steering);

    float beta = atanf(0.5f * tanf(steering));
    float yaw_rate = 0.0f;
    if (new_speed > 1e-3f) {
        yaw_rate = (new_speed * cosf(beta) * tanf(steering)) / agent->wheelbase;
    }

    state.x += new_speed * cosf(state.heading + beta) * dt;
    state.y += new_speed * sinf(state.heading + beta) * dt;
    state.z = target.z;
    state.heading = normalize_heading(state.heading + yaw_rate * dt);
    state.speed = new_speed;
    state.steering_angle = steering;
    return state;
}

static PDMRollout pdm_generate_constant_speed_rollout(Drive *env, Agent *agent, const PDMPlanningRoute *route,
                                                      IDMLaneProjection projection, float offset, float speed) {
    PDMRollout rollout = {0};
    PDMBezierPath path = {0};
    if (!pdm_build_smooth_path(env, agent, route, projection, offset, speed, &path)) {
        return rollout;
    }
    rollout.valid = 1;

    PDMTrackingState action_state = pdm_initial_tracking_state(agent);
    PDMRolloutStep action_target = {0};
    if (!pdm_sample_smooth_path(env, route, projection, offset, &path, speed * env->dt, &action_target)) {
        rollout.valid = 0;
        return rollout;
    }
    action_state = pdm_track_target_step(env, agent, action_state, action_target, speed, env->dt);
    rollout.action_step = pdm_tracking_state_to_step(action_state, action_target, env->dt, speed * env->dt);

    float horizon = pdm_agent_horizon(env, agent);
    float planning_dt = pdm_planning_dt(env);
    PDMTrackingState state = pdm_initial_tracking_state(agent);
    float prev_t = 0.0f;

    for (int step = 0; step < PDM_MAX_ROLLOUT_STEPS; step++) {
        float t = step * planning_dt;
        if (t > horizon + 1e-4f) {
            break;
        }

        float s = speed * t;
        PDMRolloutStep target_step = {0};
        if (!pdm_sample_smooth_path(env, route, projection, offset, &path, s, &target_step)) {
            rollout.valid = 0;
            break;
        }

        if (step > 0) {
            state = pdm_track_target_step(env, agent, state, target_step, speed, t - prev_t);
        }
        PDMRolloutStep rollout_step = pdm_tracking_state_to_step(state, target_step, t, s);
        rollout.steps[rollout.num_steps++] = rollout_step;
        prev_t = t;
    }

    return rollout;
}

static IDMLeader pdm_find_leader_by_offset_route_boxes(Drive *env, int agent_idx, const PDMPlanningRoute *route,
                                                       IDMLaneProjection projection, float offset) {
    Agent *agent = &env->agents[agent_idx];
    IDMLeader no_leader = idm_no_leader();
    if (!projection.valid) {
        return no_leader;
    }

    float speed = fmaxf(0.0f, agent->sim_speed_signed);
    float lookahead = clip(speed * IDM_LOOKAHEAD_TIME, IDM_MIN_LOOKAHEAD, IDM_MAX_LOOKAHEAD);
    int candidates[IDM_MAX_CANDIDATES];
    int num_candidates = idm_collect_route_candidates(env, agent_idx, lookahead, candidates, IDM_MAX_CANDIDATES);

    for (float sample_s = IDM_ROUTE_SAMPLE_DS; sample_s <= lookahead + 1e-4f; sample_s += IDM_ROUTE_SAMPLE_DS) {
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
            if (!idm_sample_hits_agent(&sample, other)) {
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

static Agent pdm_make_rollout_sample_agent(const Agent *agent, PDMRolloutStep step) {
    return idm_make_sample_agent(agent, step.x, step.y, step.z, step.heading);
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
    float other_forward_projection =
        fabsf(other->cos_heading * ego_sample->cos_heading + other->sin_heading * ego_sample->sin_heading);
    float other_lateral_projection =
        fabsf(-other->sin_heading * ego_sample->cos_heading + other->cos_heading * ego_sample->sin_heading);
    float other_half_extent =
        0.5f * other->sim_length * other_forward_projection + 0.5f * other->sim_width * other_lateral_projection;
    return rel_x + other_half_extent >= -0.5f * ego_sample->sim_length;
}

static float pdm_compute_collision_ttc(Drive *env, int agent_idx, PDMCandidateScore *candidate) {
    if (!candidate->valid || !candidate->rollout.valid) {
        return 0.0f;
    }

    Agent *agent = &env->agents[agent_idx];
    float horizon = pdm_agent_horizon(env, agent);
    float max_distance = candidate->new_speed * horizon + 0.5f * agent->sim_length + 10.0f;
    int candidates[IDM_MAX_CANDIDATES];
    int num_candidates = idm_collect_route_candidates(env, agent_idx, max_distance, candidates, IDM_MAX_CANDIDATES);
    int frontal_candidates[IDM_MAX_CANDIDATES];
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
        Agent sample = pdm_make_rollout_sample_agent(agent, step);

        for (int i = 0; i < num_frontal_candidates; i++) {
            Agent other = pdm_predict_other_agent(&env->agents[frontal_candidates[i]], step.t);
            if (idm_sample_hits_agent(&sample, &other)) {
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
        Agent sample = pdm_make_rollout_sample_agent(agent, step);
        if (idm_sample_hits_red_light(env, &sample, step.lane_idx)) {
            return step.t;
        }
    }

    return pdm_agent_horizon(env, agent);
}

static int pdm_sample_is_offroad(Drive *env, const Agent *agent, PDMRolloutStep step) {
    if (get_grid_index(env, step.x, step.y) == -1) {
        return 1;
    }

    float half_length = 0.5f * agent->sim_length;
    float half_width = 0.5f * agent->sim_width;
    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] =
            step.x + offsets[i][0] * half_length * step.cos_heading - offsets[i][1] * half_width * step.sin_heading;
        corners[i][1] =
            step.y + offsets[i][0] * half_length * step.sin_heading + offsets[i][1] * half_width * step.cos_heading;
    }

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25];
    int list_size =
        get_neighbors_entities(env, step.x, step.y, entity_list, MAX_ENTITIES_PER_CELL * 25, collision_offsets, 25);
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

        float abs_dz = fabsf(element->z[geometry_idx] - step.z);
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

    return pdm_agent_horizon(env, agent);
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
    float horizon = pdm_agent_horizon(env, agent);
    candidate->collision_ttc = pdm_compute_collision_ttc(env, agent_idx, candidate);
    candidate->traffic_light_ttc = pdm_compute_traffic_light_ttc(env, agent, candidate);
    candidate->offroad_ttc = INFINITY;
    candidate->min_ttc = fminf(candidate->collision_ttc, candidate->traffic_light_ttc);

    float ttc_score = clip(candidate->min_ttc / horizon, 0.0f, 1.0f);
    float speed_score = clip(candidate->new_speed / fmaxf(speed_limit, 1.0f), 0.0f, 1.0f);
    if (candidate->min_ttc < horizon && candidate->min_ttc <= PDM_SAFE_SPEED_TTC) {
        speed_score = 0.0f;
    }

    float danger_ttc = fminf(pdm_speed_aware_ttc(fmaxf(0.0f, agent->sim_speed_signed)), horizon);
    float collision_penalty = (candidate->min_ttc < danger_ttc) ? PDM_COLLISION_PENALTY : 0.0f;
    float center_bonus = (fabsf(candidate->offset) < 1e-4f) ? PDM_CENTER_BONUS : 0.0f;
    candidate->score = -collision_penalty + PDM_SPEED_WEIGHT * speed_score + PDM_TTC_WEIGHT * ttc_score + center_bonus;
}

static int pdm_build_placeholder_candidates(Drive *env, int agent_idx, PDMCandidateScore *candidates,
                                            int max_candidates) {
    Agent *agent = &env->agents[agent_idx];
    int count = 0;
    float speed_limit = idm_desired_speed(env, agent);
    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    float horizon = pdm_agent_horizon(env, agent);
    IDMLaneProjection projection = pdm_project_from_route_state(env, agent);
    PDMPlanningRoute planning_route = {0};
    IDMLeader offset_leaders[PDM_NUM_OFFSETS];

    if (!projection.valid || !pdm_build_planning_route(env, agent, &projection, &planning_route)) {
        return 0;
    }

    for (int offset_idx = 0; offset_idx < PDM_NUM_OFFSETS; offset_idx++) {
        offset_leaders[offset_idx] =
            pdm_find_leader_by_offset_route_boxes(env, agent_idx, &planning_route, projection, PDM_OFFSETS[offset_idx]);
    }

    for (int offset_idx = 0; offset_idx < PDM_NUM_OFFSETS; offset_idx++) {
        for (int speed_idx = 0; speed_idx < PDM_NUM_SPEED_FRACTIONS; speed_idx++) {
            if (count >= max_candidates) {
                return count;
            }

            float speed_fraction = PDM_SPEED_FRACTIONS[speed_idx];
            float target_speed = speed_fraction * speed_limit;
            float accel = pdm_compute_idm_acceleration(agent, target_speed, offset_leaders[offset_idx]);
            accel = clip(accel, -IDM_MAX_DECEL, IDM_MAX_ACCEL);

            float new_speed = current_speed + accel * env->dt;
            if (new_speed < 0.0f) {
                new_speed = 0.0f;
            }
            accel = (new_speed - current_speed) / env->dt;

            PDMRollout rollout = pdm_generate_constant_speed_rollout(env, agent, &planning_route, projection,
                                                                     PDM_OFFSETS[offset_idx], new_speed);
            candidates[count++] = (PDMCandidateScore){
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

static PDMCandidateScore pdm_select_best_feasible_candidate(Drive *env, int agent_idx, PDMCandidateScore *candidates,
                                                            int num_candidates) {
    int checked[PDM_NUM_CANDIDATES] = {0};
    Agent *agent = &env->agents[agent_idx];
    float horizon = pdm_agent_horizon(env, agent);
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

        if (!best_offroad.valid || candidates[best_idx].offroad_ttc > best_offroad.offroad_ttc ||
            (candidates[best_idx].offroad_ttc == best_offroad.offroad_ttc &&
             candidates[best_idx].score > best_offroad.score)) {
            best_offroad = candidates[best_idx];
        }
    }

    return best_offroad.valid ? best_offroad : pdm_select_best_candidate(candidates, num_candidates);
}

static int pdm_build_urgent_action_step(Drive *env, int agent_idx, float offset, float urgent_speed,
                                        PDMRolloutStep *out) {
    Agent *agent = &env->agents[agent_idx];
    *out = (PDMRolloutStep){0};

    IDMLaneProjection projection = pdm_project_from_route_state(env, agent);
    PDMPlanningRoute planning_route = {0};
    if (!projection.valid || !pdm_build_planning_route(env, agent, &projection, &planning_route)) {
        return 0;
    }

    PDMBezierPath path = {0};
    if (!pdm_build_smooth_path(env, agent, &planning_route, projection, offset, urgent_speed, &path)) {
        return 0;
    }

    PDMRolloutStep target = {0};
    if (!pdm_sample_smooth_path(env, &planning_route, projection, offset, &path, urgent_speed * env->dt, &target)) {
        return 0;
    }

    PDMTrackingState state = pdm_initial_tracking_state(agent);
    state = pdm_track_target_step(env, agent, state, target, urgent_speed, env->dt);
    *out = pdm_tracking_state_to_step(state, target, env->dt, urgent_speed * env->dt);
    return out->valid;
}

static void pdm_stop_agent(Agent *agent) {
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
}

static void pdm_apply_speed_along_route_or_heading(Drive *env, int agent_idx, float new_speed) {
    Agent *agent = &env->agents[agent_idx];
    float old_a_long = agent->a_long;
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
    agent->jerk_lat = (new_a_lat - agent->a_lat) / env->dt;
    agent->a_long = accel;
    agent->a_lat = new_a_lat;
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
    float old_a_long = agent->a_long;
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
            agent->current_route_index = step.route_idx;
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
    agent->jerk_lat = (new_a_lat - agent->a_lat) / env->dt;
    agent->a_long = accel;
    agent->a_lat = new_a_lat;
    update_agent_speed(agent);
}

static void move_pdm(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];

    if (agent->removed) {
        invalidate_agent(agent);
        return;
    }

    if (agent->stopped || agent->sim_x == INVALID_POSITION) {
        pdm_stop_agent(agent);
        return;
    }

    PDMCandidateScore candidates[PDM_NUM_CANDIDATES];
    int num_candidates = pdm_build_placeholder_candidates(env, agent_idx, candidates, PDM_NUM_CANDIDATES);
    PDMCandidateScore best = pdm_select_best_feasible_candidate(env, agent_idx, candidates, num_candidates);
    if (!best.valid) {
        pdm_apply_urgent_brake_fallback(env, agent_idx);
        return;
    }

    float danger_ttc = fminf(pdm_speed_aware_ttc(fmaxf(0.0f, agent->sim_speed_signed)), pdm_agent_horizon(env, agent));
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
