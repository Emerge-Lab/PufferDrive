#ifndef PUFFERLIB_OCEAN_DRIVE_PDM_H
#define PUFFERLIB_OCEAN_DRIVE_PDM_H

#define PDM_NUM_OFFSETS 3
#define PDM_NUM_SPEED_FRACTIONS 5
#define PDM_NUM_CANDIDATES (PDM_NUM_OFFSETS * PDM_NUM_SPEED_FRACTIONS)
#define PDM_HORIZON 4.0f
#define PDM_PLANNING_DT 0.5f
#define PDM_MAX_ROLLOUT_STEPS 9
#define PDM_DANGER_TTC 2.0f

static const float PDM_OFFSETS[PDM_NUM_OFFSETS] = {0.0f, -1.0f, 1.0f};
static const float PDM_SPEED_FRACTIONS[PDM_NUM_SPEED_FRACTIONS] = {1.0f, 0.8f, 0.6f, 0.4f, 0.2f};

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
    int lane_idx;
} PDMRolloutStep;

typedef struct {
    int valid;
    int num_steps;
    PDMRolloutStep steps[PDM_MAX_ROLLOUT_STEPS];
} PDMRollout;

typedef struct {
    float offset;
    float speed_fraction;
    float target_speed;
    float new_speed;
    float accel;
    float score;
    int valid;
    PDMRollout rollout;
} PDMCandidateScore;

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

static int pdm_sample_offset_route_pose(Drive *env, Agent *agent, IDMLaneProjection projection, float distance,
                                        float offset, PDMRolloutStep *out) {
    if (!projection.valid || agent->route == NULL || agent->route_length <= 0) {
        return 0;
    }

    float traveled_s = 0.0f;
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
            return 1;
        }

        route_idx++;
        seg_idx = 0;
        t = 0.0f;
    }

    return 0;
}

static PDMRollout pdm_generate_constant_speed_rollout(Drive *env, Agent *agent, IDMLaneProjection projection,
                                                      float offset, float speed) {
    PDMRollout rollout = {0};
    rollout.valid = 1;

    for (int step = 0; step < PDM_MAX_ROLLOUT_STEPS; step++) {
        float t = step * PDM_PLANNING_DT;
        if (t > PDM_HORIZON + 1e-4f) {
            break;
        }

        PDMRolloutStep rollout_step = {0};
        float s = speed * t;
        if (!pdm_sample_offset_route_pose(env, agent, projection, s, offset, &rollout_step)) {
            rollout.valid = 0;
            break;
        }

        rollout_step.t = t;
        rollout_step.speed = speed;
        rollout.steps[rollout.num_steps++] = rollout_step;
    }

    return rollout;
}

static IDMLeader pdm_find_leader_by_offset_route_boxes(Drive *env, int agent_idx, IDMLaneProjection projection,
                                                       float offset) {
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
        if (!pdm_sample_offset_route_pose(env, agent, projection, sample_s, offset, &sample_step)) {
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

static int pdm_build_placeholder_candidates(Drive *env, int agent_idx, PDMCandidateScore *candidates,
                                            int max_candidates) {
    Agent *agent = &env->agents[agent_idx];
    int count = 0;
    float speed_limit = idm_desired_speed(env, agent);
    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    IDMLaneProjection projection = idm_project_to_route_lanes(env, agent);
    IDMLeader offset_leaders[PDM_NUM_OFFSETS];

    if (!projection.valid) {
        return 0;
    }

    for (int offset_idx = 0; offset_idx < PDM_NUM_OFFSETS; offset_idx++) {
        offset_leaders[offset_idx] =
            pdm_find_leader_by_offset_route_boxes(env, agent_idx, projection, PDM_OFFSETS[offset_idx]);
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

            PDMRollout rollout =
                pdm_generate_constant_speed_rollout(env, agent, projection, PDM_OFFSETS[offset_idx], new_speed);
            candidates[count++] = (PDMCandidateScore){
                .offset = PDM_OFFSETS[offset_idx],
                .speed_fraction = speed_fraction,
                .target_speed = target_speed,
                .new_speed = new_speed,
                .accel = accel,
                .score = 0.0f,
                .valid = rollout.valid,
                .rollout = rollout,
            };
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

static void pdm_apply_teleport_step(Drive *env, int agent_idx, PDMCandidateScore candidate) {
    Agent *agent = &env->agents[agent_idx];
    float old_a_long = agent->a_long;
    float current_speed = fmaxf(0.0f, agent->sim_speed_signed);
    float new_speed = candidate.new_speed;
    if (new_speed < 0.0f) {
        new_speed = 0.0f;
    }
    float accel = (new_speed - current_speed) / env->dt;

    float old_heading = agent->sim_heading;
    float distance = new_speed * env->dt;
    if (!idm_advance_along_route_lanes(env, agent_idx, distance, &old_heading)) {
        agent->stopped = 1;
        pdm_stop_agent(agent);
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
    agent->steering_angle = 0.0f;
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
    PDMCandidateScore best = pdm_select_best_candidate(candidates, num_candidates);
    if (!best.valid) {
        pdm_stop_agent(agent);
        return;
    }

    pdm_apply_teleport_step(env, agent_idx, best);
}

#endif
