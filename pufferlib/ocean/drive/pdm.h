#ifndef PUFFERLIB_OCEAN_DRIVE_PDM_H
#define PUFFERLIB_OCEAN_DRIVE_PDM_H

#define PDM_NUM_OFFSETS 3
#define PDM_NUM_SPEED_FRACTIONS 5
#define PDM_NUM_CANDIDATES (PDM_NUM_OFFSETS * PDM_NUM_SPEED_FRACTIONS)
#define PDM_HORIZON 4.0f
#define PDM_PLANNING_DT 0.5f
#define PDM_DANGER_TTC 2.0f

static const float PDM_OFFSETS[PDM_NUM_OFFSETS] = {0.0f, -1.0f, 1.0f};
static const float PDM_SPEED_FRACTIONS[PDM_NUM_SPEED_FRACTIONS] = {1.0f, 0.8f, 0.6f, 0.4f, 0.2f};

typedef struct {
    float offset;
    float speed_fraction;
    float target_speed;
    float score;
    int valid;
} PDMCandidateScore;

static int pdm_build_placeholder_candidates(Drive *env, Agent *agent, PDMCandidateScore *candidates,
                                            int max_candidates) {
    int count = 0;
    float speed_limit = idm_desired_speed(env, agent);

    for (int offset_idx = 0; offset_idx < PDM_NUM_OFFSETS; offset_idx++) {
        for (int speed_idx = 0; speed_idx < PDM_NUM_SPEED_FRACTIONS; speed_idx++) {
            if (count >= max_candidates) {
                return count;
            }

            float speed_fraction = PDM_SPEED_FRACTIONS[speed_idx];
            candidates[count++] = (PDMCandidateScore){
                .offset = PDM_OFFSETS[offset_idx],
                .speed_fraction = speed_fraction,
                .target_speed = speed_fraction * speed_limit,
                .score = 0.0f,
                .valid = 1,
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
    float accel = (candidate.target_speed - current_speed) / env->dt;
    accel = clip(accel, -IDM_MAX_DECEL, IDM_MAX_ACCEL);

    float new_speed = current_speed + accel * env->dt;
    if (new_speed < 0.0f) {
        new_speed = 0.0f;
    }
    accel = (new_speed - current_speed) / env->dt;

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
    int num_candidates = pdm_build_placeholder_candidates(env, agent, candidates, PDM_NUM_CANDIDATES);
    PDMCandidateScore best = pdm_select_best_candidate(candidates, num_candidates);
    if (!best.valid) {
        pdm_stop_agent(agent);
        return;
    }

    pdm_apply_teleport_step(env, agent_idx, best);
}

#endif
