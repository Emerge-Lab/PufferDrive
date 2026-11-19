#ifndef OPENGIGAFLOW_OCEAN_DRIVE_PDM_H
#define OPENGIGAFLOW_OCEAN_DRIVE_PDM_H

// PDM-Closed style planner: proposals = lateral route offsets x fractions of the speed limit, rolled out and scored
#define PDM_LATERAL_OFFSET_COUNT 3
#define PDM_SPEED_FRACTION_COUNT 5
#define PDM_CANDIDATE_COUNT (PDM_LATERAL_OFFSET_COUNT * PDM_SPEED_FRACTION_COUNT)
#define PDM_MAX_ROLLOUT_STEPS 81
static const float PDM_LATERAL_OFFSETS_METERS[PDM_LATERAL_OFFSET_COUNT] = {0.0f, -1.0f, 1.0f};
static const float PDM_SPEED_FRACTIONS[PDM_SPEED_FRACTION_COUNT] = {1.0f, 0.75f, 0.5f, 0.25f, 0.01f};

#define PDM_MIN_DANGER_TTC_SECONDS 2.0f
#define PDM_DANGER_TTC_BUFFER_SECONDS 0.5f
#define PDM_SAFE_SPEED_TTC_SECONDS 5.0f
#define PDM_COLLISION_PENALTY 48.0f
#define PDM_SPEED_WEIGHT 3.0f
#define PDM_TTC_WEIGHT 10.0f
#define PDM_CENTER_BONUS 1.0f

#define PDM_MIN_SPACING_METERS 2.0f
#define PDM_SAFE_TIME_HEADWAY_SECONDS 2.0f
#define PDM_MAX_ACCEL_MPS2 4.0f
#define PDM_MAX_DECEL_MPS2 5.0f
#define PDM_DELTA 4.0f
#define PDM_DEFAULT_DESIRED_SPEED_MPS 15.0f
#define PDM_MIN_DESIRED_SPEED_MPS 1.0f
#define PDM_MIN_LEADER_GAP_METERS 0.1f
#define PDM_LEADER_LOOKAHEAD_SECONDS 5.0f
#define PDM_MIN_LEADER_LOOKAHEAD_METERS 20.0f
#define PDM_MAX_LEADER_LOOKAHEAD_METERS 120.0f

#define PDM_PURSUIT_LOOKAHEAD_SECONDS 0.6f
#define PDM_PURSUIT_MIN_LOOKAHEAD_METERS 3.0f
#define PDM_PURSUIT_MAX_LOOKAHEAD_METERS 15.0f
#define PDM_MIN_AIM_DISTANCE_METERS 1e-3f
#define PDM_REAR_AXLE_RATIO 0.5f // box centre sits halfway between the axles

#define PDM_ROUTE_MARGIN_METERS 10.0f
#define PDM_MAX_NEARBY_AGENTS 256
#define PDM_AGENT_QUERY_MARGIN_METERS 5.0f
#define PDM_COLLISION_QUERY_MARGIN_METERS 10.0f
#define PDM_COLLISION_BBOX_MARGIN_METERS 0.05f
#define PDM_MIN_SEGMENT_LENGTH_METERS 1e-6f
#define PDM_DISTANCE_EPSILON_METERS 1e-4f
#define PDM_TIME_EPSILON_SECONDS 1e-4f
#define PDM_SPEED_EPSILON_MPS 1e-3f
#define PDM_ACCEL_EPSILON_MPS2 1e-6f

// Point on the route, shifted sideways by a candidate's lateral offset
typedef struct {
    float x;
    float y;
    float z;
    float heading_radians;
    int lane_idx;
    int route_idx;
} PDMRoutePose;

typedef struct {
    float time_seconds;
    float x;
    float y;
    float z;
    float heading_radians;
    float speed_mps;
    float steering_angle_radians;
    int lane_idx; // route lane and slot the step is driving toward
    int route_idx;
} PDMRolloutStep;

typedef struct {
    int valid;
    int step_count;
    PDMRolloutStep first_step; // one env dt ahead: the step the agent takes
    PDMRolloutStep steps[PDM_MAX_ROLLOUT_STEPS];
} PDMRollout;

typedef struct {
    int valid;
    float lateral_offset_meters;
    float next_speed_mps;
    float min_ttc_seconds;
    float offroad_ttc_seconds;
    float score;
    PDMRollout rollout;
} PDMCandidate;

static inline float pdm_danger_ttc_seconds(float speed_mps) {
    return fmaxf(PDM_MIN_DANGER_TTC_SECONDS, speed_mps / PDM_MAX_DECEL_MPS2 + PDM_DANGER_TTC_BUFFER_SECONDS);
}

static inline float pdm_agent_horizon_seconds(const Drive *env, const Agent *agent) {
    return fmaxf(env->pdm_horizon_seconds, pdm_danger_ttc_seconds(fmaxf(0.0f, agent->sim_speed_signed)));
}

static float pdm_desired_speed(Drive *env, Agent *agent) {
    float desired_speed_mps = idm_lane_speed_limit(env, agent->current_lane_idx);
    if (desired_speed_mps <= 0.0f) {
        desired_speed_mps = idm_lane_speed_limit(env, agent->route[agent->current_route_idx]);
    }
    if (desired_speed_mps <= 0.0f) {
        desired_speed_mps = PDM_DEFAULT_DESIRED_SPEED_MPS;
    }
    return clip(desired_speed_mps, PDM_MIN_DESIRED_SPEED_MPS, env->base_max_speed_mps);
}

static float pdm_compute_acceleration(Drive *env, Agent *agent, float desired_speed_mps, IDMLeader leader) {
    float current_speed_mps = fmaxf(0.0f, agent->sim_speed_signed);
    desired_speed_mps = clip(desired_speed_mps, PDM_MIN_DESIRED_SPEED_MPS, env->base_max_speed_mps);
    float free_road_term = powf(current_speed_mps / desired_speed_mps, PDM_DELTA);
    float leader_term = 0.0f;
    if (leader.has_leader) {
        float desired_gap_meters = PDM_MIN_SPACING_METERS
            + fmaxf(0.0f,
                    current_speed_mps * PDM_SAFE_TIME_HEADWAY_SECONDS
                        + current_speed_mps * (current_speed_mps - leader.leader_speed_mps)
                            / (2.0f * sqrtf(PDM_MAX_ACCEL_MPS2 * PDM_MAX_DECEL_MPS2)));
        float gap_meters = fmaxf(leader.gap_meters, PDM_MIN_LEADER_GAP_METERS);
        leader_term = (desired_gap_meters / gap_meters) * (desired_gap_meters / gap_meters);
    }
    return PDM_MAX_ACCEL_MPS2 * (1.0f - free_road_term - leader_term);
}

// routes come from map files and are not validated on load: NULL for an entry that is no lane with geometry
static RoadMapElement *pdm_route_lane(Drive *env, const Agent *agent, int route_idx) {
    int lane_idx = agent->route[route_idx];
    if (lane_idx < 0 || lane_idx >= env->num_road_elements || env->road_elements[lane_idx].segment_size < 2) {
        return NULL;
    }
    return &env->road_elements[lane_idx];
}

static float pdm_remaining_route_distance(Drive *env, const Agent *agent, IDMLaneProjection projection) {
    float distance_meters = 0.0f;
    for (int route_idx = projection.route_idx; route_idx < agent->route_length; route_idx++) {
        RoadMapElement *lane = pdm_route_lane(env, agent, route_idx);
        if (lane == NULL) {
            break;
        }
        int is_projection_lane = route_idx == projection.route_idx;
        int start_segment_idx = is_projection_lane ? projection.segment_idx : 0;
        for (int segment_idx = start_segment_idx; segment_idx < lane->segment_size - 1; segment_idx++) {
            float segment_length_meters = idm_lane_segment_size(lane, segment_idx);
            float segment_t = (is_projection_lane && segment_idx == start_segment_idx) ? projection.t : 0.0f;
            distance_meters += (segment_length_meters < PDM_MIN_SEGMENT_LENGTH_METERS)
                ? 0.0f
                : (1.0f - segment_t) * segment_length_meters;
        }
    }
    return distance_meters;
}

// Gigaflow routes are random walks: drop the lanes behind the agent and walk on; returns the distance ahead
static float pdm_extend_route(Drive *env, Agent *agent, IDMLaneProjection *projection, float distance_meters) {
    int kept_lane_count = agent->route_length - projection->route_idx;
    assert(kept_lane_count <= MAX_ROUTE_LENGTH);
    int extended_route[MAX_ROUTE_LENGTH];
    memcpy(extended_route, agent->route + projection->route_idx, kept_lane_count * sizeof(int));
    int route_length = kept_lane_count;
    while (distance_meters < ROUTE_TARGET_DISTANCE && route_length < MAX_ROUTE_LENGTH) {
        int exit_lane_idx = pick_random_exit_lane(
            &env->rng_state,
            env->road_elements,
            extended_route[route_length - 1],
            extended_route,
            route_length);
        if (exit_lane_idx == -1) {
            break;
        }
        extended_route[route_length++] = exit_lane_idx;
        distance_meters += env->road_elements[exit_lane_idx].length;
    }
    if (route_length == kept_lane_count) {
        return distance_meters;
    }

    free(agent->route);
    agent->route = (int *) malloc(route_length * sizeof(int));
    memcpy(agent->route, extended_route, route_length * sizeof(int));
    agent->route_length = route_length;
    agent->current_route_idx = 0;
    projection->route_idx = 0;
    return distance_meters;
}

// Whether enough route lies ahead to plan on: a replay route ends where its log does, a gigaflow route is extended
static int pdm_prepare_route(Drive *env, Agent *agent, IDMLaneProjection *projection) {
    if (!projection->valid) {
        return 0;
    }
    if (env->simulation_mode == SIMULATION_MODE_REPLAY) {
        return 1;
    }
    float max_next_speed_mps
        = fminf(env->base_max_speed_mps, fmaxf(0.0f, agent->sim_speed_signed) + PDM_MAX_ACCEL_MPS2 * env->dt);
    float required_meters = pdm_agent_horizon_seconds(env, agent) * max_next_speed_mps + PDM_ROUTE_MARGIN_METERS;
    float remaining_meters = pdm_remaining_route_distance(env, agent, *projection);
    if (remaining_meters < required_meters) {
        remaining_meters = pdm_extend_route(env, agent, projection, remaining_meters);
    }
    return remaining_meters >= required_meters;
}

// Pose at route distance distance_meters if it falls on this lane; else the lane's rest is added to *traveled_meters
static int pdm_sample_lane_pose(
    RoadMapElement *lane,
    int start_segment_idx,
    float start_segment_t,
    float distance_meters,
    float lateral_offset_meters,
    float *traveled_meters,
    PDMRoutePose *pose) {
    for (int segment_idx = start_segment_idx; segment_idx < lane->segment_size - 1; segment_idx++) {
        float segment_length_meters = idm_lane_segment_size(lane, segment_idx);
        if (segment_length_meters < PDM_MIN_SEGMENT_LENGTH_METERS) {
            continue;
        }
        float segment_t = (segment_idx == start_segment_idx) ? start_segment_t : 0.0f;
        float remaining_meters = (1.0f - segment_t) * segment_length_meters;
        if (*traveled_meters + remaining_meters + PDM_DISTANCE_EPSILON_METERS < distance_meters) {
            *traveled_meters += remaining_meters;
            continue;
        }

        float pose_t = clip(segment_t + (distance_meters - *traveled_meters) / segment_length_meters, 0.0f, 1.0f);
        float heading_radians = normalize_heading(lane->headings[segment_idx]);
        float center_x = lane->x[segment_idx] + pose_t * (lane->x[segment_idx + 1] - lane->x[segment_idx]);
        float center_y = lane->y[segment_idx] + pose_t * (lane->y[segment_idx + 1] - lane->y[segment_idx]);
        pose->x = center_x - sinf(heading_radians) * lateral_offset_meters;
        pose->y = center_y + cosf(heading_radians) * lateral_offset_meters;
        pose->z = lane->z[segment_idx] + pose_t * (lane->z[segment_idx + 1] - lane->z[segment_idx]);
        pose->heading_radians = heading_radians;
        return 1;
    }
    return 0;
}

// Pose distance_meters along the route from the projection; 0 once the route ends before it
static int pdm_sample_route_pose(
    Drive *env,
    const Agent *agent,
    IDMLaneProjection projection,
    float distance_meters,
    float lateral_offset_meters,
    PDMRoutePose *pose) {
    float traveled_meters = 0.0f;
    for (int route_idx = projection.route_idx; route_idx < agent->route_length; route_idx++) {
        RoadMapElement *lane = pdm_route_lane(env, agent, route_idx);
        if (lane == NULL) {
            return 0;
        }
        int is_projection_lane = route_idx == projection.route_idx;
        if (pdm_sample_lane_pose(
                lane,
                is_projection_lane ? projection.segment_idx : 0,
                is_projection_lane ? projection.t : 0.0f,
                distance_meters,
                lateral_offset_meters,
                &traveled_meters,
                pose)) {
            pose->lane_idx = agent->route[route_idx];
            pose->route_idx = route_idx;
            return 1;
        }
    }
    return 0;
}

// Pure-pursuit aim point a speed-scaled lookahead beyond the target; the target itself near the route end
static PDMRoutePose pdm_sample_pursuit_aim(
    Drive *env,
    const Agent *agent,
    IDMLaneProjection projection,
    float lateral_offset_meters,
    float target_distance_meters,
    float speed_mps,
    PDMRoutePose target) {
    float lookahead_meters = clip(
        PDM_PURSUIT_LOOKAHEAD_SECONDS * speed_mps,
        PDM_PURSUIT_MIN_LOOKAHEAD_METERS,
        PDM_PURSUIT_MAX_LOOKAHEAD_METERS);
    PDMRoutePose aim;
    if (pdm_sample_route_pose(
            env,
            agent,
            projection,
            target_distance_meters + lookahead_meters,
            lateral_offset_meters,
            &aim)) {
        return aim;
    }
    return target;
}

// One kinematic bicycle step toward the aim point, speed driven to target_speed_mps within the accel limits
static PDMRolloutStep pdm_track_step(
    const Drive *env,
    const Agent *agent,
    PDMRolloutStep state,
    PDMRoutePose aim,
    float target_speed_mps,
    float duration_seconds) {
    float accel_mps2
        = clip((target_speed_mps - state.speed_mps) / duration_seconds, -PDM_MAX_DECEL_MPS2, PDM_MAX_ACCEL_MPS2);
    float speed_mps = clip(state.speed_mps + accel_mps2 * duration_seconds, 0.0f, env->base_max_speed_mps);

    // pure pursuit: curvature = 2 sin(angle to the aim point) / distance to it
    float aim_dx = aim.x - state.x;
    float aim_dy = aim.y - state.y;
    float aim_distance_meters = sqrtf(aim_dx * aim_dx + aim_dy * aim_dy);
    float steering_radians = 0.0f;
    if (aim_distance_meters > PDM_MIN_AIM_DISTANCE_METERS) {
        float aim_angle_radians = compute_heading_diff(atan2f(aim_dy, aim_dx), state.heading_radians);
        float curvature = 2.0f * sinf(aim_angle_radians) / aim_distance_meters;
        steering_radians = atanf(curvature * agent->wheelbase);
    }
    float max_steering_step_radians = MAX_STEERING_RATE_RADPS * duration_seconds;
    float steering_step_radians
        = clip(steering_radians - state.steering_angle_radians, -max_steering_step_radians, max_steering_step_radians);
    steering_radians = clip(state.steering_angle_radians + steering_step_radians, -STEERING_LIMIT, STEERING_LIMIT);

    float slip_angle_radians = atanf(PDM_REAR_AXLE_RATIO * tanf(steering_radians));
    float yaw_rate_radps = (speed_mps > PDM_SPEED_EPSILON_MPS)
        ? speed_mps * cosf(slip_angle_radians) * tanf(steering_radians) / agent->wheelbase
        : 0.0f;
    state.x += speed_mps * cosf(state.heading_radians + slip_angle_radians) * duration_seconds;
    state.y += speed_mps * sinf(state.heading_radians + slip_angle_radians) * duration_seconds;
    state.z = aim.z;
    state.heading_radians = normalize_heading(state.heading_radians + yaw_rate_radps * duration_seconds);
    state.speed_mps = speed_mps;
    state.steering_angle_radians = steering_radians;
    return state;
}

static PDMRolloutStep pdm_current_state(const Agent *agent) {
    return (PDMRolloutStep) {
        .x = agent->sim_x,
        .y = agent->sim_y,
        .z = agent->sim_z,
        .heading_radians = agent->sim_heading,
        .speed_mps = fmaxf(0.0f, agent->sim_speed_signed),
        .steering_angle_radians = agent->steering_angle,
    };
}

// The agent's pose after one env dt at speed_mps along the offset route; 0 when the route ends before it
static int pdm_build_first_step(
    Drive *env,
    const Agent *agent,
    IDMLaneProjection projection,
    float lateral_offset_meters,
    float speed_mps,
    PDMRolloutStep *step) {
    float distance_meters = speed_mps * env->dt;
    PDMRoutePose target;
    if (!pdm_sample_route_pose(env, agent, projection, distance_meters, lateral_offset_meters, &target)) {
        return 0;
    }
    PDMRoutePose aim
        = pdm_sample_pursuit_aim(env, agent, projection, lateral_offset_meters, distance_meters, speed_mps, target);
    *step = pdm_track_step(env, agent, pdm_current_state(agent), aim, speed_mps, env->dt);
    step->time_seconds = env->dt;
    step->lane_idx = target.lane_idx;
    step->route_idx = target.route_idx;
    return 1;
}

// Speed and distance at time_seconds: constant accel until target_speed_mps is reached, then held
static void pdm_speed_profile(
    float current_speed_mps,
    float accel_mps2,
    float target_speed_mps,
    float time_seconds,
    float max_speed_mps,
    float *speed_mps_out,
    float *distance_meters_out) {
    float reach_seconds
        = (fabsf(accel_mps2) > PDM_ACCEL_EPSILON_MPS2) ? (target_speed_mps - current_speed_mps) / accel_mps2 : INFINITY;
    if (!(reach_seconds > 0.0f)) {
        reach_seconds = INFINITY;
    }
    float speed_mps = target_speed_mps;
    float distance_meters;
    if (time_seconds <= reach_seconds) {
        speed_mps = current_speed_mps + accel_mps2 * time_seconds;
        distance_meters = current_speed_mps * time_seconds + 0.5f * accel_mps2 * time_seconds * time_seconds;
    } else {
        distance_meters = current_speed_mps * reach_seconds + 0.5f * accel_mps2 * reach_seconds * reach_seconds
            + target_speed_mps * (time_seconds - reach_seconds);
    }
    *speed_mps_out = clip(speed_mps, 0.0f, max_speed_mps);
    *distance_meters_out = fmaxf(distance_meters, 0.0f);
}

static PDMRollout pdm_generate_rollout(
    Drive *env,
    const Agent *agent,
    IDMLaneProjection projection,
    float lateral_offset_meters,
    float accel_mps2,
    float target_speed_mps) {
    PDMRollout rollout = {0};
    float current_speed_mps = fmaxf(0.0f, agent->sim_speed_signed);
    float next_speed_mps = clip(current_speed_mps + accel_mps2 * env->dt, 0.0f, env->base_max_speed_mps);
    rollout.valid
        = pdm_build_first_step(env, agent, projection, lateral_offset_meters, next_speed_mps, &rollout.first_step);
    if (!rollout.valid) {
        return rollout;
    }

    // the sim's own time step: a coarser rollout predicts a wider turn than the agent then drives
    float horizon_seconds = pdm_agent_horizon_seconds(env, agent);
    float rollout_dt_seconds = fmaxf(env->dt, horizon_seconds / (float) (PDM_MAX_ROLLOUT_STEPS - 1));
    PDMRolloutStep state = pdm_current_state(agent);
    for (int step_idx = 0; step_idx < PDM_MAX_ROLLOUT_STEPS; step_idx++) {
        float time_seconds = step_idx * rollout_dt_seconds;
        if (time_seconds > horizon_seconds + PDM_TIME_EPSILON_SECONDS) {
            break;
        }
        float speed_mps, distance_meters;
        pdm_speed_profile(
            current_speed_mps,
            accel_mps2,
            target_speed_mps,
            time_seconds,
            env->base_max_speed_mps,
            &speed_mps,
            &distance_meters);
        PDMRoutePose target;
        if (!pdm_sample_route_pose(env, agent, projection, distance_meters, lateral_offset_meters, &target)) {
            // a logged route may end inside the horizon; a gigaflow route that does is a dead end
            rollout.valid = env->simulation_mode == SIMULATION_MODE_REPLAY;
            break;
        }
        if (step_idx > 0) {
            PDMRoutePose aim = pdm_sample_pursuit_aim(
                env,
                agent,
                projection,
                lateral_offset_meters,
                distance_meters,
                speed_mps,
                target);
            state = pdm_track_step(env, agent, state, aim, speed_mps, time_seconds - state.time_seconds);
        }
        state.time_seconds = time_seconds;
        state.lane_idx = target.lane_idx;
        state.route_idx = target.route_idx;
        rollout.steps[rollout.step_count++] = state;
    }
    return rollout;
}

static Agent pdm_make_sample_agent(const Agent *ego, float x, float y, float z, float heading_radians) {
    Agent sample = idm_make_sample_agent(ego);
    sample.sim_x = x;
    sample.sim_y = y;
    sample.sim_z = z;
    sample.sim_heading = normalize_heading(heading_radians);
    sample.cos_heading = cosf(sample.sim_heading);
    sample.sin_heading = sinf(sample.sim_heading);
    copy_pose_to_prev(&sample);
    return sample;
}

static int pdm_collect_nearby_agents(Drive *env, int ego_idx, float range_meters, int *nearby_agent_indices) {
    Agent *ego = &env->agents[ego_idx];
    int nearby_count = 0;
    for (int i = 0; i < env->num_agents && nearby_count < PDM_MAX_NEARBY_AGENTS; i++) {
        int other_idx = (i < env->active_agent_count) ? env->active_agent_indices[i]
                                                      : env->static_agent_indices[i - env->active_agent_count];
        Agent *other = &env->agents[other_idx];
        float dx = other->sim_x - ego->sim_x;
        float dy = other->sim_y - ego->sim_y;
        float max_distance_meters = range_meters + 0.5f * ego->sim_length + 0.5f * other->sim_length
            + PDM_AGENT_QUERY_MARGIN_METERS + 2.0f * IDM_BBOX_MARGIN_METERS;
        if (other_idx == ego_idx || other->removed || other->sim_x == INVALID_POSITION || other->sim_valid == 0
            || dx * dx + dy * dy > max_distance_meters * max_distance_meters) {
            continue;
        }
        nearby_agent_indices[nearby_count++] = other_idx;
    }
    return nearby_count;
}

// Sample box against the other agent's box moved prediction_seconds along its velocity, both grown by the PDM margin
static int pdm_sample_hits_agent(const Agent *sample, const Agent *other, float prediction_seconds) {
    float other_x = other->sim_x + other->sim_vx * prediction_seconds;
    float other_y = other->sim_y + other->sim_vy * prediction_seconds;
    float dx = other_x - sample->sim_x;
    float dy = other_y - sample->sim_y;
    float reach_meters = 0.5f * sample->sim_length + 0.5f * other->sim_length + sample->sim_width + other->sim_width
        + IDM_COLLISION_RADIUS_MARGIN_METERS + 2.0f * IDM_BBOX_MARGIN_METERS;
    if (!idm_check_z_overlap(sample, other) || dx * dx + dy * dy > reach_meters * reach_meters) {
        return 0;
    }

    Agent grown_sample = *sample;
    grown_sample.sim_length = sample->sim_length + 2.0f * PDM_COLLISION_BBOX_MARGIN_METERS;
    grown_sample.sim_width = sample->sim_width + 2.0f * PDM_COLLISION_BBOX_MARGIN_METERS;
    Agent grown_other = *other;
    grown_other.sim_x = other_x;
    grown_other.sim_y = other_y;
    grown_other.sim_length = other->sim_length + 2.0f * (IDM_BBOX_MARGIN_METERS + PDM_COLLISION_BBOX_MARGIN_METERS);
    grown_other.sim_width = other->sim_width + 2.0f * (IDM_BBOX_MARGIN_METERS + PDM_COLLISION_BBOX_MARGIN_METERS);
    return check_obb_collision(&grown_sample, &grown_other);
}

static int pdm_first_agent_hit(
    Drive *env,
    const Agent *sample,
    const int *agent_indices,
    int agent_count,
    float prediction_seconds) {
    for (int i = 0; i < agent_count; i++) {
        if (pdm_sample_hits_agent(sample, &env->agents[agent_indices[i]], prediction_seconds)) {
            return agent_indices[i];
        }
    }
    return -1;
}

// First red light or agent box met by the ego box swept along the offset route
static IDMLeader pdm_find_leader(Drive *env, int agent_idx, IDMLaneProjection projection, float lateral_offset_meters) {
    Agent *agent = &env->agents[agent_idx];
    IDMLeader leader = idm_no_leader();
    float lookahead_meters = clip(
        fmaxf(0.0f, agent->sim_speed_signed) * PDM_LEADER_LOOKAHEAD_SECONDS,
        PDM_MIN_LEADER_LOOKAHEAD_METERS,
        PDM_MAX_LEADER_LOOKAHEAD_METERS);
    int nearby_agent_indices[PDM_MAX_NEARBY_AGENTS];
    int nearby_count = pdm_collect_nearby_agents(env, agent_idx, lookahead_meters, nearby_agent_indices);

    for (float distance_meters = IDM_ROUTE_SAMPLE_DISTANCE_METERS;
         distance_meters <= lookahead_meters + PDM_DISTANCE_EPSILON_METERS;
         distance_meters += IDM_ROUTE_SAMPLE_DISTANCE_METERS) {
        PDMRoutePose pose;
        if (!pdm_sample_route_pose(env, agent, projection, distance_meters, lateral_offset_meters, &pose)) {
            break;
        }
        Agent sample = pdm_make_sample_agent(agent, pose.x, pose.y, pose.z, pose.heading_radians);
        if (idm_sample_hits_red_light(env, &sample, pose.lane_idx)) {
            idm_update_best_leader(&leader, -1, 1, distance_meters, 0.0f);
            break;
        }
        int hit_agent_idx = pdm_first_agent_hit(env, &sample, nearby_agent_indices, nearby_count, 0.0f);
        if (hit_agent_idx != -1) {
            Agent *hit_agent = &env->agents[hit_agent_idx];
            float leader_speed_mps = hit_agent->sim_vx * sample.cos_heading + hit_agent->sim_vy * sample.sin_heading;
            idm_update_best_leader(&leader, hit_agent_idx, 0, distance_meters, leader_speed_mps);
            break;
        }
    }
    return leader;
}

// agents entirely behind the ego's rear bumper are theirs to avoid
static int pdm_agent_is_ahead_of_rear_bumper(const Agent *ego, const Agent *other) {
    float dx = other->sim_x - ego->sim_x;
    float dy = other->sim_y - ego->sim_y;
    float longitudinal_meters = dx * ego->cos_heading + dy * ego->sin_heading;
    float length_projection = fabsf(other->cos_heading * ego->cos_heading + other->sin_heading * ego->sin_heading);
    float width_projection = fabsf(-other->sin_heading * ego->cos_heading + other->cos_heading * ego->sin_heading);
    float other_half_extent_meters
        = 0.5f * other->sim_length * length_projection + 0.5f * other->sim_width * width_projection;
    return longitudinal_meters + other_half_extent_meters >= -0.5f * ego->sim_length;
}

// Time of the first rollout step that overlaps an agent moving on at its current velocity
static float pdm_collision_ttc(Drive *env, int agent_idx, const PDMCandidate *candidate, float horizon_seconds) {
    Agent *agent = &env->agents[agent_idx];
    float range_meters
        = candidate->next_speed_mps * horizon_seconds + 0.5f * agent->sim_length + PDM_COLLISION_QUERY_MARGIN_METERS;
    int ahead_agent_indices[PDM_MAX_NEARBY_AGENTS];
    int nearby_count = pdm_collect_nearby_agents(env, agent_idx, range_meters, ahead_agent_indices);
    int ahead_count = 0;
    for (int i = 0; i < nearby_count; i++) {
        if (pdm_agent_is_ahead_of_rear_bumper(agent, &env->agents[ahead_agent_indices[i]])) {
            ahead_agent_indices[ahead_count++] = ahead_agent_indices[i];
        }
    }

    for (int step_idx = 1; step_idx < candidate->rollout.step_count; step_idx++) {
        const PDMRolloutStep *step = &candidate->rollout.steps[step_idx];
        Agent sample = pdm_make_sample_agent(agent, step->x, step->y, step->z, step->heading_radians);
        if (pdm_first_agent_hit(env, &sample, ahead_agent_indices, ahead_count, step->time_seconds) != -1) {
            return step->time_seconds;
        }
    }
    return horizon_seconds;
}

// lights keep their current state over the horizon
static float pdm_traffic_light_ttc(
    Drive *env,
    const Agent *agent,
    const PDMCandidate *candidate,
    float horizon_seconds) {
    for (int step_idx = 1; step_idx < candidate->rollout.step_count; step_idx++) {
        const PDMRolloutStep *step = &candidate->rollout.steps[step_idx];
        Agent sample = pdm_make_sample_agent(agent, step->x, step->y, step->z, step->heading_radians);
        if (idm_sample_hits_red_light(env, &sample, step->lane_idx)) {
            return step->time_seconds;
        }
    }
    return horizon_seconds;
}

static int pdm_step_is_offroad(Drive *env, const Agent *agent, const PDMRolloutStep *step) {
    if (get_grid_index(env, step->x, step->y) == -1) {
        return 1;
    }
    Agent sample = *agent;
    sample.sim_x = step->x;
    sample.sim_y = step->y;
    sample.sim_z = step->z;
    sample.cos_heading = cosf(step->heading_radians);
    sample.sin_heading = sinf(step->heading_radians);
    copy_pose_to_prev(&sample); // static probe: the swept road-edge check degenerates to a static box test

    GridMapEntity entity_list[ROAD_QUERY_ENTITY_COUNT];
    int entity_count = get_neighbors_entities(
        env,
        step->x,
        step->y,
        entity_list,
        ROAD_QUERY_ENTITY_COUNT,
        ROAD_OFFSETS,
        ROAD_QUERY_CELL_COUNT);
    for (int i = 0; i < entity_count; i++) {
        RoadMapElement *element = &env->road_elements[entity_list[i].entity_idx];
        if (is_road_edge(element->type)
            && road_edge_segment_crosses_agent(element, entity_list[i].geometry_idx, &sample)) {
            return 1;
        }
    }
    return 0;
}

static float pdm_offroad_ttc(Drive *env, const Agent *agent, const PDMCandidate *candidate, float horizon_seconds) {
    for (int step_idx = 1; step_idx < candidate->rollout.step_count; step_idx++) {
        if (pdm_step_is_offroad(env, agent, &candidate->rollout.steps[step_idx])) {
            return candidate->rollout.steps[step_idx].time_seconds;
        }
    }
    return horizon_seconds;
}

static PDMCandidate pdm_build_candidate(
    Drive *env,
    int agent_idx,
    IDMLaneProjection projection,
    float lateral_offset_meters,
    IDMLeader leader,
    float target_speed_mps,
    float speed_limit_mps) {
    Agent *agent = &env->agents[agent_idx];
    float current_speed_mps = fmaxf(0.0f, agent->sim_speed_signed);
    float accel_mps2
        = clip(pdm_compute_acceleration(env, agent, target_speed_mps, leader), -PDM_MAX_DECEL_MPS2, PDM_MAX_ACCEL_MPS2);
    float next_speed_mps = current_speed_mps + accel_mps2 * env->dt;
    if (next_speed_mps < 0.0f) {
        next_speed_mps = 0.0f;
    }
    accel_mps2 = (next_speed_mps - current_speed_mps) / env->dt;

    PDMCandidate candidate = {
        .lateral_offset_meters = lateral_offset_meters,
        .next_speed_mps = next_speed_mps,
        .score = -INFINITY,
        .rollout = pdm_generate_rollout(env, agent, projection, lateral_offset_meters, accel_mps2, target_speed_mps),
    };
    candidate.valid = candidate.rollout.valid;
    if (!candidate.valid) {
        return candidate;
    }

    float horizon_seconds = pdm_agent_horizon_seconds(env, agent);
    candidate.min_ttc_seconds = fminf(
        pdm_collision_ttc(env, agent_idx, &candidate, horizon_seconds),
        pdm_traffic_light_ttc(env, agent, &candidate, horizon_seconds));
    float ttc_score = clip(candidate.min_ttc_seconds / horizon_seconds, 0.0f, 1.0f);
    float speed_score = clip(next_speed_mps / fmaxf(speed_limit_mps, PDM_MIN_DESIRED_SPEED_MPS), 0.0f, 1.0f);
    // no reward for speed toward something reached within the horizon
    if (candidate.min_ttc_seconds < horizon_seconds && candidate.min_ttc_seconds <= PDM_SAFE_SPEED_TTC_SECONDS) {
        speed_score = 0.0f;
    }
    float collision_penalty
        = (candidate.min_ttc_seconds < pdm_danger_ttc_seconds(current_speed_mps)) ? PDM_COLLISION_PENALTY : 0.0f;
    float center_bonus = (fabsf(lateral_offset_meters) < PDM_DISTANCE_EPSILON_METERS) ? PDM_CENTER_BONUS : 0.0f;
    candidate.score = -collision_penalty + PDM_SPEED_WEIGHT * speed_score + PDM_TTC_WEIGHT * ttc_score + center_bonus;
    return candidate;
}

static void pdm_build_candidates(Drive *env, int agent_idx, IDMLaneProjection projection, PDMCandidate *candidates) {
    float speed_limit_mps = pdm_desired_speed(env, &env->agents[agent_idx]);
    for (int offset_idx = 0; offset_idx < PDM_LATERAL_OFFSET_COUNT; offset_idx++) {
        float lateral_offset_meters = PDM_LATERAL_OFFSETS_METERS[offset_idx];
        IDMLeader leader = pdm_find_leader(env, agent_idx, projection, lateral_offset_meters);
        for (int speed_idx = 0; speed_idx < PDM_SPEED_FRACTION_COUNT; speed_idx++) {
            candidates[offset_idx * PDM_SPEED_FRACTION_COUNT + speed_idx] = pdm_build_candidate(
                env,
                agent_idx,
                projection,
                lateral_offset_meters,
                leader,
                PDM_SPEED_FRACTIONS[speed_idx] * speed_limit_mps,
                speed_limit_mps);
        }
    }
}

static PDMCandidate *pdm_best_unchecked_candidate(PDMCandidate *candidates, int *checked) {
    int best_idx = -1;
    for (int i = 0; i < PDM_CANDIDATE_COUNT; i++) {
        if (checked[i] || !candidates[i].valid) {
            continue;
        }
        if (best_idx == -1 || candidates[i].score > candidates[best_idx].score) {
            best_idx = i;
        }
    }
    if (best_idx == -1) {
        return NULL;
    }
    checked[best_idx] = 1;
    return &candidates[best_idx];
}

// Best-scored candidate that stays on the road; if all leave it, the one that leaves last. NULL without a valid one
static PDMCandidate *pdm_select_candidate(Drive *env, int agent_idx, PDMCandidate *candidates) {
    Agent *agent = &env->agents[agent_idx];
    float horizon_seconds = pdm_agent_horizon_seconds(env, agent);
    int checked[PDM_CANDIDATE_COUNT] = {0};
    PDMCandidate *latest_offroad = NULL;
    for (int rank = 0; rank < PDM_CANDIDATE_COUNT; rank++) {
        PDMCandidate *candidate = pdm_best_unchecked_candidate(candidates, checked);
        if (candidate == NULL) {
            break;
        }
        // the road-edge test is the costly one, so it runs in score order and stops at the first clean rollout
        candidate->offroad_ttc_seconds = pdm_offroad_ttc(env, agent, candidate, horizon_seconds);
        candidate->min_ttc_seconds = fminf(candidate->min_ttc_seconds, candidate->offroad_ttc_seconds);
        if (candidate->offroad_ttc_seconds >= horizon_seconds) {
            return candidate;
        }
        if (latest_offroad == NULL || candidate->offroad_ttc_seconds > latest_offroad->offroad_ttc_seconds
            || (candidate->offroad_ttc_seconds == latest_offroad->offroad_ttc_seconds
                && candidate->score > latest_offroad->score)) {
            latest_offroad = candidate;
        }
    }
    return latest_offroad;
}

static void pdm_set_motion(
    Drive *env,
    Agent *agent,
    float speed_mps,
    float previous_speed_mps,
    float previous_heading) {
    float accel_mps2 = (speed_mps - previous_speed_mps) / env->dt;
    float yaw_rate_radps = compute_heading_diff(agent->sim_heading, previous_heading) / env->dt;
    float lateral_accel_mps2 = speed_mps * yaw_rate_radps;
    agent->sim_vx = speed_mps * agent->cos_heading;
    agent->sim_vy = speed_mps * agent->sin_heading;
    agent->jerk_long = (accel_mps2 - agent->accel_long) / env->dt;
    agent->jerk_lat = (lateral_accel_mps2 - agent->accel_lat) / env->dt;
    agent->accel_long = accel_mps2;
    agent->accel_lat = lateral_accel_mps2;
    update_agent_speed(agent);
}

static void pdm_apply_step(Drive *env, Agent *agent, const PDMRolloutStep *step) {
    float previous_speed_mps = fmaxf(0.0f, agent->sim_speed_signed);
    float previous_heading = agent->sim_heading;
    agent->sim_x = step->x;
    agent->sim_y = step->y;
    agent->sim_z = step->z;
    agent->sim_heading = normalize_heading(step->heading_radians);
    agent->cos_heading = cosf(agent->sim_heading);
    agent->sin_heading = sinf(agent->sim_heading);
    agent->steering_angle = step->steering_angle_radians;
    agent->current_route_idx = step->route_idx;
    agent->current_lane_idx = step->lane_idx;
    pdm_set_motion(env, agent, step->speed_mps, previous_speed_mps, previous_heading);
}

// Nothing to plan on: brake as hard as allowed along the route, or straight ahead once the route has ended
static void pdm_apply_brake_fallback(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    float previous_speed_mps = fmaxf(0.0f, agent->sim_speed_signed);
    float speed_mps = fmaxf(0.0f, previous_speed_mps - PDM_MAX_DECEL_MPS2 * env->dt);
    float distance_meters = speed_mps * env->dt;
    float previous_heading = agent->sim_heading;
    if (!idm_advance_along_route_lanes_limited(env, agent_idx, distance_meters, speed_mps, &previous_heading)) {
        agent->sim_x += distance_meters * agent->cos_heading;
        agent->sim_y += distance_meters * agent->sin_heading;
    }
    agent->steering_angle = 0.0f;
    pdm_set_motion(env, agent, speed_mps, previous_speed_mps, previous_heading);
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

    IDMLaneProjection projection = idm_project_to_route_lanes(env, agent);
    PDMCandidate candidates[PDM_CANDIDATE_COUNT];
    PDMCandidate *best = NULL;
    if (pdm_prepare_route(env, agent, &projection)) {
        pdm_build_candidates(env, agent_idx, projection, candidates);
        best = pdm_select_candidate(env, agent_idx, candidates);
    }
    if (best == NULL) {
        pdm_apply_brake_fallback(env, agent_idx);
        return;
    }

    // the chosen rollout still meets something too soon: keep its path, brake as hard as allowed
    PDMRolloutStep step = best->rollout.first_step;
    float current_speed_mps = fmaxf(0.0f, agent->sim_speed_signed);
    float urgent_speed_mps = fmaxf(0.0f, current_speed_mps - PDM_MAX_DECEL_MPS2 * env->dt);
    if (best->min_ttc_seconds < pdm_danger_ttc_seconds(current_speed_mps)
        && !pdm_build_first_step(env, agent, projection, best->lateral_offset_meters, urgent_speed_mps, &step)) {
        pdm_apply_brake_fallback(env, agent_idx);
        return;
    }
    pdm_apply_step(env, agent, &step);
}

#endif
