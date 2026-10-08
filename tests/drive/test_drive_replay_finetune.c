#include "include/drive_fixture.h"
#include "include/test.h"

static void init_similarity_env(Drive *env, Agent *agent, Log *log, int *active, float *reward) {
    memset(env, 0, sizeof(*env));
    memset(agent, 0, sizeof(*agent));
    memset(log, 0, sizeof(*log));
    *agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    active[0] = 0;
    env->agents = agent;
    env->active_agent_indices = active;
    env->logs = log;
    env->rewards = reward;
    env->active_agent_count = 1;
    env->dt = 0.1f;
    env->num_goals = 3;
    env->simulation_mode = SIMULATION_MODE_REPLAY;
    env->reward_expert_similarity = 0.01f;
    agent->goal_count = 3;
    agent->metrics_array[LANE_ANGLE_IDX] = 1.0f;
}

static int test_expert_similarity_reward_is_quadratic(void) {
    Drive env;
    Agent agent;
    Log log;
    int active[1];
    float reward[1] = {0};
    float log_x[2] = {0.0f, 10.0f};
    float log_y[2] = {0.0f, 10.0f};
    int log_valid[2] = {1, 0};

    init_similarity_env(&env, &agent, &log, active, reward);
    agent.log_trajectory_x = log_x;
    agent.log_trajectory_y = log_y;
    agent.log_valid = log_valid;
    agent.trajectory_size = 2;
    agent.sim_x = 3.0f;
    agent.sim_y = 4.0f;
    env.timestep = 0;
    compute_rewards(&env, 0);
    EXPECT_NEAR(log.reward_expert_similarity, -0.01f * 25.0f, 1e-6f);
    EXPECT_NEAR(env.rewards[0], -0.01f * 25.0f, 1e-6f);

    init_similarity_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.log_trajectory_x = log_x;
    agent.log_trajectory_y = log_y;
    agent.log_valid = log_valid;
    agent.trajectory_size = 2;
    agent.sim_x = 3.0f;
    agent.sim_y = 4.0f;
    env.timestep = 1; // logged pose invalid at this step
    compute_rewards(&env, 0);
    EXPECT_NEAR(log.reward_expert_similarity, 0.0f, 1e-6f);

    init_similarity_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.log_trajectory_x = log_x;
    agent.log_trajectory_y = log_y;
    agent.log_valid = log_valid;
    agent.trajectory_size = 2;
    agent.sim_x = 3.0f;
    agent.sim_y = 4.0f;
    env.timestep = 0;
    env.simulation_mode = SIMULATION_MODE_GIGAFLOW;
    compute_rewards(&env, 0);
    EXPECT_NEAR(log.reward_expert_similarity, 0.0f, 1e-6f);
    return 0;
}

static int test_episode_max_steps_truncates(void) {
    srand(5);
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    env.episode_max_steps = 5;
    allocate(&env);
    c_reset(&env);
    EXPECT_EQ_INT(env.active_agent_count, 1);

    for (int step = 1; step <= env.episode_max_steps; step++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
        EXPECT_EQ_INT(env.truncations[0], step == env.episode_max_steps);
    }
    EXPECT_NEAR(env.log.n, 1.0f, 1e-6f);
    EXPECT_NEAR(env.log.episode_length, (float) env.episode_max_steps, 1e-6f);
    free_allocated(&env);
    return 0;
}

static int test_init_step_jitter_resamples_within_range(void) {
    srand(5);
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    env.init_step_base = 2;
    env.init_step_jitter_steps = 5;
    allocate(&env);

    int min_seen = 1 << 20;
    int max_seen = -1;
    for (int episode = 0; episode < 24; episode++) {
        env.timestep = -1;
        c_reset(&env);
        EXPECT_TRUE(env.init_step >= env.init_step_base);
        EXPECT_TRUE(env.init_step <= env.init_step_base + env.init_step_jitter_steps);
        EXPECT_EQ_INT(env.timestep, env.init_step);
        min_seen = env.init_step < min_seen ? env.init_step : min_seen;
        max_seen = env.init_step > max_seen ? env.init_step : max_seen;
    }
    EXPECT_TRUE(max_seen > min_seen);

    env.eval_mode = 1;
    env.timestep = -1;
    c_reset(&env);
    EXPECT_TRUE(env.init_step >= env.init_step_base);
    EXPECT_TRUE(env.init_step <= env.init_step_base + env.init_step_jitter_steps);
    free_allocated(&env);
    return 0;
}

static int test_static_expert_is_flagged_and_masked(void) {
    srand(5);
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    env.episode_max_steps = 10;
    env.static_expert_min_motion_m = 1e6f;
    allocate(&env);
    EXPECT_EQ_INT(env.active_agent_count, 1);
    Agent *ego = &env.agents[env.active_agent_indices[0]];
    EXPECT_TRUE(ego->log_valid[0] && ego->log_valid[env.episode_max_steps]);

    env.timestep = -1;
    c_reset(&env);
    EXPECT_EQ_INT(ego->is_static_expert, 1);
    drive_set_neutral_actions(&env);
    c_step(&env);
    EXPECT_EQ_INT(env.masks[0], 0);

    env.static_expert_min_motion_m = 0.0f;
    env.timestep = -1;
    c_reset(&env);
    EXPECT_EQ_INT(ego->is_static_expert, 0);
    drive_set_neutral_actions(&env);
    c_step(&env);
    EXPECT_EQ_INT(env.masks[0], 1);
    free_allocated(&env);
    return 0;
}

static int test_expert_similarity_kernel_is_positive_and_bounded(void) {
    Drive env;
    Agent agent;
    Log log;
    int active[1];
    float reward[1] = {0};
    float log_x[2] = {0.0f, 10.0f};
    float log_y[2] = {0.0f, 10.0f};
    int log_valid[2] = {1, 0};

    init_similarity_env(&env, &agent, &log, active, reward);
    agent.log_trajectory_x = log_x;
    agent.log_trajectory_y = log_y;
    agent.log_valid = log_valid;
    agent.trajectory_size = 2;
    agent.sim_x = 3.0f;
    agent.sim_y = 4.0f;
    env.expert_similarity_kernel_m = 5.0f;
    env.timestep = 0;
    compute_rewards(&env, 0);
    EXPECT_NEAR(log.reward_expert_similarity, 0.01f * expf(-25.0f / 25.0f), 1e-6f);

    init_similarity_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.log_trajectory_x = log_x;
    agent.log_trajectory_y = log_y;
    agent.log_valid = log_valid;
    agent.trajectory_size = 2;
    agent.sim_x = 0.0f;
    agent.sim_y = 0.0f;
    env.expert_similarity_kernel_m = 5.0f;
    env.timestep = 0;
    compute_rewards(&env, 0);
    EXPECT_NEAR(log.reward_expert_similarity, 0.01f, 1e-6f);

    init_similarity_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.log_trajectory_x = log_x;
    agent.log_trajectory_y = log_y;
    agent.log_valid = log_valid;
    agent.trajectory_size = 2;
    agent.sim_x = 0.0f;
    agent.sim_y = 0.0f;
    env.expert_similarity_kernel_m = 5.0f;
    env.timestep = 1; // invalid logged pose pays nothing, not the full reward
    compute_rewards(&env, 0);
    EXPECT_NEAR(log.reward_expert_similarity, 0.0f, 1e-6f);
    return 0;
}

static int test_expert_similarity_only_drops_rl_terms(void) {
    Drive env;
    Agent agent;
    Log log;
    int active[1];
    float reward[1] = {0};
    float log_x[1] = {0.0f};
    float log_y[1] = {0.0f};
    int log_valid[1] = {1};

    init_similarity_env(&env, &agent, &log, active, reward);
    agent.log_trajectory_x = log_x;
    agent.log_trajectory_y = log_y;
    agent.log_valid = log_valid;
    agent.trajectory_size = 1;
    agent.sim_x = 3.0f;
    agent.sim_y = 4.0f;
    agent.metrics_array[COLLISION_IDX] = 1.0f;
    agent.reward_coefs[REWARD_COEF_COLLISION] = 3.0f;
    env.expert_similarity_only = 1;
    env.timestep = 0;
    compute_rewards(&env, 0);
    EXPECT_NEAR(env.rewards[0], -0.01f * 25.0f, 1e-6f);
    EXPECT_NEAR(log.reward_collision, -3.0f, 1e-6f);
    EXPECT_NEAR(log.episode_return, -0.01f * 25.0f, 1e-6f);
    return 0;
}

static int test_expert_tracking_follows_log(void) {
    srand(5);
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    env.sdc_controller = CONTROLLER_EXPERT_TRACKING;
    env.dynamics_model = DYNAMICS_MODEL_JERK;
    env.scenario_length = 60;
    allocate(&env);
    c_reset(&env);
    EXPECT_EQ_INT(env.active_agent_count, 1);
    Agent *ego = &env.agents[env.active_agent_indices[0]];
    int num_actions = NUM_JERK_LONG_ACTIONS * NUM_JERK_LAT_ACTIONS;

    for (int step = 1; step <= env.scenario_length; step++) {
        c_step(&env);
        int chosen_action = ((int *) env.actions)[0];
        EXPECT_TRUE(chosen_action >= 0 && chosen_action < num_actions);
        for (int action = 0; action < num_actions; action++) {
            EXPECT_FINITE(env.expert_tracking_errors[action]);
            EXPECT_TRUE(env.expert_tracking_errors[chosen_action] <= env.expert_tracking_errors[action]);
        }
        if (step < env.scenario_length) {
            EXPECT_EQ_INT(env.masks[0], 1);
            EXPECT_TRUE(compute_displacement_error(ego, env.timestep) < 0.5f);
        }
    }
    EXPECT_NEAR(env.log.n, 1.0f, 1e-6f);
    EXPECT_TRUE(env.log.avg_displacement_error < 0.1f);
    EXPECT_TRUE(env.log.final_displacement_error >= 0.0f && env.log.final_displacement_error < 0.5f);
    free_allocated(&env);
    return 0;
}

static int test_reset_seeds_dynamics_state_from_log(void) {
    srand(5);
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    env.dynamics_model = DYNAMICS_MODEL_JERK;
    env.dt = 0.1f;
    allocate(&env);
    Agent *ego = &env.agents[env.active_agent_indices[0]];

    // Pick a logged step where the ego is clearly accelerating or braking
    int step = -1;
    for (int t = 1; t + 1 < ego->trajectory_size; t++) {
        if (!ego->log_valid[t - 1] || !ego->log_valid[t] || !ego->log_valid[t + 1]) {
            continue;
        }
        float speed_change = log_position_speed(ego, t + 1, 0.1f) - log_position_speed(ego, t - 1, 0.1f);
        if (fabsf(speed_change) > 0.1f) {
            step = t;
            break;
        }
    }
    EXPECT_TRUE(step > 0);

    env.init_step = step;
    env.init_step_base = step;
    env.timestep = -1;
    c_reset(&env);
    float expected_accel_long
        = (log_position_speed(ego, step + 1, env.dt) - log_position_speed(ego, step - 1, env.dt)) / (2.0f * env.dt);
    expected_accel_long = clip(expected_accel_long, ACCEL_LONG_LIMIT[0], ACCEL_LONG_LIMIT[1]);
    EXPECT_NEAR(ego->accel_long, expected_accel_long, 1e-5f);
    EXPECT_TRUE(fabsf(ego->accel_long) > 0.1f);
    EXPECT_NEAR(
        ego->accel_lat, log_position_speed(ego, step, env.dt) * compute_log_yaw_rate(ego, step, env.dt), 1e-4f);
    EXPECT_TRUE(fabsf(ego->steering_angle) <= STEERING_ANGLE_LIMIT_RAD);
    free_allocated(&env);
    return 0;
}

static int test_expert_tracking_teleport_bounds_per_step_error(void) {
    srand(5);
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    env.sdc_controller = CONTROLLER_EXPERT_TRACKING;
    env.expert_tracking_teleport = 1;
    env.dynamics_model = DYNAMICS_MODEL_JERK;
    env.dt = 0.1f;
    env.scenario_length = 120;
    allocate(&env);
    env.timestep = -1;
    c_reset(&env);
    Agent *ego = &env.agents[env.active_agent_indices[0]];

    float worst_step_error = 0.0f;
    for (int step = 1; step < env.scenario_length; step++) {
        c_step(&env);
        float step_error = compute_displacement_error(ego, env.timestep);
        worst_step_error = step_error > worst_step_error ? step_error : worst_step_error;
    }
    EXPECT_TRUE(worst_step_error < 0.3f);
    free_allocated(&env);
    return 0;
}

static int test_jerk_rear_axle_slip_offsets_box_center(void) {
    srand(5);
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    env.dynamics_model = DYNAMICS_MODEL_JERK;
    env.dt = 0.1f;
    allocate(&env);
    env.timestep = -1;
    c_reset(&env);
    int ego_idx = env.active_agent_indices[0];
    Agent *ego = &env.agents[ego_idx];
    ((int *) env.actions)[0] = 7; // zero jerk: hold the current steering and speed
    ego->sim_vx = 5.0f * ego->cos_heading;
    ego->sim_vy = 5.0f * ego->sin_heading;
    ego->steering_angle = 0.3f;
    ego->accel_lat = 25.0f * tanf(0.3f) / ego->wheelbase;
    update_agent_speed(ego);
    Agent start = *ego;

    env.jerk_rear_axle_slip = 0;
    move_dynamics(&env, 0, ego_idx);
    Agent centre_result = *ego;
    *ego = start;
    env.jerk_rear_axle_slip = 1;
    move_dynamics(&env, 0, ego_idx);
    Agent slip_result = *ego;

    EXPECT_NEAR(slip_result.sim_heading, centre_result.sim_heading, 1e-6f);
    EXPECT_TRUE(fabsf(normalize_heading(slip_result.sim_heading - start.sim_heading)) > 1e-3f);
    float rear_axle_m = REAR_AXLE_RATIO * start.wheelbase;
    float expected_dx = rear_axle_m * (slip_result.cos_heading - start.cos_heading);
    float expected_dy = rear_axle_m * (slip_result.sin_heading - start.sin_heading);
    EXPECT_NEAR(slip_result.sim_x - centre_result.sim_x, expected_dx, 1e-4f);
    EXPECT_NEAR(slip_result.sim_y - centre_result.sim_y, expected_dy, 1e-4f);
    EXPECT_TRUE(fabsf(expected_dx) + fabsf(expected_dy) > 1e-3f);
    free_allocated(&env);
    return 0;
}

static int test_seeding_flags_log_glitches(void) {
    Drive env = {0};
    env.dt = 0.1f;
    Agent agent = {0};
    enum { STEPS = 12 };
    float log_x[STEPS];
    float log_y[STEPS];
    float log_heading[STEPS];
    float log_vx[STEPS];
    float log_vy[STEPS];
    int log_valid[STEPS];
    for (int t = 0; t < STEPS; t++) {
        log_x[t] = 10.0f * t * env.dt; // steady 10 m/s along +x
        log_y[t] = 0.0f;
        log_heading[t] = 0.0f;
        log_vx[t] = 10.0f;
        log_vy[t] = 0.0f;
        log_valid[t] = 1;
    }
    log_x[6] += 0.3f; // one-sample position glitch
    agent.log_trajectory_x = log_x;
    agent.log_trajectory_y = log_y;
    agent.log_heading = log_heading;
    agent.log_velocity_x = log_vx;
    agent.log_velocity_y = log_vy;
    agent.log_valid = log_valid;
    agent.trajectory_size = STEPS;
    agent.wheelbase = 3.0f;

    seed_dynamics_state_from_log(&env, &agent, 2);
    EXPECT_NEAR(agent.accel_long, 0.0f, 1e-4f);
    EXPECT_EQ_INT(log_state_within_envelope(&env, &agent, 2), 1);
    // The five-point stencil around the glitched sample 6 corrupts steps 4, 6 and 8
    EXPECT_EQ_INT(log_state_within_envelope(&env, &agent, 4), 0);
    EXPECT_EQ_INT(log_state_within_envelope(&env, &agent, 5), 1);
    EXPECT_EQ_INT(log_state_within_envelope(&env, &agent, 6), 0);
    EXPECT_EQ_INT(log_state_within_envelope(&env, &agent, 8), 0);
    EXPECT_EQ_INT(log_state_within_envelope(&env, &agent, 10), 1);
    seed_dynamics_state_from_log(&env, &agent, 6);
    EXPECT_NEAR(agent.accel_long, ACCEL_LONG_LIMIT[0], 1e-4f);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_seeding_flags_log_glitches);
    RUN_TEST(test_jerk_rear_axle_slip_offsets_box_center);
    RUN_TEST(test_reset_seeds_dynamics_state_from_log);
    RUN_TEST(test_expert_tracking_teleport_bounds_per_step_error);
    RUN_TEST(test_expert_similarity_reward_is_quadratic);
    RUN_TEST(test_expert_similarity_only_drops_rl_terms);
    RUN_TEST(test_expert_similarity_kernel_is_positive_and_bounded);
    RUN_TEST(test_expert_tracking_follows_log);
    RUN_TEST(test_episode_max_steps_truncates);
    RUN_TEST(test_init_step_jitter_resamples_within_range);
    RUN_TEST(test_static_expert_is_flagged_and_masked);
    return test_summary(failures);
}
