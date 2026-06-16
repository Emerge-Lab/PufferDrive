#include "include/drive_fixture.h"
#include "include/test.h"

static int test_observation_size_formula(void) {
    Drive env = {0};
    env.target_type = TARGET_STATIC;
    env.num_target_waypoints = 3;
    env.reward_conditioning = 0;
    env.obs_slots_partners_n = 2;
    env.obs_slots_lane_kept = 5;
    env.obs_slots_boundary_kept = 7;
    env.obs_slots_traffic_controls_n = 4;
    int expected = EGO_FEATURES + 3 * STATIC_TARGET_FEATURES + 2 * PARTNER_FEATURES + 12 * ROAD_FEATURES
        + 4 * TRAFFIC_CONTROL_FEATURES + OBS_VALID_COUNT_FEATURES;
    EXPECT_EQ_INT(compute_observation_size(&env), expected);

    env.target_type = TARGET_DYNAMIC;
    env.reward_conditioning = 1;
    expected = EGO_FEATURES + NUM_REWARD_COEFS + 3 * DYNAMIC_TARGET_FEATURES + 2 * PARTNER_FEATURES + 12 * ROAD_FEATURES
        + 4 * TRAFFIC_CONTROL_FEATURES + OBS_VALID_COUNT_FEATURES;
    EXPECT_EQ_INT(compute_observation_size(&env), expected);
    return 0;
}

static int test_observation_zero_fill_and_valid_counts(void) {
    srand(3);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_GIGAFLOW, 1, 0);
    env.obs_slots_partners_n = 4;
    env.obs_slots_lane_n = 8;
    env.obs_slots_boundary_n = 8;
    env.obs_slots_lane_kept = 4;
    env.obs_slots_boundary_kept = 4;
    env.obs_slots_traffic_controls_n = 2;
    allocate(&env);
    c_reset(&env);

    EXPECT_EQ_INT(env.active_agent_count, 1);
    int obs_size = compute_observation_size(&env);
    float *obs = env.observations;
    int partner_base = EGO_FEATURES + env.num_target_waypoints * STATIC_TARGET_FEATURES;
    int road_base = partner_base + env.obs_slots_partners_n * PARTNER_FEATURES;
    int traffic_base = road_base + (env.obs_slots_lane_kept + env.obs_slots_boundary_kept) * ROAD_FEATURES;
    int valid_base = obs_size - OBS_VALID_COUNT_FEATURES;

    EXPECT_NEAR(obs[valid_base + 2], 0.0f, 1e-5f);
    EXPECT_TRUE(obs[valid_base] >= 0.0f && obs[valid_base] <= (float) env.obs_slots_lane_kept);
    EXPECT_TRUE(obs[valid_base + 1] >= 0.0f && obs[valid_base + 1] <= (float) env.obs_slots_boundary_kept);
    EXPECT_TRUE(obs[valid_base + 3] >= 0.0f && obs[valid_base + 3] <= (float) env.obs_slots_traffic_controls_n);

    for (int i = 0; i < env.obs_slots_partners_n * PARTNER_FEATURES; i++) {
        EXPECT_NEAR(obs[partner_base + i], 0.0f, 1e-6f);
    }
    for (int i = traffic_base + (int) obs[valid_base + 3] * TRAFFIC_CONTROL_FEATURES; i < valid_base; i++) {
        EXPECT_NEAR(obs[i], 0.0f, 1e-6f);
    }

    free_allocated(&env);
    return 0;
}

static int test_metric_terminal_flags_are_exclusive(void) {
    srand(5);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_GIGAFLOW, 2, 0);
    int agent_idx = env.active_agent_indices[0];
    Agent *agent = &env.agents[agent_idx];
    agent->sim_x = env.grid_map->top_left_x - 1000.0f;
    agent->sim_y = env.grid_map->top_left_y + 1000.0f;
    copy_pose_to_prev(agent);

    compute_metrics(&env, agent_idx, 0);

    EXPECT_NEAR(agent->metrics_array[OFFROAD_IDX], 1.0f, 1e-5f);
    EXPECT_NEAR(agent->metrics_array[COLLISION_IDX], 0.0f, 1e-5f);
    EXPECT_NEAR(agent->metrics_array[RED_LIGHT_IDX], 0.0f, 1e-5f);

    free_allocated(&env);
    return 0;
}

static void init_reward_env(Drive *env, Agent *agent, Log *log, int *active, float *reward) {
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
    env->reward_goal = 2.0f;
    env->simulation_mode = SIMULATION_GIGAFLOW;
    env->num_target_waypoints = 3;
    env->compute_eval_metrics = 0;
    agent->reward_coefs[REWARD_COEF_COLLISION] = 3.0f;
    agent->reward_coefs[REWARD_COEF_OFFROAD] = 4.0f;
    agent->reward_coefs[REWARD_COEF_STOP_LINE] = 5.0f;
    agent->metrics_array[LANE_ANGLE_IDX] = 1.0f;
}

static int test_reward_terminal_components(void) {
    Drive env;
    Agent agent;
    Log log;
    int active[1];
    float reward[1] = {0};

    init_reward_env(&env, &agent, &log, active, reward);
    agent.sim_speed = 10.0f;
    agent.metrics_array[COLLISION_IDX] = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(env.rewards[0], -4.0f, 1e-5f);
    EXPECT_NEAR(log.reward_collision, -4.0f, 1e-5f);

    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.metrics_array[OFFROAD_IDX] = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(env.rewards[0], -4.0f, 1e-5f);
    EXPECT_NEAR(log.reward_offroad, -4.0f, 1e-5f);

    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.metrics_array[RED_LIGHT_IDX] = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(env.rewards[0], -5.0f, 1e-5f);
    EXPECT_NEAR(log.reward_red_light, -5.0f, 1e-5f);

    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.metrics_array[REACHED_GOAL_IDX] = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(env.rewards[0], 2.0f, 1e-5f);
    EXPECT_NEAR(log.reward_goal, 2.0f, 1e-5f);
    return 0;
}

static int test_classic_and_jerk_action_clipping(void) {
    srand(19);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_GIGAFLOW, 1, 0);
    env.action_type = 1;
    env.dynamics_model = CLASSIC;
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    agent->sim_speed_signed = 0.0f;
    ((float (*)[2]) env.actions)[0][0] = 5.0f;
    ((float (*)[2]) env.actions)[0][1] = 5.0f;
    move_dynamics(&env, 0, env.active_agent_indices[0]);
    EXPECT_TRUE(agent->sim_speed_signed <= MAX_SPEED);
    EXPECT_TRUE(agent->steering_angle <= STEERING_LIMIT);
    free_allocated(&env);

    srand(23);
    env = drive_test_make_env(drive_carla_map(), SIMULATION_GIGAFLOW, 1, 0);
    env.action_type = 1;
    env.dynamics_model = JERK;
    agent = &env.agents[env.active_agent_indices[0]];
    agent->reward_coefs[REWARD_COEF_THROTTLE] = 1.0f;
    agent->reward_coefs[REWARD_COEF_STEER] = 1.0f;
    agent->reward_coefs[REWARD_COEF_ACC] = 1.0f;
    ((float (*)[2]) env.actions)[0][0] = -5.0f;
    ((float (*)[2]) env.actions)[0][1] = 5.0f;
    move_dynamics(&env, 0, env.active_agent_indices[0]);
    EXPECT_TRUE(agent->accel_long >= ACCEL_LONG_LIMIT[0]);
    EXPECT_TRUE(agent->accel_lat <= ACCEL_LAT_LIMIT[1]);
    EXPECT_TRUE(agent->steering_angle <= 0.55f);
    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_observation_size_formula);
    RUN_TEST(test_observation_zero_fill_and_valid_counts);
    RUN_TEST(test_metric_terminal_flags_are_exclusive);
    RUN_TEST(test_reward_terminal_components);
    RUN_TEST(test_classic_and_jerk_action_clipping);
    return test_summary(failures);
}
