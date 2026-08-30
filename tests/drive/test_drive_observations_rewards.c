#include "include/drive_fixture.h"
#include "include/test.h"

static int test_observation_size_formula(void) {
    Drive env = {0};
    env.num_goals = 3;
    env.reward_conditioning = 0;
    env.obs_slots_partners_n = 2;
    env.obs_slots_lane_kept = 5;
    env.obs_slots_boundary_kept = 7;
    env.obs_slots_traffic_controls_n = 4;
    int expected = EGO_FEATURES + 3 * GOAL_FEATURES + 2 * PARTNER_FEATURES + 5 * LANE_FEATURES + 7 * BOUNDARY_FEATURES
        + 4 * TRAFFIC_CONTROL_FEATURES + OBS_VALID_COUNT_FEATURES;
    EXPECT_EQ_INT(compute_observation_size(&env), expected);

    env.reward_conditioning = 1;
    expected = EGO_FEATURES + NUM_REWARD_COEFS + 3 * GOAL_FEATURES + 2 * PARTNER_FEATURES + 5 * LANE_FEATURES
        + 7 * BOUNDARY_FEATURES + 4 * TRAFFIC_CONTROL_FEATURES + OBS_VALID_COUNT_FEATURES;
    EXPECT_EQ_INT(compute_observation_size(&env), expected);

    env.obs_partner_relative_velocity = 1;
    expected += 2 * PARTNER_RELATIVE_VELOCITY_FEATURES;
    EXPECT_EQ_INT(compute_observation_size(&env), expected);
    return 0;
}

static int test_observation_zero_fill_and_valid_counts(void) {
    srand(3);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
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
    int partner_base = EGO_FEATURES + env.num_goals * GOAL_FEATURES;
    int road_base = partner_base + env.obs_slots_partners_n * PARTNER_FEATURES;
    int traffic_base
        = road_base + env.obs_slots_lane_kept * LANE_FEATURES + env.obs_slots_boundary_kept * BOUNDARY_FEATURES;
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
    env->simulation_mode = SIMULATION_MODE_GIGAFLOW;
    env->num_goals = 3;
    agent->goal_count = 3;
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

static int test_reward_goal_speed_gating(void) {
    Drive env;
    Agent agent;
    Log log;
    int active[1];
    float reward[1] = {0};

    // GIGAFLOW: final waypoint reached above goal-speed → goal reward gated to 0
    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.metrics_array[REACHED_GOAL_IDX] = 1.0f;
    agent.current_goal_idx = agent.goal_count;
    agent.reward_coefs[REWARD_COEF_GOAL_SPEED] = 3.0f;
    agent.sim_speed = 10.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(log.reward_goal, 0.0f, 1e-5f);

    // Same final waypoint but below goal-speed → full goal reward
    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.metrics_array[REACHED_GOAL_IDX] = 1.0f;
    agent.current_goal_idx = agent.goal_count;
    agent.reward_coefs[REWARD_COEF_GOAL_SPEED] = 3.0f;
    agent.sim_speed = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(log.reward_goal, 2.0f, 1e-5f);
    return 0;
}

static int test_reward_coef_goal_speed_pinned(void) {
    Drive env = {0};
    Agent agent = {0};
    env.reward_randomization = 1;
    env.goal_speed = 3.0f;
    rng_seed(&env.rng_state, 7);

    env.goal_speed_randomization = 0;
    for (int sample_idx = 0; sample_idx < 16; sample_idx++) {
        generate_reward_coefs(&env, &agent);
        EXPECT_NEAR(agent.reward_coefs[REWARD_COEF_GOAL_SPEED], 3.0f, 1e-6f);
    }

    env.goal_speed_randomization = 1;
    int differs_from_pinned = 0;
    for (int sample_idx = 0; sample_idx < 16; sample_idx++) {
        generate_reward_coefs(&env, &agent);
        float goal_speed = agent.reward_coefs[REWARD_COEF_GOAL_SPEED];
        EXPECT_TRUE(goal_speed >= 0.0f && goal_speed <= 20.0f);
        differs_from_pinned += fabsf(goal_speed - 3.0f) > 1e-3f;
    }
    EXPECT_TRUE(differs_from_pinned > 0);
    return 0;
}

static int test_partner_obs_relative_velocity(void) {
    // Ego heading +x at 10 m/s; partner 20 m ahead heading +y at 5 m/s -> ego-frame relative velocity (-10, 5).
    Drive env = {0};
    env.obs_norm_speed_mps = 10.0f;
    env.obs_norm_xy_offset_m = 100.0f;
    env.obs_norm_z_m = 10.0f;
    env.obs_norm_veh_length_m = 10.0f;
    env.obs_norm_veh_width_m = 5.0f;
    env.obs_range_partner_m = 100.0f;
    env.obs_slots_partners_n = 2;
    env.num_agents = 2;
    env.active_agent_count = 2;
    int active[2] = {0, 1};
    env.active_agent_indices = active;
    Agent agents[2];
    agents[0] = drive_test_agent(0.0f, 0.0f, 0.0f);
    agents[0].sim_vx = 10.0f;
    agents[1] = drive_test_agent(20.0f, 0.0f, (float) M_PI / 2.0f);
    agents[1].sim_vy = 5.0f;
    update_agent_speed(&agents[0]);
    update_agent_speed(&agents[1]);
    env.agents = agents;
    float obs[2 * (PARTNER_FEATURES + PARTNER_RELATIVE_VELOCITY_FEATURES)];
    int partner_count = 0;

    env.obs_partner_relative_velocity = 0;
    memset(obs, 0, sizeof(obs));
    int end_idx = write_partner_obs(&env, &agents[0], 0, obs, 0, &partner_count);
    EXPECT_EQ_INT(partner_count, 1);
    EXPECT_EQ_INT(end_idx, 2 * PARTNER_FEATURES);
    EXPECT_NEAR(obs[7], 0.5f, 1e-5f);

    env.obs_partner_relative_velocity = 1;
    memset(obs, 0, sizeof(obs));
    end_idx = write_partner_obs(&env, &agents[0], 0, obs, 0, &partner_count);
    EXPECT_EQ_INT(end_idx, 2 * (PARTNER_FEATURES + PARTNER_RELATIVE_VELOCITY_FEATURES));
    EXPECT_NEAR(obs[7], 0.5f, 1e-5f);
    EXPECT_NEAR(obs[PARTNER_FEATURES], -1.0f, 1e-5f);
    EXPECT_NEAR(obs[PARTNER_FEATURES + 1], 0.5f, 1e-5f);
    return 0;
}

static int test_reward_lane_align_wrong_way(void) {
    Drive env;
    Agent agent;
    Log log;
    int active[1];
    float reward[1] = {0};

    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.metrics_array[LANE_ANGLE_IDX] = -1.0f; // cos(θ_f) = -1 → driving against lane
    agent.sim_speed_signed = 5.0f;
    agent.reward_coefs[REWARD_COEF_LANE_ALIGN] = 1.0f;
    agent.reward_coefs[REWARD_COEF_VEL_ALIGN] = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_TRUE(log.reward_lane_align < 0.0f);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_observation_size_formula);
    RUN_TEST(test_observation_zero_fill_and_valid_counts);
    RUN_TEST(test_reward_terminal_components);
    RUN_TEST(test_reward_goal_speed_gating);
    RUN_TEST(test_reward_coef_goal_speed_pinned);
    RUN_TEST(test_partner_obs_relative_velocity);
    RUN_TEST(test_reward_lane_align_wrong_way);
    return test_summary(failures);
}
