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
    agent.metrics_array[STOP_SIGN_IDX] = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(env.rewards[0], -5.0f, 1e-5f);
    EXPECT_NEAR(log.reward_stop_sign, -5.0f, 1e-5f);
    EXPECT_NEAR(log.stop_sign_violation_rate, 1.0f, 1e-5f);

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

// speed bonus: weight x dt x forward lane alignment x speed's share of [from, base_max_speed_mps], not on turn legs
static int test_reward_speed_bonus(void) {
    Drive env;
    Agent agent;
    Log log;
    int active[1];
    float reward[1] = {0};
    const float cases[][3]
        = {{10.0f, 1.0f, 0.5f},
           {30.0f, 1.0f, 1.0f},
           {-5.0f, 1.0f, 0.0f},
           {10.0f, -1.0f, 0.0f},
           {10.0f, 0.5f, 0.25f},
           {0.0f, 1.0f, 0.0f}};
    for (int case_idx = 0; case_idx < (int) (sizeof(cases) / sizeof(cases[0])); case_idx++) {
        init_reward_env(&env, &agent, &log, active, reward);
        reward[0] = 0.0f;
        env.base_max_speed_mps = 20.0f;
        env.reward_speed_bonus = 4e-3f;
        agent.sim_speed_signed = cases[case_idx][0];
        agent.sim_speed = fabsf(cases[case_idx][0]);
        agent.metrics_array[LANE_ANGLE_IDX] = cases[case_idx][1];
        compute_rewards(&env, 0);
        EXPECT_NEAR(log.reward_speed_bonus, 4e-3f * env.dt * cases[case_idx][2], 1e-9f);
    }
    // counted from 5 m/s: nothing up to 5, then the share of the 15 m/s above it
    const float from_cases[][3]
        = {{3.0f, 1.0f, 0.0f}, {5.0f, 1.0f, 0.0f}, {12.5f, 1.0f, 0.5f}, {12.5f, 0.5f, 0.25f}, {25.0f, 1.0f, 1.0f}};
    for (int case_idx = 0; case_idx < (int) (sizeof(from_cases) / sizeof(from_cases[0])); case_idx++) {
        init_reward_env(&env, &agent, &log, active, reward);
        reward[0] = 0.0f;
        env.base_max_speed_mps = 20.0f;
        env.reward_speed_bonus = 4e-3f;
        env.reward_speed_bonus_from_mps = 5.0f;
        agent.sim_speed_signed = from_cases[case_idx][0];
        agent.sim_speed = from_cases[case_idx][0];
        agent.metrics_array[LANE_ANGLE_IDX] = from_cases[case_idx][1];
        compute_rewards(&env, 0);
        EXPECT_NEAR(log.reward_speed_bonus, 4e-3f * env.dt * from_cases[case_idx][2], 1e-9f);
    }

    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    env.base_max_speed_mps = 20.0f;
    agent.sim_speed_signed = 10.0f;
    agent.sim_speed = 10.0f;
    compute_rewards(&env, 0);
    float reward_without = reward[0];
    EXPECT_NEAR(log.reward_speed_bonus, 0.0f, 0.0f);
    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    env.base_max_speed_mps = 20.0f;
    env.reward_speed_bonus = 4e-3f;
    agent.sim_speed_signed = 10.0f;
    agent.sim_speed = 10.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(reward[0] - reward_without, 4e-3f * env.dt * 0.5f, 1e-9f);

    struct LatticeAgent *lattice_agent = (struct LatticeAgent *) calloc(1, sizeof(struct LatticeAgent));
    for (int turn_active = 0; turn_active < 2; turn_active++) {
        init_reward_env(&env, &agent, &log, active, reward);
        reward[0] = 0.0f;
        env.dynamics_model = DYNAMICS_MODEL_SPLINE_WERLING;
        env.lattice_agents = lattice_agent;
        env.reward_wait_full_speed_mps = 5.0f;
        env.base_max_speed_mps = 20.0f;
        env.reward_speed_bonus = 4e-3f;
        lattice_agent->turn.active = turn_active;
        agent.sim_speed_signed = 2.0f;
        agent.sim_speed = 2.0f;
        compute_rewards(&env, 0);
        EXPECT_NEAR(log.reward_speed_bonus, turn_active ? 0.0f : 4e-3f * env.dt * 0.1f, 1e-9f);
    }
    free(lattice_agent);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_observation_size_formula);
    RUN_TEST(test_observation_zero_fill_and_valid_counts);
    RUN_TEST(test_reward_terminal_components);
    RUN_TEST(test_reward_goal_speed_gating);
    RUN_TEST(test_reward_lane_align_wrong_way);
    RUN_TEST(test_reward_speed_bonus);
    return test_summary(failures);
}
