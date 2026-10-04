#include "include/drive_fixture.h"
#include "include/test.h"

#define OVERRIDE_TEST_TOLERANCE 1e-5f
#define OVERRIDE_TEST_STEPS 5

static float expected_linear_obs(int coef_idx, float raw_value) {
    float lo = REWARD_BOUNDS[coef_idx].min_val;
    float hi = REWARD_BOUNDS[coef_idx].max_val;
    float unit_fraction = fmaxf(0.0f, fminf(1.0f, (raw_value - lo) / (hi - lo + REWARD_COEF_NORMALIZATION_EPSILON)));
    return 2.0f * unit_fraction - 1.0f;
}

static float conditioning_obs(Drive *env, int row, int coef_idx) {
    return env->observations[row * compute_observation_size(env) + EGO_FEATURES + coef_idx];
}

static void set_override(Drive *env, int entry_idx, int target_agent_idx, int coef_idx, float value) {
    env->reward_coef_overrides[entry_idx] = (RewardCoefOverride) {target_agent_idx, coef_idx, value};
}

static void step_neutral(Drive *env, int steps) {
    for (int t = 0; t < steps; t++) {
        drive_set_neutral_actions(env);
        c_step(env);
    }
}

static int test_reward_coef_normalization_round_trip(void) {
    for (int coef_idx = 0; coef_idx < NUM_REWARD_COEFS; coef_idx++) {
        float lo = REWARD_BOUNDS[coef_idx].min_val;
        float hi = REWARD_BOUNDS[coef_idx].max_val;
        float normalized_values[] = {-1.0f, -0.5f, 0.0f, 0.3f, 1.0f};
        for (int i = 0; i < 5; i++) {
            float raw_value = denormalize_reward_coef(coef_idx, normalized_values[i]);
            EXPECT_NEAR(normalize_reward_coef(coef_idx, raw_value), normalized_values[i], OVERRIDE_TEST_TOLERANCE);
        }
        if (hi <= lo) {
            continue;
        }
        float raw_values[] = {lo, lo + 0.25f * (hi - lo), 0.5f * (lo + hi), hi};
        for (int i = 0; i < 4; i++) {
            float round_trip = denormalize_reward_coef(coef_idx, normalize_reward_coef(coef_idx, raw_values[i]));
            EXPECT_NEAR(round_trip, raw_values[i], OVERRIDE_TEST_TOLERANCE * (hi - lo));
            EXPECT_NEAR(
                normalize_reward_coef(coef_idx, raw_values[i]),
                expected_linear_obs(coef_idx, raw_values[i]),
                OVERRIDE_TEST_TOLERANCE);
        }
    }
    EXPECT_EQ_INT(strcmp(REWARD_COEF_NAMES[REWARD_COEF_COLLISION], "collision"), 0);
    EXPECT_EQ_INT(strcmp(REWARD_COEF_NAMES[REWARD_COEF_ACC], "acc"), 0);
    return 0;
}

// Per row: SDC gets its own collision coef, `special_agent_idx` its own offroad coef, the rest the OTHERS values.
static int check_replay_overrides(Drive *env, int special_agent_idx, float others_collision) {
    for (int row = 0; row < env->active_agent_count; row++) {
        int agent_idx = env->active_agent_indices[row];
        float *coefs = env->agents[agent_idx].reward_coefs;
        float expected_collision = agent_idx == EGO_IDX ? 5.0f : others_collision;
        EXPECT_NEAR(coefs[REWARD_COEF_COLLISION], expected_collision, 0.0f);
        EXPECT_NEAR(
            conditioning_obs(env, row, REWARD_COEF_COLLISION),
            expected_linear_obs(REWARD_COEF_COLLISION, expected_collision),
            OVERRIDE_TEST_TOLERANCE);
        if (agent_idx == EGO_IDX) {
            EXPECT_TRUE(coefs[REWARD_COEF_OFFROAD] != 1.0f && coefs[REWARD_COEF_OFFROAD] != 2.5f);
            continue;
        }
        float expected_offroad = agent_idx == special_agent_idx ? 2.5f : 1.0f;
        EXPECT_NEAR(coefs[REWARD_COEF_OFFROAD], expected_offroad, 0.0f);
        EXPECT_NEAR(
            conditioning_obs(env, row, REWARD_COEF_OFFROAD),
            expected_linear_obs(REWARD_COEF_OFFROAD, expected_offroad),
            OVERRIDE_TEST_TOLERANCE);
    }
    return 0;
}

static int test_reward_coef_overrides_replay_targets_and_reset(void) {
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 16, 0);
    env.control_mode = CONTROL_MODE_VEHICLES;
    env.goal_source = GOAL_SOURCE_GT;
    env.reward_conditioning = 1;
    env.reward_randomization = 1;
    allocate(&env);
    EXPECT_TRUE(env.active_agent_count >= 3);
    EXPECT_EQ_INT(env.active_agent_indices[0], EGO_IDX);
    int special_agent_idx = env.active_agent_indices[1];

    set_override(&env, 0, REWARD_COEF_TARGET_OTHERS, REWARD_COEF_COLLISION, -2.0f);
    set_override(&env, 1, REWARD_COEF_TARGET_OTHERS, REWARD_COEF_OFFROAD, 1.0f);
    set_override(&env, 2, special_agent_idx, REWARD_COEF_OFFROAD, 2.5f);
    set_override(&env, 3, EGO_IDX, REWARD_COEF_COLLISION, 5.0f);
    env.reward_coef_override_count = 4;
    env.timestep = -1;
    c_reset(&env);
    if (check_replay_overrides(&env, special_agent_idx, -2.0f)) {
        return 1;
    }

    float lane_align_before = env.agents[special_agent_idx].reward_coefs[REWARD_COEF_LANE_ALIGN];
    step_neutral(&env, OVERRIDE_TEST_STEPS);
    c_reset(&env);
    if (check_replay_overrides(&env, special_agent_idx, -2.0f)) {
        return 1;
    }
    // Coefs without an override keep being re-randomized at every reset.
    EXPECT_TRUE(env.agents[special_agent_idx].reward_coefs[REWARD_COEF_LANE_ALIGN] != lane_align_before);

    // A replaced table only lands at the next reset, never mid-episode.
    set_override(&env, 0, REWARD_COEF_TARGET_OTHERS, REWARD_COEF_COLLISION, 0.5f);
    step_neutral(&env, OVERRIDE_TEST_STEPS);
    EXPECT_NEAR(env.agents[special_agent_idx].reward_coefs[REWARD_COEF_COLLISION], -2.0f, 0.0f);
    c_reset(&env);
    if (check_replay_overrides(&env, special_agent_idx, 0.5f)) {
        return 1;
    }

    free_allocated(&env);
    return 0;
}

static int test_reward_coef_overrides_survive_gigaflow_respawn(void) {
    srand(5);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 8, 0);
    env.reward_conditioning = 1;
    env.reward_randomization = 1;
    set_override(&env, 0, REWARD_COEF_TARGET_OTHERS, REWARD_COEF_COLLISION, -1.0f);
    set_override(&env, 1, EGO_IDX, REWARD_COEF_COLLISION, 4.0f);
    env.reward_coef_override_count = 2;
    allocate(&env);
    c_reset(&env);
    EXPECT_TRUE(env.active_agent_count >= 2);

    for (int episode = 0; episode < 3; episode++) {
        float spawn_x_before = env.agents[env.active_agent_indices[1]].sim_x;
        step_neutral(&env, OVERRIDE_TEST_STEPS);
        c_reset(&env);
        EXPECT_TRUE(env.agents[env.active_agent_indices[1]].sim_x != spawn_x_before);
        for (int row = 0; row < env.active_agent_count; row++) {
            int agent_idx = env.active_agent_indices[row];
            if (env.agents[agent_idx].removed) {
                continue;
            }
            float expected_collision = agent_idx == EGO_IDX ? 4.0f : -1.0f;
            EXPECT_NEAR(env.agents[agent_idx].reward_coefs[REWARD_COEF_COLLISION], expected_collision, 0.0f);
            EXPECT_NEAR(
                conditioning_obs(&env, row, REWARD_COEF_COLLISION),
                expected_linear_obs(REWARD_COEF_COLLISION, expected_collision),
                OVERRIDE_TEST_TOLERANCE);
        }
    }

    free_allocated(&env);
    return 0;
}

static int test_reward_coef_override_changes_collision_reward(void) {
    Drive env = {0};
    Agent agents[2] = {drive_test_agent(0.0f, 0.0f, 0.0f), drive_test_agent(10.0f, 0.0f, 0.0f)};
    Log log = {0};
    int active[1] = {1};
    float reward[1] = {0.0f};
    env.agents = agents;
    env.active_agent_indices = active;
    env.active_agent_count = 1;
    env.logs = &log;
    env.rewards = reward;
    env.dt = 0.1f;
    env.num_goals = 3;
    env.simulation_mode = SIMULATION_MODE_GIGAFLOW;
    env.reward_conditioning = 1;
    env.reward_collision = 3.0f;
    set_override(&env, 0, REWARD_COEF_TARGET_OTHERS, REWARD_COEF_COLLISION, -2.0f);
    env.reward_coef_override_count = 1;

    generate_reward_coefs(&env, EGO_IDX);
    generate_reward_coefs(&env, 1);
    EXPECT_NEAR(agents[EGO_IDX].reward_coefs[REWARD_COEF_COLLISION], 3.0f, 0.0f);
    EXPECT_NEAR(agents[1].reward_coefs[REWARD_COEF_COLLISION], -2.0f, 0.0f);

    agents[1].goal_count = 3;
    agents[1].metrics_array[COLLISION_IDX] = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(reward[0], 2.0f, OVERRIDE_TEST_TOLERANCE);
    EXPECT_NEAR(log.reward_collision, 2.0f, OVERRIDE_TEST_TOLERANCE);
    return 0;
}

static int test_learner_rows_follow_mode(void) {
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 16, 0);
    env.control_mode = CONTROL_MODE_VEHICLES;
    env.goal_source = GOAL_SOURCE_GT;
    env.learner_agents = LEARNER_AGENTS_SDC;
    allocate(&env);
    EXPECT_TRUE(env.active_agent_count >= 2);
    for (int row = 0; row < env.active_agent_count; row++) {
        EXPECT_EQ_INT(env.learner_masks[row], env.active_agent_indices[row] == EGO_IDX);
    }
    env.learner_agents = LEARNER_AGENTS_ALL;
    assign_learner_rows(&env);
    for (int row = 0; row < env.active_agent_count; row++) {
        EXPECT_EQ_INT(env.learner_masks[row], 1);
    }
    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_reward_coef_normalization_round_trip);
    RUN_TEST(test_reward_coef_overrides_replay_targets_and_reset);
    RUN_TEST(test_reward_coef_overrides_survive_gigaflow_respawn);
    RUN_TEST(test_reward_coef_override_changes_collision_reward);
    RUN_TEST(test_learner_rows_follow_mode);
    return test_summary(failures);
}
