#include "include/drive_fixture.h"
#include "include/test.h"

#define TUNING_AGENT_COUNT 8
#define TUNING_MAX_SPEED_MPS 20.0f
#define TUNING_CONDITIONING_SPEED_SCALE 2.0f
#define WINDOW_LOW_FRACTION 0.6f
#define WINDOW_HIGH_FRACTION 0.8f
#define OUT_OF_WINDOW_MARGIN 0.01f
#define EXACT_TOLERANCE 0.0f
#define ROUND_TRIP_TOLERANCE 1e-4f
#define NORMALIZED_SPAN 2.0f
#define NARROW_SPEED_MIN 1.9f
#define NARROW_SPEED_MAX 1.95f
#define RESET_EPISODE_COUNT 3
#define MID_EPISODE_TIMESTEP 5

static const int BOUNDS_SAMPLED_COEFS[] = {
    REWARD_COEF_GOAL_RADIUS,
    REWARD_COEF_GOAL_SPEED,
    REWARD_COEF_COLLISION,
    REWARD_COEF_OFFROAD,
    REWARD_COEF_COMFORT,
    REWARD_COEF_LANE_ALIGN,
    REWARD_COEF_LANE_CENTER,
    REWARD_COEF_STOP_LINE,
    REWARD_COEF_CENTER_BIAS,
    REWARD_COEF_VEL_ALIGN,
    REWARD_COEF_OVERSPEED,
    REWARD_COEF_REVERSE,
    REWARD_COEF_SPEED,
};
#define BOUNDS_SAMPLED_COEF_COUNT ((int) (sizeof(BOUNDS_SAMPLED_COEFS) / sizeof(BOUNDS_SAMPLED_COEFS[0])))

static const int FIXED_PROFILE_COEFS[] = {
    REWARD_COEF_VELOCITY,
    REWARD_COEF_TIMESTEP,
    REWARD_COEF_THROTTLE,
    REWARD_COEF_STEER,
    REWARD_COEF_ACC,
};
#define FIXED_PROFILE_COEF_COUNT ((int) (sizeof(FIXED_PROFILE_COEFS) / sizeof(FIXED_PROFILE_COEFS[0])))

typedef struct {
    float min_values[NUM_REWARD_COEFS];
    float max_values[NUM_REWARD_COEFS];
} CoefBounds;

static Drive tuning_env_with_goal_speed(int reward_randomization, int goal_speed_randomization, unsigned int seed) {
    srand(seed);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, TUNING_AGENT_COUNT, 0);
    env.max_speed_mps = TUNING_MAX_SPEED_MPS;
    env.conditioning_accel_scale = 1.0f;
    env.conditioning_speed_scale = TUNING_CONDITIONING_SPEED_SCALE;
    env.reward_randomization = reward_randomization;
    env.reward_conditioning = 1;
    env.goal_speed_randomization = goal_speed_randomization;
    allocate(&env);
    c_reset(&env);
    return env;
}

static Drive tuning_env(int reward_randomization, unsigned int seed) {
    return tuning_env_with_goal_speed(reward_randomization, 1, seed);
}

static CoefBounds trained_limits(const Drive *env) {
    CoefBounds limits;
    for (int c = 0; c < NUM_REWARD_COEFS; c++) {
        reward_coef_limits(env, c, &limits.min_values[c], &limits.max_values[c]);
    }
    return limits;
}

static CoefBounds sub_window(const Drive *env) {
    CoefBounds limits = trained_limits(env);
    CoefBounds window = limits;
    for (int k = 0; k < BOUNDS_SAMPLED_COEF_COUNT; k++) {
        int c = BOUNDS_SAMPLED_COEFS[k];
        float span = limits.max_values[c] - limits.min_values[c];
        window.min_values[c] = limits.min_values[c] + WINDOW_LOW_FRACTION * span;
        window.max_values[c] = limits.min_values[c] + WINDOW_HIGH_FRACTION * span;
    }
    return window;
}

static int bounds_state_matches(const Drive *env, const CoefBounds *expected, int expected_overridden) {
    return env->reward_bounds_overridden == expected_overridden
        && memcmp(env->reward_coef_min, expected->min_values, sizeof(expected->min_values)) == 0
        && memcmp(env->reward_coef_max, expected->max_values, sizeof(expected->max_values)) == 0;
}

static CoefBounds current_bounds(const Drive *env) {
    CoefBounds bounds;
    memcpy(bounds.min_values, env->reward_coef_min, sizeof(bounds.min_values));
    memcpy(bounds.max_values, env->reward_coef_max, sizeof(bounds.max_values));
    return bounds;
}

static int policy_agent_count(const Drive *env) {
    int count = 0;
    for (int i = 0; i < env->active_agent_count; i++) {
        count += env->agents[env->active_agent_indices[i]].controller == CONTROLLER_POLICY;
    }
    return count;
}

static int coefs_inside_bounds(const Agent *agent, const CoefBounds *bounds) {
    for (int i = 0; i < BOUNDS_SAMPLED_COEF_COUNT; i++) {
        int c = BOUNDS_SAMPLED_COEFS[i];
        if (agent->reward_coefs[c] < bounds->min_values[c] || agent->reward_coefs[c] > bounds->max_values[c]) {
            return 0;
        }
    }
    return 1;
}

static int expect_rejected_unchanged(Drive *env, const CoefBounds *candidate, const CoefBounds *before) {
    EXPECT_EQ_INT(c_set_reward_coef_bounds(env, candidate->min_values, candidate->max_values), -1);
    EXPECT_TRUE(bounds_state_matches(env, before, 0));
    return 0;
}

static int test_bounds_default_to_training_window(void) {
    Drive env = tuning_env(1, 301);
    EXPECT_EQ_INT(env.reward_bounds_overridden, 0);
    for (int c = 0; c < NUM_REWARD_COEFS; c++) {
        if (c == REWARD_COEF_SPEED) {
            continue;
        }
        EXPECT_NEAR(env.reward_coef_min[c], REWARD_BOUNDS[c].min_val, EXACT_TOLERANCE);
        EXPECT_NEAR(env.reward_coef_max[c], REWARD_BOUNDS[c].max_val, EXACT_TOLERANCE);
    }
    EXPECT_NEAR(env.reward_coef_min[REWARD_COEF_SPEED], 1.0f / TUNING_CONDITIONING_SPEED_SCALE, EXACT_TOLERANCE);
    EXPECT_NEAR(env.reward_coef_max[REWARD_COEF_SPEED], TUNING_CONDITIONING_SPEED_SCALE, EXACT_TOLERANCE);
    free_allocated(&env);
    return 0;
}

static int test_default_sampler_ignores_the_bounds_table_for_speed(void) {
    Drive env = tuning_env(1, 303);
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    Rng rng_before = env.rng_state;
    generate_reward_coefs(&env, agent);
    float training_speed_coef = agent->reward_coefs[REWARD_COEF_SPEED];

    env.rng_state = rng_before;
    env.reward_coef_min[REWARD_COEF_SPEED] = NARROW_SPEED_MIN;
    env.reward_coef_max[REWARD_COEF_SPEED] = NARROW_SPEED_MAX;
    generate_reward_coefs(&env, agent);
    EXPECT_NEAR(agent->reward_coefs[REWARD_COEF_SPEED], training_speed_coef, EXACT_TOLERANCE);

    env.rng_state = rng_before;
    env.reward_bounds_overridden = 1;
    generate_reward_coefs(&env, agent);
    EXPECT_TRUE(agent->reward_coefs[REWARD_COEF_SPEED] >= NARROW_SPEED_MIN);
    EXPECT_TRUE(agent->reward_coefs[REWARD_COEF_SPEED] <= NARROW_SPEED_MAX);
    free_allocated(&env);
    return 0;
}

static int test_set_bounds_accepts_the_exact_trained_window(void) {
    Drive env = tuning_env(1, 307);
    CoefBounds limits = trained_limits(&env);
    EXPECT_EQ_INT(c_set_reward_coef_bounds(&env, limits.min_values, limits.max_values), 0);
    EXPECT_TRUE(bounds_state_matches(&env, &limits, 0));
    free_allocated(&env);
    return 0;
}

static int test_narrowing_one_coefficient_flags_override(void) {
    Drive env = tuning_env(1, 308);
    CoefBounds limits = trained_limits(&env);
    for (int k = 0; k < BOUNDS_SAMPLED_COEF_COUNT; k++) {
        int c = BOUNDS_SAMPLED_COEFS[k];
        CoefBounds candidate = limits;
        candidate.max_values[c] = sub_window(&env).max_values[c];
        EXPECT_EQ_INT(c_set_reward_coef_bounds(&env, candidate.min_values, candidate.max_values), 0);
        EXPECT_TRUE(bounds_state_matches(&env, &candidate, 1));
        EXPECT_EQ_INT(c_set_reward_coef_bounds(&env, limits.min_values, limits.max_values), 0);
        EXPECT_TRUE(bounds_state_matches(&env, &limits, 0));
    }
    free_allocated(&env);
    return 0;
}

static int test_trained_window_after_a_level_restores_training_sampler(void) {
    Drive reference_env = tuning_env(1, 310);
    Drive cycled_env = tuning_env(1, 310);
    CoefBounds window = sub_window(&cycled_env);
    CoefBounds limits = trained_limits(&cycled_env);
    EXPECT_EQ_INT(c_set_reward_coef_bounds(&cycled_env, window.min_values, window.max_values), 0);
    EXPECT_EQ_INT(c_set_reward_coef_bounds(&cycled_env, limits.min_values, limits.max_values), 0);
    EXPECT_EQ_INT(cycled_env.reward_bounds_overridden, 0);
    c_resample_reward_coefs(&reference_env);
    c_resample_reward_coefs(&cycled_env);
    for (int i = 0; i < reference_env.active_agent_count; i++) {
        const Agent *reference = &reference_env.agents[reference_env.active_agent_indices[i]];
        const Agent *cycled = &cycled_env.agents[cycled_env.active_agent_indices[i]];
        EXPECT_TRUE(memcmp(reference->reward_coefs, cycled->reward_coefs, sizeof(reference->reward_coefs)) == 0);
    }
    free_allocated(&reference_env);
    free_allocated(&cycled_env);
    return 0;
}

static int test_narrowing_a_fixed_profile_coefficient_is_rejected(void) {
    Drive env = tuning_env(1, 312);
    CoefBounds before = current_bounds(&env);
    CoefBounds limits = trained_limits(&env);
    CoefBounds window = sub_window(&env);
    for (int k = 0; k < FIXED_PROFILE_COEF_COUNT; k++) {
        int c = FIXED_PROFILE_COEFS[k];
        float span = limits.max_values[c] - limits.min_values[c];
        CoefBounds candidate = window;
        candidate.min_values[c] = limits.min_values[c] + WINDOW_LOW_FRACTION * span;
        if (expect_rejected_unchanged(&env, &candidate, &before)) {
            printf("  coefficient %d: narrowed min accepted\n", c);
            return 1;
        }
        candidate = window;
        candidate.max_values[c] = limits.min_values[c] + WINDOW_HIGH_FRACTION * span;
        if (expect_rejected_unchanged(&env, &candidate, &before)) {
            printf("  coefficient %d: narrowed max accepted\n", c);
            return 1;
        }
    }
    free_allocated(&env);
    return 0;
}

static int test_goal_speed_window_needs_goal_speed_randomization(void) {
    Drive fixed_env = tuning_env_with_goal_speed(1, 0, 314);
    CoefBounds before = current_bounds(&fixed_env);
    CoefBounds window = sub_window(&fixed_env);
    if (expect_rejected_unchanged(&fixed_env, &window, &before)) {
        return 1;
    }
    CoefBounds limits = trained_limits(&fixed_env);
    window.min_values[REWARD_COEF_GOAL_SPEED] = limits.min_values[REWARD_COEF_GOAL_SPEED];
    window.max_values[REWARD_COEF_GOAL_SPEED] = limits.max_values[REWARD_COEF_GOAL_SPEED];
    EXPECT_EQ_INT(c_set_reward_coef_bounds(&fixed_env, window.min_values, window.max_values), 0);
    free_allocated(&fixed_env);

    Drive randomized_env = tuning_env_with_goal_speed(1, 1, 314);
    window = sub_window(&randomized_env);
    EXPECT_EQ_INT(c_set_reward_coef_bounds(&randomized_env, window.min_values, window.max_values), 0);
    free_allocated(&randomized_env);
    return 0;
}

static int test_set_bounds_without_randomization_is_refused(void) {
    Drive env = tuning_env(0, 316);
    CoefBounds before = current_bounds(&env);
    CoefBounds window = sub_window(&env);
    if (expect_rejected_unchanged(&env, &window, &before)) {
        return 1;
    }
    CoefBounds limits = trained_limits(&env);
    if (expect_rejected_unchanged(&env, &limits, &before)) {
        return 1;
    }
    free_allocated(&env);
    return 0;
}

static int test_speed_window_comes_from_conditioning_scale(void) {
    Drive env = tuning_env(1, 309);
    CoefBounds window = trained_limits(&env);
    EXPECT_TRUE(TUNING_CONDITIONING_SPEED_SCALE > REWARD_BOUNDS[REWARD_COEF_SPEED].max_val);
    window.max_values[REWARD_COEF_SPEED] = TUNING_CONDITIONING_SPEED_SCALE;
    EXPECT_EQ_INT(c_set_reward_coef_bounds(&env, window.min_values, window.max_values), 0);
    window.min_values[REWARD_COEF_SPEED] = 1.0f / TUNING_CONDITIONING_SPEED_SCALE - OUT_OF_WINDOW_MARGIN;
    EXPECT_EQ_INT(c_set_reward_coef_bounds(&env, window.min_values, window.max_values), -1);
    free_allocated(&env);
    return 0;
}

static int test_set_bounds_rejects_invalid_and_leaves_state_unchanged(void) {
    Drive env = tuning_env(1, 311);
    CoefBounds before = current_bounds(&env);
    CoefBounds limits = trained_limits(&env);
    const float invalid_values[] = {NAN, INFINITY, -INFINITY};
    for (int c = 0; c < NUM_REWARD_COEFS; c++) {
        float span = limits.max_values[c] - limits.min_values[c];
        CoefBounds candidate = limits;
        candidate.min_values[c] = limits.min_values[c] - OUT_OF_WINDOW_MARGIN * fmaxf(span, 1.0f);
        if (expect_rejected_unchanged(&env, &candidate, &before)) {
            printf("  coefficient %d: min below the trained window\n", c);
            return 1;
        }
        candidate = limits;
        candidate.max_values[c] = limits.max_values[c] + OUT_OF_WINDOW_MARGIN * fmaxf(span, 1.0f);
        if (expect_rejected_unchanged(&env, &candidate, &before)) {
            printf("  coefficient %d: max above the trained window\n", c);
            return 1;
        }
        candidate = limits;
        candidate.min_values[c] = limits.max_values[c];
        candidate.max_values[c] = limits.min_values[c];
        if (span > 0.0f && expect_rejected_unchanged(&env, &candidate, &before)) {
            printf("  coefficient %d: inverted window\n", c);
            return 1;
        }
        for (int v = 0; v < (int) (sizeof(invalid_values) / sizeof(invalid_values[0])); v++) {
            candidate = limits;
            candidate.min_values[c] = invalid_values[v];
            if (expect_rejected_unchanged(&env, &candidate, &before)) {
                printf("  coefficient %d: non-finite min %f\n", c, invalid_values[v]);
                return 1;
            }
            candidate = limits;
            candidate.max_values[c] = invalid_values[v];
            if (expect_rejected_unchanged(&env, &candidate, &before)) {
                printf("  coefficient %d: non-finite max %f\n", c, invalid_values[v]);
                return 1;
            }
        }
    }
    free_allocated(&env);
    return 0;
}

static int test_rejection_after_a_live_set_keeps_the_live_window(void) {
    Drive env = tuning_env(1, 313);
    CoefBounds window = sub_window(&env);
    EXPECT_EQ_INT(c_set_reward_coef_bounds(&env, window.min_values, window.max_values), 0);
    CoefBounds candidate = window;
    candidate.max_values[NUM_REWARD_COEFS - 1] = NAN;
    EXPECT_EQ_INT(c_set_reward_coef_bounds(&env, candidate.min_values, candidate.max_values), -1);
    EXPECT_TRUE(bounds_state_matches(&env, &window, 1));
    free_allocated(&env);
    return 0;
}

static int test_resample_counts_policy_cars_and_skips_human(void) {
    Drive env = tuning_env(1, 317);
    int human_agent_idx = env.active_agent_indices[0];
    EXPECT_EQ_INT(c_select_human_agent(&env, human_agent_idx), 0);
    float human_coefs_before[NUM_REWARD_COEFS];
    memcpy(human_coefs_before, env.agents[human_agent_idx].reward_coefs, sizeof(human_coefs_before));

    CoefBounds window = sub_window(&env);
    EXPECT_EQ_INT(c_set_reward_coef_bounds(&env, window.min_values, window.max_values), 0);
    int expected_count = policy_agent_count(&env);
    EXPECT_EQ_INT(expected_count, env.active_agent_count - 1);
    EXPECT_EQ_INT(c_resample_reward_coefs(&env), expected_count);
    EXPECT_TRUE(memcmp(human_coefs_before, env.agents[human_agent_idx].reward_coefs, sizeof(human_coefs_before)) == 0);
    EXPECT_FALSE(coefs_inside_bounds(&env.agents[human_agent_idx], &window));
    free_allocated(&env);
    return 0;
}

static int test_resample_skips_non_policy_controllers(void) {
    Drive env = tuning_env(1, 319);
    int idm_agent_idx = env.active_agent_indices[1];
    env.agents[idm_agent_idx].controller = CONTROLLER_IDM;
    float idm_coefs_before[NUM_REWARD_COEFS];
    memcpy(idm_coefs_before, env.agents[idm_agent_idx].reward_coefs, sizeof(idm_coefs_before));
    EXPECT_EQ_INT(c_resample_reward_coefs(&env), env.active_agent_count - 1);
    EXPECT_TRUE(memcmp(idm_coefs_before, env.agents[idm_agent_idx].reward_coefs, sizeof(idm_coefs_before)) == 0);
    free_allocated(&env);
    return 0;
}

static int test_resampled_values_lie_inside_new_bounds(void) {
    Drive env = tuning_env(1, 323);
    CoefBounds window = sub_window(&env);
    int outside_before = 0;
    for (int i = 0; i < env.active_agent_count; i++) {
        outside_before += !coefs_inside_bounds(&env.agents[env.active_agent_indices[i]], &window);
    }
    EXPECT_TRUE(outside_before > 0);

    EXPECT_EQ_INT(c_set_reward_coef_bounds(&env, window.min_values, window.max_values), 0);
    EXPECT_EQ_INT(c_resample_reward_coefs(&env), env.active_agent_count);
    for (int i = 0; i < env.active_agent_count; i++) {
        EXPECT_TRUE(coefs_inside_bounds(&env.agents[env.active_agent_indices[i]], &window));
    }
    // a window, not a pinned value: drivers still differ
    const Agent *first = &env.agents[env.active_agent_indices[0]];
    for (int k = 0; k < BOUNDS_SAMPLED_COEF_COUNT; k++) {
        int c = BOUNDS_SAMPLED_COEFS[k];
        int distinct = 0;
        for (int i = 1; i < env.active_agent_count; i++) {
            distinct |= env.agents[env.active_agent_indices[i]].reward_coefs[c] != first->reward_coefs[c];
        }
        EXPECT_EQ_INT(distinct, 1);
    }
    free_allocated(&env);
    return 0;
}

static int test_resample_without_randomization_is_refused(void) {
    Drive env = tuning_env(0, 327);
    const Agent *agent = &env.agents[env.active_agent_indices[0]];
    float coefs_before[NUM_REWARD_COEFS];
    memcpy(coefs_before, agent->reward_coefs, sizeof(coefs_before));
    EXPECT_EQ_INT(c_resample_reward_coefs(&env), -1);
    EXPECT_TRUE(memcmp(coefs_before, agent->reward_coefs, sizeof(coefs_before)) == 0);
    free_allocated(&env);
    return 0;
}

static int test_live_window_survives_reset(void) {
    Drive env = tuning_env(1, 329);
    CoefBounds window = sub_window(&env);
    EXPECT_EQ_INT(c_set_reward_coef_bounds(&env, window.min_values, window.max_values), 0);
    for (int episode = 0; episode < RESET_EPISODE_COUNT; episode++) {
        env.timestep = MID_EPISODE_TIMESTEP;
        c_reset(&env);
        EXPECT_TRUE(bounds_state_matches(&env, &window, 1));
        for (int i = 0; i < env.active_agent_count; i++) {
            EXPECT_TRUE(coefs_inside_bounds(&env.agents[env.active_agent_indices[i]], &window));
        }
    }
    free_allocated(&env);
    return 0;
}

static int test_resample_is_deterministic(void) {
    Drive first_env = tuning_env(1, 331);
    Drive second_env = tuning_env(1, 331);
    CoefBounds window = sub_window(&first_env);
    EXPECT_EQ_INT(c_set_reward_coef_bounds(&first_env, window.min_values, window.max_values), 0);
    EXPECT_EQ_INT(c_set_reward_coef_bounds(&second_env, window.min_values, window.max_values), 0);
    c_resample_reward_coefs(&first_env);
    c_resample_reward_coefs(&second_env);
    EXPECT_EQ_INT(first_env.active_agent_count, second_env.active_agent_count);
    for (int i = 0; i < first_env.active_agent_count; i++) {
        const Agent *first = &first_env.agents[first_env.active_agent_indices[i]];
        const Agent *second = &second_env.agents[second_env.active_agent_indices[i]];
        EXPECT_TRUE(memcmp(first->reward_coefs, second->reward_coefs, sizeof(first->reward_coefs)) == 0);
    }
    free_allocated(&first_env);
    free_allocated(&second_env);
    return 0;
}

static float decode_reward_target(const Drive *env, int coef_idx, float normalized) {
    float limit_min, limit_max;
    reward_coef_limits(env, coef_idx, &limit_min, &limit_max);
    return limit_min + (normalized + 1.0f) / NORMALIZED_SPAN * (limit_max - limit_min);
}

static int test_live_set_moves_reward_target_obs_and_round_trips(void) {
    Drive env = tuning_env(1, 337);
    int human_agent_idx = env.active_agent_indices[0];
    EXPECT_EQ_INT(c_select_human_agent(&env, human_agent_idx), 0);
    int obs_per_agent = compute_observation_size(&env);
    int partner_row = 1;
    const float *partner_target = &env.observations[partner_row * obs_per_agent + EGO_FEATURES];
    const float *human_target = &env.observations[EGO_FEATURES];

    drive_set_neutral_actions(&env);
    c_step(&env);
    float partner_before[NUM_REWARD_COEFS];
    float human_before[NUM_REWARD_COEFS];
    memcpy(partner_before, partner_target, sizeof(partner_before));
    memcpy(human_before, human_target, sizeof(human_before));

    CoefBounds window = sub_window(&env);
    EXPECT_EQ_INT(c_set_reward_coef_bounds(&env, window.min_values, window.max_values), 0);
    EXPECT_EQ_INT(c_resample_reward_coefs(&env), env.active_agent_count - 1);
    drive_set_neutral_actions(&env);
    c_step(&env);

    EXPECT_TRUE(memcmp(partner_before, partner_target, sizeof(partner_before)) != 0);
    EXPECT_TRUE(memcmp(human_before, human_target, sizeof(human_before)) == 0);
    const Agent *partner = &env.agents[env.active_agent_indices[partner_row]];
    for (int k = 0; k < BOUNDS_SAMPLED_COEF_COUNT; k++) {
        int c = BOUNDS_SAMPLED_COEFS[k];
        float decoded = decode_reward_target(&env, c, partner_target[c]);
        EXPECT_NEAR(decoded, partner->reward_coefs[c], ROUND_TRIP_TOLERANCE * fmaxf(1.0f, fabsf(decoded)));
        EXPECT_TRUE(partner_target[c] > -1.0f && partner_target[c] < 1.0f);
    }
    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_bounds_default_to_training_window);
    RUN_TEST(test_default_sampler_ignores_the_bounds_table_for_speed);
    RUN_TEST(test_set_bounds_accepts_the_exact_trained_window);
    RUN_TEST(test_narrowing_one_coefficient_flags_override);
    RUN_TEST(test_trained_window_after_a_level_restores_training_sampler);
    RUN_TEST(test_narrowing_a_fixed_profile_coefficient_is_rejected);
    RUN_TEST(test_goal_speed_window_needs_goal_speed_randomization);
    RUN_TEST(test_set_bounds_without_randomization_is_refused);
    RUN_TEST(test_speed_window_comes_from_conditioning_scale);
    RUN_TEST(test_set_bounds_rejects_invalid_and_leaves_state_unchanged);
    RUN_TEST(test_rejection_after_a_live_set_keeps_the_live_window);
    RUN_TEST(test_resample_counts_policy_cars_and_skips_human);
    RUN_TEST(test_resample_skips_non_policy_controllers);
    RUN_TEST(test_resampled_values_lie_inside_new_bounds);
    RUN_TEST(test_resample_without_randomization_is_refused);
    RUN_TEST(test_live_window_survives_reset);
    RUN_TEST(test_resample_is_deterministic);
    RUN_TEST(test_live_set_moves_reward_target_obs_and_round_trips);
    return test_summary(failures);
}
