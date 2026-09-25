#include "include/drive_fixture.h"
#include "include/test.h"

#define TEST_HORIZON_SECONDS 1.5f
#define TEST_WHEELBASE_M 2.7f

static Drive make_spline_dyn_env(void) {
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.action_type = ACTION_TYPE_SPLINE;
    env.dynamics_model = DYNAMICS_MODEL_SPLINE;
    env.spline_horizon_seconds = TEST_HORIZON_SECONDS;
    init_spline_dynamics_fields(&env);
    env.spline_consistency_lag_count = 1;
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    agent->wheelbase = TEST_WHEELBASE_M;
    agent->reward_coefs[REWARD_COEF_THROTTLE] = 1.0f;
    agent->reward_coefs[REWARD_COEF_STEER] = 1.0f;
    agent->phantom_braking_counter = 0;
    agent->is_phantom_braker = 0;
    return env;
}

static void set_entry_state(Agent *agent, float heading, float speed_signed, float accel_long, float accel_lat) {
    agent->sim_heading = heading;
    agent->cos_heading = cosf(heading);
    agent->sin_heading = sinf(heading);
    agent->sim_vx = speed_signed * agent->cos_heading;
    agent->sim_vy = speed_signed * agent->sin_heading;
    agent->accel_long = accel_long;
    agent->accel_lat = accel_lat;
    update_agent_speed(agent);
}

static void set_raw_action(Drive *env, const float raw[SPLINE_INTENT_FEATURES]) {
    float *actions = (float *) env->actions;
    for (int feature_idx = 0; feature_idx < SPLINE_INTENT_FEATURES; feature_idx++) {
        actions[feature_idx] = raw[feature_idx];
    }
}

static float quintic_jerk(const float coefs[6], float t) {
    return 6.0f * coefs[3] + 24.0f * coefs[4] * t + 60.0f * coefs[5] * t * t;
}

static float long_limit_for(float raw) {
    return (raw < 0.0f) ? -JERK_LONG[0] : JERK_LONG[3];
}

static int test_solver_reproduces_channels(void) {
    float dt = 0.3f, T = TEST_HORIZON_SECONDS;
    float c3, c4, c5;
    solve_quintic_from_jerk_channels(dt, T, 1.2f, -2.0f, 4.0f, &c3, &c4, &c5);
    float coefs[6] = {0.0f, 0.0f, 0.0f, c3, c4, c5};
    EXPECT_NEAR(evaluate_quintic_derivative(coefs, dt, 2), 1.2f, 1e-5f);
    EXPECT_NEAR(quintic_jerk(coefs, 0.5f * T), -2.0f, 1e-4f);
    EXPECT_NEAR(quintic_jerk(coefs, T), 4.0f, 1e-4f);
    // reference values from an independent Python solve of the same 3x3 system
    solve_quintic_from_jerk_channels(dt, T, 0.0f, 4.0f, 0.0f, &c3, &c4, &c5);
    float far_only[6] = {0.0f, 0.0f, 0.0f, c3, c4, c5};
    EXPECT_NEAR(evaluate_quintic_derivative(far_only, dt, 2), 0.0f, 1e-5f);
    solve_quintic_from_jerk_channels(dt, T, 0.0f, 0.0f, 0.0f, &c3, &c4, &c5);
    EXPECT_NEAR(c3, 0.0f, 1e-9f);
    EXPECT_NEAR(c4, 0.0f, 1e-9f);
    EXPECT_NEAR(c5, 0.0f, 1e-9f);
    return 0;
}

static int test_zero_action_follows_coast_curve(void) {
    Drive env = make_spline_dyn_env();
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    set_entry_state(agent, 0.0f, 10.0f, 1.0f, 0.0f);
    float x0 = agent->sim_x, y0 = agent->sim_y, dt = env.dt;
    const float raw[SPLINE_INTENT_FEATURES] = {0};
    set_raw_action(&env, raw);
    move_dynamics(&env, 0, env.active_agent_indices[0]);
    for (int coef_idx = 3; coef_idx < 6; coef_idx++) {
        EXPECT_NEAR(agent->spline_coefs_x[coef_idx], 0.0f, 1e-6f);
        EXPECT_NEAR(agent->spline_coefs_y[coef_idx], 0.0f, 1e-6f);
    }
    EXPECT_NEAR(agent->sim_x - x0, 10.0f * dt + 0.5f * 1.0f * dt * dt, 1e-4f);
    EXPECT_NEAR(agent->sim_y - y0, 0.0f, 1e-6f);
    EXPECT_NEAR(agent->sim_speed_signed, 10.0f + 1.0f * dt, 1e-4f);
    EXPECT_NEAR(agent->accel_long, 1.0f, 1e-5f);
    EXPECT_NEAR(agent->sim_heading, 0.0f, 1e-6f);
    free_allocated(&env);
    return 0;
}

static int test_channel_one_sets_executed_accel(void) {
    const float raw_values[2] = {0.5f, -0.5f};
    for (int case_idx = 0; case_idx < 2; case_idx++) {
        Drive env = make_spline_dyn_env();
        Agent *agent = &env.agents[env.active_agent_indices[0]];
        set_entry_state(agent, 0.0f, 10.0f, 0.0f, 0.0f);
        float raw[SPLINE_INTENT_FEATURES] = {0};
        raw[0] = raw_values[case_idx];
        set_raw_action(&env, raw);
        move_dynamics(&env, 0, env.active_agent_indices[0]);
        float expected = raw[0] * long_limit_for(raw[0]) * env.dt;
        EXPECT_NEAR(agent->accel_long, expected, 1e-5f);
        EXPECT_NEAR(agent->accel_lat, 0.0f, 1e-6f);
        EXPECT_NEAR(agent->jerk_long, expected / env.dt, 1e-3f);
        free_allocated(&env);
    }
    return 0;
}

static int test_far_channels_do_not_change_executed_accel(void) {
    Drive env = make_spline_dyn_env();
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    set_entry_state(agent, 0.0f, 10.0f, 0.0f, 0.0f);
    const float raw[SPLINE_INTENT_FEATURES] = {0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f};
    set_raw_action(&env, raw);
    move_dynamics(&env, 0, env.active_agent_indices[0]);
    EXPECT_NEAR(agent->accel_long, 0.0f, 1e-4f);
    float T = env.spline_horizon_seconds;
    EXPECT_NEAR(quintic_jerk(agent->spline_coefs_x, 0.5f * T), JERK_LONG[3], 1e-3f);
    EXPECT_NEAR(quintic_jerk(agent->spline_coefs_x, T), JERK_LONG[0], 1e-3f);
    free_allocated(&env);
    return 0;
}

static int test_car_lands_on_its_curve(void) {
    Drive env = make_spline_dyn_env();
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    set_entry_state(agent, 0.4f, 8.0f, -0.5f, 0.8f);
    const float raw[SPLINE_INTENT_FEATURES] = {0.3f, -0.2f, 0.5f, 0.4f, 0.1f, -0.3f};
    set_raw_action(&env, raw);
    move_dynamics(&env, 0, env.active_agent_indices[0]);
    float dt = env.dt;
    EXPECT_NEAR(agent->sim_x, evaluate_quintic_derivative(agent->spline_coefs_x, dt, 0), 1e-4f);
    EXPECT_NEAR(agent->sim_y, evaluate_quintic_derivative(agent->spline_coefs_y, dt, 0), 1e-4f);
    EXPECT_NEAR(agent->sim_vx, evaluate_quintic_derivative(agent->spline_coefs_x, dt, 1), 1e-5f);
    EXPECT_NEAR(agent->sim_vy, evaluate_quintic_derivative(agent->spline_coefs_y, dt, 1), 1e-5f);
    for (int feature_idx = 0; feature_idx < SPLINE_INTENT_FEATURES; feature_idx++) {
        EXPECT_NEAR(agent->spline_intent[feature_idx], raw[feature_idx], 1e-7f);
    }
    free_allocated(&env);
    return 0;
}

static int test_standstill_does_not_spin(void) {
    Drive env = make_spline_dyn_env();
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    set_entry_state(agent, 0.7f, 0.0f, 0.0f, 0.0f);
    const float zero[SPLINE_INTENT_FEATURES] = {0};
    set_raw_action(&env, zero);
    move_dynamics(&env, 0, env.active_agent_indices[0]);
    EXPECT_NEAR(agent->sim_heading, 0.7f, 1e-7f);
    EXPECT_NEAR(agent->sim_speed, 0.0f, 1e-7f);

    // a pure sideways request from rest barely turns the car: the clamp scales with distance travelled
    const float sideways[SPLINE_INTENT_FEATURES] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    set_raw_action(&env, sideways);
    move_dynamics(&env, 0, env.active_agent_indices[0]);
    float arc_length_m = 0.5f * agent->sim_speed * env.dt;
    EXPECT_TRUE(fabsf(agent->sim_heading - 0.7f) <= arc_length_m * tanf(STEERING_ANGLE_LIMIT) / TEST_WHEELBASE_M + 1e-6f);
    free_allocated(&env);
    return 0;
}

static int test_heading_step_is_bounded(void) {
    Drive env = make_spline_dyn_env();
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    set_entry_state(agent, 0.0f, 3.0f, 0.0f, 0.0f);
    unsigned int state = 12345u;
    for (int step = 0; step < 40; step++) {
        float raw[SPLINE_INTENT_FEATURES];
        for (int feature_idx = 0; feature_idx < SPLINE_INTENT_FEATURES; feature_idx++) {
            state = state * 1103515245u + 12345u;
            raw[feature_idx] = ((float) ((state >> 8) & 0xFFFF) / 65535.0f) * 2.0f - 1.0f;
        }
        set_raw_action(&env, raw);
        float heading_before = agent->sim_heading;
        float speed_before = agent->sim_speed;
        move_dynamics(&env, 0, env.active_agent_indices[0]);
        float arc_length_m = 0.5f * (speed_before + agent->sim_speed) * env.dt;
        float heading_change = normalize_heading(agent->sim_heading - heading_before);
        EXPECT_TRUE(fabsf(heading_change) <= arc_length_m * tanf(STEERING_ANGLE_LIMIT) / TEST_WHEELBASE_M + 1e-5f);
        EXPECT_TRUE(fabsf(agent->steering_angle) <= STEERING_ANGLE_LIMIT + 1e-5f);
        EXPECT_FINITE(agent->sim_x);
        EXPECT_FINITE(agent->accel_long);
    }
    free_allocated(&env);
    return 0;
}

static int test_straight_reverse_keeps_heading(void) {
    Drive env = make_spline_dyn_env();
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    set_entry_state(agent, 1.0f, -1.5f, 0.0f, 0.0f);
    const float zero[SPLINE_INTENT_FEATURES] = {0};
    set_raw_action(&env, zero);
    for (int step = 0; step < 30; step++) {
        move_dynamics(&env, 0, env.active_agent_indices[0]);
        EXPECT_NEAR(agent->sim_heading, 1.0f, 1e-5f);
        EXPECT_TRUE(agent->sim_speed_signed < 0.0f);
    }
    free_allocated(&env);
    return 0;
}

static int test_phantom_braking_stops_at_zero(void) {
    Drive env = make_spline_dyn_env();
    int agent_idx = env.active_agent_indices[0];
    Agent *agent = &env.agents[agent_idx];
    agent->controller = CONTROLLER_POLICY;
    set_entry_state(agent, 0.0f, 0.5f, 0.0f, 0.0f);
    const float raw[SPLINE_INTENT_FEATURES] = {0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    set_raw_action(&env, raw);
    agent->spline_history_count = 2;
    int steps = 10;
    agent->phantom_braking_counter = steps + 1; // move_dynamics decrements before the branch
    for (int step = 0; step < steps; step++) {
        move_dynamics(&env, 0, agent_idx);
        compute_spline_rewards(&env, 0);
    }
    EXPECT_NEAR(agent->sim_speed, 0.0f, 0.0f);
    EXPECT_NEAR(agent->accel_long, 0.0f, 0.0f);
    EXPECT_NEAR(agent->accel_lat, 0.0f, 0.0f);
    EXPECT_NEAR(agent->spline_intent[0], 0.3f, 1e-7f);
    EXPECT_EQ_INT(agent->spline_history_count, 0);
    free_allocated(&env);
    return 0;
}

static int test_replanning_the_same_curve_costs_nothing(void) {
    Drive env = make_spline_dyn_env();
    int agent_idx = env.active_agent_indices[0];
    Agent *agent = &env.agents[agent_idx];
    agent->controller = CONTROLLER_POLICY;
    set_entry_state(agent, 0.0f, 10.0f, 0.0f, 0.0f);
    float dt = env.dt, T = env.spline_horizon_seconds;
    const float first[SPLINE_INTENT_FEATURES] = {0.2f, -0.1f, 0.05f, 0.0f, 0.0f, 0.0f};
    set_raw_action(&env, first);
    move_dynamics(&env, 0, agent_idx);
    compute_spline_rewards(&env, 0);

    // continue the stored curve from the new start: same accel change, same jerk at the shifted times
    const float *old = env.agents[agent_idx].spline_history_coefs_x[0];
    float delta_accel = evaluate_quintic_derivative(old, 2.0f * dt, 2) - evaluate_quintic_derivative(old, dt, 2);
    float jerk_mid = quintic_jerk(old, dt + 0.5f * T);
    float jerk_end = quintic_jerk(old, dt + T);
    float next[SPLINE_INTENT_FEATURES] = {0};
    next[0] = delta_accel / (long_limit_for(delta_accel) * dt);
    next[1] = jerk_mid / long_limit_for(jerk_mid);
    next[2] = jerk_end / long_limit_for(jerk_end);
    for (int channel_idx = 0; channel_idx < SPLINE_CHANNELS_PER_AXIS; channel_idx++) {
        EXPECT_TRUE(fabsf(next[channel_idx]) <= 1.0f);
    }
    set_raw_action(&env, next);
    move_dynamics(&env, 0, agent_idx);
    float lag1_cost = -1.0f;
    float cost = compute_spline_consistency_cost(agent, 1, env.spline_consistency_num_samples + 1, dt, &lag1_cost);
    EXPECT_NEAR(cost, 0.0f, 1e-6f);
    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_solver_reproduces_channels);
    RUN_TEST(test_zero_action_follows_coast_curve);
    RUN_TEST(test_channel_one_sets_executed_accel);
    RUN_TEST(test_far_channels_do_not_change_executed_accel);
    RUN_TEST(test_car_lands_on_its_curve);
    RUN_TEST(test_standstill_does_not_spin);
    RUN_TEST(test_heading_step_is_bounded);
    RUN_TEST(test_straight_reverse_keeps_heading);
    RUN_TEST(test_phantom_braking_stops_at_zero);
    RUN_TEST(test_replanning_the_same_curve_costs_nothing);
    return test_summary(failures);
}
