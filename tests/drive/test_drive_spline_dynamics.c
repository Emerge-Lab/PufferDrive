#include "include/drive_fixture.h"
#include "include/test.h"

// Shared setup for the move_dynamics-level spline tests below: a fresh single-agent env in
// spline mode, with a real map (needed by drive_test_make_env), a chosen planning horizon T,
// and the derived offset-limit fields populated via init_spline_dynamics_fields (mirroring what
// binding.c does at real env-init time, since this fixture never goes through binding.c).
static Drive make_spline_env(float T) {
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.action_type = ACTION_TYPE_SPLINE;
    env.dynamics_model = DYNAMICS_MODEL_JERK;
    env.spline_horizon_seconds = T;
    init_spline_dynamics_fields(&env);
    return env;
}

static void set_spline_action(
    Drive *env,
    float p1_fwd,
    float v1_fwd,
    float a1_fwd,
    float p1_left,
    float v1_left,
    float a1_left) {
    float (*actions)[6] = (float (*)[6]) env->actions;
    actions[0][0] = p1_fwd;
    actions[0][1] = v1_fwd;
    actions[0][2] = a1_fwd;
    actions[0][3] = p1_left;
    actions[0][4] = v1_left;
    actions[0][5] = a1_left;
}

static int test_straight_ahead_target_gives_near_zero_lateral_accel(void) {
    Drive env = make_spline_env(1.0f);
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    agent->sim_heading = 0.0f;
    agent->cos_heading = 1.0f;
    agent->sin_heading = 0.0f;
    agent->sim_vx = 5.0f;
    agent->sim_vy = 0.0f;
    agent->accel_long = 0.0f;
    agent->accel_lat = 0.0f;
    set_spline_action(&env, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f); // pure forward target
    move_dynamics(&env, 0, env.active_agent_indices[0]);
    EXPECT_NEAR(agent->accel_lat, 0.0f, 1e-3f);
    free_allocated(&env);
    return 0;
}

static int test_lateral_offset_sign_matches_turn_direction(void) {
    Drive env = make_spline_env(1.0f);
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    agent->sim_heading = 0.0f;
    agent->cos_heading = 1.0f;
    agent->sin_heading = 0.0f;
    agent->sim_vx = 5.0f;
    agent->sim_vy = 0.0f;
    agent->accel_long = 0.0f;
    agent->accel_lat = 0.0f;
    set_spline_action(&env, 0.3f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f); // positive p1_left = target to the left
    move_dynamics(&env, 0, env.active_agent_indices[0]);
    // Positive left offset must curve the agent left: counterclockwise, positive yaw_rate,
    // consistent with (accel_long, accel_lat) already being the (local_x, local_y) convention
    // this codebase's curvature/steering round-trip assumes (no sign flip).
    EXPECT_TRUE(agent->yaw_rate > 0.0f);
    free_allocated(&env);
    return 0;
}

static int test_infeasible_target_still_respects_accel_limits(void) {
    Drive env = make_spline_env(1.0f);
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    agent->sim_heading = 0.0f;
    agent->cos_heading = 1.0f;
    agent->sin_heading = 0.0f;
    agent->sim_vx = env.base_max_speed_mps;
    agent->sim_vy = 0.0f;
    agent->accel_long = ACCEL_LONG_LIMIT[1]; // start at max accel
    agent->accel_lat = ACCEL_LAT_LIMIT[1];   // start at max lateral accel
    // Ask for a full reversal in both axes at maximum magnitude.
    set_spline_action(&env, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f);
    move_dynamics(&env, 0, env.active_agent_indices[0]);
    float c_acc = agent->reward_coefs[REWARD_COEF_ACC];
    EXPECT_TRUE(agent->accel_long >= ACCEL_LONG_LIMIT[0] - 1e-3f);
    EXPECT_TRUE(agent->accel_long <= ACCEL_LONG_LIMIT[1] * c_acc + 1e-3f);
    EXPECT_TRUE(agent->accel_lat >= ACCEL_LAT_LIMIT[0] - 1e-3f);
    EXPECT_TRUE(agent->accel_lat <= ACCEL_LAT_LIMIT[1] + 1e-3f);
    free_allocated(&env);
    return 0;
}

static int test_discontinuous_target_realized_jerk_stays_bounded(void) {
    Drive env = make_spline_env(1.0f);
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    agent->sim_heading = 0.0f;
    agent->cos_heading = 1.0f;
    agent->sin_heading = 0.0f;
    agent->sim_vx = 5.0f;
    agent->sim_vy = 0.0f;
    agent->accel_long = 0.0f;
    agent->accel_lat = 0.0f;

    // Step 1: neutral target, starting from rest-accel — nothing dramatic should happen.
    set_spline_action(&env, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    move_dynamics(&env, 0, env.active_agent_indices[0]);

    // Step 2: flip to a maximally different target in one step (max-accel-forward and
    // max-left, from wherever step 1 landed) — this is the discontinuity the jerk clamp exists
    // to catch: the vehicle must ramp, not snap.
    set_spline_action(&env, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
    move_dynamics(&env, 0, env.active_agent_indices[0]);

    EXPECT_TRUE(agent->jerk_long >= JERK_LONG[0] - 1e-2f);
    EXPECT_TRUE(agent->jerk_long <= JERK_LONG[3] + 1e-2f);
    EXPECT_TRUE(agent->jerk_lat >= JERK_LAT[0] - 1e-2f);
    EXPECT_TRUE(agent->jerk_lat <= JERK_LAT[2] + 1e-2f);
    free_allocated(&env);
    return 0;
}

static int test_near_zero_speed_lateral_target_stays_finite(void) {
    Drive env = make_spline_env(1.0f);
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    agent->sim_heading = 0.0f;
    agent->cos_heading = 1.0f;
    agent->sin_heading = 0.0f;
    agent->sim_vx = 0.001f; // near-stationary — the fragile regime (v_eff floor)
    agent->sim_vy = 0.0f;
    agent->accel_long = 0.0f;
    agent->accel_lat = 0.0f;
    set_spline_action(&env, 0.1f, 0.5f, 0.0f, 1.0f, 0.5f, 1.0f); // lateral target at low speed
    move_dynamics(&env, 0, env.active_agent_indices[0]);
    EXPECT_FINITE(agent->steering_angle);
    EXPECT_FINITE(agent->sim_x);
    EXPECT_FINITE(agent->sim_y);
    EXPECT_FINITE(agent->sim_heading);
    free_allocated(&env);
    return 0;
}

static int test_stopped_agent_spline_action_still_clears_motion(void) {
    Drive env = {0};
    Agent agent = drive_test_agent(5.0f, 3.0f, 0.5f);
    env.agents = &agent;
    env.dt = 0.1f;
    env.action_type = ACTION_TYPE_SPLINE;
    env.dynamics_model = DYNAMICS_MODEL_JERK;
    // env.actions is intentionally left NULL here (bare env, no allocate()) — if the new spline
    // branch ever bypassed the stopped-agent early return, reading env->actions would crash,
    // making a clean pass a meaningful proof the guard still fires first.
    agent.stopped = 1;
    agent.sim_vx = 4.0f;
    update_agent_speed(&agent);
    agent.steering_angle = 0.3f;

    move_dynamics(&env, 0, 0);

    EXPECT_NEAR(agent.sim_vx, 0.0f, 1e-6f);
    EXPECT_NEAR(agent.sim_speed, 0.0f, 1e-6f);
    EXPECT_NEAR(agent.steering_angle, 0.0f, 1e-6f);
    return 0;
}

static int test_consistency_cost_math(void) {
    float dt = 0.1f;
    // curr: a straight line x(t) = t (c1=1, everything else 0).
    float curr_x[6] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float curr_y[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // prev: the exact same line, shifted so prev(t) == curr(t - dt): prev_c0 = curr_c0 - c1*dt.
    float prev_x[6] = {-1.0f * dt, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float prev_y[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    float cost_aligned = compute_spline_consistency_cost(prev_x, prev_y, curr_x, curr_y, 5, dt);
    EXPECT_NEAR(cost_aligned, 0.0f, 1e-4f);

    // A curve pointing somewhere unrelated should cost far more.
    float curr_far_x[6] = {100.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float cost_far = compute_spline_consistency_cost(prev_x, prev_y, curr_far_x, curr_y, 5, dt);
    EXPECT_TRUE(cost_far > 100.0f);
    return 0;
}

static int test_consistency_reward_guard_and_rotation(void) {
    Drive env;
    Agent agent;
    Log log;
    int active[1];
    float reward[1] = {0};
    memset(&env, 0, sizeof(env));
    memset(&agent, 0, sizeof(agent));
    memset(&log, 0, sizeof(log));
    agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    active[0] = 0;
    env.agents = &agent;
    env.active_agent_indices = active;
    env.logs = &log;
    env.rewards = reward;
    env.active_agent_count = 1;
    env.dt = 0.1f;
    env.simulation_mode = SIMULATION_MODE_GIGAFLOW;
    env.compute_eval_metrics = 0;
    env.action_type = ACTION_TYPE_SPLINE;
    env.spline_consistency_num_samples = 5;
    agent.reward_coefs[REWARD_COEF_TRAJECTORY_CONSISTENCY] = 1.0f;

    // First call ever for this agent: no previous curve exists yet, so the guard must skip the
    // term entirely, regardless of what spline_coefs_x holds.
    agent.spline_coefs_x[1] = 1.0f; // c1 = 1: a line with some slope
    agent.spline_history_valid = 0;
    reward[0] = 0.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(reward[0], 0.0f, 1e-5f);
    EXPECT_EQ_INT(agent.spline_history_valid, 1);           // now flips on...
    EXPECT_NEAR(agent.prev_spline_coefs_x[1], 1.0f, 1e-5f); // ...and curr got rotated into prev

    // Second call: continue the exact same line one step later -> near-zero penalty.
    agent.spline_coefs_x[0] = 1.0f * env.dt;
    agent.spline_coefs_x[1] = 1.0f;
    reward[0] = 0.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(reward[0], 0.0f, 1e-3f);

    // Third call: an abrupt, unrelated target -> a materially large penalty.
    agent.spline_coefs_x[0] = 500.0f;
    agent.spline_coefs_x[1] = 0.0f;
    reward[0] = 0.0f;
    compute_rewards(&env, 0);
    EXPECT_TRUE(reward[0] < -10.0f);
    return 0;
}

static int test_spline_intent_obs_block(void) {
    Drive env = make_spline_env(1.0f);
    int agent_idx = env.active_agent_indices[0];
    Agent *agent = &env.agents[agent_idx];

    int spline_obs_size = compute_observation_size(&env);
    env.action_type = ACTION_TYPE_CONTINUOUS;
    int baseline_obs_size = compute_observation_size(&env);
    env.action_type = ACTION_TYPE_SPLINE;
    EXPECT_EQ_INT(spline_obs_size - baseline_obs_size, SPLINE_INTENT_FEATURES);

    // Before the agent has ever acted there is no intent to report, so the block reads zeros.
    compute_observations(&env);
    for (int feature_idx = 0; feature_idx < SPLINE_INTENT_FEATURES; feature_idx++) {
        EXPECT_NEAR(env.observations[EGO_FEATURES + feature_idx], 0.0f, 1e-6f);
    }

    agent->sim_heading = 0.0f;
    agent->cos_heading = 1.0f;
    agent->sin_heading = 0.0f;
    agent->sim_vx = 5.0f;
    agent->sim_vy = 0.0f;
    agent->accel_long = 0.0f;
    agent->accel_lat = 0.0f;
    const float emitted[SPLINE_INTENT_FEATURES] = {0.5f, -0.25f, 0.75f, -0.4f, 0.1f, 0.6f};
    set_spline_action(&env, emitted[0], emitted[1], emitted[2], emitted[3], emitted[4], emitted[5]);
    move_dynamics(&env, 0, agent_idx);
    compute_observations(&env);

    // The block is the emitted action verbatim, unaffected by whatever the clamps did downstream.
    for (int feature_idx = 0; feature_idx < SPLINE_INTENT_FEATURES; feature_idx++) {
        EXPECT_NEAR(env.observations[EGO_FEATURES + feature_idx], emitted[feature_idx], 1e-6f);
    }

    // A reset clears it, so a recycled agent slot cannot leak last episode's intent.
    reset_agent_state(agent);
    compute_observations(&env);
    for (int feature_idx = 0; feature_idx < SPLINE_INTENT_FEATURES; feature_idx++) {
        EXPECT_NEAR(env.observations[EGO_FEATURES + feature_idx], 0.0f, 1e-6f);
    }

    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_straight_ahead_target_gives_near_zero_lateral_accel);
    RUN_TEST(test_lateral_offset_sign_matches_turn_direction);
    RUN_TEST(test_infeasible_target_still_respects_accel_limits);
    RUN_TEST(test_discontinuous_target_realized_jerk_stays_bounded);
    RUN_TEST(test_near_zero_speed_lateral_target_stays_finite);
    RUN_TEST(test_stopped_agent_spline_action_still_clears_motion);
    RUN_TEST(test_consistency_cost_math);
    RUN_TEST(test_consistency_reward_guard_and_rotation);
    RUN_TEST(test_spline_intent_obs_block);
    return test_summary(failures);
}
