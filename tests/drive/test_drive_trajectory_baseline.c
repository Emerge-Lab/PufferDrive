#include "include/drive_fixture.h"
#include "include/test.h"

#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#define TEST_HORIZON_SECONDS 1.5f
#define TEST_WHEELBASE_M 2.7f
#define RANDOM_STEP_COUNT 300
#define ROLLOUT_AGENT_COUNT 8
#define ROLLOUT_AGENT_CAPACITY 64
#define ROLLOUT_STEP_COUNT 200
#define JERK_ACTION_STRIDE 2
// measured worst at map-scale coordinates: 6e-5 m, 9e-6 m/s, 7e-5 m/s^2
#define END_POSITION_TOLERANCE_M 1e-3f
#define END_VELOCITY_TOLERANCE_MPS 1e-4f
#define ACCEL_TOLERANCE_MPS2 1e-3f
#define SENTINEL_COEF 12345.0f

enum {
    DEATH_NON_FINITE_STATE,
    DEATH_WRONG_DYNAMICS_MODEL
};

static unsigned int lcg_state = 12345u;
static float lcg_uniform(void) {
    lcg_state = lcg_state * 1103515245u + 12345u;
    return ((float) ((lcg_state >> 8) & 0xFFFF) / 65535.0f) * 2.0f - 1.0f;
}

static Drive make_env_on_map(
    const char *map_file,
    int simulation_mode,
    float dt,
    int trajectory_baseline,
    int num_agents) {
    Drive env = drive_test_env_config(map_file, simulation_mode, num_agents, 0);
    env.dynamics_model = DYNAMICS_MODEL_JERK;
    env.action_type = ACTION_TYPE_CONTINUOUS;
    env.dt = dt;
    env.spline_horizon_seconds = TEST_HORIZON_SECONDS;
    env.spline_consistency_lag_count = 1;
    env.trajectory_baseline = trajectory_baseline;
    allocate(&env);
    c_reset(&env);
    return env;
}

static Drive make_baseline_env(float dt, int trajectory_baseline, int num_agents) {
    return make_env_on_map(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, dt, trajectory_baseline, num_agents);
}

static Agent *prepare_single_agent(Drive *env, float heading, float speed_signed) {
    Agent *agent = &env->agents[env->active_agent_indices[0]];
    agent->wheelbase = TEST_WHEELBASE_M;
    agent->reward_coefs[REWARD_COEF_THROTTLE] = 1.0f;
    agent->reward_coefs[REWARD_COEF_STEER] = 1.0f;
    agent->reward_coefs[REWARD_COEF_ACC] = 1.0f;
    agent->reward_coefs[REWARD_COEF_SPEED] = 1.0f;
    agent->phantom_braking_counter = 0;
    agent->is_phantom_braker = 0;
    agent->sim_heading = heading;
    agent->cos_heading = cosf(heading);
    agent->sin_heading = sinf(heading);
    agent->sim_vx = speed_signed * agent->cos_heading;
    agent->sim_vy = speed_signed * agent->sin_heading;
    agent->accel_long = 0.0f;
    agent->accel_lat = 0.0f;
    agent->steering_angle = 0.0f;
    update_agent_speed(agent);
    return agent;
}

static void step_with_fit(Drive *env, Agent *agent, float raw_long, float raw_lat) {
    float *actions = (float *) env->actions;
    actions[0] = raw_long;
    actions[1] = raw_lat;
    int agent_idx = env->active_agent_indices[0];
    begin_baseline_trajectory(env, agent);
    move_dynamics(env, 0, agent_idx);
    fit_baseline_trajectory(env, agent);
}

static void fill_coefs(Agent *agent, float value) {
    for (int coef_idx = 0; coef_idx < 6; coef_idx++) {
        agent->spline_coefs_x[coef_idx] = value;
        agent->spline_coefs_y[coef_idx] = value;
    }
}

static int coefs_all_equal(const Agent *agent, float value) {
    for (int coef_idx = 0; coef_idx < 6; coef_idx++) {
        if (agent->spline_coefs_x[coef_idx] != value || agent->spline_coefs_y[coef_idx] != value) {
            return 0;
        }
    }
    return 1;
}

static int curve_collapsed_on_car(const Agent *agent) {
    if (agent->spline_coefs_x[0] != agent->sim_x || agent->spline_coefs_y[0] != agent->sim_y) {
        return 0;
    }
    for (int coef_idx = 1; coef_idx < 6; coef_idx++) {
        if (agent->spline_coefs_x[coef_idx] != 0.0f || agent->spline_coefs_y[coef_idx] != 0.0f) {
            return 0;
        }
    }
    return 1;
}

static int all_active_curves_collapsed(const Drive *env) {
    for (int i = 0; i < env->active_agent_count; i++) {
        if (!curve_collapsed_on_car(&env->agents[env->active_agent_indices[i]])) {
            return 0;
        }
    }
    return 1;
}

// Random full-range jerk actions drive the car into the speed caps, zero-crossing snaps and reverse.
static int check_fit_over_random_steps(float dt) {
    Drive env = make_baseline_env(dt, 1, 1);
    Agent *agent = prepare_single_agent(&env, 0.4f, 8.0f);
    lcg_state = 12345u;
    int saw_reverse = 0, saw_reverse_cap = 0;
    for (int step = 0; step < RANDOM_STEP_COUNT; step++) {
        float start_x = agent->sim_x, start_y = agent->sim_y, start_vx = agent->sim_vx, start_vy = agent->sim_vy;
        float start_accel_long = agent->accel_long, start_accel_lat = agent->accel_lat;
        float start_cos = agent->cos_heading, start_sin = agent->sin_heading;
        step_with_fit(&env, agent, lcg_uniform(), lcg_uniform());
        const float *coefs_x = agent->spline_coefs_x;
        const float *coefs_y = agent->spline_coefs_y;
        for (int coef_idx = 0; coef_idx < 6; coef_idx++) {
            EXPECT_FINITE(coefs_x[coef_idx]);
            EXPECT_FINITE(coefs_y[coef_idx]);
        }

        EXPECT_TRUE(evaluate_quintic_derivative(coefs_x, 0.0f, 0) == start_x);
        EXPECT_TRUE(evaluate_quintic_derivative(coefs_y, 0.0f, 0) == start_y);
        EXPECT_TRUE(evaluate_quintic_derivative(coefs_x, 0.0f, 1) == start_vx);
        EXPECT_TRUE(evaluate_quintic_derivative(coefs_y, 0.0f, 1) == start_vy);
        // rotated by hand with the pre-step heading, independent of the frame helpers the fit uses
        float start_ax = evaluate_quintic_derivative(coefs_x, 0.0f, 2);
        float start_ay = evaluate_quintic_derivative(coefs_y, 0.0f, 2);
        EXPECT_NEAR(start_ax * start_cos + start_ay * start_sin, start_accel_long, ACCEL_TOLERANCE_MPS2);
        EXPECT_NEAR(-start_ax * start_sin + start_ay * start_cos, start_accel_lat, ACCEL_TOLERANCE_MPS2);

        EXPECT_NEAR(evaluate_quintic_derivative(coefs_x, dt, 0), agent->sim_x, END_POSITION_TOLERANCE_M);
        EXPECT_NEAR(evaluate_quintic_derivative(coefs_y, dt, 0), agent->sim_y, END_POSITION_TOLERANCE_M);
        EXPECT_NEAR(evaluate_quintic_derivative(coefs_x, dt, 1), agent->sim_vx, END_VELOCITY_TOLERANCE_MPS);
        EXPECT_NEAR(evaluate_quintic_derivative(coefs_y, dt, 1), agent->sim_vy, END_VELOCITY_TOLERANCE_MPS);
        float end_ax = evaluate_quintic_derivative(coefs_x, dt, 2);
        float end_ay = evaluate_quintic_derivative(coefs_y, dt, 2);
        EXPECT_NEAR(end_ax * agent->cos_heading + end_ay * agent->sin_heading, agent->accel_long, ACCEL_TOLERANCE_MPS2);
        EXPECT_NEAR(-end_ax * agent->sin_heading + end_ay * agent->cos_heading, agent->accel_lat, ACCEL_TOLERANCE_MPS2);

        saw_reverse |= agent->sim_speed_signed < 0.0f;
        saw_reverse_cap |= agent->sim_speed_signed == MAX_BACKWARD_SPEED;
    }
    EXPECT_TRUE(saw_reverse);
    EXPECT_TRUE(saw_reverse_cap);
    free_allocated(&env);
    return 0;
}

static int test_fit_passes_through_both_step_ends(void) {
    EXPECT_EQ_INT(check_fit_over_random_steps(0.3f), 0);
    EXPECT_EQ_INT(check_fit_over_random_steps(0.1f), 0);
    return 0;
}

static int test_steady_cruise_fits_a_straight_line(void) {
    Drive env = make_baseline_env(0.3f, 1, 1);
    Agent *agent = prepare_single_agent(&env, 0.4f, 10.0f);
    step_with_fit(&env, agent, 0.0f, 0.0f);
    float dt = env.dt;
    for (int coef_idx = 2; coef_idx < 6; coef_idx++) {
        EXPECT_NEAR(agent->spline_coefs_x[coef_idx], 0.0f, 1e-2f);
        EXPECT_NEAR(agent->spline_coefs_y[coef_idx], 0.0f, 1e-2f);
    }
    float far_t = TEST_HORIZON_SECONDS;
    float far_x = evaluate_quintic_derivative(agent->spline_coefs_x, far_t, 0);
    float far_y = evaluate_quintic_derivative(agent->spline_coefs_y, far_t, 0);
    EXPECT_NEAR(far_x, agent->spline_coefs_x[0] + 10.0f * cosf(0.4f) * far_t, 0.1f);
    EXPECT_NEAR(far_y, agent->spline_coefs_y[0] + 10.0f * sinf(0.4f) * far_t, 0.1f);
    EXPECT_NEAR(evaluate_quintic_derivative(agent->spline_coefs_x, dt, 0), agent->sim_x, END_POSITION_TOLERANCE_M);
    free_allocated(&env);
    return 0;
}

static int test_stopped_agent_curve_stays_put(void) {
    Drive env = make_baseline_env(0.3f, 1, 1);
    Agent *agent = prepare_single_agent(&env, 0.4f, 8.0f);
    agent->accel_long = 1.0f;
    agent->stopped = 1;
    float stop_x = agent->sim_x, stop_y = agent->sim_y;
    float dt = env.dt;

    // the stop step: the jerk model zeroes the motion in place, so the curve must end at rest there
    step_with_fit(&env, agent, 1.0f, 1.0f);
    EXPECT_NEAR(evaluate_quintic_derivative(agent->spline_coefs_x, dt, 0), stop_x, END_POSITION_TOLERANCE_M);
    EXPECT_NEAR(evaluate_quintic_derivative(agent->spline_coefs_y, dt, 0), stop_y, END_POSITION_TOLERANCE_M);
    EXPECT_NEAR(evaluate_quintic_derivative(agent->spline_coefs_x, dt, 1), 0.0f, END_VELOCITY_TOLERANCE_MPS);
    EXPECT_NEAR(evaluate_quintic_derivative(agent->spline_coefs_y, dt, 1), 0.0f, END_VELOCITY_TOLERANCE_MPS);

    // from rest to rest: every coefficient past c0 is exactly zero
    step_with_fit(&env, agent, 1.0f, 1.0f);
    EXPECT_TRUE(agent->spline_coefs_x[0] == stop_x);
    EXPECT_TRUE(agent->spline_coefs_y[0] == stop_y);
    for (int coef_idx = 1; coef_idx < 6; coef_idx++) {
        EXPECT_TRUE(agent->spline_coefs_x[coef_idx] == 0.0f);
        EXPECT_TRUE(agent->spline_coefs_y[coef_idx] == 0.0f);
    }
    free_allocated(&env);
    return 0;
}

static int test_removed_agent_and_disabled_flag_leave_curve_untouched(void) {
    Drive removed_env = make_baseline_env(0.3f, 1, 1);
    Agent *removed = prepare_single_agent(&removed_env, 0.4f, 8.0f);
    removed->removed = 1;
    fill_coefs(removed, SENTINEL_COEF);
    step_with_fit(&removed_env, removed, 1.0f, 1.0f);
    EXPECT_TRUE(coefs_all_equal(removed, SENTINEL_COEF));
    free_allocated(&removed_env);

    Drive disabled_env = make_baseline_env(0.3f, 0, 1);
    Agent *moving = prepare_single_agent(&disabled_env, 0.4f, 8.0f);
    fill_coefs(moving, SENTINEL_COEF);
    step_with_fit(&disabled_env, moving, 1.0f, 1.0f);
    EXPECT_TRUE(coefs_all_equal(moving, SENTINEL_COEF));
    free_allocated(&disabled_env);
    return 0;
}

static int agent_motion_identical(const Agent *a, const Agent *b) {
    return a->sim_x == b->sim_x && a->sim_y == b->sim_y && a->sim_z == b->sim_z && a->sim_heading == b->sim_heading
        && a->sim_vx == b->sim_vx && a->sim_vy == b->sim_vy && a->accel_long == b->accel_long
        && a->accel_lat == b->accel_lat && a->jerk_long == b->jerk_long && a->jerk_lat == b->jerk_lat
        && a->steering_angle == b->steering_angle && a->stopped == b->stopped && a->removed == b->removed;
}

// Full c_step with the flag on and off must stay bit-identical in everything but the curve itself.
static int check_rollout_identical(
    const char *map_file,
    int simulation_mode,
    int num_agents,
    int collision_behavior,
    int offroad_behavior) {
    Drive baseline_env = make_env_on_map(map_file, simulation_mode, 0.3f, 1, num_agents);
    Drive plain_env = make_env_on_map(map_file, simulation_mode, 0.3f, 0, num_agents);
    baseline_env.collision_behavior = plain_env.collision_behavior = collision_behavior;
    baseline_env.offroad_behavior = plain_env.offroad_behavior = offroad_behavior;
    EXPECT_TRUE(all_active_curves_collapsed(&baseline_env));
    EXPECT_EQ_INT(baseline_env.active_agent_count, plain_env.active_agent_count);
    int active_count = baseline_env.active_agent_count;
    int obs_floats = active_count * compute_observation_size(&baseline_env);
    lcg_state = 777u;
    int fitted_agent_steps = 0, reset_steps = 0;
    float pre_step_x[ROLLOUT_AGENT_CAPACITY], pre_step_y[ROLLOUT_AGENT_CAPACITY];
    int expect_fit[ROLLOUT_AGENT_CAPACITY];
    EXPECT_TRUE(baseline_env.num_total_agents <= ROLLOUT_AGENT_CAPACITY);
    for (int step = 0; step < ROLLOUT_STEP_COUNT; step++) {
        float *baseline_actions = (float *) baseline_env.actions;
        float *plain_actions = (float *) plain_env.actions;
        for (int action_idx = 0; action_idx < active_count * JERK_ACTION_STRIDE; action_idx++) {
            baseline_actions[action_idx] = plain_actions[action_idx] = lcg_uniform();
        }
        memset(expect_fit, 0, sizeof(expect_fit));
        for (int i = 0; i < active_count; i++) {
            int agent_idx = baseline_env.active_agent_indices[i];
            const Agent *agent = &baseline_env.agents[agent_idx];
            pre_step_x[agent_idx] = agent->sim_x;
            pre_step_y[agent_idx] = agent->sim_y;
            expect_fit[agent_idx] = agent->controller == CONTROLLER_POLICY && !agent->removed;
        }
        int timestep_before = baseline_env.timestep;
        c_step(&baseline_env);
        c_step(&plain_env);
        EXPECT_EQ_INT(baseline_env.active_agent_count, active_count);
        EXPECT_TRUE(memcmp(baseline_env.observations, plain_env.observations, obs_floats * sizeof(float)) == 0);
        EXPECT_TRUE(memcmp(baseline_env.rewards, plain_env.rewards, active_count * sizeof(float)) == 0);
        EXPECT_TRUE(memcmp(baseline_env.terminals, plain_env.terminals, active_count) == 0);
        EXPECT_TRUE(memcmp(baseline_env.truncations, plain_env.truncations, active_count) == 0);
        EXPECT_TRUE(memcmp(baseline_env.masks, plain_env.masks, active_count) == 0);
        EXPECT_EQ_INT(baseline_env.timestep, plain_env.timestep);
        // c_reset inside c_step respawns every car after the move, so each curve must collapse onto its new car
        int episode_reset = baseline_env.timestep != timestep_before + 1;
        if (episode_reset) {
            EXPECT_TRUE(all_active_curves_collapsed(&baseline_env));
            reset_steps++;
        }
        for (int i = 0; i < active_count; i++) {
            int agent_idx = baseline_env.active_agent_indices[i];
            EXPECT_EQ_INT(agent_idx, plain_env.active_agent_indices[i]);
            const Agent *baseline_agent = &baseline_env.agents[agent_idx];
            const Agent *plain_agent = &plain_env.agents[agent_idx];
            EXPECT_TRUE(agent_motion_identical(baseline_agent, plain_agent));
            EXPECT_TRUE(coefs_all_equal(plain_agent, 0.0f));
            if (episode_reset || !expect_fit[agent_idx]) {
                continue;
            }
            // the curve must start from the state the move started from, not the one it produced
            EXPECT_TRUE(baseline_agent->spline_coefs_x[0] == pre_step_x[agent_idx]);
            EXPECT_TRUE(baseline_agent->spline_coefs_y[0] == pre_step_y[agent_idx]);
            fitted_agent_steps++;
        }
    }
    EXPECT_TRUE(fitted_agent_steps > 0);
    EXPECT_TRUE(reset_steps > 0);
    free_allocated(&baseline_env);
    free_allocated(&plain_env);
    return 0;
}

static int test_rollout_is_identical_with_flag_on_and_off(void) {
    const int ignore = INFRACTION_BEHAVIOR_IGNORE;
    EXPECT_EQ_INT(
        check_rollout_identical(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, ROLLOUT_AGENT_COUNT, ignore, ignore),
        0);
    EXPECT_EQ_INT(
        check_rollout_identical(
            drive_carla_map(),
            SIMULATION_MODE_GIGAFLOW,
            ROLLOUT_AGENT_COUNT,
            INFRACTION_BEHAVIOR_STOP,
            INFRACTION_BEHAVIOR_REMOVE),
        0);
    EXPECT_EQ_INT(check_rollout_identical(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, ignore, ignore), 0);
    return 0;
}

static void run_death_case(int death_case) {
    Drive env = make_baseline_env(0.3f, 1, 1);
    Agent *agent = prepare_single_agent(&env, 0.4f, 8.0f);
    begin_baseline_trajectory(&env, agent);
    move_dynamics(&env, 0, env.active_agent_indices[0]);
    if (death_case == DEATH_NON_FINITE_STATE) {
        agent->sim_vx = NAN;
    } else {
        env.dynamics_model = DYNAMICS_MODEL_SPLINE;
    }
    fit_baseline_trajectory(&env, agent);
}

static int aborts_in_child(int death_case) {
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) {
        struct rlimit no_core_dump = {0, 0};
        if (setrlimit(RLIMIT_CORE, &no_core_dump) != 0 || freopen("/dev/null", "w", stderr) == NULL) {
            _exit(2);
        }
        run_death_case(death_case);
        _exit(0);
    }
    if (pid < 0) {
        perror("fork failed");
        return 0;
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

static int test_fit_asserts_abort_on_broken_invariants(void) {
    EXPECT_TRUE(aborts_in_child(DEATH_NON_FINITE_STATE));
    EXPECT_TRUE(aborts_in_child(DEATH_WRONG_DYNAMICS_MODEL));
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_fit_passes_through_both_step_ends);
    RUN_TEST(test_steady_cruise_fits_a_straight_line);
    RUN_TEST(test_stopped_agent_curve_stays_put);
    RUN_TEST(test_removed_agent_and_disabled_flag_leave_curve_untouched);
    RUN_TEST(test_rollout_is_identical_with_flag_on_and_off);
    RUN_TEST(test_fit_asserts_abort_on_broken_invariants);
    return test_summary(failures);
}
