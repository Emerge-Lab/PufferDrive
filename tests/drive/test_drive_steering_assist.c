#include "include/drive_fixture.h"
#include "include/test.h"

#define ASSIST_TEST_DT_SECONDS 0.1f
#define ASSIST_TEST_POLICY_DT_SECONDS 0.3f
#define ASSIST_TEST_MAX_SPEED_MPS 20.0f
#define ASSIST_TEST_SPEED_SCALE 1.5f
#define ASSIST_TEST_AGENT_COUNT 4
#define ASSIST_TEST_SUBSTEPS 18
#define ASSIST_TEST_STEP_COUNT 40
#define EXACT_TOLERANCE 0.0f
#define SPEED_TOLERANCE_MPS 1e-5f
// float32 rounding of weight += clip(...) on a 1/60 s step against the 0.5 s ramp
#define RAMP_BOUND_SLACK 1e-6f
#define REFERENCE_SENTINEL_RAD 123.0f
#define HUMAN_ACTIVE_IDX 0
#define SELECTOR_ROLLING_SPEED_MPS 0.51f
#define SELECTOR_REFUSED_SPEED_MPS 4.0f
#define SELECTOR_SETTLE_STEP_COUNT 30
#define OFF_WINDOW_STEERING_NORM -0.4f
#define HUMAN_INPUT_DIRECT 0
#define HUMAN_INPUT_PEDAL_AT_THRESHOLD 1
#define HUMAN_INPUT_FEET_OFF 2
#define HELD_THROTTLE_NORM 0.3f
#define JUST_RELEASED_NORM 0.0499f
#define LONG_BENCH_SPEED_MPS 10.0f
#define CONTRACT_ACCEL_MIN_MPS2 -9.0f
#define CONTRACT_ACCEL_MAX_MPS2 4.0f
#define CONTRACT_STOP_DECEL_MPS2 3.0f
// any nonzero end-of-step speed: the policy did not choose to stop
#define MOVING_REFERENCE_SPEED_MPS 7.0f

typedef struct {
    Drive env;
    GridMap offgrid_map;
    Agent agent;
    float actions[1][2];
} AssistBench;

// The zeroed GridMap reports every query off-grid, which makes update_agent_z a no-op.
static void assist_bench_init(AssistBench *bench, float speed_mps) {
    memset(bench, 0, sizeof(AssistBench));
    bench->agent = drive_test_agent(0.0f, 0.0f, 0.3f);
    for (int c = 0; c < NUM_REWARD_COEFS; c++) {
        bench->agent.reward_coefs[c] = 1.0f;
    }
    bench->agent.controller = CONTROLLER_HUMAN;
    bench->agent.sim_vx = speed_mps * bench->agent.cos_heading;
    bench->agent.sim_vy = speed_mps * bench->agent.sin_heading;
    update_agent_speed(&bench->agent);
    copy_pose_to_prev(&bench->agent);
    bench->env.agents = &bench->agent;
    bench->env.num_agents = 1;
    bench->env.grid_map = &bench->offgrid_map;
    bench->env.actions = (float *) bench->actions;
    bench->env.action_type = ACTION_TYPE_CONTINUOUS;
    bench->env.dynamics_model = DYNAMICS_MODEL_JERK;
    bench->env.dt = ASSIST_TEST_DT_SECONDS;
    bench->env.base_max_speed_mps = ASSIST_TEST_MAX_SPEED_MPS;
    bench->env.reset_accel_on_stop = 1;
    bench->env.human_grip_mode = GRIP_MODE_KINEMATIC;
    bench->env.human_agent_idx = 0;
}

static Drive assist_full_env(int dynamics_model, int substeps) {
    srand(11);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, ASSIST_TEST_AGENT_COUNT, 0);
    env.max_speed_mps = ASSIST_TEST_MAX_SPEED_MPS;
    env.conditioning_accel_scale = 1.0f;
    env.conditioning_speed_scale = ASSIST_TEST_SPEED_SCALE;
    env.reset_accel_on_stop = 1;
    env.dt = substeps > 1 ? ASSIST_TEST_POLICY_DT_SECONDS : ASSIST_TEST_DT_SECONDS;
    env.action_type = ACTION_TYPE_CONTINUOUS;
    env.dynamics_model = dynamics_model;
    allocate(&env);
    c_reset(&env);
    if (substeps > 1) {
        c_set_substepping(&env, SUBSTEP_MODE_ALL, substeps);
    }
    c_select_human_agent(&env, env.active_agent_indices[HUMAN_ACTIVE_IDX]);
    return env;
}

static void set_actions(Drive *env, int step) {
    float (*actions)[2] = (float (*)[2]) env->actions;
    for (int i = 0; i < env->active_agent_count; i++) {
        actions[i][0] = 0.6f * cosf(0.11f * (float) (step + i));
        actions[i][1] = sinf(0.17f * (float) (step + 3 * i));
    }
}

static Agent *human_agent(Drive *env) {
    return &env->agents[env->active_agent_indices[HUMAN_ACTIVE_IDX]];
}

static unsigned long long hash_motion(unsigned long long hash, const Drive *env) {
    for (int i = 0; i < env->num_total_agents; i++) {
        const Agent *a = &env->agents[i];
        float fields[] = {a->sim_x, a->sim_y, a->sim_heading, a->sim_vx, a->sim_vy, a->yaw_rate,
                          a->accel_long, a->accel_lat, a->jerk_long, a->jerk_lat, a->steering_angle};
        const unsigned char *bytes = (const unsigned char *) fields;
        for (size_t b = 0; b < sizeof(fields); b++) {
            hash ^= bytes[b];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

static void set_human_command(Drive *env, int step, int human_input) {
    float steering_norm = 0.8f * sinf(0.2f * (float) step);
    if (human_input == HUMAN_INPUT_DIRECT) {
        c_set_human_input(env, steering_norm, 1.0f);
        return;
    }
    if (human_input == HUMAN_INPUT_FEET_OFF) {
        c_set_human_pedals(env, steering_norm, JUST_RELEASED_NORM, JUST_RELEASED_NORM, 0.0f, 0);
        return;
    }
    // alternate feet so each pedal alone, held exactly at the threshold, must take over
    int on_throttle = step % 2 == 0;
    float threshold = ASSIST_PEDAL_RELEASED_FRACTION;
    c_set_human_pedals(env, steering_norm, on_throttle ? threshold : 0.0f, on_throttle ? 0.0f : threshold, 0.0f, 0);
}

static unsigned long long drive_with_assist_input(
    int assist_mode, float blend_target, int substeps, int human_input, int *saw_valid_out) {
    Drive env = assist_full_env(DYNAMICS_MODEL_JERK, substeps);
    c_set_assist_mode(&env, assist_mode, blend_target);
    unsigned long long hash = 1469598103934665603ULL;
    *saw_valid_out = 0;
    for (int step = 0; step < ASSIST_TEST_STEP_COUNT; step++) {
        set_actions(&env, step);
        set_human_command(&env, step, human_input);
        for (int s = 0; s < env.substeps_per_policy_step && substeps > 1; s++) {
            c_substep(&env);
            hash = hash_motion(hash, &env);
        }
        c_step(&env);
        hash = hash_motion(hash, &env);
        *saw_valid_out |= env.assist_reference_valid;
    }
    free_allocated(&env);
    return hash;
}

static unsigned long long drive_with_assist(int assist_mode, float blend_target, int substeps, int *saw_valid_out) {
    return drive_with_assist_input(assist_mode, blend_target, substeps, HUMAN_INPUT_DIRECT, saw_valid_out);
}

static int test_integrate_jerk_does_not_write_the_agent(void) {
    AssistBench bench;
    assist_bench_init(&bench, 8.0f);
    bench.agent.accel_long = 1.2f;
    bench.agent.accel_lat = -2.0f;
    bench.agent.steering_angle = 0.1f;
    Agent before;
    memcpy(&before, &bench.agent, sizeof(Agent));
    float jerks[][2] = {{4.0f, 2.0f}, {-15.0f, -4.0f}, {0.0f, 0.0f}, {-4.0f, 4.0f}};
    for (int k = 0; k < 4; k++) {
        for (int phantom = 0; phantom <= 1; phantom++) {
            JerkStep step = integrate_jerk(&bench.env, &bench.agent, jerks[k][0], jerks[k][1], 0.3f, phantom);
            EXPECT_FINITE(step.steering_angle);
            EXPECT_TRUE(memcmp(&before, &bench.agent, sizeof(Agent)) == 0);
        }
    }
    return 0;
}

static int test_jerk_step_obeys_the_named_steering_limits(void) {
    AssistBench bench;
    assist_bench_init(&bench, 3.0f);
    bench.agent.steering_angle = 0.5f;
    bench.agent.accel_lat = 3.9f;
    JerkStep step = integrate_jerk(&bench.env, &bench.agent, 0.0f, 4.0f, 0.3f, 0);
    EXPECT_TRUE(step.steering_angle <= JERK_STEERING_LIMIT_RAD);
    EXPECT_NEAR(step.steering_angle, JERK_STEERING_LIMIT_RAD, EXACT_TOLERANCE);
    bench.agent.steering_angle = 0.0f;
    step = integrate_jerk(&bench.env, &bench.agent, 0.0f, 4.0f, 0.3f, 0);
    EXPECT_NEAR(step.steering_angle, JERK_STEERING_RATE_LIMIT_RAD_PER_S * 0.3f, 1e-7f);
    EXPECT_NEAR(JERK_STEERING_RATE_LIMIT_RAD_PER_S, 0.6f, EXACT_TOLERANCE);
    EXPECT_NEAR(JERK_STEERING_LIMIT_RAD, 0.55f, EXACT_TOLERANCE);
    EXPECT_NEAR(JERK_CURVATURE_MIN_SPEED_MPS, 1.0f, EXACT_TOLERANCE);
    return 0;
}

static int expect_move_matches_prediction(AssistBench *bench, float j_long_action, float j_lat_action, int phantom) {
    bench->actions[0][0] = j_long_action;
    bench->actions[0][1] = j_lat_action;
    bench->agent.controller = CONTROLLER_POLICY;
    bench->agent.phantom_braking_counter = phantom ? 2 : 0;
    float j_long, j_lat;
    decode_jerk_action(&bench->env, 0, &j_long, &j_lat);
    if (phantom) {
        j_long = JERK_LONG[0];
    }
    float accel_lat_before = bench->agent.accel_lat;
    JerkStep predicted = integrate_jerk(&bench->env, &bench->agent, j_long, j_lat, bench->env.dt, phantom);
    move_dynamics(&bench->env, 0, 0);
    EXPECT_NEAR(bench->agent.accel_long, predicted.accel_long, EXACT_TOLERANCE);
    EXPECT_NEAR(bench->agent.accel_lat, predicted.accel_lat, EXACT_TOLERANCE);
    EXPECT_NEAR(bench->agent.jerk_long, predicted.jerk_long, EXACT_TOLERANCE);
    EXPECT_NEAR(bench->agent.jerk_lat, (predicted.accel_lat - accel_lat_before) / bench->env.dt, EXACT_TOLERANCE);
    EXPECT_NEAR(bench->agent.steering_angle, predicted.steering_angle, EXACT_TOLERANCE);
    EXPECT_NEAR(bench->agent.sim_speed_signed, predicted.speed_mps, SPEED_TOLERANCE_MPS);
    return 0;
}

static int test_move_dynamics_applies_exactly_what_integrate_jerk_predicts(void) {
    float actions[][2] = {{1.0f, 0.4f}, {-0.3f, -1.0f}, {0.0f, 0.0f}, {-1.0f, 1.0f}, {0.5f, -0.2f}};
    for (int phantom = 0; phantom <= 1; phantom++) {
        AssistBench bench;
        assist_bench_init(&bench, 6.0f);
        bench.env.is_first_integration = 0; // keeps the phantom counter where the test put it
        for (int step = 0; step < 25; step++) {
            if (expect_move_matches_prediction(&bench, actions[step % 5][0], actions[step % 5][1], phantom)) {
                printf("  at step %d phantom %d\n", step, phantom);
                return 1;
            }
        }
    }
    return 0;
}

static int test_default_assist_is_off_and_commands_exactly_the_driver(void) {
    Drive env = assist_full_env(DYNAMICS_MODEL_JERK, 1);
    EXPECT_EQ_INT(env.assist_mode, ASSIST_MODE_OFF);
    EXPECT_NEAR(env.assist_blend_weight, 0.0f, EXACT_TOLERANCE);
    EXPECT_EQ_INT(env.assist_reference_valid, 0);
    c_set_human_input(&env, 0.37f, 0.0f);
    c_step(&env);
    EXPECT_TRUE(human_agent(&env)->steering_angle == -0.37f * HUMAN_STEERING_LIMIT_RAD);
    free_allocated(&env);
    return 0;
}

// direct input never leaves neutral, so FULL has no forward gear to drive here
static int test_display_nudge_and_zero_blend_leave_the_sim_bit_identical_to_off(void) {
    int substep_counts[] = {1, 6};
    for (int k = 0; k < 2; k++) {
        int saw_valid = 0;
        unsigned long long off = drive_with_assist(ASSIST_MODE_OFF, 0.0f, substep_counts[k], &saw_valid);
        EXPECT_EQ_INT(saw_valid, 0);
        unsigned long long display = drive_with_assist(ASSIST_MODE_DISPLAY, 1.0f, substep_counts[k], &saw_valid);
        EXPECT_EQ_INT(saw_valid, 1);
        unsigned long long nudge = drive_with_assist(ASSIST_MODE_NUDGE, 1.0f, substep_counts[k], &saw_valid);
        EXPECT_EQ_INT(saw_valid, 1);
        unsigned long long full = drive_with_assist(ASSIST_MODE_FULL, 1.0f, substep_counts[k], &saw_valid);
        EXPECT_EQ_INT(saw_valid, 1);
        unsigned long long zero_blend = drive_with_assist(ASSIST_MODE_BLEND, 0.0f, substep_counts[k], &saw_valid);
        EXPECT_EQ_INT(saw_valid, 1);
        unsigned long long half_blend = drive_with_assist(ASSIST_MODE_BLEND, 0.5f, substep_counts[k], &saw_valid);
        EXPECT_TRUE(display == off);
        EXPECT_TRUE(nudge == off);
        EXPECT_TRUE(full == off);
        EXPECT_TRUE(zero_blend == off);
        EXPECT_TRUE(half_blend != off);
    }
    return 0;
}

static int test_reference_is_one_policy_step_of_the_humans_own_action_row(void) {
    int substep_counts[] = {1, ASSIST_TEST_SUBSTEPS};
    for (int k = 0; k < 2; k++) {
        Drive env = assist_full_env(DYNAMICS_MODEL_JERK, substep_counts[k]);
        c_set_assist_mode(&env, ASSIST_MODE_DISPLAY, 0.0f);
        float policy_dt = env.dt;
        for (int step = 0; step < 12; step++) {
            set_actions(&env, step);
            c_set_human_input(&env, -0.3f, 0.5f);
            float j_long, j_lat;
            decode_jerk_action(&env, HUMAN_ACTIVE_IDX, &j_long, &j_lat);
            JerkStep expected = integrate_jerk(&env, human_agent(&env), j_long, j_lat, policy_dt, 0);
            if (substep_counts[k] > 1) {
                c_substep(&env);
            } else {
                c_step(&env);
            }
            EXPECT_EQ_INT(env.assist_reference_valid, 1);
            EXPECT_TRUE(env.assist_reference_steering_rad == expected.steering_angle);
            EXPECT_TRUE(env.assist_reference_accel_mps2 == expected.accel_long);
            for (int s = 1; s < env.substeps_per_policy_step && substep_counts[k] > 1; s++) {
                c_substep(&env);
            }
            if (substep_counts[k] > 1) {
                c_step(&env);
            }
        }
        free_allocated(&env);
    }
    return 0;
}

static int test_reference_keeps_refreshing_while_blend_ramps_out(void) {
    Drive env = assist_full_env(DYNAMICS_MODEL_JERK, 1);
    c_set_assist_mode(&env, ASSIST_MODE_BLEND, 1.0f);
    for (int step = 0; step < 10; step++) {
        set_actions(&env, step);
        c_set_human_input(&env, 0.5f, 1.0f);
        c_step(&env);
    }
    c_set_assist_mode(&env, ASSIST_MODE_OFF, 0.0f);
    int refreshed_count = 0;
    for (int step = 10; step < 20; step++) {
        set_actions(&env, step);
        float j_long, j_lat;
        decode_jerk_action(&env, HUMAN_ACTIVE_IDX, &j_long, &j_lat);
        JerkStep expected = integrate_jerk(&env, human_agent(&env), j_long, j_lat, env.dt, 0);
        float weight_after = fmaxf(0.0f, env.assist_blend_weight - env.dt / ASSIST_RAMP_SECONDS);
        c_step(&env);
        if (weight_after <= RAMP_BOUND_SLACK) {
            continue;
        }
        EXPECT_TRUE(env.assist_reference_steering_rad == expected.steering_angle);
        refreshed_count++;
    }
    EXPECT_TRUE(refreshed_count >= 3);
    EXPECT_NEAR(env.assist_blend_weight, 0.0f, EXACT_TOLERANCE);
    free_allocated(&env);
    return 0;
}

static int test_reference_refreshes_only_on_the_first_substep(void) {
    Drive env = assist_full_env(DYNAMICS_MODEL_JERK, ASSIST_TEST_SUBSTEPS);
    c_set_assist_mode(&env, ASSIST_MODE_DISPLAY, 0.0f);
    float (*actions)[2] = (float (*)[2]) env.actions;
    c_set_human_input(&env, 0.0f, 2.0f);
    for (int step = 0; step < 4; step++) {
        actions[HUMAN_ACTIVE_IDX][0] = 0.5f;
        actions[HUMAN_ACTIVE_IDX][1] = 1.0f;
        c_substep(&env);
        float first_reference = env.assist_reference_steering_rad;
        actions[HUMAN_ACTIVE_IDX][1] = -1.0f;
        for (int s = 1; s < env.substeps_per_policy_step; s++) {
            c_substep(&env);
            EXPECT_TRUE(env.assist_reference_steering_rad == first_reference);
        }
        c_step(&env);
        EXPECT_TRUE(env.assist_reference_steering_rad == first_reference);
    }
    free_allocated(&env);
    return 0;
}

static int test_reference_is_not_computed_while_off(void) {
    Drive env = assist_full_env(DYNAMICS_MODEL_JERK, 6);
    env.assist_reference_steering_rad = REFERENCE_SENTINEL_RAD;
    for (int step = 0; step < 5; step++) {
        set_actions(&env, step);
        for (int s = 0; s < env.substeps_per_policy_step; s++) {
            c_substep(&env);
        }
        c_step(&env);
    }
    EXPECT_EQ_INT(env.assist_reference_valid, 0);
    EXPECT_NEAR(env.assist_reference_steering_rad, REFERENCE_SENTINEL_RAD, EXACT_TOLERANCE);
    free_allocated(&env);
    return 0;
}

static int test_full_blend_steers_exactly_the_reference_after_the_ramp(void) {
    int substep_counts[] = {1, ASSIST_TEST_SUBSTEPS};
    for (int k = 0; k < 2; k++) {
        Drive env = assist_full_env(DYNAMICS_MODEL_JERK, substep_counts[k]);
        c_set_assist_mode(&env, ASSIST_MODE_BLEND, 1.0f);
        float step_dt = env.dt / (float) env.substeps_per_policy_step;
        int ramp_substeps = (int) ceilf(ASSIST_RAMP_SECONDS / step_dt) + 1;
        int substeps_taken = 0;
        for (int step = 0; step < ASSIST_TEST_STEP_COUNT; step++) {
            set_actions(&env, step);
            c_set_human_input(&env, 0.9f, 1.0f);
            int count = substep_counts[k] > 1 ? env.substeps_per_policy_step : 1;
            for (int s = 0; s < count; s++) {
                if (substep_counts[k] > 1) {
                    c_substep(&env);
                } else {
                    c_step(&env);
                }
                substeps_taken++;
                if (substeps_taken < ramp_substeps) {
                    continue;
                }
                EXPECT_NEAR(env.assist_blend_weight, 1.0f, EXACT_TOLERANCE);
                EXPECT_TRUE(human_agent(&env)->steering_angle == env.assist_reference_steering_rad);
            }
            if (substep_counts[k] > 1) {
                c_step(&env);
            }
        }
        EXPECT_TRUE(human_agent(&env)->steering_angle != -0.9f * HUMAN_STEERING_LIMIT_RAD);
        free_allocated(&env);
    }
    return 0;
}

static int expect_weight_step(float before, float after, float target, float step_dt) {
    float bound = step_dt / ASSIST_RAMP_SECONDS + RAMP_BOUND_SLACK;
    EXPECT_TRUE(fabsf(after - before) <= bound);
    EXPECT_TRUE(fabsf(after - target) <= fabsf(before - target));
    EXPECT_TRUE(after >= 0.0f && after <= 1.0f);
    return 0;
}

static int test_mode_switches_never_move_the_weight_faster_than_the_ramp(void) {
    Drive env = assist_full_env(DYNAMICS_MODEL_JERK, ASSIST_TEST_SUBSTEPS);
    float step_dt = env.dt / (float) env.substeps_per_policy_step;
    int modes[] = {ASSIST_MODE_BLEND, ASSIST_MODE_DISPLAY, ASSIST_MODE_BLEND, ASSIST_MODE_BLEND, ASSIST_MODE_OFF,
                   ASSIST_MODE_NUDGE};
    float targets[] = {1.0f, 1.0f, 0.4f, 0.0f, 1.0f, 1.0f};
    int reached_target_count = 0;
    for (int phase = 0; phase < 6; phase++) {
        // switch mid-window, as the menu button does
        for (int step = 0; step < 4; step++) {
            set_actions(&env, step);
            c_set_human_input(&env, -0.6f, 1.0f);
            for (int s = 0; s < env.substeps_per_policy_step; s++) {
                if (step == 0 && s == 7) {
                    c_set_assist_mode(&env, modes[phase], targets[phase]);
                }
                float target = env.assist_mode == ASSIST_MODE_BLEND ? env.assist_blend_target : 0.0f;
                float before = env.assist_blend_weight;
                c_substep(&env);
                if (expect_weight_step(before, env.assist_blend_weight, target, step_dt)) {
                    printf("  phase %d step %d substep %d\n", phase, step, s);
                    return 1;
                }
            }
            c_step(&env);
        }
        float target = modes[phase] == ASSIST_MODE_BLEND ? targets[phase] : 0.0f;
        reached_target_count += env.assist_blend_weight == target;
    }
    EXPECT_EQ_INT(reached_target_count, 6);
    free_allocated(&env);
    return 0;
}

static int test_command_change_within_a_window_is_bounded_by_the_ramp(void) {
    AssistBench bench;
    assist_bench_init(&bench, 10.0f);
    bench.actions[0][0] = 0.0f;
    bench.actions[0][1] = 1.0f;
    bench.env.assist_mode = ASSIST_MODE_BLEND;
    bench.env.assist_blend_target = 1.0f;
    bench.env.human_steering_cmd_norm = -0.5f;
    float substep_dt = ASSIST_TEST_POLICY_DT_SECONDS / ASSIST_TEST_SUBSTEPS;
    bench.env.dt = substep_dt;
    float human_rad = human_steering_angle(&bench.env);
    for (int s = 0; s < ASSIST_TEST_SUBSTEPS; s++) {
        float before = assisted_steering_angle(&bench.env, human_rad);
        advance_assist(&bench.env, 0, &bench.agent, ASSIST_TEST_POLICY_DT_SECONDS, substep_dt, s == 0);
        if (s == 0) {
            continue;
        }
        float after = assisted_steering_angle(&bench.env, human_rad);
        float gap = fabsf(bench.env.assist_reference_steering_rad - human_rad);
        EXPECT_TRUE(fabsf(after - before) <= (substep_dt / ASSIST_RAMP_SECONDS) * gap + RAMP_BOUND_SLACK);
        EXPECT_TRUE(fabsf(after - before) > 0.0f);
    }
    return 0;
}

static int test_blend_endpoints_are_exact(void) {
    AssistBench bench;
    assist_bench_init(&bench, 10.0f);
    bench.env.assist_reference_valid = 1;
    float humans[] = {0.0f, -0.0f, 0.31f, -0.6f, 1e-8f};
    float references[] = {0.2f, -0.55f, 0.0f, 0.33f};
    for (int h = 0; h < 5; h++) {
        for (int r = 0; r < 4; r++) {
            bench.env.assist_reference_steering_rad = references[r];
            bench.env.assist_blend_weight = 0.0f;
            float at_zero = assisted_steering_angle(&bench.env, humans[h]);
            EXPECT_TRUE(memcmp(&at_zero, &humans[h], sizeof(float)) == 0);
            bench.env.assist_blend_weight = 1.0f;
            EXPECT_TRUE(assisted_steering_angle(&bench.env, humans[h]) == references[r]);
            bench.env.assist_blend_weight = 0.25f;
            EXPECT_NEAR(
                assisted_steering_angle(&bench.env, humans[h]), 0.75f * humans[h] + 0.25f * references[r], 1e-7f);
        }
    }
    bench.env.assist_reference_valid = 0;
    bench.env.assist_blend_weight = 1.0f;
    EXPECT_TRUE(assisted_steering_angle(&bench.env, 0.31f) == 0.31f);
    return 0;
}

static int test_setter_rejects_bad_mode_and_target_and_keeps_state(void) {
    AssistBench bench;
    assist_bench_init(&bench, 0.0f);
    EXPECT_EQ_INT(c_set_assist_mode(&bench.env, ASSIST_MODE_BLEND, 0.25f), 0);
    int bad_modes[] = {ASSIST_MODE_OFF - 1, ASSIST_MODE_FULL + 1, 99};
    for (int k = 0; k < 3; k++) {
        EXPECT_EQ_INT(c_set_assist_mode(&bench.env, bad_modes[k], 0.5f), -1);
    }
    float bad_targets[] = {NAN, INFINITY, -INFINITY, -0.001f, 1.001f, -1.0f};
    for (int k = 0; k < 6; k++) {
        EXPECT_EQ_INT(c_set_assist_mode(&bench.env, ASSIST_MODE_DISPLAY, bad_targets[k]), -1);
    }
    EXPECT_EQ_INT(bench.env.assist_mode, ASSIST_MODE_BLEND);
    EXPECT_NEAR(bench.env.assist_blend_target, 0.25f, EXACT_TOLERANCE);
    EXPECT_EQ_INT(c_set_assist_mode(&bench.env, ASSIST_MODE_NUDGE, 1.0f), 0);
    EXPECT_EQ_INT(c_set_assist_mode(&bench.env, ASSIST_MODE_FULL, 1.0f), 0);
    EXPECT_EQ_INT(bench.env.assist_mode, ASSIST_MODE_FULL);
    EXPECT_EQ_INT(c_set_assist_mode(&bench.env, ASSIST_MODE_OFF, 0.0f), 0);
    EXPECT_NEAR(bench.env.assist_blend_weight, 0.0f, EXACT_TOLERANCE);
    return 0;
}

static int test_reference_is_invalid_for_classic_dynamics(void) {
    AssistBench bench;
    assist_bench_init(&bench, 8.0f);
    bench.env.dynamics_model = DYNAMICS_MODEL_CLASSIC;
    bench.env.assist_mode = ASSIST_MODE_BLEND;
    bench.env.assist_blend_target = 1.0f;
    bench.env.assist_blend_weight = 1.0f;
    bench.env.assist_reference_valid = 1;
    bench.env.human_steering_cmd_norm = 0.4f;
    advance_assist(&bench.env, 0, &bench.agent, ASSIST_TEST_DT_SECONDS, ASSIST_TEST_DT_SECONDS, 1);
    EXPECT_EQ_INT(bench.env.assist_reference_valid, 0);
    move_human(&bench.env, 0);
    EXPECT_TRUE(bench.agent.steering_angle == -0.4f * HUMAN_STEERING_LIMIT_RAD);
    return 0;
}

static int test_reference_is_invalid_for_a_removed_or_stopped_human(void) {
    for (int which = 0; which < 2; which++) {
        AssistBench bench;
        assist_bench_init(&bench, 8.0f);
        bench.env.assist_mode = ASSIST_MODE_DISPLAY;
        bench.env.assist_reference_valid = 1;
        bench.agent.removed = which == 0;
        bench.agent.stopped = which == 1;
        advance_assist(&bench.env, 0, &bench.agent, ASSIST_TEST_DT_SECONDS, ASSIST_TEST_DT_SECONDS, 1);
        EXPECT_EQ_INT(bench.env.assist_reference_valid, 0);
    }
    return 0;
}

static int test_selecting_another_car_invalidates_the_reference(void) {
    Drive env = assist_full_env(DYNAMICS_MODEL_JERK, 1);
    c_set_assist_mode(&env, ASSIST_MODE_DISPLAY, 0.0f);
    set_actions(&env, 0);
    c_step(&env);
    EXPECT_EQ_INT(env.assist_reference_valid, 1);
    EXPECT_EQ_INT(c_select_human_agent(&env, env.active_agent_indices[1]), 0);
    EXPECT_EQ_INT(env.assist_reference_valid, 0);
    EXPECT_EQ_INT(c_select_human_agent(&env, -1), 0);
    EXPECT_EQ_INT(env.assist_reference_valid, 0);
    free_allocated(&env);
    return 0;
}

static int test_non_first_substeps_never_compute_a_reference(void) {
    AssistBench bench;
    assist_bench_init(&bench, 8.0f);
    bench.env.assist_mode = ASSIST_MODE_DISPLAY;
    bench.actions[0][1] = 1.0f;
    advance_assist(&bench.env, 0, &bench.agent, ASSIST_TEST_POLICY_DT_SECONDS, ASSIST_TEST_DT_SECONDS, 0);
    EXPECT_EQ_INT(bench.env.assist_reference_valid, 0);
    advance_assist(&bench.env, 0, &bench.agent, ASSIST_TEST_POLICY_DT_SECONDS, ASSIST_TEST_DT_SECONDS, 1);
    EXPECT_EQ_INT(bench.env.assist_reference_valid, 1);
    float j_long, j_lat;
    decode_jerk_action(&bench.env, 0, &j_long, &j_lat);
    JerkStep expected = integrate_jerk(&bench.env, &bench.agent, j_long, j_lat, ASSIST_TEST_POLICY_DT_SECONDS, 0);
    EXPECT_TRUE(bench.env.assist_reference_steering_rad == expected.steering_angle);
    return 0;
}

static int test_full_mode_is_the_fifth_mode(void) {
    EXPECT_EQ_INT(ASSIST_MODE_FULL, 4);
    EXPECT_TRUE(ASSIST_MODE_FULL > ASSIST_MODE_NUDGE);
    return 0;
}

static int test_off_at_zero_weight_drops_the_reference_and_does_nothing_else(void) {
    AssistBench bench;
    assist_bench_init(&bench, 8.0f);
    bench.env.assist_mode = ASSIST_MODE_OFF;
    bench.env.assist_blend_target = 1.0f;
    bench.env.assist_reference_valid = 1;
    bench.env.assist_reference_steering_rad = REFERENCE_SENTINEL_RAD;
    bench.actions[0][1] = 1.0f;
    for (int is_first = 0; is_first < 2; is_first++) {
        advance_assist(&bench.env, 0, &bench.agent, ASSIST_TEST_POLICY_DT_SECONDS, ASSIST_TEST_DT_SECONDS, is_first);
        EXPECT_EQ_INT(bench.env.assist_reference_valid, 0);
        EXPECT_NEAR(bench.env.assist_reference_steering_rad, REFERENCE_SENTINEL_RAD, EXACT_TOLERANCE);
        EXPECT_NEAR(bench.env.assist_blend_weight, 0.0f, EXACT_TOLERANCE);
    }
    return 0;
}

static int test_weight_never_moves_without_a_valid_reference(void) {
    int modes[] = {ASSIST_MODE_BLEND, ASSIST_MODE_DISPLAY, ASSIST_MODE_OFF};
    for (int k = 0; k < 3; k++) {
        AssistBench bench;
        assist_bench_init(&bench, 8.0f);
        bench.env.assist_mode = modes[k];
        bench.env.assist_blend_target = 1.0f;
        bench.env.assist_blend_weight = 0.6f;
        bench.env.assist_reference_valid = 0;
        advance_assist(&bench.env, 0, &bench.agent, ASSIST_TEST_POLICY_DT_SECONDS, ASSIST_TEST_DT_SECONDS, 0);
        EXPECT_EQ_INT(bench.env.assist_reference_valid, 0);
        EXPECT_NEAR(bench.env.assist_blend_weight, 0.0f, EXACT_TOLERANCE);
        advance_assist(&bench.env, 0, &bench.agent, ASSIST_TEST_POLICY_DT_SECONDS, ASSIST_TEST_DT_SECONDS, 0);
        EXPECT_NEAR(bench.env.assist_blend_weight, 0.0f, EXACT_TOLERANCE);
    }
    return 0;
}

static int test_a_stopped_human_drops_a_ramped_blend_to_the_driver(void) {
    AssistBench bench;
    assist_bench_init(&bench, 8.0f);
    bench.env.assist_mode = ASSIST_MODE_BLEND;
    bench.env.assist_blend_target = 1.0f;
    bench.env.assist_blend_weight = 1.0f;
    bench.env.assist_reference_valid = 1;
    bench.agent.stopped = 1;
    advance_assist(&bench.env, 0, &bench.agent, ASSIST_TEST_POLICY_DT_SECONDS, ASSIST_TEST_DT_SECONDS, 1);
    EXPECT_EQ_INT(bench.env.assist_reference_valid, 0);
    EXPECT_NEAR(bench.env.assist_blend_weight, 0.0f, EXACT_TOLERANCE);
    EXPECT_TRUE(assisted_steering_angle(&bench.env, 0.31f) == 0.31f);
    return 0;
}

static int test_first_ramp_step_starts_from_zero_with_a_fresh_reference(void) {
    AssistBench bench;
    assist_bench_init(&bench, 8.0f);
    bench.actions[0][0] = 0.2f;
    bench.actions[0][1] = -1.0f;
    bench.env.assist_mode = ASSIST_MODE_BLEND;
    bench.env.assist_blend_target = 1.0f;
    float j_long, j_lat;
    decode_jerk_action(&bench.env, 0, &j_long, &j_lat);
    JerkStep expected = integrate_jerk(&bench.env, &bench.agent, j_long, j_lat, ASSIST_TEST_POLICY_DT_SECONDS, 0);
    advance_assist(&bench.env, 0, &bench.agent, ASSIST_TEST_POLICY_DT_SECONDS, ASSIST_TEST_DT_SECONDS, 1);
    EXPECT_EQ_INT(bench.env.assist_reference_valid, 1);
    EXPECT_TRUE(bench.env.assist_reference_steering_rad == expected.steering_angle);
    EXPECT_NEAR(bench.env.assist_blend_weight, ASSIST_TEST_DT_SECONDS / ASSIST_RAMP_SECONDS, RAMP_BOUND_SLACK);
    return 0;
}

static void run_off_window(Drive *env, int step) {
    set_actions(env, step);
    c_set_human_input(env, OFF_WINDOW_STEERING_NORM, 1.0f);
    for (int s = 0; s < env->substeps_per_policy_step; s++) {
        c_substep(env);
    }
    c_step(env);
}

static int expect_mode_entered_mid_window_waits_for_a_fresh_reference(int entered_mode) {
    Drive env = assist_full_env(DYNAMICS_MODEL_JERK, ASSIST_TEST_SUBSTEPS);
    float substep_dt = env.dt / (float) env.substeps_per_policy_step;
    c_set_assist_mode(&env, ASSIST_MODE_BLEND, 1.0f);
    for (int step = 0; step < 6; step++) {
        run_off_window(&env, step);
    }
    EXPECT_TRUE(env.assist_blend_weight > 0.0f);
    c_set_assist_mode(&env, ASSIST_MODE_OFF, 0.0f);
    for (int step = 6; step < 16; step++) {
        run_off_window(&env, step);
    }
    EXPECT_EQ_INT(env.assist_reference_valid, 0);
    EXPECT_NEAR(env.assist_blend_weight, 0.0f, EXACT_TOLERANCE);
    set_actions(&env, 16);
    c_substep(&env);
    c_set_assist_mode(&env, entered_mode, 1.0f);
    for (int s = 1; s < env.substeps_per_policy_step; s++) {
        c_substep(&env);
        EXPECT_EQ_INT(env.assist_reference_valid, 0);
        EXPECT_NEAR(env.assist_blend_weight, 0.0f, EXACT_TOLERANCE);
        EXPECT_TRUE(human_agent(&env)->steering_angle == -OFF_WINDOW_STEERING_NORM * HUMAN_STEERING_LIMIT_RAD);
    }
    c_step(&env);
    EXPECT_EQ_INT(env.assist_reference_valid, 0);
    set_actions(&env, 17);
    float j_long, j_lat;
    decode_jerk_action(&env, HUMAN_ACTIVE_IDX, &j_long, &j_lat);
    JerkStep expected = integrate_jerk(&env, human_agent(&env), j_long, j_lat, env.dt, 0);
    c_substep(&env);
    EXPECT_EQ_INT(env.assist_reference_valid, 1);
    EXPECT_TRUE(env.assist_reference_steering_rad == expected.steering_angle);
    float expected_weight = entered_mode == ASSIST_MODE_BLEND ? substep_dt / ASSIST_RAMP_SECONDS : 0.0f;
    EXPECT_NEAR(env.assist_blend_weight, expected_weight, RAMP_BOUND_SLACK);
    free_allocated(&env);
    return 0;
}

static int test_entering_any_assist_mid_window_never_acts_on_a_stale_reference(void) {
    int modes[] = {ASSIST_MODE_BLEND, ASSIST_MODE_DISPLAY, ASSIST_MODE_NUDGE, ASSIST_MODE_FULL};
    for (int k = 0; k < 4; k++) {
        if (expect_mode_entered_mid_window_waits_for_a_fresh_reference(modes[k])) {
            printf("  entered mode %d\n", modes[k]);
            return 1;
        }
    }
    return 0;
}

static int test_reset_clears_the_reference_and_the_weight(void) {
    Drive env = assist_full_env(DYNAMICS_MODEL_JERK, 1);
    c_set_assist_mode(&env, ASSIST_MODE_BLEND, 1.0f);
    for (int step = 0; step < 3; step++) {
        set_actions(&env, step);
        c_step(&env);
    }
    EXPECT_EQ_INT(env.assist_reference_valid, 1);
    EXPECT_TRUE(env.assist_blend_weight > 0.0f);
    c_reset(&env);
    EXPECT_EQ_INT(env.assist_reference_valid, 0);
    EXPECT_NEAR(env.assist_blend_weight, 0.0f, EXACT_TOLERANCE);
    EXPECT_EQ_INT(env.assist_mode, ASSIST_MODE_BLEND);
    EXPECT_EQ_INT(c_select_human_agent(&env, -1), 0);
    env.assist_reference_valid = 1;
    env.assist_blend_weight = 1.0f;
    c_reset(&env);
    EXPECT_EQ_INT(env.assist_reference_valid, 0);
    EXPECT_NEAR(env.assist_blend_weight, 0.0f, EXACT_TOLERANCE);
    free_allocated(&env);
    return 0;
}

// --- Automatic gearbox D/R selector ---

typedef struct {
    Drive env;
    GridMap offgrid_map;
    Agent agent;
} SelectorBench;

static void selector_bench_init(SelectorBench *bench, float speed_mps, int gearbox_mode, int gear) {
    memset(bench, 0, sizeof(SelectorBench));
    bench->agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    bench->agent.reward_coefs[REWARD_COEF_SPEED] = ASSIST_TEST_SPEED_SCALE;
    bench->agent.sim_vx = speed_mps;
    update_agent_speed(&bench->agent);
    copy_pose_to_prev(&bench->agent);
    bench->env.agents = &bench->agent;
    bench->env.num_agents = 1;
    bench->env.grid_map = &bench->offgrid_map;
    bench->env.dt = ASSIST_TEST_DT_SECONDS;
    bench->env.base_max_speed_mps = ASSIST_TEST_MAX_SPEED_MPS;
    bench->env.reset_accel_on_stop = 1;
    bench->env.human_grip_mode = GRIP_MODE_KINEMATIC;
    bench->env.human_gearbox_mode = gearbox_mode;
    bench->env.human_gear = gear;
    bench->env.human_engine_rpm = ENGINE_IDLE_RPM;
    bench->env.human_longitudinal_mode = HUMAN_LONGITUDINAL_POWERTRAIN;
}

static int selector_gear_after(float speed_mps, int gearbox_mode, int gear, int shift_request, float clutch_norm) {
    SelectorBench bench;
    selector_bench_init(&bench, speed_mps, gearbox_mode, gear);
    bench.env.human_clutch_norm = clutch_norm;
    bench.env.human_brake_norm = 1.0f;
    bench.env.human_shift_request = shift_request;
    move_human(&bench.env, 0);
    return bench.env.human_gear;
}

static int test_automatic_left_paddle_selects_reverse_at_a_standstill(void) {
    float speeds[] = {0.0f, AUTO_SELECTOR_MAX_SPEED_MPS, -AUTO_SELECTOR_MAX_SPEED_MPS, 0.2f};
    for (int k = 0; k < 4; k++) {
        EXPECT_EQ_INT(selector_gear_after(speeds[k], GEARBOX_MODE_AUTOMATIC, 1, -1, 0.0f), GEAR_REVERSE);
        EXPECT_EQ_INT(selector_gear_after(speeds[k], GEARBOX_MODE_AUTOMATIC, GEAR_NEUTRAL, -1, 0.0f), GEAR_REVERSE);
        EXPECT_EQ_INT(selector_gear_after(speeds[k], GEARBOX_MODE_AUTOMATIC, GEAR_REVERSE, -1, 0.0f), GEAR_REVERSE);
    }
    return 0;
}

static int test_automatic_right_paddle_leaves_reverse_for_first_at_a_standstill(void) {
    EXPECT_EQ_INT(selector_gear_after(0.0f, GEARBOX_MODE_AUTOMATIC, GEAR_REVERSE, +1, 0.0f), 1);
    EXPECT_EQ_INT(selector_gear_after(-0.3f, GEARBOX_MODE_AUTOMATIC, GEAR_REVERSE, +1, 0.0f), 1);
    EXPECT_EQ_INT(selector_gear_after(0.0f, GEARBOX_MODE_AUTOMATIC, 1, +1, 0.0f), 1);
    return 0;
}

static int test_selector_request_moves_between_reverse_and_first_only(void) {
    SelectorBench bench;
    selector_bench_init(&bench, 0.0f, GEARBOX_MODE_AUTOMATIC, GEAR_REVERSE);
    apply_automatic_selector_request(&bench.env, +1, 0.0f);
    EXPECT_EQ_INT(bench.env.human_gear, 1);
    bench.env.human_gear = 3;
    apply_automatic_selector_request(&bench.env, +1, 0.0f);
    EXPECT_EQ_INT(bench.env.human_gear, 3);
    apply_automatic_selector_request(&bench.env, 0, 0.0f);
    EXPECT_EQ_INT(bench.env.human_gear, 3);
    apply_automatic_selector_request(&bench.env, -1, 0.0f);
    EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    return 0;
}

static int test_automatic_selector_refuses_while_rolling(void) {
    float speeds[] = {SELECTOR_ROLLING_SPEED_MPS, -SELECTOR_ROLLING_SPEED_MPS, SELECTOR_REFUSED_SPEED_MPS};
    for (int k = 0; k < 3; k++) {
        EXPECT_TRUE(selector_gear_after(speeds[k], GEARBOX_MODE_AUTOMATIC, 1, -1, 0.0f) != GEAR_REVERSE);
        EXPECT_EQ_INT(selector_gear_after(speeds[k], GEARBOX_MODE_AUTOMATIC, GEAR_REVERSE, +1, 0.0f), GEAR_REVERSE);
    }
    EXPECT_NEAR(AUTO_SELECTOR_MAX_SPEED_MPS, 0.5f, EXACT_TOLERANCE);
    return 0;
}

static int test_refused_selector_request_is_not_remembered(void) {
    SelectorBench bench;
    selector_bench_init(&bench, SELECTOR_REFUSED_SPEED_MPS, GEARBOX_MODE_AUTOMATIC, 1);
    bench.env.human_brake_norm = 1.0f;
    bench.env.human_shift_request = -1;
    move_human(&bench.env, 0);
    EXPECT_EQ_INT(bench.env.human_shift_request, 0);
    for (int step = 0; step < SELECTOR_SETTLE_STEP_COUNT; step++) {
        move_human(&bench.env, 0);
    }
    EXPECT_NEAR(bench.agent.sim_speed, 0.0f, EXACT_TOLERANCE);
    EXPECT_EQ_INT(bench.env.human_gear, 1);
    return 0;
}

static int test_reverse_selected_by_paddle_drives_backwards_and_stays_in_reverse(void) {
    SelectorBench bench;
    selector_bench_init(&bench, 0.0f, GEARBOX_MODE_AUTOMATIC, 1);
    bench.env.human_shift_request = -1;
    bench.env.human_throttle_norm = 1.0f;
    for (int step = 0; step < SELECTOR_SETTLE_STEP_COUNT; step++) {
        move_human(&bench.env, 0);
        EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    }
    EXPECT_TRUE(bench.agent.sim_speed_signed < -AUTO_SELECTOR_MAX_SPEED_MPS);
    return 0;
}

static int test_manual_gearbox_ignores_the_selector(void) {
    EXPECT_EQ_INT(selector_gear_after(0.0f, GEARBOX_MODE_MANUAL, 1, -1, 0.0f), 1);
    EXPECT_EQ_INT(selector_gear_after(0.0f, GEARBOX_MODE_MANUAL, GEAR_REVERSE, +1, 0.0f), GEAR_REVERSE);
    EXPECT_TRUE(selector_gear_after(0.0f, GEARBOX_MODE_MANUAL, 1, -1, 1.0f) != GEAR_REVERSE);
    return 0;
}


// --- Longitudinal assist ---

#define LONG_REFERENCE_COUNT 8
static const float LONG_REFERENCES_MPS2[LONG_REFERENCE_COUNT] = {-20.0f, -9.0f, -3.3f, 0.0f, 0.7f, 2.5f, 4.0f, 10.0f};
// the reference clipped to the human car's [-9, 4] m/s^2 envelope
static const float LONG_REFERENCES_CLIPPED_MPS2[LONG_REFERENCE_COUNT] = {-9.0f, -9.0f, -3.3f, 0.0f, 0.7f, 2.5f, 4.0f, 4.0f};
#define LONG_HUMAN_COUNT 5
static const float LONG_HUMANS_MPS2[LONG_HUMAN_COUNT] = {0.0f, -0.0f, 1.7f, -8.9f, 1e-8f};

static void long_bench_init(AssistBench *bench, int assist_mode, float weight, float throttle, float brake, int gear) {
    assist_bench_init(bench, LONG_BENCH_SPEED_MPS);
    bench->env.assist_mode = assist_mode;
    bench->env.assist_blend_weight = weight;
    bench->env.assist_reference_valid = 1;
    bench->env.assist_reference_speed_mps = MOVING_REFERENCE_SPEED_MPS;
    bench->env.human_throttle_norm = throttle;
    bench->env.human_brake_norm = brake;
    bench->env.human_gear = gear;
}

static int same_bits(float a, float b) {
    return memcmp(&a, &b, sizeof(float)) == 0;
}

static int test_long_assist_without_a_reference_is_the_driver_bit_for_bit(void) {
    float weights[] = {0.0f, 0.5f, 1.0f};
    int gears[] = {GEAR_REVERSE, GEAR_NEUTRAL, 1, GEARBOX_FORWARD_GEAR_COUNT};
    for (int mode = ASSIST_MODE_OFF; mode <= ASSIST_MODE_FULL; mode++) {
        for (int w = 0; w < 3; w++) {
            for (int g = 0; g < 4; g++) {
                AssistBench bench;
                long_bench_init(&bench, mode, weights[w], 0.0f, 0.0f, gears[g]);
                bench.env.assist_reference_valid = 0;
                bench.env.assist_reference_accel_mps2 = 3.0f;
                for (int h = 0; h < LONG_HUMAN_COUNT; h++) {
                    EXPECT_TRUE(same_bits(assisted_accel(&bench.env, LONG_HUMANS_MPS2[h], LONG_BENCH_SPEED_MPS), LONG_HUMANS_MPS2[h]));
                }
            }
        }
    }
    return 0;
}

static int test_full_with_feet_off_in_a_forward_gear_drives_the_clipped_reference(void) {
    float released[][2] = {{0.0f, 0.0f}, {JUST_RELEASED_NORM, 0.0f}, {0.0f, JUST_RELEASED_NORM},
                           {JUST_RELEASED_NORM, JUST_RELEASED_NORM}};
    float weights[] = {0.0f, 0.4f, 1.0f};
    for (int gear = 1; gear <= GEARBOX_FORWARD_GEAR_COUNT; gear++) {
        for (int p = 0; p < 4; p++) {
            for (int w = 0; w < 3; w++) {
                AssistBench bench;
                long_bench_init(&bench, ASSIST_MODE_FULL, weights[w], released[p][0], released[p][1], gear);
                for (int r = 0; r < LONG_REFERENCE_COUNT; r++) {
                    bench.env.assist_reference_accel_mps2 = LONG_REFERENCES_MPS2[r];
                    for (int h = 0; h < LONG_HUMAN_COUNT; h++) {
                        EXPECT_TRUE(assisted_accel(&bench.env, LONG_HUMANS_MPS2[h], LONG_BENCH_SPEED_MPS) == LONG_REFERENCES_CLIPPED_MPS2[r]);
                    }
                }
            }
        }
    }
    EXPECT_NEAR(ASSIST_PEDAL_RELEASED_FRACTION, 0.05f, EXACT_TOLERANCE);
    EXPECT_NEAR(HUMAN_ACCEL_LIMIT_MPS2[0], CONTRACT_ACCEL_MIN_MPS2, EXACT_TOLERANCE);
    EXPECT_NEAR(HUMAN_ACCEL_LIMIT_MPS2[1], CONTRACT_ACCEL_MAX_MPS2, EXACT_TOLERANCE);
    return 0;
}

static int test_full_hands_back_on_either_pedal_or_outside_a_forward_gear(void) {
    float threshold = ASSIST_PEDAL_RELEASED_FRACTION;
    float pedals_gears[][3] = {{threshold, 0.0f, 1}, {0.0f, threshold, 1}, {1.0f, 0.0f, 3}, {0.0f, 1.0f, 3},
                               {threshold, threshold, 2}, {0.0f, 0.0f, GEAR_NEUTRAL}, {0.0f, 0.0f, GEAR_REVERSE},
                               {1.0f, 0.0f, GEAR_REVERSE}};
    float weights[] = {0.0f, 0.6f, 1.0f};
    for (int k = 0; k < 8; k++) {
        for (int w = 0; w < 3; w++) {
            AssistBench bench;
            long_bench_init(&bench, ASSIST_MODE_FULL, weights[w], pedals_gears[k][0], pedals_gears[k][1],
                            (int) pedals_gears[k][2]);
            for (int r = 0; r < LONG_REFERENCE_COUNT; r++) {
                bench.env.assist_reference_accel_mps2 = LONG_REFERENCES_MPS2[r];
                for (int h = 0; h < LONG_HUMAN_COUNT; h++) {
                    EXPECT_TRUE(same_bits(assisted_accel(&bench.env, LONG_HUMANS_MPS2[h], LONG_BENCH_SPEED_MPS), LONG_HUMANS_MPS2[h]));
                }
            }
        }
    }
    return 0;
}

static int test_blend_shares_the_longitudinal_command_with_the_clipped_reference(void) {
    for (int r = 0; r < LONG_REFERENCE_COUNT; r++) {
        for (int h = 0; h < LONG_HUMAN_COUNT; h++) {
            AssistBench bench;
            long_bench_init(&bench, ASSIST_MODE_BLEND, 0.0f, 0.0f, 0.0f, 1);
            bench.env.assist_reference_accel_mps2 = LONG_REFERENCES_MPS2[r];
            float human = LONG_HUMANS_MPS2[h];
            EXPECT_TRUE(same_bits(assisted_accel(&bench.env, human, LONG_BENCH_SPEED_MPS), human));
            bench.env.assist_blend_weight = 1.0f;
            EXPECT_TRUE(assisted_accel(&bench.env, human, LONG_BENCH_SPEED_MPS) == LONG_REFERENCES_CLIPPED_MPS2[r]);
            bench.env.assist_blend_weight = 0.25f;
            EXPECT_NEAR(assisted_accel(&bench.env, human, LONG_BENCH_SPEED_MPS), 0.75f * human + 0.25f * LONG_REFERENCES_CLIPPED_MPS2[r], 1e-6f);
            bench.env.human_throttle_norm = 1.0f;
            bench.env.human_gear = GEAR_REVERSE;
            EXPECT_NEAR(assisted_accel(&bench.env, human, LONG_BENCH_SPEED_MPS), 0.75f * human + 0.25f * LONG_REFERENCES_CLIPPED_MPS2[r], 1e-6f);
        }
    }
    return 0;
}

static int test_display_and_nudge_leave_the_longitudinal_command_alone_once_ramped_out(void) {
    int modes[] = {ASSIST_MODE_OFF, ASSIST_MODE_DISPLAY, ASSIST_MODE_NUDGE};
    for (int m = 0; m < 3; m++) {
        AssistBench bench;
        long_bench_init(&bench, modes[m], 0.0f, 0.0f, 0.0f, 1);
        for (int r = 0; r < LONG_REFERENCE_COUNT; r++) {
            bench.env.assist_reference_accel_mps2 = LONG_REFERENCES_MPS2[r];
            for (int h = 0; h < LONG_HUMAN_COUNT; h++) {
                EXPECT_TRUE(same_bits(assisted_accel(&bench.env, LONG_HUMANS_MPS2[h], LONG_BENCH_SPEED_MPS), LONG_HUMANS_MPS2[h]));
            }
        }
        // a weight still easing out of BLEND keeps blending, exactly as the steering does
        bench.env.assist_blend_weight = 0.5f;
        bench.env.assist_reference_accel_mps2 = 2.0f;
        EXPECT_NEAR(assisted_accel(&bench.env, -2.0f, LONG_BENCH_SPEED_MPS), 0.0f, 1e-6f);
    }
    return 0;
}

static void long_selector_bench(SelectorBench *bench, float speed_mps, int gearbox_mode, int gear, int assist_mode,
                                float throttle, float brake, float reference_mps2) {
    selector_bench_init(bench, speed_mps, gearbox_mode, gear);
    bench->env.assist_mode = assist_mode;
    bench->env.assist_reference_valid = 1;
    bench->env.assist_reference_accel_mps2 = reference_mps2;
    bench->env.assist_reference_speed_mps = MOVING_REFERENCE_SPEED_MPS;
    bench->env.human_throttle_norm = throttle;
    bench->env.human_brake_norm = brake;
}

static int test_full_with_feet_off_moves_the_car_on_the_reference(void) {
    const float *expected_accels = LONG_REFERENCES_CLIPPED_MPS2;
    int start_gears[] = {GEAR_NEUTRAL, 1, 3};
    for (int g = 0; g < 3; g++) {
        for (int r = 0; r < LONG_REFERENCE_COUNT; r++) {
            SelectorBench bench;
            long_selector_bench(&bench, LONG_BENCH_SPEED_MPS, GEARBOX_MODE_AUTOMATIC, start_gears[g], ASSIST_MODE_FULL,
                                0.0f, 0.0f, LONG_REFERENCES_MPS2[r]);
            move_human(&bench.env, 0);
            EXPECT_TRUE(bench.env.human_gear >= 1);
            EXPECT_TRUE(bench.agent.accel_long == expected_accels[r]);
            EXPECT_NEAR(bench.agent.sim_speed_signed, LONG_BENCH_SPEED_MPS + 0.5f * expected_accels[r] * ASSIST_TEST_DT_SECONDS,
                        SPEED_TOLERANCE_MPS);
        }
    }
    return 0;
}

static int expect_same_motion_as_off(float speed_mps, int gearbox_mode, int gear, int assist_mode, float throttle,
                                     float brake, float weight) {
    SelectorBench assisted;
    SelectorBench off;
    long_selector_bench(&assisted, speed_mps, gearbox_mode, gear, assist_mode, throttle, brake, 3.0f);
    long_selector_bench(&off, speed_mps, gearbox_mode, gear, ASSIST_MODE_OFF, throttle, brake, 3.0f);
    assisted.env.assist_blend_weight = weight;
    off.env.assist_reference_valid = 0;
    for (int step = 0; step < SELECTOR_SETTLE_STEP_COUNT; step++) {
        move_human(&assisted.env, 0);
        move_human(&off.env, 0);
        EXPECT_TRUE(memcmp(&assisted.agent, &off.agent, sizeof(Agent)) == 0);
        EXPECT_EQ_INT(assisted.env.human_gear, off.env.human_gear);
        EXPECT_TRUE(same_bits(assisted.env.human_engine_rpm, off.env.human_engine_rpm));
    }
    return 0;
}

static int test_full_with_a_pedal_or_outside_drive_moves_the_car_exactly_like_off(void) {
    float threshold = ASSIST_PEDAL_RELEASED_FRACTION;
    EXPECT_FALSE(expect_same_motion_as_off(LONG_BENCH_SPEED_MPS, GEARBOX_MODE_AUTOMATIC, 1, ASSIST_MODE_FULL, 0.6f, 0.0f, 0.0f));
    EXPECT_FALSE(expect_same_motion_as_off(LONG_BENCH_SPEED_MPS, GEARBOX_MODE_AUTOMATIC, 1, ASSIST_MODE_FULL, threshold, 0.0f, 0.0f));
    EXPECT_FALSE(expect_same_motion_as_off(LONG_BENCH_SPEED_MPS, GEARBOX_MODE_AUTOMATIC, 2, ASSIST_MODE_FULL, 0.0f, threshold, 0.0f));
    EXPECT_FALSE(expect_same_motion_as_off(-1.0f, GEARBOX_MODE_AUTOMATIC, GEAR_REVERSE, ASSIST_MODE_FULL, 0.0f, 0.0f, 0.0f));
    EXPECT_FALSE(expect_same_motion_as_off(LONG_BENCH_SPEED_MPS, GEARBOX_MODE_MANUAL, GEAR_NEUTRAL, ASSIST_MODE_FULL, 0.0f, 0.0f, 0.0f));
    EXPECT_FALSE(expect_same_motion_as_off(LONG_BENCH_SPEED_MPS, GEARBOX_MODE_AUTOMATIC, 1, ASSIST_MODE_DISPLAY, 0.0f, 0.0f, 0.0f));
    EXPECT_FALSE(expect_same_motion_as_off(LONG_BENCH_SPEED_MPS, GEARBOX_MODE_AUTOMATIC, 1, ASSIST_MODE_NUDGE, 0.0f, 0.0f, 0.0f));
    EXPECT_FALSE(expect_same_motion_as_off(LONG_BENCH_SPEED_MPS, GEARBOX_MODE_AUTOMATIC, 1, ASSIST_MODE_BLEND, 0.0f, 0.0f, 0.0f));
    SelectorBench feet_off;
    long_selector_bench(&feet_off, LONG_BENCH_SPEED_MPS, GEARBOX_MODE_AUTOMATIC, 1, ASSIST_MODE_FULL, 0.0f, 0.0f, 3.0f);
    move_human(&feet_off.env, 0);
    EXPECT_TRUE(feet_off.agent.accel_long == 3.0f);
    return 0;
}

static int test_speed_clip_and_zero_crossing_still_apply_to_the_assisted_command(void) {
    SelectorBench bench;
    long_selector_bench(&bench, 0.2f, GEARBOX_MODE_AUTOMATIC, 1, ASSIST_MODE_FULL, 0.0f, 0.0f, -9.0f);
    move_human(&bench.env, 0);
    EXPECT_TRUE(bench.agent.sim_speed_signed == 0.0f);
    EXPECT_TRUE(bench.agent.accel_long == 0.0f);
    float top_speed_mps = ASSIST_TEST_MAX_SPEED_MPS * ASSIST_TEST_SPEED_SCALE;
    long_selector_bench(&bench, top_speed_mps - 0.01f, GEARBOX_MODE_MANUAL, GEARBOX_FORWARD_GEAR_COUNT,
                        ASSIST_MODE_FULL, 0.0f, 0.0f, 4.0f);
    move_human(&bench.env, 0);
    EXPECT_NEAR(bench.agent.sim_speed_signed, top_speed_mps, SPEED_TOLERANCE_MPS);
    EXPECT_TRUE(bench.agent.accel_long == 4.0f);
    return 0;
}

static int test_full_with_feet_off_differs_from_off_but_a_pedal_at_the_threshold_does_not(void) {
    int substep_counts[] = {1, 6};
    for (int k = 0; k < 2; k++) {
        int saw_valid = 0;
        int n = substep_counts[k];
        unsigned long long off_held = drive_with_assist_input(ASSIST_MODE_OFF, 0.0f, n, HUMAN_INPUT_PEDAL_AT_THRESHOLD, &saw_valid);
        unsigned long long full_held = drive_with_assist_input(ASSIST_MODE_FULL, 1.0f, n, HUMAN_INPUT_PEDAL_AT_THRESHOLD, &saw_valid);
        EXPECT_EQ_INT(saw_valid, 1);
        EXPECT_TRUE(full_held == off_held);
        unsigned long long off_free = drive_with_assist_input(ASSIST_MODE_OFF, 0.0f, n, HUMAN_INPUT_FEET_OFF, &saw_valid);
        unsigned long long full_free = drive_with_assist_input(ASSIST_MODE_FULL, 1.0f, n, HUMAN_INPUT_FEET_OFF, &saw_valid);
        unsigned long long display_free = drive_with_assist_input(ASSIST_MODE_DISPLAY, 1.0f, n, HUMAN_INPUT_FEET_OFF, &saw_valid);
        unsigned long long nudge_free = drive_with_assist_input(ASSIST_MODE_NUDGE, 1.0f, n, HUMAN_INPUT_FEET_OFF, &saw_valid);
        unsigned long long zero_blend_free = drive_with_assist_input(ASSIST_MODE_BLEND, 0.0f, n, HUMAN_INPUT_FEET_OFF, &saw_valid);
        EXPECT_TRUE(full_free != off_free);
        EXPECT_TRUE(display_free == off_free);
        EXPECT_TRUE(nudge_free == off_free);
        EXPECT_TRUE(zero_blend_free == off_free);
    }
    return 0;
}

static int accel_follows_clipped_reference(const Drive *env) {
    const Agent *agent = &env->agents[env->active_agent_indices[HUMAN_ACTIVE_IDX]];
    float clipped = fminf(fmaxf(env->assist_reference_accel_mps2, CONTRACT_ACCEL_MIN_MPS2), CONTRACT_ACCEL_MAX_MPS2);
    int stopped_by_zero_crossing = agent->sim_speed_signed == 0.0f && agent->accel_long == 0.0f;
    return agent->accel_long == clipped || stopped_by_zero_crossing;
}

static int test_full_feet_off_drives_the_reference_accel_but_never_blends_steering(void) {
    Drive env = assist_full_env(DYNAMICS_MODEL_JERK, 1);
    c_set_assist_mode(&env, ASSIST_MODE_FULL, 1.0f);
    int followed_count = 0;
    for (int step = 0; step < ASSIST_TEST_STEP_COUNT; step++) {
        set_actions(&env, step);
        set_human_command(&env, step, HUMAN_INPUT_FEET_OFF);
        c_step(&env);
        EXPECT_EQ_INT(env.assist_reference_valid, 1);
        EXPECT_TRUE(human_agent(&env)->steering_angle == -env.human_steering_cmd_norm * HUMAN_STEERING_LIMIT_RAD);
        EXPECT_TRUE(accel_follows_clipped_reference(&env));
        followed_count += human_agent(&env)->accel_long != 0.0f;
    }
    EXPECT_TRUE(followed_count > 0);
    free_allocated(&env);
    return 0;
}

static int test_blend_at_one_drives_the_reference_accel_after_the_ramp(void) {
    Drive env = assist_full_env(DYNAMICS_MODEL_JERK, 1);
    c_set_assist_mode(&env, ASSIST_MODE_BLEND, 1.0f);
    int ramp_steps = (int) ceilf(ASSIST_RAMP_SECONDS / env.dt) + 1;
    int followed_count = 0;
    for (int step = 0; step < ASSIST_TEST_STEP_COUNT; step++) {
        set_actions(&env, step);
        c_set_human_pedals(&env, 0.4f, 0.8f, 0.0f, 0.0f, 0);
        c_step(&env);
        if (step < ramp_steps) {
            continue;
        }
        EXPECT_NEAR(env.assist_blend_weight, 1.0f, EXACT_TOLERANCE);
        EXPECT_TRUE(accel_follows_clipped_reference(&env));
        followed_count += human_agent(&env)->accel_long != 0.0f;
    }
    EXPECT_TRUE(followed_count > 0);
    free_allocated(&env);
    return 0;
}


#define REST_SPEED_COUNT 3
static const float REST_SPEEDS_MPS[REST_SPEED_COUNT] = {0.0f, -0.0f, -1.0f};

static int test_a_reference_never_brakes_a_car_at_rest_into_reverse(void) {
    for (int k = 0; k < REST_SPEED_COUNT; k++) {
        float speed_mps = REST_SPEEDS_MPS[k];
        for (int r = 0; r < LONG_REFERENCE_COUNT; r++) {
            float held = fmaxf(LONG_REFERENCES_CLIPPED_MPS2[r], 0.0f);
            AssistBench bench;
            long_bench_init(&bench, ASSIST_MODE_FULL, 0.0f, 0.0f, 0.0f, 1);
            bench.env.assist_reference_accel_mps2 = LONG_REFERENCES_MPS2[r];
            EXPECT_TRUE(assisted_accel(&bench.env, -5.0f, speed_mps) == held);
            bench.env.assist_mode = ASSIST_MODE_BLEND;
            bench.env.assist_blend_weight = 1.0f;
            EXPECT_TRUE(assisted_accel(&bench.env, -5.0f, speed_mps) == held);
            bench.env.assist_blend_weight = 0.5f;
            EXPECT_NEAR(assisted_accel(&bench.env, -3.0f, speed_mps), -1.5f + 0.5f * held, 1e-6f);
        }
    }
    return 0;
}

static int test_the_drivers_own_command_at_rest_is_untouched(void) {
    for (int k = 0; k < REST_SPEED_COUNT; k++) {
        AssistBench bench;
        long_bench_init(&bench, ASSIST_MODE_FULL, 0.0f, 0.0f, 1.0f, 1);
        bench.env.assist_reference_accel_mps2 = -4.0f;
        for (int h = 0; h < LONG_HUMAN_COUNT; h++) {
            EXPECT_TRUE(same_bits(assisted_accel(&bench.env, LONG_HUMANS_MPS2[h], REST_SPEEDS_MPS[k]), LONG_HUMANS_MPS2[h]));
        }
        bench.env.assist_mode = ASSIST_MODE_BLEND;
        EXPECT_TRUE(same_bits(assisted_accel(&bench.env, -8.9f, REST_SPEEDS_MPS[k]), -8.9f));
        bench.env.assist_reference_valid = 0;
        bench.env.assist_blend_weight = 1.0f;
        EXPECT_TRUE(same_bits(assisted_accel(&bench.env, -8.9f, REST_SPEEDS_MPS[k]), -8.9f));
    }
    return 0;
}

static int test_a_negative_reference_still_brakes_a_car_rolling_forward(void) {
    float rolling_speeds_mps[] = {1e-6f, 0.2f, LONG_BENCH_SPEED_MPS};
    for (int k = 0; k < 3; k++) {
        AssistBench bench;
        long_bench_init(&bench, ASSIST_MODE_FULL, 0.0f, 0.0f, 0.0f, 1);
        bench.env.assist_reference_accel_mps2 = -20.0f;
        EXPECT_TRUE(assisted_accel(&bench.env, 1.0f, rolling_speeds_mps[k]) == CONTRACT_ACCEL_MIN_MPS2);
        bench.env.assist_mode = ASSIST_MODE_BLEND;
        bench.env.assist_blend_weight = 1.0f;
        EXPECT_TRUE(assisted_accel(&bench.env, 1.0f, rolling_speeds_mps[k]) == CONTRACT_ACCEL_MIN_MPS2);
    }
    return 0;
}

static int expect_held_at_rest(int assist_mode, float reference_mps2) {
    SelectorBench bench;
    long_selector_bench(&bench, 0.0f, GEARBOX_MODE_AUTOMATIC, 1, assist_mode, 0.0f, 0.0f, reference_mps2);
    bench.env.assist_blend_weight = 1.0f;
    for (int step = 0; step < SELECTOR_SETTLE_STEP_COUNT; step++) {
        move_human(&bench.env, 0);
        EXPECT_TRUE(bench.env.human_gear >= 1);
        EXPECT_TRUE(bench.agent.sim_speed_signed == 0.0f);
    }
    return 0;
}

static int test_feet_off_at_rest_with_a_braking_policy_holds_the_car(void) {
    float references[] = {-0.01f, -1.5f, -9.0f, -20.0f};
    for (int r = 0; r < 4; r++) {
        EXPECT_FALSE(expect_held_at_rest(ASSIST_MODE_FULL, references[r]));
        EXPECT_FALSE(expect_held_at_rest(ASSIST_MODE_BLEND, references[r]));
    }
    return 0;
}

static int test_a_braking_policy_never_speeds_up_a_car_rolling_back_in_drive(void) {
    int modes[] = {ASSIST_MODE_FULL, ASSIST_MODE_BLEND};
    for (int m = 0; m < 2; m++) {
        SelectorBench bench;
        long_selector_bench(&bench, -0.3f, GEARBOX_MODE_AUTOMATIC, 1, modes[m], 0.0f, 0.0f, -3.0f);
        bench.env.assist_blend_weight = 1.0f;
        for (int step = 0; step < SELECTOR_SETTLE_STEP_COUNT; step++) {
            move_human(&bench.env, 0);
            EXPECT_TRUE(bench.env.human_gear >= 1);
            EXPECT_TRUE(bench.agent.sim_speed_signed >= -0.3f);
        }
    }
    return 0;
}

static int test_a_positive_reference_still_pulls_away_from_rest(void) {
    int modes[] = {ASSIST_MODE_FULL, ASSIST_MODE_BLEND};
    for (int m = 0; m < 2; m++) {
        SelectorBench bench;
        long_selector_bench(&bench, 0.0f, GEARBOX_MODE_AUTOMATIC, 1, modes[m], 0.0f, 0.0f, 2.0f);
        bench.env.assist_blend_weight = 1.0f;
        move_human(&bench.env, 0);
        EXPECT_TRUE(bench.agent.accel_long == 2.0f);
        EXPECT_NEAR(bench.agent.sim_speed_signed, 0.5f * 2.0f * ASSIST_TEST_DT_SECONDS, SPEED_TOLERANCE_MPS);
    }
    return 0;
}

static int test_braking_to_a_stop_then_staying_stopped(void) {
    SelectorBench bench;
    long_selector_bench(&bench, 0.5f, GEARBOX_MODE_AUTOMATIC, 1, ASSIST_MODE_FULL, 0.0f, 0.0f, -9.0f);
    float previous_speed_mps = bench.agent.sim_speed_signed;
    int stopped = 0;
    for (int step = 0; step < SELECTOR_SETTLE_STEP_COUNT; step++) {
        move_human(&bench.env, 0);
        EXPECT_TRUE(bench.agent.sim_speed_signed >= 0.0f);
        EXPECT_TRUE(bench.agent.sim_speed_signed <= previous_speed_mps);
        previous_speed_mps = bench.agent.sim_speed_signed;
        stopped |= bench.agent.sim_speed_signed == 0.0f;
    }
    EXPECT_TRUE(stopped);
    EXPECT_TRUE(bench.agent.sim_speed_signed == 0.0f);
    return 0;
}

static int expect_braking_policy_never_reverses_the_car(int assist_mode) {
    Drive env = assist_full_env(DYNAMICS_MODEL_JERK, 1);
    c_set_assist_mode(&env, assist_mode, 1.0f);
    float (*actions)[2] = (float (*)[2]) env.actions;
    int saw_rest = 0;
    for (int step = 0; step < ASSIST_TEST_STEP_COUNT; step++) {
        for (int i = 0; i < env.active_agent_count; i++) {
            actions[i][0] = -1.0f;
            actions[i][1] = 0.0f;
        }
        c_set_human_pedals(&env, 0.0f, 0.0f, 0.0f, 0.0f, 0);
        c_step(&env);
        EXPECT_TRUE(human_agent(&env)->sim_speed_signed >= 0.0f);
        saw_rest |= human_agent(&env)->sim_speed_signed == 0.0f && env.assist_reference_valid;
    }
    free_allocated(&env);
    // BLEND ramps in over idle creep, so only FULL is sure to be at rest with a live reference
    EXPECT_TRUE(saw_rest || assist_mode != ASSIST_MODE_FULL);
    return 0;
}

static int test_a_braking_policy_never_reverses_the_human_car(void) {
    EXPECT_FALSE(expect_braking_policy_never_reverses_the_car(ASSIST_MODE_FULL));
    EXPECT_FALSE(expect_braking_policy_never_reverses_the_car(ASSIST_MODE_BLEND));
    return 0;
}


#define FORWARD_SPEED_COUNT 3
static const float FORWARD_SPEEDS_MPS[FORWARD_SPEED_COUNT] = {1e-6f, 0.06f, LONG_BENCH_SPEED_MPS};

static float stop_rule_reference(int r) {
    return fminf(LONG_REFERENCES_CLIPPED_MPS2[r], -CONTRACT_STOP_DECEL_MPS2);
}

static int test_a_policy_that_chose_to_stop_brakes_a_moving_car_at_least_the_stop_decel(void) {
    float stop_speeds[] = {0.0f, -0.0f};
    for (int z = 0; z < 2; z++) {
        for (int k = 0; k < FORWARD_SPEED_COUNT; k++) {
            for (int r = 0; r < LONG_REFERENCE_COUNT; r++) {
                AssistBench bench;
                long_bench_init(&bench, ASSIST_MODE_FULL, 0.0f, 0.0f, 0.0f, 1);
                bench.env.assist_reference_speed_mps = stop_speeds[z];
                bench.env.assist_reference_accel_mps2 = LONG_REFERENCES_MPS2[r];
                EXPECT_TRUE(assisted_accel(&bench.env, 1.0f, FORWARD_SPEEDS_MPS[k]) == stop_rule_reference(r));
                bench.env.assist_mode = ASSIST_MODE_BLEND;
                bench.env.assist_blend_weight = 1.0f;
                EXPECT_TRUE(assisted_accel(&bench.env, 1.0f, FORWARD_SPEEDS_MPS[k]) == stop_rule_reference(r));
                bench.env.assist_blend_weight = 0.5f;
                EXPECT_NEAR(assisted_accel(&bench.env, 1.0f, FORWARD_SPEEDS_MPS[k]), 0.5f + 0.5f * stop_rule_reference(r), 1e-6f);
            }
        }
    }
    EXPECT_NEAR(ASSIST_STOP_DECEL_MPS2, CONTRACT_STOP_DECEL_MPS2, EXACT_TOLERANCE);
    return 0;
}

static int test_the_stop_rule_never_brakes_harder_than_a_stronger_reference(void) {
    float strong_references[] = {-3.0f, -5.5f, -9.0f, -20.0f};
    float expected[] = {-3.0f, -5.5f, -9.0f, -9.0f};
    for (int r = 0; r < 4; r++) {
        AssistBench bench;
        long_bench_init(&bench, ASSIST_MODE_FULL, 0.0f, 0.0f, 0.0f, 1);
        bench.env.assist_reference_speed_mps = 0.0f;
        bench.env.assist_reference_accel_mps2 = strong_references[r];
        EXPECT_TRUE(assisted_accel(&bench.env, 0.0f, LONG_BENCH_SPEED_MPS) == expected[r]);
    }
    return 0;
}

static int test_the_stop_rule_needs_an_exact_zero_reference_speed(void) {
    float near_zero_speeds[] = {1e-6f, -1e-6f, 0.5f, -2.0f};
    for (int z = 0; z < 4; z++) {
        for (int r = 0; r < LONG_REFERENCE_COUNT; r++) {
            AssistBench bench;
            long_bench_init(&bench, ASSIST_MODE_FULL, 0.0f, 0.0f, 0.0f, 1);
            bench.env.assist_reference_speed_mps = near_zero_speeds[z];
            bench.env.assist_reference_accel_mps2 = LONG_REFERENCES_MPS2[r];
            EXPECT_TRUE(assisted_accel(&bench.env, 1.0f, LONG_BENCH_SPEED_MPS) == LONG_REFERENCES_CLIPPED_MPS2[r]);
        }
    }
    return 0;
}

static int test_the_stop_rule_leaves_a_car_at_rest_and_the_driver_alone(void) {
    for (int r = 0; r < LONG_REFERENCE_COUNT; r++) {
        for (int k = 0; k < REST_SPEED_COUNT; k++) {
            AssistBench bench;
            long_bench_init(&bench, ASSIST_MODE_FULL, 0.0f, 0.0f, 0.0f, 1);
            bench.env.assist_reference_speed_mps = 0.0f;
            bench.env.assist_reference_accel_mps2 = LONG_REFERENCES_MPS2[r];
            EXPECT_TRUE(assisted_accel(&bench.env, -5.0f, REST_SPEEDS_MPS[k]) == fmaxf(LONG_REFERENCES_CLIPPED_MPS2[r], 0.0f));
            bench.env.human_brake_norm = 1.0f;
            EXPECT_TRUE(same_bits(assisted_accel(&bench.env, 1.7f, LONG_BENCH_SPEED_MPS), 1.7f));
            bench.env.assist_mode = ASSIST_MODE_BLEND;
            EXPECT_TRUE(same_bits(assisted_accel(&bench.env, 1.7f, LONG_BENCH_SPEED_MPS), 1.7f));
        }
    }
    return 0;
}

static int expect_creep_stops_and_holds(int assist_mode) {
    SelectorBench bench;
    long_selector_bench(&bench, 0.06f, GEARBOX_MODE_AUTOMATIC, 1, assist_mode, 0.0f, 0.0f, 0.0f);
    bench.env.assist_reference_speed_mps = 0.0f;
    bench.env.assist_blend_weight = 1.0f;
    move_human(&bench.env, 0);
    EXPECT_TRUE(bench.agent.sim_speed_signed == 0.0f);
    for (int step = 0; step < SELECTOR_SETTLE_STEP_COUNT; step++) {
        move_human(&bench.env, 0);
        EXPECT_TRUE(bench.agent.sim_speed_signed == 0.0f);
    }
    return 0;
}

static int test_a_creeping_car_whose_policy_chose_to_stop_reaches_zero_and_holds(void) {
    EXPECT_FALSE(expect_creep_stops_and_holds(ASSIST_MODE_FULL));
    EXPECT_FALSE(expect_creep_stops_and_holds(ASSIST_MODE_BLEND));
    return 0;
}

static int test_the_reference_speed_is_the_policy_steps_end_speed(void) {
    float speeds[] = {8.0f, 0.03f, 0.0f};
    for (int k = 0; k < 3; k++) {
        AssistBench bench;
        assist_bench_init(&bench, speeds[k]);
        bench.env.assist_mode = ASSIST_MODE_DISPLAY;
        bench.env.assist_reference_speed_mps = REFERENCE_SENTINEL_RAD;
        bench.actions[0][0] = -1.0f;
        advance_assist(&bench.env, 0, &bench.agent, ASSIST_TEST_POLICY_DT_SECONDS, ASSIST_TEST_DT_SECONDS, 1);
        float j_long, j_lat;
        decode_jerk_action(&bench.env, 0, &j_long, &j_lat);
        JerkStep expected = integrate_jerk(&bench.env, &bench.agent, j_long, j_lat, ASSIST_TEST_POLICY_DT_SECONDS, 0);
        EXPECT_TRUE(bench.env.assist_reference_speed_mps == expected.speed_mps);
    }
    return 0;
}

static int expect_full_sim_creep_comes_to_rest(int assist_mode) {
    Drive env = assist_full_env(DYNAMICS_MODEL_JERK, 1);
    c_set_assist_mode(&env, assist_mode, 1.0f);
    float (*actions)[2] = (float (*)[2]) env.actions;
    int rest_step_count = 0;
    for (int step = 0; step < 2 * ASSIST_TEST_STEP_COUNT; step++) {
        for (int i = 0; i < env.active_agent_count; i++) {
            actions[i][0] = -1.0f;
            actions[i][1] = 0.0f;
        }
        c_set_human_pedals(&env, 0.0f, step < 3 ? 0.06f : 0.0f, 0.0f, 0.0f, 0);
        c_step(&env);
        EXPECT_TRUE(human_agent(&env)->sim_speed_signed >= 0.0f);
        rest_step_count = human_agent(&env)->sim_speed_signed == 0.0f ? rest_step_count + 1 : 0;
    }
    free_allocated(&env);
    EXPECT_TRUE(rest_step_count >= ASSIST_TEST_STEP_COUNT);
    return 0;
}

static int test_a_braking_policy_brings_a_creeping_human_car_to_rest(void) {
    EXPECT_FALSE(expect_full_sim_creep_comes_to_rest(ASSIST_MODE_FULL));
    EXPECT_FALSE(expect_full_sim_creep_comes_to_rest(ASSIST_MODE_BLEND));
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_integrate_jerk_does_not_write_the_agent);
    RUN_TEST(test_jerk_step_obeys_the_named_steering_limits);
    RUN_TEST(test_move_dynamics_applies_exactly_what_integrate_jerk_predicts);
    RUN_TEST(test_default_assist_is_off_and_commands_exactly_the_driver);
    RUN_TEST(test_display_nudge_and_zero_blend_leave_the_sim_bit_identical_to_off);
    RUN_TEST(test_reference_is_one_policy_step_of_the_humans_own_action_row);
    RUN_TEST(test_reference_keeps_refreshing_while_blend_ramps_out);
    RUN_TEST(test_reference_refreshes_only_on_the_first_substep);
    RUN_TEST(test_reference_is_not_computed_while_off);
    RUN_TEST(test_full_blend_steers_exactly_the_reference_after_the_ramp);
    RUN_TEST(test_mode_switches_never_move_the_weight_faster_than_the_ramp);
    RUN_TEST(test_command_change_within_a_window_is_bounded_by_the_ramp);
    RUN_TEST(test_blend_endpoints_are_exact);
    RUN_TEST(test_setter_rejects_bad_mode_and_target_and_keeps_state);
    RUN_TEST(test_reference_is_invalid_for_classic_dynamics);
    RUN_TEST(test_reference_is_invalid_for_a_removed_or_stopped_human);
    RUN_TEST(test_selecting_another_car_invalidates_the_reference);
    RUN_TEST(test_non_first_substeps_never_compute_a_reference);
    RUN_TEST(test_full_mode_is_the_fifth_mode);
    RUN_TEST(test_off_at_zero_weight_drops_the_reference_and_does_nothing_else);
    RUN_TEST(test_weight_never_moves_without_a_valid_reference);
    RUN_TEST(test_a_stopped_human_drops_a_ramped_blend_to_the_driver);
    RUN_TEST(test_first_ramp_step_starts_from_zero_with_a_fresh_reference);
    RUN_TEST(test_entering_any_assist_mid_window_never_acts_on_a_stale_reference);
    RUN_TEST(test_reset_clears_the_reference_and_the_weight);
    RUN_TEST(test_automatic_left_paddle_selects_reverse_at_a_standstill);
    RUN_TEST(test_automatic_right_paddle_leaves_reverse_for_first_at_a_standstill);
    RUN_TEST(test_selector_request_moves_between_reverse_and_first_only);
    RUN_TEST(test_automatic_selector_refuses_while_rolling);
    RUN_TEST(test_refused_selector_request_is_not_remembered);
    RUN_TEST(test_reverse_selected_by_paddle_drives_backwards_and_stays_in_reverse);
    RUN_TEST(test_manual_gearbox_ignores_the_selector);
    RUN_TEST(test_long_assist_without_a_reference_is_the_driver_bit_for_bit);
    RUN_TEST(test_full_with_feet_off_in_a_forward_gear_drives_the_clipped_reference);
    RUN_TEST(test_full_hands_back_on_either_pedal_or_outside_a_forward_gear);
    RUN_TEST(test_blend_shares_the_longitudinal_command_with_the_clipped_reference);
    RUN_TEST(test_display_and_nudge_leave_the_longitudinal_command_alone_once_ramped_out);
    RUN_TEST(test_full_with_feet_off_moves_the_car_on_the_reference);
    RUN_TEST(test_full_with_a_pedal_or_outside_drive_moves_the_car_exactly_like_off);
    RUN_TEST(test_speed_clip_and_zero_crossing_still_apply_to_the_assisted_command);
    RUN_TEST(test_full_with_feet_off_differs_from_off_but_a_pedal_at_the_threshold_does_not);
    RUN_TEST(test_full_feet_off_drives_the_reference_accel_but_never_blends_steering);
    RUN_TEST(test_blend_at_one_drives_the_reference_accel_after_the_ramp);
    RUN_TEST(test_a_reference_never_brakes_a_car_at_rest_into_reverse);
    RUN_TEST(test_the_drivers_own_command_at_rest_is_untouched);
    RUN_TEST(test_a_negative_reference_still_brakes_a_car_rolling_forward);
    RUN_TEST(test_feet_off_at_rest_with_a_braking_policy_holds_the_car);
    RUN_TEST(test_a_braking_policy_never_speeds_up_a_car_rolling_back_in_drive);
    RUN_TEST(test_a_positive_reference_still_pulls_away_from_rest);
    RUN_TEST(test_braking_to_a_stop_then_staying_stopped);
    RUN_TEST(test_a_braking_policy_never_reverses_the_human_car);
    RUN_TEST(test_a_policy_that_chose_to_stop_brakes_a_moving_car_at_least_the_stop_decel);
    RUN_TEST(test_the_stop_rule_never_brakes_harder_than_a_stronger_reference);
    RUN_TEST(test_the_stop_rule_needs_an_exact_zero_reference_speed);
    RUN_TEST(test_the_stop_rule_leaves_a_car_at_rest_and_the_driver_alone);
    RUN_TEST(test_a_creeping_car_whose_policy_chose_to_stop_reaches_zero_and_holds);
    RUN_TEST(test_the_reference_speed_is_the_policy_steps_end_speed);
    RUN_TEST(test_a_braking_policy_brings_a_creeping_human_car_to_rest);
    return test_summary(failures);
}
