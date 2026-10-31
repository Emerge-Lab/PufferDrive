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

static unsigned long long drive_with_assist(int assist_mode, float blend_target, int substeps, int *saw_valid_out) {
    Drive env = assist_full_env(DYNAMICS_MODEL_JERK, substeps);
    c_set_assist_mode(&env, assist_mode, blend_target);
    unsigned long long hash = 1469598103934665603ULL;
    *saw_valid_out = 0;
    for (int step = 0; step < ASSIST_TEST_STEP_COUNT; step++) {
        set_actions(&env, step);
        c_set_human_input(&env, 0.8f * sinf(0.2f * (float) step), 1.0f);
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
    return test_summary(failures);
}
