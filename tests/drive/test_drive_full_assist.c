#include "include/drive_fixture.h"
#include "include/test.h"

#define FULL_TEST_DT_SECONDS 0.1f
#define FULL_TEST_POLICY_DT_SECONDS 0.3f
#define FULL_TEST_MAX_SPEED_MPS 20.0f
#define FULL_TEST_SPEED_SCALE 1.5f
#define FULL_TEST_AGENT_COUNT 4
#define FULL_TEST_SUBSTEPS 6
#define FULL_TEST_STEP_COUNT 30
#define FULL_TEST_SEED 23
#define HUMAN_ACTIVE_IDX 0
#define OTHER_ACTIVE_IDX 1
#define EXACT_TOLERANCE 0.0f
#define SPEED_TOLERANCE_MPS 1e-5f
#define CRUISE_SPEED_MPS 8.0f
#define REVERSING_SPEED_MPS -1.0f
#define WHEEL_STEERING_NORM 0.3f
#define OTHER_WHEEL_STEERING_NORM -0.7f
#define PRESSED_PEDAL_NORM 0.5f
#define POLICY_ACCEL_ACTION 0.6f
#define POLICY_TURN_ACTION -0.8f
#define MOVING_REFERENCE_SPEED_MPS 7.0f
#define FAR_LEFT_SET_STEERING_RAD 2.0f
#define SET_STEERING_RAD 0.05f
#define SET_STEERING_SETTLE_STEPS 20
#define PREVIEW_STEP_COUNT 3
#define PREVIEW_PIECE_COUNTS 4
#define ROAD_DROPOUT_KEPT_SLOTS 8
#define UPSHIFT_TEST_SPEED_MPS 15.0f
#define STYLE_WINDOW_LOW_FRACTION 0.25f
#define STYLE_WINDOW_HIGH_FRACTION 0.75f
#define REALISTIC_GRIP_SPEED_MPS 15.0f

static const int PREVIEW_PIECES[PREVIEW_PIECE_COUNTS] = {1, 3, 18, 30};
static const int FULL_DYNAMICS_MODES[] = {FULL_ASSIST_DYNAMICS_JERK_SPEED, FULL_ASSIST_DYNAMICS_HUMAN,
                                          FULL_ASSIST_DYNAMICS_JERK, FULL_ASSIST_DYNAMICS_SET_STEERING};
#define FULL_DYNAMICS_MODE_COUNT 4

typedef struct {
    Drive env;
    GridMap offgrid_map;
    Agent agent;
    float actions[1][2];
} FullBench;

// The zeroed GridMap reports every query off-grid, which makes update_agent_z a no-op.
static void full_bench_init(FullBench *bench, float speed_mps, int full_assist_dynamics) {
    memset(bench, 0, sizeof(FullBench));
    bench->agent = drive_test_agent(0.0f, 0.0f, 0.2f);
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
    bench->env.dt = FULL_TEST_DT_SECONDS;
    bench->env.base_max_speed_mps = FULL_TEST_MAX_SPEED_MPS;
    bench->env.human_max_speed_mps = FULL_TEST_MAX_SPEED_MPS;
    bench->env.conditioning_speed_scale = FULL_TEST_SPEED_SCALE;
    bench->env.reset_accel_on_stop = 1;
    bench->env.human_grip_mode = GRIP_MODE_KINEMATIC;
    bench->env.human_agent_idx = 0;
    bench->env.human_gearbox_mode = GEARBOX_MODE_AUTOMATIC;
    bench->env.human_gear = 1;
    bench->env.human_engine_rpm = ENGINE_IDLE_RPM;
    bench->env.assist_mode = ASSIST_MODE_FULL;
    bench->env.assist_reference_valid = 1;
    bench->env.assist_reference_speed_mps = MOVING_REFERENCE_SPEED_MPS;
    bench->env.full_assist_dynamics = full_assist_dynamics;
    bench->actions[0][0] = POLICY_ACCEL_ACTION;
    bench->actions[0][1] = POLICY_TURN_ACTION;
    c_set_human_pedals(&bench->env, WHEEL_STEERING_NORM, 0.0f, 0.0f, 0.0f, 0);
}

static int same_bits(float a, float b) {
    return memcmp(&a, &b, sizeof(float)) == 0;
}

static Drive full_env(int dynamics_model, unsigned int seed) {
    srand(seed);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, FULL_TEST_AGENT_COUNT, 0);
    env.max_speed_mps = FULL_TEST_MAX_SPEED_MPS;
    env.conditioning_accel_scale = 1.0f;
    env.conditioning_speed_scale = FULL_TEST_SPEED_SCALE;
    env.reset_accel_on_stop = 1;
    env.dt = FULL_TEST_POLICY_DT_SECONDS;
    env.action_type = ACTION_TYPE_CONTINUOUS;
    env.dynamics_model = dynamics_model;
    // road dropout makes the observation writers draw from rng_state
    env.obs_slots_lane_kept = ROAD_DROPOUT_KEPT_SLOTS;
    env.obs_slots_boundary_kept = ROAD_DROPOUT_KEPT_SLOTS;
    env.reward_randomization = 1;
    allocate(&env);
    c_reset(&env);
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
    return &env->agents[env->human_agent_idx];
}

// Two envs own different heap buffers, so their agents compare equal once the pointers are masked.
static int same_agent_state(const Agent *a, const Agent *b) {
    Agent masked_a = *a;
    Agent masked_b = *b;
    Agent *masked[2] = {&masked_a, &masked_b};
    for (int k = 0; k < 2; k++) {
        masked[k]->log_trajectory_x = masked[k]->log_trajectory_y = masked[k]->log_trajectory_z = NULL;
        masked[k]->log_heading = masked[k]->log_velocity_x = masked[k]->log_velocity_y = NULL;
        masked[k]->log_length = masked[k]->log_width = masked[k]->log_height = NULL;
        masked[k]->log_valid = NULL;
        masked[k]->route = NULL;
    }
    if (memcmp(&masked_a, &masked_b, sizeof(Agent)) != 0) {
        return 0;
    }
    return a->route_length == 0 || memcmp(a->route, b->route, a->route_length * sizeof(int)) == 0;
}

static size_t observation_bytes(Drive *env) {
    return (size_t) compute_observation_size(env) * env->active_agent_count * sizeof(float);
}


static int test_begin_preview_refuses_when_there_is_nothing_to_imagine(void) {
    Drive env = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
    int human_agent_idx = env.human_agent_idx;
    EXPECT_EQ_INT(c_begin_preview(&env), HUMAN_ACTIVE_IDX);
    EXPECT_EQ_INT(c_begin_preview(&env), -1);
    EXPECT_EQ_INT(env.preview_slot, HUMAN_ACTIVE_IDX);
    c_end_preview(&env);
    EXPECT_EQ_INT(env.preview_slot, -1);

    human_agent(&env)->stopped = 1;
    EXPECT_EQ_INT(c_begin_preview(&env), -1);
    human_agent(&env)->stopped = 0;
    human_agent(&env)->removed = 1;
    EXPECT_EQ_INT(c_begin_preview(&env), -1);
    human_agent(&env)->removed = 0;

    env.dynamics_model = DYNAMICS_MODEL_CLASSIC;
    EXPECT_EQ_INT(c_begin_preview(&env), -1);
    env.dynamics_model = DYNAMICS_MODEL_JERK;

    c_select_human_agent(&env, -1);
    EXPECT_EQ_INT(c_begin_preview(&env), -1);
    EXPECT_EQ_INT(env.preview_slot, -1);

    c_select_human_agent(&env, env.active_agent_indices[OTHER_ACTIVE_IDX]);
    EXPECT_EQ_INT(c_begin_preview(&env), OTHER_ACTIVE_IDX);
    c_end_preview(&env);
    EXPECT_TRUE(human_agent_idx != env.human_agent_idx);
    free_allocated(&env);
    return 0;
}

static int test_preview_step_refuses_without_a_preview_or_pieces(void) {
    Drive env = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
    float steering_rad[1] = {0.0f};
    EXPECT_EQ_INT(c_preview_step(&env, 1, steering_rad), -1);
    EXPECT_EQ_INT(c_begin_preview(&env), HUMAN_ACTIVE_IDX);
    EXPECT_EQ_INT(c_preview_step(&env, 0, steering_rad), -1);
    EXPECT_EQ_INT(c_preview_step(&env, -1, steering_rad), -1);
    c_end_preview(&env);
    c_end_preview(&env);
    EXPECT_EQ_INT(env.preview_slot, -1);
    free_allocated(&env);
    return 0;
}

static int test_preview_step_integrates_the_slot_action_in_pieces_with_the_jerk_model(void) {
    for (int p = 0; p < PREVIEW_PIECE_COUNTS; p++) {
        int piece_count = PREVIEW_PIECES[p];
        Drive env = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
        set_actions(&env, 0);
        for (int step = 0; step < 4; step++) {
            c_step(&env);
        }
        int slot = c_begin_preview(&env);
        EXPECT_EQ_INT(slot, HUMAN_ACTIVE_IDX);
        Agent imagined = *human_agent(&env);
        float piece_dt = env.dt / (float) piece_count;
        float steering_rad[piece_count];
        for (int step = 0; step < PREVIEW_STEP_COUNT; step++) {
            float (*actions)[2] = (float (*)[2]) env.actions;
            actions[slot][0] = 0.4f * (float) (step + 1) - 0.9f;
            actions[slot][1] = 0.5f - 0.3f * (float) step;
            float j_long, j_lat;
            decode_jerk_action(&env, slot, &j_long, &j_lat);
            EXPECT_EQ_INT(c_preview_step(&env, piece_count, steering_rad), 0);
            for (int s = 0; s < piece_count; s++) {
                apply_jerk_step(&env, &imagined, j_long, j_lat, piece_dt, 0);
                EXPECT_TRUE(same_bits(steering_rad[s], imagined.steering_angle));
            }
            EXPECT_TRUE(memcmp(&imagined, human_agent(&env), sizeof(Agent)) == 0);
        }
        c_end_preview(&env);
        free_allocated(&env);
    }
    return 0;
}

static int test_preview_step_rebuilds_only_the_human_observation_row(void) {
    Drive env = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
    set_actions(&env, 0);
    c_step(&env);
    int obs_per_agent = compute_observation_size(&env);
    size_t bytes = observation_bytes(&env);
    float *real_obs = (float *) malloc(bytes);
    float *imagined_obs = (float *) malloc(bytes);
    memcpy(real_obs, env.observations, bytes);
    int slot = c_begin_preview(&env);
    float steering_rad[FULL_TEST_SUBSTEPS];
    Rng rng_before_step = env.rng_state;
    EXPECT_EQ_INT(c_preview_step(&env, FULL_TEST_SUBSTEPS, steering_rad), 0);
    memcpy(imagined_obs, env.observations, bytes);
    for (int i = 0; i < env.active_agent_count; i++) {
        int row_equal = memcmp(&imagined_obs[i * obs_per_agent], &real_obs[i * obs_per_agent],
                               obs_per_agent * sizeof(float)) == 0;
        EXPECT_TRUE(row_equal == (i != slot));
    }
    env.rng_state = rng_before_step;
    PartnerScanEntry partner_scan[env.num_agents];
    fill_partner_scan(&env, partner_scan);
    write_agent_observation(&env, slot, partner_scan);
    EXPECT_TRUE(memcmp(env.observations, imagined_obs, bytes) == 0);
    c_end_preview(&env);
    EXPECT_TRUE(memcmp(env.observations, real_obs, bytes) == 0);
    free(real_obs);
    free(imagined_obs);
    free_allocated(&env);
    return 0;
}

static int run_preview(Drive *env) {
    float (*actions)[2] = (float (*)[2]) env->actions;
    int slot = c_begin_preview(env);
    if (slot < 0) {
        return slot;
    }
    float saved_action[2] = {actions[slot][0], actions[slot][1]};
    float steering_rad[FULL_TEST_SUBSTEPS];
    for (int step = 0; step < PREVIEW_STEP_COUNT; step++) {
        actions[slot][0] = -1.0f + 0.7f * (float) step;
        actions[slot][1] = 1.0f - 0.9f * (float) step;
        c_preview_step(env, FULL_TEST_SUBSTEPS, steering_rad);
    }
    c_end_preview(env);
    // the caller owns the action buffer, exactly as the demo loop rewrites it after a look-ahead
    actions[slot][0] = saved_action[0];
    actions[slot][1] = saved_action[1];
    return slot;
}

static int test_end_preview_restores_agents_observations_and_rng_exactly(void) {
    Drive env = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
    set_actions(&env, 0);
    c_step(&env);
    size_t agent_bytes = (size_t) env.num_total_agents * sizeof(Agent);
    size_t bytes = observation_bytes(&env);
    Agent *agents_before = (Agent *) malloc(agent_bytes);
    float *obs_before = (float *) malloc(bytes);
    memcpy(agents_before, env.agents, agent_bytes);
    memcpy(obs_before, env.observations, bytes);
    Rng rng_before = env.rng_state;
    EXPECT_EQ_INT(run_preview(&env), HUMAN_ACTIVE_IDX);
    EXPECT_TRUE(memcmp(agents_before, env.agents, agent_bytes) == 0);
    EXPECT_TRUE(memcmp(obs_before, env.observations, bytes) == 0);
    EXPECT_TRUE(memcmp(&rng_before, &env.rng_state, sizeof(Rng)) == 0);
    free(agents_before);
    free(obs_before);
    free_allocated(&env);
    return 0;
}

static int expect_previews_leave_trajectories_identical(int assist_mode, int substeps) {
    Drive with_previews = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
    Drive without_previews = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
    Drive *envs[2] = {&with_previews, &without_previews};
    for (int e = 0; e < 2; e++) {
        c_set_assist_mode(envs[e], assist_mode, 1.0f);
        if (substeps > 1) {
            c_set_substepping(envs[e], SUBSTEP_MODE_ALL, substeps);
        }
    }
    int preview_count = 0;
    for (int step = 0; step < FULL_TEST_STEP_COUNT; step++) {
        for (int e = 0; e < 2; e++) {
            set_actions(envs[e], step);
            c_set_human_pedals(envs[e], 0.5f * sinf(0.3f * (float) step), 0.0f, 0.0f, 0.0f, 0);
        }
        preview_count += run_preview(&with_previews) >= 0;
        for (int s = 0; s < substeps && substeps > 1; s++) {
            c_substep(&with_previews);
            c_substep(&without_previews);
        }
        c_step(&with_previews);
        c_step(&without_previews);
        for (int a = 0; a < with_previews.num_total_agents; a++) {
            EXPECT_TRUE(same_agent_state(&with_previews.agents[a], &without_previews.agents[a]));
        }
        EXPECT_TRUE(memcmp(with_previews.observations, without_previews.observations, observation_bytes(&with_previews)) == 0);
        EXPECT_TRUE(memcmp(&with_previews.rng_state, &without_previews.rng_state, sizeof(Rng)) == 0);
        EXPECT_TRUE(memcmp(with_previews.rewards, without_previews.rewards, with_previews.active_agent_count * sizeof(float)) == 0);
    }
    EXPECT_TRUE(preview_count > 0);
    free_allocated(&with_previews);
    free_allocated(&without_previews);
    return 0;
}

static int test_previews_never_change_later_trajectories(void) {
    EXPECT_FALSE(expect_previews_leave_trajectories_identical(ASSIST_MODE_OFF, 1));
    EXPECT_FALSE(expect_previews_leave_trajectories_identical(ASSIST_MODE_FULL, 1));
    EXPECT_FALSE(expect_previews_leave_trajectories_identical(ASSIST_MODE_FULL, FULL_TEST_SUBSTEPS));
    return 0;
}


// Verbatim copy of the pre-split compute_observations loop, kept as the oracle.
static void reference_compute_observations(Drive *env) {
    int obs_per_agent = compute_observation_size(env);
    memset(env->observations, 0, obs_per_agent * env->active_agent_count * sizeof(float));
    float (*obs_matrix)[obs_per_agent] = (float (*)[obs_per_agent]) env->observations;
    PartnerScanEntry partner_scan[env->num_agents];
    for (int j = 0; j < env->num_agents; j++) {
        int index = (j < env->active_agent_count) ? env->active_agent_indices[j]
                                                  : env->static_agent_indices[j - env->active_agent_count];
        Agent *agent = &env->agents[index];
        partner_scan[j] = (PartnerScanEntry) {agent->sim_x, agent->sim_y, agent->sim_z, index};
    }
    for (int i = 0; i < env->active_agent_count; i++) {
        float *obs = &obs_matrix[i][0];
        int agent_idx = env->active_agent_indices[i];
        Agent *ego = &env->agents[agent_idx];
        int partner_count = 0;
        int lane_count = 0;
        int boundary_count = 0;
        int traffic_control_count = 0;
        int obs_idx = 0;

        obs_idx = write_ego_obs(env, ego, obs, obs_idx);
        obs_idx = write_reward_target_obs(env, ego, obs, obs_idx);
        obs_idx = write_partner_obs(env, ego, agent_idx, partner_scan, obs, obs_idx, &partner_count);
        obs_idx = write_road_obs(env, ego, obs, obs_idx, &lane_count, &boundary_count);
        obs_idx = write_traffic_control_obs(env, ego, obs, obs_idx, &traffic_control_count);
        obs[obs_idx++] = (float) lane_count;
        obs[obs_idx++] = (float) boundary_count;
        obs[obs_idx++] = (float) partner_count;
        obs[obs_idx++] = (float) traffic_control_count;
    }
}

static int test_split_observations_match_the_pre_split_loop_for_every_agent(void) {
    Drive env = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
    size_t bytes = observation_bytes(&env);
    float *expected_obs = (float *) malloc(bytes);
    for (int step = 0; step < FULL_TEST_STEP_COUNT; step++) {
        set_actions(&env, step);
        c_step(&env);
        Rng rng_before = env.rng_state;
        reference_compute_observations(&env);
        memcpy(expected_obs, env.observations, bytes);
        Rng rng_after_reference = env.rng_state;
        env.rng_state = rng_before;
        compute_observations(&env);
        EXPECT_TRUE(memcmp(expected_obs, env.observations, bytes) == 0);
        EXPECT_TRUE(memcmp(&rng_after_reference, &env.rng_state, sizeof(Rng)) == 0);
    }
    free(expected_obs);
    free_allocated(&env);
    return 0;
}

static int test_apply_jerk_step_writes_exactly_the_integrated_step(void) {
    float jerks[][2] = {{4.0f, 2.0f}, {-15.0f, -4.0f}, {0.0f, 0.0f}, {-4.0f, 4.0f}};
    float dts[] = {FULL_TEST_DT_SECONDS, FULL_TEST_POLICY_DT_SECONDS / (float) FULL_TEST_SUBSTEPS};
    for (int k = 0; k < 4; k++) {
        for (int d = 0; d < 2; d++) {
            for (int phantom = 0; phantom <= 1; phantom++) {
                FullBench bench;
                full_bench_init(&bench, CRUISE_SPEED_MPS, FULL_ASSIST_DYNAMICS_JERK);
                bench.agent.accel_long = 1.1f;
                bench.agent.accel_lat = -0.7f;
                bench.agent.steering_angle = 0.08f;
                Agent expected = bench.agent;
                JerkStep step = integrate_jerk(&bench.env, &expected, jerks[k][0], jerks[k][1], dts[d], phantom);
                integrate_bicycle_arc(&expected, step.signed_curvature, step.speed_mps, step.speed_before_mps, dts[d]);
                expected.jerk_long = step.jerk_long;
                expected.jerk_lat = (step.accel_lat - expected.accel_lat) / dts[d];
                expected.accel_long = step.accel_long;
                expected.accel_lat = step.accel_lat;
                expected.steering_angle = step.steering_angle;
                apply_jerk_step(&bench.env, &bench.agent, jerks[k][0], jerks[k][1], dts[d], phantom);
                EXPECT_TRUE(memcmp(&expected, &bench.agent, sizeof(Agent)) == 0);
            }
        }
    }
    return 0;
}


static int test_full_assist_dynamics_setter_validates_and_keeps_state(void) {
    FullBench bench;
    full_bench_init(&bench, 0.0f, FULL_ASSIST_DYNAMICS_HUMAN);
    for (int m = 0; m < FULL_DYNAMICS_MODE_COUNT; m++) {
        EXPECT_EQ_INT(c_set_full_assist_dynamics(&bench.env, FULL_DYNAMICS_MODES[m]), 0);
        EXPECT_EQ_INT(bench.env.full_assist_dynamics, FULL_DYNAMICS_MODES[m]);
    }
    int bad_modes[] = {FULL_ASSIST_DYNAMICS_JERK_SPEED - 1, FULL_ASSIST_DYNAMICS_SET_STEERING + 1, 99};
    for (int b = 0; b < 3; b++) {
        EXPECT_EQ_INT(c_set_full_assist_dynamics(&bench.env, bad_modes[b]), -1);
        EXPECT_EQ_INT(bench.env.full_assist_dynamics, FULL_ASSIST_DYNAMICS_SET_STEERING);
    }
    Drive fresh = {0};
    EXPECT_EQ_INT(fresh.full_assist_dynamics, FULL_ASSIST_DYNAMICS_JERK_SPEED);
    return 0;
}

static int test_assist_steering_setter_rejects_a_non_finite_valid_angle(void) {
    FullBench bench;
    full_bench_init(&bench, 0.0f, FULL_ASSIST_DYNAMICS_SET_STEERING);
    EXPECT_EQ_INT(c_set_assist_steering(&bench.env, 1, SET_STEERING_RAD), 0);
    EXPECT_EQ_INT(bench.env.assist_set_steering_valid, 1);
    EXPECT_TRUE(bench.env.assist_set_steering_rad == SET_STEERING_RAD);
    EXPECT_EQ_INT(c_set_assist_steering(&bench.env, 1, NAN), -1);
    EXPECT_EQ_INT(c_set_assist_steering(&bench.env, 1, INFINITY), -1);
    EXPECT_EQ_INT(c_set_assist_steering(&bench.env, 1, -INFINITY), -1);
    EXPECT_EQ_INT(bench.env.assist_set_steering_valid, 1);
    EXPECT_TRUE(bench.env.assist_set_steering_rad == SET_STEERING_RAD);
    EXPECT_EQ_INT(c_set_assist_steering(&bench.env, 0, 0.0f), 0);
    EXPECT_EQ_INT(bench.env.assist_set_steering_valid, 0);
    return 0;
}

// Speed from the jerk model; the arc from the given steering angle, with the realistic grip cap.
static Agent expected_jerk_speed_move(FullBench *bench, float steering_angle_rad) {
    Agent expected = bench->agent;
    copy_pose_to_prev(&expected);
    float j_long, j_lat;
    decode_jerk_action(&bench->env, 0, &j_long, &j_lat);
    JerkStep step = integrate_jerk(&bench->env, &bench->agent, j_long, j_lat, bench->env.dt, 0);
    float speed_old = bench->agent.sim_speed_signed;
    float speed_new = fminf(step.speed_mps, bench->env.human_max_speed_mps);
    float signed_curvature = tanf(steering_angle_rad) / expected.wheelbase;
    float accel_lat_demand = speed_new * speed_new * signed_curvature;
    if (bench->env.human_grip_mode == GRIP_MODE_REALISTIC && fabsf(accel_lat_demand) > HUMAN_TIRE_GRIP_MPS2) {
        signed_curvature = copysignf(HUMAN_TIRE_GRIP_MPS2 / (speed_new * speed_new), signed_curvature);
    }
    integrate_bicycle_arc(&expected, signed_curvature, speed_new, speed_old, bench->env.dt);
    float accel_lat_new = speed_new * speed_new * signed_curvature;
    expected.jerk_long = step.jerk_long;
    expected.jerk_lat = (accel_lat_new - bench->agent.accel_lat) / bench->env.dt;
    expected.accel_long = step.accel_long;
    expected.accel_lat = accel_lat_new;
    expected.steering_angle = steering_angle_rad;
    return expected;
}

static Agent expected_jerk_move(FullBench *bench) {
    Agent expected = bench->agent;
    copy_pose_to_prev(&expected);
    float j_long, j_lat;
    decode_jerk_action(&bench->env, 0, &j_long, &j_lat);
    apply_jerk_step(&bench->env, &expected, j_long, j_lat, bench->env.dt, 0);
    return expected;
}

static int test_jerk_speed_drives_speed_on_the_jerk_model_and_steers_by_the_wheel(void) {
    int grip_modes[] = {GRIP_MODE_KINEMATIC, GRIP_MODE_REALISTIC};
    float speeds_mps[] = {CRUISE_SPEED_MPS, REALISTIC_GRIP_SPEED_MPS};
    for (int g = 0; g < 2; g++) {
        for (int k = 0; k < 2; k++) {
            FullBench bench;
            full_bench_init(&bench, speeds_mps[k], FULL_ASSIST_DYNAMICS_JERK_SPEED);
            bench.env.human_grip_mode = grip_modes[g];
            for (int step = 0; step < FULL_TEST_STEP_COUNT; step++) {
                Agent expected = expected_jerk_speed_move(&bench, -WHEEL_STEERING_NORM * HUMAN_STEERING_LIMIT_RAD);
                move_human(&bench.env, 0, 0);
                EXPECT_TRUE(memcmp(&expected, &bench.agent, sizeof(Agent)) == 0);
            }
        }
    }
    return 0;
}

static int test_jerk_speed_obeys_your_own_speed_cap(void) {
    FullBench bench;
    full_bench_init(&bench, FULL_TEST_MAX_SPEED_MPS - 0.01f, FULL_ASSIST_DYNAMICS_JERK_SPEED);
    bench.agent.accel_long = ACCEL_LONG_LIMIT[1];
    bench.actions[0][0] = 1.0f;
    move_human(&bench.env, 0, 0);
    EXPECT_NEAR(bench.agent.sim_speed_signed, FULL_TEST_MAX_SPEED_MPS, SPEED_TOLERANCE_MPS);
    return 0;
}

static int test_jerk_drives_speed_and_steering_and_ignores_the_wheel(void) {
    float wheel_norms[] = {WHEEL_STEERING_NORM, OTHER_WHEEL_STEERING_NORM};
    for (int w = 0; w < 2; w++) {
        FullBench bench;
        full_bench_init(&bench, CRUISE_SPEED_MPS, FULL_ASSIST_DYNAMICS_JERK);
        c_set_human_pedals(&bench.env, wheel_norms[w], 0.0f, 0.0f, 0.0f, 0);
        for (int step = 0; step < FULL_TEST_STEP_COUNT; step++) {
            Agent expected = expected_jerk_move(&bench);
            move_human(&bench.env, 0, 0);
            EXPECT_TRUE(memcmp(&expected, &bench.agent, sizeof(Agent)) == 0);
        }
    }
    return 0;
}

static int test_set_steering_steers_by_the_jerk_model_until_an_angle_arrives(void) {
    FullBench set_bench;
    FullBench jerk_bench;
    full_bench_init(&set_bench, CRUISE_SPEED_MPS, FULL_ASSIST_DYNAMICS_SET_STEERING);
    full_bench_init(&jerk_bench, CRUISE_SPEED_MPS, FULL_ASSIST_DYNAMICS_JERK);
    for (int step = 0; step < FULL_TEST_STEP_COUNT; step++) {
        move_human(&set_bench.env, 0, 0);
        move_human(&jerk_bench.env, 0, 0);
        EXPECT_TRUE(memcmp(&set_bench.agent, &jerk_bench.agent, sizeof(Agent)) == 0);
    }
    return 0;
}

static int test_set_steering_moves_toward_the_set_angle_at_the_jerk_rate_and_ignores_the_wheel(void) {
    float wheel_norms[] = {WHEEL_STEERING_NORM, OTHER_WHEEL_STEERING_NORM};
    float max_change_rad = JERK_STEERING_RATE_LIMIT_RAD_PER_S * FULL_TEST_DT_SECONDS;
    for (int w = 0; w < 2; w++) {
        FullBench bench;
        full_bench_init(&bench, CRUISE_SPEED_MPS, FULL_ASSIST_DYNAMICS_SET_STEERING);
        c_set_human_pedals(&bench.env, wheel_norms[w], 0.0f, 0.0f, 0.0f, 0);
        EXPECT_EQ_INT(c_set_assist_steering(&bench.env, 1, SET_STEERING_RAD), 0);
        for (int step = 0; step < SET_STEERING_SETTLE_STEPS; step++) {
            float previous_rad = bench.agent.steering_angle;
            float change_rad = fminf(fmaxf(SET_STEERING_RAD - previous_rad, -max_change_rad), max_change_rad);
            Agent expected = expected_jerk_speed_move(&bench, previous_rad + change_rad);
            move_human(&bench.env, 0, 0);
            EXPECT_TRUE(memcmp(&expected, &bench.agent, sizeof(Agent)) == 0);
        }
        EXPECT_TRUE(bench.agent.steering_angle == SET_STEERING_RAD);

        EXPECT_EQ_INT(c_set_assist_steering(&bench.env, 1, -SET_STEERING_RAD), 0);
        move_human(&bench.env, 0, 0);
        EXPECT_NEAR(bench.agent.steering_angle, SET_STEERING_RAD - max_change_rad, 1e-7f);
    }
    return 0;
}

static int test_set_steering_is_clipped_to_the_jerk_steering_limit(void) {
    float set_angles_rad[] = {FAR_LEFT_SET_STEERING_RAD, -FAR_LEFT_SET_STEERING_RAD};
    for (int k = 0; k < 2; k++) {
        FullBench bench;
        full_bench_init(&bench, CRUISE_SPEED_MPS, FULL_ASSIST_DYNAMICS_SET_STEERING);
        c_set_assist_steering(&bench.env, 1, set_angles_rad[k]);
        for (int step = 0; step < SET_STEERING_SETTLE_STEPS; step++) {
            move_human(&bench.env, 0, 0);
            EXPECT_TRUE(fabsf(bench.agent.steering_angle) <= JERK_STEERING_LIMIT_RAD);
        }
        EXPECT_TRUE(bench.agent.steering_angle == copysignf(JERK_STEERING_LIMIT_RAD, set_angles_rad[k]));
    }
    return 0;
}

static int test_a_pressed_pedal_hands_every_full_dynamics_back_to_the_driver(void) {
    float pedals[][2] = {{PRESSED_PEDAL_NORM, 0.0f}, {0.0f, PRESSED_PEDAL_NORM},
                         {ASSIST_PEDAL_RELEASED_FRACTION, 0.0f}, {0.0f, ASSIST_PEDAL_RELEASED_FRACTION}};
    for (int m = 0; m < FULL_DYNAMICS_MODE_COUNT; m++) {
        for (int p = 0; p < 4; p++) {
            FullBench full;
            FullBench off;
            full_bench_init(&full, CRUISE_SPEED_MPS, FULL_DYNAMICS_MODES[m]);
            full_bench_init(&off, CRUISE_SPEED_MPS, FULL_DYNAMICS_MODES[m]);
            off.env.assist_mode = ASSIST_MODE_OFF;
            c_set_assist_steering(&full.env, 1, SET_STEERING_RAD);
            for (int step = 0; step < FULL_TEST_STEP_COUNT; step++) {
                c_set_human_pedals(&full.env, WHEEL_STEERING_NORM, pedals[p][0], pedals[p][1], 0.0f, 0);
                c_set_human_pedals(&off.env, WHEEL_STEERING_NORM, pedals[p][0], pedals[p][1], 0.0f, 0);
                move_human(&full.env, 0, 0);
                move_human(&off.env, 0, 0);
                EXPECT_TRUE(memcmp(&full.agent, &off.agent, sizeof(Agent)) == 0);
                EXPECT_EQ_INT(full.env.human_gear, off.env.human_gear);
                EXPECT_TRUE(same_bits(full.env.human_engine_rpm, off.env.human_engine_rpm));
            }
        }
    }
    return 0;
}

static int test_jerk_paths_still_run_the_powertrain(void) {
    int jerk_modes[] = {FULL_ASSIST_DYNAMICS_JERK_SPEED, FULL_ASSIST_DYNAMICS_JERK, FULL_ASSIST_DYNAMICS_SET_STEERING};
    for (int m = 0; m < 3; m++) {
        FullBench bench;
        full_bench_init(&bench, UPSHIFT_TEST_SPEED_MPS, jerk_modes[m]);
        float rpm_before = bench.env.human_engine_rpm;
        move_human(&bench.env, 0, 0);
        EXPECT_TRUE(bench.env.human_gear > 1);
        EXPECT_TRUE(bench.env.human_engine_rpm != rpm_before);
    }
    return 0;
}

static int test_set_steering_validity_resets_on_select_and_reset(void) {
    Drive env = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
    c_set_assist_steering(&env, 1, SET_STEERING_RAD);
    c_select_human_agent(&env, env.active_agent_indices[OTHER_ACTIVE_IDX]);
    EXPECT_EQ_INT(env.assist_set_steering_valid, 0);
    c_set_assist_steering(&env, 1, SET_STEERING_RAD);
    c_reset(&env);
    EXPECT_EQ_INT(env.assist_set_steering_valid, 0);
    free_allocated(&env);
    return 0;
}


static void reverse_bench_init(FullBench *bench, float speed_mps, int gear, float reference_speed_mps) {
    full_bench_init(bench, speed_mps, FULL_ASSIST_DYNAMICS_HUMAN);
    bench->env.human_gear = gear;
    bench->env.assist_reference_speed_mps = reference_speed_mps;
    bench->env.assist_reference_accel_mps2 = reference_speed_mps < 0.0f ? -1.0f : 1.0f;
    bench->env.reset_accel_on_stop = 1;
}

static int test_full_selects_reverse_when_the_policy_step_ends_moving_backward(void) {
    FullBench bench;
    reverse_bench_init(&bench, 0.0f, 1, -MOVING_REFERENCE_SPEED_MPS);
    move_human(&bench.env, 0, 0);
    EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    EXPECT_TRUE(bench.agent.sim_speed_signed < 0.0f);

    reverse_bench_init(&bench, 0.0f, GEAR_REVERSE, MOVING_REFERENCE_SPEED_MPS);
    move_human(&bench.env, 0, 0);
    EXPECT_EQ_INT(bench.env.human_gear, 1);
    EXPECT_TRUE(bench.agent.sim_speed_signed > 0.0f);

    reverse_bench_init(&bench, REVERSING_SPEED_MPS, GEAR_REVERSE, -MOVING_REFERENCE_SPEED_MPS);
    move_human(&bench.env, 0, 0);
    EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    return 0;
}

static int test_full_reverse_selector_holds_while_rolling_or_without_the_policy(void) {
    FullBench bench;
    reverse_bench_init(&bench, CRUISE_SPEED_MPS, 1, -MOVING_REFERENCE_SPEED_MPS);
    move_human(&bench.env, 0, 0);
    EXPECT_TRUE(bench.env.human_gear >= 1);

    reverse_bench_init(&bench, 0.0f, 1, -MOVING_REFERENCE_SPEED_MPS);
    bench.env.human_gearbox_mode = GEARBOX_MODE_MANUAL;
    move_human(&bench.env, 0, 0);
    EXPECT_EQ_INT(bench.env.human_gear, 1);

    int other_modes[] = {ASSIST_MODE_OFF, ASSIST_MODE_DISPLAY, ASSIST_MODE_BLEND, ASSIST_MODE_NUDGE};
    for (int m = 0; m < 4; m++) {
        reverse_bench_init(&bench, 0.0f, 1, -MOVING_REFERENCE_SPEED_MPS);
        bench.env.assist_mode = other_modes[m];
        move_human(&bench.env, 0, 0);
        EXPECT_TRUE(bench.env.human_gear >= 1);
    }

    reverse_bench_init(&bench, 0.0f, 1, -MOVING_REFERENCE_SPEED_MPS);
    c_set_human_pedals(&bench.env, 0.0f, 0.0f, PRESSED_PEDAL_NORM, 0.0f, 0);
    move_human(&bench.env, 0, 0);
    EXPECT_TRUE(bench.env.human_gear >= 1);

    reverse_bench_init(&bench, 0.0f, 1, -MOVING_REFERENCE_SPEED_MPS);
    bench.env.assist_reference_valid = 0;
    move_human(&bench.env, 0, 0);
    EXPECT_TRUE(bench.env.human_gear >= 1);
    return 0;
}

static int test_a_driver_paddle_wins_over_the_automatic_reverse_request(void) {
    FullBench bench;
    reverse_bench_init(&bench, 0.0f, GEAR_REVERSE, -MOVING_REFERENCE_SPEED_MPS);
    c_set_human_pedals(&bench.env, 0.0f, 0.0f, 0.0f, 0.0f, 1);
    move_human(&bench.env, 0, 0);
    EXPECT_EQ_INT(bench.env.human_gear, 1);
    return 0;
}


static void style_window(Drive *env, float *min_values, float *max_values) {
    for (int c = 0; c < NUM_REWARD_COEFS; c++) {
        float limit_min, limit_max;
        reward_coef_limits(env, c, &limit_min, &limit_max);
        float range = limit_max - limit_min;
        min_values[c] = limit_min + STYLE_WINDOW_LOW_FRACTION * range;
        max_values[c] = limit_min + STYLE_WINDOW_HIGH_FRACTION * range;
    }
}

static float expected_style_centre(Drive *env, int c, float min_value, float max_value) {
    const RewardBound *bounds = env->reward_log_sampling ? REWARD_BOUNDS_LOG : REWARD_BOUNDS;
    if (bounds[c].log_scale && min_value > 0.0f) {
        return sqrtf(min_value * max_value);
    }
    return 0.5f * (min_value + max_value);
}

static int expect_styled_coefs(Drive *env, const float *own_coefs, int expect_style) {
    const Agent *agent = &env->agents[env->human_agent_idx];
    for (int c = 0; c < NUM_REWARD_COEFS; c++) {
        int use_style = expect_style && reward_coef_reads_bounds(env, c);
        float expected = use_style ? env->human_style_coefs[c] : own_coefs[c];
        if (!same_bits(agent->reward_coefs[c], expected)) {
            printf("  coefficient %d is %f, expected %f\n", c, agent->reward_coefs[c], expected);
            return 1;
        }
    }
    return 0;
}

static int test_style_setter_validates_the_window_and_keeps_state(void) {
    Drive env = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
    float min_values[NUM_REWARD_COEFS], max_values[NUM_REWARD_COEFS];
    float limit_min, limit_max;
    reward_coef_limits(&env, REWARD_COEF_COLLISION, &limit_min, &limit_max);
    float bad_values[] = {NAN, INFINITY, limit_min - 1.0f, limit_max + 1.0f};
    for (int b = 0; b < 4; b++) {
        for (int side = 0; side < 2; side++) {
            style_window(&env, min_values, max_values);
            float *edited = side == 0 ? min_values : max_values;
            edited[REWARD_COEF_COLLISION] = bad_values[b];
            EXPECT_EQ_INT(c_set_human_driving_style(&env, 1, min_values, max_values), -1);
            EXPECT_EQ_INT(env.human_style_active, 0);
        }
    }
    style_window(&env, min_values, max_values);
    min_values[REWARD_COEF_COLLISION] = max_values[REWARD_COEF_COLLISION] + 0.1f;
    EXPECT_EQ_INT(c_set_human_driving_style(&env, 1, min_values, max_values), -1);
    EXPECT_EQ_INT(env.human_style_active, 0);
    free_allocated(&env);
    return 0;
}

static int test_style_is_the_window_centre_geometric_on_log_scale(void) {
    for (int log_sampling = 0; log_sampling <= 1; log_sampling++) {
        Drive env = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
        env.reward_log_sampling = log_sampling;
        float min_values[NUM_REWARD_COEFS], max_values[NUM_REWARD_COEFS];
        style_window(&env, min_values, max_values);
        EXPECT_EQ_INT(c_set_human_driving_style(&env, 1, min_values, max_values), 0);
        EXPECT_EQ_INT(env.human_style_active, 1);
        int saw_geometric = 0;
        for (int c = 0; c < NUM_REWARD_COEFS; c++) {
            float expected = expected_style_centre(&env, c, min_values[c], max_values[c]);
            EXPECT_TRUE(same_bits(env.human_style_coefs[c], expected));
            saw_geometric |= expected != 0.5f * (min_values[c] + max_values[c]);
        }
        EXPECT_EQ_INT(saw_geometric, log_sampling);
        free_allocated(&env);
    }
    return 0;
}

static int test_style_applies_only_under_full_and_only_to_shaped_coefficients(void) {
    Drive env = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
    float own_coefs[NUM_REWARD_COEFS];
    memcpy(own_coefs, human_agent(&env)->reward_coefs, sizeof(own_coefs));
    EXPECT_TRUE(memcmp(own_coefs, env.human_own_coefs, sizeof(own_coefs)) == 0);
    float min_values[NUM_REWARD_COEFS], max_values[NUM_REWARD_COEFS];
    style_window(&env, min_values, max_values);

    EXPECT_EQ_INT(c_set_human_driving_style(&env, 1, min_values, max_values), 0);
    EXPECT_FALSE(expect_styled_coefs(&env, own_coefs, 0));
    EXPECT_EQ_INT(c_set_assist_mode(&env, ASSIST_MODE_FULL, 1.0f), 0);
    EXPECT_FALSE(expect_styled_coefs(&env, own_coefs, 1));
    EXPECT_TRUE(human_agent(&env)->reward_coefs[REWARD_COEF_COLLISION] != own_coefs[REWARD_COEF_COLLISION]);
    int other_modes[] = {ASSIST_MODE_OFF, ASSIST_MODE_DISPLAY, ASSIST_MODE_BLEND, ASSIST_MODE_NUDGE};
    for (int m = 0; m < 4; m++) {
        EXPECT_EQ_INT(c_set_assist_mode(&env, other_modes[m], 0.5f), 0);
        EXPECT_FALSE(expect_styled_coefs(&env, own_coefs, 0));
    }
    EXPECT_EQ_INT(c_set_assist_mode(&env, ASSIST_MODE_FULL, 1.0f), 0);
    EXPECT_EQ_INT(c_set_human_driving_style(&env, 0, min_values, max_values), 0);
    EXPECT_EQ_INT(env.human_style_active, 0);
    EXPECT_FALSE(expect_styled_coefs(&env, own_coefs, 0));
    free_allocated(&env);
    return 0;
}

static int test_capability_and_unrandomized_goal_speed_are_never_styled(void) {
    Drive env = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
    int never_styled[] = {REWARD_COEF_THROTTLE, REWARD_COEF_STEER, REWARD_COEF_ACC, REWARD_COEF_VELOCITY,
                          REWARD_COEF_TIMESTEP};
    for (int k = 0; k < 5; k++) {
        EXPECT_EQ_INT(reward_coef_reads_bounds(&env, never_styled[k]), 0);
    }
    EXPECT_EQ_INT(reward_coef_reads_bounds(&env, REWARD_COEF_GOAL_SPEED), env.goal_speed_randomization);
    env.goal_speed_randomization = 0;
    float own_coefs[NUM_REWARD_COEFS];
    memcpy(own_coefs, env.human_own_coefs, sizeof(own_coefs));
    float min_values[NUM_REWARD_COEFS], max_values[NUM_REWARD_COEFS];
    style_window(&env, min_values, max_values);
    c_set_assist_mode(&env, ASSIST_MODE_FULL, 1.0f);
    EXPECT_EQ_INT(c_set_human_driving_style(&env, 1, min_values, max_values), 0);
    EXPECT_TRUE(same_bits(human_agent(&env)->reward_coefs[REWARD_COEF_GOAL_SPEED], own_coefs[REWARD_COEF_GOAL_SPEED]));
    for (int k = 0; k < 5; k++) {
        EXPECT_TRUE(same_bits(human_agent(&env)->reward_coefs[never_styled[k]], own_coefs[never_styled[k]]));
    }
    free_allocated(&env);
    return 0;
}

static int test_max_speed_writes_your_own_speed_coefficient_not_the_style(void) {
    Drive env = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
    float min_values[NUM_REWARD_COEFS], max_values[NUM_REWARD_COEFS];
    style_window(&env, min_values, max_values);
    c_set_assist_mode(&env, ASSIST_MODE_FULL, 1.0f);
    EXPECT_EQ_INT(c_set_human_driving_style(&env, 1, min_values, max_values), 0);
    float capped_speed_mps = env.base_max_speed_mps / env.conditioning_speed_scale;
    EXPECT_EQ_INT(c_set_human_max_speed(&env, capped_speed_mps), 0);
    EXPECT_TRUE(same_bits(env.human_own_coefs[REWARD_COEF_SPEED], capped_speed_mps / env.base_max_speed_mps));
    EXPECT_TRUE(same_bits(human_agent(&env)->reward_coefs[REWARD_COEF_SPEED], env.human_style_coefs[REWARD_COEF_SPEED]));
    EXPECT_NEAR(env.human_max_speed_mps, capped_speed_mps, EXACT_TOLERANCE);
    c_set_assist_mode(&env, ASSIST_MODE_OFF, 0.0f);
    EXPECT_TRUE(same_bits(human_agent(&env)->reward_coefs[REWARD_COEF_SPEED], capped_speed_mps / env.base_max_speed_mps));
    free_allocated(&env);
    return 0;
}

static int test_switching_cars_restores_the_previous_car_and_styles_the_new_one(void) {
    Drive env = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
    int first_idx = env.human_agent_idx;
    float first_own_coefs[NUM_REWARD_COEFS];
    memcpy(first_own_coefs, env.human_own_coefs, sizeof(first_own_coefs));
    float min_values[NUM_REWARD_COEFS], max_values[NUM_REWARD_COEFS];
    style_window(&env, min_values, max_values);
    c_set_assist_mode(&env, ASSIST_MODE_FULL, 1.0f);
    EXPECT_EQ_INT(c_set_human_driving_style(&env, 1, min_values, max_values), 0);

    int second_idx = env.active_agent_indices[OTHER_ACTIVE_IDX];
    EXPECT_EQ_INT(c_select_human_agent(&env, second_idx), 0);
    EXPECT_TRUE(memcmp(env.agents[first_idx].reward_coefs, first_own_coefs, sizeof(first_own_coefs)) == 0);
    float second_own_coefs[NUM_REWARD_COEFS];
    memcpy(second_own_coefs, env.human_own_coefs, sizeof(second_own_coefs));
    EXPECT_TRUE(same_bits(second_own_coefs[REWARD_COEF_SPEED], env.human_max_speed_mps / env.base_max_speed_mps));
    EXPECT_TRUE(same_bits(second_own_coefs[REWARD_COEF_THROTTLE], 1.0f));
    EXPECT_FALSE(expect_styled_coefs(&env, second_own_coefs, 1));
    free_allocated(&env);
    return 0;
}

static int test_style_survives_an_episode_reset(void) {
    Drive env = full_env(DYNAMICS_MODEL_JERK, FULL_TEST_SEED);
    float min_values[NUM_REWARD_COEFS], max_values[NUM_REWARD_COEFS];
    style_window(&env, min_values, max_values);
    c_set_assist_mode(&env, ASSIST_MODE_FULL, 1.0f);
    EXPECT_EQ_INT(c_set_human_driving_style(&env, 1, min_values, max_values), 0);
    int human_agent_idx = env.human_agent_idx;
    c_reset(&env);
    EXPECT_EQ_INT(env.human_agent_idx, human_agent_idx);
    EXPECT_EQ_INT(env.human_style_active, 1);
    float own_coefs[NUM_REWARD_COEFS];
    memcpy(own_coefs, env.human_own_coefs, sizeof(own_coefs));
    EXPECT_FALSE(expect_styled_coefs(&env, own_coefs, 1));
    free_allocated(&env);
    return 0;
}


#define GEARLESS_RAMP_STEP_COUNT 20
#define GEARLESS_HOLD_STEP_COUNT 30
#define JERK_RATE_TOLERANCE 1e-3f

static float jerk_long_max(void) {
    return JERK_LONG[NUM_JERK_LONG_ACTIONS - 1];
}

static void gearless_bench_init(FullBench *bench, float speed_mps, int gear) {
    full_bench_init(bench, speed_mps, FULL_ASSIST_DYNAMICS_HUMAN);
    bench->env.assist_mode = ASSIST_MODE_OFF;
    bench->env.assist_reference_valid = 0;
    bench->env.human_gear = gear;
    c_set_human_gearbox_mode(&bench->env, GEARBOX_MODE_NONE);
}

static void press(FullBench *bench, float throttle, float brake, int shift_request) {
    c_set_human_pedals(&bench->env, 0.0f, throttle, brake, 0.0f, shift_request);
}

static int test_gearbox_mode_setter_accepts_none_and_collapses_forward_gears(void) {
    FullBench bench;
    full_bench_init(&bench, 0.0f, FULL_ASSIST_DYNAMICS_HUMAN);
    int bad_modes[] = {GEARBOX_MODE_MANUAL - 1, GEARBOX_MODE_NONE + 1};
    for (int b = 0; b < 2; b++) {
        EXPECT_EQ_INT(c_set_human_gearbox_mode(&bench.env, bad_modes[b]), -1);
        EXPECT_EQ_INT(bench.env.human_gearbox_mode, GEARBOX_MODE_AUTOMATIC);
    }
    int gears[] = {GEAR_NEUTRAL, 1, GEARBOX_FORWARD_GEAR_COUNT, GEAR_REVERSE};
    int expected[] = {1, 1, 1, GEAR_REVERSE};
    for (int g = 0; g < 4; g++) {
        bench.env.human_gearbox_mode = GEARBOX_MODE_AUTOMATIC;
        bench.env.human_gear = gears[g];
        EXPECT_EQ_INT(c_set_human_gearbox_mode(&bench.env, GEARBOX_MODE_NONE), 0);
        EXPECT_EQ_INT(bench.env.human_gearbox_mode, GEARBOX_MODE_NONE);
        EXPECT_EQ_INT(bench.env.human_gear, expected[g]);
    }
    EXPECT_EQ_INT(c_set_human_gearbox_mode(&bench.env, GEARBOX_MODE_MANUAL), 0);
    EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    return 0;
}

static int test_gearless_full_throttle_ramps_at_the_jerk_limit_to_the_accel_limit(void) {
    FullBench bench;
    gearless_bench_init(&bench, 0.0f, 1);
    press(&bench, 1.0f, 0.0f, 0);
    float previous_accel_mps2 = 0.0f;
    for (int step = 0; step < GEARLESS_RAMP_STEP_COUNT; step++) {
        move_human(&bench.env, 0, 0);
        float expected_mps2 = fminf(previous_accel_mps2 + jerk_long_max() * FULL_TEST_DT_SECONDS, ACCEL_LONG_LIMIT[1]);
        EXPECT_NEAR(bench.agent.accel_long, expected_mps2, JERK_RATE_TOLERANCE);
        EXPECT_TRUE(bench.agent.sim_speed_signed > 0.0f);
        EXPECT_NEAR(bench.env.human_engine_rpm, ENGINE_IDLE_RPM, EXACT_TOLERANCE);
        EXPECT_EQ_INT(bench.env.human_gear, 1);
        previous_accel_mps2 = bench.agent.accel_long;
    }
    EXPECT_NEAR(bench.agent.accel_long, ACCEL_LONG_LIMIT[1], EXACT_TOLERANCE);
    return 0;
}

static int test_gearless_release_falls_to_zero_then_holds_speed(void) {
    FullBench bench;
    gearless_bench_init(&bench, CRUISE_SPEED_MPS, 1);
    bench.agent.accel_long = ACCEL_LONG_LIMIT[1];
    press(&bench, 0.0f, 0.0f, 0);
    move_human(&bench.env, 0, 0);
    float expected_mps2 = fmaxf(ACCEL_LONG_LIMIT[1] + JERK_LONG[0] * FULL_TEST_DT_SECONDS, 0.0f);
    EXPECT_NEAR(bench.agent.accel_long, expected_mps2, JERK_RATE_TOLERANCE);
    for (int step = 0; step < GEARLESS_RAMP_STEP_COUNT; step++) {
        move_human(&bench.env, 0, 0);
    }
    EXPECT_NEAR(bench.agent.accel_long, 0.0f, EXACT_TOLERANCE);
    float coasting_speed_mps = bench.agent.sim_speed_signed;
    for (int step = 0; step < GEARLESS_HOLD_STEP_COUNT; step++) {
        move_human(&bench.env, 0, 0);
        EXPECT_TRUE(bench.agent.sim_speed_signed == coasting_speed_mps);
    }
    return 0;
}

static int test_gearless_full_brake_stops_and_never_reverses(void) {
    FullBench bench;
    gearless_bench_init(&bench, CRUISE_SPEED_MPS, 1);
    press(&bench, 0.0f, 1.0f, 0);
    move_human(&bench.env, 0, 0);
    EXPECT_NEAR(bench.agent.accel_long, JERK_LONG[0] * FULL_TEST_DT_SECONDS, JERK_RATE_TOLERANCE);
    int stopped = 0;
    for (int step = 0; step < 4 * GEARLESS_HOLD_STEP_COUNT; step++) {
        move_human(&bench.env, 0, 0);
        EXPECT_TRUE(bench.agent.accel_long >= ACCEL_LONG_LIMIT[0]);
        EXPECT_TRUE(bench.agent.sim_speed_signed >= 0.0f);
        stopped |= bench.agent.sim_speed_signed == 0.0f;
    }
    EXPECT_TRUE(stopped);
    EXPECT_TRUE(bench.agent.sim_speed_signed == 0.0f);
    EXPECT_EQ_INT(bench.env.human_gear, 1);
    return 0;
}

static int test_gearless_throttle_and_brake_at_rest_stay_at_rest(void) {
    FullBench bench;
    gearless_bench_init(&bench, 0.0f, 1);
    press(&bench, 1.0f, 1.0f, 0);
    for (int step = 0; step < GEARLESS_HOLD_STEP_COUNT; step++) {
        move_human(&bench.env, 0, 0);
        EXPECT_TRUE(bench.agent.sim_speed_signed == 0.0f);
    }
    return 0;
}

static int test_gearless_paddle_selects_reverse_only_at_rest_and_reverse_is_capped(void) {
    FullBench bench;
    gearless_bench_init(&bench, CRUISE_SPEED_MPS, 1);
    press(&bench, 0.0f, 0.0f, -1);
    move_human(&bench.env, 0, 0);
    EXPECT_EQ_INT(bench.env.human_gear, 1);

    gearless_bench_init(&bench, 0.0f, 1);
    press(&bench, 0.0f, 0.0f, -1);
    move_human(&bench.env, 0, 0);
    EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    press(&bench, 1.0f, 0.0f, 0);
    for (int step = 0; step < 2 * GEARLESS_HOLD_STEP_COUNT; step++) {
        move_human(&bench.env, 0, 0);
        EXPECT_TRUE(bench.agent.sim_speed_signed <= 0.0f);
        EXPECT_TRUE(bench.agent.sim_speed_signed >= MAX_BACKWARD_SPEED);
    }
    EXPECT_NEAR(bench.agent.sim_speed_signed, MAX_BACKWARD_SPEED, SPEED_TOLERANCE_MPS);

    press(&bench, 0.0f, 1.0f, 0);
    for (int step = 0; step < 2 * GEARLESS_HOLD_STEP_COUNT; step++) {
        move_human(&bench.env, 0, 0);
        EXPECT_TRUE(bench.agent.sim_speed_signed <= 0.0f);
    }
    EXPECT_TRUE(bench.agent.sim_speed_signed == 0.0f);
    EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);

    press(&bench, 0.0f, 0.0f, 1);
    move_human(&bench.env, 0, 0);
    EXPECT_EQ_INT(bench.env.human_gear, 1);
    return 0;
}

static int test_gearless_obeys_your_own_speed_cap(void) {
    FullBench bench;
    gearless_bench_init(&bench, FULL_TEST_MAX_SPEED_MPS - 0.01f, 1);
    bench.agent.accel_long = ACCEL_LONG_LIMIT[1];
    press(&bench, 1.0f, 0.0f, 0);
    for (int step = 0; step < GEARLESS_HOLD_STEP_COUNT; step++) {
        move_human(&bench.env, 0, 0);
        EXPECT_TRUE(bench.agent.sim_speed_signed <= FULL_TEST_MAX_SPEED_MPS);
    }
    EXPECT_NEAR(bench.agent.sim_speed_signed, FULL_TEST_MAX_SPEED_MPS, SPEED_TOLERANCE_MPS);
    return 0;
}

static int test_gearless_under_full_with_the_human_dynamics_still_drives_jerk_speed(void) {
    FullBench gearless;
    FullBench jerk_speed;
    full_bench_init(&gearless, CRUISE_SPEED_MPS, FULL_ASSIST_DYNAMICS_HUMAN);
    c_set_human_gearbox_mode(&gearless.env, GEARBOX_MODE_NONE);
    full_bench_init(&jerk_speed, CRUISE_SPEED_MPS, FULL_ASSIST_DYNAMICS_JERK_SPEED);
    for (int step = 0; step < FULL_TEST_STEP_COUNT; step++) {
        move_human(&gearless.env, 0, 0);
        move_human(&jerk_speed.env, 0, 0);
        EXPECT_TRUE(memcmp(&gearless.agent, &jerk_speed.agent, sizeof(Agent)) == 0);
        EXPECT_NEAR(gearless.env.human_engine_rpm, ENGINE_IDLE_RPM, EXACT_TOLERANCE);
    }
    return 0;
}

static int test_gearless_under_full_follows_the_policy_into_reverse_at_rest(void) {
    FullBench bench;
    full_bench_init(&bench, 0.0f, FULL_ASSIST_DYNAMICS_HUMAN);
    c_set_human_gearbox_mode(&bench.env, GEARBOX_MODE_NONE);
    bench.env.assist_reference_speed_mps = -MOVING_REFERENCE_SPEED_MPS;
    bench.actions[0][0] = -1.0f;
    move_human(&bench.env, 0, 0);
    EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    for (int step = 0; step < GEARLESS_HOLD_STEP_COUNT; step++) {
        move_human(&bench.env, 0, 0);
    }
    EXPECT_TRUE(bench.agent.sim_speed_signed < 0.0f);
    EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    bench.env.assist_reference_speed_mps = MOVING_REFERENCE_SPEED_MPS;
    move_human(&bench.env, 0, 0);
    EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    return 0;
}

static int test_a_manual_gearbox_in_neutral_hands_every_jerk_dynamics_back_to_the_driver(void) {
    int jerk_modes[] = {FULL_ASSIST_DYNAMICS_JERK_SPEED, FULL_ASSIST_DYNAMICS_JERK, FULL_ASSIST_DYNAMICS_SET_STEERING};
    for (int m = 0; m < 3; m++) {
        FullBench full;
        FullBench off;
        full_bench_init(&full, CRUISE_SPEED_MPS, jerk_modes[m]);
        full_bench_init(&off, CRUISE_SPEED_MPS, jerk_modes[m]);
        off.env.assist_mode = ASSIST_MODE_OFF;
        FullBench *benches[2] = {&full, &off};
        for (int k = 0; k < 2; k++) {
            benches[k]->env.human_gearbox_mode = GEARBOX_MODE_MANUAL;
            benches[k]->env.human_gear = GEAR_NEUTRAL;
        }
        c_set_assist_steering(&full.env, 1, SET_STEERING_RAD);
        for (int step = 0; step < FULL_TEST_STEP_COUNT; step++) {
            move_human(&full.env, 0, 0);
            move_human(&off.env, 0, 0);
            EXPECT_TRUE(memcmp(&full.agent, &off.agent, sizeof(Agent)) == 0);
            EXPECT_EQ_INT(full.env.human_gear, GEAR_NEUTRAL);
        }
    }
    return 0;
}

static int test_a_manual_gearbox_in_gear_is_driven_by_the_policy(void) {
    FullBench bench;
    full_bench_init(&bench, CRUISE_SPEED_MPS, FULL_ASSIST_DYNAMICS_JERK_SPEED);
    bench.env.human_gearbox_mode = GEARBOX_MODE_MANUAL;
    bench.env.human_gear = 3;
    Agent expected = expected_jerk_speed_move(&bench, -WHEEL_STEERING_NORM * HUMAN_STEERING_LIMIT_RAD);
    move_human(&bench.env, 0, 0);
    EXPECT_TRUE(memcmp(&expected, &bench.agent, sizeof(Agent)) == 0);

    full_bench_init(&bench, CRUISE_SPEED_MPS, FULL_ASSIST_DYNAMICS_JERK);
    bench.env.human_gearbox_mode = GEARBOX_MODE_MANUAL;
    bench.env.human_gear = 3;
    expected = expected_jerk_move(&bench);
    move_human(&bench.env, 0, 0);
    EXPECT_TRUE(memcmp(&expected, &bench.agent, sizeof(Agent)) == 0);
    return 0;
}

#define TAKEOVER_STEP_COUNT 120
#define BACKWARD_ROLL_MPS -1.5f

static void press_pedals(FullBench *bench, float throttle, float brake, int shift_request) {
    c_set_human_pedals(&bench->env, 0.0f, throttle, brake, 0.0f, shift_request);
}

// FULL picks R for a policy reversing from rest, so the selector is in an assist-chosen R.
static void assist_chosen_reverse(FullBench *bench, int gearbox_mode) {
    reverse_bench_init(bench, 0.0f, 1, -MOVING_REFERENCE_SPEED_MPS);
    bench->env.human_gearbox_mode = gearbox_mode;
    press_pedals(bench, 0.0f, 0.0f, 0);
    move_human(&bench->env, 0, 0);
}

static int expect_stops_in_drive_then_throttle_goes_forward(FullBench *bench) {
    for (int step = 0; step < TAKEOVER_STEP_COUNT; step++) {
        float speed_before_mps = bench->agent.sim_speed_signed;
        move_human(&bench->env, 0, 0);
        if (bench->env.human_gear == GEAR_REVERSE) {
            continue;
        }
        EXPECT_TRUE(fabsf(speed_before_mps) <= AUTO_SELECTOR_MAX_SPEED_MPS);
    }
    EXPECT_TRUE(bench->agent.sim_speed_signed == 0.0f);
    EXPECT_EQ_INT(bench->env.human_gear, 1);
    EXPECT_EQ_INT(bench->env.human_reverse_chosen_by_assist, 0);
    press_pedals(bench, 1.0f, 0.0f, 0);
    for (int step = 0; step < TAKEOVER_STEP_COUNT; step++) {
        move_human(&bench->env, 0, 0);
        EXPECT_TRUE(bench->agent.sim_speed_signed >= 0.0f);
    }
    EXPECT_TRUE(bench->agent.sim_speed_signed > 0.0f);
    EXPECT_TRUE(bench->env.human_gear >= 1);
    return 0;
}

static int test_full_marks_the_reverse_it_chose(void) {
    int gearboxes[] = {GEARBOX_MODE_AUTOMATIC, GEARBOX_MODE_NONE};
    for (int g = 0; g < 2; g++) {
        FullBench bench;
        assist_chosen_reverse(&bench, gearboxes[g]);
        EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
        EXPECT_EQ_INT(bench.env.human_reverse_chosen_by_assist, 1);
        EXPECT_EQ_INT(c_set_human_gear(&bench.env, GEAR_REVERSE), 0);
        EXPECT_EQ_INT(bench.env.human_reverse_chosen_by_assist, 0);
    }
    return 0;
}

static int test_switching_the_assist_off_in_its_reverse_returns_to_drive_at_rest(void) {
    int gearboxes[] = {GEARBOX_MODE_AUTOMATIC, GEARBOX_MODE_NONE};
    for (int g = 0; g < 2; g++) {
        FullBench bench;
        assist_chosen_reverse(&bench, gearboxes[g]);
        bench.env.assist_mode = ASSIST_MODE_OFF;
        press_pedals(&bench, 0.0f, 1.0f, 0);
        EXPECT_FALSE(expect_stops_in_drive_then_throttle_goes_forward(&bench));
    }
    return 0;
}

static int test_a_brake_takeover_in_the_assists_reverse_returns_to_drive_at_rest(void) {
    int gearboxes[] = {GEARBOX_MODE_AUTOMATIC, GEARBOX_MODE_NONE};
    for (int g = 0; g < 2; g++) {
        FullBench bench;
        assist_chosen_reverse(&bench, gearboxes[g]);
        press_pedals(&bench, 0.0f, 1.0f, 0);
        EXPECT_FALSE(expect_stops_in_drive_then_throttle_goes_forward(&bench));
    }
    return 0;
}

static int test_a_reverse_the_driver_chose_stays_in_reverse(void) {
    FullBench bench;
    reverse_bench_init(&bench, 0.0f, 1, MOVING_REFERENCE_SPEED_MPS);
    bench.env.assist_mode = ASSIST_MODE_OFF;
    press_pedals(&bench, 0.0f, 1.0f, -1);
    move_human(&bench.env, 0, 0);
    EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    EXPECT_EQ_INT(bench.env.human_reverse_chosen_by_assist, 0);
    press_pedals(&bench, 0.0f, 1.0f, 0);
    for (int step = 0; step < TAKEOVER_STEP_COUNT; step++) {
        move_human(&bench.env, 0, 0);
        EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    }
    press_pedals(&bench, 1.0f, 0.0f, 0);
    for (int step = 0; step < TAKEOVER_STEP_COUNT; step++) {
        move_human(&bench.env, 0, 0);
        EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    }
    EXPECT_NEAR(bench.agent.sim_speed_signed, MAX_BACKWARD_SPEED, SPEED_TOLERANCE_MPS);
    return 0;
}

static int test_full_alone_keeps_its_reverse_while_the_policy_reverses(void) {
    FullBench bench;
    assist_chosen_reverse(&bench, GEARBOX_MODE_AUTOMATIC);
    for (int step = 0; step < TAKEOVER_STEP_COUNT; step++) {
        move_human(&bench.env, 0, 0);
        EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
        EXPECT_TRUE(bench.agent.sim_speed_signed <= 0.0f);
    }
    EXPECT_TRUE(bench.agent.sim_speed_signed < 0.0f);
    EXPECT_EQ_INT(bench.env.human_reverse_chosen_by_assist, 1);
    return 0;
}

static int test_a_car_still_rolling_back_at_takeover_keeps_reverse_until_it_stops(void) {
    FullBench bench;
    assist_chosen_reverse(&bench, GEARBOX_MODE_AUTOMATIC);
    bench.agent.sim_vx = BACKWARD_ROLL_MPS * bench.agent.cos_heading;
    bench.agent.sim_vy = BACKWARD_ROLL_MPS * bench.agent.sin_heading;
    update_agent_speed(&bench.agent);
    bench.agent.accel_long = 0.0f;
    bench.env.assist_mode = ASSIST_MODE_OFF;
    press_pedals(&bench, 0.0f, 0.3f, 0);
    int saw_reverse_while_rolling = 0;
    for (int step = 0; step < TAKEOVER_STEP_COUNT; step++) {
        float speed_before_mps = bench.agent.sim_speed_signed;
        move_human(&bench.env, 0, 0);
        if (fabsf(speed_before_mps) > AUTO_SELECTOR_MAX_SPEED_MPS) {
            EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
            saw_reverse_while_rolling = 1;
        }
    }
    EXPECT_TRUE(saw_reverse_while_rolling);
    EXPECT_TRUE(bench.agent.sim_speed_signed == 0.0f);
    EXPECT_EQ_INT(bench.env.human_gear, 1);
    return 0;
}

static int test_a_manual_gearbox_is_never_shifted_out_of_reverse(void) {
    FullBench bench;
    reverse_bench_init(&bench, 0.0f, GEAR_REVERSE, -MOVING_REFERENCE_SPEED_MPS);
    bench.env.human_gearbox_mode = GEARBOX_MODE_MANUAL;
    bench.env.human_reverse_chosen_by_assist = 1;
    bench.env.assist_mode = ASSIST_MODE_OFF;
    press_pedals(&bench, 0.0f, 1.0f, 0);
    for (int step = 0; step < TAKEOVER_STEP_COUNT; step++) {
        move_human(&bench.env, 0, 0);
        EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    }
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_begin_preview_refuses_when_there_is_nothing_to_imagine);
    RUN_TEST(test_preview_step_refuses_without_a_preview_or_pieces);
    RUN_TEST(test_preview_step_integrates_the_slot_action_in_pieces_with_the_jerk_model);
    RUN_TEST(test_preview_step_rebuilds_only_the_human_observation_row);
    RUN_TEST(test_end_preview_restores_agents_observations_and_rng_exactly);
    RUN_TEST(test_previews_never_change_later_trajectories);
    RUN_TEST(test_split_observations_match_the_pre_split_loop_for_every_agent);
    RUN_TEST(test_apply_jerk_step_writes_exactly_the_integrated_step);
    RUN_TEST(test_full_assist_dynamics_setter_validates_and_keeps_state);
    RUN_TEST(test_assist_steering_setter_rejects_a_non_finite_valid_angle);
    RUN_TEST(test_jerk_speed_drives_speed_on_the_jerk_model_and_steers_by_the_wheel);
    RUN_TEST(test_jerk_speed_obeys_your_own_speed_cap);
    RUN_TEST(test_jerk_drives_speed_and_steering_and_ignores_the_wheel);
    RUN_TEST(test_set_steering_steers_by_the_jerk_model_until_an_angle_arrives);
    RUN_TEST(test_set_steering_moves_toward_the_set_angle_at_the_jerk_rate_and_ignores_the_wheel);
    RUN_TEST(test_set_steering_is_clipped_to_the_jerk_steering_limit);
    RUN_TEST(test_a_pressed_pedal_hands_every_full_dynamics_back_to_the_driver);
    RUN_TEST(test_jerk_paths_still_run_the_powertrain);
    RUN_TEST(test_a_manual_gearbox_in_neutral_hands_every_jerk_dynamics_back_to_the_driver);
    RUN_TEST(test_a_manual_gearbox_in_gear_is_driven_by_the_policy);
    RUN_TEST(test_set_steering_validity_resets_on_select_and_reset);
    RUN_TEST(test_full_selects_reverse_when_the_policy_step_ends_moving_backward);
    RUN_TEST(test_full_reverse_selector_holds_while_rolling_or_without_the_policy);
    RUN_TEST(test_a_driver_paddle_wins_over_the_automatic_reverse_request);
    RUN_TEST(test_full_marks_the_reverse_it_chose);
    RUN_TEST(test_switching_the_assist_off_in_its_reverse_returns_to_drive_at_rest);
    RUN_TEST(test_a_brake_takeover_in_the_assists_reverse_returns_to_drive_at_rest);
    RUN_TEST(test_a_reverse_the_driver_chose_stays_in_reverse);
    RUN_TEST(test_full_alone_keeps_its_reverse_while_the_policy_reverses);
    RUN_TEST(test_a_car_still_rolling_back_at_takeover_keeps_reverse_until_it_stops);
    RUN_TEST(test_a_manual_gearbox_is_never_shifted_out_of_reverse);
    RUN_TEST(test_style_setter_validates_the_window_and_keeps_state);
    RUN_TEST(test_style_is_the_window_centre_geometric_on_log_scale);
    RUN_TEST(test_style_applies_only_under_full_and_only_to_shaped_coefficients);
    RUN_TEST(test_capability_and_unrandomized_goal_speed_are_never_styled);
    RUN_TEST(test_max_speed_writes_your_own_speed_coefficient_not_the_style);
    RUN_TEST(test_switching_cars_restores_the_previous_car_and_styles_the_new_one);
    RUN_TEST(test_style_survives_an_episode_reset);
    RUN_TEST(test_gearbox_mode_setter_accepts_none_and_collapses_forward_gears);
    RUN_TEST(test_gearless_full_throttle_ramps_at_the_jerk_limit_to_the_accel_limit);
    RUN_TEST(test_gearless_release_falls_to_zero_then_holds_speed);
    RUN_TEST(test_gearless_full_brake_stops_and_never_reverses);
    RUN_TEST(test_gearless_throttle_and_brake_at_rest_stay_at_rest);
    RUN_TEST(test_gearless_paddle_selects_reverse_only_at_rest_and_reverse_is_capped);
    RUN_TEST(test_gearless_obeys_your_own_speed_cap);
    RUN_TEST(test_gearless_under_full_with_the_human_dynamics_still_drives_jerk_speed);
    RUN_TEST(test_gearless_under_full_follows_the_policy_into_reverse_at_rest);
    return test_summary(failures);
}
