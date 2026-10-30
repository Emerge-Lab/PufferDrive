#include "include/drive_fixture.h"
#include "include/test.h"

#define HUMAN_TEST_DT_SECONDS 0.1f
#define HUMAN_TEST_MAX_SPEED_MPS 20.0f
#define HUMAN_TEST_AGENT_COUNT 4
#define JERK_PATH_STEERING_RATE_LIMIT_RPS 0.6f
#define JERK_PATH_STEERING_POSITION_LIMIT_RAD 0.55f
#define STEERING_RATIO_MIN 14.0f
#define STEERING_RATIO_MAX 18.0f
#define EXACT_TOLERANCE 0.0f
#define SPEED_TOLERANCE_MPS 1e-6f
#define CURVATURE_TOLERANCE_PER_M 1e-6f
#define ACCEL_TOLERANCE_MPS2 1e-3f
#define ARC_TOLERANCE_M 1e-4f
#define CIRCLE_CLOSURE_TOLERANCE_M 2e-3f
#define CIRCLE_STEP_COUNT 64
#define PARTNER_OFFSET_AHEAD_M 12.0f
#define COLLISION_OVERLAP_OFFSET_M 0.5f
#define OBS_MATCH_TOLERANCE 1e-4f
#define COMFORT_JERK_THRESHOLD_MPS3 5.0f
// Few enough steps that a full-lock turn cannot wrap the heading past the half circle.
#define TURN_DIRECTION_STEP_COUNT 3
#define SIZE_PROFILE_AGENT_COUNT 16
#define CAR_PROFILE_MIN_LENGTH_M 2.0f
#define CAR_PROFILE_MAX_LENGTH_M 5.5f
#define CAR_PROFILE_MIN_WIDTH_M 1.5f
#define CAR_PROFILE_MAX_WIDTH_M 2.5f

typedef struct {
    Drive env;
    GridMap offgrid_map;
    Agent agent;
} HumanBench;

// The zeroed GridMap reports every query off-grid, which makes update_agent_z a no-op.
static void human_bench_init(HumanBench *bench, float heading_rad, float speed_mps) {
    memset(bench, 0, sizeof(HumanBench));
    bench->agent = drive_test_agent(0.0f, 0.0f, heading_rad);
    bench->agent.reward_coefs[REWARD_COEF_SPEED] = 1.0f;
    bench->agent.reward_coefs[REWARD_COEF_ACC] = 1.0f;
    bench->agent.sim_vx = speed_mps * bench->agent.cos_heading;
    bench->agent.sim_vy = speed_mps * bench->agent.sin_heading;
    update_agent_speed(&bench->agent);
    copy_pose_to_prev(&bench->agent);
    bench->env.agents = &bench->agent;
    bench->env.num_agents = 1;
    bench->env.grid_map = &bench->offgrid_map;
    bench->env.dt = HUMAN_TEST_DT_SECONDS;
    bench->env.base_max_speed_mps = HUMAN_TEST_MAX_SPEED_MPS;
    bench->env.human_grip_mode = GRIP_MODE_KINEMATIC;
}

static Drive human_full_env(
    const char *map_file,
    int simulation_mode,
    int num_agents,
    int collision_behavior,
    unsigned int seed) {
    srand(seed);
    Drive env = drive_test_env_config(map_file, simulation_mode, num_agents, 0);
    env.max_speed_mps = HUMAN_TEST_MAX_SPEED_MPS;
    env.conditioning_accel_scale = 1.0f;
    env.collision_behavior = collision_behavior;
    allocate(&env);
    c_reset(&env);
    return env;
}

static Drive human_randomized_env(int num_agents, unsigned int seed) {
    srand(seed);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, num_agents, 0);
    env.max_speed_mps = HUMAN_TEST_MAX_SPEED_MPS;
    env.conditioning_accel_scale = 1.0f;
    env.reward_randomization = 1;
    allocate(&env);
    c_reset(&env);
    return env;
}

static void place_agent_ahead_of(Agent *agent, const Agent *reference, float distance_ahead_m) {
    agent->sim_x = reference->sim_x + distance_ahead_m * reference->cos_heading;
    agent->sim_y = reference->sim_y + distance_ahead_m * reference->sin_heading;
    agent->sim_z = reference->sim_z;
    agent->sim_heading = reference->sim_heading;
    agent->cos_heading = reference->cos_heading;
    agent->sin_heading = reference->sin_heading;
    copy_pose_to_prev(agent);
}

static float commanded_curvature_per_m(const Agent *agent, float steering_norm) {
    return tanf(-steering_norm * HUMAN_STEERING_LIMIT_RAD) / agent->wheelbase;
}

static float achieved_curvature_per_m(const Agent *agent) {
    return agent->yaw_rate / agent->sim_speed_signed;
}

// Verbatim copy of the pre-extraction inline integrator from move_dynamics, kept as the oracle.
static void reference_bicycle_arc(Agent *agent, float signed_curvature, float v_new, float signed_v, float dt) {
    float heading_x = agent->cos_heading;
    float heading_y = agent->sin_heading;
    float d = 0.5f * (v_new + signed_v) * dt;
    float theta = d * signed_curvature;
    float dx_local, dy_local;

    if (fabsf(signed_curvature) < 1e-5f || fabsf(theta) < 1e-5f) {
        dx_local = d;
        dy_local = 0.0f;
    } else {
        dx_local = sinf(theta) / signed_curvature;
        dy_local = (1.0f - cosf(theta)) / signed_curvature;
    }

    float dx = dx_local * heading_x - dy_local * heading_y;
    float dy = dx_local * heading_y + dy_local * heading_x;

    agent->sim_x += dx;
    agent->sim_y += dy;
    agent->sim_heading = normalize_heading(agent->sim_heading + theta);
    agent->cos_heading = cosf(agent->sim_heading);
    agent->sin_heading = sinf(agent->sim_heading);
    agent->sim_vx = v_new * cosf(agent->sim_heading);
    agent->sim_vy = v_new * sinf(agent->sim_heading);
    agent->yaw_rate = v_new * signed_curvature;
    update_agent_speed(agent);
}

// A right-lock command (+1) is a NEGATIVE road-wheel angle, because positive curvature is a left turn.
static int test_human_steering_is_exact_position_command(void) {
    static const float steering_norms[] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.25f, 0.5f, 1.0f, 3.0f};
    static const float expected_angles_rad[] = {
        HUMAN_STEERING_LIMIT_RAD,
        HUMAN_STEERING_LIMIT_RAD,
        0.5f * HUMAN_STEERING_LIMIT_RAD,
        0.0f,
        -0.25f * HUMAN_STEERING_LIMIT_RAD,
        -0.5f * HUMAN_STEERING_LIMIT_RAD,
        -HUMAN_STEERING_LIMIT_RAD,
        -HUMAN_STEERING_LIMIT_RAD,
    };
    int case_count = (int) (sizeof(steering_norms) / sizeof(steering_norms[0]));
    for (int i = 0; i < case_count; i++) {
        HumanBench bench;
        human_bench_init(&bench, 0.0f, 5.0f);
        EXPECT_EQ_INT(c_set_human_input(&bench.env, steering_norms[i], 0.0f), 0);
        move_human(&bench.env, 0);
        EXPECT_NEAR(bench.agent.steering_angle, expected_angles_rad[i], EXACT_TOLERANCE);
    }
    EXPECT_NEAR(HUMAN_STEERING_LIMIT_RAD, JERK_PATH_STEERING_POSITION_LIMIT_RAD, EXACT_TOLERANCE);
    return 0;
}

// The property a driver can see, stated without reference to any internal sign: turning the
// wheel right must move the car right. Asserting steering_angle instead is blind to a negation.
static int test_wheel_right_steers_the_car_right(void) {
    static const float speeds_mps[] = {3.0f, 8.0f, 15.0f};
    static const float steering_norms[] = {-1.0f, -0.4f, 0.4f, 1.0f};
    int speed_count = (int) (sizeof(speeds_mps) / sizeof(speeds_mps[0]));
    int steering_count = (int) (sizeof(steering_norms) / sizeof(steering_norms[0]));

    for (int grip_mode = GRIP_MODE_KINEMATIC; grip_mode <= GRIP_MODE_REALISTIC; grip_mode++) {
        for (int s = 0; s < speed_count; s++) {
            for (int w = 0; w < steering_count; w++) {
                float steering_norm = steering_norms[w];
                HumanBench bench;
                human_bench_init(&bench, 0.0f, speeds_mps[s]);
                bench.env.human_grip_mode = grip_mode;
                EXPECT_EQ_INT(c_set_human_input(&bench.env, steering_norm, 0.0f), 0);

                for (int step = 0; step < TURN_DIRECTION_STEP_COUNT; step++) {
                    float heading_before_rad = bench.agent.sim_heading;
                    move_human(&bench.env, 0);
                    float heading_change_rad = normalize_heading(bench.agent.sim_heading - heading_before_rad);
                    if (steering_norm > 0.0f) {
                        EXPECT_TRUE(heading_change_rad < 0.0f);
                        EXPECT_TRUE(bench.agent.yaw_rate < 0.0f);
                    } else {
                        EXPECT_TRUE(heading_change_rad > 0.0f);
                        EXPECT_TRUE(bench.agent.yaw_rate > 0.0f);
                    }
                }
                // Started at the origin heading +x, so right of the start line is negative y.
                if (steering_norm > 0.0f) {
                    EXPECT_TRUE(bench.agent.sim_y < 0.0f);
                } else {
                    EXPECT_TRUE(bench.agent.sim_y > 0.0f);
                }
                EXPECT_TRUE(bench.agent.sim_x > 0.0f);
            }
        }
    }
    return 0;
}

static int test_centred_wheel_drives_straight(void) {
    for (int grip_mode = GRIP_MODE_KINEMATIC; grip_mode <= GRIP_MODE_REALISTIC; grip_mode++) {
        HumanBench bench;
        human_bench_init(&bench, 0.0f, 10.0f);
        bench.env.human_grip_mode = grip_mode;
        EXPECT_EQ_INT(c_set_human_input(&bench.env, 0.0f, 0.0f), 0);
        for (int step = 0; step < TURN_DIRECTION_STEP_COUNT; step++) {
            move_human(&bench.env, 0);
            EXPECT_NEAR(bench.agent.sim_heading, 0.0f, EXACT_TOLERANCE);
            EXPECT_NEAR(bench.agent.sim_y, 0.0f, EXACT_TOLERANCE);
            EXPECT_NEAR(bench.agent.yaw_rate, 0.0f, EXACT_TOLERANCE);
        }
        EXPECT_TRUE(bench.agent.sim_x > 0.0f);
    }
    return 0;
}

// The chain the negation in move_human depends on: positive curvature is a left turn.
static int test_positive_curvature_turns_left(void) {
    const float speed_mps = 10.0f;
    {
        Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
        integrate_bicycle_arc(&agent, 0.1f, speed_mps, speed_mps, HUMAN_TEST_DT_SECONDS);
        EXPECT_TRUE(agent.sim_heading > 0.0f);
        EXPECT_TRUE(agent.sim_y > 0.0f);
        EXPECT_TRUE(agent.yaw_rate > 0.0f);
    }
    {
        Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
        integrate_bicycle_arc(&agent, -0.1f, speed_mps, speed_mps, HUMAN_TEST_DT_SECONDS);
        EXPECT_TRUE(agent.sim_heading < 0.0f);
        EXPECT_TRUE(agent.sim_y < 0.0f);
        EXPECT_TRUE(agent.yaw_rate < 0.0f);
    }
    // A positive road-wheel angle makes positive curvature, so it too is a left turn.
    HumanBench bench;
    human_bench_init(&bench, 0.0f, speed_mps);
    EXPECT_TRUE(tanf(HUMAN_STEERING_LIMIT_RAD) / bench.agent.wheelbase > 0.0f);

    // Full chain for a right-lock command: command > 0, angle < 0, curvature < 0, heading falls.
    EXPECT_EQ_INT(c_set_human_input(&bench.env, 1.0f, 0.0f), 0);
    move_human(&bench.env, 0);
    EXPECT_TRUE(bench.agent.steering_angle < 0.0f);
    EXPECT_TRUE(tanf(bench.agent.steering_angle) / bench.agent.wheelbase < 0.0f);
    EXPECT_TRUE(bench.agent.yaw_rate < 0.0f);
    EXPECT_TRUE(bench.agent.sim_heading < 0.0f);
    return 0;
}

static int test_human_steering_is_symmetric_and_monotone(void) {
    const int sample_count = 41;
    float previous_angle_rad = INFINITY;
    for (int i = 0; i < sample_count; i++) {
        float steering_norm = -1.0f + 2.0f * (float) i / (float) (sample_count - 1);
        HumanBench right_bench, left_bench;
        human_bench_init(&right_bench, 0.0f, 5.0f);
        human_bench_init(&left_bench, 0.0f, 5.0f);
        c_set_human_input(&right_bench.env, steering_norm, 0.0f);
        c_set_human_input(&left_bench.env, -steering_norm, 0.0f);
        move_human(&right_bench.env, 0);
        move_human(&left_bench.env, 0);
        EXPECT_NEAR(right_bench.agent.steering_angle, -left_bench.agent.steering_angle, EXACT_TOLERANCE);
        EXPECT_TRUE(right_bench.agent.steering_angle < previous_angle_rad);
        previous_angle_rad = right_bench.agent.steering_angle;
    }
    return 0;
}

static int test_human_steering_has_no_rate_limit(void) {
    HumanBench bench;
    human_bench_init(&bench, 0.0f, 5.0f);
    bench.agent.steering_angle = HUMAN_STEERING_LIMIT_RAD;
    c_set_human_input(&bench.env, 1.0f, 0.0f);
    move_human(&bench.env, 0);
    EXPECT_NEAR(bench.agent.steering_angle, -HUMAN_STEERING_LIMIT_RAD, EXACT_TOLERANCE);
    float steering_rate_rps = 2.0f * HUMAN_STEERING_LIMIT_RAD / HUMAN_TEST_DT_SECONDS;
    EXPECT_TRUE(steering_rate_rps > JERK_PATH_STEERING_RATE_LIMIT_RPS);
    return 0;
}

static int test_human_steering_ratio_matches_wheel_range(void) {
    float half_lock_rad = HUMAN_WHEEL_RANGE_DEG * 0.5f * (float) M_PI / 180.0f;
    EXPECT_NEAR(half_lock_rad / HUMAN_STEERING_RATIO, HUMAN_STEERING_LIMIT_RAD, 1e-6f);
    EXPECT_TRUE(HUMAN_STEERING_RATIO > STEERING_RATIO_MIN);
    EXPECT_TRUE(HUMAN_STEERING_RATIO < STEERING_RATIO_MAX);
    return 0;
}

static int test_human_accel_is_commanded_not_integrated(void) {
    HumanBench bench;
    human_bench_init(&bench, 0.0f, 5.0f);

    c_set_human_input(&bench.env, 0.0f, 1.7f);
    move_human(&bench.env, 0);
    EXPECT_NEAR(bench.agent.accel_long, 1.7f, EXACT_TOLERANCE);

    c_set_human_input(&bench.env, 0.0f, -3.0f);
    move_human(&bench.env, 0);
    EXPECT_NEAR(bench.agent.accel_long, -3.0f, EXACT_TOLERANCE);

    c_set_human_input(&bench.env, 0.0f, 100.0f);
    move_human(&bench.env, 0);
    EXPECT_NEAR(bench.agent.accel_long, HUMAN_ACCEL_LIMIT_MPS2[1], EXACT_TOLERANCE);

    c_set_human_input(&bench.env, 0.0f, -100.0f);
    move_human(&bench.env, 0);
    EXPECT_NEAR(bench.agent.accel_long, HUMAN_ACCEL_LIMIT_MPS2[0], EXACT_TOLERANCE);
    return 0;
}

// The human car is not jerk-conditioned: it uses its own envelope, not the policy's scaled one.
static int test_human_accel_envelope_is_independent_of_the_policy(void) {
    HumanBench bench;
    human_bench_init(&bench, 0.0f, 5.0f);
    bench.agent.reward_coefs[REWARD_COEF_ACC] = 2.0f;

    c_set_human_input(&bench.env, 0.0f, 100.0f);
    move_human(&bench.env, 0);
    EXPECT_NEAR(bench.agent.accel_long, HUMAN_ACCEL_LIMIT_MPS2[1], EXACT_TOLERANCE);

    c_set_human_input(&bench.env, 0.0f, -100.0f);
    move_human(&bench.env, 0);
    EXPECT_NEAR(bench.agent.accel_long, HUMAN_ACCEL_LIMIT_MPS2[0], EXACT_TOLERANCE);

    EXPECT_TRUE(HUMAN_ACCEL_LIMIT_MPS2[1] > ACCEL_LONG_LIMIT[1]);
    EXPECT_TRUE(HUMAN_ACCEL_LIMIT_MPS2[0] < ACCEL_LONG_LIMIT[0]);
    return 0;
}

static int test_human_jerk_recorded_but_never_consumed(void) {
    HumanBench clean_bench, dirty_bench;
    human_bench_init(&clean_bench, 0.3f, 8.0f);
    human_bench_init(&dirty_bench, 0.3f, 8.0f);
    dirty_bench.agent.jerk_long = 900.0f;
    dirty_bench.agent.jerk_lat = -900.0f;

    c_set_human_input(&clean_bench.env, 0.4f, 1.25f);
    c_set_human_input(&dirty_bench.env, 0.4f, 1.25f);
    move_human(&clean_bench.env, 0);
    move_human(&dirty_bench.env, 0);

    EXPECT_NEAR(dirty_bench.agent.sim_x, clean_bench.agent.sim_x, EXACT_TOLERANCE);
    EXPECT_NEAR(dirty_bench.agent.sim_y, clean_bench.agent.sim_y, EXACT_TOLERANCE);
    EXPECT_NEAR(dirty_bench.agent.sim_heading, clean_bench.agent.sim_heading, EXACT_TOLERANCE);
    EXPECT_NEAR(dirty_bench.agent.sim_speed_signed, clean_bench.agent.sim_speed_signed, EXACT_TOLERANCE);
    EXPECT_NEAR(dirty_bench.agent.accel_long, clean_bench.agent.accel_long, EXACT_TOLERANCE);
    EXPECT_NEAR(dirty_bench.agent.accel_lat, clean_bench.agent.accel_lat, EXACT_TOLERANCE);
    EXPECT_NEAR(dirty_bench.agent.yaw_rate, clean_bench.agent.yaw_rate, EXACT_TOLERANCE);
    EXPECT_NEAR(dirty_bench.agent.jerk_long, clean_bench.agent.jerk_long, EXACT_TOLERANCE);
    EXPECT_NEAR(dirty_bench.agent.jerk_lat, clean_bench.agent.jerk_lat, EXACT_TOLERANCE);

    EXPECT_NEAR(clean_bench.agent.jerk_long, 1.25f / HUMAN_TEST_DT_SECONDS, 1e-4f);
    EXPECT_NEAR(clean_bench.agent.jerk_lat, clean_bench.agent.accel_lat / HUMAN_TEST_DT_SECONDS, 1e-4f);
    return 0;
}

static int test_human_speed_uses_trapezoidal_integration(void) {
    HumanBench bench;
    human_bench_init(&bench, 0.0f, 0.0f);
    c_set_human_input(&bench.env, 0.0f, 2.0f);

    move_human(&bench.env, 0);
    EXPECT_NEAR(bench.agent.sim_speed_signed, 0.1f, SPEED_TOLERANCE_MPS);
    EXPECT_NEAR(bench.agent.sim_x, 0.005f, ARC_TOLERANCE_M);

    move_human(&bench.env, 0);
    EXPECT_NEAR(bench.agent.sim_speed_signed, 0.3f, SPEED_TOLERANCE_MPS);
    EXPECT_NEAR(bench.agent.sim_x, 0.005f + 0.02f, ARC_TOLERANCE_M);
    return 0;
}

static int test_human_speed_clamped_to_conditioned_band(void) {
    {
        HumanBench bench;
        human_bench_init(&bench, 0.0f, HUMAN_TEST_MAX_SPEED_MPS - 0.1f);
        c_set_human_input(&bench.env, 0.0f, ACCEL_LONG_LIMIT[1]);
        for (int step = 0; step < 10; step++) {
            move_human(&bench.env, 0);
            EXPECT_TRUE(bench.agent.sim_speed_signed <= HUMAN_TEST_MAX_SPEED_MPS);
        }
        EXPECT_NEAR(bench.agent.sim_speed_signed, HUMAN_TEST_MAX_SPEED_MPS, SPEED_TOLERANCE_MPS);
    }
    {
        HumanBench bench;
        human_bench_init(&bench, 0.0f, 5.0f);
        bench.agent.reward_coefs[REWARD_COEF_SPEED] = 0.5f;
        c_set_human_input(&bench.env, 0.0f, ACCEL_LONG_LIMIT[1]);
        for (int step = 0; step < 40; step++) {
            move_human(&bench.env, 0);
        }
        EXPECT_NEAR(bench.agent.sim_speed_signed, 0.5f * HUMAN_TEST_MAX_SPEED_MPS, SPEED_TOLERANCE_MPS);
    }
    {
        HumanBench bench;
        human_bench_init(&bench, 0.0f, MAX_BACKWARD_SPEED + 0.1f);
        c_set_human_input(&bench.env, 0.0f, ACCEL_LONG_LIMIT[0]);
        for (int step = 0; step < 10; step++) {
            move_human(&bench.env, 0);
            EXPECT_TRUE(bench.agent.sim_speed_signed >= MAX_BACKWARD_SPEED);
        }
        EXPECT_NEAR(bench.agent.sim_speed_signed, MAX_BACKWARD_SPEED, SPEED_TOLERANCE_MPS);
    }
    return 0;
}

// The human path has no jerk state, so a stop clears the acceleration whatever reset_accel_on_stop says.
static int test_human_braking_to_stop_snaps_to_zero(void) {
    for (int reset_accel_on_stop = 0; reset_accel_on_stop <= 1; reset_accel_on_stop++) {
        HumanBench bench;
        human_bench_init(&bench, 0.0f, 0.1f);
        bench.env.reset_accel_on_stop = reset_accel_on_stop;
        c_set_human_input(&bench.env, 0.0f, ACCEL_LONG_LIMIT[0]);
        move_human(&bench.env, 0);
        EXPECT_NEAR(bench.agent.sim_speed_signed, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(bench.agent.sim_speed, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(bench.agent.accel_long, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(bench.agent.accel_lat, 0.0f, EXACT_TOLERANCE);
        EXPECT_TRUE(bench.agent.sim_x > 0.0f);
        EXPECT_NEAR(bench.agent.sim_x, 0.005f, ARC_TOLERANCE_M);
    }
    return 0;
}

// Pedals latch powertrain mode; a direct command must take the car back.
static int test_direct_command_overrides_the_powertrain(void) {
    HumanBench bench;
    human_bench_init(&bench, 0.0f, 5.0f);
    bench.env.human_gearbox_mode = GEARBOX_MODE_AUTOMATIC;
    bench.env.human_gear = 2;
    bench.env.human_engine_rpm = ENGINE_IDLE_RPM;

    EXPECT_EQ_INT(c_set_human_pedals(&bench.env, 0.0f, 1.0f, 0.0f, 0.0f, 0), 0);
    EXPECT_EQ_INT(bench.env.human_longitudinal_mode, HUMAN_LONGITUDINAL_POWERTRAIN);
    move_human(&bench.env, 0);
    EXPECT_TRUE(bench.agent.accel_long > 0.0f);

    EXPECT_EQ_INT(c_set_human_input(&bench.env, 0.0f, -3.0f), 0);
    EXPECT_EQ_INT(bench.env.human_longitudinal_mode, HUMAN_LONGITUDINAL_DIRECT);
    move_human(&bench.env, 0);
    EXPECT_NEAR(bench.agent.accel_long, -3.0f, EXACT_TOLERANCE);

    EXPECT_EQ_INT(c_set_human_pedals(&bench.env, 0.0f, 1.0f, 0.0f, 0.0f, 0), 0);
    EXPECT_EQ_INT(bench.env.human_longitudinal_mode, HUMAN_LONGITUDINAL_POWERTRAIN);
    move_human(&bench.env, 0);
    EXPECT_TRUE(bench.agent.accel_long > 0.0f);
    return 0;
}

// The stop reset zeroes accel_long, so the recorded jerk must be taken before it, as the jerk path does.
static int test_human_stop_reset_preserves_driver_jerk(void) {
    {
        HumanBench bench;
        human_bench_init(&bench, 0.0f, 0.1f);
        bench.env.reset_accel_on_stop = 1;
        bench.agent.accel_long = ACCEL_LONG_LIMIT[0];
        c_set_human_input(&bench.env, 0.0f, ACCEL_LONG_LIMIT[0]);
        move_human(&bench.env, 0);
        EXPECT_NEAR(bench.agent.sim_speed_signed, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(bench.agent.accel_long, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(bench.agent.jerk_long, 0.0f, EXACT_TOLERANCE);
        EXPECT_TRUE(fabsf(bench.agent.jerk_long) <= COMFORT_JERK_THRESHOLD_MPS3);
    }
    {
        HumanBench bench;
        human_bench_init(&bench, 0.0f, 0.1f);
        bench.env.reset_accel_on_stop = 1;
        c_set_human_input(&bench.env, 0.0f, ACCEL_LONG_LIMIT[0]);
        move_human(&bench.env, 0);
        EXPECT_NEAR(bench.agent.sim_speed_signed, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(bench.agent.accel_long, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(bench.agent.jerk_long, ACCEL_LONG_LIMIT[0] / HUMAN_TEST_DT_SECONDS, 1e-3f);
    }
    return 0;
}

static int test_human_reverses_from_standstill(void) {
    HumanBench bench;
    human_bench_init(&bench, 0.0f, 0.0f);
    bench.env.reset_accel_on_stop = 1;
    c_set_human_input(&bench.env, 0.0f, -2.0f);

    move_human(&bench.env, 0);
    EXPECT_NEAR(bench.agent.sim_speed_signed, -0.1f, SPEED_TOLERANCE_MPS);
    EXPECT_NEAR(bench.agent.sim_x, -0.005f, ARC_TOLERANCE_M);
    EXPECT_NEAR(bench.agent.accel_long, -2.0f, EXACT_TOLERANCE);

    move_human(&bench.env, 0);
    EXPECT_NEAR(bench.agent.sim_speed_signed, -0.3f, SPEED_TOLERANCE_MPS);

    for (int step = 0; step < 40; step++) {
        move_human(&bench.env, 0);
        EXPECT_TRUE(bench.agent.sim_speed_signed >= MAX_BACKWARD_SPEED);
    }
    EXPECT_NEAR(bench.agent.sim_speed_signed, MAX_BACKWARD_SPEED, SPEED_TOLERANCE_MPS);
    EXPECT_TRUE(bench.agent.sim_x < -1.0f);
    return 0;
}

static int test_human_grip_kinematic_follows_front_wheels(void) {
    static const float speeds_mps[] = {0.5f, 2.0f, 6.0f, 12.0f, HUMAN_TEST_MAX_SPEED_MPS};
    static const float steering_norms[] = {-1.0f, -0.4f, 0.3f, 1.0f};
    int speed_count = (int) (sizeof(speeds_mps) / sizeof(speeds_mps[0]));
    int steering_count = (int) (sizeof(steering_norms) / sizeof(steering_norms[0]));
    for (int s = 0; s < speed_count; s++) {
        for (int w = 0; w < steering_count; w++) {
            HumanBench bench;
            human_bench_init(&bench, 0.7f, speeds_mps[s]);
            c_set_human_input(&bench.env, steering_norms[w], 0.0f);
            move_human(&bench.env, 0);
            float expected_curvature = commanded_curvature_per_m(&bench.agent, steering_norms[w]);
            EXPECT_NEAR(achieved_curvature_per_m(&bench.agent), expected_curvature, CURVATURE_TOLERANCE_PER_M);
            float speed = bench.agent.sim_speed_signed;
            EXPECT_NEAR(bench.agent.accel_lat, speed * speed * expected_curvature, ACCEL_TOLERANCE_MPS2);
        }
    }
    return 0;
}

static int test_human_grip_modes_agree_below_tire_limit(void) {
    static const float steering_norms[] = {-1.0f, 1.0f};
    for (int w = 0; w < 2; w++) {
        for (int speed_tenths = 5; speed_tenths <= 200; speed_tenths += 5) {
            float speed_mps = (float) speed_tenths / 10.0f;
            HumanBench kinematic_bench, realistic_bench;
            human_bench_init(&kinematic_bench, 0.0f, speed_mps);
            human_bench_init(&realistic_bench, 0.0f, speed_mps);
            realistic_bench.env.human_grip_mode = GRIP_MODE_REALISTIC;
            c_set_human_input(&kinematic_bench.env, steering_norms[w], 0.0f);
            c_set_human_input(&realistic_bench.env, steering_norms[w], 0.0f);
            move_human(&kinematic_bench.env, 0);
            move_human(&realistic_bench.env, 0);
            float demand_mps2
                = speed_mps * speed_mps * commanded_curvature_per_m(&kinematic_bench.agent, steering_norms[w]);
            if (fabsf(demand_mps2) > HUMAN_TIRE_GRIP_MPS2) {
                continue;
            }
            EXPECT_NEAR(realistic_bench.agent.sim_x, kinematic_bench.agent.sim_x, EXACT_TOLERANCE);
            EXPECT_NEAR(realistic_bench.agent.sim_y, kinematic_bench.agent.sim_y, EXACT_TOLERANCE);
            EXPECT_NEAR(realistic_bench.agent.sim_heading, kinematic_bench.agent.sim_heading, EXACT_TOLERANCE);
            EXPECT_NEAR(realistic_bench.agent.yaw_rate, kinematic_bench.agent.yaw_rate, EXACT_TOLERANCE);
            EXPECT_NEAR(realistic_bench.agent.accel_lat, kinematic_bench.agent.accel_lat, EXACT_TOLERANCE);
        }
    }
    return 0;
}

static int test_human_grip_realistic_understeers_above_tire_limit(void) {
    static const float steering_norms[] = {-1.0f, 1.0f};
    for (int w = 0; w < 2; w++) {
        for (int speed_tenths = 5; speed_tenths <= 200; speed_tenths += 5) {
            float speed_mps = (float) speed_tenths / 10.0f;
            HumanBench kinematic_bench, realistic_bench;
            human_bench_init(&kinematic_bench, 0.0f, speed_mps);
            human_bench_init(&realistic_bench, 0.0f, speed_mps);
            realistic_bench.env.human_grip_mode = GRIP_MODE_REALISTIC;
            c_set_human_input(&kinematic_bench.env, steering_norms[w], 0.0f);
            c_set_human_input(&realistic_bench.env, steering_norms[w], 0.0f);
            move_human(&kinematic_bench.env, 0);
            move_human(&realistic_bench.env, 0);

            float commanded_curvature = commanded_curvature_per_m(&kinematic_bench.agent, steering_norms[w]);
            float realistic_curvature = achieved_curvature_per_m(&realistic_bench.agent);
            EXPECT_TRUE(fabsf(realistic_bench.agent.accel_lat) <= HUMAN_TIRE_GRIP_MPS2 + ACCEL_TOLERANCE_MPS2);
            EXPECT_TRUE(realistic_curvature * commanded_curvature > 0.0f);
            EXPECT_NEAR(realistic_bench.agent.steering_angle, kinematic_bench.agent.steering_angle, EXACT_TOLERANCE);

            float demand_mps2 = speed_mps * speed_mps * commanded_curvature;
            if (fabsf(demand_mps2) <= HUMAN_TIRE_GRIP_MPS2) {
                continue;
            }
            EXPECT_TRUE(fabsf(realistic_curvature) < fabsf(commanded_curvature));
            EXPECT_TRUE(fabsf(realistic_curvature) < fabsf(achieved_curvature_per_m(&kinematic_bench.agent)));
            EXPECT_NEAR(fabsf(realistic_bench.agent.accel_lat), HUMAN_TIRE_GRIP_MPS2, ACCEL_TOLERANCE_MPS2);
        }
    }
    return 0;
}

static int test_human_grip_realistic_safe_at_standstill(void) {
    HumanBench bench;
    human_bench_init(&bench, 0.0f, 0.0f);
    bench.env.human_grip_mode = GRIP_MODE_REALISTIC;
    c_set_human_input(&bench.env, 1.0f, 0.0f);
    move_human(&bench.env, 0);
    EXPECT_FINITE(bench.agent.sim_x);
    EXPECT_FINITE(bench.agent.sim_heading);
    EXPECT_NEAR(bench.agent.accel_lat, 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(bench.agent.yaw_rate, 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(bench.agent.sim_heading, 0.0f, EXACT_TOLERANCE);
    return 0;
}

static int test_human_stopped_and_removed_agents(void) {
    {
        HumanBench bench;
        human_bench_init(&bench, 0.4f, 9.0f);
        bench.agent.stopped = 1;
        bench.agent.steering_angle = 0.3f;
        c_set_human_input(&bench.env, 1.0f, ACCEL_LONG_LIMIT[1]);
        move_human(&bench.env, 0);
        EXPECT_NEAR(bench.agent.sim_speed, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(bench.agent.steering_angle, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(bench.agent.sim_x, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(bench.agent.sim_y, 0.0f, EXACT_TOLERANCE);
    }
    {
        HumanBench bench;
        human_bench_init(&bench, 0.4f, 9.0f);
        bench.agent.removed = 1;
        c_set_human_input(&bench.env, 1.0f, ACCEL_LONG_LIMIT[1]);
        move_human(&bench.env, 0);
        EXPECT_EQ_INT(bench.agent.sim_valid, 0);
        EXPECT_NEAR(bench.agent.sim_x, INVALID_POSITION, 1e-3f);
    }
    return 0;
}

static int test_bicycle_arc_straight_line_fallback(void) {
    const float speed_mps = 10.0f;
    const float distance_m = speed_mps * HUMAN_TEST_DT_SECONDS;
    {
        Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
        integrate_bicycle_arc(&agent, 0.0f, speed_mps, speed_mps, HUMAN_TEST_DT_SECONDS);
        EXPECT_NEAR(agent.sim_x, distance_m, EXACT_TOLERANCE);
        EXPECT_NEAR(agent.sim_y, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(agent.sim_heading, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(agent.yaw_rate, 0.0f, EXACT_TOLERANCE);
    }
    {
        float below_epsilon_curvature = 0.5f * BICYCLE_ARC_EPSILON;
        Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
        integrate_bicycle_arc(&agent, below_epsilon_curvature, speed_mps, speed_mps, HUMAN_TEST_DT_SECONDS);
        EXPECT_NEAR(agent.sim_x, distance_m, EXACT_TOLERANCE);
        EXPECT_NEAR(agent.sim_y, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(agent.sim_heading, distance_m * below_epsilon_curvature, EXACT_TOLERANCE);
    }
    {
        float tiny_distance_m = 1e-6f;
        float tiny_speed_mps = tiny_distance_m / HUMAN_TEST_DT_SECONDS;
        float large_curvature = 0.2f;
        Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
        integrate_bicycle_arc(&agent, large_curvature, tiny_speed_mps, tiny_speed_mps, HUMAN_TEST_DT_SECONDS);
        EXPECT_NEAR(agent.sim_x, tiny_distance_m, EXACT_TOLERANCE);
        EXPECT_NEAR(agent.sim_y, 0.0f, EXACT_TOLERANCE);
    }
    return 0;
}

static int test_bicycle_arc_lands_on_geometric_circle(void) {
    static const float curvatures_per_m[] = {-0.3f, -0.05f, 0.02f, 0.18f};
    static const float headings_rad[] = {0.0f, 1.1f, -2.4f};
    static const float speeds_mps[] = {-2.0f, 3.0f, 14.0f};
    int curvature_count = (int) (sizeof(curvatures_per_m) / sizeof(curvatures_per_m[0]));
    int heading_count = (int) (sizeof(headings_rad) / sizeof(headings_rad[0]));
    int speed_count = (int) (sizeof(speeds_mps) / sizeof(speeds_mps[0]));
    for (int c = 0; c < curvature_count; c++) {
        for (int h = 0; h < heading_count; h++) {
            for (int s = 0; s < speed_count; s++) {
                float curvature = curvatures_per_m[c];
                float heading_rad = headings_rad[h];
                float speed_mps = speeds_mps[s];
                float radius_m = 1.0f / curvature;
                float centre_x = -radius_m * sinf(heading_rad);
                float centre_y = radius_m * cosf(heading_rad);
                float theta_rad = speed_mps * HUMAN_TEST_DT_SECONDS * curvature;

                Agent agent = drive_test_agent(0.0f, 0.0f, heading_rad);
                integrate_bicycle_arc(&agent, curvature, speed_mps, speed_mps, HUMAN_TEST_DT_SECONDS);

                float offset_x = agent.sim_x - centre_x;
                float offset_y = agent.sim_y - centre_y;
                EXPECT_NEAR(sqrtf(offset_x * offset_x + offset_y * offset_y), fabsf(radius_m), ARC_TOLERANCE_M);
                EXPECT_NEAR(agent.sim_x, centre_x + radius_m * sinf(heading_rad + theta_rad), ARC_TOLERANCE_M);
                EXPECT_NEAR(agent.sim_y, centre_y - radius_m * cosf(heading_rad + theta_rad), ARC_TOLERANCE_M);
                EXPECT_NEAR(agent.sim_heading, normalize_heading(heading_rad + theta_rad), 1e-5f);
                EXPECT_NEAR(agent.yaw_rate, speed_mps * curvature, 1e-5f);
            }
        }
    }
    return 0;
}

static int test_bicycle_arc_closes_a_full_circle(void) {
    const float speed_mps = 10.0f;
    const float distance_per_step_m = speed_mps * HUMAN_TEST_DT_SECONDS;
    float theta_per_step_rad = 2.0f * (float) M_PI / (float) CIRCLE_STEP_COUNT;
    float curvature_per_m = theta_per_step_rad / distance_per_step_m;
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    for (int step = 0; step < CIRCLE_STEP_COUNT; step++) {
        integrate_bicycle_arc(&agent, curvature_per_m, speed_mps, speed_mps, HUMAN_TEST_DT_SECONDS);
    }
    EXPECT_NEAR(agent.sim_x, 0.0f, CIRCLE_CLOSURE_TOLERANCE_M);
    EXPECT_NEAR(agent.sim_y, 0.0f, CIRCLE_CLOSURE_TOLERANCE_M);
    EXPECT_NEAR(normalize_heading(agent.sim_heading), 0.0f, 1e-4f);
    return 0;
}

static int test_bicycle_arc_matches_pre_extraction_code(void) {
    static const float curvatures_per_m[] = {0.0f, 1e-6f, -1e-6f, 0.004f, -0.12f, 0.37f};
    static const float headings_rad[] = {0.0f, 0.9f, -1.7f, 3.0f};
    int curvature_count = (int) (sizeof(curvatures_per_m) / sizeof(curvatures_per_m[0]));
    int heading_count = (int) (sizeof(headings_rad) / sizeof(headings_rad[0]));
    for (int c = 0; c < curvature_count; c++) {
        for (int h = 0; h < heading_count; h++) {
            for (int speed_tenths = -20; speed_tenths <= 200; speed_tenths += 7) {
                float speed_new_mps = (float) speed_tenths / 10.0f;
                float speed_old_mps = speed_new_mps - 0.25f;
                Agent extracted = drive_test_agent(3.0f, -4.0f, headings_rad[h]);
                Agent reference = drive_test_agent(3.0f, -4.0f, headings_rad[h]);
                integrate_bicycle_arc(
                    &extracted,
                    curvatures_per_m[c],
                    speed_new_mps,
                    speed_old_mps,
                    HUMAN_TEST_DT_SECONDS);
                reference_bicycle_arc(
                    &reference,
                    curvatures_per_m[c],
                    speed_new_mps,
                    speed_old_mps,
                    HUMAN_TEST_DT_SECONDS);
                EXPECT_NEAR(extracted.sim_x, reference.sim_x, EXACT_TOLERANCE);
                EXPECT_NEAR(extracted.sim_y, reference.sim_y, EXACT_TOLERANCE);
                EXPECT_NEAR(extracted.sim_heading, reference.sim_heading, EXACT_TOLERANCE);
                EXPECT_NEAR(extracted.sim_vx, reference.sim_vx, EXACT_TOLERANCE);
                EXPECT_NEAR(extracted.sim_vy, reference.sim_vy, EXACT_TOLERANCE);
                EXPECT_NEAR(extracted.yaw_rate, reference.yaw_rate, EXACT_TOLERANCE);
                EXPECT_NEAR(extracted.sim_speed_signed, reference.sim_speed_signed, EXACT_TOLERANCE);
            }
        }
    }
    EXPECT_NEAR(BICYCLE_ARC_EPSILON, 1e-5f, EXACT_TOLERANCE);
    return 0;
}

static int test_human_agent_defaults_to_nobody_driving(void) {
    Drive env = human_full_env(
        drive_carla_map(),
        SIMULATION_MODE_GIGAFLOW,
        HUMAN_TEST_AGENT_COUNT,
        INFRACTION_BEHAVIOR_IGNORE,
        101);
    EXPECT_EQ_INT(env.human_agent_idx, -1);
    EXPECT_EQ_INT(env.human_grip_mode, GRIP_MODE_KINEMATIC);
    for (int i = 0; i < env.active_agent_count; i++) {
        EXPECT_EQ_INT(env.agents[env.active_agent_indices[i]].ignores_infraction_behavior, 0);
        EXPECT_TRUE(env.agents[env.active_agent_indices[i]].controller != CONTROLLER_HUMAN);
    }
    free_allocated(&env);
    return 0;
}

static int test_human_agent_selection_handover(void) {
    Drive env = human_full_env(
        drive_carla_map(),
        SIMULATION_MODE_GIGAFLOW,
        HUMAN_TEST_AGENT_COUNT,
        INFRACTION_BEHAVIOR_IGNORE,
        103);
    int human_agent_idx = env.active_agent_indices[0];
    Agent *agent = &env.agents[human_agent_idx];
    agent->steering_angle = 0.4f;
    agent->accel_long = 1.1f;
    agent->accel_lat = -2.2f;
    agent->jerk_long = 7.0f;
    agent->jerk_lat = -7.0f;
    c_set_human_input(&env, 1.0f, 2.0f);

    EXPECT_EQ_INT(c_select_human_agent(&env, human_agent_idx), 0);
    EXPECT_EQ_INT(env.human_agent_idx, human_agent_idx);
    EXPECT_EQ_INT(agent->controller, CONTROLLER_HUMAN);
    EXPECT_EQ_INT(agent->ignores_infraction_behavior, 1);
    EXPECT_NEAR(env.human_steering_cmd_norm, 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(env.human_accel_long_cmd_mps2, 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(agent->steering_angle, 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(agent->accel_long, 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(agent->accel_lat, 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(agent->jerk_long, 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(agent->jerk_lat, 0.0f, EXACT_TOLERANCE);

    EXPECT_EQ_INT(c_select_human_agent(&env, -1), 0);
    EXPECT_EQ_INT(env.human_agent_idx, -1);
    EXPECT_EQ_INT(agent->controller, CONTROLLER_POLICY);
    EXPECT_EQ_INT(agent->ignores_infraction_behavior, 0);
    free_allocated(&env);
    return 0;
}

static int test_human_agent_selection_restores_previous_controller(void) {
    Drive env = human_full_env(
        drive_carla_map(),
        SIMULATION_MODE_GIGAFLOW,
        HUMAN_TEST_AGENT_COUNT,
        INFRACTION_BEHAVIOR_IGNORE,
        107);
    int first_idx = env.active_agent_indices[0];
    int second_idx = env.active_agent_indices[1];
    env.agents[first_idx].controller = CONTROLLER_STATIC;
    env.agents[second_idx].controller = CONTROLLER_IDM;

    EXPECT_EQ_INT(c_select_human_agent(&env, first_idx), 0);
    EXPECT_EQ_INT(env.agents[first_idx].controller, CONTROLLER_HUMAN);

    EXPECT_EQ_INT(c_select_human_agent(&env, second_idx), 0);
    EXPECT_EQ_INT(env.agents[first_idx].controller, CONTROLLER_STATIC);
    EXPECT_EQ_INT(env.agents[first_idx].ignores_infraction_behavior, 0);
    EXPECT_EQ_INT(env.agents[second_idx].controller, CONTROLLER_HUMAN);

    EXPECT_EQ_INT(c_select_human_agent(&env, -1), 0);
    EXPECT_EQ_INT(env.agents[second_idx].controller, CONTROLLER_IDM);
    EXPECT_EQ_INT(env.agents[second_idx].ignores_infraction_behavior, 0);
    free_allocated(&env);
    return 0;
}

static int test_human_agent_selection_rejects_out_of_range(void) {
    Drive env = human_full_env(
        drive_carla_map(),
        SIMULATION_MODE_GIGAFLOW,
        HUMAN_TEST_AGENT_COUNT,
        INFRACTION_BEHAVIOR_IGNORE,
        109);
    int human_agent_idx = env.active_agent_indices[0];
    c_select_human_agent(&env, human_agent_idx);
    c_set_human_input(&env, 0.5f, 1.0f);

    EXPECT_EQ_INT(c_select_human_agent(&env, env.num_total_agents), -1);
    EXPECT_EQ_INT(c_select_human_agent(&env, env.num_total_agents + 100), -1);
    EXPECT_EQ_INT(c_select_human_agent(&env, -2), -1);
    EXPECT_EQ_INT(env.human_agent_idx, human_agent_idx);
    EXPECT_EQ_INT(env.agents[human_agent_idx].controller, CONTROLLER_HUMAN);
    EXPECT_NEAR(env.human_steering_cmd_norm, 0.5f, EXACT_TOLERANCE);
    EXPECT_NEAR(env.human_accel_long_cmd_mps2, 1.0f, EXACT_TOLERANCE);
    free_allocated(&env);
    return 0;
}

static int test_human_input_rejects_non_finite(void) {
    Drive env = human_full_env(
        drive_carla_map(),
        SIMULATION_MODE_GIGAFLOW,
        HUMAN_TEST_AGENT_COUNT,
        INFRACTION_BEHAVIOR_IGNORE,
        113);
    EXPECT_EQ_INT(c_set_human_input(&env, 0.25f, 1.5f), 0);
    EXPECT_NEAR(env.human_steering_cmd_norm, 0.25f, EXACT_TOLERANCE);
    EXPECT_NEAR(env.human_accel_long_cmd_mps2, 1.5f, EXACT_TOLERANCE);

    EXPECT_EQ_INT(c_set_human_input(&env, NAN, 1.0f), -1);
    EXPECT_EQ_INT(c_set_human_input(&env, 0.0f, NAN), -1);
    EXPECT_EQ_INT(c_set_human_input(&env, INFINITY, 1.0f), -1);
    EXPECT_EQ_INT(c_set_human_input(&env, 0.0f, -INFINITY), -1);
    EXPECT_NEAR(env.human_steering_cmd_norm, 0.25f, EXACT_TOLERANCE);
    EXPECT_NEAR(env.human_accel_long_cmd_mps2, 1.5f, EXACT_TOLERANCE);

    EXPECT_EQ_INT(c_set_human_input(&env, 9.0f, 1000.0f), 0);
    EXPECT_NEAR(env.human_steering_cmd_norm, 1.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(env.human_accel_long_cmd_mps2, 1000.0f, EXACT_TOLERANCE);
    EXPECT_EQ_INT(c_set_human_input(&env, -9.0f, 0.0f), 0);
    EXPECT_NEAR(env.human_steering_cmd_norm, -1.0f, EXACT_TOLERANCE);
    free_allocated(&env);
    return 0;
}

static int test_human_grip_mode_rejects_unknown(void) {
    Drive env = human_full_env(
        drive_carla_map(),
        SIMULATION_MODE_GIGAFLOW,
        HUMAN_TEST_AGENT_COUNT,
        INFRACTION_BEHAVIOR_IGNORE,
        127);
    EXPECT_EQ_INT(c_set_human_grip_mode(&env, GRIP_MODE_REALISTIC), 0);
    EXPECT_EQ_INT(env.human_grip_mode, GRIP_MODE_REALISTIC);
    EXPECT_EQ_INT(c_set_human_grip_mode(&env, -1), -1);
    EXPECT_EQ_INT(c_set_human_grip_mode(&env, 2), -1);
    EXPECT_EQ_INT(c_set_human_grip_mode(&env, 9999), -1);
    EXPECT_EQ_INT(env.human_grip_mode, GRIP_MODE_REALISTIC);
    EXPECT_EQ_INT(c_set_human_grip_mode(&env, GRIP_MODE_KINEMATIC), 0);
    EXPECT_EQ_INT(env.human_grip_mode, GRIP_MODE_KINEMATIC);
    free_allocated(&env);
    return 0;
}

static int test_human_car_survives_reset(void) {
    {
        Drive env = human_full_env(
            drive_carla_map(),
            SIMULATION_MODE_GIGAFLOW,
            HUMAN_TEST_AGENT_COUNT,
            INFRACTION_BEHAVIOR_IGNORE,
            131);
        int human_agent_idx = env.active_agent_indices[0];
        c_select_human_agent(&env, human_agent_idx);
        c_set_human_input(&env, 0.6f, 1.0f);
        c_step(&env);
        env.timestep = 5;
        c_reset(&env);
        EXPECT_EQ_INT(env.human_agent_idx, human_agent_idx);
        EXPECT_EQ_INT(env.agents[human_agent_idx].controller, CONTROLLER_HUMAN);
        EXPECT_EQ_INT(env.agents[human_agent_idx].ignores_infraction_behavior, 1);
        EXPECT_NEAR(env.human_steering_cmd_norm, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(env.human_accel_long_cmd_mps2, 0.0f, EXACT_TOLERANCE);
        free_allocated(&env);
    }
    {
        Drive env = human_full_env(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, INFRACTION_BEHAVIOR_IGNORE, 137);
        int human_agent_idx = env.active_agent_indices[0];
        c_select_human_agent(&env, human_agent_idx);
        c_set_human_input(&env, 0.6f, 1.0f);
        c_step(&env);
        env.timestep = 5;
        c_reset(&env);
        EXPECT_EQ_INT(env.human_agent_idx, human_agent_idx);
        EXPECT_EQ_INT(env.agents[human_agent_idx].controller, CONTROLLER_HUMAN);
        EXPECT_EQ_INT(env.agents[human_agent_idx].ignores_infraction_behavior, 1);
        EXPECT_NEAR(env.human_steering_cmd_norm, 0.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(env.human_accel_long_cmd_mps2, 0.0f, EXACT_TOLERANCE);
        free_allocated(&env);
    }
    {
        Drive env = human_full_env(
            drive_carla_map(),
            SIMULATION_MODE_GIGAFLOW,
            HUMAN_TEST_AGENT_COUNT,
            INFRACTION_BEHAVIOR_IGNORE,
            139);
        int human_agent_idx = env.active_agent_indices[0];
        c_select_human_agent(&env, human_agent_idx);
        c_set_human_input(&env, 0.6f, 1.0f);
        EXPECT_EQ_INT(env.timestep, 0);
        c_reset(&env);
        EXPECT_EQ_INT(env.human_agent_idx, human_agent_idx);
        EXPECT_EQ_INT(env.agents[human_agent_idx].controller, CONTROLLER_HUMAN);
        EXPECT_NEAR(env.human_steering_cmd_norm, 0.0f, EXACT_TOLERANCE);
        free_allocated(&env);
    }
    return 0;
}

static int test_human_coefs_pinned_to_a_fixed_profile(void) {
    Drive env = human_randomized_env(HUMAN_TEST_AGENT_COUNT, 211);
    int human_agent_idx = env.active_agent_indices[0];
    const Agent *human = &env.agents[human_agent_idx];

    int randomization_is_live = 0;
    for (int i = 1; i < env.active_agent_count; i++) {
        const Agent *other = &env.agents[env.active_agent_indices[i]];
        randomization_is_live |= (other->reward_coefs[REWARD_COEF_ACC] != 1.0f);
    }
    EXPECT_EQ_INT(randomization_is_live, 1);

    EXPECT_EQ_INT(c_select_human_agent(&env, human_agent_idx), 0);
    float expected_speed_coef = env.human_max_speed_mps / env.base_max_speed_mps;
    EXPECT_NEAR(human->reward_coefs[REWARD_COEF_SPEED], expected_speed_coef, EXACT_TOLERANCE);
    EXPECT_NEAR(human->reward_coefs[REWARD_COEF_ACC], 1.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(human->reward_coefs[REWARD_COEF_THROTTLE], 1.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(human->reward_coefs[REWARD_COEF_STEER], 1.0f, EXACT_TOLERANCE);

    for (int episode = 0; episode < 3; episode++) {
        env.timestep = 5;
        c_reset(&env);
        EXPECT_EQ_INT(env.human_agent_idx, human_agent_idx);
        EXPECT_NEAR(human->reward_coefs[REWARD_COEF_SPEED], expected_speed_coef, EXACT_TOLERANCE);
        EXPECT_NEAR(human->reward_coefs[REWARD_COEF_ACC], 1.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(human->reward_coefs[REWARD_COEF_THROTTLE], 1.0f, EXACT_TOLERANCE);
        EXPECT_NEAR(human->reward_coefs[REWARD_COEF_STEER], 1.0f, EXACT_TOLERANCE);
    }
    free_allocated(&env);
    return 0;
}

static int test_human_max_speed_rejects_values_outside_trained_band(void) {
    Drive env = human_full_env(
        drive_carla_map(),
        SIMULATION_MODE_GIGAFLOW,
        HUMAN_TEST_AGENT_COUNT,
        INFRACTION_BEHAVIOR_IGNORE,
        167);
    float lowest_mps = env.base_max_speed_mps / env.conditioning_speed_scale;
    float highest_mps = env.base_max_speed_mps * env.conditioning_speed_scale;

    EXPECT_EQ_INT(c_set_human_max_speed(&env, lowest_mps), 0);
    EXPECT_EQ_INT(c_set_human_max_speed(&env, highest_mps), 0);
    EXPECT_EQ_INT(c_set_human_max_speed(&env, env.base_max_speed_mps), 0);
    EXPECT_NEAR(env.human_max_speed_mps, env.base_max_speed_mps, EXACT_TOLERANCE);

    EXPECT_EQ_INT(c_set_human_max_speed(&env, lowest_mps - 0.5f), -1);
    EXPECT_EQ_INT(c_set_human_max_speed(&env, highest_mps + 0.5f), -1);
    EXPECT_EQ_INT(c_set_human_max_speed(&env, 0.0f), -1);
    EXPECT_EQ_INT(c_set_human_max_speed(&env, -5.0f), -1);
    EXPECT_EQ_INT(c_set_human_max_speed(&env, NAN), -1);
    EXPECT_EQ_INT(c_set_human_max_speed(&env, INFINITY), -1);
    EXPECT_NEAR(env.human_max_speed_mps, env.base_max_speed_mps, EXACT_TOLERANCE);
    free_allocated(&env);
    return 0;
}

static int test_human_max_speed_drives_the_speed_cap(void) {
    Drive env = human_full_env(
        drive_carla_map(),
        SIMULATION_MODE_GIGAFLOW,
        HUMAN_TEST_AGENT_COUNT,
        INFRACTION_BEHAVIOR_IGNORE,
        173);
    int human_agent_idx = env.active_agent_indices[0];
    Agent *human = &env.agents[human_agent_idx];
    float capped_speed_mps = env.base_max_speed_mps / env.conditioning_speed_scale;

    EXPECT_EQ_INT(c_set_human_max_speed(&env, capped_speed_mps), 0);
    EXPECT_EQ_INT(c_select_human_agent(&env, human_agent_idx), 0);
    EXPECT_NEAR(human->reward_coefs[REWARD_COEF_SPEED], capped_speed_mps / env.base_max_speed_mps, EXACT_TOLERANCE);

    c_set_human_input(&env, 0.0f, ACCEL_LONG_LIMIT[1]);
    for (int step = 0; step < 200; step++) {
        move_human(&env, human_agent_idx);
        EXPECT_TRUE(human->sim_speed_signed <= capped_speed_mps + SPEED_TOLERANCE_MPS);
    }
    EXPECT_NEAR(human->sim_speed_signed, capped_speed_mps, SPEED_TOLERANCE_MPS);

    EXPECT_EQ_INT(c_set_human_max_speed(&env, env.base_max_speed_mps), 0);
    EXPECT_NEAR(human->reward_coefs[REWARD_COEF_SPEED], 1.0f, EXACT_TOLERANCE);
    for (int step = 0; step < 200; step++) {
        move_human(&env, human_agent_idx);
    }
    EXPECT_NEAR(human->sim_speed_signed, env.base_max_speed_mps, SPEED_TOLERANCE_MPS);
    free_allocated(&env);
    return 0;
}

static int test_infraction_behavior_skips_flagged_agent(void) {
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    agent.ignores_infraction_behavior = 1;
    apply_infraction_behavior(&agent, INFRACTION_BEHAVIOR_STOP);
    apply_infraction_behavior(&agent, INFRACTION_BEHAVIOR_REMOVE);
    EXPECT_EQ_INT(agent.stopped, 0);
    EXPECT_EQ_INT(agent.removed, 0);

    agent.ignores_infraction_behavior = 0;
    apply_infraction_behavior(&agent, INFRACTION_BEHAVIOR_STOP);
    EXPECT_EQ_INT(agent.stopped, 1);
    apply_infraction_behavior(&agent, INFRACTION_BEHAVIOR_REMOVE);
    EXPECT_EQ_INT(agent.removed, 1);
    return 0;
}

static int test_human_collision_scored_but_not_enforced(void) {
    {
        Drive env = human_full_env(
            drive_carla_map(),
            SIMULATION_MODE_GIGAFLOW,
            HUMAN_TEST_AGENT_COUNT,
            INFRACTION_BEHAVIOR_STOP,
            139);
        int human_agent_idx = env.active_agent_indices[0];
        int partner_agent_idx = env.active_agent_indices[1];
        c_select_human_agent(&env, human_agent_idx);
        place_agent_ahead_of(&env.agents[partner_agent_idx], &env.agents[human_agent_idx], COLLISION_OVERLAP_OFFSET_M);

        compute_metrics(&env, human_agent_idx, 0);
        compute_metrics(&env, partner_agent_idx, 1);

        EXPECT_NEAR(env.agents[human_agent_idx].metrics_array[COLLISION_IDX], 1.0f, EXACT_TOLERANCE);
        EXPECT_EQ_INT(env.agents[human_agent_idx].stopped, 0);
        EXPECT_EQ_INT(env.agents[human_agent_idx].removed, 0);
        EXPECT_TRUE(env.agents[human_agent_idx].first_collision_partner_idx != -1);
        EXPECT_NEAR(env.agents[partner_agent_idx].metrics_array[COLLISION_IDX], 1.0f, EXACT_TOLERANCE);
        EXPECT_EQ_INT(env.agents[partner_agent_idx].stopped, 1);
        free_allocated(&env);
    }
    {
        Drive env = human_full_env(
            drive_carla_map(),
            SIMULATION_MODE_GIGAFLOW,
            HUMAN_TEST_AGENT_COUNT,
            INFRACTION_BEHAVIOR_REMOVE,
            139);
        int human_agent_idx = env.active_agent_indices[0];
        int partner_agent_idx = env.active_agent_indices[1];
        c_select_human_agent(&env, human_agent_idx);
        place_agent_ahead_of(&env.agents[partner_agent_idx], &env.agents[human_agent_idx], COLLISION_OVERLAP_OFFSET_M);

        compute_metrics(&env, human_agent_idx, 0);
        compute_metrics(&env, partner_agent_idx, 1);

        EXPECT_NEAR(env.agents[human_agent_idx].metrics_array[COLLISION_IDX], 1.0f, EXACT_TOLERANCE);
        EXPECT_EQ_INT(env.agents[human_agent_idx].removed, 0);
        EXPECT_EQ_INT(env.agents[partner_agent_idx].removed, 1);
        free_allocated(&env);
    }
    return 0;
}

static int test_human_agent_is_a_full_participant(void) {
    Drive env = human_full_env(
        drive_carla_map(),
        SIMULATION_MODE_GIGAFLOW,
        HUMAN_TEST_AGENT_COUNT,
        INFRACTION_BEHAVIOR_IGNORE,
        149);
    int human_log_idx = 0;
    int human_agent_idx = env.active_agent_indices[human_log_idx];
    int observer_log_idx = 1;
    int observer_agent_idx = env.active_agent_indices[observer_log_idx];
    c_select_human_agent(&env, human_agent_idx);

    int found_in_active_list = 0;
    for (int i = 0; i < env.active_agent_count; i++) {
        found_in_active_list |= (env.active_agent_indices[i] == human_agent_idx);
    }
    EXPECT_EQ_INT(found_in_active_list, 1);

    place_agent_ahead_of(&env.agents[human_agent_idx], &env.agents[observer_agent_idx], PARTNER_OFFSET_AHEAD_M);
    c_set_human_input(&env, 0.0f, ACCEL_LONG_LIMIT[1]);
    c_step(&env);

    int observation_size = compute_observation_size(&env);
    EXPECT_TRUE(drive_all_finite(&env.observations[human_log_idx * observation_size], observation_size));
    EXPECT_FINITE(env.rewards[human_log_idx]);
    EXPECT_TRUE(env.logs[human_log_idx].episode_length > 0.0f);
    EXPECT_NEAR(env.agents[human_agent_idx].metrics_array[OFFROAD_IDX], 0.0f, EXACT_TOLERANCE);

    place_agent_ahead_of(&env.agents[human_agent_idx], &env.agents[observer_agent_idx], PARTNER_OFFSET_AHEAD_M);
    compute_observations(&env);
    int partner_base = EGO_FEATURES + env.num_goals * GOAL_FEATURES;
    const float *observer_obs = &env.observations[observer_log_idx * observation_size];
    float expected_rel_x = PARTNER_OFFSET_AHEAD_M / env.obs_norm_xy_offset_m;
    float expected_width = env.agents[human_agent_idx].sim_width / env.obs_norm_veh_width_m;
    int human_slot_found = 0;
    for (int slot = 0; slot < env.obs_slots_partners_n; slot++) {
        const float *row = &observer_obs[partner_base + slot * partner_feature_count(&env)];
        if (fabsf(row[0] - expected_rel_x) > OBS_MATCH_TOLERANCE) {
            continue;
        }
        if (fabsf(row[1]) > OBS_MATCH_TOLERANCE || fabsf(row[4] - expected_width) > OBS_MATCH_TOLERANCE) {
            continue;
        }
        human_slot_found = 1;
    }
    EXPECT_EQ_INT(human_slot_found, 1);
    free_allocated(&env);
    return 0;
}

// c_step only moves active agents, so a non-active one must be refused rather than handed over frozen.
static int test_human_selection_rejects_a_non_active_agent(void) {
    Drive env = human_full_env(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, INFRACTION_BEHAVIOR_IGNORE, 151);
    EXPECT_TRUE(env.static_agent_count > 0);
    int static_agent_idx = env.static_agent_indices[0];
    int previous_controller = env.agents[static_agent_idx].controller;
    EXPECT_EQ_INT(env.agents[static_agent_idx].active_agent, 0);
    EXPECT_TRUE(static_agent_idx < env.num_total_agents);

    EXPECT_EQ_INT(c_select_human_agent(&env, static_agent_idx), -1);
    EXPECT_EQ_INT(env.human_agent_idx, -1);
    EXPECT_EQ_INT(env.agents[static_agent_idx].controller, previous_controller);
    EXPECT_EQ_INT(env.agents[static_agent_idx].ignores_infraction_behavior, 0);

    int active_agent_idx = env.active_agent_indices[0];
    EXPECT_EQ_INT(c_select_human_agent(&env, active_agent_idx), 0);
    c_set_human_input(&env, 0.4f, 1.0f);
    EXPECT_EQ_INT(c_select_human_agent(&env, static_agent_idx), -1);
    EXPECT_EQ_INT(env.human_agent_idx, active_agent_idx);
    EXPECT_EQ_INT(env.agents[active_agent_idx].controller, CONTROLLER_HUMAN);
    EXPECT_NEAR(env.human_steering_cmd_norm, 0.4f, EXACT_TOLERANCE);
    EXPECT_NEAR(env.human_accel_long_cmd_mps2, 1.0f, EXACT_TOLERANCE);
    free_allocated(&env);
    return 0;
}

static int test_human_reclaim_drops_a_no_longer_active_agent(void) {
    Drive env = human_full_env(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, INFRACTION_BEHAVIOR_IGNORE, 163);
    int human_agent_idx = env.active_agent_indices[0];
    c_select_human_agent(&env, human_agent_idx);
    env.agents[human_agent_idx].active_agent = 0;
    env.timestep = 5;
    c_reset(&env);
    EXPECT_EQ_INT(env.human_agent_idx, -1);
    free_allocated(&env);
    return 0;
}

// The demo wants car-sized traffic without turning on eval_mode, which changes other behaviour.
static int test_car_size_profile_narrows_spawn_sizes(void) {
    float training_min_length_m = INFINITY;
    float training_max_length_m = -INFINITY;
    for (int size_profile = VEHICLE_SIZE_PROFILE_TRAINING; size_profile <= VEHICLE_SIZE_PROFILE_CAR; size_profile++) {
        srand(311);
        Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, SIZE_PROFILE_AGENT_COUNT, 0);
        env.max_speed_mps = HUMAN_TEST_MAX_SPEED_MPS;
        env.conditioning_accel_scale = 1.0f;
        env.vehicle_size_profile = size_profile;
        allocate(&env);
        c_reset(&env);
        EXPECT_EQ_INT(env.eval_mode, 0);
        EXPECT_TRUE(env.active_agent_count > 0);

        for (int i = 0; i < env.active_agent_count; i++) {
            const Agent *agent = &env.agents[env.active_agent_indices[i]];
            EXPECT_TRUE(agent->sim_width <= agent->sim_length);
            if (size_profile == VEHICLE_SIZE_PROFILE_CAR) {
                EXPECT_TRUE(agent->sim_length >= CAR_PROFILE_MIN_LENGTH_M);
                EXPECT_TRUE(agent->sim_length <= CAR_PROFILE_MAX_LENGTH_M);
                EXPECT_TRUE(agent->sim_width >= CAR_PROFILE_MIN_WIDTH_M);
                EXPECT_TRUE(agent->sim_width <= CAR_PROFILE_MAX_WIDTH_M);
            } else {
                training_min_length_m = fminf(training_min_length_m, agent->sim_length);
                training_max_length_m = fmaxf(training_max_length_m, agent->sim_length);
            }
        }
        free_allocated(&env);
    }
    // The training profile must actually be wider, or the knob does nothing.
    EXPECT_TRUE(training_min_length_m < CAR_PROFILE_MIN_LENGTH_M || training_max_length_m > CAR_PROFILE_MAX_LENGTH_M);
    return 0;
}

static int test_human_car_moves_through_c_step(void) {
    Drive env = human_full_env(
        drive_carla_map(),
        SIMULATION_MODE_GIGAFLOW,
        HUMAN_TEST_AGENT_COUNT,
        INFRACTION_BEHAVIOR_IGNORE,
        157);
    int human_agent_idx = env.active_agent_indices[0];
    c_select_human_agent(&env, human_agent_idx);
    c_set_human_input(&env, 1.0f, ACCEL_LONG_LIMIT[1]);
    c_step(&env);
    EXPECT_NEAR(env.agents[human_agent_idx].steering_angle, -HUMAN_STEERING_LIMIT_RAD, EXACT_TOLERANCE);
    EXPECT_NEAR(env.agents[human_agent_idx].accel_long, ACCEL_LONG_LIMIT[1], EXACT_TOLERANCE);
    EXPECT_TRUE(env.agents[human_agent_idx].sim_speed > 0.0f);
    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_wheel_right_steers_the_car_right);
    RUN_TEST(test_centred_wheel_drives_straight);
    RUN_TEST(test_positive_curvature_turns_left);
    RUN_TEST(test_human_steering_is_exact_position_command);
    RUN_TEST(test_human_steering_is_symmetric_and_monotone);
    RUN_TEST(test_human_steering_has_no_rate_limit);
    RUN_TEST(test_human_steering_ratio_matches_wheel_range);
    RUN_TEST(test_human_accel_is_commanded_not_integrated);
    RUN_TEST(test_human_accel_envelope_is_independent_of_the_policy);
    RUN_TEST(test_human_jerk_recorded_but_never_consumed);
    RUN_TEST(test_human_speed_uses_trapezoidal_integration);
    RUN_TEST(test_human_speed_clamped_to_conditioned_band);
    RUN_TEST(test_human_braking_to_stop_snaps_to_zero);
    RUN_TEST(test_direct_command_overrides_the_powertrain);
    RUN_TEST(test_human_stop_reset_preserves_driver_jerk);
    RUN_TEST(test_human_reverses_from_standstill);
    RUN_TEST(test_human_grip_kinematic_follows_front_wheels);
    RUN_TEST(test_human_grip_modes_agree_below_tire_limit);
    RUN_TEST(test_human_grip_realistic_understeers_above_tire_limit);
    RUN_TEST(test_human_grip_realistic_safe_at_standstill);
    RUN_TEST(test_human_stopped_and_removed_agents);
    RUN_TEST(test_bicycle_arc_straight_line_fallback);
    RUN_TEST(test_bicycle_arc_lands_on_geometric_circle);
    RUN_TEST(test_bicycle_arc_closes_a_full_circle);
    RUN_TEST(test_bicycle_arc_matches_pre_extraction_code);
    RUN_TEST(test_human_agent_defaults_to_nobody_driving);
    RUN_TEST(test_human_agent_selection_handover);
    RUN_TEST(test_human_agent_selection_restores_previous_controller);
    RUN_TEST(test_human_agent_selection_rejects_out_of_range);
    RUN_TEST(test_human_input_rejects_non_finite);
    RUN_TEST(test_human_grip_mode_rejects_unknown);
    RUN_TEST(test_human_car_survives_reset);
    RUN_TEST(test_human_coefs_pinned_to_a_fixed_profile);
    RUN_TEST(test_human_max_speed_rejects_values_outside_trained_band);
    RUN_TEST(test_human_max_speed_drives_the_speed_cap);
    RUN_TEST(test_infraction_behavior_skips_flagged_agent);
    RUN_TEST(test_human_collision_scored_but_not_enforced);
    RUN_TEST(test_human_agent_is_a_full_participant);
    RUN_TEST(test_human_selection_rejects_a_non_active_agent);
    RUN_TEST(test_human_reclaim_drops_a_no_longer_active_agent);
    RUN_TEST(test_car_size_profile_narrows_spawn_sizes);
    RUN_TEST(test_human_car_moves_through_c_step);
    return test_summary(failures);
}
