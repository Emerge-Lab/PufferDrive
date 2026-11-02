#include "include/drive_fixture.h"
#include "include/test.h"

#define SKID_TEST_DT_SECONDS (1.0f / 60.0f)
#define SKID_TEST_MAX_SPEED_MPS 20.0f
#define RUN_UP_ACCEL_MPS2 3.0f
#define RUN_UP_STEP_COUNT 360
#define SETTLE_STEP_COUNT 60
#define MANOEUVRE_STEP_COUNT 300
#define BRAKE_MAX_STEP_COUNT 600
#define BRAKE_END_SPEED_MPS 1.0f
#define FULL_LOCK_STEP_COUNT 120
#define PARKING_SPEED_MPS 1.5f
#define SLOW_SPEED_MPS 3.0f
#define SLOW_YAW_RELATIVE_TOLERANCE 0.07f
#define STRAIGHT_POSITION_TOLERANCE_M 0.05f
#define STRAIGHT_SPEED_TOLERANCE_MPS 1e-3f
#define LOCK_POSITION_TOLERANCE_M 5e-3f
#define YAW_TOLERANCE_RPS 1e-5f
#define STEADY_CORNER_NORM -0.03f
#define STEP_STEER_NORM -0.6f
#define MODERATE_CORNER_NORM -0.015f
#define STABLE_SLIDE_DEG 2.0f
#define STEP_STEER_MAX_SLIDE_DEG 20.0f
#define SPIN_SLIDE_DEG 90.0f
#define STEADY_YAW_MIN_FRACTION 0.5f
#define LATERAL_TOLERANCE_MPS2 0.5f
#define SPEED_RATE_TOLERANCE_MPS2 0.5f
#define SPEED_RISE_TOLERANCE_MPS 1e-5f
#define SLIDE_SPEED_FLOOR_MPS 0.1f
#define SPIN_BRAKE_MPS2 -8.5f
#define DEG_PER_RAD (180.0f / (float) M_PI)
#define SLOPE_PROBE_SLIP_RAD 1e-5f
#define LARGE_SLIP_RAD 1e3f
#define PEAK_SEARCH_COUNT 2000
#define PEAK_SEARCH_MAX_SLIP_RAD 0.5f
#define TEST_PEAK_FORCE_N 6000.0f

static const float MODERATE_BRAKES_MPS2[] = {-3.0f, -6.0f};
static const float MODERATE_CORNER_NORMS[] = {MODERATE_CORNER_NORM, STEADY_CORNER_NORM};

typedef struct {
    Drive env;
    GridMap offgrid_map;
    Agent agent;
} SkidBench;

typedef struct {
    float max_slide_deg;
    float max_abs_accel_lat_mps2;
    float max_abs_speed_rate_mps2;
    int speed_rose_while_braking;
    int all_finite;
} ManoeuvreStats;

// The zeroed GridMap reports every query off-grid, which makes update_agent_z a no-op.
static void skid_bench_init(SkidBench *bench, int grip_mode, float speed_mps) {
    memset(bench, 0, sizeof(SkidBench));
    bench->agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    for (int c = 0; c < NUM_REWARD_COEFS; c++) {
        bench->agent.reward_coefs[c] = 1.0f;
    }
    bench->agent.controller = CONTROLLER_HUMAN;
    bench->agent.sim_vx = speed_mps;
    update_agent_speed(&bench->agent);
    copy_pose_to_prev(&bench->agent);
    bench->env.agents = &bench->agent;
    bench->env.num_agents = 1;
    bench->env.grid_map = &bench->offgrid_map;
    bench->env.dt = SKID_TEST_DT_SECONDS;
    bench->env.base_max_speed_mps = SKID_TEST_MAX_SPEED_MPS;
    bench->env.human_max_speed_mps = SKID_TEST_MAX_SPEED_MPS;
    bench->env.human_agent_idx = 0;
    bench->env.reset_accel_on_stop = 1;
    bench->env.human_grip_mode = grip_mode;
}

static void drive_steps(SkidBench *bench, float steering_norm, float accel_mps2, int step_count) {
    for (int step = 0; step < step_count; step++) {
        c_set_human_input(&bench->env, steering_norm, accel_mps2);
        move_human(&bench->env, 0, 0);
    }
}

static void run_up(SkidBench *bench) {
    drive_steps(bench, 0.0f, RUN_UP_ACCEL_MPS2, RUN_UP_STEP_COUNT);
}

// angle between where the car points and where it is going
static float slide_deg(const Agent *agent) {
    if (agent->sim_speed < SLIDE_SPEED_FLOOR_MPS) {
        return 0.0f;
    }
    return fabsf(normalize_heading(atan2f(agent->sim_vy, agent->sim_vx) - agent->sim_heading)) * DEG_PER_RAD;
}

static int agent_is_finite(const Agent *agent) {
    float fields[] = {agent->sim_x, agent->sim_y, agent->sim_heading, agent->sim_vx, agent->sim_vy, agent->yaw_rate,
                      agent->accel_long, agent->accel_lat, agent->jerk_long, agent->jerk_lat, agent->steering_angle};
    for (size_t k = 0; k < sizeof(fields) / sizeof(fields[0]); k++) {
        if (!isfinite(fields[k])) {
            return 0;
        }
    }
    return 1;
}

// Braking stops at BRAKE_END_SPEED_MPS: direct input past rest would drive the car backward.
static ManoeuvreStats manoeuvre(SkidBench *bench, float steering_norm, float accel_mps2, int step_count) {
    ManoeuvreStats stats = {0.0f, 0.0f, 0.0f, 0, 1};
    for (int step = 0; step < step_count; step++) {
        if (accel_mps2 < 0.0f && bench->agent.sim_speed < BRAKE_END_SPEED_MPS) {
            break;
        }
        float speed_before_mps = bench->agent.sim_speed;
        c_set_human_input(&bench->env, steering_norm, accel_mps2);
        move_human(&bench->env, 0, 0);
        const Agent *agent = &bench->agent;
        stats.all_finite &= agent_is_finite(agent);
        stats.max_slide_deg = fmaxf(stats.max_slide_deg, slide_deg(agent));
        stats.max_abs_accel_lat_mps2 = fmaxf(stats.max_abs_accel_lat_mps2, fabsf(agent->accel_lat));
        float speed_rate_mps2 = fabsf(agent->sim_speed - speed_before_mps) / bench->env.dt;
        stats.max_abs_speed_rate_mps2 = fmaxf(stats.max_abs_speed_rate_mps2, speed_rate_mps2);
        stats.speed_rose_while_braking
            |= accel_mps2 < 0.0f && agent->sim_speed > speed_before_mps + SPEED_RISE_TOLERANCE_MPS;
    }
    return stats;
}

static ManoeuvreStats braking_in_a_corner(float steering_norm, float brake_mps2) {
    SkidBench bench;
    skid_bench_init(&bench, GRIP_MODE_SKID, 0.0f);
    run_up(&bench);
    drive_steps(&bench, steering_norm, 0.0f, SETTLE_STEP_COUNT);
    return manoeuvre(&bench, steering_norm, brake_mps2, BRAKE_MAX_STEP_COUNT);
}

static int expect_stats_within_the_tyres(ManoeuvreStats stats) {
    EXPECT_TRUE(stats.all_finite);
    EXPECT_FALSE(stats.speed_rose_while_braking);
    EXPECT_TRUE(stats.max_abs_accel_lat_mps2 <= HUMAN_TIRE_GRIP_MPS2 + LATERAL_TOLERANCE_MPS2);
    return 0;
}

static int test_tire_side_force_has_the_stiffness_slope_and_the_peak(void) {
    float stiffnesses[] = {HUMAN_FRONT_CORNERING_STIFFNESS_N_PER_RAD, HUMAN_REAR_CORNERING_STIFFNESS_N_PER_RAD};
    for (int k = 0; k < 2; k++) {
        float slope = tire_side_force_n(SLOPE_PROBE_SLIP_RAD, stiffnesses[k], TEST_PEAK_FORCE_N) / SLOPE_PROBE_SLIP_RAD;
        EXPECT_NEAR(slope / stiffnesses[k], 1.0f, 1e-3f);
        float peak_n = 0.0f;
        for (int s = 1; s <= PEAK_SEARCH_COUNT; s++) {
            float slip_rad = PEAK_SEARCH_MAX_SLIP_RAD * (float) s / (float) PEAK_SEARCH_COUNT;
            peak_n = fmaxf(peak_n, tire_side_force_n(slip_rad, stiffnesses[k], TEST_PEAK_FORCE_N));
        }
        EXPECT_NEAR(peak_n / TEST_PEAK_FORCE_N, 1.0f, 1e-4f);
        float sliding_n = TEST_PEAK_FORCE_N * sinf(HUMAN_TIRE_SHAPE_FACTOR * (float) M_PI / 2.0f);
        EXPECT_NEAR(tire_side_force_n(LARGE_SLIP_RAD, stiffnesses[k], TEST_PEAK_FORCE_N), sliding_n, 1.0f);
        EXPECT_TRUE(tire_side_force_n(-0.1f, stiffnesses[k], TEST_PEAK_FORCE_N)
                    == -tire_side_force_n(0.1f, stiffnesses[k], TEST_PEAK_FORCE_N));
        EXPECT_TRUE(tire_side_force_n(0.1f, stiffnesses[k], 0.0f) == 0.0f);
        EXPECT_TRUE(tire_side_force_n(0.1f, stiffnesses[k], -1.0f) == 0.0f);
    }
    return 0;
}

static int test_grip_mode_setter_accepts_skid(void) {
    SkidBench bench;
    skid_bench_init(&bench, GRIP_MODE_KINEMATIC, 0.0f);
    EXPECT_EQ_INT(c_set_human_grip_mode(&bench.env, GRIP_MODE_SKID), 0);
    EXPECT_EQ_INT(bench.env.human_grip_mode, GRIP_MODE_SKID);
    EXPECT_EQ_INT(c_set_human_grip_mode(&bench.env, GRIP_MODE_SKID + 1), -1);
    EXPECT_EQ_INT(bench.env.human_grip_mode, GRIP_MODE_SKID);
    return 0;
}

static int test_a_straight_line_matches_the_kinematic_car(void) {
    SkidBench skid;
    SkidBench kinematic;
    skid_bench_init(&skid, GRIP_MODE_SKID, 0.0f);
    skid_bench_init(&kinematic, GRIP_MODE_KINEMATIC, 0.0f);
    run_up(&skid);
    run_up(&kinematic);
    EXPECT_NEAR(skid.agent.sim_speed, kinematic.agent.sim_speed, STRAIGHT_SPEED_TOLERANCE_MPS);
    EXPECT_NEAR(skid.agent.sim_x, kinematic.agent.sim_x, STRAIGHT_POSITION_TOLERANCE_M);
    EXPECT_TRUE(skid.agent.sim_y == 0.0f);
    EXPECT_TRUE(skid.agent.sim_heading == 0.0f);
    EXPECT_TRUE(skid.agent.yaw_rate == 0.0f);
    return 0;
}

static int test_parking_speed_full_lock_is_the_kinematic_arc(void) {
    float locks[] = {-1.0f, 1.0f};
    for (int k = 0; k < 2; k++) {
        SkidBench skid;
        SkidBench kinematic;
        skid_bench_init(&skid, GRIP_MODE_SKID, PARKING_SPEED_MPS);
        skid_bench_init(&kinematic, GRIP_MODE_KINEMATIC, PARKING_SPEED_MPS);
        for (int step = 0; step < FULL_LOCK_STEP_COUNT; step++) {
            drive_steps(&skid, locks[k], 0.0f, 1);
            drive_steps(&kinematic, locks[k], 0.0f, 1);
            EXPECT_NEAR(skid.agent.yaw_rate, kinematic.agent.yaw_rate, YAW_TOLERANCE_RPS);
            EXPECT_NEAR(slide_deg(&skid.agent), 0.0f, 1e-3f);
        }
        EXPECT_NEAR(skid.agent.sim_x, kinematic.agent.sim_x, LOCK_POSITION_TOLERANCE_M);
        EXPECT_NEAR(skid.agent.sim_y, kinematic.agent.sim_y, LOCK_POSITION_TOLERANCE_M);
    }
    return 0;
}

static int test_slow_full_lock_stays_close_to_the_kinematic_arc(void) {
    SkidBench skid;
    SkidBench kinematic;
    skid_bench_init(&skid, GRIP_MODE_SKID, SLOW_SPEED_MPS);
    skid_bench_init(&kinematic, GRIP_MODE_KINEMATIC, SLOW_SPEED_MPS);
    drive_steps(&skid, -1.0f, 0.0f, FULL_LOCK_STEP_COUNT);
    drive_steps(&kinematic, -1.0f, 0.0f, FULL_LOCK_STEP_COUNT);
    EXPECT_NEAR(skid.agent.yaw_rate / kinematic.agent.yaw_rate, 1.0f, SLOW_YAW_RELATIVE_TOLERANCE);
    return 0;
}

static int test_a_steady_corner_is_stable_and_turns_a_little_less_than_realistic(void) {
    SkidBench skid;
    SkidBench realistic;
    skid_bench_init(&skid, GRIP_MODE_SKID, 0.0f);
    skid_bench_init(&realistic, GRIP_MODE_REALISTIC, 0.0f);
    run_up(&skid);
    run_up(&realistic);
    ManoeuvreStats stats = manoeuvre(&skid, STEADY_CORNER_NORM, 0.0f, MANOEUVRE_STEP_COUNT);
    drive_steps(&realistic, STEADY_CORNER_NORM, 0.0f, MANOEUVRE_STEP_COUNT);
    EXPECT_FALSE(expect_stats_within_the_tyres(stats));
    EXPECT_TRUE(stats.max_slide_deg < STABLE_SLIDE_DEG);
    EXPECT_TRUE(skid.agent.yaw_rate > STEADY_YAW_MIN_FRACTION * realistic.agent.yaw_rate);
    EXPECT_TRUE(skid.agent.yaw_rate < realistic.agent.yaw_rate);
    return 0;
}

static int test_a_big_step_steer_understeers_without_spinning(void) {
    SkidBench skid;
    skid_bench_init(&skid, GRIP_MODE_SKID, 0.0f);
    run_up(&skid);
    ManoeuvreStats stats = manoeuvre(&skid, STEP_STEER_NORM, 0.0f, MANOEUVRE_STEP_COUNT);
    EXPECT_FALSE(expect_stats_within_the_tyres(stats));
    EXPECT_TRUE(stats.max_slide_deg > STABLE_SLIDE_DEG);
    EXPECT_TRUE(stats.max_slide_deg < STEP_STEER_MAX_SLIDE_DEG);
    float kinematic_yaw_rps = skid.agent.sim_speed * tanf(-STEP_STEER_NORM * HUMAN_STEERING_LIMIT_RAD) / skid.agent.wheelbase;
    EXPECT_TRUE(skid.agent.yaw_rate > 0.0f);
    EXPECT_TRUE(skid.agent.yaw_rate < kinematic_yaw_rps);
    return 0;
}

static int test_moderate_braking_in_a_corner_stays_stable(void) {
    for (int c = 0; c < 2; c++) {
        for (int b = 0; b < 2; b++) {
            ManoeuvreStats stats = braking_in_a_corner(MODERATE_CORNER_NORMS[c], MODERATE_BRAKES_MPS2[b]);
            EXPECT_FALSE(expect_stats_within_the_tyres(stats));
            EXPECT_TRUE(stats.max_slide_deg < STABLE_SLIDE_DEG);
            EXPECT_TRUE(stats.max_abs_speed_rate_mps2 <= -HUMAN_ACCEL_LIMIT_MPS2[0] + SPEED_RATE_TOLERANCE_MPS2);
        }
    }
    return 0;
}

static int test_emergency_braking_in_a_corner_spins_the_car(void) {
    for (int c = 0; c < 2; c++) {
        ManoeuvreStats stats = braking_in_a_corner(MODERATE_CORNER_NORMS[c], SPIN_BRAKE_MPS2);
        EXPECT_FALSE(expect_stats_within_the_tyres(stats));
        EXPECT_TRUE(stats.max_slide_deg > SPIN_SLIDE_DEG);
    }
    return 0;
}

// Contract: no speed-rate spikes at the blend band or after a spin.
static int test_a_spinning_car_never_slows_faster_than_the_hardest_brake(void) {
    for (int c = 0; c < 2; c++) {
        ManoeuvreStats stats = braking_in_a_corner(MODERATE_CORNER_NORMS[c], SPIN_BRAKE_MPS2);
        if (stats.max_abs_speed_rate_mps2 > -HUMAN_ACCEL_LIMIT_MPS2[0] + SPEED_RATE_TOLERANCE_MPS2) {
            printf("  corner norm %f: total speed fell at %f m/s^2\n", MODERATE_CORNER_NORMS[c], stats.max_abs_speed_rate_mps2);
        }
        EXPECT_TRUE(stats.max_abs_speed_rate_mps2 <= -HUMAN_ACCEL_LIMIT_MPS2[0] + SPEED_RATE_TOLERANCE_MPS2);
    }
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_tire_side_force_has_the_stiffness_slope_and_the_peak);
    RUN_TEST(test_grip_mode_setter_accepts_skid);
    RUN_TEST(test_a_straight_line_matches_the_kinematic_car);
    RUN_TEST(test_parking_speed_full_lock_is_the_kinematic_arc);
    RUN_TEST(test_slow_full_lock_stays_close_to_the_kinematic_arc);
    RUN_TEST(test_a_steady_corner_is_stable_and_turns_a_little_less_than_realistic);
    RUN_TEST(test_a_big_step_steer_understeers_without_spinning);
    RUN_TEST(test_moderate_braking_in_a_corner_stays_stable);
    RUN_TEST(test_emergency_braking_in_a_corner_spins_the_car);
    RUN_TEST(test_a_spinning_car_never_slows_faster_than_the_hardest_brake);
    return test_summary(failures);
}
