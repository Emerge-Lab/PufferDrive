#include "include/drive_fixture.h"
#include "include/test.h"

#define POWERTRAIN_TEST_DT_SECONDS 0.1f
#define POWERTRAIN_TEST_BASE_SPEED_MPS 20.0f
#define POWERTRAIN_TEST_SPEED_SCALE 1.5f
#define POWERTRAIN_TEST_AGENT_COUNT 4
#define TAPER_TEST_SPEED_MPS 10.0f
#define SETTLE_STEP_COUNT 600
#define HOLD_STEP_COUNT 300
#define EXACT_TOLERANCE 0.0f
#define SPEED_TOLERANCE_MPS 1e-6f
#define RPM_TOLERANCE 1e-2f
#define TORQUE_TOLERANCE_NM 1e-3f
#define ENGAGEMENT_TOLERANCE 1e-6f
#define ACCEL_TOLERANCE_MPS2 1e-4f

typedef struct {
    Drive env;
    GridMap offgrid_map;
    Agent agent;
} PowertrainBench;

// The zeroed GridMap reports every query off-grid, which makes update_agent_z a no-op.
static void powertrain_bench_init(PowertrainBench *bench, float speed_mps, int gearbox_mode, int gear) {
    memset(bench, 0, sizeof(PowertrainBench));
    bench->agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    bench->agent.reward_coefs[REWARD_COEF_SPEED] = POWERTRAIN_TEST_SPEED_SCALE;
    bench->agent.sim_vx = speed_mps;
    update_agent_speed(&bench->agent);
    copy_pose_to_prev(&bench->agent);
    bench->env.agents = &bench->agent;
    bench->env.num_agents = 1;
    bench->env.grid_map = &bench->offgrid_map;
    bench->env.dt = POWERTRAIN_TEST_DT_SECONDS;
    bench->env.base_max_speed_mps = POWERTRAIN_TEST_BASE_SPEED_MPS;
    bench->env.conditioning_speed_scale = POWERTRAIN_TEST_SPEED_SCALE;
    bench->env.reset_accel_on_stop = 1;
    bench->env.human_grip_mode = GRIP_MODE_KINEMATIC;
    bench->env.human_gearbox_mode = gearbox_mode;
    bench->env.human_gear = gear;
    bench->env.human_engine_rpm = ENGINE_IDLE_RPM;
    bench->env.human_longitudinal_mode = HUMAN_LONGITUDINAL_POWERTRAIN;
}

static Drive powertrain_full_env(int num_agents, unsigned int seed) {
    srand(seed);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, num_agents, 0);
    env.max_speed_mps = POWERTRAIN_TEST_BASE_SPEED_MPS;
    env.conditioning_accel_scale = 1.0f;
    env.conditioning_speed_scale = POWERTRAIN_TEST_SPEED_SCALE;
    env.reset_accel_on_stop = 1;
    allocate(&env);
    c_reset(&env);
    return env;
}

static int test_gear_ratios_are_ordered(void) {
    EXPECT_TRUE(gear_ratio(GEAR_REVERSE) < 0.0f);
    EXPECT_NEAR(gear_ratio(GEAR_REVERSE), -GEAR_REVERSE_RATIO, EXACT_TOLERANCE);
    EXPECT_NEAR(gear_ratio(GEAR_NEUTRAL), 0.0f, EXACT_TOLERANCE);
    for (int gear = 1; gear < GEARBOX_FORWARD_GEAR_COUNT; gear++) {
        EXPECT_TRUE(gear_ratio(gear) > gear_ratio(gear + 1));
        EXPECT_TRUE(gear_ratio(gear) > 0.0f);
    }
    EXPECT_NEAR(gear_ratio(1), GEAR_RATIOS[0], EXACT_TOLERANCE);
    EXPECT_NEAR(gear_ratio(GEARBOX_FORWARD_GEAR_COUNT), GEAR_RATIOS[GEARBOX_FORWARD_GEAR_COUNT - 1], EXACT_TOLERANCE);
    return 0;
}

static int test_engine_rpm_is_linear_in_speed(void) {
    EXPECT_NEAR(engine_rpm_for_speed(0.0f, GEAR_NEUTRAL), ENGINE_IDLE_RPM, EXACT_TOLERANCE);
    EXPECT_NEAR(engine_rpm_for_speed(25.0f, GEAR_NEUTRAL), ENGINE_IDLE_RPM, EXACT_TOLERANCE);

    for (int gear = 1; gear <= GEARBOX_FORWARD_GEAR_COUNT; gear++) {
        float rpm_at_five = engine_rpm_for_speed(5.0f, gear);
        EXPECT_NEAR(engine_rpm_for_speed(10.0f, gear), 2.0f * rpm_at_five, RPM_TOLERANCE);
        EXPECT_NEAR(engine_rpm_for_speed(15.0f, gear), 3.0f * rpm_at_five, RPM_TOLERANCE);
        EXPECT_NEAR(engine_rpm_for_speed(-5.0f, gear), rpm_at_five, EXACT_TOLERANCE);
        EXPECT_NEAR(engine_rpm_for_speed(0.0f, gear), 0.0f, EXACT_TOLERANCE);
        EXPECT_TRUE(rpm_at_five > 0.0f);
    }
    float reverse_rpm = engine_rpm_for_speed(-5.0f, GEAR_REVERSE);
    EXPECT_TRUE(reverse_rpm > 0.0f);
    EXPECT_NEAR(engine_rpm_for_speed(5.0f, GEAR_REVERSE), reverse_rpm, EXACT_TOLERANCE);

    for (int gear = 1; gear < GEARBOX_FORWARD_GEAR_COUNT; gear++) {
        EXPECT_TRUE(engine_rpm_for_speed(10.0f, gear) > engine_rpm_for_speed(10.0f, gear + 1));
    }
    return 0;
}

static int test_engine_torque_curve_shape(void) {
    EXPECT_NEAR(engine_torque_nm(ENGINE_STALL_RPM - 1.0f, 1.0f), 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(engine_torque_nm(0.0f, 1.0f), 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(engine_torque_nm(ENGINE_REDLINE_RPM + 1.0f, 1.0f), 0.0f, EXACT_TOLERANCE);
    EXPECT_TRUE(engine_torque_nm(ENGINE_STALL_RPM, 1.0f) > 0.0f);
    EXPECT_TRUE(engine_torque_nm(ENGINE_REDLINE_RPM, 1.0f) > 0.0f);

    float peak_torque_nm = engine_torque_nm(ENGINE_PEAK_TORQUE_RPM, 1.0f);
    EXPECT_NEAR(peak_torque_nm, ENGINE_PEAK_TORQUE_NM, TORQUE_TOLERANCE_NM);
    for (float rpm = ENGINE_STALL_RPM; rpm <= ENGINE_REDLINE_RPM; rpm += 50.0f) {
        float torque_nm = engine_torque_nm(rpm, 1.0f);
        EXPECT_TRUE(torque_nm >= 0.0f);
        EXPECT_TRUE(torque_nm <= peak_torque_nm + TORQUE_TOLERANCE_NM);
    }
    for (float offset_rpm = 100.0f; offset_rpm <= 2000.0f; offset_rpm += 100.0f) {
        EXPECT_NEAR(
            engine_torque_nm(ENGINE_PEAK_TORQUE_RPM - offset_rpm, 1.0f),
            engine_torque_nm(ENGINE_PEAK_TORQUE_RPM + offset_rpm, 1.0f),
            TORQUE_TOLERANCE_NM);
    }
    for (int throttle_tenths = 0; throttle_tenths <= 10; throttle_tenths++) {
        float throttle_norm = (float) throttle_tenths / 10.0f;
        EXPECT_NEAR(
            engine_torque_nm(ENGINE_PEAK_TORQUE_RPM, throttle_norm),
            throttle_norm * peak_torque_nm,
            TORQUE_TOLERANCE_NM);
    }
    return 0;
}

static int test_clutch_engagement_profile(void) {
    EXPECT_NEAR(clutch_engagement(0.0f), 1.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(clutch_engagement(CLUTCH_ENGAGE_FRACTION), 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(clutch_engagement(1.0f), 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(clutch_engagement(0.5f * CLUTCH_ENGAGE_FRACTION), 0.5f, ENGAGEMENT_TOLERANCE);

    float previous_engagement = 2.0f;
    for (int pedal_hundredths = 0; pedal_hundredths <= 100; pedal_hundredths++) {
        float clutch_norm = (float) pedal_hundredths / 100.0f;
        float engagement = clutch_engagement(clutch_norm);
        EXPECT_TRUE(engagement >= 0.0f);
        EXPECT_TRUE(engagement <= 1.0f);
        EXPECT_TRUE(engagement <= previous_engagement);
        previous_engagement = engagement;
    }
    return 0;
}

static int test_clutch_slip_lets_engine_outrun_driveline(void) {
    PowertrainBench bench;
    powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_MANUAL, 1);
    bench.env.human_throttle_norm = 1.0f;
    bench.env.human_clutch_norm = 0.3f;
    float launch_accel_mps2 = human_powertrain_accel_mps2(&bench.env, &bench.agent, 0);
    EXPECT_EQ_INT(bench.env.human_engine_stalled, 0);
    EXPECT_TRUE(bench.env.human_engine_rpm > engine_rpm_for_speed(0.0f, 1));
    EXPECT_TRUE(bench.env.human_engine_rpm > ENGINE_STALL_RPM);
    EXPECT_TRUE(launch_accel_mps2 > 0.0f);

    float previous_accel_mps2 = -1.0f;
    for (int pedal_tenths = 5; pedal_tenths >= 1; pedal_tenths--) {
        PowertrainBench rolling_bench;
        powertrain_bench_init(&rolling_bench, 8.0f, GEARBOX_MODE_MANUAL, 2);
        rolling_bench.env.human_throttle_norm = 1.0f;
        rolling_bench.env.human_clutch_norm = (float) pedal_tenths / 10.0f;
        float accel_mps2 = human_powertrain_accel_mps2(&rolling_bench.env, &rolling_bench.agent, 0);
        EXPECT_TRUE(accel_mps2 > previous_accel_mps2);
        previous_accel_mps2 = accel_mps2;
    }

    PowertrainBench disengaged_bench;
    powertrain_bench_init(&disengaged_bench, 8.0f, GEARBOX_MODE_MANUAL, 2);
    disengaged_bench.env.human_throttle_norm = 1.0f;
    disengaged_bench.env.human_clutch_norm = 1.0f;
    float coasting_accel_mps2 = human_powertrain_accel_mps2(&disengaged_bench.env, &disengaged_bench.agent, 0);
    EXPECT_TRUE(coasting_accel_mps2 < 0.0f);
    EXPECT_NEAR(disengaged_bench.env.human_engine_rpm, ENGINE_IDLE_RPM + ENGINE_FREE_REV_GAIN, RPM_TOLERANCE);
    return 0;
}

static int test_manual_stalls_in_gear_at_standstill(void) {
    PowertrainBench bench;
    powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_MANUAL, 1);
    bench.env.human_throttle_norm = 1.0f;
    float stalled_accel_mps2 = human_powertrain_accel_mps2(&bench.env, &bench.agent, 0);
    EXPECT_EQ_INT(bench.env.human_engine_stalled, 1);
    EXPECT_NEAR(bench.env.human_engine_rpm, 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(stalled_accel_mps2, 0.0f, EXACT_TOLERANCE);

    bench.env.human_clutch_norm = 1.0f;
    human_powertrain_accel_mps2(&bench.env, &bench.agent, 0);
    EXPECT_EQ_INT(bench.env.human_engine_stalled, 0);
    EXPECT_TRUE(bench.env.human_engine_rpm > ENGINE_STALL_RPM);

    PowertrainBench neutral_bench;
    powertrain_bench_init(&neutral_bench, 0.0f, GEARBOX_MODE_MANUAL, GEAR_NEUTRAL);
    human_powertrain_accel_mps2(&neutral_bench.env, &neutral_bench.agent, 0);
    EXPECT_EQ_INT(neutral_bench.env.human_engine_stalled, 0);
    return 0;
}

// Only a fully engaged clutch can drag the engine under; any slip keeps it turning.
static int test_stall_requires_full_engagement(void) {
    for (int pedal_hundredths = 0; pedal_hundredths <= 50; pedal_hundredths += 5) {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_MANUAL, 1);
        bench.env.human_clutch_norm = (float) pedal_hundredths / 100.0f;
        human_powertrain_accel_mps2(&bench.env, &bench.agent, 0);
        int expect_stalled = (clutch_engagement(bench.env.human_clutch_norm) >= 1.0f);
        EXPECT_EQ_INT(bench.env.human_engine_stalled, expect_stalled);
    }
    return 0;
}

static int test_automatic_never_stalls(void) {
    for (int gear = 1; gear <= GEARBOX_FORWARD_GEAR_COUNT; gear++) {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_AUTOMATIC, gear);
        float accel_mps2 = human_powertrain_accel_mps2(&bench.env, &bench.agent, 0);
        EXPECT_EQ_INT(bench.env.human_engine_stalled, 0);
        EXPECT_TRUE(bench.env.human_engine_rpm >= ENGINE_IDLE_RPM);
        EXPECT_TRUE(accel_mps2 > 0.0f);
    }
    PowertrainBench reverse_bench;
    powertrain_bench_init(&reverse_bench, 0.0f, GEARBOX_MODE_AUTOMATIC, GEAR_REVERSE);
    human_powertrain_accel_mps2(&reverse_bench.env, &reverse_bench.agent, 0);
    EXPECT_EQ_INT(reverse_bench.env.human_engine_stalled, 0);
    return 0;
}

static int test_manual_shift_requires_the_clutch(void) {
    {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 10.0f, GEARBOX_MODE_MANUAL, 3);
        EXPECT_EQ_INT(c_set_human_pedals(&bench.env, 0.0f, 0.0f, 0.0f, CLUTCH_ENGAGE_FRACTION - 0.01f, 1), 0);
        move_human(&bench.env, 0);
        EXPECT_EQ_INT(bench.env.human_gear, 3);
    }
    {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 10.0f, GEARBOX_MODE_MANUAL, 3);
        EXPECT_EQ_INT(c_set_human_pedals(&bench.env, 0.0f, 0.0f, 0.0f, CLUTCH_ENGAGE_FRACTION, 1), 0);
        move_human(&bench.env, 0);
        EXPECT_EQ_INT(bench.env.human_gear, 4);
    }
    {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 10.0f, GEARBOX_MODE_MANUAL, 3);
        c_set_human_pedals(&bench.env, 0.0f, 0.0f, 0.0f, 1.0f, -1);
        move_human(&bench.env, 0);
        EXPECT_EQ_INT(bench.env.human_gear, 2);
    }
    return 0;
}

static int test_manual_shift_is_consumed_by_one_step(void) {
    PowertrainBench bench;
    powertrain_bench_init(&bench, 10.0f, GEARBOX_MODE_MANUAL, 2);
    c_set_human_pedals(&bench.env, 0.0f, 0.0f, 0.0f, 1.0f, 1);
    EXPECT_EQ_INT(bench.env.human_shift_request, 1);
    move_human(&bench.env, 0);
    EXPECT_EQ_INT(bench.env.human_gear, 3);
    EXPECT_EQ_INT(bench.env.human_shift_request, 0);
    for (int step = 0; step < 5; step++) {
        move_human(&bench.env, 0);
        EXPECT_EQ_INT(bench.env.human_gear, 3);
    }
    return 0;
}

static int test_manual_shift_does_not_wrap(void) {
    {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 15.0f, GEARBOX_MODE_MANUAL, GEARBOX_FORWARD_GEAR_COUNT);
        c_set_human_pedals(&bench.env, 0.0f, 0.0f, 0.0f, 1.0f, 1);
        move_human(&bench.env, 0);
        EXPECT_EQ_INT(bench.env.human_gear, GEARBOX_FORWARD_GEAR_COUNT);
    }
    {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_MANUAL, GEAR_REVERSE);
        c_set_human_pedals(&bench.env, 0.0f, 0.0f, 0.0f, 1.0f, -1);
        move_human(&bench.env, 0);
        EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    }
    return 0;
}

static int test_automatic_shifts_on_rpm_thresholds(void) {
    {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_AUTOMATIC, 2);
        float upshift_speed_mps = 1.05f * AUTO_UPSHIFT_RPM / engine_rpm_for_speed(1.0f, 2);
        select_automatic_gear(&bench.env, upshift_speed_mps);
        EXPECT_EQ_INT(bench.env.human_gear, 3);
    }
    {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_AUTOMATIC, 4);
        float downshift_speed_mps = 0.95f * AUTO_DOWNSHIFT_RPM / engine_rpm_for_speed(1.0f, 4);
        select_automatic_gear(&bench.env, downshift_speed_mps);
        EXPECT_EQ_INT(bench.env.human_gear, 3);
    }
    {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_AUTOMATIC, GEAR_NEUTRAL);
        select_automatic_gear(&bench.env, 0.0f);
        EXPECT_TRUE(bench.env.human_gear >= 1);
    }
    {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_AUTOMATIC, 1);
        select_automatic_gear(&bench.env, 0.0f);
        EXPECT_EQ_INT(bench.env.human_gear, 1);
    }
    return 0;
}

static int test_automatic_does_not_oscillate_at_steady_speed(void) {
    for (int speed_mps_int = 3; speed_mps_int <= 30; speed_mps_int += 3) {
        float speed_mps = (float) speed_mps_int;
        PowertrainBench bench;
        powertrain_bench_init(&bench, speed_mps, GEARBOX_MODE_AUTOMATIC, 1);
        for (int step = 0; step < 40; step++) {
            select_automatic_gear(&bench.env, speed_mps);
        }
        int settled_gear = bench.env.human_gear;
        for (int step = 0; step < 40; step++) {
            select_automatic_gear(&bench.env, speed_mps);
            EXPECT_EQ_INT(bench.env.human_gear, settled_gear);
        }
        float settled_rpm = engine_rpm_for_speed(speed_mps, settled_gear);
        EXPECT_TRUE(settled_rpm <= AUTO_UPSHIFT_RPM || settled_gear == GEARBOX_FORWARD_GEAR_COUNT);
        EXPECT_TRUE(settled_rpm >= AUTO_DOWNSHIFT_RPM || settled_gear == 1);
    }
    return 0;
}

static int test_automatic_stays_in_reverse(void) {
    {
        PowertrainBench bench;
        powertrain_bench_init(&bench, -1.5f, GEARBOX_MODE_AUTOMATIC, GEAR_REVERSE);
        c_set_human_pedals(&bench.env, 0.0f, 1.0f, 0.0f, 0.0f, 0);
        for (int step = 0; step < 100; step++) {
            move_human(&bench.env, 0);
            EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
        }
    }
    // Reverse selected while still rolling forward spins the box past its upshift threshold.
    for (int speed_mps_int = 8; speed_mps_int <= 24; speed_mps_int += 4) {
        float rolling_speed_mps = (float) speed_mps_int;
        EXPECT_TRUE(engine_rpm_for_speed(rolling_speed_mps, GEAR_REVERSE) > AUTO_UPSHIFT_RPM);
        PowertrainBench bench;
        powertrain_bench_init(&bench, rolling_speed_mps, GEARBOX_MODE_AUTOMATIC, GEAR_REVERSE);
        for (int step = 0; step < 20; step++) {
            select_automatic_gear(&bench.env, rolling_speed_mps);
            EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
        }
    }
    return 0;
}

static int test_brake_holds_against_idle_creep(void) {
    PowertrainBench bench;
    powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_AUTOMATIC, 1);
    c_set_human_pedals(&bench.env, 0.0f, 0.0f, 1.0f, 0.0f, 0);
    float standstill_accel_mps2 = human_powertrain_accel_mps2(&bench.env, &bench.agent, 0);
    EXPECT_NEAR(standstill_accel_mps2, 0.0f, ACCEL_TOLERANCE_MPS2);

    for (int step = 0; step < HOLD_STEP_COUNT; step++) {
        move_human(&bench.env, 0);
        EXPECT_NEAR(bench.agent.sim_speed_signed, 0.0f, EXACT_TOLERANCE);
    }
    EXPECT_NEAR(bench.agent.sim_x, 0.0f, EXACT_TOLERANCE);
    return 0;
}

// The stop is exact whatever reset_accel_on_stop says: that flag is the jerk path's, not the human's.
static int test_brake_stops_the_car_exactly_under_both_flag_settings(void) {
    for (int reset_accel_on_stop = 0; reset_accel_on_stop <= 1; reset_accel_on_stop++) {
        for (int start_speed_mps_int = 5; start_speed_mps_int <= 30; start_speed_mps_int += 5) {
            PowertrainBench bench;
            powertrain_bench_init(&bench, (float) start_speed_mps_int, GEARBOX_MODE_AUTOMATIC, 4);
            bench.env.reset_accel_on_stop = reset_accel_on_stop;
            c_set_human_pedals(&bench.env, 0.0f, 0.0f, 1.0f, 0.0f, 0);
            float previous_x_m = bench.agent.sim_x;
            for (int step = 0; step < SETTLE_STEP_COUNT; step++) {
                move_human(&bench.env, 0);
                EXPECT_TRUE(bench.agent.sim_speed_signed >= 0.0f);
                EXPECT_TRUE(bench.agent.sim_x >= previous_x_m);
                previous_x_m = bench.agent.sim_x;
            }
            EXPECT_NEAR(bench.agent.sim_speed_signed, 0.0f, EXACT_TOLERANCE);
            EXPECT_NEAR(bench.agent.accel_long, 0.0f, EXACT_TOLERANCE);
            float settled_x_m = bench.agent.sim_x;
            for (int step = 0; step < HOLD_STEP_COUNT; step++) {
                move_human(&bench.env, 0);
                EXPECT_NEAR(bench.agent.sim_speed_signed, 0.0f, EXACT_TOLERANCE);
                EXPECT_NEAR(bench.agent.sim_x, settled_x_m, EXACT_TOLERANCE);
            }
        }
    }
    return 0;
}

static int test_engine_restart_recovers_a_stalled_car(void) {
    PowertrainBench bench;
    powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_MANUAL, 1);
    c_set_human_pedals(&bench.env, 0.0f, 1.0f, 0.0f, 0.0f, 0);
    move_human(&bench.env, 0);
    EXPECT_EQ_INT(bench.env.human_engine_stalled, 1);
    EXPECT_NEAR(bench.env.human_engine_rpm, 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(bench.agent.sim_speed_signed, 0.0f, EXACT_TOLERANCE);

    EXPECT_EQ_INT(c_restart_human_engine(&bench.env), 0);
    EXPECT_EQ_INT(bench.env.human_engine_stalled, 0);
    EXPECT_NEAR(bench.env.human_engine_rpm, ENGINE_IDLE_RPM, EXACT_TOLERANCE);

    c_set_human_pedals(&bench.env, 0.0f, 0.8f, 0.0f, 0.3f, 0);
    for (int step = 0; step < 100; step++) {
        move_human(&bench.env, 0);
    }
    EXPECT_EQ_INT(bench.env.human_engine_stalled, 0);
    EXPECT_TRUE(bench.agent.sim_speed_signed > 0.0f);
    EXPECT_TRUE(bench.agent.sim_x > 0.0f);
    return 0;
}

// Feathering the clutch is not an ignition switch, and a restart with it up stalls straight back.
static int test_restart_is_the_only_way_out_of_a_stall(void) {
    {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_MANUAL, 1);
        bench.env.human_throttle_norm = 1.0f;
        human_powertrain_accel_mps2(&bench.env, &bench.agent, 0);
        EXPECT_EQ_INT(bench.env.human_engine_stalled, 1);
        for (int pedal_tenths = 1; pedal_tenths <= 5; pedal_tenths++) {
            bench.env.human_clutch_norm = (float) pedal_tenths / 10.0f;
            float accel_mps2 = human_powertrain_accel_mps2(&bench.env, &bench.agent, 0);
            EXPECT_EQ_INT(bench.env.human_engine_stalled, 1);
            EXPECT_NEAR(bench.env.human_engine_rpm, 0.0f, EXACT_TOLERANCE);
            EXPECT_NEAR(accel_mps2, 0.0f, EXACT_TOLERANCE);
        }
    }
    {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_MANUAL, 1);
        human_powertrain_accel_mps2(&bench.env, &bench.agent, 0);
        EXPECT_EQ_INT(c_restart_human_engine(&bench.env), 0);
        float accel_mps2 = human_powertrain_accel_mps2(&bench.env, &bench.agent, 0);
        EXPECT_EQ_INT(bench.env.human_engine_stalled, 1);
        EXPECT_NEAR(accel_mps2, 0.0f, EXACT_TOLERANCE);
    }
    return 0;
}

// The idle governor floors a running engine at idle, so a fully engaged clutch can read above
// the driveline; engine braking is computed from the governed figure.
static int test_idle_governor_floors_reported_rpm(void) {
    PowertrainBench bench;
    powertrain_bench_init(&bench, TAPER_TEST_SPEED_MPS, GEARBOX_MODE_MANUAL, GEARBOX_FORWARD_GEAR_COUNT);
    bench.env.human_throttle_norm = 1.0f;
    float driveline_rpm = engine_rpm_for_speed(TAPER_TEST_SPEED_MPS, GEARBOX_FORWARD_GEAR_COUNT);
    EXPECT_TRUE(driveline_rpm < ENGINE_IDLE_RPM);
    human_powertrain_accel_mps2(&bench.env, &bench.agent, 0);
    EXPECT_EQ_INT(bench.env.human_engine_stalled, 0);
    EXPECT_NEAR(bench.env.human_engine_rpm, ENGINE_IDLE_RPM, EXACT_TOLERANCE);
    return 0;
}

static int test_idle_creep_in_automatic(void) {
    PowertrainBench bench;
    powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_AUTOMATIC, GEAR_NEUTRAL);
    c_set_human_pedals(&bench.env, 0.0f, 0.0f, 0.0f, 0.0f, 0);
    for (int step = 0; step < 40; step++) {
        move_human(&bench.env, 0);
    }
    EXPECT_TRUE(bench.env.human_gear >= 1);
    EXPECT_TRUE(bench.agent.sim_speed_signed > 0.0f);
    EXPECT_TRUE(bench.agent.sim_x > 0.0f);

    c_set_human_pedals(&bench.env, 0.0f, 0.0f, 1.0f, 0.0f, 0);
    for (int step = 0; step < HOLD_STEP_COUNT; step++) {
        move_human(&bench.env, 0);
    }
    EXPECT_NEAR(bench.agent.sim_speed_signed, 0.0f, EXACT_TOLERANCE);
    float stopped_x_m = bench.agent.sim_x;
    for (int step = 0; step < HOLD_STEP_COUNT; step++) {
        move_human(&bench.env, 0);
        EXPECT_NEAR(bench.agent.sim_x, stopped_x_m, EXACT_TOLERANCE);
    }
    return 0;
}

static int test_reverse_accelerates_backwards_to_the_floor(void) {
    {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_AUTOMATIC, GEAR_REVERSE);
        c_set_human_pedals(&bench.env, 0.0f, 0.6f, 0.0f, 0.0f, 0);
        for (int step = 0; step < SETTLE_STEP_COUNT; step++) {
            move_human(&bench.env, 0);
            EXPECT_TRUE(bench.agent.sim_speed_signed >= MAX_BACKWARD_SPEED);
        }
        EXPECT_NEAR(bench.agent.sim_speed_signed, MAX_BACKWARD_SPEED, SPEED_TOLERANCE_MPS);
        EXPECT_TRUE(bench.agent.sim_x < 0.0f);
    }
    {
        PowertrainBench bench;
        powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_MANUAL, GEAR_REVERSE);
        c_set_human_pedals(&bench.env, 0.0f, 0.6f, 0.0f, 0.3f, 0);
        for (int step = 0; step < SETTLE_STEP_COUNT; step++) {
            move_human(&bench.env, 0);
            EXPECT_TRUE(bench.agent.sim_speed_signed >= MAX_BACKWARD_SPEED);
        }
        EXPECT_EQ_INT(bench.env.human_engine_stalled, 0);
        EXPECT_NEAR(bench.agent.sim_speed_signed, MAX_BACKWARD_SPEED, SPEED_TOLERANCE_MPS);
        EXPECT_TRUE(bench.agent.sim_x < 0.0f);
    }
    return 0;
}

static int test_powertrain_accel_stays_in_envelope(void) {
    for (int gearbox_mode = 0; gearbox_mode <= 1; gearbox_mode++) {
        for (int gear = GEAR_REVERSE; gear <= GEARBOX_FORWARD_GEAR_COUNT; gear++) {
            for (int speed_tenths = -40; speed_tenths <= 400; speed_tenths += 13) {
                for (int throttle_fifths = 0; throttle_fifths <= 5; throttle_fifths++) {
                    for (int brake_halves = 0; brake_halves <= 2; brake_halves++) {
                        for (int clutch_fifths = 0; clutch_fifths <= 5; clutch_fifths++) {
                            PowertrainBench bench;
                            powertrain_bench_init(&bench, (float) speed_tenths / 10.0f, gearbox_mode, gear);
                            bench.env.human_throttle_norm = (float) throttle_fifths / 5.0f;
                            bench.env.human_brake_norm = (float) brake_halves / 2.0f;
                            bench.env.human_clutch_norm = (float) clutch_fifths / 5.0f;
                            float accel_mps2 = human_powertrain_accel_mps2(&bench.env, &bench.agent, 0);
                            EXPECT_FINITE(accel_mps2);
                            EXPECT_TRUE(accel_mps2 >= HUMAN_ACCEL_LIMIT_MPS2[0]);
                            EXPECT_TRUE(accel_mps2 <= HUMAN_ACCEL_LIMIT_MPS2[1]);
                            EXPECT_FINITE(bench.env.human_engine_rpm);
                            EXPECT_TRUE(bench.env.human_engine_rpm >= 0.0f);
                        }
                    }
                }
            }
        }
    }
    return 0;
}

static int test_accel_tapers_with_gear(void) {
    float previous_accel_mps2 = HUMAN_ACCEL_LIMIT_MPS2[1] + 1.0f;
    for (int gear = 1; gear <= GEARBOX_FORWARD_GEAR_COUNT; gear++) {
        PowertrainBench bench;
        powertrain_bench_init(&bench, TAPER_TEST_SPEED_MPS, GEARBOX_MODE_MANUAL, gear);
        bench.env.human_throttle_norm = 1.0f;
        float accel_mps2 = human_powertrain_accel_mps2(&bench.env, &bench.agent, 0);
        EXPECT_EQ_INT(bench.env.human_engine_stalled, 0);
        EXPECT_TRUE(accel_mps2 < previous_accel_mps2);
        previous_accel_mps2 = accel_mps2;
    }
    EXPECT_TRUE(previous_accel_mps2 > 0.0f);
    return 0;
}

static int test_speed_settles_exactly_on_the_cap(void) {
    PowertrainBench bench;
    powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_AUTOMATIC, GEAR_NEUTRAL);
    bench.agent.reward_coefs[REWARD_COEF_SPEED] = 1.0f;
    float cap_mps = bench.env.base_max_speed_mps * bench.agent.reward_coefs[REWARD_COEF_SPEED];
    c_set_human_pedals(&bench.env, 0.0f, 1.0f, 0.0f, 0.0f, 0);
    for (int step = 0; step < SETTLE_STEP_COUNT; step++) {
        move_human(&bench.env, 0);
        EXPECT_TRUE(bench.agent.sim_speed_signed <= cap_mps);
    }
    EXPECT_NEAR(bench.agent.sim_speed_signed, cap_mps, SPEED_TOLERANCE_MPS);
    return 0;
}

static int test_powertrain_setters_validate(void) {
    PowertrainBench bench;
    powertrain_bench_init(&bench, 0.0f, GEARBOX_MODE_AUTOMATIC, GEAR_NEUTRAL);

    EXPECT_EQ_INT(c_set_human_gear(&bench.env, GEAR_REVERSE), 0);
    EXPECT_EQ_INT(c_set_human_gear(&bench.env, GEARBOX_FORWARD_GEAR_COUNT), 0);
    EXPECT_EQ_INT(c_set_human_gear(&bench.env, GEAR_REVERSE - 1), -1);
    EXPECT_EQ_INT(c_set_human_gear(&bench.env, GEARBOX_FORWARD_GEAR_COUNT + 1), -1);
    EXPECT_EQ_INT(bench.env.human_gear, GEARBOX_FORWARD_GEAR_COUNT);

    EXPECT_EQ_INT(c_set_human_gearbox_mode(&bench.env, GEARBOX_MODE_MANUAL), 0);
    EXPECT_EQ_INT(c_set_human_gearbox_mode(&bench.env, GEARBOX_MODE_AUTOMATIC), 0);
    EXPECT_EQ_INT(c_set_human_gearbox_mode(&bench.env, -1), -1);
    EXPECT_EQ_INT(c_set_human_gearbox_mode(&bench.env, 2), -1);
    EXPECT_EQ_INT(bench.env.human_gearbox_mode, GEARBOX_MODE_AUTOMATIC);

    EXPECT_EQ_INT(c_set_human_pedals(&bench.env, 0.0f, 0.5f, 0.25f, 0.75f, 1), 0);
    EXPECT_EQ_INT(c_set_human_pedals(&bench.env, NAN, 0.0f, 0.0f, 0.0f, 0), -1);
    EXPECT_EQ_INT(c_set_human_pedals(&bench.env, 0.0f, NAN, 0.0f, 0.0f, 0), -1);
    EXPECT_EQ_INT(c_set_human_pedals(&bench.env, 0.0f, 0.0f, INFINITY, 0.0f, 0), -1);
    EXPECT_EQ_INT(c_set_human_pedals(&bench.env, 0.0f, 0.0f, 0.0f, -INFINITY, 0), -1);
    EXPECT_EQ_INT(c_set_human_pedals(&bench.env, 0.0f, 0.0f, 0.0f, 0.0f, 2), -1);
    EXPECT_EQ_INT(c_set_human_pedals(&bench.env, 0.0f, 0.0f, 0.0f, 0.0f, -2), -1);
    EXPECT_NEAR(bench.env.human_throttle_norm, 0.5f, EXACT_TOLERANCE);
    EXPECT_NEAR(bench.env.human_brake_norm, 0.25f, EXACT_TOLERANCE);
    EXPECT_NEAR(bench.env.human_clutch_norm, 0.75f, EXACT_TOLERANCE);

    EXPECT_EQ_INT(c_set_human_pedals(&bench.env, 5.0f, 9.0f, -3.0f, 7.0f, 0), 0);
    EXPECT_NEAR(bench.env.human_steering_cmd_norm, 1.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(bench.env.human_throttle_norm, 1.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(bench.env.human_brake_norm, 0.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(bench.env.human_clutch_norm, 1.0f, EXACT_TOLERANCE);
    EXPECT_EQ_INT(bench.env.human_longitudinal_mode, HUMAN_LONGITUDINAL_POWERTRAIN);
    return 0;
}

static int test_longitudinal_mode_defaults_to_direct(void) {
    Drive env = powertrain_full_env(POWERTRAIN_TEST_AGENT_COUNT, 311);
    EXPECT_EQ_INT(env.human_longitudinal_mode, HUMAN_LONGITUDINAL_DIRECT);
    EXPECT_EQ_INT(env.human_gearbox_mode, GEARBOX_MODE_AUTOMATIC);
    EXPECT_EQ_INT(env.human_gear, GEAR_NEUTRAL);
    EXPECT_EQ_INT(env.human_engine_stalled, 0);
    EXPECT_NEAR(env.human_engine_rpm, ENGINE_IDLE_RPM, EXACT_TOLERANCE);
    EXPECT_NEAR(env.human_max_speed_mps, env.base_max_speed_mps * env.conditioning_speed_scale, EXACT_TOLERANCE);
    free_allocated(&env);
    return 0;
}

// The direct path must stay reachable: a DIRECT command drives the car, not the pedals.
static int test_direct_mode_ignores_the_powertrain(void) {
    PowertrainBench bench;
    powertrain_bench_init(&bench, 5.0f, GEARBOX_MODE_AUTOMATIC, 2);
    bench.env.human_longitudinal_mode = HUMAN_LONGITUDINAL_DIRECT;
    bench.env.human_throttle_norm = 1.0f;
    c_set_human_input(&bench.env, 0.0f, -2.0f);
    move_human(&bench.env, 0);
    EXPECT_NEAR(bench.agent.accel_long, -2.0f, EXACT_TOLERANCE);
    EXPECT_NEAR(bench.env.human_engine_rpm, ENGINE_IDLE_RPM, EXACT_TOLERANCE);
    EXPECT_EQ_INT(bench.env.human_gear, 2);
    return 0;
}

static int test_policy_dynamics_keep_their_own_accel_cap(void) {
    static const float conditioning_coefs[] = {1.0f, 2.0f};
    for (int i = 0; i < 2; i++) {
        srand(41);
        Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
        env.action_type = ACTION_TYPE_CONTINUOUS;
        env.dynamics_model = DYNAMICS_MODEL_JERK;
        int agent_idx = env.active_agent_indices[0];
        Agent *agent = &env.agents[agent_idx];
        agent->reward_coefs[REWARD_COEF_ACC] = conditioning_coefs[i];
        agent->reward_coefs[REWARD_COEF_THROTTLE] = 1.0f;
        agent->reward_coefs[REWARD_COEF_STEER] = 1.0f;
        agent->reward_coefs[REWARD_COEF_SPEED] = 1.0f;
        ((float (*)[2]) env.actions)[0][0] = 10.0f;
        ((float (*)[2]) env.actions)[0][1] = 0.0f;
        for (int step = 0; step < 20; step++) {
            move_dynamics(&env, 0, agent_idx);
        }
        EXPECT_NEAR(agent->accel_long, ACCEL_LONG_LIMIT[1] * conditioning_coefs[i], 1e-5f);
        EXPECT_TRUE(agent->accel_long <= ACCEL_LONG_LIMIT[1] * conditioning_coefs[i] + 1e-5f);
        free_allocated(&env);
    }
    return 0;
}

static int test_powertrain_setters_do_not_touch_policy_traffic(void) {
    Drive baseline_env = powertrain_full_env(POWERTRAIN_TEST_AGENT_COUNT, 43);
    Drive poked_env = powertrain_full_env(POWERTRAIN_TEST_AGENT_COUNT, 43);

    EXPECT_EQ_INT(c_set_human_gearbox_mode(&poked_env, GEARBOX_MODE_MANUAL), 0);
    EXPECT_EQ_INT(c_set_human_gear(&poked_env, GEAR_REVERSE), 0);
    EXPECT_EQ_INT(c_set_human_pedals(&poked_env, 1.0f, 1.0f, 1.0f, 1.0f, 1), 0);
    EXPECT_EQ_INT(c_set_human_grip_mode(&poked_env, GRIP_MODE_REALISTIC), 0);

    for (int step = 0; step < 20; step++) {
        drive_set_neutral_actions(&baseline_env);
        drive_set_neutral_actions(&poked_env);
        c_step(&baseline_env);
        c_step(&poked_env);
    }

    EXPECT_EQ_INT(poked_env.active_agent_count, baseline_env.active_agent_count);
    for (int i = 0; i < baseline_env.active_agent_count; i++) {
        const Agent *baseline_agent = &baseline_env.agents[baseline_env.active_agent_indices[i]];
        const Agent *poked_agent = &poked_env.agents[poked_env.active_agent_indices[i]];
        EXPECT_EQ_INT(poked_agent->controller, baseline_agent->controller);
        EXPECT_NEAR(poked_agent->sim_x, baseline_agent->sim_x, EXACT_TOLERANCE);
        EXPECT_NEAR(poked_agent->sim_y, baseline_agent->sim_y, EXACT_TOLERANCE);
        EXPECT_NEAR(poked_agent->sim_heading, baseline_agent->sim_heading, EXACT_TOLERANCE);
        EXPECT_NEAR(poked_agent->sim_speed_signed, baseline_agent->sim_speed_signed, EXACT_TOLERANCE);
        EXPECT_NEAR(poked_agent->accel_long, baseline_agent->accel_long, EXACT_TOLERANCE);
        EXPECT_NEAR(poked_agent->steering_angle, baseline_agent->steering_angle, EXACT_TOLERANCE);
    }
    free_allocated(&baseline_env);
    free_allocated(&poked_env);
    return 0;
}

static int test_human_powertrain_car_drives_through_c_step(void) {
    Drive env = powertrain_full_env(POWERTRAIN_TEST_AGENT_COUNT, 47);
    int human_agent_idx = env.active_agent_indices[0];
    Agent *human = &env.agents[human_agent_idx];
    EXPECT_EQ_INT(c_select_human_agent(&env, human_agent_idx), 0);
    EXPECT_EQ_INT(c_set_human_gearbox_mode(&env, GEARBOX_MODE_AUTOMATIC), 0);
    EXPECT_EQ_INT(c_set_human_pedals(&env, 0.0f, 1.0f, 0.0f, 0.0f, 0), 0);

    for (int step = 0; step < 15; step++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
    }
    EXPECT_TRUE(human->sim_speed_signed > 0.0f);
    EXPECT_TRUE(env.human_gear >= 1);
    EXPECT_TRUE(env.human_engine_rpm >= ENGINE_IDLE_RPM);
    EXPECT_TRUE(human->accel_long <= HUMAN_ACCEL_LIMIT_MPS2[1]);
    EXPECT_TRUE(human->accel_long >= HUMAN_ACCEL_LIMIT_MPS2[0]);
    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_gear_ratios_are_ordered);
    RUN_TEST(test_engine_rpm_is_linear_in_speed);
    RUN_TEST(test_engine_torque_curve_shape);
    RUN_TEST(test_clutch_engagement_profile);
    RUN_TEST(test_clutch_slip_lets_engine_outrun_driveline);
    RUN_TEST(test_manual_stalls_in_gear_at_standstill);
    RUN_TEST(test_stall_requires_full_engagement);
    RUN_TEST(test_automatic_never_stalls);
    RUN_TEST(test_engine_restart_recovers_a_stalled_car);
    RUN_TEST(test_restart_is_the_only_way_out_of_a_stall);
    RUN_TEST(test_idle_governor_floors_reported_rpm);
    RUN_TEST(test_manual_shift_requires_the_clutch);
    RUN_TEST(test_manual_shift_is_consumed_by_one_step);
    RUN_TEST(test_manual_shift_does_not_wrap);
    RUN_TEST(test_automatic_shifts_on_rpm_thresholds);
    RUN_TEST(test_automatic_does_not_oscillate_at_steady_speed);
    RUN_TEST(test_automatic_stays_in_reverse);
    RUN_TEST(test_brake_holds_against_idle_creep);
    RUN_TEST(test_brake_stops_the_car_exactly_under_both_flag_settings);
    RUN_TEST(test_idle_creep_in_automatic);
    RUN_TEST(test_reverse_accelerates_backwards_to_the_floor);
    RUN_TEST(test_powertrain_accel_stays_in_envelope);
    RUN_TEST(test_accel_tapers_with_gear);
    RUN_TEST(test_speed_settles_exactly_on_the_cap);
    RUN_TEST(test_powertrain_setters_validate);
    RUN_TEST(test_longitudinal_mode_defaults_to_direct);
    RUN_TEST(test_direct_mode_ignores_the_powertrain);
    RUN_TEST(test_policy_dynamics_keep_their_own_accel_cap);
    RUN_TEST(test_powertrain_setters_do_not_touch_policy_traffic);
    RUN_TEST(test_human_powertrain_car_drives_through_c_step);
    return test_summary(failures);
}
