#include "include/drive_fixture.h"
#include "include/test.h"

#define STOP_GO_DT_SECONDS (1.0f / 60.0f)
#define STOP_GO_MAX_SPEED_MPS 20.0f
#define CREEP_SPEED_MPS 0.01f
#define WRONG_WAY_CREEP_MPS 1e-3f
#define DIRECT_BRAKE_MPS2 -5.0f
#define DIRECT_PULL_MPS2 2.0f
#define PULL_AWAY_THROTTLE 0.6f
#define PULL_AWAY_STEP_COUNT 10
#define PULL_AWAY_MIN_SPEED_MPS 0.01f
#define LONG_DRIVE_STEP_COUNT 180
#define BACKWARD_ROLL_MPS -1.0f
#define BLEND_BACKING_SPEED_MPS -1.1f
#define BLEND_WEIGHT 0.5f
#define POLICY_STOP_ACCEL_MPS2 -4.5f
#define POLICY_BACKING_SPEED_MPS -0.1f
#define BLEND_PULL_AWAY_MIN_SPEED_MPS 1.0f
#define MOVING_FORWARD_MPS 2.0f
#define ROLLING_FORWARD_MPS 1.0f
#define EXACT_TOLERANCE 0.0f

static const float WRONG_SIGN_CREEPS_MPS[] = {1e-9f, 5.6e-17f, 1e-4f};
#define WRONG_SIGN_CREEP_COUNT 3

typedef struct {
    Drive env;
    GridMap offgrid_map;
    Agent agent;
} StopGoBench;

// The zeroed GridMap reports every query off-grid, which makes update_agent_z a no-op.
static void bench_init(StopGoBench *bench, int grip_mode, int gearbox_mode, int gear, float forward_speed_mps) {
    memset(bench, 0, sizeof(StopGoBench));
    bench->agent = drive_test_agent(0.0f, 0.0f, 0.4f);
    for (int c = 0; c < NUM_REWARD_COEFS; c++) {
        bench->agent.reward_coefs[c] = 1.0f;
    }
    bench->agent.controller = CONTROLLER_HUMAN;
    bench->agent.sim_vx = forward_speed_mps * bench->agent.cos_heading;
    bench->agent.sim_vy = forward_speed_mps * bench->agent.sin_heading;
    update_agent_speed(&bench->agent);
    copy_pose_to_prev(&bench->agent);
    bench->env.agents = &bench->agent;
    bench->env.num_agents = 1;
    bench->env.grid_map = &bench->offgrid_map;
    bench->env.dt = STOP_GO_DT_SECONDS;
    bench->env.base_max_speed_mps = STOP_GO_MAX_SPEED_MPS;
    bench->env.human_max_speed_mps = STOP_GO_MAX_SPEED_MPS;
    bench->env.human_agent_idx = 0;
    bench->env.reset_accel_on_stop = 1;
    bench->env.human_grip_mode = grip_mode;
    bench->env.human_gearbox_mode = gearbox_mode;
    bench->env.human_gear = gear;
    bench->env.human_engine_rpm = ENGINE_IDLE_RPM;
}

static float forward_speed_mps(const Agent *agent) {
    return agent->sim_vx * agent->cos_heading + agent->sim_vy * agent->sin_heading;
}

static void step_direct(StopGoBench *bench, float accel_mps2) {
    c_set_human_input(&bench->env, 0.0f, accel_mps2);
    move_human(&bench->env, 0, 0);
}

static void step_pedals(StopGoBench *bench, float throttle, float brake, int shift_request) {
    c_set_human_pedals(&bench->env, 0.0f, throttle, brake, 0.0f, shift_request);
    move_human(&bench->env, 0, 0);
}

static int test_braking_through_zero_stops_the_car_in_either_gear(void) {
    int grips[] = {GRIP_MODE_KINEMATIC, GRIP_MODE_SKID};
    int gears[] = {1, GEAR_REVERSE};
    for (int g = 0; g < 2; g++) {
        for (int r = 0; r < 2; r++) {
            float direction = gears[r] == GEAR_REVERSE ? -1.0f : 1.0f;
            StopGoBench bench;
            bench_init(&bench, grips[g], GEARBOX_MODE_AUTOMATIC, gears[r], direction * CREEP_SPEED_MPS);
            step_direct(&bench, direction * DIRECT_BRAKE_MPS2);
            EXPECT_TRUE(bench.agent.sim_speed_signed == 0.0f);
            EXPECT_TRUE(bench.agent.accel_long == 0.0f);
            EXPECT_TRUE(forward_speed_mps(&bench.agent) == 0.0f);
        }
    }
    return 0;
}

// Crossing zero in either direction lands exactly at rest for one step (the symmetric snap); the next step pulls away.
static int test_a_wrong_way_creep_is_snapped_to_rest_then_pulls_away(void) {
    int grips[] = {GRIP_MODE_KINEMATIC, GRIP_MODE_REALISTIC, GRIP_MODE_SKID};
    int gears[] = {1, GEAR_REVERSE};
    for (int g = 0; g < 3; g++) {
        for (int r = 0; r < 2; r++) {
            float direction = gears[r] == GEAR_REVERSE ? -1.0f : 1.0f;
            StopGoBench bench;
            bench_init(&bench, grips[g], GEARBOX_MODE_AUTOMATIC, gears[r], -direction * WRONG_WAY_CREEP_MPS);
            step_direct(&bench, direction * DIRECT_PULL_MPS2);
            EXPECT_TRUE(bench.agent.sim_speed_signed == 0.0f);
            EXPECT_TRUE(bench.agent.accel_long == 0.0f);
            for (int step = 1; step < PULL_AWAY_STEP_COUNT; step++) {
                step_direct(&bench, direction * DIRECT_PULL_MPS2);
            }
            EXPECT_TRUE(direction * bench.agent.sim_speed_signed > PULL_AWAY_MIN_SPEED_MPS);
        }
    }
    return 0;
}

// The investigated freeze: skid, assist off, a rounding-sized creep against the gear, and the throttle held.
static int expect_skid_pulls_away_from_a_tiny_wrong_sign_creep(int gearbox_mode, int gear) {
    float direction = gear == GEAR_REVERSE ? -1.0f : 1.0f;
    for (int k = 0; k < WRONG_SIGN_CREEP_COUNT; k++) {
        StopGoBench bench;
        bench_init(&bench, GRIP_MODE_SKID, gearbox_mode, gear, -direction * WRONG_SIGN_CREEPS_MPS[k]);
        int landed_at_rest = 0;
        for (int step = 0; step < PULL_AWAY_STEP_COUNT; step++) {
            step_pedals(&bench, PULL_AWAY_THROTTLE, 0.0f, 0);
            landed_at_rest |= bench.agent.sim_vx == 0.0f && bench.agent.sim_vy == 0.0f;
            EXPECT_TRUE(direction * forward_speed_mps(&bench.agent) >= 0.0f);
        }
        if (!(direction * forward_speed_mps(&bench.agent) > PULL_AWAY_MIN_SPEED_MPS)) {
            printf("  gearbox %d gear %d creep %g: forward speed %g\n", gearbox_mode, gear, WRONG_SIGN_CREEPS_MPS[k],
                   forward_speed_mps(&bench.agent));
        }
        EXPECT_TRUE(landed_at_rest);
        EXPECT_TRUE(direction * forward_speed_mps(&bench.agent) > PULL_AWAY_MIN_SPEED_MPS);
        EXPECT_EQ_INT(bench.env.human_gear == GEAR_REVERSE, gear == GEAR_REVERSE);
    }
    return 0;
}

static int test_skid_throttle_never_freezes_on_a_wrong_sign_creep(void) {
    EXPECT_FALSE(expect_skid_pulls_away_from_a_tiny_wrong_sign_creep(GEARBOX_MODE_AUTOMATIC, 1));
    EXPECT_FALSE(expect_skid_pulls_away_from_a_tiny_wrong_sign_creep(GEARBOX_MODE_NONE, 1));
    EXPECT_FALSE(expect_skid_pulls_away_from_a_tiny_wrong_sign_creep(GEARBOX_MODE_AUTOMATIC, GEAR_REVERSE));
    EXPECT_FALSE(expect_skid_pulls_away_from_a_tiny_wrong_sign_creep(GEARBOX_MODE_NONE, GEAR_REVERSE));
    return 0;
}

static int test_a_skid_stop_without_a_slide_leaves_no_velocity_at_all(void) {
    StopGoBench bench;
    bench_init(&bench, GRIP_MODE_SKID, GEARBOX_MODE_AUTOMATIC, 1, CREEP_SPEED_MPS);
    for (int step = 0; step < PULL_AWAY_STEP_COUNT && bench.agent.sim_speed_signed != 0.0f; step++) {
        step_pedals(&bench, 0.0f, 1.0f, 0);
    }
    EXPECT_TRUE(bench.agent.sim_speed_signed == 0.0f);
    EXPECT_TRUE(bench.agent.sim_vx == 0.0f && bench.agent.sim_vy == 0.0f);
    for (int step = 0; step < PULL_AWAY_STEP_COUNT; step++) {
        step_pedals(&bench, 0.0f, 1.0f, 0);
        EXPECT_TRUE(bench.agent.sim_vx == 0.0f && bench.agent.sim_vy == 0.0f);
    }
    return 0;
}

#define SLIDE_SIDE_SPEED_MPS 3.0f
#define SLIDE_FORWARD_SPEED_MPS 2.0f
#define SLIDE_BRAKE_STEP_COUNT 30

static void set_slide(StopGoBench *bench, float forward_mps, float side_mps) {
    Agent *agent = &bench->agent;
    agent->sim_vx = forward_mps * agent->cos_heading - side_mps * agent->sin_heading;
    agent->sim_vy = forward_mps * agent->sin_heading + side_mps * agent->cos_heading;
    update_agent_speed(agent);
}

static int test_braking_a_pure_sideways_slide_never_creates_forward_speed(void) {
    int gears[] = {1, GEAR_REVERSE};
    for (int r = 0; r < 2; r++) {
        StopGoBench bench;
        bench_init(&bench, GRIP_MODE_SKID, GEARBOX_MODE_AUTOMATIC, gears[r], 0.0f);
        set_slide(&bench, 0.0f, SLIDE_SIDE_SPEED_MPS);
        float previous_speed_mps = bench.agent.sim_speed;
        for (int step = 0; step < SLIDE_BRAKE_STEP_COUNT; step++) {
            step_pedals(&bench, 0.0f, 1.0f, 0);
            if (fabsf(forward_speed_mps(&bench.agent)) >= 1e-6f) {
                printf("  gear %d step %d: braked sideways slide gained forward speed %g (total %g)\n", gears[r], step,
                       forward_speed_mps(&bench.agent), bench.agent.sim_speed);
            }
            EXPECT_TRUE(fabsf(forward_speed_mps(&bench.agent)) < 1e-6f);
            EXPECT_TRUE(bench.agent.sim_speed <= previous_speed_mps + 1e-6f);
            previous_speed_mps = bench.agent.sim_speed;
        }
    }
    return 0;
}

static int test_braking_a_slide_shrinks_the_forward_speed_toward_rest_and_never_past_it(void) {
    float forward_speeds_mps[] = {SLIDE_FORWARD_SPEED_MPS, -SLIDE_FORWARD_SPEED_MPS};
    for (int f = 0; f < 2; f++) {
        StopGoBench bench;
        bench_init(&bench, GRIP_MODE_SKID, GEARBOX_MODE_AUTOMATIC, 1, 0.0f);
        set_slide(&bench, forward_speeds_mps[f], SLIDE_SIDE_SPEED_MPS);
        float previous_forward_mps = forward_speed_mps(&bench.agent);
        for (int step = 0; step < 4 * SLIDE_BRAKE_STEP_COUNT; step++) {
            step_pedals(&bench, 0.0f, 1.0f, 0);
            float forward_mps = forward_speed_mps(&bench.agent);
            EXPECT_TRUE(forward_mps * forward_speeds_mps[f] >= -1e-6f);
            EXPECT_TRUE(fabsf(forward_mps) <= fabsf(previous_forward_mps) + 1e-5f);
            previous_forward_mps = forward_mps;
        }
        EXPECT_TRUE(fabsf(previous_forward_mps) < 1e-5f);
    }
    return 0;
}

static int test_a_braked_car_rolling_back_in_drive_is_only_stopped_never_pushed_forward(void) {
    int gearboxes[] = {GEARBOX_MODE_AUTOMATIC, GEARBOX_MODE_MANUAL};
    for (int g = 0; g < 2; g++) {
        StopGoBench bench;
        bench_init(&bench, GRIP_MODE_KINEMATIC, gearboxes[g], 1, BACKWARD_ROLL_MPS);
        bench.env.human_clutch_norm = 1.0f;
        bench.env.human_brake_norm = 1.0f;
        bench.env.human_longitudinal_mode = HUMAN_LONGITUDINAL_POWERTRAIN;
        float speed_mps = bench.agent.sim_speed_signed;
        float accel_mps2 = human_powertrain_accel_mps2(&bench.env, &bench.agent, 0);
        EXPECT_TRUE(accel_mps2 > 0.0f);
        EXPECT_TRUE(accel_mps2 * bench.env.dt <= fabsf(speed_mps) + 1e-6f);
        for (int step = 0; step < LONG_DRIVE_STEP_COUNT; step++) {
            c_set_human_pedals(&bench.env, 0.0f, 0.0f, 1.0f, gearboxes[g] == GEARBOX_MODE_MANUAL ? 1.0f : 0.0f, 0);
            move_human(&bench.env, 0, 0);
            EXPECT_TRUE(bench.agent.sim_speed_signed <= 0.0f);
        }
        EXPECT_TRUE(bench.agent.sim_speed_signed == 0.0f);
    }
    return 0;
}

static int test_a_braked_car_at_rest_is_held_against_idle_creep(void) {
    StopGoBench bench;
    bench_init(&bench, GRIP_MODE_KINEMATIC, GEARBOX_MODE_AUTOMATIC, 1, 0.0f);
    for (int step = 0; step < LONG_DRIVE_STEP_COUNT; step++) {
        step_pedals(&bench, 0.0f, 1.0f, 0);
        EXPECT_TRUE(bench.agent.sim_speed_signed == 0.0f);
    }
    for (int step = 0; step < PULL_AWAY_STEP_COUNT; step++) {
        step_pedals(&bench, 0.0f, 0.0f, 0);
    }
    EXPECT_TRUE(bench.agent.sim_speed_signed > 0.0f);
    return 0;
}

static void blend_bench(StopGoBench *bench, float speed_mps, float reference_speed_mps, float throttle) {
    bench_init(bench, GRIP_MODE_KINEMATIC, GEARBOX_MODE_AUTOMATIC, 1, speed_mps);
    bench->env.assist_mode = ASSIST_MODE_BLEND;
    bench->env.assist_blend_weight = BLEND_WEIGHT;
    bench->env.assist_blend_target = BLEND_WEIGHT;
    bench->env.assist_reference_valid = 1;
    bench->env.assist_reference_accel_mps2 = POLICY_STOP_ACCEL_MPS2;
    bench->env.assist_reference_speed_mps = reference_speed_mps;
    bench->env.human_throttle_norm = throttle;
}

static int test_with_the_throttle_down_a_stopping_policy_never_brakes_you_under_blend(void) {
    float human_mps2 = 1.5f;
    float stop_speeds[] = {0.0f, POLICY_BACKING_SPEED_MPS};
    for (int k = 0; k < 2; k++) {
        StopGoBench bench;
        blend_bench(&bench, MOVING_FORWARD_MPS, stop_speeds[k], PULL_AWAY_THROTTLE);
        EXPECT_NEAR(assisted_accel(&bench.env, human_mps2, MOVING_FORWARD_MPS), (1.0f - BLEND_WEIGHT) * human_mps2, 1e-6f);
        bench.env.human_throttle_norm = ASSIST_PEDAL_RELEASED_FRACTION;
        EXPECT_NEAR(assisted_accel(&bench.env, human_mps2, MOVING_FORWARD_MPS), (1.0f - BLEND_WEIGHT) * human_mps2, 1e-6f);
    }
    return 0;
}

static int test_with_the_throttle_released_the_old_blend_rules_stand(void) {
    float human_mps2 = 1.5f;
    StopGoBench bench;
    blend_bench(&bench, MOVING_FORWARD_MPS, 0.0f, 0.0f);
    float stop_rule_mps2 = fminf(POLICY_STOP_ACCEL_MPS2, -ASSIST_STOP_DECEL_MPS2);
    EXPECT_NEAR(assisted_accel(&bench.env, human_mps2, MOVING_FORWARD_MPS),
                (1.0f - BLEND_WEIGHT) * human_mps2 + BLEND_WEIGHT * stop_rule_mps2, 1e-6f);
    blend_bench(&bench, MOVING_FORWARD_MPS, POLICY_BACKING_SPEED_MPS, 0.0f);
    EXPECT_NEAR(assisted_accel(&bench.env, human_mps2, MOVING_FORWARD_MPS),
                (1.0f - BLEND_WEIGHT) * human_mps2 + BLEND_WEIGHT * POLICY_STOP_ACCEL_MPS2, 1e-6f);
    blend_bench(&bench, 0.0f, POLICY_BACKING_SPEED_MPS, 0.0f);
    EXPECT_NEAR(assisted_accel(&bench.env, human_mps2, 0.0f), (1.0f - BLEND_WEIGHT) * human_mps2, 1e-6f);
    return 0;
}

static int test_the_throttle_rule_leaves_a_moving_policy_and_full_alone(void) {
    float human_mps2 = 1.5f;
    StopGoBench bench;
    blend_bench(&bench, MOVING_FORWARD_MPS, MOVING_FORWARD_MPS, PULL_AWAY_THROTTLE);
    EXPECT_NEAR(assisted_accel(&bench.env, human_mps2, MOVING_FORWARD_MPS),
                (1.0f - BLEND_WEIGHT) * human_mps2 + BLEND_WEIGHT * POLICY_STOP_ACCEL_MPS2, 1e-6f);
    blend_bench(&bench, MOVING_FORWARD_MPS, POLICY_BACKING_SPEED_MPS, PULL_AWAY_THROTTLE);
    bench.env.assist_mode = ASSIST_MODE_FULL;
    EXPECT_TRUE(assisted_accel(&bench.env, human_mps2, MOVING_FORWARD_MPS) == human_mps2);
    bench.env.human_throttle_norm = 0.0f;
    EXPECT_TRUE(assisted_accel(&bench.env, human_mps2, MOVING_FORWARD_MPS) == POLICY_STOP_ACCEL_MPS2);
    return 0;
}

// The reported regression: Blend 0.5 with a policy that wants to back up, stopped in D, throttle down.
static int test_a_blended_car_stopped_in_drive_pulls_away_on_the_throttle(void) {
    StopGoBench bench;
    blend_bench(&bench, 0.0f, POLICY_BACKING_SPEED_MPS, PULL_AWAY_THROTTLE);
    for (int step = 0; step < LONG_DRIVE_STEP_COUNT; step++) {
        step_pedals(&bench, PULL_AWAY_THROTTLE, 0.0f, 0);
        bench.env.assist_reference_speed_mps = POLICY_BACKING_SPEED_MPS;
        bench.env.assist_reference_accel_mps2 = POLICY_STOP_ACCEL_MPS2;
    }
    EXPECT_TRUE(bench.agent.sim_speed_signed > BLEND_PULL_AWAY_MIN_SPEED_MPS);
    EXPECT_TRUE(bench.env.human_gear >= 1);
    return 0;
}

static int test_drive_is_selected_from_reverse_while_rolling_back_and_the_throttle_goes_forward(void) {
    int gearboxes[] = {GEARBOX_MODE_AUTOMATIC, GEARBOX_MODE_NONE};
    for (int g = 0; g < 2; g++) {
        StopGoBench bench;
        bench_init(&bench, GRIP_MODE_KINEMATIC, gearboxes[g], GEAR_REVERSE, BLEND_BACKING_SPEED_MPS);
        step_pedals(&bench, PULL_AWAY_THROTTLE, 0.0f, 1);
        EXPECT_EQ_INT(bench.env.human_gear, 1);
        float previous_speed_mps = bench.agent.sim_speed_signed;
        for (int step = 0; step < LONG_DRIVE_STEP_COUNT; step++) {
            step_pedals(&bench, PULL_AWAY_THROTTLE, 0.0f, 0);
            EXPECT_TRUE(bench.agent.sim_speed_signed >= previous_speed_mps);
            previous_speed_mps = bench.agent.sim_speed_signed;
        }
        EXPECT_TRUE(bench.agent.sim_speed_signed > BLEND_PULL_AWAY_MIN_SPEED_MPS);
    }
    return 0;
}

static int test_reverse_is_still_refused_while_rolling_forward(void) {
    int gearboxes[] = {GEARBOX_MODE_AUTOMATIC, GEARBOX_MODE_NONE};
    for (int g = 0; g < 2; g++) {
        StopGoBench bench;
        bench_init(&bench, GRIP_MODE_KINEMATIC, gearboxes[g], 1, ROLLING_FORWARD_MPS);
        EXPECT_TRUE(ROLLING_FORWARD_MPS > AUTO_SELECTOR_MAX_SPEED_MPS);
        step_pedals(&bench, 0.0f, 0.0f, -1);
        EXPECT_EQ_INT(bench.env.human_gear, 1);
        bench_init(&bench, GRIP_MODE_KINEMATIC, gearboxes[g], 1, AUTO_SELECTOR_MAX_SPEED_MPS);
        step_pedals(&bench, 0.0f, 0.0f, -1);
        EXPECT_EQ_INT(bench.env.human_gear, GEAR_REVERSE);
    }
    return 0;
}

static int test_blend_steering_override_replaces_the_policy_angle(void) {
    StopGoBench bench;
    blend_bench(&bench, MOVING_FORWARD_MPS, MOVING_FORWARD_MPS, 0.0f);
    bench.env.assist_reference_steering_rad = 0.2f;
    float wheel_rad = -0.1f;
    EXPECT_NEAR(assisted_steering_angle(&bench.env, wheel_rad), 0.5f * wheel_rad + 0.5f * 0.2f, 1e-7f);
    EXPECT_EQ_INT(c_set_blend_steering(&bench.env, 1, -0.3f), 0);
    EXPECT_NEAR(blend_reference_steering_angle(&bench.env), -0.3f, EXACT_TOLERANCE);
    EXPECT_NEAR(assisted_steering_angle(&bench.env, wheel_rad), 0.5f * wheel_rad + 0.5f * -0.3f, 1e-7f);
    EXPECT_EQ_INT(c_set_blend_steering(&bench.env, 1, NAN), -1);
    EXPECT_EQ_INT(c_set_blend_steering(&bench.env, 1, INFINITY), -1);
    EXPECT_NEAR(blend_reference_steering_angle(&bench.env), -0.3f, EXACT_TOLERANCE);
    EXPECT_EQ_INT(c_set_blend_steering(&bench.env, 0, 0.0f), 0);
    EXPECT_NEAR(blend_reference_steering_angle(&bench.env), 0.2f, EXACT_TOLERANCE);
    bench.env.assist_blend_weight = 0.0f;
    c_set_blend_steering(&bench.env, 1, -0.3f);
    EXPECT_TRUE(assisted_steering_angle(&bench.env, wheel_rad) == wheel_rad);
    return 0;
}

static int test_blend_steering_override_is_cleared_by_selecting_a_car_and_by_reset(void) {
    srand(5);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 4, 0);
    env.dynamics_model = DYNAMICS_MODEL_JERK;
    env.action_type = ACTION_TYPE_CONTINUOUS;
    allocate(&env);
    c_reset(&env);
    c_select_human_agent(&env, env.active_agent_indices[0]);
    EXPECT_EQ_INT(c_set_blend_steering(&env, 1, 0.1f), 0);
    c_select_human_agent(&env, env.active_agent_indices[1]);
    EXPECT_EQ_INT(env.assist_blend_steering_valid, 0);
    EXPECT_EQ_INT(c_set_blend_steering(&env, 1, 0.1f), 0);
    c_reset(&env);
    EXPECT_EQ_INT(env.assist_blend_steering_valid, 0);
    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_braking_through_zero_stops_the_car_in_either_gear);
    RUN_TEST(test_a_wrong_way_creep_is_snapped_to_rest_then_pulls_away);
    RUN_TEST(test_skid_throttle_never_freezes_on_a_wrong_sign_creep);
    RUN_TEST(test_a_skid_stop_without_a_slide_leaves_no_velocity_at_all);
    RUN_TEST(test_braking_a_pure_sideways_slide_never_creates_forward_speed);
    RUN_TEST(test_braking_a_slide_shrinks_the_forward_speed_toward_rest_and_never_past_it);
    RUN_TEST(test_a_braked_car_rolling_back_in_drive_is_only_stopped_never_pushed_forward);
    RUN_TEST(test_a_braked_car_at_rest_is_held_against_idle_creep);
    RUN_TEST(test_with_the_throttle_down_a_stopping_policy_never_brakes_you_under_blend);
    RUN_TEST(test_with_the_throttle_released_the_old_blend_rules_stand);
    RUN_TEST(test_the_throttle_rule_leaves_a_moving_policy_and_full_alone);
    RUN_TEST(test_a_blended_car_stopped_in_drive_pulls_away_on_the_throttle);
    RUN_TEST(test_drive_is_selected_from_reverse_while_rolling_back_and_the_throttle_goes_forward);
    RUN_TEST(test_reverse_is_still_refused_while_rolling_forward);
    RUN_TEST(test_blend_steering_override_replaces_the_policy_angle);
    RUN_TEST(test_blend_steering_override_is_cleared_by_selecting_a_car_and_by_reset);
    return test_summary(failures);
}
