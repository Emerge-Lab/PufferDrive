#include "include/drive_fixture.h"
#include "include/test.h"

// Stop line spans y in [-2, 2] at x = 0 and controls traffic moving in +x.
static int stop_sign_lane = 100;

static TrafficControlElement drive_test_stop_sign(void) {
    TrafficControlElement stop_sign = {0};
    stop_sign.type = TRAFFIC_CONTROL_TYPE_STOP_SIGN;
    stop_sign.stop_line[1] = -2.0f;
    stop_sign.stop_line[4] = 2.0f;
    stop_sign.heading = 0.0f;
    stop_sign.num_controlled_lanes = 1;
    stop_sign.controlled_lanes = &stop_sign_lane;
    return stop_sign;
}

static Drive drive_test_stop_sign_env(TrafficControlElement *stop_sign) {
    Drive env = {0};
    env.num_traffic_elements = 1;
    env.traffic_elements = stop_sign;
    return env;
}

static Agent drive_test_stop_sign_agent(float x, float speed) {
    Agent agent = drive_test_agent(x, 0.0f, 0.0f);
    agent.current_lane_idx = stop_sign_lane;
    agent.previous_lane_idx = stop_sign_lane;
    agent.sim_speed = speed;
    agent.sim_speed_signed = speed;
    agent.sim_vx = speed;
    return agent;
}

static void cross_stop_line(Agent *agent) {
    agent->prev_x = -3.0f;
    agent->prev_y = 0.0f;
    agent->prev_cos_heading = 1.0f;
    agent->prev_sin_heading = 0.0f;
    agent->sim_x = 1.0f;
    agent->sim_speed = 2.0f;
    agent->sim_speed_signed = 2.0f;
    agent->sim_vx = 2.0f;
}

static int test_crossing_without_stop_flags(void) {
    TrafficControlElement stop_sign = drive_test_stop_sign();
    Drive env = drive_test_stop_sign_env(&stop_sign);
    Agent agent = drive_test_stop_sign_agent(-3.0f, 2.0f);
    cross_stop_line(&agent);
    EXPECT_TRUE(check_stop_sign_violation(&env, &agent));
    return 0;
}

static int test_required_stop_allows_crossing(void) {
    TrafficControlElement stop_sign = drive_test_stop_sign();
    Drive env = drive_test_stop_sign_env(&stop_sign);
    Agent agent = drive_test_stop_sign_agent(-3.0f, 0.0f);
    for (int i = 0; i < STOP_SIGN_REQUIRED_STOP_TIMESTEPS; i++) {
        EXPECT_FALSE(check_stop_sign_violation(&env, &agent));
    }
    EXPECT_EQ_INT(agent.stop_sign_stopped_timestep_count, STOP_SIGN_REQUIRED_STOP_TIMESTEPS);

    cross_stop_line(&agent);
    EXPECT_FALSE(check_stop_sign_violation(&env, &agent));
    EXPECT_EQ_INT(agent.stop_sign_stopped_timestep_count, 0);
    return 0;
}

static int test_partial_stop_flags_crossing(void) {
    TrafficControlElement stop_sign = drive_test_stop_sign();
    Drive env = drive_test_stop_sign_env(&stop_sign);
    Agent agent = drive_test_stop_sign_agent(-3.0f, 0.0f);
    for (int i = 0; i < STOP_SIGN_REQUIRED_STOP_TIMESTEPS - 1; i++) {
        EXPECT_FALSE(check_stop_sign_violation(&env, &agent));
    }

    cross_stop_line(&agent);
    EXPECT_TRUE(check_stop_sign_violation(&env, &agent));
    return 0;
}

static int test_moving_resets_partial_stop(void) {
    TrafficControlElement stop_sign = drive_test_stop_sign();
    Drive env = drive_test_stop_sign_env(&stop_sign);
    Agent agent = drive_test_stop_sign_agent(-3.0f, 0.0f);
    for (int i = 0; i < STOP_SIGN_REQUIRED_STOP_TIMESTEPS - 1; i++) {
        EXPECT_FALSE(check_stop_sign_violation(&env, &agent));
    }
    agent.sim_speed = 1.0f;
    EXPECT_FALSE(check_stop_sign_violation(&env, &agent));
    EXPECT_EQ_INT(agent.stop_sign_stopped_timestep_count, 0);
    return 0;
}

static int test_lane_match_does_not_gate_crossing(void) {
    TrafficControlElement stop_sign = drive_test_stop_sign();
    Drive env = drive_test_stop_sign_env(&stop_sign);
    Agent agent = drive_test_stop_sign_agent(-3.0f, 2.0f);
    agent.current_lane_idx = -1;
    cross_stop_line(&agent);
    EXPECT_TRUE(check_stop_sign_violation(&env, &agent));
    return 0;
}

static int test_beyond_five_meters_does_not_accumulate_stop(void) {
    TrafficControlElement stop_sign = drive_test_stop_sign();
    Drive env = drive_test_stop_sign_env(&stop_sign);
    Agent agent = drive_test_stop_sign_agent(-5.0f, 0.0f);
    EXPECT_FALSE(check_stop_sign_violation(&env, &agent));
    EXPECT_EQ_INT(agent.stop_sign_stopped_timestep_count, 1);

    agent.sim_x = -5.1f;
    copy_pose_to_prev(&agent);
    EXPECT_FALSE(check_stop_sign_violation(&env, &agent));
    EXPECT_EQ_INT(agent.stop_sign_stopped_timestep_count, 0);
    return 0;
}

static int test_stop_after_line_does_not_qualify(void) {
    TrafficControlElement stop_sign = drive_test_stop_sign();
    Drive env = drive_test_stop_sign_env(&stop_sign);
    Agent agent = drive_test_stop_sign_agent(1.0f, 0.0f);
    for (int i = 0; i < STOP_SIGN_REQUIRED_STOP_TIMESTEPS; i++) {
        EXPECT_FALSE(check_stop_sign_violation(&env, &agent));
    }
    EXPECT_EQ_INT(agent.stop_sign_stopped_timestep_count, 0);
    return 0;
}

static int test_leaving_proximity_resets_stop(void) {
    TrafficControlElement stop_sign = drive_test_stop_sign();
    Drive env = drive_test_stop_sign_env(&stop_sign);
    Agent agent = drive_test_stop_sign_agent(-3.0f, 0.0f);
    EXPECT_FALSE(check_stop_sign_violation(&env, &agent));
    EXPECT_EQ_INT(agent.stop_sign_stopped_timestep_count, 1);

    agent.sim_x = -20.0f;
    copy_pose_to_prev(&agent);
    EXPECT_FALSE(check_stop_sign_violation(&env, &agent));
    EXPECT_EQ_INT(agent.stop_sign_stopped_timestep_count, 0);
    return 0;
}

static int test_wrong_heading_is_ignored(void) {
    TrafficControlElement stop_sign = drive_test_stop_sign();
    Drive env = drive_test_stop_sign_env(&stop_sign);
    Agent agent = drive_test_stop_sign_agent(-3.0f, 0.0f);
    agent.sim_heading = M_PI;
    agent.cos_heading = -1.0f;
    agent.sin_heading = 0.0f;
    copy_pose_to_prev(&agent);
    EXPECT_FALSE(check_stop_sign_violation(&env, &agent));
    EXPECT_EQ_INT(agent.stop_sign_stopped_timestep_count, 0);
    return 0;
}

static int test_ninety_degree_heading_is_in_scope(void) {
    TrafficControlElement stop_sign = drive_test_stop_sign();
    Drive env = drive_test_stop_sign_env(&stop_sign);
    Agent agent = drive_test_stop_sign_agent(-3.0f, 0.0f);
    agent.sim_heading = M_PI / 2.0f;
    agent.cos_heading = 0.0f;
    agent.sin_heading = 1.0f;
    copy_pose_to_prev(&agent);
    EXPECT_FALSE(check_stop_sign_violation(&env, &agent));
    EXPECT_EQ_INT(agent.stop_sign_stopped_timestep_count, 1);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_crossing_without_stop_flags);
    RUN_TEST(test_required_stop_allows_crossing);
    RUN_TEST(test_partial_stop_flags_crossing);
    RUN_TEST(test_moving_resets_partial_stop);
    RUN_TEST(test_lane_match_does_not_gate_crossing);
    RUN_TEST(test_beyond_five_meters_does_not_accumulate_stop);
    RUN_TEST(test_stop_after_line_does_not_qualify);
    RUN_TEST(test_leaving_proximity_resets_stop);
    RUN_TEST(test_wrong_heading_is_ignored);
    RUN_TEST(test_ninety_degree_heading_is_in_scope);
    return test_summary(failures);
}
