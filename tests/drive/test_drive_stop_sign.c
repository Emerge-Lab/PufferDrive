#include "include/drive_fixture.h"
#include "include/test.h"

// Stop line spans y in [-2, 2] at x = 0, traffic flows +x; the trigger box is centred on it,
// 1.7 m deep on each side. Agents are 4 m long, so a stop with the nose at the line overlaps it.
static int stop_lanes[1] = {100};

static TrafficControlElement drive_test_stop_sign(void) {
    TrafficControlElement tc = {0};
    tc.type = TRAFFIC_CONTROL_TYPE_STOP_SIGN;
    tc.stop_line[1] = -2.0f;
    tc.stop_line[4] = 2.0f;
    tc.heading = 0.0f;
    tc.num_controlled_lanes = 1;
    tc.controlled_lanes = stop_lanes;
    return tc;
}

static Drive drive_test_stop_env(TrafficControlElement *tc, Agent *agent) {
    Drive env = {0};
    env.num_traffic_elements = 1;
    env.traffic_elements = tc;
    env.agents = agent;
    return env;
}

static Agent drive_test_stop_agent(float x, float y, float heading) {
    Agent agent = drive_test_agent(x, y, heading);
    reset_agent_state(&agent);
    agent.current_lane_idx = 100;
    agent.previous_lane_idx = 100;
    return agent;
}

// One sim step: the agent moved prev -> cur with the given speed (positive = forward).
static void drive_test_step_to(Agent *agent, float x, float y, float speed) {
    copy_pose_to_prev(agent);
    agent->sim_x = x;
    agent->sim_y = y;
    agent->sim_speed = fabsf(speed);
    agent->sim_speed_signed = speed;
}

static void drive_test_turn_to(Agent *agent, float heading) {
    agent->sim_heading = heading;
    agent->cos_heading = cosf(heading);
    agent->sin_heading = sinf(heading);
}

static int test_approach_acquires_target(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-15.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -14.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, 0);
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 0);
    return 0;
}

static int test_far_sign_not_targeted(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-26.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -25.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    return 0;
}

static int test_stop_in_box_then_cross_no_flag(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    // nose 0.5 m before the line, standing still
    drive_test_step_to(&agent, -2.5f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 1);
    drive_test_step_to(&agent, -0.5f, 0.0f, 3.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 0.5f, 0.0f, 3.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, 0);
    EXPECT_EQ_INT(agent.stop_sign_last_failed_idx, -1);
    return 0;
}

static int test_cross_without_stop_flags_once(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, -0.5f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 0.5f, 0.0f, 10.0f);
    EXPECT_TRUE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    EXPECT_EQ_INT(agent.stop_sign_last_failed_idx, 0);
    // still inside the scaled box: the run sign is not re-targeted, nothing fires twice
    drive_test_step_to(&agent, 1.5f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    return 0;
}

static int test_stop_before_box_does_not_count(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    // nose 4 m before the line: outside the 1.7 m deep trigger box
    drive_test_step_to(&agent, -6.0f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 0);
    drive_test_step_to(&agent, -0.5f, 0.0f, 5.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 0.5f, 0.0f, 5.0f);
    EXPECT_TRUE(update_stop_sign_state(&env, 0));
    return 0;
}

static int test_slow_roll_in_box_does_not_count(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, -2.5f, 0.0f, 0.3f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 0);
    drive_test_step_to(&agent, -0.5f, 0.0f, 0.3f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 0.5f, 0.0f, 0.3f);
    EXPECT_TRUE(update_stop_sign_state(&env, 0));
    return 0;
}

static int test_swerve_around_line_flags(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    // acquired on the lane, then crossing 6 m beside the painted line
    drive_test_step_to(&agent, -0.5f, 6.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 0.5f, 6.0f, 10.0f);
    EXPECT_TRUE(update_stop_sign_state(&env, 0));
    return 0;
}

static int test_parallel_lane_never_targeted(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 6.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 6.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    drive_test_step_to(&agent, -0.5f, 6.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 0.5f, 6.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    return 0;
}

static int test_opposing_direction_not_targeted(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(6.0f, 0.0f, (float) M_PI);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, 5.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    return 0;
}

static int test_cross_street_not_targeted(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(1.0f, -6.0f, (float) M_PI / 2.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, 1.0f, -5.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    return 0;
}

static int test_reversing_not_targeted(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-4.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -5.0f, 0.0f, -1.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    return 0;
}

static int test_different_z_level_not_targeted(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    agent.sim_z = 6.0f;
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    return 0;
}

static int test_state_clears_beyond_proximity(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, -0.5f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 0.5f, 0.0f, 10.0f);
    EXPECT_TRUE(update_stop_sign_state(&env, 0));
    drive_test_step_to(&agent, 25.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_last_failed_idx, -1);
    // a completed stop is forgotten the same way, so a later approach must stop again
    Agent again = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    env.agents = &again;
    drive_test_step_to(&again, -2.5f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(again.stop_sign_stop_completed, 1);
    drive_test_step_to(&again, 25.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(again.stop_sign_target_idx, -1);
    EXPECT_EQ_INT(again.stop_sign_stop_completed, 0);
    return 0;
}

static int test_spawn_inside_box_counts_as_stopped(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-1.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 1);
    drive_test_step_to(&agent, 1.0f, 0.0f, 2.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    return 0;
}

// Mid U-turn: standing still inside the box while pointed 115 deg off the lane, then aligning and crossing.
static int test_standstill_before_targeting_counts(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-0.7f, 0.0f, 2.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -0.7f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    EXPECT_EQ_INT(agent.stop_sign_standstill_idx, 0);
    drive_test_turn_to(&agent, 0.0f);
    drive_test_step_to(&agent, -0.3f, 0.0f, 1.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, 0);
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 1);
    drive_test_step_to(&agent, 0.5f, 0.0f, 1.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_last_failed_idx, -1);
    return 0;
}

static int test_standstill_outside_box_before_targeting_does_not_count(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-6.0f, 0.0f, 2.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -6.0f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_standstill_idx, -1);
    drive_test_turn_to(&agent, 0.0f);
    drive_test_step_to(&agent, -0.5f, 0.0f, 5.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 0);
    drive_test_step_to(&agent, 0.5f, 0.0f, 5.0f);
    EXPECT_TRUE(update_stop_sign_state(&env, 0));
    return 0;
}

static int test_standstill_clears_beyond_proximity(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    Agent agent = drive_test_stop_agent(-0.7f, 0.0f, 2.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -0.7f, 0.0f, 0.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_standstill_idx, 0);
    drive_test_step_to(&agent, -25.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_standstill_idx, -1);
    return 0;
}

static int test_traffic_light_ignored(void) {
    TrafficControlElement tc = drive_test_stop_sign();
    tc.type = TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
    Agent agent = drive_test_stop_agent(-12.0f, 0.0f, 0.0f);
    Drive env = drive_test_stop_env(&tc, &agent);
    drive_test_step_to(&agent, -10.0f, 0.0f, 10.0f);
    EXPECT_FALSE(update_stop_sign_state(&env, 0));
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    return 0;
}

static int test_obb_collision_unchanged_by_shared_sat(void) {
    Agent ego = drive_test_agent(0.0f, 0.0f, 0.0f);
    Agent other = drive_test_agent(3.0f, 1.5f, 0.5f);
    EXPECT_TRUE(check_obb_collision(&ego, &other));
    other = drive_test_agent(6.0f, 0.0f, 0.0f);
    EXPECT_FALSE(check_obb_collision(&ego, &other));
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_approach_acquires_target);
    RUN_TEST(test_far_sign_not_targeted);
    RUN_TEST(test_stop_in_box_then_cross_no_flag);
    RUN_TEST(test_cross_without_stop_flags_once);
    RUN_TEST(test_stop_before_box_does_not_count);
    RUN_TEST(test_slow_roll_in_box_does_not_count);
    RUN_TEST(test_swerve_around_line_flags);
    RUN_TEST(test_parallel_lane_never_targeted);
    RUN_TEST(test_opposing_direction_not_targeted);
    RUN_TEST(test_cross_street_not_targeted);
    RUN_TEST(test_reversing_not_targeted);
    RUN_TEST(test_different_z_level_not_targeted);
    RUN_TEST(test_state_clears_beyond_proximity);
    RUN_TEST(test_spawn_inside_box_counts_as_stopped);
    RUN_TEST(test_standstill_before_targeting_counts);
    RUN_TEST(test_standstill_outside_box_before_targeting_does_not_count);
    RUN_TEST(test_standstill_clears_beyond_proximity);
    RUN_TEST(test_traffic_light_ignored);
    RUN_TEST(test_obb_collision_unchanged_by_shared_sat);
    return test_summary(failures);
}
