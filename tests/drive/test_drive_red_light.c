#include "include/drive_fixture.h"
#include "include/test.h"

// Stop line spans y in [-2, 2] at x = 0, traffic flows +x. Controlled lanes only
// matter for the spawn overlap check; violation detection is purely geometric.
#define LIGHT_STATE_COUNT 16
static int light_states[LIGHT_STATE_COUNT];
static int light_lanes[2] = {100, 101};

static TrafficControlElement drive_test_light(void) {
    TrafficControlElement tc = {0};
    tc.type = TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
    tc.stop_line[1] = -2.0f;
    tc.stop_line[4] = 2.0f;
    tc.heading = 0.0f;
    tc.state_size = LIGHT_STATE_COUNT;
    tc.states = light_states;
    tc.num_controlled_lanes = 2;
    tc.controlled_lanes = light_lanes;
    return tc;
}

static void set_light(int state) {
    for (int t = 0; t < LIGHT_STATE_COUNT; t++) {
        light_states[t] = state;
    }
}

static Drive drive_test_light_env(TrafficControlElement *tc) {
    Drive env = {0};
    env.num_traffic_elements = 1;
    env.traffic_elements = tc;
    env.timestep = 5;
    return env;
}

// Agent length 4: rear bumper sits 2 m behind center; violation = rear crossing.
static Agent drive_test_driving_agent(float prev_x, float cur_x, float y) {
    Agent agent = drive_test_agent(prev_x, y, 0.0f);
    agent.sim_x = cur_x;
    agent.current_lane_idx = 100;
    agent.previous_lane_idx = 100;
    return agent;
}

static int test_red_crossing_flags(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    Agent agent = drive_test_driving_agent(1.5f, 2.5f, 0.0f);
    env.agents = &agent;
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_straddle_then_proceed_on_red_flags(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    // nose already over the line when the light turned red; proceeding completes entry
    Agent agent = drive_test_driving_agent(-0.5f, 2.5f, 0.0f);
    env.agents = &agent;
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_green_crossing_then_red_inside_no_flag(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_GREEN);
    Agent agent = drive_test_driving_agent(1.5f, 2.5f, 0.0f);
    env.agents = &agent;
    EXPECT_TRUE(!check_red_light_violation(&env, 0));
    // light turns red once the agent is fully inside the junction
    set_light(TRAFFIC_CONTROL_STATE_RED);
    agent = drive_test_driving_agent(4.0f, 5.0f, 0.0f);
    EXPECT_TRUE(!check_red_light_violation(&env, 0));
    return 0;
}

static int test_stationary_straddle_no_flag(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    Agent agent = drive_test_driving_agent(-0.5f, -0.5f, 0.0f);
    env.agents = &agent;
    EXPECT_TRUE(!check_red_light_violation(&env, 0));
    // the overlap check (spawn semantics) still sees the straddle
    EXPECT_TRUE(check_agent_on_stop_line(&env, &agent, false));
    return 0;
}

static int test_crossing_fires_once(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    Agent agent = drive_test_driving_agent(2.5f, 3.5f, 0.0f);
    env.agents = &agent;
    EXPECT_TRUE(!check_red_light_violation(&env, 0));
    return 0;
}

static int test_opposing_lane_bypass_flags(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    // crossing 5 m beside the painted line (opposing lane) is still a violation
    Agent agent = drive_test_driving_agent(1.5f, 2.5f, 5.0f);
    env.agents = &agent;
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_crossing_beyond_virtual_extension_no_flag(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    Agent agent = drive_test_driving_agent(1.5f, 2.5f, 18.0f);
    env.agents = &agent;
    EXPECT_TRUE(!check_red_light_violation(&env, 0));
    return 0;
}

static int test_exiting_direction_no_flag(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    // opposing-direction agent leaves the junction across the line region
    Agent agent = drive_test_agent(1.0f, 0.0f, (float) M_PI);
    agent.sim_x = -1.0f;
    env.agents = &agent;
    EXPECT_TRUE(!check_red_light_violation(&env, 0));
    return 0;
}

static int test_diagonal_entry_flags(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    // 1.2 rad off the light heading is still entering (< pi/2 gate)
    Agent agent = drive_test_agent(0.5f, 0.0f, 1.2f);
    agent.sim_x = 1.0f;
    env.agents = &agent;
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_perpendicular_crossing_no_flag(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    // 1.8 rad off the light heading is not entering (>= pi/2 gate)
    Agent agent = drive_test_agent(-0.5f, 0.0f, 1.8f);
    agent.sim_x = 0.0f;
    env.agents = &agent;
    EXPECT_TRUE(!check_red_light_violation(&env, 0));
    return 0;
}

static int test_different_z_level_no_flag(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    Agent agent = drive_test_driving_agent(1.5f, 2.5f, 0.0f);
    agent.sim_z = 6.0f;
    env.agents = &agent;
    EXPECT_TRUE(!check_red_light_violation(&env, 0));
    return 0;
}

static int test_no_lane_assignment_still_flags(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    Agent agent = drive_test_driving_agent(1.5f, 2.5f, 0.0f);
    agent.current_lane_idx = -1;
    agent.previous_lane_idx = -1;
    env.agents = &agent;
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_red_crossing_flags);
    RUN_TEST(test_straddle_then_proceed_on_red_flags);
    RUN_TEST(test_green_crossing_then_red_inside_no_flag);
    RUN_TEST(test_stationary_straddle_no_flag);
    RUN_TEST(test_crossing_fires_once);
    RUN_TEST(test_opposing_lane_bypass_flags);
    RUN_TEST(test_crossing_beyond_virtual_extension_no_flag);
    RUN_TEST(test_exiting_direction_no_flag);
    RUN_TEST(test_diagonal_entry_flags);
    RUN_TEST(test_perpendicular_crossing_no_flag);
    RUN_TEST(test_different_z_level_no_flag);
    RUN_TEST(test_no_lane_assignment_still_flags);
    return test_summary(failures);
}
