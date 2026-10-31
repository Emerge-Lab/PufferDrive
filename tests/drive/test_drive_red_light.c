#include "include/drive_fixture.h"
#include "include/test.h"

// Stop line spans y in [-2, 2] at x = 0, traffic flows +x. Controlled lanes only
// matter for the spawn overlap check; violation detection is purely geometric.
#define LIGHT_STATE_COUNT 16
#define TEST_HALF_LENGTH_M 2.0f // drive_test_agent is 4 m long: front edge = center + 2 m at heading 0
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

// Violation = on red the line lies between the tail clearance point (rear - 1 m) and the front edge, moving or
// not; arguments are front-edge x (rear = front - 4 m).
static Agent drive_test_driving_agent(float prev_front_x, float cur_front_x, float y) {
    Agent agent = drive_test_agent(prev_front_x - TEST_HALF_LENGTH_M, y, 0.0f);
    agent.sim_x = cur_front_x - TEST_HALF_LENGTH_M;
    agent.current_lane_idx = 100;
    agent.previous_lane_idx = 100;
    return agent;
}

static int test_red_crossing_flags(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    Agent agent = drive_test_driving_agent(-0.5f, 0.5f, 0.0f);
    env.agents = &agent;
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_front_past_line_on_red_flags(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    // nose in the junction, rear before the line: creeping on red is a violation however the front got there
    Agent agent = drive_test_driving_agent(0.5f, 2.5f, 0.0f);
    env.agents = &agent;
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_stationary_straddle_flags(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    // parked with the tail on the line when the light turns red
    Agent agent = drive_test_driving_agent(1.5f, 1.5f, 0.0f);
    env.agents = &agent;
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    EXPECT_TRUE(check_agent_on_stop_line(&env, &agent, false));
    return 0;
}

static int test_green_crossing_then_red_inside_no_flag(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_GREEN);
    Agent agent = drive_test_driving_agent(-0.5f, 0.5f, 0.0f);
    env.agents = &agent;
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    // light turns red once the rear (front - 4 m) is more than the 1 m clearance past the line
    set_light(TRAFFIC_CONTROL_STATE_RED);
    agent = drive_test_driving_agent(5.5f, 6.5f, 0.0f);
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_tail_clearance_on_red(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    // rear 0.9 m past the line: the clearance point still sits before it
    Agent agent = drive_test_driving_agent(4.9f, 4.9f, 0.0f);
    env.agents = &agent;
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    agent = drive_test_driving_agent(5.1f, 5.1f, 0.0f);
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_rear_crossing_on_red_flags(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    // entered on yellow, red while straddling: the rear (front - 4 m) clears the line on red
    Agent agent = drive_test_driving_agent(3.5f, 4.5f, 0.0f);
    env.agents = &agent;
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_violation_fires_once_per_stay(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    Agent agent = drive_test_driving_agent(1.5f, 1.5f, 0.0f);
    env.agents = &agent;
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    // a green phase spent on the line does not re-arm the latch
    set_light(TRAFFIC_CONTROL_STATE_GREEN);
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    set_light(TRAFFIC_CONTROL_STATE_RED);
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    // rear 3.5 m past the line: footprint off it, re-armed
    agent.sim_x = 7.5f - TEST_HALF_LENGTH_M;
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    // reversed back onto it
    agent.sim_x = 1.5f - TEST_HALF_LENGTH_M;
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_opposing_lane_bypass_flags(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    // crossing 5 m beside the painted line (opposing lane) is still a violation
    Agent agent = drive_test_driving_agent(-0.5f, 0.5f, 5.0f);
    env.agents = &agent;
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_crossing_beyond_virtual_extension_no_flag(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    Agent agent = drive_test_driving_agent(-0.5f, 0.5f, 18.0f);
    env.agents = &agent;
    EXPECT_FALSE(check_red_light_violation(&env, 0));
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
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_diagonal_entry_flags(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    // 1.2 rad off the light heading is still entering (< pi/2 gate); the front edge sits at x 0.22
    Agent agent = drive_test_agent(-1.5f, 0.0f, 1.2f);
    agent.sim_x = -0.5f;
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
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_different_z_level_no_flag(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    Agent agent = drive_test_driving_agent(-0.5f, 0.5f, 0.0f);
    agent.sim_z = 6.0f;
    env.agents = &agent;
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_no_lane_assignment_still_flags(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    set_light(TRAFFIC_CONTROL_STATE_RED);
    Agent agent = drive_test_driving_agent(-0.5f, 0.5f, 0.0f);
    agent.current_lane_idx = -1;
    agent.previous_lane_idx = -1;
    env.agents = &agent;
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_training_margin_flags_before_line(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    env.red_light_train_front_margin_m = 0.3f;
    set_light(TRAFFIC_CONTROL_STATE_RED);
    Agent agent = drive_test_driving_agent(-0.5f, -0.4f, 0.0f);
    env.agents = &agent;
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    agent = drive_test_driving_agent(-0.5f, -0.2f, 0.0f);
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_training_rear_margin_extends_tail_clearance(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    env.red_light_train_rear_margin_m = 1.0f;
    set_light(TRAFFIC_CONTROL_STATE_RED);
    // rear 2.5 m past the line clears 1 m + margin
    Agent agent = drive_test_driving_agent(6.5f, 6.5f, 0.0f);
    env.agents = &agent;
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    agent = drive_test_driving_agent(5.9f, 5.9f, 0.0f);
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    // eval semantics: the real 1 m clearance
    env.eval_mode = 1;
    agent = drive_test_driving_agent(5.9f, 5.9f, 0.0f);
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    agent = drive_test_driving_agent(4.9f, 4.9f, 0.0f);
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_eval_semantics_drop_training_margin(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    env.red_light_train_front_margin_m = 0.3f;
    env.eval_mode = 1;
    set_light(TRAFFIC_CONTROL_STATE_RED);
    Agent agent = drive_test_driving_agent(-0.5f, -0.2f, 0.0f);
    env.agents = &agent;
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    agent = drive_test_driving_agent(-0.1f, 0.1f, 0.0f);
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    // rendering the training distribution keeps the training margin
    env.eval_training_render = 1;
    agent = drive_test_driving_agent(-0.5f, -0.2f, 0.0f);
    EXPECT_TRUE(check_red_light_violation(&env, 0));
    // co-sim shadow envs use eval semantics without eval_mode
    env.eval_mode = 0;
    env.eval_training_render = 0;
    env.cosim_eval_semantics = 1;
    agent = drive_test_driving_agent(-0.5f, -0.2f, 0.0f);
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    return 0;
}

static int test_perceived_size_margin_never_inflates_detection(void) {
    TrafficControlElement tc = drive_test_light();
    Drive env = drive_test_light_env(&tc);
    env.eval_mode = 1;
    env.eval_perceived_size_margin_m = 0.1f;
    set_light(TRAFFIC_CONTROL_STATE_RED);
    // eval observes the ego 0.2 m longer; the detector keeps the real front edge, which stops 5 cm short
    Agent agent = drive_test_driving_agent(-0.15f, -0.05f, 0.0f);
    env.agents = &agent;
    EXPECT_FALSE(check_red_light_violation(&env, 0));
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_red_crossing_flags);
    RUN_TEST(test_front_past_line_on_red_flags);
    RUN_TEST(test_stationary_straddle_flags);
    RUN_TEST(test_green_crossing_then_red_inside_no_flag);
    RUN_TEST(test_tail_clearance_on_red);
    RUN_TEST(test_rear_crossing_on_red_flags);
    RUN_TEST(test_violation_fires_once_per_stay);
    RUN_TEST(test_opposing_lane_bypass_flags);
    RUN_TEST(test_crossing_beyond_virtual_extension_no_flag);
    RUN_TEST(test_exiting_direction_no_flag);
    RUN_TEST(test_diagonal_entry_flags);
    RUN_TEST(test_perpendicular_crossing_no_flag);
    RUN_TEST(test_different_z_level_no_flag);
    RUN_TEST(test_no_lane_assignment_still_flags);
    RUN_TEST(test_training_margin_flags_before_line);
    RUN_TEST(test_training_rear_margin_extends_tail_clearance);
    RUN_TEST(test_eval_semantics_drop_training_margin);
    RUN_TEST(test_perceived_size_margin_never_inflates_detection);
    return test_summary(failures);
}
