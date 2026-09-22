#include "include/drive_fixture.h"
#include "include/test.h"

static int test_heading_normalization(void) {
    EXPECT_NEAR(normalize_heading(3.0f * (float) M_PI), (float) M_PI, 1e-5f);
    EXPECT_NEAR(normalize_heading(-3.0f * (float) M_PI), -(float) M_PI, 1e-5f);
    EXPECT_NEAR(compute_heading_diff(1.5f * (float) M_PI, 0.0f), -0.5f * (float) M_PI, 1e-5f);
    EXPECT_NEAR(compute_heading_diff(-1.5f * (float) M_PI, 0.0f), 0.5f * (float) M_PI, 1e-5f);
    return 0;
}

static int test_point_to_segment_distance_cases(void) {
    EXPECT_NEAR(compute_point_to_segment_distance(0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f), 1.0f, 1e-5f);
    EXPECT_NEAR(compute_point_to_segment_distance(2.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f), 1.0f, 1e-5f);
    EXPECT_NEAR(compute_point_to_segment_distance(0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f), 0.0f, 1e-5f);
    EXPECT_NEAR(compute_point_to_segment_distance(3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f), 5.0f, 1e-5f);
    return 0;
}

static int test_obb_collision_cases(void) {
    Agent ego = drive_test_agent(0.0f, 0.0f, 0.0f);
    Agent other = drive_test_agent(3.9f, 0.0f, 0.0f);
    EXPECT_TRUE(check_obb_collision(&ego, &other));

    other = drive_test_agent(4.1f, 0.0f, 0.0f);
    EXPECT_FALSE(check_obb_collision(&ego, &other));

    other = drive_test_agent(4.0f, 0.0f, 0.0f);
    EXPECT_TRUE(check_obb_collision(&ego, &other));

    other = drive_test_agent(0.0f, 0.0f, 0.0f);
    other.sim_z = 2.0f;
    EXPECT_FALSE(check_obb_collision(&ego, &other));

    other = drive_test_agent(1.0f, 0.0f, (float) M_PI / 4.0f);
    EXPECT_TRUE(check_obb_collision(&ego, &other));

    other = drive_test_agent(4.4f, 0.0f, (float) M_PI / 4.0f);
    EXPECT_FALSE(check_obb_collision(&ego, &other));

    return 0;
}

static int test_obb_collision_dimensions(void) {
    Agent ego = drive_test_agent(0.0f, 0.0f, 0.0f);
    ego.sim_length = 12.0f;
    ego.sim_width = 2.5f;
    update_agent_radius(&ego);

    Agent other = drive_test_agent(7.9f, 0.0f, 0.0f);
    EXPECT_TRUE(check_obb_collision(&ego, &other));
    other = drive_test_agent(8.1f, 0.0f, 0.0f);
    EXPECT_FALSE(check_obb_collision(&ego, &other));

    ego = drive_test_agent(0.0f, 0.0f, 0.0f);
    ego.sim_width = 1.0f;
    update_agent_radius(&ego);
    other = drive_test_agent(0.0f, 1.4f, 0.0f);
    other.sim_width = 1.0f;
    update_agent_radius(&other);
    EXPECT_FALSE(check_obb_collision(&ego, &other));
    other = drive_test_agent(0.0f, 0.9f, 0.0f);
    other.sim_width = 1.0f;
    update_agent_radius(&other);
    EXPECT_TRUE(check_obb_collision(&ego, &other));

    return 0;
}

static int test_moving_obb_collision_cases(void) {
    Agent ego = drive_test_agent(8.0f, 0.0f, 0.0f);
    Agent other = drive_test_agent(0.0f, 0.0f, 0.0f);
    ego.prev_x = -8.0f;
    EXPECT_TRUE(check_moving_obb_collision(&ego, &other, 16.0f, 0.0f));

    ego = drive_test_agent(3.0f, 0.0f, 0.0f);
    other = drive_test_agent(-3.0f, 0.0f, 0.0f);
    ego.prev_x = -3.0f;
    other.prev_x = 3.0f;
    EXPECT_TRUE(check_moving_obb_collision(&ego, &other, 6.0f, 6.0f));

    ego = drive_test_agent(0.0f, 0.0f, 0.0f);
    other = drive_test_agent(0.0f, 3.0f, 0.0f);
    ego.prev_x = -5.0f;
    other.prev_x = -5.0f;
    EXPECT_FALSE(check_moving_obb_collision(&ego, &other, 5.0f, 5.0f));

    return 0;
}

static int test_moving_obb_collision_dimensions(void) {
    Agent ego = drive_test_agent(10.0f, 0.0f, 0.0f);
    ego.sim_length = 2.0f;
    ego.sim_width = 2.0f;
    update_agent_radius(&ego);
    ego.prev_x = -10.0f;
    Agent other = drive_test_agent(0.0f, 0.0f, 0.0f);
    other.sim_length = 1.0f;
    other.sim_width = 1.0f;
    update_agent_radius(&other);
    EXPECT_TRUE(check_moving_obb_collision(&ego, &other, 20.0f, 0.0f));

    other = drive_test_agent(0.0f, 2.0f, 0.0f);
    other.sim_length = 1.0f;
    other.sim_width = 1.0f;
    update_agent_radius(&other);
    EXPECT_FALSE(check_moving_obb_collision(&ego, &other, 20.0f, 0.0f));

    return 0;
}

static int test_collision_check_filters(void) {
    Drive env = {0};
    Agent agents[5] = {0};
    agents[0] = drive_test_agent(0.0f, 0.0f, 0.0f);
    agents[1] = drive_test_agent(0.0f, 0.0f, 0.0f);
    agents[1].removed = 1;
    agents[2] = drive_test_agent(INVALID_POSITION, 0.0f, 0.0f);
    agents[3] = drive_test_agent(0.0f, 0.0f, 0.0f);
    agents[3].sim_valid = 0;
    agents[4] = drive_test_agent(3.9f, 0.0f, 0.0f);

    env.agents = agents;
    env.num_total_agents = 5;
    env.num_agents = 2;

    EXPECT_EQ_INT(collision_check(&env, &agents[0]), 4);
    return 0;
}

static int test_segment_aabb_cases(void) {
    float through_a[2] = {-3.0f, 0.0f};
    float through_b[2] = {3.0f, 0.0f};
    EXPECT_TRUE(check_segment_intersects_aabb(through_a, through_b, 1.0f, 1.0f));

    float parallel_a[2] = {-3.0f, 2.0f};
    float parallel_b[2] = {3.0f, 2.0f};
    EXPECT_FALSE(check_segment_intersects_aabb(parallel_a, parallel_b, 1.0f, 1.0f));

    float inside[2] = {0.5f, 0.5f};
    EXPECT_TRUE(check_segment_intersects_aabb(inside, inside, 1.0f, 1.0f));

    float outside[2] = {2.0f, 0.0f};
    EXPECT_FALSE(check_segment_intersects_aabb(outside, outside, 1.0f, 1.0f));

    float tangent_a[2] = {-3.0f, 1.0f};
    float tangent_b[2] = {3.0f, 1.0f};
    EXPECT_TRUE(check_segment_intersects_aabb(tangent_a, tangent_b, 1.0f, 1.0f));

    return 0;
}

static int test_swept_line_box_cases(void) {
    Agent agent = drive_test_agent(4.0f, 0.0f, 0.0f);
    agent.prev_x = -4.0f;
    EXPECT_TRUE(check_segment_crosses_moving_box(0.0f, -3.0f, 0.0f, 3.0f, &agent));

    agent = drive_test_agent(0.0f, 5.0f, 0.0f);
    agent.prev_y = -5.0f;
    EXPECT_TRUE(check_segment_crosses_moving_box(-10.0f, 0.0f, 10.0f, 0.0f, &agent));

    agent = drive_test_agent(4.0f, 0.0f, 0.0f);
    agent.prev_x = -4.0f;
    EXPECT_FALSE(check_segment_crosses_moving_box(-2.0f, 5.0f, 2.0f, 5.0f, &agent));

    return 0;
}

static int test_stop_line_red_light_gating(void) {
    Drive env = {0};
    TrafficControlElement tc = {0};
    int red_state[1] = {TRAFFIC_CONTROL_STATE_RED};
    int state[1] = {TRAFFIC_CONTROL_STATE_RED};
    int controlled_lane[1] = {1};
    Agent agent = drive_test_agent(1.0f, 0.0f, 0.0f);

    agent.prev_x = -3.0f;
    tc.type = TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
    tc.state_size = 1;
    tc.states = state;
    tc.stop_line[0] = 0.0f;
    tc.stop_line[1] = -1.0f;
    tc.stop_line[3] = 0.0f;
    tc.stop_line[4] = 1.0f;
    tc.heading = 0.0f;
    tc.num_controlled_lanes = 1;
    tc.controlled_lanes = controlled_lane;
    env.num_traffic_elements = 1;
    env.traffic_elements = &tc;
    env.timestep = 0;

    EXPECT_TRUE(check_agent_on_stop_line(&env, &agent, false));

    state[0] = TRAFFIC_CONTROL_STATE_GREEN;
    EXPECT_FALSE(check_agent_on_stop_line(&env, &agent, false));
    state[0] = TRAFFIC_CONTROL_STATE_YELLOW;
    EXPECT_FALSE(check_agent_on_stop_line(&env, &agent, false));
    EXPECT_TRUE(check_agent_on_stop_line(&env, &agent, true)); // yellow counts when spawn-mode flag is set
    state[0] = TRAFFIC_CONTROL_STATE_OFF;
    EXPECT_FALSE(check_agent_on_stop_line(&env, &agent, false));
    state[0] = TRAFFIC_CONTROL_STATE_RED;

    env.timestep = 1;
    EXPECT_FALSE(check_agent_on_stop_line(&env, &agent, false));
    env.timestep = 0;

    controlled_lane[0] = 2;
    EXPECT_FALSE(check_agent_on_stop_line(&env, &agent, false));
    controlled_lane[0] = 1;

    agent = drive_test_agent(1.0f, 0.0f, (float) M_PI);
    agent.prev_x = -3.0f;
    EXPECT_FALSE(check_agent_on_stop_line(&env, &agent, false));

    agent = drive_test_agent(11.0f, 0.0f, 0.0f);
    agent.prev_x = -1.0f;
    EXPECT_FALSE(check_agent_on_stop_line(&env, &agent, false));

    agent = drive_test_agent(1.0f, 1.4f, 0.0f);
    agent.prev_x = -3.0f;
    agent.prev_y = 1.4f;
    agent.sim_width = 0.4f;
    update_agent_radius(&agent);
    tc.states = red_state;
    EXPECT_TRUE(check_agent_on_stop_line(&env, &agent, false));

    return 0;
}

static int test_stop_line_stationary_on_red(void) {
    Drive env = {0};
    TrafficControlElement tc = {0};
    int state[1] = {TRAFFIC_CONTROL_STATE_RED};
    int controlled_lane[1] = {1};
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);

    tc.type = TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
    tc.state_size = 1;
    tc.states = state;
    tc.stop_line[0] = 0.0f;
    tc.stop_line[1] = -1.0f;
    tc.stop_line[3] = 0.0f;
    tc.stop_line[4] = 1.0f;
    tc.heading = 0.0f;
    tc.num_controlled_lanes = 1;
    tc.controlled_lanes = controlled_lane;
    env.num_traffic_elements = 1;
    env.traffic_elements = &tc;
    env.timestep = 0;

    EXPECT_TRUE(check_agent_on_stop_line(&env, &agent, false));
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_heading_normalization);
    RUN_TEST(test_point_to_segment_distance_cases);
    RUN_TEST(test_obb_collision_cases);
    RUN_TEST(test_obb_collision_dimensions);
    RUN_TEST(test_moving_obb_collision_cases);
    RUN_TEST(test_moving_obb_collision_dimensions);
    RUN_TEST(test_collision_check_filters);
    RUN_TEST(test_segment_aabb_cases);
    RUN_TEST(test_swept_line_box_cases);
    RUN_TEST(test_stop_line_red_light_gating);
    RUN_TEST(test_stop_line_stationary_on_red);
    return test_summary(failures);
}
