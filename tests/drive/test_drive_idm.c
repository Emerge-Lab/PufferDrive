#include "include/drive_fixture.h"
#include "include/test.h"

static void make_lane(RoadMapElement *lane, float *x, float *y, float *z, float *headings, float speed_limit) {
    lane->type = LANE_SURFACE_STREET;
    lane->segment_size = 2;
    lane->x = x;
    lane->y = y;
    lane->z = z;
    lane->headings = headings;
    lane->speed_limit = speed_limit;
}

static int test_leader_defaults_and_update(void) {
    IDMLeader leader = idm_no_leader();
    EXPECT_EQ_INT(leader.has_leader, 0);
    EXPECT_EQ_INT(leader.leader_agent_idx, -1);
    EXPECT_TRUE(isinf(leader.gap_meters));

    idm_update_best_leader(&leader, 3, 0, -1.0f, -5.0f);
    EXPECT_EQ_INT(leader.has_leader, 1);
    EXPECT_EQ_INT(leader.leader_agent_idx, 3);
    EXPECT_NEAR(leader.gap_meters, IDM_MIN_SPACING_METERS, 1e-5f);
    EXPECT_NEAR(leader.leader_speed_mps, 0.0f, 1e-5f);

    idm_update_best_leader(&leader, 4, 0, 10.0f, 3.0f);
    EXPECT_EQ_INT(leader.leader_agent_idx, 3);
    idm_update_best_leader(&leader, 5, 1, 0.05f, 7.0f);
    EXPECT_EQ_INT(leader.leader_agent_idx, 5);
    EXPECT_EQ_INT(leader.is_traffic_light, 1);
    EXPECT_NEAR(leader.gap_meters, IDM_MIN_SPACING_METERS, 1e-5f);
    EXPECT_NEAR(leader.leader_speed_mps, 7.0f, 1e-5f);
    return 0;
}

static int test_z_overlap(void) {
    Agent a = drive_test_agent(0.0f, 0.0f, 0.0f);
    Agent b = drive_test_agent(0.0f, 0.0f, 0.0f);
    b.sim_z = 1.4f;
    EXPECT_TRUE(idm_check_z_overlap(&a, &b));
    b.sim_z = 2.0f;
    EXPECT_FALSE(idm_check_z_overlap(&a, &b));
    return 0;
}

static int test_desired_speed_fallback_and_lane_limit(void) {
    Drive env = {0};
    RoadMapElement lanes[2] = {0};
    float x0[2] = {0.0f, 10.0f};
    float y0[2] = {0.0f, 0.0f};
    float z0[2] = {0.0f, 0.0f};
    float h0[1] = {0.0f};
    float x1[2] = {10.0f, 20.0f};
    float y1[2] = {0.0f, 0.0f};
    float z1[2] = {0.0f, 0.0f};
    float h1[1] = {0.0f};
    int route[1] = {1};
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);

    make_lane(&lanes[0], x0, y0, z0, h0, 0.0f);
    make_lane(&lanes[1], x1, y1, z1, h1, 12.0f);
    env.road_elements = lanes;
    env.num_road_elements = 2;
    env.base_max_speed_mps = 20.0f;
    agent.current_lane_idx = 0;
    agent.route = route;
    agent.route_length = 1;
    agent.current_route_idx = 0;
    EXPECT_NEAR(idm_desired_speed(&env, &agent), 12.0f, 1e-5f);

    lanes[1].speed_limit = 0.0f;
    EXPECT_NEAR(idm_desired_speed(&env, &agent), IDM_DEFAULT_DESIRED_SPEED_MPS, 1e-5f);
    return 0;
}

static int test_route_projection_and_advance(void) {
    Drive env = {0};
    RoadMapElement lanes[2] = {0};
    float x0[2] = {0.0f, 10.0f};
    float y0[2] = {0.0f, 0.0f};
    float z0[2] = {0.0f, 0.0f};
    float h0[1] = {0.0f};
    float x1[2] = {10.0f, 20.0f};
    float y1[2] = {0.0f, 0.0f};
    float z1[2] = {0.0f, 0.0f};
    float h1[1] = {0.0f};
    int route[2] = {0, 1};
    Agent agent = drive_test_agent(5.0f, 0.0f, 0.0f);

    make_lane(&lanes[0], x0, y0, z0, h0, 10.0f);
    make_lane(&lanes[1], x1, y1, z1, h1, 10.0f);
    env.road_elements = lanes;
    env.num_road_elements = 2;
    agent.route = route;
    agent.route_length = 2;
    agent.current_route_idx = 0;

    IDMLaneProjection projection = idm_project_to_route_lanes(&env, &agent);
    EXPECT_EQ_INT(projection.valid, 1);
    EXPECT_EQ_INT(projection.lane_idx, 0);
    EXPECT_EQ_INT(projection.segment_idx, 0);
    EXPECT_NEAR(projection.t, 0.5f, 1e-5f);

    EXPECT_EQ_INT(idm_set_projected_agent_pose(&env, &agent, projection, 7.0f), 1);
    EXPECT_NEAR(agent.sim_x, 12.0f, 1e-5f);
    EXPECT_NEAR(agent.sim_y, 0.0f, 1e-5f);

    agent = drive_test_agent(5.0f, 0.0f, 0.0f);
    agent.route = route;
    agent.route_length = 2;
    agent.current_route_idx = 0;
    env.agents = &agent;
    float old_heading = 0.0f;
    EXPECT_EQ_INT(idm_advance_along_route_lanes(&env, 0, 7.0f, &old_heading), 1);
    EXPECT_EQ_INT(agent.current_route_idx, 1);
    EXPECT_NEAR(agent.sim_x, 12.0f, 1e-5f);
    return 0;
}

static int test_red_yellow_traffic_light_obstacle(void) {
    Drive env = {0};
    TrafficControlElement tc = {0};
    int state[1] = {TRAFFIC_CONTROL_STATE_RED};
    int controlled[1] = {2};
    Agent sample = drive_test_agent(5.0f, 0.0f, 0.0f);
    sample.prev_x = 4.0f;

    tc.type = TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
    tc.state_size = 1;
    tc.states = state;
    tc.stop_line[0] = 5.0f;
    tc.stop_line[1] = -2.0f;
    tc.stop_line[3] = 5.0f;
    tc.stop_line[4] = 2.0f;
    tc.heading = 0.0f;
    tc.num_controlled_lanes = 1;
    tc.controlled_lanes = controlled;
    env.traffic_elements = &tc;
    env.num_traffic_elements = 1;
    env.timestep = 0;

    EXPECT_TRUE(idm_sample_hits_red_light(&env, &sample, 2));
    state[0] = TRAFFIC_CONTROL_STATE_YELLOW;
    EXPECT_TRUE(idm_sample_hits_red_light(&env, &sample, 2));
    state[0] = TRAFFIC_CONTROL_STATE_GREEN;
    EXPECT_FALSE(idm_sample_hits_red_light(&env, &sample, 2));
    EXPECT_FALSE(idm_sample_hits_red_light(&env, &sample, 3));
    return 0;
}

static int test_leader_selection_and_move_idm(void) {
    Drive env = {0};
    RoadMapElement lane = {0};
    float x[2] = {0.0f, 50.0f};
    float y[2] = {0.0f, 0.0f};
    float z[2] = {0.0f, 0.0f};
    float h[1] = {0.0f};
    Agent agents[2] = {0};
    int ego_route[1] = {0};
    int other_route[1] = {0};

    make_lane(&lane, x, y, z, h, 10.0f);
    agents[0] = drive_test_agent(0.0f, 0.0f, 0.0f);
    agents[1] = drive_test_agent(10.0f, 0.0f, 0.0f);
    agents[0].route = ego_route;
    agents[0].route_length = 1;
    agents[0].current_route_idx = 0;
    agents[0].current_lane_idx = 0;
    agents[1].route = other_route;
    agents[1].route_length = 1;
    agents[1].current_route_idx = 0;
    agents[1].current_lane_idx = 0;
    agents[1].sim_vx = 2.0f;
    update_agent_speed(&agents[1]);

    env.road_elements = &lane;
    env.num_road_elements = 1;
    env.agents = agents;
    env.num_agents = 1;
    env.num_total_agents = 2;
    env.dt = 0.1f;
    env.base_max_speed_mps = 20.0f;

    IDMLeader leader = idm_find_leader_by_route_boxes(&env, 0);
    EXPECT_EQ_INT(leader.has_leader, 1);
    EXPECT_EQ_INT(leader.leader_agent_idx, 1);
    EXPECT_FALSE(leader.is_traffic_light);

    float before = agents[0].sim_x;
    move_idm(&env, 0);
    EXPECT_TRUE(agents[0].sim_x >= before);
    EXPECT_FINITE(agents[0].sim_speed);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_leader_defaults_and_update);
    RUN_TEST(test_z_overlap);
    RUN_TEST(test_desired_speed_fallback_and_lane_limit);
    RUN_TEST(test_route_projection_and_advance);
    RUN_TEST(test_red_yellow_traffic_light_obstacle);
    RUN_TEST(test_leader_selection_and_move_idm);
    return test_summary(failures);
}
