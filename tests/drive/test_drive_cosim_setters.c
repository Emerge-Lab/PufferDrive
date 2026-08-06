#include "include/drive_fixture.h"
#include "include/test.h"

// Defensive coverage for the co-sim external-state setters (drive.h "Co-simulation external-state
// setters" section). These are called from Python (pufferlib/ocean/cosim/*) every tick to overwrite
// agent state from CARLA/nuplan

// ---------------------------------------------------------------------------
// c_set_agent_sizes
// ---------------------------------------------------------------------------

static int test_set_agent_sizes_updates_dimensions_radius_and_wheelbase(void) {
    // Overwriting bounding-box size must also refresh the derived radius (collision broad-phase)
    // and wheelbase (steering derivation in c_set_agent_states) -- both are easy to forget since
    // neither is read back in the same function.
    Drive env = {0};
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f); // spawns at length=4, width=2
    env.agents = &agent;
    env.num_total_agents = 1;

    int idx[1] = {0};
    float length[1] = {8.0f};
    float width[1] = {3.0f};
    c_set_agent_sizes(&env, 1, idx, length, width);

    EXPECT_NEAR(agent.sim_length, 8.0f, 1e-6f);
    EXPECT_NEAR(agent.sim_width, 3.0f, 1e-6f);
    EXPECT_NEAR(agent.radius, 0.5f * sqrtf(8.0f * 8.0f + 3.0f * 3.0f), 1e-6f);
    EXPECT_NEAR(agent.wheelbase, 0.6f * 8.0f, 1e-6f);
    return 0;
}

static int test_set_agent_sizes_skips_out_of_range_index(void) {
    // Indices outside [0, num_total_agents) must be skipped, not clamped or crashed on; the valid
    // entries in the same batch must still apply.
    Drive env = {0};
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    env.agents = &agent;
    env.num_total_agents = 1;

    int idx[3] = {-1, 0, 5};
    float length[3] = {1.0f, 8.0f, 1.0f};
    float width[3] = {1.0f, 3.0f, 1.0f};
    c_set_agent_sizes(&env, 3, idx, length, width);

    EXPECT_NEAR(agent.sim_length, 8.0f, 1e-6f);
    EXPECT_NEAR(agent.sim_width, 3.0f, 1e-6f);
    return 0;
}

// ---------------------------------------------------------------------------
// c_set_traffic_light_states
// ---------------------------------------------------------------------------

static TrafficControlElement make_traffic_element(int type, int state_size, int *states_storage) {
    TrafficControlElement element = {0};
    element.type = type;
    element.state_size = state_size;
    element.states = states_storage;
    return element;
}

static int test_set_traffic_light_states_writes_current_timestep_for_lights_only(void) {
    // Only TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT elements are written, only at env->timestep, and every
    // other timestep slot must be left alone.
    int light_states[5] = {99, 99, 99, 99, 99};
    int stop_sign_states[5] = {99, 99, 99, 99, 99};
    TrafficControlElement elements[2];
    elements[0] = make_traffic_element(TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT, 5, light_states);
    elements[1] = make_traffic_element(TRAFFIC_CONTROL_TYPE_STOP_SIGN, 5, stop_sign_states);

    Drive env = {0};
    env.traffic_elements = elements;
    env.num_traffic_elements = 2;
    env.timestep = 2;

    int new_states[2] = {3, 7}; // one entry per traffic element, indexed like env->traffic_elements
    c_set_traffic_light_states(&env, new_states);

    EXPECT_EQ_INT(light_states[2], 3); // written at the current timestep
    EXPECT_EQ_INT(light_states[0], 99);
    EXPECT_EQ_INT(light_states[1], 99);
    EXPECT_EQ_INT(light_states[3], 99);
    EXPECT_EQ_INT(light_states[4], 99);
    for (int t = 0; t < 5; t++) {
        EXPECT_EQ_INT(stop_sign_states[t], 99); // non-light element untouched
    }
    return 0;
}

static int test_set_traffic_light_states_skips_out_of_range_or_missing_states(void) {
    // A timestep outside [0, state_size) or a NULL states buffer (e.g. an element with no logged
    // schedule) must be skipped rather than read/written out of bounds.
    int light_states[3] = {99, 99, 99};
    TrafficControlElement elements[2];
    elements[0] = make_traffic_element(TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT, 3, light_states);
    elements[1] = make_traffic_element(TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT, 3, NULL);

    Drive env = {0};
    env.traffic_elements = elements;
    env.num_traffic_elements = 2;
    env.timestep = 10; // out of range for state_size=3

    int new_states[2] = {1, 1};
    c_set_traffic_light_states(&env, new_states); // must not crash on the NULL-states element either

    EXPECT_EQ_INT(light_states[0], 99);
    EXPECT_EQ_INT(light_states[1], 99);
    EXPECT_EQ_INT(light_states[2], 99);
    return 0;
}


// ---------------------------------------------------------------------------
// c_set_agent_states
// ---------------------------------------------------------------------------

static int test_set_agent_states_teleport_resets_prev_pose(void) {
    // An externally-set pose is a teleport: prev_* must follow it. If prev stays at
    // the old pose, every prev-based swept check (find_lane_and_offroad's road-edge
    // crossing, compute_metrics' goal-reach segment) sweeps the whole gap and fires
    // spuriously -- observed as an on-lane teleported ego always classified offroad
    // with dead lane observations.
    Drive env = {0};
    Agent agent = drive_test_agent(500.0f, 500.0f, 1.0f);
    agent.prev_x = 500.0f;
    agent.prev_y = 500.0f;
    env.agents = &agent;
    env.num_total_agents = 1;

    int idx[1] = {0};
    float x[1] = {10.0f}, y[1] = {20.0f}, z[1] = {0.0f}, h[1] = {0.5f};
    float vx[1] = {1.0f}, vy[1] = {0.0f}, yr[1] = {0.0f}, al[1] = {0.0f};
    c_set_agent_states(&env, 1, idx, x, y, z, h, vx, vy, yr, al);

    EXPECT_NEAR(agent.prev_x, agent.sim_x, 1e-6f);
    EXPECT_NEAR(agent.prev_y, agent.sim_y, 1e-6f);
    EXPECT_NEAR(agent.prev_cos_heading, agent.cos_heading, 1e-6f);
    EXPECT_NEAR(agent.prev_sin_heading, agent.sin_heading, 1e-6f);
    return 0;
}

// ---------------------------------------------------------------------------
// c_set_agent_goals
// ---------------------------------------------------------------------------

static int test_set_agent_goals_sets_positions_lane_and_count(void) {
    // Externally-set goals must land in the bin frame (world_mean subtracted), report an accurate
    // goal_count, and reset the current_goal_* window to slot 0 -- all fields that used to be left
    // stale from whatever a previous internal goal generation wrote. With no grid map (as here),
    // goal->lane snapping (find_goal_lane) must degrade to list_goal_lane = -1, the "no GPS
    // lane-distance" convention shared with GOAL_SOURCE_GT -- never read a NULL grid.
    Drive env = {0};
    env.world_mean_x = 100.0f;
    env.world_mean_y = 200.0f;
    Agent agent = {0};
    agent.list_goal_lane[0] = 42; // stale value from a prior internal goal generation
    agent.goal_count = 99;        // stale value: must be overwritten, not left stale
    env.agents = &agent;
    env.num_total_agents = 1;

    float gx[3] = {110.0f, 130.0f, 150.0f};
    float gy[3] = {205.0f, 215.0f, 225.0f};
    float gz[3] = {1.0f, 2.0f, 3.0f};
    float gdx[3] = {1.0f, 1.0f, 1.0f};
    float gdy[3] = {0.0f, 0.0f, 0.0f};
    c_set_agent_goals(&env, 0, 3, gx, gy, gz, gdx, gdy);

    EXPECT_EQ_INT(agent.goal_count, 3);
    EXPECT_EQ_INT(agent.current_goal_idx, 0);
    for (int w = 0; w < 3; w++) {
        EXPECT_NEAR(agent.list_goal_x[w], gx[w] - env.world_mean_x, 1e-6f);
        EXPECT_NEAR(agent.list_goal_y[w], gy[w] - env.world_mean_y, 1e-6f);
        EXPECT_NEAR(agent.list_goal_z[w], gz[w], 1e-6f);
        EXPECT_EQ_INT(agent.list_goal_lane[w], -1);
    }
    EXPECT_NEAR(agent.current_goal_x, agent.list_goal_x[0], 1e-6f);
    EXPECT_NEAR(agent.current_goal_y, agent.list_goal_y[0], 1e-6f);
    EXPECT_NEAR(agent.current_goal_z, agent.list_goal_z[0], 1e-6f);
    return 0;
}

static int test_set_agent_goals_caps_at_max_goals(void) {
    // A caller passing more waypoints than MAX_GOALS must be clamped, and goal_count must reflect the
    // clamped count (not the raw waypoint count), or the obs window in write_reward_target_obs reads
    // past what was actually written.
    Drive env = {0};
    Agent agent = {0};
    env.agents = &agent;
    env.num_total_agents = 1;

    int requested = MAX_GOALS + 5;
    float gx[MAX_GOALS + 5], gy[MAX_GOALS + 5], gz[MAX_GOALS + 5];
    float gdir[MAX_GOALS + 5];
    for (int w = 0; w < requested; w++) {
        gx[w] = (float) w;
        gy[w] = (float) w;
        gz[w] = 0.0f;
        gdir[w] = 0.0f;
    }
    c_set_agent_goals(&env, 0, requested, gx, gy, gz, gdir, gdir);

    EXPECT_EQ_INT(agent.goal_count, MAX_GOALS);
    EXPECT_EQ_INT(agent.list_goal_lane[MAX_GOALS - 1], -1);
    EXPECT_NEAR(agent.list_goal_x[MAX_GOALS - 1], (float) (MAX_GOALS - 1), 1e-6f);
    return 0;
}

static int test_set_agent_goals_out_of_range_agent_idx_is_noop(void) {
    Drive env = {0};
    Agent agent = {0};
    agent.goal_count = 5; // must survive untouched
    env.agents = &agent;
    env.num_total_agents = 1;

    float gx[1] = {1.0f}, gy[1] = {1.0f}, gz[1] = {1.0f}, gdir[1] = {0.0f};
    c_set_agent_goals(&env, 5, 1, gx, gy, gz, gdir, gdir); // agent_idx out of range

    EXPECT_EQ_INT(agent.goal_count, 5);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_set_agent_sizes_updates_dimensions_radius_and_wheelbase);
    RUN_TEST(test_set_agent_sizes_skips_out_of_range_index);
    RUN_TEST(test_set_traffic_light_states_writes_current_timestep_for_lights_only);
    RUN_TEST(test_set_traffic_light_states_skips_out_of_range_or_missing_states);
    RUN_TEST(test_set_agent_states_teleport_resets_prev_pose);
    RUN_TEST(test_set_agent_goals_sets_positions_lane_and_count);
    RUN_TEST(test_set_agent_goals_caps_at_max_goals);
    RUN_TEST(test_set_agent_goals_out_of_range_agent_idx_is_noop);
    return test_summary(failures);
}
