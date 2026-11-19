#include "include/drive_fixture.h"
#include "include/test.h"

static int test_set_agent_sizes_updates_dimensions_radius_and_wheelbase(void) {
    // a new size also refreshes the derived radius and wheelbase
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

static int test_set_agent_sizes_rejects_out_of_range_index_and_bad_size(void) {
    // a bad index or non-positive size fails the whole call and leaves the agent untouched
    Drive env = {0};
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    env.agents = &agent;
    env.num_total_agents = 1;

    int bad_idx[1] = {5};
    float length[1] = {8.0f};
    float width[1] = {3.0f};
    EXPECT_EQ_INT(c_set_agent_sizes(&env, 1, bad_idx, length, width), -1);
    EXPECT_NEAR(agent.sim_length, 4.0f, 1e-6f);

    int idx[1] = {0};
    float zero_width[1] = {0.0f};
    EXPECT_EQ_INT(c_set_agent_sizes(&env, 1, idx, length, zero_width), -1);
    EXPECT_NEAR(agent.sim_length, 4.0f, 1e-6f);
    EXPECT_EQ_INT(c_set_agent_sizes(&env, 1, idx, length, width), 0);
    EXPECT_NEAR(agent.sim_length, 8.0f, 1e-6f);
    return 0;
}

static TrafficControlElement make_traffic_element(int type, int state_size, int *states_storage) {
    TrafficControlElement element = {0};
    element.type = type;
    element.state_size = state_size;
    element.states = states_storage;
    return element;
}

static int test_set_traffic_light_states_writes_current_and_next_timestep_for_lights_only(void) {
    // only lights are written, at the timestep and the next one c_step reads before the next sync
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
    EXPECT_EQ_INT(c_set_traffic_light_states(&env, new_states, NULL), 0);

    EXPECT_EQ_INT(light_states[2], 3); // written at the current timestep
    EXPECT_EQ_INT(light_states[3], 3); // and at the one c_step reads next
    EXPECT_EQ_INT(light_states[0], 99);
    EXPECT_EQ_INT(light_states[1], 99);
    EXPECT_EQ_INT(light_states[4], 99);
    for (int t = 0; t < 5; t++) {
        EXPECT_EQ_INT(stop_sign_states[t], 99); // non-light element untouched
    }
    return 0;
}

static int test_set_traffic_light_states_rejects_out_of_range_timestep_and_state(void) {
    // a bad timestep or state fails the call; a light without a schedule (NULL states) is skipped
    int light_states[3] = {99, 99, 99};
    TrafficControlElement elements[2];
    elements[0] = make_traffic_element(TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT, 3, NULL);
    elements[1] = make_traffic_element(TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT, 3, light_states);

    Drive env = {0};
    env.traffic_elements = elements;
    env.num_traffic_elements = 2;

    int new_states[2] = {1, 1};
    env.timestep = 10; // out of range for state_size=3
    EXPECT_EQ_INT(c_set_traffic_light_states(&env, new_states, NULL), -1);
    EXPECT_EQ_INT(light_states[0], 99);
    EXPECT_EQ_INT(light_states[1], 99);
    EXPECT_EQ_INT(light_states[2], 99);

    env.timestep = 1;
    int bad_states[2] = {1, TRAFFIC_CONTROL_STATE_OFF + 1};
    EXPECT_EQ_INT(c_set_traffic_light_states(&env, bad_states, NULL), -1);
    EXPECT_EQ_INT(light_states[1], 99);
    EXPECT_EQ_INT(c_set_traffic_light_states(&env, new_states, NULL), 0);
    EXPECT_EQ_INT(light_states[1], 1);
    EXPECT_EQ_INT(light_states[2], 1);

    env.timestep = 2; // last slot of the schedule: no next timestep to fill, still a success
    int last_states[2] = {3, 3};
    EXPECT_EQ_INT(c_set_traffic_light_states(&env, last_states, NULL), 0);
    EXPECT_EQ_INT(light_states[2], 3);
    return 0;
}

#define TIMING_STATE_COUNT 12

static Drive make_timing_env(TrafficControlElement *elements, int element_count, int (*storage)[TIMING_STATE_COUNT]) {
    for (int i = 0; i < element_count; i++) {
        for (int t = 0; t < TIMING_STATE_COUNT; t++) {
            storage[i][t] = 99;
        }
        int type = i == element_count - 1 ? TRAFFIC_CONTROL_TYPE_STOP_SIGN : TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
        elements[i] = make_traffic_element(type, TIMING_STATE_COUNT, storage[i]);
    }
    Drive env = {0};
    env.traffic_elements = elements;
    env.num_traffic_elements = element_count;
    env.dt = 0.1f;
    env.traffic_light_yellow_max_s = 3.0f;
    return env;
}

static int test_set_traffic_light_states_yellow_timing_patches_schedule(void) {
    // yellow with r s left stays yellow through ts + ceil(r / dt) - 1, then red; other states get no patch
    int storage[3][TIMING_STATE_COUNT];
    TrafficControlElement elements[3];
    Drive env = make_timing_env(elements, 3, storage);
    int *yellow_light = storage[0], *green_light = storage[1], *stop_sign = storage[2];

    env.timestep = 2;
    int new_states[3] = {TRAFFIC_CONTROL_STATE_YELLOW, TRAFFIC_CONTROL_STATE_GREEN, 7};
    float remaining_s[3] = {0.5f, 0.0f, 0.0f}; // 5 steps: yellow at 2..6, red at 7
    EXPECT_EQ_INT(c_set_traffic_light_states(&env, new_states, remaining_s), 0);
    EXPECT_EQ_INT(yellow_light[1], 99);
    for (int t = 2; t <= 6; t++) {
        EXPECT_EQ_INT(yellow_light[t], TRAFFIC_CONTROL_STATE_YELLOW);
    }
    EXPECT_EQ_INT(yellow_light[7], TRAFFIC_CONTROL_STATE_RED);
    EXPECT_EQ_INT(yellow_light[8], 99);
    EXPECT_EQ_INT(green_light[2], TRAFFIC_CONTROL_STATE_GREEN);
    EXPECT_EQ_INT(green_light[3], TRAFFIC_CONTROL_STATE_GREEN);
    EXPECT_EQ_INT(green_light[4], 99);
    for (int t = 0; t < TIMING_STATE_COUNT; t++) {
        EXPECT_EQ_INT(stop_sign[t], 99);
    }

    env.timestep = 0; // under one step left: the held step still counts, red right after it
    float short_remaining_s[3] = {0.05f, 0.0f, 0.0f};
    EXPECT_EQ_INT(c_set_traffic_light_states(&env, new_states, short_remaining_s), 0);
    EXPECT_EQ_INT(yellow_light[0], TRAFFIC_CONTROL_STATE_YELLOW);
    EXPECT_EQ_INT(yellow_light[1], TRAFFIC_CONTROL_STATE_YELLOW);
    EXPECT_EQ_INT(yellow_light[2], TRAFFIC_CONTROL_STATE_RED);

    env.timestep = 9; // the schedule end truncates the patch
    float long_remaining_s[3] = {3.0f, 0.0f, 0.0f};
    EXPECT_EQ_INT(c_set_traffic_light_states(&env, new_states, long_remaining_s), 0);
    for (int t = 9; t < TIMING_STATE_COUNT; t++) {
        EXPECT_EQ_INT(yellow_light[t], TRAFFIC_CONTROL_STATE_YELLOW);
    }
    return 0;
}

static int test_set_traffic_light_states_yellow_timing_validation(void) {
    // The obs flag makes the timing mandatory; NaN, negative or above-the-training-max seconds fail the call.
    int storage[2][TIMING_STATE_COUNT];
    TrafficControlElement elements[2];
    Drive env = make_timing_env(elements, 2, storage);
    env.timestep = 1;
    int new_states[2] = {TRAFFIC_CONTROL_STATE_YELLOW, 7};

    env.obs_traffic_light_yellow_remaining = 1;
    EXPECT_EQ_INT(c_set_traffic_light_states(&env, new_states, NULL), -1);
    EXPECT_EQ_INT(storage[0][1], 99);
    float at_max_s[2] = {3.0f, 0.0f};
    EXPECT_EQ_INT(c_set_traffic_light_states(&env, new_states, at_max_s), 0);
    EXPECT_EQ_INT(storage[0][1], TRAFFIC_CONTROL_STATE_YELLOW);

    float nan_s[2] = {NAN, 0.0f};
    float negative_s[2] = {-0.1f, 0.0f};
    float above_max_s[2] = {3.5f, 0.0f};
    EXPECT_EQ_INT(c_set_traffic_light_states(&env, new_states, nan_s), -1);
    EXPECT_EQ_INT(c_set_traffic_light_states(&env, new_states, negative_s), -1);
    EXPECT_EQ_INT(c_set_traffic_light_states(&env, new_states, above_max_s), -1);
    return 0;
}

static int test_set_agent_states_teleport_resets_prev_pose(void) {
    // a set pose is a teleport: prev follows, or the swept offroad and goal checks would sweep the jump
    Drive env = {0};
    Agent agent = drive_test_agent(500.0f, 500.0f, 1.0f);
    agent.prev_x = 500.0f;
    agent.prev_y = 500.0f;
    env.agents = &agent;
    env.num_total_agents = 1;

    int idx[1] = {0};
    float x[1] = {10.0f}, y[1] = {20.0f}, z[1] = {0.0f}, h[1] = {0.5f};
    float vx[1] = {1.0f}, vy[1] = {0.0f}, yr[1] = {0.0f}, al[1] = {0.0f};
    EXPECT_EQ_INT(c_set_agent_states(&env, 1, idx, x, y, z, h, vx, vy, yr, al, NULL, NULL), 0);

    EXPECT_NEAR(agent.prev_x, agent.sim_x, 1e-6f);
    EXPECT_NEAR(agent.prev_y, agent.sim_y, 1e-6f);
    EXPECT_NEAR(agent.prev_cos_heading, agent.cos_heading, 1e-6f);
    EXPECT_NEAR(agent.prev_sin_heading, agent.sin_heading, 1e-6f);
    return 0;
}

static int test_set_agent_states_seconds_stopped_injects_or_preserves(void) {
    // NULL seconds_stopped keeps c_step's own accumulation, an array injects it
    Drive env = {0};
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    agent.seconds_stopped = 12.5f;
    env.agents = &agent;
    env.num_total_agents = 1;

    int idx[1] = {0};
    float x[1] = {1.0f}, y[1] = {2.0f}, z[1] = {0.0f}, h[1] = {0.0f};
    float vx[1] = {0.0f}, vy[1] = {0.0f}, yr[1] = {0.0f}, al[1] = {0.0f};

    EXPECT_EQ_INT(c_set_agent_states(&env, 1, idx, x, y, z, h, vx, vy, yr, al, NULL, NULL), 0);
    EXPECT_NEAR(agent.seconds_stopped, 12.5f, 1e-6f);

    float seconds_stopped[1] = {3.0f};
    EXPECT_EQ_INT(c_set_agent_states(&env, 1, idx, x, y, z, h, vx, vy, yr, al, seconds_stopped, NULL), 0);
    EXPECT_NEAR(agent.seconds_stopped, 3.0f, 1e-6f);

    float negative_seconds_stopped[1] = {-1.0f};
    EXPECT_EQ_INT(c_set_agent_states(&env, 1, idx, x, y, z, h, vx, vy, yr, al, negative_seconds_stopped, NULL), -1);
    EXPECT_NEAR(agent.seconds_stopped, 3.0f, 1e-6f);
    return 0;
}

static int test_set_agent_states_steering_is_state_below_1mps(void) {
    // handed-over steering is kept even at standstill, where the yaw-rate derivation would read 0
    Drive env = {0};
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    env.agents = &agent;
    env.num_total_agents = 1;

    int idx[1] = {0};
    float x[1] = {1.0f}, y[1] = {2.0f}, z[1] = {0.0f}, h[1] = {0.0f};
    float vy[1] = {0.0f}, yr[1] = {0.0f}, al[1] = {0.0f};

    float stopped[1] = {0.0f}, steering[1] = {0.14f};
    EXPECT_EQ_INT(c_set_agent_states(&env, 1, idx, x, y, z, h, stopped, vy, yr, al, NULL, steering), 0);
    EXPECT_NEAR(agent.steering_angle, 0.14f, 1e-6f);
    EXPECT_NEAR(agent.accel_lat, 0.0f, 1e-6f);

    float crawling[1] = {0.5f};
    EXPECT_EQ_INT(c_set_agent_states(&env, 1, idx, x, y, z, h, crawling, vy, yr, al, NULL, steering), 0);
    EXPECT_NEAR(agent.steering_angle, 0.14f, 1e-6f);
    EXPECT_NEAR(agent.accel_lat, 0.25f * tanf(0.14f) / agent.wheelbase, 1e-6f);

    EXPECT_EQ_INT(c_set_agent_states(&env, 1, idx, x, y, z, h, stopped, vy, yr, al, NULL, NULL), 0);
    EXPECT_NEAR(agent.steering_angle, 0.0f, 1e-6f);

    float not_finite[1] = {NAN};
    EXPECT_EQ_INT(c_set_agent_states(&env, 1, idx, x, y, z, h, stopped, vy, yr, al, NULL, not_finite), -1);
    return 0;
}

static int test_set_agent_goals_sets_positions_lane_and_count(void) {
    // goals in the bin frame, goal_count set, window reset to slot 0; without a grid map no lane (-1)
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
    EXPECT_EQ_INT(c_set_agent_goals(&env, 0, 3, gx, gy, gz, gdx, gdy), 0);

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

static int test_set_agent_goals_rejects_more_than_max_goals(void) {
    // zero or more than MAX_GOALS waypoints fail the call and leave the goal window untouched
    Drive env = {0};
    Agent agent = {0};
    agent.goal_count = 2;
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
    EXPECT_EQ_INT(c_set_agent_goals(&env, 0, requested, gx, gy, gz, gdir, gdir), -1);
    EXPECT_EQ_INT(agent.goal_count, 2);
    EXPECT_EQ_INT(c_set_agent_goals(&env, 0, 0, gx, gy, gz, gdir, gdir), -1);
    EXPECT_EQ_INT(agent.goal_count, 2);
    EXPECT_EQ_INT(c_set_agent_goals(&env, 0, MAX_GOALS, gx, gy, gz, gdir, gdir), 0);
    EXPECT_EQ_INT(agent.goal_count, MAX_GOALS);
    return 0;
}

static int test_set_agent_goals_rejects_out_of_range_agent_idx(void) {
    Drive env = {0};
    Agent agent = {0};
    agent.goal_count = 5; // must survive untouched
    env.agents = &agent;
    env.num_total_agents = 1;

    float gx[1] = {1.0f}, gy[1] = {1.0f}, gz[1] = {1.0f}, gdir[1] = {0.0f};
    EXPECT_EQ_INT(c_set_agent_goals(&env, 5, 1, gx, gy, gz, gdir, gdir), -1); // agent_idx out of range

    EXPECT_EQ_INT(agent.goal_count, 5);
    return 0;
}

static int test_set_stop_signs_retires_map_signs_and_appends_external_ones(void) {
    // map stop signs retired in place, external ones appended in the bin frame, every stop-sign latch reset
    int light_states[3] = {1, 1, 1};
    TrafficControlElement *elements = (TrafficControlElement *) calloc(2, sizeof(TrafficControlElement));
    elements[0] = make_traffic_element(TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT, 3, light_states);
    elements[1] = make_traffic_element(TRAFFIC_CONTROL_TYPE_STOP_SIGN, 0, NULL);
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    agent.stop_sign_target_idx = 1;
    agent.stop_sign_stop_completed = 1;
    agent.stop_sign_last_failed_idx = 1;
    Drive env = {0};
    env.traffic_elements = elements;
    env.num_traffic_elements = 2;
    env.agents = &agent;
    env.num_total_agents = 1;
    env.world_mean_x = 100.0f;
    env.world_mean_y = -50.0f;

    float lines[12] = {110.0f, -52.0f, 0.5f, 110.0f, -48.0f, 0.5f, 130.0f, -60.0f, 0.0f, 134.0f, -60.0f, 0.0f};
    float headings[2] = {0.0f, 1.5f};
    EXPECT_EQ_INT(c_set_stop_signs(&env, 2, lines, headings), 0);
    EXPECT_EQ_INT(env.num_traffic_elements, 4);
    EXPECT_EQ_INT(env.traffic_elements[0].type, TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT);
    EXPECT_EQ_INT(env.traffic_elements[0].states[1], 1);
    EXPECT_EQ_INT(env.traffic_elements[1].type, TRAFFIC_CONTROL_TYPE_NONE);
    EXPECT_EQ_INT(env.traffic_elements[2].type, TRAFFIC_CONTROL_TYPE_STOP_SIGN);
    EXPECT_NEAR(env.traffic_elements[2].stop_line[0], 10.0f, 1e-6f);
    EXPECT_NEAR(env.traffic_elements[2].stop_line[2], 0.5f, 1e-6f);
    EXPECT_NEAR(env.traffic_elements[2].stop_line[4], 2.0f, 1e-6f);
    EXPECT_NEAR(env.traffic_elements[3].heading, 1.5f, 1e-6f);
    EXPECT_EQ_INT(env.traffic_elements[3].state_size, 0);
    EXPECT_EQ_INT(env.traffic_elements[3].num_controlled_lanes, 0);
    EXPECT_EQ_INT(env.traffic_elements[3].junction_id, -1);
    EXPECT_EQ_INT(agent.stop_sign_target_idx, -1);
    EXPECT_EQ_INT(agent.stop_sign_stop_completed, 0);
    EXPECT_EQ_INT(agent.stop_sign_last_failed_idx, -1);
    free(env.traffic_elements);
    return 0;
}

static int test_set_stop_signs_rejects_bad_input_untouched(void) {
    // bad input fails the whole call and leaves the elements; zero external signs still retire the map's
    TrafficControlElement *elements = (TrafficControlElement *) calloc(1, sizeof(TrafficControlElement));
    elements[0] = make_traffic_element(TRAFFIC_CONTROL_TYPE_STOP_SIGN, 0, NULL);
    Drive env = {0};
    env.traffic_elements = elements;
    env.num_traffic_elements = 1;

    float degenerate[6] = {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f};
    float nan_line[6] = {0.0f, NAN, 0.0f, 0.0f, 4.0f, 0.0f};
    float heading[1] = {0.0f};
    EXPECT_EQ_INT(c_set_stop_signs(&env, -1, degenerate, heading), -1);
    EXPECT_EQ_INT(c_set_stop_signs(&env, 1, degenerate, heading), -1);
    EXPECT_EQ_INT(c_set_stop_signs(&env, 1, nan_line, heading), -1);
    EXPECT_EQ_INT(env.num_traffic_elements, 1);
    EXPECT_EQ_INT(env.traffic_elements[0].type, TRAFFIC_CONTROL_TYPE_STOP_SIGN);
    EXPECT_EQ_INT(c_set_stop_signs(&env, 0, NULL, NULL), 0);
    EXPECT_EQ_INT(env.num_traffic_elements, 1);
    EXPECT_EQ_INT(env.traffic_elements[0].type, TRAFFIC_CONTROL_TYPE_NONE);
    free(env.traffic_elements);
    return 0;
}

static int test_set_traffic_light_lines_moves_lights_only_and_rejects_bad_input(void) {
    // only the addressed lights move (bin frame), schedules stay; any bad input fails the whole call
    int light_states[2] = {1, 1};
    TrafficControlElement elements[2];
    elements[0] = make_traffic_element(TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT, 2, light_states);
    elements[1] = make_traffic_element(TRAFFIC_CONTROL_TYPE_STOP_SIGN, 0, NULL);
    Drive env = {0};
    env.traffic_elements = elements;
    env.num_traffic_elements = 2;
    env.world_mean_x = 100.0f;
    env.world_mean_y = -50.0f;

    int idx[1] = {0};
    float line[6] = {110.0f, -52.0f, 0.5f, 110.0f, -48.0f, 0.5f};
    EXPECT_EQ_INT(c_set_traffic_light_lines(&env, 1, idx, line), 0);
    EXPECT_NEAR(elements[0].stop_line[0], 10.0f, 1e-6f);
    EXPECT_NEAR(elements[0].stop_line[1], -2.0f, 1e-6f);
    EXPECT_NEAR(elements[0].stop_line[2], 0.5f, 1e-6f);
    EXPECT_NEAR(elements[0].stop_line[3], 10.0f, 1e-6f);
    EXPECT_NEAR(elements[0].stop_line[4], 2.0f, 1e-6f);
    EXPECT_EQ_INT(elements[0].states[0], 1);

    int stop_sign_idx[1] = {1};
    EXPECT_EQ_INT(c_set_traffic_light_lines(&env, 1, stop_sign_idx, line), -1);
    EXPECT_NEAR(elements[1].stop_line[0], 0.0f, 1e-6f);
    int bad_idx[1] = {7};
    EXPECT_EQ_INT(c_set_traffic_light_lines(&env, 1, bad_idx, line), -1);
    float degenerate[6] = {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f};
    EXPECT_EQ_INT(c_set_traffic_light_lines(&env, 1, idx, degenerate), -1);
    float nan_line[6] = {0.0f, NAN, 0.0f, 0.0f, 4.0f, 0.0f};
    EXPECT_EQ_INT(c_set_traffic_light_lines(&env, 1, idx, nan_line), -1);
    EXPECT_NEAR(elements[0].stop_line[0], 10.0f, 1e-6f);
    EXPECT_EQ_INT(c_set_traffic_light_lines(&env, 0, NULL, NULL), 0);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_set_agent_sizes_updates_dimensions_radius_and_wheelbase);
    RUN_TEST(test_set_agent_sizes_rejects_out_of_range_index_and_bad_size);
    RUN_TEST(test_set_traffic_light_states_writes_current_and_next_timestep_for_lights_only);
    RUN_TEST(test_set_traffic_light_states_rejects_out_of_range_timestep_and_state);
    RUN_TEST(test_set_traffic_light_states_yellow_timing_patches_schedule);
    RUN_TEST(test_set_traffic_light_states_yellow_timing_validation);
    RUN_TEST(test_set_stop_signs_retires_map_signs_and_appends_external_ones);
    RUN_TEST(test_set_stop_signs_rejects_bad_input_untouched);
    RUN_TEST(test_set_traffic_light_lines_moves_lights_only_and_rejects_bad_input);
    RUN_TEST(test_set_agent_states_teleport_resets_prev_pose);
    RUN_TEST(test_set_agent_states_seconds_stopped_injects_or_preserves);
    RUN_TEST(test_set_agent_states_steering_is_state_below_1mps);
    RUN_TEST(test_set_agent_goals_sets_positions_lane_and_count);
    RUN_TEST(test_set_agent_goals_rejects_more_than_max_goals);
    RUN_TEST(test_set_agent_goals_rejects_out_of_range_agent_idx);
    return test_summary(failures);
}
