#include "include/drive_fixture.h"
#include "include/test.h"

static int test_metric_offroad_outside_grid(void) {
    srand(5);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 2, 0);
    int agent_idx = 0;
    Agent *agent = &env.agents[agent_idx];
    agent->sim_x = env.grid_map->top_left_x - 1000.0f;
    agent->sim_y = env.grid_map->top_left_y + 1000.0f;
    copy_pose_to_prev(agent);

    compute_metrics(&env, agent_idx);

    EXPECT_NEAR(agent->metrics_array[OFFROAD_IDX], 1.0f, 1e-5f);
    EXPECT_NEAR(agent->metrics_array[COLLISION_IDX], 0.0f, 1e-5f);
    EXPECT_NEAR(agent->metrics_array[RED_LIGHT_IDX], 0.0f, 1e-5f);

    free_allocated(&env);
    return 0;
}

static int test_metric_invalid_position_resets(void) {
    Drive env = {0};
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    Log log = {0};
    env.agents = &agent;
    env.num_total_agents = 1;
    env.logs = &log;
    agent.sim_x = INVALID_POSITION;
    agent.metrics_array[COLLISION_IDX] = 1.0f;
    agent.metrics_array[OFFROAD_IDX] = 1.0f;

    compute_metrics(&env, 0);

    EXPECT_NEAR(agent.metrics_array[COLLISION_IDX], 0.0f, 1e-6f);
    EXPECT_NEAR(agent.metrics_array[OFFROAD_IDX], 0.0f, 1e-6f);
    EXPECT_NEAR(agent.metrics_array[RED_LIGHT_IDX], 0.0f, 1e-6f);
    return 0;
}

static int test_metric_on_road_lane_alignment(void) {
    srand(7);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    int agent_idx = 0;
    Agent *agent = &env.agents[agent_idx];

    compute_metrics(&env, agent_idx);

    EXPECT_TRUE(agent->current_lane_idx != -1);
    EXPECT_TRUE(agent->metrics_array[LANE_ANGLE_IDX] >= -1.0f && agent->metrics_array[LANE_ANGLE_IDX] <= 1.0f);
    EXPECT_NEAR(agent->metrics_array[OFFROAD_IDX], 0.0f, 1e-5f);

    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_metric_offroad_outside_grid);
    RUN_TEST(test_metric_invalid_position_resets);
    RUN_TEST(test_metric_on_road_lane_alignment);
    return test_summary(failures);
}
