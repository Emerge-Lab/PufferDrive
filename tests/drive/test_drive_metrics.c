#include "include/drive_fixture.h"
#include "include/test.h"

static int test_metric_offroad_outside_grid(void) {
    srand(5);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 2, 0);
    int agent_idx = env.active_agent_indices[0];
    Agent *agent = &env.agents[agent_idx];
    agent->sim_x = env.grid_map->top_left_x - 1000.0f;
    agent->sim_y = env.grid_map->top_left_y + 1000.0f;
    copy_pose_to_prev(agent);

    compute_metrics(&env, agent_idx, 0);

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
    env.logs = &log;
    agent.sim_x = INVALID_POSITION;
    agent.metrics_array[COLLISION_IDX] = 1.0f;
    agent.metrics_array[OFFROAD_IDX] = 1.0f;

    compute_metrics(&env, 0, 0);

    EXPECT_NEAR(agent.metrics_array[COLLISION_IDX], 0.0f, 1e-6f);
    EXPECT_NEAR(agent.metrics_array[OFFROAD_IDX], 0.0f, 1e-6f);
    EXPECT_NEAR(agent.metrics_array[RED_LIGHT_IDX], 0.0f, 1e-6f);
    return 0;
}

static int test_metric_on_road_lane_alignment(void) {
    srand(7);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    int agent_idx = env.active_agent_indices[0];
    Agent *agent = &env.agents[agent_idx];

    compute_metrics(&env, agent_idx, 0);

    EXPECT_TRUE(agent->current_lane_idx != -1);
    EXPECT_TRUE(agent->metrics_array[LANE_ANGLE_IDX] >= -1.0f && agent->metrics_array[LANE_ANGLE_IDX] <= 1.0f);
    EXPECT_NEAR(agent->metrics_array[OFFROAD_IDX], 0.0f, 1e-5f);

    free_allocated(&env);
    return 0;
}

static void place_agent_on_goal(Agent *agent, int goal_idx, float goal_speed, float sim_speed) {
    agent->current_goal_idx = goal_idx;
    agent->list_goal_x[goal_idx] = agent->sim_x;
    agent->list_goal_y[goal_idx] = agent->sim_y;
    agent->list_goal_z[goal_idx] = agent->sim_z;
    agent->current_goal_x = agent->sim_x;
    agent->current_goal_y = agent->sim_y;
    agent->current_goal_z = agent->sim_z;
    agent->reward_coefs[REWARD_COEF_GOAL_RADIUS] = 5.0f;
    agent->reward_coefs[REWARD_COEF_GOAL_SPEED] = goal_speed;
    agent->sim_speed = sim_speed;
    copy_pose_to_prev(agent);
}

static int test_metric_final_goal_requires_speed(void) {
    srand(11);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    int agent_idx = env.active_agent_indices[0];
    Agent *agent = &env.agents[agent_idx];
    EXPECT_TRUE(agent->goal_count >= 2);
    int final_idx = agent->goal_count - 1;

    // Paper semantics: final goal is not consumed while faster than goal speed.
    env.goal_reach_requires_speed = 1;
    place_agent_on_goal(agent, final_idx, 3.0f, 10.0f);
    compute_metrics(&env, agent_idx, 0);
    EXPECT_NEAR(agent->metrics_array[REACHED_GOAL_IDX], 0.0f, 1e-6f);
    EXPECT_EQ_INT(agent->current_goal_idx, final_idx);

    // Below goal speed the same final goal is consumed.
    place_agent_on_goal(agent, final_idx, 3.0f, 1.0f);
    compute_metrics(&env, agent_idx, 0);
    EXPECT_NEAR(agent->metrics_array[REACHED_GOAL_IDX], 1.0f, 1e-6f);
    EXPECT_EQ_INT(agent->current_goal_idx, agent->goal_count);

    // Intermediate waypoints are consumed at any speed.
    place_agent_on_goal(agent, 0, 3.0f, 10.0f);
    compute_metrics(&env, agent_idx, 0);
    EXPECT_NEAR(agent->metrics_array[REACHED_GOAL_IDX], 1.0f, 1e-6f);
    EXPECT_EQ_INT(agent->current_goal_idx, 1);

    // Legacy semantics: final goal is consumed on entry regardless of speed.
    env.goal_reach_requires_speed = 0;
    place_agent_on_goal(agent, final_idx, 3.0f, 10.0f);
    compute_metrics(&env, agent_idx, 0);
    EXPECT_NEAR(agent->metrics_array[REACHED_GOAL_IDX], 1.0f, 1e-6f);
    EXPECT_EQ_INT(agent->current_goal_idx, agent->goal_count);

    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_metric_offroad_outside_grid);
    RUN_TEST(test_metric_invalid_position_resets);
    RUN_TEST(test_metric_on_road_lane_alignment);
    RUN_TEST(test_metric_final_goal_requires_speed);
    return test_summary(failures);
}
