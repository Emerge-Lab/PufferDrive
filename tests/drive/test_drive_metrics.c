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

#define NUPLAN_TEST_TRAJECTORY_STEPS 21

// Straight expert route east along y=0, one meter per step.
static void nuplan_test_build_expert_route(Agent *agent, float *traj_x, float *traj_y, int *valid) {
    for (int t = 0; t < NUPLAN_TEST_TRAJECTORY_STEPS; t++) {
        traj_x[t] = (float) t;
        traj_y[t] = 0.0f;
        valid[t] = 1;
    }
    agent->log_trajectory_x = traj_x;
    agent->log_trajectory_y = traj_y;
    agent->log_valid = valid;
    agent->trajectory_size = NUPLAN_TEST_TRAJECTORY_STEPS;
    agent->expert_route_idx = 0;
}

static int test_nuplan_progress_follows_expert_route(void) {
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    float traj_x[NUPLAN_TEST_TRAJECTORY_STEPS], traj_y[NUPLAN_TEST_TRAJECTORY_STEPS];
    int valid[NUPLAN_TEST_TRAJECTORY_STEPS];
    nuplan_test_build_expert_route(&agent, traj_x, traj_y, valid);

    agent.prev_x = 0.0f;
    agent.prev_y = 0.0f;
    agent.sim_x = 3.0f;
    EXPECT_NEAR(nuplan_expert_route_progress(&agent), 3.0f, 1e-4f);

    // Driving back along the route registers as negative progress.
    agent.prev_x = 3.0f;
    agent.sim_x = 1.0f;
    EXPECT_NEAR(nuplan_expert_route_progress(&agent), -2.0f, 1e-4f);

    // Lateral motion is orthogonal to the route, so it is not progress.
    agent.prev_x = 1.0f;
    agent.prev_y = 0.0f;
    agent.sim_y = 5.0f;
    EXPECT_NEAR(nuplan_expert_route_progress(&agent), 0.0f, 1e-4f);
    return 0;
}

static int test_nuplan_comfort_flags_violation(void) {
    Drive env = {0};
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    Log log = {0};
    float traj_x[NUPLAN_TEST_TRAJECTORY_STEPS], traj_y[NUPLAN_TEST_TRAJECTORY_STEPS];
    int valid[NUPLAN_TEST_TRAJECTORY_STEPS];
    nuplan_test_build_expert_route(&agent, traj_x, traj_y, valid);
    env.agents = &agent;
    env.logs = &log;
    env.dt = 0.1f;
    env.num_agents = 1;
    env.active_agent_count = 1;
    int active_indices[1] = {0};
    env.active_agent_indices = active_indices;

    agent.accel_long = NUPLAN_MAX_LON_ACCEL - 0.1f;
    accumulate_nuplan_metrics(&env, 0, &log);
    EXPECT_NEAR(log.nuplan_comfort_violation, 0.0f, 1e-6f);

    agent.accel_long = NUPLAN_MAX_LON_ACCEL + 0.1f;
    accumulate_nuplan_metrics(&env, 0, &log);
    EXPECT_NEAR(log.nuplan_comfort_violation, 1.0f, 1e-6f);
    return 0;
}

static int test_nuplan_score_matches_devkit_weights(void) {
    Drive env = {0};
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    Log log = {0};
    float traj_x[NUPLAN_TEST_TRAJECTORY_STEPS], traj_y[NUPLAN_TEST_TRAJECTORY_STEPS];
    int valid[NUPLAN_TEST_TRAJECTORY_STEPS];
    nuplan_test_build_expert_route(&agent, traj_x, traj_y, valid);
    env.dt = 0.1f;
    env.init_step = 0;
    env.scenario_length = NUPLAN_TEST_TRAJECTORY_STEPS - 1;

    // Expert covers 20m; ego covers 10m with no infraction and no violation.
    log.episode_length = 20.0f;
    log.nuplan_ego_progress_meters = 10.0f;
    calculate_nuplan_score(&env, &agent, &log);

    EXPECT_NEAR(log.nuplan_ego_progress_along_expert_route, 0.5f, 1e-5f);
    EXPECT_NEAR(log.nuplan_ego_is_making_progress, 1.0f, 1e-6f);
    EXPECT_NEAR(log.nuplan_driving_direction_compliance, 1.0f, 1e-6f);
    EXPECT_NEAR(log.nuplan_time_to_collision_within_bound, 1.0f, 1e-6f);
    EXPECT_NEAR(log.nuplan_ego_is_comfortable, 1.0f, 1e-6f);
    EXPECT_NEAR(log.nuplan_speed_limit_compliance, 1.0f, 1e-6f);
    // (5*0.5 + 5 + 4 + 2) / 16
    EXPECT_NEAR(log.nuplan_score, 13.5f / 16.0f, 1e-5f);

    // An at-fault collision zeroes the whole score through the multiplier.
    log.at_fault_collision_rate = 1.0f;
    calculate_nuplan_score(&env, &agent, &log);
    EXPECT_NEAR(log.nuplan_score, 0.0f, 1e-6f);
    return 0;
}

static int test_nuplan_wrong_way_window_scores_direction(void) {
    Drive env = {0};
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    Log log = {0};
    float traj_x[NUPLAN_TEST_TRAJECTORY_STEPS], traj_y[NUPLAN_TEST_TRAJECTORY_STEPS];
    int valid[NUPLAN_TEST_TRAJECTORY_STEPS];
    nuplan_test_build_expert_route(&agent, traj_x, traj_y, valid);
    env.dt = 0.1f;
    env.init_step = 0;
    env.scenario_length = NUPLAN_TEST_TRAJECTORY_STEPS - 1;
    log.episode_length = 20.0f;

    log.nuplan_worst_window_progress_meters = -1.0f;
    calculate_nuplan_score(&env, &agent, &log);
    EXPECT_NEAR(log.nuplan_driving_direction_compliance, 1.0f, 1e-6f);

    log.nuplan_worst_window_progress_meters = -4.0f;
    calculate_nuplan_score(&env, &agent, &log);
    EXPECT_NEAR(log.nuplan_driving_direction_compliance, 0.5f, 1e-6f);

    log.nuplan_worst_window_progress_meters = -7.0f;
    calculate_nuplan_score(&env, &agent, &log);
    EXPECT_NEAR(log.nuplan_driving_direction_compliance, 0.0f, 1e-6f);
    EXPECT_NEAR(log.nuplan_score, 0.0f, 1e-6f);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_metric_offroad_outside_grid);
    RUN_TEST(test_metric_invalid_position_resets);
    RUN_TEST(test_metric_on_road_lane_alignment);
    RUN_TEST(test_nuplan_progress_follows_expert_route);
    RUN_TEST(test_nuplan_comfort_flags_violation);
    RUN_TEST(test_nuplan_score_matches_devkit_weights);
    RUN_TEST(test_nuplan_wrong_way_window_scores_direction);
    return test_summary(failures);
}
