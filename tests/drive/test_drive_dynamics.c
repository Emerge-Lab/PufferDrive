#include "include/drive_fixture.h"
#include "include/test.h"

static int test_classic_action_clipping(void) {
    srand(19);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.action_type = 1;
    env.dynamics_model = DYNAMICS_MODEL_CLASSIC;
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    agent->sim_speed_signed = MAX_SPEED * 0.9f;
    agent->steering_angle = STEERING_LIMIT * 0.9f;
    ((float (*)[2]) env.actions)[0][0] = ACCELERATION_VALUES[6] * 10.0f; // large acceleration
    ((float (*)[2]) env.actions)[0][1] = STEERING_VALUES[8] * 10.0f;     // large steering
    move_dynamics(&env, 0, env.active_agent_indices[0]);
    EXPECT_TRUE(agent->sim_speed_signed <= MAX_SPEED);
    EXPECT_TRUE(agent->steering_angle <= STEERING_LIMIT);
    free_allocated(&env);
    return 0;
}

static int test_jerk_action_clipping(void) {
    srand(23);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.action_type = 1;
    env.dynamics_model = DYNAMICS_MODEL_JERK;
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    agent->sim_speed_signed = MAX_SPEED * 0.9f;
    agent->steering_angle = STEERING_LIMIT * 0.9f;
    ((float (*)[2]) env.actions)[0][0] = ACCELERATION_VALUES[0] * 10.0f; // large braking
    ((float (*)[2]) env.actions)[0][1] = STEERING_VALUES[0] * 10.0f;     // large steering
    move_dynamics(&env, 0, env.active_agent_indices[0]);
    EXPECT_TRUE(agent->accel_long >= ACCEL_LONG_LIMIT[0]);
    EXPECT_TRUE(agent->accel_lat <= ACCEL_LAT_LIMIT[1]);
    EXPECT_TRUE(agent->steering_angle <= STEERING_LIMIT);
    free_allocated(&env);
    return 0;
}

static int test_dynamics_stopped_agent_clears_motion(void) {
    Drive env = {0};
    Agent agent = drive_test_agent(5.0f, 3.0f, 0.5f);
    env.agents = &agent;
    env.dt = 0.1f;
    agent.stopped = 1;
    agent.sim_vx = 4.0f;
    update_agent_speed(&agent);
    agent.steering_angle = 0.3f;

    move_dynamics(&env, 0, 0);

    EXPECT_NEAR(agent.sim_vx, 0.0f, 1e-6f);
    EXPECT_NEAR(agent.sim_speed, 0.0f, 1e-6f);
    EXPECT_NEAR(agent.steering_angle, 0.0f, 1e-6f);
    return 0;
}

static int test_dynamics_removed_agent_invalidated(void) {
    Drive env = {0};
    Agent agent = drive_test_agent(5.0f, 3.0f, 0.5f);
    env.agents = &agent;
    env.dt = 0.1f;
    agent.removed = 1;

    move_dynamics(&env, 0, 0);

    EXPECT_EQ_INT(agent.sim_valid, 0);
    EXPECT_NEAR(agent.sim_x, INVALID_POSITION, 1e-3f);
    return 0;
}

static int test_neutral_actions_zero_out(void) {
    {
        Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
        env.action_type = 0;
        env.dynamics_model = DYNAMICS_MODEL_CLASSIC;
        drive_set_neutral_actions(&env);
        int action_val = ((int *) env.actions)[0];
        int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
        int acceleration_index = action_val / num_steer;
        int steering_index = action_val % num_steer;
        EXPECT_NEAR(ACCELERATION_VALUES[acceleration_index], 0.0f, 1e-6f);
        EXPECT_NEAR(STEERING_VALUES[steering_index], 0.0f, 1e-6f);
        free_allocated(&env);
    }
    {
        Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
        env.action_type = 0;
        env.dynamics_model = DYNAMICS_MODEL_JERK;
        drive_set_neutral_actions(&env);
        int action_val = ((int *) env.actions)[0];
        int num_lat = sizeof(JERK_LAT) / sizeof(JERK_LAT[0]);
        int j_long_idx = action_val / num_lat;
        int j_lat_idx = action_val % num_lat;
        EXPECT_NEAR(JERK_LONG[j_long_idx], 0.0f, 1e-6f);
        EXPECT_NEAR(JERK_LAT[j_lat_idx], 0.0f, 1e-6f);
        free_allocated(&env);
    }
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_classic_action_clipping);
    RUN_TEST(test_jerk_action_clipping);
    RUN_TEST(test_dynamics_stopped_agent_clears_motion);
    RUN_TEST(test_dynamics_removed_agent_invalidated);
    RUN_TEST(test_neutral_actions_zero_out);
    return test_summary(failures);
}
