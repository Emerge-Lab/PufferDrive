#include "include/drive_fixture.h"
#include "include/test.h"

static int test_jerk_action_clipping(void) {
    srand(23);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    agent->sim_speed_signed = env.base_max_speed_mps * 0.9f;
    agent->steering_angle = STEERING_LIMIT * 0.9f;
    ((float (*)[2]) env.actions)[0][0] = -10.0f; // large braking
    ((float (*)[2]) env.actions)[0][1] = -10.0f; // large steering
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

// Braking to a standstill zeroes the acceleration but must report the driver's jerk
static int test_jerk_stop_reset_keeps_driver_jerk(void) {
    srand(29);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.dt = 0.3f;
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    agent->reward_coefs[REWARD_COEF_THROTTLE] = 1.0f;
    agent->reward_coefs[REWARD_COEF_STEER] = 1.0f;
    agent->reward_coefs[REWARD_COEF_ACC] = 1.0f;
    agent->reward_coefs[REWARD_COEF_SPEED] = 1.0f;
    float (*actions)[2] = (float (*)[2]) env.actions;
    const float braking_jerk_action = JERK_LONG[1] / -JERK_LONG[0]; // decodes to -4 m/s^3
    const float stop_speed_mps = 0.3f;
    const float stop_accel_mps2 = -3.0f;
    {
        agent->sim_vx = stop_speed_mps * agent->cos_heading;
        agent->sim_vy = stop_speed_mps * agent->sin_heading;
        update_agent_speed(agent);
        agent->accel_long = stop_accel_mps2;
        actions[0][0] = 0.0f;
        actions[0][1] = 0.0f;
        move_dynamics(&env, 0, env.active_agent_indices[0]);
        EXPECT_NEAR(agent->sim_speed, 0.0f, 1e-6f);
        EXPECT_NEAR(agent->accel_long, 0.0f, 1e-6f);
        EXPECT_NEAR(agent->jerk_long, 0.0f, 1e-6f);
    }
    {
        agent->sim_vx = stop_speed_mps * agent->cos_heading;
        agent->sim_vy = stop_speed_mps * agent->sin_heading;
        update_agent_speed(agent);
        agent->accel_long = stop_accel_mps2;
        actions[0][0] = braking_jerk_action;
        actions[0][1] = 0.0f;
        move_dynamics(&env, 0, env.active_agent_indices[0]);
        EXPECT_NEAR(agent->sim_speed, 0.0f, 1e-6f);
        EXPECT_NEAR(agent->accel_long, 0.0f, 1e-6f);
        EXPECT_NEAR(agent->jerk_long, JERK_LONG[1], 1e-3f);
    }
    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_jerk_action_clipping);
    RUN_TEST(test_dynamics_stopped_agent_clears_motion);
    RUN_TEST(test_dynamics_removed_agent_invalidated);
    RUN_TEST(test_jerk_stop_reset_keeps_driver_jerk);
    return test_summary(failures);
}
