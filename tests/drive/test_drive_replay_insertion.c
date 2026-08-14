#include "include/drive_fixture.h"
#include "include/test.h"

// The checked-in nuPlan scenario has 57 agents; 27 are valid at step 0 and 30 appear later.
#define NUPLAN_TOTAL_AGENT_COUNT 57
#define NUPLAN_VALID_AT_INIT_COUNT 27
#define NUPLAN_LATE_AGENT_COUNT 30

static int test_eval_replay_creates_late_appearing_agents(void) {
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    env.control_mode = CONTROL_MODE_VEHICLES;
    env.eval_mode = 1;
    allocate(&env);
    c_reset(&env);

    EXPECT_EQ_INT(env.num_total_agents, NUPLAN_TOTAL_AGENT_COUNT);
    EXPECT_EQ_INT(env.num_agents, NUPLAN_TOTAL_AGENT_COUNT);
    EXPECT_EQ_INT(env.active_agent_count, 1);
    EXPECT_EQ_INT(env.static_agent_count, NUPLAN_TOTAL_AGENT_COUNT - 1);
    EXPECT_TRUE(env.expert_static_agent_count >= NUPLAN_LATE_AGENT_COUNT);

    int invalid_at_reset = 0;
    for (int i = 0; i < env.num_total_agents; i++) {
        if (env.agents[i].sim_x == INVALID_POSITION) {
            invalid_at_reset++;
        }
    }
    EXPECT_TRUE(invalid_at_reset >= NUPLAN_LATE_AGENT_COUNT);

    free_allocated(&env);
    return 0;
}

static int test_training_replay_still_skips_late_appearing_agents(void) {
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    env.control_mode = CONTROL_MODE_VEHICLES;
    env.eval_mode = 0;
    allocate(&env);
    c_reset(&env);

    EXPECT_EQ_INT(env.num_agents, NUPLAN_VALID_AT_INIT_COUNT);

    free_allocated(&env);
    return 0;
}

static void fill_parked_log(
    Agent *agent,
    int steps,
    float x,
    float y,
    float *log_x,
    float *log_y,
    float *log_z,
    float *log_heading,
    float *log_vx,
    float *log_vy,
    float *log_length,
    float *log_width,
    float *log_height,
    int *log_valid) {
    for (int t = 0; t < steps; t++) {
        log_x[t] = x;
        log_y[t] = y;
        log_z[t] = 0.0f;
        log_heading[t] = 0.0f;
        log_vx[t] = 0.0f;
        log_vy[t] = 0.0f;
        log_length[t] = 4.0f;
        log_width[t] = 2.0f;
        log_height[t] = 1.5f;
        log_valid[t] = 1;
    }
    agent->type = VEHICLE;
    agent->trajectory_size = steps;
    agent->log_trajectory_x = log_x;
    agent->log_trajectory_y = log_y;
    agent->log_trajectory_z = log_z;
    agent->log_heading = log_heading;
    agent->log_velocity_x = log_vx;
    agent->log_velocity_y = log_vy;
    agent->log_length = log_length;
    agent->log_width = log_width;
    agent->log_height = log_height;
    agent->log_valid = log_valid;
    agent->sim_length = 4.0f;
    agent->sim_width = 2.0f;
    agent->sim_height = 1.5f;
    update_agent_radius(agent);
}

static int test_insertion_held_while_overlapping_active(void) {
    enum { STEPS = 2 };
    Drive env = {0};
    env.simulation_mode = SIMULATION_MODE_REPLAY;
    env.control_mode = CONTROL_MODE_VEHICLES;

    Agent agents[2] = {0};
    agents[0] = drive_test_agent(10.0f, 0.0f, 0.0f);
    float x1[STEPS], y1[STEPS], z1[STEPS], h1[STEPS], vx1[STEPS], vy1[STEPS], l1[STEPS], w1[STEPS], hh1[STEPS];
    int v1[STEPS];
    fill_parked_log(&agents[1], STEPS, 10.0f, 0.0f, x1, y1, z1, h1, vx1, vy1, l1, w1, hh1, v1);
    v1[0] = 0; // appears at t=1, right on top of the active agent
    invalidate_agent(&agents[1]);

    int active_indices[1] = {0};
    env.agents = agents;
    env.num_total_agents = 2;
    env.num_agents = 1;
    env.active_agent_count = 1;
    env.active_agent_indices = active_indices;
    env.timestep = 1;

    move_expert(&env, 1);
    EXPECT_TRUE(agents[1].sim_x == INVALID_POSITION);
    EXPECT_EQ_INT(agents[1].sim_valid, 0);

    // Active agent clears the spot: the held insertion succeeds on the next attempt
    agents[0].sim_x = 30.0f;
    agents[0].prev_x = 30.0f;
    move_expert(&env, 1);
    EXPECT_EQ_INT(agents[1].sim_valid, 1);
    EXPECT_NEAR(agents[1].sim_x, 10.0f, 1e-6f);

    return 0;
}

static int test_remove_bad_trajectories_blocks_reinsertion(void) {
    enum { STEPS = 8 };
    Drive env = {0};
    env.simulation_mode = SIMULATION_MODE_REPLAY;
    env.control_mode = CONTROL_MODE_VEHICLES;
    env.scenario_length = STEPS;
    env.dt = 0.1f;

    Agent agents[2] = {0};
    float x0[STEPS], y0[STEPS], z0[STEPS], h0[STEPS], vx0[STEPS], vy0[STEPS], l0[STEPS], w0[STEPS], hh0[STEPS];
    int v0[STEPS];
    fill_parked_log(&agents[0], STEPS, 0.0f, 0.0f, x0, y0, z0, h0, vx0, vy0, l0, w0, hh0, v0);
    for (int t = 0; t < STEPS; t++) {
        x0[t] = (float) t; // drives straight through the parked agent at x=4
        vx0[t] = 10.0f;
    }
    float x1[STEPS], y1[STEPS], z1[STEPS], h1[STEPS], vx1[STEPS], vy1[STEPS], l1[STEPS], w1[STEPS], hh1[STEPS];
    int v1[STEPS];
    fill_parked_log(&agents[1], STEPS, 4.0f, 0.0f, x1, y1, z1, h1, vx1, vy1, l1, w1, hh1, v1);

    int active_indices[1] = {0};
    int static_indices[1] = {1};
    int expert_indices[1] = {1};
    env.agents = agents;
    env.num_total_agents = 2;
    env.num_agents = 2;
    env.active_agent_count = 1;
    env.active_agent_indices = active_indices;
    env.static_agent_count = 1;
    env.static_agent_indices = static_indices;
    env.expert_static_agent_count = 1;
    env.expert_static_agent_indices = expert_indices;
    env.timestep = 0;

    remove_bad_trajectories(&env);

    for (int t = 0; t < STEPS; t++) {
        EXPECT_EQ_INT(agents[1].log_valid[t], 0);
    }
    EXPECT_TRUE(agents[1].sim_x == INVALID_POSITION);

    // Regression: the removed agent must not be re-inserted by move_expert at later steps
    env.timestep = 1;
    move_expert(&env, 1);
    EXPECT_TRUE(agents[1].sim_x == INVALID_POSITION);

    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_eval_replay_creates_late_appearing_agents);
    RUN_TEST(test_training_replay_still_skips_late_appearing_agents);
    RUN_TEST(test_insertion_held_while_overlapping_active);
    RUN_TEST(test_remove_bad_trajectories_blocks_reinsertion);
    return test_summary(failures);
}
