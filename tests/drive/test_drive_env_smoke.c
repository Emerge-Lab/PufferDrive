#include "include/drive_fixture.h"
#include "include/test.h"

static int run_case(const char *name, const char *map_file, int simulation_mode, int num_agents) {
    srand(7);
    Drive env = drive_test_make_env(map_file, simulation_mode, num_agents, 0);
    int obs_size = compute_observation_size(&env);

    EXPECT_TRUE(env.num_agents > 0);
    EXPECT_TRUE(env.observations != NULL);
    EXPECT_TRUE(env.rewards != NULL);
    EXPECT_TRUE(drive_all_finite(env.observations, env.num_agents * obs_size));

    int saw_log = 0;
    for (int t = 0; t < env.scenario_length + 5; t++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
        EXPECT_TRUE(drive_all_finite(env.observations, env.num_agents * obs_size));
        EXPECT_TRUE(drive_all_finite(env.rewards, env.num_agents));
        if (env.log.n > 0.0f) {
            saw_log = 1;
        }
        for (int i = 0; i < env.num_agents; i++) {
            Agent *agent = &env.agents[i];
            int terminal_flags = (agent->metrics_array[COLLISION_IDX] > 0.0f)
                + (agent->metrics_array[OFFROAD_IDX] > 0.0f) + (agent->metrics_array[RED_LIGHT_IDX] > 0.0f);
            EXPECT_TRUE(terminal_flags <= 1);
        }
    }

    EXPECT_TRUE(saw_log);
    printf("case %s active=%d log_n=%.0f\n", name, env.num_agents, env.log.n);
    free_allocated(&env);
    return 0;
}

static int test_carla_gigaflow_load_step_log(void) {
    return run_case("carla-gigaflow", drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 32);
}

static int test_nuplan_gigaflow_load_step_log(void) {
    return run_case("nuplan-gigaflow", drive_nuplan_map(), SIMULATION_MODE_GIGAFLOW, 32);
}

static int test_nuplan_replay_load_step_log(void) {
    return run_case("nuplan-replay", drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1);
}

static int test_truncation_and_episode_log(void) {
    srand(11);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 8, 0);
    env.scenario_length = 3;
    allocate(&env);
    c_reset(&env);

    for (int t = 0; t < 3; t++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
    }

    EXPECT_TRUE(env.log.n > 0.0f);
    for (int i = 0; i < env.num_agents; i++) {
        EXPECT_EQ_INT(env.truncations[i], 1);
    }

    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_carla_gigaflow_load_step_log);
    RUN_TEST(test_nuplan_gigaflow_load_step_log);
    RUN_TEST(test_nuplan_replay_load_step_log);
    RUN_TEST(test_truncation_and_episode_log);
    return test_summary(failures);
}
