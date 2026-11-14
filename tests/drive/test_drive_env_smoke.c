#include "include/drive_fixture.h"
#include "include/test.h"

static int run_case(const char *name, const char *map_file, int simulation_mode, int num_agents) {
    srand(7);
    Drive env = drive_test_make_env(map_file, simulation_mode, num_agents, 0);
    int obs_size = compute_observation_size(&env);

    EXPECT_TRUE(env.active_agent_count > 0);
    EXPECT_TRUE(env.observations != NULL);
    EXPECT_TRUE(env.rewards != NULL);
    EXPECT_TRUE(drive_all_finite(env.observations, env.active_agent_count * obs_size));

    int saw_log = 0;
    for (int t = 0; t < env.scenario_length + 5; t++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
        EXPECT_TRUE(drive_all_finite(env.observations, env.active_agent_count * obs_size));
        EXPECT_TRUE(drive_all_finite(env.rewards, env.active_agent_count));
        if (env.log.n > 0.0f) {
            saw_log = 1;
        }
        for (int i = 0; i < env.active_agent_count; i++) {
            Agent *agent = &env.agents[env.active_agent_indices[i]];
            int terminal_flags = (agent->metrics_array[COLLISION_IDX] > 0.0f)
                + (agent->metrics_array[OFFROAD_IDX] > 0.0f) + (agent->metrics_array[RED_LIGHT_IDX] > 0.0f);
            EXPECT_TRUE(terminal_flags <= 1);
        }
    }

    EXPECT_TRUE(saw_log);
    printf("case %s active=%d log_n=%.0f\n", name, env.active_agent_count, env.log.n);
    free_allocated(&env);
    return 0;
}

static int test_carla_gigaflow_load_step_log(void) {
    return run_case("carla-gigaflow", drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 32);
}

static int test_failed_init_spawn_keeps_policy_slot(void) {
    // more agents than Town01 has room for: failed spawns stay addressable, removed and masked
    int requested_agent_count = 3000;
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, requested_agent_count, 0);
    EXPECT_EQ_INT(env.active_agent_count, requested_agent_count);
    int removed_agent_count = 0;
    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        removed_agent_count += agent->removed;
        EXPECT_EQ_INT(agent->sim_valid, !agent->removed);
        EXPECT_EQ_INT(env.masks[i], !agent->removed);
    }
    EXPECT_TRUE(removed_agent_count > 0);
    free_allocated(&env);
    return 0;
}

static int test_ground_truth_trajectories_fit_the_buffer_width(void) {
    Drive env = drive_test_make_env(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    int agent_count = env.active_agent_count;
    Agent *agent = &env.agents[env.active_agent_indices[0]];
    int buffer_widths[2] = {agent->trajectory_size / 2, agent->trajectory_size + 7};
    for (int w = 0; w < 2; w++) {
        int num_timesteps = buffer_widths[w];
        int cell_count = agent_count * num_timesteps;
        float *xyz_heading = (float *) calloc(4 * cell_count, sizeof(float));
        int *valid = (int *) malloc(cell_count * sizeof(int));
        int *ids = (int *) calloc(2 * agent_count, sizeof(int));
        for (int k = 0; k < cell_count; k++) {
            valid[k] = -1;
        }
        c_get_global_ground_truth_trajectories(
            &env,
            num_timesteps,
            xyz_heading,
            xyz_heading + cell_count,
            xyz_heading + 2 * cell_count,
            xyz_heading + 3 * cell_count,
            valid,
            ids,
            ids + agent_count);
        for (int t = 0; t < num_timesteps; t++) {
            int expected_valid = t < agent->trajectory_size ? agent->log_valid[t] : 0;
            EXPECT_EQ_INT(valid[t], expected_valid);
        }
        free(xyz_heading);
        free(valid);
        free(ids);
    }
    free_allocated(&env);
    return 0;
}

static int test_nuplan_gigaflow_load_step_log(void) {
    return run_case("nuplan-gigaflow", drive_nuplan_map(), SIMULATION_MODE_GIGAFLOW, 32);
}

static int test_nuplan_replay_load_step_log(void) {
    return run_case("nuplan-replay", drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1);
}

static int test_early_reset_logs_and_resets(void) {
    srand(7);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 32, 0);
    EXPECT_TRUE(env.active_agent_count >= 4);
    env.termination_mode = 1;
    env.inactive_agent_threshold = 0.4f;

    int removed_target = env.active_agent_count / 2 + 1;
    for (int i = 0; i < removed_target; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        invalidate_agent(agent);
        agent->removed = 1;
    }
    drive_set_neutral_actions(&env);
    c_step(&env);

    EXPECT_TRUE(env.log.n > 0.0f);
    EXPECT_EQ_INT(env.autoreset_pending, 1);

    drive_set_neutral_actions(&env);
    c_step(&env);
    EXPECT_EQ_INT(env.autoreset_pending, 0);
    free_allocated(&env);
    return 0;
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
    EXPECT_EQ_INT(env.timestep, 3);
    EXPECT_EQ_INT(env.autoreset_pending, 1);
    for (int i = 0; i < env.active_agent_count; i++) {
        EXPECT_EQ_INT(env.truncations[i], 1);
        EXPECT_EQ_INT(env.masks[i], 0);
    }

    drive_set_neutral_actions(&env);
    c_step(&env);
    EXPECT_EQ_INT(env.autoreset_pending, 0);
    EXPECT_EQ_INT(env.timestep, env.init_step);
    for (int i = 0; i < env.active_agent_count; i++) {
        Agent *agent = &env.agents[env.active_agent_indices[i]];
        int expected_mask = !(agent->stopped || agent->removed || agent->is_blind_partner || agent->is_phantom_braker);
        EXPECT_EQ_INT(env.truncations[i], 0);
        EXPECT_EQ_INT(env.terminals[i], 0);
        EXPECT_TRUE(env.rewards[i] == 0.0f);
        EXPECT_EQ_INT(env.masks[i], expected_mask);
    }

    free_allocated(&env);
    return 0;
}

static int test_stagger_first_episode(void) {
    const int scenario_length = 40;
    const int min_horizon = 5;
    const int num_seeds = 6;
    int first_episode_ends[6];
    srand(5);
    for (int seed_idx = 0; seed_idx < num_seeds; seed_idx++) {
        Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 8, 0);
        env.scenario_length = scenario_length;
        env.init_step_min_horizon = min_horizon;
        env.stagger_first_episode = 1;
        env.init_seed = (uint64_t) seed_idx;
        allocate(&env);
        c_reset(&env);
        EXPECT_EQ_INT(env.first_reset_pending, 0);
        EXPECT_TRUE(env.episode_end_timestep >= min_horizon);
        EXPECT_TRUE(env.episode_end_timestep <= scenario_length);
        first_episode_ends[seed_idx] = env.episode_end_timestep;
        free_allocated(&env);
    }
    int distinct_end_count = 0;
    for (int seed_idx = 1; seed_idx < num_seeds; seed_idx++) {
        distinct_end_count += first_episode_ends[seed_idx] != first_episode_ends[0];
    }
    EXPECT_TRUE(distinct_end_count > 0);

    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 8, 0);
    env.scenario_length = scenario_length;
    env.init_step_min_horizon = min_horizon;
    env.stagger_first_episode = 1;
    env.init_seed = 3;
    allocate(&env);
    c_reset(&env);
    EXPECT_EQ_INT(env.episode_end_timestep, first_episode_ends[3]);
    int first_episode_end = env.episode_end_timestep;
    int step_count = 0;
    while (!env.autoreset_pending && step_count <= scenario_length) {
        drive_set_neutral_actions(&env);
        c_step(&env);
        step_count++;
    }
    EXPECT_EQ_INT(env.autoreset_pending, 1);
    EXPECT_EQ_INT(step_count, first_episode_end);
    EXPECT_EQ_INT(env.timestep, first_episode_end);
    for (int i = 0; i < env.active_agent_count; i++) {
        EXPECT_EQ_INT(env.truncations[i], 1);
    }

    drive_set_neutral_actions(&env);
    c_step(&env);
    EXPECT_EQ_INT(env.autoreset_pending, 0);
    EXPECT_EQ_INT(env.timestep, env.init_step);
    EXPECT_EQ_INT(env.episode_end_timestep, scenario_length);
    for (int t = 0; t < scenario_length; t++) {
        drive_set_neutral_actions(&env);
        c_step(&env);
    }
    EXPECT_EQ_INT(env.autoreset_pending, 1);
    EXPECT_EQ_INT(env.timestep, scenario_length);
    free_allocated(&env);

    Drive eval_env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 8, 0);
    eval_env.scenario_length = scenario_length;
    eval_env.init_step_min_horizon = min_horizon;
    eval_env.stagger_first_episode = 1;
    eval_env.eval_mode = 1;
    allocate(&eval_env);
    c_reset(&eval_env);
    EXPECT_EQ_INT(eval_env.first_reset_pending, 0);
    EXPECT_EQ_INT(eval_env.episode_end_timestep, scenario_length);
    free_allocated(&eval_env);
    return 0;
}

static int test_eval_spawn_keeps_edge_clearance(void) {
    const char *town04_map = DRIVE_TEST_REPO_ROOT "/pufferlib/resources/drive/binaries/carla/opendrive__Town04.bin";
    const int reset_count = 8;
    const int agent_count = 50;
    Drive env = drive_test_env_config(town04_map, SIMULATION_MODE_GIGAFLOW, agent_count, 0);
    env.eval_mode = 1;
    allocate(&env);
    c_reset(&env);
    EXPECT_EQ_INT(env.active_agent_count, agent_count);

    for (int reset_idx = 0; reset_idx < reset_count; reset_idx++) {
        for (int i = 0; i < env.active_agent_count; i++) {
            Agent *agent = &env.agents[env.active_agent_indices[i]];
            EXPECT_EQ_INT(agent->removed, 0);
            EXPECT_FALSE(check_spawn_offroad(&env, agent, EVAL_SPAWN_EDGE_CLEARANCE_M));
        }
        env.timestep = env.scenario_length;
        c_reset(&env);
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
    RUN_TEST(test_early_reset_logs_and_resets);
    RUN_TEST(test_stagger_first_episode);
    RUN_TEST(test_eval_spawn_keeps_edge_clearance);
    RUN_TEST(test_failed_init_spawn_keeps_policy_slot);
    RUN_TEST(test_ground_truth_trajectories_fit_the_buffer_width);
    return test_summary(failures);
}
