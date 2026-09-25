#include "include/drive_fixture.h"
#include "include/test.h"

#include <sys/wait.h>
#include <unistd.h>

static Drive create_test_env_with_cache_modes(int use_map_cache, int use_neighbor_cache, int num_agents) {
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, num_agents, use_map_cache);
    env.use_neighbor_cache = use_neighbor_cache;
    allocate(&env);
    c_reset(&env);
    return env;
}

static size_t compute_neighbor_cache_bytes(GridMap *grid_map) {
    if (grid_map->neighbor_cache_entities == NULL) {
        return 0;
    }
    int grid_cell_count = grid_map->grid_cols * grid_map->grid_rows;
    size_t bytes = grid_cell_count * sizeof(GridMapEntity *) + (grid_cell_count + 1) * sizeof(int);
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        bytes += grid_map->neighbor_cache_count[grid_index] * sizeof(GridMapEntity);
    }
    return bytes;
}

static int test_all_cache_modes_produce_identical_step_outputs(void) {
    const int steps = 20;
    srand(12345);
    Drive baseline = create_test_env_with_cache_modes(0, 0, 32);
    int obs_count = baseline.active_agent_count * compute_observation_size(&baseline);
    int agent_count = baseline.active_agent_count;
    float *obs_log = (float *) malloc(steps * obs_count * sizeof(float));
    float *rew_log = (float *) malloc(steps * agent_count * sizeof(float));
    unsigned char *term_log = (unsigned char *) malloc(steps * agent_count * sizeof(unsigned char));
    unsigned char *trunc_log = (unsigned char *) malloc(steps * agent_count * sizeof(unsigned char));

    for (int timestep = 0; timestep < steps; timestep++) {
        drive_set_neutral_actions(&baseline);
        c_step(&baseline);
        memcpy(&obs_log[timestep * obs_count], baseline.observations, obs_count * sizeof(float));
        memcpy(&rew_log[timestep * agent_count], baseline.rewards, agent_count * sizeof(float));
        memcpy(&term_log[timestep * agent_count], baseline.terminals, agent_count * sizeof(unsigned char));
        memcpy(&trunc_log[timestep * agent_count], baseline.truncations, agent_count * sizeof(unsigned char));
    }
    free_allocated(&baseline);

    for (int use_map_cache = 0; use_map_cache <= 1; use_map_cache++) {
        for (int use_neighbor_cache = 0; use_neighbor_cache <= 1; use_neighbor_cache++) {
            if (use_map_cache == 0 && use_neighbor_cache == 0) {
                continue;
            }
            drive_map_cache_clear();
            srand(12345);
            Drive env = create_test_env_with_cache_modes(use_map_cache, use_neighbor_cache, 32);
            EXPECT_EQ_INT(env.active_agent_count, agent_count);
            EXPECT_EQ_INT(env.active_agent_count * compute_observation_size(&env), obs_count);
            for (int timestep = 0; timestep < steps; timestep++) {
                drive_set_neutral_actions(&env);
                c_step(&env);
                EXPECT_EQ_INT(memcmp(&obs_log[timestep * obs_count], env.observations, obs_count * sizeof(float)), 0);
                EXPECT_EQ_INT(memcmp(&rew_log[timestep * agent_count], env.rewards, agent_count * sizeof(float)), 0);
                EXPECT_EQ_INT(
                    memcmp(&term_log[timestep * agent_count], env.terminals, agent_count * sizeof(unsigned char)),
                    0);
                EXPECT_EQ_INT(
                    memcmp(&trunc_log[timestep * agent_count], env.truncations, agent_count * sizeof(unsigned char)),
                    0);
            }
            free_allocated(&env);
        }
    }

    free(obs_log);
    free(rew_log);
    free(term_log);
    free(trunc_log);
    drive_map_cache_clear();
    return 0;
}

static int test_cache_mode_matrix_has_expected_map_and_neighbor_allocations(void) {
    for (int use_map_cache = 0; use_map_cache <= 1; use_map_cache++) {
        for (int use_neighbor_cache = 0; use_neighbor_cache <= 1; use_neighbor_cache++) {
            drive_map_cache_clear();
            Drive first = create_test_env_with_cache_modes(use_map_cache, use_neighbor_cache, 8);
            Drive second = create_test_env_with_cache_modes(use_map_cache, use_neighbor_cache, 8);

            EXPECT_EQ_INT(drive_map_cache_live_count(), use_map_cache);
            if (use_map_cache) {
                EXPECT_TRUE(first.shared_map == second.shared_map);
                EXPECT_TRUE(first.grid_map == second.grid_map);
                EXPECT_TRUE(first.road_elements == second.road_elements);
                EXPECT_EQ_INT(first.shared_map->ref_count, 2);
            } else {
                EXPECT_TRUE(first.shared_map == NULL);
                EXPECT_TRUE(second.shared_map == NULL);
                EXPECT_TRUE(first.grid_map != second.grid_map);
                EXPECT_TRUE(first.road_elements != second.road_elements);
            }

            if (use_neighbor_cache) {
                EXPECT_TRUE(first.obs_neighbor_scratch == NULL);
                EXPECT_TRUE(second.obs_neighbor_scratch == NULL);
                size_t first_neighbor_cache_bytes = compute_neighbor_cache_bytes(first.grid_map);
                size_t second_neighbor_cache_bytes = compute_neighbor_cache_bytes(second.grid_map);
                EXPECT_TRUE(first_neighbor_cache_bytes > 0);
                EXPECT_TRUE(first_neighbor_cache_bytes == second_neighbor_cache_bytes);
                size_t unique_neighbor_cache_bytes = first_neighbor_cache_bytes;
                if (first.grid_map->neighbor_cache_entities != second.grid_map->neighbor_cache_entities) {
                    unique_neighbor_cache_bytes += second_neighbor_cache_bytes;
                }
                EXPECT_TRUE(unique_neighbor_cache_bytes == first_neighbor_cache_bytes * (use_map_cache ? 1 : 2));
            } else {
                EXPECT_TRUE(first.grid_map->neighbor_cache_entities == NULL);
                EXPECT_TRUE(second.grid_map->neighbor_cache_entities == NULL);
                EXPECT_TRUE(first.obs_neighbor_scratch != NULL);
                EXPECT_TRUE(second.obs_neighbor_scratch != NULL);
                EXPECT_TRUE(first.obs_neighbor_scratch != second.obs_neighbor_scratch);
                size_t total_neighbor_scratch_bytes
                    = (first.grid_map->total_entities + second.grid_map->total_entities) * sizeof(GridMapEntity);
                EXPECT_TRUE(total_neighbor_scratch_bytes == 2 * first.grid_map->total_entities * sizeof(GridMapEntity));
            }

            drive_set_neutral_actions(&first);
            drive_set_neutral_actions(&second);
            c_step(&first);
            c_step(&second);
            free_allocated(&first);
            if (use_map_cache) {
                EXPECT_EQ_INT(drive_map_cache_live_count(), 1);
                EXPECT_EQ_INT(second.shared_map->ref_count, 1);
            }
            free_allocated(&second);
            EXPECT_EQ_INT(drive_map_cache_live_count(), 0);
        }
    }
    drive_map_cache_clear();
    return 0;
}

static int test_mixed_neighbor_modes_share_map_and_populate_neighbor_cache(void) {
    for (int first_env_uses_neighbor_cache = 0; first_env_uses_neighbor_cache <= 1; first_env_uses_neighbor_cache++) {
        drive_map_cache_clear();
        Drive first = create_test_env_with_cache_modes(1, first_env_uses_neighbor_cache, 8);
        Drive second = create_test_env_with_cache_modes(1, !first_env_uses_neighbor_cache, 8);

        EXPECT_EQ_INT(drive_map_cache_live_count(), 1);
        EXPECT_TRUE(first.shared_map == second.shared_map);
        EXPECT_TRUE(first.grid_map == second.grid_map);
        EXPECT_EQ_INT(first.shared_map->ref_count, 2);
        EXPECT_TRUE(first.grid_map->neighbor_cache_entities != NULL);
        EXPECT_TRUE(first.grid_map->neighbor_cache_count != NULL);
        EXPECT_TRUE(compute_neighbor_cache_bytes(first.grid_map) > 0);
        Drive *env_without_neighbor_cache = first_env_uses_neighbor_cache ? &second : &first;
        Drive *env_with_neighbor_cache = first_env_uses_neighbor_cache ? &first : &second;
        EXPECT_TRUE(env_without_neighbor_cache->obs_neighbor_scratch != NULL);
        EXPECT_TRUE(env_with_neighbor_cache->obs_neighbor_scratch == NULL);

        free_allocated(env_with_neighbor_cache);
        EXPECT_EQ_INT(drive_map_cache_live_count(), 1);
        EXPECT_EQ_INT(env_without_neighbor_cache->shared_map->ref_count, 1);
        drive_set_neutral_actions(env_without_neighbor_cache);
        c_step(env_without_neighbor_cache);
        free_allocated(env_without_neighbor_cache);
        EXPECT_EQ_INT(drive_map_cache_live_count(), 0);
    }
    drive_map_cache_clear();
    return 0;
}

static int test_repeated_single_map_lifetimes_reuse_one_cache_slot(void) {
    drive_map_cache_clear();
    for (int use_neighbor_cache = 0; use_neighbor_cache <= 1; use_neighbor_cache++) {
        for (int cycle = 0; cycle < 3; cycle++) {
            Drive env = create_test_env_with_cache_modes(1, use_neighbor_cache, 8);
            free_allocated(&env);
            EXPECT_EQ_INT(g_map_cache_count, 1);
            EXPECT_EQ_INT(drive_map_cache_live_count(), 0);
        }
    }
    drive_map_cache_clear();
    return 0;
}

static int test_forked_child_owns_and_frees_new_map_cache_entry(void) {
    drive_map_cache_clear();
    Drive warm = create_test_env_with_cache_modes(1, 1, 8);
    free_allocated(&warm);
    int parent_size_before_fork = g_map_cache_count;
    int fds[2];
    EXPECT_EQ_INT(pipe(fds), 0);

    pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        Drive child = create_test_env_with_cache_modes(1, 0, 8);
        int live_after_build = drive_map_cache_live_count();
        free_allocated(&child);
        int live_after_close = drive_map_cache_live_count();
        int payload[2] = {live_after_build, live_after_close};
        write(fds[1], payload, sizeof(payload));
        close(fds[1]);
        _exit(0);
    }

    close(fds[1]);
    int payload[2] = {-1, -1};
    int status = 0;
    read(fds[0], payload, sizeof(payload));
    close(fds[0]);
    waitpid(pid, &status, 0);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ_INT(WEXITSTATUS(status), 0);
    EXPECT_EQ_INT(payload[0], 1);
    EXPECT_EQ_INT(payload[1], 0);
    EXPECT_EQ_INT(g_map_cache_count, parent_size_before_fork);

    drive_map_cache_clear();
    return 0;
}

static int test_forked_child_reuses_preloaded_map_cache_entry(void) {
    drive_map_cache_clear();
    Drive preload_config = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 8, 1);
    preload_config.use_neighbor_cache = 1;
    const char *map_files[] = {drive_carla_map()};
    pid_t parent_pid = getpid();

    EXPECT_EQ_INT(preload_map_cache(&preload_config, map_files, 1), 1);
    EXPECT_EQ_INT(drive_map_cache_live_count(), 1);
    struct SharedMapData *preloaded = map_cache_lookup(&preload_config);
    EXPECT_TRUE(preloaded != NULL);
    EXPECT_EQ_INT(preloaded->ref_count, 1);
    EXPECT_EQ_INT(preloaded->owner_pid, parent_pid);

    int fds[2];
    EXPECT_EQ_INT(pipe(fds), 0);
    pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        Drive child = create_test_env_with_cache_modes(1, 1, 8);
        int payload[] = {
            child.shared_map == preloaded,
            child.grid_map == preloaded->grid_map,
            child.road_elements == preloaded->road_elements,
            child.shared_map->owner_pid == parent_pid,
            drive_map_cache_live_count(),
            child.shared_map->ref_count,
            0,
            0,
        };
        free_allocated(&child);
        payload[6] = drive_map_cache_live_count();
        payload[7] = preloaded->ref_count;
        write(fds[1], payload, sizeof(payload));
        close(fds[1]);
        _exit(0);
    }

    close(fds[1]);
    int payload[8] = {0};
    int status = 0;
    read(fds[0], payload, sizeof(payload));
    close(fds[0]);
    waitpid(pid, &status, 0);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ_INT(WEXITSTATUS(status), 0);
    EXPECT_EQ_INT(payload[0], 1);
    EXPECT_EQ_INT(payload[1], 1);
    EXPECT_EQ_INT(payload[2], 1);
    EXPECT_EQ_INT(payload[3], 1);
    EXPECT_EQ_INT(payload[4], 1);
    EXPECT_EQ_INT(payload[5], 2);
    EXPECT_EQ_INT(payload[6], 1);
    EXPECT_EQ_INT(payload[7], 1);
    EXPECT_EQ_INT(preloaded->ref_count, 1);

    EXPECT_EQ_INT(release_preloaded_map_cache(&preload_config, map_files, 1), 0);
    EXPECT_EQ_INT(drive_map_cache_live_count(), 0);
    free(preload_config.map_name);
    drive_map_cache_clear();
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_all_cache_modes_produce_identical_step_outputs);
    RUN_TEST(test_cache_mode_matrix_has_expected_map_and_neighbor_allocations);
    RUN_TEST(test_mixed_neighbor_modes_share_map_and_populate_neighbor_cache);
    RUN_TEST(test_repeated_single_map_lifetimes_reuse_one_cache_slot);
    RUN_TEST(test_forked_child_owns_and_frees_new_map_cache_entry);
    RUN_TEST(test_forked_child_reuses_preloaded_map_cache_entry);
    return test_summary(failures);
}
