#include "include/drive_fixture.h"
#include "include/test.h"

#include <sys/wait.h>
#include <unistd.h>

static int test_obs_reward_done_parity_cache_on_vs_off(void) {
    const int steps = 20;
    srand(12345);
    Drive off = drive_test_make_env(drive_carla_map(), SIMULATION_GIGAFLOW, 32, 0);
    int obs_count = off.active_agent_count * compute_observation_size(&off);
    int agent_count = off.active_agent_count;
    float *obs_log = (float *) malloc(steps * obs_count * sizeof(float));
    float *rew_log = (float *) malloc(steps * agent_count * sizeof(float));
    unsigned char *term_log = (unsigned char *) malloc(steps * agent_count * sizeof(unsigned char));
    unsigned char *trunc_log = (unsigned char *) malloc(steps * agent_count * sizeof(unsigned char));

    for (int t = 0; t < steps; t++) {
        drive_set_neutral_actions(&off);
        c_step(&off);
        memcpy(&obs_log[t * obs_count], off.observations, obs_count * sizeof(float));
        memcpy(&rew_log[t * agent_count], off.rewards, agent_count * sizeof(float));
        memcpy(&term_log[t * agent_count], off.terminals, agent_count * sizeof(unsigned char));
        memcpy(&trunc_log[t * agent_count], off.truncations, agent_count * sizeof(unsigned char));
    }
    free_allocated(&off);

    srand(12345);
    Drive on = drive_test_make_env(drive_carla_map(), SIMULATION_GIGAFLOW, 32, 1);
    EXPECT_EQ_INT(on.active_agent_count, agent_count);
    EXPECT_EQ_INT(on.active_agent_count * compute_observation_size(&on), obs_count);
    for (int t = 0; t < steps; t++) {
        drive_set_neutral_actions(&on);
        c_step(&on);
        EXPECT_EQ_INT(memcmp(&obs_log[t * obs_count], on.observations, obs_count * sizeof(float)), 0);
        EXPECT_EQ_INT(memcmp(&rew_log[t * agent_count], on.rewards, agent_count * sizeof(float)), 0);
        EXPECT_EQ_INT(memcmp(&term_log[t * agent_count], on.terminals, agent_count * sizeof(unsigned char)), 0);
        EXPECT_EQ_INT(memcmp(&trunc_log[t * agent_count], on.truncations, agent_count * sizeof(unsigned char)), 0);
    }
    free_allocated(&on);
    free(obs_log);
    free(rew_log);
    free(term_log);
    free(trunc_log);
    drive_map_cache_clear();
    return 0;
}

static int close_order_case(const int *order) {
    drive_map_cache_clear();
    srand(9);
    Drive envs[3];
    for (int i = 0; i < 3; i++) {
        envs[i] = drive_test_make_env(drive_carla_map(), SIMULATION_GIGAFLOW, 8, 1);
        drive_set_neutral_actions(&envs[i]);
        c_step(&envs[i]);
    }
    EXPECT_EQ_INT(g_map_cache_count, 1);
    EXPECT_EQ_INT(drive_map_cache_live_count(), 1);
    EXPECT_EQ_INT(g_map_cache[0]->ref_count, 3);
    for (int i = 0; i < 3; i++) {
        free_allocated(&envs[order[i]]);
    }
    EXPECT_EQ_INT(drive_map_cache_live_count(), 0);
    drive_map_cache_clear();
    return 0;
}

static int test_multi_env_close_orderings_do_not_crash(void) {
    const int order_a[3] = {0, 1, 2};
    const int order_b[3] = {2, 1, 0};
    const int order_c[3] = {1, 2, 0};
    const int order_d[3] = {1, 0, 2};
    EXPECT_EQ_INT(close_order_case(order_a), 0);
    EXPECT_EQ_INT(close_order_case(order_b), 0);
    EXPECT_EQ_INT(close_order_case(order_c), 0);
    EXPECT_EQ_INT(close_order_case(order_d), 0);
    return 0;
}

static int test_cache_size_bounded_by_unique_maps(void) {
    drive_map_cache_clear();
    for (int cycle = 0; cycle < 3; cycle++) {
        Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_GIGAFLOW, 8, 1);
        free_allocated(&env);
        EXPECT_EQ_INT(g_map_cache_count, 1);
        EXPECT_EQ_INT(drive_map_cache_live_count(), 0);
    }
    drive_map_cache_clear();
    return 0;
}

static int test_forked_child_can_build_and_free_its_own_entry(void) {
    drive_map_cache_clear();
    Drive warm = drive_test_make_env(drive_carla_map(), SIMULATION_GIGAFLOW, 8, 1);
    free_allocated(&warm);
    int parent_size_before_fork = g_map_cache_count;
    int fds[2];
    EXPECT_EQ_INT(pipe(fds), 0);

    pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        Drive child = drive_test_make_env(drive_carla_map(), SIMULATION_GIGAFLOW, 8, 1);
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

int main(void) {
    int failures = 0;
    RUN_TEST(test_obs_reward_done_parity_cache_on_vs_off);
    RUN_TEST(test_multi_env_close_orderings_do_not_crash);
    RUN_TEST(test_cache_size_bounded_by_unique_maps);
    RUN_TEST(test_forked_child_can_build_and_free_its_own_entry);
    return test_summary(failures);
}
