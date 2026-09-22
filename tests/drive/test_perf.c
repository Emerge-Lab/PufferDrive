#include "include/drive_fixture.h"
#include "include/test.h"

#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec / 1e9;
}

static int test_simulator_raw_perf(void) {
    const double timeout = 5.0;
    const int baseline_sps = 24690;
    const float threshold = 0.8f * (float) baseline_sps;
    srand(17);
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 32, 0);
    EXPECT_EQ_INT(env.num_agents, 32);

    int tick = 0;
    double start = now_seconds();
    while (now_seconds() - start < timeout) {
        int *actions = (int *) env.actions;
        for (int i = 0; i < env.num_agents; i++) {
            actions[i] = rand() % (7 * 9);
        }
        c_step(&env);
        tick++;
    }
    double elapsed = now_seconds() - start;
    float sps = (float) env.num_agents * (float) tick / (float) elapsed;
    printf("Steps per second (SPS): %.1f\n", sps);
    printf("Ticks: %d elapsed: %.3f\n", tick, elapsed);

    EXPECT_TRUE(sps >= threshold);
    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_simulator_raw_perf);
    return test_summary(failures);
}
