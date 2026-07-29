#include "include/drive_fixture.h"
#include "include/test.h"

static int test_render_default_frame(void) {
    srand(13);
    SetTraceLogLevel(LOG_WARNING);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 8, 0);
    env.render_mode = RENDER_WINDOW;
    allocate(&env);
    c_reset(&env);

    c_render(&env, VIEW_MODE_DEFAULT);
    EXPECT_TRUE(env.client != NULL);

    close_client(env.client);
    env.client = NULL;
    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_render_default_frame);
    return test_summary(failures);
}
