#include "drive.h"

#include "puffernet.h"

#include <time.h>
#include <unistd.h>

void demo() {
    Drive env = {
        .action_type = DISCRETE,
        .dynamics_model = CLASSIC,
        .dt = 0.1f,
        .episode_length = 91,
        .reward_collision = -0.1f,
        .reward_offroad = -0.1f,
        .map_name = "resources/drive/demo_replay_map.bin",
    };
    if (allocate(&env) != 0) {
        return;
    }
    c_reset(&env);
    c_render(&env);
    Weights *weights = load_weights("resources/drive/drive_weights.bin");
    int logit_sizes[1] = {action_dim_classic_discrete()};
    PufferNet *net = make_puffernet(weights, env.num_agents, env.obs_size, 256, 4, logit_sizes, 1);
    int accel_delta = 2;
    int steer_delta = 4;
    while (!WindowShouldClose()) {
        float *actions = env.actions;
        forward_puffernet(net, env.observations, env.actions);
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            int accel_idx = 3;
            int steer_idx = 4;
            if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
                accel_idx += accel_delta;
                if (accel_idx > 6) {
                    accel_idx = 6;
                }
            }
            if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
                accel_idx -= accel_delta;
                if (accel_idx < 0) {
                    accel_idx = 0;
                }
            }
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                steer_idx += steer_delta;
                if (steer_idx < 0) {
                    steer_idx = 0;
                }
            }
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                steer_idx -= steer_delta;
                if (steer_idx > 8) {
                    steer_idx = 8;
                }
            }
            actions[EGO_IDX]
                = (float) (accel_idx * ((int) (sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]))) + steer_idx);
        }
        c_step(&env);
        c_render(&env);
    }

    close_client(env.client);
    free_allocated(&env);
    free_puffernet(net);
    free(weights);
}

void performance_test() {
    long test_time = 10;
    Drive env = {
        .action_type = DISCRETE,
        .dynamics_model = CLASSIC,
        .dt = 0.1f,
        .episode_length = 91,
        .map_name = "resources/drive/demo_replay_map.bin",
    };
    if (allocate(&env) != 0) {
        return;
    }
    c_reset(&env);

    Weights *weights = load_weights("resources/drive/drive_weights.bin");
    int logit_sizes[1] = {action_dim_classic_discrete()};
    PufferNet *net = make_puffernet(weights, env.num_agents, env.obs_size, 256, 4, logit_sizes, 1);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        forward_puffernet(net, env.observations, env.actions);
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (long) (i * env.num_agents) / (end - start));
    free_allocated(&env);
    free_puffernet(net);
    free(weights);
}

int main() {
    demo();
    return 0;
}
