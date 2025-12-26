#include "drivenet.h"
#include <string.h>
#include "../env_config.h"

// Use this test if the network changes to ensure that the forward pass
// matches the torch implementation to the 3rd or ideally 4th decimal place
void test_drivenet() {
    int num_obs = 1848;
    int num_actions = 2;
    int num_agents = 4;

    float *observations = calloc(num_agents * num_obs, sizeof(float));
    for (int i = 0; i < num_obs * num_agents; i++) {
        observations[i] = i % 7;
    }

    int *actions = calloc(num_agents * num_actions, sizeof(int));

    // Weights* weights = load_weights("resources/drive/puffer_drive_weights.bin");
    Weights *weights = load_weights("puffer_drive_weights.bin");
    DriveNet *net = init_drivenet(weights, num_agents, CLASSIC);

    forward(net, observations, actions);
    for (int i = 0; i < num_agents * num_actions; i++) {
        printf("idx: %d, action: %d, logits:", i, actions[i]);
        for (int j = 0; j < num_actions; j++) {
            printf(" %.6f", net->actor->output[i * num_actions + j]);
        }
        printf("\n");
    }
    free_drivenet(net);
    free(weights);
}

void demo() {

    // The settings below are hardcoded for demo purposes. Since the policy was
    // trained with these exact settings, note that changing them may lead to
    // weird behavior.
    Drive env = {
        .human_agent_idx = 0,
        .action_type = 0, // Discrete
        .dynamics_model = CLASSIC, // Classic dynamics
        .reward_vehicle_collision = -1.0f,
        .reward_offroad_collision = -1.0f,
        .reward_goal = 1.0f,
        .reward_goal_post_respawn = 0.25f,
        .goal_radius = 2.0f,
        .goal_behavior = 1,
        .goal_target_distance = 0.0f,
        .goal_speed = 0.0f,
        .dt = 0.1f,
        .episode_length = 1000,
        .termination_mode = 0,
        .collision_behavior = 0,
        .offroad_behavior = 0,
        .init_steps = 0,
        .init_mode = 0,
        .control_mode = 0,
        .map_name = "resources/drive/map_town_02_carla.bin",
    };
    allocate(&env);
    c_reset(&env);
    c_render(&env);
    Weights *weights = load_weights("resources/drive/puffer_drive_gljhhrl6.bin");
    DriveNet *net = init_drivenet(weights, env.active_agent_count, env.dynamics_model);

    int accel_delta = 2;
    int steer_delta = 4;
    while (!WindowShouldClose()) {
        int *actions = (int *)env.actions; // Single integer per agent

        forward(net, env.observations, actions);

        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (env.dynamics_model == CLASSIC) {
                // Classic dynamics: acceleration and steering
                int accel_idx = 3; // neutral (0 m/s²)
                int steer_idx = 6; // neutral (0.0 steering)

                if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
                    accel_idx += accel_delta;
                    if (accel_idx > 6)
                        accel_idx = 6;
                }
                if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
                    accel_idx -= accel_delta;
                    if (accel_idx < 0)
                        accel_idx = 0;
                }
                if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                    steer_idx += steer_delta; // Increase steering index for left turn
                    if (steer_idx > 12)
                        steer_idx = 12;
                }
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                    steer_idx -= steer_delta; // Decrease steering index for right turn
                    if (steer_idx < 0)
                        steer_idx = 0;
                }

                // Encode into single integer: action = accel_idx * 13 + steer_idx
                actions[env.human_agent_idx] = accel_idx * 13 + steer_idx;

            } else if (env.dynamics_model == JERK) {
                // Jerk dynamics: longitudinal and lateral jerk
                // JERK_LONG[4] = {-15.0f, -4.0f, 0.0f, 4.0f}
                // JERK_LAT[3] = {-4.0f, 0.0f, 4.0f}
                int jerk_long_idx = 2; // neutral (0.0)
                int jerk_lat_idx = 1;  // neutral (0.0)

                if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
                    jerk_long_idx = 3; // acceleration (4.0)
                }
                if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
                    jerk_long_idx = 0; // hard braking (-15.0)
                }
                if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                    jerk_lat_idx = 2; // left turn (4.0)
                }
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                    jerk_lat_idx = 0; // right turn (-4.0)
                }

                // Encode into single integer: action = jerk_long_idx * 3 + jerk_lat_idx
                actions[env.human_agent_idx] = jerk_long_idx * 3 + jerk_lat_idx;
            }
        }

        c_step(&env);
        c_render(&env);
    }

    close_client(env.client);
    free_allocated(&env);
    free_drivenet(net);
    free(weights);
}

void performance_test() {
    // Read configuration from INI file
    env_init_config conf = {0};
    const char *ini_file = "pufferlib/config/ocean/drive.ini";
    if (ini_parse(ini_file, handler, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s. Cannot determine environment configuration.\n", ini_file);
        exit(1);
    }

    long test_time = 10;
    Drive env = {
        .human_agent_idx = 0,
        .dynamics_model = conf.dynamics_model,
        .reward_vehicle_collision = conf.reward_vehicle_collision,
        .reward_offroad_collision = conf.reward_offroad_collision,
        .goal_radius = conf.goal_radius,
        .dt = conf.dt,
        .map_name = "resources/drive/binaries/map_000.bin",
        .init_steps = conf.init_steps,
    };
    clock_t start_time, end_time;
    double cpu_time_used;
    start_time = clock();
    allocate(&env);
    c_reset(&env);
    end_time = clock();
    cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Init time: %f\n", cpu_time_used);

    long start = time(NULL);
    int i = 0;
    int (*actions)[2] = (int (*)[2])env.actions;

    while (time(NULL) - start < test_time) {
        // Set random actions for all agents
        for (int j = 0; j < env.active_agent_count; j++) {
            int accel = rand() % 7;
            int steer = rand() % 13;
            actions[j][0] = accel; // -1, 0, or 1
            actions[j][1] = steer; // Random steering
        }

        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (i * env.active_agent_count) / (end - start));
    free_allocated(&env);
}

int main() {
    // performance_test();
    demo();
    // test_drivenet();
    return 0;
}
