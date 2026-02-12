#include "drivenet.h"
#include "error.h"
#include "libgen.h"
#include "../env_config.h"
#include <string.h>

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

int demo(const char *map_name, const char *policy_name, int show_grid, int obs_only, int lasers, int show_human_logs,
         int frame_skip, const char *view_mode, const char *output_topdown, const char *output_agent, int num_maps,
         int zoom_in) {

    // Parse configuration from INI file
    env_init_config conf = {0};
    const char *ini_file = "pufferlib/config/ocean/drive.ini";
    if (ini_parse(ini_file, handler, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s. Cannot determine environment configuration.\n", ini_file);
        return -1;
    }

    char map_buffer[100];
    if (map_name == NULL) {
        srand(time(NULL));
        int random_map = rand() % num_maps;
        sprintf(map_buffer, "%s/map_%03d.bin", conf.map_dir, random_map);
        map_name = map_buffer;
    }

    // Initialize environment with all config values from INI [env] section
    Drive env = {
        .action_type = conf.action_type,
        .dynamics_model = conf.dynamics_model,
        .reward_vehicle_collision = conf.reward_vehicle_collision,
        .reward_offroad_collision = conf.reward_offroad_collision,
        .reward_goal = conf.reward_goal,
        .reward_goal_post_respawn = conf.reward_goal_post_respawn,
        .goal_radius = conf.goal_radius,
        .goal_behavior = conf.goal_behavior,
        .goal_target_distance = conf.goal_target_distance,
        .goal_speed = conf.goal_speed,
        .dt = conf.dt,
        .episode_length = conf.episode_length,
        .termination_mode = conf.termination_mode,
        .collision_behavior = conf.collision_behavior,
        .offroad_behavior = conf.offroad_behavior,
        .init_steps = conf.init_steps,
        .init_mode = conf.init_mode,
        .control_mode = conf.control_mode,
        .map_name = (char *)map_name,
    };
    allocate(&env);
    if (env.active_agent_count == 0) {
        fprintf(stderr, "Error: No active agents found in map '%s' with init_mode=%d. Cannot run demo.\n", env.map_name,
                conf.init_mode);
        free_allocated(&env);
        return -1;
    }
    c_reset(&env);
    c_render(&env);
    Weights *weights = load_weights((char *)policy_name);
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
    return 0;
}

void performance_test() {

    long test_time = 10;
    Drive env = {
        .human_agent_idx = 0,
        .dynamics_model = CLASSIC, // Classic dynamics
        .action_type = 0,          // Discrete
        .map_name = "resources/drive/binaries/map_000.bin",
        .dt = 0.1f,
        .init_steps = 0,
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

int main(int argc, char *argv[]) {
    // Visualization-only parameters (not in [env] section)
    int show_grid = 0;
    int obs_only = 0;
    int lasers = 0;
    int show_human_logs = 0;
    int frame_skip = 1;
    int zoom_in = 0;
    const char *view_mode = "both";

    // File paths and num_maps (not in [env] section)
    const char *map_name = NULL;
    const char *policy_name = "resources/drive/puffer_drive_weights.bin";
    const char *output_topdown = NULL;
    const char *output_agent = NULL;
    int num_maps = 1;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--show-grid") == 0) {
            show_grid = 1;
        } else if (strcmp(argv[i], "--obs-only") == 0) {
            obs_only = 1;
        } else if (strcmp(argv[i], "--lasers") == 0) {
            lasers = 1;
        } else if (strcmp(argv[i], "--log-trajectories") == 0) {
            show_human_logs = 1;
        } else if (strcmp(argv[i], "--frame-skip") == 0) {
            if (i + 1 < argc) {
                frame_skip = atoi(argv[i + 1]);
                i++;
                if (frame_skip <= 0) {
                    frame_skip = 1;
                }
            }
        } else if (strcmp(argv[i], "--zoom-in") == 0) {
            zoom_in = 1;
        } else if (strcmp(argv[i], "--view") == 0) {
            if (i + 1 < argc) {
                view_mode = argv[i + 1];
                i++;
                if (strcmp(view_mode, "both") != 0 && strcmp(view_mode, "topdown") != 0 &&
                    strcmp(view_mode, "agent") != 0) {
                    fprintf(stderr, "Error: --view must be 'both', 'topdown', or 'agent'\n");
                    return 1;
                }
            } else {
                fprintf(stderr, "Error: --view option requires a value (both/topdown/agent)\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--map-name") == 0) {
            if (i + 1 < argc) {
                map_name = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --map-name option requires a map file path\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--policy-name") == 0) {
            if (i + 1 < argc) {
                policy_name = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --policy-name option requires a policy file path\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--output-topdown") == 0) {
            if (i + 1 < argc) {
                output_topdown = argv[i + 1];
                i++;
            }
        } else if (strcmp(argv[i], "--output-agent") == 0) {
            if (i + 1 < argc) {
                output_agent = argv[i + 1];
                i++;
            }
        } else if (strcmp(argv[i], "--num-maps") == 0) {
            if (i + 1 < argc) {
                num_maps = atoi(argv[i + 1]);
                i++;
            }
        }
    }

    // performance_test();
    demo(map_name, policy_name, show_grid, obs_only, lasers, show_human_logs, frame_skip, view_mode, output_topdown,
         output_agent, num_maps, zoom_in);
    // test_drivenet();
    return 0;
}
