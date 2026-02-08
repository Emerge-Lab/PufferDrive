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
        observations[i] = i % 8;
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

void test_load_map_binary() {
    printf("=== Testing load_map_binary ===\n");

    // Initialize environment
    Drive *env = (Drive *)calloc(1, sizeof(Drive));

    // Test with a known map file
    const char *test_map = "resources/drive/binaries/NewMaps/map_000.bin";

    printf("Loading map: %s\n", test_map);
    load_map_binary(test_map, env);

    // Validation checks
    printf("\n--- Map Load Results ---\n");
    printf("SDC track index: %d\n", env->sdc_track_index);
    printf("Num tracks to predict: %d\n", env->num_tracks_to_predict);
    printf("Num objects (agents): %d\n", env->num_objects);
    printf("Num roads: %d\n", env->num_roads);

    // Sanity checks
    int passed = 1;

    if (env->num_objects <= 0) {
        printf("❌ FAIL: num_objects is %d (expected > 0)\n", env->num_objects);
        passed = 0;
    } else {
        printf("✅ PASS: num_objects = %d\n", env->num_objects);
    }

    if (env->num_roads <= 0) {
        printf("❌ FAIL: num_roads is %d (expected > 0)\n", env->num_roads);
        passed = 0;
    } else {
        printf("✅ PASS: num_roads = %d\n", env->num_roads);
    }

    // Check first agent
    if (env->agents != NULL && env->num_objects > 0) {
        Agent *first_agent = &env->agents[0];
        printf("\n--- First Agent Details ---\n");
        printf("ID: %d\n", first_agent->id);
        printf("Type: %d\n", first_agent->type);
        printf("Trajectory length: %d\n", first_agent->trajectory_length);
        printf("Width: %.2f, Length: %.2f, Height: %.2f\n", first_agent->sim_width, first_agent->sim_length,
               first_agent->sim_height);

        if (first_agent->trajectory_length > 0) {
            printf("First position: (%.2f, %.2f, %.2f)\n", first_agent->log_trajectory_x[0],
                   first_agent->log_trajectory_y[0], first_agent->log_trajectory_z[0]);
            printf("✅ PASS: Agent trajectory data loaded\n");
        } else {
            printf("❌ FAIL: Agent trajectory_length is 0\n");
            passed = 0;
        }
    }

    // Check first road
    if (env->road_elements != NULL && env->num_roads > 0) {
        RoadMapElement *first_road = &env->road_elements[0];
        printf("\n--- First Road Details ---\n");
        printf("ID: %d\n", first_road->id);
        printf("Type: %d\n", first_road->type);
        printf("Segment length: %d\n", first_road->segment_length);

        if (first_road->segment_length > 0) {
            printf("First point: (%.2f, %.2f, %.2f)\n", first_road->x[0], first_road->y[0], first_road->z[0]);
            printf("✅ PASS: Road geometry data loaded\n");
        } else {
            printf("❌ FAIL: Road segment_length is 0\n");
            passed = 0;
        }
    }

    printf("\n=== Test Result: %s ===\n\n", passed ? "✅ ALL PASSED" : "❌ SOME FAILED");

    // Cleanup
    // ... (add proper cleanup if needed)
}

void demo() {
    // Read configuration from INI file
    env_init_config conf = {0};
    const char *ini_file = "pufferlib/config/ocean/drive.ini";
    if (ini_parse(ini_file, handler, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s. Cannot determine environment configuration.\n", ini_file);
        exit(1);
    }

    // Note: Use below hardcoded settings for 2.0 demo purposes. Since the policy was
    // trained with these exact settings, changing them may lead to
    // weird behavior.
    // Drive env = {
    //     .human_agent_idx = 0,
    //     .action_type = 0,          // Discrete
    //     .dynamics_model = CLASSIC, // Classic dynamics
    //     .reward_vehicle_collision = -1.0f,
    //     .reward_offroad_collision = -1.0f,
    //     .reward_goal = 1.0f,
    //     .reward_goal_post_respawn = 0.25f,
    //     .goal_radius = 2.0f,
    //     .goal_behavior = 1,
    //     .goal_target_distance = 30.0f,
    //     .goal_speed = 10.0f,
    //     .dt = 0.1f,
    //     .episode_length = 300,
    //     .termination_mode = 0,
    //     .collision_behavior = 0,
    //     .offroad_behavior = 0,
    //     .init_steps = 0,
    //     .init_mode = 0,
    //     .control_mode = 0,
    //     .map_name = "resources/drive/map_town_02_carla.bin",
    // };

    Drive env = {
        .human_agent_idx = 0,
        .action_type = 0, // Demo doesn't support continuous action space
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
        .map_name = "resources/drive/binaries/carla/carla_3D/map_001.bin",
    };
    allocate(&env);
    c_reset(&env);
    c_render(&env);
    Weights *weights = load_weights("resources/drive/puffer_drive_weights.bin");
    DriveNet *net = init_drivenet(weights, env.active_agent_count, env.dynamics_model);

    int accel_delta = 1;
    int steer_delta = 2;
    while (!WindowShouldClose()) {
        int *actions = (int *)env.actions; // Single integer per agent

        if (!IsKeyDown(KEY_LEFT_SHIFT)) {
            forward(net, env.observations, actions);
        } else {
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

int main() {
    test_load_map_binary();
    // performance_test();
    demo();
    // test_drivenet();
    return 0;
}
