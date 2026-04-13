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
        observations[i] = i % 8;
    }

    int *actions = calloc(num_agents * num_actions, sizeof(int));

    // Weights* weights = load_weights("resources/drive/puffer_drive_weights.bin");
    Weights *weights = load_weights("puffer_drive_weights.bin");
    DriveNet *net = init_drivenet(weights, num_agents, CLASSIC, 1);

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

void demo(const char *map_name_arg, const char *policy_name_arg, int view_mode, int draw_traces) {
    // Read configuration from INI file
    env_init_config conf = {0};
    const char *ini_file = "pufferlib/config/ocean/drive.ini";
    if (ini_parse(ini_file, handler, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s. Cannot determine environment configuration.\n", ini_file);
        exit(1);
    }

    // Set different seed each time
    srand(time(NULL));

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

    AgentSpawnSettings spawn_settings = {
        .max_agents_in_sim = conf.max_agents_per_env,
        .min_w = conf.spawn_width_min,
        .max_w = conf.spawn_width_max,
        .min_l = conf.spawn_length_min,
        .max_l = conf.spawn_length_max,
        .h = conf.spawn_height,
    };

    Drive env = {
        .action_type = conf.action_type,
        .dynamics_model = conf.dynamics_model,
        .reward_vehicle_collision = conf.reward_vehicle_collision,
        .reward_offroad_collision = conf.reward_offroad_collision,
        .reward_lane_align = conf.reward_lane_align,
        .reward_lane_center = conf.reward_lane_center,
        .reward_goal = conf.reward_goal,
        .reward_goal_post_respawn = conf.reward_goal_post_respawn,
        .goal_radius = conf.goal_radius,
        .min_goal_speed = conf.min_goal_speed,
        .goal_behavior = conf.goal_behavior,
        .reward_randomization = conf.reward_randomization,
        .reward_conditioning = conf.reward_conditioning,
        .turn_off_normalization = conf.turn_off_normalization,
        .min_goal_distance = conf.min_goal_distance,
        .max_goal_distance = conf.max_goal_distance,
        .max_goal_speed = conf.max_goal_speed,
        .dt = conf.dt,
        .episode_length = conf.episode_length,
        .termination_mode = conf.termination_mode,
        .collision_behavior = conf.collision_behavior,
        .offroad_behavior = conf.offroad_behavior,
        .observation_window_size = conf.observation_window_size,
        .polyline_reduction_threshold = conf.polyline_reduction_threshold,
        .polyline_max_segment_length = conf.polyline_max_segment_length,
        .init_steps = conf.init_steps,
        .init_mode = conf.init_mode,
        .control_mode = conf.control_mode,
        .spawn_settings = spawn_settings,
        .reward_bounds =
            {
                {conf.reward_bound_goal_radius_min, conf.reward_bound_goal_radius_max},
                {conf.reward_bound_collision_min, conf.reward_bound_collision_max},
                {conf.reward_bound_offroad_min, conf.reward_bound_offroad_max},
                {conf.reward_bound_comfort_min, conf.reward_bound_comfort_max},
                {conf.reward_bound_lane_align_min, conf.reward_bound_lane_align_max},
                {conf.reward_bound_lane_center_min, conf.reward_bound_lane_center_max},
                {conf.reward_bound_velocity_min, conf.reward_bound_velocity_max},
                {conf.reward_bound_traffic_light_min, conf.reward_bound_traffic_light_max},
                {conf.reward_bound_center_bias_min, conf.reward_bound_center_bias_max},
                {conf.reward_bound_vel_align_min, conf.reward_bound_vel_align_max},
                {conf.reward_bound_overspeed_min, conf.reward_bound_overspeed_max},
                {conf.reward_bound_timestep_min, conf.reward_bound_timestep_max},
                {conf.reward_bound_reverse_min, conf.reward_bound_reverse_max},
                {conf.reward_bound_throttle_min, conf.reward_bound_throttle_max},
                {conf.reward_bound_steer_min, conf.reward_bound_steer_max},
                {conf.reward_bound_acc_min, conf.reward_bound_acc_max},
            },
        .map_name = "resources/drive/binaries/Town01/map_000.bin",
        .render_mode = RENDER_WINDOW,
        .partner_obs_radius = conf.partner_obs_radius,
    };

    if (conf.init_mode == INIT_VARIABLE_AGENT_NUMBER) {
        env.num_agents = conf.min_agents_per_env + rand() % (conf.max_agents_per_env - conf.min_agents_per_env + 1);
    }

    allocate(&env);
    if (env.active_agent_count == 0) {
        fprintf(stderr, "Error: No active agents found in map '%s' with init_mode=%d. Cannot run demo.\n", env.map_name,
                conf.init_mode);
        free_allocated(&env);
        return;
    }
    c_reset(&env);
    c_render(&env, view_mode, draw_traces);
    const char *weights_path = policy_name_arg ? policy_name_arg : "resources/drive/new_render_single_agent.bin";
    Weights *weights = load_weights(weights_path);
    if (!weights) {
        fprintf(stderr, "Error: Failed to load weights from '%s'\n", weights_path);
        free_allocated(&env);
        return;
    }
    DriveNet *net = init_drivenet(weights, env.active_agent_count, env.dynamics_model, env.reward_conditioning);

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
        c_render(&env, view_mode, draw_traces);
    }

    close_client(env.client);
    free_allocated(&env);
    free_drivenet(net);
    free(weights);
    return;
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
    const char *map_name = NULL;
    const char *policy_name = NULL;
    int view_mode = VIEW_MODE_SIM_STATE; // Default: full sim-state bird's-eye view
    int draw_traces = 1;                 // Default: show logged trajectories

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--map-name") == 0) {
            if (i + 1 < argc) {
                map_name = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --map-name requires a map file path\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--policy-name") == 0) {
            if (i + 1 < argc) {
                policy_name = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --policy-name requires a policy file path\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--view") == 0) {
            if (i + 1 < argc) {
                const char *v = argv[i + 1];
                i++;
                if (strcmp(v, "sim_state") == 0 || strcmp(v, "topdown") == 0) {
                    view_mode = VIEW_MODE_SIM_STATE;
                } else if (strcmp(v, "bev") == 0 || strcmp(v, "agent") == 0) {
                    view_mode = VIEW_MODE_BEV_AGENT_OBS;
                } else if (strcmp(v, "persp") == 0) {
                    view_mode = VIEW_MODE_AGENT_PERSP;
                } else {
                    fprintf(stderr, "Error: --view must be 'sim_state', 'bev', 'persp', or 'zoom_out'\n");
                    return 1;
                }
            } else {
                fprintf(stderr, "Error: --view requires a value (sim_state/bev/persp/zoom_out)\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--no-traces") == 0) {
            draw_traces = 0;
        }
    }

    // performance_test();
    demo(map_name, policy_name, view_mode, draw_traces);
    // test_drivenet();
    return 0;
}
