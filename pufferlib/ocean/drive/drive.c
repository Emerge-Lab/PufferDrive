#include "drive.h"
#include "drivenet.h"
#include <string.h>
#include "../env_config.h"

// Use this test if the network changes to ensure that the forward pass
// matches the torch implementation to the 3rd or ideally 4th decimal place
void test_drivenet() {
    int num_obs = 1848;
    int num_actions = 2;
    int num_agents = 4;

    float* observations = calloc(num_agents*num_obs, sizeof(float));
    for (int i = 0; i < num_obs*num_agents; i++) {
        observations[i] = i % 7;
    }

    int* actions = calloc(num_agents*num_actions, sizeof(int));

    //Weights* weights = load_weights("resources/drive/puffer_drive_weights.bin");
    Weights* weights = load_weights("puffer_drive_weights.bin");
    DriveNet* net = init_drivenet(weights, num_agents);

    forward(net, observations, actions);
    for (int i = 0; i < num_agents*num_actions; i++) {
        printf("idx: %d, action: %d, logits:", i, actions[i]);
        for (int j = 0; j < num_actions; j++) {
            printf(" %.6f", net->actor->output[i*num_actions + j]);
        }
        printf("\n");
    }
    free_drivenet(net);
    free(weights);
}

void demo() {
    // Read configuration from INI file
    env_init_config conf = {0};
    const char* ini_file = "pufferlib/config/ocean/drive.ini";
    if(ini_parse(ini_file, handler, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s. Cannot determine environment configuration.\n", ini_file);
        exit(1);
    }

    Drive env = {
        .human_agent_idx = 0,
        .dynamics_model = conf.dynamics_model,
        .reward_vehicle_collision = conf.reward_vehicle_collision,
        .reward_offroad_collision = conf.reward_offroad_collision,
        .reward_ade = conf.reward_ade,
        .goal_radius = conf.goal_radius,
        .dt = conf.dt,
	    .map_name = "resources/drive/binaries/map_000.bin",
        .control_non_vehicles = conf.control_non_vehicles,
        .init_steps = conf.init_steps,
        .control_all_agents = conf.control_all_agents,
        .policy_agents_per_env = conf.num_policy_controlled_agents,
        .deterministic_agent_selection = conf.deterministic_agent_selection,
    };
    allocate(&env);
    c_reset(&env);
    c_render(&env);
    Weights* weights = load_weights("resources/drive/puffer_drive_weights.bin");
    DriveNet* net = init_drivenet(weights, env.active_agent_count, env.dynamics_model);
    //Client* client = make_client(&env);
    int accel_delta = 2;
    int steer_delta = 4;
    while (!WindowShouldClose()) {
        // Handle camera controls
        int (*actions)[2] = (int(*)[2])env.actions;
        forward(net, env.observations, env.actions);
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            actions[env.human_agent_idx][0] = 3;
            actions[env.human_agent_idx][1] = 6;
            if(IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)){
                actions[env.human_agent_idx][0] += accel_delta;
                // Cap acceleration to maximum of 6
                if(actions[env.human_agent_idx][0] > 6) {
                    actions[env.human_agent_idx][0] = 6;
                }
            }
            if(IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)){
                actions[env.human_agent_idx][0] -= accel_delta;
                // Cap acceleration to minimum of 0
                if(actions[env.human_agent_idx][0] < 0) {
                    actions[env.human_agent_idx][0] = 0;
                }
            }
            if(IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)){
                actions[env.human_agent_idx][1] += steer_delta;
                // Cap steering to minimum of 0
                if(actions[env.human_agent_idx][1] < 0) {
                    actions[env.human_agent_idx][1] = 0;
                }
            }
            if(IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)){
                actions[env.human_agent_idx][1] -= steer_delta;
                // Cap steering to maximum of 12
                if(actions[env.human_agent_idx][1] > 12) {
                    actions[env.human_agent_idx][1] = 12;
                }
            }
            if(IsKeyPressed(KEY_TAB)){
                env.human_agent_idx = (env.human_agent_idx + 1) % env.active_agent_count;
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
    const char* ini_file = "pufferlib/config/ocean/drive.ini";
    if(ini_parse(ini_file, handler, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s. Cannot determine environment configuration.\n", ini_file);
        exit(1);
    }

    long test_time = 10;
    Drive env = {
        .human_agent_idx = 0,
        .dynamics_model = conf.dynamics_model,
        .reward_vehicle_collision = conf.reward_vehicle_collision,
        .reward_offroad_collision = conf.reward_offroad_collision,
        .reward_ade = conf.reward_ade,
        .goal_radius = conf.goal_radius,
        .dt = conf.dt,
	    .map_name = "resources/drive/binaries/map_000.bin",
        .control_non_vehicles = conf.control_non_vehicles,
        .init_steps = conf.init_steps,
        .control_all_agents = conf.control_all_agents,
        .policy_agents_per_env = conf.num_policy_controlled_agents,
        .deterministic_agent_selection = conf.deterministic_agent_selection,
    };
    clock_t start_time, end_time;
    double cpu_time_used;
    start_time = clock();
    allocate(&env);
    c_reset(&env);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Init time: %f\n", cpu_time_used);

    long start = time(NULL);
    int i = 0;
    int (*actions)[2] = (int(*)[2])env.actions;

    while (time(NULL) - start < test_time) {
        // Set random actions for all agents
        for(int j = 0; j < env.active_agent_count; j++) {
            int accel = rand() % 7;
            int steer = rand() % 13;
            actions[j][0] = accel;  // -1, 0, or 1
            actions[j][1] = steer;  // Random steering
        }

        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (i*env.active_agent_count) / (end - start));
    free_allocated(&env);
}

int main() {
    //performance_test();
    demo();
    //test_drivenet();
    return 0;
}
<<<<<<< HEAD
=======
void demo() {

    Drive env = {
        .dynamics_model = CLASSIC,
        .human_agent_idx = 0,
        .reward_vehicle_collision = -0.1f,
        .reward_offroad_collision = -0.1f,
        .reward_ade = -0.0f,
        .goal_radius = 2.0f,
	    .map_name = "resources/drive/binaries/map_000.bin",
        .spawn_immunity_timer = 50,
    };
    allocate(&env);
    c_reset(&env);
    c_render(&env);
    Weights* weights = load_weights("resources/drive/puffer_drive_weights.bin", 595925);
    DriveNet* net = init_drivenet(weights, env.active_agent_count);
    //Client* client = make_client(&env);
    int accel_delta = 2;
    int steer_delta = 4;
    while (!WindowShouldClose()) {
        // Handle camera controls
        int (*actions)[2] = (int(*)[2])env.actions;
        forward(net, env.observations, env.actions);
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            actions[env.human_agent_idx][0] = 3;
            actions[env.human_agent_idx][1] = 6;
            if(IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)){
                actions[env.human_agent_idx][0] += accel_delta;
                // Cap acceleration to maximum of 6
                if(actions[env.human_agent_idx][0] > 6) {
                    actions[env.human_agent_idx][0] = 6;
                }
            }
            if(IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)){
                actions[env.human_agent_idx][0] -= accel_delta;
                // Cap acceleration to minimum of 0
                if(actions[env.human_agent_idx][0] < 0) {
                    actions[env.human_agent_idx][0] = 0;
                }
            }
            if(IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)){
                actions[env.human_agent_idx][1] += steer_delta;
                // Cap steering to minimum of 0
                if(actions[env.human_agent_idx][1] < 0) {
                    actions[env.human_agent_idx][1] = 0;
                }
            }
            if(IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)){
                actions[env.human_agent_idx][1] -= steer_delta;
                // Cap steering to maximum of 12
                if(actions[env.human_agent_idx][1] > 12) {
                    actions[env.human_agent_idx][1] = 12;
                }
            }
            if(IsKeyPressed(KEY_TAB)){
                env.human_agent_idx = (env.human_agent_idx + 1) % env.active_agent_count;
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


static int run_cmd(const char *cmd) {
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "[ffmpeg] command failed (%d): %s\n", rc, cmd);
    }
    return rc;
}

// Make a high-quality GIF from numbered PNG frames like frame_000.png
static int make_gif_from_frames(const char *pattern, int fps,
                                const char *palette_path,
                                const char *out_gif) {
    char cmd[1024];

    // 1) Generate palette (no quotes needed for simple filter)
    //    NOTE: if your frames start at 000, you don't need -start_number.
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -y -framerate %d -i %s -vf palettegen %s",
             fps, pattern, palette_path);
    if (run_cmd(cmd) != 0) return -1;

    // 2) Use palette to encode the GIF
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -y -framerate %d -i %s -i %s -lavfi paletteuse -loop 0 %s",
             fps, pattern, palette_path, out_gif);
    if (run_cmd(cmd) != 0) return -1;

    return 0;
}

int eval_gif(const char* map_name, int show_grid, int obs_only, int lasers, int log_trajectories, int frame_skip, float goal_radius) {

    // Use default if no map provided
    if (map_name == NULL) {
        map_name = "resources/drive/binaries/map_000.bin";
    }

    if (frame_skip <= 0) {
        frame_skip = 1;  // Default: render every frame
    }

    // Check if map file exists
    FILE* map_file = fopen(map_name, "rb");
    if (map_file == NULL) {
        RAISE_FILE_ERROR(map_name);
    }
    fclose(map_file);

    // Make env
    Drive env = {
        .dynamics_model = CLASSIC,
        .reward_vehicle_collision = -0.1f,
        .reward_offroad_collision = -0.1f,
        .reward_ade = -0.0f,
	.collision_behaviour = 0,
	.offroad_behaviour = 0,
        .goal_radius = goal_radius,
	    .map_name = map_name,
        .spawn_immunity_timer = 50
    };
    allocate(&env);

    // Set which vehicle to focus on for obs mode
    env.human_agent_idx = 0;

    c_reset(&env);
    // Make client for rendering
    Client* client = (Client*)calloc(1, sizeof(Client));
    env.client = client;

    SetConfigFlags(FLAG_WINDOW_HIDDEN);

    SetTargetFPS(6000);

    float map_width = env.map_corners[2] - env.map_corners[0];
    float map_height = env.map_corners[3] - env.map_corners[1];

    printf("Map size: %.1fx%.1f\n", map_width, map_height);
    float scale = 6.0f; // Can be used to increase the video quality

    // Calculate video width and height; round to nearest even number
    int img_width = (int)roundf(map_width * scale / 2.0f) * 2;
    int img_height = (int)roundf(map_height * scale / 2.0f) * 2;
    InitWindow(img_width, img_height, "Puffer Drive");
    SetConfigFlags(FLAG_MSAA_4X_HINT);

    // Load cpt into network
    Weights* weights = load_weights("resources/drive/puffer_drive_weights.bin", 595925);
    DriveNet* net = init_drivenet(weights, env.active_agent_count);

    int frame_count = 91;
    char filename[256];
    int log_trajectory = log_trajectories;

    printf("Starting video recording...\n");

    // Create video recorders
    VideoRecorder topdown_recorder;
    if (!OpenVideo(&topdown_recorder, "resources/drive/output_topdown.mp4", img_width, img_height)) {
        CloseWindow();
        return -1;
    }

    int rendered_frames = 0;
    double startTime = GetTime();

    // Generate top-down view video
    printf("Recording top-down view...\n");

    for(int i = 0; i < frame_count; i++) {
        // Only render every frame_skip frames
        if (i % frame_skip == 0) {
            renderTopDownView(&env, client, map_height, 0, 0, 0, frame_count, NULL, log_trajectories, show_grid);
            WriteFrame(&topdown_recorder, img_width, img_height);
            rendered_frames++;
        }

            int (*actions)[2] = (int(*)[2])env.actions;
            forward(net, env.observations, env.actions);
            c_step(&env);
    }

    // Reset environment for agent view
    c_reset(&env);
    CloseVideo(&topdown_recorder);

    VideoRecorder agent_recorder;
    if (!OpenVideo(&agent_recorder, "resources/drive/output_agent.mp4", img_width, img_height)) {
        CloseWindow();
        return -1;
    }

    for(int i = 0; i < frame_count; i++) {

        if (i % frame_skip == 0) {
            renderAgentView(&env, client, map_height, obs_only, lasers, show_grid);
            WriteFrame(&agent_recorder, img_width, img_height);
            rendered_frames++;
        }

            int (*actions)[2] = (int(*)[2])env.actions;
            forward(net, env.observations, env.actions);
            c_step(&env);
    }

    double endTime = GetTime();
    double elapsedTime = endTime - startTime;
    double writeFPS = (elapsedTime > 0) ? rendered_frames / elapsedTime : 0;

    printf("Wrote %d frames in %.2f seconds (%.2f FPS)\n",
           rendered_frames, elapsedTime, writeFPS);

    // Close video recorders
    CloseVideo(&agent_recorder);
    CloseWindow();

    // Clean up resources
    free(client);
    free_allocated(&env);
    free_drivenet(net);
    free(weights);
    return 0;
}

void performance_test() {
    long test_time = 10;
    Drive env = {
        .dynamics_model = CLASSIC,
        .human_agent_idx = 0,
	    .map_name = "resources/drive/binaries/map_000.bin"
    };
    clock_t start_time, end_time;
    double cpu_time_used;
    start_time = clock();
    allocate(&env);
    c_reset(&env);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Init time: %f\n", cpu_time_used);

    long start = time(NULL);
    int i = 0;
    int (*actions)[2] = (int(*)[2])env.actions;

    while (time(NULL) - start < test_time) {
        // Set random actions for all agents
        for(int j = 0; j < env.active_agent_count; j++) {
            int accel = rand() % 7;
            int steer = rand() % 13;
            actions[j][0] = accel;  // -1, 0, or 1
            actions[j][1] = steer;  // Random steering
        }

        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (i*env.active_agent_count) / (end - start));
    free_allocated(&env);
}

int main(int argc, char* argv[]) {
    int show_grid = 0;
    int obs_only = 0;
    int lasers = 0;
    int log_trajectories = 1;
    int frame_skip = 1;
    float goal_radius = 2.0f;
    const char* map_name = NULL;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--show-grid") == 0) {
            show_grid = 1;
        } else if (strcmp(argv[i], "--obs-only") == 0) {
            obs_only = 1;
        } else if (strcmp(argv[i], "--lasers") == 0) {
            lasers = 1;
        } else if (strcmp(argv[i], "--log-trajectories") == 0) {
            log_trajectories = 1;
        } else if (strcmp(argv[i], "--frame-skip") == 0) {
            if (i + 1 < argc) {
                frame_skip = atoi(argv[i + 1]);
                i++; // Skip the next argument since we consumed it
                if (frame_skip <= 0) {
                    frame_skip = 1; // Ensure valid value
                }
            }
        } else if (strcmp(argv[i], "--goal-radius") == 0) {
            if (i + 1 < argc) {
                goal_radius = atof(argv[i + 1]);
                i++;
                if (goal_radius <= 0) {
                    goal_radius = 2.0f; // Ensure valid value
                }
            }
        } else if (strcmp(argv[i], "--map-name") == 0) {
            // Check if there's a next argument for the map path
            if (i + 1 < argc) {
                map_name = argv[i + 1];
                i++; // Skip the next argument since we used it as map path
            } else {
                fprintf(stderr, "Error: --map-name option requires a map file path\n");
                return 1;
            }
        }
    }

    eval_gif(map_name, show_grid, obs_only, lasers, log_trajectories, frame_skip, goal_radius);
    //demo();
    //performance_test();
    return 0;
}
>>>>>>> e423d459 (Initial commit)
