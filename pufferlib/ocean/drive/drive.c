#include "drive.h"
#include "drivenet.h"
#include <sys/stat.h>
#include <sys/types.h>
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

void create_directory_recursive(const char* path) {
    char tmp[256];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    
    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
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
int eval_gif(const char* map_name, 
             const char* policy_name, 
             int show_grid, 
             int obs_only, 
             int lasers, 
             int log_trajectories, 
             int frame_skip, 
             float goal_radius, 
             int population_play,
             int control_non_vehicles, 
             int init_steps, 
             int control_all_agents, 
             int policy_agents_per_env, 
             int deterministic_selection, 
             const char* view_mode, 
             const char* output_topdown, 
             const char* output_agent, 
             int num_maps, 
             int scenario_length_override) {

    char map_buffer[100];
    if (map_name == NULL) {
        srand(time(NULL));
        int random_map = rand() % num_maps;
        sprintf(map_buffer, "resources/drive/binaries/map_%03d.bin", random_map);
        map_name = map_buffer;
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

    FILE* policy_file = fopen(policy_name, "rb");
    if (policy_file == NULL) {
        RAISE_FILE_ERROR(policy_name);
    }
    fclose(policy_file);

    Drive env = {
        .dynamics_model = CLASSIC,
        .reward_vehicle_collision = -0.5f,
        .reward_offroad_collision = -0.2f,
        .reward_ade = -0.0f,
        .goal_radius = goal_radius,
        .map_name = (char*)map_name,
        .control_non_vehicles = control_non_vehicles,
        .init_steps = init_steps,
        .control_all_agents = control_all_agents,
        .policy_agents_per_env = policy_agents_per_env,
        .deterministic_agent_selection = deterministic_selection
    };
    env.scenario_length = (scenario_length_override > 0) ? scenario_length_override : TRAJECTORY_LENGTH_DEFAULT;
    allocate(&env);

    // Set which vehicle to focus on for obs mode
    env.human_agent_idx = 0;

    const char* topdown_path = output_topdown ? output_topdown : "resources/drive/output_topdown.mp4";
    const char* agent_path = output_agent ? output_agent : "resources/drive/output_agent.mp4";

    char dir_path[256];
    const char* last_slash = strrchr(topdown_path, '/');
    if (last_slash) {
        size_t dir_len = last_slash - topdown_path;
        strncpy(dir_path, topdown_path, dir_len);
        dir_path[dir_len] = '\0';
        create_directory_recursive(dir_path);
    }
    c_reset(&env);
    // Make client for rendering
    Client* client = (Client*)calloc(1, sizeof(Client));
    env.client = client;

    SetConfigFlags(FLAG_WINDOW_HIDDEN);

    SetTargetFPS(6000);

    float map_width = env.grid_map->bottom_right_x - env.grid_map->top_left_x;
    float map_height = env.grid_map->top_left_y - env.grid_map->bottom_right_y;

    printf("Map size: %.1fx%.1f\n", map_width, map_height);
    float scale = 6.0f; // Can be used to increase the video quality

    // Calculate video width and height; round to nearest even number
    int img_width = (int)roundf(map_width * scale / 2.0f) * 2;
    int img_height = (int)roundf(map_height * scale / 2.0f) * 2;
    InitWindow(img_width, img_height, "Puffer Drive");
    SetConfigFlags(FLAG_MSAA_4X_HINT);

    // Population play variables
    struct timespec ts;
    int ego_agent_id = 0;
    int num_co_players = 0;
    int max_obs = 7 + 63*7 + 200*7; // 7 self features, 63 objects with 7 features, 200 road points with 7 features (= 1848)

    if (population_play) {
        clock_gettime(CLOCK_REALTIME, &ts);
        srand(ts.tv_nsec);
        ego_agent_id = rand() % env.active_agent_count;
        num_co_players = env.active_agent_count - 1;
        
        printf("Population play enabled: ego_agent_id=%d, num_co_players=%d, total_agents=%d\n", 
               ego_agent_id, num_co_players, env.active_agent_count);
        
        for (int i = 0; i < env.active_agent_count; i++) {
            if (i != ego_agent_id) {
                env.entities[env.active_agent_indices[i]].is_ego = false;
                env.entities[env.active_agent_indices[i]].is_co_player = true;
            } else {
                env.entities[env.active_agent_indices[i]].is_ego = true;
                env.entities[env.active_agent_indices[i]].is_co_player = false;
            }
        }
        env.human_agent_idx = ego_agent_id;
    }

    // Load weights and initialize networks
    Weights* weights = NULL;
    Weights* co_player_policy_weights = NULL;
    DriveNet* net = NULL;
    DriveNet* ego_net = NULL;
    DriveNet* co_player_net = NULL;
    int* co_player_actions = NULL;
    int* ego_actions = NULL;
    float* co_player_obs = NULL;
    float* ego_agent_obs = NULL;

    if (population_play) {
        weights = load_weights(policy_name);
        ego_net = init_drivenet(weights, 1);

        co_player_policy_weights = load_weights("resources/drive/puffer_drive_co_player.bin");
        co_player_net = init_drivenet(co_player_policy_weights, num_co_players);

        co_player_actions = (int*)calloc((env.active_agent_count-1)*2, sizeof(int));
        ego_actions = (int*)calloc(1*2, sizeof(int));
        co_player_obs = (float*)calloc(num_co_players*max_obs, sizeof(float));
        ego_agent_obs = (float*)calloc(1*max_obs, sizeof(float));
        
        if (!co_player_actions || !ego_actions || !co_player_obs || !ego_agent_obs) {
            printf("ERROR: Failed to allocate memory for population play\n");
            return -1;
        }
        
        printf("Population play memory allocated successfully\n");
        printf("DEBUG: max_obs=%d, ego_agent_id=%d, offset=%d\n", 
               max_obs, ego_agent_id, ego_agent_id * max_obs);
        fflush(stdout);
    } else {
        weights = load_weights(policy_name);
        printf("Active agents in map: %d\n", env.active_agent_count);
        net = init_drivenet(weights, env.active_agent_count);
    }

    int frame_count = env.scenario_length > 0 ? env.scenario_length : TRAJECTORY_LENGTH_DEFAULT;
    int log_trajectory = log_trajectories;
    char filename_topdown[256];
    char filename_agent[256];

    if (output_topdown != NULL && output_agent != NULL) {
        strcpy(filename_topdown, output_topdown);
        strcpy(filename_agent, output_agent);
    } else {
        char policy_base[256];
        strcpy(policy_base, policy_name);
        *strrchr(policy_base, '.') = '\0';

        char map[256];
        strcpy(map, basename((char*)map_name));
        *strrchr(map, '.') = '\0';

        // Create gifs directory if it doesn't exist
        char gifs_dir[256];
        sprintf(gifs_dir, "%s/gifs", policy_base);
        mkdir(gifs_dir, 0755);

        sprintf(filename_topdown, "%s/gifs/%s_topdown.mp4", policy_base, map);
        sprintf(filename_agent, "%s/gifs/%s_agent.mp4", policy_base, map);
    }

    bool render_topdown = (strcmp(view_mode, "both") == 0 || strcmp(view_mode, "topdown") == 0);
    bool render_agent = (strcmp(view_mode, "both") == 0 || strcmp(view_mode, "agent") == 0);

    printf("Rendering: %s\n", view_mode);
    printf("DEBUG: Checking observation buffer - env.observations=%p\n", (void*)env.observations);
    if (env.observations == NULL) {
        printf("ERROR: Observations not allocated!\n");
        return -1;
    }
    printf("DEBUG: Observation buffer is valid\n");
    fflush(stdout);

    int rendered_frames = 0;
    double startTime = GetTime();

    VideoRecorder topdown_recorder, agent_recorder;

    if (render_topdown) {
        if (!OpenVideo(&topdown_recorder, topdown_path, img_width, img_height)) {
            CloseWindow();
            return -1;
        }
    }

    if (render_agent) {
        if (!OpenVideo(&agent_recorder, agent_path, img_width, img_height)) {
            if (render_topdown) CloseVideo(&topdown_recorder);
            CloseWindow();
            return -1;
        }
    }

    if (render_topdown) {
        printf("Recording topdown view...\n");
        printf("DEBUG: frame_count=%d, population_play=%d, ego_agent_id=%d\n", 
               frame_count, population_play, ego_agent_id);
        printf("DEBUG: active_agent_count=%d, max_obs=%d\n", env.active_agent_count, max_obs);
        fflush(stdout);
        for(int i = 0; i < frame_count; i++) {
            if (i % frame_skip == 0) {
                if (i == 0) {
                    printf("DEBUG: About to render frame 0\n");
                    printf("DEBUG: Calling renderTopDownView...\n");
                    fflush(stdout);
                }
                renderTopDownView(&env, client, map_height, 0, 0, 0, frame_count, NULL, log_trajectories, show_grid);
                if (i == 0) {
                    printf("DEBUG: Render complete, about to write frame\n");
                    fflush(stdout);
                }
                WriteFrame(&topdown_recorder, img_width, img_height);
                rendered_frames++;
                if (i == 0) {
                    printf("DEBUG: Frame written successfully\n");
                    fflush(stdout);
                }
            }
            int (*actions)[2] = (int(*)[2])env.actions;
            
            if (population_play) {
                if (i == 0) {
                    printf("DEBUG: Starting population play logic\n");
                    printf("DEBUG: ego_agent_id=%d, max_obs=%d, active_agent_count=%d\n", 
                           ego_agent_id, max_obs, env.active_agent_count);
                    printf("DEBUG: Expected total obs buffer size: %d floats (%zu bytes)\n",
                           env.active_agent_count * max_obs,
                           env.active_agent_count * max_obs * sizeof(float));
                    printf("DEBUG: env.observations pointer: %p\n", (void*)env.observations);
                    printf("DEBUG: ego_agent_obs pointer: %p\n", (void*)ego_agent_obs);
                    printf("DEBUG: co_player_obs pointer: %p\n", (void*)co_player_obs);
                    if (env.observations == NULL) {
                        printf("ERROR: env.observations is NULL!\n");
                        return -1;
                    }
                    if (ego_agent_obs == NULL) {
                        printf("ERROR: ego_agent_obs is NULL!\n");
                        return -1;
                    }
                    
                    // Check if ego_agent_id is within bounds
                    if (ego_agent_id >= env.active_agent_count) {
                        printf("ERROR: ego_agent_id (%d) >= active_agent_count (%d)!\n",
                               ego_agent_id, env.active_agent_count);
                        return -1;
                    }
                    
                    printf("DEBUG: About to copy ego observation (offset=%d, size=%zu bytes)\n", 
                           ego_agent_id * max_obs, max_obs * sizeof(float));
                    printf("DEBUG: Source address: %p\n", (void*)&env.observations[ego_agent_id * max_obs]);
                    printf("DEBUG: Attempting to read first observation value...\n");
                    fflush(stdout);
                    float test_val = env.observations[ego_agent_id * max_obs];
                    printf("DEBUG: First obs value: %f\n", test_val);
                    fflush(stdout);
                }
                // Copy ego agent observation
                printf("DEBUG: About to memcpy...\n");
                fflush(stdout);
                memcpy(ego_agent_obs, &env.observations[ego_agent_id * max_obs], max_obs * sizeof(float));
                printf("DEBUG: memcpy complete\n");
                fflush(stdout);

                int co_obs_offset = 0;
                for (int j = 0; j < env.active_agent_count; j++) {
                    if (j == ego_agent_id) continue;
                    memcpy(&co_player_obs[co_obs_offset],
                        &env.observations[j * max_obs],
                        max_obs * sizeof(float));
                    co_obs_offset += max_obs;
                }

                // Get actions from both networks
                forward(ego_net, ego_agent_obs, ego_actions);
                forward(co_player_net, co_player_obs, co_player_actions);

                // Assign ego action
                actions[ego_agent_id][0] = ego_actions[0];
                actions[ego_agent_id][1] = ego_actions[1];

                // Assign co-player actions with correct indexing
                int co_player_idx = 0;
                for (int j = 0; j < env.active_agent_count; j++) {
                    if (j == ego_agent_id) continue;
                    actions[j][0] = co_player_actions[co_player_idx * 2];
                    actions[j][1] = co_player_actions[co_player_idx * 2 + 1];
                    co_player_idx++;
                }
            } else {
                forward(net, env.observations, (int*)env.actions);
            }
            c_step(&env);
        }
    }

    if (render_agent) {
        c_reset(&env);
        printf("Recording agent view...\n");
        for(int i = 0; i < frame_count; i++) {
            if (i % frame_skip == 0) {
                renderAgentView(&env, client, map_height, obs_only, lasers, show_grid);
                WriteFrame(&agent_recorder, img_width, img_height);
                rendered_frames++;
            }
            int (*actions)[2] = (int(*)[2])env.actions;
            
            if (population_play) {
                memcpy(ego_agent_obs, &env.observations[ego_agent_id * max_obs], max_obs * sizeof(float));

                int co_obs_offset = 0;
                for (int j = 0; j < env.active_agent_count; j++) {
                    if (j == ego_agent_id) continue;
                    memcpy(&co_player_obs[co_obs_offset],
                        &env.observations[j * max_obs],
                        max_obs * sizeof(float));
                    co_obs_offset += max_obs;
                }

                forward(ego_net, ego_agent_obs, ego_actions);
                forward(co_player_net, co_player_obs, co_player_actions);

                actions[ego_agent_id][0] = ego_actions[0];
                actions[ego_agent_id][1] = ego_actions[1];

                int co_player_idx = 0;
                for (int j = 0; j < env.active_agent_count; j++) {
                    if (j == ego_agent_id) continue;
                    actions[j][0] = co_player_actions[co_player_idx * 2];
                    actions[j][1] = co_player_actions[co_player_idx * 2 + 1];
                    co_player_idx++;
                }
            } else {
                forward(net, env.observations, (int*)env.actions);
            }
            c_step(&env);
        }
    }

    double endTime = GetTime();
    double elapsedTime = endTime - startTime;
    double writeFPS = (elapsedTime > 0) ? rendered_frames / elapsedTime : 0;

    printf("Wrote %d frames in %.2f seconds (%.2f FPS) to %s \n",
           rendered_frames, elapsedTime, writeFPS, filename_topdown);

    if (render_topdown) {
        CloseVideo(&topdown_recorder);
    }
    if (render_agent) {
        CloseVideo(&agent_recorder);
    }
    CloseWindow();

    // Free networks properly based on mode
    if (population_play) {
        if (ego_net) free_drivenet(ego_net);
        if (co_player_net) free_drivenet(co_player_net);
        if (co_player_actions) free(co_player_actions);
        if (ego_actions) free(ego_actions);
        if (co_player_obs) free(co_player_obs);
        if (ego_agent_obs) free(ego_agent_obs);
        if (weights) free(weights);
        if (co_player_policy_weights) free(co_player_policy_weights);
    } else {
        if (net) free_drivenet(net);
        if (weights) free(weights);
    }

    // Clean up resources
    free(client);
    free_allocated(&env);
    
    return 0;
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

int main(int argc, char* argv[]) {
    int show_grid = 0;
    int obs_only = 0;
    int lasers = 0;
    int log_trajectories = 1;
    int frame_skip = 1;
    float goal_radius = 2.0f;
    int init_steps = 0;
    const char* map_name = NULL;
    int population_play = 1;
    int control_all_agents = 0;
    int deterministic_selection = 0;
    int policy_agents_per_env = -1;
    int control_non_vehicles = 0;

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
        } else if (strcmp(argv[i], "--init-steps") == 0) {
            if (i + 1 < argc) {
                init_steps = atoi(argv[i + 1]);
                i++;
                if (init_steps < 0) {
                    init_steps = 0; // Ensure non-negative
                }
                if (init_steps > TRAJECTORY_LENGTH-1) {
                    init_steps = TRAJECTORY_LENGTH-1; // Upper bound
                }
            }
        } else if (strcmp(argv[i], "--frame-skip") == 0) {
            if (i + 1 < argc) {
                frame_skip = atoi(argv[i + 1]);
                i++; // Skip the next argument since we consumed it
                if (frame_skip <= 0) {
                    frame_skip = 1; // Ensure valid value
                }
            }
        } else if (strcmp(argv[i], "--control-non-vehicles") == 0) {
            control_non_vehicles = 1;
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
        }else if (strcmp(argv[i], "--population-play") == 0) {
            population_play = 1;
        } else if (strcmp(argv[i], "--pure-self-play") == 0) {
            control_all_agents = 1;
        } else if (strcmp(argv[i], "--num-policy-controlled-agents") == 0) {
            if (i + 1 < argc) {
                policy_agents_per_env = atoi(argv[i + 1]);
                i++;
            }
        } else if (strcmp(argv[i], "--deterministic-selection") == 0) {
            deterministic_selection = 1;
        }
    }

    eval_gif(map_name, show_grid, obs_only, lasers, log_trajectories, frame_skip,
             goal_radius,population_play, control_non_vehicles, init_steps,
             control_all_agents, policy_agents_per_env, deterministic_selection);
    //demo();
    //performance_test();
    demo();
    //test_drivenet();
    return 0;
}
