#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <math.h>
#include <raylib.h>
#include "rlgl.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "error.h"
#include "drivenet.h"
#include "libgen.h"
#include "../env_config.h"
#define TRAJECTORY_LENGTH_DEFAULT 91

typedef struct {
    int pipefd[2];
    pid_t pid;
} VideoRecorder;

bool OpenVideo(VideoRecorder *recorder, const char *output_filename, int width, int height) {
    if (pipe(recorder->pipefd) == -1) {
        fprintf(stderr, "Failed to create pipe\n");
        return false;
    }

    recorder->pid = fork();
    if (recorder->pid == -1) {
        fprintf(stderr, "Failed to fork\n");
        return false;
    }

    char size_str[64];
    snprintf(size_str, sizeof(size_str), "%dx%d", width, height);

    if (recorder->pid == 0) { // Child process: run ffmpeg
        close(recorder->pipefd[1]);
        dup2(recorder->pipefd[0], STDIN_FILENO);
        close(recorder->pipefd[0]);
        // Close all other file descriptors to prevent leaks
        for (int fd = 3; fd < 256; fd++) {
            close(fd);
        }
        execlp("ffmpeg", "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", size_str, "-r", "30", "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "23", "-loglevel", "error",
               output_filename, NULL);
        TraceLog(LOG_ERROR, "Failed to launch ffmpeg");
        return false;
    }

    close(recorder->pipefd[0]); // Close read end in parent
    return true;
}

void WriteFrame(VideoRecorder *recorder, int width, int height) {
    unsigned char *screen_data = rlReadScreenPixels(width, height);
    write(recorder->pipefd[1], screen_data, width * height * 4 * sizeof(*screen_data));
    RL_FREE(screen_data);
}

void CloseVideo(VideoRecorder *recorder) {
    close(recorder->pipefd[1]);
    waitpid(recorder->pid, NULL, 0);
}

void renderTopDownView(Drive *env, Client *client, int map_height, int obs, int lasers, int trajectories,
                       int frame_count, float *path, int log_trajectories, int show_grid, int img_width, int img_height,
                       int zoom_in) {
    BeginDrawing();

    // Top-down orthographic camera
    Camera3D camera = {0};

    if (zoom_in) {                                       // Zoom in on part of the map
        camera.position = (Vector3){0.0f, 0.0f, 500.0f}; // above the scene
        camera.target = (Vector3){0.0f, 0.0f, 0.0f};     // look at origin
        camera.fovy = map_height;
    } else { // Show full map
        camera.position = (Vector3){env->grid_map->top_left_x, env->grid_map->bottom_right_y, 500.0f};
        camera.target = (Vector3){env->grid_map->top_left_x, env->grid_map->bottom_right_y, 0.0f};
        camera.fovy = 2 * map_height;
    }

    camera.up = (Vector3){0.0f, -1.0f, 0.0f};
    camera.projection = CAMERA_ORTHOGRAPHIC;

    client->width = img_width;
    client->height = img_height;

    Color road = (Color){35, 35, 37, 255};
    ClearBackground(road);
    BeginMode3D(camera);
    rlEnableDepthTest();

    // Draw human replay trajectories if enabled
    if (log_trajectories) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int idx = env->active_agent_indices[i];
            Vector3 prev_point = {0};
            bool has_prev = false;

            for (int j = 0; j < env->entities[idx].array_size; j++) {
                float x = env->entities[idx].traj_x[j];
                float y = env->entities[idx].traj_y[j];
                float valid = env->entities[idx].traj_valid[j];

                if (!valid) {
                    has_prev = false;
                    continue;
                }

                Vector3 curr_point = {x, y, 0.5f};

                if (has_prev) {
                    DrawLine3D(prev_point, curr_point, Fade(LIGHTGREEN, 0.6f));
                }

                prev_point = curr_point;
                has_prev = true;
            }
        }
    }

    // Draw agent trajs
    if (trajectories) {
        for (int i = 0; i < frame_count; i++) {
            DrawSphere((Vector3){path[i * 2], path[i * 2 + 1], 0.8f}, 0.5f, YELLOW);
        }
    }

    // Draw scene
    draw_scene(env, client, 1, obs, lasers, show_grid);
    EndMode3D();
    EndDrawing();
}

void renderAgentView(Drive *env, Client *client, int map_height, int obs_only, int lasers, int show_grid) {
    // Agent perspective camera following the selected agent
    int agent_idx = env->active_agent_indices[env->human_agent_idx];
    Entity *agent = &env->entities[agent_idx];

    BeginDrawing();

    Camera3D camera = {0};
    // Position camera behind and above the agent
    camera.position =
        (Vector3){agent->x - (25.0f * cosf(agent->heading)), agent->y - (25.0f * sinf(agent->heading)), 15.0f};
    camera.target = (Vector3){agent->x + 40.0f * cosf(agent->heading), agent->y + 40.0f * sinf(agent->heading), 1.0f};
    camera.up = (Vector3){0.0f, 0.0f, 1.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    Color road = (Color){35, 35, 37, 255};

    ClearBackground(road);
    BeginMode3D(camera);
    rlEnableDepthTest();
    draw_scene(env, client, 0, obs_only, lasers, show_grid); // mode=0 for agent view
    EndMode3D();
    EndDrawing();
}

static int run_cmd(const char *cmd) {
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "[ffmpeg] command failed (%d): %s\n", rc, cmd);
    }
    return rc;
}

// Make a high-quality GIF from numbered PNG frames like frame_000.png
static int make_gif_from_frames(const char *pattern, int fps, const char *palette_path, const char *out_gif) {
    char cmd[1024];

    // 1) Generate palette (no quotes needed for simple filter)
    //    NOTE: if your frames start at 000, you don't need -start_number.
    snprintf(cmd, sizeof(cmd), "ffmpeg -y -framerate %d -i %s -vf palettegen %s", fps, pattern, palette_path);
    if (run_cmd(cmd) != 0)
        return -1;

    // 2) Use palette to encode the GIF
    snprintf(cmd, sizeof(cmd), "ffmpeg -y -framerate %d -i %s -i %s -lavfi paletteuse -loop 0 %s", fps, pattern,
             palette_path, out_gif);
    if (run_cmd(cmd) != 0)
        return -1;

    return 0;
}

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

    // Read configuration from INI file
    env_init_config conf = {0};
    const char *ini_file = "pufferlib/config/ocean/drive.ini";
    if (ini_parse(ini_file, handler, &conf) < 0) {
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
        .init_steps = conf.init_steps,
        .collision_behavior = conf.collision_behavior,
        .offroad_behavior = conf.offroad_behavior,
        .goal_behavior = conf.goal_behavior,
        .goal_target_distance = conf.goal_target_distance,
        .action_type = conf.action_type,
        .episode_length = conf.episode_length,
        .termination_mode = conf.termination_mode,
        .control_mode = 0, // Vehicles
        .init_mode = 0,    // Init all valid
    };
    allocate(&env);
    c_reset(&env);
    c_render(&env);
    Weights *weights = load_weights("resources/drive/puffer_drive_weights.bin");
    DriveNet *net = init_drivenet(weights, env.active_agent_count, env.dynamics_model);

    int accel_delta = 2;
    int steer_delta = 4;
    while (!WindowShouldClose()) {
        // Handle camera controls
        int (*actions)[2] = (int (*)[2])env.actions;
        forward(net, env.observations, env.actions);
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            actions[env.human_agent_idx][0] = 3;
            actions[env.human_agent_idx][1] = 6;
            if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
                actions[env.human_agent_idx][0] += accel_delta;
                // Cap acceleration to maximum of 6
                if (actions[env.human_agent_idx][0] > 6) {
                    actions[env.human_agent_idx][0] = 6;
                }
            }
            if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
                actions[env.human_agent_idx][0] -= accel_delta;
                // Cap acceleration to minimum of 0
                if (actions[env.human_agent_idx][0] < 0) {
                    actions[env.human_agent_idx][0] = 0;
                }
            }
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                actions[env.human_agent_idx][1] += steer_delta;
                // Cap steering to minimum of 0
                if (actions[env.human_agent_idx][1] < 0) {
                    actions[env.human_agent_idx][1] = 0;
                }
            }
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                actions[env.human_agent_idx][1] -= steer_delta;
                // Cap steering to maximum of 12
                if (actions[env.human_agent_idx][1] > 12) {
                    actions[env.human_agent_idx][1] = 12;
                }
            }
            if (IsKeyPressed(KEY_TAB)) {
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
        .reward_ade = conf.reward_ade,
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

int eval_gif(const char *map_name, const char *policy_name, int show_grid, int obs_only, int lasers,
             int log_trajectories, int frame_skip, const char *view_mode, const char *output_topdown,
             const char *output_agent, int num_maps, int zoom_in) {

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

    if (frame_skip <= 0) {
        frame_skip = 1;
    }

    // Check if map file exists
    FILE *map_file = fopen(map_name, "rb");
    if (map_file == NULL) {
        RAISE_FILE_ERROR(map_name);
    }
    fclose(map_file);

    FILE *policy_file = fopen(policy_name, "rb");
    if (policy_file == NULL) {
        RAISE_FILE_ERROR(policy_name);
    }
    fclose(policy_file);

    // Initialize environment with all config values from INI [env] section
    Drive env = {
        .action_type = conf.action_type,
        .dynamics_model = conf.dynamics_model,
        .reward_vehicle_collision = conf.reward_vehicle_collision,
        .reward_offroad_collision = conf.reward_offroad_collision,
        .reward_goal = conf.reward_goal,
        .reward_goal_post_respawn = conf.reward_goal_post_respawn,
        .reward_ade = conf.reward_ade,
        .goal_radius = conf.goal_radius,
        .goal_behavior = conf.goal_behavior,
        .goal_target_distance = conf.goal_target_distance,
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

    // Check if map has any active agents
    if (env.active_agent_count == 0) {
        fprintf(stderr, "Error: Map %s has no controllable agents\n", map_name);
        free_allocated(&env);
        return -1;
    }

    // Set which vehicle to focus on for obs mode
    int random_agent_idx = (rand() % 10) + 1;
    env.human_agent_idx = random_agent_idx;

    c_reset(&env);

    // Make client for rendering
    Client *client = (Client *)calloc(1, sizeof(Client));
    env.client = client;

    SetConfigFlags(FLAG_WINDOW_HIDDEN);
    SetTargetFPS(6000);

    float map_width = env.grid_map->bottom_right_x - env.grid_map->top_left_x;
    float map_height = env.grid_map->top_left_y - env.grid_map->bottom_right_y;

    printf("Map size: %.1fx%.1f\n", map_width, map_height);
    float scale = 6.0f;

    int img_width = (int)roundf(map_width * scale / 2.0f) * 2;
    int img_height = (int)roundf(map_height * scale / 2.0f) * 2;

    InitWindow(img_width, img_height, "Puffer Drive");
    SetConfigFlags(FLAG_MSAA_4X_HINT);

    // Load the textures and models
    client->puffers = LoadTexture("resources/puffers_128.png");
    client->cars[0] = LoadModel("resources/drive/RedCar.glb");
    client->cars[1] = LoadModel("resources/drive/WhiteCar.glb");
    client->cars[2] = LoadModel("resources/drive/BlueCar.glb");
    client->cars[3] = LoadModel("resources/drive/YellowCar.glb");
    client->cars[4] = LoadModel("resources/drive/GreenCar.glb");
    client->cars[5] = LoadModel("resources/drive/GreyCar.glb");
    client->cyclist = LoadModel("resources/drive/cyclist.glb");
    client->pedestrian = LoadModel("resources/drive/pedestrian.glb");

    Weights *weights = load_weights(policy_name);
    printf("Active agents in map: %d\n", env.active_agent_count);
    DriveNet *net = init_drivenet(weights, env.active_agent_count, env.dynamics_model);

    int frame_count = env.episode_length > 0 ? env.episode_length : TRAJECTORY_LENGTH_DEFAULT;
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
        strcpy(map, basename((char *)map_name));
        *strrchr(map, '.') = '\0';

        char video_dir[256];
        sprintf(video_dir, "%s/video", policy_base);
        char mkdir_cmd[512];
        snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p \"%s\"", video_dir);
        system(mkdir_cmd);

        sprintf(filename_topdown, "%s/video/%s_topdown.mp4", policy_base, map);
        sprintf(filename_agent, "%s/video/%s_agent.mp4", policy_base, map);
    }

    bool render_topdown = (strcmp(view_mode, "both") == 0 || strcmp(view_mode, "topdown") == 0);
    bool render_agent = (strcmp(view_mode, "both") == 0 || strcmp(view_mode, "agent") == 0);

    printf("Rendering: %s\n", view_mode);

    int rendered_frames = 0;
    double startTime = GetTime();

    VideoRecorder topdown_recorder, agent_recorder;

    if (render_topdown) {
        if (!OpenVideo(&topdown_recorder, filename_topdown, img_width, img_height)) {
            CloseWindow();
            return -1;
        }
    }

    if (render_agent) {
        if (!OpenVideo(&agent_recorder, filename_agent, img_width, img_height)) {
            if (render_topdown)
                CloseVideo(&topdown_recorder);
            CloseWindow();
            return -1;
        }
    }

    if (render_topdown) {
        printf("Recording topdown view...\n");
        for (int i = 0; i < frame_count; i++) {
            if (i % frame_skip == 0) {
                renderTopDownView(&env, client, map_height, 0, 0, 0, frame_count, NULL, log_trajectories, show_grid,
                                  img_width, img_height, zoom_in);
                WriteFrame(&topdown_recorder, img_width, img_height);
                rendered_frames++;
            }
            forward(net, env.observations, (int *)env.actions);
            c_step(&env);
        }
    }

    if (render_agent) {
        c_reset(&env);
        printf("Recording agent view...\n");
        for (int i = 0; i < frame_count; i++) {
            int human_idx = env.active_agent_indices[env.human_agent_idx];
            if (env.entities[human_idx].respawn_count > 0) {
                break;
            }
            if (i % frame_skip == 0) {
                renderAgentView(&env, client, map_height, obs_only, lasers, show_grid);
                WriteFrame(&agent_recorder, img_width, img_height);
                rendered_frames++;
            }
            forward(net, env.observations, (int *)env.actions);
            c_step(&env);
        }
    }

    double endTime = GetTime();
    double elapsedTime = endTime - startTime;
    double writeFPS = (elapsedTime > 0) ? rendered_frames / elapsedTime : 0;

    printf("Wrote %d frames in %.2f seconds (%.2f FPS) to %s\n", rendered_frames, elapsedTime, writeFPS,
           filename_topdown);

    if (render_topdown) {
        CloseVideo(&topdown_recorder);
    }
    if (render_agent) {
        CloseVideo(&agent_recorder);
    }
    CloseWindow();

    free(client);
    free_allocated(&env);
    free_drivenet(net);
    free(weights);
    return 0;
}

int main(int argc, char *argv[]) {
    // Check if we're running in eval_gif mode (with CLI arguments)
    if (argc > 1) {
        // Visualization-only parameters (not in [env] section)
        int show_grid = 0;
        int obs_only = 0;
        int lasers = 0;
        int log_trajectories = 1;
        int frame_skip = 1;
        int zoom_in = 1;
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
                log_trajectories = 1;
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
            } else {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
                return 1;
            }
        }

        return eval_gif(map_name, policy_name, show_grid, obs_only, lasers, log_trajectories, frame_skip, view_mode,
                        output_topdown, output_agent, num_maps, zoom_in);
    } else {
        // No command line arguments - run demo mode by default
        demo();
        return 0;
    }
}
