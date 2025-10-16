#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <math.h>
#include <raylib.h>
#include "rlgl.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "error.h"
#include "drive.c"
#include "libgen.h"

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
        execlp("ffmpeg", "ffmpeg",
               "-y",
               "-f", "rawvideo",
               "-pix_fmt", "rgba",
               "-s", size_str,
               "-r", "30",
               "-i", "-",
               "-c:v", "libx264",
               "-pix_fmt", "yuv420p",
               "-preset", "ultrafast",
               "-crf", "23",
               "-loglevel", "error",
               output_filename,
               NULL);
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

void renderTopDownView(Drive* env, Client* client, int map_height, int obs, int lasers, int trajectories, int frame_count, float* path, int log_trajectories, int show_grid) {

    BeginDrawing();

    // Top-down orthographic camera
    Camera3D camera = {0};
    camera.position = (Vector3){ 0.0f, 0.0f, 500.0f };  // above the scene
    camera.target   = (Vector3){ 0.0f, 0.0f, 0.0f };  // look at origin
    camera.up       = (Vector3){ 0.0f, -1.0f, 0.0f };
    camera.fovy     = map_height;
    camera.projection = CAMERA_ORTHOGRAPHIC;

    Color road = (Color){35, 35, 37, 255};
    ClearBackground(road);
    BeginMode3D(camera);
    rlEnableDepthTest();

    // Draw human replay trajectories if enabled
    if(log_trajectories){
    for(int i=0; i<env->active_agent_count; i++){
        int idx = env->active_agent_indices[i];
        Vector3 prev_point = {0};
        bool has_prev = false;

        for(int j=0; j<TRAJECTORY_LENGTH; j++){
            float x = env->entities[idx].traj_x[j];
            float y = env->entities[idx].traj_y[j];
            float valid = env->entities[idx].traj_valid[j];

            if(!valid) {
                has_prev = false;
                continue;
            }

            Vector3 curr_point = {x, y, 0.5f};

            if(has_prev) {
                DrawLine3D(prev_point, curr_point, Fade(LIGHTGREEN, 0.6f));
            }

            prev_point = curr_point;
            has_prev = true;
        }
    }
}

    // Draw agent trajs
    if(trajectories){
        for(int i=0; i<frame_count; i++){
            DrawSphere((Vector3){path[i*2], path[i*2 +1], 0.8f}, 0.5f, YELLOW);
        }
    }

    // Draw scene
    draw_scene(env, client, 1, obs, lasers, show_grid);
    EndMode3D();
    EndDrawing();
}

void renderAgentView(Drive* env, Client* client, int map_height, int obs_only, int lasers, int show_grid) {
    // Agent perspective camera following the selected agent
    int agent_idx = env->active_agent_indices[env->human_agent_idx];
    Entity* agent = &env->entities[agent_idx];

    BeginDrawing();

    Camera3D camera = {0};
    // Position camera behind and above the agent
    camera.position = (Vector3){
        agent->x - (25.0f * cosf(agent->heading)),
        agent->y - (25.0f * sinf(agent->heading)),
        15.0f
    };
    camera.target = (Vector3){
        agent->x + 40.0f * cosf(agent->heading),
        agent->y + 40.0f * sinf(agent->heading),
        1.0f
    };
    camera.up = (Vector3){ 0.0f, 0.0f, 1.0f };
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

int eval_gif(const char* map_name, const char* policy_name, int show_grid, int obs_only, int lasers, int log_trajectories, int frame_skip, float goal_radius, int oracle_mode, int use_rc, int use_ec, int use_dc, const char* view_mode) {

    char map_buffer[100];
    if (map_name == NULL) {
        srand(time(NULL));
        int random_map = rand() % 100;
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

    // Make env
    Drive env = {
        .dynamics_model = CLASSIC,
        // .reward_ade = -0.0f,
        .goal_radius = goal_radius,
	      .map_name = map_name,
        // .spawn_immunity_timer = 50,
        .use_rc = use_rc,
        .use_ec = use_ec,
        .use_dc = use_dc,
        .oracle_mode = oracle_mode,
        // Reward conditioning weight bounds (defaults)
        .collision_weight_lb = -1.0f,
        .collision_weight_ub = -1.0f,
        .offroad_weight_lb = -0.4f,
        .offroad_weight_ub = -0.4f,
        .goal_weight_lb = 1.0f,
        .goal_weight_ub = 1.0f,
        // Entropy conditioning weight bounds (from training config)
        .entropy_weight_lb = 0.001f,
        .entropy_weight_ub = 0.001f,
        // Discount conditioning weight bounds (from training config)
        .discount_weight_lb = 0.98f,
        .discount_weight_ub = 0.98f
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
    Weights* weights = load_weights(policy_name);
    printf("Active agents in map: %d\n", env.active_agent_count);
    printf("Model config: oracle=%d, use_rc=%d, use_ec=%d, use_dc=%d\n", oracle_mode, use_rc, use_ec, use_dc);
    DriveNet* net = init_drivenet(weights, env.active_agent_count, use_rc, use_ec, use_dc, oracle_mode);

    int frame_count = 91;
    int log_trajectory = log_trajectories;
    char filename_topdown[256];
    char filename_agent[256];

    char policy_base[256];
    strcpy(policy_base, policy_name);
    *strrchr(policy_base, '.') = '\0';

    char map[256];
    strcpy(map, basename((char*)map_name));
    *strrchr(map, '.') = '\0';

    sprintf(filename_topdown, "%s/gifs/%s_topdown.mp4", policy_base, map);
    sprintf(filename_agent, "%s/gifs/%s_agent.mp4", policy_base, map);

    // Determine which views to render
    bool render_topdown = (strcmp(view_mode, "both") == 0 || strcmp(view_mode, "topdown") == 0);
    bool render_agent = (strcmp(view_mode, "both") == 0 || strcmp(view_mode, "agent") == 0);

    printf("Starting video recording...\n");
    printf("Rendering: %s\n", view_mode);

    int rendered_frames = 0;
    double startTime = GetTime();

    // Create video recorders for requested views
    VideoRecorder topdown_recorder, agent_recorder;

    if (render_topdown) {
        if (!OpenVideo(&topdown_recorder, filename_topdown, img_width, img_height)) {
            CloseWindow();
            return -1;
        }
    }

    if (render_agent) {
        if (!OpenVideo(&agent_recorder, filename_agent, img_width, img_height)) {
            if (render_topdown) CloseVideo(&topdown_recorder);
            CloseWindow();
            return -1;
        }
    }

    // Render topdown view if requested
    if (render_topdown) {
        printf("Recording top-down view...\n");
        for(int i = 0; i < frame_count; i++) {
            if (i % frame_skip == 0) {
                renderTopDownView(&env, client, map_height, 0, 0, 0, frame_count, NULL, log_trajectories, show_grid);
                WriteFrame(&topdown_recorder, img_width, img_height);
                rendered_frames++;
            }
            int (*actions)[2] = (int(*)[2])env.actions;
            forward(net, env.observations, env.actions);
            c_step(&env);
            for(int i=0; i < env.active_agent_count; i++)
            {
                printf("Agent %d - %f %f %f %f %f\n", i, env.logs[i].perf, env.logs[i].episode_return, env.logs[i].collision_rate, env.logs[i].offroad_rate, env.logs[i].completion_rate);
            }
        }

    }

    // Reset environment for agent view
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
            forward(net, env.observations, env.actions);
            c_step(&env);
        }
    }

    double endTime = GetTime();
    double elapsedTime = endTime - startTime;
    double writeFPS = (elapsedTime > 0) ? rendered_frames / elapsedTime : 0;

    printf("Wrote %d frames in %.2f seconds (%.2f FPS) to %s \n",
           rendered_frames, elapsedTime, writeFPS, filename_topdown);

    // Close video recorders (parallel encoding happens here)
    if (render_topdown) {
        printf("Finalizing topdown video...\n");
        CloseVideo(&topdown_recorder);
    }
    if (render_agent) {
        printf("Finalizing agent video...\n");
        CloseVideo(&agent_recorder);
    }
    CloseWindow();

    // Clean up resources
    free(client);
    free_allocated(&env);
    free_drivenet(net);
    free(weights);
    return 0;
}

int main(int argc, char* argv[]) {
    int show_grid = 0;
    int obs_only = 0;
    int lasers = 0;
    int log_trajectories = 1;
    int frame_skip = 1;
    float goal_radius = 2.0f;
    const char* map_name = NULL;
    const char* policy_name = "resources/drive/puffer_drive_weights.bin";

    int oracle_mode = 1;
    int use_rc = 1;
    int use_ec = 1;
    int use_dc = 1;
    const char* view_mode = "both";  // "both", "topdown", "agent"

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
        } else if (strcmp(argv[i], "--policy-name") == 0) {
            if (i + 1 < argc) {
                policy_name = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --policy-name option requires a policy file path\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--oracle") == 0) {
            if (i + 1 < argc) {
                oracle_mode = atoi(argv[i + 1]);
                i++;
            }
        } else if (strcmp(argv[i], "--use-rc") == 0) {
            if (i + 1 < argc) {
                use_rc = atoi(argv[i + 1]);
                i++;
            }
        } else if (strcmp(argv[i], "--use-ec") == 0) {
            if (i + 1 < argc) {
                use_ec = atoi(argv[i + 1]);
                i++;
            }
        } else if (strcmp(argv[i], "--use-dc") == 0) {
            if (i + 1 < argc) {
                use_dc = atoi(argv[i + 1]);
                i++;
            }
        } else if (strcmp(argv[i], "--view") == 0) {
            if (i + 1 < argc) {
                view_mode = argv[i + 1];
                i++;
                // Validate view_mode
                if (strcmp(view_mode, "both") != 0 &&
                    strcmp(view_mode, "topdown") != 0 &&
                    strcmp(view_mode, "agent") != 0) {
                    fprintf(stderr, "Error: --view must be 'both', 'topdown', or 'agent'\n");
                    return 1;
                }
            } else {
                fprintf(stderr, "Error: --view option requires a value (both/topdown/agent)\n");
                return 1;
            }
        }
    }

    eval_gif(map_name, policy_name, show_grid, obs_only, lasers, log_trajectories, frame_skip, goal_radius, oracle_mode, use_rc, use_ec, use_dc, view_mode);
    return 0;
}
