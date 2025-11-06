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

        for(int j = 0; j < env->entities[idx].array_size; j++){
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
             int scenario_length_override,
             int k_scenarios,
             int use_rc,
             int use_ec, 
             int use_dc,
             int oracle_mode,
             int ego_reward_conditioned,
             int ego_entropy_conditioned,
             int ego_discount_conditioned,
             int co_player_reward_conditioned,
             int co_player_entropy_conditioned,
             int co_player_discount_conditioned,
             float ego_collision_weight,
             float ego_offroad_weight,
             float ego_goal_weight,
             float ego_entropy_weight,
             float ego_discount_weight,
             float co_player_collision_weight,
             float co_player_offroad_weight,
             float co_player_goal_weight,
             float co_player_entropy_weight,
             float co_player_discount_weight) {

    // Parse configuration from INI file
    env_init_config conf = {0};
    const char* ini_file = "pufferlib/config/ocean/drive.ini";
    if(ini_parse(ini_file, handler, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s. Cannot determine environment configuration.\n", ini_file);
        return -1;
    }
    
    char map_buffer[100];
    if (map_name == NULL) {
        srand(time(NULL));
        int random_map = rand() % num_maps;
        sprintf(map_buffer, "resources/drive/binaries/map_%03d.bin", random_map);
        map_name = map_buffer;
    }

    if (frame_skip <= 0) {
        frame_skip = 1;
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
        .dynamics_model = conf.dynamics_model,
        .reward_vehicle_collision = conf.reward_vehicle_collision,
        .reward_offroad_collision = conf.reward_offroad_collision,
        .reward_ade = conf.reward_ade,
        .goal_radius = goal_radius,
        .dt = conf.dt,
        .map_name = (char*)map_name,
        .control_non_vehicles = conf.control_non_vehicles,
        .init_steps = conf.init_steps,
        .control_all_agents = conf.control_all_agents,
        .policy_agents_per_env = conf.num_policy_controlled_agents,
        .deterministic_agent_selection = conf.deterministic_agent_selection,
        .population_play = population_play,
    };
    env.scenario_length = (scenario_length_override > 0) ? scenario_length_override :
                          (conf.scenario_length > 0) ? conf.scenario_length : TRAJECTORY_LENGTH_DEFAULT;
    allocate(&env);

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
    float scale = 6.0f;

    int img_width = (int)roundf(map_width * scale / 2.0f) * 2;
    int img_height = (int)roundf(map_height * scale / 2.0f) * 2;
    InitWindow(img_width, img_height, "Puffer Drive");
    SetConfigFlags(FLAG_MSAA_4X_HINT);

    // Population play variables
    struct timespec ts;
    int ego_agent_id = 0;
    int num_co_players = 0;
    
    // Calculate conditioning sizes
    int ego_conditioning_size = 0;
    if (ego_reward_conditioned) ego_conditioning_size += 3;
    if (ego_entropy_conditioned) ego_conditioning_size += 1;
    if (ego_discount_conditioned) ego_conditioning_size += 1;
    
    int co_player_conditioning_size = 0;
    if (co_player_reward_conditioned) co_player_conditioning_size += 3;
    if (co_player_entropy_conditioned) co_player_conditioning_size += 1;
    if (co_player_discount_conditioned) co_player_conditioning_size += 1;
    
    // Determine base observation size based on dynamics model
    int base_ego_size = (conf.dynamics_model == JERK) ? 9 : 6;  // 6 base + 3 JERK or just 6 base
    int remaining_obs_size = 63*7 + 200*7;  // Objects + road points
    int max_obs = base_ego_size + 1 + remaining_obs_size;  // +1 for respawn flag at the end
    
    // Size WITH conditioning for ego and co-players
    int ego_obs_with_conditioning = base_ego_size + ego_conditioning_size + remaining_obs_size + 1;
    int co_player_obs_with_conditioning = base_ego_size + co_player_conditioning_size + remaining_obs_size + 1;

    if (population_play) {
        clock_gettime(CLOCK_REALTIME, &ts);
        srand(ts.tv_nsec);
        ego_agent_id = rand() % env.active_agent_count;
        num_co_players = env.active_agent_count - 1;
        
        printf("Population play enabled: ego_agent_id=%d, num_co_players=%d, total_agents=%d\n", 
               ego_agent_id, num_co_players, env.active_agent_count);
        printf("Will run %d scenario attempts with same roles\n", k_scenarios);
        printf("Dynamics model: %s\n", (conf.dynamics_model == JERK) ? "JERK" : "NON-JERK");
        printf("Base ego size: %d, max_obs: %d\n", base_ego_size, max_obs);
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
        printf("DEBUG: Loading ego weights from: %s\n", policy_name);
        weights = load_weights(policy_name);
        if (weights == NULL) {
            printf("ERROR: Failed to load weights from %s\n", policy_name);
            CloseWindow();
            return -1;
        }
        ego_net = init_drivenet(weights, 1, 
                                conf.dynamics_model,
                                ego_reward_conditioned, 
                                ego_entropy_conditioned, 
                                ego_discount_conditioned);
        if (ego_net == NULL) {
            printf("ERROR: Failed to initialize ego network\n");
            CloseWindow();
            return -1;
        }

        printf("DEBUG: Loading co-player weights from: resources/drive/puffer_adaptive_driving_agent_co_player.bin\n");
        co_player_policy_weights = load_weights("resources/drive/puffer_adaptive_driving_agent_co_player.bin");
        if (co_player_policy_weights == NULL) {
            printf("ERROR: Failed to load co-player weights\n");
            CloseWindow();
            return -1;
        }
        co_player_net = init_drivenet(co_player_policy_weights,
                                      num_co_players, 
                                      conf.dynamics_model,
                                      co_player_reward_conditioned,
                                      co_player_entropy_conditioned, 
                                      co_player_discount_conditioned);
        if (co_player_net == NULL) {
            printf("ERROR: Failed to initialize co-player network\n");
            CloseWindow();
            return -1;
        }

        co_player_actions = (int*)calloc((env.active_agent_count-1)*2, sizeof(int));
        ego_actions = (int*)calloc(1*2, sizeof(int));
        co_player_obs = (float*)calloc(num_co_players*co_player_obs_with_conditioning, sizeof(float));
        ego_agent_obs = (float*)calloc(1*ego_obs_with_conditioning, sizeof(float));
        
        if (!co_player_actions || !ego_actions || !co_player_obs || !ego_agent_obs) {
            printf("ERROR: Failed to allocate memory for population play\n");
            return -1;
        }
        
        printf("Population play memory allocated successfully\n");
    } else {
        weights = load_weights(policy_name);
        printf("Active agents in map: %d\n", env.active_agent_count);
        net = init_drivenet(weights, env.active_agent_count, conf.dynamics_model, use_rc, use_ec, use_dc);
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
        
        for (int scenario_idx = 0; scenario_idx < k_scenarios; scenario_idx++) {
            printf("\n=== Scenario attempt %d/%d ===\n", scenario_idx + 1, k_scenarios);
            fflush(stdout);
            
            if (scenario_idx > 0) {
                c_reset(&env);
            }
            
            if (population_play) {
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
            else{
                for (int i = 0; i < env.active_agent_count; i++) {
                    env.entities[env.active_agent_indices[i]].is_ego = true;
                    env.entities[env.active_agent_indices[i]].is_co_player = false;
                }
            }
            
            for(int i = 0; i < frame_count; i++) {
                if (i % frame_skip == 0) {
                    renderTopDownView(&env, client, map_height, 0, 0, 0, frame_count, NULL, log_trajectories, show_grid);
                    WriteFrame(&topdown_recorder, img_width, img_height);
                    rendered_frames++;
                }
                
                int (*actions)[2] = (int(*)[2])env.actions;
                
                if (population_play) {
                    // Copy and condition ego agent observation
                    float* ego_src = &env.observations[ego_agent_id * max_obs];
                    
                    if (ego_conditioning_size > 0) {
                        if (conf.dynamics_model == JERK) {
                            // Copy first 6 base observations
                            memcpy(ego_agent_obs, ego_src, 6 * sizeof(float));
                            
                            // Copy JERK dynamics (indices 6-8: steering_angle, a_long, a_lat)
                            memcpy(&ego_agent_obs[6], &ego_src[6], 3 * sizeof(float));
                            
                            // Add conditioning values after JERK dynamics (starting at index 9)
                            int cond_idx = 9;
                            if (ego_reward_conditioned) {
                                ego_agent_obs[cond_idx++] = ego_collision_weight;
                                ego_agent_obs[cond_idx++] = ego_offroad_weight;
                                ego_agent_obs[cond_idx++] = ego_goal_weight;
                            }
                            if (ego_entropy_conditioned) {
                                ego_agent_obs[cond_idx++] = ego_entropy_weight;
                            }
                            if (ego_discount_conditioned) {
                                ego_agent_obs[cond_idx++] = ego_discount_weight;
                            }
                            
                            // Copy remaining observations (objects + road points) and respawn flag
                            memcpy(&ego_agent_obs[cond_idx], &ego_src[9], (max_obs - 9) * sizeof(float));
                        } else {
                            // Non-JERK: Copy first 6 base observations
                            memcpy(ego_agent_obs, ego_src, 6 * sizeof(float));
                            
                            // Add conditioning values after first 6 (starting at index 6)
                            int cond_idx = 6;
                            if (ego_reward_conditioned) {
                                ego_agent_obs[cond_idx++] = ego_collision_weight;
                                ego_agent_obs[cond_idx++] = ego_offroad_weight;
                                ego_agent_obs[cond_idx++] = ego_goal_weight;
                            }
                            if (ego_entropy_conditioned) {
                                ego_agent_obs[cond_idx++] = ego_entropy_weight;
                            }
                            if (ego_discount_conditioned) {
                                ego_agent_obs[cond_idx++] = ego_discount_weight;
                            }
                            
                            // Copy remaining observations and respawn flag
                            memcpy(&ego_agent_obs[cond_idx], &ego_src[6], (max_obs - 6) * sizeof(float));
                        }
                    } else {
                        // No conditioning, just copy
                        memcpy(ego_agent_obs, ego_src, max_obs * sizeof(float));
                    }

                    // Copy and condition co-player observations
                    int co_obs_offset = 0;
                    for (int j = 0; j < env.active_agent_count; j++) {
                        if (j == ego_agent_id) continue;
                        
                        float* co_src = &env.observations[j * max_obs];
                        float* co_dest = &co_player_obs[co_obs_offset];
                        
                        if (co_player_conditioning_size > 0) {
                            if (conf.dynamics_model == JERK) {
                                // Copy first 6 base + 3 JERK dynamics
                                memcpy(co_dest, co_src, 6 * sizeof(float));
                                memcpy(&co_dest[6], &co_src[6], 3 * sizeof(float));
                                
                                // Add conditioning after JERK (index 9)
                                int cond_idx = 9;
                                if (co_player_reward_conditioned) {
                                    co_dest[cond_idx++] = co_player_collision_weight;
                                    co_dest[cond_idx++] = co_player_offroad_weight;
                                    co_dest[cond_idx++] = co_player_goal_weight;
                                }
                                if (co_player_entropy_conditioned) {
                                    co_dest[cond_idx++] = co_player_entropy_weight;
                                }
                                if (co_player_discount_conditioned) {
                                    co_dest[cond_idx++] = co_player_discount_weight;
                                }
                                
                                memcpy(&co_dest[cond_idx], &co_src[9], (max_obs - 9) * sizeof(float));
                            } else {
                                // Non-JERK: Copy first 6 base observations
                                memcpy(co_dest, co_src, 6 * sizeof(float));
                                
                                int cond_idx = 6;
                                if (co_player_reward_conditioned) {
                                    co_dest[cond_idx++] = co_player_collision_weight;
                                    co_dest[cond_idx++] = co_player_offroad_weight;
                                    co_dest[cond_idx++] = co_player_goal_weight;
                                }
                                if (co_player_entropy_conditioned) {
                                    co_dest[cond_idx++] = co_player_entropy_weight;
                                }
                                if (co_player_discount_conditioned) {
                                    co_dest[cond_idx++] = co_player_discount_weight;
                                }
                                
                                memcpy(&co_dest[cond_idx], &co_src[6], (max_obs - 6) * sizeof(float));
                            }
                            co_obs_offset += co_player_obs_with_conditioning;
                        } else {
                            // No conditioning, just copy
                            memcpy(co_dest, co_src, max_obs * sizeof(float));
                            co_obs_offset += max_obs;
                        }
                    }

                    // Get actions from both networks
                    forward(ego_net, ego_agent_obs, ego_actions);
                    forward(co_player_net, co_player_obs, co_player_actions);

                    // Assign ego action
                    actions[ego_agent_id][0] = ego_actions[0];
                    actions[ego_agent_id][1] = ego_actions[1];

                    // Assign co-player actions
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
    }

    if (render_agent) {
        c_reset(&env);
        printf("Recording agent view...\n");
        
        for (int scenario_idx = 0; scenario_idx < k_scenarios; scenario_idx++) {
            printf("\n=== Agent view scenario attempt %d/%d ===\n", scenario_idx + 1, k_scenarios);
            fflush(stdout);
            
            if (scenario_idx > 0) {
                c_reset(&env);
            }
            
            if (population_play) {
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
            
            for(int i = 0; i < frame_count; i++) {
                if (i % frame_skip == 0) {
                    renderAgentView(&env, client, map_height, obs_only, lasers, show_grid);
                    WriteFrame(&agent_recorder, img_width, img_height);
                    rendered_frames++;
                }
                
                int (*actions)[2] = (int(*)[2])env.actions;
                
                if (population_play) {
                    // Copy and condition ego agent observation
                    float* ego_src = &env.observations[ego_agent_id * max_obs];
                    
                    if (ego_conditioning_size > 0) {
                        if (conf.dynamics_model == JERK) {
                            memcpy(ego_agent_obs, ego_src, 6 * sizeof(float));
                            memcpy(&ego_agent_obs[6], &ego_src[6], 3 * sizeof(float));
                            
                            int cond_idx = 9;
                            if (ego_reward_conditioned) {
                                ego_agent_obs[cond_idx++] = ego_collision_weight;
                                ego_agent_obs[cond_idx++] = ego_offroad_weight;
                                ego_agent_obs[cond_idx++] = ego_goal_weight;
                            }
                            if (ego_entropy_conditioned) {
                                ego_agent_obs[cond_idx++] = ego_entropy_weight;
                            }
                            if (ego_discount_conditioned) {
                                ego_agent_obs[cond_idx++] = ego_discount_weight;
                            }
                            
                            memcpy(&ego_agent_obs[cond_idx], &ego_src[9], (max_obs - 9) * sizeof(float));
                        } else {
                            memcpy(ego_agent_obs, ego_src, 6 * sizeof(float));
                            
                            int cond_idx = 6;
                            if (ego_reward_conditioned) {
                                ego_agent_obs[cond_idx++] = ego_collision_weight;
                                ego_agent_obs[cond_idx++] = ego_offroad_weight;
                                ego_agent_obs[cond_idx++] = ego_goal_weight;
                            }
                            if (ego_entropy_conditioned) {
                                ego_agent_obs[cond_idx++] = ego_entropy_weight;
                            }
                            if (ego_discount_conditioned) {
                                ego_agent_obs[cond_idx++] = ego_discount_weight;
                            }
                            
                            memcpy(&ego_agent_obs[cond_idx], &ego_src[6], (max_obs - 6) * sizeof(float));
                        }
                    } else {
                        memcpy(ego_agent_obs, ego_src, max_obs * sizeof(float));
                    }

                    // Copy and condition co-player observations
                    int co_obs_offset = 0;
                    for (int j = 0; j < env.active_agent_count; j++) {
                        if (j == ego_agent_id) continue;
                        
                        float* co_src = &env.observations[j * max_obs];
                        float* co_dest = &co_player_obs[co_obs_offset];
                        
                        if (co_player_conditioning_size > 0) {
                            if (conf.dynamics_model == JERK) {
                                memcpy(co_dest, co_src, 6 * sizeof(float));
                                memcpy(&co_dest[6], &co_src[6], 3 * sizeof(float));
                                
                                int cond_idx = 9;
                                if (co_player_reward_conditioned) {
                                    co_dest[cond_idx++] = co_player_collision_weight;
                                    co_dest[cond_idx++] = co_player_offroad_weight;
                                    co_dest[cond_idx++] = co_player_goal_weight;
                                }
                                if (co_player_entropy_conditioned) {
                                    co_dest[cond_idx++] = co_player_entropy_weight;
                                }
                                if (co_player_discount_conditioned) {
                                    co_dest[cond_idx++] = co_player_discount_weight;
                                }
                                
                                memcpy(&co_dest[cond_idx], &co_src[9], (max_obs - 9) * sizeof(float));
                            } else {
                                memcpy(co_dest, co_src, 6 * sizeof(float));
                                
                                int cond_idx = 6;
                                if (co_player_reward_conditioned) {
                                    co_dest[cond_idx++] = co_player_collision_weight;
                                    co_dest[cond_idx++] = co_player_offroad_weight;
                                    co_dest[cond_idx++] = co_player_goal_weight;
                                }
                                if (co_player_entropy_conditioned) {
                                    co_dest[cond_idx++] = co_player_entropy_weight;
                                }
                                if (co_player_discount_conditioned) {
                                    co_dest[cond_idx++] = co_player_discount_weight;
                                }
                                
                                memcpy(&co_dest[cond_idx], &co_src[6], (max_obs - 6) * sizeof(float));
                            }
                            co_obs_offset += co_player_obs_with_conditioning;
                        } else {
                            memcpy(co_dest, co_src, max_obs * sizeof(float));
                            co_obs_offset += max_obs;
                        }
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
int main(int argc, char* argv[]) {
    int show_grid = 0;
    int obs_only = 0;
    int lasers = 0;
    int log_trajectories = 1;
    int frame_skip = 1;
    float goal_radius = 2.0f;
    int init_steps = 0;
    const char* map_name = NULL;
    const char* policy_name = "resources/drive/puffer_drive_weights.bin";
    int control_all_agents = 0;
    int deterministic_selection = 0;
    int policy_agents_per_env = -1;
    int control_non_vehicles = 0;
    int num_maps = 100;
    int scenario_length_cli = -1;
    int population_play = 0;
    int k_scenarios = 1;  // Default: run scenario once
    int use_rc = 0;
    int use_ec = 0;
    int use_dc = 0;
    int oracle_mode = 0;

    // Conditioning flags for ego agent
    int ego_reward_conditioned = 0;
    int ego_entropy_conditioned = 0;
    int ego_discount_conditioned = 0;
    
    // Conditioning flags for co-players
    int co_player_reward_conditioned = 0;
    int co_player_entropy_conditioned = 0;
    int co_player_discount_conditioned = 0;
    
    // Ego agent conditioning weights
    float ego_collision_weight = -1.0f;
    float ego_offroad_weight = -0.4f;
    float ego_goal_weight = 1;
    float ego_entropy_weight = 0.0f;
    float ego_discount_weight = 0.98f;
    
    // Co-player conditioning weights
    float co_player_collision_weight = -1.0f;
    float co_player_offroad_weight = -0.4f;
    float co_player_goal_weight = 1.0f;
    float co_player_entropy_weight = 0.0f;
    float co_player_discount_weight = 0.98f;

    const char* view_mode = "both";  // "both", "topdown", "agent"
    const char* output_topdown = NULL;
    const char* output_agent = NULL;

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
        } else if (strcmp(argv[i], "--view") == 0) {
            if (i + 1 < argc) {
                view_mode = argv[i + 1];
                i++;
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
        } else if (strcmp(argv[i], "--init-steps") == 0) {
            if (i + 1 < argc) {
                init_steps = atoi(argv[i + 1]);
                i++;
                if (init_steps < 0) {
                    init_steps = 0;
                }
            }
        } else if (strcmp(argv[i], "--control-non-vehicles") == 0) {
            control_non_vehicles = 1;
        } else if (strcmp(argv[i], "--pure-self-play") == 0) {
            control_all_agents = 1;
        } else if (strcmp(argv[i], "--num-policy-controlled-agents") == 0) {
            if (i + 1 < argc) {
                policy_agents_per_env = atoi(argv[i + 1]);
                i++;
            }
        } else if (strcmp(argv[i], "--deterministic-selection") == 0) {
            deterministic_selection = 1;
        } else if (strcmp(argv[i], "--num-maps") == 0) {
            if (i + 1 < argc) {
                num_maps = atoi(argv[i + 1]);
                i++;
            }
        } else if (strcmp(argv[i], "--scenario-length") == 0) {
            if (i + 1 < argc) {
                scenario_length_cli = atoi(argv[i + 1]);
                i++;
            }
        } else if (strcmp(argv[i], "--k-scenarios") == 0) {
            if (i + 1 < argc) {
                k_scenarios = atoi(argv[i + 1]);
                i++;
                if (k_scenarios < 1) {
                    k_scenarios = 1;
                }
            }
        } else if (strcmp(argv[i], "--population-play") == 0) {
            population_play = 1;
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
        }else if (strcmp(argv[i], "--population-play") == 0) {
            population_play = 1;

        }
    }

    eval_gif(map_name, policy_name, show_grid, obs_only, lasers, log_trajectories, 
         frame_skip, goal_radius, population_play, control_non_vehicles, 
         init_steps, control_all_agents, policy_agents_per_env, 
         deterministic_selection, view_mode, output_topdown, output_agent, 
         num_maps, scenario_length_cli, k_scenarios, use_rc, use_ec, use_dc,
         oracle_mode, ego_reward_conditioned, ego_entropy_conditioned, 
         ego_discount_conditioned, co_player_reward_conditioned, 
         co_player_entropy_conditioned, co_player_discount_conditioned,
         ego_collision_weight, ego_offroad_weight, ego_goal_weight, 
         ego_entropy_weight, ego_discount_weight, co_player_collision_weight,
         co_player_offroad_weight, co_player_goal_weight, 
         co_player_entropy_weight, co_player_discount_weight);
    return 0;
}