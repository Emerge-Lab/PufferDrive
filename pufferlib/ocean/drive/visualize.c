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


typedef struct {
    int ego_conditioning_size;
    int co_player_conditioning_size;
    int base_ego_size;
    int max_obs;
    int ego_obs_with_conditioning;
    int co_player_obs_with_conditioning;
} ObservationDimensions;

typedef struct {
    DriveNet* ego_net;
    DriveNet* co_player_net;
    Weights* ego_weights;
    Weights* co_player_weights;
    int* ego_actions;
    int* co_player_actions;
    float* ego_obs;
    float* co_player_obs;
} PopulationPlayState;

typedef struct {
    VideoRecorder topdown;
    VideoRecorder agent;
    bool render_topdown;
    bool render_agent;
} RenderState;

// ============================================================================
// Helper Functions
// ============================================================================

static bool load_config(env_init_config* conf, const char* ini_file) {
    if (ini_parse(ini_file, handler, conf) < 0) {
        fprintf(stderr, "Error: Could not load %s. Cannot determine environment configuration.\n", ini_file);
        return false;
    }
    return true;
}

static bool validate_file_exists(const char* filepath) {
    FILE* file = fopen(filepath, "rb");
    if (file == NULL) {
        RAISE_FILE_ERROR(filepath);
        return false;
    }
    fclose(file);
    return true;
}

static const char* resolve_map_name(char* map_buffer, const char* map_name, int num_maps) {
    if (map_name != NULL) {
        return map_name;
    }
    
    srand(time(NULL));
    int random_map = rand() % num_maps;
    sprintf(map_buffer, "resources/drive/binaries/map_%03d.bin", random_map);
    return map_buffer;
}

static void initialize_drive_env(Drive* env, const env_init_config* conf, 
                                 const char* map_name, float goal_radius,
                                 int population_play, int scenario_length_override, int use_rc, int use_ec, int use_dc) {
    env->dynamics_model = conf->dynamics_model;
    env->reward_vehicle_collision = conf->reward_vehicle_collision;
    env->reward_offroad_collision = conf->reward_offroad_collision;
    env->reward_ade = conf->reward_ade;
    env->goal_radius = goal_radius;
    env->dt = conf->dt;
    env->map_name = (char*)map_name;
    env->control_non_vehicles = conf->control_non_vehicles;
    env->init_steps = conf->init_steps;
    env->control_all_agents = conf->control_all_agents;
    env->policy_agents_per_env = conf->num_policy_controlled_agents;
    env->deterministic_agent_selection = conf->deterministic_agent_selection;
    env->population_play = population_play;
    env->scenario_length = (scenario_length_override > 0) ? scenario_length_override :
                          (conf->scenario_length > 0) ? conf->scenario_length : TRAJECTORY_LENGTH_DEFAULT;
    env->human_agent_idx = 0;

    env->use_rc = use_rc;
    env->use_ec = use_ec;
    env->use_dc = use_dc;

    if (use_rc){
        env->offroad_weight_lb = conf->reward_offroad_weight_lb;
        env->offroad_weight_ub = conf->reward_offroad_weight_ub;
        env->collision_weight_lb = conf->reward_collision_weight_lb;
        env->collision_weight_ub = conf->reward_collision_weight_ub;
        env->goal_weight_lb = conf->reward_goal_weight_lb;
        env->goal_weight_ub = conf->reward_goal_weight_ub;
    }
    if (use_ec){
        env->entropy_weight_lb = conf->entropy_weight_lb;
        env->entropy_weight_ub = conf->entropy_weight_ub;
    }
    if (use_dc){
        env->discount_weight_lb = conf->discount_weight_lb;
        env->discount_weight_ub = conf->discount_weight_ub;
    }

    

}

static ObservationDimensions calculate_observation_dimensions(
    const env_init_config* conf,
    int ego_reward_conditioned, int ego_entropy_conditioned, int ego_discount_conditioned,
    int co_player_reward_conditioned, int co_player_entropy_conditioned, int co_player_discount_conditioned) {
    
    ObservationDimensions dims = {0};
    
    // Calculate conditioning sizes
    if (ego_reward_conditioned) dims.ego_conditioning_size += 3;
    if (ego_entropy_conditioned) dims.ego_conditioning_size += 1;
    if (ego_discount_conditioned) dims.ego_conditioning_size += 1;
    
    if (co_player_reward_conditioned) dims.co_player_conditioning_size += 3;
    if (co_player_entropy_conditioned) dims.co_player_conditioning_size += 1;
    if (co_player_discount_conditioned) dims.co_player_conditioning_size += 1;
    
    // Determine base observation size based on dynamics model
    dims.base_ego_size = (conf->dynamics_model == JERK) ? 9 : 6;
    int remaining_obs_size = 63*7 + 200*7;  // Objects + road points
    dims.max_obs = dims.base_ego_size + 1 + remaining_obs_size;  // +1 for respawn flag
    
    // Sizes with conditioning
    dims.ego_obs_with_conditioning = dims.base_ego_size + dims.ego_conditioning_size + remaining_obs_size + 1;
    dims.co_player_obs_with_conditioning = dims.base_ego_size + dims.co_player_conditioning_size + remaining_obs_size + 1;
    
    return dims;
}

static int select_ego_agent(int active_agent_count) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    srand(ts.tv_nsec);
    return rand() % active_agent_count;
}

static void print_population_play_info(int ego_agent_id, int num_co_players, 
                                       int total_agents, int k_scenarios,
                                       const env_init_config* conf, 
                                       const ObservationDimensions* dims) {
    printf("Population play enabled: ego_agent_id=%d, num_co_players=%d, total_agents=%d\n", 
           ego_agent_id, num_co_players, total_agents);
    printf("Will run %d scenario attempts with same roles\n", k_scenarios);
    printf("Dynamics model: %s\n", (conf->dynamics_model == JERK) ? "JERK" : "NON-JERK");
    printf("Base ego size: %d, max_obs: %d\n", dims->base_ego_size, dims->max_obs);
}

static PopulationPlayState* initialize_population_play(
    const char* policy_name,
    int num_co_players,
    int active_agent_count,
    const env_init_config* conf,
    const ObservationDimensions* dims,
    int ego_reward_conditioned, int ego_entropy_conditioned, int ego_discount_conditioned,
    int co_player_reward_conditioned, int co_player_entropy_conditioned, int co_player_discount_conditioned) {
    
    PopulationPlayState* state = (PopulationPlayState*)calloc(1, sizeof(PopulationPlayState));
    if (!state) return NULL;
    
    // Load ego weights
    printf("DEBUG: Loading ego weights from: %s\n", policy_name);
    state->ego_weights = load_weights(policy_name);
    if (!state->ego_weights) {
        printf("ERROR: Failed to load weights from %s\n", policy_name);
        free(state);
        return NULL;
    }
    
    // Initialize ego network
    state->ego_net = init_drivenet(state->ego_weights, 1, conf->dynamics_model,
                                   ego_reward_conditioned, ego_entropy_conditioned, ego_discount_conditioned);
    if (!state->ego_net) {
        printf("ERROR: Failed to initialize ego network\n");
        free(state->ego_weights);
        free(state);
        return NULL;
    }
    
    // Load co-player weights
    printf("DEBUG: Loading co-player weights from: resources/drive/puffer_adaptive_driving_agent_co_player.bin\n");
    state->co_player_weights = load_weights("resources/drive/puffer_adaptive_driving_agent_co_player.bin");
    if (!state->co_player_weights) {
        printf("ERROR: Failed to load co-player weights\n");
        free_drivenet(state->ego_net);
        free(state->ego_weights);
        free(state);
        return NULL;
    }
    
    // Initialize co-player network
    state->co_player_net = init_drivenet(state->co_player_weights, num_co_players, conf->dynamics_model,
                                         co_player_reward_conditioned, co_player_entropy_conditioned, co_player_discount_conditioned);
    if (!state->co_player_net) {
        printf("ERROR: Failed to initialize co-player network\n");
        free(state->co_player_weights);
        free_drivenet(state->ego_net);
        free(state->ego_weights);
        free(state);
        return NULL;
    }
    
    // Allocate memory for actions and observations
    state->co_player_actions = (int*)calloc((active_agent_count-1)*2, sizeof(int));
    state->ego_actions = (int*)calloc(1*2, sizeof(int));
    state->co_player_obs = (float*)calloc(num_co_players * dims->co_player_obs_with_conditioning, sizeof(float));
    state->ego_obs = (float*)calloc(1 * dims->ego_obs_with_conditioning, sizeof(float));
    
    if (!state->co_player_actions || !state->ego_actions || 
        !state->co_player_obs || !state->ego_obs) {
        printf("ERROR: Failed to allocate memory for population play\n");
        free(state->co_player_actions);
        free(state->ego_actions);
        free(state->co_player_obs);
        free(state->ego_obs);
        free_drivenet(state->co_player_net);
        free(state->co_player_weights);
        free_drivenet(state->ego_net);
        free(state->ego_weights);
        free(state);
        return NULL;
    }
    
    printf("Population play memory allocated successfully\n");
    return state;
}

static void free_population_play_state(PopulationPlayState* state) {
    if (!state) return;
    
    if (state->ego_net) free_drivenet(state->ego_net);
    if (state->co_player_net) free_drivenet(state->co_player_net);
    if (state->co_player_actions) free(state->co_player_actions);
    if (state->ego_actions) free(state->ego_actions);
    if (state->co_player_obs) free(state->co_player_obs);
    if (state->ego_obs) free(state->ego_obs);
    if (state->ego_weights) free(state->ego_weights);
    if (state->co_player_weights) free(state->co_player_weights);
    free(state);
}

static void create_output_directory(const char* path) {
    char dir_path[256];
    const char* last_slash = strrchr(path, '/');
    if (last_slash) {
        size_t dir_len = last_slash - path;
        strncpy(dir_path, path, dir_len);
        dir_path[dir_len] = '\0';
        create_directory_recursive(dir_path);
    }
}

static void setup_output_paths(char* topdown_path, char* agent_path,
                               const char* output_topdown, const char* output_agent,
                               const char* policy_name, const char* map_name) {
    if (output_topdown && output_agent) {
        strcpy(topdown_path, output_topdown);
        strcpy(agent_path, output_agent);
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

        sprintf(topdown_path, "%s/video/%s_topdown.mp4", policy_base, map);
        sprintf(agent_path, "%s/video/%s_agent.mp4", policy_base, map);
    }
}

static bool initialize_window_and_rendering(Drive* env, Client* client) {
    SetConfigFlags(FLAG_WINDOW_HIDDEN);
    SetTargetFPS(6000);

    float map_width = env->grid_map->bottom_right_x - env->grid_map->top_left_x;
    float map_height = env->grid_map->top_left_y - env->grid_map->bottom_right_y;

    printf("Map size: %.1fx%.1f\n", map_width, map_height);
    float scale = 6.0f;

    int img_width = (int)roundf(map_width * scale / 2.0f) * 2;
    int img_height = (int)roundf(map_height * scale / 2.0f) * 2;
    InitWindow(img_width, img_height, "Puffer Drive");
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    
    return true;
}

static void assign_agent_roles(Drive* env, int ego_agent_id, bool population_play) {
    if (population_play) {
        for (int i = 0; i < env->active_agent_count; i++) {
            if (i != ego_agent_id) {
                env->entities[env->active_agent_indices[i]].is_ego = false;
                env->entities[env->active_agent_indices[i]].is_co_player = true;
            } else {
                env->entities[env->active_agent_indices[i]].is_ego = true;
                env->entities[env->active_agent_indices[i]].is_co_player = false;
            }
        }
        env->human_agent_idx = ego_agent_id;
    } else {
        for (int i = 0; i < env->active_agent_count; i++) {
            env->entities[env->active_agent_indices[i]].is_ego = true;
            env->entities[env->active_agent_indices[i]].is_co_player = false;
        }
    }
}

// ============================================================================
// Observation Conditioning
// ============================================================================

static void condition_observation(float* dest, const float* src,
                                  const ObservationDimensions* dims,
                                  bool is_jerk, int conditioning_size,
                                  float collision_weight, float offroad_weight, float goal_weight,
                                  float entropy_weight, float discount_weight,
                                  bool reward_conditioned, bool entropy_conditioned, bool discount_conditioned) {
    if (conditioning_size > 0) {
        if (is_jerk) {
            // Copy first 6 base + 3 JERK dynamics
            memcpy(dest, src, 6 * sizeof(float));
            memcpy(&dest[6], &src[6], 3 * sizeof(float));
            
            // Add conditioning after JERK (index 9)
            int cond_idx = 9;
            if (reward_conditioned) {
                dest[cond_idx++] = collision_weight;
                dest[cond_idx++] = offroad_weight;
                dest[cond_idx++] = goal_weight;
            }
            if (entropy_conditioned) {
                dest[cond_idx++] = entropy_weight;
            }
            if (discount_conditioned) {
                dest[cond_idx++] = discount_weight;
            }
            
            // Copy remaining observations
            memcpy(&dest[cond_idx], &src[9], (dims->max_obs - 9) * sizeof(float));
        } else {
            // Non-JERK: Copy first 6 base observations
            memcpy(dest, src, 6 * sizeof(float));
            
            // Add conditioning after first 6 (starting at index 6)
            int cond_idx = 6;
            if (reward_conditioned) {
                dest[cond_idx++] = collision_weight;
                dest[cond_idx++] = offroad_weight;
                dest[cond_idx++] = goal_weight;
            }
            if (entropy_conditioned) {
                dest[cond_idx++] = entropy_weight;
            }
            if (discount_conditioned) {
                dest[cond_idx++] = discount_weight;
            }
            
            // Copy remaining observations
            memcpy(&dest[cond_idx], &src[6], (dims->max_obs - 6) * sizeof(float));
        }
    } else {
        // No conditioning, just copy
        memcpy(dest, src, dims->max_obs * sizeof(float));
    }
}

static void prepare_population_play_observations(
    Drive* env, PopulationPlayState* state, int ego_agent_id,
    const ObservationDimensions* dims, bool is_jerk,
    float ego_collision_weight, float ego_offroad_weight, float ego_goal_weight,
    float ego_entropy_weight, float ego_discount_weight,
    float co_player_collision_weight, float co_player_offroad_weight, float co_player_goal_weight,
    float co_player_entropy_weight, float co_player_discount_weight,
    bool ego_reward_conditioned, bool ego_entropy_conditioned, bool ego_discount_conditioned,
    bool co_player_reward_conditioned, bool co_player_entropy_conditioned, bool co_player_discount_conditioned) {
    
    // Condition ego observation
    float* ego_src = &env->observations[ego_agent_id * dims->max_obs];
    condition_observation(state->ego_obs, ego_src, dims, is_jerk, dims->ego_conditioning_size,
                         ego_collision_weight, ego_offroad_weight, ego_goal_weight,
                         ego_entropy_weight, ego_discount_weight,
                         ego_reward_conditioned, ego_entropy_conditioned, ego_discount_conditioned);
    
    // Condition co-player observations
    int co_obs_offset = 0;
    for (int j = 0; j < env->active_agent_count; j++) {
        if (j == ego_agent_id) continue;
        
        float* co_src = &env->observations[j * dims->max_obs];
        float* co_dest = &state->co_player_obs[co_obs_offset];
        
        condition_observation(co_dest, co_src, dims, is_jerk, dims->co_player_conditioning_size,
                             co_player_collision_weight, co_player_offroad_weight, co_player_goal_weight,
                             co_player_entropy_weight, co_player_discount_weight,
                             co_player_reward_conditioned, co_player_entropy_conditioned, co_player_discount_conditioned);
        
        co_obs_offset += dims->co_player_obs_with_conditioning;
    }
}

static void execute_population_play_actions(Drive* env, PopulationPlayState* state, int ego_agent_id) {
    int (*actions)[2] = (int(*)[2])env->actions;
    
    // Get actions from both networks
    forward(state->ego_net, state->ego_obs, state->ego_actions);
    forward(state->co_player_net, state->co_player_obs, state->co_player_actions);
    
    // Assign ego action
    actions[ego_agent_id][0] = state->ego_actions[0];
    actions[ego_agent_id][1] = state->ego_actions[1];
    
    // Assign co-player actions
    int co_player_idx = 0;
    for (int j = 0; j < env->active_agent_count; j++) {
        if (j == ego_agent_id) continue;
        actions[j][0] = state->co_player_actions[co_player_idx * 2];
        actions[j][1] = state->co_player_actions[co_player_idx * 2 + 1];
        co_player_idx++;
    }
}

// ============================================================================
// Rendering Functions
// ============================================================================

typedef struct {
    float ego_collision_weight;
    float ego_offroad_weight;
    float ego_goal_weight;
    float ego_entropy_weight;
    float ego_discount_weight;
    float co_player_collision_weight;
    float co_player_offroad_weight;
    float co_player_goal_weight;
    float co_player_entropy_weight;
    float co_player_discount_weight;
    bool ego_reward_conditioned;
    bool ego_entropy_conditioned;
    bool ego_discount_conditioned;
    bool co_player_reward_conditioned;
    bool co_player_entropy_conditioned;
    bool co_player_discount_conditioned;
} ConditioningWeights;



// Unified rendering mode enum
typedef enum {
    RENDER_MODE_TOPDOWN,
    RENDER_MODE_AGENT
} RenderMode;

// Unified render frame function that handles both modes
static void render_frame(Drive* env, Client* client, float map_height, 
                        RenderMode mode, int obs_only, int lasers, 
                        int log_trajectories, int show_grid) {
    if (mode == RENDER_MODE_TOPDOWN) {
        renderTopDownView(env, client, map_height, obs_only, lasers, 
                         0,      // trajectories (set to 0, not used in refactor)
                         0,      // frame_count (not used in this context)
                         NULL,   // path (not used)
                         log_trajectories, show_grid);
    } else {
        renderAgentView(env, client, map_height, obs_only, lasers, show_grid);
    }
}

// Main scenario rendering function
static void render_scenario(
    Drive* env, Client* client, DriveNet* net, PopulationPlayState* pop_state,
    int ego_agent_id, int frame_count, int frame_skip, int k_scenarios,
    bool population_play, bool is_jerk, float map_height,
    const ObservationDimensions* dims, const ConditioningWeights* weights,
    VideoRecorder* recorder, int img_width, int img_height,
    RenderMode render_mode,
    int show_grid, int obs_only, int lasers, int log_trajectories) {

    int rendered_frames = 0;
    
    for (int scenario_idx = 0; scenario_idx < k_scenarios; scenario_idx++) {
        printf("\n=== Scenario attempt %d/%d ===\n", scenario_idx + 1, k_scenarios);
        fflush(stdout);
        
        if (scenario_idx > 0) {
            c_reset(env);
        }
        
        assign_agent_roles(env, ego_agent_id, population_play);
        
        for (int i = 0; i < frame_count; i++) {
            if (i % frame_skip == 0) {
                render_frame(env, client, map_height, render_mode, 
                           obs_only, lasers, log_trajectories, show_grid);
                WriteFrame(recorder, img_width, img_height);
                rendered_frames++;
            }
            
            if (population_play) {
                prepare_population_play_observations(
                    env, pop_state, ego_agent_id, dims, is_jerk,
                    weights->ego_collision_weight, weights->ego_offroad_weight, 
                    weights->ego_goal_weight, weights->ego_entropy_weight, 
                    weights->ego_discount_weight,
                    weights->co_player_collision_weight, weights->co_player_offroad_weight, 
                    weights->co_player_goal_weight, weights->co_player_entropy_weight, 
                    weights->co_player_discount_weight,
                    weights->ego_reward_conditioned, weights->ego_entropy_conditioned, 
                    weights->ego_discount_conditioned,
                    weights->co_player_reward_conditioned, weights->co_player_entropy_conditioned, 
                    weights->co_player_discount_conditioned);
                
                execute_population_play_actions(env, pop_state, ego_agent_id);
            } else {
                forward(net, env->observations, (int*)env->actions);
            }
            
            c_step(env);
        }
    }
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

    // Load configuration
    env_init_config conf = {0};
    if (!load_config(&conf, "pufferlib/config/ocean/drive.ini")) {
        return -1;
    }
    
    // Resolve map name
    char map_buffer[100];
    map_name = resolve_map_name(map_buffer, map_name, num_maps);
    
    // Validate frame skip
    if (frame_skip <= 0) {
        frame_skip = 1;
    }
    
    // Validate files exist
    if (!validate_file_exists(map_name) || !validate_file_exists(policy_name)) {
        return -1;
    }
    
    // Initialize environment
    Drive env = {0};
    initialize_drive_env(&env, &conf, map_name, goal_radius, population_play, 
                        scenario_length_override, use_rc, use_ec, use_dc);
    allocate(&env);
    
    // Setup output paths
    char topdown_path[256], agent_path[256];
    setup_output_paths(topdown_path, agent_path, output_topdown, output_agent, 
                      policy_name, map_name);
    create_output_directory(topdown_path);
    
    // Reset environment and create client
    c_reset(&env);
    Client* client = (Client*)calloc(1, sizeof(Client));
    env.client = client;
    
    // Initialize window and rendering
    if (!initialize_window_and_rendering(&env, client)) {
        free(client);
        free_allocated(&env);
        return -1;
    }
    
    float map_width = env.grid_map->bottom_right_x - env.grid_map->top_left_x;
    float map_height = env.grid_map->top_left_y - env.grid_map->bottom_right_y;
    float scale = 6.0f;
    int img_width = (int)roundf(map_width * scale / 2.0f) * 2;
    int img_height = (int)roundf(map_height * scale / 2.0f) * 2;
    
    // Calculate observation dimensions
    ObservationDimensions dims = calculate_observation_dimensions(
        &conf, ego_reward_conditioned, ego_entropy_conditioned, ego_discount_conditioned,
        co_player_reward_conditioned, co_player_entropy_conditioned, 
        co_player_discount_conditioned);
    
    // Setup conditioning weights
    ConditioningWeights cond_weights = {
        .ego_collision_weight = ego_collision_weight,
        .ego_offroad_weight = ego_offroad_weight,
        .ego_goal_weight = ego_goal_weight,
        .ego_entropy_weight = ego_entropy_weight,
        .ego_discount_weight = ego_discount_weight,
        .co_player_collision_weight = co_player_collision_weight,
        .co_player_offroad_weight = co_player_offroad_weight,
        .co_player_goal_weight = co_player_goal_weight,
        .co_player_entropy_weight = co_player_entropy_weight,
        .co_player_discount_weight = co_player_discount_weight,
        .ego_reward_conditioned = ego_reward_conditioned,
        .ego_entropy_conditioned = ego_entropy_conditioned,
        .ego_discount_conditioned = ego_discount_conditioned,
        .co_player_reward_conditioned = co_player_reward_conditioned,
        .co_player_entropy_conditioned = co_player_entropy_conditioned,
        .co_player_discount_conditioned = co_player_discount_conditioned
    };
    
    // Initialize networks based on mode
    PopulationPlayState* pop_state = NULL;
    DriveNet* net = NULL;
    Weights* weights = NULL;
    int ego_agent_id = 0;
    int num_co_players = 0;
    
    if (population_play) {
        ego_agent_id = select_ego_agent(env.active_agent_count);
        num_co_players = env.active_agent_count - 1;
        print_population_play_info(ego_agent_id, num_co_players, env.active_agent_count, 
                                   k_scenarios, &conf, &dims);
        
        pop_state = initialize_population_play(
            policy_name, num_co_players, env.active_agent_count, &conf, &dims,
            ego_reward_conditioned, ego_entropy_conditioned, ego_discount_conditioned,
            co_player_reward_conditioned, co_player_entropy_conditioned, 
            co_player_discount_conditioned);
        
        if (!pop_state) {
            CloseWindow();
            free(client);
            free_allocated(&env);
            return -1;
        }
    } else {
        weights = load_weights(policy_name);
        printf("Active agents in map: %d\n", env.active_agent_count);
        net = init_drivenet(weights, env.active_agent_count, conf.dynamics_model, 0, 0, 0);
    }
    
    // Validate observations
    printf("DEBUG: Checking observation buffer - env.observations=%p\n", 
           (void*)env.observations);
    if (env.observations == NULL) {
        printf("ERROR: Observations not allocated!\n");
        if (population_play) free_population_play_state(pop_state);
        else { 
            if (net) free_drivenet(net); 
            if (weights) free(weights); 
        }
        CloseWindow();
        free(client);
        free_allocated(&env);
        return -1;
    }
    printf("DEBUG: Observation buffer is valid\n");
    fflush(stdout);
    
    int frame_count = env.scenario_length > 0 ? env.scenario_length : TRAJECTORY_LENGTH_DEFAULT;
    bool render_topdown = (strcmp(view_mode, "both") == 0 || strcmp(view_mode, "topdown") == 0);
    bool render_agent = (strcmp(view_mode, "both") == 0 || strcmp(view_mode, "agent") == 0);
    bool is_jerk = (conf.dynamics_model == JERK);
    
    // Open video recorders
    VideoRecorder topdown_recorder, agent_recorder;
    if (render_topdown && !OpenVideo(&topdown_recorder, topdown_path, img_width, img_height)) {
        if (population_play) free_population_play_state(pop_state);
        else { 
            if (net) free_drivenet(net); 
            if (weights) free(weights); 
        }
        CloseWindow();
        free(client);
        free_allocated(&env);
        return -1;
    }
    
    if (render_agent && !OpenVideo(&agent_recorder, agent_path, img_width, img_height)) {
        if (render_topdown) CloseVideo(&topdown_recorder);
        if (population_play) free_population_play_state(pop_state);
        else { 
            if (net) free_drivenet(net); 
            if (weights) free(weights); 
        }
        CloseWindow();
        free(client);
        free_allocated(&env);
        return -1;
    }
    
    printf("Rendering: %s\n", view_mode);
    double startTime = GetTime();
    
    // Render topdown view
    if (render_topdown) {
        printf("Recording topdown view...\n");
        render_scenario(&env, client, net, pop_state, ego_agent_id, frame_count, frame_skip, 
                       k_scenarios, population_play, is_jerk, map_height, &dims, &cond_weights,
                       &topdown_recorder, img_width, img_height, RENDER_MODE_TOPDOWN,
                       show_grid, obs_only, lasers, log_trajectories);
    }
    
    // Render agent view
    if (render_agent) {
        c_reset(&env);
        printf("Recording agent view...\n");
        render_scenario(&env, client, net, pop_state, ego_agent_id, frame_count, frame_skip, 
                       k_scenarios, population_play, is_jerk, map_height, &dims, &cond_weights,
                       &agent_recorder, img_width, img_height, RENDER_MODE_AGENT,
                       show_grid, obs_only, lasers, log_trajectories);
    }
    
    // Calculate and print statistics
    double endTime = GetTime();
    double elapsedTime = endTime - startTime;
    int total_frames = (render_topdown ? 1 : 0) + (render_agent ? 1 : 0);
    total_frames *= (frame_count / frame_skip) * k_scenarios;
    double writeFPS = (elapsedTime > 0) ? total_frames / elapsedTime : 0;
    
    printf("Wrote %d frames in %.2f seconds (%.2f FPS) to %s\n",
           total_frames, elapsedTime, writeFPS, topdown_path);
    
    // Cleanup
    if (render_topdown) CloseVideo(&topdown_recorder);
    if (render_agent) CloseVideo(&agent_recorder);
    CloseWindow();
    
    if (population_play) {
        free_population_play_state(pop_state);
    } else {
        if (net) free_drivenet(net);
        if (weights) free(weights);
    }
    
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
    int use_rc = 1;
    int use_ec = 1;
    int use_dc = 1;
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