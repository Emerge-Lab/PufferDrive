#include <time.h>
#include <unistd.h>
#include "drive.h"
#include "puffernet.h"
#include <sys/wait.h>
#include <math.h>
#include <raylib.h>
#include "rlgl.h"
#include <stdlib.h>
#include <stdio.h>
#include "error.h"

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
        execlp("ffmpeg", "ffmpeg",
               "-y",
               "-f", "rawvideo",
               "-pix_fmt", "rgba",
               "-s", size_str,
               "-r", "30",
               "-i", "-",
               "-c:v", "libx264",
               "-pix_fmt", "yuv420p",
               "-preset", "fast",
               "-crf", "18",
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

typedef struct DriveNet DriveNet;
struct DriveNet {
    int num_agents;
    float* obs_self;
    float* obs_partner;
    float* obs_road;
    float* partner_linear_output;
    float* road_linear_output;
    float* partner_layernorm_output;
    float* road_layernorm_output;
    float* partner_linear_output_two;
    float* road_linear_output_two;
    Linear* ego_encoder;
    Linear* road_encoder;
    Linear* partner_encoder;
    LayerNorm* ego_layernorm;
    LayerNorm* road_layernorm;
    LayerNorm* partner_layernorm;
    Linear* ego_encoder_two;
    Linear* road_encoder_two;
    Linear* partner_encoder_two;
    MaxDim1* partner_max;
    MaxDim1* road_max;
    CatDim1* cat1;
    CatDim1* cat2;
    GELU* gelu;
    Linear* shared_embedding;
    ReLU* relu;
    LSTM* lstm;
    Linear* actor;
    Linear* value_fn;
    Multidiscrete* multidiscrete;
};

DriveNet* init_drivenet(Weights* weights, int num_agents) {
    DriveNet* net = calloc(1, sizeof(DriveNet));
    int hidden_size = 256;
    int input_size = 64;

    net->num_agents = num_agents;
    net->obs_self = calloc(num_agents*7, sizeof(float)); // 7 features
    net->obs_partner = calloc(num_agents*63*7, sizeof(float)); // 63 objects, 7 features
    net->obs_road = calloc(num_agents*200*13, sizeof(float)); // 200 objects, 13 features
    net->partner_linear_output = calloc(num_agents*63*input_size, sizeof(float));
    net->road_linear_output = calloc(num_agents*200*input_size, sizeof(float));
    net->partner_linear_output_two = calloc(num_agents*63*input_size, sizeof(float));
    net->road_linear_output_two = calloc(num_agents*200*input_size, sizeof(float));
    net->partner_layernorm_output = calloc(num_agents*63*input_size, sizeof(float));
    net->road_layernorm_output = calloc(num_agents*200*input_size, sizeof(float));
    net->ego_encoder = make_linear(weights, num_agents, 7, input_size);
    net->ego_layernorm = make_layernorm(weights, num_agents, input_size);
    net->ego_encoder_two = make_linear(weights, num_agents, input_size, input_size);
    net->road_encoder = make_linear(weights, num_agents, 13, input_size);
    net->road_layernorm = make_layernorm(weights, num_agents, input_size);
    net->road_encoder_two = make_linear(weights, num_agents, input_size, input_size);
    net->partner_encoder = make_linear(weights, num_agents, 7, input_size);
    net->partner_layernorm = make_layernorm(weights, num_agents, input_size);
    net->partner_encoder_two = make_linear(weights, num_agents, input_size, input_size);
    net->partner_max = make_max_dim1(num_agents, 63, input_size);
    net->road_max = make_max_dim1(num_agents, 200, input_size);
    net->cat1 = make_cat_dim1(num_agents, input_size, input_size);
    net->cat2 = make_cat_dim1(num_agents, input_size + input_size, input_size);
    net->gelu = make_gelu(num_agents, 3*input_size);
    net->shared_embedding = make_linear(weights, num_agents, input_size*3, hidden_size);
    net->relu = make_relu(num_agents, hidden_size);
    net->actor = make_linear(weights, num_agents, hidden_size, 20);
    net->value_fn = make_linear(weights, num_agents, hidden_size, 1);
    net->lstm = make_lstm(weights, num_agents, hidden_size, 256);
    memset(net->lstm->state_h, 0, num_agents*256*sizeof(float));
    memset(net->lstm->state_c, 0, num_agents*256*sizeof(float));
    int logit_sizes[2] = {7, 13};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 2);
    return net;
}

void free_drivenet(DriveNet* net) {
    free(net->obs_self);
    free(net->obs_partner);
    free(net->obs_road);
    free(net->partner_linear_output);
    free(net->road_linear_output);
    free(net->partner_linear_output_two);
    free(net->road_linear_output_two);
    free(net->partner_layernorm_output);
    free(net->road_layernorm_output);
    free(net->ego_encoder);
    free(net->road_encoder);
    free(net->partner_encoder);
    free(net->ego_layernorm);
    free(net->road_layernorm);
    free(net->partner_layernorm);
    free(net->ego_encoder_two);
    free(net->road_encoder_two);
    free(net->partner_encoder_two);
    free(net->partner_max);
    free(net->road_max);
    free(net->cat1);
    free(net->cat2);
    free(net->gelu);
    free(net->shared_embedding);
    free(net->relu);
    free(net->multidiscrete);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net);
}

void forward(DriveNet* net, float* observations, int* actions) {
    // Clear previous observations
    memset(net->obs_self, 0, net->num_agents * 7 * sizeof(float));
    memset(net->obs_partner, 0, net->num_agents * 63 * 7 * sizeof(float));
    memset(net->obs_road, 0, net->num_agents * 200 * 13 * sizeof(float));

    // Reshape observations into 2D boards and additional features
    float (*obs_self)[7] = (float (*)[7])net->obs_self;
    float (*obs_partner)[63][7] = (float (*)[63][7])net->obs_partner;
    float (*obs_road)[200][13] = (float (*)[200][13])net->obs_road;

    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b * (7 + 63*7 + 200*7);  // offset for each batch
        int partner_offset = b_offset + 7;
        int road_offset = b_offset + 7 + 63*7;
        // Process self observation
        for(int i = 0; i < 7; i++) {
            obs_self[b][i] = observations[b_offset + i];
        }

        // Process partner observation
        for(int i = 0; i < 63; i++) {
            for(int j = 0; j < 7; j++) {
                obs_partner[b][i][j] = observations[partner_offset + i*7 + j];
            }
        }

        // Process road observation
        for(int i = 0; i < 200; i++) {
            for(int j = 0; j < 7; j++) {
                obs_road[b][i][j] = observations[road_offset + i*7 + j];
            }
            for(int j = 0; j < 7; j++) {
                if(j == observations[road_offset+i*7 + 6]) {
                    obs_road[b][i][6 + j] = 1.0f;
                } else {
                    obs_road[b][i][6 + j] = 0.0f;
                }
            }
        }
    }

    // Forward pass through the network
    linear(net->ego_encoder, net->obs_self);
    layernorm(net->ego_layernorm, net->ego_encoder->output);
    linear(net->ego_encoder_two, net->ego_layernorm->output);
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < 63; obj++) {
            // Get the 7 features for this object
            float* obj_features = &net->obs_partner[b*63*7 + obj*7];
            // Apply linear layer to this object
            _linear(obj_features, net->partner_encoder->weights, net->partner_encoder->bias,
                   &net->partner_linear_output[b*63*64 + obj*64], 1, 7, 64);
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < 63; obj++) {
            float* after_first = &net->partner_linear_output[b*63*64 + obj*64];
            _layernorm(after_first, net->partner_layernorm->weights, net->partner_layernorm->bias,
                        &net->partner_layernorm_output[b*63*64 + obj*64], 1, 64);
        }
    }
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < 63; obj++) {
            // Get the 7 features for this object
            float* obj_features = &net->partner_layernorm_output[b*63*64 + obj*64];
            // Apply linear layer to this object
            _linear(obj_features, net->partner_encoder_two->weights, net->partner_encoder_two->bias,
                   &net->partner_linear_output_two[b*63*64 + obj*64], 1, 64, 64);

        }
    }

    // Process road objects: apply linear to each object individually
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < 200; obj++) {
            // Get the 13 features for this object
            float* obj_features = &net->obs_road[b*200*13 + obj*13];
            // Apply linear layer to this object
            _linear(obj_features, net->road_encoder->weights, net->road_encoder->bias,
                   &net->road_linear_output[b*200*64 + obj*64], 1, 13, 64);
        }
    }

    // Apply layer norm and second linear to each road object
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < 200; obj++) {
            float* after_first = &net->road_linear_output[b*200*64 + obj*64];
            _layernorm(after_first, net->road_layernorm->weights, net->road_layernorm->bias,
                        &net->road_layernorm_output[b*200*64 + obj*64], 1, 64);
        }
    }
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < 200; obj++) {
            float* after_first = &net->road_layernorm_output[b*200*64 + obj*64];
            _linear(after_first, net->road_encoder_two->weights, net->road_encoder_two->bias,
                    &net->road_linear_output_two[b*200*64 + obj*64], 1, 64, 64);
        }
    }

    max_dim1(net->partner_max, net->partner_linear_output_two);
    max_dim1(net->road_max, net->road_linear_output_two);
    cat_dim1(net->cat1, net->ego_encoder_two->output, net->road_max->output);
    cat_dim1(net->cat2, net->cat1->output, net->partner_max->output);
    gelu(net->gelu, net->cat2->output);
    linear(net->shared_embedding, net->gelu->output);
    relu(net->relu, net->shared_embedding->output);
    lstm(net->lstm, net->relu->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);

    // Get action by taking argmax of actor output
    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);
}
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

    float map_width = env.grid_map->bottom_right_x - env.grid_map->top_left_x;
    float map_height = env.grid_map->top_left_y - env.grid_map->bottom_right_y;

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
