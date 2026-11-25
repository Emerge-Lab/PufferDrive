#include <signal.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <time.h>
#include "error.h"

// CURRENT

// EGL is optional: systems without EGL headers keep using the Xvfb/Mesa path.
#if defined(__linux__) && defined(__has_include)
#if __has_include(<EGL/egl.h>)
#define DRIVE_HAS_EGL 1
#endif
#endif

#ifdef DRIVE_HAS_EGL
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>
#include "egl_headless.h"
#endif

// Render modes
#define RENDER_WINDOW 0
#define RENDER_HEADLESS 1

// View modes
#define VIEW_MODE_SIM_STATE 0
#define VIEW_MODE_BEV_AGENT_OBS 1
#define VIEW_MODE_AGENT_PERSP 2
#define VIEW_MODE_SENSOR_NOISE 3
#define VIEW_MODE_FOLLOW_AGENT 4

// Order of entities in rendering (lower is rendered first)
#define Z_ROAD_SURFACE 0.0f
#define Z_ROAD_MARKINGS 0.05f // Lane lines, road lines, traces
#define Z_AGENT_DETAILS 0.2f  // Arrow, goal markers, obs overlays
#define Z_AGENTS 2.0f         // Vehicles, cyclists, pedestrians

// Entity Types
#define NONE 0
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3
#define ROAD_LANE 4
#define ROAD_LINE 5
#define ROAD_EDGE 6
#define STOP_SIGN 7
#define CROSSWALK 8
#define SPEED_BUMP 9
#define DRIVEWAY 10

#define INVALID_POSITION -10000.0f

// Trajectory Length
#define TRAJECTORY_LENGTH 91

// Initialization modes
#define INIT_ALL_VALID 0
#define INIT_ONLY_CONTROLLABLE_AGENTS 1

// Control modes
#define CONTROL_VEHICLES 0
#define CONTROL_AGENTS 1
#define CONTROL_WOSAC 2
#define CONTROL_SDC_ONLY 3
#define CONTROL_MIXED_PLAY 4
#define CONTROL_INFERRED_EXPERT_ACTIONS 5
#define CONTROL_REPLAY_LOGS 6

// Controller modes
#define CONTROLLER_STATIC 0
#define CONTROLLER_POLICY 1
#define CONTROLLER_REPLAY 2

// Minimum distance to goal position
#define MIN_DISTANCE_TO_GOAL 2.0f

// Threshold for data filtering
#define ADE_THRESHOLD 2.0f

// Dynamics models
#define CLASSIC 0
#define JERK 1
#define DELTA_LOCAL 2
// Grid cell size: Depends on resolution of data
// Formula: 3 * (2 + GRID_CELL_SIZE*sqrt(2)/resolution) => For each entity type in
// gridmap, diagonal poly-lines -> sqrt(2), include diagonal ends -> 2
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL 30

// Observation constants
#define MAX_ROAD_SEGMENT_OBSERVATIONS 128

// Maximum number of agents per scene
#ifndef MAX_AGENTS
#define MAX_AGENTS 32
#endif
// Active cars in a spawned fleet. Observation width stays MAX_AGENTS.
#define ACTOR_CAP 256
#define STOP_AGENT 1
#define REMOVE_AGENT 2

#define ROAD_FEATURES 7
#define ROAD_FEATURES_ONEHOT 13
#define PARTNER_FEATURES 7
// Four corners of the ego box, closest agents only. Center view stays 31 slots.
#define CORNER_VIEWS 4
#define CORNER_PARTNERS 16
#define DRIVE_CORNER_OBS (CORNER_VIEWS * CORNER_PARTNERS * PARTNER_FEATURES)
#define DRIVE_MAX_OBS(ego_dim)                                                                                         \
    ((ego_dim) + PARTNER_FEATURES * (MAX_AGENTS - 1) + DRIVE_CORNER_OBS +                                              \
     ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS)

// Ego features depend on dynamics model
#define EGO_FEATURES 12
#define EGO_FEATURES_JERK 15

#define LAMBDA_CONDITIONING_IDX 0
#define REWARD_COLLISION_IDX 1
#define REWARD_OFFROAD_COLLISION_IDX 2
#define REWARD_GOAL_IDX 3

// Observation normalization constants
#define MAX_SPEED 100.0f
#define MAX_VEH_LEN 30.0f
#define MAX_VEH_WIDTH 15.0f
#define MAX_VEH_HEIGHT 10.0f
#define MIN_REL_GOAL_COORD -1000.0f
#define MAX_REL_GOAL_COORD 1000.0f
#define MIN_REL_AGENT_POS -1000.0f
#define MAX_REL_AGENT_POS 1000.0f
#define MAX_ORIENTATION_RAD 2 * PI
#define MIN_RG_COORD -1000.0f
#define MAX_RG_COORD 1000.0f
#define MAX_ROAD_SCALE 100.0f
#define MAX_ROAD_SEGMENT_LENGTH 100.0f

// Goal behavior
#define GOAL_REMOVE 0
#define GOAL_GENERATE_NEW 1
#define GOAL_STOP 2

// Forward goal sampling for spawned fleets. Matches the pufferflow lane
// search: stitch lane ends within 1 m, stay 2 m short of the next car,
// and pick a point in the [goal_distance_min, goal_distance_max] band.
#define LANE_STITCH 1.0f
#define LANE_HEADING_DOT 0.5f
#define LANE_CAR_RADIUS 2.5f
#define GOAL_FOLLOW_GAP 2.0f
#define LANE_NODE_SNAP 4.0f
#define LANE_SUCC_MAX 8
#define SPAWN_SPEED_MAX 22.352f
#define SPAWN_CAR_LENGTH 4.3f
#define SPAWN_CAR_WIDTH 1.8f
#define METER_TO_MILE (1.0f / 1609.344f)

// Jerk action space (for JERK dynamics model)
static const float JERK_LONG[4] = {-15.0f, -4.0f, 0.0f, 4.0f};
static const float JERK_LAT[3] = {-4.0f, 0.0f, 4.0f};

// Classic action space. Same grid as pufferflow bicycle_integrate:
// two independent categoricals, accel ±16 m/s^2, steer ±0.70 rad,
// an odd count so the center bin is coast, and a ±45 m/s speed clamp.
#define NUM_ACCEL_BINS 257
#define ACCEL_MIN -16.0f
#define ACCEL_MAX 16.0f

#define NUM_STEER_BINS 257
#define STEER_MIN -0.70f
#define STEER_MAX 0.70f
#define ACCEL_COAST_BIN ((NUM_ACCEL_BINS - 1) / 2)
#define STEER_COAST_BIN ((NUM_STEER_BINS - 1) / 2)
#define DYN_SPEED_MAX 45.0f
#define NUM_DX_BINS 51
#define NUM_DY_BINS 51
#define NUM_YAW_BINS 127
#define DELTA_MAX_DX 3.5f
#define DELTA_MAX_DY 0.1f
#define DELTA_MAX_DYAW (M_PI / 6.0)

// Kinematic constraint parameters for delta-local model
#define A_LONG_MAX 8.0f    // m/s^2  — max longitudinal accel/brake
#define MAX_STEER 0.7f     // rad    — max effective steering angle
#define YAW_ACCEL_MAX 8.0f // rad/s^2 — max change in yaw rate per second

static int COLLECT_EXPERT_TELEPORT = 1;
static float ACCELERATION_VALUES[NUM_ACCEL_BINS];
static float STEERING_VALUES[NUM_STEER_BINS];
static float DELTA_DX_VALUES[NUM_DX_BINS];
static float DELTA_DY_VALUES[NUM_DY_BINS];
static float DELTA_YAW_VALUES[NUM_YAW_BINS];

static const float offsets[4][2] = {
    {-1, 1}, // top-left
    {1, 1},  // top-right
    {1, -1}, // bottom-right
    {-1, -1} // bottom-left
};

static const int collision_offsets[25][2] = {
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, // Top row
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1}, // Second row
    {-2, 0},  {-1, 0},  {0, 0},  {1, 0},  {2, 0},  // Middle row (including center)
    {-2, 1},  {-1, 1},  {0, 1},  {1, 1},  {2, 1},  // Fourth row
    {-2, 2},  {-1, 2},  {0, 2},  {1, 2},  {2, 2}   // Bottom row
};

#define MIN(a, b) ((a) < (b) ? (a) : (b))

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};
const Color LIGHTGREEN = (Color){152, 255, 152, 255};
const Color LIGHTYELLOW = (Color){255, 255, 152, 255};
const Color SOFT_YELLOW = (Color){245, 245, 220, 255};
const Color ROAD_COLOR = (Color){35, 35, 37, 255};
const Color LIGHTBLUE = (Color){167, 204, 255, 255};
const Color DEEPBLUE = (Color){45, 112, 226, 255};
const Color EXPERT_REPLAY = (Color){162, 220, 183, 255};
const Color EXPERT_REPLAY_SMALL = (Color){95, 112, 93, 255};
const Color LIGHT_ORANGE = (Color){255, 160, 80, 255};
const Color LIGHT_PURPLE = (Color){204, 204, 255, 255};

typedef struct Drive Drive;
typedef struct Client Client;
typedef struct Log Log;

struct Log {
    float episode_return;
    float episode_length;
    float score;
    float goals_reached_this_episode;
    float goals_sampled_this_episode;
    float offroad_rate;
    float collision_rate;
    float completion_rate;
    float offroad_per_agent;
    float collisions_per_agent;
    float dnf_rate;
    float speed_at_goal;
    float active_agent_count;
    float expert_static_agent_count;
    float static_agent_count;
    float perc_other;
    float route_progress; // Longitudinal continuous route completion
    float displacement_error_avg; // ADE
    float displacement_samples;
    float n;
    float failures_per_mile;
    // Cruise metrics. Steps are summed over the episode, then pooled.
    float speeding_steps;
    float cruise_steps;
    float ff_steps;
    float ff_gauss;
    float safe_cruising_score;
    // Sums of per-agent rates. vec_log divides by n; my_log divides by cruise_agents.
    float speeding_rate;
    float free_flow_efficiency;
    float cruise_agents;
};

typedef struct Entity Entity;
struct Entity {
    int scenario_id;
    int type;
    int id;
    int array_size;
    float *traj_x;
    float *traj_y;
    float *traj_z;
    float *traj_vx;
    float *traj_vy;
    float *traj_vz;
    float *traj_heading;
    int *traj_valid;
    float *expert_accel;
    float *expert_steering;
    float *expert_delta_x;
    float *expert_delta_y;
    float *expert_delta_yaw;
    float *inferred_traj_x;
    float *inferred_traj_y;
    float inferred_ade;
    float width;
    float length;
    float height;
    float goal_position_x;
    float goal_position_y;
    float goal_position_z;
    float init_goal_x;
    float init_goal_y;
    int mark_as_expert;
    int collision_state;
    float init_dist_to_goal; // Distance to goal at episode start (for progress normalization)
    int offroad_state;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float heading;
    float heading_x;
    float heading_y;
    int valid;
    int failure_before_goal;
    float goals_reached_this_episode;
    float goals_sampled_this_episode;
    int current_goal_reached;
    int active_agent;
    int controller;
    int stopped;
    int removed;
    float a_long;
    float a_lat;
    float jerk_long;
    float jerk_lat;
    float steering_angle;
    float wheelbase;
    // Lambda conditioning (human regularization)
    float lambda;
    // Reward conditioning
    float reward_collision_cond;
    float reward_offroad_cond;
    float reward_goal_cond;
    float prev_action_dx;
    float prev_action_dy;
    float prev_action_dyaw;
    float jerk_yaw;
    int lane_node;
    float distance_m;
    float failure_steps;
    // Private cruise setpoint in m/s. Not copied into partner observations.
    float cruise_target;
};

void free_entity(Entity *entity) {
    // free trajectory arrays
    free(entity->traj_x);
    free(entity->traj_y);
    free(entity->traj_z);
    free(entity->traj_vx);
    free(entity->traj_vy);
    free(entity->traj_vz);
    free(entity->traj_heading);
    free(entity->traj_valid);
    free(entity->inferred_traj_x);
    free(entity->inferred_traj_y);
    free(entity->expert_accel);
    free(entity->expert_steering);
    free(entity->expert_delta_x);
    free(entity->expert_delta_y);
    free(entity->expert_delta_yaw);
}

// Utility functions

static float gaussian_noise(float sigma) {
    if (sigma <= 0.0f)
        return 0.0f;
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sigma * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

float relative_distance_2d(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float distance = sqrtf(dx * dx + dy * dy);
    return distance;
}

float clip(float value, float min, float max) {
    if (value < min)
        return min;
    if (value > max)
        return max;
    return value;
}

float compute_route_progress_and_errors(Entity *e, float px, float py, int init_steps, int timestep,
                                        float *displacement_out) {
    if (e->traj_x == NULL || e->traj_valid == NULL) {
        *displacement_out = 0.0f;
        return 0.0f;
    }

    float cumulative = 0.0f;
    float d_p = 0.0f;  // arc length at init_steps
    float d_q = 0.0f;  // arc length at end (total)
    float d_xt = 0.0f; // arc length at closest point to (px, py)
    float best_dist_sq = 1e30f;
    float prev_x = e->traj_x[0], prev_y = e->traj_y[0];

    // Clamp timestep into the trajectory range for displacement lookup
    int t_lookup = timestep;
    if (t_lookup < 0)
        t_lookup = 0;
    if (t_lookup >= e->array_size)
        t_lookup = e->array_size - 1;

    for (int t = 0; t < e->array_size; t++) {
        if (!e->traj_valid[t]) {
            prev_x = e->traj_x[t];
            prev_y = e->traj_y[t];
            continue;
        }
        if (t > 0 && e->traj_valid[t - 1]) {
            float dx = e->traj_x[t] - prev_x;
            float dy = e->traj_y[t] - prev_y;
            cumulative += sqrtf(dx * dx + dy * dy);
        }
        if (t == init_steps)
            d_p = cumulative;
        d_q = cumulative;

        float dx = px - e->traj_x[t];
        float dy = py - e->traj_y[t];
        float dist_sq = dx * dx + dy * dy;
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            d_xt = cumulative;
        }
        prev_x = e->traj_x[t];
        prev_y = e->traj_y[t];
    }

    if (e->traj_valid[t_lookup] && e->traj_x[t_lookup] != INVALID_POSITION) {
        float ex = e->traj_x[t_lookup];
        float ey = e->traj_y[t_lookup];
        *displacement_out = sqrtf((px - ex) * (px - ex) + (py - ey) * (py - ey));
    } else {
        *displacement_out = -1.0f; // sentinel: invalid
    }

    float denom = d_q - d_p;
    if (denom < 1e-3f)
        return 0.0f;
    float ratio = (d_xt - d_p) / denom;
    return ratio < 0.0f ? 0.0f : ratio;
}

typedef struct GridMapEntity GridMapEntity;
struct GridMapEntity {
    int entity_idx;
    int geometry_idx;
};

typedef struct GridMap GridMap;
struct GridMap {
    float top_left_x;
    float top_left_y;
    float bottom_right_x;
    float bottom_right_y;
    int grid_cols;
    int grid_rows;
    int cell_size_x;
    int cell_size_y;
    int *cell_entities_count; // number of entities in each cell of the GridMap
    GridMapEntity **cells;    // list of gridEntities in each cell of the GridMap
    // Extras/Optimizations
    int vision_range;
    int *neighbor_cache_count;               // number of entities in each cells neighbor cache
    GridMapEntity **neighbor_cache_entities; // preallocated array to hold neighbor entities
};

struct Drive {
    Client *client;
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;
    unsigned char *truncations;
    Log log;
    Log *logs;
    int num_agents;
    int active_agent_count;
    int *active_agent_indices;
    int action_type;
    int human_agent_idx;
    Entity *entities;
    int num_entities;
    int num_actors;
    int num_objects;
    int num_roads;
    int static_agent_count;
    int *static_agent_indices;
    int expert_static_agent_count;
    int *expert_static_agent_indices;
    int timestep;
    int init_steps;
    int dynamics_model;
    GridMap *grid_map;
    int *neighbor_offsets;
    int episode_length;
    int termination_mode;
    char *map_name;
    float world_mean_x;
    float world_mean_y;
    float dt;
    int fix_rewards;
    float reward_goal;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    int fix_lambdas;
    float lambda_value;
    float goal_radius;
    int logs_capacity;
    int goal_behavior;
    float goal_target_distance;
    char *ini_file;
    char scenario_id[16];
    int collision_behavior;
    int offroad_behavior;
    int sdc_track_index;
    int num_tracks_to_predict;
    int *tracks_to_predict_indices;
    int init_mode;
    int control_mode;
    int sdc_controller;
    int non_sdc_controller;
    int non_vehicle_controller;
    int max_controlled_agents;
    int render_mode;
    // Noise configuration
    float dynamics_noise_pos;
    float dynamics_noise_heading;
    float obs_partner_noise_pos;
    float obs_partner_noise_speed;
    float obs_noise_road;
    bool async_resets;
    int spawn_agents;
    float goal_distance_min;
    float goal_distance_max;
    // Stage 2 cruise task. Weight 0 leaves the goal reward unchanged.
    float target_speed;
    float speed_reward_weight;
    float speed_reward_sigma;
    float accel_penalty;
    int terminate_on_failure;
    int lane_n;
    int lane_polys;
    float *lane_x;
    float *lane_y;
    float *lane_hx;
    float *lane_hy;
    float *lane_dist;
    float *lane_block;
    float *lane_heap_d;
    int *lane_next;
    int *lane_poly_start;
    int *lane_heap_i;
    int *lane_order;
    int lane_heap_cap;
};

void add_log(Drive *env) {
    for (int i = 0; i < env->active_agent_count; i++) {
        Entity *e = &env->entities[env->active_agent_indices[i]];

        env->log.goals_reached_this_episode += e->goals_reached_this_episode;
        env->log.goals_sampled_this_episode += e->goals_sampled_this_episode;
        env->log.offroad_rate += env->logs[i].offroad_rate;
        env->log.collision_rate += env->logs[i].collision_rate;
        env->log.offroad_per_agent += env->logs[i].offroad_per_agent;
        env->log.collisions_per_agent += env->logs[i].collisions_per_agent;

        // Route progress ratio
        if (env->logs[i].route_progress > 0.0f) {
            // Already set to 1.0 because agent reached goal
            env->log.route_progress += env->logs[i].route_progress;
        } else {
            // Agent timed out without reaching goal: compute from final position
            float unused_disp = 0.0f;
            env->log.route_progress += compute_route_progress_and_errors(e, e->x, e->y, env->init_steps, env->timestep,
                                                                         &unused_disp);
        }

        if (env->logs[i].displacement_samples > 0.0f) {
            env->log.displacement_error_avg += env->logs[i].displacement_error_avg / env->logs[i].displacement_samples;
        }

        float frac_goal_reached = e->goals_reached_this_episode / e->goals_sampled_this_episode;
        float threshold = 1.0f;
        if (e->goals_sampled_this_episode > 1) {
            threshold = (e->goals_sampled_this_episode - 1.0f) / e->goals_sampled_this_episode;
        }
        if (frac_goal_reached >= threshold && !e->failure_before_goal) {
            env->log.score += 1.0f;
        }
        if (!e->failure_before_goal && frac_goal_reached < 1.0f) {
            env->log.dnf_rate += 1.0f;
        }
        env->log.speed_at_goal += env->logs[i].speed_at_goal;
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        float miles = e->distance_m * METER_TO_MILE;
        if (miles > 1.0e-3f) {
            env->log.failures_per_mile += e->failure_steps / miles;
        }

        env->log.speeding_steps += env->logs[i].speeding_steps;
        env->log.cruise_steps += env->logs[i].cruise_steps;
        env->log.ff_steps += env->logs[i].ff_steps;
        env->log.ff_gauss += env->logs[i].ff_gauss;
        float cruise_n = env->logs[i].cruise_steps;
        if (cruise_n > 0.0f) {
            float speeding_i = env->logs[i].speeding_steps / cruise_n;
            // Blocked for the whole episode: no free-flow step, score the match as 1.
            float freeflow_i =
                env->logs[i].ff_steps > 0.0f ? env->logs[i].ff_gauss / env->logs[i].ff_steps : 1.0f;
            int incident_free = env->logs[i].collision_rate == 0.0f && env->logs[i].offroad_rate == 0.0f;
            float safe_i = incident_free ? (1.0f - speeding_i) * freeflow_i : 0.0f;
            if (safe_i < 0.0f)
                safe_i = 0.0f;
            env->log.speeding_rate += speeding_i;
            env->log.free_flow_efficiency += freeflow_i;
            env->log.safe_cruising_score += safe_i;
            env->log.cruise_agents += 1.0f;
        }

        env->log.active_agent_count += env->active_agent_count;
        env->log.expert_static_agent_count += env->expert_static_agent_count;
        env->log.static_agent_count += env->static_agent_count;
        int total = env->active_agent_count + env->static_agent_count;
        env->log.perc_other += (float)env->static_agent_count / (float)total;
        env->log.n += 1;
    }
}
void init_action_space() {
    // Classic discrete action space. An even count has no zero command.
    assert(NUM_ACCEL_BINS % 2 == 1);
    assert(NUM_STEER_BINS % 2 == 1);
    float accel_step = (ACCEL_MAX - ACCEL_MIN) / (float)(NUM_ACCEL_BINS - 1);
    for (int i = 0; i < NUM_ACCEL_BINS; i++) {
        ACCELERATION_VALUES[i] = ACCEL_MIN + i * accel_step;
    }

    float steer_step = (STEER_MAX - STEER_MIN) / (float)(NUM_STEER_BINS - 1);
    for (int i = 0; i < NUM_STEER_BINS; i++) {
        STEERING_VALUES[i] = STEER_MIN + i * steer_step;
    }
    ACCELERATION_VALUES[ACCEL_COAST_BIN] = 0.0f;
    STEERING_VALUES[STEER_COAST_BIN] = 0.0f;
    // Delta-local discrete action space
    float dx_step = (2.0f * DELTA_MAX_DX) / (NUM_DX_BINS - 1);
    for (int i = 0; i < NUM_DX_BINS; i++) {
        DELTA_DX_VALUES[i] = -DELTA_MAX_DX + i * dx_step;
    }
    float dy_step = (2.0f * DELTA_MAX_DY) / (NUM_DY_BINS - 1);
    for (int i = 0; i < NUM_DY_BINS; i++) {
        DELTA_DY_VALUES[i] = -DELTA_MAX_DY + i * dy_step;
    }
    float yaw_step = (2.0f * DELTA_MAX_DYAW) / (NUM_YAW_BINS - 1);
    for (int i = 0; i < NUM_YAW_BINS; i++) {
        DELTA_YAW_VALUES[i] = -DELTA_MAX_DYAW + i * yaw_step;
    }
}

Entity *load_map_binary(const char *filename, Drive *env) {
    FILE *file = fopen(filename, "rb");
    if (!file)
        return NULL;

    // Read scenario_id
    fread(env->scenario_id, sizeof(char), 16, file);

    // Read sdc_track_index
    fread(&env->sdc_track_index, sizeof(int), 1, file);

    // Read tracks_to_predict
    fread(&env->num_tracks_to_predict, sizeof(int), 1, file);
    if (env->num_tracks_to_predict > 0) {
        env->tracks_to_predict_indices = (int *)malloc(env->num_tracks_to_predict * sizeof(int));

        for (int i = 0; i < env->num_tracks_to_predict; i++) {
            fread(&env->tracks_to_predict_indices[i], sizeof(int), 1, file);
        }
    } else {
        env->tracks_to_predict_indices = NULL;
    }

    fread(&env->num_objects, sizeof(int), 1, file);
    fread(&env->num_roads, sizeof(int), 1, file);
    env->num_entities = env->num_objects + env->num_roads;
    Entity *entities = (Entity *)malloc(env->num_entities * sizeof(Entity));

    for (int i = 0; i < env->num_entities; i++) {
        // Read base entity data
        fread(&entities[i].scenario_id, sizeof(int), 1, file);
        fread(&entities[i].type, sizeof(int), 1, file);
        fread(&entities[i].id, sizeof(int), 1, file);
        fread(&entities[i].array_size, sizeof(int), 1, file);
        // Allocate arrays based on type
        int size = entities[i].array_size;
        entities[i].traj_x = (float *)malloc(size * sizeof(float));
        entities[i].traj_y = (float *)malloc(size * sizeof(float));
        entities[i].traj_z = (float *)malloc(size * sizeof(float));
        if (entities[i].type == VEHICLE || entities[i].type == PEDESTRIAN ||
            entities[i].type == CYCLIST) { // Object type
            // Allocate arrays for object-specific data
            entities[i].traj_vx = (float *)malloc(size * sizeof(float));
            entities[i].traj_vy = (float *)malloc(size * sizeof(float));
            entities[i].traj_vz = (float *)malloc(size * sizeof(float));
            entities[i].traj_heading = (float *)malloc(size * sizeof(float));
            entities[i].traj_valid = (int *)malloc(size * sizeof(int));
            entities[i].inferred_traj_x = (float *)calloc(size, sizeof(float));
            entities[i].inferred_traj_y = (float *)calloc(size, sizeof(float));
            entities[i].expert_accel = (float *)malloc(size * sizeof(float));
            entities[i].expert_steering = (float *)malloc(size * sizeof(float));
            entities[i].expert_delta_x = (float *)malloc(size * sizeof(float));
            entities[i].expert_delta_y = (float *)malloc(size * sizeof(float));
            entities[i].expert_delta_yaw = (float *)malloc(size * sizeof(float));
        } else {
            // Roads don't use these arrays
            entities[i].traj_vx = NULL;
            entities[i].traj_vy = NULL;
            entities[i].traj_vz = NULL;
            entities[i].traj_heading = NULL;
            entities[i].traj_valid = NULL;
            entities[i].inferred_traj_x = NULL;
            entities[i].inferred_traj_y = NULL;
            entities[i].expert_accel = NULL;
            entities[i].expert_steering = NULL;
            entities[i].expert_delta_x = NULL;
            entities[i].expert_delta_y = NULL;
            entities[i].expert_delta_yaw = NULL;
        }
        // Read array data
        fread(entities[i].traj_x, sizeof(float), size, file);
        fread(entities[i].traj_y, sizeof(float), size, file);
        fread(entities[i].traj_z, sizeof(float), size, file);
        if (entities[i].type == VEHICLE || entities[i].type == PEDESTRIAN ||
            entities[i].type == CYCLIST) { // Object type
            fread(entities[i].traj_vx, sizeof(float), size, file);
            fread(entities[i].traj_vy, sizeof(float), size, file);
            fread(entities[i].traj_vz, sizeof(float), size, file);
            fread(entities[i].traj_heading, sizeof(float), size, file);
            fread(entities[i].traj_valid, sizeof(int), size, file);
            fread(entities[i].expert_accel, sizeof(float), size, file);
            fread(entities[i].expert_steering, sizeof(float), size, file);
            fread(entities[i].expert_delta_x, sizeof(float), size, file);
            fread(entities[i].expert_delta_y, sizeof(float), size, file);
            fread(entities[i].expert_delta_yaw, sizeof(float), size, file);
        }
        // Read remaining scalar fields
        fread(&entities[i].width, sizeof(float), 1, file);
        fread(&entities[i].length, sizeof(float), 1, file);
        fread(&entities[i].height, sizeof(float), 1, file);
        fread(&entities[i].goal_position_x, sizeof(float), 1, file);
        fread(&entities[i].goal_position_y, sizeof(float), 1, file);
        fread(&entities[i].goal_position_z, sizeof(float), 1, file);
        fread(&entities[i].mark_as_expert, sizeof(int), 1, file);
    }

    fclose(file);
    return entities;
}

void set_start_position(Drive *env) {
    for (int i = 0; i < env->num_entities; i++) {
        int is_active = 0;
        for (int j = 0; j < env->active_agent_count; j++) {
            if (env->active_agent_indices[j] == i) {
                is_active = 1;
                break;
            }
        }
        Entity *e = &env->entities[i];

        // Clamp init_steps to ensure we don't go out of bounds
        int step = env->init_steps;
        if (step >= e->array_size)
            step = e->array_size - 1;
        if (step < 0)
            step = 0;

        e->x = e->traj_x[step];
        e->y = e->traj_y[step];
        e->z = e->traj_z[step];
        if (e->type > CYCLIST || e->type == 0) {
            continue;
        }
        if (is_active == 0) {
            e->vx = 0;
            e->vy = 0;
            e->vz = 0;
            e->failure_before_goal = 0;
        } else {
            e->vx = e->traj_vx[step];
            e->vy = e->traj_vy[step];
            e->vz = e->traj_vz[step];
        }
        e->heading = e->traj_heading[step];
        e->heading_x = cosf(e->heading);
        e->heading_y = sinf(e->heading);
        e->valid = e->traj_valid[step];
        e->collision_state = 0;
        e->offroad_state = 0;
        e->stopped = 0;
        e->removed = 0;

        // Dynamics
        e->a_long = 0.0f;
        e->a_lat = 0.0f;
        e->jerk_long = 0.0f;
        e->jerk_lat = 0.0f;
        e->steering_angle = 0.0f;
        e->wheelbase = e->length;
        // Initialize prev_action_* from the agent's actual starting motion so the
        // kinematic constraints don't artificially throttle step 1.
        float speed_init = e->vx * e->heading_x + e->vy * e->heading_y;
        e->prev_action_dx = speed_init * env->dt;
        e->prev_action_dy = 0.0f;
        e->prev_action_dyaw = 0.0f;
    }
}

int getGridIndex(Drive *env, float x1, float y1) {
    if (env->grid_map->top_left_x >= env->grid_map->bottom_right_x ||
        env->grid_map->bottom_right_y >= env->grid_map->top_left_y) {
        return -1; // Invalid grid coordinates
    }

    float relativeX = x1 - env->grid_map->top_left_x;     // Distance from left
    float relativeY = y1 - env->grid_map->bottom_right_y; // Distance from bottom
    int gridX = (int)(relativeX / GRID_CELL_SIZE);        // Column index
    int gridY = (int)(relativeY / GRID_CELL_SIZE);        // Row index
    if (gridX < 0 || gridX >= env->grid_map->grid_cols || gridY < 0 || gridY >= env->grid_map->grid_rows) {
        return -1; // Return -1 for out of bounds
    }
    int index = (gridY * env->grid_map->grid_cols) + gridX;
    return index;
}

void add_entity_to_grid(Drive *env, int grid_index, int entity_idx, int geometry_idx, int *cell_entities_insert_index) {
    if (grid_index == -1) {
        return;
    }

    int count = cell_entities_insert_index[grid_index];
    if (count >= env->grid_map->cell_entities_count[grid_index]) {
        printf("Error: Exceeded precomputed entity count for grid cell %d. Current count: %d, Max count(Precomputed): "
               "%d\n",
               grid_index, count, env->grid_map->cell_entities_count[grid_index]);
        return;
    }

    env->grid_map->cells[grid_index][count].entity_idx = entity_idx;
    env->grid_map->cells[grid_index][count].geometry_idx = geometry_idx;
    cell_entities_insert_index[grid_index] = count + 1;
}

void init_grid_map(Drive *env) {
    // Allocate memory for the grid map structure
    env->grid_map = (GridMap *)malloc(sizeof(GridMap));

    // Find top left and bottom right points of the map
    float top_left_x;
    float top_left_y;
    float bottom_right_x;
    float bottom_right_y;
    int first_valid_point = 0;
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type > 3 && env->entities[i].type < 7) {
            // Check all points in the trajectory for road elements
            Entity *e = &env->entities[i];
            for (int j = 0; j < e->array_size; j++) {
                if (e->traj_x[j] == INVALID_POSITION)
                    continue;
                if (e->traj_y[j] == INVALID_POSITION)
                    continue;
                if (!first_valid_point) {
                    top_left_x = bottom_right_x = e->traj_x[j];
                    top_left_y = bottom_right_y = e->traj_y[j];
                    first_valid_point = true;
                    continue;
                }
                if (e->traj_x[j] < top_left_x)
                    top_left_x = e->traj_x[j];
                if (e->traj_x[j] > bottom_right_x)
                    bottom_right_x = e->traj_x[j];
                if (e->traj_y[j] > top_left_y)
                    top_left_y = e->traj_y[j];
                if (e->traj_y[j] < bottom_right_y)
                    bottom_right_y = e->traj_y[j];
            }
        }
    }

    env->grid_map->top_left_x = top_left_x;
    env->grid_map->top_left_y = top_left_y;
    env->grid_map->bottom_right_x = bottom_right_x;
    env->grid_map->bottom_right_y = bottom_right_y;
    env->grid_map->cell_size_x = GRID_CELL_SIZE;
    env->grid_map->cell_size_y = GRID_CELL_SIZE;

    // Calculate grid dimensions
    float grid_width = bottom_right_x - top_left_x;
    float grid_height = top_left_y - bottom_right_y;
    env->grid_map->grid_cols = ceil(grid_width / GRID_CELL_SIZE);
    env->grid_map->grid_rows = ceil(grid_height / GRID_CELL_SIZE);
    int grid_cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->cells = (GridMapEntity **)calloc(grid_cell_count, sizeof(GridMapEntity *));
    env->grid_map->cell_entities_count = (int *)calloc(grid_cell_count, sizeof(int));

    // Calculate number of entities in each grid cell
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type > 3 && env->entities[i].type < 7) {
            for (int j = 0; j < env->entities[i].array_size - 1; j++) {
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j + 1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j + 1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                env->grid_map->cell_entities_count[grid_index]++;
            }
        }
    }
    int cell_entities_insert_index[grid_cell_count]; // Helper array for insertion index
    memset(cell_entities_insert_index, 0, grid_cell_count * sizeof(int));

    // Initialize grid cells
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        env->grid_map->cells[grid_index] =
            (GridMapEntity *)calloc(env->grid_map->cell_entities_count[grid_index], sizeof(GridMapEntity));
    }
    for (int i = 0; i < grid_cell_count; i++) {
        if (cell_entities_insert_index[i] != 0) {
            printf("Error: cell_entities_insert_index[%d] not zero during initialization.\n", i);
            cell_entities_insert_index[i] = 0;
        }
    }

    // Populate grid cells
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type > 3 &&
            env->entities[i].type < 7) { // NOTE: Only Road Edges, Lines, and Lanes in grid map
            for (int j = 0; j < env->entities[i].array_size - 1; j++) {
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j + 1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j + 1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                add_entity_to_grid(env, grid_index, i, j, cell_entities_insert_index);
            }
        }
    }
}

void init_neighbor_offsets(Drive *env) {
    // Allocate memory for the offsets
    env->neighbor_offsets = (int *)calloc(env->grid_map->vision_range * env->grid_map->vision_range * 2, sizeof(int));
    // neighbor offsets in a spiral pattern
    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0;                  // Current x offset
    int y = 0;                  // Current y offset
    int dir = 0;                // Current direction (0: right, 1: up, 2: left, 3: down)
    int steps_to_take = 1;      // Number of steps in current direction
    int steps_taken = 0;        // Steps taken in current direction
    int segments_completed = 0; // Count of direction segments completed
    int total = 0;              // Total offsets added
    int max_offsets = env->grid_map->vision_range * env->grid_map->vision_range;
    // Start at center (0,0)
    int curr_idx = 0;
    env->neighbor_offsets[curr_idx++] = 0; // x offset
    env->neighbor_offsets[curr_idx++] = 0; // y offset
    total++;
    // Generate spiral pattern
    while (total < max_offsets) {
        // Move in current direction
        x += dx[dir];
        y += dy[dir];
        // Only add if within vision range bounds
        if (abs(x) <= env->grid_map->vision_range / 2 && abs(y) <= env->grid_map->vision_range / 2) {
            env->neighbor_offsets[curr_idx++] = x;
            env->neighbor_offsets[curr_idx++] = y;
            total++;
        }
        steps_taken++;
        // Check if we need to change direction
        if (steps_taken != steps_to_take)
            continue;
        steps_taken = 0;     // Reset steps taken
        dir = (dir + 1) % 4; // Change direction (clockwise: right->up->left->down)
        segments_completed++;
        // Increase step length every two direction changes
        if (segments_completed % 2 == 0) {
            steps_to_take++;
        }
    }
}

void cache_neighbor_offsets(Drive *env) {
    int count = 0;
    int cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->neighbor_cache_entities = (GridMapEntity **)calloc(cell_count, sizeof(GridMapEntity *));
    env->grid_map->neighbor_cache_count = (int *)calloc(cell_count + 1, sizeof(int));
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int current_cell_neighbor_count = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows)
                continue;
            int grid_count = env->grid_map->cell_entities_count[grid_index];
            current_cell_neighbor_count += grid_count;
        }
        env->grid_map->neighbor_cache_count[i] = current_cell_neighbor_count;
        count += current_cell_neighbor_count;
        if (current_cell_neighbor_count == 0) {
            env->grid_map->neighbor_cache_entities[i] = NULL;
            continue;
        }
        env->grid_map->neighbor_cache_entities[i] =
            (GridMapEntity *)calloc(current_cell_neighbor_count, sizeof(GridMapEntity));
    }

    env->grid_map->neighbor_cache_count[cell_count] = count;
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int base_index = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows)
                continue;
            int grid_count = env->grid_map->cell_entities_count[grid_index];

            // Skip if no entities or source is NULL
            if (grid_count == 0 || env->grid_map->cells[grid_index] == NULL) {
                continue;
            }

            int src_idx = grid_index;
            int dst_idx = base_index;
            // Copy grid_count pairs (entity_idx, geometry_idx) at once
            memcpy(&env->grid_map->neighbor_cache_entities[i][dst_idx], env->grid_map->cells[src_idx],
                   grid_count * sizeof(GridMapEntity));
            base_index += grid_count;
        }
    }
}

int get_neighbor_cache_entities(Drive *env, int cell_idx, GridMapEntity *entities, int max_entities) {
    GridMap *grid_map = env->grid_map;
    if (cell_idx < 0 || cell_idx >= (grid_map->grid_cols * grid_map->grid_rows)) {
        return 0; // Invalid cell index
    }

    int count = grid_map->neighbor_cache_count[cell_idx];
    // Limit to available space
    if (count > max_entities) {
        count = max_entities;
    }
    memcpy(entities, grid_map->neighbor_cache_entities[cell_idx], count * sizeof(GridMapEntity));
    return count;
}

void set_means(Drive *env) {
    float mean_x = 0.0f;
    float mean_y = 0.0f;
    int64_t point_count = 0;

    // Compute single mean for all entities (vehicles and roads)
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN ||
            env->entities[i].type == CYCLIST) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                // Assume a validity flag exists (e.g., valid[j]); adjust if not available
                if (env->entities[i].traj_valid[j]) { // Add validity check if applicable
                    point_count++;
                    mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
                    mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
                }
            }
        } else if (env->entities[i].type >= 4) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                point_count++;
                mean_x += (env->entities[i].traj_x[j] - mean_x) / point_count;
                mean_y += (env->entities[i].traj_y[j] - mean_y) / point_count;
            }
        }
    }
    env->world_mean_x = mean_x;
    env->world_mean_y = mean_y;
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN ||
            env->entities[i].type == CYCLIST || env->entities[i].type >= 4) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                if (env->entities[i].traj_x[j] == INVALID_POSITION)
                    continue;
                env->entities[i].traj_x[j] -= mean_x;
                env->entities[i].traj_y[j] -= mean_y;
            }
            env->entities[i].goal_position_x -= mean_x;
            env->entities[i].goal_position_y -= mean_y;
        }
    }
}

void move_expert(Drive *env, float *actions, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    int t = env->timestep;
    if (t < 0 || t >= agent->array_size) {
        agent->x = INVALID_POSITION;
        agent->y = INVALID_POSITION;
        agent->z = 0.0f;
        agent->heading = 0.0f;
        agent->heading_x = 1.0f;
        agent->heading_y = 0.0f;
        return;
    }
    if (agent->traj_valid && agent->traj_valid[t] == 0) {
        agent->x = INVALID_POSITION;
        agent->y = INVALID_POSITION;
        agent->z = 0.0f;
        agent->heading = 0.0f;
        agent->heading_x = 1.0f;
        agent->heading_y = 0.0f;
        return;
    }
    agent->x = agent->traj_x[t];
    agent->y = agent->traj_y[t];
    agent->z = agent->traj_z[t];
    agent->heading = agent->traj_heading[t];
    agent->heading_x = cosf(agent->heading);
    agent->heading_y = sinf(agent->heading);

    if (agent->traj_vx != NULL) {
        agent->vx = agent->traj_vx[t];
        agent->vy = agent->traj_vy[t];
        agent->vz = agent->traj_vz[t];
    } else if (t > 0 && env->dt > 0.0f) {
        agent->vx = (agent->traj_x[t] - agent->traj_x[t - 1]) / env->dt;
        agent->vy = (agent->traj_y[t] - agent->traj_y[t - 1]) / env->dt;
        agent->vz = (agent->traj_z[t] - agent->traj_z[t - 1]) / env->dt;
    }
}

bool check_line_intersection(float p1[2], float p2[2], float q1[2], float q2[2]) {
    if (fmax(p1[0], p2[0]) < fmin(q1[0], q2[0]) || fmin(p1[0], p2[0]) > fmax(q1[0], q2[0]) ||
        fmax(p1[1], p2[1]) < fmin(q1[1], q2[1]) || fmin(p1[1], p2[1]) > fmax(q1[1], q2[1]))
        return false;

    // Calculate vectors
    float dx1 = p2[0] - p1[0];
    float dy1 = p2[1] - p1[1];
    float dx2 = q2[0] - q1[0];
    float dy2 = q2[1] - q1[1];

    // Calculate cross products
    float cross = dx1 * dy2 - dy1 * dx2;

    // If lines are parallel
    if (cross == 0)
        return false;

    // Calculate relative vectors between start points
    float dx3 = p1[0] - q1[0];
    float dy3 = p1[1] - q1[1];

    // Calculate parameters for intersection point
    float s = (dx1 * dy3 - dy1 * dx3) / cross;
    float t = (dx2 * dy3 - dy2 * dx3) / cross;

    // Check if intersection point lies within both line segments
    return (s >= 0 && s <= 1 && t >= 0 && t <= 1);
}

int checkNeighbors(Drive *env, float x, float y, GridMapEntity *entity_list, int max_size,
                   const int (*local_offsets)[2], int offset_size) {
    // Get the grid index for the given position (x, y)
    int index = getGridIndex(env, x, y);
    if (index == -1)
        return 0; // Return 0 size if position invalid
    // Calculate 2D grid coordinates
    int cellsX = env->grid_map->grid_cols;
    int gridX = index % cellsX;
    int gridY = index / cellsX;
    int entity_list_count = 0;
    // Fill the provided array
    for (int i = 0; i < offset_size; i++) {
        int nx = gridX + local_offsets[i][0];
        int ny = gridY + local_offsets[i][1];
        // Ensure the neighbor is within grid bounds
        if (nx < 0 || nx >= env->grid_map->grid_cols || ny < 0 || ny >= env->grid_map->grid_rows)
            continue;
        int neighborIndex = ny * env->grid_map->grid_cols + nx;
        int count = env->grid_map->cell_entities_count[neighborIndex];
        // Add entities from this cell to the list
        for (int j = 0; j < count && entity_list_count < max_size; j++) {
            int entityId = env->grid_map->cells[neighborIndex][j].entity_idx;
            int geometry_idx = env->grid_map->cells[neighborIndex][j].geometry_idx;
            entity_list[entity_list_count].entity_idx = entityId;
            entity_list[entity_list_count].geometry_idx = geometry_idx;
            entity_list_count += 1;
        }
    }
    return entity_list_count;
}

int check_aabb_collision(Entity *car1, Entity *car2) {
    // Get car corners in world space
    float cos1 = car1->heading_x;
    float sin1 = car1->heading_y;
    float cos2 = car2->heading_x;
    float sin2 = car2->heading_y;

    // Calculate half dimensions
    float half_len1 = car1->length * 0.5f;
    float half_width1 = car1->width * 0.5f;
    float half_len2 = car2->length * 0.5f;
    float half_width2 = car2->width * 0.5f;

    // Calculate car1's corners in world space
    float car1_corners[4][2] = {
        {car1->x + (half_len1 * cos1 - half_width1 * sin1), car1->y + (half_len1 * sin1 + half_width1 * cos1)},
        {car1->x + (half_len1 * cos1 + half_width1 * sin1), car1->y + (half_len1 * sin1 - half_width1 * cos1)},
        {car1->x + (-half_len1 * cos1 - half_width1 * sin1), car1->y + (-half_len1 * sin1 + half_width1 * cos1)},
        {car1->x + (-half_len1 * cos1 + half_width1 * sin1), car1->y + (-half_len1 * sin1 - half_width1 * cos1)}};

    // Calculate car2's corners in world space
    float car2_corners[4][2] = {
        {car2->x + (half_len2 * cos2 - half_width2 * sin2), car2->y + (half_len2 * sin2 + half_width2 * cos2)},
        {car2->x + (half_len2 * cos2 + half_width2 * sin2), car2->y + (half_len2 * sin2 - half_width2 * cos2)},
        {car2->x + (-half_len2 * cos2 - half_width2 * sin2), car2->y + (-half_len2 * sin2 + half_width2 * cos2)},
        {car2->x + (-half_len2 * cos2 + half_width2 * sin2), car2->y + (-half_len2 * sin2 - half_width2 * cos2)}};

    // Get the axes to check (normalized vectors perpendicular to each edge)
    float axes[4][2] = {
        {cos1, sin1},  // Car1's length axis
        {-sin1, cos1}, // Car1's width axis
        {cos2, sin2},  // Car2's length axis
        {-sin2, cos2}  // Car2's width axis
    };

    // Check each axis
    for (int i = 0; i < 4; i++) {
        float min1 = INFINITY, max1 = -INFINITY;
        float min2 = INFINITY, max2 = -INFINITY;

        // Project car1's corners onto the axis
        for (int j = 0; j < 4; j++) {
            float proj = car1_corners[j][0] * axes[i][0] + car1_corners[j][1] * axes[i][1];
            min1 = fminf(min1, proj);
            max1 = fmaxf(max1, proj);
        }

        // Project car2's corners onto the axis
        for (int j = 0; j < 4; j++) {
            float proj = car2_corners[j][0] * axes[i][0] + car2_corners[j][1] * axes[i][1];
            min2 = fminf(min2, proj);
            max2 = fmaxf(max2, proj);
        }

        // If there's a gap on this axis, the boxes don't intersect
        if (max1 < min2 || min1 > max2) {
            return 0; // No collision
        }
    }

    // If we get here, there's no separating axis, so the boxes intersect
    return 1; // Collision
}

int collision_check(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];

    if (agent->x == INVALID_POSITION)
        return -1;

    // Skip collision checking for pedestrians because they are often too
    // close to other entities at initialization.
    if (agent->type == PEDESTRIAN)
        return -1;

    int car_collided_with_index = -1;

    if (agent->removed == 1)
        return car_collided_with_index; // Skip removed entities

    // Waymo keeps the 32-slot scan. A spawned fleet has to see every car.
    int scan_limit = MAX_AGENTS;
    if (env->spawn_agents > 0)
        scan_limit = env->active_agent_count;
    for (int i = 0; i < scan_limit; i++) {
        int index = -1;
        if (i < env->active_agent_count) {
            index = env->active_agent_indices[i];
        } else if (env->spawn_agents <= 0 && i < env->num_actors && env->static_agent_count > 0) {
            index = env->static_agent_indices[i - env->active_agent_count];
        }
        if (index == -1)
            continue;
        if (index == agent_idx)
            continue;
        Entity *entity = &env->entities[index];
        if (entity->removed == 1)
            continue; // Skip removed entities
        float x1 = entity->x;
        float y1 = entity->y;
        float dist = ((x1 - agent->x) * (x1 - agent->x) + (y1 - agent->y) * (y1 - agent->y));
        if (dist > 225.0f)
            continue;
        if (check_aabb_collision(agent, entity)) {
            car_collided_with_index = index;
            break;
        }
    }

    return car_collided_with_index;
}

void compute_agent_metrics(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];

    if (agent->x == INVALID_POSITION)
        return; // invalid agent position

    int collided_with_agent = 0;
    int collided_with_road = 0;

    // Get agent bb measurements
    float half_length = agent->length / 2.0f;
    float half_width = agent->width / 2.0f;
    float cos_heading = cosf(agent->heading);
    float sin_heading = sinf(agent->heading);
    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] =
            agent->x + (offsets[i][0] * half_length * cos_heading - offsets[i][1] * half_width * sin_heading);
        corners[i][1] =
            agent->y + (offsets[i][0] * half_length * sin_heading + offsets[i][1] * half_width * cos_heading);
    }

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25]; // Array big enough for all neighboring cells
    int list_size =
        checkNeighbors(env, agent->x, agent->y, entity_list, MAX_ENTITIES_PER_CELL * 25, collision_offsets, 25);

    // Iterate through road entities and check for off-road event
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_idx == -1)
            continue;
        if (entity_list[i].entity_idx == agent_idx)
            continue;
        Entity *entity;
        entity = &env->entities[entity_list[i].entity_idx];

        // Check for offroad collision with road edges (only for vehicles and cyclists)
        if (entity->type == ROAD_EDGE && agent->type != PEDESTRIAN) {
            int geometry_idx = entity_list[i].geometry_idx;
            float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
            float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};
            for (int k = 0; k < 4; k++) { // Check each edge of the bounding box
                int next = (k + 1) % 4;
                if (check_line_intersection(corners[k], corners[next], start, end)) {
                    collided_with_road = 1;
                    break;
                }
            }
        }

        if (collided_with_road == 1)
            break;
    }
    // Check for collisions with other agents (skip for pedestrians)
    int car_collided_with_index = -1;
    if (agent->type != PEDESTRIAN) {
        car_collided_with_index = collision_check(env, agent_idx);
        if (car_collided_with_index != -1) {
            collided_with_agent = 1;
            agent->collision_state = 1;
        }
    }

    if (collided_with_agent) {
        if (env->collision_behavior == STOP_AGENT && !agent->stopped) {
            agent->stopped = 1;
            agent->vx = agent->vy = 0.0f;
        } else if (env->collision_behavior == REMOVE_AGENT && !agent->removed) {
            Entity *agent_collided = &env->entities[car_collided_with_index];
            agent->removed = 1;
            agent_collided->removed = 1;
            agent->x = agent->y = -10000.0f;
            agent_collided->x = agent_collided->y = -10000.0f;
        }
    }
    if (collided_with_road) {
        agent->offroad_state = 1;
        if (env->offroad_behavior == STOP_AGENT && !agent->stopped) {
            agent->stopped = 1;
            agent->vx = agent->vy = 0.0f;
        } else if (env->offroad_behavior == REMOVE_AGENT && !agent->removed) {
            agent->removed = 1;
            agent->x = agent->y = -10000.0f;
        }
    }

    return;
}

bool should_control_agent(Drive *env, int agent_idx, int control_limit) {
    // Check if we have room for more agents or are already at capacity
    if (env->active_agent_count >= control_limit) {
        return false;
    }

    Entity *entity = &env->entities[agent_idx];

    // SDC-only modes: only the SDC entity is policy/inferred-action controlled.
    // INFERRED_EXPERT_ACTIONS now drives only the SDC; everyone else replays logs.
    if (env->control_mode == CONTROL_SDC_ONLY || env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS) {
        return agent_idx == env->sdc_track_index;
    }

    bool is_vehicle = (entity->type == VEHICLE);
    bool is_ped_or_bike = (entity->type == PEDESTRIAN || entity->type == CYCLIST);
    bool type_is_valid = false;

    switch (env->control_mode) {
    case CONTROL_WOSAC:
    case CONTROL_REPLAY_LOGS:
        // Valid types only, ignore expert flag and goal distance
        return (is_vehicle || is_ped_or_bike);

    case CONTROL_VEHICLES:
        type_is_valid = is_vehicle;
        break;

    default:
        type_is_valid = (is_vehicle || is_ped_or_bike);
        break;
    }

    // Filter invalid types or experts
    if (!type_is_valid || entity->mark_as_expert) {
        return false;
    }

    // Check distance to goal in agent's local frame
    float cos_heading = cosf(entity->traj_heading[0]);
    float sin_heading = sinf(entity->traj_heading[0]);
    float goal_dx = entity->goal_position_x - entity->traj_x[0];
    float goal_dy = entity->goal_position_y - entity->traj_y[0];

    // Transform to agent's local frame
    float local_goal_x = goal_dx * cos_heading + goal_dy * sin_heading;
    float local_goal_y = -goal_dx * sin_heading + goal_dy * cos_heading;
    float distance_to_goal = relative_distance_2d(0, 0, local_goal_x, local_goal_y);

    return distance_to_goal >= MIN_DISTANCE_TO_GOAL;
}

static int resolve_agent_controller(Drive *env, int agent_idx, int is_active, int replay_by_default) {
    if (env->control_mode == CONTROL_REPLAY_LOGS) {
        return CONTROLLER_REPLAY;
    }
    if (replay_by_default) {
        return CONTROLLER_REPLAY;
    }

    int requested_controller;
    if (env->sdc_track_index >= 0 && agent_idx == env->sdc_track_index) {
        requested_controller = env->sdc_controller;
    } else if (env->entities[agent_idx].type == VEHICLE) {
        requested_controller = env->non_sdc_controller;
    } else {
        requested_controller = env->non_vehicle_controller;
    }

    if (requested_controller == CONTROLLER_POLICY && !is_active) {
        return CONTROLLER_STATIC;
    }

    return requested_controller;
}

void set_active_agents(Drive *env) {

    // Initialize
    env->active_agent_count = 0;        // Policy-controlled agents
    env->static_agent_count = 0;        // Non-moving background agents
    env->expert_static_agent_count = 0; // Expert replay agents (non-controlled)
    env->num_actors = 0;                // Total agents created (there is always the SDC)

    int actor_limit = MAX_AGENTS;
    if (env->spawn_agents > MAX_AGENTS) {
        actor_limit = env->spawn_agents;
        if (actor_limit > ACTOR_CAP)
            actor_limit = ACTOR_CAP;
    }
    int active_agent_indices[ACTOR_CAP];
    int static_agent_indices[ACTOR_CAP];
    int expert_static_agent_indices[ACTOR_CAP];

    if (env->num_agents == 0) {
        env->num_agents = actor_limit;
    }

    int control_limit;
    if (env->control_mode == CONTROL_MIXED_PLAY) {
        control_limit = MIN(env->max_controlled_agents, env->num_agents);
    } else {
        control_limit = env->num_agents;
    }

    // SDC-only modes: only the SDC is active; all other valid agents replay
    // their logs as expert-static background. Forces force_replay below.
    bool sdc_only_mode =
        (env->control_mode == CONTROL_SDC_ONLY || env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS);

    // If we have a SDC index (WOMD), initialize it first:
    int sdc_index = env->sdc_track_index;

    if (sdc_index >= 0) {
        active_agent_indices[0] = sdc_index;
        env->num_actors++;
        env->active_agent_count++;
        env->entities[sdc_index].active_agent = 1;
        env->entities[sdc_index].controller = resolve_agent_controller(env, sdc_index, 1, 0);
    }

    // Iterate through entities to find agents to create and/or control
    for (int i = 0; i < env->num_objects && env->num_actors < actor_limit; i++) {

        // Skip if its the SDC
        if (i == sdc_index) {
            continue;
        }

        Entity *entity = &env->entities[i];

        // Spawned cars have no logged pose. Their valid flag stays clear so
        // the map-centering mean ignores the blank origin point.
        if (env->spawn_agents <= 0 && entity->traj_valid[env->init_steps] != 1) {
            continue;
        }

        // Determine if entity should be created
        bool should_create = false;
        if (env->init_mode == INIT_ALL_VALID) {
            should_create = true; // All valid entities
        } else if (env->control_mode == CONTROL_VEHICLES) {
            should_create = (entity->type == VEHICLE);
        } else { // Control all agents
            should_create = (entity->type == VEHICLE || entity->type == PEDESTRIAN || entity->type == CYCLIST);
        }

        if (!should_create)
            continue;

        env->num_actors++;

        // Determine if this agent should be policy-controlled.
        // Logged scenes keep the goal-distance filter. A spawned fleet is
        // placed on the lane graph after this count, so every vehicle is live.
        bool is_controlled = false;
        if (env->spawn_agents > 0) {
            is_controlled = env->active_agent_count < control_limit && entity->type == VEHICLE;
        } else {
            is_controlled = should_control_agent(env, i, control_limit);
        }

        if (is_controlled) {
            active_agent_indices[env->active_agent_count] = i;
            env->active_agent_count++;
            env->entities[i].active_agent = 1;
            env->entities[i].controller = resolve_agent_controller(env, i, 1, 0);
        } else if (env->init_mode != INIT_ONLY_CONTROLLABLE_AGENTS) {
            static_agent_indices[env->static_agent_count] = i;
            env->static_agent_count++; // Includes expert replay and static agents
            env->entities[i].active_agent = 0;
            int requested_controller =
                env->entities[i].type == VEHICLE ? env->non_sdc_controller : env->non_vehicle_controller;
            int replay_by_default = env->entities[i].mark_as_expert != 0 || env->active_agent_count >= control_limit ||
                                    (sdc_only_mode && requested_controller == CONTROLLER_POLICY);
            env->entities[i].controller = resolve_agent_controller(env, i, 0, replay_by_default);
            // Replay-controlled backgrounds move through the expert list so
            // move_expert teleports them along their logged trajectories.
            if (env->entities[i].controller == CONTROLLER_REPLAY) {
                expert_static_agent_indices[env->expert_static_agent_count] = i;
                env->expert_static_agent_count++;
                env->entities[i].mark_as_expert = 1;
            }
        }
    }

    // Set up initial active agents
    env->active_agent_indices = (int *)malloc(env->active_agent_count * sizeof(int));
    env->static_agent_indices = (int *)malloc(env->static_agent_count * sizeof(int));
    env->expert_static_agent_indices = (int *)malloc(env->expert_static_agent_count * sizeof(int));
    for (int i = 0; i < env->active_agent_count; i++) {
        env->active_agent_indices[i] = active_agent_indices[i];
    };
    for (int i = 0; i < env->static_agent_count; i++) {
        env->static_agent_indices[i] = static_agent_indices[i];
    }
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        env->expert_static_agent_indices[i] = expert_static_agent_indices[i];
    }
    // printf("Total actors: %d, Active agents: %d, Static agents: %d, Expert static agents: %d\n", env->num_actors,
    //        env->active_agent_count, env->static_agent_count, env->expert_static_agent_count);
    // printf("Control mode: %d, max controlled agents: %d\n", env->control_mode, env->max_controlled_agents);

    return;
}

void remove_bad_trajectories(Drive *env) {

    if (env->control_mode != CONTROL_WOSAC && env->control_mode != CONTROL_INFERRED_EXPERT_ACTIONS &&
        env->control_mode != CONTROL_REPLAY_LOGS) {
        return;
    }

    set_start_position(env);
    int collided_agents[env->active_agent_count];
    int collided_with_indices[env->active_agent_count];
    memset(collided_agents, 0, env->active_agent_count * sizeof(int));
    for (int i = 0; i < env->active_agent_count; ++i) {
        collided_with_indices[i] = -1;
    }
    // move experts through trajectories to check for collisions and remove as illegal agents
    for (int t = 0; t < env->episode_length; t++) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            move_expert(env, env->actions, agent_idx);
        }
        for (int i = 0; i < env->expert_static_agent_count; i++) {
            int expert_idx = env->expert_static_agent_indices[i];
            if (env->entities[expert_idx].x == INVALID_POSITION)
                continue;
            move_expert(env, env->actions, expert_idx);
        }
        // check collisions
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            env->entities[agent_idx].collision_state = 0;
            int collided_with_index = collision_check(env, agent_idx);
            if ((collided_with_index >= 0) && collided_agents[i] == 0) {
                collided_agents[i] = 1;
                collided_with_indices[i] = collided_with_index;
            }
        }
        env->timestep++;
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        if (collided_with_indices[i] == -1)
            continue;
        for (int j = 0; j < env->static_agent_count; j++) {
            int static_agent_idx = env->static_agent_indices[j];
            if (static_agent_idx != collided_with_indices[i])
                continue;
            env->entities[static_agent_idx].traj_x[0] = INVALID_POSITION;
            env->entities[static_agent_idx].traj_y[0] = INVALID_POSITION;
        }
    }
    env->timestep = 0;
}

static void install_spawn_fleet(Drive *env);
static void free_lanes(Drive *env);
static void build_lanes(Drive *env);
static void place_spawned(Drive *env);
void sample_new_goal(Drive *env, int agent_idx);

void init(Drive *env) {
    env->human_agent_idx = 0;
    env->timestep = 0;
    init_action_space();
    env->entities = load_map_binary(env->map_name, env);
    if (env->spawn_agents > 0) {
        install_spawn_fleet(env);
    }
    set_means(env);
    init_grid_map(env);
    env->grid_map->vision_range = 21; // TODO: Why is this hardcoded?
    init_neighbor_offsets(env);
    cache_neighbor_offsets(env);
    env->logs_capacity = 0;
    set_active_agents(env);
    env->logs_capacity = env->active_agent_count;
    remove_bad_trajectories(env);
    if (env->spawn_agents > 0) {
        if (env->goal_distance_min <= 0.0f || env->goal_distance_max <= env->goal_distance_min) {
            env->goal_distance_min = 5.0f;
            env->goal_distance_max = 50.0f;
        }
        build_lanes(env);
        place_spawned(env);
    } else {
        set_start_position(env);
    }

    for (int x = 0; x < env->active_agent_count; x++) {
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].init_goal_x = env->entities[agent_idx].goal_position_x;
        env->entities[agent_idx].init_goal_y = env->entities[agent_idx].goal_position_y;
    }

    env->logs = (Log *)calloc(env->active_agent_count, sizeof(Log));
    env->dynamics_noise_pos = 0.0f; // 0.0125f; Gigaflow
    env->dynamics_noise_heading = 0.0f;
    env->obs_noise_road = 0.0f;
    if (env->async_resets != 0 && env->async_resets != 1) {
        env->async_resets = 1; // Default to async resets
    }
}

void close_client(Client *client);

void c_close(Drive *env) {
    if (env->client != NULL) {
        close_client(env->client);
        env->client = NULL;
    }
    for (int i = 0; i < env->num_entities; i++) {
        free_entity(&env->entities[i]);
    }
    free(env->entities);
    free(env->active_agent_indices);
    free(env->logs);
    // GridMap cleanup
    int grid_cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        free(env->grid_map->cells[grid_index]);
    }
    free(env->grid_map->cells);
    free(env->grid_map->cell_entities_count);
    free(env->neighbor_offsets);

    for (int i = 0; i < grid_cell_count; i++) {
        free(env->grid_map->neighbor_cache_entities[i]);
    }
    free(env->grid_map->neighbor_cache_entities);
    free(env->grid_map->neighbor_cache_count);
    free(env->grid_map);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    free(env->ini_file);
    free(env->tracks_to_predict_indices);
    free_lanes(env);
    env->tracks_to_predict_indices = NULL;
}

void allocate(Drive *env) {
    init(env);
    int ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES;
    int max_obs = DRIVE_MAX_OBS(ego_dim);
    env->observations = (float *)calloc(env->active_agent_count * max_obs, sizeof(float));

    // All discrete models use a single int per agent (cast to float storage)
    // All continuous models use action_dim floats per agent
    if (env->action_type == 1) { // continuous
        int action_dim = (env->dynamics_model == DELTA_LOCAL) ? 3 : 2;
        env->actions = (float *)calloc(env->active_agent_count * action_dim, sizeof(float));
    } else { // discrete
        int actions_per_agent = (env->dynamics_model == DELTA_LOCAL) ? 3 : (env->dynamics_model == CLASSIC) ? 2 : 1;
        env->actions = (float *)calloc(env->active_agent_count * actions_per_agent, sizeof(int));
    }

    env->rewards = (float *)calloc(env->active_agent_count, sizeof(float));
    env->terminals = (unsigned char *)calloc(env->active_agent_count, sizeof(unsigned char));
    env->truncations = (unsigned char *)calloc(env->active_agent_count, sizeof(unsigned char));
}

void free_allocated(Drive *env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->truncations);
    c_close(env);
}

float clipSpeed(float speed) {
    const float maxSpeed = MAX_SPEED;
    if (speed > maxSpeed)
        return maxSpeed;
    if (speed < -maxSpeed)
        return -maxSpeed;
    return speed;
}

float normalize_heading(float heading) {
    if (heading > M_PI)
        heading -= 2 * M_PI;
    if (heading < -M_PI)
        heading += 2 * M_PI;
    return heading;
}

// The logged body is longer than the wheelbase.
static float bicycle_wheelbase(float length) {
    float wheelbase = 0.6f * length;
    if (wheelbase < 1.0f) {
        wheelbase = 1.0f;
    }
    return wheelbase;
}

// Ackermann bicycle. The rear axle has no sideslip and yaws at
// v tan(steer) / wheelbase. The body center is half a wheelbase ahead.
// Same integrator as pufferflow bicycle_integrate, stepped by env->dt.
static void bicycle_integrate(float *x, float *y, float *heading, float *vx, float *vy, float length,
                              float acceleration, float steering, float dt) {
    float c = cosf(*heading);
    float s = sinf(*heading);
    float v = (*vx) * c + (*vy) * s + acceleration * dt;
    float wheelbase = bicycle_wheelbase(length);
    float half = 0.5f * wheelbase;
    float rx = *x - half * c;
    float ry = *y - half * s;
    if (v > DYN_SPEED_MAX) {
        v = DYN_SPEED_MAX;
    }
    if (v < -DYN_SPEED_MAX) {
        v = -DYN_SPEED_MAX;
    }
    if (steering > STEER_MAX) {
        steering = STEER_MAX;
    } else if (steering < STEER_MIN) {
        steering = STEER_MIN;
    }
    float tsteer = tanf(steering);
    float dth = v * tsteer / wheelbase * dt;
    float dx_b, dy_b;
    if (fabsf(dth) < 1.0e-4f) {
        dx_b = v * dt;
        dy_b = 0.0f;
    } else {
        float radius = v * dt / dth;
        dx_b = radius * sinf(dth);
        dy_b = radius * (1.0f - cosf(dth));
    }
    rx += dx_b * c - dy_b * s;
    ry += dx_b * s + dy_b * c;
    *heading = normalize_heading(*heading + dth);
    c = cosf(*heading);
    s = sinf(*heading);
    *x = rx + half * c;
    *y = ry + half * s;
    float v_lat = v * tsteer * 0.5f;
    *vx = v * c - v_lat * s;
    *vy = v * s + v_lat * c;
}

void move_dynamics(Drive *env, int action_idx, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    if (agent->removed)
        return;

    if (agent->stopped) {
        agent->vx = 0.0f;
        agent->vy = 0.0f;
        return;
    }

    if (env->dynamics_model == CLASSIC) {
        float acceleration = 0.0f;
        float steering = 0.0f;

        if (env->action_type == 1) { // continuous, scaled onto the same command box
            float (*action_array_f)[2] = (float (*)[2])env->actions;
            acceleration = action_array_f[action_idx][0] * ACCEL_MAX;
            steering = action_array_f[action_idx][1] * STEER_MAX;
        } else { // discrete: two independent bins, accel then steer
            int *action_array = (int *)env->actions;
            int acceleration_index = action_array[action_idx * 2 + 0];
            int steering_index = action_array[action_idx * 2 + 1];
            if (acceleration_index < 0 || acceleration_index >= NUM_ACCEL_BINS)
                acceleration_index = ACCEL_COAST_BIN;
            if (steering_index < 0 || steering_index >= NUM_STEER_BINS)
                steering_index = STEER_COAST_BIN;
            acceleration = ACCELERATION_VALUES[acceleration_index];
            steering = STEERING_VALUES[steering_index];
        }

        float x = agent->x;
        float y = agent->y;
        float heading = agent->heading;
        float vx = agent->vx;
        float vy = agent->vy;
        bicycle_integrate(&x, &y, &heading, &vx, &vy, agent->length, acceleration, steering, env->dt);
        agent->a_long = acceleration;
        agent->x = x;
        agent->y = y;
        agent->heading = heading;
        agent->heading_x = cosf(heading);
        agent->heading_y = sinf(heading);
        agent->vx = vx;
        agent->vy = vy;

    } else if (env->dynamics_model == DELTA_LOCAL) {
        float action_dx, action_dy, action_dyaw;

        if (env->action_type == 1) { // continuous
            float *action_base = (float *)env->actions;
            action_dx = action_base[action_idx * 3 + 0] * DELTA_MAX_DX;
            action_dy = action_base[action_idx * 3 + 1] * DELTA_MAX_DY;
            action_dyaw = action_base[action_idx * 3 + 2] * DELTA_MAX_DYAW;
        } else { // discrete — independent MultiDiscrete: 3 ints per agent
            int *action_array = (int *)env->actions;
            int dx_idx = action_array[action_idx * 3 + 0];
            int dy_idx = action_array[action_idx * 3 + 1];
            int yaw_idx = action_array[action_idx * 3 + 2];
            if (dx_idx < 0 || dx_idx >= NUM_DX_BINS)
                dx_idx = NUM_DX_BINS / 2;
            if (dy_idx < 0 || dy_idx >= NUM_DY_BINS)
                dy_idx = NUM_DY_BINS / 2;
            if (yaw_idx < 0 || yaw_idx >= NUM_YAW_BINS)
                yaw_idx = NUM_YAW_BINS / 2;

            action_dx = DELTA_DX_VALUES[dx_idx];
            action_dy = DELTA_DY_VALUES[dy_idx];
            action_dyaw = DELTA_YAW_VALUES[yaw_idx];
        }

        action_dx = clip(action_dx, -DELTA_MAX_DX, DELTA_MAX_DX);
        action_dy = clip(action_dy, -DELTA_MAX_DY, DELTA_MAX_DY);
        action_dyaw = normalize_heading(action_dyaw);

        float raw_dx = action_dx, raw_dy = action_dy, raw_dyaw = action_dyaw;

        // ----- Kinematic constraints (delta-local) -----
        // Each constraint operates on the action AFTER prior constraints have clipped it.
        // prev_action_* stores the previously-executed (post-constraint) values.

        // A. Longitudinal acceleration bound.
        //    Implied previous speed = prev_action_dx / dt. Limit change to ±A_LONG_MAX*dt.
        {
            float prev_speed = agent->prev_action_dx / env->dt;
            float speed_lo = prev_speed - A_LONG_MAX * env->dt;
            float speed_hi = prev_speed + A_LONG_MAX * env->dt;
            float new_speed_proposed = action_dx / env->dt;
            float new_speed = clip(new_speed_proposed, speed_lo, speed_hi);
            action_dx = new_speed * env->dt;
        }

        // B. Lateral motion bicycle envelope: |dy| ≤ |dx| * tan(MAX_STEER).
        //    Eliminates lateral sliding and side-shimmy at low / zero forward speed.
        {
            float max_dy = fabsf(action_dx) * tanf(MAX_STEER);
            float prev_action_dy = action_dy;
            action_dy = clip(action_dy, -max_dy, max_dy);
        }

        float jerk_dx = action_dx - agent->prev_action_dx;
        float jerk_dy = action_dy - agent->prev_action_dy;
        float jerk_dyaw = normalize_heading(action_dyaw - agent->prev_action_dyaw);

        agent->jerk_long = jerk_dx;
        agent->jerk_lat = jerk_dy;
        agent->jerk_yaw = jerk_dyaw;

        agent->prev_action_dx = action_dx;
        agent->prev_action_dy = action_dy;
        agent->prev_action_dyaw = action_dyaw;

        // if (raw_dx != action_dx)
        //     printf("clipped dx: %.3f -> %.3f\n", raw_dx, action_dx);
        // if (raw_dy != action_dy)
        //     printf("clipped dy: %.3f -> %.3f\n", raw_dy, action_dy);
        // if (raw_dyaw != action_dyaw)
        //     printf("clipped dyaw: %.3f -> %.3f\n", raw_dyaw, action_dyaw);

        float cos_h = agent->heading_x;
        float sin_h = agent->heading_y;
        float global_dx = cos_h * action_dx - sin_h * action_dy;
        float global_dy = sin_h * action_dx + cos_h * action_dy;

        agent->x += global_dx;
        agent->y += global_dy;
        agent->heading = normalize_heading(agent->heading + action_dyaw);
        agent->heading_x = cosf(agent->heading);
        agent->heading_y = sinf(agent->heading);
        agent->vx = global_dx / env->dt;
        agent->vy = global_dy / env->dt;
    } else {
        // JERK dynamics model
        // Extract action components
        float a_long, a_lat;
        if (env->action_type == 1) { // continuous
            float (*action_array_f)[2] = (float (*)[2])env->actions;

            // Asymmetric scaling for longitudinal jerk to match discrete action space
            // Discrete: JERK_LONG = [-15, -4, 0, 4] (more braking than acceleration)
            float a_long_action = action_array_f[action_idx][0]; // [-1, 1]
            if (a_long_action < 0) {
                a_long = a_long_action * (-JERK_LONG[0]); // Negative: [-1, 0] → [-15, 0] (braking)
            } else {
                a_long = a_long_action * JERK_LONG[3]; // Positive: [0, 1] → [0, 4] (acceleration)
            }

            // Symmetric scaling for lateral jerk
            a_lat = action_array_f[action_idx][1] * JERK_LAT[2];
        } else { // discrete
            // Interpret action as a single integer: a = long_idx * num_lat + lat_idx
            int *action_array = (int *)env->actions;
            int num_lat = sizeof(JERK_LAT) / sizeof(JERK_LAT[0]);
            int action_val = action_array[action_idx];
            int a_long_idx = action_val / num_lat;
            int a_lat_idx = action_val % num_lat;
            a_long = JERK_LONG[a_long_idx];
            a_lat = JERK_LAT[a_lat_idx];
        }

        // Calculate new acceleration
        float a_long_new = agent->a_long + a_long * env->dt;
        float a_lat_new = agent->a_lat + a_lat * env->dt;

        // Make it easy to stop with 0 accel
        if (agent->a_long * a_long_new < 0) {
            a_long_new = 0.0f;
        } else {
            a_long_new = clip(a_long_new, -5.0f, 2.5f);
        }

        if (agent->a_lat * a_lat_new < 0) {
            a_lat_new = 0.0f;
        } else {
            a_lat_new = clip(a_lat_new, -4.0f, 4.0f);
        }

        // Calculate new velocity
        float v_dot_heading = agent->vx * agent->heading_x + agent->vy * agent->heading_y;
        float signed_v = copysignf(sqrtf(agent->vx * agent->vx + agent->vy * agent->vy), v_dot_heading);
        float v_new = signed_v + 0.5f * (a_long_new + agent->a_long) * env->dt;

        // Make it easy to stop with 0 vel
        if (signed_v * v_new < 0) {
            v_new = 0.0f;
        } else {
            v_new = clip(v_new, -2.0f, 20.0f);
        }

        // Calculate new steering angle
        float signed_curvature = a_lat_new / fmaxf(v_new * v_new, 1e-5f);
        signed_curvature = copysignf(fmaxf(fabsf(signed_curvature), 1e-5f), signed_curvature);
        float steering_angle = atanf(signed_curvature * agent->wheelbase);
        float delta_steer = clip(steering_angle - agent->steering_angle, -0.6f * env->dt, 0.6f * env->dt);
        float new_steering_angle = clip(agent->steering_angle + delta_steer, -0.55f, 0.55f);

        // Update curvature and accel to account for limited steering
        signed_curvature = tanf(new_steering_angle) / agent->wheelbase;
        a_lat_new = v_new * v_new * signed_curvature;

        // Calculate resulting movement using bicycle dynamics
        float d = 0.5f * (v_new + signed_v) * env->dt;
        float theta = d * signed_curvature;
        float dx_local, dy_local;

        if (fabsf(signed_curvature) < 1e-5f || fabsf(theta) < 1e-5f) {
            dx_local = d;
            dy_local = 0.0f;
        } else {
            dx_local = sinf(theta) / signed_curvature;
            dy_local = (1.0f - cosf(theta)) / signed_curvature;
        }

        float dx = dx_local * agent->heading_x - dy_local * agent->heading_y;
        float dy = dx_local * agent->heading_y + dy_local * agent->heading_x;

        // Update everything
        agent->x += dx;
        agent->y += dy;
        agent->jerk_long = (a_long_new - agent->a_long) / env->dt;
        agent->jerk_lat = (a_lat_new - agent->a_lat) / env->dt;
        agent->a_long = a_long_new;
        agent->a_lat = a_lat_new;
        agent->heading = normalize_heading(agent->heading + theta);
        agent->heading_x = cosf(agent->heading);
        agent->heading_y = sinf(agent->heading);
        agent->vx = v_new * agent->heading_x;
        agent->vy = v_new * agent->heading_y;
        agent->steering_angle = new_steering_angle;
    }

    return;
}

void apply_dynamics_noise(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    if (agent->removed || agent->stopped)
        return;

    if (env->dynamics_noise_pos > 0.0f) {
        agent->x += gaussian_noise(env->dynamics_noise_pos);
        agent->y += gaussian_noise(env->dynamics_noise_pos);
    }
    if (env->dynamics_noise_heading > 0.0f) {
        agent->heading += gaussian_noise(env->dynamics_noise_heading);
        agent->heading = normalize_heading(agent->heading);
        agent->heading_x = cosf(agent->heading);
        agent->heading_y = sinf(agent->heading);
    }
}

static inline int is_in_track_to_predicts(Drive *env, int agent_idx) {
    if (env->tracks_to_predict_indices == NULL || env->num_tracks_to_predict == 0) {
        return 0;
    }
    for (int k = 0; k < env->num_tracks_to_predict; k++) {
        if (env->tracks_to_predict_indices[k] == agent_idx) {
            return 1;
        }
    }
    return 0;
}

void c_get_global_agent_state(Drive *env, float *x_out, float *y_out, float *z_out, float *heading_out, int *id_out,
                              float *length_out, float *width_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Entity *agent = &env->entities[agent_idx];

        // For WOSAC, we need the original world coordinates, so we add the world means back
        x_out[i] = agent->x + env->world_mean_x;
        y_out[i] = agent->y + env->world_mean_y;
        z_out[i] = agent->z;
        heading_out[i] = agent->heading;
        id_out[i] = agent->id;
        length_out[i] = agent->length;
        width_out[i] = agent->width;
    }
}

void c_get_global_ground_truth_trajectories(Drive *env, float *x_out, float *y_out, float *z_out, float *heading_out,
                                            int *valid_out, int *id_out, bool *is_vehicle_out,
                                            bool *is_track_to_predict_out, char *scenario_id_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Entity *agent = &env->entities[agent_idx];
        id_out[i] = agent->id;
        is_vehicle_out[i] = agent->type == VEHICLE;
        is_track_to_predict_out[i] = is_in_track_to_predicts(env, agent_idx);

        // The scenario_id is an array of 16 char
        memcpy(scenario_id_out + (i * 16), env->scenario_id, 16);

        for (int t = env->init_steps; t < agent->array_size; t++) {
            int out_idx = i * (agent->array_size - env->init_steps) + (t - env->init_steps);
            // Add world means back to get original world coordinates
            x_out[out_idx] = agent->traj_x[t] + env->world_mean_x;
            y_out[out_idx] = agent->traj_y[t] + env->world_mean_y;
            z_out[out_idx] = agent->traj_z[t];
            heading_out[out_idx] = agent->traj_heading[t];
            valid_out[out_idx] = agent->traj_valid[t];
        }
    }
}

void c_get_road_edge_counts(Drive *env, int *num_polylines_out, int *total_points_out) {
    int count = 0, points = 0;
    for (int i = env->num_objects; i < env->num_entities; i++) {
        if (env->entities[i].type == ROAD_EDGE) {
            count++;
            points += env->entities[i].array_size;
        }
    }
    *num_polylines_out = count;
    *total_points_out = points;
}

void c_get_road_edge_polylines(Drive *env, float *x_out, float *y_out, int *lengths_out, char *scenario_ids_out) {
    int poly_idx = 0, pt_idx = 0;
    for (int i = env->num_objects; i < env->num_entities; i++) {
        Entity *e = &env->entities[i];
        if (e->type == ROAD_EDGE) {
            lengths_out[poly_idx] = e->array_size;

            char *scenario_id_ptr = scenario_ids_out + poly_idx * 16;
            memcpy(scenario_id_ptr, env->scenario_id, 16);

            for (int j = 0; j < e->array_size; j++) {
                x_out[pt_idx] = e->traj_x[j] + env->world_mean_x;
                y_out[pt_idx] = e->traj_y[j] + env->world_mean_y;
                pt_idx++;
            }
            poly_idx++;
        }
    }
}

// Partner slot from an origin in the ego frame. Center uses the ego position.
// A corner camera uses that corner. Width, length, heading, and speed are the partner's.
static void write_partner_obs(float *dst, const Entity *ego, const Entity *other, float origin_x, float origin_y,
                              float noise_pos, float noise_speed) {
    float cos_heading = ego->heading_x;
    float sin_heading = ego->heading_y;
    float dx = other->x - origin_x;
    float dy = other->y - origin_y;
    float rel_x = dx * cos_heading + dy * sin_heading;
    float rel_y = -dx * sin_heading + dy * cos_heading;
    dst[0] = rel_x * 0.02f + gaussian_noise(noise_pos);
    dst[1] = rel_y * 0.02f + gaussian_noise(noise_pos);
    dst[2] = other->width / MAX_VEH_WIDTH + gaussian_noise(noise_pos);
    dst[3] = other->length / MAX_VEH_LEN + gaussian_noise(noise_pos);
    // cos(a-b), sin(a-b)
    dst[4] = other->heading_x * ego->heading_x + other->heading_y * ego->heading_y;
    dst[5] = other->heading_y * ego->heading_x - other->heading_x * ego->heading_y;
    float speed = sqrtf(other->vx * other->vx + other->vy * other->vy);
    float along = other->vx * other->heading_x + other->vy * other->heading_y;
    dst[6] = copysignf(speed, along) / MAX_SPEED + gaussian_noise(noise_speed);
}

// Keep the closest partners. ids/dists stay sorted by ascending distance.
static void insert_corner_candidate(int *ids, float *dists, int *count, int index, float dist) {
    int n = *count;
    if (n == CORNER_PARTNERS && dist >= dists[n - 1])
        return;
    int slot = n < CORNER_PARTNERS ? n : CORNER_PARTNERS - 1;
    if (n < CORNER_PARTNERS)
        n++;
    while (slot > 0 && dist < dists[slot - 1]) {
        dists[slot] = dists[slot - 1];
        ids[slot] = ids[slot - 1];
        slot--;
    }
    dists[slot] = dist;
    ids[slot] = index;
    *count = n;
}

void compute_observations(Drive *env) {
    int ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES;
    int max_obs = DRIVE_MAX_OBS(ego_dim);
    memset(env->observations, 0, max_obs * env->active_agent_count * sizeof(float));
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    for (int i = 0; i < env->active_agent_count; i++) {
        float *obs = &observations[i][0];
        Entity *ego_entity = &env->entities[env->active_agent_indices[i]];
        if (ego_entity->type > 3)
            break;

        float cos_heading = ego_entity->heading_x;
        float sin_heading = ego_entity->heading_y;
        float speed_magnitude = sqrtf(ego_entity->vx * ego_entity->vx + ego_entity->vy * ego_entity->vy);
        float v_dot_heading = ego_entity->vx * ego_entity->heading_x + ego_entity->vy * ego_entity->heading_y;
        float signed_speed = copysignf(speed_magnitude, v_dot_heading);

        // Set goal distances
        float goal_x = ego_entity->goal_position_x - ego_entity->x;
        float goal_y = ego_entity->goal_position_y - ego_entity->y;

        // Rotate to ego vehicle's frame
        float rel_goal_x = goal_x * cos_heading + goal_y * sin_heading;
        float rel_goal_y = -goal_x * sin_heading + goal_y * cos_heading;

        // Conditioning observations (idx 0-3)
        obs[LAMBDA_CONDITIONING_IDX] = ego_entity->lambda;
        obs[REWARD_COLLISION_IDX] = ego_entity->reward_collision_cond;
        obs[REWARD_OFFROAD_COLLISION_IDX] = ego_entity->reward_offroad_cond;
        obs[REWARD_GOAL_IDX] = ego_entity->reward_goal_cond;

        // Other ego features
        obs[4] = rel_goal_x * 0.005f;
        obs[5] = rel_goal_y * 0.005f;
        obs[6] = signed_speed / MAX_SPEED;
        obs[7] = ego_entity->width / MAX_VEH_WIDTH;
        obs[8] = ego_entity->length / MAX_VEH_LEN;
        obs[9] = (ego_entity->collision_state > 0) ? 1.0f : 0.0f;

        if (env->dynamics_model == JERK) {
            obs[10] = ego_entity->steering_angle / M_PI;
            // Asymmetric normalization for a_long to match action space
            obs[11] =
                (ego_entity->a_long < 0) ? ego_entity->a_long / (-JERK_LONG[0]) : ego_entity->a_long / JERK_LONG[3];
            obs[12] = ego_entity->a_lat / JERK_LAT[2];
            obs[13] = 0.0f;
            // Add normalized entity type (VEHICLE=1, PEDESTRIAN=2, CYCLIST=3)
            obs[14] = ego_entity->type / 3.0f;
        } else {
            obs[10] = 0.0f; // Can be deleted later
            obs[11] = ego_entity->type / 3.0f;
        }
        // Relative Pos of other cars. Waymo keeps logged order and the
        // 32-slot cap. A larger spawned fleet keeps the nearest 31 so the
        // observation width does not change between curriculum stages.
        int obs_idx = ego_dim;
        int cars_seen = 0;
        int partner_ids[ACTOR_CAP];
        int partner_n = 0;
        int self_index = env->active_agent_indices[i];
        if (env->spawn_agents > 0 && env->active_agent_count > MAX_AGENTS) {
            int sel[MAX_AGENTS - 1];
            float sel_d[MAX_AGENTS - 1];
            int nsel = 0;
            for (int j = 0; j < env->active_agent_count; j++) {
                int index = env->active_agent_indices[j];
                if (index == self_index)
                    continue;
                Entity *candidate = &env->entities[index];
                if (candidate->removed || candidate->type > 3)
                    continue;
                float cdx = candidate->x - ego_entity->x;
                float cdy = candidate->y - ego_entity->y;
                float cdist = cdx * cdx + cdy * cdy;
                if (cdist > 2500.0f)
                    continue;
                if (nsel < MAX_AGENTS - 1) {
                    sel[nsel] = index;
                    sel_d[nsel] = cdist;
                    nsel++;
                } else {
                    int worst = 0;
                    for (int k = 1; k < nsel; k++) {
                        if (sel_d[k] > sel_d[worst])
                            worst = k;
                    }
                    if (cdist < sel_d[worst]) {
                        sel[worst] = index;
                        sel_d[worst] = cdist;
                    }
                }
            }
            for (int j = 0; j < nsel; j++)
                partner_ids[partner_n++] = sel[j];
        } else {
            for (int j = 0; j < MAX_AGENTS; j++) {
                int index = -1;
                if (j < env->active_agent_count) {
                    index = env->active_agent_indices[j];
                } else if (j < env->num_actors && env->static_agent_count > 0) {
                    index = env->static_agent_indices[j - env->active_agent_count];
                }
                if (index == -1)
                    continue;
                if (env->entities[index].type > 3)
                    break;
                partner_ids[partner_n++] = index;
            }
        }
        for (int p = 0; p < partner_n; p++) {
            int index = partner_ids[p];
            if (index == self_index)
                continue; // Skip self, but don't increment obs_idx
            Entity *other_entity = &env->entities[index];
            if (ego_entity->removed)
                continue;
            if (other_entity->removed)
                continue;
            float dx = other_entity->x - ego_entity->x;
            float dy = other_entity->y - ego_entity->y;
            if (dx * dx + dy * dy > 2500.0f)
                continue;
            write_partner_obs(&obs[obs_idx], ego_entity, other_entity, ego_entity->x, ego_entity->y,
                              env->obs_partner_noise_pos, env->obs_partner_noise_speed);
            cars_seen++;
            obs_idx += PARTNER_FEATURES;
        }
        int remaining_partner_obs = (MAX_AGENTS - 1 - cars_seen) * PARTNER_FEATURES;
        memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
        obs_idx += remaining_partner_obs;

        // Corner cameras. offsets are (longitudinal, lateral) signs:
        // rear-left, front-left, front-right, rear-right. The rectangle is
        // opaque, so a partner whose center lies in the inward quadrant is hidden.
        // Range matches the center view, 50 m, so relative positions stay in [-1, 1].
        float half_length = ego_entity->length * 0.5f;
        float half_width = ego_entity->width * 0.5f;
        float corner_x[CORNER_VIEWS];
        float corner_y[CORNER_VIEWS];
        int corner_ids[CORNER_VIEWS][CORNER_PARTNERS];
        float corner_dist[CORNER_VIEWS][CORNER_PARTNERS];
        int corner_count[CORNER_VIEWS];
        for (int c = 0; c < CORNER_VIEWS; c++) {
            float ox = offsets[c][0];
            float oy = offsets[c][1];
            corner_x[c] = ego_entity->x + (ox * half_length * cos_heading - oy * half_width * sin_heading);
            corner_y[c] = ego_entity->y + (ox * half_length * sin_heading + oy * half_width * cos_heading);
            corner_count[c] = 0;
        }
        int corner_scan = (env->spawn_agents > 0) ? env->active_agent_count : MAX_AGENTS;
        for (int j = 0; j < corner_scan; j++) {
            int index = -1;
            if (j < env->active_agent_count) {
                index = env->active_agent_indices[j];
            } else if (env->spawn_agents <= 0 && j < env->num_actors && env->static_agent_count > 0) {
                index = env->static_agent_indices[j - env->active_agent_count];
            }
            if (index == -1 || index == self_index)
                continue;
            Entity *other_entity = &env->entities[index];
            if (other_entity->type > 3 || other_entity->removed || ego_entity->removed)
                continue;
            for (int c = 0; c < CORNER_VIEWS; c++) {
                float dx = other_entity->x - corner_x[c];
                float dy = other_entity->y - corner_y[c];
                float rel_x = dx * cos_heading + dy * sin_heading;
                float rel_y = -dx * sin_heading + dy * cos_heading;
                float ox = offsets[c][0];
                float oy = offsets[c][1];
                if (rel_x * ox < 0.0f && rel_y * oy < 0.0f)
                    continue;
                float dist = dx * dx + dy * dy;
                if (dist > 2500.0f)
                    continue;
                insert_corner_candidate(corner_ids[c], corner_dist[c], &corner_count[c], index, dist);
            }
        }
        for (int c = 0; c < CORNER_VIEWS; c++) {
            for (int p = 0; p < corner_count[c]; p++) {
                Entity *other_entity = &env->entities[corner_ids[c][p]];
                write_partner_obs(&obs[obs_idx], ego_entity, other_entity, corner_x[c], corner_y[c],
                                  env->obs_partner_noise_pos, env->obs_partner_noise_speed);
                obs_idx += PARTNER_FEATURES;
            }
            int pad = (CORNER_PARTNERS - corner_count[c]) * PARTNER_FEATURES;
            memset(&obs[obs_idx], 0, pad * sizeof(float));
            obs_idx += pad;
        }
        // map observations
        GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25];
        int grid_idx = getGridIndex(env, ego_entity->x, ego_entity->y);

        int list_size = get_neighbor_cache_entities(env, grid_idx, entity_list, MAX_ROAD_SEGMENT_OBSERVATIONS);

        for (int k = 0; k < list_size; k++) {
            int entity_idx = entity_list[k].entity_idx;
            int geometry_idx = entity_list[k].geometry_idx;

            // Validate entity_idx before accessing
            if (entity_idx < 0 || entity_idx >= env->num_entities) {
                printf("ERROR: Invalid entity_idx %d (max: %d)\n", entity_idx, env->num_entities - 1);
                continue;
            }

            Entity *entity = &env->entities[entity_idx];

            // Validate geometry_idx before accessing
            if (geometry_idx < 0 || geometry_idx >= entity->array_size) {
                printf("ERROR: Invalid geometry_idx %d for entity %d (max: %d)\n", geometry_idx, entity_idx,
                       entity->array_size - 1);
                continue;
            }
            float start_x = entity->traj_x[geometry_idx];
            float start_y = entity->traj_y[geometry_idx];
            float end_x = entity->traj_x[geometry_idx + 1];
            float end_y = entity->traj_y[geometry_idx + 1];
            float mid_x = (start_x + end_x) / 2.0f;
            float mid_y = (start_y + end_y) / 2.0f;
            float rel_x = mid_x - ego_entity->x;
            float rel_y = mid_y - ego_entity->y;
            float x_obs = rel_x * cos_heading + rel_y * sin_heading;
            float y_obs = -rel_x * sin_heading + rel_y * cos_heading;
            float length = relative_distance_2d(mid_x, mid_y, end_x, end_y);
            float width = 0.1;
            // Calculate angle from ego to midpoint (vector from ego to midpoint)
            float dx = end_x - mid_x;
            float dy = end_y - mid_y;
            float dx_norm = dx;
            float dy_norm = dy;
            float hypot = sqrtf(dx * dx + dy * dy);
            if (hypot > 0) {
                dx_norm /= hypot;
                dy_norm /= hypot;
            }
            // Compute sin and cos of relative angle directly without atan2f
            float cos_angle = dx_norm * cos_heading + dy_norm * sin_heading;
            float sin_angle = -dx_norm * sin_heading + dy_norm * cos_heading;
            obs[obs_idx] = x_obs * 0.02f;
            obs[obs_idx + 1] = y_obs * 0.02f;
            obs[obs_idx + 2] = length / MAX_ROAD_SEGMENT_LENGTH;
            obs[obs_idx + 3] = width / MAX_ROAD_SCALE;
            obs[obs_idx + 4] = cos_angle;
            obs[obs_idx + 5] = sin_angle;
            obs[obs_idx + 6] = entity->type - 4.0f;
            obs_idx += 7;
        }
        int remaining_obs = (MAX_ROAD_SEGMENT_OBSERVATIONS - list_size) * 7;
        // Set the entire block to 0 at once
        memset(&obs[obs_idx], 0, remaining_obs * sizeof(float));
    }
}

static float *g_lane_dist_sort;

static int cmp_lane_dist_desc(const void *a, const void *b) {
    float da = g_lane_dist_sort[*(const int *)a];
    float db = g_lane_dist_sort[*(const int *)b];
    if (da < db)
        return 1;
    if (da > db)
        return -1;
    return 0;
}

static void free_lanes(Drive *env) {
    free(env->lane_x);
    env->lane_x = NULL;
    env->lane_y = NULL;
    env->lane_hx = NULL;
    env->lane_hy = NULL;
    env->lane_dist = NULL;
    env->lane_block = NULL;
    env->lane_heap_d = NULL;
    env->lane_next = NULL;
    env->lane_poly_start = NULL;
    env->lane_heap_i = NULL;
    env->lane_order = NULL;
    env->lane_n = 0;
    env->lane_polys = 0;
    env->lane_heap_cap = 0;
}

static void blank_vehicle(Entity *e) {
    memset(e, 0, sizeof(Entity));
    e->type = VEHICLE;
    e->array_size = 1;
    e->length = SPAWN_CAR_LENGTH;
    e->width = SPAWN_CAR_WIDTH;
    e->height = 1.5f;
    e->lane_node = -1;
    e->x = INVALID_POSITION;
    e->y = INVALID_POSITION;
    e->traj_x = (float *)calloc(1, sizeof(float));
    e->traj_y = (float *)calloc(1, sizeof(float));
    e->traj_z = (float *)calloc(1, sizeof(float));
    e->traj_vx = (float *)calloc(1, sizeof(float));
    e->traj_vy = (float *)calloc(1, sizeof(float));
    e->traj_vz = (float *)calloc(1, sizeof(float));
    e->traj_heading = (float *)calloc(1, sizeof(float));
    e->traj_valid = (int *)calloc(1, sizeof(int));
    e->inferred_traj_x = (float *)calloc(1, sizeof(float));
    e->inferred_traj_y = (float *)calloc(1, sizeof(float));
    e->expert_accel = (float *)calloc(1, sizeof(float));
    e->expert_steering = (float *)calloc(1, sizeof(float));
    e->expert_delta_x = (float *)calloc(1, sizeof(float));
    e->expert_delta_y = (float *)calloc(1, sizeof(float));
    e->expert_delta_yaw = (float *)calloc(1, sizeof(float));
}

static void install_spawn_fleet(Drive *env) {
    int spawn = env->spawn_agents;
    if (spawn <= 0 || env->entities == NULL)
        return;
    if (spawn > ACTOR_CAP)
        spawn = ACTOR_CAP;
    int nroads = env->num_roads;
    int nobj = env->num_objects;
    Entity *old = env->entities;
    Entity *neu = (Entity *)calloc((size_t)spawn + (size_t)nroads, sizeof(Entity));
    if (neu == NULL)
        return;
    for (int i = 0; i < nroads; i++)
        neu[spawn + i] = old[nobj + i];
    for (int i = 0; i < nobj; i++)
        free_entity(&old[i]);
    free(old);
    for (int i = 0; i < spawn; i++)
        blank_vehicle(&neu[i]);
    env->entities = neu;
    env->num_objects = spawn;
    env->num_entities = spawn + nroads;
    env->spawn_agents = spawn;
    env->sdc_track_index = -1;
    free(env->tracks_to_predict_indices);
    env->tracks_to_predict_indices = NULL;
    env->num_tracks_to_predict = 0;
}

static void build_lanes(Drive *env) {
    free_lanes(env);
    int n = 0;
    int polys = 0;
    for (int i = 0; i < env->num_entities; i++) {
        Entity *e = &env->entities[i];
        if (e->type != ROAD_LANE || e->array_size < 2)
            continue;
        n += e->array_size;
        polys++;
    }
    if (n < 2 || polys < 1)
        return;
    int heap_cap = n * 4;
    size_t fcount = (size_t)n * 6 + (size_t)heap_cap;
    size_t icount = (size_t)n * 3 + (size_t)polys + (size_t)heap_cap;
    char *buf = (char *)calloc(1, fcount * sizeof(float) + icount * sizeof(int));
    if (buf == NULL)
        return;
    float *f = (float *)buf;
    int *ip = (int *)(f + fcount);
    env->lane_n = n;
    env->lane_polys = polys;
    env->lane_heap_cap = heap_cap;
    env->lane_x = f;
    env->lane_y = f + n;
    env->lane_hx = f + 2 * n;
    env->lane_hy = f + 3 * n;
    env->lane_dist = f + 4 * n;
    env->lane_block = f + 5 * n;
    env->lane_heap_d = f + 6 * n;
    env->lane_next = ip;
    env->lane_order = ip + n;
    env->lane_heap_i = ip + 2 * n;
    env->lane_poly_start = ip + 2 * n + heap_cap;

    int cursor = 0;
    int pi = 0;
    for (int i = 0; i < env->num_entities; i++) {
        Entity *e = &env->entities[i];
        if (e->type != ROAD_LANE || e->array_size < 2)
            continue;
        env->lane_poly_start[pi++] = cursor;
        for (int k = 0; k < e->array_size; k++) {
            int a = k;
            int b = k + 1;
            if (b == e->array_size) {
                a = k - 1;
                b = k;
            }
            float dx = e->traj_x[b] - e->traj_x[a];
            float dy = e->traj_y[b] - e->traj_y[a];
            float hyp = sqrtf(dx * dx + dy * dy);
            if (hyp > 0.0f) {
                dx /= hyp;
                dy /= hyp;
            }
            env->lane_x[cursor] = e->traj_x[k];
            env->lane_y[cursor] = e->traj_y[k];
            env->lane_hx[cursor] = dx;
            env->lane_hy[cursor] = dy;
            env->lane_next[cursor] = (k + 1 < e->array_size) ? cursor + 1 : -1;
            cursor++;
        }
    }
}

static int lane_succ(Drive *env, int node, int *out) {
    int nxt = env->lane_next[node];
    if (nxt >= 0) {
        out[0] = nxt;
        return 1;
    }
    int nout = 0;
    float x = env->lane_x[node];
    float y = env->lane_y[node];
    float hx = env->lane_hx[node];
    float hy = env->lane_hy[node];
    float stitch2 = LANE_STITCH * LANE_STITCH;
    for (int p = 0; p < env->lane_polys; p++) {
        int s = env->lane_poly_start[p];
        if (s == node)
            continue;
        float dx = env->lane_x[s] - x;
        float dy = env->lane_y[s] - y;
        float d2 = dx * dx + dy * dy;
        if (d2 > stitch2)
            continue;
        float hdot = hx * env->lane_hx[s] + hy * env->lane_hy[s];
        if (hdot <= LANE_HEADING_DOT)
            continue;
        if (d2 > 1.0e-6f) {
            float dist = sqrtf(d2);
            float sdot = (dx / dist) * hx + (dy / dist) * hy;
            if (sdot <= LANE_HEADING_DOT)
                continue;
        }
        if (nout >= LANE_SUCC_MAX)
            break;
        out[nout++] = s;
    }
    return nout;
}

static void lane_heap_push(Drive *env, int *nheap, float d, int node) {
    if (*nheap >= env->lane_heap_cap)
        return;
    int i = (*nheap)++;
    env->lane_heap_d[i] = d;
    env->lane_heap_i[i] = node;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (env->lane_heap_d[p] <= env->lane_heap_d[i])
            break;
        float td = env->lane_heap_d[p];
        int ti = env->lane_heap_i[p];
        env->lane_heap_d[p] = env->lane_heap_d[i];
        env->lane_heap_i[p] = env->lane_heap_i[i];
        env->lane_heap_d[i] = td;
        env->lane_heap_i[i] = ti;
        i = p;
    }
}

static int lane_heap_pop(Drive *env, int *nheap, float *d, int *node) {
    if (*nheap <= 0)
        return 0;
    *d = env->lane_heap_d[0];
    *node = env->lane_heap_i[0];
    (*nheap)--;
    if (*nheap == 0)
        return 1;
    env->lane_heap_d[0] = env->lane_heap_d[*nheap];
    env->lane_heap_i[0] = env->lane_heap_i[*nheap];
    int i = 0;
    while (1) {
        int l = 2 * i + 1;
        int r = l + 1;
        int s = i;
        if (l < *nheap && env->lane_heap_d[l] < env->lane_heap_d[s])
            s = l;
        if (r < *nheap && env->lane_heap_d[r] < env->lane_heap_d[s])
            s = r;
        if (s == i)
            break;
        float td = env->lane_heap_d[i];
        int ti = env->lane_heap_i[i];
        env->lane_heap_d[i] = env->lane_heap_d[s];
        env->lane_heap_i[i] = env->lane_heap_i[s];
        env->lane_heap_d[s] = td;
        env->lane_heap_i[s] = ti;
        i = s;
    }
    return 1;
}

static int car_blocks_node(Drive *env, int node, int self_idx) {
    float x = env->lane_x[node];
    float y = env->lane_y[node];
    float hx = env->lane_hx[node];
    float hy = env->lane_hy[node];
    float r2 = LANE_CAR_RADIUS * LANE_CAR_RADIUS;
    for (int i = 0; i < env->active_agent_count; i++) {
        int idx = env->active_agent_indices[i];
        if (idx == self_idx)
            continue;
        Entity *o = &env->entities[idx];
        if (o->removed || o->x == INVALID_POSITION)
            continue;
        float dx = o->x - x;
        float dy = o->y - y;
        if (dx * dx + dy * dy > r2)
            continue;
        if (o->heading_x * hx + o->heading_y * hy > LANE_HEADING_DOT)
            return 1;
    }
    return 0;
}

// Local Dijkstra, stopped at goal_distance_max * 3. Does not walk past the
// next aligned car. A full-map search is too slow on Town06.
static void lane_search(Drive *env, int start, int self_idx) {
    int n = env->lane_n;
    for (int i = 0; i < n; i++) {
        env->lane_dist[i] = -1.0f;
        env->lane_block[i] = 1.0e30f;
    }
    if (start < 0 || start >= n)
        return;
    float path_cap = env->goal_distance_max * 3.0f + GOAL_FOLLOW_GAP;
    env->lane_dist[start] = 0.0f;
    int nheap = 0;
    lane_heap_push(env, &nheap, 0.0f, start);
    while (nheap > 0) {
        float popped;
        int u;
        if (!lane_heap_pop(env, &nheap, &popped, &u))
            break;
        if (popped > env->lane_dist[u] + 1.0e-4f)
            continue;
        if (u != start && car_blocks_node(env, u, self_idx)) {
            env->lane_block[u] = env->lane_dist[u];
            continue;
        }
        int succ[LANE_SUCC_MAX];
        int fn = lane_succ(env, u, succ);
        for (int k = 0; k < fn; k++) {
            int v = succ[k];
            float dx = env->lane_x[v] - env->lane_x[u];
            float dy = env->lane_y[v] - env->lane_y[u];
            float nd = env->lane_dist[u] + sqrtf(dx * dx + dy * dy);
            if (nd > path_cap)
                continue;
            if (env->lane_dist[v] >= 0.0f && nd >= env->lane_dist[v])
                continue;
            env->lane_dist[v] = nd;
            lane_heap_push(env, &nheap, nd, v);
        }
    }
    int reached = 0;
    for (int i = 0; i < n; i++) {
        if (env->lane_dist[i] >= 0.0f)
            env->lane_order[reached++] = i;
    }
    g_lane_dist_sort = env->lane_dist;
    qsort(env->lane_order, (size_t)reached, sizeof(int), cmp_lane_dist_desc);
    for (int t = 0; t < reached; t++) {
        int u = env->lane_order[t];
        float block = env->lane_block[u];
        int succ[LANE_SUCC_MAX];
        int fn = lane_succ(env, u, succ);
        for (int k = 0; k < fn; k++) {
            int v = succ[k];
            if (env->lane_dist[v] < 0.0f)
                continue;
            if (env->lane_dist[v] + 1.0e-3f < env->lane_dist[u])
                continue;
            if (env->lane_block[v] < block)
                block = env->lane_block[v];
        }
        env->lane_block[u] = block;
    }
}

static int snap_lane_node(Drive *env, Entity *e) {
    if (env->lane_n < 2)
        return 0;
    int best = -1;
    float best_d = LANE_NODE_SNAP * LANE_NODE_SNAP;
    for (int i = 0; i < env->lane_n; i++) {
        float dx = env->lane_x[i] - e->x;
        float dy = env->lane_y[i] - e->y;
        float d2 = dx * dx + dy * dy;
        if (d2 >= best_d)
            continue;
        float dot = e->heading_x * env->lane_hx[i] + e->heading_y * env->lane_hy[i];
        if (dot <= LANE_HEADING_DOT)
            continue;
        best_d = d2;
        best = i;
    }
    if (best < 0)
        return 0;
    e->lane_node = best;
    return 1;
}

static int pick_forward_goal(Drive *env, Entity *e, int self_idx, int count_goal) {
    if (env->lane_n < 2)
        return 0;
    int node = e->lane_node;
    if (node < 0 || node >= env->lane_n)
        return 0;
    lane_search(env, node, self_idx);
    float lo2 = env->goal_distance_min * env->goal_distance_min;
    float hi2 = env->goal_distance_max * env->goal_distance_max;
    int seen = 0;
    float pick_x = e->x;
    float pick_y = e->y;
    for (int i = 0; i < env->lane_n; i++) {
        float path = env->lane_dist[i];
        if (path <= 0.0f)
            continue;
        float limit = env->lane_block[i] >= 1.0e29f ? 1.0e30f : env->lane_block[i] - GOAL_FOLLOW_GAP;
        if (path > limit)
            continue;
        float dx = env->lane_x[i] - e->x;
        float dy = env->lane_y[i] - e->y;
        float d2 = dx * dx + dy * dy;
        if (d2 < lo2 || d2 > hi2)
            continue;
        if (dx * e->heading_x + dy * e->heading_y <= 0.0f)
            continue;
        seen++;
        if (rand() % seen == 0) {
            pick_x = env->lane_x[i];
            pick_y = env->lane_y[i];
        }
    }
    if (seen == 0)
        return 0;
    e->goal_position_x = pick_x;
    e->goal_position_y = pick_y;
    if (count_goal)
        e->goals_sampled_this_episode += 1.0f;
    return 1;
}

static void sit_on_lane(Drive *env, Entity *e, int k) {
    e->x = env->lane_x[k];
    e->y = env->lane_y[k];
    e->z = 0.0f;
    e->heading_x = env->lane_hx[k];
    e->heading_y = env->lane_hy[k];
    e->heading = atan2f(e->heading_y, e->heading_x);
    e->lane_node = k;
    e->valid = 1;
    e->removed = 0;
    e->stopped = 0;
}

static int spot_taken(Drive *env, int agent_idx) {
    Entity *e = &env->entities[agent_idx];
    for (int i = 0; i < env->active_agent_count; i++) {
        int idx = env->active_agent_indices[i];
        if (idx == agent_idx)
            continue;
        Entity *o = &env->entities[idx];
        if (o->x == INVALID_POSITION || o->removed)
            continue;
        if (check_aabb_collision(e, o))
            return 1;
    }
    return 0;
}

static int place_car(Drive *env, int agent_idx) {
    Entity *e = &env->entities[agent_idx];
    int npts = env->lane_n;
    if (npts < 2)
        return 0;
    for (int attempt = 0; attempt < 1024; attempt++) {
        int k = rand() % npts;
        if (env->lane_hx[k] == 0.0f && env->lane_hy[k] == 0.0f)
            continue;
        sit_on_lane(env, e, k);
        if (spot_taken(env, agent_idx)) {
            e->x = INVALID_POSITION;
            e->y = INVALID_POSITION;
            continue;
        }
        e->collision_state = 0;
        e->offroad_state = 0;
        compute_agent_metrics(env, agent_idx);
        if (e->collision_state || e->offroad_state) {
            e->collision_state = 0;
            e->offroad_state = 0;
            e->x = INVALID_POSITION;
            e->y = INVALID_POSITION;
            continue;
        }
        return 1;
    }
    sit_on_lane(env, e, 0);
    e->collision_state = 0;
    e->offroad_state = 0;
    return 1;
}

static void place_spawned(Drive *env) {
    int n = env->active_agent_count;
    for (int i = 0; i < n; i++) {
        Entity *e = &env->entities[env->active_agent_indices[i]];
        e->x = INVALID_POSITION;
        e->y = INVALID_POSITION;
        e->vx = 0.0f;
        e->vy = 0.0f;
        e->collision_state = 0;
        e->offroad_state = 0;
        e->removed = 0;
        e->stopped = 0;
        e->lane_node = -1;
    }
    for (int i = 0; i < n; i++)
        place_car(env, env->active_agent_indices[i]);
    for (int round = 0; round < 64; round++) {
        int fail = -1;
        for (int i = 0; i < n; i++) {
            int idx = env->active_agent_indices[i];
            Entity *e = &env->entities[idx];
            if (e->x == INVALID_POSITION)
                continue;
            if (!pick_forward_goal(env, e, idx, 0)) {
                fail = idx;
                break;
            }
        }
        if (fail < 0)
            break;
        env->entities[fail].x = INVALID_POSITION;
        env->entities[fail].y = INVALID_POSITION;
        place_car(env, fail);
    }
    for (int i = 0; i < n; i++) {
        int idx = env->active_agent_indices[i];
        Entity *e = &env->entities[idx];
        if (e->x == INVALID_POSITION)
            place_car(env, idx);
        float dx = e->goal_position_x - e->x;
        float dy = e->goal_position_y - e->y;
        float ahead = dx * e->heading_x + dy * e->heading_y;
        float dist = sqrtf(dx * dx + dy * dy);
        if (ahead <= 0.0f || dist < env->goal_distance_min || dist > env->goal_distance_max)
            sample_new_goal(env, idx);
        float speed = ((float)rand() / (float)RAND_MAX) * SPAWN_SPEED_MAX;
        e->vx = speed * e->heading_x;
        e->vy = speed * e->heading_y;
        e->vz = 0.0f;
        e->init_goal_x = e->goal_position_x;
        e->init_goal_y = e->goal_position_y;
        e->collision_state = 0;
        e->offroad_state = 0;
    }
}

void sample_new_goal(Drive *env, int agent_idx) {
    // Samples a new goal position based on the existing road lane points
    Entity *agent = &env->entities[agent_idx];
    float best_x = agent->x;
    float best_y = agent->y;
    float best_distance_error = 1e30f;

    // Sample points from all road lanes
    for (int i = env->num_objects; i < env->num_entities; i++) {
        if (env->entities[i].type != ROAD_LANE)
            continue;

        Entity *lane = &env->entities[i];

        // Check every point in the lane
        for (int j = 0; j < lane->array_size; j++) {
            float point_x = lane->traj_x[j];
            float point_y = lane->traj_y[j];

            // Calculate vector from agent to point
            float to_point_x = point_x - agent->x;
            float to_point_y = point_y - agent->y;

            // Check if point is ahead of agent
            float dot = to_point_x * agent->heading_x + to_point_y * agent->heading_y;
            if (dot <= 0.0f)
                continue;

            // Calculate distance to point
            float distance = sqrtf(to_point_x * to_point_x + to_point_y * to_point_y);

            // Find point closest to target distance
            float distance_error = fabsf(distance - env->goal_target_distance);
            if (distance_error < best_distance_error) {
                best_distance_error = distance_error;
                best_x = point_x;
                best_y = point_y;
            }
        }
    }

    // If no valid goal found, use another agent's initial goal
    if (best_distance_error >= 1e30f && env->active_agent_count > 1) {
        int other_idx = env->active_agent_indices[(agent_idx + 1) % env->active_agent_count];
        best_x = env->entities[other_idx].init_goal_x;
        best_y = env->entities[other_idx].init_goal_y;
    }

    agent->goal_position_x = best_x;
    agent->goal_position_y = best_y;
    agent->goals_sampled_this_episode += 1;
}

// One cruise setpoint, in m/s, shared by every agent.
static float shared_cruise_target(Drive *env) {
    return env->target_speed;
}

void c_reset(Drive *env) {
    env->timestep = env->init_steps;
    if (env->spawn_agents > 0)
        place_spawned(env);
    else
        set_start_position(env);
    for (int x = 0; x < env->active_agent_count; x++) {
        env->logs[x] = (Log){0};
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].failure_before_goal = 0;
        env->entities[agent_idx].distance_m = 0.0f;
        env->entities[agent_idx].failure_steps = 0.0f;
        env->entities[agent_idx].goals_reached_this_episode = 0.0f;
        env->entities[agent_idx].goals_sampled_this_episode = 1.0f;
        env->entities[agent_idx].current_goal_reached = 0;
        env->entities[agent_idx].stopped = 0;
        env->entities[agent_idx].removed = 0;
        env->entities[agent_idx].cruise_target = shared_cruise_target(env);

        if (env->goal_behavior == GOAL_GENERATE_NEW) {
            env->entities[agent_idx].goal_position_x = env->entities[agent_idx].init_goal_x;
            env->entities[agent_idx].goal_position_y = env->entities[agent_idx].init_goal_y;
        }

        // Snapshot initial distance to goal for progress normalization
        env->entities[agent_idx].init_dist_to_goal =
            relative_distance_2d(env->entities[agent_idx].x, env->entities[agent_idx].y,
                                 env->entities[agent_idx].goal_position_x, env->entities[agent_idx].goal_position_y);

        // Conditioning
        // printf("lam %f \n", env->lambda_value);

        if (env->fix_lambdas) {
            env->entities[agent_idx].lambda = env->lambda_value;
        } else {
            env->entities[agent_idx].lambda = (float)rand() / (float)RAND_MAX * 0.5;
        }

        if (env->fix_rewards) {
            env->entities[agent_idx].reward_collision_cond = env->reward_vehicle_collision;
            env->entities[agent_idx].reward_offroad_cond = env->reward_offroad_collision;
            env->entities[agent_idx].reward_goal_cond = env->reward_goal;
        } else {
            float u = (float)rand() / (float)RAND_MAX;
            float range = 0.1f - env->reward_vehicle_collision;
            env->entities[agent_idx].reward_collision_cond = env->reward_vehicle_collision + u * range;

            u = (float)rand() / (float)RAND_MAX;
            range = 0.001f - env->reward_offroad_collision;
            env->entities[agent_idx].reward_offroad_cond = env->reward_offroad_collision + u * range;
            env->entities[agent_idx].reward_goal_cond = env->reward_goal;
        }
        compute_agent_metrics(env, agent_idx);
    }
    compute_observations(env);
}

static void override_action_with_expert(Drive *env, int action_idx, int agent_idx, int t) {
    Entity *agent = &env->entities[agent_idx];

    if (t < 0 || t >= agent->array_size)
        return;

    if (env->dynamics_model == DELTA_LOCAL) {
        if (agent->expert_delta_x == NULL)
            return;
        if (agent->expert_delta_x[t] == -1.0f && agent->expert_delta_y[t] == -1.0f)
            return;

        if (env->action_type == 1) {
            float *action_base = (float *)env->actions;
            action_base[action_idx * 3 + 0] = agent->expert_delta_x[t] / DELTA_MAX_DX;
            action_base[action_idx * 3 + 1] = agent->expert_delta_y[t] / DELTA_MAX_DY;
            action_base[action_idx * 3 + 2] = agent->expert_delta_yaw[t] / DELTA_MAX_DYAW;
        } else {
            int best_dx = 0, best_dy = 0, best_yaw = 0;
            for (int j = 1; j < NUM_DX_BINS; j++)
                if (fabsf(agent->expert_delta_x[t] - DELTA_DX_VALUES[j]) <
                    fabsf(agent->expert_delta_x[t] - DELTA_DX_VALUES[best_dx]))
                    best_dx = j;
            for (int j = 1; j < NUM_DY_BINS; j++)
                if (fabsf(agent->expert_delta_y[t] - DELTA_DY_VALUES[j]) <
                    fabsf(agent->expert_delta_y[t] - DELTA_DY_VALUES[best_dy]))
                    best_dy = j;
            for (int j = 1; j < NUM_YAW_BINS; j++)
                if (fabsf(agent->expert_delta_yaw[t] - DELTA_YAW_VALUES[j]) <
                    fabsf(agent->expert_delta_yaw[t] - DELTA_YAW_VALUES[best_yaw]))
                    best_yaw = j;
            int *action_array = (int *)env->actions;
            action_array[action_idx * 3 + 0] = best_dx;
            action_array[action_idx * 3 + 1] = best_dy;
            action_array[action_idx * 3 + 2] = best_yaw;
        }
    } else {
        if (agent->expert_accel == NULL || agent->expert_steering == NULL)
            return;
        if (agent->expert_accel[t] == -1.0f || agent->expert_steering[t] == -1.0f)
            return;

        if (env->action_type == 1) {
            float (*action_array_f)[2] = (float (*)[2])env->actions;
            action_array_f[action_idx][0] = agent->expert_accel[t] / ACCEL_MAX;
            action_array_f[action_idx][1] = agent->expert_steering[t] / STEER_MAX;
        } else {
            int best_accel_idx = 0;
            float min_accel_diff = fabsf(agent->expert_accel[t] - ACCELERATION_VALUES[0]);
            for (int j = 1; j < NUM_ACCEL_BINS; j++) {
                float diff = fabsf(agent->expert_accel[t] - ACCELERATION_VALUES[j]);
                if (diff < min_accel_diff) {
                    min_accel_diff = diff;
                    best_accel_idx = j;
                }
            }
            int best_steer_idx = 0;
            float min_steer_diff = fabsf(agent->expert_steering[t] - STEERING_VALUES[0]);
            for (int j = 1; j < NUM_STEER_BINS; j++) {
                float diff = fabsf(agent->expert_steering[t] - STEERING_VALUES[j]);
                if (diff < min_steer_diff) {
                    min_steer_diff = diff;
                    best_steer_idx = j;
                }
            }
            int *action_array = (int *)env->actions;
            action_array[action_idx * 2 + 0] = best_accel_idx;
            action_array[action_idx * 2 + 1] = best_steer_idx;
        }
    }
}

// Closest vehicle ahead in a ±15° cone. Gap is bumper-to-bumper along the ego heading.
// Returns 0 when the lane ahead is empty. v_lead is that vehicle's speed along the same axis.
static int lead_vehicle(Drive *env, int agent_idx, float *gap_out, float *v_lead_out) {
    Entity *ego = &env->entities[agent_idx];
    const float cone = 15.0f * (float)M_PI / 180.0f;
    int counts[2] = {env->active_agent_count, env->static_agent_count};
    int *lists[2] = {env->active_agent_indices, env->static_agent_indices};
    float best_gap = 1.0e30f;
    float best_v = 0.0f;
    int found = 0;
    for (int pass = 0; pass < 2; pass++) {
        for (int j = 0; j < counts[pass]; j++) {
            int index = lists[pass][j];
            if (index == agent_idx)
                continue;
            Entity *other = &env->entities[index];
            if (other->removed || other->type != VEHICLE || other->x == INVALID_POSITION)
                continue;
            float dx = other->x - ego->x;
            float dy = other->y - ego->y;
            float fwd = dx * ego->heading_x + dy * ego->heading_y;
            if (fwd <= 0.0f)
                continue;
            float left = -dx * ego->heading_y + dy * ego->heading_x;
            if (fabsf(atan2f(left, fwd)) > cone)
                continue;
            float gap = fwd - 0.5f * ego->length - 0.5f * other->length;
            if (gap < 0.0f)
                gap = 0.0f;
            if (gap < best_gap) {
                best_gap = gap;
                best_v = other->vx * ego->heading_x + other->vy * ego->heading_y;
                found = 1;
            }
        }
    }
    if (!found)
        return 0;
    *gap_out = best_gap;
    *v_lead_out = best_v;
    return 1;
}

// 2-second rule: free flow when the bumper gap clears d0 + 2*v and the lead is not closing inside 3 s.
static int free_flow_step(Drive *env, int agent_idx, float signed_speed) {
    float gap = 0.0f;
    float v_lead = 0.0f;
    if (!lead_vehicle(env, agent_idx, &gap, &v_lead))
        return 1;
    int gap_ok = gap > (2.5f + 2.0f * signed_speed);
    float closing = signed_speed - v_lead;
    float ttc = gap / fmaxf(closing, 1.0e-6f);
    int closing_ok = (signed_speed <= v_lead) || (ttc > 3.0f);
    return gap_ok && closing_ok;
}

static void record_cruise_metrics(Drive *env, int log_idx, int agent_idx, float signed_speed) {
    Log *lg = &env->logs[log_idx];
    Entity *agent = &env->entities[agent_idx];
    float v_tgt = agent->cruise_target;
    lg->cruise_steps += 1.0f;
    float eps = fmaxf(1.0f, 0.1f * v_tgt);
    if (signed_speed > v_tgt + eps)
        lg->speeding_steps += 1.0f;
    if (!free_flow_step(env, agent_idx, signed_speed))
        return;
    float sigma = env->speed_reward_sigma > 0.0f ? env->speed_reward_sigma : 2.5f;
    float dv = signed_speed - v_tgt;
    lg->ff_steps += 1.0f;
    lg->ff_gauss += expf(-(dv * dv) / (2.0f * sigma * sigma));
}

static void move_agent_with_controller(Drive *env, int action_idx, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];

    if (agent->controller == CONTROLLER_STATIC) {
        return;
    }

    if (agent->controller == CONTROLLER_REPLAY) {
        move_expert(env, env->actions, agent_idx);
        return;
    }

    if (action_idx >= 0) {
        move_dynamics(env, action_idx, agent_idx);
    }
}

void c_step(Drive *env) {
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
    memset(env->truncations, 0, env->active_agent_count * sizeof(unsigned char));

    // If async_resets is off and we already finished, just re-assert done signals
    if (!env->async_resets && (env->timestep + 1) >= env->episode_length) {
        for (int i = 0; i < env->active_agent_count; i++) {
            env->truncations[i] = 1;
            env->terminals[i] = 1;
        }
        return;
    }

    // printf("t = %d \n", env->timestep);

    // Re-assert terminal for agents already terminated in a prior step
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        if (env->entities[agent_idx].removed) {
            env->terminals[i] = 1;
        }
    }

    env->timestep++;

    // Move background agents according to their per-agent controller.
    for (int i = 0; i < env->static_agent_count; i++) {
        int background_idx = env->static_agent_indices[i];
        if (env->entities[background_idx].x == INVALID_POSITION)
            continue;
        move_agent_with_controller(env, -1, background_idx);
    }

    // Process actions for all active agents
    for (int i = 0; i < env->active_agent_count; i++) {
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];

        if (env->entities[agent_idx].removed)
            continue;

        if (env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS) {
            override_action_with_expert(env, i, agent_idx, env->timestep - 1);
        }

        move_agent_with_controller(env, i, agent_idx);

        if (env->entities[agent_idx].controller == CONTROLLER_POLICY) {
            // Apply sensor noise to policy-driven dynamics only.
            apply_dynamics_noise(env, agent_idx);
        }
    }

    // Speed is the post-dynamics value. Collision handling below can zero it.
    for (int i = 0; i < env->active_agent_count; i++) {
        if (env->terminals[i])
            continue;
        int agent_idx = env->active_agent_indices[i];
        Entity *agent = &env->entities[agent_idx];
        if (agent->x == INVALID_POSITION)
            continue;
        float along = agent->vx * agent->heading_x + agent->vy * agent->heading_y;
        float spd = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
        record_cruise_metrics(env, i, agent_idx, copysignf(spd, along));
    }

    // Compute rewards
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        int agent_is_done = env->terminals[i];

        env->entities[agent_idx].collision_state = 0;
        env->entities[agent_idx].offroad_state = 0;

        if (agent_is_done)
            continue;

        compute_agent_metrics(env, agent_idx);

        int collision_state = env->entities[agent_idx].collision_state;
        int offroad_state = env->entities[agent_idx].offroad_state;

        // Charge on every overlapping step, as in the ezmuwiyj Waymo run.
        // Stage 2 still stops after one charge because terminate_on_failure
        // removes the car before the next step.
        if (collision_state == 1 && !agent_is_done) {
            float r_collision = env->entities[agent_idx].reward_collision_cond;
            env->rewards[i] += r_collision;
            env->logs[i].episode_return += r_collision;
            env->logs[i].collision_rate = 1.0f;
            env->logs[i].collisions_per_agent += 1.0f;
        }

        if (offroad_state == 1 && !agent_is_done) {
            float r_off = env->entities[agent_idx].reward_offroad_cond;
            env->rewards[i] += r_off;
            env->logs[i].episode_return += r_off;
            env->logs[i].offroad_rate = 1.0f;
            env->logs[i].offroad_per_agent += 1.0f;
        }

        if ((collision_state || offroad_state) && !agent_is_done) {
            env->entities[agent_idx].failure_before_goal = 1;
            env->entities[agent_idx].failure_steps += 1.0f;
            // End this agent. A later cruise return must not pay for the crash.
            if (env->terminate_on_failure) {
                env->terminals[i] = 1;
                env->entities[agent_idx].removed = 1;
                env->entities[agent_idx].stopped = 1;
                env->entities[agent_idx].vx = 0.0f;
                env->entities[agent_idx].vy = 0.0f;
            }
        } else if (!agent_is_done && env->speed_reward_weight > 0.0f && env->speed_reward_sigma > 0.0f) {
            float along = env->entities[agent_idx].vx * env->entities[agent_idx].heading_x +
                          env->entities[agent_idx].vy * env->entities[agent_idx].heading_y;
            float spd = sqrtf(env->entities[agent_idx].vx * env->entities[agent_idx].vx +
                              env->entities[agent_idx].vy * env->entities[agent_idx].vy);
            float v = copysignf(spd, along);
            float dv = v - env->entities[agent_idx].cruise_target;
            float sig2 = env->speed_reward_sigma * env->speed_reward_sigma;
            float r_speed = env->speed_reward_weight * expf(-(dv * dv) / (2.0f * sig2));
            env->rewards[i] += r_speed;
            env->logs[i].episode_return += r_speed;
        }
        if (!agent_is_done && env->accel_penalty > 0.0f && env->dynamics_model == CLASSIC) {
            float a = env->entities[agent_idx].a_long;
            float r_acc = -env->accel_penalty * a * a;
            env->rewards[i] += r_acc;
            env->logs[i].episode_return += r_acc;
        }
        if (!agent_is_done) {
            float step_speed = sqrtf(env->entities[agent_idx].vx * env->entities[agent_idx].vx +
                                     env->entities[agent_idx].vy * env->entities[agent_idx].vy);
            env->entities[agent_idx].distance_m += step_speed * env->dt;
        }

        // Check if agent reached goal
        float distance_to_goal =
            relative_distance_2d(env->entities[agent_idx].x, env->entities[agent_idx].y,
                                 env->entities[agent_idx].goal_position_x, env->entities[agent_idx].goal_position_y);

        float current_speed = sqrtf(env->entities[agent_idx].vx * env->entities[agent_idx].vx +
                                    env->entities[agent_idx].vy * env->entities[agent_idx].vy);

        // Arrival is the radius. Stage 2 pays once, samples a new point, and
        // does not end the episode.
        bool within_distance = distance_to_goal < env->goal_radius;

        if (within_distance && !env->entities[agent_idx].current_goal_reached &&
            !env->terminals[i]) {
            float r_goal = env->entities[agent_idx].reward_goal_cond;
            env->rewards[i] += r_goal;
            env->logs[i].episode_return += r_goal;
            env->entities[agent_idx].goals_reached_this_episode += 1.0f;
            env->entities[agent_idx].current_goal_reached = 1;
            env->logs[i].speed_at_goal = current_speed;

            if (env->goal_behavior == GOAL_GENERATE_NEW) {
                int assigned = 0;
                if (env->spawn_agents > 0) {
                    snap_lane_node(env, &env->entities[agent_idx]);
                    assigned = pick_forward_goal(env, &env->entities[agent_idx], agent_idx, 1);
                }
                if (!assigned)
                    sample_new_goal(env, agent_idx);
                float new_dist =
                    relative_distance_2d(env->entities[agent_idx].x, env->entities[agent_idx].y,
                                         env->entities[agent_idx].goal_position_x,
                                         env->entities[agent_idx].goal_position_y);
                if (new_dist >= env->goal_radius)
                    env->entities[agent_idx].current_goal_reached = 0;
            }
        }

        // Per-step metrics accumulation (only while agent is alive)
        if (!agent_is_done && !env->entities[agent_idx].removed) {
            float disp_err = 0.0f;
            compute_route_progress_and_errors(&env->entities[agent_idx], env->entities[agent_idx].x,
                                              env->entities[agent_idx].y, env->init_steps, env->timestep, &disp_err);

            if (disp_err >= 0.0f) {
                env->logs[i].displacement_error_avg += disp_err;
                env->logs[i].displacement_samples += 1.0f;
            }
        }
    }

    // Handle agents
    if (env->goal_behavior == GOAL_REMOVE) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            if (env->entities[agent_idx].current_goal_reached) {
                // Route progress = 1 by definition when goal is reached
                env->logs[i].route_progress = 1.0f;
                env->terminals[i] = 1;
                env->entities[agent_idx].removed = 1;
                env->entities[agent_idx].stopped = 1;
                env->entities[agent_idx].vx = env->entities[agent_idx].vy = 0.0f;
                env->entities[agent_idx].x = INVALID_POSITION;
                env->entities[agent_idx].y = INVALID_POSITION;
            }
        }

    } else if (env->goal_behavior == GOAL_STOP) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            int reached_goal = env->entities[agent_idx].current_goal_reached;
            if (reached_goal) {
                env->terminals[i] = 1; // Mark agent as done
                env->entities[agent_idx].stopped = 1;
                env->entities[agent_idx].vx = env->entities[agent_idx].vy = 0.0f;
            }
        }
    }

    // Truncate episode when all agents are terminated
    int all_terminated = 1;
    if (env->termination_mode == 1) {
        for (int i = 0; i < env->active_agent_count; i++) {
            if (!env->terminals[i]) {
                all_terminated = 0;
                break;
            }
        }
    }
    int reached_max_episode_len = (env->timestep + 1) >= env->episode_length;
    if (reached_max_episode_len || (all_terminated && env->termination_mode == 1)) {
        for (int i = 0; i < env->active_agent_count; i++) {
            env->truncations[i] = 1;
        }
        add_log(env);
        if (env->async_resets) {
            c_reset(env);
        }
        return;
    }

    compute_observations(env);
}

void c_collect_expert_data(Drive *env, float *expert_actions_discrete_out, float *expert_obs_out) {
    int ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES;
    int max_obs = DRIVE_MAX_OBS(ego_dim);

    int saved_control_mode = env->control_mode;
    int saved_goal_behavior = env->goal_behavior;
    int saved_episode_length = env->episode_length;
    int saved_termination_mode = env->termination_mode;

    env->control_mode = CONTROL_INFERRED_EXPERT_ACTIONS;
    env->episode_length = TRAJECTORY_LENGTH + 1;
    env->goal_behavior = GOAL_STOP;
    env->termination_mode = 0;
    env->init_steps = 0;

    c_reset(env); // Reset all agents to their initial position

    for (int t = 0; t < TRAJECTORY_LENGTH; t++) {
        int obs_offset = t * env->active_agent_count * max_obs;
        memcpy(&expert_obs_out[obs_offset], env->observations, env->active_agent_count * max_obs * sizeof(float));

        // Write expert actions into env->actions for this timestep
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            override_action_with_expert(env, i, agent_idx, t);
        }

        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            Entity *agent = &env->entities[agent_idx];

            if (agent->inferred_traj_x != NULL) {
                agent->inferred_traj_x[t] = agent->x;
                agent->inferred_traj_y[t] = agent->y;
            }
        }
        if (t < TRAJECTORY_LENGTH - 1) {
            c_step(env);

            if (COLLECT_EXPERT_TELEPORT) {
                // Overwrite active agent positions with ground truth after c_step
                // so that obs are computed from true positions, while static agents
                // were advanced by c_step as a side effect.
                for (int i = 0; i < env->active_agent_count; i++) {
                    int agent_idx = env->active_agent_indices[i];
                    Entity *agent = &env->entities[agent_idx];
                    int next = t + 1;
                    if (next < agent->array_size && agent->traj_valid && agent->traj_valid[next]) {
                        agent->x = agent->traj_x[next];
                        agent->y = agent->traj_y[next];
                        agent->z = agent->traj_z[next];
                        agent->heading = agent->traj_heading[next];
                        agent->heading_x = cosf(agent->heading);
                        agent->heading_y = sinf(agent->heading);
                        agent->vx = agent->traj_vx[next];
                        agent->vy = agent->traj_vy[next];
                        agent->vz = agent->traj_vz[next];
                    }
                }
                compute_observations(env);
            }
        }

        // Record actions
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            Entity *agent = &env->entities[agent_idx];

            int disc_off = t * env->active_agent_count + i;

            bool valid;
            if (env->dynamics_model == DELTA_LOCAL) {
                valid = (t < agent->array_size && agent->expert_delta_x != NULL && agent->expert_delta_x[t] != -1.0f);
            } else {
                valid = (t < agent->array_size && agent->expert_accel != NULL && agent->expert_accel[t] != -1.0f);
            }

            if (valid) {
                if (env->dynamics_model == DELTA_LOCAL) {
                    int *ai = (int *)env->actions;
                    int dx_idx = ai[i * 3 + 0];
                    int dy_idx = ai[i * 3 + 1];
                    int yaw_idx = ai[i * 3 + 2];
                    if (env->action_type == 1) {
                        float *ab = (float *)env->actions;
                        float dx = ab[i * 3 + 0] * DELTA_MAX_DX;
                        float dy = ab[i * 3 + 1] * DELTA_MAX_DY;
                        float dyaw = ab[i * 3 + 2] * DELTA_MAX_DYAW;
                        dx_idx = 0;
                        dy_idx = 0;
                        yaw_idx = 0;
                        for (int j = 1; j < NUM_DX_BINS; j++)
                            if (fabsf(dx - DELTA_DX_VALUES[j]) < fabsf(dx - DELTA_DX_VALUES[dx_idx]))
                                dx_idx = j;
                        for (int j = 1; j < NUM_DY_BINS; j++)
                            if (fabsf(dy - DELTA_DY_VALUES[j]) < fabsf(dy - DELTA_DY_VALUES[dy_idx]))
                                dy_idx = j;
                        for (int j = 1; j < NUM_YAW_BINS; j++)
                            if (fabsf(dyaw - DELTA_YAW_VALUES[j]) < fabsf(dyaw - DELTA_YAW_VALUES[yaw_idx]))
                                yaw_idx = j;
                    }
                    int base = (t * env->active_agent_count + i) * 3;
                    expert_actions_discrete_out[base + 0] = (float)dx_idx;
                    expert_actions_discrete_out[base + 1] = (float)dy_idx;
                    expert_actions_discrete_out[base + 2] = (float)yaw_idx;
                } else if (env->dynamics_model == CLASSIC) {
                    int base = (t * env->active_agent_count + i) * 2;
                    if (env->action_type == 1) {
                        float (*af)[2] = (float (*)[2])env->actions;
                        float accel = af[i][0] * ACCEL_MAX;
                        float steer = af[i][1] * STEER_MAX;
                        int best_a = 0, best_s = 0;
                        for (int j = 1; j < NUM_ACCEL_BINS; j++) {
                            if (fabsf(accel - ACCELERATION_VALUES[j]) < fabsf(accel - ACCELERATION_VALUES[best_a]))
                                best_a = j;
                        }
                        for (int j = 1; j < NUM_STEER_BINS; j++) {
                            if (fabsf(steer - STEERING_VALUES[j]) < fabsf(steer - STEERING_VALUES[best_s]))
                                best_s = j;
                        }
                        expert_actions_discrete_out[base + 0] = (float)best_a;
                        expert_actions_discrete_out[base + 1] = (float)best_s;
                    } else {
                        int *ai = (int *)env->actions;
                        expert_actions_discrete_out[base + 0] = (float)ai[i * 2 + 0];
                        expert_actions_discrete_out[base + 1] = (float)ai[i * 2 + 1];
                    }
                } else {
                    if (env->action_type == 1) {
                        expert_actions_discrete_out[disc_off] = 0.0f;
                    } else {
                        int *ai = (int *)env->actions;
                        expert_actions_discrete_out[disc_off] = (float)ai[i];
                    }
                }
            } else {
                if (env->dynamics_model == DELTA_LOCAL) {
                    int base = (t * env->active_agent_count + i) * 3;
                    expert_actions_discrete_out[base + 0] = -1.0f;
                    expert_actions_discrete_out[base + 1] = -1.0f;
                    expert_actions_discrete_out[base + 2] = -1.0f;
                } else if (env->dynamics_model == CLASSIC) {
                    int base = (t * env->active_agent_count + i) * 2;
                    expert_actions_discrete_out[base + 0] = -1.0f;
                    expert_actions_discrete_out[base + 1] = -1.0f;
                } else {
                    int disc_off = t * env->active_agent_count + i;
                    expert_actions_discrete_out[disc_off] = -1.0f;
                }
            }
        }
    }

    // Filter data: Compute per-agent ADE and mark unfit trajectories
    for (int i = 0; i < env->active_agent_count; i++) {
        int idx = env->active_agent_indices[i];
        Entity *e = &env->entities[idx];
        float total_error = 0.0f;
        int count = 0;
        int is_static = 1;

        if (e->inferred_traj_x && e->traj_valid) {
            for (int t = 0; t < TRAJECTORY_LENGTH; t++) {
                if (!e->traj_valid[t])
                    continue;
                float dx = e->inferred_traj_x[t] - e->traj_x[t];
                float dy = e->inferred_traj_y[t] - e->traj_y[t];
                // printf("t %d error %.2f\n", t, sqrtf(dx * dx + dy * dy));
                total_error += sqrtf(dx * dx + dy * dy);
                count++;
                if (count > 1 && (e->traj_x[t] != e->traj_x[0] || e->traj_y[t] != e->traj_y[0]))
                    is_static = 0;
            }
        }

        e->inferred_ade = (count > 0) ? total_error / count : -1.0f;

        int min_valid = 10;
        int unfit;
        if (COLLECT_EXPERT_TELEPORT) {
            unfit = (count < min_valid || is_static);
        } else {
            unfit = (e->inferred_ade < 0.0f || e->inferred_ade > ADE_THRESHOLD || count < min_valid || is_static);
        }

        if (unfit) {
            for (int t = 0; t < TRAJECTORY_LENGTH; t++) {
                if (env->dynamics_model == DELTA_LOCAL) {
                    int base = (t * env->active_agent_count + i) * 3;
                    expert_actions_discrete_out[base + 0] = -1.0f;
                    expert_actions_discrete_out[base + 1] = -1.0f;
                    expert_actions_discrete_out[base + 2] = -1.0f;
                } else {
                    int disc_off = t * env->active_agent_count + i;
                    expert_actions_discrete_out[disc_off] = -1.0f;
                }
            }
        }
    }

    env->control_mode = saved_control_mode;
    env->goal_behavior = saved_goal_behavior;
    env->episode_length = saved_episode_length;
    env->termination_mode = saved_termination_mode;
}

typedef struct Client Client;

struct Client {
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Model cars[6];
    Model cyclist;
    Model pedestrian;
    ModelAnimation *cycle_anim;
    int car_assignments[MAX_AGENTS];
    Vector3 default_camera_position;
    Vector3 default_camera_target;
    int recorder_pipefd[2];
    pid_t recorder_pid;
    pid_t xvfb_pid;
    int xvfb_display_num;
    int render_mode;
    int follow_ready;
    int follow_agent;
    int egl_mode;
    Mesh road_tri_mesh;
    Material road_material;
    float *road_line_verts;
    unsigned char *road_line_colors;
    int road_line_count;
    int road_cache_valid;
#ifdef DRIVE_HAS_EGL
    unsigned int pbo[2];
#endif
    int pbo_index;
    int pbo_frame_count;
};

static int g_glfw_ready = 0;
#ifdef DRIVE_HAS_EGL
static int g_egl_available = -1;
#endif
static pid_t g_xvfb_pid = 0;
static int g_xvfb_display_num = 0;

void stop_recorder(Client *client) {
    if (client->recorder_pid > 0) {
        close(client->recorder_pipefd[1]);
        waitpid(client->recorder_pid, NULL, 0);
        client->recorder_pid = 0;
    }
}

void start_recorder(Client *client, const char *scenario_id, int width, int height) {
    stop_recorder(client);

    if (pipe(client->recorder_pipefd) == -1) {
        fprintf(stderr, "Failed to create pipe for recorder\n");
        return;
    }

    char size_str[64];
    snprintf(size_str, sizeof(size_str), "%dx%d", width, height);

    char filename[256];
    snprintf(filename, sizeof(filename), "%s.mp4", scenario_id);

    client->recorder_pid = fork();
    if (client->recorder_pid == -1) {
        fprintf(stderr, "Failed to fork recorder\n");
        close(client->recorder_pipefd[0]);
        close(client->recorder_pipefd[1]);
        return;
    }

    if (client->recorder_pid == 0) {
        close(client->recorder_pipefd[1]);
        dup2(client->recorder_pipefd[0], STDIN_FILENO);
        close(client->recorder_pipefd[0]);
        for (int fd = 3; fd < 256; fd++)
            close(fd);
        execlp("ffmpeg", "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", size_str, "-r", "30", "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "23", "-loglevel", "error",
               filename, NULL);
        fprintf(stderr, "execlp ffmpeg failed\n");
        _exit(1);
    }
    close(client->recorder_pipefd[0]);

#ifdef DRIVE_HAS_EGL
    int pipe_sz = fcntl(client->recorder_pipefd[1], F_SETPIPE_SZ, 4 * 1024 * 1024);
    if (pipe_sz > 0) {
        fprintf(stderr, "[drive] Pipe buffer set to %d bytes\n", pipe_sz);
    }
#endif
}

Client *make_client(Drive *env) {

    Client *client = (Client *)calloc(1, sizeof(Client));
    client->render_mode = env->render_mode;

    if (env->render_mode == RENDER_WINDOW) {
        client->width = 1280;
        client->height = 704;
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        SetTargetFPS(30);

        // Set up camera for interactive window
        Vector3 target_pos = {0, 0, 1}; // Y is up, Z is depth

        client->default_camera_position = (Vector3){
            0,      // Same X as target
            120.0f, // 20 units above target
            175.0f  // 20 units behind target
        };
        client->default_camera_target = target_pos;
        client->camera.position = client->default_camera_position;
        client->camera.target = client->default_camera_target;
        client->camera.up = (Vector3){0.0f, -1.0f, 0.0f}; // Y is up
        client->camera.fovy = 45.0f;
        client->camera.projection = CAMERA_PERSPECTIVE;

        SetTraceLogLevel(LOG_WARNING); // Only show warnings and errors
        InitWindow(client->width, client->height, "PufferDrive");
    } else { // Headless rendering
        // Clean-eval videos use a fixed 16:9 frame. This also matches the
        // persistent EGL pbuffer model: every env records into the same size.
        client->width = 1920;
        client->height = 1080;

        if (!g_glfw_ready) {
            if (getenv("DISPLAY") == NULL) {
                g_xvfb_display_num = 100 + (getpid() % 900);
                char display_str[16];
                char lock_file[32];
                char socket_file[32];
                snprintf(display_str, sizeof(display_str), ":%d", g_xvfb_display_num);
                snprintf(lock_file, sizeof(lock_file), "/tmp/.X%d-lock", g_xvfb_display_num);
                snprintf(socket_file, sizeof(socket_file), "/tmp/.X11-unix/X%d", g_xvfb_display_num);

                FILE *f = fopen(lock_file, "r");
                if (f) {
                    pid_t pid = -1;
                    fscanf(f, "%d", &pid);
                    fclose(f);
                    if (pid > 0 && kill(pid, 0) != 0) {
                        unlink(lock_file);
                        unlink(socket_file);
                    }
                }

                g_xvfb_pid = fork();
                if (g_xvfb_pid == 0) {
                    close(STDOUT_FILENO);
                    close(STDERR_FILENO);
                    execlp("Xvfb", "Xvfb", display_str, "-screen", "0", "1280x720x24", "+extension", "GLX", "-ac",
                           "-noreset", NULL);
                    _exit(1);
                }
                setenv("DISPLAY", display_str, 1);
                for (int i = 0; i < 20 && access(lock_file, F_OK) != 0; i++)
                    usleep(100000);
                usleep(200000);

                client->xvfb_pid = g_xvfb_pid;
                client->xvfb_display_num = g_xvfb_display_num;
            }

            SetConfigFlags(FLAG_WINDOW_HIDDEN);
            SetTargetFPS(6000);
            SetTraceLogLevel(LOG_WARNING);
            InitWindow(client->width, client->height, "PufferDrive");
            g_glfw_ready = 1;
        }
    }

    if (!IsWindowReady()) {
        fprintf(stderr, "WARNING: Failed to initialize render window. Rendering disabled.\n");
        free(client);
        return NULL;
    }

#ifdef DRIVE_HAS_EGL
    if (env->render_mode == RENDER_HEADLESS && g_egl_available != 0) {
        if (!g_egl_ctx.active) {
            if (egl_headless_init((int)client->width, (int)client->height) && egl_switch_to_gpu()) {
                rlglInit((int)client->width, (int)client->height);
                rlViewport(0, 0, (int)client->width, (int)client->height);
                rlEnableDepthTest();
                g_egl_available = 1;
            }
            if (g_egl_available < 0) {
                g_egl_available = 0;
                fprintf(stderr, "[drive] EGL GPU unavailable, using Xvfb/Mesa rendering\n");
            }
        } else {
            egl_headless_resize((int)client->width, (int)client->height);
            rlViewport(0, 0, (int)client->width, (int)client->height);
        }
        if (g_egl_available == 1)
            client->egl_mode = 1;
    }
#endif

    // Load assets
    client->cars[0] = LoadModel("resources/drive/RedCar.glb");
    client->cars[1] = LoadModel("resources/drive/WhiteCar.glb");
    client->cars[2] = LoadModel("resources/drive/BlueCar.glb");
    client->cars[3] = LoadModel("resources/drive/YellowCar.glb");
    client->cars[4] = LoadModel("resources/drive/GreenCar.glb");
    client->cars[5] = LoadModel("resources/drive/GreyCar.glb");
    client->cyclist = LoadModel("resources/drive/cyclist.glb");
    client->pedestrian = LoadModel("resources/drive/pedestrian.glb");
    int animCountCyc = 0;
    client->cycle_anim = LoadModelAnimations("resources/drive/cyclist.glb", &animCountCyc);
    for (int i = 0; i < MAX_AGENTS; i++) {
        client->car_assignments[i] = (rand() % 4) + 1;
    }

    return client;
}

// Camera control functions
void handle_camera_controls(Client *client) {
    static Vector2 prev_mouse_pos = {0};
    static bool is_dragging = false;
    float camera_move_speed = 0.5f;

    // Handle mouse drag for camera movement
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        prev_mouse_pos = GetMousePosition();
        is_dragging = true;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        is_dragging = false;
    }

    if (is_dragging) {
        Vector2 current_mouse_pos = GetMousePosition();
        Vector2 delta = {(current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
                         -(current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed};

        // Update camera position (only X and Y)
        client->camera.position.x += delta.x;
        client->camera.position.y += delta.y;

        // Update camera target (only X and Y)
        client->camera.target.x += delta.x;
        client->camera.target.y += delta.y;

        prev_mouse_pos = current_mouse_pos;
    }

    // Handle mouse wheel for zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        float zoom_factor = 1.0f - (wheel * 0.1f);
        // Calculate the current direction vector from target to position
        Vector3 direction = {client->camera.position.x - client->camera.target.x,
                             client->camera.position.y - client->camera.target.y,
                             client->camera.position.z - client->camera.target.z};

        // Scale the direction vector by the zoom factor
        direction.x *= zoom_factor;
        direction.y *= zoom_factor;
        direction.z *= zoom_factor;

        // Update the camera position based on the scaled direction
        client->camera.position.x = client->camera.target.x + direction.x;
        client->camera.position.y = client->camera.target.y + direction.y;
        client->camera.position.z = client->camera.target.z + direction.z;
    }
}

void draw_agent_obs(Drive *env, int agent_index, int mode, int obs_only, int lasers) {
    // Diamond dimensions
    float diamond_height = 3.0f; // Total height of diamond
    float diamond_width = 1.5f;  // Width of diamond
    float diamond_z = 8.0f;      // Base Z position

    // Define diamond points
    Vector3 top_point = (Vector3){0.0f, 0.0f, diamond_z + diamond_height / 2};    // Top point
    Vector3 bottom_point = (Vector3){0.0f, 0.0f, diamond_z - diamond_height / 2}; // Bottom point
    Vector3 front_point = (Vector3){0.0f, diamond_width / 2, diamond_z};          // Front point
    Vector3 back_point = (Vector3){0.0f, -diamond_width / 2, diamond_z};          // Back point
    Vector3 left_point = (Vector3){-diamond_width / 2, 0.0f, diamond_z};          // Left point
    Vector3 right_point = (Vector3){diamond_width / 2, 0.0f, diamond_z};          // Right point

    // Draw the diamond faces
    // Top pyramid
    if (mode == 0) {
        DrawTriangle3D(top_point, front_point, right_point, PUFF_CYAN); // Front-right face
        DrawTriangle3D(top_point, right_point, back_point, PUFF_CYAN);  // Back-right face
        DrawTriangle3D(top_point, back_point, left_point, PUFF_CYAN);   // Back-left face
        DrawTriangle3D(top_point, left_point, front_point, PUFF_CYAN);  // Front-left face

        // Bottom pyramid
        DrawTriangle3D(bottom_point, right_point, front_point, PUFF_CYAN); // Front-right face
        DrawTriangle3D(bottom_point, back_point, right_point, PUFF_CYAN);  // Back-right face
        DrawTriangle3D(bottom_point, left_point, back_point, PUFF_CYAN);   // Back-left face
        DrawTriangle3D(bottom_point, front_point, left_point, PUFF_CYAN);  // Front-left face
    }
    if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
        return;
    }

    int ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES;
    int max_obs = DRIVE_MAX_OBS(ego_dim);
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    float *agent_obs = &observations[agent_index][0];
    // self
    int active_idx = env->active_agent_indices[agent_index];
    float heading_self_x = env->entities[active_idx].heading_x;
    float heading_self_y = env->entities[active_idx].heading_y;
    float px = env->entities[active_idx].x;
    float py = env->entities[active_idx].y;
    // draw goal
    float goal_x = agent_obs[0] * 200;
    float goal_y = agent_obs[1] * 200;

    int agent_type = env->entities[active_idx].type;
    Color goal_color = LIGHTBLUE;
    if (agent_type == PEDESTRIAN)
        goal_color = LIGHT_ORANGE;
    else if (agent_type == CYCLIST)
        goal_color = LIGHT_PURPLE;

    if (mode == 0) { // agent-relative coordinates
        DrawSphere((Vector3){goal_x, goal_y, Z_AGENT_DETAILS}, 0.5f, goal_color);
        DrawCircle3D((Vector3){goal_x, goal_y, Z_AGENT_DETAILS}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f,
                     Fade(goal_color, 0.3f));
    }

    if (mode == 1) { // world coordinates

        float goal_x_world = px + (goal_x * heading_self_x - goal_y * heading_self_y);
        float goal_y_world = py + (goal_x * heading_self_y + goal_y * heading_self_x);
        DrawSphere((Vector3){goal_x_world, goal_y_world, Z_AGENT_DETAILS}, 0.5f, goal_color);
        DrawCircle3D((Vector3){goal_x_world, goal_y_world, Z_AGENT_DETAILS}, env->goal_radius, (Vector3){0, 0, 1},
                     90.0f, Fade(goal_color, 0.3f));
    }
    // First draw other agent observations
    int obs_idx = ego_dim; // Start after ego obs
    for (int j = 0; j < MAX_AGENTS - 1; j++) {
        if (agent_obs[obs_idx] == 0 || agent_obs[obs_idx + 1] == 0) {
            obs_idx += 7; // Move to next agent observation
            continue;
        }
        // Draw position of other agents
        float x = agent_obs[obs_idx] * 50;
        float y = agent_obs[obs_idx + 1] * 50;
        if (lasers && mode == 0) {
            DrawLine3D((Vector3){0, 0, 0}, (Vector3){x, y, Z_AGENT_DETAILS}, ORANGE);
        }

        float partner_x = px + (x * heading_self_x - y * heading_self_y);
        float partner_y = py + (x * heading_self_y + y * heading_self_x);
        if (lasers && mode == 1) {
            DrawLine3D((Vector3){px, py, Z_AGENT_DETAILS}, (Vector3){partner_x, partner_y, Z_AGENT_DETAILS}, ORANGE);
        }

        float half_width = 0.5 * agent_obs[obs_idx + 2] * MAX_VEH_WIDTH;
        float half_len = 0.5 * agent_obs[obs_idx + 3] * MAX_VEH_LEN;
        float theta_x = agent_obs[obs_idx + 4];
        float theta_y = agent_obs[obs_idx + 5];
        float partner_angle = atan2f(theta_y, theta_x);
        float cos_heading = cosf(partner_angle);
        float sin_heading = sinf(partner_angle);
        Vector3 corners[4] = {
            (Vector3){x + (half_len * cos_heading - half_width * sin_heading),
                      y + (half_len * sin_heading + half_width * cos_heading), Z_AGENT_DETAILS},
            (Vector3){x + (half_len * cos_heading + half_width * sin_heading),
                      y + (half_len * sin_heading - half_width * cos_heading), Z_AGENT_DETAILS},
            (Vector3){x + (-half_len * cos_heading + half_width * sin_heading),
                      y + (-half_len * sin_heading - half_width * cos_heading), Z_AGENT_DETAILS},
            (Vector3){x + (-half_len * cos_heading - half_width * sin_heading),
                      y + (-half_len * sin_heading + half_width * cos_heading), Z_AGENT_DETAILS},
        };

        if (mode == 0) {
            for (int j = 0; j < 4; j++) {
                DrawLine3D(corners[j], corners[(j + 1) % 4], ORANGE);
            }
        }

        if (mode == 1) {
            Vector3 world_corners[4];
            for (int j = 0; j < 4; j++) {
                float lx = corners[j].x;
                float ly = corners[j].y;

                world_corners[j].x = px + (lx * heading_self_x - ly * heading_self_y);
                world_corners[j].y = py + (lx * heading_self_y + ly * heading_self_x);
                world_corners[j].z = 1;
            }
            for (int j = 0; j < 4; j++) {
                DrawLine3D(world_corners[j], world_corners[(j + 1) % 4], ORANGE);
            }
        }

        // draw an arrow above the car pointing in the direction that the partner is going
        float arrow_length = 2.5f;
        float arrow_x = x + arrow_length * cosf(partner_angle);
        float arrow_y = y + arrow_length * sinf(partner_angle);
        float arrow_x_world;
        float arrow_y_world;
        if (mode == 0) {
            DrawLine3D((Vector3){x, y, Z_AGENT_DETAILS}, (Vector3){arrow_x, arrow_y, Z_AGENT_DETAILS}, PUFF_WHITE);
        }
        if (mode == 1) {
            arrow_x_world = px + (arrow_x * heading_self_x - arrow_y * heading_self_y);
            arrow_y_world = py + (arrow_x * heading_self_y + arrow_y * heading_self_x);
            DrawLine3D((Vector3){partner_x, partner_y, Z_AGENT_DETAILS},
                       (Vector3){arrow_x_world, arrow_y_world, Z_AGENT_DETAILS}, PUFF_WHITE);
        }
        // Calculate perpendicular offsets for arrow head
        float arrow_size = 0.3f; // Size of the arrow head
        float dx = arrow_x - x;
        float dy = arrow_y - y;
        float length = sqrtf(dx * dx + dy * dy);
        if (length > 0) {
            // Normalize direction vector
            dx /= length;
            dy /= length;

            // Calculate perpendicular vector
            float perp_x = -dy * arrow_size;
            float perp_y = dx * arrow_size;

            float arrow_x_end1 = arrow_x - dx * arrow_size + perp_x;
            float arrow_y_end1 = arrow_y - dy * arrow_size + perp_y;
            float arrow_x_end2 = arrow_x - dx * arrow_size - perp_x;
            float arrow_y_end2 = arrow_y - dy * arrow_size - perp_y;

            // Draw the two lines forming the arrow head
            if (mode == 0) {
                DrawLine3D((Vector3){arrow_x, arrow_y, 0.0}, (Vector3){arrow_x_end1, arrow_y_end1, 0.0}, PUFF_WHITE);
                DrawLine3D((Vector3){arrow_x, arrow_y, 0.0}, (Vector3){arrow_x_end2, arrow_y_end2, 0.0}, PUFF_WHITE);
            }

            if (mode == 1) {
                float arrow_x_end1_world = px + (arrow_x_end1 * heading_self_x - arrow_y_end1 * heading_self_y);
                float arrow_y_end1_world = py + (arrow_x_end1 * heading_self_y + arrow_y_end1 * heading_self_x);
                float arrow_x_end2_world = px + (arrow_x_end2 * heading_self_x - arrow_y_end2 * heading_self_y);
                float arrow_y_end2_world = py + (arrow_x_end2 * heading_self_y + arrow_y_end2 * heading_self_x);
                DrawLine3D((Vector3){arrow_x_world, arrow_y_world, 0.0},
                           (Vector3){arrow_x_end1_world, arrow_y_end1_world, 0.0}, PUFF_WHITE);
                DrawLine3D((Vector3){arrow_x_world, arrow_y_world, 0.0},
                           (Vector3){arrow_x_end2_world, arrow_y_end2_world, 0.0}, PUFF_WHITE);
            }
        }

        obs_idx += PARTNER_FEATURES; // Move to next agent observation (7 values per agent)
    }
    // Then draw map observations
    int map_start_idx = ego_dim + PARTNER_FEATURES * (MAX_AGENTS - 1) + DRIVE_CORNER_OBS;
    for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {          // Loop through potential map entities
        int entity_idx = map_start_idx + k * 7;
        if (agent_obs[entity_idx] == 0 && agent_obs[entity_idx + 1] == 0) {
            continue;
        }
        Color lineColor = BLUE; // Default color
        int entity_type = (int)agent_obs[entity_idx + 6];
        // Choose color based on entity type
        if (entity_type + 4 != ROAD_EDGE) {
            continue;
        }
        lineColor = PUFF_CYAN;
        // For road segments, draw line between start and end points
        float x_middle = agent_obs[entity_idx] * 50;
        float y_middle = agent_obs[entity_idx + 1] * 50;
        float rel_angle_x = (agent_obs[entity_idx + 4]);
        float rel_angle_y = (agent_obs[entity_idx + 5]);
        float rel_angle = atan2f(rel_angle_y, rel_angle_x);
        float segment_length = agent_obs[entity_idx + 2] * MAX_ROAD_SEGMENT_LENGTH;
        // Calculate endpoint using the relative angle directly
        // Calculate endpoint directly
        float x_start = x_middle - segment_length * cosf(rel_angle);
        float y_start = y_middle - segment_length * sinf(rel_angle);
        float x_end = x_middle + segment_length * cosf(rel_angle);
        float y_end = y_middle + segment_length * sinf(rel_angle);

        if (lasers && mode == 0) {
            DrawLine3D((Vector3){0, 0, 0}, (Vector3){x_middle, y_middle, 1}, lineColor);
        }

        if (mode == 1) {
            float x_middle_world = px + (x_middle * heading_self_x - y_middle * heading_self_y);
            float y_middle_world = py + (x_middle * heading_self_y + y_middle * heading_self_x);
            float x_start_world = px + (x_start * heading_self_x - y_start * heading_self_y);
            float y_start_world = py + (x_start * heading_self_y + y_start * heading_self_x);
            float x_end_world = px + (x_end * heading_self_x - y_end * heading_self_y);
            float y_end_world = py + (x_end * heading_self_y + y_end * heading_self_x);
            DrawCube((Vector3){x_middle_world, y_middle_world, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start_world, y_start_world, 1}, (Vector3){x_end_world, y_end_world, 1}, BLUE);
            if (lasers)
                DrawLine3D((Vector3){px, py, 1}, (Vector3){x_middle_world, y_middle_world, 1}, lineColor);
        }
        if (mode == 0) {
            DrawCube((Vector3){x_middle, y_middle, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start, y_start, 1}, (Vector3){x_end, y_end, 1}, BLUE);
        }
    }
}

void draw_road_edge(Drive *env, float start_x, float start_y, float end_x, float end_y) {
    Color CURB_TOP = (Color){220, 220, 220, 255};  // Top surface - lightest
    Color CURB_SIDE = (Color){180, 180, 180, 255}; // Side faces - medium
    Color CURB_BOTTOM = (Color){160, 160, 160, 255};
    // Calculate curb dimensions
    float curb_height = 0.5f; // Height of the curb
    float curb_width = 0.3f;  // Width/thickness of the curb
    float road_z = 0.0f;      // Ensure z-level for roads is below agents

    // Calculate direction vector between start and end
    Vector3 direction = {end_x - start_x, end_y - start_y, 0.0f};

    // Calculate length of the segment
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y);

    // Normalize direction vector
    Vector3 normalized_dir = {direction.x / length, direction.y / length, 0.0f};

    // Calculate perpendicular vector for width
    Vector3 perpendicular = {-normalized_dir.y, normalized_dir.x, 0.0f};

    // Calculate the four bottom corners of the curb
    Vector3 b1 = {start_x - perpendicular.x * curb_width / 2, start_y - perpendicular.y * curb_width / 2, road_z};
    Vector3 b2 = {start_x + perpendicular.x * curb_width / 2, start_y + perpendicular.y * curb_width / 2, road_z};
    Vector3 b3 = {end_x + perpendicular.x * curb_width / 2, end_y + perpendicular.y * curb_width / 2, road_z};
    Vector3 b4 = {end_x - perpendicular.x * curb_width / 2, end_y - perpendicular.y * curb_width / 2, road_z};

    // Draw the curb faces
    // Bottom face
    DrawTriangle3D(b1, b2, b3, CURB_BOTTOM);
    DrawTriangle3D(b1, b3, b4, CURB_BOTTOM);

    // Top face (raised by curb_height)
    Vector3 t1 = {b1.x, b1.y, b1.z + curb_height};
    Vector3 t2 = {b2.x, b2.y, b2.z + curb_height};
    Vector3 t3 = {b3.x, b3.y, b3.z + curb_height};
    Vector3 t4 = {b4.x, b4.y, b4.z + curb_height};
    DrawTriangle3D(t1, t3, t2, CURB_TOP);
    DrawTriangle3D(t1, t4, t3, CURB_TOP);

    // Side faces
    DrawTriangle3D(b1, t1, b2, CURB_SIDE);
    DrawTriangle3D(t1, t2, b2, CURB_SIDE);
    DrawTriangle3D(b2, t2, b3, CURB_SIDE);
    DrawTriangle3D(t2, t3, b3, CURB_SIDE);
    DrawTriangle3D(b3, t3, b4, CURB_SIDE);
    DrawTriangle3D(t3, t4, b4, CURB_SIDE);
    DrawTriangle3D(b4, t4, b1, CURB_SIDE);
    DrawTriangle3D(t4, t1, b1, CURB_SIDE);
}

void build_road_cache(Drive *env, Client *client) {
    int tri_count = 0;
    int line_count = 0;
    for (int i = 0; i < env->num_entities; i++) {
        Entity *road = &env->entities[i];
        if (road->type < ROAD_LANE || road->type > ROAD_EDGE)
            continue;
        int segs = road->array_size - 1;
        if (segs <= 0)
            continue;
        if (road->type == ROAD_EDGE)
            tri_count += segs * 12;
        else if (road->type == ROAD_LANE || road->type == ROAD_LINE)
            line_count += segs;
    }

    int num_tri_verts = tri_count * 3;
    float *tri_verts = num_tri_verts > 0 ? (float *)RL_CALLOC(num_tri_verts * 3, sizeof(float)) : NULL;
    unsigned char *tri_colors =
        num_tri_verts > 0 ? (unsigned char *)RL_CALLOC(num_tri_verts * 4, sizeof(unsigned char)) : NULL;

    free(client->road_line_verts);
    free(client->road_line_colors);
    client->road_line_verts = line_count > 0 ? (float *)malloc(line_count * 2 * 3 * sizeof(float)) : NULL;
    client->road_line_colors = line_count > 0 ? (unsigned char *)malloc(line_count * 2 * 4) : NULL;
    client->road_line_count = 0;
    int actual_tri_count = 0;

#define PUSH_TRI(vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, cr, cg, cb, ca)                                          \
    do {                                                                                                               \
        int _ti = actual_tri_count * 9;                                                                                \
        int _ci = actual_tri_count * 12;                                                                               \
        tri_verts[_ti + 0] = vx1;                                                                                      \
        tri_verts[_ti + 1] = vy1;                                                                                      \
        tri_verts[_ti + 2] = vz1;                                                                                      \
        tri_verts[_ti + 3] = vx2;                                                                                      \
        tri_verts[_ti + 4] = vy2;                                                                                      \
        tri_verts[_ti + 5] = vz2;                                                                                      \
        tri_verts[_ti + 6] = vx3;                                                                                      \
        tri_verts[_ti + 7] = vy3;                                                                                      \
        tri_verts[_ti + 8] = vz3;                                                                                      \
        for (int _v = 0; _v < 3; _v++) {                                                                               \
            tri_colors[_ci + _v * 4 + 0] = cr;                                                                         \
            tri_colors[_ci + _v * 4 + 1] = cg;                                                                         \
            tri_colors[_ci + _v * 4 + 2] = cb;                                                                         \
            tri_colors[_ci + _v * 4 + 3] = ca;                                                                         \
        }                                                                                                              \
        actual_tri_count++;                                                                                            \
    } while (0)

#define PUSH_LINE(vx1, vy1, vz1, vx2, vy2, vz2, cr, cg, cb, ca)                                                        \
    do {                                                                                                               \
        int _li = client->road_line_count * 6;                                                                         \
        int _ci = client->road_line_count * 8;                                                                         \
        client->road_line_verts[_li + 0] = vx1;                                                                        \
        client->road_line_verts[_li + 1] = vy1;                                                                        \
        client->road_line_verts[_li + 2] = vz1;                                                                        \
        client->road_line_verts[_li + 3] = vx2;                                                                        \
        client->road_line_verts[_li + 4] = vy2;                                                                        \
        client->road_line_verts[_li + 5] = vz2;                                                                        \
        for (int _v = 0; _v < 2; _v++) {                                                                               \
            client->road_line_colors[_ci + _v * 4 + 0] = cr;                                                           \
            client->road_line_colors[_ci + _v * 4 + 1] = cg;                                                           \
            client->road_line_colors[_ci + _v * 4 + 2] = cb;                                                           \
            client->road_line_colors[_ci + _v * 4 + 3] = ca;                                                           \
        }                                                                                                              \
        client->road_line_count++;                                                                                     \
    } while (0)

    for (int i = 0; i < env->num_entities; i++) {
        Entity *road = &env->entities[i];
        if (road->type < ROAD_LANE || road->type > ROAD_EDGE)
            continue;
        for (int j = 0; j < road->array_size - 1; j++) {
            float sx = road->traj_x[j];
            float sy = road->traj_y[j];
            float ex = road->traj_x[j + 1];
            float ey = road->traj_y[j + 1];
            if (sx == INVALID_POSITION || sy == INVALID_POSITION || ex == INVALID_POSITION || ey == INVALID_POSITION)
                continue;
            float sz = Z_ROAD_MARKINGS;
            float ez = Z_ROAD_MARKINGS;

            if (road->type == ROAD_EDGE) {
                sz = Z_ROAD_SURFACE;
                ez = Z_ROAD_SURFACE;
                float curb_height = 0.5f;
                float curb_width = 0.3f;
                float dx = ex - sx;
                float dy = ey - sy;
                float len = sqrtf(dx * dx + dy * dy);
                if (len < 1e-6f)
                    continue;
                float nx = -dy / len;
                float ny = dx / len;
                float hw = curb_width * 0.5f;

                float b1x = sx - nx * hw, b1y = sy - ny * hw, b2x = sx + nx * hw, b2y = sy + ny * hw;
                float b3x = ex + nx * hw, b3y = ey + ny * hw, b4x = ex - nx * hw, b4y = ey - ny * hw;
                float t1z = sz + curb_height, t2z = sz + curb_height;
                float t3z = ez + curb_height, t4z = ez + curb_height;

                PUSH_TRI(b1x, b1y, sz, b2x, b2y, sz, b3x, b3y, ez, 160, 160, 160, 255);
                PUSH_TRI(b1x, b1y, sz, b3x, b3y, ez, b4x, b4y, ez, 160, 160, 160, 255);
                PUSH_TRI(b1x, b1y, t1z, b3x, b3y, t3z, b2x, b2y, t2z, 220, 220, 220, 255);
                PUSH_TRI(b1x, b1y, t1z, b4x, b4y, t4z, b3x, b3y, t3z, 220, 220, 220, 255);
                PUSH_TRI(b1x, b1y, sz, b1x, b1y, t1z, b2x, b2y, sz, 180, 180, 180, 255);
                PUSH_TRI(b1x, b1y, t1z, b2x, b2y, t2z, b2x, b2y, sz, 180, 180, 180, 255);
                PUSH_TRI(b2x, b2y, sz, b2x, b2y, t2z, b3x, b3y, ez, 180, 180, 180, 255);
                PUSH_TRI(b2x, b2y, t2z, b3x, b3y, t3z, b3x, b3y, ez, 180, 180, 180, 255);
                PUSH_TRI(b3x, b3y, ez, b3x, b3y, t3z, b4x, b4y, ez, 180, 180, 180, 255);
                PUSH_TRI(b3x, b3y, t3z, b4x, b4y, t4z, b4x, b4y, ez, 180, 180, 180, 255);
                PUSH_TRI(b4x, b4y, ez, b4x, b4y, t4z, b1x, b1y, sz, 180, 180, 180, 255);
                PUSH_TRI(b4x, b4y, t4z, b1x, b1y, t1z, b1x, b1y, sz, 180, 180, 180, 255);
            } else if (road->type == ROAD_LANE && client->road_line_verts) {
                Color c = Fade(SOFT_YELLOW, 0.25f);
                PUSH_LINE(sx, sy, sz, ex, ey, ez, c.r, c.g, c.b, c.a);
            } else if (road->type == ROAD_LINE && client->road_line_verts) {
                PUSH_LINE(sx, sy, sz, ex, ey, ez, 255, 255, 255, 255);
            }
        }
    }

#undef PUSH_TRI
#undef PUSH_LINE

    if (actual_tri_count > 0) {
        Mesh mesh = {0};
        mesh.vertexCount = actual_tri_count * 3;
        mesh.triangleCount = actual_tri_count;
        mesh.vertices = tri_verts;
        mesh.colors = tri_colors;
        UploadMesh(&mesh, false);
        client->road_tri_mesh = mesh;
        client->road_material = LoadMaterialDefault();
    } else {
        RL_FREE(tri_verts);
        RL_FREE(tri_colors);
    }

    client->road_cache_valid = 1;
    fprintf(stderr, "[drive] Road cache: %d triangles, %d lines\n", actual_tri_count, client->road_line_count);
}

void draw_road_cached(Client *client) {
    if (client->road_tri_mesh.vertexCount > 0)
        DrawMesh(client->road_tri_mesh, client->road_material, MatrixIdentity());

    if (client->road_line_count > 0) {
        rlSetLineWidth(2.0f);
        rlBegin(RL_LINES);
        int nv = client->road_line_count * 2;
        for (int i = 0; i < nv; i++) {
            rlColor4ub(client->road_line_colors[i * 4], client->road_line_colors[i * 4 + 1],
                       client->road_line_colors[i * 4 + 2], client->road_line_colors[i * 4 + 3]);
            rlVertex3f(client->road_line_verts[i * 3], client->road_line_verts[i * 3 + 1],
                       client->road_line_verts[i * 3 + 2]);
        }
        rlEnd();
    }
}

void draw_inferred_trajectory(Drive *env, int agent_array_idx) {
    int idx = env->active_agent_indices[agent_array_idx];
    Entity *e = &env->entities[idx];
    if (!e->inferred_traj_x || !e->traj_valid)
        return;

    float zg = Z_ROAD_MARKINGS, zs = Z_ROAD_MARKINGS + 0.01f;

    for (int t = 0; t < e->array_size - 1; t++) {
        if (!e->traj_valid[t])
            continue;

        // Ground truth (green)
        DrawSphere((Vector3){e->traj_x[t], e->traj_y[t], zg}, 0.15f, LIGHTGREEN);
        if (e->traj_valid[t + 1])
            DrawLine3D((Vector3){e->traj_x[t], e->traj_y[t], zg}, (Vector3){e->traj_x[t + 1], e->traj_y[t + 1], zg},
                       Fade(LIGHTGREEN, 0.6f));

        // Inferred (purple)
        DrawSphere((Vector3){e->inferred_traj_x[t], e->inferred_traj_y[t], zs}, 0.2f, LIGHT_PURPLE);
        DrawLine3D((Vector3){e->inferred_traj_x[t], e->inferred_traj_y[t], zs},
                   (Vector3){e->inferred_traj_x[t + 1], e->inferred_traj_y[t + 1], zs}, Fade(LIGHT_PURPLE, 0.6f));

        // Error shading (red)
        if (e->traj_valid[t + 1]) {
            Vector3 a = {e->traj_x[t], e->traj_y[t], zg};
            Vector3 b = {e->inferred_traj_x[t], e->inferred_traj_y[t], zg};
            Vector3 c = {e->traj_x[t + 1], e->traj_y[t + 1], zg};
            Vector3 d = {e->inferred_traj_x[t + 1], e->inferred_traj_y[t + 1], zg};
            Color err = Fade(PUFF_RED, 0.15f);
            DrawTriangle3D(a, b, c, err);
            DrawTriangle3D(c, b, d, err);
            DrawTriangle3D(c, b, a, err);
            DrawTriangle3D(d, b, c, err);
        }
    }
}

void draw_scene(Drive *env, Client *client, int mode, int obs_only, int lasers, int show_grid) {

    int has_inferred = 0;
    if (show_grid) {
        float grid_start_x = env->grid_map->top_left_x;
        float grid_start_y = env->grid_map->bottom_right_y;
        for (int i = 0; i < env->grid_map->grid_cols; i++) {
            for (int j = 0; j < env->grid_map->grid_rows; j++) {
                float x = grid_start_x + i * GRID_CELL_SIZE;
                float y = grid_start_y + j * GRID_CELL_SIZE;
                DrawCubeWires((Vector3){x + GRID_CELL_SIZE / 2, y + GRID_CELL_SIZE / 2, 0.0f}, GRID_CELL_SIZE,
                              GRID_CELL_SIZE, 0.1f, Fade(PUFF_BACKGROUND2, 0.3f));
            }
        }
    }

    // Draw expert trajectories for verification modes
    if (env->control_mode == CONTROL_REPLAY_LOGS || env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS) {
        Color traj_color = (env->control_mode == CONTROL_REPLAY_LOGS) ? EXPERT_REPLAY : LIGHT_PURPLE;
        for (int i = 0; i < env->active_agent_count; i++) {
            int idx = env->active_agent_indices[i];
            Entity *e = &env->entities[idx];
            for (int t = env->init_steps; t < e->array_size - 1; t++) {
                if (e->traj_valid[t] && e->traj_valid[t + 1]) {
                    DrawLine3D((Vector3){e->traj_x[t], e->traj_y[t], Z_ROAD_MARKINGS},
                               (Vector3){e->traj_x[t + 1], e->traj_y[t + 1], Z_ROAD_MARKINGS}, Fade(traj_color, 0.4f));
                }
                if (e->traj_valid[t]) {
                    DrawSphere((Vector3){e->traj_x[t], e->traj_y[t], Z_ROAD_MARKINGS}, 0.15f, traj_color);
                }
            }
        }

        if (env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS) {
            has_inferred = 1;
            for (int i = 0; i < env->active_agent_count; i++) {
                draw_inferred_trajectory(env, i);
            }
        }
    }

    if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0 && client->road_cache_valid) {
        draw_road_cached(client);
    }

    // Draw entities
    for (int i = 0; i < env->num_entities; i++) {
        // Draw objects
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN ||
            env->entities[i].type == CYCLIST) {

            bool is_active_agent = false;
            bool is_static_agent = false;
            int agent_index = -1;
            for (int j = 0; j < env->active_agent_count; j++) {
                if (env->active_agent_indices[j] == i) {
                    agent_index = j;
                    if (!env->entities[agent_index].removed) {
                        is_active_agent = true;
                        break;
                    }
                }
            }

            for (int j = 0; j < env->expert_static_agent_count; j++) {
                if (env->expert_static_agent_indices[j] == i) {
                    is_static_agent = true;
                    break;
                }
            }

            for (int j = 0; j < env->static_agent_count; j++) {
                if (env->static_agent_indices[j] == i) {
                    is_static_agent = true;
                    break;
                }
            }

            if (!is_active_agent && !is_static_agent) {
                continue;
            }

            if (env->entities[i].removed) {
                continue;
            }

            Vector3 position;
            float heading;
            position = (Vector3){env->entities[i].x, env->entities[i].y, Z_AGENTS};
            heading = env->entities[i].heading;

            // Create size vector
            Vector3 size = {env->entities[i].length, env->entities[i].width, env->entities[i].height};

            bool is_expert = (!is_active_agent) && (env->entities[i].mark_as_expert == 1);

            // Save current transform
            if (mode == 1) {
                float cos_heading = env->entities[i].heading_x;
                float sin_heading = env->entities[i].heading_y;

                // Calculate half dimensions
                float half_len = env->entities[i].length * 0.5f;
                float half_width = env->entities[i].width * 0.5f;

                // Calculate the four corners of the collision box
                Vector3 corners[4] = {
                    (Vector3){position.x + (half_len * cos_heading - half_width * sin_heading),
                              position.y + (half_len * sin_heading + half_width * cos_heading), position.z},
                    (Vector3){position.x + (half_len * cos_heading + half_width * sin_heading),
                              position.y + (half_len * sin_heading - half_width * cos_heading), position.z},
                    (Vector3){position.x + (-half_len * cos_heading + half_width * sin_heading),
                              position.y + (-half_len * sin_heading - half_width * cos_heading), position.z},
                    (Vector3){position.x + (-half_len * cos_heading - half_width * sin_heading),
                              position.y + (-half_len * sin_heading + half_width * cos_heading), position.z},

                };

                if (agent_index == env->human_agent_idx &&
                    !env->entities[env->active_agent_indices[agent_index]].current_goal_reached) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }

                if ((obs_only || IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx) {
                    continue;
                }

                // Pick agent color (reference scheme)
                Color agent_color = GRAY;
                if (is_expert) {
                    if (env->entities[i].type == PEDESTRIAN || env->entities[i].type == CYCLIST)
                        agent_color = EXPERT_REPLAY_SMALL;
                    else
                        agent_color = EXPERT_REPLAY;
                }
                if (is_active_agent) {
                    if (env->control_mode == CONTROL_REPLAY_LOGS)
                        agent_color = EXPERT_REPLAY;
                    else if (env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS)
                        agent_color = LIGHT_PURPLE;
                    else if (env->entities[i].type == PEDESTRIAN)
                        agent_color = LIGHT_ORANGE;
                    else if (env->entities[i].type == CYCLIST)
                        agent_color = LIGHT_PURPLE;
                    else
                        agent_color = BLUE;
                }
                if (is_active_agent && env->entities[i].collision_state > 0)
                    agent_color = RED;
                if (is_active_agent && env->entities[i].offroad_state > 0)
                    agent_color = LIGHTYELLOW;

                // Filled cube + wireframe outline at the same color
                rlPushMatrix();
                rlTranslatef(position.x, position.y, position.z);
                rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f);
                DrawCube((Vector3){0.0f, 0.0f, 0.0f}, size.x, size.y, 1.0f, Fade(agent_color, 0.5f));
                DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, size.x, size.y, 1.0f, agent_color);
                rlPopMatrix();

                // Heading arrow scaled by speed
                float agent_speed =
                    sqrtf(env->entities[i].vx * env->entities[i].vx + env->entities[i].vy * env->entities[i].vy);
                float speed_frac = fminf(agent_speed / MAX_SPEED, 1.0f);
                float arrow_len = half_len * 1.5f + speed_frac * 15.0f;
                float shaft_w = 0.4f;
                float head_len = 1.0f;
                float head_w = 1.4f;
                float az = position.z + 0.51f;

                float shaft_end = arrow_len - head_len;
                Vector3 s_start_l = {position.x - sin_heading * shaft_w * 0.5f,
                                     position.y + cos_heading * shaft_w * 0.5f, az};
                Vector3 s_start_r = {position.x + sin_heading * shaft_w * 0.5f,
                                     position.y - cos_heading * shaft_w * 0.5f, az};
                Vector3 s_end_l = {position.x + cos_heading * shaft_end - sin_heading * shaft_w * 0.5f,
                                   position.y + sin_heading * shaft_end + cos_heading * shaft_w * 0.5f, az};
                Vector3 s_end_r = {position.x + cos_heading * shaft_end + sin_heading * shaft_w * 0.5f,
                                   position.y + sin_heading * shaft_end - cos_heading * shaft_w * 0.5f, az};
                DrawTriangle3D(s_start_l, s_start_r, s_end_r, agent_color);
                DrawTriangle3D(s_start_l, s_end_r, s_end_l, agent_color);
                DrawTriangle3D(s_end_r, s_start_r, s_start_l, agent_color);
                DrawTriangle3D(s_end_l, s_end_r, s_start_l, agent_color);

                Vector3 tip = {position.x + cos_heading * arrow_len, position.y + sin_heading * arrow_len, az};
                Vector3 head_l = {position.x + cos_heading * shaft_end - sin_heading * head_w * 0.5f,
                                  position.y + sin_heading * shaft_end + cos_heading * head_w * 0.5f, az};
                Vector3 head_r = {position.x + cos_heading * shaft_end + sin_heading * head_w * 0.5f,
                                  position.y + sin_heading * shaft_end - cos_heading * head_w * 0.5f, az};
                DrawTriangle3D(head_l, head_r, tip, agent_color);
                DrawTriangle3D(tip, head_r, head_l, agent_color);

            } else { // Agent view
                rlPushMatrix();
                // Translate to position, rotate around Y axis, then draw
                rlTranslatef(position.x, position.y, position.z);
                rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f); // Convert radians to degrees

                // Select car model (skip index 0)
                Model car_model = client->cars[(i % 5) + 1]; // Cycles through indices 1-5

                if (env->control_mode == CONTROL_REPLAY_LOGS || env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS) {
                    car_model = client->cars[1]; // White car for verification modes
                } else if (agent_index == env->human_agent_idx) {
                    car_model = client->cars[0];
                } else if (is_active_agent) {
                    car_model = client->cars[(i % 5) + 1];
                    if (env->entities[i].collision_state > 0) {
                        car_model = client->cars[0];
                    }
                }
                // Draw obs for selected agent index
                if (agent_index == env->human_agent_idx &&
                    (!env->entities[env->active_agent_indices[agent_index]].current_goal_reached ||
                     env->goal_behavior == GOAL_GENERATE_NEW || env->goal_behavior == GOAL_STOP)) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }

                // Draw cube for cars static and active
                // Calculate scale factors based on desired size and model dimensions
                BoundingBox bounds = GetModelBoundingBox(car_model);
                Vector3 model_size = {bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y,
                                      bounds.max.z - bounds.min.z};
                Vector3 scale = {size.x / model_size.x, size.y / model_size.y, size.z / model_size.z};

                if (env->entities[i].type == CYCLIST) {
                    scale = (Vector3){0.01, 0.01, 0.01};
                    car_model = client->cyclist;
                }
                if (env->entities[i].type == PEDESTRIAN) {
                    scale = (Vector3){2, 2, 2};
                    car_model = client->pedestrian;
                }
                DrawModelEx(car_model, (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, WHITE);
                {
                    float half_len = env->entities[i].length * 0.5f;
                    float half_width = env->entities[i].width * 0.5f;
                    Vector3 corners[4] = {
                        (Vector3){half_len, -half_width, 0},  // Front-left
                        (Vector3){half_len, half_width, 0},   // Front-right
                        (Vector3){-half_len, half_width, 0},  // Back-right
                        (Vector3){-half_len, -half_width, 0}, // Back-left
                    };
                    Color wire_color = GRAY;
                    if (!is_active_agent && env->entities[i].mark_as_expert == 1)
                        wire_color = EXPERT_REPLAY;
                    if (is_active_agent) {
                        if (env->control_mode == CONTROL_REPLAY_LOGS)
                            wire_color = EXPERT_REPLAY;
                        else if (env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS)
                            wire_color = LIGHT_PURPLE;
                        else
                            wire_color = BLUE; // Policy-controlled
                    }
                    if (is_active_agent && env->entities[i].collision_state > 0)
                        wire_color = RED;

                    if (is_active_agent && env->entities[i].collision_state > 0)
                        wire_color = LIGHTYELLOW;

                    rlSetLineWidth(2.0f);
                    for (int j = 0; j < 4; j++) {
                        DrawLine3D(corners[j], corners[(j + 1) % 4], wire_color);
                    }
                }
                rlPopMatrix();
            }

            // FPV Camera Control
            if (IsKeyDown(KEY_SPACE) && env->human_agent_idx == agent_index) {
                Vector3 camera_position = (Vector3){position.x - (25.0f * cosf(heading)),
                                                    position.y - (25.0f * sinf(heading)), position.z + 15};

                Vector3 camera_target = (Vector3){position.x + 40.0f * cosf(heading),
                                                  position.y + 40.0f * sinf(heading), position.z - 5.0f};
                client->camera.position = camera_position;
                client->camera.target = camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }
            if (IsKeyReleased(KEY_SPACE)) {
                client->camera.position = client->default_camera_position;
                client->camera.target = client->default_camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }
            // Draw goal position for active agents
            if (!is_active_agent || env->entities[i].valid == 0) {
                continue;
            }
            if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0 && env->control_mode != CONTROL_REPLAY_LOGS &&
                env->control_mode != CONTROL_INFERRED_EXPERT_ACTIONS) {
                // Match clean-eval: goal markers are bright lime, with a faint
                // radius circle on the road surface.
                Color goal_color = LIME;
                DrawSphere((Vector3){env->entities[i].goal_position_x, env->entities[i].goal_position_y, Z_AGENTS},
                           1.5f, goal_color);
                DrawCircle3D(
                    (Vector3){env->entities[i].goal_position_x, env->entities[i].goal_position_y, Z_ROAD_MARKINGS},
                    env->goal_radius, (Vector3){0, 0, 1}, 90.0f, Fade(goal_color, 0.3f));
            }
        }
        // Draw road elements
        if (env->entities[i].type <= 3 || env->entities[i].type >= 7) {
            continue;
        }
        if (client->road_cache_valid) {
            continue;
        }
        for (int j = 0; j < env->entities[i].array_size - 1; j++) {
            Vector3 start = {env->entities[i].traj_x[j], env->entities[i].traj_y[j], Z_ROAD_MARKINGS};
            Vector3 end = {env->entities[i].traj_x[j + 1], env->entities[i].traj_y[j + 1], Z_ROAD_MARKINGS};
            Color lineColor = GRAY;
            if (env->entities[i].type == ROAD_LANE)
                lineColor = Fade(SOFT_YELLOW, 0.25f);
            else if (env->entities[i].type == ROAD_LINE)
                lineColor = WHITE;
            else if (env->entities[i].type == ROAD_EDGE)
                lineColor = Fade(WHITE, 0.7f);
            else if (env->entities[i].type == DRIVEWAY)
                lineColor = RED;

            if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
                if (env->entities[i].type == ROAD_EDGE) {
                    draw_road_edge(env, start.x, start.y, end.x, end.y);
                } else if (env->entities[i].type == ROAD_LANE || env->entities[i].type == ROAD_LINE) {
                    // Draw road lanes and lines as purple lines
                    rlSetLineWidth(2.0f);
                    DrawLine3D(start, end, lineColor);
                }
            }
        }
    }

    EndMode3D();

    // Draw per-agent ADE labels projected to screen for interactive verify mode
    if (has_inferred) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int idx = env->active_agent_indices[i];
            Entity *e = &env->entities[idx];
            if (e->inferred_ade < 0)
                continue;
            int mid = e->array_size / 2;
            Vector2 sp = GetWorldToScreen((Vector3){e->traj_x[mid], e->traj_y[mid], Z_AGENT_DETAILS}, client->camera);
            int sx = (int)sp.x, sy = (int)sp.y;
            if (sx < 0 || sx > client->width || sy < 0 || sy > client->height)
                continue;
            char buf[32];
            // Check if unfit (same criteria as c_collect_expert_data)
            int valid_count = 0;
            int is_static = 1;
            if (e->traj_valid) {
                for (int t = 0; t < e->array_size; t++) {
                    if (!e->traj_valid[t])
                        continue;
                    valid_count++;
                    if (valid_count > 1 && (e->traj_x[t] != e->traj_x[0] || e->traj_y[t] != e->traj_y[0]))
                        is_static = 0;
                }
            }
            int unfit = (e->inferred_ade < 0.0f || e->inferred_ade > ADE_THRESHOLD || valid_count < 3 || is_static);
            Color label_color = unfit ? PUFF_RED : PUFF_WHITE;
            snprintf(buf, sizeof(buf), "ADE:%.2fm", e->inferred_ade);
            DrawText(buf, sx - MeasureText(buf, 16) / 2, sy, 16, label_color);
        }
    }

    if (mode == 1 && env->render_mode != RENDER_HEADLESS) {
        float cam_x = 0.0f, cam_y = 0.0f;
        float fovy = env->grid_map->top_left_y - env->grid_map->bottom_right_y;

        if (env->sdc_track_index >= 0 && env->control_mode == CONTROL_SDC_ONLY &&
            !env->entities[env->sdc_track_index].removed) {
            cam_x = env->entities[env->sdc_track_index].x;
            cam_y = env->entities[env->sdc_track_index].y;
            fovy = 150.0f;
        }
        float scale = client->height / fovy;

        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];

            if (env->entities[agent_idx].removed || env->entities[agent_idx].x == INVALID_POSITION) {
                continue;
            }

            int sx = (int)(-(env->entities[agent_idx].x - cam_x) * scale) + client->width / 2 + 20;
            int sy = (int)((env->entities[agent_idx].y - cam_y) * scale) + client->height / 2 - 25;
            if (sx < 0 || sx > client->width || sy < 0 || sy > client->height)
                continue;
            char text[32];
            snprintf(text, sizeof(text), agent_idx == env->sdc_track_index ? "sdc" : "%d", agent_idx);
            DrawText(text, sx - MeasureText(text, 20) / 2, sy, 20, PUFF_WHITE);
        }
    }
}

void draw_thick_line_3d(Vector3 a, Vector3 b, float thickness, Color color) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float len = sqrtf(dx * dx + dy * dy);
    if (len < 1e-6f)
        return;
    float nx = -dy / len * thickness * 0.5f;
    float ny = dx / len * thickness * 0.5f;

    Vector3 p0 = {a.x + nx, a.y + ny, a.z};
    Vector3 p1 = {a.x - nx, a.y - ny, a.z};
    Vector3 p2 = {b.x - nx, b.y - ny, b.z};
    Vector3 p3 = {b.x + nx, b.y + ny, b.z};

    DrawTriangle3D(p0, p1, p2, color);
    DrawTriangle3D(p0, p2, p3, color);
    DrawTriangle3D(p2, p1, p0, color);
    DrawTriangle3D(p3, p2, p0, color);
}

void draw_sensor_noise_view(Drive *env, Client *client) {
    int sdc_array_idx = 0;
    if (env->sdc_track_index >= 0) {
        for (int i = 0; i < env->active_agent_count; i++) {
            if (env->active_agent_indices[i] == env->sdc_track_index) {
                sdc_array_idx = i;
                break;
            }
        }
    }

    int ego_idx = env->active_agent_indices[sdc_array_idx];
    Entity *ego = &env->entities[ego_idx];
    if (ego->removed == 1)
        return;

    float cos_h = ego->heading_x;
    float sin_h = ego->heading_y;
    int num_samples = 5;

    for (int j = 0; j < MAX_AGENTS; j++) {
        int index = -1;
        if (j < env->active_agent_count)
            index = env->active_agent_indices[j];
        else if (j < env->num_actors && env->static_agent_count > 0)
            index = env->static_agent_indices[j - env->active_agent_count];
        if (index == -1 || index == ego_idx)
            continue;

        Entity *other = &env->entities[index];
        if (other->type > CYCLIST)
            continue;
        if (other->x == INVALID_POSITION)
            continue;
        if (other->removed == 1)
            continue;

        float dx = other->x - ego->x;
        float dy = other->y - ego->y;
        if ((dx * dx + dy * dy) > 2500.0f)
            continue;

        float gt_cos = other->heading_x;
        float gt_sin = other->heading_y;
        float gt_hl = other->length * 0.5f;
        float gt_hw = other->width * 0.5f;

        // Ground truth box (green)
        Vector3 gt[4] = {
            {other->x + gt_hl * gt_cos - gt_hw * gt_sin, other->y + gt_hl * gt_sin + gt_hw * gt_cos, Z_AGENTS},
            {other->x + gt_hl * gt_cos + gt_hw * gt_sin, other->y + gt_hl * gt_sin - gt_hw * gt_cos, Z_AGENTS},
            {other->x - gt_hl * gt_cos + gt_hw * gt_sin, other->y - gt_hl * gt_sin - gt_hw * gt_cos, Z_AGENTS},
            {other->x - gt_hl * gt_cos - gt_hw * gt_sin, other->y - gt_hl * gt_sin + gt_hw * gt_cos, Z_AGENTS},
        };
        for (int k = 0; k < 4; k++)
            draw_thick_line_3d(gt[k], gt[(k + 1) % 4], 0.45f, LIGHTGREEN);

        // Noisy perception samples (purple)
        for (int s = 0; s < num_samples; s++) {
            float rel_x = dx * cos_h + dy * sin_h;
            float rel_y = -dx * sin_h + dy * cos_h;
            float noisy_rx = rel_x + gaussian_noise(env->obs_partner_noise_pos) / 0.02f;
            float noisy_ry = rel_y + gaussian_noise(env->obs_partner_noise_pos) / 0.02f;
            float px = ego->x + noisy_rx * cos_h - noisy_ry * sin_h;
            float py = ego->y + noisy_rx * sin_h + noisy_ry * cos_h;

            float p_hw =
                (other->width / MAX_VEH_WIDTH + gaussian_noise(env->obs_partner_noise_pos)) * MAX_VEH_WIDTH * 0.5f;
            float p_hl =
                (other->length / MAX_VEH_LEN + gaussian_noise(env->obs_partner_noise_pos)) * MAX_VEH_LEN * 0.5f;

            Vector3 pc[4] = {
                {px + p_hl * gt_cos - p_hw * gt_sin, py + p_hl * gt_sin + p_hw * gt_cos, Z_AGENTS + 0.01f},
                {px + p_hl * gt_cos + p_hw * gt_sin, py + p_hl * gt_sin - p_hw * gt_cos, Z_AGENTS + 0.01f},
                {px - p_hl * gt_cos + p_hw * gt_sin, py - p_hl * gt_sin - p_hw * gt_cos, Z_AGENTS + 0.01f},
                {px - p_hl * gt_cos - p_hw * gt_sin, py - p_hl * gt_sin + p_hw * gt_cos, Z_AGENTS + 0.01f},
            };
            Color sc = Fade(LIGHT_PURPLE, 0.6f);
            for (int k = 0; k < 4; k++)
                draw_thick_line_3d(pc[k], pc[(k + 1) % 4], 0.4f, sc);
        }
    }

    // Ego box (cyan)
    float hl = ego->length * 0.5f;
    float hw = ego->width * 0.5f;
    Vector3 ec[4] = {
        {ego->x + hl * cos_h - hw * sin_h, ego->y + hl * sin_h + hw * cos_h, Z_AGENTS},
        {ego->x + hl * cos_h + hw * sin_h, ego->y + hl * sin_h - hw * cos_h, Z_AGENTS},
        {ego->x - hl * cos_h + hw * sin_h, ego->y - hl * sin_h - hw * cos_h, Z_AGENTS},
        {ego->x - hl * cos_h - hw * sin_h, ego->y - hl * sin_h + hw * cos_h, Z_AGENTS},
    };
    for (int k = 0; k < 4; k++)
        draw_thick_line_3d(ec[k], ec[(k + 1) % 4], 0.5f, DEEPBLUE);

    // Ego heading arrow
    Vector3 as = {ego->x, ego->y, Z_AGENTS};
    Vector3 ae = {ego->x + cos_h * hl * 2.0f, ego->y + sin_h * hl * 2.0f, Z_AGENTS};
    draw_thick_line_3d(as, ae, 0.25f, DEEPBLUE);
}

static void write_all_bytes(int fd, const unsigned char *data, size_t size) {
    size_t remaining = size;
    const unsigned char *p = data;
    while (remaining > 0) {
        ssize_t written = write(fd, p, remaining);
        if (written < 0) {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "[drive] write failed errno=%d(%s)\n", errno, strerror(errno));
            break;
        }
        p += written;
        remaining -= (size_t)written;
    }
}

#ifdef DRIVE_HAS_EGL
static void write_gl_frame_flipped(int fd, unsigned char *ptr, int width, int height) {
    int row_bytes = width * 4;
    int rows_remaining = height;
    int row_top = 0;
    int iov_max = 1024;

    while (rows_remaining > 0) {
        int chunk = rows_remaining < iov_max ? rows_remaining : iov_max;
        struct iovec iov[1024];
        size_t chunk_bytes = 0;
        for (int i = 0; i < chunk; i++) {
            int src_row = height - 1 - (row_top + i);
            iov[i].iov_base = ptr + (size_t)src_row * row_bytes;
            iov[i].iov_len = row_bytes;
            chunk_bytes += row_bytes;
        }

        struct iovec *cur = iov;
        int cur_cnt = chunk;
        size_t cur_remaining = chunk_bytes;
        while (cur_remaining > 0) {
            ssize_t written = writev(fd, cur, cur_cnt);
            if (written < 0) {
                if (errno == EINTR)
                    continue;
                fprintf(stderr, "[drive-pbo] writev failed errno=%d(%s)\n", errno, strerror(errno));
                return;
            }
            cur_remaining -= (size_t)written;
            size_t consumed = (size_t)written;
            while (cur_cnt > 0 && consumed >= cur[0].iov_len) {
                consumed -= cur[0].iov_len;
                cur++;
                cur_cnt--;
            }
            if (cur_cnt > 0 && consumed > 0) {
                cur[0].iov_base = (unsigned char *)cur[0].iov_base + consumed;
                cur[0].iov_len -= consumed;
            }
        }

        row_top += chunk;
        rows_remaining -= chunk;
    }
}
#endif

void c_render(Drive *env, int view_mode, int draw_traces) {

    // Create client on first render call
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    if (env->client == NULL) {
        return; // Silently skip rendering
    }

    Client *client = env->client;

    if (env->render_mode == RENDER_HEADLESS && !client->road_cache_valid) {
        build_road_cache(env, client);
    }

    if (env->render_mode == RENDER_HEADLESS && client->recorder_pid <= 0) {
        start_recorder(client, env->scenario_id, (int)client->width, (int)client->height);
    }

    if (env->render_mode == RENDER_HEADLESS) { // Headless rendering via ffmpeg
        Camera3D camera = {0};

        if (view_mode == VIEW_MODE_SIM_STATE || view_mode == VIEW_MODE_FOLLOW_AGENT) {
            int track = -1;
            if (view_mode == VIEW_MODE_FOLLOW_AGENT && env->active_agent_count > 0) {
                int current_ok = 0;
                if (client->follow_ready) {
                    for (int i = 0; i < env->active_agent_count; i++) {
                        int idx = env->active_agent_indices[i];
                        if (idx == client->follow_agent && !env->entities[idx].removed &&
                            env->entities[idx].x != INVALID_POSITION) {
                            current_ok = 1;
                            break;
                        }
                    }
                }
                if (!client->follow_ready || env->timestep <= env->init_steps || !current_ok) {
                    int choices[ACTOR_CAP];
                    int nchoices = 0;
                    for (int i = 0; i < env->active_agent_count && nchoices < ACTOR_CAP; i++) {
                        int idx = env->active_agent_indices[i];
                        if (env->entities[idx].removed || env->entities[idx].x == INVALID_POSITION)
                            continue;
                        choices[nchoices++] = idx;
                    }
                    if (nchoices > 0) {
                        client->follow_agent = choices[rand() % nchoices];
                        client->follow_ready = 1;
                    }
                }
                if (client->follow_ready)
                    track = client->follow_agent;
            } else if (env->sdc_track_index >= 0 && env->control_mode == CONTROL_SDC_ONLY) {
                track = env->sdc_track_index;
            }
            if (track >= 0) {
                // Zoomed orthographic shot, one agent for the whole episode.
                Entity *followed = &env->entities[track];
                camera.position = (Vector3){followed->x, followed->y, 400.0f};
                camera.target = (Vector3){followed->x, followed->y, 0.0f};
                camera.up = (Vector3){0.0f, -1.0f, 0.0f};
                camera.projection = CAMERA_ORTHOGRAPHIC;
                camera.fovy = 75.0f;
            } else {
                // Orthographic bird's-eye view over the entire map.
                float map_width = fabsf(env->grid_map->bottom_right_x - env->grid_map->top_left_x);
                float map_height = fabsf(env->grid_map->top_left_y - env->grid_map->bottom_right_y);
                float map_center_x = (env->grid_map->top_left_x + env->grid_map->bottom_right_x) * 0.5f;
                float map_center_y = (env->grid_map->top_left_y + env->grid_map->bottom_right_y) * 0.5f;
                float aspect = client->width / client->height;
                float fovy = fmaxf(map_height, map_width / aspect) * 1.05f;
                camera.position = (Vector3){map_center_x, map_center_y, 400.0f};
                camera.target = (Vector3){map_center_x, map_center_y, 0.0f};
                camera.up = (Vector3){0.0f, -1.0f, 0.0f};
                camera.projection = CAMERA_ORTHOGRAPHIC;
                camera.fovy = fovy;
            }
            BeginDrawing();
            ClearBackground(ROAD_COLOR);
            BeginMode3D(camera);

            if (draw_traces) { // Show logged trajectories of active agents and expert static agents
                for (int i = 0; i < env->active_agent_count; i++) {
                    int idx = env->active_agent_indices[i];
                    int t_end = env->episode_length;
                    if (t_end > env->entities[idx].array_size)
                        t_end = env->entities[idx].array_size;
                    for (int t = env->init_steps; t < t_end; t++) {
                        if (env->entities[idx].traj_valid && !env->entities[idx].traj_valid[t])
                            continue;
                        Color agent_color = LIGHTBLUE;
                        if (env->entities[idx].type == PEDESTRIAN) {
                            agent_color = LIGHT_ORANGE;
                        } else if (env->entities[idx].type == CYCLIST) {
                            agent_color = LIGHT_PURPLE;
                        }
                        DrawPoint3D(
                            (Vector3){env->entities[idx].traj_x[t], env->entities[idx].traj_y[t], Z_AGENT_DETAILS},
                            agent_color);
                    }
                }

                for (int i = 0; i < env->expert_static_agent_count; i++) {
                    int idx = env->expert_static_agent_indices[i];
                    int t_end = env->episode_length;
                    if (t_end > env->entities[idx].array_size)
                        t_end = env->entities[idx].array_size;
                    for (int t = env->init_steps; t < t_end; t++) {
                        if (env->entities[idx].traj_valid && !env->entities[idx].traj_valid[t])
                            continue;
                        DrawPoint3D(
                            (Vector3){env->entities[idx].traj_x[t], env->entities[idx].traj_y[t], Z_AGENT_DETAILS},
                            EXPERT_REPLAY);
                    }
                }
            }

            draw_scene(env, client, 1, 0, 0, 0);

            // Draw timestep
            DrawText(TextFormat("t=%d", env->timestep), 10, 10, 26, PUFF_WHITE);

        } else if (view_mode == VIEW_MODE_BEV_AGENT_OBS) {
            // Orthographic bird's-eye view centered on the selected agent,
            // showing only that agent's observations
            int agent_idx = env->active_agent_indices[env->human_agent_idx];
            Entity *agent = &env->entities[agent_idx];

            Camera3D camera = {0};
            camera.position = (Vector3){agent->x, agent->y, 400.0f};
            camera.target = (Vector3){agent->x, agent->y, 0.0f};
            camera.up = (Vector3){0.0f, -1.0f, 0.0f};
            camera.projection = CAMERA_ORTHOGRAPHIC;
            camera.fovy = env->grid_map->vision_range * GRID_CELL_SIZE * 2.0f;

            BeginDrawing();
            ClearBackground(ROAD_COLOR);
            BeginMode3D(camera);
            draw_scene(env, client, 1, 1, 0, 0);

        } else if (view_mode == VIEW_MODE_SENSOR_NOISE) {
            // Sensor noise visualization: BEV centered on SDC showing
            // ground truth (green) vs noisy perception (purple)
            int sdc_array_idx = 0;
            if (env->sdc_track_index >= 0) {
                for (int i = 0; i < env->active_agent_count; i++) {
                    if (env->active_agent_indices[i] == env->sdc_track_index) {
                        sdc_array_idx = i;
                        break;
                    }
                }
            }
            int ego_idx = env->active_agent_indices[sdc_array_idx];
            Entity *sdc = &env->entities[ego_idx];

            camera.position = (Vector3){sdc->x, sdc->y, 400.0f};
            camera.target = (Vector3){sdc->x, sdc->y, 0.0f};
            camera.up = (Vector3){0.0f, -1.0f, 0.0f};
            camera.projection = CAMERA_ORTHOGRAPHIC;
            camera.fovy = 100.0f;

            BeginDrawing();
            ClearBackground(ROAD_COLOR);
            BeginMode3D(camera);

            // Draw roads first
            for (int i = 0; i < env->num_entities; i++) {
                if (env->entities[i].type < ROAD_LANE || env->entities[i].type > ROAD_EDGE)
                    continue;
                for (int j = 0; j < env->entities[i].array_size - 1; j++) {
                    Vector3 start = {env->entities[i].traj_x[j], env->entities[i].traj_y[j], Z_ROAD_MARKINGS};
                    Vector3 end = {env->entities[i].traj_x[j + 1], env->entities[i].traj_y[j + 1], Z_ROAD_MARKINGS};
                    if (env->entities[i].type == ROAD_EDGE)
                        draw_road_edge(env, start.x, start.y, end.x, end.y);
                    else {
                        Color c = (env->entities[i].type == ROAD_LANE) ? Fade(SOFT_YELLOW, 0.25f) : WHITE;
                        rlSetLineWidth(2.0f);
                        DrawLine3D(start, end, c);
                    }
                }
            }

            draw_sensor_noise_view(env, client);
            EndMode3D();

        } else { // First-person perspective from a selected agent
            int agent_idx = env->active_agent_indices[env->human_agent_idx];
            Entity *agent = &env->entities[agent_idx];

            Camera3D camera = {0};
            // Position camera behind and above the agent
            camera.position =
                (Vector3){agent->x - (25.0f * cosf(agent->heading)), agent->y - (25.0f * sinf(agent->heading)), 15.0f};
            camera.target =
                (Vector3){agent->x + 40.0f * cosf(agent->heading), agent->y + 40.0f * sinf(agent->heading), 1.0f};
            camera.up = (Vector3){0.0f, 0.0f, 1.0f};
            camera.fovy = 60.0f;
            camera.projection = CAMERA_PERSPECTIVE;

            BeginDrawing();
            ClearBackground(ROAD_COLOR);
            BeginMode3D(camera);
            draw_scene(env, client, 0, 0, 0, 1);
        }

#ifdef DRIVE_HAS_EGL
        if (client->egl_mode) {
            rlDrawRenderBatchActive();
        } else
#endif
        {
            EndDrawing();
        }

        int w = (int)client->width;
        int h = (int)client->height;
        int frame_bytes = w * h * 4;

#ifdef DRIVE_HAS_EGL
        if (client->egl_mode) {
            if (client->pbo[0] == 0) {
                glGenBuffers(2, client->pbo);
                for (int i = 0; i < 2; i++) {
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, client->pbo[i]);
                    glBufferData(GL_PIXEL_PACK_BUFFER, frame_bytes, NULL, GL_STREAM_READ);
                }
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            }

            int curr = client->pbo_index;
            int prev = 1 - curr;

            glBindBuffer(GL_PIXEL_PACK_BUFFER, client->pbo[curr]);
            glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, 0);
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

            if (client->pbo_frame_count > 0) {
                glBindBuffer(GL_PIXEL_PACK_BUFFER, client->pbo[prev]);
                unsigned char *ptr = (unsigned char *)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
                if (ptr) {
                    write_gl_frame_flipped(client->recorder_pipefd[1], ptr, w, h);
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                } else {
                    fprintf(stderr, "[drive-pbo] frame=%d glMapBuffer returned NULL, GL error=0x%x\n",
                            client->pbo_frame_count, glGetError());
                }
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            }

            client->pbo_index = prev;
            client->pbo_frame_count++;
        } else
#endif
        {
            unsigned char *screen_data = rlReadScreenPixels(w, h);
            if (screen_data) {
                write_all_bytes(client->recorder_pipefd[1], screen_data, (size_t)frame_bytes);
                RL_FREE(screen_data);
            }
        }
    } else { // Pop-up window
        BeginDrawing();
        ClearBackground(ROAD_COLOR);
        BeginMode3D(client->camera);
        handle_camera_controls(env->client);
        draw_scene(env, client, 0, 0, 0, 0);

        if (IsKeyPressed(KEY_TAB) && env->active_agent_count > 0) {
            env->human_agent_idx = (env->human_agent_idx + 1) % env->active_agent_count;
        }

        DrawText(TextFormat("Timestep: %d", env->timestep), 10, 65, 20, PUFF_WHITE);
        DrawText(TextFormat("Controlling agent: %d \n", env->human_agent_idx), 10, 85, 20, PUFF_WHITE);
        if (env->control_mode == CONTROL_REPLAY_LOGS) {
            DrawText("Mode: Replay logs", 10, 20, 20, EXPERT_REPLAY);
        } else if (env->control_mode == CONTROL_INFERRED_EXPERT_ACTIONS) {
            DrawText("Mode: Inferred expert actions", 10, 20, 20, LIGHT_PURPLE);
            if (env->dynamics_model == DELTA_LOCAL) {
                DrawText("Delta-local dynamics model", 10, 45, 20, LIGHT_PURPLE);
            } else if (env->dynamics_model == CLASSIC) {
                DrawText("Classic dynamics model", 10, 45, 20, LIGHT_PURPLE);
            }
            // DrawText(env->action_type == 1 ? "Actions: continuous" : "Actions: discrete", 10, 65, 20, LIGHT_PURPLE);
        }

        int human_idx = env->active_agent_indices[env->human_agent_idx];

        Color action_color = IsKeyDown(KEY_LEFT_SHIFT) ? YELLOW : PUFF_WHITE;

        if (env->action_type == 0) { // discrete
            int *action_array = (int *)env->actions;

            bool can_take_control =
                !(env->control_mode != CONTROL_REPLAY_LOGS || env->control_mode != CONTROL_INFERRED_EXPERT_ACTIONS);

            if (env->dynamics_model == CLASSIC && can_take_control) {
                int accel_idx = action_array[env->human_agent_idx * 2 + 0];
                int steer_idx = action_array[env->human_agent_idx * 2 + 1];
                float accel_value = ACCELERATION_VALUES[accel_idx];
                float steer_value = STEERING_VALUES[steer_idx];

                DrawText(TextFormat("Acceleration: %.2f m/s^2", accel_value), 10, 110, 20, action_color);
                DrawText(TextFormat("Steering: %.3f", steer_value), 10, 130, 20, action_color);
            } else if (env->dynamics_model == JERK) {
                int action_val = action_array[env->human_agent_idx];
                int num_lat = 3;
                int jerk_long_idx = action_val / num_lat;
                int jerk_lat_idx = action_val % num_lat;
                float jerk_long_value = JERK_LONG[jerk_long_idx];
                float jerk_lat_value = JERK_LAT[jerk_lat_idx];

                DrawText(TextFormat("Longitudinal Jerk: %.2f m/s^3", jerk_long_value), 10, 110, 20, action_color);
                DrawText(TextFormat("Lateral Jerk: %.2f m/s^3", jerk_lat_value), 10, 130, 20, action_color);
            }
        } else { // continuous
            float (*action_array_f)[2] = (float (*)[2])env->actions;
            DrawText(TextFormat("Acceleration: %.2f", action_array_f[env->human_agent_idx][0]), 10, 110, 20,
                     action_color);
            DrawText(TextFormat("Steering: %.2f", action_array_f[env->human_agent_idx][1]), 10, 130, 20, action_color);
        }

        int status_y = 150;
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            DrawText("[shift pressed]", 10, status_y, 20, PUFF_WHITE);
            status_y += 20;
        }
        if (IsKeyDown(KEY_SPACE)) {
            DrawText("[space pressed]", 10, status_y, 20, PUFF_WHITE);
            status_y += 20;
        }
        if (IsKeyDown(KEY_LEFT_CONTROL)) {
            DrawText("[ctrl pressed]", 10, status_y, 20, PUFF_WHITE);
            status_y += 20;
        }

        DrawText("Controls: SHIFT + W/S - Accelerate/Brake, SHIFT + A/D - Steer, TAB - Switch Agent", 10,
                 client->height - 30, 20, PUFF_WHITE);
        DrawText(TextFormat("Grid Rows: %d", env->grid_map->grid_rows), 10, status_y, 20, PUFF_WHITE);
        DrawText(TextFormat("Grid Cols: %d", env->grid_map->grid_cols), 10, status_y + 20, 20, PUFF_WHITE);
        EndDrawing();
    }
}

void close_client(Client *client) {
#ifdef DRIVE_HAS_EGL
    if (client->egl_mode && client->pbo_frame_count > 0 && client->pbo[0] != 0) {
        int prev = 1 - client->pbo_index;
        int w = (int)client->width;
        int h = (int)client->height;
        glBindBuffer(GL_PIXEL_PACK_BUFFER, client->pbo[prev]);
        unsigned char *ptr = (unsigned char *)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
        if (ptr) {
            write_gl_frame_flipped(client->recorder_pipefd[1], ptr, w, h);
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        } else {
            fprintf(stderr, "[drive-pbo] final glMapBuffer returned NULL, GL error=0x%x\n", glGetError());
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        glDeleteBuffers(2, client->pbo);
    }
#endif

    if (client->road_cache_valid && client->road_tri_mesh.vertexCount > 0) {
        UnloadMesh(client->road_tri_mesh);
    }
    free(client->road_line_verts);
    free(client->road_line_colors);

    stop_recorder(client);

    for (int i = 0; i < 6; i++)
        UnloadModel(client->cars[i]);
    UnloadModel(client->cyclist);
    UnloadModel(client->pedestrian);

    if (client->render_mode == RENDER_WINDOW) {
        CloseWindow();
    }

    free(client);
}
