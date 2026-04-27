#ifndef SIM_DRIVE_H
#define SIM_DRIVE_H

#include "datatypes.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INVALID_POSITION -10000.0f
#define EGO_IDX 0

// Control modes
#define CONTROL_VEHICLES 0
#define CONTROL_SDC_ONLY 1

// Simulation modes
#define SIMULATION_GIGAFLOW 0
#define SIMULATION_REPLAY 1

// Lane selection scoring
#define LANE_SELECTION_DISTANCE_WEIGHT 0.7f
#define LANE_SELECTION_HEADING_WEIGHT 0.3f
#define LANE_DISTANCE_NORMALIZATION 4.0f
#define LANE_SWITCH_THRESHOLD 0.05f // Hysteresis: new lane must be 5% better to switch
#define LANE_ALIGN_COS_THRESHOLD 0.5f

// Collision and distance thresholds
#define MAX_CHECKED_LANES 32
#define TRAFFIC_CONTROL_DIST_SQ (10.0f * 10.0f)
#define STOP_LINE_EXTENSION_FACTOR 1.5f
#define STOP_LINE_HEADING_THRESHOLD (M_PI / 4.0f)

// Collision/Infraction behaviors
#define STOP_AGENT 1
#define REMOVE_AGENT 2

#define MAX_SPEED 40.0f
#define LANE_WIDTH 3.7f

// Grid Map
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL 30

// TARGET_TYPE modes (controls what target info is in observations)
#define TARGET_STATIC 0
#define TARGET_DYNAMIC 1
#define MAX_ROUTE_LENGTH 64
#define ROUTE_TARGET_DISTANCE 1000.0f
#define ROUTE_EXIT_RANDOM_TOP_N 3

// Observation Space
#define EGO_FEATURES 9
#define PARTNER_FEATURES 9
#define ROAD_FEATURES 7
#define TRAFFIC_CONTROL_FEATURES 7
#define STATIC_TARGET_FEATURES 3
#define DYNAMIC_TARGET_FEATURES 5
#define OBS_COUNT_FEATURES 4

// Traffic light generation
#define TL_DEFAULT_RED_DURATION 2.0f
#define TL_DEFAULT_YELLOW_DURATION 3.0f
#define TL_DEFAULT_GREEN_DURATION 10.0f
#define TL_EPISODE_DISABLE_PROB 0.20f
#define TL_INDIVIDUAL_REMOVE_PROB 0.20f
#define TL_ALWAYS_GREEN_PROB 0.05f

// 2.5D Z estimation
#define Z_BUFFER 4.0f
#define Z_NUM_PT_AVG 30

typedef struct {
    float z_dis;
    float euclidean_dis;
    float z;
} DepthPoint;

static const int ROAD_OFFSETS[25][2]
    = {{-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {-2, -1}, {-1, -1}, {0, -1}, {1, -1},
       {2, -1},  {-2, 0},  {-1, 0}, {0, 0},  {1, 0},  {2, 0},   {-2, 1},  {-1, 1}, {0, 1},
       {1, 1},   {2, 1},   {-2, 2}, {-1, 2}, {0, 2},  {1, 2},   {2, 2}};

// Dynamics Models
#define CLASSIC 0
#define JERK 1
// Action Types
#define DISCRETE 0
#define CONTINUOUS 1

static const float JERK_LONG[4] = {-15.0f, -4.0f, 0.0f, 4.0f};
static const float JERK_LAT[3] = {-4.0f, 0.0f, 4.0f};

static const float ACCELERATION_VALUES[7] = {-4.0000f, -2.6670f, -1.3330f, -0.0000f, 1.3330f, 2.6670f, 4.0000f};
static const float STEERING_VALUES[9] = {-0.667f, -0.500f, -0.333f, -0.167f, 0.000f, 0.167f, 0.333f, 0.500f, 0.667f};

typedef struct Drive Drive;
typedef struct Client Client;
typedef struct Log Log;
typedef struct Agent Agent;
typedef struct RoadMapElement RoadMapElement;
typedef struct TrafficControlElement TrafficControlElement;
typedef struct GridMapEntity GridMapEntity;
typedef struct GridMap GridMap;

struct Log {
    float n;
    float episode_return;
    float episode_length;
    float score;
    float offroad_rate;
    float collision_rate;
    float red_light_violation_rate;
    float num_goals_reached;
    float comfort_violation_count;

    float velocity_progress_sum;
    float lane_center_rate;
    float dnf_rate;
    float avg_speed_per_agent;
    float total_distance_travelled;
    float total_infractions;
};

struct GridMapEntity {
    int entity_idx;
    int geometry_idx;
};

struct GridMap {
    float top_left_x;
    float top_left_y;
    float bottom_right_x;
    float bottom_right_y;
    int grid_cols;
    int grid_rows;
    int vision_range;
    int *cell_entities_count;
    int *neighbor_cache_count;
    int *grid_index_drivable;
    int num_drivable_grid_cell;
    GridMapEntity **cells;
    GridMapEntity **neighbor_cache_entities;
};

struct Drive {
    Client *client;
    Log log;
    Log *logs;
    // Rollout buffers
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;
    unsigned char *truncations;
    unsigned char *masks;
    // Grid map fields
    GridMap *grid_map;
    int *neighbor_offsets;
    // Entity fields
    Agent *agents;
    RoadMapElement *road_elements;
    TrafficControlElement *traffic_elements;
    int num_sim_agents;        // All valid agents: [active | moving_log | static_log]
    int num_max_agents;        // Max agents to keep active in the simulation (0 for no limit)
    int num_agents;            // Number of active agents in the simulation
    int num_moving_log_agents; // Number of log agents that moves during the scenario
    int num_road_elements;
    int num_traffic_elements;
    int num_objects;
    // Simulation state fields
    int timestep;
    // Env parameters
    float dt;
    int init_step;
    int dynamics_model;
    int action_type;
    int episode_length;
    int control_mode;
    int simulation_mode;
    int replay_expert_actions;
    int termination_mode;
    float inactive_agent_threshold;
    // Reward coefficients
    float reward_goal;
    float reward_collision;
    float reward_offroad;
    float reward_comfort;
    float reward_lane_align;
    float reward_vel_align;
    float reward_lane_center;
    float reward_center_bias;
    float reward_velocity;
    float reward_reverse;
    float reward_stop_line;
    float reward_timestep;
    float reward_overspeed;
    float spawn_initial_speed;
    float goal_radius;
    float goal_speed;
    float path_spacing;
    float min_goal_spacing;
    float max_goal_spacing;
    int num_target_waypoints;
    int target_type;
    int eval_mode;
    int min_agents_per_env;
    int max_agents_per_env;
    int collision_behavior;
    int offroad_behavior;
    int red_light_behavior;
    int traffic_control_scope;
    unsigned int rng;
    // Metadata fields
    char scenario_id[128];
    char dataset_name[32];
    char *map_name;
    int log_length;
    float log_dt;
    // Observation parameters
    int max_boundary_segment_observations;
    int max_lane_segment_observations;
    int max_partner_observations;
    int max_traffic_control_observations;
    float max_goal_position;
    float max_position;
    float max_veh_len;
    float max_veh_width;
    float max_road_segment_length;
    float max_road_segment_width;
    float max_traffic_control_distance;
    float agent_obs_max_dist;
    float road_obs_front_dist;
    float road_obs_behind_dist;
    float road_obs_side_dist;
    int obs_size;
};

#include "dataloader.h"

// ========================================
// Utility Functions
// ========================================

static float compute_euclidean_distance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

static int compare_depthpoint(const void *a, const void *b) {
    float diff = ((const DepthPoint *) a)->euclidean_dis - ((const DepthPoint *) b)->euclidean_dis;
    return (diff > 0.0f) - (diff < 0.0f);
}

static float clip(float value, float min, float max) {
    return value < min ? min : (value > max ? max : value);
}

static float normalize_heading(float heading) {
    heading = fmodf(heading, 2.0f * M_PI);
    if (heading > M_PI) {
        heading -= 2.0f * M_PI;
    } else if (heading < -M_PI) {
        heading += 2.0f * M_PI;
    }
    return heading;
}

static float compute_heading_diff(float heading1, float heading2) {
    return normalize_heading(heading1 - heading2);
}

static float random_uniform(float min_val, float max_val) {
    return min_val + ((float) rand() / (float) RAND_MAX) * (max_val - min_val);
}

static inline void zero_agent_velocity_state(Agent *agent) {
    agent->rear_vx = 0.0f;
    agent->rear_vy = 0.0f;
    agent->sim_vx = 0.0f;
    agent->sim_vy = 0.0f;
    agent->yaw_rate = 0.0f;
    agent->sim_speed = 0.0f;
    agent->sim_speed_signed = 0.0f;
    agent->accel_long = 0.0f;
    agent->accel_lat = 0.0f;
    agent->jerk_long = 0.0f;
    agent->jerk_lat = 0.0f;
}

static inline int action_dim_classic_discrete(void) {
    return (int) (sizeof(ACCELERATION_VALUES) / sizeof(ACCELERATION_VALUES[0]))
        * (int) (sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]));
}

static inline int action_dim_jerk_discrete(void) {
    return (int) (sizeof(JERK_LONG) / sizeof(JERK_LONG[0])) * (int) (sizeof(JERK_LAT) / sizeof(JERK_LAT[0]));
}

static void reset_agent_state(Agent *agent) {
    agent->stopped = 0;
    agent->removed = 0;
    agent->current_lane_idx = -1;
    agent->previous_lane_idx = -1;
    agent->current_route_index = 0;
    agent->accel_long = 0.0f;
    agent->accel_lat = 0.0f;
    agent->jerk_long = 0.0f;
    agent->jerk_lat = 0.0f;
    agent->steering_angle = 0.0f;
    agent->distance_since_spawn = 0.0f;
    agent->closest_path_idx_wp = 0;
}

static void invalidate_agent(Agent *agent) {
    agent->rear_x = INVALID_POSITION;
    agent->rear_y = INVALID_POSITION;
    agent->sim_x = INVALID_POSITION;
    agent->sim_y = INVALID_POSITION;
    agent->sim_z = 0.0f;
    agent->sim_heading = 0.0f;
    agent->cos_heading = 1.0f;
    agent->sin_heading = 0.0f;
    zero_agent_velocity_state(agent);
    agent->steering_angle = 0.0f;
    agent->sim_valid = 0;
    agent->removed = 1;
}

static inline void update_agent_rear_from_center(Agent *agent) {
    float rear_offset = 0.5f * agent->wheelbase;
    agent->rear_x = agent->sim_x - (rear_offset * agent->cos_heading);
    agent->rear_y = agent->sim_y - (rear_offset * agent->sin_heading);
}

static inline void update_agent_radius(Agent *agent) {
    agent->radius = 0.5f * sqrtf(agent->sim_length * agent->sim_length + agent->sim_width * agent->sim_width);
}

static inline void apply_infraction_behavior(Agent *agent, int behavior) {
    if (behavior == STOP_AGENT && !agent->stopped) {
        agent->stopped = 1;
    } else if (behavior == REMOVE_AGENT && !agent->removed) {
        agent->removed = 1;
    }
}

static inline void update_agent_center_from_rear(Agent *agent) {
    float center_offset = 0.5f * agent->wheelbase;
    agent->sim_x = agent->rear_x + (center_offset * agent->cos_heading);
    agent->sim_y = agent->rear_y + (center_offset * agent->sin_heading);
}

static inline void update_agent_speed(Agent *agent) {
    float speed = sqrtf(agent->rear_vx * agent->rear_vx + agent->rear_vy * agent->rear_vy);
    float v_dot_heading = agent->rear_vx * agent->cos_heading + agent->rear_vy * agent->sin_heading;
    agent->sim_speed = speed;
    agent->sim_speed_signed = copysignf(speed, v_dot_heading);
    agent->sim_vx = agent->rear_vx - (agent->yaw_rate * 0.5f * agent->wheelbase * agent->sin_heading);
    agent->sim_vy = agent->rear_vy + (agent->yaw_rate * 0.5f * agent->wheelbase * agent->cos_heading);
}

static inline float compute_log_yaw_rate(Agent *agent, int timestep, float dt) {
    int prev_t = timestep - 1;
    int next_t = timestep + 1;
    int has_prev = (prev_t >= 0) && (agent->log_valid[prev_t] == 1);
    int has_next = (next_t < agent->trajectory_length) && (agent->log_valid[next_t] == 1);

    if (has_prev && has_next) {
        float dtheta = normalize_heading(agent->log_heading[next_t] - agent->log_heading[prev_t]);
        return dtheta / (2.0f * dt);
    }
    if (has_next) {
        float dtheta = normalize_heading(agent->log_heading[next_t] - agent->log_heading[timestep]);
        return dtheta / dt;
    }
    if (has_prev) {
        float dtheta = normalize_heading(agent->log_heading[timestep] - agent->log_heading[prev_t]);
        return dtheta / dt;
    }

    return 0.0f;
}

static inline void project_vector_to_ego_frame(
    const Agent *ego,
    float world_x,
    float world_y,
    float *ego_x,
    float *ego_y) {
    *ego_x = world_x * ego->cos_heading + world_y * ego->sin_heading;
    *ego_y = -world_x * ego->sin_heading + world_y * ego->cos_heading;
}

static inline void project_point_to_ego_frame(
    const Agent *ego,
    float world_x,
    float world_y,
    float *ego_x,
    float *ego_y) {
    project_vector_to_ego_frame(ego, world_x - ego->rear_x, world_y - ego->rear_y, ego_x, ego_y);
}

// ========================================
// Grid Map Functions
// ========================================

static int get_grid_index(Drive *env, float x1, float y1) {
    if (env->grid_map->top_left_x >= env->grid_map->bottom_right_x
        || env->grid_map->bottom_right_y >= env->grid_map->top_left_y) {
        return -1;
    }
    float relativeX = x1 - env->grid_map->top_left_x;
    float relativeY = y1 - env->grid_map->bottom_right_y;
    int gridX = (int) (relativeX / GRID_CELL_SIZE);
    int gridY = (int) (relativeY / GRID_CELL_SIZE);
    if (gridX < 0 || gridX >= env->grid_map->grid_cols || gridY < 0 || gridY >= env->grid_map->grid_rows) {
        return -1;
    }
    return (gridY * env->grid_map->grid_cols) + gridX;
}

static void add_entity_to_grid(
    Drive *env,
    int grid_index,
    int entity_idx,
    int geometry_idx,
    int *cell_entities_insert_index) {
    if (grid_index == -1) {
        return;
    }

    int count = cell_entities_insert_index[grid_index];
    if (count >= env->grid_map->cell_entities_count[grid_index]) {
        return;
    }

    env->grid_map->cells[grid_index][count].entity_idx = entity_idx;
    env->grid_map->cells[grid_index][count].geometry_idx = geometry_idx;
    cell_entities_insert_index[grid_index] = count + 1;
}

static void init_grid_map(Drive *env) {
    env->grid_map = (GridMap *) malloc(sizeof(GridMap));
    env->grid_map->num_drivable_grid_cell = 0;

    float top_left_x = 0.0f, top_left_y = 0.0f, bottom_right_x = 0.0f, bottom_right_y = 0.0f;
    bool first_valid_point = false;
    for (int i = 0; i < env->num_road_elements; i++) {
        if (!is_road(env->road_elements[i].type)) {
            continue;
        }
        RoadMapElement *element = &env->road_elements[i];
        for (int j = 0; j < element->segment_length; j++) {
            if (element->x[j] == INVALID_POSITION || element->y[j] == INVALID_POSITION) {
                continue;
            }
            if (!first_valid_point) {
                top_left_x = bottom_right_x = element->x[j];
                top_left_y = bottom_right_y = element->y[j];
                first_valid_point = true;
                continue;
            }
            if (element->x[j] < top_left_x) {
                top_left_x = element->x[j];
            }
            if (element->x[j] > bottom_right_x) {
                bottom_right_x = element->x[j];
            }
            if (element->y[j] > top_left_y) {
                top_left_y = element->y[j];
            }
            if (element->y[j] < bottom_right_y) {
                bottom_right_y = element->y[j];
            }
        }
    }
    env->grid_map->top_left_x = top_left_x;
    env->grid_map->top_left_y = top_left_y;
    env->grid_map->bottom_right_x = bottom_right_x;
    env->grid_map->bottom_right_y = bottom_right_y;

    float grid_width = bottom_right_x - top_left_x;
    float grid_height = top_left_y - bottom_right_y;
    env->grid_map->grid_cols = ceil(grid_width / GRID_CELL_SIZE);
    env->grid_map->grid_rows = ceil(grid_height / GRID_CELL_SIZE);
    int grid_cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->cells = (GridMapEntity **) calloc(grid_cell_count, sizeof(GridMapEntity *));
    env->grid_map->cell_entities_count = (int *) calloc(grid_cell_count, sizeof(int));
    // First pass to count entities in each grid cell
    for (int i = 0; i < env->num_road_elements; i++) {
        if (!is_road(env->road_elements[i].type)) {
            continue;
        }
        RoadMapElement *element = &env->road_elements[i];
        for (int j = 0; j < element->segment_length - 1; j++) {
            float x_center = (element->x[j] + element->x[j + 1]) / 2;
            float y_center = (element->y[j] + element->y[j + 1]) / 2;
            int grid_index = get_grid_index(env, x_center, y_center);
            if (grid_index == -1) {
                continue;
            }
            env->grid_map->cell_entities_count[grid_index]++;
        }
    }
    // Allocate grid cells based on counts
    int *cell_entities_insert_index = (int *) calloc(grid_cell_count, sizeof(int));
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        int count = env->grid_map->cell_entities_count[grid_index];
        env->grid_map->cells[grid_index] = (GridMapEntity *) calloc(count, sizeof(GridMapEntity));
    }
    // Track which grid cells have drivable lanes
    bool *drivable_grid_seen = (bool *) calloc(grid_cell_count, sizeof(bool));
    for (int i = 0; i < env->num_road_elements; i++) {
        if (!is_road(env->road_elements[i].type)) {
            continue;
        }
        RoadMapElement *element = &env->road_elements[i];
        for (int j = 0; j < element->segment_length - 1; j++) {
            float x_center = (element->x[j] + element->x[j + 1]) / 2;
            float y_center = (element->y[j] + element->y[j + 1]) / 2;
            int grid_index = get_grid_index(env, x_center, y_center);
            if (grid_index == -1) {
                continue;
            }
            add_entity_to_grid(env, grid_index, i, j, cell_entities_insert_index);
            if (is_drivable_road_lane(element->type) && !drivable_grid_seen[grid_index]) {
                drivable_grid_seen[grid_index] = true;
                env->grid_map->num_drivable_grid_cell++;
            }
        }
    }
    // Create a compact array of drivable grid cell indices for quick access
    env->grid_map->grid_index_drivable = (int *) malloc(env->grid_map->num_drivable_grid_cell * sizeof(int));
    int drivable_idx = 0;
    for (int i = 0; i < grid_cell_count; i++) {
        if (drivable_grid_seen[i]) {
            env->grid_map->grid_index_drivable[drivable_idx++] = i;
        }
    }
    free(drivable_grid_seen);
    free(cell_entities_insert_index);
}

static void init_neighbor_offsets(Drive *env) {
    int vr = env->grid_map->vision_range;
    env->neighbor_offsets = (int *) calloc(vr * vr * 2, sizeof(int));
    // Spiral pattern generation
    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0, y = 0, dir = 0, steps_taken = 0, segments_completed = 0, total = 0, curr_idx = 0;
    int steps_to_take = 1;
    int max_offsets = vr * vr;
    env->neighbor_offsets[curr_idx++] = 0;
    env->neighbor_offsets[curr_idx++] = 0;
    total++;
    while (total < max_offsets) {
        x += dx[dir];
        y += dy[dir];
        if (abs(x) <= vr / 2 && abs(y) <= vr / 2) {
            env->neighbor_offsets[curr_idx++] = x;
            env->neighbor_offsets[curr_idx++] = y;
            total++;
        }
        steps_taken++;
        if (steps_taken != steps_to_take) {
            continue;
        }
        steps_taken = 0;
        dir = (dir + 1) % 4;
        segments_completed++;
        if (segments_completed % 2 == 0) {
            steps_to_take++;
        }
    }
}

static void cache_neighbor_offsets(Drive *env) {
    int count = 0;
    int cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->neighbor_cache_entities = (GridMapEntity **) calloc(cell_count, sizeof(GridMapEntity *));
    env->grid_map->neighbor_cache_count = (int *) calloc(cell_count + 1, sizeof(int));
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int current_cell_neighbor_count = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows) {
                continue;
            }
            int grid_count = env->grid_map->cell_entities_count[grid_index];
            current_cell_neighbor_count += grid_count;
        }
        env->grid_map->neighbor_cache_count[i] = current_cell_neighbor_count;
        count += current_cell_neighbor_count;
        if (current_cell_neighbor_count == 0) {
            env->grid_map->neighbor_cache_entities[i] = NULL;
            continue;
        }
        env->grid_map->neighbor_cache_entities[i]
            = (GridMapEntity *) calloc(current_cell_neighbor_count, sizeof(GridMapEntity));
    }

    env->grid_map->neighbor_cache_count[cell_count] = count;
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols;
        int cell_y = i / env->grid_map->grid_cols;
        int base_index = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows) {
                continue;
            }
            int grid_count = env->grid_map->cell_entities_count[grid_index];
            // Skip if no entities or source is NULL
            if (grid_count == 0 || env->grid_map->cells[grid_index] == NULL) {
                continue;
            }
            // Copy grid_count pairs (entity_idx, geometry_idx) at once
            memcpy(
                &env->grid_map->neighbor_cache_entities[i][base_index],
                env->grid_map->cells[grid_index],
                grid_count * sizeof(GridMapEntity));
            base_index += grid_count;
        }
    }
}

static int get_neighbors_entities(
    Drive *env,
    float x,
    float y,
    GridMapEntity *entity_list,
    int max_size,
    const int (*local_offsets)[2],
    int offset_size) {
    int index = get_grid_index(env, x, y);
    if (index == -1) {
        return 0;
    }
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
        if (nx < 0 || nx >= env->grid_map->grid_cols || ny < 0 || ny >= env->grid_map->grid_rows) {
            continue;
        }
        int neighbor_idx = ny * env->grid_map->grid_cols + nx;
        int count = env->grid_map->cell_entities_count[neighbor_idx];
        // Add entities from this cell to the list
        for (int j = 0; j < count && entity_list_count < max_size; j++) {
            entity_list[entity_list_count].entity_idx = env->grid_map->cells[neighbor_idx][j].entity_idx;
            entity_list[entity_list_count].geometry_idx = env->grid_map->cells[neighbor_idx][j].geometry_idx;
            entity_list_count += 1;
        }
    }
    return entity_list_count;
}

// ========================================
// Road Utility Functions
// ========================================

static float compute_lane_length(RoadMapElement *lane) {
    float length = 0.0f;
    for (int i = 1; i < lane->segment_length; i++) {
        float dx = lane->x[i] - lane->x[i - 1];
        float dy = lane->y[i] - lane->y[i - 1];
        length += sqrtf(dx * dx + dy * dy);
    }
    return length;
}

static float compute_lane_progress(
    RoadMapElement *lane,
    float pos_x,
    float pos_y,
    float cos_heading,
    float sin_heading,
    bool align_heading) {
    float best_progress = 0.0f;
    float best_dist_sq = 1e30f;

    for (int pass = 0; pass < 2; pass++) {
        float progress = 0.0f;
        for (int i = 0; i < lane->segment_length - 1; i++) {
            float x0 = lane->x[i];
            float y0 = lane->y[i];
            float x1 = lane->x[i + 1];
            float y1 = lane->y[i + 1];
            float dx = x1 - x0;
            float dy = y1 - y0;
            float seg_len_sq = dx * dx + dy * dy;
            float seg_len = sqrtf(seg_len_sq);

            if (seg_len_sq <= 1e-6f) {
                continue;
            }
            if (align_heading && pass == 0 && dx * cos_heading + dy * sin_heading < 0.0f) {
                progress += seg_len;
                continue;
            }

            float t = ((pos_x - x0) * dx + (pos_y - y0) * dy) / seg_len_sq;
            t = fmaxf(0.0f, fminf(1.0f, t));
            float proj_x = x0 + t * dx;
            float proj_y = y0 + t * dy;
            float dist_sq = (pos_x - proj_x) * (pos_x - proj_x) + (pos_y - proj_y) * (pos_y - proj_y);
            if (dist_sq < best_dist_sq) {
                best_dist_sq = dist_sq;
                best_progress = progress + t * seg_len;
            }
            progress += seg_len;
        }
        if (!align_heading || best_dist_sq < 1e30f) {
            break;
        }
    }

    return best_progress;
}

static DepthPoint compute_z_distance_to_road_segment(const Agent *agent, const RoadMapElement *lane, int geometry_idx) {
    float dx = agent->sim_x - lane->x[geometry_idx];
    float dy = agent->sim_y - lane->y[geometry_idx];
    float dz = agent->sim_z - lane->z[geometry_idx];

    DepthPoint point;
    point.z_dis = fabsf(dz);
    point.euclidean_dis = sqrtf(dx * dx + dy * dy + dz * dz);
    point.z = lane->z[geometry_idx];
    return point;
}

static void update_agent_z(Drive *env, Agent *agent) {
    static const int z_offsets[9][2] = {
        {-1, -1},
        {0, -1},
        {1, -1},
        {-1, 0},
        {0, 0},
        {1, 0},
        {-1, 1},
        {0, 1},
        {1, 1},
    };

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 9];
    int list_size
        = get_neighbors_entities(env, agent->sim_x, agent->sim_y, entity_list, MAX_ENTITIES_PER_CELL * 9, z_offsets, 9);
    if (list_size <= 0) {
        return;
    }

    DepthPoint neighbors[list_size];
    int valid_count = 0;
    int current_lane_count = 0;
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_idx == -1) {
            continue;
        }
        const RoadMapElement *entity = &env->road_elements[entity_list[i].entity_idx];
        DepthPoint point = compute_z_distance_to_road_segment(agent, entity, entity_list[i].geometry_idx);
        if (point.z_dis >= Z_BUFFER) {
            continue;
        }
        if (entity_list[i].entity_idx == agent->current_lane_idx) {
            if (current_lane_count < valid_count) {
                neighbors[valid_count] = neighbors[current_lane_count];
            }
            neighbors[current_lane_count++] = point;
            valid_count++;
        } else {
            neighbors[valid_count++] = point;
        }
    }

    int neighbor_count = (current_lane_count > 0) ? current_lane_count : valid_count;
    if (neighbor_count <= 0) {
        return;
    }

    if (neighbor_count > Z_NUM_PT_AVG) {
        qsort(neighbors, neighbor_count, sizeof(DepthPoint), compare_depthpoint);
        neighbor_count = Z_NUM_PT_AVG;
    }
    float sum_z = 0.0f;
    for (int i = 0; i < neighbor_count; i++) {
        sum_z += neighbors[i].z;
    }
    agent->sim_z = sum_z / neighbor_count;
}

// ========================================
// Route/Path/Goal Functions
// ========================================

static int get_closest_waypoint_index_on_path(Agent *agent) {
    if (agent->path == NULL || agent->path->num_waypoints == 0) {
        return 0;
    }
    const float MAX_DIST_SQ = 10000.0f;
    const int LOOKAHEAD = 10;
    const int LOOKBACK = 10;

    int hint_idx = agent->closest_path_idx_wp;
    if (hint_idx < 0 || hint_idx >= agent->path->num_waypoints) {
        hint_idx = 0;
    }
    int best_idx = hint_idx;
    float min_dist_sq = MAX_DIST_SQ;

    // Try windowed search first, fallback to full search if no candidate found
    int start_idx = fmax(0, hint_idx - LOOKBACK);
    int end_idx = fmin(agent->path->num_waypoints, hint_idx + LOOKAHEAD);
    for (int pass = 0; pass < 2; pass++) {
        for (int i = start_idx; i < end_idx; i++) {
            struct Waypoint *wp = &agent->path->waypoints[i];
            float dx = wp->x - agent->rear_x;
            float dy = wp->y - agent->rear_y;
            if (dx * agent->cos_heading + dy * agent->sin_heading < 0.0f) {
                continue;
            }
            if (agent->cos_heading * wp->cos_heading + agent->sin_heading * wp->sin_heading < 0.0f) {
                continue;
            }
            float dist_sq = dx * dx + dy * dy;
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                best_idx = i;
            }
        }
        if (min_dist_sq < MAX_DIST_SQ) {
            break;
        }
        // Fallback: full search
        start_idx = 0;
        end_idx = agent->path->num_waypoints;
    }
    return best_idx;
}

static void build_path(Drive *env, Agent *agent) {
    // NOTE: This function assumes the agent's route is already set.
    // It interpolates waypoints along the route lanes at fixed spacing.
    // It is a mid-level representation between route and low-level goals waypoints.
    float waypoints_spacing = env->path_spacing;

    if (agent->path != NULL) {
        free(agent->path);
    }
    agent->path = (struct Path *) malloc(sizeof(struct Path));

    int wp_count = 0;
    float prev_x, prev_y, prev_z, prev_s;

    // Interpolate waypoints along route lanes
    for (int route_idx = 0; route_idx < agent->route_length && wp_count < MAX_NUM_WP_PATH; route_idx++) {
        int lane_idx = agent->route[route_idx];
        RoadMapElement *lane = &env->road_elements[lane_idx];

        for (int i = 0; i < lane->segment_length && wp_count < MAX_NUM_WP_PATH; i++) {
            float curr_x = lane->x[i];
            float curr_y = lane->y[i];
            float curr_z = lane->z[i];

            // First point: add directly
            if (wp_count == 0) {
                agent->path->waypoints[0]
                    = (struct Waypoint) {.x = curr_x, .y = curr_y, .z = curr_z, .s = 0.0f, .lane_idx = lane_idx};
                prev_x = curr_x;
                prev_y = curr_y;
                prev_z = curr_z;
                prev_s = 0.0f;
                wp_count++;
                continue;
            }

            float dx = curr_x - prev_x;
            float dy = curr_y - prev_y;
            float dz = curr_z - prev_z;
            float seg_len = sqrtf(dx * dx + dy * dy + dz * dz);
            if (seg_len < 1e-6f) {
                prev_x = curr_x;
                prev_y = curr_y;
                prev_z = curr_z;
                continue;
            }

            float curr_s = prev_s + seg_len;

            // Interpolate waypoints within this segment
            float target_s = (float) wp_count * waypoints_spacing;
            while (target_s < curr_s && wp_count < MAX_NUM_WP_PATH) {
                float t = (target_s - prev_s) / seg_len;
                agent->path->waypoints[wp_count] = (struct Waypoint) {
                    .x = prev_x + t * dx,
                    .y = prev_y + t * dy,
                    .z = prev_z + t * dz,
                    .s = target_s,
                    .lane_idx = lane_idx,
                };
                wp_count++;
                target_s = (float) wp_count * waypoints_spacing;
            }

            prev_x = curr_x;
            prev_y = curr_y;
            prev_z = curr_z;
            prev_s = curr_s;
        }
    }

    agent->path->num_waypoints = wp_count;
    if (wp_count < 2) {
        return;
    }

    // Compute heading (tangent angle) and cache trig values
    for (int i = 0; i < wp_count - 1; i++) {
        float dx = agent->path->waypoints[i + 1].x - agent->path->waypoints[i].x;
        float dy = agent->path->waypoints[i + 1].y - agent->path->waypoints[i].y;
        float heading = atan2f(dy, dx);
        agent->path->waypoints[i].heading = heading;
        agent->path->waypoints[i].cos_heading = cosf(heading);
        agent->path->waypoints[i].sin_heading = sinf(heading);
    }
    // Last waypoint copies from second-to-last
    agent->path->waypoints[wp_count - 1].heading = agent->path->waypoints[wp_count - 2].heading;
    agent->path->waypoints[wp_count - 1].cos_heading = agent->path->waypoints[wp_count - 2].cos_heading;
    agent->path->waypoints[wp_count - 1].sin_heading = agent->path->waypoints[wp_count - 2].sin_heading;

    // Reset search hint before lookup — prior index is invalid on a freshly built path.
    agent->closest_path_idx_wp = 0;
    agent->closest_path_idx_wp = get_closest_waypoint_index_on_path(agent);
}

static int generate_random_route(
    RoadMapElement *road_elements,
    int start_lane_idx,
    int *route,
    int max_route_length,
    float *route_distance,
    float agent_x,
    float agent_y,
    float agent_cos_heading,
    float agent_sin_heading,
    const int *prev_route,
    int prev_route_length) {
    int route_length = 0;
    int current_lane_idx = start_lane_idx;
    route[route_length++] = current_lane_idx;

    RoadMapElement *current_lane = &road_elements[current_lane_idx];
    *route_distance += current_lane->length
        - compute_lane_progress(current_lane, agent_x, agent_y, agent_cos_heading, agent_sin_heading, true);

    // Random walk through lane graph. Prefer novel exits, but keep extending toward target distance.
    while (*route_distance < ROUTE_TARGET_DISTANCE && route_length < max_route_length) {
        if (current_lane_idx == -1) {
            break;
        }

        current_lane = &road_elements[current_lane_idx];
        int best_priority = 3;
        int top_exits[ROUTE_EXIT_RANDOM_TOP_N];
        float top_dists[ROUTE_EXIT_RANDOM_TOP_N];
        int top_count = 0;

        for (int e = 0; e < current_lane->num_exits; e++) {
            int exit_lane_idx = current_lane->exit_lanes[e];
            if (exit_lane_idx == -1) {
                continue;
            }
            int in_route = 0;
            for (int r = 0; r < route_length; r++) {
                if (route[r] == exit_lane_idx) {
                    in_route = 1;
                    break;
                }
            }

            int in_prev = 0;
            if (!in_route) {
                for (int p = 0; p < prev_route_length; p++) {
                    if (prev_route[p] == exit_lane_idx) {
                        in_prev = 1;
                        break;
                    }
                }
            }

            int priority = in_route ? 2 : (in_prev ? 1 : 0);
            RoadMapElement *exit_lane = &road_elements[exit_lane_idx];
            int endpoint_idx = exit_lane->segment_length - 1;
            float dx = exit_lane->x[endpoint_idx] - agent_x;
            float dy = exit_lane->y[endpoint_idx] - agent_y;
            float dist_sq = dx * dx + dy * dy;

            if (priority < best_priority) {
                best_priority = priority;
                top_count = 0;
            }
            if (priority != best_priority) {
                continue;
            }

            int insert_idx = top_count;
            while (insert_idx > 0 && dist_sq > top_dists[insert_idx - 1]) {
                if (insert_idx < ROUTE_EXIT_RANDOM_TOP_N) {
                    top_exits[insert_idx] = top_exits[insert_idx - 1];
                    top_dists[insert_idx] = top_dists[insert_idx - 1];
                }
                insert_idx--;
            }
            if (insert_idx >= ROUTE_EXIT_RANDOM_TOP_N) {
                continue;
            }
            top_exits[insert_idx] = exit_lane_idx;
            top_dists[insert_idx] = dist_sq;
            if (top_count < ROUTE_EXIT_RANDOM_TOP_N) {
                top_count++;
            }
        }

        if (top_count == 0) {
            break;
        }

        int chosen_exit_idx = top_exits[rand() % top_count];
        route[route_length++] = chosen_exit_idx;
        *route_distance += road_elements[chosen_exit_idx].length;
        current_lane_idx = chosen_exit_idx;
    }

    return route_length;
}

static int compute_new_route(Drive *env, Agent *agent, int current_lane_idx) {
    int temp_route[MAX_ROUTE_LENGTH];
    float route_distance = 0.0f;
    const int *prev_route = agent->route;
    int prev_route_length = (prev_route == NULL) ? 0 : agent->route_length;
    int route_length = generate_random_route(
        env->road_elements,
        current_lane_idx,
        temp_route,
        MAX_ROUTE_LENGTH,
        &route_distance,
        agent->sim_x,
        agent->sim_y,
        agent->cos_heading,
        agent->sin_heading,
        prev_route,
        prev_route_length);

    if (route_length == 0) {
        return 0;
    }

    // Free old route and allocate new one
    if (agent->route != NULL) {
        free(agent->route);
    }
    agent->route_length = route_length;
    agent->route = (int *) malloc(route_length * sizeof(int));
    for (int i = 0; i < route_length; i++) {
        agent->route[i] = temp_route[i];
    }

    agent->current_route_index = 0;

    return 1;
}

static int compute_goals(Drive *env, Agent *agent) {
    struct Path *path = agent->path;
    if (path == NULL || path->num_waypoints <= 0) {
        invalidate_agent(agent);
        return 0;
    }

    float goal_spacings[MAX_TARGET_POINTS];

    int min_steps = (int) ceilf(env->min_goal_spacing / env->path_spacing);
    int max_steps = (int) floorf(env->max_goal_spacing / env->path_spacing);
    int span = max_steps - min_steps + 1;
    float total_spacing = 0.0f;
    for (int i = 0; i < env->num_target_waypoints; i++) {
        goal_spacings[i] = (float) (min_steps + rand() % span) * env->path_spacing;
        total_spacing += goal_spacings[i];
    }

    int base_idx = get_closest_waypoint_index_on_path(agent);
    float base_s = path->waypoints[base_idx].s;
    float needed_s = base_s + total_spacing;
    float path_end_s = path->waypoints[path->num_waypoints - 1].s;

    if (env->simulation_mode == SIMULATION_REPLAY) {
        // In replay mode, if agent is near end of map, remove it
        if (base_idx >= path->num_waypoints - 3) {
            invalidate_agent(agent);
            return 0;
        }
    }

    if (needed_s >= path_end_s) {
        // Current agent is reaching the end of the path
        if (env->simulation_mode == SIMULATION_GIGAFLOW) {
            // Compute a new random route and path for the agent to continue on
            int start_lane_idx
                = (agent->current_lane_idx != -1) ? agent->current_lane_idx : path->waypoints[base_idx].lane_idx;
            if (!compute_new_route(env, agent, start_lane_idx)) {
                invalidate_agent(agent);
                printf(
                    "[GIGAFLOW WARNING] -> Failed to compute new route for agent %d. Removing from simulation.\n",
                    agent->id);
                return 0;
            }
            build_path(env, agent);
            path = agent->path;
            RoadMapElement *start_lane = &env->road_elements[start_lane_idx];
            float start_s = compute_lane_progress(
                start_lane,
                agent->sim_x,
                agent->sim_y,
                agent->cos_heading,
                agent->sin_heading,
                true);
            base_idx = path->num_waypoints - 1;
            for (int i = 0; i < path->num_waypoints; i++) {
                if (path->waypoints[i].s >= start_s) {
                    base_idx = i;
                    break;
                }
            }
            agent->closest_path_idx_wp = base_idx;
            base_s = path->waypoints[base_idx].s;
            needed_s = base_s + total_spacing;
            path_end_s = path->waypoints[path->num_waypoints - 1].s;
            if (needed_s >= path_end_s) {
                invalidate_agent(agent);
                printf(
                    "[GIGAFLOW ERROR] -> New route for agent %d is too short for goal generation. Removing from "
                    "simulation.\n",
                    agent->id);
                return 0;
            }
        } else if (env->simulation_mode == SIMULATION_REPLAY) {
            // Place remaining goals at end of path
            float goal_x[MAX_TARGET_POINTS];
            float goal_y[MAX_TARGET_POINTS];
            float goal_z[MAX_TARGET_POINTS];
            int valid_goal_count = 0;
            float cumulative_spacing = 0.0f;
            int end_idx = path->num_waypoints - 1;

            for (int i = 0; i < env->num_target_waypoints - 1; i++) {
                cumulative_spacing += goal_spacings[i];
                float target_s = base_s + cumulative_spacing;
                if (target_s >= path_end_s) {
                    break;
                }

                int wp_idx = end_idx;
                for (int j = base_idx + 1; j < path->num_waypoints; j++) {
                    if (path->waypoints[j].s >= target_s) {
                        wp_idx = j;
                        break;
                    }
                }
                if (wp_idx >= end_idx) {
                    break;
                }

                goal_x[valid_goal_count] = path->waypoints[wp_idx].x;
                goal_y[valid_goal_count] = path->waypoints[wp_idx].y;
                goal_z[valid_goal_count] = path->waypoints[wp_idx].z;
                valid_goal_count++;
            }

            goal_x[valid_goal_count] = path->waypoints[end_idx].x;
            goal_y[valid_goal_count] = path->waypoints[end_idx].y;
            goal_z[valid_goal_count] = path->waypoints[end_idx].z;
            valid_goal_count++;

            int start_idx = env->num_target_waypoints - valid_goal_count;
            for (int i = 0; i < start_idx; i++) {
                agent->goal_positions_x[i] = 0.0f;
                agent->goal_positions_y[i] = 0.0f;
                agent->goal_positions_z[i] = 0.0f;
            }
            for (int i = 0; i < valid_goal_count; i++) {
                int goal_idx = start_idx + i;
                agent->goal_positions_x[goal_idx] = goal_x[i];
                agent->goal_positions_y[goal_idx] = goal_y[i];
                agent->goal_positions_z[goal_idx] = goal_z[i];
            }
            agent->current_goal_idx = start_idx;
            agent->goal_position_x = agent->goal_positions_x[start_idx];
            agent->goal_position_y = agent->goal_positions_y[start_idx];
            agent->goal_position_z = agent->goal_positions_z[start_idx];
            return 1;
        }
    }

    // Place N goals along the path at random spacing intervals from current position
    float cumulative_spacing = 0.0f;
    for (int i = 0; i < env->num_target_waypoints; i++) {
        cumulative_spacing += goal_spacings[i];
        float target_s = base_s + cumulative_spacing;
        // Find waypoint at or past target_s
        int wp_idx = path->num_waypoints - 1;
        for (int j = base_idx + 1; j < path->num_waypoints; j++) {
            if (path->waypoints[j].s >= target_s) {
                wp_idx = j;
                break;
            }
        }
        agent->goal_positions_x[i] = path->waypoints[wp_idx].x;
        agent->goal_positions_y[i] = path->waypoints[wp_idx].y;
        agent->goal_positions_z[i] = path->waypoints[wp_idx].z;
    }
    agent->current_goal_idx = 0;
    agent->goal_position_x = agent->goal_positions_x[0];
    agent->goal_position_y = agent->goal_positions_y[0];
    agent->goal_position_z = agent->goal_positions_z[0];
    return 1;
}

// ========================================
// Metrics/Collision Functions
// ========================================

static bool check_line_intersection(float p1[2], float p2[2], float q1[2], float q2[2]) {
    if (fmaxf(p1[0], p2[0]) < fminf(q1[0], q2[0]) || fminf(p1[0], p2[0]) > fmaxf(q1[0], q2[0])
        || fmaxf(p1[1], p2[1]) < fminf(q1[1], q2[1]) || fminf(p1[1], p2[1]) > fmaxf(q1[1], q2[1])) {
        return false;
    }
    float dx1 = p2[0] - p1[0];
    float dy1 = p2[1] - p1[1];
    float dx2 = q2[0] - q1[0];
    float dy2 = q2[1] - q1[1];
    float cross = dx1 * dy2 - dy1 * dx2;
    if (cross == 0) {
        return false;
    }

    float dx3 = p1[0] - q1[0];
    float dy3 = p1[1] - q1[1];
    float s = (dx1 * dy3 - dy1 * dx3) / cross;
    float t = (dx2 * dy3 - dy2 * dx3) / cross;
    return (s >= 0 && s <= 1 && t >= 0 && t <= 1);
}

static void compute_agent_corners(Agent *agent, float corners[4][2]) {
    static const float offsets[4][2] = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}};
    float half_length = agent->sim_length / 2.0f;
    float half_width = agent->sim_width / 2.0f;

    for (int i = 0; i < 4; i++) {
        corners[i][0] = agent->sim_x
            + (offsets[i][0] * half_length * agent->cos_heading - offsets[i][1] * half_width * agent->sin_heading);
        corners[i][1] = agent->sim_y
            + (offsets[i][0] * half_length * agent->sin_heading + offsets[i][1] * half_width * agent->cos_heading);
    }
}

static bool check_agent_corners_cross_stop_line(float corners[4][2], TrafficControlElement *traffic) {
    float sl_dx = traffic->stop_line[3] - traffic->stop_line[0];
    float sl_dy = traffic->stop_line[4] - traffic->stop_line[1];
    float ext = (STOP_LINE_EXTENSION_FACTOR - 1.0f) * 0.5f;
    float ext_p1[2] = {traffic->stop_line[0] - ext * sl_dx, traffic->stop_line[1] - ext * sl_dy};
    float ext_p2[2] = {traffic->stop_line[3] + ext * sl_dx, traffic->stop_line[4] + ext * sl_dy};

    for (int k = 0; k < 4; k++) {
        if (k == 2) {
            continue;
        }
        int next = (k + 1) % 4;
        if (check_line_intersection(corners[k], corners[next], ext_p1, ext_p2)) {
            return true;
        }
    }

    return false;
}

static bool red_light_for_lane_in_range(
    Drive *env,
    TrafficControlElement *traffic,
    int lane_idx,
    float agent_x,
    float agent_y) {
    if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
        return false;
    }
    if (traffic->num_controlled_lanes == 0) {
        return false;
    }
    if (env->timestep >= traffic->state_length) {
        return false;
    }
    if (traffic->states[env->timestep] != TRAFFIC_CONTROL_STATE_RED) {
        return false;
    }
    int controls_lane = 0;
    for (int j = 0; j < traffic->num_controlled_lanes; j++) {
        if (traffic->controlled_lanes[j] == lane_idx) {
            controls_lane = 1;
            break;
        }
    }
    if (!controls_lane) {
        return false;
    }
    float mid_x = (traffic->stop_line[0] + traffic->stop_line[3]) * 0.5f;
    float mid_y = (traffic->stop_line[1] + traffic->stop_line[4]) * 0.5f;
    float dx = agent_x - mid_x;
    float dy = agent_y - mid_y;
    return (dx * dx + dy * dy) <= TRAFFIC_CONTROL_DIST_SQ;
}

static bool check_stop_line_crossing(Drive *env, Agent *agent, int current_lane_idx, float corners[4][2]) {
    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *traffic = &env->traffic_elements[i];
        if (!red_light_for_lane_in_range(env, traffic, current_lane_idx, agent->sim_x, agent->sim_y)) {
            continue;
        }
        // Heading check: agent must be heading towards the stop line
        float heading_diff = compute_heading_diff(agent->sim_heading, traffic->heading);
        if (fabsf(heading_diff) > STOP_LINE_HEADING_THRESHOLD) {
            continue;
        }
        if (check_agent_corners_cross_stop_line(corners, traffic)) {
            return true;
        }
    }
    return false;
}

static bool check_lane_change_red_light(Drive *env, Agent *agent) {
    if (agent->previous_lane_idx == agent->current_lane_idx) {
        return false;
    }
    if (agent->previous_lane_idx == -1 || agent->current_lane_idx == -1) {
        return false;
    }
    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *traffic = &env->traffic_elements[i];
        if (red_light_for_lane_in_range(env, traffic, agent->current_lane_idx, agent->sim_x, agent->sim_y)) {
            return true;
        }
    }
    return false;
}

static bool check_red_light_violation(Drive *env, Agent *agent) {
    int current_lane_idx = agent->current_lane_idx;
    if (current_lane_idx == -1) {
        return false;
    }

    float corners[4][2];
    compute_agent_corners(agent, corners);

    if (check_stop_line_crossing(env, agent, current_lane_idx, corners)) {
        return true;
    }

    if (check_lane_change_red_light(env, agent)) {
        return true;
    }

    return false;
}

static bool check_obb_collision(Agent *car1, Agent *car2) {
    // Early z-axis rejection
    float car1_top = car1->sim_z + car1->sim_height;
    float car2_top = car2->sim_z + car2->sim_height;
    if (car1_top < car2->sim_z || car2_top < car1->sim_z) {
        return false;
    }

    float cos1 = car1->cos_heading;
    float sin1 = car1->sin_heading;
    float cos2 = car2->cos_heading;
    float sin2 = car2->sin_heading;
    float car1_corners[4][2];
    compute_agent_corners(car1, car1_corners);
    float car2_corners[4][2];
    compute_agent_corners(car2, car2_corners);

    float axes[4][2] = {{cos1, sin1}, {-sin1, cos1}, {cos2, sin2}, {-sin2, cos2}};

    for (int i = 0; i < 4; i++) {
        float min1 = INFINITY, max1 = -INFINITY;
        float min2 = INFINITY, max2 = -INFINITY;
        for (int j = 0; j < 4; j++) {
            float proj1 = car1_corners[j][0] * axes[i][0] + car1_corners[j][1] * axes[i][1];
            min1 = fminf(min1, proj1);
            max1 = fmaxf(max1, proj1);
            float proj2 = car2_corners[j][0] * axes[i][0] + car2_corners[j][1] * axes[i][1];
            min2 = fminf(min2, proj2);
            max2 = fmaxf(max2, proj2);
        }
        if (max1 < min2 || min1 > max2) {
            return false;
        }
    }
    return true;
}

static int collision_check(Drive *env, Agent *agent) {
    if (agent->removed) {
        return -1;
    }

    int car_collided_with_index = -1;
    // Linear over all actors; pair radius quick-check prunes before OBB SAT.
    for (int i = 0; i < env->num_sim_agents; i++) {
        Agent *other_agent = &env->agents[i];
        if (agent == other_agent || other_agent->removed) {
            continue;
        }
        float threshold = agent->radius + other_agent->radius + 0.5f;
        float ddx = other_agent->sim_x - agent->sim_x;
        float ddy = other_agent->sim_y - agent->sim_y;
        if (ddx * ddx + ddy * ddy > threshold * threshold) {
            continue;
        }
        if (check_obb_collision(agent, other_agent)) {
            car_collided_with_index = i;
            break;
        }
    }

    return car_collided_with_index;
}

static void add_log(Drive *env) {
    env->log = (Log) {0};
    int safe_timestep = (env->timestep > 0) ? env->timestep : 1;
    for (int i = 0; i < env->num_agents; i++) {
        Agent *agent = &env->agents[i];
        int offroad = env->logs[i].offroad_rate;
        env->log.offroad_rate += offroad;
        int collided = env->logs[i].collision_rate;
        env->log.collision_rate += collided;
        int red_light_violations = env->logs[i].red_light_violation_rate;
        env->log.red_light_violation_rate += red_light_violations;
        int total_infractions = (offroad || collided || red_light_violations) ? 1 : 0;
        float avg_speed_per_agent = env->logs[i].avg_speed_per_agent;
        env->log.avg_speed_per_agent += avg_speed_per_agent / safe_timestep;
        int num_goals_reached = env->logs[i].num_goals_reached;
        env->log.num_goals_reached += num_goals_reached;
        if (!agent->removed && !agent->stopped) {
            if (num_goals_reached >= 1) {
                env->log.score += 1.0f;
            } else {
                env->log.dnf_rate += 1.0f;
            }
        }
        env->log.total_distance_travelled += agent->distance_since_spawn;
        if (total_infractions > 0) {
            env->log.total_infractions += 1.0f;
        }
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        env->log.n += 1;
    }
}

// ========================================
// Initialization Functions
// ========================================

static void generate_traffic_light_states(Drive *env) {
    int steps = env->episode_length;
    float dt = env->dt;

    if (steps <= 0) {
        return;
    }

    // Ensure all traffic light elements have a states array of the correct length
    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *traffic = &env->traffic_elements[i];
        if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            continue;
        }
        if (traffic->states && traffic->state_length != steps) {
            free(traffic->states);
            traffic->states = NULL;
        }
        if (traffic->states == NULL) {
            traffic->states = (int *) malloc(steps * sizeof(int));
            if (traffic->states == NULL) {
                traffic->state_length = 0;
                continue;
            }
        }
        traffic->state_length = steps;
    }

    // 20% chance: disable ALL lights for this episode
    int disable_all = (!env->eval_mode) && (random_uniform(0.0f, 1.0f) < TL_EPISODE_DISABLE_PROB);

    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *traffic = &env->traffic_elements[i];
        if (traffic->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT || traffic->states == NULL
            || traffic->state_length <= 0) {
            continue;
        }

        int fill_steps = steps;
        if (traffic->state_length < fill_steps) {
            fill_steps = traffic->state_length;
        }

        if (disable_all) {
            for (int t = 0; t < fill_steps; t++) {
                traffic->states[t] = TRAFFIC_CONTROL_STATE_OFF;
            }
            continue;
        }

        if (!env->eval_mode) {
            // Individual removal
            if (random_uniform(0.0f, 1.0f) < TL_INDIVIDUAL_REMOVE_PROB) {
                for (int t = 0; t < fill_steps; t++) {
                    traffic->states[t] = TRAFFIC_CONTROL_STATE_OFF;
                }
                continue;
            }
            // Always green
            if (random_uniform(0.0f, 1.0f) < TL_ALWAYS_GREEN_PROB) {
                for (int t = 0; t < fill_steps; t++) {
                    traffic->states[t] = TRAFFIC_CONTROL_STATE_GREEN;
                }
                continue;
            }
        }

        // Compute phase durations
        float dur_green, dur_yellow, dur_red;
        if (env->eval_mode) {
            dur_green = TL_DEFAULT_GREEN_DURATION;
            dur_yellow = TL_DEFAULT_YELLOW_DURATION;
            dur_red = TL_DEFAULT_RED_DURATION;
        } else {
            dur_green = random_uniform(0.1 * TL_DEFAULT_GREEN_DURATION, TL_DEFAULT_GREEN_DURATION);
            dur_yellow = random_uniform(0.5f * TL_DEFAULT_YELLOW_DURATION, 0.75f * TL_DEFAULT_YELLOW_DURATION);
            dur_red = random_uniform(0.15f * TL_DEFAULT_RED_DURATION, 5.0f * TL_DEFAULT_RED_DURATION);
        }

        int steps_green = (int) (dur_green / dt);
        if (steps_green < 1) {
            steps_green = 1;
        }
        int steps_yellow = (int) (dur_yellow / dt);
        if (steps_yellow < 1) {
            steps_yellow = 1;
        }
        int steps_red = (int) (dur_red / dt);
        if (steps_red < 1) {
            steps_red = 1;
        }
        int cycle_length = steps_green + steps_yellow + steps_red;

        // Random phase offset
        int offset = rand() % cycle_length;

        // Fill states: GREEN -> YELLOW -> RED -> repeat
        for (int t = 0; t < fill_steps; t++) {
            int phase = (t + offset) % cycle_length;
            if (phase < steps_green) {
                traffic->states[t] = TRAFFIC_CONTROL_STATE_GREEN;
            } else if (phase < steps_green + steps_yellow) {
                traffic->states[t] = TRAFFIC_CONTROL_STATE_YELLOW;
            } else {
                traffic->states[t] = TRAFFIC_CONTROL_STATE_RED;
            }
        }
    }
}

static bool check_spawn_collision(Drive *env, int num_existing_agents, Agent *tmp_agent) {
    float min_safe_dist_sq = (tmp_agent->sim_length + 5.0f) * (tmp_agent->sim_length + 5.0f);

    for (int i = 0; i < num_existing_agents; i++) {
        Agent *other = &env->agents[i];

        if (other->removed || other->sim_valid != 1) {
            continue;
        }

        float dx = other->sim_x - tmp_agent->sim_x;
        float dy = other->sim_y - tmp_agent->sim_y;
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq > min_safe_dist_sq) {
            continue;
        }
        if (check_obb_collision(tmp_agent, other)) {
            return true;
        }
    }

    return false;
}

static bool check_spawn_offroad(Drive *env, Agent *tmp_agent) {
    // Increase length and width slightly for spawn offroad check
    Agent scaled = *tmp_agent;
    scaled.sim_length *= 1.1f;
    scaled.sim_width *= 1.1f;

    float corners[4][2];
    compute_agent_corners(&scaled, corners);

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25];
    int list_size = get_neighbors_entities(
        env,
        tmp_agent->sim_x,
        tmp_agent->sim_y,
        entity_list,
        MAX_ENTITIES_PER_CELL * 25,
        ROAD_OFFSETS,
        25);

    for (int i = 0; i < list_size; i++) {
        int entity_idx = entity_list[i].entity_idx;
        int geometry_idx = entity_list[i].geometry_idx;
        RoadMapElement *element = &env->road_elements[entity_idx];

        if (is_road_edge(element->type)) {
            float abs_dz = fabsf(element->z[geometry_idx] - tmp_agent->sim_z);
            if (abs_dz > Z_BUFFER) {
                continue;
            }
            float start[2] = {element->x[geometry_idx], element->y[geometry_idx]};
            float end[2] = {element->x[geometry_idx + 1], element->y[geometry_idx + 1]};
            for (int k = 0; k < 4; k++) {
                int next = (k + 1) % 4;
                if (check_line_intersection(corners[k], corners[next], start, end)) {
                    return true;
                }
            }
        }
    }
    return false;
}

static bool check_spawn_red_light_violation(Drive *env, Agent *tmp_agent) {
    float corners[4][2];
    compute_agent_corners(tmp_agent, corners);
    return check_stop_line_crossing(env, tmp_agent, tmp_agent->current_lane_idx, corners);
}

static bool create_agent(Drive *env, Agent *agent, int num_agents) {
    // Free existing route on reset
    if (agent->route != NULL) {
        free(agent->route);
        agent->route = NULL;
    }
    agent->route_length = 0;
    agent->current_route_index = 0;

    // Initialize identity fields
    agent->type = VEHICLE;
    agent->control_state = CONTROL_STATE_ACTIVE;

    // Default vehicle dimensions
    // length: [0.8, 7.0] m
    // width: [0.8, 3.0] m
    // width = min(width, length)
    float spawn_length, spawn_width;
    if (env->eval_mode) {
        // Fixed size for eval mode
        spawn_length = random_uniform(2.0f, 5.5f);
        spawn_width = random_uniform(1.5f, 2.5f);
    } else {
        // Random size for training mode
        spawn_length = random_uniform(0.8f, 7.0f);
        spawn_width = random_uniform(0.8f, 3.0f);
    }
    if (spawn_width > spawn_length) {
        spawn_width = spawn_length;
    }
    float spawn_height = 1.5f; // Fixed height

    // Set spawn position on start lane
    float spawn_x, spawn_y, spawn_z, spawn_heading;
    RoadMapElement *start_lane;
    int start_lane_idx;
    bool is_agent_spawned = false;

    // Sampling rejection loop
    // TARGET: Only one attempt should be sufficient in most cases
    const int MAX_SPAWN_ATTEMPTS = 30;
    for (int attempt = 0; attempt < MAX_SPAWN_ATTEMPTS; attempt++) {
        int chosen_lane_idx = -1;

        int list_idx = rand() % env->grid_map->num_drivable_grid_cell;
        int grid_idx = env->grid_map->grid_index_drivable[list_idx];

        GridMapEntity cell_candidates[MAX_ENTITIES_PER_CELL];
        int candidate_count = 0;

        for (int i = 0; i < env->grid_map->cell_entities_count[grid_idx]; i++) {
            GridMapEntity entity = env->grid_map->cells[grid_idx][i];

            if (is_drivable_road_lane(env->road_elements[entity.entity_idx].type)) {
                if (candidate_count >= MAX_ENTITIES_PER_CELL) {
                    continue;
                }
                cell_candidates[candidate_count++] = entity;
            }
        }

        if (candidate_count == 0) {
            continue;
        }

        GridMapEntity chosen_entity = cell_candidates[rand() % candidate_count];
        chosen_lane_idx = chosen_entity.entity_idx;

        start_lane_idx = chosen_lane_idx;
        start_lane = &env->road_elements[start_lane_idx];

        spawn_x = start_lane->x[chosen_entity.geometry_idx];
        spawn_y = start_lane->y[chosen_entity.geometry_idx];
        spawn_z = start_lane->z[chosen_entity.geometry_idx];
        spawn_heading = start_lane->headings[chosen_entity.geometry_idx];

        Agent tmp_agent = {0};
        tmp_agent.sim_x = spawn_x;
        tmp_agent.sim_y = spawn_y;
        tmp_agent.sim_z = spawn_z;
        tmp_agent.sim_heading = spawn_heading;
        tmp_agent.cos_heading = cosf(spawn_heading);
        tmp_agent.sin_heading = sinf(spawn_heading);
        tmp_agent.yaw_rate = 0.0f;
        tmp_agent.sim_length = spawn_length;
        tmp_agent.sim_width = spawn_width;
        tmp_agent.sim_height = spawn_height;
        update_agent_radius(&tmp_agent);
        tmp_agent.current_lane_idx = start_lane_idx;

        if (check_spawn_collision(env, num_agents, &tmp_agent)) {
            continue;
        }

        if (check_spawn_offroad(env, &tmp_agent)) {
            continue;
        }

        if (check_spawn_red_light_violation(env, &tmp_agent)) {
            continue;
        }

        is_agent_spawned = true;
        break;
    }

    if (!is_agent_spawned) {
        printf("[GIGAFLOW WARNING] -> Failed to find a collision-free spawn position for agent %d\n", agent->id);
        return is_agent_spawned;
    }

    // Update simulation state
    agent->sim_x = spawn_x;
    agent->sim_y = spawn_y;
    agent->sim_z = spawn_z;
    agent->sim_heading = spawn_heading;
    agent->cos_heading = cosf(spawn_heading);
    agent->sin_heading = sinf(spawn_heading);
    agent->sim_length = spawn_length;
    agent->sim_width = spawn_width;
    agent->sim_height = spawn_height;
    update_agent_radius(agent);
    agent->sim_valid = 1;
    agent->wheelbase = 0.6f * spawn_length; // Estimate wheelbase as 60% of length
    agent->current_lane_idx = start_lane_idx;
    update_agent_rear_from_center(agent);

    // Optional launch speed for gigaflow debugging and curriculum. Keep the
    // initial velocity aligned with the spawn heading.
    float spawn_speed = clip(env->spawn_initial_speed, 0.0f, MAX_SPEED);
    agent->yaw_rate = 0.0f;
    agent->rear_vx = spawn_speed * agent->cos_heading;
    agent->rear_vy = spawn_speed * agent->sin_heading;
    update_agent_speed(agent);

    return is_agent_spawned;
}

static void set_agent_at_init_log_step(Drive *env) {
    assert(env->init_step >= 0);
    assert(env->init_step < env->log_length);
    for (int i = 0; i < env->num_sim_agents; i++) {
        Agent *agent = &env->agents[i];
        int step = env->init_step;
        if (agent->log_valid[step] != 1) {
            invalidate_agent(agent);
            continue;
        }
        agent->sim_x = agent->log_trajectory_x[step];
        agent->sim_y = agent->log_trajectory_y[step];
        agent->sim_z = agent->log_trajectory_z[step];
        agent->sim_heading = agent->log_heading[step];
        agent->cos_heading = cosf(agent->sim_heading);
        agent->sin_heading = sinf(agent->sim_heading);
        agent->sim_valid = agent->log_valid[step];
        agent->sim_length = agent->log_length[step];
        agent->sim_width = agent->log_width[step];
        agent->sim_height = agent->log_height[step];
        update_agent_radius(agent);
        agent->wheelbase = 0.6f * agent->sim_length;
        if (agent->control_state == CONTROL_STATE_STATIC) {
            zero_agent_velocity_state(agent);
            continue;
        }
        agent->yaw_rate = compute_log_yaw_rate(agent, step, env->dt);
        update_agent_rear_from_center(agent);
        agent->rear_vx = agent->log_velocity_x[step] + (agent->yaw_rate * 0.5f * agent->wheelbase * agent->sin_heading);
        agent->rear_vy = agent->log_velocity_y[step] - (agent->yaw_rate * 0.5f * agent->wheelbase * agent->cos_heading);
        update_agent_speed(agent);
    }
}

static void spawn_active_agents(Drive *env) {
    for (int i = 0; i < env->num_agents; i++) {
        Agent *agent = &env->agents[i];
        reset_agent_state(agent);

        if (!create_agent(env, agent, i)) {
            invalidate_agent(agent);
            fprintf(stderr, "[GIGAFLOW WARNING] -> Failed to spawn agent %d\n", i);
            continue;
        }

        if (!compute_new_route(env, agent, agent->current_lane_idx)) {
            invalidate_agent(agent);
            fprintf(stderr, "[GIGAFLOW WARNING] -> Failed to initialize route for agent %d\n", i);
        }
    }
}

void set_active_agents(Drive *env) {
    env->num_agents = 0;
    env->num_moving_log_agents = 0;
    int num_valid_agents = 0;

    if (env->simulation_mode == SIMULATION_GIGAFLOW) {
        int num_agents_to_create;
        if (env->num_max_agents == 0) {
            int lo = env->min_agents_per_env > 0 ? env->min_agents_per_env : 1;
            int hi = env->max_agents_per_env >= lo ? env->max_agents_per_env : lo;
            printf(
                "[GIGAFLOW INFO] -> min_agents_per_env: %d, max_agents_per_env: %d\n",
                env->min_agents_per_env,
                env->max_agents_per_env);
            printf("[GIGAFLOW INFO] -> Randomly sampling number of agents to create between %d and %d\n", lo, hi);
            num_agents_to_create = lo + rand() % (hi - lo + 1);
            printf("[GIGAFLOW INFO] -> Created %d agents in this episode\n", num_agents_to_create);
        } else {
            num_agents_to_create = env->num_max_agents;
        }
        free(env->agents);
        env->agents = (Agent *) calloc(num_agents_to_create, sizeof(Agent));
        env->num_sim_agents = num_agents_to_create;
        env->num_agents = num_agents_to_create;
        return;
    }

    Agent *raw_agents = env->agents;
    Agent *compacted_agents = (Agent *) calloc(env->num_sim_agents, sizeof(Agent));
    int *kept = (int *) calloc(env->num_sim_agents, sizeof(int));
    int max_active = env->num_max_agents == 0 ? env->num_sim_agents : env->num_max_agents;
    bool is_log_replay = (env->control_mode == CONTROL_SDC_ONLY);

    // First pass: collect controlled agents up to the max_active limit
    for (int i = 0; i < env->num_sim_agents; i++) {
        Agent *agent = &raw_agents[i];
        if (agent->log_valid[env->init_step] == 0 || env->num_agents >= max_active) {
            continue;
        }

        bool should_control = false;
        if (is_log_replay) {
            should_control = i == EGO_IDX;
        } else {
            should_control = agent->control_state == CONTROL_STATE_ACTIVE;
        }

        if (!should_control) {
            continue;
        }
        compacted_agents[num_valid_agents] = *agent;
        compacted_agents[num_valid_agents].control_state = CONTROL_STATE_ACTIVE;
        num_valid_agents++;
        kept[i] = 1;
        env->num_agents++;
    }

    // Second pass: collect moving-log/replay agents
    for (int i = 0; i < env->num_sim_agents; i++) {
        Agent *agent = &raw_agents[i];
        if (kept[i]) {
            continue;
        }

        bool should_move = false;
        if (is_log_replay) {
            should_move = true;
        } else if (agent->log_valid[env->init_step] == 0) {
            continue;
        } else {
            should_move = agent->control_state <= CONTROL_STATE_MOVING;
        }

        if (!should_move) {
            continue;
        }

        compacted_agents[num_valid_agents] = *agent;
        compacted_agents[num_valid_agents].control_state = CONTROL_STATE_MOVING;
        num_valid_agents++;
        kept[i] = 1;
        env->num_moving_log_agents++;
    }

    // Third pass: collect remaining valid log agents
    for (int i = 0; i < env->num_sim_agents; i++) {
        Agent *agent = &raw_agents[i];
        if (kept[i] || agent->log_valid[env->init_step] == 0) {
            continue;
        }
        compacted_agents[num_valid_agents] = *agent;
        compacted_agents[num_valid_agents].control_state = CONTROL_STATE_STATIC;
        num_valid_agents++;
        kept[i] = 1;
    }

    for (int i = 0; i < env->num_sim_agents; i++) {
        if (kept[i]) {
            continue;
        }
        free_agent(&raw_agents[i]);
    }

    free(kept);
    free(raw_agents);

    if (env->num_agents == 0) {
        printf(
            "[ERROR] -> No control agents found in the logs at the initial step. Please check the log data and initial "
            "step configuration.\n");
        for (int i = 0; i < num_valid_agents; i++) {
            free_agent(&compacted_agents[i]);
        }
        free(compacted_agents);
        env->agents = NULL;
        env->num_sim_agents = 0;
        return;
    }

    env->agents = (Agent *) realloc(compacted_agents, num_valid_agents * sizeof(Agent));
    env->num_sim_agents = num_valid_agents;
}

static void move_expert(Drive *env, int agent_idx) {
    if (env->simulation_mode == SIMULATION_GIGAFLOW) {
        printf("[GIGAFLOW ERROR] -> move_expert() called in GIGAFLOW mode\n");
        return;
    }
    bool is_log_replay = (env->control_mode == CONTROL_SDC_ONLY);

    Agent *agent = &env->agents[agent_idx];
    int t = env->timestep;
    if (t < 0 || t >= agent->trajectory_length || agent->log_valid[t] == 0) {
        invalidate_agent(agent);
        return;
    }
    agent->sim_x = agent->log_trajectory_x[t];
    agent->sim_y = agent->log_trajectory_y[t];
    agent->sim_z = agent->log_trajectory_z[t];
    agent->sim_heading = agent->log_heading[t];
    agent->cos_heading = cosf(agent->sim_heading);
    agent->sin_heading = sinf(agent->sim_heading);
    agent->sim_valid = agent->log_valid[t];
    if (is_log_replay) {
        agent->sim_length = agent->log_length[t];
        agent->sim_width = agent->log_width[t];
        agent->sim_height = agent->log_height[t];
        update_agent_radius(agent);
        agent->wheelbase = 0.6f * agent->sim_length;
    }
    agent->yaw_rate = compute_log_yaw_rate(agent, t, env->dt);
    agent->sim_vx = agent->log_velocity_x[t];
    agent->sim_vy = agent->log_velocity_y[t];
    agent->sim_speed = sqrtf(agent->sim_vx * agent->sim_vx + agent->sim_vy * agent->sim_vy);
    update_agent_rear_from_center(agent);
}

static int compute_observation_size(Drive *env) {
    int target_features = (env->target_type == TARGET_STATIC) ? STATIC_TARGET_FEATURES
        : (env->target_type == TARGET_DYNAMIC)                ? DYNAMIC_TARGET_FEATURES
                                                              : 0;
    return EGO_FEATURES + PARTNER_FEATURES * env->max_partner_observations
        + ROAD_FEATURES * (env->max_lane_segment_observations + env->max_boundary_segment_observations)
        + TRAFFIC_CONTROL_FEATURES * env->max_traffic_control_observations + OBS_COUNT_FEATURES
        + env->num_target_waypoints * target_features;
}

int init(Drive *env) {
    env->timestep = 0;
    if (load_map_binary(env->map_name, env) != 0) {
        fprintf(stderr, "[ERROR] -> Failed to load map binary: %s\n", env->map_name);
        return -1;
    }
    for (int i = 0; i < env->num_road_elements; i++) {
        env->road_elements[i].length = compute_lane_length(&env->road_elements[i]);
    }
    init_grid_map(env);
    int vision_half_range = (int) ceilf(
        fmaxf(fmaxf(env->road_obs_front_dist, env->road_obs_behind_dist), env->road_obs_side_dist) / GRID_CELL_SIZE);
    env->grid_map->vision_range = 2 * vision_half_range + 1;
    init_neighbor_offsets(env);
    cache_neighbor_offsets(env);
    set_active_agents(env);
    env->logs = (Log *) calloc(env->num_agents, sizeof(Log));
    env->obs_size = compute_observation_size(env);
    return 0;
}

void free_env(Drive *env) {
    for (int i = 0; i < env->num_sim_agents; i++) {
        free_agent(&env->agents[i]);
    }
    for (int i = 0; i < env->num_road_elements; i++) {
        free_road_element(&env->road_elements[i]);
    }
    for (int i = 0; i < env->num_traffic_elements; i++) {
        free_traffic_element(&env->traffic_elements[i]);
    }
    free(env->agents);
    free(env->road_elements);
    free(env->traffic_elements);
    free(env->logs);
    free(env->neighbor_offsets);
    if (env->grid_map) {
        int grid_cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
        for (int i = 0; i < grid_cell_count; i++) {
            free(env->grid_map->cells[i]);
        }
        free(env->grid_map->cells);
        free(env->grid_map->cell_entities_count);
        free(env->grid_map->grid_index_drivable);
        for (int i = 0; i < grid_cell_count; i++) {
            free(env->grid_map->neighbor_cache_entities[i]);
        }
        free(env->grid_map->neighbor_cache_entities);
        free(env->grid_map->neighbor_cache_count);
        free(env->grid_map);
    }
}

void c_close(Drive *env) {
    free_env(env);
}

int allocate(Drive *env) {
    if (init(env) != 0) {
        return -1;
    }
    env->observations = (float *) calloc(env->num_agents * env->obs_size, sizeof(float));
    env->actions = (float *) calloc(env->num_agents * 2, sizeof(float));
    env->rewards = (float *) calloc(env->num_agents, sizeof(float));
    env->terminals = (unsigned char *) calloc(env->num_agents, sizeof(unsigned char));
    env->truncations = (unsigned char *) calloc(env->num_agents, sizeof(unsigned char));
    env->masks = (unsigned char *) calloc(env->num_agents, sizeof(unsigned char));
    return 0;
}

void free_allocated(Drive *env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->truncations);
    free(env->masks);
    c_close(env);
}

// ========================================
// Core Simulation Functions
// ========================================

static void compute_metrics(Drive *env, int i) {
    Agent *agent = &env->agents[i];

    for (int j = 0; j < NUM_METRICS; j++) {
        agent->metrics_array[j] = 0.0f;
    }

    if (agent->removed) {
        return; // invalid agent position
    }

    // Current agent is offgrid, treat as offroad
    if (get_grid_index(env, agent->sim_x, agent->sim_y) == -1) {
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        apply_infraction_behavior(agent, env->offroad_behavior);
        return;
    }

    bool is_offroad = false;

    // Track best candidate by combined distance/heading score
    float best_score = 1e9f;
    int lane_idx = -1;
    float signed_lane_distance = 0.0f, lane_heading = 0.0f;

    float corners[4][2];
    compute_agent_corners(agent, corners);

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25];
    int list_size = get_neighbors_entities(
        env,
        agent->sim_x,
        agent->sim_y,
        entity_list,
        MAX_ENTITIES_PER_CELL * 25,
        ROAD_OFFSETS,
        25);

    if (list_size <= 0) {
        is_offroad = true;
    }

    // Vehicle-width based distance threshold (3x width)
    float max_distance_threshold = 3.0f * agent->sim_width;

    int checked_lanes[MAX_CHECKED_LANES];
    int num_checked_lanes = 0;

    // Loop through road entities and compute associated metrics (offroad, lane alignment)
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_idx == -1) {
            continue;
        }

        int entity_idx = entity_list[i].entity_idx;
        int geometry_idx = entity_list[i].geometry_idx;
        RoadMapElement *element = &env->road_elements[entity_idx];

        // Check for offroad collision with road edges
        if (is_road_edge(element->type)) {
            float start[2] = {element->x[geometry_idx], element->y[geometry_idx]};
            float end[2] = {element->x[geometry_idx + 1], element->y[geometry_idx + 1]};
            float abs_dz = fabsf(element->z[geometry_idx] - agent->sim_z);
            if (abs_dz > Z_BUFFER) {
                continue;
            }
            for (int k = 0; k < 4; k++) { // Check each edge of the bounding box
                int next = (k + 1) % 4;
                if (check_line_intersection(corners[k], corners[next], start, end)) {
                    is_offroad = true;
                    break;
                }
            }
        }

        if (is_offroad) {
            break;
        }

        if (!is_drivable_road_lane(element->type)) {
            continue;
        }

        int already_checked = 0;
        for (int c = 0; c < num_checked_lanes; c++) {
            if (checked_lanes[c] == entity_idx) {
                already_checked = 1;
                break;
            }
        }
        if (already_checked) {
            continue;
        }
        if (num_checked_lanes < MAX_CHECKED_LANES) {
            checked_lanes[num_checked_lanes++] = entity_idx;
        }

        // Find closest segment on this lane (signed distance: left = negative, right = positive)
        int closest_seg_idx = 0;
        float signed_dist = 1e9f;
        int num_segments = element->segment_length - 1;
        if (num_segments >= 1) {
            float min_dist_sq = 1e18f;
            float closest_cross = 0.0f;
            for (int seg_idx = 0; seg_idx < num_segments; seg_idx++) {
                float seg_start_x = element->x[seg_idx];
                float seg_start_y = element->y[seg_idx];
                float seg_end_x = element->x[seg_idx + 1];
                float seg_end_y = element->y[seg_idx + 1];
                float seg_dx = seg_end_x - seg_start_x;
                float seg_dy = seg_end_y - seg_start_y;
                float seg_length_sq = seg_dx * seg_dx + seg_dy * seg_dy;
                float to_agent_x = agent->sim_x - seg_start_x;
                float to_agent_y = agent->sim_y - seg_start_y;
                float cross = seg_dx * to_agent_y - seg_dy * to_agent_x;
                float dist_sq;
                if (seg_length_sq > 1e-6f) {
                    float t = (to_agent_x * seg_dx + to_agent_y * seg_dy) / seg_length_sq;
                    if (t <= 0.0f) {
                        dist_sq = to_agent_x * to_agent_x + to_agent_y * to_agent_y;
                    } else if (t >= 1.0f) {
                        float dxe = agent->sim_x - seg_end_x;
                        float dye = agent->sim_y - seg_end_y;
                        dist_sq = dxe * dxe + dye * dye;
                    } else {
                        dist_sq = (cross * cross) / seg_length_sq;
                    }
                } else {
                    dist_sq = to_agent_x * to_agent_x + to_agent_y * to_agent_y;
                }
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    closest_seg_idx = seg_idx;
                    closest_cross = cross;
                }
            }
            float abs_dist_val = sqrtf(min_dist_sq);
            signed_dist = (closest_cross >= 0.0f) ? -abs_dist_val : abs_dist_val;
        }

        float abs_dist = fabsf(signed_dist);
        if (abs_dist > max_distance_threshold) {
            continue;
        }

        // Multi-segment lane heading (more weight on center segment)
        float avg_lane_heading = 0.0f;
        float total_weight = 0.0f;
        int seg_start = (closest_seg_idx > 0) ? (closest_seg_idx - 1) : closest_seg_idx;
        int seg_end
            = (closest_seg_idx < element->segment_length - 2) ? (closest_seg_idx + 1) : (element->segment_length - 2);
        for (int seg_idx = seg_start; seg_idx <= seg_end; seg_idx++) {
            if (seg_idx < 0 || seg_idx >= element->segment_length - 1) {
                continue;
            }
            float seg_heading = element->headings[seg_idx];
            float weight = (seg_idx == closest_seg_idx) ? 2.0f : 1.0f;
            if (total_weight == 0.0f) {
                avg_lane_heading = seg_heading;
            } else {
                float angle_diff = compute_heading_diff(seg_heading, avg_lane_heading);
                avg_lane_heading += weight * angle_diff / (total_weight + weight);
            }
            total_weight += weight;
        }

        float heading_diff = compute_heading_diff(agent->sim_heading, avg_lane_heading);
        float heading_penalty = fabsf(heading_diff) / M_PI;
        float distance_penalty = abs_dist / LANE_DISTANCE_NORMALIZATION;
        float score
            = LANE_SELECTION_DISTANCE_WEIGHT * distance_penalty + LANE_SELECTION_HEADING_WEIGHT * heading_penalty;
        if (agent->current_lane_idx != entity_idx && agent->current_lane_idx != -1) {
            score += LANE_SWITCH_THRESHOLD;
        }

        if (score < best_score) {
            best_score = score;
            lane_idx = entity_idx;
            signed_lane_distance = signed_dist;
            lane_heading = avg_lane_heading;
        }
    }

    // Update lane alignment metric (running average)
    if (lane_idx != -1) {
        agent->previous_lane_idx = agent->current_lane_idx;
        agent->current_lane_idx = lane_idx;

        // Lane distance and angle metrics
        // x_f = lateral offset from lane center (left = negative, right = positive)
        agent->metrics_array[LANE_DIST_IDX] = signed_lane_distance;
        // theta_f = angle relative to lane heading
        float theta_f = compute_heading_diff(agent->sim_heading, lane_heading);
        agent->metrics_array[LANE_ANGLE_IDX] = cosf(theta_f); // Store cos(θ_f)
    } else {
        // Agent not on any lane
        agent->previous_lane_idx = -1;
        agent->current_lane_idx = -1;
        agent->metrics_array[LANE_DIST_IDX] = LANE_DISTANCE_NORMALIZATION; // Max distance (far from lane)
        agent->metrics_array[LANE_ANGLE_IDX] = 0.0f;                       // Perpendicular (no alignment)
    }

    agent->closest_path_idx_wp = get_closest_waypoint_index_on_path(agent);

    // Speed limit metric
    float target_speed = 15.0f; // Default target speed
    int current_lane_idx = agent->current_lane_idx;
    if (current_lane_idx != -1 && env->road_elements[current_lane_idx].speed_limit > 0) {
        target_speed = env->road_elements[current_lane_idx].speed_limit;
    }
    agent->metrics_array[SPEED_LIMIT_IDX] = (agent->sim_speed > target_speed + 2.0f) ? 1.0f : 0.0f;

    // Velocity metric (GIGAFLOW) - forward progress aligned with lane
    const float VELOCITY_MIN_SPEED = 2.5f; // m/s
    if (agent->sim_speed_signed > VELOCITY_MIN_SPEED && lane_idx != -1) {
        float cos_theta = agent->metrics_array[LANE_ANGLE_IDX];
        agent->metrics_array[VELOCITY_PROGRESS_IDX] = fmaxf(cos_theta, 0.0f);
    } else {
        agent->metrics_array[VELOCITY_PROGRESS_IDX] = 0.0f;
    }

    // Comfort metric: asymmetric longitudinal thresholds allow stronger braking than acceleration.
    const float COMFORT_ACCEL_THRESHOLD = 3.0f; // m/s²
    const float COMFORT_JERK_THRESHOLD = 5.0f;  // m/s³
    int accel_violation
        = (fabsf(agent->accel_long) > COMFORT_ACCEL_THRESHOLD) + (fabsf(agent->accel_lat) > COMFORT_ACCEL_THRESHOLD);
    int jerk_violation
        = (fabsf(agent->jerk_long) > COMFORT_JERK_THRESHOLD || fabsf(agent->jerk_lat) > COMFORT_JERK_THRESHOLD) ? 1 : 0;
    agent->metrics_array[COMFORT_VIOLATION_IDX] = (float) (accel_violation + jerk_violation);

    if (is_offroad) {
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        apply_infraction_behavior(agent, env->offroad_behavior);
        return;
    }

    int car_collided_with_index = collision_check(env, agent);
    if (car_collided_with_index != -1) {
        agent->metrics_array[COLLISION_IDX] = 1.0f;
        apply_infraction_behavior(agent, env->collision_behavior);
        return;
    }

    if (env->max_traffic_control_observations && check_red_light_violation(env, agent)) {
        agent->metrics_array[RED_LIGHT_IDX] = 1.0f;
        apply_infraction_behavior(agent, env->red_light_behavior);
        return;
    }

    // Goal reaching
    float distance_to_goal
        = compute_euclidean_distance(agent->sim_x, agent->sim_y, agent->goal_position_x, agent->goal_position_y);
    float goal_z_dist = fabsf(agent->sim_z - agent->goal_position_z);
    if (distance_to_goal < 2.0f && goal_z_dist < Z_BUFFER) {
        agent->metrics_array[REACH_GOAL_IDX] = 1.0f;
    }
}

static void compute_rewards(Drive *env, int i) {
    Agent *agent = &env->agents[i];

    // Collision reward
    if (agent->metrics_array[COLLISION_IDX] > 0.0f) {
        // Velocity-dependent penalty: incentivizes braking before unavoidable collision.
        // At max speed (~20 m/s): extra -2.0 on top of base coefficient.
        float reward_collision = -(env->reward_collision + 0.1f * agent->sim_speed);
        env->rewards[i] += reward_collision;
        env->logs[i].episode_return += reward_collision;
        env->logs[i].collision_rate = 1.0f;
    }

    // Offroad reward
    if (agent->metrics_array[OFFROAD_IDX] > 0.0f) {
        float reward_offroad = -env->reward_offroad;
        env->rewards[i] += reward_offroad;
        env->logs[i].episode_return += reward_offroad;
        env->logs[i].offroad_rate = 1.0f;
    }

    // Red light violation reward
    if (agent->metrics_array[RED_LIGHT_IDX] > 0.0f) {
        float reward_red_light = -env->reward_stop_line;
        env->rewards[i] += reward_red_light;
        env->logs[i].red_light_violation_rate = 1.0f;
        env->logs[i].episode_return += reward_red_light;
    }

    // Goal reward
    if (agent->metrics_array[REACH_GOAL_IDX] > 0.0f) {
        float reward_goal = env->reward_goal;
        env->rewards[i] += reward_goal;
        env->logs[i].episode_return += reward_goal;
    }

    // Get lane angle metric: cos(θ_f) where θ_f = heading diff from lane
    float cos_theta = agent->metrics_array[LANE_ANGLE_IDX];
    float theta_f = acosf(fminf(fmaxf(cos_theta, -1.0f), 1.0f)); // Get |θ_f| from cos

    // Rl-align: min(cos,0) + vel_align*min(cos*v,0) + 0.0025*(1-|θ|/(π/2))
    float against_lane_penalty = fminf(cos_theta, 0.0f); // negative when >90 degrees off
    float vel_aligned_penalty = env->reward_vel_align * fminf(cos_theta * agent->sim_speed, 0.0f);
    float alignment_bonus = 0.0025f * (1.0f - theta_f / (M_PI / 2.0f));
    float lane_align_reward
        = env->reward_lane_align * env->dt * (against_lane_penalty + vel_aligned_penalty + alignment_bonus);
    env->rewards[i] += lane_align_reward;
    env->logs[i].episode_return += lane_align_reward;

    // Rl-center: -α * dt * (|x_f - bias| - 0.05 / exp(|x_f - bias| - 0.5))
    float lane_center_distance = agent->metrics_array[LANE_DIST_IDX];
    float adjusted_dist = fabsf(lane_center_distance - env->reward_center_bias);
    float exp_decay = 0.05f / expf(adjusted_dist - 0.5f);
    float lane_center_reward = -env->reward_lane_center * env->dt * ((cos_theta > 0.5f) * adjusted_dist - exp_decay);
    env->rewards[i] += lane_center_reward;
    env->logs[i].lane_center_rate += fabsf(lane_center_distance) < 0.5f ? 1.0f : 0.0f;
    env->logs[i].episode_return += lane_center_reward;

    // Comfort reward
    float comfort_violations = agent->metrics_array[COMFORT_VIOLATION_IDX];
    float comfort_penalty = -env->reward_comfort * comfort_violations;
    env->rewards[i] += comfort_penalty;
    env->logs[i].comfort_violation_count += comfort_violations;
    env->logs[i].episode_return += comfort_penalty;

    // Velocity reward
    float velocity_progress = agent->metrics_array[VELOCITY_PROGRESS_IDX];
    float velocity_reward = env->reward_velocity * env->dt * velocity_progress;
    env->rewards[i] += velocity_reward;
    env->logs[i].velocity_progress_sum += velocity_progress;
    env->logs[i].episode_return += velocity_reward;

    // Timestep reward
    float accel = sqrtf(agent->accel_long * agent->accel_long + agent->accel_lat * agent->accel_lat);
    if (agent->sim_speed > 0.01f || accel > 0.01f) {
        float timestep_penalty = -env->reward_timestep * env->dt;
        env->rewards[i] += timestep_penalty;
        env->logs[i].episode_return += timestep_penalty;
    }

    // Reverse reward
    if (agent->sim_speed_signed < -0.01f) {
        float reverse_penalty = -env->reward_reverse * env->dt;
        env->rewards[i] += reverse_penalty;
        env->logs[i].episode_return += reverse_penalty;
    }

    // Speed limit reward
    float speed_reward = -env->reward_overspeed * agent->metrics_array[SPEED_LIMIT_IDX];
    env->rewards[i] += speed_reward;
    env->logs[i].avg_speed_per_agent += agent->sim_speed;
    env->logs[i].episode_return += speed_reward;
}

static int write_ego_obs(Drive *env, Agent *ego, float *obs, int obs_idx) {
    obs[obs_idx++] = ego->sim_speed_signed / MAX_SPEED;
    obs[obs_idx++] = ego->sim_width / env->max_veh_width;
    obs[obs_idx++] = ego->sim_length / env->max_veh_len;
    obs[obs_idx++] = ego->steering_angle / STEERING_VALUES[8];
    obs[obs_idx++] = (ego->accel_long < 0) ? ego->accel_long / (-JERK_LONG[0]) : ego->accel_long / JERK_LONG[3];
    obs[obs_idx++] = ego->accel_lat / JERK_LAT[2];
    obs[obs_idx++] = fmaxf(-1.0f, fminf(1.0f, ego->metrics_array[LANE_DIST_IDX] / LANE_DISTANCE_NORMALIZATION));
    obs[obs_idx++] = ego->metrics_array[LANE_ANGLE_IDX];
    float lane_speed_limit
        = (ego->current_lane_idx != -1) ? env->road_elements[ego->current_lane_idx].speed_limit : -1.0f;
    obs[obs_idx++] = lane_speed_limit / MAX_SPEED;
    return obs_idx;
}

static int write_reward_target_obs(Drive *env, Agent *ego, float *obs, int obs_idx) {
    if (env->target_type == TARGET_STATIC) {
        for (int wp = 0; wp < env->num_target_waypoints; wp++) {
            if (wp < ego->current_goal_idx) {
                obs[obs_idx++] = 0.0f;
                obs[obs_idx++] = 0.0f;
                obs[obs_idx++] = 0.0f;
                continue;
            }
            float rel_goal_x, rel_goal_y;
            project_point_to_ego_frame(
                ego,
                ego->goal_positions_x[wp],
                ego->goal_positions_y[wp],
                &rel_goal_x,
                &rel_goal_y);
            obs[obs_idx++] = rel_goal_x / env->max_goal_position;
            obs[obs_idx++] = rel_goal_y / env->max_goal_position;
            obs[obs_idx++] = (ego->goal_positions_z[wp] - ego->sim_z) / Z_BUFFER;
        }
        return obs_idx;
    }

    if (env->target_type == TARGET_DYNAMIC && ego->path != NULL && ego->path->num_waypoints > 0) {
        for (int wp = 0; wp < env->num_target_waypoints; wp++) {
            int wp_index = fmin(ego->closest_path_idx_wp + wp, ego->path->num_waypoints - 1);
            if (wp_index < 0) {
                wp_index = 0;
            }
            struct Waypoint *waypoint = &ego->path->waypoints[wp_index];
            float rel_wp_x, rel_wp_y, rel_heading_x, rel_heading_y;
            float wp_z = waypoint->z - ego->sim_z;
            project_point_to_ego_frame(ego, waypoint->x, waypoint->y, &rel_wp_x, &rel_wp_y);
            project_vector_to_ego_frame(
                ego,
                waypoint->cos_heading,
                waypoint->sin_heading,
                &rel_heading_x,
                &rel_heading_y);
            obs[obs_idx++] = rel_wp_x / env->max_position;
            obs[obs_idx++] = rel_wp_y / env->max_position;
            obs[obs_idx++] = wp_z / Z_BUFFER;
            obs[obs_idx++] = rel_heading_x;
            obs[obs_idx++] = rel_heading_y;
        }
        return obs_idx;
    }

    return obs_idx + DYNAMIC_TARGET_FEATURES * env->num_target_waypoints;
}

static int write_partner_obs(Drive *env, Agent *ego, int agent_idx, float *obs, int obs_idx, int *partner_count) {
    typedef struct {
        int index;
        float dist_sq;
        float dx;
        float dy;
        float dz;
    } AgentDistance;
    AgentDistance candidates[env->num_sim_agents - 1];
    int candidate_count = 0;
    float agent_obs_max_dist_sq = env->agent_obs_max_dist * env->agent_obs_max_dist;
    for (int j = 0; j < env->num_sim_agents; j++) {
        if (j == agent_idx) {
            continue;
        }
        Agent *other_entity = &env->agents[j];
        float dx = other_entity->sim_x - ego->rear_x;
        float dy = other_entity->sim_y - ego->rear_y;
        float dz = other_entity->sim_z - ego->sim_z;
        float dist_sq = dx * dx + dy * dy + dz * dz;
        if (dist_sq > agent_obs_max_dist_sq || fabsf(dz) > Z_BUFFER) {
            continue;
        }
        candidates[candidate_count].index = j;
        candidates[candidate_count].dist_sq = dist_sq;
        candidates[candidate_count].dx = dx;
        candidates[candidate_count].dy = dy;
        candidates[candidate_count].dz = dz;
        candidate_count++;
    }

    int cars_seen = 0;
    int num_agents_to_observe
        = (candidate_count < env->max_partner_observations) ? candidate_count : env->max_partner_observations;
    for (int k = 0; k < num_agents_to_observe; k++) {
        int min_idx = k;
        for (int j = k + 1; j < candidate_count; j++) {
            if (candidates[j].dist_sq < candidates[min_idx].dist_sq) {
                min_idx = j;
            }
        }
        if (min_idx != k) {
            AgentDistance tmp = candidates[k];
            candidates[k] = candidates[min_idx];
            candidates[min_idx] = tmp;
        }
    }

    for (int j = 0; j < num_agents_to_observe; j++) {
        Agent *other = &env->agents[candidates[j].index];
        float rel_x, rel_y, rel_heading_x, rel_heading_y, rel_vx, rel_vy;
        project_vector_to_ego_frame(ego, candidates[j].dx, candidates[j].dy, &rel_x, &rel_y);
        project_vector_to_ego_frame(ego, other->cos_heading, other->sin_heading, &rel_heading_x, &rel_heading_y);
        project_vector_to_ego_frame(ego, other->sim_vx, other->sim_vy, &rel_vx, &rel_vy);
        obs[obs_idx++] = rel_x / env->max_position;
        obs[obs_idx++] = rel_y / env->max_position;
        obs[obs_idx++] = candidates[j].dz / Z_BUFFER;
        obs[obs_idx++] = other->sim_length / env->max_veh_len;
        obs[obs_idx++] = other->sim_width / env->max_veh_width;
        obs[obs_idx++] = rel_heading_x;
        obs[obs_idx++] = rel_heading_y;
        obs[obs_idx++] = rel_vx / MAX_SPEED;
        obs[obs_idx++] = rel_vy / MAX_SPEED;
        cars_seen++;
    }

    *partner_count = cars_seen;
    return obs_idx + (env->max_partner_observations - cars_seen) * PARTNER_FEATURES;
}

static int write_road_obs(Drive *env, Agent *ego, float *obs, int obs_idx, int *lane_count, int *boundary_count) {
    int grid_idx = get_grid_index(env, ego->rear_x, ego->rear_y);
    int list_size = 0;
    const GridMapEntity *entity_list = NULL;
    if (!(grid_idx < 0 || grid_idx >= (env->grid_map->grid_cols * env->grid_map->grid_rows))) {
        list_size = env->grid_map->neighbor_cache_count[grid_idx];
        entity_list = env->grid_map->neighbor_cache_entities[grid_idx];
    }

    int lane_obs_idx = obs_idx;
    int boundary_obs_idx = lane_obs_idx + env->max_lane_segment_observations * ROAD_FEATURES;
    obs_idx = boundary_obs_idx + env->max_boundary_segment_observations * ROAD_FEATURES;

    float *lanes_dest = &obs[lane_obs_idx];
    float *boundaries_dest = &obs[boundary_obs_idx];
    int lanes_collected = 0;
    int boundaries_collected = 0;

    for (int k = 0; k < list_size; k++) {
        if (lanes_collected >= env->max_lane_segment_observations
            && boundaries_collected >= env->max_boundary_segment_observations) {
            break;
        }
        int entity_idx = entity_list[k].entity_idx;
        int geometry_idx = entity_list[k].geometry_idx;
        RoadMapElement *element = &env->road_elements[entity_idx];
        int is_lane = is_road_lane(element->type);
        int is_edge = is_road_edge(element->type);
        if (!is_lane && !is_edge) {
            continue;
        }

        float start_x = element->x[geometry_idx];
        float start_y = element->y[geometry_idx];
        float start_z = element->z[geometry_idx];
        float end_x = element->x[geometry_idx + 1];
        float end_y = element->y[geometry_idx + 1];
        float end_z = element->z[geometry_idx + 1];
        float mid_x = (start_x + end_x) / 2.0f;
        float mid_y = (start_y + end_y) / 2.0f;
        float mid_z = (start_z + end_z) / 2.0f;
        float rel_x, rel_y;
        float rel_z = mid_z - ego->sim_z;
        project_point_to_ego_frame(ego, mid_x, mid_y, &rel_x, &rel_y);
        if (rel_x < -env->road_obs_behind_dist || rel_x > env->road_obs_front_dist) {
            continue;
        }
        if (fabsf(rel_y) > env->road_obs_side_dist || fabsf(rel_z) > Z_BUFFER) {
            continue;
        }

        float dx = end_x - mid_x;
        float dy = end_y - mid_y;
        float length = sqrtf(dx * dx + dy * dy);
        float dx_norm = (length > 0) ? dx / length : dx;
        float dy_norm = (length > 0) ? dy / length : dy;
        float cos_angle, sin_angle;
        project_vector_to_ego_frame(ego, dx_norm, dy_norm, &cos_angle, &sin_angle);

        float *target = is_lane ? lanes_dest : boundaries_dest;
        int *counter = is_lane ? &lanes_collected : &boundaries_collected;
        int cap = is_lane ? env->max_lane_segment_observations : env->max_boundary_segment_observations;
        if (*counter >= cap) {
            continue;
        }
        int base = (*counter)++ * ROAD_FEATURES;
        target[base] = rel_x / env->max_position;
        target[base + 1] = rel_y / env->max_position;
        target[base + 2] = rel_z / Z_BUFFER;
        target[base + 3] = length / env->max_road_segment_length;
        target[base + 4] = LANE_WIDTH / env->max_road_segment_width;
        target[base + 5] = cos_angle;
        target[base + 6] = sin_angle;
    }

    *lane_count = lanes_collected;
    *boundary_count = boundaries_collected;
    return obs_idx;
}

static int write_traffic_control_obs(Drive *env, Agent *ego, float *obs, int obs_idx, int *traffic_control_count) {
    typedef struct {
        int idx;
        float dist_sq;
    } TrafficControlDist;
    TrafficControlDist traffic_controls[env->num_traffic_elements > 0 ? env->num_traffic_elements : 1];
    int num_visible_controls = 0;
    float tc_max_dist_sq = env->max_traffic_control_distance * env->max_traffic_control_distance;

    for (int j = 0; j < env->num_traffic_elements; j++) {
        TrafficControlElement *traffic = &env->traffic_elements[j];
        if (!traffic_control_in_scope(traffic->type, env->traffic_control_scope)) {
            continue;
        }
        float mid_x = (traffic->stop_line[0] + traffic->stop_line[3]) * 0.5f;
        float mid_y = (traffic->stop_line[1] + traffic->stop_line[4]) * 0.5f;
        float mid_z = (traffic->stop_line[2] + traffic->stop_line[5]) * 0.5f;
        float dx = mid_x - ego->rear_x;
        float dy = mid_y - ego->rear_y;
        float dz = mid_z - ego->sim_z;
        float dist_sq = dx * dx + dy * dy + dz * dz;
        if (dist_sq > tc_max_dist_sq || fabsf(dz) > Z_BUFFER) {
            continue;
        }
        traffic_controls[num_visible_controls].idx = j;
        traffic_controls[num_visible_controls].dist_sq = dist_sq;
        num_visible_controls++;
    }

    int num_controls_to_observe = (num_visible_controls < env->max_traffic_control_observations)
        ? num_visible_controls
        : env->max_traffic_control_observations;
    for (int k = 0; k < num_controls_to_observe; k++) {
        int min_idx = k;
        for (int j = k + 1; j < num_visible_controls; j++) {
            if (traffic_controls[j].dist_sq < traffic_controls[min_idx].dist_sq) {
                min_idx = j;
            }
        }
        if (min_idx != k) {
            TrafficControlDist temp = traffic_controls[k];
            traffic_controls[k] = traffic_controls[min_idx];
            traffic_controls[min_idx] = temp;
        }
    }

    int controls_added = 0;
    for (int j = 0; j < num_controls_to_observe && controls_added < env->max_traffic_control_observations; j++) {
        TrafficControlElement *traffic = &env->traffic_elements[traffic_controls[j].idx];
        float rel_x1, rel_y1, rel_x2, rel_y2;
        project_point_to_ego_frame(ego, traffic->stop_line[0], traffic->stop_line[1], &rel_x1, &rel_y1);
        project_point_to_ego_frame(ego, traffic->stop_line[3], traffic->stop_line[4], &rel_x2, &rel_y2);
        float rel_z = (traffic->stop_line[2] + traffic->stop_line[5]) * 0.5f - ego->sim_z;
        int state = TRAFFIC_CONTROL_STATE_UNKNOWN;
        if (traffic->type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            state = (env->timestep >= 0 && env->timestep < traffic->state_length && traffic->states != NULL)
                ? traffic->states[env->timestep]
                : TRAFFIC_CONTROL_STATE_OFF;
        }

        obs[obs_idx++] = rel_x1 / env->max_position;
        obs[obs_idx++] = rel_y1 / env->max_position;
        obs[obs_idx++] = rel_x2 / env->max_position;
        obs[obs_idx++] = rel_y2 / env->max_position;
        obs[obs_idx++] = rel_z / Z_BUFFER;
        obs[obs_idx++] = traffic->type;
        obs[obs_idx++] = state;
        controls_added++;
    }

    *traffic_control_count = controls_added;
    return obs_idx + (env->max_traffic_control_observations - controls_added) * TRAFFIC_CONTROL_FEATURES;
}

static void compute_observations(Drive *env) {
    int max_obs = env->obs_size;

    memset(env->observations, 0, (size_t) max_obs * env->num_agents * sizeof(float));
    float (*observations)[max_obs] = (float (*)[max_obs]) env->observations;
    for (int i = 0; i < env->num_agents; i++) {
        float *obs = &observations[i][0];
        Agent *ego = &env->agents[i];
        int partner_count = 0;
        int lane_count = 0;
        int boundary_count = 0;
        int traffic_control_count = 0;
        int obs_idx = 0;

        obs_idx = write_ego_obs(env, ego, obs, obs_idx);
        obs_idx = write_reward_target_obs(env, ego, obs, obs_idx);
        obs_idx = write_partner_obs(env, ego, i, obs, obs_idx, &partner_count);
        obs_idx = write_road_obs(env, ego, obs, obs_idx, &lane_count, &boundary_count);
        obs_idx = write_traffic_control_obs(env, ego, obs, obs_idx, &traffic_control_count);
        obs[obs_idx++] = (float) lane_count;
        obs[obs_idx++] = (float) boundary_count;
        obs[obs_idx++] = (float) partner_count;
        obs[obs_idx++] = (float) traffic_control_count;
        assert(obs_idx == max_obs);
    }
}

static void move_dynamics(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];

    if (agent->removed) {
        invalidate_agent(agent);
        return;
    }
    if (agent->stopped) {
        zero_agent_velocity_state(agent);
        return;
    }

    float speed = agent->sim_speed_signed;
    float steering = agent->steering_angle;
    float accel_long_new = 0.0f, accel_lat_new = 0.0f, yaw_rate = 0.0f;

    if (env->dynamics_model == CLASSIC) {
        float accel = 0.0f, steer_req = 0.0f;

        if (env->action_type == DISCRETE) {
            int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
            int val = (int) env->actions[agent_idx];
            assert(val >= 0);
            assert(val < action_dim_classic_discrete());
            accel = ACCELERATION_VALUES[val / num_steer];
            steer_req = STEERING_VALUES[val % num_steer];
        } else if (env->action_type == CONTINUOUS) {
            float (*act)[2] = (float (*)[2]) env->actions;
            accel = act[agent_idx][0] * ACCELERATION_VALUES[6];
            steer_req = act[agent_idx][1] * STEERING_VALUES[8];
        }

        // Steering rate and position limits
        float delta_steer = clip(steer_req - steering, -0.6f * env->dt, 0.6f * env->dt);
        steering = clip(steering + delta_steer, -0.667f, 0.667f);
        // Update state
        speed = clip(speed + accel * env->dt, -MAX_SPEED, MAX_SPEED);
        yaw_rate = (speed * tanf(steering)) / agent->wheelbase;
        // Simple bicycle model kinematics
        agent->rear_x += speed * agent->cos_heading * env->dt;
        agent->rear_y += speed * agent->sin_heading * env->dt;
        agent->sim_heading = normalize_heading(agent->sim_heading + yaw_rate * env->dt);
        // Accelerations update (not used in kinematics)
        accel_long_new = accel;
        accel_lat_new = speed * yaw_rate;

    } else if (env->dynamics_model == JERK) {
        float j_long = 0.0f, j_lat = 0.0f;

        if (env->action_type == DISCRETE) {
            int num_lat = sizeof(JERK_LAT) / sizeof(JERK_LAT[0]);
            int val = (int) env->actions[agent_idx];
            assert(val >= 0);
            assert(val < action_dim_jerk_discrete());
            j_long = JERK_LONG[val / num_lat];
            j_lat = JERK_LAT[val % num_lat];
        } else if (env->action_type == CONTINUOUS) {
            float (*act)[2] = (float (*)[2]) env->actions;
            float jl_act = act[agent_idx][0];
            j_long = (jl_act < 0) ? jl_act * (-JERK_LONG[0]) : jl_act * JERK_LONG[3];
            j_lat = act[agent_idx][1] * JERK_LAT[2];
        }

        // Integrate jerk to get new accelerations
        accel_long_new = agent->accel_long + (j_long * env->dt);
        accel_long_new = (agent->accel_long * accel_long_new < 0) ? 0.0f : clip(accel_long_new, -5.0f, 2.5f);
        // Lateral acceleration update with steering influence
        accel_lat_new = agent->accel_lat + (j_lat * env->dt);
        accel_lat_new = (agent->accel_lat * accel_lat_new < 0) ? 0.0f : clip(accel_lat_new, -4.0f, 4.0f);
        // Velocity (Trapezoidal)
        float v_new = clip(speed + 0.5f * (accel_long_new + agent->accel_long) * env->dt, -2.0f, 20.0f);
        if (speed * v_new < 0) {
            v_new = 0.0f;
        }
        // Curvature and Steering
        float v_eff = fmaxf(fabsf(v_new), 1.0f);
        float curvature = accel_lat_new / (v_eff * v_eff);
        float steer_req = atanf(curvature * agent->wheelbase);
        float delta_steer = clip(steer_req - steering, -0.6f * env->dt, 0.6f * env->dt);
        steering = clip(steering + delta_steer, -0.55f, 0.55f);
        // Recalculate based on limits
        float final_curv = tanf(steering) / agent->wheelbase;
        accel_lat_new = v_new * v_new * final_curv;
        float dist = 0.5f * (v_new + speed) * env->dt;
        float theta = dist * final_curv;
        // Local to Global translation
        float dx_l = (fabsf(theta) < 1e-5f) ? dist : sinf(theta) / final_curv;
        float dy_l = (fabsf(theta) < 1e-5f) ? 0.0f : (1.0f - cosf(theta)) / final_curv;
        // Apply rotation to get global dx, dy
        agent->rear_x += dx_l * agent->cos_heading - dy_l * agent->sin_heading;
        agent->rear_y += dx_l * agent->sin_heading + dy_l * agent->cos_heading;
        agent->sim_heading = normalize_heading(agent->sim_heading + theta);
        speed = v_new;
        yaw_rate = speed * final_curv;
    }
    // Vehicle dynamics state update
    agent->steering_angle = steering;
    agent->jerk_long = (accel_long_new - agent->accel_long) / env->dt;
    agent->jerk_lat = (accel_lat_new - agent->accel_lat) / env->dt;
    agent->accel_long = accel_long_new;
    agent->accel_lat = accel_lat_new;
    agent->yaw_rate = yaw_rate;
    agent->cos_heading = cosf(agent->sim_heading);
    agent->sin_heading = sinf(agent->sim_heading);
    agent->rear_vx = speed * agent->cos_heading;
    agent->rear_vy = speed * agent->sin_heading;
    update_agent_center_from_rear(agent);
    update_agent_speed(agent);
    update_agent_z(env, agent);
}

void c_reset(Drive *env) {
    env->timestep = env->init_step;

    if (env->simulation_mode == SIMULATION_REPLAY) {
        set_agent_at_init_log_step(env);
    } else if (env->simulation_mode == SIMULATION_GIGAFLOW) {
        generate_traffic_light_states(env);
        spawn_active_agents(env);
    }

    for (int i = 0; i < env->num_agents; i++) {
        env->logs[i] = (Log) {0};
        Agent *agent = &env->agents[i];

        if (env->simulation_mode == SIMULATION_GIGAFLOW && agent->removed) {
            continue;
        }
        reset_agent_state(agent);
        build_path(env, agent);
        if (!compute_goals(env, agent)) {
            continue;
        }
        compute_metrics(env, i);
    }
    compute_observations(env);
}

void c_step(Drive *env) {
    memset(env->rewards, 0, env->num_agents * sizeof(float));
    memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));
    memset(env->truncations, 0, env->num_agents * sizeof(unsigned char));
    memset(env->masks, 0, env->num_agents * sizeof(unsigned char));
    env->timestep++;

    for (int i = 0; i < env->num_agents; i++) {
        Agent *a = &env->agents[i];
        env->masks[i] = !(a->stopped || a->removed);
    }

    for (int i = 0; i < env->num_moving_log_agents; i++) {
        move_expert(env, env->num_agents + i);
    }
    for (int i = 0; i < env->num_agents; i++) {
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
        if (env->replay_expert_actions) {
            move_expert(env, i);
        } else {
            move_dynamics(env, i);
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        if (env->agents[i].stopped || env->agents[i].removed) {
            continue;
        }
        compute_metrics(env, i);
        compute_rewards(env, i);
    }

    int count_inactive = 0;
    for (int i = 0; i < env->num_agents; i++) {
        if (env->agents[i].removed || env->agents[i].stopped) {
            env->terminals[i] = 1;
            count_inactive++;
        }
    }
    if (env->timestep == env->episode_length
        || (env->termination_mode == 1 && (float) count_inactive / env->num_agents > env->inactive_agent_threshold)) {
        for (int i = 0; i < env->num_agents; i++) {
            env->truncations[i] = 1;
        }
        add_log(env);
        c_reset(env);
        return;
    }

    for (int i = 0; i < env->num_agents; i++) {
        Agent *agent = &env->agents[i];
        if (agent->metrics_array[REACH_GOAL_IDX] <= 0.0f) {
            continue;
        }
        agent->current_goal_idx++;
        if (agent->current_goal_idx == env->num_target_waypoints) {
            env->logs[i].num_goals_reached += 1;
            if (!compute_goals(env, agent)) {
                env->truncations[i] = 1;
            }
            continue;
        }
        agent->goal_position_x = agent->goal_positions_x[agent->current_goal_idx];
        agent->goal_position_y = agent->goal_positions_y[agent->current_goal_idx];
        agent->goal_position_z = agent->goal_positions_z[agent->current_goal_idx];
    }

    compute_observations(env);
}

#include "render.h"

#endif // SIM_DRIVE_H
