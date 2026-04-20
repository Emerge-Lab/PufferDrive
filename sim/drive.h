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

// Lane selection scoring
#define LANE_SELECTION_DISTANCE_WEIGHT 0.7f
#define LANE_SELECTION_HEADING_WEIGHT 0.3f
#define LANE_DISTANCE_NORMALIZATION 4.0f
#define LANE_SWITCH_THRESHOLD 0.05f // Hysteresis: new lane must be 5% better to switch
#define LANE_ALIGN_COS_THRESHOLD 0.5f
#define MAX_CHECKED_LANES 32

// Collision State
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2

// Grid Map
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL 30
#define VISION_RANGE 21

// Collision/Infraction behaviors
#define STOP_AGENT 1
#define REMOVE_AGENT 2

// Observation Space
#define EGO_FEATURES 7
#define PARTNER_FEATURES 8
#define ROAD_FEATURES 7
#define TRAFFIC_CONTROL_FEATURES 0
#define OBS_PARTNER_SLOTS 20
#define OBS_LANE_SLOTS 40
#define OBS_BOUNDARY_SLOTS 40
#define OBS_TRAFFIC_CONTROL_SLOTS 0
#define OBS_COUNT_FEATURES 4

#define OBS_SIZE                                                                                                       \
    (EGO_FEATURES + (PARTNER_FEATURES * OBS_PARTNER_SLOTS) + (ROAD_FEATURES * OBS_LANE_SLOTS)                          \
     + (ROAD_FEATURES * OBS_BOUNDARY_SLOTS) + (TRAFFIC_CONTROL_FEATURES * OBS_TRAFFIC_CONTROL_SLOTS)                   \
     + OBS_COUNT_FEATURES)

// Observation normalization
#define MAX_SPEED 100.0f
#define MAX_VEH_LEN 30.0f
#define MAX_VEH_WIDTH 15.0f
#define MAX_VEH_HEIGHT 10.0f
#define MAX_ROAD_SCALE 100.0f
#define MAX_ROAD_SEGMENT_LENGTH 100.0f

// Observation scaling factors
#define OBS_GOAL_SCALE 0.005f
#define OBS_SPEED_SCALE 0.01f
#define OBS_POSITION_SCALE 0.02f

// Distance thresholds
#define COLLISION_DIST_SQ (15.0f * 15.0f)
#define OBS_DIST_SQ (100.0f * 100.0f)

// 2.5D Z estimation
#define Z_BUFFER 4.0f
#define Z_NUM_PT_AVG 30

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

typedef struct {
    float z_dis;
    float euclidean_dis;
    float z;
} DepthPoint;

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
    float completion_rate;
    float dnf_rate;
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
    int *neighbor_cache_offsets;
    int *grid_index_drivable;
    int num_drivable_grid_cell;
    GridMapEntity **cells;
    GridMapEntity *neighbor_cache_entities;
};

struct Drive {
    Client *client;
    Log log;
    Log *logs;
    // Rollout buffers
    float *observations;
    float *actions;
    float *rewards;
    float *terminals;
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
    // Reward coefficients
    float reward_collision;
    float reward_offroad;
    float reward_goal_post_respawn;
    float reward_collision_post_respawn;
    // Infraction behaviors (0=none, STOP_AGENT, REMOVE_AGENT)
    int collision_behavior;
    int offroad_behavior;
    unsigned int rng;
    // Metadata fields
    char scenario_id[128];
    char dataset_name[32];
    char *map_name;
    int log_length;
    float log_dt;
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

// Normalize heading to [-pi, pi]
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

static inline void update_agent_rear_from_center(Agent *agent) {
    float rear_offset = 0.5f * agent->wheelbase;
    agent->rear_x = agent->sim_x - (rear_offset * agent->cos_heading);
    agent->rear_y = agent->sim_y - (rear_offset * agent->sin_heading);
}

static inline void update_agent_center_from_rear(Agent *agent) {
    float center_offset = 0.5f * agent->wheelbase;
    agent->sim_x = agent->rear_x + (center_offset * agent->cos_heading);
    agent->sim_y = agent->rear_y + (center_offset * agent->sin_heading);
}

static inline void zero_agent_velocity_state(Agent *agent) {
    agent->rear_vx = 0.0f;
    agent->rear_vy = 0.0f;
    agent->sim_vx = 0.0f;
    agent->sim_vy = 0.0f;
    agent->yaw_rate = 0.0f;
    agent->sim_speed = 0.0f;
    agent->sim_speed_signed = 0.0f;
    agent->a_long = 0.0f;
    agent->a_lat = 0.0f;
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
    agent->respawn_timestep = -1;
    agent->reached_goal = 0;
    agent->collided_before_goal = 0;
    agent->reached_goal_this_episode = 0;
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
}

static inline void update_agent_speed_from_center_velocity(Agent *agent) {
    float center_offset = 0.5f * agent->wheelbase;
    agent->rear_vx = agent->sim_vx + (agent->yaw_rate * center_offset * agent->sin_heading);
    agent->rear_vy = agent->sim_vy - (agent->yaw_rate * center_offset * agent->cos_heading);
    float speed = sqrtf(agent->rear_vx * agent->rear_vx + agent->rear_vy * agent->rear_vy);
    float v_dot_heading = agent->rear_vx * agent->cos_heading + agent->rear_vy * agent->sin_heading;
    agent->sim_speed = speed;
    agent->sim_speed_signed = copysignf(speed, v_dot_heading);
}

static inline void update_agent_speed_from_rear_velocity(Agent *agent) {
    float center_offset = 0.5f * agent->wheelbase;
    agent->sim_vx = agent->rear_vx - (agent->yaw_rate * center_offset * agent->sin_heading);
    agent->sim_vy = agent->rear_vy + (agent->yaw_rate * center_offset * agent->cos_heading);
    float speed = sqrtf(agent->rear_vx * agent->rear_vx + agent->rear_vy * agent->rear_vy);
    float v_dot_heading = agent->rear_vx * agent->cos_heading + agent->rear_vy * agent->sin_heading;
    agent->sim_speed = speed;
    agent->sim_speed_signed = copysignf(speed, v_dot_heading);
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

static inline void initialize_agent_dynamics_from_log_step(Agent *agent, int step, float dt) {
    assert(step >= 0);
    assert(step < agent->trajectory_length);

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
    agent->wheelbase = 0.6f * agent->sim_length;
    agent->steering_angle = 0.0f;
    agent->a_long = 0.0f;
    agent->a_lat = 0.0f;
    agent->jerk_long = 0.0f;
    agent->jerk_lat = 0.0f;
    update_agent_rear_from_center(agent);

    if (agent->control_state == CONTROL_STATE_STATIC) {
        zero_agent_velocity_state(agent);
        return;
    }

    agent->yaw_rate = compute_log_yaw_rate(agent, step, dt);
    agent->sim_vx = agent->log_velocity_x[step];
    agent->sim_vy = agent->log_velocity_y[step];
    update_agent_speed_from_center_velocity(agent);
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
    env->grid_map = (GridMap *) calloc(1, sizeof(GridMap));

    float top_left_x = 0.0f;
    float top_left_y = 0.0f;
    float bottom_right_x = 0.0f;
    float bottom_right_y = 0.0f;
    bool first_valid_point = false;
    for (int i = 0; i < env->num_road_elements; i++) {
        if (!is_road(env->road_elements[i].type)) {
            continue;
        }
        RoadMapElement *element = &env->road_elements[i];
        for (int j = 0; j < element->segment_length; j++) {
            if (element->x[j] == INVALID_POSITION) {
                continue;
            }
            if (element->y[j] == INVALID_POSITION) {
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

    int *cell_entities_insert_index = (int *) calloc(grid_cell_count, sizeof(int));

    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        int count = env->grid_map->cell_entities_count[grid_index];
        env->grid_map->cells[grid_index] = (GridMapEntity *) calloc(count, sizeof(GridMapEntity));
    }

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

    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0, y = 0, dir = 0;
    int steps_to_take = 1, steps_taken = 0, segments_completed = 0;
    int total = 0, max_offsets = vr * vr;
    int curr_idx = 0;
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
    int vr = env->grid_map->vision_range;
    int cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->neighbor_cache_count = (int *) calloc(cell_count, sizeof(int));
    env->grid_map->neighbor_cache_offsets = (int *) calloc(cell_count + 1, sizeof(int));

    int total = 0;
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols;
        int cell_y = i / env->grid_map->grid_cols;
        int cell_total = 0;
        for (int j = 0; j < vr * vr; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows) {
                continue;
            }
            cell_total += env->grid_map->cell_entities_count[env->grid_map->grid_cols * y + x];
        }
        env->grid_map->neighbor_cache_count[i] = cell_total;
        env->grid_map->neighbor_cache_offsets[i] = total;
        total += cell_total;
    }
    env->grid_map->neighbor_cache_offsets[cell_count] = total;
    env->grid_map->neighbor_cache_entities = (GridMapEntity *) calloc(total, sizeof(GridMapEntity));

    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols;
        int cell_y = i / env->grid_map->grid_cols;
        int write_idx = env->grid_map->neighbor_cache_offsets[i];
        for (int j = 0; j < vr * vr; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows) {
                continue;
            }
            int grid_index = env->grid_map->grid_cols * y + x;
            int grid_count = env->grid_map->cell_entities_count[grid_index];
            if (grid_count == 0) {
                continue;
            }
            memcpy(
                &env->grid_map->neighbor_cache_entities[write_idx],
                env->grid_map->cells[grid_index],
                grid_count * sizeof(GridMapEntity));
            write_idx += grid_count;
        }
    }
}

static int get_neighbor_cache_entities(Drive *env, int cell_idx, GridMapEntity *entities, int max_entities) {
    if (cell_idx < 0 || cell_idx >= (env->grid_map->grid_cols * env->grid_map->grid_rows)) {
        return 0;
    }
    int count = env->grid_map->neighbor_cache_count[cell_idx];
    if (count > max_entities) {
        count = max_entities;
    }
    if (count == 0) {
        return 0;
    }
    int offset = env->grid_map->neighbor_cache_offsets[cell_idx];
    memcpy(entities, &env->grid_map->neighbor_cache_entities[offset], count * sizeof(GridMapEntity));
    return count;
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
    int cellsX = env->grid_map->grid_cols;
    int gridX = index % cellsX;
    int gridY = index / cellsX;
    int entity_list_count = 0;

    for (int i = 0; i < offset_size; i++) {
        int nx = gridX + local_offsets[i][0];
        int ny = gridY + local_offsets[i][1];
        if (nx < 0 || nx >= env->grid_map->grid_cols || ny < 0 || ny >= env->grid_map->grid_rows) {
            continue;
        }
        int neighbor_idx = ny * env->grid_map->grid_cols + nx;
        int count = env->grid_map->cell_entities_count[neighbor_idx];
        for (int j = 0; j < count && entity_list_count < max_size; j++) {
            entity_list[entity_list_count] = env->grid_map->cells[neighbor_idx][j];
            entity_list_count++;
        }
    }
    return entity_list_count;
}

// ========================================
// Road Utility Functions
// ========================================

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
    if (agent->sim_x == INVALID_POSITION || agent->removed) {
        return -1;
    }

    int car_collided_with_index = -1;
    // Linear over all actors; quick-check prunes at COLLISION_QUICK_CHECK_DIST.
    for (int i = 0; i < env->num_sim_agents; i++) {
        Agent *other_agent = &env->agents[i];
        if (agent == other_agent) {
            continue;
        }
        if (other_agent->sim_x == INVALID_POSITION || other_agent->removed) {
            continue;
        }

        float ddx = other_agent->sim_x - agent->sim_x;
        float ddy = other_agent->sim_y - agent->sim_y;
        if (ddx * ddx + ddy * ddy > COLLISION_DIST_SQ) {
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
    for (int i = 0; i < env->num_agents; i++) {
        Agent *agent = &env->agents[i];
        if (agent->reached_goal_this_episode) {
            env->log.completion_rate += 1.0f;
        }
        int offroad = env->logs[i].offroad_rate;
        env->log.offroad_rate += env->logs[i].offroad_rate;
        int collided = env->logs[i].collision_rate;
        env->log.collision_rate += collided;
        if (agent->reached_goal_this_episode && !agent->collided_before_goal) {
            env->log.score += 1.0f;
        }
        if (!offroad && !collided && !agent->reached_goal_this_episode) {
            env->log.dnf_rate += 1.0f;
        }
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        env->log.n += 1;
    }
}

// ========================================
// Initialization Functions
// ========================================

static void set_agent_at_init_log_step(Drive *env) {
    assert(env->init_step >= 0);
    assert(env->init_step < env->log_length);
    for (int i = 0; i < env->num_sim_agents; i++) {
        Agent *agent = &env->agents[i];
        int step = env->init_step;
        initialize_agent_dynamics_from_log_step(agent, step, env->dt);
        agent->collision_state = NO_COLLISION;
        agent->respawn_timestep = -1;
        agent->reached_goal = 0;
        agent->collided_before_goal = 0;
    }
}

static bool is_valid_agent(const Agent *agent) {
    return agent->log_valid[0] == 1 && agent->type <= CYCLIST && agent->type != UNKNOWN;
}

void compact_agents(Drive *env) {
    env->num_agents = 0;
    env->num_moving_log_agents = 0;
    int num_valid_agents = 0;

    if (env->num_sim_agents == 0) {
        free(env->agents);
        env->agents = NULL;
        return;
    }

    Agent *raw_agents = env->agents;
    Agent *compacted_agents = (Agent *) calloc(env->num_sim_agents, sizeof(Agent));
    int *kept = (int *) calloc(env->num_sim_agents, sizeof(int));
    int max_active = env->num_max_agents == 0 ? env->num_sim_agents : env->num_max_agents;

    // First pass: collect active agents up to the max_active limit
    for (int i = 0; i < env->num_sim_agents; i++) {
        Agent *agent = &raw_agents[i];
        if (!is_valid_agent(agent)) {
            continue;
        }
        if (agent->control_state != CONTROL_STATE_ACTIVE || env->num_agents >= max_active) {
            continue;
        }
        compacted_agents[num_valid_agents++] = *agent;
        kept[i] = 1;
        env->num_agents++;
    }

    // Second pass: collect moving-log agents
    for (int i = 0; i < env->num_sim_agents; i++) {
        Agent *agent = &raw_agents[i];
        if (!is_valid_agent(agent) || kept[i] || agent->control_state != CONTROL_STATE_MOVING) {
            continue;
        }
        compacted_agents[num_valid_agents++] = *agent;
        kept[i] = 1;
        env->num_moving_log_agents++;
    }

    // Third pass: collect remaining valid log agents
    for (int i = 0; i < env->num_sim_agents; i++) {
        Agent *agent = &raw_agents[i];
        if (!is_valid_agent(agent) || kept[i]) {
            continue;
        }
        compacted_agents[num_valid_agents++] = *agent;
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

    if (num_valid_agents == 0) {
        free(compacted_agents);
        env->agents = NULL;
        return;
    }

    env->agents = (Agent *) realloc(compacted_agents, num_valid_agents * sizeof(Agent));
    env->num_sim_agents = num_valid_agents;
}

static void move_expert(Drive *env, int agent_idx) {
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
    agent->yaw_rate = compute_log_yaw_rate(agent, t, env->dt);
    agent->sim_vx = agent->log_velocity_x[t];
    agent->sim_vy = agent->log_velocity_y[t];
    update_agent_rear_from_center(agent);
    update_agent_speed_from_center_velocity(agent);
}

static void remove_bad_trajectories(Drive *env) {
    set_agent_at_init_log_step(env);
    int collided_agents[env->num_agents];
    int collided_with_indices[env->num_agents];
    memset(collided_agents, 0, env->num_agents * sizeof(int));
    for (int i = 0; i < env->num_agents; ++i) {
        collided_with_indices[i] = -1;
    }

    for (int t = 0; t < env->log_length; t++) {
        for (int i = 0; i < env->num_agents; i++) {
            move_expert(env, i);
        }
        for (int i = 0; i < env->num_moving_log_agents; i++) {
            int expert_idx = env->num_agents + i;
            if (env->agents[expert_idx].sim_x == INVALID_POSITION) {
                continue;
            }
            move_expert(env, expert_idx);
        }
        for (int i = 0; i < env->num_agents; i++) {
            Agent *agent = &env->agents[i];
            int collided_with_index = collision_check(env, agent);
            if ((collided_with_index >= 0) && collided_agents[i] == 0) {
                collided_agents[i] = 1;
                collided_with_indices[i] = collided_with_index;
            }
        }
        env->timestep++;
    }

    for (int i = 0; i < env->num_agents; i++) {
        int collided_with_index = collided_with_indices[i];
        // Layout after compact_agents is [active | log]; only invalidate log agents.
        if (collided_with_index < env->num_agents || collided_with_index == -1) {
            continue;
        }
        env->agents[collided_with_index].log_trajectory_x[0] = INVALID_POSITION;
        env->agents[collided_with_index].log_trajectory_y[0] = INVALID_POSITION;
    }
    env->timestep = 0;
}

int init(Drive *env) {
    env->timestep = 0;
    if (load_map_binary(env->map_name, env) != 0) {
        return -1;
    }
    init_grid_map(env);
    env->grid_map->vision_range = VISION_RANGE;
    init_neighbor_offsets(env);
    cache_neighbor_offsets(env);
    compact_agents(env);
    remove_bad_trajectories(env);
    set_agent_at_init_log_step(env);
    env->logs = (Log *) calloc(env->num_agents, sizeof(Log));
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
        if (env->grid_map->cells) {
            for (int i = 0; i < grid_cell_count; i++) {
                free(env->grid_map->cells[i]);
            }
            free(env->grid_map->cells);
        }
        free(env->grid_map->cell_entities_count);
        free(env->grid_map->grid_index_drivable);
        free(env->grid_map->neighbor_cache_entities);
        free(env->grid_map->neighbor_cache_count);
        free(env->grid_map->neighbor_cache_offsets);
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
    env->observations = (float *) calloc(env->num_agents * OBS_SIZE, sizeof(float));
    env->actions = (float *) calloc(env->num_agents * 2, sizeof(float));
    env->rewards = (float *) calloc(env->num_agents, sizeof(float));
    env->terminals = (float *) calloc(env->num_agents, sizeof(float));
    return 0;
}

void free_allocated(Drive *env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
}

// ========================================
// Core Simulation Functions
// ========================================

void respawn_agent(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    initialize_agent_dynamics_from_log_step(agent, env->init_step, env->dt);
    agent->reached_goal = 0;
    agent->respawn_timestep = env->timestep;
}

static void compute_metrics(Drive *env, Agent *agent) {
    for (int i = 0; i < NUM_METRICS; i++) {
        agent->metrics_array[i] = 0.0f;
    }
    agent->collision_state = NO_COLLISION;

    if (agent->sim_x == INVALID_POSITION) {
        return;
    }

    bool is_offroad = false;

    // Track best candidate by combined distance/heading score
    float best_score = 1e9f;
    int lane_idx = -1;
    float signed_lane_distance = 0.0f, lane_heading = 0.0f;

    float corners[4][2];
    compute_agent_corners(agent, corners);

    static const int road_offsets[25][2]
        = {{-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {-2, -1}, {-1, -1}, {0, -1}, {1, -1},
           {2, -1},  {-2, 0},  {-1, 0}, {0, 0},  {1, 0},  {2, 0},   {-2, 1},  {-1, 1}, {0, 1},
           {1, 1},   {2, 1},   {-2, 2}, {-1, 2}, {0, 2},  {1, 2},   {2, 2}};

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL * 25];
    int list_size = get_neighbors_entities(
        env,
        agent->sim_x,
        agent->sim_y,
        entity_list,
        MAX_ENTITIES_PER_CELL * 25,
        road_offsets,
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

    // Early returns after offroad and collision enforce mutual exclusivity of terminal flags.
    if (is_offroad) {
        agent->collision_state = OFFROAD;
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        if (env->offroad_behavior == STOP_AGENT) {
            agent->stopped = 1;
        } else if (env->offroad_behavior == REMOVE_AGENT) {
            agent->removed = 1;
            invalidate_agent(agent);
        }
        return;
    }

    int car_collided_with_index = collision_check(env, agent);
    if (car_collided_with_index != -1
        && (agent->respawn_timestep != -1 || env->agents[car_collided_with_index].respawn_timestep != -1)) {
        car_collided_with_index = -1;
    }
    if (car_collided_with_index != -1) {
        agent->collision_state = VEHICLE_COLLISION;
        agent->metrics_array[COLLISION_IDX] = 1.0f;
        if (env->collision_behavior == STOP_AGENT) {
            agent->stopped = 1;
        } else if (env->collision_behavior == REMOVE_AGENT) {
            agent->removed = 1;
            invalidate_agent(agent);
        }
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
        env->rewards[i] += env->reward_collision;
        env->logs[i].episode_return += env->reward_collision;
        env->logs[i].collision_rate = 1.0f;
    }

    // Offroad reward
    if (agent->metrics_array[OFFROAD_IDX] > 0.0f) {
        env->rewards[i] += env->reward_offroad;
        env->logs[i].episode_return += env->reward_offroad;
        env->logs[i].offroad_rate = 1.0f;
    }

    if (agent->metrics_array[COLLISION_IDX] + agent->metrics_array[OFFROAD_IDX] > 0.0f) {
        if (!agent->reached_goal_this_episode) {
            agent->collided_before_goal = 1;
        }
    }

    // Goal reward
    if (agent->metrics_array[REACH_GOAL_IDX] > 0.0f) {
        if (agent->respawn_timestep != -1) {
            env->rewards[i] += env->reward_goal_post_respawn;
            env->logs[i].episode_return += env->reward_goal_post_respawn;
        } else {
            env->rewards[i] += 1.0f;
            env->logs[i].episode_return += 1.0f;
        }
        agent->reached_goal = 1;
        agent->reached_goal_this_episode = 1;
    }
}

void compute_observations(Drive *env) {
    memset(env->observations, 0, OBS_SIZE * env->num_agents * sizeof(float));
    float (*observations)[OBS_SIZE] = (float (*)[OBS_SIZE]) env->observations;
    for (int i = 0; i < env->num_agents; i++) {
        float *obs = &observations[i][0];
        Agent *ego = &env->agents[i];
        int obs_idx = 0;
        int partner_count = 0;
        int lane_count = 0;
        int boundary_count = 0;
        int traffic_control_count = 0;

        // ====== Ego observations ======
        float goal_x = ego->goal_position_x - ego->sim_x;
        float goal_y = ego->goal_position_y - ego->sim_y;
        float rel_goal_x = goal_x * ego->cos_heading + goal_y * ego->sin_heading;
        float rel_goal_y = -goal_x * ego->sin_heading + goal_y * ego->cos_heading;
        obs[obs_idx++] = rel_goal_x * OBS_GOAL_SCALE;
        obs[obs_idx++] = rel_goal_y * OBS_GOAL_SCALE;
        obs[obs_idx++] = sqrtf(ego->sim_vx * ego->sim_vx + ego->sim_vy * ego->sim_vy) * OBS_SPEED_SCALE;
        obs[obs_idx++] = ego->sim_width / MAX_VEH_WIDTH;
        obs[obs_idx++] = ego->sim_length / MAX_VEH_LEN;
        obs[obs_idx++] = (ego->collision_state > NO_COLLISION) ? 1 : 0;
        obs[obs_idx++] = (ego->respawn_timestep != -1) ? 1 : 0;

        // ====== Reward conditioning and target observations ======
        // To fill

        // ===== Partner observations =====
        typedef struct {
            int index;
            float dist_sq;
            float dx;
            float dy;
            float dz;
        } AgentDistance;
        AgentDistance candidates[env->num_sim_agents - 1];
        int candidate_count = 0;
        for (int j = 0; j < env->num_sim_agents; j++) {
            if (j == i) {
                continue;
            }
            Agent *other_entity = &env->agents[j];
            if (other_entity->respawn_timestep != -1) {
                continue;
            }
            float dx = other_entity->sim_x - ego->sim_x;
            float dy = other_entity->sim_y - ego->sim_y;
            float dz = other_entity->sim_z - ego->sim_z;
            float dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq > OBS_DIST_SQ || fabsf(dz) > Z_BUFFER) {
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
        if (candidate_count > 0) {
            int num_agents_to_observe = (candidate_count < OBS_PARTNER_SLOTS) ? candidate_count : OBS_PARTNER_SLOTS;

            // Partial selection sort: surface the k closest candidates, leave the rest unordered.
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
                float dx = candidates[j].dx;
                float dy = candidates[j].dy;
                float dz = candidates[j].dz;
                float rel_x = dx * ego->cos_heading + dy * ego->sin_heading;
                float rel_y = -dx * ego->sin_heading + dy * ego->cos_heading;
                float rel_heading_x = other->cos_heading * ego->cos_heading + other->sin_heading * ego->sin_heading;
                float rel_heading_y = other->sin_heading * ego->cos_heading - other->cos_heading * ego->sin_heading;

                obs[obs_idx++] = rel_x * OBS_POSITION_SCALE;
                obs[obs_idx++] = rel_y * OBS_POSITION_SCALE;
                obs[obs_idx++] = dz / Z_BUFFER;
                obs[obs_idx++] = other->sim_width / MAX_VEH_WIDTH;
                obs[obs_idx++] = other->sim_length / MAX_VEH_LEN;
                obs[obs_idx++] = rel_heading_x;
                obs[obs_idx++] = rel_heading_y;
                float other_speed = sqrtf(other->sim_vx * other->sim_vx + other->sim_vy * other->sim_vy);
                obs[obs_idx++] = other_speed / MAX_SPEED;
                cars_seen++;
            }
        }
        partner_count = cars_seen;
        int remaining_partner_obs = (OBS_PARTNER_SLOTS - cars_seen) * PARTNER_FEATURES;
        memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
        obs_idx += remaining_partner_obs;

        // ===== Road observations =====
        GridMapEntity entity_list[OBS_LANE_SLOTS + OBS_BOUNDARY_SLOTS];
        int grid_idx = get_grid_index(env, ego->sim_x, ego->sim_y);
        int list_size = get_neighbor_cache_entities(env, grid_idx, entity_list, OBS_LANE_SLOTS + OBS_BOUNDARY_SLOTS);

        int lane_obs_idx = obs_idx;
        int boundary_obs_idx = lane_obs_idx + OBS_LANE_SLOTS * ROAD_FEATURES;
        obs_idx = boundary_obs_idx + OBS_BOUNDARY_SLOTS * ROAD_FEATURES;

        float *lanes_dest = &obs[lane_obs_idx];
        float *boundaries_dest = &obs[boundary_obs_idx];
        int lanes_collected = 0;
        int boundaries_collected = 0;

        for (int k = 0; k < list_size; k++) {
            if (lanes_collected >= OBS_LANE_SLOTS && boundaries_collected >= OBS_BOUNDARY_SLOTS) {
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
            float rel_x = mid_x - ego->sim_x;
            float rel_y = mid_y - ego->sim_y;
            float rel_z = mid_z - ego->sim_z;
            float x_obs = rel_x * ego->cos_heading + rel_y * ego->sin_heading;
            float y_obs = -rel_x * ego->sin_heading + rel_y * ego->cos_heading;

            if (fabsf(rel_z) > Z_BUFFER) {
                continue;
            }

            float dx = end_x - mid_x;
            float dy = end_y - mid_y;
            float length = sqrtf(dx * dx + dy * dy);
            float dx_norm = (length > 0) ? dx / length : dx;
            float dy_norm = (length > 0) ? dy / length : dy;
            float cos_angle = dx_norm * ego->cos_heading + dy_norm * ego->sin_heading;
            float sin_angle = -dx_norm * ego->sin_heading + dy_norm * ego->cos_heading;

            float *target;
            int *counter;
            int cap;
            if (is_lane) {
                target = lanes_dest;
                counter = &lanes_collected;
                cap = OBS_LANE_SLOTS;
            } else {
                target = boundaries_dest;
                counter = &boundaries_collected;
                cap = OBS_BOUNDARY_SLOTS;
            }
            if (*counter >= cap) {
                continue;
            }
            int base = (*counter)++ * ROAD_FEATURES;
            target[base] = x_obs * OBS_POSITION_SCALE;
            target[base + 1] = y_obs * OBS_POSITION_SCALE;
            target[base + 2] = rel_z / Z_BUFFER;
            target[base + 3] = length / MAX_ROAD_SEGMENT_LENGTH;
            target[base + 4] = 0.1f / MAX_ROAD_SCALE;
            target[base + 5] = cos_angle;
            target[base + 6] = sin_angle;
        }

        lane_count = lanes_collected;
        boundary_count = boundaries_collected;
        memset(
            &obs[lane_obs_idx + lanes_collected * ROAD_FEATURES],
            0,
            (OBS_LANE_SLOTS - lanes_collected) * ROAD_FEATURES * sizeof(float));
        memset(
            &obs[boundary_obs_idx + boundaries_collected * ROAD_FEATURES],
            0,
            (OBS_BOUNDARY_SLOTS - boundaries_collected) * ROAD_FEATURES * sizeof(float));

        // ===== Traffic control observations =====
        typedef struct {
            int idx;
            float dist_sq;
        } TrafficControlDist;
        TrafficControlDist traffic_controls[env->num_traffic_elements > 0 ? env->num_traffic_elements : 1];
        int num_visible_controls = 0;

        // Collect traffic controls within range
        for (int j = 0; j < env->num_traffic_elements; j++) {
            TrafficControlElement *traffic = &env->traffic_elements[j];
            // if (!traffic_control_in_scope(traffic->type, env->traffic_control_scope)) {
            //     continue;
            // }

            float mid_x = (traffic->stop_line[0] + traffic->stop_line[3]) * 0.5f;
            float mid_y = (traffic->stop_line[1] + traffic->stop_line[4]) * 0.5f;
            float mid_z = (traffic->stop_line[2] + traffic->stop_line[5]) * 0.5f;
            float dx = mid_x - ego->sim_x;
            float dy = mid_y - ego->sim_y;
            float dz = mid_z - ego->sim_z;
            float abs_dz = fabsf(dz);
            float dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq > OBS_DIST_SQ || abs_dz > Z_BUFFER) {
                continue;
            }

            traffic_controls[num_visible_controls].idx = j;
            traffic_controls[num_visible_controls].dist_sq = dist_sq;
            num_visible_controls++;
        }

        // Partial selection sort: find K closest (O(N*K))
        // int num_controls_to_observe = (num_visible_controls < env->max_traffic_control_observations)
        //     ? num_visible_controls
        //     : env->max_traffic_control_observations;
        int num_controls_to_observe = 0;
        int max_traffic_control_observations = OBS_TRAFFIC_CONTROL_SLOTS;
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

        // Add observations for closest traffic controls
        int controls_added = 0;
        for (int j = 0; j < num_controls_to_observe && controls_added < max_traffic_control_observations; j++) {
            TrafficControlElement *traffic = &env->traffic_elements[traffic_controls[j].idx];
            // Stop line start point
            float dx1 = traffic->stop_line[0] - ego->sim_x;
            float dy1 = traffic->stop_line[1] - ego->sim_y;
            float rel_x1 = dx1 * ego->cos_heading + dy1 * ego->sin_heading;
            float rel_y1 = -dx1 * ego->sin_heading + dy1 * ego->cos_heading;
            // Stop line end point
            float dx2 = traffic->stop_line[3] - ego->sim_x;
            float dy2 = traffic->stop_line[4] - ego->sim_y;
            float rel_x2 = dx2 * ego->cos_heading + dy2 * ego->sin_heading;
            float rel_y2 = -dx2 * ego->sin_heading + dy2 * ego->cos_heading;
            float rel_z = (traffic->stop_line[2] + traffic->stop_line[5]) * 0.5f - ego->sim_z;

            int state;
            if (traffic->type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
                if (env->timestep >= 0 && env->timestep < traffic->state_length && traffic->states != NULL) {
                    state = traffic->states[env->timestep];
                } else {
                    state = TRAFFIC_CONTROL_STATE_OFF;
                }
            } else {
                state = TRAFFIC_CONTROL_STATE_UNKNOWN;
            }

            obs[obs_idx++] = rel_x1 * OBS_POSITION_SCALE;
            obs[obs_idx++] = rel_y1 * OBS_POSITION_SCALE;
            obs[obs_idx++] = rel_x2 * OBS_POSITION_SCALE;
            obs[obs_idx++] = rel_y2 * OBS_POSITION_SCALE;
            obs[obs_idx++] = rel_z / Z_BUFFER;
            obs[obs_idx++] = traffic->type;
            obs[obs_idx++] = state;
            controls_added++;
        }

        // Zero out remaining traffic control slots
        int remaining_traffic_obs = (max_traffic_control_observations - controls_added) * TRAFFIC_CONTROL_FEATURES;
        memset(&obs[obs_idx], 0, remaining_traffic_obs * sizeof(float));
        obs_idx += remaining_traffic_obs;
        traffic_control_count = controls_added;

        obs[obs_idx++] = (float) lane_count;
        obs[obs_idx++] = (float) boundary_count;
        obs[obs_idx++] = (float) partner_count;
        obs[obs_idx++] = (float) traffic_control_count;
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
    float a_long_new = 0.0f, a_lat_new = 0.0f;

    if (env->dynamics_model == CLASSIC) {
        float accel = 0.0f, steer_req = 0.0f;

        if (env->action_type == DISCRETE) { // Discrete
            int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
            int val = (int) env->actions[agent_idx];
            assert(val >= 0);
            assert(val < action_dim_classic_discrete());
            accel = ACCELERATION_VALUES[val / num_steer];
            steer_req = STEERING_VALUES[val % num_steer];
        } else if (env->action_type == CONTINUOUS) { // Continuous
            float (*act)[2] = (float (*)[2]) env->actions;
            accel = act[agent_idx][0] * ACCELERATION_VALUES[6];
            steer_req = act[agent_idx][1] * STEERING_VALUES[8];
        }

        // Steering rate and position limits
        float delta_steer = clip(steer_req - steering, -0.6f * env->dt, 0.6f * env->dt);
        steering = clip(steering + delta_steer, -0.667f, 0.667f);
        // Update state
        speed = clip(speed + accel * env->dt, -MAX_SPEED, MAX_SPEED);
        float yaw_rate = (speed * tanf(steering)) / agent->wheelbase;
        // Simple bicycle model kinematics
        agent->rear_x += speed * agent->cos_heading * env->dt;
        agent->rear_y += speed * agent->sin_heading * env->dt;
        agent->sim_heading = normalize_heading(agent->sim_heading + yaw_rate * env->dt);
        // Accelerations update (not used in kinematics)
        a_long_new = accel;
        a_lat_new = speed * yaw_rate;

    } else if (env->dynamics_model == JERK) {
        float j_long = 0.0f, j_lat = 0.0f;

        if (env->action_type == DISCRETE) { // Discrete
            int num_lat = sizeof(JERK_LAT) / sizeof(JERK_LAT[0]);
            int val = (int) env->actions[agent_idx];
            assert(val >= 0);
            assert(val < action_dim_jerk_discrete());
            j_long = JERK_LONG[val / num_lat];
            j_lat = JERK_LAT[val % num_lat];
        } else if (env->action_type == CONTINUOUS) { // Continuous
            float (*act)[2] = (float (*)[2]) env->actions;
            float jl_act = act[agent_idx][0];
            j_long = (jl_act < 0) ? jl_act * (-JERK_LONG[0]) : jl_act * JERK_LONG[3];
            j_lat = act[agent_idx][1] * JERK_LAT[2];
        }

        // Integrate jerk to get new accelerations
        a_long_new = agent->a_long + (j_long * env->dt);
        a_long_new = (agent->a_long * a_long_new < 0) ? 0.0f : clip(a_long_new, -5.0f, 2.5f);
        // Lateral acceleration update with steering influence
        a_lat_new = agent->a_lat + (j_lat * env->dt);
        a_lat_new = (agent->a_lat * a_lat_new < 0) ? 0.0f : clip(a_lat_new, -4.0f, 4.0f);
        // Velocity (Trapezoidal)
        float v_new = clip(speed + 0.5f * (a_long_new + agent->a_long) * env->dt, -2.0f, 20.0f);
        if (speed * v_new < 0) {
            v_new = 0.0f;
        }
        // Curvature and Steering
        float v_eff = fmaxf(fabsf(v_new), 1.0f);
        float curvature = a_lat_new / (v_eff * v_eff);
        float steer_req = atanf(curvature * agent->wheelbase);
        float delta_steer = clip(steer_req - steering, -0.6f * env->dt, 0.6f * env->dt);
        steering = clip(steering + delta_steer, -0.55f, 0.55f);
        // Recalculate based on limits
        float final_curv = tanf(steering) / agent->wheelbase;
        a_lat_new = v_new * v_new * final_curv;
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
    }
    // Vehicle dynamics state update
    agent->steering_angle = steering;
    agent->jerk_long = (a_long_new - agent->a_long) / env->dt;
    agent->jerk_lat = (a_lat_new - agent->a_lat) / env->dt;
    agent->a_long = a_long_new;
    agent->a_lat = a_lat_new;
    agent->cos_heading = cosf(agent->sim_heading);
    agent->sin_heading = sinf(agent->sim_heading);
    agent->rear_vx = speed * agent->cos_heading;
    agent->rear_vy = speed * agent->sin_heading;
    update_agent_center_from_rear(agent);
    update_agent_speed_from_rear_velocity(agent);
    update_agent_z(env, agent);
}

void c_reset(Drive *env) {
    env->timestep = env->init_step;
    set_agent_at_init_log_step(env);
    for (int i = 0; i < env->num_agents; i++) {
        env->logs[i] = (Log) {0};
        Agent *agent = &env->agents[i];
        reset_agent_state(agent);
        compute_metrics(env, agent);
    }
    compute_observations(env);
}

void c_step(Drive *env) {
    memset(env->rewards, 0, env->num_agents * sizeof(float));
    memset(env->terminals, 0, env->num_agents * sizeof(float));
    env->timestep++;

    if (env->timestep == env->episode_length) {
        add_log(env);
        c_reset(env);
        return;
    }

    for (int i = 0; i < env->num_moving_log_agents; i++) {
        move_expert(env, env->num_agents + i);
    }
    for (int i = 0; i < env->num_agents; i++) {
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
        env->agents[i].collision_state = NO_COLLISION;
        move_dynamics(env, i);
    }

    for (int i = 0; i < env->num_agents; i++) {
        Agent *agent = &env->agents[i];

        if (agent->stopped || agent->removed) {
            env->terminals[i] = 1;
            continue;
        }

        compute_metrics(env, agent);
        compute_rewards(env, i);

        if (agent->stopped || agent->removed) {
            env->terminals[i] = 1;
        }
        if (agent->reached_goal) {
            respawn_agent(env, i);
        }
    }

    compute_observations(env);
}

#include "render.h"

#endif // SIM_DRIVE_H
