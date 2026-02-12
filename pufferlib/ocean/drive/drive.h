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
#include "datatypes.h"

// templates for bringing in some datatypes we might use later
// if we do this we can have a transition phase in which we can start testing with
// the types from datatypes, without losing the ones defined in this file.
#define DT_STOP_SIGN STOP_SIGN
#define DT_CROSSWALK CROSSWALK

// remove the values from datatypes.h
#undef NONE
#undef VEHICLE
#undef PEDESTRIAN
#undef CYCLIST
#undef ROAD_LANE
#undef ROAD_LINE
#undef ROAD_EDGE
#undef STOP_SIGN
#undef CROSSWALK
#undef SPEED_BUMP
#undef DRIVEWAY

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

// Initialization modes
#define INIT_ALL_VALID 0
#define INIT_ONLY_CONTROLLABLE_AGENTS 1
#define RANDOM_AGENTS 2

// Control modes
#define CONTROL_VEHICLES 0
#define CONTROL_AGENTS 1
#define CONTROL_WOSAC 2
#define CONTROL_SDC_ONLY 3

// Minimum distance to goal position
#define MIN_DISTANCE_TO_GOAL 2.0f

// Actions
#define NOOP 0

// Dynamics Models
#define CLASSIC 0
#define JERK 1

// Collision state
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2

// Metrics array indices
#define COLLISION_IDX 0
#define OFFROAD_IDX 1
#define REACHED_GOAL_IDX 2
#define LANE_ALIGNED_IDX 3

// Grid cell size
#define GRID_CELL_SIZE 5.0f

// Observation constants
#define MAX_ROAD_SEGMENT_OBSERVATIONS 128
#ifndef MAX_AGENTS // TODO: Needs to be replaced with MAX_PARTNER_OBS(agents in obs_radius) throughout observations code
                   // and with env->max_agents_in_sim throughout all agent for loops
#define MAX_AGENTS 64
#endif
#define STOP_AGENT 1
#define REMOVE_AGENT 2

#define ROAD_FEATURES 8
#define ROAD_FEATURES_ONEHOT 14
#define PARTNER_FEATURES 8

// Ego features depend on dynamics model
#define EGO_FEATURES_CLASSIC 8
#define EGO_FEATURES_JERK 11

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
#define GOAL_RESPAWN 0
#define GOAL_GENERATE_NEW 1
#define GOAL_STOP 2

// Offsets
#define COLLISION_RANGE 5
#define Z_RANGE 3
#define Z_BUFFER 4.0f // 4.0m buffer for different z level checking

// Rejection sampling parameters
#define MAX_SPAWN_ATTEMPTS 30
#define MAX_SPAWNS_ATTEMPTS_WITH_DIMENSION_CHANGES 30

// Jerk action space (for JERK dynamics model)
static const float JERK_LONG[4] = {-15.0f, -4.0f, 0.0f, 4.0f};
static const float JERK_LAT[3] = {-4.0f, 0.0f, 4.0f};

// Classic action space (for CLASSIC dynamics model)
static const float ACCELERATION_VALUES[7] = {-4.0000f, -2.6670f, -1.3330f, -0.0000f, 1.3330f, 2.6670f, 4.0000f};
static const float STEERING_VALUES[13] = {-1.000f, -0.833f, -0.667f, -0.500f, -0.333f, -0.167f, 0.000f,
                                          0.167f,  0.333f,  0.500f,  0.667f,  0.833f,  1.000f};

static const float offsets[4][2] = {
    {-1, 1}, // top-left
    {1, 1},  // top-right
    {1, -1}, // bottom-right
    {-1, -1} // bottom-left
};

static inline void generate_offsets(int offsets[][2], int offset_range) {
    int half_grid = offset_range / 2;
    int left_most = -half_grid;
    int right_most = offset_range + left_most - 1;
    int index = 0;
    for (int dy = left_most; dy <= right_most; dy++) {
        for (int dx = left_most; dx <= right_most; dx++) {
            offsets[index][0] = dx;
            offsets[index][1] = dy;
            index++;
        }
    }
}

// Offset arrays
int collision_offsets[COLLISION_RANGE * COLLISION_RANGE][2] = {0};
int z_offsets[Z_RANGE * Z_RANGE][2] = {0};

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};
const Color LIGHTGREEN = (Color){152, 255, 152, 255};
const Color LIGHTYELLOW = (Color){255, 255, 152, 255};
const Color SOFT_YELLOW = (Color){245, 245, 220, 255};

struct timespec ts;

typedef struct AgentSpawnSettings AgentSpawnSettings;
typedef struct Drive Drive;
typedef struct Client Client;
typedef struct Log Log;
typedef struct Agent Agent;
typedef struct RoadMapElement RoadMapElement;
typedef struct TrafficControlElement TrafficControlElement;

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
    float n;
    float lane_alignment_rate;
    float speed_at_goal;
    float active_agent_count;
    float expert_static_agent_count;
    float static_agent_count;
};

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

struct AgentSpawnSettings {
    int max_agents_in_sim; // max number of agents in sim(max Agent struct objects allocated)
    float min_w;
    float max_w;
    float min_l;
    float max_l;
    float h;
};

struct Drive {
    Client *client;
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;
    Log log;
    Log *logs;
    int num_agents; // Max controlled agents
    int active_agent_count;
    int *active_agent_indices;
    int action_type;
    int human_agent_idx;
    Agent *agents;
    RoadMapElement *road_elements;
    int *road_scenario_ids;
    TrafficControlElement *traffic_elements;
    AgentSpawnSettings spawn_settings;
    int num_created_agents; // number of agents created in the sim
    int num_traffic_elements;
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
    float reward_vehicle_collision;
    float reward_offroad_collision;
    char *map_name;
    float world_mean_x;
    float world_mean_y;
    float world_mean_z;
    float dt;
    float reward_goal;
    float reward_goal_post_respawn;
    float goal_radius;
    float goal_speed;
    int logs_capacity;
    int goal_behavior;
    float goal_target_distance;
    char *ini_file;
    char *scenario_id;
    int collision_behavior;
    int offroad_behavior;
    int sdc_track_index;
    int num_tracks_to_predict;
    int *tracks_to_predict_indices;
    int init_mode;
    int control_mode;
};

// ========================================
// Forward declaration placeholders
// ========================================

void move_expert(Drive *env, float *actions, int agent_idx);
float point_to_segment_distance_2d(float px, float py, float x1, float y1, float x2, float y2);
void init_goal_positions(Drive *env);
float clipSpeed(float speed);
void sample_new_goal(Drive *env, int agent_idx);
int check_lane_aligned(Agent *car, RoadMapElement *lane, int geometry_idx);

// ========================================
// Utility Functions
// ========================================

// rename to: compare_agent_dist
float relative_distance(float a, float b) {
    float distance = sqrtf(powf(a - b, 2));
    return distance;
}

// NOTE: Valentin renamed to compute_euclidean_distance, we will have to re-think
float relative_distance_3d(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    float distance = sqrtf(dx * dx + dy * dy + dz * dz);
    return distance;
}

float clip(float value, float min, float max) {
    if (value < min)
        return min;
    if (value > max)
        return max;
    return value;
}

float normalize_heading(float heading) {
    if (heading > M_PI)
        heading -= 2 * M_PI;
    if (heading < -M_PI)
        heading += 2 * M_PI;
    return heading;
}

// Note: added for 2.5D
typedef struct {
    float dis;
    float z;
} DepthPoint;

// Note: added for 2.5D
DepthPoint compute_z_distance_to_road_segment(Agent *agent, RoadMapElement *lane, int geomtery_idx) {
    float agent_position_z = agent->sim_z;
    float road_z = lane->z[geomtery_idx];
    float dis = fabsf(road_z - agent_position_z); // Start with vertical distance
    DepthPoint point;
    point.dis = dis;
    point.z = road_z;
    return point;
}

// Note: added for 2.5D
int compare_depthpoint(const void *a, const void *b) {
    float diff = ((DepthPoint *)a)->dis - ((DepthPoint *)b)->dis;
    return (diff > 0) - (diff < 0); // returns 1, 0, or -1
}

// void compute_heading_diff(void){}

static float random_uniform(float min_val, float max_val) {
    float scale = (float)rand() / (float)RAND_MAX;
    return min_val + scale * (max_val - min_val);
}

// void mixed_uniform(void){}

// void generate_reward_coefs(void){}

// void normalize_reward_coef(void){}

// void find_lane_index_by_id(void){}

void set_means(Drive *env) {
    float mean_x = 0.0f;
    float mean_y = 0.0f;
    float mean_z = 0.0f;
    int64_t point_count = 0;

    // Compute single mean for all agents and road elements
    if (env->init_mode != RANDOM_AGENTS) {
        for (int i = 0; i < env->num_objects; i++) {
            Agent *agent = &env->agents[i];
            for (int j = 0; j < agent->trajectory_length; j++) {
                if (agent->log_valid && agent->log_valid[j]) {
                    point_count++;
                    mean_x += (agent->log_trajectory_x[j] - mean_x) / point_count;
                    mean_y += (agent->log_trajectory_y[j] - mean_y) / point_count;
                    mean_z += (agent->log_trajectory_z[j] - mean_z) / point_count;
                }
            }
        }
    }
    for (int i = 0; i < env->num_roads; i++) {
        RoadMapElement *road = &env->road_elements[i];
        for (int j = 0; j < road->segment_length; j++) {
            point_count++;
            mean_x += (road->x[j] - mean_x) / point_count;
            mean_y += (road->y[j] - mean_y) / point_count;
            mean_z += (road->z[j] - mean_z) / point_count;
        }
    }
    env->world_mean_x = mean_x;
    env->world_mean_y = mean_y;
    env->world_mean_z = mean_z;
    if (env->init_mode != RANDOM_AGENTS) {
        for (int i = 0; i < env->num_objects; i++) {
            Agent *agent = &env->agents[i];
            for (int j = 0; j < agent->trajectory_length; j++) {
                if (agent->log_trajectory_x[j] == INVALID_POSITION)
                    continue;
                agent->log_trajectory_x[j] -= mean_x;
                agent->log_trajectory_y[j] -= mean_y;
                agent->log_trajectory_z[j] -= mean_z;
            }
            agent->goal_position_x -= mean_x;
            agent->goal_position_y -= mean_y;
            agent->goal_position_z -= mean_z;
        }
    }
    for (int i = 0; i < env->num_roads; i++) {
        RoadMapElement *road = &env->road_elements[i];
        for (int j = 0; j < road->segment_length; j++) {
            if (road->x[j] == INVALID_POSITION)
                continue;
            road->x[j] -= mean_x;
            road->y[j] -= mean_y;
            road->z[j] -= mean_z;
        }
    }
}

// ========================================
// Grid Map Functions
// ========================================

// rename to: get_grid_index
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
    for (int i = 0; i < env->num_roads; i++) {
        RoadMapElement *road = &env->road_elements[i];
        if (road->type == ROAD_LANE || road->type == ROAD_LINE || road->type == ROAD_EDGE) {
            for (int j = 0; j < road->segment_length; j++) {
                if (road->x[j] == INVALID_POSITION)
                    continue;
                if (road->y[j] == INVALID_POSITION)
                    continue;
                if (!first_valid_point) {
                    top_left_x = bottom_right_x = road->x[j];
                    top_left_y = bottom_right_y = road->y[j];
                    first_valid_point = true;
                    continue;
                }
                if (road->x[j] < top_left_x)
                    top_left_x = road->x[j];
                if (road->x[j] > bottom_right_x)
                    bottom_right_x = road->x[j];
                if (road->y[j] > top_left_y)
                    top_left_y = road->y[j];
                if (road->y[j] < bottom_right_y)
                    bottom_right_y = road->y[j];
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
    for (int i = 0; i < env->num_roads; i++) {
        RoadMapElement *road = &env->road_elements[i];
        if (road->type == ROAD_LANE || road->type == ROAD_LINE || road->type == ROAD_EDGE) {
            for (int j = 0; j < road->segment_length - 1; j++) {
                float x_center = (road->x[j] + road->x[j + 1]) / 2;
                float y_center = (road->y[j] + road->y[j + 1]) / 2;
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
    for (int i = 0; i < env->num_roads; i++) {
        RoadMapElement *road = &env->road_elements[i];
        if (road->type == ROAD_LANE || road->type == ROAD_LINE ||
            road->type == ROAD_EDGE) { // NOTE: Only Road Edges, Lines, and Lanes in grid map
            for (int j = 0; j < road->segment_length - 1; j++) {
                float x_center = (road->x[j] + road->x[j + 1]) / 2;
                float y_center = (road->y[j] + road->y[j + 1]) / 2;
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

GridMapEntity *checkNeighbors(Drive *env, float x, float y, const int (*local_offsets)[2], int offset_size,
                              int *list_count) {
    int index = getGridIndex(env, x, y);
    if (index == -1)
        return NULL;

    int cellsX = env->grid_map->grid_cols;
    int gridX = index % cellsX;
    int gridY = index / cellsX;
    int entity_list_count = 0;

    // Calculate entities count in neighboring cells
    int min_neighbor_index = INT16_MAX;
    for (int i = 0; i < offset_size; i++) {
        int nx = gridX + local_offsets[i][0];
        int ny = gridY + local_offsets[i][1];
        if (nx < 0 || nx >= env->grid_map->grid_cols || ny < 0 || ny >= env->grid_map->grid_rows)
            continue;
        int neighborIndex = ny * env->grid_map->grid_cols + nx;
        min_neighbor_index = fmin(min_neighbor_index, neighborIndex);
        int count = env->grid_map->cell_entities_count[neighborIndex];
        entity_list_count += count;
    }

    int entered_entity_count = 0;

    // Fill entity_list with neighboring entities
    GridMapEntity *entity_list = (GridMapEntity *)calloc(entity_list_count, sizeof(GridMapEntity));
    for (int i = 0; i < offset_size; i++) {
        int nx = gridX + local_offsets[i][0];
        int ny = gridY + local_offsets[i][1];
        if (nx < 0 || nx >= env->grid_map->grid_cols || ny < 0 || ny >= env->grid_map->grid_rows)
            continue;
        int neighborIndex = ny * env->grid_map->grid_cols + nx;
        int count = env->grid_map->cell_entities_count[neighborIndex];
        if (count > 0) {
            memcpy(&entity_list[entered_entity_count], env->grid_map->cells[neighborIndex],
                   (size_t)count * sizeof(GridMapEntity));
        }
        entered_entity_count += count;
    }

    if (entered_entity_count != entity_list_count) {
        printf("Error: Mismatch in entered_entity_count (%d) and entity_list_count (%d)\n", entered_entity_count,
               entity_list_count);
    }

    *(list_count) = entity_list_count;
    return entity_list;
}

// ========================================
// Map Loading Functions
// ========================================

void load_map_binary(const char *filename, Drive *env) {
    FILE *file = fopen(filename, "rb");
    if (!file)
        return;

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

    env->agents = (Agent *)calloc(env->num_objects, sizeof(Agent));
    env->road_elements = (RoadMapElement *)calloc(env->num_roads, sizeof(RoadMapElement));
    env->road_scenario_ids = (int *)calloc(env->num_roads, sizeof(int));

    if (env->init_mode != RANDOM_AGENTS) {
        env->spawn_settings.max_agents_in_sim = env->num_objects;
    }

    int total_entities = env->num_objects + env->num_roads;
    int agent_idx = 0;
    int road_idx = 0;
    for (int i = 0; i < total_entities; i++) {
        int scenario_id = 0;
        int type = 0;
        int id = 0;
        int array_size = 0;
        float width = 0.0f, length = 0.0f, height = 0.0f;
        float goal_x = 0.0f, goal_y = 0.0f, goal_z = 0.0f;
        int mark_as_expert = 0;

        // Read base entity data
        fread(&scenario_id, sizeof(int), 1, file);
        fread(&type, sizeof(int), 1, file);
        fread(&id, sizeof(int), 1, file);
        fread(&array_size, sizeof(int), 1, file);

        if (i < env->num_objects) {
            Agent *agent = &env->agents[agent_idx];
            agent->id = id;
            agent->type = type;
            agent->scenario_id = scenario_id;
            agent->trajectory_length = array_size;

            agent->log_trajectory_x = (float *)malloc(array_size * sizeof(float));
            agent->log_trajectory_y = (float *)malloc(array_size * sizeof(float));
            agent->log_trajectory_z = (float *)malloc(array_size * sizeof(float));
            agent->log_velocity_x = (float *)malloc(array_size * sizeof(float));
            agent->log_velocity_y = (float *)malloc(array_size * sizeof(float));
            agent->log_heading = (float *)malloc(array_size * sizeof(float));
            agent->log_valid = (int *)malloc(array_size * sizeof(int));

            fread(agent->log_trajectory_x, sizeof(float), array_size, file);
            fread(agent->log_trajectory_y, sizeof(float), array_size, file);
            fread(agent->log_trajectory_z, sizeof(float), array_size, file);
            fread(agent->log_velocity_x, sizeof(float), array_size, file);
            fread(agent->log_velocity_y, sizeof(float), array_size, file);

            // Skip velocity z (unused)
            float *tmp_vz = (float *)malloc(array_size * sizeof(float));
            fread(tmp_vz, sizeof(float), array_size, file);
            free(tmp_vz);

            fread(agent->log_heading, sizeof(float), array_size, file);
            fread(agent->log_valid, sizeof(int), array_size, file);

            // Read remaining scalar fields
            fread(&width, sizeof(float), 1, file);
            fread(&length, sizeof(float), 1, file);
            fread(&height, sizeof(float), 1, file);
            fread(&goal_x, sizeof(float), 1, file);
            fread(&goal_y, sizeof(float), 1, file);
            fread(&goal_z, sizeof(float), 1, file);
            fread(&mark_as_expert, sizeof(int), 1, file);

            agent->sim_width = width;
            agent->sim_length = length;
            agent->sim_height = height;
            agent->goal_position_x = goal_x;
            agent->goal_position_y = goal_y;
            agent->goal_position_z = goal_z;
            agent->mark_as_expert = mark_as_expert;
            agent_idx++;
        } else {
            RoadMapElement *road = &env->road_elements[road_idx];
            road->id = id;
            road->type = type;
            road->segment_length = array_size;

            road->x = (float *)malloc(array_size * sizeof(float));
            road->y = (float *)malloc(array_size * sizeof(float));
            road->z = (float *)malloc(array_size * sizeof(float));

            fread(road->x, sizeof(float), array_size, file);
            fread(road->y, sizeof(float), array_size, file);
            fread(road->z, sizeof(float), array_size, file);

            // Read and discard remaining scalar fields
            fread(&width, sizeof(float), 1, file);
            fread(&length, sizeof(float), 1, file);
            fread(&height, sizeof(float), 1, file);
            fread(&goal_x, sizeof(float), 1, file);
            fread(&goal_y, sizeof(float), 1, file);
            fread(&goal_z, sizeof(float), 1, file);
            fread(&mark_as_expert, sizeof(int), 1, file);

            env->road_scenario_ids[road_idx] = scenario_id;
            road_idx++;
        }
    }

    fclose(file);
}

// ========================================
// Road Utility Functions
// ========================================

// void compute_multi_segment_alignment(void){}

// void get_drivable_lane_indices(void){}

static void get_random_point_on_lane(RoadMapElement *lane, float *out_x, float *out_y, float *out_z,
                                     float *out_heading) {
    // If the lane is sparse, we should interpolate between points for better accuracy.
    int seg_idx = rand() % (lane->segment_length - 1);

    float x0 = lane->x[seg_idx];
    float y0 = lane->y[seg_idx];
    float z0 = lane->z[seg_idx];

    if (lane->segment_length != 1) {
        float dx = lane->x[seg_idx + 1] - x0;
        float dy = lane->y[seg_idx + 1] - y0;
        *out_heading = atan2f(dy, dx);
    } else {
        *out_heading = rand() / (float)RAND_MAX * 2 * M_PI - M_PI;
    }

    *out_x = x0;
    *out_y = y0;
    *out_z = z0;
}

// void compute_lane_length(void){}

// void compute_remaining_lane_distance(void){}

// void find_closest_segment_on_lane(void){}

// void compute_log_trajectory_distance(void){}

// ========================================
// Route/Path/Goal Functions
// ========================================

// void get_closest_waypoint_index_on_path(void){}

// void build_path(void){}

// void generate_random_route(void){}

// void compute_route_distance(void){}

// void compute_new_route(void){}

// void compute_new_goal(void){}

// ========================================
// Metrics/Collision Functions
// ========================================

// void compute_displacement_error(void){}

// void check_red_light_violation(void){}

int check_aabb_collision(Agent *car1, Agent *car2) {
    // Get car corners in world space
    float cos1 = cosf(car1->sim_heading);
    float sin1 = sinf(car1->sim_heading);
    float cos2 = cosf(car2->sim_heading);
    float sin2 = sinf(car2->sim_heading);

    // Calculate half dimensions
    float half_len1 = car1->sim_length * 0.5f;
    float half_width1 = car1->sim_width * 0.5f;
    float half_len2 = car2->sim_length * 0.5f;
    float half_width2 = car2->sim_width * 0.5f;

    // Calculate car1's corners in world space
    float car1_corners[4][2] = {
        {car1->sim_x + (half_len1 * cos1 - half_width1 * sin1), car1->sim_y + (half_len1 * sin1 + half_width1 * cos1)},
        {car1->sim_x + (half_len1 * cos1 + half_width1 * sin1), car1->sim_y + (half_len1 * sin1 - half_width1 * cos1)},
        {car1->sim_x + (-half_len1 * cos1 - half_width1 * sin1),
         car1->sim_y + (-half_len1 * sin1 + half_width1 * cos1)},
        {car1->sim_x + (-half_len1 * cos1 + half_width1 * sin1),
         car1->sim_y + (-half_len1 * sin1 - half_width1 * cos1)}};

    // Calculate car2's corners in world space
    float car2_corners[4][2] = {
        {car2->sim_x + (half_len2 * cos2 - half_width2 * sin2), car2->sim_y + (half_len2 * sin2 + half_width2 * cos2)},
        {car2->sim_x + (half_len2 * cos2 + half_width2 * sin2), car2->sim_y + (half_len2 * sin2 - half_width2 * cos2)},
        {car2->sim_x + (-half_len2 * cos2 - half_width2 * sin2),
         car2->sim_y + (-half_len2 * sin2 + half_width2 * cos2)},
        {car2->sim_x + (-half_len2 * cos2 + half_width2 * sin2),
         car2->sim_y + (-half_len2 * sin2 - half_width2 * cos2)}};

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

// Note: added to support 2.5D
int check_z_collision(Agent *car1, Agent *car2) {
    float car1_bottom = car1->sim_z;
    float car1_top = car1->sim_z + car1->sim_height;
    float car2_bottom = car2->sim_z;
    float car2_top = car2->sim_z + car2->sim_height;

    // Check for overlap in the z-axis
    if (car1_top < car2_bottom || car2_top < car1_bottom) {
        return 0; // No collision
    }
    return 1; // Collision
}

int collision_check(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];

    if (agent->sim_x == INVALID_POSITION)
        return -1;

    int car_collided_with_index = -1;

    if (agent->respawn_timestep != -1)
        return car_collided_with_index; // Skip respawning entities

    for (int i = 0; i < env->spawn_settings.max_agents_in_sim; i++) {
        int index = -1;
        if (i < env->active_agent_count) {
            index = env->active_agent_indices[i];
        } else if (i < env->num_created_agents) {
            index = env->static_agent_indices[i - env->active_agent_count];
        }
        if (index == -1)
            continue;
        if (index == agent_idx)
            continue;
        Agent *entity = &env->agents[index];
        if (entity->respawn_timestep != -1)
            continue; // Skip respawning entities
        float x1 = entity->sim_x;
        float y1 = entity->sim_y;
        float dist = ((x1 - agent->sim_x) * (x1 - agent->sim_x) + (y1 - agent->sim_y) * (y1 - agent->sim_y));
        if (dist > 225.0f)
            continue;
        if (check_aabb_collision(agent, entity)) {
            agent->aabb_collision_state = 1;
            if (check_z_collision(agent, entity)) {
                car_collided_with_index = index;
                break;
            }
        }
    }

    return car_collided_with_index;
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

static bool check_offroad(Drive *env, Agent *agent) {
    if (agent->removed || agent->sim_x == INVALID_POSITION)
        return false;

    // Compute the corners of agent's bounding box
    float half_length = agent->sim_length / 2.0f;
    float half_width = agent->sim_width / 2.0f;
    float cos_heading = cosf(agent->sim_heading);
    float sin_heading = sinf(agent->sim_heading);
    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] =
            agent->sim_x + (offsets[i][0] * half_length * cos_heading - offsets[i][1] * half_width * sin_heading);
        corners[i][1] =
            agent->sim_y + (offsets[i][0] * half_length * sin_heading + offsets[i][1] * half_width * cos_heading);
    }

    int list_size = 0;
    GridMapEntity *entity_list = checkNeighbors(env, agent->sim_x, agent->sim_y, collision_offsets,
                                                COLLISION_RANGE * COLLISION_RANGE, &list_size);
    bool offroad = false;
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_idx == -1)
            continue;
        RoadMapElement *entity;
        entity = &env->road_elements[entity_list[i].entity_idx];

        // Check for offroad collision with road edges
        if (entity->type == ROAD_EDGE) {
            int geometry_idx = entity_list[i].geometry_idx;
            if (entity->z[geometry_idx] > agent->sim_z + agent->sim_height / 2.0f ||
                entity->z[geometry_idx] < agent->sim_z - agent->sim_height / 2.0f)
                continue; // Edge is at a different z level

            // TODO: Edge cases still possible here(Need to check intersection of each edge of cuboid with 3D road edge
            // lines)
            float start[2] = {entity->x[geometry_idx], entity->y[geometry_idx]};
            float end[2] = {entity->x[geometry_idx + 1], entity->y[geometry_idx + 1]};
            for (int k = 0; k < 4; k++) {
                int next = (k + 1) % 4;
                if (check_line_intersection(corners[k], corners[next], start, end)) {
                    offroad = true;
                }
            }
        }
    }
    free(entity_list);
    return offroad;
}

void add_log(Drive *env) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Agent *agent = &env->agents[agent_idx];

        env->log.goals_reached_this_episode += agent->goals_reached_this_episode;
        env->log.goals_sampled_this_episode += agent->goals_sampled_this_episode;

        int offroad = env->logs[i].offroad_rate;
        env->log.offroad_rate += offroad;
        int collided = env->logs[i].collision_rate;
        env->log.collision_rate += collided;
        float offroad_per_agent = env->logs[i].offroad_per_agent;
        env->log.offroad_per_agent += offroad_per_agent;
        float collisions_per_agent = env->logs[i].collisions_per_agent;
        env->log.collisions_per_agent += collisions_per_agent;

        float frac_goal_reached = agent->goals_reached_this_episode / agent->goals_sampled_this_episode;

        // Update score, which is an aggregate measure whether the agent fully solved its task
        // Note: When resampling goals, performance is relative to the number of goals sampled
        float threshold = 0.99f; // Default threshold for 1 goal
        if (agent->goals_sampled_this_episode == 2.0f) {
            threshold = 0.5f; // Require ≥50% completion for 2 goals
        } else if (agent->goals_sampled_this_episode < 5.0f) {
            threshold = 0.8f; // Require ≥80% completion for 3-4 goals
        } else {
            threshold = 0.9f; // Require ≥90% completion for 5+ goals
        }

        int collision_occurred =
            (env->goal_behavior == GOAL_RESPAWN) ? agent->collided_before_goal : env->logs[i].collision_rate;
        if (frac_goal_reached > threshold && !collision_occurred) {
            env->log.score += 1.0f;
        }
        if (!offroad && !collided && frac_goal_reached < 1.0f) {
            env->log.dnf_rate += 1.0f;
        }
        int lane_aligned = env->logs[i].lane_alignment_rate;
        env->log.lane_alignment_rate += lane_aligned;
        env->log.speed_at_goal += env->logs[i].speed_at_goal;
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        // Log composition counts per agent so vec_log averaging recovers the per-env value
        env->log.active_agent_count += env->active_agent_count;
        env->log.expert_static_agent_count += env->expert_static_agent_count;
        env->log.static_agent_count += env->static_agent_count;
        env->log.n += 1;
    }
}

// ========================================
// Initialization Functions
// ========================================

void reset_agent_metrics(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    agent->metrics_array[COLLISION_IDX] = 0.0f;    // vehicle collision
    agent->metrics_array[OFFROAD_IDX] = 0.0f;      // offroad
    agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f; // lane aligned
    agent->collision_state = 0;
    agent->aabb_collision_state = 0;
}

// void reset_agent_state(void){}

static bool check_spawn_collision(Drive *env, int agents_to_check, float spawn_x, float spawn_y, float spawn_heading,
                                  float spawn_length, float spawn_width) {
    // Create a temporary agent structure for collision checking
    Agent temp_agent;
    temp_agent.sim_x = spawn_x;
    temp_agent.sim_y = spawn_y;
    temp_agent.sim_heading = spawn_heading;
    temp_agent.sim_length = spawn_length;
    temp_agent.sim_width = spawn_width;

    float min_safe_dist_sq = (spawn_length + GRID_CELL_SIZE) * (spawn_length + GRID_CELL_SIZE);

    for (int i = 0; i < agents_to_check; i++) {
        Agent *other = &env->agents[i];

        if (other->sim_x == INVALID_POSITION || other->sim_valid != 1)
            continue;

        // Quick distance check first
        float dx = other->sim_x - spawn_x;
        float dy = other->sim_y - spawn_y;
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq > min_safe_dist_sq)
            continue;

        if (check_aabb_collision(&temp_agent, other)) {
            if (check_z_collision(&temp_agent, other)) {
                return true;
            }
        }
    }

    return false;
}

static bool check_spawn_offroad(Drive *env, float spawn_x, float spawn_y, float spawn_z, float spawn_heading,
                                float spawn_length, float spawn_width, float spawn_height) {
    Agent temp_agent = {.sim_x = spawn_x,
                        .sim_y = spawn_y,
                        .sim_z = spawn_z,
                        .sim_heading = spawn_heading,
                        .sim_length = spawn_length,
                        .sim_width = spawn_width,
                        .sim_height = spawn_height};

    if (check_offroad(env, &temp_agent)) {
        return true;
    }
    return false;
}

static int spawn_agent(Drive *env, int agent_idx, int agents_to_check) {
    Agent *agent = &env->agents[agent_idx];

    if (agent->route != NULL) {
        free(agent->route);
        agent->route = NULL;
    }

    // Initialize identity fields
    agent->id = agent_idx;
    agent->type = VEHICLE;
    agent->active_agent = 1;
    agent->mark_as_expert = 0;

    // TODO: Pre-compute drivable lanes in env to speed up?
    int drivable_lanes[env->num_roads];
    int num_drivable = 0;
    for (int i = 0; i < env->num_roads && num_drivable < env->num_roads; i++) {
        if (env->road_elements[i].type == ROAD_LANE) {
            drivable_lanes[num_drivable++] = i;
        }
    }

    if (num_drivable == 0)
        raise_error_with_message(ERROR_UNKNOWN,
                                 "No drivable lanes found in the environment with %d created agents at agent_idx %d.",
                                 env->num_created_agents, agent_idx);

    // TODO: for eval mode
    // spawn_length = random_uniform(2.0f, 5.5f);
    // spawn_width = random_uniform(1.5f, 2.5f);

    AgentSpawnSettings spawn_settings = env->spawn_settings;

    float spawn_length = random_uniform(spawn_settings.min_l, spawn_settings.max_l);
    float spawn_width = random_uniform(spawn_settings.min_w, spawn_settings.max_w);

    // Design Choice(we don't have wide vehicles on roads)
    if (spawn_width > spawn_length)
        spawn_width = spawn_length;
    float spawn_height = 1.5f; // Design Choice: Doesn't matter as we don't have flying cars in 2026

    // Set spawn position on start lane
    float spawn_x, spawn_y, spawn_z, spawn_heading;
    RoadMapElement *start_lane;
    int start_lane_idx;

    // Sampling rejection loop
    for (int attempt = 0; attempt < MAX_SPAWN_ATTEMPTS; attempt++) {
        // TODO: Uniform sampling can lead to clustering in small lanes
        start_lane_idx = drivable_lanes[rand() % num_drivable];
        start_lane = &env->road_elements[start_lane_idx];

        get_random_point_on_lane(start_lane, &spawn_x, &spawn_y, &spawn_z, &spawn_heading);

        spawn_z += spawn_height / 2.0f; // Adjust z to be at the center of the agent's height

        // Check for collision with existing/already-reset agents
        if (check_spawn_collision(env, agents_to_check, spawn_x, spawn_y, spawn_heading, spawn_length, spawn_width))
            continue;

        // Check for offroad
        if (check_spawn_offroad(env, spawn_x, spawn_y, spawn_z, spawn_heading, spawn_length, spawn_width, spawn_height))
            continue;

        break;
    }

    // Update simulation state
    agent->sim_x = spawn_x;
    agent->sim_y = spawn_y;
    agent->sim_z = spawn_z;
    agent->sim_heading = spawn_heading;
    agent->sim_length = spawn_length;
    agent->sim_width = spawn_width;
    agent->sim_height = spawn_height;
    agent->sim_valid = 1;
    agent->wheelbase = 0.6f * spawn_length; // Estimate wheelbase as 60% of length

    // Explicitly initialize velocity to zero
    agent->sim_vx = 0.0f;
    agent->sim_vy = 0.0f;
    agent->sim_speed = 0.0f;
    agent->sim_speed_signed = 0.0f;

    // Compute initial route
    // if (!compute_new_route(env, agent_idx, start_lane->id)) {
    //     printf("[GIGAFLOW WARNING] -> Failed to compute a new route for agent %d\n", agent->id);
    //     return 0; // Failed to compute new goal
    // }

    return 1;
}

void set_start_position(Drive *env) {
    if (env->init_mode == RANDOM_AGENTS) {
        return;
    }
    for (int i = 0; i < env->num_objects; i++) {
        int is_active = 0;
        for (int j = 0; j < env->active_agent_count; j++) {
            if (env->active_agent_indices[j] == i) {
                is_active = 1;
                break;
            }
        }
        Agent *e = &env->agents[i];

        // Clamp init_steps to ensure we don't go out of bounds
        int step = env->init_steps;
        if (step >= e->trajectory_length)
            step = e->trajectory_length - 1;
        if (step < 0)
            step = 0;

        e->sim_x = e->log_trajectory_x[step];
        e->sim_y = e->log_trajectory_y[step];
        e->sim_z = e->log_trajectory_z[step];
        if (e->type > CYCLIST || e->type == 0) {
            continue;
        }
        if (is_active == 0) {
            e->sim_vx = 0;
            e->sim_vy = 0;
            e->collided_before_goal = 0;
        } else {
            e->sim_vx = e->log_velocity_x[env->init_steps];
            e->sim_vy = e->log_velocity_y[env->init_steps];
        }
        e->sim_heading = e->log_heading[env->init_steps];
        e->heading_x = cosf(e->sim_heading);
        e->heading_y = sinf(e->sim_heading);
        e->sim_valid = e->log_valid[env->init_steps];
        e->collision_state = 0;
        e->aabb_collision_state = 0;
        e->metrics_array[COLLISION_IDX] = 0.0f;    // vehicle collision
        e->metrics_array[OFFROAD_IDX] = 0.0f;      // offroad
        e->metrics_array[REACHED_GOAL_IDX] = 0.0f; // reached goal
        e->metrics_array[LANE_ALIGNED_IDX] = 0.0f; // lane aligned
        e->respawn_timestep = -1;
        e->stopped = 0;
        e->removed = 0;
        e->respawn_count = 0;

        // Dynamics
        e->a_long = 0.0f;
        e->a_lat = 0.0f;
        e->jerk_long = 0.0f;
        e->jerk_lat = 0.0f;
        e->steering_angle = 0.0f;
        e->wheelbase = 0.6f * e->sim_length;
    }
}

bool should_control_agent(Drive *env, int agent_idx) {

    // Check if we have room for more agents or are already at capacity
    if (env->active_agent_count >= env->num_agents) {
        return false;
    }

    Agent *entity = &env->agents[agent_idx];

    // TODO: Move this elsewhere or remove
    entity->sim_width *= 0.7f;
    entity->sim_length *= 0.7f;

    if (env->control_mode == CONTROL_SDC_ONLY) {
        return agent_idx == env->sdc_track_index;
    }

    bool is_vehicle = (entity->type == VEHICLE);
    bool is_ped_or_bike = (entity->type == PEDESTRIAN || entity->type == CYCLIST);
    bool type_is_valid = false;

    switch (env->control_mode) {
    case CONTROL_WOSAC:
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
    float cos_heading = cosf(entity->log_heading[0]);
    float sin_heading = sinf(entity->log_heading[0]);
    float goal_dx = entity->goal_position_x - entity->log_trajectory_x[0];
    float goal_dy = entity->goal_position_y - entity->log_trajectory_y[0];
    float goal_dz = entity->goal_position_z - entity->log_trajectory_z[0];

    // Transform to agent's local frame
    float local_goal_x = goal_dx * cos_heading + goal_dy * sin_heading;
    float local_goal_y = -goal_dx * sin_heading + goal_dy * cos_heading;
    float distance_to_goal = relative_distance_3d(0, 0, 0, local_goal_x, local_goal_y, goal_dz);
    return distance_to_goal >= MIN_DISTANCE_TO_GOAL;
}

int spawn_active_agents(Drive *env, int num_agents_to_create) {
    // Free any pre-existing agents allocated during map loading
    free_agents(env->agents, env->num_objects);

    env->agents = (Agent *)calloc(num_agents_to_create, sizeof(Agent));

    int successfully_created = 0;
    for (int i = 0; i < num_agents_to_create; i++) {
        int created = 0;
        for (int attempt = 0; attempt < MAX_SPAWNS_ATTEMPTS_WITH_DIMENSION_CHANGES; attempt++) {
            if (spawn_agent(env, i, successfully_created)) {
                successfully_created++;
                created = 1;
                break;
            } else {
            }
        }
        if (!created) {
            // Failed spawn: ensure agent is properly invalidated
            printf("WARNING: Failed to spawn agent %d after %d attempts with changed settings. Marking as removed.\n",
                   i, MAX_SPAWNS_ATTEMPTS_WITH_DIMENSION_CHANGES);
            env->agents[i].sim_x = INVALID_POSITION;
            env->agents[i].sim_y = INVALID_POSITION;
            env->agents[i].sim_valid = 0;
            env->agents[i].removed = 1;
        }
    }

    int created_cnt = 0;
    Agent *created_agents = (Agent *)calloc(successfully_created, sizeof(Agent));
    for (int i = 0; i < num_agents_to_create; i++) {
        if (env->agents[i].removed || !env->agents[i].sim_valid) {
            continue;
        }
        created_agents[created_cnt] = env->agents[i]; // Direct assignment works as random agents mode has no
                                                      // dynamically allocated fields(logged data)
        created_cnt++;
    }

    // Free agents array with invalid agents and replace with array of successfully created agents
    free(env->agents);
    env->agents = created_agents;

    return successfully_created;
}

void spawn_agents_with_counts(Drive *env) {
    // Currently only creates active agents
    int num_agents_to_create = env->num_agents;

    int successfully_created = spawn_active_agents(env, num_agents_to_create);
    env->num_created_agents = successfully_created;

    // Free all agent cnt arrays
    free(env->active_agent_indices);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);

    env->active_agent_indices = (int *)malloc(env->num_created_agents * sizeof(int));
    env->static_agent_indices = NULL;
    env->expert_static_agent_indices = NULL;

    for (int i = 0; i < env->num_created_agents; i++)
        env->active_agent_indices[i] = i;

    // Only active agents supported currently
    env->active_agent_count = env->num_created_agents;
    env->static_agent_count = 0;
    env->expert_static_agent_count = 0;

    return;
}

void set_active_agents(Drive *env) {

    env->active_agent_count = 0;        // Policy-controlled agents
    env->static_agent_count = 0;        // Non-moving background agents
    env->expert_static_agent_count = 0; // Expert replay agents (non-controlled)
    env->num_created_agents = 0;        // Total agents created

    if (env->num_agents == 0) {
        printf("Warning: num_agents is 0, defaulting to max_agents_in_sim (%d)\n",
               env->spawn_settings.max_agents_in_sim);
        env->num_agents = env->spawn_settings.max_agents_in_sim;
    }

    if (env->init_mode == RANDOM_AGENTS) {
        spawn_agents_with_counts(env);
        env->num_objects = env->num_created_agents;
        return;
    }

    // For other modes(agents from data)
    int *active_agent_indices = (int *)malloc(env->spawn_settings.max_agents_in_sim * sizeof(int));
    int *static_agent_indices = (int *)malloc(env->spawn_settings.max_agents_in_sim * sizeof(int));
    int *expert_static_agent_indices = (int *)malloc(env->spawn_settings.max_agents_in_sim * sizeof(int));

    // Iterate through entities to find agents to create and/or control
    for (int i = 0; i < env->num_objects && env->num_created_agents < env->spawn_settings.max_agents_in_sim; i++) {
        Agent *entity = &env->agents[i];

        // Skip if not valid at initialization
        if (entity->log_valid[env->init_steps] != 1) {
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

        env->num_created_agents++;

        // Determine if this agent should be policy-controlled
        bool is_controlled = false;

        is_controlled = should_control_agent(env, i);

        if (is_controlled) {
            active_agent_indices[env->active_agent_count] = i;
            env->active_agent_count++;
            env->agents[i].active_agent = 1;
        } else if (env->init_mode != INIT_ONLY_CONTROLLABLE_AGENTS) {
            static_agent_indices[env->static_agent_count] = i;
            env->static_agent_count++;
            env->agents[i].active_agent = 0;
            if (env->agents[i].mark_as_expert == 1 || env->active_agent_count == env->num_agents) {
                expert_static_agent_indices[env->expert_static_agent_count] = i;
                env->expert_static_agent_count++;
                env->agents[i].mark_as_expert = 1;
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

    return;
}

void remove_bad_trajectories(Drive *env) {

    if (env->control_mode != CONTROL_WOSAC) {
        return; // Leave all trajectories in WOSAC control mode
    }

    if (env->init_mode == RANDOM_AGENTS) {
        return; // No trajectories in random agents mode
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
            if (env->agents[expert_idx].sim_x == INVALID_POSITION)
                continue;
            move_expert(env, env->actions, expert_idx);
        }
        // check collisions
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            env->agents[agent_idx].collision_state = 0;
            env->agents[agent_idx].aabb_collision_state = 0;
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
            env->agents[static_agent_idx].log_trajectory_x[0] = INVALID_POSITION;
            env->agents[static_agent_idx].log_trajectory_y[0] = INVALID_POSITION;
            env->agents[static_agent_idx].log_trajectory_z[0] = INVALID_POSITION;
        }
    }
    env->timestep = 0;
}

void init(Drive *env) {
    env->human_agent_idx = 0;
    env->timestep = 0;
    load_map_binary(env->map_name, env);
    set_means(env);
    init_grid_map(env);
    generate_offsets(collision_offsets, COLLISION_RANGE);
    generate_offsets(z_offsets, Z_RANGE);
    env->grid_map->vision_range = 21;
    init_neighbor_offsets(env);
    cache_neighbor_offsets(env);
    env->logs_capacity = 0;
    set_active_agents(env);
    env->logs_capacity = env->active_agent_count;
    remove_bad_trajectories(env);
    set_start_position(env);
    init_goal_positions(env);
    env->logs = (Log *)calloc(env->active_agent_count, sizeof(Log));
}

void c_close(Drive *env) {
    free_agents(env->agents, env->num_objects);
    for (int i = 0; i < env->num_roads; i++) {
        free_road_element(&env->road_elements[i]);
    }
    free(env->road_elements);
    free(env->road_scenario_ids);
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
    free(env->tracks_to_predict_indices);
    free(env->ini_file);
}

void allocate(Drive *env) {
    init(env);
    int ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int max_obs = ego_dim + PARTNER_FEATURES * (MAX_AGENTS - 1) + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS;
    env->observations = (float *)calloc(env->active_agent_count * max_obs, sizeof(float));
    env->actions = (float *)calloc(env->active_agent_count * 2, sizeof(float));
    env->rewards = (float *)calloc(env->active_agent_count, sizeof(float));
    env->terminals = (unsigned char *)calloc(env->active_agent_count, sizeof(unsigned char));
}

void free_allocated(Drive *env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
}

// ========================================
// Extra C API Functions
// ========================================

static inline int get_track_id_or_placeholder(Drive *env, int agent_idx) {
    if (env->tracks_to_predict_indices == NULL || env->num_tracks_to_predict == 0) {
        return -1;
    }
    for (int k = 0; k < env->num_tracks_to_predict; k++) {
        if (env->tracks_to_predict_indices[k] == agent_idx) {
            return env->tracks_to_predict_indices[k];
        }
    }
    return -1;
}

void c_get_global_agent_state(Drive *env, float *x_out, float *y_out, float *z_out, float *heading_out, int *id_out,
                              float *length_out, float *width_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Agent *agent = &env->agents[agent_idx];

        // For WOSAC, we need the original world coordinates, so we add the world means back
        x_out[i] = agent->sim_x + env->world_mean_x;
        y_out[i] = agent->sim_y + env->world_mean_y;
        z_out[i] = agent->sim_z + env->world_mean_z;
        heading_out[i] = agent->sim_heading;
        id_out[i] = get_track_id_or_placeholder(env, agent_idx);
        length_out[i] = agent->sim_length;
        width_out[i] = agent->sim_width;
    }
}

void c_get_global_ground_truth_trajectories(Drive *env, float *x_out, float *y_out, float *z_out, float *heading_out,
                                            int *valid_out, int *id_out, int *scenario_id_out) {
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        Agent *agent = &env->agents[agent_idx];
        id_out[i] = get_track_id_or_placeholder(env, agent_idx);
        scenario_id_out[i] = agent->scenario_id;

        for (int t = env->init_steps; t < agent->trajectory_length; t++) {
            int out_idx = i * (agent->trajectory_length - env->init_steps) + (t - env->init_steps);
            // Add world means back to get original world coordinates
            x_out[out_idx] = agent->log_trajectory_x[t] + env->world_mean_x;
            y_out[out_idx] = agent->log_trajectory_y[t] + env->world_mean_y;
            z_out[out_idx] = agent->log_trajectory_z[t] + env->world_mean_z;
            heading_out[out_idx] = agent->log_heading[t];
            valid_out[out_idx] = agent->log_valid[t];
        }
    }
}

void c_get_road_edge_counts(Drive *env, int *num_polylines_out, int *total_points_out) {
    int count = 0, points = 0;
    for (int i = 0; i < env->num_roads; i++) {
        if (env->road_elements[i].type == ROAD_EDGE) {
            count++;
            points += env->road_elements[i].segment_length;
        }
    }
    *num_polylines_out = count;
    *total_points_out = points;
}

void c_get_road_edge_polylines(Drive *env, float *x_out, float *y_out, int *lengths_out, int *scenario_ids_out) {
    int poly_idx = 0, pt_idx = 0;
    for (int i = 0; i < env->num_roads; i++) {
        RoadMapElement *road = &env->road_elements[i];
        if (road->type == ROAD_EDGE) {
            lengths_out[poly_idx] = road->segment_length;
            scenario_ids_out[poly_idx] = env->road_scenario_ids[i];
            for (int j = 0; j < road->segment_length; j++) {
                x_out[pt_idx] = road->x[j] + env->world_mean_x;
                y_out[pt_idx] = road->y[j] + env->world_mean_y;
                pt_idx++;
            }
            poly_idx++;
        }
    }
}

// ========================================
// Core Simulation Functions
// ========================================

// rename to: compute_metrics
void compute_agent_metrics(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];

    reset_agent_metrics(env, agent_idx);

    if (agent->sim_x == INVALID_POSITION)
        return; // invalid agent position

    int collided = 0;
    float half_length = agent->sim_length / 2.0f;
    float half_width = agent->sim_width / 2.0f;
    float cos_heading = cosf(agent->sim_heading);
    float sin_heading = sinf(agent->sim_heading);
    float min_distance = (float)INT16_MAX;

    int closest_lane_entity_idx = -1;
    int closest_lane_geometry_idx = -1;

    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] =
            agent->sim_x + (offsets[i][0] * half_length * cos_heading - offsets[i][1] * half_width * sin_heading);
        corners[i][1] =
            agent->sim_y + (offsets[i][0] * half_length * sin_heading + offsets[i][1] * half_width * cos_heading);
    }
    int list_size = 0;
    GridMapEntity *entity_list = checkNeighbors(env, agent->sim_x, agent->sim_y, collision_offsets,
                                                COLLISION_RANGE * COLLISION_RANGE, &list_size);
    for (int i = 0; i < list_size; i++) {
        if (entity_list[i].entity_idx == -1)
            continue;
        if (entity_list[i].entity_idx == agent_idx)
            continue;
        RoadMapElement *entity;
        entity = &env->road_elements[entity_list[i].entity_idx];

        // Check for offroad collision with road edges
        if (entity->type == ROAD_EDGE) {
            int geometry_idx = entity_list[i].geometry_idx;
            if (entity->z[geometry_idx] > agent->sim_z + Z_BUFFER || entity->z[geometry_idx] < agent->sim_z - Z_BUFFER)
                continue; // Edge is at a different z level
            float start[2] = {entity->x[geometry_idx], entity->y[geometry_idx]};
            float end[2] = {entity->x[geometry_idx + 1], entity->y[geometry_idx + 1]};
            for (int k = 0; k < 4; k++) { // Check each edge of the bounding box
                int next = (k + 1) % 4;
                if (check_line_intersection(corners[k], corners[next], start, end)) {
                    collided = OFFROAD;
                    break;
                }
            }
        }

        if (collided == OFFROAD)
            break;

        // Find closest point on the road centerline to the agent
        if (entity->type == ROAD_LANE) {
            int entity_idx = entity_list[i].entity_idx;
            int geometry_idx = entity_list[i].geometry_idx;

            float start[2] = {entity->x[geometry_idx], entity->y[geometry_idx]};
            float end[2] = {entity->x[geometry_idx + 1], entity->y[geometry_idx + 1]};

            float dist = point_to_segment_distance_2d(agent->sim_x, agent->sim_y, start[0], start[1], end[0], end[1]);
            float heading_diff = fabsf(atan2f(end[1] - start[1], end[0] - start[0]) - agent->sim_heading);

            // Normalize heading difference to [0, pi]
            if (heading_diff > M_PI)
                heading_diff = 2.0f * M_PI - heading_diff;

            // Penalize if heading differs by more than 30 degrees
            if (heading_diff > (M_PI / 6.0f))
                dist += 3.0f;

            if (dist < min_distance) {
                min_distance = dist;
                closest_lane_entity_idx = entity_idx;
                closest_lane_geometry_idx = geometry_idx;
            }
        }
    }

    // check if aligned with closest lane and set current lane
    // 4.0m threshold: agents more than 4 meters from any lane are considered off-road
    if (min_distance > 4.0f || closest_lane_entity_idx == -1) {
        agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f;
        agent->current_lane_index = -1;
    } else {
        agent->current_lane_index = closest_lane_entity_idx;
        int lane_aligned =
            check_lane_aligned(agent, &env->road_elements[closest_lane_entity_idx], closest_lane_geometry_idx);
        agent->metrics_array[LANE_ALIGNED_IDX] = lane_aligned;
    }

    // Check for vehicle collisions
    int car_collided_with_index = collision_check(env, agent_idx);
    if (car_collided_with_index != -1)
        collided = VEHICLE_COLLISION;

    agent->collision_state = collided;

    if (collided == VEHICLE_COLLISION) {
        if (env->collision_behavior == STOP_AGENT && !agent->stopped) {
            agent->stopped = 1;
            agent->sim_vx = agent->sim_vy = 0.0f;
        } else if (env->collision_behavior == REMOVE_AGENT && !agent->removed) {
            Agent *agent_collided = &env->agents[car_collided_with_index];
            agent->removed = 1;
            agent_collided->removed = 1;
            agent->sim_x = agent->sim_y = -10000.0f;
            agent_collided->sim_x = agent_collided->sim_y = -10000.0f;
        }
    }
    if (collided == OFFROAD) {
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        if (env->offroad_behavior == STOP_AGENT && !agent->stopped) {
            agent->stopped = 1;
            agent->sim_vx = agent->sim_vy = 0.0f;
        } else if (env->offroad_behavior == REMOVE_AGENT && !agent->removed) {
            agent->removed = 1;
            agent->sim_x = agent->sim_y = -10000.0f;
        }
    }
    free(entity_list);
    return;
}

// void compute_rewards(void){}

void compute_observations(Drive *env) {
    int ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int max_obs = ego_dim + PARTNER_FEATURES * (MAX_AGENTS - 1) + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS;
    memset(env->observations, 0, max_obs * env->active_agent_count * sizeof(float));
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    for (int i = 0; i < env->active_agent_count; i++) {
        float *obs = &observations[i][0];
        int ego_idx = env->active_agent_indices[i];
        Agent *ego_entity = &env->agents[ego_idx];
        if (ego_entity->type > CYCLIST)
            break;

        float cos_heading = cosf(ego_entity->sim_heading);
        float sin_heading = sinf(ego_entity->sim_heading);
        float speed_magnitude =
            sqrtf(ego_entity->sim_vx * ego_entity->sim_vx + ego_entity->sim_vy * ego_entity->sim_vy);
        float v_dot_heading = ego_entity->sim_vx * cos_heading + ego_entity->sim_vy * sin_heading;
        float signed_speed = copysignf(speed_magnitude, v_dot_heading);

        // Set goal distances
        float goal_x = ego_entity->goal_position_x - ego_entity->sim_x;
        float goal_y = ego_entity->goal_position_y - ego_entity->sim_y;
        float goal_z = ego_entity->goal_position_z - ego_entity->sim_z;

        // Rotate to ego vehicle's frame
        float rel_goal_x = goal_x * cos_heading + goal_y * sin_heading;
        float rel_goal_y = -goal_x * sin_heading + goal_y * cos_heading;

        float rel_goal_z = goal_z; // No rotation needed for vertical component
        obs[0] = rel_goal_x * 0.005f;
        obs[1] = rel_goal_y * 0.005f;
        obs[2] = rel_goal_z * 0.005f;
        obs[3] = signed_speed / MAX_SPEED;
        obs[4] = ego_entity->sim_width / MAX_VEH_WIDTH;
        obs[5] = ego_entity->sim_length / MAX_VEH_LEN;
        obs[6] = (ego_entity->collision_state > 0) ? 1.0f : 0.0f;

        if (env->dynamics_model == JERK) {
            obs[7] = ego_entity->steering_angle / M_PI;
            // Asymmetric normalization for a_long to match action space
            obs[8] =
                (ego_entity->a_long < 0) ? ego_entity->a_long / (-JERK_LONG[0]) : ego_entity->a_long / JERK_LONG[3];
            obs[9] = ego_entity->a_lat / JERK_LAT[2];
            obs[10] = (ego_entity->respawn_timestep != -1) ? 1 : 0;
        } else {
            obs[7] = (ego_entity->respawn_timestep != -1) ? 1 : 0;
        }

        // Relative Pos of other cars
        int obs_idx = ego_dim;
        int cars_seen = 0;
        for (int j = 0; j < MAX_AGENTS; j++) {
            int index = -1;
            if (j < env->active_agent_count) {
                index = env->active_agent_indices[j];
            } else if (j < env->num_created_agents) {
                index = env->static_agent_indices[j - env->active_agent_count];
            }
            if (index == -1)
                continue;
            if (env->agents[index].type > CYCLIST)
                break;
            if (index == env->active_agent_indices[i])
                continue; // Skip self, but don't increment obs_idx
            Agent *other_entity = &env->agents[index];
            if (ego_entity->respawn_timestep != -1)
                continue;
            if (other_entity->respawn_timestep != -1)
                continue;
            // Store original relative positions
            float dx = other_entity->sim_x - ego_entity->sim_x;
            float dy = other_entity->sim_y - ego_entity->sim_y;
            float dz = other_entity->sim_z - ego_entity->sim_z;
            float dist = (dx * dx + dy * dy + dz * dz);
            if (dist > 2500.0f)
                continue;
            // Rotate to ego vehicle's frame
            float rel_x = dx * cos_heading + dy * sin_heading;
            float rel_y = -dx * sin_heading + dy * cos_heading;
            float rel_z = dz; // No rotation needed for vertical component
            // Store observations with correct indexing
            obs[obs_idx] = rel_x * 0.02f;
            obs[obs_idx + 1] = rel_y * 0.02f;
            obs[obs_idx + 2] = rel_z * 0.02f;
            obs[obs_idx + 3] = other_entity->sim_width / MAX_VEH_WIDTH;
            obs[obs_idx + 4] = other_entity->sim_length / MAX_VEH_LEN;
            // relative heading
            float other_cos = cosf(other_entity->sim_heading);
            float other_sin = sinf(other_entity->sim_heading);
            float rel_heading_x =
                other_cos * cos_heading + other_sin * sin_heading; // cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
            float rel_heading_y =
                other_sin * cos_heading - other_cos * sin_heading; // sin(a-b) = sin(a)cos(b) - cos(a)sin(b)

            obs[obs_idx + 5] = rel_heading_x;
            obs[obs_idx + 6] = rel_heading_y;
            // relative speed
            float other_speed_magnitude =
                sqrtf(other_entity->sim_vx * other_entity->sim_vx + other_entity->sim_vy * other_entity->sim_vy);
            float other_v_dot_heading = other_entity->sim_vx * other_cos + other_entity->sim_vy * other_sin;
            float other_signed_speed = copysignf(other_speed_magnitude, other_v_dot_heading);
            obs[obs_idx + 7] = other_signed_speed / MAX_SPEED;
            cars_seen++;
            obs_idx += 8; // Move to next observation slot
        }
        int remaining_partner_obs = (MAX_AGENTS - 1 - cars_seen) * 8;
        memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
        obs_idx += remaining_partner_obs;
        // map observations
        GridMapEntity entity_list[MAX_ROAD_SEGMENT_OBSERVATIONS];
        int grid_idx = getGridIndex(env, ego_entity->sim_x, ego_entity->sim_y);

        int list_size = get_neighbor_cache_entities(env, grid_idx, entity_list, MAX_ROAD_SEGMENT_OBSERVATIONS);

        for (int k = 0; k < list_size; k++) {
            int entity_idx = entity_list[k].entity_idx;
            int geometry_idx = entity_list[k].geometry_idx;

            // Validate entity_idx before accessing
            if (entity_idx < 0 || entity_idx >= env->num_roads) {
                printf("ERROR: Invalid road_idx %d (max: %d)\n", entity_idx, env->num_roads - 1);
                continue;
            }

            RoadMapElement *entity = &env->road_elements[entity_idx];

            // Validate geometry_idx before accessing
            if (geometry_idx < 0 || geometry_idx >= entity->segment_length) {
                printf("ERROR: Invalid geometry_idx %d for road %d (max: %d)\n", geometry_idx, entity_idx,
                       entity->segment_length - 1);
                continue;
            }
            float start_x = entity->x[geometry_idx];
            float start_y = entity->y[geometry_idx];
            float start_z = entity->z[geometry_idx];
            float end_x = entity->x[geometry_idx + 1];
            float end_y = entity->y[geometry_idx + 1];
            float end_z = entity->z[geometry_idx + 1];
            float mid_x = (start_x + end_x) / 2.0f;
            float mid_y = (start_y + end_y) / 2.0f;
            float mid_z = (start_z + end_z) / 2.0f;
            float rel_x = mid_x - ego_entity->sim_x;
            float rel_y = mid_y - ego_entity->sim_y;
            float rel_z = mid_z - ego_entity->sim_z;
            float x_obs = rel_x * cos_heading + rel_y * sin_heading;
            float y_obs = -rel_x * sin_heading + rel_y * cos_heading;
            float z_obs = rel_z;
            float length = relative_distance_3d(mid_x, mid_y, mid_z, end_x, end_y, end_z);
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
            obs[obs_idx + 2] = z_obs * 0.02f;
            obs[obs_idx + 3] = length / MAX_ROAD_SEGMENT_LENGTH;
            obs[obs_idx + 4] = width / MAX_ROAD_SCALE;
            obs[obs_idx + 5] = cos_angle;
            obs[obs_idx + 6] = sin_angle;
            obs[obs_idx + 7] = entity->type - 4.0f;
            obs_idx += 8;
        }
        int remaining_obs = (MAX_ROAD_SEGMENT_OBSERVATIONS - list_size) * 8;
        // Set the entire block to 0 at once
        memset(&obs[obs_idx], 0, remaining_obs * sizeof(float));
    }
}

void respawn_agent(Drive *env, int agent_idx) {

    if (env->init_mode == RANDOM_AGENTS) {
        spawn_agents_with_counts(env);
        return;
    }

    Agent *agent = &env->agents[agent_idx];
    agent->sim_x = agent->log_trajectory_x[0];
    agent->sim_y = agent->log_trajectory_y[0];
    agent->sim_z = agent->log_trajectory_z[0];
    agent->sim_heading = agent->log_heading[0];
    agent->heading_x = cosf(agent->sim_heading);
    agent->heading_y = sinf(agent->sim_heading);
    agent->sim_vx = agent->log_velocity_x[0];
    agent->sim_vy = agent->log_velocity_y[0];
    agent->metrics_array[COLLISION_IDX] = 0.0f;
    agent->metrics_array[OFFROAD_IDX] = 0.0f;
    agent->metrics_array[REACHED_GOAL_IDX] = 0.0f;
    agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f;

    agent->respawn_timestep = env->timestep;
    agent->collided_before_goal = 0;
    agent->stopped = 0;
    agent->removed = 0;
    agent->a_long = 0.0f;
    agent->a_lat = 0.0f;
    agent->jerk_long = 0.0f;
    agent->jerk_lat = 0.0f;
    agent->steering_angle = 0.0f;
}

void move_expert(Drive *env, float *actions, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    int t = env->timestep;
    if (t < 0 || t >= agent->trajectory_length) {
        agent->sim_x = INVALID_POSITION;
        agent->sim_y = INVALID_POSITION;
        agent->sim_z = 0.0f;
        agent->sim_heading = 0.0f;
        agent->heading_x = 1.0f;
        agent->heading_y = 0.0f;
        return;
    }
    if (agent->log_valid && agent->log_valid[t] == 0) {
        agent->sim_x = INVALID_POSITION;
        agent->sim_y = INVALID_POSITION;
        agent->sim_z = 0.0f;
        agent->sim_heading = 0.0f;
        agent->heading_x = 1.0f;
        agent->heading_y = 0.0f;
        return;
    }
    agent->sim_x = agent->log_trajectory_x[t];
    agent->sim_y = agent->log_trajectory_y[t];
    agent->sim_z = agent->log_trajectory_z[t];
    agent->sim_heading = agent->log_heading[t];
    agent->heading_x = cosf(agent->sim_heading);
    agent->heading_y = sinf(agent->sim_heading);
}

void move_dynamics(Drive *env, int action_idx, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    if (agent->removed)
        return;

    if (agent->stopped) {
        agent->sim_vx = 0.0f;
        agent->sim_vy = 0.0f;
        return;
    }

    if (env->dynamics_model == CLASSIC) {
        // Classic dynamics model
        float acceleration = 0.0f;
        float steering = 0.0f;

        if (env->action_type == 1) { // continuous
            float (*action_array_f)[2] = (float (*)[2])env->actions;
            acceleration = action_array_f[action_idx][0];
            steering = action_array_f[action_idx][1];

            acceleration *= ACCELERATION_VALUES[6];
            steering *= STEERING_VALUES[12];
        } else { // discrete
            // Interpret action as a single integer: a = accel_idx * num_steer + steer_idx
            int *action_array = (int *)env->actions;
            int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
            int action_val = action_array[action_idx];
            int acceleration_index = action_val / num_steer;
            int steering_index = action_val % num_steer;
            acceleration = ACCELERATION_VALUES[acceleration_index];
            steering = STEERING_VALUES[steering_index];
        }

        // Current state
        float x = agent->sim_x;
        float y = agent->sim_y;
        float heading = agent->sim_heading;
        float vx = agent->sim_vx;
        float vy = agent->sim_vy;

        // Calculate current speed (signed based on direction relative to heading)
        float speed_magnitude = sqrtf(vx * vx + vy * vy);
        float v_dot_heading = vx * cosf(heading) + vy * sinf(heading);
        float signed_speed = copysignf(speed_magnitude, v_dot_heading);

        // Update speed with acceleration
        signed_speed = signed_speed + acceleration * env->dt;
        signed_speed = clipSpeed(signed_speed);
        // Compute yaw rate
        float beta = tanh(.5 * tanf(steering));

        // New heading
        float yaw_rate = (signed_speed * cosf(beta) * tanf(steering)) / agent->sim_length;

        // New velocity
        float new_vx = signed_speed * cosf(heading + beta);
        float new_vy = signed_speed * sinf(heading + beta);

        // Update position
        x = x + (new_vx * env->dt);
        y = y + (new_vy * env->dt);
        heading = heading + yaw_rate * env->dt;

        // Apply updates to the agent's state
        agent->sim_x = x;
        agent->sim_y = y;
        agent->sim_heading = heading;
        agent->heading_x = cosf(heading);
        agent->heading_y = sinf(heading);
        agent->sim_vx = new_vx;
        agent->sim_vy = new_vy;
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
        float v_dot_heading = agent->sim_vx * cosf(agent->sim_heading) + agent->sim_vy * sinf(agent->sim_heading);
        float signed_v = copysignf(sqrtf(agent->sim_vx * agent->sim_vx + agent->sim_vy * agent->sim_vy), v_dot_heading);
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

        float cos_heading = cosf(agent->sim_heading);
        float sin_heading = sinf(agent->sim_heading);
        float dx = dx_local * cos_heading - dy_local * sin_heading;
        float dy = dx_local * sin_heading + dy_local * cos_heading;

        // Update everything
        agent->sim_x += dx;
        agent->sim_y += dy;
        agent->jerk_long = (a_long_new - agent->a_long) / env->dt;
        agent->jerk_lat = (a_lat_new - agent->a_lat) / env->dt;
        agent->a_long = a_long_new;
        agent->a_lat = a_lat_new;
        agent->sim_heading = normalize_heading(agent->sim_heading + theta);
        agent->heading_x = cosf(agent->sim_heading);
        agent->heading_y = sinf(agent->sim_heading);
        agent->sim_vx = v_new * agent->heading_x;
        agent->sim_vy = v_new * agent->heading_y;
        agent->steering_angle = new_steering_angle;
    }

    // To update agent's z-coordinate based on road elevation of 20 nearest elements
    int list_size = 0;
    GridMapEntity *entity_list =
        checkNeighbors(env, agent->sim_x, agent->sim_y, z_offsets, Z_RANGE * Z_RANGE, &list_size);
    if (list_size > 0) {
        DepthPoint road_neighbours[list_size];
        int valid_count = 0;
        // store an array masuring the distance of the agent with each road segment nearby
        for (int i = 0; i < list_size; i++) {
            if (entity_list[i].entity_idx == -1)
                continue;
            RoadMapElement *entity = &env->road_elements[entity_list[i].entity_idx];
            if (entity->type == ROAD_EDGE || entity->type == ROAD_LINE || entity->type == ROAD_LANE) {
                int geometry_idx = entity_list[i].geometry_idx;
                DepthPoint val = compute_z_distance_to_road_segment(agent, entity, geometry_idx);
                if (val.dis < Z_BUFFER) {
                    road_neighbours[valid_count++] = val;
                }
            }
        }

        if (valid_count > 0) {

            qsort(road_neighbours, valid_count, sizeof(DepthPoint), compare_depthpoint);
            int check_count = (valid_count < 30) ? valid_count : 30;
            float sum_z = 0.0f;
            for (int i = 0; i < check_count; i++) {
                sum_z += road_neighbours[i].z;
            }
            agent->sim_z = sum_z / (check_count);
        }
    }
    // Free allocated memory
    free(entity_list);
    return;
}

void c_reset(Drive *env) {
    env->timestep = env->init_steps;
    set_start_position(env);
    init_goal_positions(env);
    for (int x = 0; x < env->active_agent_count; x++) {
        env->logs[x] = (Log){0};
        int agent_idx = env->active_agent_indices[x];
        Agent *agent = &env->agents[agent_idx];
        agent->respawn_timestep = -1;
        agent->respawn_count = 0;
        agent->collided_before_goal = 0;
        agent->goals_reached_this_episode = 0.0f;
        // Initialize to 1 because there is one goal in the data file
        agent->goals_sampled_this_episode = 1.0f;
        agent->current_goal_reached = 0;
        agent->metrics_array[COLLISION_IDX] = 0.0f;
        agent->metrics_array[OFFROAD_IDX] = 0.0f;
        agent->metrics_array[REACHED_GOAL_IDX] = 0.0f;
        agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f;
        agent->stopped = 0;
        agent->removed = 0;

        if (env->goal_behavior == GOAL_GENERATE_NEW) {
            agent->goal_position_x = agent->init_goal_x;
            agent->goal_position_y = agent->init_goal_y;
            agent->goal_position_z = agent->init_goal_z;
        }

        compute_agent_metrics(env, agent_idx);
    }
    compute_observations(env);
}

void c_step(Drive *env) {
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
    env->timestep++;

    int originals_remaining = 0;
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        // Keep flag true if there is at least one agent that has not been respawned yet
        if (env->agents[agent_idx].respawn_count == 0) {
            originals_remaining = 1;
            break;
        }
    }

    if (env->timestep == env->episode_length || (!originals_remaining && env->termination_mode == 1)) {
        add_log(env);
        c_reset(env);
        return;
    }

    // Move static experts
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        int expert_idx = env->expert_static_agent_indices[i];
        if (env->agents[expert_idx].sim_x == INVALID_POSITION)
            continue;
        move_expert(env, env->actions, expert_idx);
    }
    // Process actions for all active agents
    for (int i = 0; i < env->active_agent_count; i++) {
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];
        env->agents[agent_idx].collision_state = 0;
        env->agents[agent_idx].aabb_collision_state = 0;
        float prev_vx = env->agents[agent_idx].sim_vx;
        float prev_vy = env->agents[agent_idx].sim_vy;

        move_dynamics(env, i, agent_idx);

        // Tiny jerk penalty for smoothness
        if (env->dynamics_model == CLASSIC) {
            float delta_vx = env->agents[agent_idx].sim_vx - prev_vx;
            float delta_vy = env->agents[agent_idx].sim_vy - prev_vy;
            float jerk_penalty = -0.0002f * sqrtf(delta_vx * delta_vx + delta_vy * delta_vy) / env->dt;
            env->rewards[i] += jerk_penalty;
            env->logs[i].episode_return += jerk_penalty;
        }
    }

    // Compute rewards
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        env->agents[agent_idx].collision_state = 0;
        env->agents[agent_idx].aabb_collision_state = 0;
        compute_agent_metrics(env, agent_idx);
        int collision_state = env->agents[agent_idx].collision_state;

        if (collision_state > 0) {
            if (collision_state == VEHICLE_COLLISION) {
                env->rewards[i] += env->reward_vehicle_collision;
                env->logs[i].episode_return += env->reward_vehicle_collision;
                env->logs[i].collision_rate = 1.0f;
                env->logs[i].collisions_per_agent += 1.0f;
            } else if (collision_state == OFFROAD) {
                env->rewards[i] += env->reward_offroad_collision;
                env->logs[i].episode_return += env->reward_offroad_collision;
                env->logs[i].offroad_rate = 1.0f;
                env->logs[i].offroad_per_agent += 1.0f;
            }

            env->agents[agent_idx].collided_before_goal = 1;
        }

        float distance_to_goal =
            relative_distance_3d(env->agents[agent_idx].sim_x, env->agents[agent_idx].sim_y,
                                 env->agents[agent_idx].sim_z, env->agents[agent_idx].goal_position_x,
                                 env->agents[agent_idx].goal_position_y, env->agents[agent_idx].goal_position_z);

        float current_speed = sqrtf(env->agents[agent_idx].sim_vx * env->agents[agent_idx].sim_vx +
                                    env->agents[agent_idx].sim_vy * env->agents[agent_idx].sim_vy);

        // Reward agent if it is within X meters of goal and speed is below threshold
        bool within_distance = distance_to_goal < env->goal_radius;
        bool within_speed = current_speed <= env->goal_speed;

        if (within_distance && within_speed && !env->agents[agent_idx].current_goal_reached) {
            if (env->goal_behavior == GOAL_RESPAWN && env->agents[agent_idx].respawn_timestep != -1) {
                env->rewards[i] += env->reward_goal_post_respawn;
                env->logs[i].episode_return += env->reward_goal_post_respawn;
                env->agents[agent_idx].current_goal_reached = 1;
            } else if (env->goal_behavior == GOAL_GENERATE_NEW && (!env->agents[agent_idx].current_goal_reached)) {
                env->rewards[i] += env->reward_goal;
                env->logs[i].episode_return += env->reward_goal;
                sample_new_goal(env, agent_idx);
                env->agents[agent_idx].current_goal_reached = 0;
                env->agents[agent_idx].goals_reached_this_episode += 1.0f;
            } else { // Zero out the velocity so that the agent stops at the goal
                env->rewards[i] = env->reward_goal;
                env->logs[i].episode_return = env->reward_goal;
                env->agents[agent_idx].stopped = 1;
                env->agents[agent_idx].sim_vx = env->agents[agent_idx].sim_vy = 0.0f;
                env->agents[agent_idx].goals_reached_this_episode += 1.0f;
            }
            env->agents[agent_idx].metrics_array[REACHED_GOAL_IDX] = 1.0f;
            env->logs[i].speed_at_goal = current_speed;
        }

        int lane_aligned = env->agents[agent_idx].metrics_array[LANE_ALIGNED_IDX];
        env->logs[i].lane_alignment_rate = lane_aligned;
    }

    if (env->goal_behavior == GOAL_RESPAWN) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            int reached_goal = env->agents[agent_idx].metrics_array[REACHED_GOAL_IDX];
            if (reached_goal) {
                respawn_agent(env, agent_idx);
                env->agents[agent_idx].respawn_count++;
            }
        }
    } else if (env->goal_behavior == GOAL_STOP) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            int reached_goal = env->agents[agent_idx].metrics_array[REACHED_GOAL_IDX];
            if (reached_goal) {
                env->agents[agent_idx].stopped = 1;
                env->agents[agent_idx].sim_vx = env->agents[agent_idx].sim_vy = 0.0f;
            }
        }
    }

    compute_observations(env);
}

// ========================================
// Render Functions (eventually will move to render.h)
// ========================================

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
    int car_assignments[MAX_AGENTS]; // To keep car model assignments consistent per vehicle
    Vector3 default_camera_position;
    Vector3 default_camera_target;
};

Client *make_client(Drive *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));
    client->width = 1280;
    client->height = 704;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(client->width, client->height, "PufferDrive");
    SetTargetFPS(30);
    client->puffers = LoadTexture("resources/puffers_128.png");
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
    // Get initial target position from first active agent
    Vector3 target_pos = {
        0,
        0, // Y is up
        1  // Z is depth
    };

    // Set up camera to look at target from above and behind
    client->default_camera_position = (Vector3){
        0,      // Same X as target
        120.0f, // 20 units above target
        40.0f   // 20 units behind target
    };
    client->default_camera_target = target_pos;
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3){0.0f, -1.0f, 0.0f}; // Y is up
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;
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

    int ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int max_obs = ego_dim + PARTNER_FEATURES * (MAX_AGENTS - 1) + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS;
    float (*observations)[max_obs] = (float (*)[max_obs])env->observations;
    float *agent_obs = &observations[agent_index][0];
    // self
    int active_idx = env->active_agent_indices[agent_index];
    float heading_self_x = env->agents[active_idx].heading_x;
    float heading_self_y = env->agents[active_idx].heading_y;
    float px = env->agents[active_idx].sim_x;
    float py = env->agents[active_idx].sim_y;
    float pz = env->agents[active_idx].sim_z;
    // draw goal
    float goal_x = agent_obs[0] * 200;
    float goal_y = agent_obs[1] * 200;
    float goal_z = agent_obs[2] * 200;

    if (mode == 0) {
        DrawSphere((Vector3){goal_x, goal_y, goal_z}, 0.5f, LIGHTGREEN);
        DrawCircle3D((Vector3){goal_x, goal_y, goal_z}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f,
                     Fade(LIGHTGREEN, 0.3f));
    }

    if (mode == 1) {
        float goal_x_world = px + (goal_x * heading_self_x - goal_y * heading_self_y);
        float goal_y_world = py + (goal_x * heading_self_y + goal_y * heading_self_x);
        float goal_z_world = pz + goal_z;
        DrawSphere((Vector3){goal_x_world, goal_y_world, goal_z_world}, 0.5f, LIGHTGREEN);
        DrawCircle3D((Vector3){goal_x_world, goal_y_world, goal_z_world}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f,
                     Fade(LIGHTGREEN, 0.3f));
    }
    // First draw other agent observations
    int obs_idx = ego_dim; // Start after ego obs
    for (int j = 0; j < MAX_AGENTS - 1; j++) {
        if (agent_obs[obs_idx] == 0 || agent_obs[obs_idx + 1] == 0) {
            obs_idx += 8; // Move to next agent observation
            continue;
        }
        // Draw position of other agents
        float x = agent_obs[obs_idx] * 50;
        float y = agent_obs[obs_idx + 1] * 50;
        float z = agent_obs[obs_idx + 2] * 50;
        if (lasers && mode == 0) {
            DrawLine3D((Vector3){0, 0, 0}, (Vector3){x, y, z}, ORANGE);
        }

        float partner_x = px + (x * heading_self_x - y * heading_self_y);
        float partner_y = py + (x * heading_self_y + y * heading_self_x);
        float partner_z = pz + z;
        if (lasers && mode == 1) {
            DrawLine3D((Vector3){px, py, pz}, (Vector3){partner_x, partner_y, partner_z}, ORANGE);
        }

        float half_width = 0.5 * agent_obs[obs_idx + 3] * MAX_VEH_WIDTH;
        float half_len = 0.5 * agent_obs[obs_idx + 4] * MAX_VEH_LEN;
        float theta_x = agent_obs[obs_idx + 5];
        float theta_y = agent_obs[obs_idx + 6];
        float partner_angle = atan2f(theta_y, theta_x);
        float cos_heading = cosf(partner_angle);
        float sin_heading = sinf(partner_angle);
        Vector3 corners[4] = {
            (Vector3){x + (half_len * cos_heading - half_width * sin_heading),
                      y + (half_len * sin_heading + half_width * cos_heading), z},
            (Vector3){x + (half_len * cos_heading + half_width * sin_heading),
                      y + (half_len * sin_heading - half_width * cos_heading), z},
            (Vector3){x + (-half_len * cos_heading + half_width * sin_heading),
                      y + (-half_len * sin_heading - half_width * cos_heading), z},
            (Vector3){x + (-half_len * cos_heading - half_width * sin_heading),
                      y + (-half_len * sin_heading + half_width * cos_heading), z},
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
                float lz = corners[j].z;

                world_corners[j].x = px + (lx * heading_self_x - ly * heading_self_y);
                world_corners[j].y = py + (lx * heading_self_y + ly * heading_self_x);
                world_corners[j].z = pz + lz;
            }
            for (int j = 0; j < 4; j++) {
                DrawLine3D(world_corners[j], world_corners[(j + 1) % 4], ORANGE);
            }
        }

        // draw an arrow above the car pointing in the direction that the partner is going
        float arrow_length = 4.5f;
        float arrow_x = x + arrow_length * cosf(partner_angle);
        float arrow_y = y + arrow_length * sinf(partner_angle);
        float arrow_z = z;
        float arrow_x_world;
        float arrow_y_world;
        float arrow_z_world;
        if (mode == 0) {
            DrawLine3D((Vector3){x, y, z}, (Vector3){arrow_x, arrow_y, arrow_z}, PUFF_WHITE);
        }
        if (mode == 1) {
            arrow_x_world = px + (arrow_x * heading_self_x - arrow_y * heading_self_y);
            arrow_y_world = py + (arrow_x * heading_self_y + arrow_y * heading_self_x);
            arrow_z_world = pz + arrow_z;
            DrawLine3D((Vector3){partner_x, partner_y, partner_z},
                       (Vector3){arrow_x_world, arrow_y_world, arrow_z_world}, PUFF_WHITE);
        }
        // Calculate perpendicular offsets for arrow head
        float arrow_size = 0.8f; // Size of the arrow head
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
            float arrow_z_end = arrow_z;

            // Draw the two lines forming the arrow head
            if (mode == 0) {
                DrawLine3D((Vector3){arrow_x, arrow_y, arrow_z}, (Vector3){arrow_x_end1, arrow_y_end1, arrow_z_end},
                           PUFF_WHITE);
                DrawLine3D((Vector3){arrow_x, arrow_y, arrow_z}, (Vector3){arrow_x_end2, arrow_y_end2, arrow_z_end},
                           PUFF_WHITE);
            }

            if (mode == 1) {
                float arrow_x_end1_world = px + (arrow_x_end1 * heading_self_x - arrow_y_end1 * heading_self_y);
                float arrow_y_end1_world = py + (arrow_x_end1 * heading_self_y + arrow_y_end1 * heading_self_x);
                float arrow_x_end2_world = px + (arrow_x_end2 * heading_self_x - arrow_y_end2 * heading_self_y);
                float arrow_y_end2_world = py + (arrow_x_end2 * heading_self_y + arrow_y_end2 * heading_self_x);
                float arrow_z_end_world = pz + arrow_z_end;
                DrawLine3D((Vector3){arrow_x_world, arrow_y_world, arrow_z_world},
                           (Vector3){arrow_x_end1_world, arrow_y_end1_world, arrow_z_end_world}, PUFF_WHITE);
                DrawLine3D((Vector3){arrow_x_world, arrow_y_world, arrow_z_world},
                           (Vector3){arrow_x_end2_world, arrow_y_end2_world, arrow_z_end_world}, PUFF_WHITE);
            }
        }

        obs_idx += PARTNER_FEATURES; // Move to next agent observation (8 values per agent)
    }
    // Then draw map observations
    int map_start_idx = ego_dim + PARTNER_FEATURES * (MAX_AGENTS - 1); // Start after agent observations
    for (int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {          // Loop through potential map entities
        int entity_idx = map_start_idx + k * 8;
        if (agent_obs[entity_idx] == 0 && agent_obs[entity_idx + 1] == 0) {
            continue;
        }
        Color lineColor = BLUE; // Default color
        int entity_type = (int)agent_obs[entity_idx + 7];
        // Choose color based on entity type
        if (entity_type + 4 != ROAD_EDGE) {
            continue;
        }
        lineColor = PUFF_CYAN;
        // For road segments, draw line between start and end points
        float x_middle = agent_obs[entity_idx] * 50;
        float y_middle = agent_obs[entity_idx + 1] * 50;
        float z_middle = agent_obs[entity_idx + 2] * 50;
        float rel_angle_x = (agent_obs[entity_idx + 5]);
        float rel_angle_y = (agent_obs[entity_idx + 6]);
        float rel_angle = atan2f(rel_angle_y, rel_angle_x);
        float segment_length = agent_obs[entity_idx + 3] * MAX_ROAD_SEGMENT_LENGTH;
        // Calculate endpoint using the relative angle directly
        // Calculate endpoint directly
        float x_start = x_middle - segment_length * cosf(rel_angle);
        float y_start = y_middle - segment_length * sinf(rel_angle);
        float x_end = x_middle + segment_length * cosf(rel_angle);
        float y_end = y_middle + segment_length * sinf(rel_angle);

        if (lasers && mode == 0) {
            DrawLine3D((Vector3){0, 0, 0}, (Vector3){x_middle, y_middle, z_middle}, lineColor);
        }

        if (mode == 1) {
            float x_middle_world = px + (x_middle * heading_self_x - y_middle * heading_self_y);
            float y_middle_world = py + (x_middle * heading_self_y + y_middle * heading_self_x);
            float x_start_world = px + (x_start * heading_self_x - y_start * heading_self_y);
            float y_start_world = py + (x_start * heading_self_y + y_start * heading_self_x);
            float x_end_world = px + (x_end * heading_self_x - y_end * heading_self_y);
            float y_end_world = py + (x_end * heading_self_y + y_end * heading_self_x);
            DrawCube((Vector3){x_middle_world, y_middle_world, pz}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start_world, y_start_world, pz}, (Vector3){x_end_world, y_end_world, pz}, BLUE);
            if (lasers)
                DrawLine3D((Vector3){px, py, pz}, (Vector3){x_middle_world, y_middle_world, pz}, lineColor);
        }
        if (mode == 0) {
            DrawCube((Vector3){x_middle, y_middle, z_middle}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start, y_start, z_middle}, (Vector3){x_end, y_end, z_middle}, BLUE);
        }
    }
}

void draw_road_edge(Drive *env, float start_x, float start_y, float end_x, float end_y, float start_z, float end_z) {
    Color CURB_TOP = (Color){220, 220, 220, 255};  // Top surface - lightest
    Color CURB_SIDE = (Color){180, 180, 180, 255}; // Side faces - medium
    Color CURB_BOTTOM = (Color){160, 160, 160, 255};
    // Calculate curb dimensions
    float curb_height = 0.5f; // Height of the curb
    float curb_width = 0.3f;  // Width/thickness of the curb

    Vector3 direction = {
        end_x - start_x,
        end_y - start_y,
    };

    // Calculate length of the segment
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);

    // Normalize direction vector
    Vector3 normalized_dir = {
        direction.x / length,
        direction.y / length,
    };

    // Calculate perpendicular vector for width
    Vector3 perpendicular = {-normalized_dir.y, normalized_dir.x, 0.0f};

    // Calculate the four bottom corners of the curb
    Vector3 b1 = {start_x - perpendicular.x * curb_width / 2, start_y - perpendicular.y * curb_width / 2, start_z};
    Vector3 b2 = {start_x + perpendicular.x * curb_width / 2, start_y + perpendicular.y * curb_width / 2, start_z};
    Vector3 b3 = {end_x + perpendicular.x * curb_width / 2, end_y + perpendicular.y * curb_width / 2, end_z};
    Vector3 b4 = {end_x - perpendicular.x * curb_width / 2, end_y - perpendicular.y * curb_width / 2, end_z};

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

void draw_scene(Drive *env, Client *client, int mode, int obs_only, int lasers, int show_grid) {

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

    // Draw agents
    for (int i = 0; i < env->num_objects; i++) {
        Agent *agent = &env->agents[i];
        if (agent->type == VEHICLE || agent->type == PEDESTRIAN || agent->type == CYCLIST) {
            // Check if this vehicle is an active agent
            bool is_active_agent = false;
            bool is_static_agent = false;
            int agent_index = -1;
            for (int j = 0; j < env->active_agent_count; j++) {
                if (env->active_agent_indices[j] == i) {
                    is_active_agent = true;
                    agent_index = j;
                    break;
                }
            }
            for (int j = 0; j < env->static_agent_count; j++) {
                if (env->static_agent_indices[j] == i) {
                    is_static_agent = true;
                    break;
                }
            }
            // HIDE CARS ON RESPAWN - IMPORTANT TO KNOW VISUAL SETTING
            if ((!is_active_agent && !is_static_agent) || agent->respawn_timestep != -1) {
                continue;
            }
            Vector3 position = (Vector3){agent->sim_x, agent->sim_y, agent->sim_z};
            float heading = agent->sim_heading;
            // Create size vector
            Vector3 size = {agent->sim_length, agent->sim_width, agent->sim_height};

            bool is_expert = (!is_active_agent) && (agent->mark_as_expert == 1);

            // Save current transform
            if (mode == 1) {
                float cos_heading = agent->heading_x;
                float sin_heading = agent->heading_y;

                // Calculate half dimensions
                float half_len = agent->sim_length * 0.5f;
                float half_width = agent->sim_width * 0.5f;

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

                if (agent_index == env->human_agent_idx && !env->agents[agent_index].metrics_array[REACHED_GOAL_IDX]) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }

                if ((obs_only || IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx) {
                    continue;
                }

                // --- Draw the car  ---
                Color car_color = GRAY; // default for static
                if (is_expert)
                    car_color = GOLD; // expert replay
                if (is_active_agent)
                    car_color = BLUE; // policy-controlled
                if (is_active_agent && agent->aabb_collision_state > 0)
                    car_color = LIGHTGREEN;
                if (is_active_agent && agent->collision_state > 0)
                    car_color = RED;
                rlSetLineWidth(3.0f);
                for (int j = 0; j < 4; j++) {
                    DrawLine3D(corners[j], corners[(j + 1) % 4], car_color);
                }
                // --- Draw a heading arrow pointing forward ---
                Vector3 arrowStart = position;
                Vector3 arrowEnd = {position.x + cos_heading * half_len * 1.5f, // extend arrow beyond car
                                    position.y + sin_heading * half_len * 1.5f, position.z};

                DrawLine3D(arrowStart, arrowEnd, car_color);
                DrawSphere(arrowEnd, 0.2f, car_color); // arrow tip

            } else { // Agent view
                rlPushMatrix();
                // Translate to position, rotate around Y axis, then draw
                rlTranslatef(position.x, position.y, position.z);
                rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f); // Convert radians to degrees

                // Select car model
                Model car_model = client->cars[i % 6]; // Default: cycle through all 6 car sprites

                if (agent_index == env->human_agent_idx) {
                    car_model = client->cars[0]; // Ego agent always uses red car (cars[0])
                } else if (is_active_agent) {
                    car_model = client->cars[(i % 5) + 1];
                    if (agent->aabb_collision_state > 0) {
                        car_model = client->cars[4]; // AABB Collided agents use green car
                    }
                    if (agent->collision_state > 0) {
                        car_model = client->cars[0]; // Collided agents use red
                    }
                }
                // Draw obs for selected agent index
                if (agent_index == env->human_agent_idx &&
                    (!env->agents[agent_index].metrics_array[REACHED_GOAL_IDX] ||
                     env->goal_behavior == GOAL_GENERATE_NEW || env->goal_behavior == GOAL_STOP)) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }

                // Draw cube for cars static and active
                // Calculate scale factors based on desired size and model dimensions
                BoundingBox bounds = GetModelBoundingBox(car_model);
                Vector3 model_size = {bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y,
                                      bounds.max.z - bounds.min.z};
                Vector3 scale = {size.x / model_size.x, size.y / model_size.y, size.z / model_size.z};
                // if((obs_only ||  IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx){
                //     rlPopMatrix();
                //     continue;
                // }
                if (agent->type == CYCLIST) {
                    scale = (Vector3){0.01, 0.01, 0.01};
                    car_model = client->cyclist;
                }
                if (agent->type == PEDESTRIAN) {
                    scale = (Vector3){2, 2, 2};
                    car_model = client->pedestrian;
                }
                DrawModelEx(car_model, (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, WHITE);
                {
                    float half_len = agent->sim_length * 0.5f;
                    float half_width = agent->sim_width * 0.5f;
                    Vector3 corners[4] = {
                        (Vector3){half_len, -half_width, 0},  // Front-left
                        (Vector3){half_len, half_width, 0},   // Front-right
                        (Vector3){-half_len, half_width, 0},  // Back-right
                        (Vector3){-half_len, -half_width, 0}, // Back-left
                    };
                    Color wire_color = GRAY; // static
                    if (!is_active_agent && agent->mark_as_expert == 1)
                        wire_color = GOLD; // expert replay
                    if (is_active_agent)
                        wire_color = BLUE; // policy
                    if (is_active_agent && agent->aabb_collision_state > 0)
                        wire_color = LIGHTGREEN;
                    if (is_active_agent && agent->collision_state > 0)
                        wire_color = RED;
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
            if (!is_active_agent || agent->sim_valid == 0) {
                continue;
            }
            if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
                DrawSphere(
                    (Vector3){
                        agent->goal_position_x,
                        agent->goal_position_y,
                        agent->goal_position_z,
                    },
                    0.5f, DARKGREEN);

                DrawCircle3D(
                    (Vector3){
                        agent->goal_position_x,
                        agent->goal_position_y,
                        agent->goal_position_z,
                    },
                    env->goal_radius, (Vector3){0, 0, 1}, 90.0f, Fade(LIGHTGREEN, 0.3f));
            }
        }
    }

    // Draw road elements
    for (int i = 0; i < env->num_roads; i++) {
        RoadMapElement *road = &env->road_elements[i];
        for (int j = 0; j < road->segment_length - 1; j++) {
            Vector3 start = {road->x[j], road->y[j], road->z[j]};
            Vector3 end = {road->x[j + 1], road->y[j + 1], road->z[j + 1]};
            Color lineColor = GRAY;
            if (road->type == ROAD_LANE)
                lineColor = Fade(SOFT_YELLOW, 0.25f);
            else if (road->type == ROAD_LINE)
                lineColor = WHITE;
            else if (road->type == ROAD_EDGE)
                lineColor = WHITE;
            else if (road->type == DRIVEWAY)
                lineColor = RED;

            if (!IsKeyDown(KEY_LEFT_CONTROL) && obs_only == 0) {
                if (road->type == ROAD_EDGE) {
                    draw_road_edge(env, start.x, start.y, end.x, end.y, start.z, end.z);
                } else if (road->type == ROAD_LANE || road->type == ROAD_LINE) {
                    // Draw road lanes and lines as purple lines
                    rlSetLineWidth(2.0f);
                    DrawLine3D(start, end, lineColor);
                }
            }
        }
    }

    EndMode3D();

    // Draw track indices for the tracks to predict
    if (mode == 1 && env->control_mode == CONTROL_WOSAC) {
        float map_height = env->grid_map->top_left_y - env->grid_map->bottom_right_y;
        float pixels_per_world_unit = client->height / map_height;

        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            Agent *agent = &env->agents[agent_idx];
            // Ignore respawned agents
            if (agent->respawn_timestep != -1) {
                continue;
            }
            int womd_track_idx = env->tracks_to_predict_indices[i];

            float raw_x = -agent->sim_x * pixels_per_world_unit;
            float raw_y = agent->sim_y * pixels_per_world_unit;

            int screen_x = (int)raw_x + client->width / 2 + 20;
            int screen_y = (int)raw_y + client->height / 2 - 25;

            if (screen_x >= 0 && screen_x <= client->width && screen_y >= 0 && screen_y <= client->height) {
                char text[32];
                snprintf(text, sizeof(text), "%d", womd_track_idx);
                int text_width = MeasureText(text, 20);
                DrawText(text, screen_x - text_width / 2, screen_y, 20, PUFF_WHITE);
            }
        }
    }
}

void c_render(Drive *env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client *client = env->client;
    BeginDrawing();
    Color road = (Color){35, 35, 37, 255};
    ClearBackground(road);
    BeginMode3D(client->camera);
    handle_camera_controls(env->client);
    draw_scene(env, client, 0, 0, 0, 0);

    if (IsKeyPressed(KEY_TAB)) {
        env->human_agent_idx = (env->human_agent_idx + 1) % env->active_agent_count;
    }

    // Draw debug info
    DrawText(TextFormat("Camera Position: (%.2f, %.2f, %.2f)", client->camera.position.x, client->camera.position.y,
                        client->camera.position.z),
             10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Camera Target: (%.2f, %.2f, %.2f)", client->camera.target.x, client->camera.target.y,
                        client->camera.target.z),
             10, 30, 20, PUFF_WHITE);
    DrawText(TextFormat("Timestep: %d", env->timestep), 10, 50, 20, PUFF_WHITE);

    int human_idx = env->active_agent_indices[env->human_agent_idx];
    DrawText(TextFormat("Controlling Agent: %d", env->human_agent_idx), 10, 70, 20, PUFF_WHITE);
    DrawText(TextFormat("Agent Index: %d", human_idx), 10, 90, 20, PUFF_WHITE);

    // Display current action values - yellow when controlling, white otherwise
    Color action_color = IsKeyDown(KEY_LEFT_SHIFT) ? YELLOW : PUFF_WHITE;

    if (env->action_type == 0) { // discrete
        int *action_array = (int *)env->actions;
        int action_val = action_array[env->human_agent_idx];

        if (env->dynamics_model == CLASSIC) {
            int num_steer = 13;
            int accel_idx = action_val / num_steer;
            int steer_idx = action_val % num_steer;
            float accel_value = ACCELERATION_VALUES[accel_idx];
            float steer_value = STEERING_VALUES[steer_idx];

            DrawText(TextFormat("Acceleration: %.2f m/s^2", accel_value), 10, 110, 20, action_color);
            DrawText(TextFormat("Steering: %.3f", steer_value), 10, 130, 20, action_color);
        } else if (env->dynamics_model == JERK) {
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
        DrawText(TextFormat("Acceleration: %.2f", action_array_f[env->human_agent_idx][0]), 10, 110, 20, action_color);
        DrawText(TextFormat("Steering: %.2f", action_array_f[env->human_agent_idx][1]), 10, 130, 20, action_color);
    }

    // Show key press status
    int status_y = 150;
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
        DrawText("[shift pressed]", 10, status_y, 20, YELLOW);
        status_y += 20;
    }
    if (IsKeyDown(KEY_SPACE)) {
        DrawText("[space pressed]", 10, status_y, 20, YELLOW);
        status_y += 20;
    }
    if (IsKeyDown(KEY_LEFT_CONTROL)) {
        DrawText("[ctrl pressed]", 10, status_y, 20, YELLOW);
        status_y += 20;
    }

    // Controls help
    DrawText("Controls: SHIFT + W/S - Accelerate/Brake, SHIFT + A/D - Steer, TAB - Switch Agent", 10,
             client->height - 30, 20, PUFF_WHITE);

    DrawText(TextFormat("Grid Rows: %d", env->grid_map->grid_rows), 10, status_y, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Cols: %d", env->grid_map->grid_cols), 10, status_y + 20, 20, PUFF_WHITE);
    EndDrawing();
}

void close_client(Client *client) {
    for (int i = 0; i < 6; i++) {
        UnloadModel(client->cars[i]);
    }
    UnloadTexture(client->puffers);
    CloseWindow();
    free(client);
}

// ========================================
// Other Functions (things that will be refactored into other functions)
// ========================================

int check_lane_aligned(Agent *car, RoadMapElement *lane, int geometry_idx) {
    // Validate lane geometry length
    if (!lane || lane->segment_length < 2)
        return 0;

    // Clamp geometry index to valid segment range [0, segment_length-2]
    if (geometry_idx < 0)
        geometry_idx = 0;
    if (geometry_idx >= lane->segment_length - 1)
        geometry_idx = lane->segment_length - 2;

    // Compute local lane segment heading
    float heading_x1, heading_y1;
    if (geometry_idx > 0) {
        heading_x1 = lane->x[geometry_idx] - lane->x[geometry_idx - 1];
        heading_y1 = lane->y[geometry_idx] - lane->y[geometry_idx - 1];
    } else {
        // For first segment, just use the forward direction
        heading_x1 = lane->x[geometry_idx + 1] - lane->x[geometry_idx];
        heading_y1 = lane->y[geometry_idx + 1] - lane->y[geometry_idx];
    }

    float heading_x2 = lane->x[geometry_idx + 1] - lane->x[geometry_idx];
    float heading_y2 = lane->y[geometry_idx + 1] - lane->y[geometry_idx];

    float heading_1 = atan2f(heading_y1, heading_x1);
    float heading_2 = atan2f(heading_y2, heading_x2);
    float heading = (heading_1 + heading_2) / 2.0f;

    // Normalize to [-pi, pi]
    if (heading > M_PI)
        heading -= 2.0f * M_PI;
    if (heading < -M_PI)
        heading += 2.0f * M_PI;

    // Compute heading difference
    float car_heading = car->sim_heading; // radians
    float heading_diff = fabsf(car_heading - heading);

    if (heading_diff > M_PI)
        heading_diff = 2.0f * M_PI - heading_diff;

    // within 15 degrees
    return (heading_diff < (M_PI / 12.0f)) ? 1 : 0;
}

float point_to_segment_distance_2d(float px, float py, float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;

    if (dx == 0 && dy == 0) {
        // The segment is a point
        return sqrtf((px - x1) * (px - x1) + (py - y1) * (py - y1));
    }

    // Calculate the t that minimizes the distance
    float t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);

    // Clamp t to the segment
    if (t < 0)
        t = 0;
    else if (t > 1)
        t = 1;

    // Find the closest point on the segment
    float closestX = x1 + t * dx;
    float closestY = y1 + t * dy;

    // Return the distance from p to the closest point
    return sqrtf((px - closestX) * (px - closestX) + (py - closestY) * (py - closestY));
}

void init_goal_positions(Drive *env) {
    for (int x = 0; x < env->active_agent_count; x++) {
        int agent_idx = env->active_agent_indices[x];
        if (env->init_mode == RANDOM_AGENTS) {
            sample_new_goal(env, agent_idx);
        }
        Agent *agent = &env->agents[agent_idx];
        agent->init_goal_x = agent->goal_position_x;
        agent->init_goal_y = agent->goal_position_y;
        agent->init_goal_z = agent->goal_position_z;
    }
}

float clipSpeed(float speed) {
    const float maxSpeed = MAX_SPEED;
    if (speed > maxSpeed)
        return maxSpeed;
    if (speed < -maxSpeed)
        return -maxSpeed;
    return speed;
}

float normalize_value(float value, float min, float max) { return (value - min) / (max - min); }

void sample_new_goal(Drive *env, int agent_idx) {
    // Samples a new goal position based on the existing road lane points
    Agent *agent = &env->agents[agent_idx];
    float best_x = agent->sim_x;
    float best_y = agent->sim_y;
    float best_z = agent->sim_z;
    float best_distance_error = 1e30f;

    // Sample points from all road lanes
    for (int i = 0; i < env->num_roads; i++) {
        RoadMapElement *lane = &env->road_elements[i];
        if (lane->type != ROAD_LANE)
            continue;

        // Check every point in the lane
        for (int j = 0; j < lane->segment_length; j++) {
            float point_x = lane->x[j];
            float point_y = lane->y[j];
            float point_z = lane->z[j];

            // Calculate vector from agent to point
            float to_point_x = point_x - agent->sim_x;
            float to_point_y = point_y - agent->sim_y;

            // Check if point is ahead of agent
            float dot = to_point_x * agent->heading_x + to_point_y * agent->heading_y;
            float mod_to_pt = sqrtf(to_point_x * to_point_x + to_point_y * to_point_y);
            float mod_heading = atan2f(agent->heading_y, agent->heading_x);
            float cos_theta = dot / (mod_to_pt * mod_heading);
            if (cos_theta <= 0.0f) // Maybe increase threshold to have points in the direction of travel but not
                                   // necessarily perfectly ahead?
                continue;

            // Calculate distance to point
            float distance = sqrtf(to_point_x * to_point_x + to_point_y * to_point_y);

            // Find point closest to target distance
            float distance_error = fabsf(distance - env->goal_target_distance);
            if (distance_error < best_distance_error) {
                best_distance_error = distance_error;
                best_x = point_x;
                best_y = point_y;
                best_z = point_z;
            }
        }
    }

    // If no valid goal found, use another agent's initial goal
    // raise_error_with_message(ERROR_UNHANDLED_CASE, "No valid goal found for agent %d at (x,y,z)=(%f,%f,%f), using
    // another agent's initial goal", agent_idx, agent->sim_x, agent->sim_y, agent->sim_z);
    if (best_distance_error >= 1e30f && env->active_agent_count > 1) {
        int other_idx = env->active_agent_indices[(agent_idx + 1) % env->active_agent_count];
        best_x = env->agents[other_idx].init_goal_x;
        best_y = env->agents[other_idx].init_goal_y;
        best_z = env->agents[other_idx].init_goal_z;
    }

    agent->goal_position_x = best_x;
    agent->goal_position_y = best_y;
    agent->goal_position_z = best_z;
    agent->goals_sampled_this_episode += 1.0f;
}
