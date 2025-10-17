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

// Trajectory Length
#define TRAJECTORY_LENGTH 91

// Actions
#define NOOP 0

// Dynamics Models
#define CLASSIC 0
#define INVERTIBLE_BICYLE 1
#define DELTA_LOCAL 2
#define STATE_DYNAMICS 3

// collision state
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2

// Metrics array indices
#define COLLISION_IDX 0
#define OFFROAD_IDX 1
#define REACHED_GOAL_IDX 2
#define LANE_ALIGNED_IDX 3
#define AVG_DISPLACEMENT_ERROR_IDX 4

// grid cell size
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL 30    // Depends on resolution of data Formula: 3 * (2 + GRID_CELL_SIZE*sqrt(2)/resolution) => For each entity type in gridmap, diagonal poly-lines -> sqrt(2), include diagonal ends -> 2

// Max road segment observation entities
#define MAX_ROAD_SEGMENT_OBSERVATIONS 200
#define MAX_CARS 64
// Observation Space Constants
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

// Acceleration Values
static const float ACCELERATION_VALUES[7] = {-4.0000f, -2.6670f, -1.3330f, -0.0000f,  1.3330f,  2.6670f,  4.0000f};
// static const float STEERING_VALUES[13] = {-3.1420f, -2.6180f, -2.0940f, -1.5710f, -1.0470f, -0.5240f,  0.0000f,  0.5240f,
//          1.0470f,  1.5710f,  2.0940f,  2.6180f,  3.1420f};
static const float STEERING_VALUES[13] = {-1.000f, -0.833f, -0.667f, -0.500f, -0.333f, -0.167f, 0.000f, 0.167f, 0.333f, 0.500f, 0.667f, 0.833f, 1.000f};
static const float offsets[4][2] = {
        {-1, 1},  // top-left
        {1, 1},   // top-right
        {1, -1},  // bottom-right
        {-1, -1}  // bottom-left
    };

static const int collision_offsets[25][2] = {
    {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},  // Top row
    {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},  // Second row
    {-2,  0}, {-1,  0}, {0,  0}, {1,  0}, {2,  0},  // Middle row (including center)
    {-2,  1}, {-1,  1}, {0,  1}, {1,  1}, {2,  1},  // Fourth row
    {-2,  2}, {-1,  2}, {0,  2}, {1,  2}, {2,  2}   // Bottom row
};

struct timespec ts;

typedef struct Drive Drive;
typedef struct Client Client;
typedef struct Log Log;

struct Log {
    float episode_return;
    float episode_length;
    float perf;
    float score;
    float offroad_rate;
    float collision_rate;
    float clean_collision_rate;
    float completion_rate;
    float dnf_rate;
    float n;
    float lane_alignment_rate;
    float avg_displacement_error;
};

typedef struct Entity Entity;
struct Entity {
    int type;
    int array_size;
    float* traj_x;
    float* traj_y;
    float* traj_z;
    float* traj_vx;
    float* traj_vy;
    float* traj_vz;
    float* traj_heading;
    int* traj_valid;
    float width;
    float length;
    float height;
    float goal_position_x;
    float goal_position_y;
    float goal_position_z;
    int mark_as_expert;
    int collision_state;
    float metrics_array[5]; // metrics_array: [collision, offroad, reached_goal, lane_aligned, avg_displacement_error]
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
    int respawn_timestep;
    int collided_before_goal;
    int reached_goal_this_episode;
    int active_agent;
    float cumulative_displacement;
    int displacement_sample_count;
    float goal_radius;
};

void free_entity(Entity* entity){
    // free trajectory arrays
    free(entity->traj_x);
    free(entity->traj_y);
    free(entity->traj_z);
    free(entity->traj_vx);
    free(entity->traj_vy);
    free(entity->traj_vz);
    free(entity->traj_heading);
    free(entity->traj_valid);
}

float relative_distance(float a, float b){
    float distance = sqrtf(powf(a - b, 2));
    return distance;
}

float relative_distance_2d(float x1, float y1, float x2, float y2){
    float dx = x2 - x1;
    float dy = y2 - y1;
    float distance = sqrtf(dx*dx + dy*dy);
    return distance;
}

float compute_displacement_error(Entity* agent, int timestep) {
    // Check if timestep is within valid range
    if (timestep < 0 || timestep >= agent->array_size) {
        return 0.0f;
    }

    // Check if reference trajectory is valid at this timestep
    if (!agent->traj_valid[timestep]) {
        return 0.0f;
    }

    // Get reference position at current timestep, skip invalid ones
    float ref_x = agent->traj_x[timestep];
    float ref_y = agent->traj_y[timestep];

    if (ref_x == -10000.0f || ref_y == -10000.0f) {
        return 0.0f;
    }

    // Compute deltas: Euclidean distance between actual and reference position
    float dx = agent->x - ref_x;
    float dy = agent->y - ref_y;
    float displacement = sqrtf(dx*dx + dy*dy);

    return displacement;
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
    int* cell_entities_count;  // number of entities in each cell of the GridMap
    GridMapEntity** cells;  // list of gridEntities in each cell of the GridMap

    // Extras/Optimizations
    int vision_range;
    int* neighbor_cache_count; // number of entities in each cells neighbor cache
    GridMapEntity** neighbor_cache_entities; // preallocated array to hold neighbor entities
};

struct Drive {
    Client* client;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    Log log;
    Log* logs;
    int num_agents;
    int active_agent_count;
    int* active_agent_indices;
    int action_type;
    int human_agent_idx;
    Entity* entities;
    int num_entities;
    int num_cars;
    int num_objects;
    int num_roads;
    int static_car_count;
    int* static_car_indices;
    int expert_static_car_count;
    int* expert_static_car_indices;
    int timestep;
    int dynamics_model;
    GridMap* grid_map;
    int* neighbor_offsets;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    float reward_ade;
    char* map_name;
    float world_mean_x;
    float world_mean_y;
    int spawn_immunity_timer;
    float reward_goal_post_respawn;
    float reward_vehicle_collision_post_respawn;
    float goal_radius;
    char* ini_file;
};

void add_log(Drive* env) {
    for(int i = 0; i < env->active_agent_count; i++){
        Entity* e = &env->entities[env->active_agent_indices[i]];
        if(e->reached_goal_this_episode){
            env->log.completion_rate += 1.0f;
        }
        int offroad = env->logs[i].offroad_rate;
        env->log.offroad_rate += offroad;
        int collided = env->logs[i].collision_rate;
        env->log.collision_rate += collided;
        int clean_collided = env->logs[i].clean_collision_rate;
        env->log.clean_collision_rate += clean_collided;
        if(e->reached_goal_this_episode && !e->collided_before_goal){
            env->log.score += 1.0f;
            env->log.perf += 1.0f;
        }
        if(!offroad && !collided && !e->reached_goal_this_episode){
            env->log.dnf_rate += 1.0f;
        }
        int lane_aligned = env->logs[i].lane_alignment_rate;
        env->log.lane_alignment_rate += lane_aligned;
        float displacement_error = env->logs[i].avg_displacement_error;
        env->log.avg_displacement_error += displacement_error;
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        env->log.n += 1;
    }
}

Entity* load_map_binary(const char* filename, Drive* env) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;
    fread(&env->num_objects, sizeof(int), 1, file);
    fread(&env->num_roads, sizeof(int), 1, file);
    env->num_entities = env->num_objects + env->num_roads;
    Entity* entities = (Entity*)malloc(env->num_entities * sizeof(Entity));
    for (int i = 0; i < env->num_entities; i++) {
	// Read base entity data
        fread(&entities[i].type, sizeof(int), 1, file);
        fread(&entities[i].array_size, sizeof(int), 1, file);
        // Allocate arrays based on type
        int size = entities[i].array_size;
        entities[i].traj_x = (float*)malloc(size * sizeof(float));
        entities[i].traj_y = (float*)malloc(size * sizeof(float));
        entities[i].traj_z = (float*)malloc(size * sizeof(float));
        if (entities[i].type == 1 || entities[i].type == 2 || entities[i].type == 3) {  // Object type
            // Allocate arrays for object-specific data
            entities[i].traj_vx = (float*)malloc(size * sizeof(float));
            entities[i].traj_vy = (float*)malloc(size * sizeof(float));
            entities[i].traj_vz = (float*)malloc(size * sizeof(float));
            entities[i].traj_heading = (float*)malloc(size * sizeof(float));
            entities[i].traj_valid = (int*)malloc(size * sizeof(int));
        } else {
            // Roads don't use these arrays
            entities[i].traj_vx = NULL;
            entities[i].traj_vy = NULL;
            entities[i].traj_vz = NULL;
            entities[i].traj_heading = NULL;
            entities[i].traj_valid = NULL;
        }
        // Read array data
        fread(entities[i].traj_x, sizeof(float), size, file);
        fread(entities[i].traj_y, sizeof(float), size, file);
        fread(entities[i].traj_z, sizeof(float), size, file);
        if (entities[i].type == 1 || entities[i].type == 2 || entities[i].type == 3) {  // Object type
            fread(entities[i].traj_vx, sizeof(float), size, file);
            fread(entities[i].traj_vy, sizeof(float), size, file);
            fread(entities[i].traj_vz, sizeof(float), size, file);
            fread(entities[i].traj_heading, sizeof(float), size, file);
            fread(entities[i].traj_valid, sizeof(int), size, file);
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

void set_start_position(Drive* env){
    //InitWindow(800, 600, "GPU Drive");
    //BeginDrawing();
    for(int i = 0; i < env->num_entities; i++){
        int is_active = 0;
        for(int j = 0; j < env->active_agent_count; j++){
            if(env->active_agent_indices[j] == i){
                is_active = 1;
                break;
            }
        }
        Entity* e = &env->entities[i];
        e->x = e->traj_x[0];
        e->y = e->traj_y[0];
        e->z = e->traj_z[0];

        if(e->type >3 || e->type == 0){
            continue;
        }
        if(is_active == 0){
            e->vx = 0;
            e->vy = 0;
            e->vz = 0;
            e->collided_before_goal = 0;
        } else{
            e->vx = e->traj_vx[0];
            e->vy = e->traj_vy[0];
            e->vz = e->traj_vz[0];
        }
        e->heading = e->traj_heading[0];
        e->heading_x = cosf(e->heading);
        e->heading_y = sinf(e->heading);
        e->valid = e->traj_valid[0];
        e->collision_state = 0;
        e->metrics_array[COLLISION_IDX] = 0.0f; // vehicle collision
        e->metrics_array[OFFROAD_IDX] = 0.0f; // offroad
        e->metrics_array[REACHED_GOAL_IDX] = 0.0f; // reached goal
        e->metrics_array[LANE_ALIGNED_IDX] = 0.0f; // lane aligned
        e->metrics_array[AVG_DISPLACEMENT_ERROR_IDX] = 0.0f; // avg displacement error
        e->cumulative_displacement = 0.0f;
        e->displacement_sample_count = 0;
        e->respawn_timestep = -1;
    }
    //EndDrawing();
}

int getGridIndex(Drive* env, float x1, float y1) {
    if (env->grid_map->top_left_x >= env->grid_map->bottom_right_x || env->grid_map->bottom_right_y >= env->grid_map->top_left_y) {
        return -1;  // Invalid grid coordinates
    }

    float relativeX = x1 - env->grid_map->top_left_x;  // Distance from left
    float relativeY = y1 - env->grid_map->bottom_right_y;  // Distance from bottom
    int gridX = (int)(relativeX / GRID_CELL_SIZE);  // Column index
    int gridY = (int)(relativeY / GRID_CELL_SIZE);  // Row index
    if (gridX < 0 || gridX >= env->grid_map->grid_cols || gridY < 0 || gridY >= env->grid_map->grid_rows) {
        return -1;  // Return -1 for out of bounds
    }
    int index = (gridY*env->grid_map->grid_cols) + gridX;
    return index;
}

void add_entity_to_grid(Drive* env, int grid_index, int entity_idx, int geometry_idx, int* cell_entities_insert_index){
    if(grid_index == -1){
        return;
    }

    int count = cell_entities_insert_index[grid_index];
    if(count >= env->grid_map->cell_entities_count[grid_index]) {
        printf("Error: Exceeded precomputed entity count for grid cell %d. Current count: %d, Max count(Precomputed): %d\n", grid_index, count, env->grid_map->cell_entities_count[grid_index]);
        return;
    }

    env->grid_map->cells[grid_index][count].entity_idx = entity_idx;
    env->grid_map->cells[grid_index][count].geometry_idx = geometry_idx;
    cell_entities_insert_index[grid_index] = count + 1;
}

void init_grid_map(Drive* env){
    // Allocate memory for the grid map structure
    env->grid_map = (GridMap*)malloc(sizeof(GridMap));

    // Find top left and bottom right points of the map
    float top_left_x;
    float top_left_y;
    float bottom_right_x;
    float bottom_right_y;
    int first_valid_point = 0;
    for(int i = 0; i < env->num_entities; i++){
        if(env->entities[i].type > 3 && env->entities[i].type < 7){
            // Check all points in the trajectory for road elements
            Entity* e = &env->entities[i];
            for(int j = 0; j < e->array_size; j++){
                if(e->traj_x[j] == -10000) continue;
                if(e->traj_y[j] == -10000) continue;
                if(!first_valid_point) {
                    top_left_x = bottom_right_x = e->traj_x[j];
                    top_left_y = bottom_right_y = e->traj_y[j];
                    first_valid_point = true;
                    continue;
                }
                if(e->traj_x[j] < top_left_x) top_left_x = e->traj_x[j];
                if(e->traj_x[j] > bottom_right_x) bottom_right_x = e->traj_x[j];
                if(e->traj_y[j] > top_left_y) top_left_y = e->traj_y[j];
                if(e->traj_y[j] < bottom_right_y) bottom_right_y = e->traj_y[j];
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
    int grid_cell_count = env->grid_map->grid_cols*env->grid_map->grid_rows;
    env->grid_map->cells = (GridMapEntity**)calloc(grid_cell_count, sizeof(GridMapEntity*));
    env->grid_map->cell_entities_count = (int*)calloc(grid_cell_count, sizeof(int));

    // Calculate number of entities in each grid cell
    for(int i = 0; i < env->num_entities; i++){
        if(env->entities[i].type > 3 && env->entities[i].type < 7){
            for(int j = 0; j < env->entities[i].array_size - 1; j++){
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j+1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j+1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                env->grid_map->cell_entities_count[grid_index]++;
            }
        }
    }
    int cell_entities_insert_index[grid_cell_count];   // Helper array for insertion index
    memset(cell_entities_insert_index, 0, grid_cell_count * sizeof(int));

    // Initialize grid cells
    for(int grid_index = 0; grid_index < grid_cell_count; grid_index++){
        env->grid_map->cells[grid_index] = (GridMapEntity*)calloc(env->grid_map->cell_entities_count[grid_index], sizeof(GridMapEntity));
    }
    for(int i = 0;i<grid_cell_count;i++){
        if(cell_entities_insert_index[i] != 0){
            printf("Error: cell_entities_insert_index[%d] not zero during initialization.\n", i);
            cell_entities_insert_index[i] = 0;
        }
    }

    // Populate grid cells
    for(int i = 0; i < env->num_entities; i++){
        if(env->entities[i].type > 3 && env->entities[i].type < 7){         // NOTE: Only Road Edges, Lines, and Lanes in grid map
            for(int j = 0; j < env->entities[i].array_size - 1; j++){
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j+1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j+1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                add_entity_to_grid(env, grid_index, i, j, cell_entities_insert_index);
            }
        }
    }
}

void init_neighbor_offsets(Drive* env) {
    // Allocate memory for the offsets
    env->neighbor_offsets = (int*)calloc(env->grid_map->vision_range*env->grid_map->vision_range*2, sizeof(int));
    // neighbor offsets in a spiral pattern
    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0;    // Current x offset
    int y = 0;    // Current y offset
    int dir = 0;  // Current direction (0: right, 1: up, 2: left, 3: down)
    int steps_to_take = 1; // Number of steps in current direction
    int steps_taken = 0;   // Steps taken in current direction
    int segments_completed = 0; // Count of direction segments completed
    int total = 0; // Total offsets added
    int max_offsets = env->grid_map->vision_range*env->grid_map->vision_range;
    // Start at center (0,0)
    int curr_idx = 0;
    env->neighbor_offsets[curr_idx++] = 0;  // x offset
    env->neighbor_offsets[curr_idx++] = 0;  // y offset
    total++;
    // Generate spiral pattern
    while (total < max_offsets) {
        // Move in current direction
        x += dx[dir];
        y += dy[dir];
        // Only add if within vision range bounds
        if (abs(x) <= env->grid_map->vision_range/2 && abs(y) <= env->grid_map->vision_range/2) {
            env->neighbor_offsets[curr_idx++] = x;
            env->neighbor_offsets[curr_idx++] = y;
            total++;
        }
        steps_taken++;
        // Check if we need to change direction
        if(steps_taken != steps_to_take) continue;
        steps_taken = 0;  // Reset steps taken
        dir = (dir + 1) % 4;  // Change direction (clockwise: right->up->left->down)
        segments_completed++;
        // Increase step length every two direction changes
        if (segments_completed % 2 == 0) {
            steps_to_take++;
        }
    }
}

void cache_neighbor_offsets(Drive* env){
    int count = 0;
    int cell_count = env->grid_map->grid_cols*env->grid_map->grid_rows;
    env->grid_map->neighbor_cache_entities = (GridMapEntity**)calloc(cell_count, sizeof(GridMapEntity*));
    env->grid_map->neighbor_cache_count = (int*)calloc(cell_count + 1, sizeof(int));
    for(int i = 0; i < cell_count; i++){
        int cell_x = i % env->grid_map->grid_cols;  // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int current_cell_neighbor_count = 0;
        for(int j = 0; j < env->grid_map->vision_range*env->grid_map->vision_range; j++){
            int x = cell_x + env->neighbor_offsets[j*2];
            int y = cell_y + env->neighbor_offsets[j*2+1];
            int grid_index = env->grid_map->grid_cols*y + x;
            if(x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows) continue;
            int grid_count = env->grid_map->cell_entities_count[grid_index];
            current_cell_neighbor_count += grid_count;
        }
        env->grid_map->neighbor_cache_count[i] = current_cell_neighbor_count;
        count += current_cell_neighbor_count;
        if(current_cell_neighbor_count == 0) {
            env->grid_map->neighbor_cache_entities[i] = NULL;
            continue;
        }
        env->grid_map->neighbor_cache_entities[i] = (GridMapEntity*)calloc(current_cell_neighbor_count, sizeof(GridMapEntity));
    }

    env->grid_map->neighbor_cache_count[cell_count] = count;
    for(int i = 0; i < cell_count; i ++){
        int cell_x = i % env->grid_map->grid_cols;  // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int base_index = 0;
        for(int j = 0; j < env->grid_map->vision_range*env->grid_map->vision_range; j++){
            int x = cell_x + env->neighbor_offsets[j*2];
            int y = cell_y + env->neighbor_offsets[j*2+1];
            int grid_index = env->grid_map->grid_cols*y + x;
            if(x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows) continue;
            int grid_count = env->grid_map->cell_entities_count[grid_index];

            // Skip if no entities or source is NULL
            if(grid_count == 0 || env->grid_map->cells[grid_index] == NULL) {
                continue;
            }

            int src_idx = grid_index;
            int dst_idx = base_index;
            // Copy grid_count pairs (entity_idx, geometry_idx) at once
            memcpy(&env->grid_map->neighbor_cache_entities[i][dst_idx],
                env->grid_map->cells[src_idx],
                grid_count * sizeof(GridMapEntity));
            // for(int k = 0; k < grid_count; k++){
            //     env->grid_map->neighbor_cache_entities[i][dst_idx + k] = env->grid_map->cells[src_idx][k];
            // }
            base_index += grid_count;
        }
    }
}

int get_neighbor_cache_entities(Drive* env, int cell_idx, GridMapEntity* entities, int max_entities) {
    GridMap* grid_map = env->grid_map;
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

void set_means(Drive* env) {
    float mean_x = 0.0f;
    float mean_y = 0.0f;
    int64_t point_count = 0;

    // Compute single mean for all entities (vehicles and roads)
    for (int i = 0; i < env->num_entities; i++) {
        if (env->entities[i].type == VEHICLE) {
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
        if (env->entities[i].type == VEHICLE || env->entities[i].type >= 4) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                if(env->entities[i].traj_x[j] == -10000) continue;
                env->entities[i].traj_x[j] -= mean_x;
                env->entities[i].traj_y[j] -= mean_y;
            }
            env->entities[i].goal_position_x -= mean_x;
            env->entities[i].goal_position_y -= mean_y;
        }
    }

}

void move_expert(Drive* env, float* actions, int agent_idx){
    Entity* agent = &env->entities[agent_idx];
    agent->x = agent->traj_x[env->timestep];
    agent->y = agent->traj_y[env->timestep];
    agent->z = agent->traj_z[env->timestep];
    agent->heading = agent->traj_heading[env->timestep];
    agent->heading_x = cosf(agent->heading);
    agent->heading_y = sinf(agent->heading);
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
    if (cross == 0) return false;

    // Calculate relative vectors between start points
    float dx3 = p1[0] - q1[0];
    float dy3 = p1[1] - q1[1];

    // Calculate parameters for intersection point
    float s = (dx1 * dy3 - dy1 * dx3) / cross;
    float t = (dx2 * dy3 - dy2 * dx3) / cross;

    // Check if intersection point lies within both line segments
    return (s >= 0 && s <= 1 && t >= 0 && t <= 1);
}

int checkNeighbors(Drive* env, float x, float y, GridMapEntity* entity_list, int max_size, const int (*local_offsets)[2], int offset_size) {
    // Get the grid index for the given position (x, y)
    int index = getGridIndex(env, x, y);
    if (index == -1) return 0;  // Return 0 size if position invalid
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
        if(nx < 0 || nx >= env->grid_map->grid_cols || ny < 0 || ny >= env->grid_map->grid_rows) continue;
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

int check_aabb_collision(Entity* car1, Entity* car2) {
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
        {car1->x + (-half_len1 * cos1 + half_width1 * sin1), car1->y + (-half_len1 * sin1 - half_width1 * cos1)}
    };

    // Calculate car2's corners in world space
    float car2_corners[4][2] = {
        {car2->x + (half_len2 * cos2 - half_width2 * sin2), car2->y + (half_len2 * sin2 + half_width2 * cos2)},
        {car2->x + (half_len2 * cos2 + half_width2 * sin2), car2->y + (half_len2 * sin2 - half_width2 * cos2)},
        {car2->x + (-half_len2 * cos2 - half_width2 * sin2), car2->y + (-half_len2 * sin2 + half_width2 * cos2)},
        {car2->x + (-half_len2 * cos2 + half_width2 * sin2), car2->y + (-half_len2 * sin2 - half_width2 * cos2)}
    };

    // Get the axes to check (normalized vectors perpendicular to each edge)
    float axes[4][2] = {
        {cos1, sin1},           // Car1's length axis
        {-sin1, cos1},          // Car1's width axis
        {cos2, sin2},           // Car2's length axis
        {-sin2, cos2}           // Car2's width axis
    };

    // Check each axis
    for(int i = 0; i < 4; i++) {
        float min1 = INFINITY, max1 = -INFINITY;
        float min2 = INFINITY, max2 = -INFINITY;

        // Project car1's corners onto the axis
        for(int j = 0; j < 4; j++) {
            float proj = car1_corners[j][0] * axes[i][0] + car1_corners[j][1] * axes[i][1];
            min1 = fminf(min1, proj);
            max1 = fmaxf(max1, proj);
        }

        // Project car2's corners onto the axis
        for(int j = 0; j < 4; j++) {
            float proj = car2_corners[j][0] * axes[i][0] + car2_corners[j][1] * axes[i][1];
            min2 = fminf(min2, proj);
            max2 = fmaxf(max2, proj);
        }

        // If there's a gap on this axis, the boxes don't intersect
        if(max1 < min2 || min1 > max2) {
            return 0;  // No collision
        }
    }

    // If we get here, there's no separating axis, so the boxes intersect
    return 1;  // Collision
}

int collision_check(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];

    if(agent->x == -10000.0f ) return -1;

    int car_collided_with_index = -1;

    for(int i = 0; i < MAX_CARS; i++){
        int index = -1;
        if(i < env->active_agent_count){
            index = env->active_agent_indices[i];
        } else if (i < env->num_cars){
            index = env->static_car_indices[i - env->active_agent_count];
        }
        if(index == -1) continue;
        if(index == agent_idx) continue;
        Entity* entity = &env->entities[index];
        float x1 = entity->x;
        float y1 = entity->y;
        float dist = ((x1 - agent->x)*(x1 - agent->x) + (y1 - agent->y)*(y1 - agent->y));
        if(dist > 225.0f) continue;
        if(check_aabb_collision(agent, entity)) {
            car_collided_with_index = index;
            break;
        }
    }

    return car_collided_with_index;
}

int check_lane_aligned(Entity* car, Entity* lane, int geometry_idx) {
    // Validate lane geometry length
    if (!lane || lane->array_size < 2) return 0;

    // Clamp geometry index to valid segment range [0, array_size-2]
    if (geometry_idx < 0) geometry_idx = 0;
    if (geometry_idx >= lane->array_size - 1) geometry_idx = lane->array_size - 2;

    // Compute local lane segment heading
    float heading_x1, heading_y1;
    if (geometry_idx > 0) {
        heading_x1 = lane->traj_x[geometry_idx] - lane->traj_x[geometry_idx - 1];
        heading_y1 = lane->traj_y[geometry_idx] - lane->traj_y[geometry_idx - 1];
    } else {
        // For first segment, just use the forward direction
        heading_x1 = lane->traj_x[geometry_idx + 1] - lane->traj_x[geometry_idx];
        heading_y1 = lane->traj_y[geometry_idx + 1] - lane->traj_y[geometry_idx];
    }

    float heading_x2 = lane->traj_x[geometry_idx + 1] - lane->traj_x[geometry_idx];
    float heading_y2 = lane->traj_y[geometry_idx + 1] - lane->traj_y[geometry_idx];

    float heading_1 = atan2f(heading_y1, heading_x1);
    float heading_2 = atan2f(heading_y2, heading_x2);
    float heading = (heading_1 + heading_2) / 2.0f;

    // Normalize to [-pi, pi]
    if (heading > M_PI) heading -= 2.0f * M_PI;
    if (heading < -M_PI) heading += 2.0f * M_PI;

    // Compute heading difference
    float car_heading = car->heading; // radians
    float heading_diff = fabsf(car_heading - heading);

    if (heading_diff > M_PI) heading_diff = 2.0f * M_PI - heading_diff;

    return (heading_diff < (M_PI / 6.0f)) ? 1 : 0; // within 30 degrees
}

void reset_agent_metrics(Drive* env, int agent_idx){
    Entity* agent = &env->entities[agent_idx];
    agent->metrics_array[COLLISION_IDX] = 0.0f; // vehicle collision
    agent->metrics_array[OFFROAD_IDX] = 0.0f; // offroad
    agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f; // lane aligned
    agent->metrics_array[AVG_DISPLACEMENT_ERROR_IDX] = 0.0f;
    agent->collision_state = 0;
}

void compute_agent_metrics(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];

    reset_agent_metrics(env, agent_idx);

    if(agent->x == -10000.0f ) return; // invalid agent position

    // Compute displacement error
    float displacement_error = compute_displacement_error(agent, env->timestep);

    if (displacement_error > 0.0f) { // Only count valid displacements
        agent->cumulative_displacement += displacement_error;
        agent->displacement_sample_count++;

        // Compute running average
        agent->metrics_array[AVG_DISPLACEMENT_ERROR_IDX] =
            agent->cumulative_displacement / agent->displacement_sample_count;
    }

    int collided = 0;
    float half_length = agent->length/2.0f;
    float half_width = agent->width/2.0f;
    float cos_heading = cosf(agent->heading);
    float sin_heading = sinf(agent->heading);
    float min_distance = (float)INT16_MAX;

    int closest_lane_entity_idx = -1;
    int closest_lane_geometry_idx = -1;

    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] = agent->x + (offsets[i][0]*half_length*cos_heading - offsets[i][1]*half_width*sin_heading);
        corners[i][1] = agent->y + (offsets[i][0]*half_length*sin_heading + offsets[i][1]*half_width*cos_heading);
    }

    GridMapEntity entity_list[MAX_ENTITIES_PER_CELL*25];  // Array big enough for all neighboring cells
    int list_size = checkNeighbors(env, agent->x, agent->y, entity_list, MAX_ENTITIES_PER_CELL*25, collision_offsets, 25);
    for (int i = 0; i < list_size ; i++) {
        if(entity_list[i].entity_idx == -1) continue;
        if(entity_list[i].entity_idx == agent_idx) continue;
        Entity* entity;
        entity = &env->entities[entity_list[i].entity_idx];

        // Check for offroad collision with road edges
        if(entity->type == ROAD_EDGE) {
            int geometry_idx = entity_list[i].geometry_idx;
            float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
            float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};
            for (int k = 0; k < 4; k++) { // Check each edge of the bounding box
                int next = (k + 1) % 4;
                if (check_line_intersection(corners[k], corners[next], start, end)) {
                    collided = OFFROAD;
                    break;
                }
            }
        }

        if (collided == OFFROAD) break;

        // Find closest point on the road centerline to the agent
        if(entity->type == ROAD_LANE) {
            int entity_idx = entity_list[i].entity_idx;
            int geometry_idx = entity_list[i].geometry_idx;

            float lane_x = entity->traj_x[geometry_idx];
            float lane_y = entity->traj_y[geometry_idx];

            int lane_size = entity->array_size;
            if(geometry_idx == lane_size - 1) continue;

            float lane_x_next = entity->traj_x[geometry_idx + 1];
            float lane_y_next = entity->traj_y[geometry_idx + 1];

            float dx_lane = lane_x_next - lane_x;
            float dy_lane = lane_y_next - lane_y;

            float lane_heading = atan2f(dy_lane, dx_lane);

            float dist = ((lane_x - agent->x)*(lane_x - agent->x) + (lane_y - agent->y)*(lane_y - agent->y));
            float angle_diff = fabsf(agent->heading - lane_heading);
            if(dist < min_distance && angle_diff < (M_PI / 2.0f)) {
                min_distance = dist;
                closest_lane_entity_idx = entity_idx;
                closest_lane_geometry_idx = geometry_idx;
            }
        }
    }

    // check if aligned with closest lane
    if (min_distance > 4.0f) {
        agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f;
    } else {
        int lane_aligned = check_lane_aligned(agent, &env->entities[closest_lane_entity_idx], closest_lane_geometry_idx);
        agent->metrics_array[LANE_ALIGNED_IDX] = lane_aligned ? 1.0f : 0.0f;
    }


    // Check for vehicle collisions
    int car_collided_with_index = collision_check(env, agent_idx);
    if (car_collided_with_index != -1) collided = VEHICLE_COLLISION;

    agent->collision_state = collided;

    // spawn immunity for collisions with other agent cars as agent_idx respawns
    int is_active_agent = env->entities[agent_idx].active_agent;
    int respawned = env->entities[agent_idx].respawn_timestep != -1;
    int exceeded_spawn_immunity_agent = (env->timestep - env->entities[agent_idx].respawn_timestep) >= env->spawn_immunity_timer;

    if(collided == VEHICLE_COLLISION && is_active_agent == 1 && respawned){
        agent->collision_state = 0;
    }

    if(collided == OFFROAD) {
        agent->metrics_array[OFFROAD_IDX] = 1.0f;
        return;
    }
    if(car_collided_with_index == -1) return;

    // spawn immunity for collisions with other cars who just respawned
    int respawned_collided_with_car = env->entities[car_collided_with_index].respawn_timestep != -1;
    int exceeded_spawn_immunity_collided_with_car = (env->timestep - env->entities[car_collided_with_index].respawn_timestep) >= env->spawn_immunity_timer;
    int within_spawn_immunity_collided_with_car = (env->timestep - env->entities[car_collided_with_index].respawn_timestep) < env->spawn_immunity_timer;

    if (respawned_collided_with_car) {
        agent->collision_state = 0;
        agent->metrics_array[COLLISION_IDX] = 0.0f;
    }


    return;
}

int valid_active_agent(Drive* env, int agent_idx){
    float cos_heading = cosf(env->entities[agent_idx].traj_heading[0]);
    float sin_heading = sinf(env->entities[agent_idx].traj_heading[0]);
    float goal_x = env->entities[agent_idx].goal_position_x - env->entities[agent_idx].traj_x[0];
    float goal_y = env->entities[agent_idx].goal_position_y - env->entities[agent_idx].traj_y[0];
    // Rotate to ego vehicle's frame
    float rel_goal_x = goal_x*cos_heading + goal_y*sin_heading;
    float rel_goal_y = -goal_x*sin_heading + goal_y*cos_heading;
    float distance_to_goal = relative_distance_2d(0, 0, rel_goal_x, rel_goal_y);
    env->entities[agent_idx].width *= 0.7f;
    env->entities[agent_idx].length *= 0.7f;
    if(distance_to_goal >= 2.0f && env->entities[agent_idx].mark_as_expert == 0 && env->active_agent_count < env->num_agents){
        return distance_to_goal;
    }
    return 0;
}

void set_active_agents(Drive* env){
    env->active_agent_count = 0;
    env->static_car_count = 0;
    env->num_cars = 1;
    env->expert_static_car_count = 0;
    int active_agent_indices[MAX_CARS];
    int static_car_indices[MAX_CARS];
    int expert_static_car_indices[MAX_CARS];

    if(env->num_agents ==0){
        env->num_agents = MAX_CARS;
    }
    int first_agent_id = env->num_objects-1;
    float distance_to_goal = valid_active_agent(env, first_agent_id);
    if(distance_to_goal){
        env->active_agent_count = 1;
        active_agent_indices[0] = first_agent_id;
        env->entities[first_agent_id].active_agent = 1;
        env->num_cars = 1;
    } else {
        env->active_agent_count = 0;
        env->num_cars = 0;
    }
    for(int i = 0; i < env->num_objects-1 && env->num_cars < MAX_CARS; i++){
        if(env->entities[i].type != 1) continue;
        if(env->entities[i].traj_valid[0] != 1) continue;
        env->num_cars++;
        float distance_to_goal = valid_active_agent(env, i);
        if(distance_to_goal > 0){
            active_agent_indices[env->active_agent_count] = i;
            env->active_agent_count++;
            env->entities[i].active_agent = 1;
        } else {
            static_car_indices[env->static_car_count] = i;
            env->static_car_count++;
            env->entities[i].active_agent = 0;
            if(env->entities[i].mark_as_expert == 1 || (distance_to_goal >=2.0f && env->active_agent_count == env->num_agents)){
                expert_static_car_indices[env->expert_static_car_count] = i;
                env->expert_static_car_count++;
                env->entities[i].mark_as_expert = 1;
            }
        }
    }
    // set up initial active agents
    env->active_agent_indices = (int*)malloc(env->active_agent_count * sizeof(int));
    env->static_car_indices = (int*)malloc(env->static_car_count * sizeof(int));
    env->expert_static_car_indices = (int*)malloc(env->expert_static_car_count * sizeof(int));
    for(int i=0;i<env->active_agent_count;i++){
        env->active_agent_indices[i] = active_agent_indices[i];
    };
    for(int i=0;i<env->static_car_count;i++){
        env->static_car_indices[i] = static_car_indices[i];

    }
    for(int i=0;i<env->expert_static_car_count;i++){
        env->expert_static_car_indices[i] = expert_static_car_indices[i];
    }
    return;
}

void remove_bad_trajectories(Drive* env){
    set_start_position(env);
    int collided_agents[env->active_agent_count];
    int collided_with_indices[env->active_agent_count];
    memset(collided_agents, 0, env->active_agent_count * sizeof(int));
    // move experts through trajectories to check for collisions and remove as illegal agents
    for(int t = 0; t < TRAJECTORY_LENGTH; t++){
        for(int i = 0; i < env->active_agent_count; i++){
            int agent_idx = env->active_agent_indices[i];
            move_expert(env, env->actions, agent_idx);
        }
        for(int i = 0; i < env->expert_static_car_count; i++){
            int expert_idx = env->expert_static_car_indices[i];
            if(env->entities[expert_idx].x == -10000) continue;
            move_expert(env, env->actions, expert_idx);
        }
        // check collisions
        for(int i = 0; i < env->active_agent_count; i++){
            int agent_idx = env->active_agent_indices[i];
            env->entities[agent_idx].collision_state = 0;
            int collided_with_index = collision_check(env, agent_idx);
            if((collided_with_index >= 0) && collided_agents[i] == 0){
                collided_agents[i] = 1;
                collided_with_indices[i] = collided_with_index;
            }
        }
        env->timestep++;
    }

    for(int i = 0; i< env->active_agent_count; i++){
        if(collided_with_indices[i] == -1) continue;
        for(int j = 0; j < env->static_car_count; j++){
            int static_car_idx = env->static_car_indices[j];
            if(static_car_idx != collided_with_indices[i]) continue;
            env->entities[static_car_idx].traj_x[0] = -10000;
            env->entities[static_car_idx].traj_y[0] = -10000;
        }
    }
    env->timestep = 0;
}

void init(Drive* env){
    env->human_agent_idx = 0;
    env->timestep = 0;
    env->entities = load_map_binary(env->map_name, env);
    env->dynamics_model = CLASSIC;
    set_means(env);
    init_grid_map(env);
    env->grid_map->vision_range = 21;
    init_neighbor_offsets(env);
    cache_neighbor_offsets(env);
    set_active_agents(env);
    remove_bad_trajectories(env);
    set_start_position(env);
    env->logs = (Log*)calloc(env->active_agent_count, sizeof(Log));
}

void c_close(Drive* env){
    for(int i = 0; i < env->num_entities; i++){
        free_entity(&env->entities[i]);
    }
    free(env->entities);
    free(env->active_agent_indices);
    free(env->logs);
    // GridMap cleanup
    int grid_cell_count = env->grid_map->grid_cols*env->grid_map->grid_rows;
    for(int grid_index = 0; grid_index < grid_cell_count; grid_index++){
        free(env->grid_map->cells[grid_index]);
    }
    free(env->grid_map->cells);
    free(env->grid_map->cell_entities_count);
    free(env->neighbor_offsets);

    for(int i = 0; i < grid_cell_count; i++){
        free(env->grid_map->neighbor_cache_entities[i]);
    }
    free(env->grid_map->neighbor_cache_entities);
    free(env->grid_map->neighbor_cache_count);
    free(env->grid_map);
    free(env->static_car_indices);
    free(env->expert_static_car_indices);
    // free(env->map_name);
    free(env->ini_file);
}

void allocate(Drive* env){
    init(env);
    int max_obs = 7 + 7*(MAX_CARS - 1) + 7*MAX_ROAD_SEGMENT_OBSERVATIONS;
    env->observations = (float*)calloc(env->active_agent_count*max_obs, sizeof(float));
    env->actions = (float*)calloc(env->active_agent_count*2, sizeof(float));
    env->rewards = (float*)calloc(env->active_agent_count, sizeof(float));
    env->terminals= (unsigned char*)calloc(env->active_agent_count, sizeof(unsigned char));
}

void free_allocated(Drive* env){
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
}

float clipSpeed(float speed) {
    const float maxSpeed = MAX_SPEED;
    if (speed > maxSpeed) return maxSpeed;
    if (speed < -maxSpeed) return -maxSpeed;
    return speed;
}

float normalize_heading(float heading){
    if(heading > M_PI) heading -= 2*M_PI;
    if(heading < -M_PI) heading += 2*M_PI;
    return heading;
}

void move_dynamics(Drive* env, int action_idx, int agent_idx){
    if(env->dynamics_model == CLASSIC){
        Entity* agent = &env->entities[agent_idx];
        float acceleration = 0.0f;
        float steering = 0.0f;

        if (env->action_type == 1) { // continuous
            float (*action_array_f)[2] = (float(*)[2])env->actions;
            acceleration = action_array_f[action_idx][0];
            steering = action_array_f[action_idx][1];
        } else { // discrete
            int (*action_array)[2] = (int(*)[2])env->actions;
            int acceleration_index = action_array[action_idx][0];
            int steering_index = action_array[action_idx][1];

            acceleration = ACCELERATION_VALUES[acceleration_index];
            steering = STEERING_VALUES[steering_index];
        }

        // Current state
        float x = agent->x;
        float y = agent->y;
        float heading = agent->heading;
        float vx = agent->vx;
        float vy = agent->vy;

        // Calculate current speed
        float speed = sqrtf(vx*vx + vy*vy);

        // Time step (adjust as needed)
        const float dt = 0.1f;
        // Update speed with acceleration
        speed = speed + 0.5f*acceleration*dt;
        // if (speed < 0) speed = 0;  // Prevent going backward
        speed = clipSpeed(speed);
        // compute yaw rate
        float beta = tanh(.5*tanf(steering));
        // new heading
        float yaw_rate = (speed*cosf(beta)*tanf(steering)) / agent->length;
        // new velocity
        float new_vx = speed*cosf(heading + beta);
        float new_vy = speed*sinf(heading + beta);
        // Update position
        x = x + (new_vx*dt);
        y = y + (new_vy*dt);
        heading = heading + yaw_rate*dt;
        // heading = normalize_heading(heading);
        // Apply updates to the agent's state
        agent->x = x;
        agent->y = y;
        agent->heading = heading;
        agent->heading_x = cosf(heading);
        agent->heading_y = sinf(heading);
        agent->vx = new_vx;
        agent->vy = new_vy;
    }
    return;
}

float normalize_value(float value, float min, float max){
    return (value - min) / (max - min);
}

float reverse_normalize_value(float value, float min, float max){
    return value*50.0f;
}

void compute_observations(Drive* env) {
    int max_obs = 7 + 7*(MAX_CARS - 1) + 7*MAX_ROAD_SEGMENT_OBSERVATIONS;
    memset(env->observations, 0, max_obs*env->active_agent_count*sizeof(float));
    float (*observations)[max_obs] = (float(*)[max_obs])env->observations;
    for(int i = 0; i < env->active_agent_count; i++) {
        float* obs = &observations[i][0];
        Entity* ego_entity = &env->entities[env->active_agent_indices[i]];
        if(ego_entity->type > 3) break;
        if(ego_entity->respawn_timestep != -1) {
            obs[6] = 1;
            //continue;
        }
        float cos_heading = ego_entity->heading_x;
        float sin_heading = ego_entity->heading_y;
        float ego_speed = sqrtf(ego_entity->vx*ego_entity->vx + ego_entity->vy*ego_entity->vy);
        // Set goal distances
        float goal_x = ego_entity->goal_position_x - ego_entity->x;
        float goal_y = ego_entity->goal_position_y - ego_entity->y;
        // Rotate to ego vehicle's frame
        float rel_goal_x = goal_x*cos_heading + goal_y*sin_heading;
        float rel_goal_y = -goal_x*sin_heading + goal_y*cos_heading;
        //obs[0] = normalize_value(rel_goal_x, MIN_REL_GOAL_COORD, MAX_REL_GOAL_COORD);
        //obs[1] = normalize_value(rel_goal_y, MIN_REL_GOAL_COORD, MAX_REL_GOAL_COORD);
        obs[0] = rel_goal_x* 0.005f;
        obs[1] = rel_goal_y* 0.005f;
        //obs[2] = ego_speed / MAX_SPEED;
        obs[2] = ego_speed * 0.01f;
        obs[3] = ego_entity->width / MAX_VEH_WIDTH;
        obs[4] = ego_entity->length / MAX_VEH_LEN;
        obs[5] = (ego_entity->collision_state > 0) ? 1.0f : 0.0f;

        // Relative Pos of other cars
        int obs_idx = 7;  // Start after goal distances
        int cars_seen = 0;
        for(int j = 0; j < MAX_CARS; j++) {
            int index = -1;
            if(j < env->active_agent_count){
                index = env->active_agent_indices[j];
            } else if (j < env->num_cars){
                index = env->static_car_indices[j - env->active_agent_count];
            }
            if(index == -1) continue;
            if(env->entities[index].type > 3) break;
            if(index == env->active_agent_indices[i]) continue;  // Skip self, but don't increment obs_idx
            Entity* other_entity = &env->entities[index];
            if(ego_entity->respawn_timestep != -1) continue;
            if(other_entity->respawn_timestep != -1) continue;
            // Store original relative positions
            float dx = other_entity->x - ego_entity->x;
            float dy = other_entity->y - ego_entity->y;
            float dist = (dx*dx + dy*dy);
            if(dist > 2500.0f) continue;
            // Rotate to ego vehicle's frame
            float rel_x = dx*cos_heading + dy*sin_heading;
            float rel_y = -dx*sin_heading + dy*cos_heading;
            // Store observations with correct indexing
            obs[obs_idx] = rel_x * 0.02f;
            obs[obs_idx + 1] = rel_y * 0.02f;
            obs[obs_idx + 2] = other_entity->width / MAX_VEH_WIDTH;
            obs[obs_idx + 3] = other_entity->length / MAX_VEH_LEN;
            // relative heading
            float rel_heading_x = other_entity->heading_x * ego_entity->heading_x +
                     other_entity->heading_y * ego_entity->heading_y;  // cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
            float rel_heading_y = other_entity->heading_y * ego_entity->heading_x -
                                other_entity->heading_x * ego_entity->heading_y;  // sin(a-b) = sin(a)cos(b) - cos(a)sin(b)

            obs[obs_idx + 4] = rel_heading_x;
            obs[obs_idx + 5] = rel_heading_y;
            // obs[obs_idx + 4] = cosf(rel_heading) / MAX_ORIENTATION_RAD;
            // obs[obs_idx + 5] = sinf(rel_heading) / MAX_ORIENTATION_RAD;
            // // relative speed
            float other_speed = sqrtf(other_entity->vx*other_entity->vx + other_entity->vy*other_entity->vy);
            obs[obs_idx + 6] = other_speed / MAX_SPEED;
            cars_seen++;
            obs_idx += 7;  // Move to next observation slot
        }
        int remaining_partner_obs = (MAX_CARS - 1 - cars_seen) * 7;
        memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
        obs_idx += remaining_partner_obs;
        // map observations
        GridMapEntity entity_list[MAX_ENTITIES_PER_CELL*25];
        int grid_idx = getGridIndex(env, ego_entity->x, ego_entity->y);

        int list_size = get_neighbor_cache_entities(env, grid_idx, entity_list, MAX_ROAD_SEGMENT_OBSERVATIONS);

        for(int k = 0; k < list_size; k++) {
            int entity_idx = entity_list[k].entity_idx;
            int geometry_idx = entity_list[k].geometry_idx;

            // Validate entity_idx before accessing
            if(entity_idx < 0 || entity_idx >= env->num_entities) {
                printf("ERROR: Invalid entity_idx %d (max: %d)\n", entity_idx, env->num_entities-1);
                continue;
            }

            Entity* entity = &env->entities[entity_idx];

            // Validate geometry_idx before accessing
            if(geometry_idx < 0 || geometry_idx >= entity->array_size) {
                printf("ERROR: Invalid geometry_idx %d for entity %d (max: %d)\n",
                       geometry_idx, entity_idx, entity->array_size-1);
                continue;
            }
            float start_x = entity->traj_x[geometry_idx];
            float start_y = entity->traj_y[geometry_idx];
            float end_x = entity->traj_x[geometry_idx+1];
            float end_y = entity->traj_y[geometry_idx+1];
            float mid_x = (start_x + end_x) / 2.0f;
            float mid_y = (start_y + end_y) / 2.0f;
            float rel_x = mid_x - ego_entity->x;
            float rel_y = mid_y - ego_entity->y;
            float x_obs = rel_x*cos_heading + rel_y*sin_heading;
            float y_obs = -rel_x*sin_heading + rel_y*cos_heading;
            float length = relative_distance_2d(mid_x, mid_y, end_x, end_y);
            float width = 0.1;
            // Calculate angle from ego to midpoint (vector from ego to midpoint)
            float dx = end_x - mid_x;
            float dy = end_y - mid_y;
            float dx_norm = dx;
            float dy_norm = dy;
            float hypot = sqrtf(dx*dx + dy*dy);
            if(hypot > 0) {
                dx_norm /= hypot;
                dy_norm /= hypot;
            }
            // Compute sin and cos of relative angle directly without atan2f
            float cos_angle = dx_norm*cos_heading + dy_norm*sin_heading;
            float sin_angle = -dx_norm*sin_heading + dy_norm*cos_heading;
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

void c_reset(Drive* env){
    env->timestep = 0;
    set_start_position(env);
    for(int x = 0;x<env->active_agent_count; x++){
        env->logs[x] = (Log){0};
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].respawn_timestep = -1;
        env->entities[agent_idx].collided_before_goal = 0;
        env->entities[agent_idx].reached_goal_this_episode = 0;
        env->entities[agent_idx].metrics_array[COLLISION_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[OFFROAD_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[AVG_DISPLACEMENT_ERROR_IDX] = 0.0f;
        env->entities[agent_idx].cumulative_displacement = 0.0f;
        env->entities[agent_idx].displacement_sample_count = 0;

        compute_agent_metrics(env, agent_idx);
    }
    compute_observations(env);
}

void respawn_agent(Drive* env, int agent_idx){
    env->entities[agent_idx].x = env->entities[agent_idx].traj_x[0];
    env->entities[agent_idx].y = env->entities[agent_idx].traj_y[0];
    env->entities[agent_idx].heading = env->entities[agent_idx].traj_heading[0];
    env->entities[agent_idx].heading_x = cosf(env->entities[agent_idx].heading);
    env->entities[agent_idx].heading_y = sinf(env->entities[agent_idx].heading);
    env->entities[agent_idx].vx = env->entities[agent_idx].traj_vx[0];
    env->entities[agent_idx].vy = env->entities[agent_idx].traj_vy[0];
    env->entities[agent_idx].metrics_array[COLLISION_IDX] = 0.0f;
    env->entities[agent_idx].metrics_array[OFFROAD_IDX] = 0.0f;
    env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 0.0f;
    env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX] = 0.0f;
    env->entities[agent_idx].metrics_array[AVG_DISPLACEMENT_ERROR_IDX] = 0.0f;
    env->entities[agent_idx].cumulative_displacement = 0.0f;
    env->entities[agent_idx].displacement_sample_count = 0;
    env->entities[agent_idx].respawn_timestep = env->timestep;
}

void c_step(Drive* env){
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
    env->timestep++;
    if(env->timestep == TRAJECTORY_LENGTH){
        add_log(env);
	    c_reset(env);
        return;
    }

    // Move statix experts
    for (int i = 0; i < env->expert_static_car_count; i++) {
        int expert_idx = env->expert_static_car_indices[i];
        if(env->entities[expert_idx].x == -10000.0f) continue;
        move_expert(env, env->actions, expert_idx);
    }
    // Process actions for all active agents
    for(int i = 0; i < env->active_agent_count; i++){
        env->logs[i].score = 0.0f;
	    env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];
        env->entities[agent_idx].collision_state = 0;
        move_dynamics(env, i, agent_idx);
        // move_expert(env, env->actions, agent_idx);
    }
    for(int i = 0; i < env->active_agent_count; i++){
        int agent_idx = env->active_agent_indices[i];
        env->entities[agent_idx].collision_state = 0;
        //if(env->entities[agent_idx].respawn_timestep != -1) continue;
        compute_agent_metrics(env, agent_idx);
        int collision_state = env->entities[agent_idx].collision_state;

        if(collision_state > 0){
            if(collision_state == VEHICLE_COLLISION && env->entities[agent_idx].respawn_timestep == -1){
                if(env->entities[agent_idx].respawn_timestep != -1) {
                    env->rewards[i] = env->reward_vehicle_collision_post_respawn;
                    env->logs[i].episode_return += env->reward_vehicle_collision_post_respawn;
                } else {
                    env->rewards[i] = env->reward_vehicle_collision;
                    env->logs[i].episode_return += env->reward_vehicle_collision;
                    env->logs[i].clean_collision_rate = 1.0f;
                }
                env->logs[i].collision_rate = 1.0f;
            }
            else if(collision_state == OFFROAD){
                env->rewards[i] = env->reward_offroad_collision;
                env->logs[i].offroad_rate = 1.0f;
                env->logs[i].episode_return += env->reward_offroad_collision;
            }
            if(!env->entities[agent_idx].reached_goal_this_episode){
                env->entities[agent_idx].collided_before_goal = 1;
            }
        }

        float distance_to_goal = relative_distance_2d(
                env->entities[agent_idx].x,
                env->entities[agent_idx].y,
                env->entities[agent_idx].goal_position_x,
                env->entities[agent_idx].goal_position_y);
        // Reward agent if it is within X meters of goal
        if(distance_to_goal < env->goal_radius){
            if(env->entities[agent_idx].respawn_timestep != -1){
                env->rewards[i] += env->reward_goal_post_respawn;
                env->logs[i].episode_return += env->reward_goal_post_respawn;
            } else {
                env->rewards[i] += 1.0f;
                env->logs[i].episode_return += 1.0f;
                //env->terminals[i] = 1;
            }
            env->entities[agent_idx].reached_goal_this_episode = 1;
            env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 1.0f;
	    }

        int lane_aligned = env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX];
        if(lane_aligned){
        //     env->rewards[i] += 0.01f;
        //     env->logs[i].episode_return += 0.01f;
            env->logs[i].lane_alignment_rate = 1.0f;
        }

        // Apply ADE reward
        float current_ade = env->entities[agent_idx].metrics_array[AVG_DISPLACEMENT_ERROR_IDX];
        if(current_ade > 0.0f && env->reward_ade != 0.0f) {
            float ade_reward = env->reward_ade * current_ade;
            env->rewards[i] += ade_reward;
            env->logs[i].episode_return += ade_reward;
        }
        env->logs[i].avg_displacement_error = current_ade;
    }

    for(int i = 0; i < env->active_agent_count; i++){
        int agent_idx = env->active_agent_indices[i];
        int reached_goal = env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX];
        if(reached_goal){
            respawn_agent(env, agent_idx);
            //env->entities[agent_idx].x = -10000;
            //env->entities[agent_idx].y = -10000;
            //env->entities[agent_idx].respawn_timestep = env->timestep;
        }
    }
    compute_observations(env);
}

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};
const Color LIGHTGREEN = (Color){152, 255, 152, 255};

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
    Model cars[6];
    int car_assignments[MAX_CARS];  // To keep car model assignments consistent per vehicle
    Vector3 default_camera_position;
    Vector3 default_camera_target;
};

Client* make_client(Drive* env){
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = 1280;
    client->height = 704;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(client->width, client->height, "PufferLib Ray GPU Drive");
    SetTargetFPS(30);
    client->puffers = LoadTexture("resources/puffers_128.png");
    client->cars[0] = LoadModel("resources/drive/RedCar.glb");
    client->cars[1] = LoadModel("resources/drive/WhiteCar.glb");
    client->cars[2] = LoadModel("resources/drive/BlueCar.glb");
    client->cars[3] = LoadModel("resources/drive/YellowCar.glb");
    client->cars[4] = LoadModel("resources/drive/GreenCar.glb");
    client->cars[5] = LoadModel("resources/drive/GreyCar.glb");
    for (int i = 0; i < MAX_CARS; i++) {
        client->car_assignments[i] = (rand() % 4) + 1;
    }
    // Get initial target position from first active agent
    Vector3 target_pos = {
        0,
        0,  // Y is up
        1   // Z is depth
    };

    // Set up camera to look at target from above and behind
    client->default_camera_position = (Vector3){
        0,           // Same X as target
        120.0f,   // 20 units above target
        175.0f    // 20 units behind target
    };
    client->default_camera_target = target_pos;
    client->camera.position = client->default_camera_position;
    client->camera.target = client->default_camera_target;
    client->camera.up = (Vector3){ 0.0f, -1.0f, 0.0f };  // Y is up
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->camera_zoom = 1.0f;
    return client;
}

// Camera control functions
void handle_camera_controls(Client* client) {
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
        Vector2 delta = {
            (current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
            -(current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed
        };

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
        Vector3 direction = {
            client->camera.position.x - client->camera.target.x,
            client->camera.position.y - client->camera.target.y,
            client->camera.position.z - client->camera.target.z
        };

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

void draw_agent_obs(Drive* env, int agent_index, int mode, int obs_only, int lasers){
    // Diamond dimensions
    float diamond_height = 3.0f;    // Total height of diamond
    float diamond_width = 1.5f;     // Width of diamond
    float diamond_z = 8.0f;         // Base Z position

    // Define diamond points
    Vector3 top_point = (Vector3){0.0f, 0.0f, diamond_z + diamond_height/2};     // Top point
    Vector3 bottom_point = (Vector3){0.0f, 0.0f, diamond_z - diamond_height/2};  // Bottom point
    Vector3 front_point = (Vector3){0.0f, diamond_width/2, diamond_z};           // Front point
    Vector3 back_point = (Vector3){0.0f, -diamond_width/2, diamond_z};           // Back point
    Vector3 left_point = (Vector3){-diamond_width/2, 0.0f, diamond_z};           // Left point
    Vector3 right_point = (Vector3){diamond_width/2, 0.0f, diamond_z};           // Right point

    // Draw the diamond faces
    // Top pyramid

    if(mode ==0){
        DrawTriangle3D(top_point, front_point, right_point, PUFF_CYAN);    // Front-right face
        DrawTriangle3D(top_point, right_point, back_point, PUFF_CYAN);     // Back-right face
        DrawTriangle3D(top_point, back_point, left_point, PUFF_CYAN);      // Back-left face
        DrawTriangle3D(top_point, left_point, front_point, PUFF_CYAN);     // Front-left face

        // Bottom pyramid
        DrawTriangle3D(bottom_point, right_point, front_point, PUFF_CYAN); // Front-right face
        DrawTriangle3D(bottom_point, back_point, right_point, PUFF_CYAN);  // Back-right face
        DrawTriangle3D(bottom_point, left_point, back_point, PUFF_CYAN);   // Back-left face
        DrawTriangle3D(bottom_point, front_point, left_point, PUFF_CYAN);  // Front-left face
    }
    if(!IsKeyDown(KEY_LEFT_CONTROL) && obs_only==0){
        return;
    }

    int max_obs = 7 + 7*(MAX_CARS - 1) + 7*MAX_ROAD_SEGMENT_OBSERVATIONS;
    float (*observations)[max_obs] = (float(*)[max_obs])env->observations;
    float* agent_obs = &observations[agent_index][0];
    // self
    int active_idx = env->active_agent_indices[agent_index];
    float heading_self_x = env->entities[active_idx].heading_x;
    float heading_self_y = env->entities[active_idx].heading_y;
    float px = env->entities[active_idx].x;
    float py = env->entities[active_idx].y;
    // draw goal
    float goal_x = agent_obs[0] * 200;
    float goal_y = agent_obs[1] * 200;
    if(mode == 0 ){
        DrawSphere((Vector3){goal_x, goal_y, 1}, 0.5f, LIGHTGREEN);
        DrawCircle3D((Vector3){goal_x, goal_y, 0.1f}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f, Fade(LIGHTGREEN, 0.3f));
    }

    if (mode == 1){
        float goal_x_world = px + (goal_x * heading_self_x - goal_y*heading_self_y);
        float goal_y_world = py + (goal_x * heading_self_y + goal_y*heading_self_x);
        DrawSphere((Vector3){goal_x_world, goal_y_world, 1}, 0.5f, LIGHTGREEN);
        DrawCircle3D((Vector3){goal_x_world, goal_y_world, 0.1f}, env->goal_radius, (Vector3){0, 0, 1}, 90.0f, Fade(LIGHTGREEN, 0.3f));
    }
    // First draw other agent observations
    int obs_idx = 7;  // Start after goal distances
    for(int j = 0; j < MAX_CARS - 1; j++) {
        if(agent_obs[obs_idx] == 0 || agent_obs[obs_idx + 1] == 0) {
            obs_idx += 7;  // Move to next agent observation
            continue;
        }
        // Draw position of other agents
        float x = agent_obs[obs_idx] * 50;
        float y = agent_obs[obs_idx + 1] * 50;
        if(lasers && mode == 0){
            DrawLine3D(
                (Vector3){0, 0, 0},
                (Vector3){x, y, 1},
                ORANGE
            );
        }

        float partner_x = px + (x*heading_self_x - y*heading_self_y);
        float partner_y = py + (x*heading_self_y + y*heading_self_x);
        if(lasers && mode ==1){
            DrawLine3D(
                (Vector3){px, py, 1},
                (Vector3){partner_x,partner_y,1},
                ORANGE
            );
        }

        float half_width = 0.5*agent_obs[obs_idx + 2]*MAX_VEH_WIDTH;
        float half_len = 0.5*agent_obs[obs_idx + 3]*MAX_VEH_LEN;
        float theta_x = agent_obs[obs_idx + 4];
        float theta_y = agent_obs[obs_idx + 5];
        float partner_angle = atan2f(theta_y, theta_x);
        float cos_heading = cosf(partner_angle);
        float sin_heading = sinf(partner_angle);
        Vector3 corners[4] = {
            (Vector3){
                x + (half_len * cos_heading - half_width * sin_heading),
                y + (half_len * sin_heading + half_width * cos_heading),
                1
            },
            (Vector3){
                x + (half_len * cos_heading + half_width * sin_heading),
                y + (half_len * sin_heading - half_width * cos_heading),
                1
            },
           (Vector3){
                x + (-half_len * cos_heading + half_width * sin_heading),
                y + (-half_len * sin_heading - half_width * cos_heading),
                1
            },
           (Vector3){
                x + (-half_len * cos_heading - half_width * sin_heading),
                y + (-half_len * sin_heading + half_width * cos_heading),
                1
            },
        };

        if(mode ==0){
            for (int j = 0; j < 4; j++) {
                DrawLine3D(corners[j], corners[(j+1)%4], ORANGE);
            }
        }

        if(mode ==1){
            Vector3 world_corners[4];
            for (int j = 0; j < 4; j++) {
                float lx = corners[j].x;
                float ly = corners[j].y;

                world_corners[j].x = px + (lx * heading_self_x - ly * heading_self_y);
                world_corners[j].y = py + (lx * heading_self_y + ly * heading_self_x);
                world_corners[j].z = 1;
            }
            for (int j = 0; j < 4; j++) {
                DrawLine3D(world_corners[j], world_corners[(j+1)%4], ORANGE);
            }
        }

        // draw an arrow above the car pointing in the direction that the partner is going
        float arrow_length = 7.5f;
        float arrow_x = x + arrow_length*cosf(partner_angle);
        float arrow_y = y + arrow_length*sinf(partner_angle);
        float arrow_x_world;
        float arrow_y_world;
        if(mode ==0){
            DrawLine3D((Vector3){x, y, 1}, (Vector3){arrow_x, arrow_y, 1}, PUFF_WHITE);
        }
        if(mode == 1){
            arrow_x_world = px + (arrow_x * heading_self_x - arrow_y*heading_self_y);
            arrow_y_world = py + (arrow_x * heading_self_y + arrow_y*heading_self_x);
            DrawLine3D((Vector3){partner_x, partner_y, 1}, (Vector3){arrow_x_world, arrow_y_world, 1}, PUFF_WHITE);
        }
        // Calculate perpendicular offsets for arrow head
        float arrow_size = 2.0f;  // Size of the arrow head
        float dx = arrow_x - x;
        float dy = arrow_y - y;
        float length = sqrtf(dx*dx + dy*dy);
        if (length > 0) {
            // Normalize direction vector
            dx /= length;
            dy /= length;

            // Calculate perpendicular vector

            float perp_x = -dy * arrow_size;
            float perp_y = dx * arrow_size;

            float arrow_x_end1 = arrow_x - dx*arrow_size + perp_x;
            float arrow_y_end1 = arrow_y - dy*arrow_size + perp_y;
            float arrow_x_end2 = arrow_x - dx*arrow_size - perp_x;
            float arrow_y_end2 = arrow_y - dy*arrow_size - perp_y;

            // Draw the two lines forming the arrow head
            if(mode ==0){
                DrawLine3D(
                    (Vector3){arrow_x, arrow_y, 1},
                    (Vector3){arrow_x_end1, arrow_y_end1, 1},
                    PUFF_WHITE
                );
                DrawLine3D(
                    (Vector3){arrow_x, arrow_y, 1},
                    (Vector3){arrow_x_end2, arrow_y_end2, 1},
                    PUFF_WHITE
                );
            }

            if(mode==1){
                float arrow_x_end1_world = px + (arrow_x_end1 * heading_self_x - arrow_y_end1*heading_self_y);
                float arrow_y_end1_world = py + (arrow_x_end1 * heading_self_y + arrow_y_end1*heading_self_x);
                float arrow_x_end2_world = px + (arrow_x_end2 * heading_self_x - arrow_y_end2*heading_self_y);
                float arrow_y_end2_world = py + (arrow_x_end2 * heading_self_y + arrow_y_end2*heading_self_x);
                DrawLine3D(
                    (Vector3){arrow_x_world, arrow_y_world, 1},
                    (Vector3){arrow_x_end1_world, arrow_y_end1_world, 1},
                    PUFF_WHITE
                );
                DrawLine3D(
                    (Vector3){arrow_x_world, arrow_y_world, 1},
                    (Vector3){arrow_x_end2_world, arrow_y_end2_world, 1},
                    PUFF_WHITE
                );

            }
        }

        obs_idx += 7;  // Move to next agent observation (7 values per agent)
    }
    // Then draw map observations
    int map_start_idx = 7 + 7*(MAX_CARS - 1);  // Start after agent observations
    for(int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {  // Loop through potential map entities
        int entity_idx = map_start_idx + k*7;
        if(agent_obs[entity_idx] == 0 && agent_obs[entity_idx + 1] == 0){
            continue;
        }
        Color lineColor = BLUE;  // Default color
        int entity_type = (int)agent_obs[entity_idx + 6];
        // Choose color based on entity type
        if(entity_type+4 != ROAD_EDGE){
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
        float x_start = x_middle - segment_length*cosf(rel_angle);
        float y_start = y_middle - segment_length*sinf(rel_angle);
        float x_end = x_middle + segment_length*cosf(rel_angle);
        float y_end = y_middle + segment_length*sinf(rel_angle);


        if(lasers && mode ==0){
            DrawLine3D((Vector3){0,0,0}, (Vector3){x_middle, y_middle, 1}, lineColor);
        }

        if(mode ==1){
            float x_middle_world = px + (x_middle*heading_self_x - y_middle*heading_self_y);
            float y_middle_world = py + (x_middle*heading_self_y + y_middle*heading_self_x);
            float x_start_world = px + (x_start*heading_self_x - y_start*heading_self_y);
            float y_start_world = py + (x_start*heading_self_y + y_start*heading_self_x);
            float x_end_world = px + (x_end*heading_self_x - y_end*heading_self_y);
            float y_end_world = py + (x_end*heading_self_y + y_end*heading_self_x);
            DrawCube((Vector3){x_middle_world, y_middle_world, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start_world, y_start_world, 1}, (Vector3){x_end_world, y_end_world, 1}, BLUE);
            if(lasers) DrawLine3D((Vector3){px,py,1}, (Vector3){x_middle_world, y_middle_world, 1}, lineColor);
        }
        if(mode ==0){
            DrawCube((Vector3){x_middle, y_middle, 1}, 0.5f, 0.5f, 0.5f, lineColor);
            DrawLine3D((Vector3){x_start, y_start, 1}, (Vector3){x_end, y_end, 1}, BLUE);
        }
    }
}

void draw_road_edge(Drive* env, float start_x, float start_y, float end_x, float end_y){
    Color CURB_TOP = (Color){220, 220, 220, 255};      // Top surface - lightest
    Color CURB_SIDE = (Color){180, 180, 180, 255};     // Side faces - medium
    Color CURB_BOTTOM = (Color){160, 160, 160, 255};
                    // Calculate curb dimensions
    float curb_height = 0.5f;  // Height of the curb
    float curb_width = 0.3f;   // Width/thickness of the curb
    float road_z = 0.2f;       // Ensure z-level for roads is below agents

    // Calculate direction vector between start and end
    Vector3 direction = {
        end_x - start_x,
        end_y - start_y,
        0.0f
    };

    // Calculate length of the segment
    float length = sqrtf(direction.x * direction.x + direction.y * direction.y);

    // Normalize direction vector
    Vector3 normalized_dir = {
        direction.x / length,
        direction.y / length,
        0.0f
    };

    // Calculate perpendicular vector for width
    Vector3 perpendicular = {
        -normalized_dir.y,
        normalized_dir.x,
        0.0f
    };

    // Calculate the four bottom corners of the curb
    Vector3 b1 = {
        start_x - perpendicular.x * curb_width/2,
        start_y - perpendicular.y * curb_width/2,
        road_z
    };
    Vector3 b2 = {
        start_x + perpendicular.x * curb_width/2,
        start_y + perpendicular.y * curb_width/2,
        road_z
    };
    Vector3 b3 = {
        end_x + perpendicular.x * curb_width/2,
        end_y + perpendicular.y * curb_width/2,
        road_z
    };
    Vector3 b4 = {
        end_x - perpendicular.x * curb_width/2,
        end_y - perpendicular.y * curb_width/2,
        road_z
    };

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

void draw_scene(Drive* env, Client* client, int mode, int obs_only, int lasers, int show_grid){
   // Draw a grid to help with orientation
    // DrawGrid(20, 1.0f);
    DrawLine3D((Vector3){env->grid_map->top_left_x, env->grid_map->top_left_y, 0}, (Vector3){env->grid_map->bottom_right_x, env->grid_map->top_left_y, 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->grid_map->top_left_x, env->grid_map->bottom_right_y, 0}, (Vector3){env->grid_map->top_left_x, env->grid_map->top_left_y, 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->grid_map->bottom_right_x, env->grid_map->bottom_right_y, 0}, (Vector3){env->grid_map->bottom_right_x, env->grid_map->top_left_y, 0}, PUFF_CYAN);
    DrawLine3D((Vector3){env->grid_map->top_left_x, env->grid_map->bottom_right_y, 0}, (Vector3){env->grid_map->bottom_right_x, env->grid_map->bottom_right_y, 0}, PUFF_CYAN);
    for(int i = 0; i < env->num_entities; i++) {
        // Draw cars
        if(env->entities[i].type == 1 || env->entities[i].type == 2) {
            // Check if this vehicle is an active agent
            bool is_active_agent = false;
            bool is_static_car = false;
            int agent_index = -1;
            for(int j = 0; j < env->active_agent_count; j++) {
                if(env->active_agent_indices[j] == i) {
                    is_active_agent = true;
                    agent_index = j;
                    break;
                }
            }
            for(int j = 0; j < env->static_car_count; j++) {
                if(env->static_car_indices[j] == i) {
                    is_static_car = true;
                    break;
                }
            }
            // HIDE CARS ON RESPAWN - IMPORTANT TO KNOW VISUAL SETTING
            if((!is_active_agent && !is_static_car) || env->entities[i].respawn_timestep != -1){
                continue;
            }
            Vector3 position;
            float heading;
            position = (Vector3){
                env->entities[i].x,
                env->entities[i].y,
                1
            };
            heading = env->entities[i].heading;
            // Create size vector
            Vector3 size = {
                env->entities[i].length,
                env->entities[i].width,
                env->entities[i].height
            };

            // Save current transform
            if(mode==1){
                float cos_heading = env->entities[i].heading_x;
                float sin_heading = env->entities[i].heading_y;

                // Calculate half dimensions
                float half_len = env->entities[i].length * 0.5f;
                float half_width = env->entities[i].width * 0.5f;

                // Calculate the four corners of the collision box
                Vector3 corners[4] = {
                    (Vector3){
                        position.x + (half_len * cos_heading - half_width * sin_heading),
                        position.y + (half_len * sin_heading + half_width * cos_heading),
                        position.z
                    },


                    (Vector3){
                        position.x + (half_len * cos_heading + half_width * sin_heading),
                        position.y + (half_len * sin_heading - half_width * cos_heading),
                        position.z
                    },
                   (Vector3){
                        position.x + (-half_len * cos_heading + half_width * sin_heading),
                        position.y + (-half_len * sin_heading - half_width * cos_heading),
                        position.z
                    },
                   (Vector3){
                        position.x + (-half_len * cos_heading - half_width * sin_heading),
                        position.y + (-half_len * sin_heading + half_width * cos_heading),
                        position.z
                    },


                };

                if(agent_index == env->human_agent_idx && !env->entities[agent_index].metrics_array[REACHED_GOAL_IDX]) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }
                if((obs_only ||  IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx){
                    continue;
                }

                // --- Draw the car  ---

                Vector3 carPos = { position.x, position.y, position.z };
                Color car_color = GRAY;
                if(is_active_agent){
                    car_color = BLUE;
                }
                if(is_active_agent && env->entities[i].collision_state > 0) {
                    car_color = RED;
                }
                rlSetLineWidth(3.0f);
                for (int j = 0; j < 4; j++) {
                    DrawLine3D(corners[j], corners[(j+1)%4], car_color);
                }
                // --- Draw a heading arrow pointing forward ---
                Vector3 arrowStart = position;
                Vector3 arrowEnd = {
                    position.x + cos_heading * half_len * 1.5f, // extend arrow beyond car
                    position.y + sin_heading * half_len * 1.5f,
                    position.z
                };

                DrawLine3D(arrowStart, arrowEnd, car_color);
                DrawSphere(arrowEnd, 0.2f, car_color);  // arrow tip

            }
            else {
                rlPushMatrix();
                // Translate to position, rotate around Y axis, then draw
                rlTranslatef(position.x, position.y, position.z);
                rlRotatef(heading*RAD2DEG, 0.0f, 0.0f, 1.0f);  // Convert radians to degrees
                // Determine color based on active status and other conditions
                Color object_color = PUFF_BACKGROUND2;  // Default color for non-active vehicles
                Color outline_color = PUFF_CYAN;
                Model car_model = client->cars[5];
                if(is_active_agent){
                    car_model = client->cars[client->car_assignments[i %64]];
                }
                if(agent_index == env->human_agent_idx){
                    object_color = PUFF_CYAN;
                    outline_color = PUFF_WHITE;
                }
                if(is_active_agent && env->entities[i].collision_state > 0) {
                    car_model = client->cars[0];  // Collided agent
                }
                // Draw obs for human selected agent
                if(agent_index == env->human_agent_idx && !env->entities[agent_index].metrics_array[REACHED_GOAL_IDX]) {
                    draw_agent_obs(env, agent_index, mode, obs_only, lasers);
                }
                // Draw cube for cars static and active
                // Calculate scale factors based on desired size and model dimensions

                BoundingBox bounds = GetModelBoundingBox(car_model);
                Vector3 model_size = {
                    bounds.max.x - bounds.min.x,
                    bounds.max.y - bounds.min.y,
                    bounds.max.z - bounds.min.z
                };
                Vector3 scale = {
                    size.x / model_size.x,
                    size.y / model_size.y,
                    size.z / model_size.z
                };
                if((obs_only ||  IsKeyDown(KEY_LEFT_CONTROL)) && agent_index != env->human_agent_idx){
                    rlPopMatrix();
                    continue;
                }

                DrawModelEx(car_model, (Vector3){0, 0, 0}, (Vector3){1, 0, 0}, 90.0f, scale, WHITE);
                rlPopMatrix();
            }

            // FPV Camera Control
            if(IsKeyDown(KEY_SPACE) && env->human_agent_idx== agent_index){
                if(env->entities[agent_index].metrics_array[REACHED_GOAL_IDX]){
                    env->human_agent_idx = rand() % env->active_agent_count;
                }
                Vector3 camera_position = (Vector3){
                        position.x - (25.0f * cosf(heading)),
                        position.y - (25.0f * sinf(heading)),
                        position.z + 15
                };

                Vector3 camera_target = (Vector3){
                    position.x + 40.0f * cosf(heading),
                    position.y + 40.0f * sinf(heading),
                    position.z - 5.0f
                };
                client->camera.position = camera_position;
                client->camera.target = camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }
            if(IsKeyReleased(KEY_SPACE)){
                client->camera.position = client->default_camera_position;
                client->camera.target = client->default_camera_target;
                client->camera.up = (Vector3){0, 0, 1};
            }
            // Draw goal position for active agents

            if(!is_active_agent || env->entities[i].valid == 0) {
                continue;
            }
            if(!IsKeyDown(KEY_LEFT_CONTROL) && obs_only==0){
                DrawSphere((Vector3){
                    env->entities[i].goal_position_x,
                    env->entities[i].goal_position_y,
                    1
                }, 0.5f, DARKGREEN);

                DrawCircle3D((Vector3){
                    env->entities[i].goal_position_x,
                    env->entities[i].goal_position_y,
                    0.1f
                }, env->goal_radius, (Vector3){0, 0, 1}, 90.0f, Fade(LIGHTGREEN, 0.3f));
            }
        }
        // Draw road elements
        if(env->entities[i].type <=3 && env->entities[i].type >= 7){
            continue;
        }
        for(int j = 0; j < env->entities[i].array_size - 1; j++) {
            Vector3 start = {
                env->entities[i].traj_x[j],
                env->entities[i].traj_y[j],
                1
            };
            Vector3 end = {
                env->entities[i].traj_x[j + 1],
                env->entities[i].traj_y[j + 1],
                1
            };
            Color lineColor = GRAY;
            if (env->entities[i].type == ROAD_LANE) lineColor = GRAY;
            else if (env->entities[i].type == ROAD_LINE) lineColor = BLUE;
            else if (env->entities[i].type == ROAD_EDGE) lineColor = WHITE;
            else if (env->entities[i].type == DRIVEWAY) lineColor = RED;
            if(env->entities[i].type != ROAD_EDGE){
                continue;
            }
            if(!IsKeyDown(KEY_LEFT_CONTROL) && obs_only==0){
                draw_road_edge(env, start.x, start.y, end.x, end.y);
            }
        }
    }
    if(show_grid) {
    // Draw grid cells using the stored bounds
    float grid_start_x = env->grid_map->top_left_x;
    float grid_start_y = env->grid_map->bottom_right_y;
    for(int i = 0; i < env->grid_map->grid_cols; i++) {
        for(int j = 0; j < env->grid_map->grid_rows; j++) {
            float x = grid_start_x + i*GRID_CELL_SIZE;
            float y = grid_start_y + j*GRID_CELL_SIZE;
            DrawCubeWires(
                (Vector3){x + GRID_CELL_SIZE/2, y + GRID_CELL_SIZE/2, 1},
                GRID_CELL_SIZE, GRID_CELL_SIZE, 0.1f, PUFF_BACKGROUND2);
        }
        }
    }

    EndMode3D();

}

void saveTopDownImage(Drive* env, Client* client, const char *filename, RenderTexture2D target, int map_height, int obs, int lasers, int trajectories, int frame_count, float* path, int log_trajectories, int show_grid){
    // Top-down orthographic camera
    Camera3D camera = {0};
    camera.position = (Vector3){ 0.0f, 0.0f, 500.0f };  // above the scene
    camera.target   = (Vector3){ 0.0f, 0.0f, 0.0f };  // look at origin
    camera.up       = (Vector3){ 0.0f, -1.0f, 0.0f };
    camera.fovy     = map_height;
    camera.projection = CAMERA_ORTHOGRAPHIC;
    Color road = (Color){35, 35, 37, 255};

    BeginTextureMode(target);
        ClearBackground(road);
        BeginMode3D(camera);
            rlEnableDepthTest();

            // Draw log trajectories FIRST (in background at lower Z-level)
            if(log_trajectories){
                for(int i=0; i<env->active_agent_count;i++){
                    int idx = env->active_agent_indices[i];
                    for(int j=0; j<TRAJECTORY_LENGTH;j++){
                        float x = env->entities[idx].traj_x[j];
                        float y = env->entities[idx].traj_y[j];
                        float valid = env->entities[idx].traj_valid[j];
                        if(!valid) continue;
                        DrawSphere((Vector3){x,y,0.5f}, 0.3f, Fade(LIGHTGREEN, 0.6f));
                    }
                }
            }

            // Draw current path trajectories SECOND (slightly higher than log trajectories)
            if(trajectories){
                for(int i=0; i<frame_count; i++){
                    DrawSphere((Vector3){path[i*2], path[i*2 +1], 0.8f}, 0.5f, YELLOW);
                }
            }

            // Draw main scene LAST (on top)
            draw_scene(env, client, 1, obs, lasers, show_grid);

        EndMode3D();
    EndTextureMode();

    // save to file
    Image img = LoadImageFromTexture(target.texture);
    ImageFlipVertical(&img);
    ExportImage(img, filename);
    UnloadImage(img);
}

void saveAgentViewImage(Drive* env, Client* client, const char *filename, RenderTexture2D target, int map_height, int obs_only, int lasers, int show_grid) {
    // Agent perspective camera following the human agent
    int agent_idx = env->active_agent_indices[env->human_agent_idx];
    Entity* agent = &env->entities[agent_idx];

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

    BeginTextureMode(target);
        ClearBackground(road);
        BeginMode3D(camera);
            rlEnableDepthTest();
            draw_scene(env, client, 0, obs_only, lasers, show_grid); // mode=0 for agent view
        EndMode3D();
    EndTextureMode();

    // Save to file
    Image img = LoadImageFromTexture(target.texture);
    ImageFlipVertical(&img);
    ExportImage(img, filename);
    UnloadImage(img);
}

void c_render(Drive* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client* client = env->client;
    BeginDrawing();
    Color road = (Color){35, 35, 37, 255};
    ClearBackground(road);
    BeginMode3D(client->camera);
    handle_camera_controls(env->client);
    draw_scene(env, client, 0, 0, 0, 0);
    // Draw debug info
    DrawText(TextFormat("Camera Position: (%.2f, %.2f, %.2f)",
        client->camera.position.x,
        client->camera.position.y,
        client->camera.position.z), 10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Camera Target: (%.2f, %.2f, %.2f)",
        client->camera.target.x,
        client->camera.target.y,
        client->camera.target.z), 10, 30, 20, PUFF_WHITE);
    DrawText(TextFormat("Timestep: %d", env->timestep), 10, 50, 20, PUFF_WHITE);
    // acceleration & steering
    int human_idx = env->active_agent_indices[env->human_agent_idx];
    DrawText(TextFormat("Controlling Agent: %d", env->human_agent_idx), 10, 70, 20, PUFF_WHITE);
    DrawText(TextFormat("Agent Index: %d", human_idx), 10, 90, 20, PUFF_WHITE);
    // Controls help
    DrawText("Controls: W/S - Accelerate/Brake, A/D - Steer, 1-4 - Switch Agent",
             10, client->height - 30, 20, PUFF_WHITE);
    // acceleration & steering
    if (env->action_type == 1) { // continuous (float)
        float (*action_array_f)[2] = (float(*)[2])env->actions;
        DrawText(TextFormat("Acceleration: %.2f", action_array_f[env->human_agent_idx][0]), 10, 110, 20, PUFF_WHITE);
        DrawText(TextFormat("Steering: %.2f", action_array_f[env->human_agent_idx][1]), 10, 130, 20, PUFF_WHITE);
    } else { // discrete (int)
        int (*action_array)[2] = (int(*)[2])env->actions;
        DrawText(TextFormat("Acceleration: %d", action_array[env->human_agent_idx][0]), 10, 110, 20, PUFF_WHITE);
        DrawText(TextFormat("Steering: %d", action_array[env->human_agent_idx][1]), 10, 130, 20, PUFF_WHITE);
    }
    DrawText(TextFormat("Grid Rows: %d", env->grid_map->grid_rows), 10, 150, 20, PUFF_WHITE);
    DrawText(TextFormat("Grid Cols: %d", env->grid_map->grid_cols), 10, 170, 20, PUFF_WHITE);
    EndDrawing();
}

void close_client(Client* client){
    for (int i = 0; i < 6; i++) {
        UnloadModel(client->cars[i]);
    }
    UnloadTexture(client->puffers);
    CloseWindow();
    free(client);
}
