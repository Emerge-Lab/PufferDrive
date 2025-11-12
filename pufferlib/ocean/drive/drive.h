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

#define INVALID_POSITION -10000.0f

// Trajectory Length
#define TRAJECTORY_LENGTH 91

// Initialization modes
#define INIT_ALL_VALID 0
#define INIT_ONLY_CONTROLLABLE_AGENTS 1

// Control modes
#define CONTROL_VEHICLES 0
#define CONTROL_AGENTS 1
#define CONTROL_TRACKS_TO_PREDICT 2

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
#define AVG_DISPLACEMENT_ERROR_IDX 4

// Grid cell size
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL 30    // Depends on resolution of data Formula: 3 * (2 + GRID_CELL_SIZE*sqrt(2)/resolution) => For each entity type in gridmap, diagonal poly-lines -> sqrt(2), include diagonal ends -> 2

// Max road segment observation entities
#define MAX_ROAD_SEGMENT_OBSERVATIONS 200
#define MAX_AGENTS 64
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
#define STOP_AGENT 1
#define REMOVE_AGENT 2

//GOAL BEHAVIOUR
#define GOAL_RESPAWN 0
#define GOAL_GENERATE_NEW 1
#define GOAL_STOP 2

// Jerk action space (for JERK dynamics model)
static const float JERK_LONG[4] = {-15.0f, -4.0f, 0.0f, 4.0f};
static const float JERK_LAT[3] = {-4.0f, 0.0f, 4.0f};

// Classic action space (for CLASSIC dynamics model)
static const float ACCELERATION_VALUES[7] = {-4.0000f, -2.6670f, -1.3330f, -0.0000f, 1.3330f, 2.6670f, 4.0000f};
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
typedef struct Graph Graph;
typedef struct AdjListNode AdjListNode;
typedef struct Adaptive_Agent_Log Adaptive_Agent_Log;

struct Log {
    float episode_return;
    float episode_length;
    float score;
    float offroad_rate;
    float collision_rate;
    float num_goals_reached;
    float completion_rate;
    float dnf_rate;
    float n;
    float lane_alignment_rate;
    float avg_displacement_error;
    float active_agent_count;
    float expert_static_agent_count;
    float static_agent_count;
    // Conditioning metrics
    float avg_collision_weight;
    float avg_offroad_weight;
    float avg_goal_weight;
    float avg_entropy_weight;
    float avg_discount_weight;
    float avg_offroad_per_agent;
    float avg_collisions_per_agent;
};

typedef struct Entity Entity;
struct Entity {
    int scenario_id;
    int type;
    int id;
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
    float init_goal_x;
    float init_goal_y;
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
    int current_lane_idx;
    int valid;
    int respawn_timestep;
    int respawn_count;
    int collided_before_goal;
    int sampled_new_goal;
    int reached_goal_this_episode;
    int num_goals_reached;
    int active_agent;
    float cumulative_displacement;
    int displacement_sample_count;
    float goal_radius;
    int stopped; // 0/1 -> freeze if set
    int removed; //0/1 -> remove from sim if set

    // Jerk dynamics
    float a_long;
    float a_lat;
    float jerk_long;
    float jerk_lat;
    float steering_angle;
    float wheelbase;
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

struct Adaptive_Agent_Log {
    // Array of Log structs, one per scenario
    Log* scenario_logs;

    // Delta metrics: last scenario - first scenario (single values per agent)
    float delta_completion_rate;
    float delta_score;
    float delta_perf;
    float delta_collision_rate;
    float delta_offroad_rate;
    float delta_num_goals_reached;
    float delta_dnf_rate;
    float delta_lane_alignment_rate;
    float delta_avg_displacement_error;
    float delta_episode_return;

    int num_scenarios;
};

// Utility functions
float compute_delta_percent(float first, float last) {
    if (fabs(first) < 0.0001f) {
        return 0.0f;
    }
    return (last - first) / first * 100.0f;
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

float clip(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
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

    if (ref_x == INVALID_POSITION || ref_y == INVALID_POSITION) {
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
    Graph* topology_graph;
    int num_entities;
    int num_actors;
    int num_objects;
    int num_roads;
    int static_agent_count;
    int* static_agent_indices;
    int expert_static_agent_count;
    int* expert_static_agent_indices;
    int timestep;
    int init_steps;
    int dynamics_model;
    GridMap* grid_map;
    int* neighbor_offsets;
    int scenario_length;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    float reward_ade;
    char* map_name;
    float world_mean_x;
    float world_mean_y;
    float dt;
    float reward_goal;
    float reward_goal_post_respawn;
    float goal_radius;
    int max_controlled_agents;
    int logs_capacity;
    int goal_behavior;
    char* ini_file;
    char* scenario_id;
    int collision_behavior;
    int offroad_behavior;
    int sdc_track_index;
    int num_tracks_to_predict;
    int* tracks_to_predict_indices;
    int init_mode;
    int control_mode;

    // Reward conditioning
    bool use_rc;
    float collision_weight_lb;
    float collision_weight_ub;
    float offroad_weight_lb;
    float offroad_weight_ub;
    float goal_weight_lb;
    float goal_weight_ub;
    float* collision_weights;
    float* offroad_weights;
    float* goal_weights;
    // Entropy conditioning
    bool use_ec;
    float entropy_weight_lb;
    float entropy_weight_ub;
    float* entropy_weights;
    // Discount conditioning
    bool use_dc;
    float discount_weight_lb;
    float discount_weight_ub;
    float* discount_weights;
    // Adaptive driving agent
    Adaptive_Agent_Log* ada_log;
    Adaptive_Agent_Log** ada_logs;
    bool adaptive_driving_agent;
    int k_scenarios;
    int current_scenario;
};

void add_log(Drive* env) {
    int scenario = -1;
    if (env->adaptive_driving_agent && env->ada_logs != NULL) {
        scenario = env->current_scenario % env->k_scenarios;
    }

    for(int i = 0; i < env->active_agent_count; i++){
        Entity* e = &env->entities[env->active_agent_indices[i]];

        if(e->reached_goal_this_episode){
            env->log.completion_rate += 1.0f;
        }
        int offroad = env->logs[i].offroad_rate;
        env->log.offroad_rate += offroad;
        int collided = env->logs[i].collision_rate;
        env->log.collision_rate += collided;
        float avg_offroad_per_agent = env->logs[i].avg_offroad_per_agent;
        env->log.avg_offroad_per_agent += avg_offroad_per_agent;
        float avg_collisions_per_agent = env->logs[i].avg_collisions_per_agent;
        env->log.avg_collisions_per_agent += avg_collisions_per_agent;
        int num_goals_reached = env->logs[i].num_goals_reached;
        env->log.num_goals_reached += num_goals_reached;
        if(e->reached_goal_this_episode && !e->collided_before_goal){
            env->log.score += 1.0f;
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
        // Log composition counts per agent so vec_log averaging recovers the per-env value
        env->log.active_agent_count += env->active_agent_count;
        env->log.expert_static_agent_count += env->expert_static_agent_count;
        env->log.static_agent_count += env->static_agent_count;
        env->log.n += 1;

        // ADDITIONALLY, if adaptive, also track per-scenario for delta computation
        if (env->adaptive_driving_agent && env->ada_logs != NULL) {
            Log* scenario_log = &env->ada_logs[i]->scenario_logs[scenario];

            if (e->reached_goal_this_episode)
                scenario_log->completion_rate += 1.0f;

            scenario_log->offroad_rate += offroad;
            scenario_log->collision_rate += collided;
            scenario_log->num_goals_reached += num_goals_reached;

            if (e->reached_goal_this_episode && !e->collided_before_goal) {
                scenario_log->score += 1.0f;
            }

            if (!offroad && !collided && !e->reached_goal_this_episode) {
                scenario_log->dnf_rate += 1.0f;
            }

            scenario_log->lane_alignment_rate += lane_aligned;
            scenario_log->avg_displacement_error += displacement_error;
            scenario_log->episode_length += env->logs[i].episode_length;
            scenario_log->episode_return += env->logs[i].episode_return;

            scenario_log->active_agent_count += env->active_agent_count;
            scenario_log->expert_static_agent_count += env->expert_static_agent_count;
            scenario_log->static_agent_count += env->static_agent_count;
            scenario_log->n += 1.0f;

            // Compute delta metrics when completing last scenario
            if (scenario == env->k_scenarios - 1) {
                Log* first_log = &env->ada_logs[i]->scenario_logs[0];
                Log* last_log = &env->ada_logs[i]->scenario_logs[env->k_scenarios - 1];

                // Compute all delta metrics
                env->ada_logs[i]->delta_completion_rate =
                    compute_delta_percent(first_log->completion_rate, last_log->completion_rate);
                env->ada_logs[i]->delta_score =
                    compute_delta_percent(first_log->score, last_log->score);
                env->ada_logs[i]->delta_perf =
                    compute_delta_percent(first_log->score, last_log->score);
                env->ada_logs[i]->delta_collision_rate =
                    compute_delta_percent(first_log->collision_rate, last_log->collision_rate);
                env->ada_logs[i]->delta_offroad_rate =
                    compute_delta_percent(first_log->offroad_rate, last_log->offroad_rate);
                env->ada_logs[i]->delta_num_goals_reached =
                    compute_delta_percent(first_log->num_goals_reached, last_log->num_goals_reached);
                env->ada_logs[i]->delta_dnf_rate =
                    compute_delta_percent(first_log->dnf_rate, last_log->dnf_rate);
                env->ada_logs[i]->delta_lane_alignment_rate =
                    compute_delta_percent(first_log->lane_alignment_rate, last_log->lane_alignment_rate);

                float first_ade = (first_log->n != 0.0f) ?
                    first_log->avg_displacement_error / first_log->n : 0.0f;
                float last_ade = (last_log->n != 0.0f) ?
                    last_log->avg_displacement_error / last_log->n : 0.0f;
                env->ada_logs[i]->delta_avg_displacement_error =
                    compute_delta_percent(first_ade, last_ade);

                float first_return = (first_log->n != 0.0f) ?
                    first_log->episode_return / first_log->n : 0.0f;
                float last_return = (last_log->n != 0.0f) ?
                    last_log->episode_return / last_log->n : 0.0f;
                env->ada_logs[i]->delta_episode_return =
                    compute_delta_percent(first_return, last_return);
            }
        }
    }
}


struct AdjListNode {
    int dest;
    struct AdjListNode* next;
};

struct Graph {
    int V;
    struct AdjListNode** array;
};

// Function to create a new adjacency list node
struct AdjListNode* newAdjListNode(int dest) {
    struct AdjListNode* newNode = malloc(sizeof(struct AdjListNode));
    newNode->dest = dest;
    newNode->next = NULL;
    return newNode;
}

// Function to create a graph of V vertices
struct Graph* createGraph(int V) {
    struct Graph* graph = malloc(sizeof(struct Graph));
    graph->V = V;
    graph->array = calloc(V, sizeof(struct AdjListNode*));
    return graph;
}

// Function to get next lanes from a given lane entity index
// Returns the number of next lanes found, fills next_lanes array with entity indices
int getNextLanes(struct Graph* graph, int entity_idx, int* next_lanes, int max_lanes) {
    if (!graph || entity_idx < 0 || entity_idx >= graph->V) {
        return 0;
    }

    int count = 0;
    struct AdjListNode* node = graph->array[entity_idx];

    while (node && count < max_lanes) {
        next_lanes[count] = node->dest;
        count++;
        node = node->next;
    }

    return count;
}

// Function to free the topology graph
void freeTopologyGraph(struct Graph* graph) {
    if (!graph) return;

    for (int i = 0; i < graph->V; i++) {
        struct AdjListNode* node = graph->array[i];
        while (node) {
            struct AdjListNode* temp = node;
            node = node->next;
            free(temp);
        }
    }

    free(graph->array);
    free(graph);
}


Entity* load_map_binary(const char* filename, Drive* env) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;


    // Read sdc_track_index
    fread(&env->sdc_track_index, sizeof(int), 1, file);

    // Read tracks_to_predict
    fread(&env->num_tracks_to_predict, sizeof(int), 1, file);
    if (env->num_tracks_to_predict > 0) {
        env->tracks_to_predict_indices = (int*)malloc(env->num_tracks_to_predict * sizeof(int));

        for (int i = 0; i < env->num_tracks_to_predict; i++) {
            fread(&env->tracks_to_predict_indices[i], sizeof(int), 1, file);
        }
    } else {
        env->tracks_to_predict_indices = NULL;
    }

    fread(&env->num_objects, sizeof(int), 1, file);
    fread(&env->num_roads, sizeof(int), 1, file);
    env->num_entities = env->num_objects + env->num_roads;
    Entity* entities = (Entity*)malloc(env->num_entities * sizeof(Entity));
    for (int i = 0; i < env->num_entities; i++) {
	    // Read base entity data
        fread(&entities[i].scenario_id, sizeof(int), 1, file);
        fread(&entities[i].type, sizeof(int), 1, file);
        fread(&entities[i].id, sizeof(int), 1, file);
        fread(&entities[i].array_size, sizeof(int), 1, file);
        // Allocate arrays based on type
        int size = entities[i].array_size;
        entities[i].traj_x = (float*)malloc(size * sizeof(float));
        entities[i].traj_y = (float*)malloc(size * sizeof(float));
        entities[i].traj_z = (float*)malloc(size * sizeof(float));
        if (entities[i].type == VEHICLE || entities[i].type == PEDESTRIAN || entities[i].type == CYCLIST) {  // Object type
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
        if (entities[i].type == VEHICLE || entities[i].type == PEDESTRIAN || entities[i].type == CYCLIST) {  // Object type
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

        // Clamp init_steps to ensure we don't go out of bounds
        int step = env->init_steps;
        if (step >= e->array_size) step = e->array_size - 1;
        if (step < 0) step = 0;

        e->x = e->traj_x[step];
        e->y = e->traj_y[step];
        e->z = e->traj_z[step];

        if(e->type > CYCLIST || e->type == 0){
            continue;
        }
        if(is_active == 0){
            e->vx = 0;
            e->vy = 0;
            e->vz = 0;
            e->collided_before_goal = 0;
        } else {
            e->vx = e->traj_vx[env->init_steps];
            e->vy = e->traj_vy[env->init_steps];
            e->vz = e->traj_vz[env->init_steps];
        }
        e->heading = e->traj_heading[env->init_steps];
        e->heading_x = cosf(e->heading);
        e->heading_y = sinf(e->heading);
        e->valid = e->traj_valid[env->init_steps];
        e->collision_state = 0;
        e->metrics_array[COLLISION_IDX] = 0.0f; // vehicle collision
        e->metrics_array[OFFROAD_IDX] = 0.0f; // offroad
        e->metrics_array[REACHED_GOAL_IDX] = 0.0f; // reached goal
        e->metrics_array[LANE_ALIGNED_IDX] = 0.0f; // lane aligned
        e->metrics_array[AVG_DISPLACEMENT_ERROR_IDX] = 0.0f; // avg displacement error
        e->cumulative_displacement = 0.0f;
        e->displacement_sample_count = 0;
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
        e->wheelbase = 0.6f * e->length;
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


void init_topology_graph(Drive* env){
    // Count ROAD_LANE entities
    int road_lane_count = 0;
    for(int i = 0; i < env->num_entities; i++){
        if(env->entities[i].type == ROAD_LANE){
            road_lane_count++;
        }
    }

    if(road_lane_count == 0){
        env->topology_graph = NULL;
        return;
    }

    // Create graph with all entities as vertices (we'll only use ROAD_LANE indices)
    env->topology_graph = createGraph(env->num_entities);

    // Connect ROAD_LANE entities based on geometric connectivity
    for(int i = 0; i < env->num_entities; i++){
        if(env->entities[i].type != ROAD_LANE) continue;

        Entity* lane_i = &env->entities[i];
        if(lane_i->array_size < 2) continue; // Need at least 2 points

        // Get end point of current lane
        float end_x = lane_i->traj_x[lane_i->array_size - 1];
        float end_y = lane_i->traj_y[lane_i->array_size - 1];
        float end_vector_x = lane_i->traj_x[lane_i->array_size - 1] - lane_i->traj_x[lane_i->array_size - 2];
        float end_vector_y = lane_i->traj_y[lane_i->array_size - 1] - lane_i->traj_y[lane_i->array_size - 2];
        float end_heading = atan2f(end_vector_y, end_vector_x);

        // Find lanes that start near this lane's end
        for(int j = 0; j < env->num_entities; j++){
            if(i == j || env->entities[j].type != ROAD_LANE) continue;

            Entity* lane_j = &env->entities[j];
            if(lane_j->array_size < 2) continue;

            // Get start point of potential next lane
            float start_x = lane_j->traj_x[0];
            float start_y = lane_j->traj_y[0];
            float start_vector_x = lane_j->traj_x[1] - lane_j->traj_x[0];
            float start_vector_y = lane_j->traj_y[1] - lane_j->traj_y[0];
            float start_heading = atan2f(start_vector_y, start_vector_x);

            // Check if end of lane_i is close to start of lane_j
            float distance = relative_distance_2d(end_x, end_y, start_x, start_y);
            float heading_diff = fabsf(end_heading - start_heading);

            // Lane connectivity thresholds:
            // - 0.01m distance: lanes must connect within 1cm (very strict for clean topology)
            // - 0.1 (~5.7 degrees) heading difference: allow slight curves
            if(distance < 0.01f && heading_diff < 0.1f){
                // Add directed edge from i to j (lane i connects to lane j)
                struct AdjListNode* node = newAdjListNode(j);
                node->next = env->topology_graph->array[i];
                env->topology_graph->array[i] = node;
            }
        }
    }
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
                if(e->traj_x[j] == INVALID_POSITION) continue;
                if(e->traj_y[j] == INVALID_POSITION) continue;
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
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN || env->entities[i].type == CYCLIST) {
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
        if (env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN || env->entities[i].type == CYCLIST || env->entities[i].type >= 4) {
            for (int j = 0; j < env->entities[i].array_size; j++) {
                if(env->entities[i].traj_x[j] == INVALID_POSITION) continue;
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

    if(agent->x == INVALID_POSITION ) return -1;

    int car_collided_with_index = -1;

    if (agent->respawn_timestep != -1) return car_collided_with_index; // Skip respawning entities

    for(int i = 0; i < MAX_AGENTS; i++){
        int index = -1;
        if(i < env->active_agent_count){
            index = env->active_agent_indices[i];
        } else if (i < env->num_actors){
            index = env->static_agent_indices[i - env->active_agent_count];
        }
        if(index == -1) continue;
        if(index == agent_idx) continue;
        Entity* entity = &env->entities[index];
        if (entity->respawn_timestep != -1) continue; // Skip respawning entities
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

    // within 15 degrees
    return (heading_diff < (M_PI / 12.0f)) ? 1 : 0;
}

void reset_agent_metrics(Drive* env, int agent_idx){
    Entity* agent = &env->entities[agent_idx];
    agent->metrics_array[COLLISION_IDX] = 0.0f; // vehicle collision
    agent->metrics_array[OFFROAD_IDX] = 0.0f; // offroad
    agent->metrics_array[LANE_ALIGNED_IDX] = 0.0f; // lane aligned
    agent->metrics_array[AVG_DISPLACEMENT_ERROR_IDX] = 0.0f;
    agent->collision_state = 0;
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
    if (t < 0) t = 0;
    else if (t > 1) t = 1;

    // Find the closest point on the segment
    float closestX = x1 + t * dx;
    float closestY = y1 + t * dy;

    // Return the distance from p to the closest point
    return sqrtf((px - closestX) * (px - closestX) + (py - closestY) * (py - closestY));
}

void compute_agent_metrics(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];

    reset_agent_metrics(env, agent_idx);

    if(agent->x == INVALID_POSITION ) return; // invalid agent position

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

            float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
            float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};

            float dist = point_to_segment_distance_2d(agent->x, agent->y, start[0], start[1], end[0], end[1]);
            float heading_diff = fabsf(atan2f(end[1]-start[1], end[0]-start[0]) - agent->heading);

            // Normalize heading difference to [0, pi]
            if (heading_diff > M_PI) heading_diff = 2.0f * M_PI - heading_diff;

            // Penalize if heading differs by more than 30 degrees
            if (heading_diff > (M_PI / 6.0f)) dist += 3.0f;

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
        agent->current_lane_idx = -1;
    } else {
        agent->current_lane_idx = closest_lane_entity_idx;

        int lane_aligned = check_lane_aligned(agent, &env->entities[closest_lane_entity_idx], closest_lane_geometry_idx);
        agent->metrics_array[LANE_ALIGNED_IDX] = lane_aligned;
    }

    // Check for vehicle collisions
    int car_collided_with_index = collision_check(env, agent_idx);
    if (car_collided_with_index != -1) collided = VEHICLE_COLLISION;

    agent->collision_state = collided;

    return;
}

bool should_control_agent(Drive* env, int agent_idx){

    // Check if we have room for more agents or are already at capacity
    if (env->active_agent_count >= env->num_agents) {
        return false;
    }

    Entity* entity = &env->entities[agent_idx];

    // Shrink agent size for collision checking
    entity->width *= 0.7f; // TODO: Move this somewhere else
    entity->length *= 0.7f;

    // Special mode: control only agents in prediction track list
    if (env->control_mode == CONTROL_TRACKS_TO_PREDICT) {
        for (int j = 0; j < env->num_tracks_to_predict; j++) {
            if (env->tracks_to_predict_indices[j] == agent_idx) {
                return true;
            }
        }
        return false;
    }

    // Standard mode: check type, distance to goal, and expert status
    bool type_is_controllable = false;
    if (env->control_mode == CONTROL_VEHICLES) {
        type_is_controllable = (entity->type == VEHICLE);
    } else {  // CONTROL_AGENTS mode
        type_is_controllable = (entity->type == VEHICLE || entity->type == PEDESTRIAN || entity->type == CYCLIST);
    }

    if (!type_is_controllable || entity->mark_as_expert) {
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

void set_active_agents(Drive* env){

    // Initialize
    env->active_agent_count = 0; // Policy-controlled agents
    env->static_agent_count = 0; // Non-moving background agents
    env->expert_static_agent_count = 0; // Expert replay agents (non-controlled)
    env->num_actors = 0; // Total agents created

    int active_agent_indices[MAX_AGENTS];
    int static_agent_indices[MAX_AGENTS];
    int expert_static_agent_indices[MAX_AGENTS];

    if(env->num_agents == 0){
        env->num_agents = MAX_AGENTS;
    }

    // Iterate through entities to find agents to create and/or control
    for(int i = 0; i < env->num_objects && env->num_actors < MAX_AGENTS; i++){

        Entity* entity = &env->entities[i];

        // Skip if not valid at initialization
        if (entity->traj_valid[env->init_steps] != 1) {
            continue;
        }

        // Determine if entity should be created
        bool should_create = false;
        if (env->init_mode == INIT_ALL_VALID) {
            should_create = true;  // All valid entities
        } else if (env->control_mode == CONTROL_VEHICLES) {
            should_create = (entity->type == VEHICLE);
        } else {  // Control all agents
            should_create = (entity->type == VEHICLE || entity->type == PEDESTRIAN || entity->type == CYCLIST);
        }

        if (!should_create) continue;

        env->num_actors++;

        // Determine if this agent should be policy-controlled
        bool is_controlled = false;
        is_controlled = should_control_agent(env, i);

        if (is_controlled && env->active_agent_count >= env->max_controlled_agents && env->max_controlled_agents != -1) {
            is_controlled = false;
            entity->mark_as_expert = 1;
        }

        if(is_controlled){
            active_agent_indices[env->active_agent_count] = i;
            env->active_agent_count++;
            env->entities[i].active_agent = 1;
        } else if (env->init_mode != INIT_ONLY_CONTROLLABLE_AGENTS) {
            static_agent_indices[env->static_agent_count] = i;
            env->static_agent_count++;
            env->entities[i].active_agent = 0;
            if(env->entities[i].mark_as_expert == 1 || env->active_agent_count == env->num_agents) {
                expert_static_agent_indices[env->expert_static_agent_count] = i;
                env->expert_static_agent_count++;
                env->entities[i].mark_as_expert = 1;
            }
        }
    }

    // Set up initial active agents
    env->active_agent_indices = (int*)malloc(env->active_agent_count * sizeof(int));
    env->static_agent_indices = (int*)malloc(env->static_agent_count * sizeof(int));
    env->expert_static_agent_indices = (int*)malloc(env->expert_static_agent_count * sizeof(int));
    for(int i=0;i<env->active_agent_count;i++){
        env->active_agent_indices[i] = active_agent_indices[i];
    }
    for(int i=0;i<env->static_agent_count;i++){
        env->static_agent_indices[i] = static_agent_indices[i];
    }
    for(int i=0;i<env->expert_static_agent_count;i++){
        env->expert_static_agent_indices[i] = expert_static_agent_indices[i];
    }
    return;
}

void remove_bad_trajectories(Drive* env){

    if (env->control_mode != CONTROL_TRACKS_TO_PREDICT) {
        return; // Leave all trajectories in WOSAC control mode
    }

    set_start_position(env);
    int collided_agents[env->active_agent_count];
    int collided_with_indices[env->active_agent_count];
    memset(collided_agents, 0, env->active_agent_count * sizeof(int));
    for (int i = 0; i < env->active_agent_count; ++i) {
        collided_with_indices[i] = -1;
    }
    // move experts through trajectories to check for collisions and remove as illegal agents
    for(int t = 0; t < env->scenario_length; t++){
        for(int i = 0; i < env->active_agent_count; i++){
            int agent_idx = env->active_agent_indices[i];
            move_expert(env, env->actions, agent_idx);
        }
        for(int i = 0; i < env->expert_static_agent_count; i++){
            int expert_idx = env->expert_static_agent_indices[i];
            if(env->entities[expert_idx].x == INVALID_POSITION) continue;
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
        for(int j = 0; j < env->static_agent_count; j++){
            int static_agent_idx = env->static_agent_indices[j];
            if(static_agent_idx != collided_with_indices[i]) continue;
            env->entities[static_agent_idx].traj_x[0] = INVALID_POSITION;
            env->entities[static_agent_idx].traj_y[0] = INVALID_POSITION;
        }
    }
    env->timestep = 0;
}

void init_goal_positions(Drive* env){
    for(int x = 0;x<env->active_agent_count; x++){
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].init_goal_x = env->entities[agent_idx].goal_position_x;
        env->entities[agent_idx].init_goal_y = env->entities[agent_idx].goal_position_y;
    }
}

Adaptive_Agent_Log* create_adaptive_agent_log(int num_scenarios) {
    Adaptive_Agent_Log* log = (Adaptive_Agent_Log*)calloc(1, sizeof(Adaptive_Agent_Log));

    log->num_scenarios = num_scenarios;

    // Allocate array of Log structs (calloc initializes all fields to 0)
    log->scenario_logs = (Log*)calloc(num_scenarios, sizeof(Log));

    return log;
}

void free_adaptive_agent_log(Adaptive_Agent_Log* log) {
    if (!log) return;

    free(log->scenario_logs);
    free(log);
}

void init(Drive* env){
    env->human_agent_idx = 0;
    env->timestep = 0;
    env->entities = load_map_binary(env->map_name, env);
    set_means(env);
    init_grid_map(env);
    if (env->goal_behavior==GOAL_GENERATE_NEW) init_topology_graph(env);
    env->grid_map->vision_range = 21;
    init_neighbor_offsets(env);
    cache_neighbor_offsets(env);
    env->logs_capacity = 0;
    set_active_agents(env);
    env->logs_capacity = env->active_agent_count;
    remove_bad_trajectories(env);
    set_start_position(env);
    init_goal_positions(env);
    env->logs = (Log*)calloc(env->active_agent_count, sizeof(Log));

    if (env->use_rc) {
        env->collision_weights = (float*)calloc(env->active_agent_count, sizeof(float));
        env->offroad_weights = (float*)calloc(env->active_agent_count, sizeof(float));
        env->goal_weights = (float*)calloc(env->active_agent_count, sizeof(float));
    }
    if (env->use_ec) {
        env->entropy_weights = (float*)calloc(env->active_agent_count, sizeof(float));
    }
    if (env->use_dc) {
        env->discount_weights = (float*)calloc(env->active_agent_count, sizeof(float));
    }

    if (env->adaptive_driving_agent) {
        env->ada_logs = (Adaptive_Agent_Log**)calloc(env->active_agent_count, sizeof(Adaptive_Agent_Log*));
        for (int i = 0; i < env->active_agent_count; i++) {
            env->ada_logs[i] = create_adaptive_agent_log(env->k_scenarios);
        }
    }
}

void c_close(Drive* env){
    for(int i = 0; i < env->num_entities; i++){
        free_entity(&env->entities[i]);
    }
    free(env->entities);
    free(env->active_agent_indices);
    free(env->logs);
    // Adaptive agent logs cleanup
    if (env->adaptive_driving_agent && env->ada_logs != NULL) {
        for (int i = 0; i < env->active_agent_count; i++) {
            free_adaptive_agent_log(env->ada_logs[i]);
        }
        free(env->ada_logs);
    }
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
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    freeTopologyGraph(env->topology_graph);
    // free(env->map_name);
    free(env->ini_file);

}

void allocate(Drive* env){
    init(env);
    int base_ego_dim = (env->dynamics_model == JERK) ? 10 : 7;
    int conditioning_dims = (env->use_rc ? 3 : 0) + (env->use_ec ? 1 : 0) + (env->use_dc ? 1 : 0);
    int ego_dim = base_ego_dim + conditioning_dims;

    if (env->use_rc) {
        env->collision_weights = (float*)calloc(env->active_agent_count, sizeof(float));
        env->offroad_weights = (float*)calloc(env->active_agent_count, sizeof(float));
        env->goal_weights = (float*)calloc(env->active_agent_count, sizeof(float));
    }
    if (env->use_ec) {
        env->entropy_weights = (float*)calloc(env->active_agent_count, sizeof(float));
    }
    if (env->use_dc) {
        env->discount_weights = (float*)calloc(env->active_agent_count, sizeof(float));
    }

    int max_obs = ego_dim + 7*(MAX_AGENTS - 1) + 7*MAX_ROAD_SEGMENT_OBSERVATIONS;
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

    if (env->use_rc) {
        free(env->collision_weights);
        free(env->offroad_weights);
        free(env->goal_weights);
    }
    if (env->use_ec) {
        free(env->entropy_weights);
    }
    if (env->use_dc) {
        free(env->discount_weights);
    }

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

float normalize_value(float value, float min, float max){
    return (value - min) / (max - min);
}

void move_dynamics(Drive* env, int action_idx, int agent_idx){
    Entity* agent = &env->entities[agent_idx];
    if (agent->removed) return;
    if (agent->stopped) {
        agent->vx = 0.0f;
        agent->vy = 0.0f;
        return;
    }

    if (agent->collision_state != 0) {
        return;
    }

    if (env->dynamics_model == CLASSIC) {
        // Classic dynamics model
        float acceleration = 0.0f;
        float steering = 0.0f;

        if (env->action_type == 1) { // continuous
            float (*action_array_f)[2] = (float(*)[2])env->actions;
            acceleration = action_array_f[action_idx][0];
            steering = action_array_f[action_idx][1];

            acceleration *= ACCELERATION_VALUES[6];
            steering *= STEERING_VALUES[12];
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

        // Update speed with acceleration
        speed = speed + 0.5f*acceleration*env->dt;
        speed = clipSpeed(speed);

        // Compute yaw rate
        float beta = tanh(.5*tanf(steering));

        // New heading
        float yaw_rate = (speed*cosf(beta)*tanf(steering)) / agent->length;

        // New velocity
        float new_vx = speed*cosf(heading + beta);
        float new_vy = speed*sinf(heading + beta);

        // Update position
        x = x + (new_vx*env->dt);
        y = y + (new_vy*env->dt);
        heading = heading + yaw_rate*env->dt;

        // Apply updates to the agent's state
        agent->x = x;
        agent->y = y;
        agent->heading = heading;
        agent->heading_x = cosf(heading);
        agent->heading_y = sinf(heading);
        agent->vx = new_vx;
        agent->vy = new_vy;
    } else {
        // JERK dynamics model
        // Extract action components
        float a_long, a_lat;
        if (env->action_type == 1) { // continuous
            float (*action_array_f)[2] = (float(*)[2])env->actions;

            // Asymmetric scaling for longitudinal jerk to match discrete action space
            // Discrete: JERK_LONG = [-15, -4, 0, 4] (more braking than acceleration)
            float a_long_action = action_array_f[action_idx][0];  // [-1, 1]
            if (a_long_action < 0) {
                a_long = a_long_action * (-JERK_LONG[0]);  // Negative: [-1, 0] → [-15, 0] (braking)
            } else {
                a_long = a_long_action * JERK_LONG[3];   // Positive: [0, 1] → [0, 4] (acceleration)
            }

            // Symmetric scaling for lateral jerk
            a_lat = action_array_f[action_idx][1] * JERK_LAT[2];
        } else { // discrete
            int (*action_array)[2] = (int(*)[2])env->actions;
            int a_long_idx = action_array[action_idx][0];
            int a_lat_idx = action_array[action_idx][1];
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
        float signed_v = copysignf(sqrtf(agent->vx*agent->vx + agent->vy*agent->vy), v_dot_heading);
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

void c_get_global_agent_state(Drive* env, float* x_out, float* y_out, float* z_out, float* heading_out, int* id_out) {
    for(int i = 0; i < env->active_agent_count; i++){
        int agent_idx = env->active_agent_indices[i];
        Entity* agent = &env->entities[agent_idx];
        // For WOSAC, we need the original world coordinates, so we add the world means back
        x_out[i] = agent->x + env->world_mean_x;
        y_out[i] = agent->y + env->world_mean_y;
        z_out[i] = agent->z;
        heading_out[i] = agent->heading;
        id_out[i] = agent->id;
    }
}

void c_get_global_ground_truth_trajectories(Drive* env, float* x_out, float* y_out, float* z_out, float* heading_out, int* valid_out, int* id_out, int* scenario_id_out) {
    for(int i = 0; i < env->active_agent_count; i++){
        int agent_idx = env->active_agent_indices[i];
        Entity* agent = &env->entities[agent_idx];
        id_out[i] = agent->id;
        scenario_id_out[i] = agent->scenario_id;

        for(int t = env->init_steps; t < agent->array_size; t++){
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

void compute_observations(Drive* env) {
    int base_ego_dim = (env->dynamics_model == JERK) ? 10 : 7;
    int conditioning_dims = (env->use_rc ? 3 : 0) + (env->use_ec ? 1 : 0) + (env->use_dc ? 1 : 0);
    int ego_dim = base_ego_dim + conditioning_dims;
    int max_obs = ego_dim + 7*(MAX_AGENTS - 1) + 7*MAX_ROAD_SEGMENT_OBSERVATIONS;

    memset(env->observations, 0, max_obs*env->active_agent_count*sizeof(float));
    float (*observations)[max_obs] = (float(*)[max_obs])env->observations;

    for(int i = 0; i < env->active_agent_count; i++) {
        float* obs = &observations[i][0];
        Entity* ego_entity = &env->entities[env->active_agent_indices[i]];
        if(ego_entity->type > 3) break;

        float cos_heading = ego_entity->heading_x;
        float sin_heading = ego_entity->heading_y;
        float ego_speed = sqrtf(ego_entity->vx*ego_entity->vx + ego_entity->vy*ego_entity->vy);

        // Set goal distances
        float goal_x = ego_entity->goal_position_x - ego_entity->x;
        float goal_y = ego_entity->goal_position_y - ego_entity->y;

        // Rotate to ego vehicle's frame
        float rel_goal_x = goal_x*cos_heading + goal_y*sin_heading;
        float rel_goal_y = -goal_x*sin_heading + goal_y*cos_heading;

        obs[0] = rel_goal_x* 0.005f;
        obs[1] = rel_goal_y* 0.005f;
        obs[2] = ego_speed / MAX_SPEED;
        obs[3] = ego_entity->width / MAX_VEH_WIDTH;
        obs[4] = ego_entity->length / MAX_VEH_LEN;
        obs[5] = (ego_entity->collision_state > 0) ? 1.0f : 0.0f;
        obs[6] = (ego_entity->respawn_timestep != -1) ? 1 : 0;

        if (env->dynamics_model == JERK) {
            obs[7] = ego_entity->steering_angle / M_PI;
            // Asymmetric normalization for a_long to match action space
            obs[8] = (ego_entity->a_long < 0) ? ego_entity->a_long / (-JERK_LONG[0]) : ego_entity->a_long / JERK_LONG[3];
            obs[9] = ego_entity->a_lat / JERK_LAT[2];
        }

        int obs_idx = (env->dynamics_model == JERK) ? 10 : 7;
        // Add conditioning weights to observations
        if (env->use_rc) {
            obs[obs_idx++] = env->collision_weights[i];
            obs[obs_idx++] = env->offroad_weights[i];
            obs[obs_idx++] = env->goal_weights[i];
        }
        if (env->use_ec) {
            obs[obs_idx++] = env->entropy_weights[i];
        }
        if (env->use_dc) {
            obs[obs_idx++] = env->discount_weights[i];
        }

        // Relative Pos of other cars
        int cars_seen = 0;
        for(int j = 0; j < MAX_AGENTS; j++) {
            int index = -1;
            if(j < env->active_agent_count){
                index = env->active_agent_indices[j];
            } else if (j < env->num_actors){
                index = env->static_agent_indices[j - env->active_agent_count];
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
        // Add conditioning weights to observations
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
        int remaining_partner_obs = (MAX_AGENTS - 1 - cars_seen) * 7;
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

static int find_forward_projection_on_lane(Entity* lane, Entity* agent, int* out_segment_idx, float* out_fraction) {
    int best_idx = -1;
    float best_dist_sq = 1e30f;

    for (int i = 1; i < lane->array_size; i++) {
        float x0 = lane->traj_x[i - 1];
        float y0 = lane->traj_y[i - 1];
        float x1 = lane->traj_x[i];
        float y1 = lane->traj_y[i];
        float dx = x1 - x0;
        float dy = y1 - y0;
        float seg_len_sq = dx * dx + dy * dy;
        if (seg_len_sq < 1e-6f) continue;

        float to_agent_x = agent->x - x0;
        float to_agent_y = agent->y - y0;
        float t = (to_agent_x * dx + to_agent_y * dy) / seg_len_sq;
        if (t < 0.0f) t = 0.0f;
        else if (t > 1.0f) t = 1.0f;

        float proj_x = x0 + t * dx;
        float proj_y = y0 + t * dy;

        float rel_x = proj_x - agent->x;
        float rel_y = proj_y - agent->y;
        float forward = rel_x * agent->heading_x + rel_y * agent->heading_y;
        if (forward < 0.0f) continue;

        float dist_sq = rel_x * rel_x + rel_y * rel_y;
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_idx = i;
            *out_fraction = t;
        }
    }

    if (best_idx != -1) {
        *out_segment_idx = best_idx;
        return 1;
    }

    return 0;
}

void compute_new_goal(Drive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];
    int current_lane = agent->current_lane_idx;

    if (current_lane == -1) return; // No current lane

    // Target distance: 40m ahead along the lane topology from agent's current position
    float target_distance = 40.0f;
    int current_entity = current_lane;
    Entity* lane = &env->entities[current_entity];

    int initial_segment_idx = 1;
    float initial_fraction = 0.0f;
    if (!find_forward_projection_on_lane(lane, agent, &initial_segment_idx, &initial_fraction)) {
        int forward_idx = -1;
        for (int i = 0; i < lane->array_size; i++) {
            float to_point_x = lane->traj_x[i] - agent->x;
            float to_point_y = lane->traj_y[i] - agent->y;
            float dot = to_point_x * agent->heading_x + to_point_y * agent->heading_y;
            if (dot > 0.0f) {
                forward_idx = i;
                break;
            }
        }

        if (forward_idx == -1) {
            agent->goal_position_x = lane->traj_x[lane->array_size - 1];
            agent->goal_position_y = lane->traj_y[lane->array_size - 1];
            agent->sampled_new_goal = 0;
            return;
        }

        initial_segment_idx = forward_idx;
        if (initial_segment_idx == 0) initial_segment_idx = 1;
        initial_fraction = 0.0f;
    }

    float remaining_distance = target_distance;
    int first_lane = 1;

    // Traverse the topology graph starting from the vehicle's position forward
    while (current_entity != -1) {
        lane = &env->entities[current_entity];

        int start_idx = first_lane ? initial_segment_idx : 1;
        // Ensure start_idx is at least 1 to avoid accessing traj_x[i-1] with i=0
        if (start_idx < 1) start_idx = 1;
        first_lane = 0;

        for (int i = start_idx; i < lane->array_size; i++) {
            float prev_x = lane->traj_x[i - 1];
            float prev_y = lane->traj_y[i - 1];
            float next_x = lane->traj_x[i];
            float next_y = lane->traj_y[i];
            float seg_dx = next_x - prev_x;
            float seg_dy = next_y - prev_y;
            float segment_length = relative_distance_2d(prev_x, prev_y, next_x, next_y);

            if (remaining_distance <= segment_length) {
                agent->goal_position_x = next_x;
                agent->goal_position_y = next_y;
                agent->sampled_new_goal = 0;
                return;
            }

            remaining_distance -= segment_length;
        }

        int connected_lanes[5];
        int num_connected = getNextLanes(env->topology_graph, current_entity, connected_lanes, 5);

        if (num_connected == 0) {
            agent->goal_position_x = lane->traj_x[lane->array_size - 1];
            agent->goal_position_y = lane->traj_y[lane->array_size - 1];
            agent->sampled_new_goal = 0;
            return; // No further lanes to traverse
        }

        int random_idx = agent_idx % num_connected;
        current_entity = connected_lanes[random_idx];
    }
}

void c_reset(Drive* env){
    env->timestep = env->init_steps;
    set_start_position(env);

    // Initialize conditioning weights
    if (env->use_rc) {
        for(int i = 0; i < env->active_agent_count; i++) {
            env->collision_weights[i] = ((float)rand() / RAND_MAX) * (env->collision_weight_ub - env->collision_weight_lb) + env->collision_weight_lb;
            env->offroad_weights[i] = ((float)rand() / RAND_MAX) * (env->offroad_weight_ub - env->offroad_weight_lb) + env->offroad_weight_lb;
            env->goal_weights[i] = ((float)rand() / RAND_MAX) * (env->goal_weight_ub - env->goal_weight_lb) + env->goal_weight_lb;
        }
    }

    if (env->use_ec) {
        for(int i = 0; i < env->active_agent_count; i++) {
            env->entropy_weights[i] = ((float)rand() / RAND_MAX) * (env->entropy_weight_ub - env->entropy_weight_lb) + env->entropy_weight_lb;
        }
    }

    if (env->use_dc) {
        for(int i = 0; i < env->active_agent_count; i++) {
            env->discount_weights[i] = ((float)rand() / RAND_MAX) * (env->discount_weight_ub - env->discount_weight_lb) + env->discount_weight_lb;
        }
    }

    for(int x = 0;x<env->active_agent_count; x++){
        env->logs[x] = (Log){0};
        int agent_idx = env->active_agent_indices[x];
        env->entities[agent_idx].respawn_timestep = -1;
        env->entities[agent_idx].respawn_count = 0;
        env->entities[agent_idx].collided_before_goal = 0;
        env->entities[agent_idx].reached_goal_this_episode = 0;
        env->entities[agent_idx].metrics_array[COLLISION_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[OFFROAD_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX] = 0.0f;
        env->entities[agent_idx].metrics_array[AVG_DISPLACEMENT_ERROR_IDX] = 0.0f;
        env->entities[agent_idx].cumulative_displacement = 0.0f;
        env->entities[agent_idx].displacement_sample_count = 0;
	    env->entities[agent_idx].stopped = 0;
        env->entities[agent_idx].removed = 0;

        if (env->goal_behavior==GOAL_GENERATE_NEW) {
            env->entities[agent_idx].goal_position_x = env->entities[agent_idx].init_goal_x;
            env->entities[agent_idx].goal_position_y = env->entities[agent_idx].init_goal_y;
            env->entities[agent_idx].sampled_new_goal = 0;
        }

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
    env->entities[agent_idx].stopped = 0;
    env->entities[agent_idx].removed = 0;
    env->entities[agent_idx].a_long = 0.0f;
    env->entities[agent_idx].a_lat = 0.0f;
    env->entities[agent_idx].jerk_long = 0.0f;
    env->entities[agent_idx].jerk_lat = 0.0f;
    env->entities[agent_idx].steering_angle = 0.0f;
}

void c_step(Drive* env){
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(unsigned char));
    env->timestep++;
    if(env->timestep == env->scenario_length){
        add_log(env);
	    c_reset(env);
        if (env->adaptive_driving_agent){
            env->current_scenario += 1;
        }
        return;
    }

    // Move static experts
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        int expert_idx = env->expert_static_agent_indices[i];
        if(env->entities[expert_idx].x == INVALID_POSITION) continue;
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
            if(collision_state == VEHICLE_COLLISION){
                env->rewards[i] = env->reward_vehicle_collision;
                env->logs[i].episode_return += env->reward_vehicle_collision;
                env->logs[i].collision_rate = 1.0f;
                env->logs[i].avg_collisions_per_agent += 1.0f;
            }
            else if(collision_state == OFFROAD){
                env->rewards[i] = env->reward_offroad_collision;
                env->logs[i].offroad_rate = 1.0f;
                env->logs[i].episode_return += env->reward_offroad_collision;
                env->logs[i].avg_offroad_per_agent += 1.0f;
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
                env->rewards[i] += env->reward_goal;
                env->logs[i].episode_return += env->reward_goal;
                env->entities[agent_idx].sampled_new_goal = 1;
                env->logs[i].num_goals_reached += 1;
            }
            env->entities[agent_idx].reached_goal_this_episode = 1;
            env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX] = 1.0f;
	    }

        if(env->entities[agent_idx].sampled_new_goal){
            if (env->goal_behavior==GOAL_GENERATE_NEW) {
                compute_new_goal(env, agent_idx);
            }
            else if (env->goal_behavior==GOAL_STOP) {
                env->entities[agent_idx].stopped = 1;
                env->entities[agent_idx].vx=env->entities[agent_idx].vy = 0.0f;
            }
        }


        int lane_aligned = env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX];
        env->logs[i].lane_alignment_rate = lane_aligned;

        // Apply ADE reward
        float current_ade = env->entities[agent_idx].metrics_array[AVG_DISPLACEMENT_ERROR_IDX];
        if(current_ade > 0.0f && env->reward_ade != 0.0f) {
            float ade_reward = env->reward_ade * current_ade;
            env->rewards[i] += ade_reward;
            env->logs[i].episode_return += ade_reward;
        }
        env->logs[i].avg_displacement_error = current_ade;
    }

    if (env->goal_behavior==GOAL_RESPAWN) {
        for(int i = 0; i < env->active_agent_count; i++){
            int agent_idx = env->active_agent_indices[i];
            int reached_goal = env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX];
            if(reached_goal){
                respawn_agent(env, agent_idx);
                env->entities[agent_idx].respawn_count++;
            }
        }
    }
    else if (env->goal_behavior==GOAL_STOP) {
        for(int i = 0; i < env->active_agent_count; i++){
            int agent_idx = env->active_agent_indices[i];
            int reached_goal = env->entities[agent_idx].metrics_array[REACHED_GOAL_IDX];
            if(reached_goal){
                env->entities[agent_idx].stopped = 1;
                env->entities[agent_idx].vx=env->entities[agent_idx].vy = 0.0f;
            }
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
    int car_assignments[MAX_AGENTS];  // To keep car model assignments consistent per vehicle
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
    for (int i = 0; i < MAX_AGENTS; i++) {
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

    int max_obs = 10 + 7*(MAX_AGENTS - 1) + 7*MAX_ROAD_SEGMENT_OBSERVATIONS;
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
    int obs_idx = 10;  // Start after ego obs
    for(int j = 0; j < MAX_AGENTS - 1; j++) {
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
    int map_start_idx = 7 + 7*(MAX_AGENTS - 1);  // Start after agent observations
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
        // Draw objects
        if(env->entities[i].type == VEHICLE || env->entities[i].type == PEDESTRIAN || env->entities[i].type == CYCLIST) {
            // Check if this vehicle is an active agent
            bool is_active_agent = false;
            bool is_static_agent = false;
            int agent_index = -1;
            for(int j = 0; j < env->active_agent_count; j++) {
                if(env->active_agent_indices[j] == i) {
                    is_active_agent = true;
                    agent_index = j;
                    break;
                }
            }
            for(int j = 0; j < env->static_agent_count; j++) {
                if(env->static_agent_indices[j] == i) {
                    is_static_agent = true;
                    break;
                }
            }
            // HIDE CARS ON RESPAWN - IMPORTANT TO KNOW VISUAL SETTING
            if((!is_active_agent && !is_static_agent) || env->entities[i].respawn_timestep != -1){
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

            bool is_expert = (!is_active_agent) && (env->entities[i].mark_as_expert == 1);

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
                Color car_color = GRAY;              // default for static
                if (is_expert) car_color = GOLD;      // expert replay
                if (is_active_agent) car_color = BLUE; // policy-controlled
                if (is_active_agent && env->entities[i].collision_state > 0) car_color = RED;
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
                // Determine color based on status
                Color object_color = PUFF_BACKGROUND2;  // fill color unused for model tint
                Color outline_color = PUFF_CYAN;        // not used for model tint
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
                {
                    float cos_heading = env->entities[i].heading_x;
                    float sin_heading = env->entities[i].heading_y;
                    float half_len = env->entities[i].length * 0.5f;
                    float half_width = env->entities[i].width * 0.5f;
                    Vector3 corners[4] = {
                        (Vector3){ 0 + ( half_len * cos_heading - half_width * sin_heading), 0 + ( half_len * sin_heading + half_width * cos_heading), 0 },
                        (Vector3){ 0 + ( half_len * cos_heading + half_width * sin_heading), 0 + ( half_len * sin_heading - half_width * cos_heading), 0 },
                        (Vector3){ 0 + (-half_len * cos_heading + half_width * sin_heading), 0 + (-half_len * sin_heading - half_width * cos_heading), 0 },
                        (Vector3){ 0 + (-half_len * cos_heading - half_width * sin_heading), 0 + (-half_len * sin_heading + half_width * cos_heading), 0 },
                    };
                    Color wire_color = GRAY;                 // static
                    if (!is_active_agent && env->entities[i].mark_as_expert == 1) wire_color = GOLD;  // expert replay
                    if (is_active_agent) wire_color = BLUE;   // policy
                    if (is_active_agent && env->entities[i].collision_state > 0) wire_color = RED;
                    rlSetLineWidth(2.0f);
                    for (int j = 0; j < 4; j++) {
                        DrawLine3D(corners[j], corners[(j+1)%4], wire_color);
                    }
                }
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

    // Draw track indices for the tracks to predict
    if (mode == 1 && env->control_mode == CONTROL_TRACKS_TO_PREDICT) {
        float map_width = env->grid_map->bottom_right_x - env->grid_map->top_left_x;
        float map_height = env->grid_map->top_left_y - env->grid_map->bottom_right_y;
        float pixels_per_world_unit = client->height / map_height;

        for (int i = 0; i < env->active_agent_count; i++) {
            // Ignore respawned agents
            if (env->entities[i].respawn_timestep != -1) {
                continue;
            }
            int agent_idx = env->active_agent_indices[i];
            int womd_track_idx = env->tracks_to_predict_indices[i];

            float raw_x = -env->entities[agent_idx].x * pixels_per_world_unit;
            float raw_y = env->entities[agent_idx].y * pixels_per_world_unit;

            int screen_x = (int)raw_x + client->width/2 + 20;
            int screen_y = (int)raw_y + client->height/2 - 25;

            if (screen_x >= 0 && screen_x <= client->width &&
                screen_y >= 0 && screen_y <= client->height) {
                char text[32];
                snprintf(text, sizeof(text), "%d", womd_track_idx);
                int text_width = MeasureText(text, 20);
                DrawText(text, screen_x - text_width/2, screen_y, 20, PUFF_WHITE);
            }
        }
    }
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
                    for(int j=0; j<env->entities[idx].array_size;j++){
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
