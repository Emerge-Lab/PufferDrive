#ifndef SIM_DRIVE_H
#define SIM_DRIVE_H

#include "datatypes.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INVALID_POSITION -10000.0f

// Simulation constants
#define TRAJECTORY_LENGTH 91 // Discretized Waymo scenarios
#define SIM_DT 0.1f

// Agent limits
#ifndef MAX_AGENTS
#define MAX_AGENTS 64
#endif

// Dynamics models
#define CLASSIC 0

// Collision State
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2

// Grid Map
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL 10
#define SLOTS_PER_CELL (MAX_ENTITIES_PER_CELL * 2 + 1)
#define VISION_RANGE 21

// Observation Space
#define MAX_ROAD_SEGMENT_OBSERVATIONS 75

#define PARTNER_FEATURES 7
#define EGO_FEATURES 7
#define ROAD_FEATURES 7

#define OBS_SIZE (EGO_FEATURES + PARTNER_FEATURES * (MAX_AGENTS - 1) + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS)

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
#define MIN_DISTANCE_TO_GOAL 5.0f
#define COLLISION_DIST_SQ 225.0f
#define OBS_DIST_SQ 2500.0f

// Action space
#define NUM_ACCEL_BINS 7
#define NUM_STEER_BINS 13

static const float ACCELERATION_VALUES[NUM_ACCEL_BINS]
    = {-4.0000f, -2.6670f, -1.3330f, -0.0000f, 1.3330f, 2.6670f, 4.0000f};

static const float STEERING_VALUES[NUM_STEER_BINS]
    = {-1.000f, -0.833f, -0.667f, -0.500f, -0.333f, -0.167f, 0.000f, 0.167f, 0.333f, 0.500f, 0.667f, 0.833f, 1.000f};

static const int collision_offsets[25][2]
    = {{-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {-2, -1}, {-1, -1}, {0, -1}, {1, -1},
       {2, -1},  {-2, 0},  {-1, 0}, {0, 0},  {1, 0},  {2, 0},   {-2, 1},  {-1, 1}, {0, 1},
       {1, 1},   {2, 1},   {-2, 2}, {-1, 2}, {0, 2},  {1, 2},   {2, 2}};

// Forward declarations
typedef struct Drive Drive;
typedef struct Client Client;
typedef struct Log Log;
typedef struct Agent Agent;
typedef struct RoadMapElement RoadMapElement;
typedef struct TrafficControlElement TrafficControlElement;

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
};

struct Drive {
    Client *client;
    float *observations;
    float *actions;
    float *rewards;
    float *terminals;
    Log log;
    Log *logs;
    Agent *agents;
    RoadMapElement *road_elements;
    TrafficControlElement *traffic_elements;
    // Entity fields
    int num_agents;
    int max_agents;
    int num_total_agents;
    int num_actors;
    int num_road_elements;
    int num_traffic_elements;
    int num_objects;
    int active_agent_count;
    int *active_agent_indices;
    int static_agent_count;
    int *static_agent_indices;
    int expert_static_agent_count;
    int *expert_static_agent_indices;
    int timestep;
    int dynamics_model;
    // Grid map fields
    float *map_corners;
    int *grid_cells;
    int grid_cols;
    int grid_rows;
    int vision_range;
    int *neighbor_offsets;
    int *neighbor_cache_entities;
    int *neighbor_cache_indices;
    // Reward coefficients
    float reward_vehicle_collision;
    float reward_offroad_collision;
    float reward_goal_post_respawn;
    float reward_vehicle_collision_post_respawn;
    unsigned int rng;
    // Metadata fields
    char scenario_id[128];
    char dataset_name[32];
    char *map_name;
    int log_length;

    float log_dt;
    int human_agent_idx;
};

// ========================================
// Utility Functions
// ========================================

float compute_euclidean_distance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

float clip(float value, float min, float max) {
    return value < min ? min : (value > max ? max : value);
}

// Normalize heading to [-pi, pi]
float normalize_heading(float heading) {
    heading = fmodf(heading, 2.0f * M_PI);
    if (heading > M_PI) {
        heading -= 2.0f * M_PI;
    } else if (heading < -M_PI) {
        heading += 2.0f * M_PI;
    }
    return heading;
}

// ========================================
// Grid Map Functions
// ========================================

int get_grid_index(Drive *env, float x1, float y1) {
    if (env->map_corners[0] >= env->map_corners[2] || env->map_corners[1] >= env->map_corners[3]) {
        return -1;
    }
    float relativeX = x1 - env->map_corners[0];
    float relativeY = y1 - env->map_corners[1];
    int gridX = (int) (relativeX / GRID_CELL_SIZE);
    int gridY = (int) (relativeY / GRID_CELL_SIZE);
    if (gridX < 0 || gridX >= env->grid_cols || gridY < 0 || gridY >= env->grid_rows) {
        return -1;
    }
    return (gridY * env->grid_cols) + gridX;
}

void add_entity_to_grid(Drive *env, int grid_index, int entity_idx, int geometry_idx) {
    if (grid_index == -1) {
        return;
    }

    int base_index = grid_index * SLOTS_PER_CELL;
    int count = env->grid_cells[base_index];
    if (count >= MAX_ENTITIES_PER_CELL) {
        return;
    }

    env->grid_cells[base_index + count * 2 + 1] = entity_idx;
    env->grid_cells[base_index + count * 2 + 2] = geometry_idx;
    env->grid_cells[base_index] = count + 1;
}

void init_grid_map(Drive *env) {
    float top_left_x, top_left_y, bottom_right_x, bottom_right_y;
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
            if (element->y[j] < top_left_y) {
                top_left_y = element->y[j];
            }
            if (element->y[j] > bottom_right_y) {
                bottom_right_y = element->y[j];
            }
        }
    }

    env->map_corners = (float *) calloc(4, sizeof(float));
    env->map_corners[0] = top_left_x;
    env->map_corners[1] = top_left_y;
    env->map_corners[2] = bottom_right_x;
    env->map_corners[3] = bottom_right_y;

    float grid_width = bottom_right_x - top_left_x;
    float grid_height = bottom_right_y - top_left_y;
    env->grid_cols = ceil(grid_width / GRID_CELL_SIZE);
    env->grid_rows = ceil(grid_height / GRID_CELL_SIZE);
    int grid_cell_count = env->grid_cols * env->grid_rows;
    env->grid_cells = (int *) calloc(grid_cell_count * SLOTS_PER_CELL, sizeof(int));

    for (int i = 0; i < env->num_road_elements; i++) {
        if (!is_road(env->road_elements[i].type)) {
            continue;
        }
        RoadMapElement *element = &env->road_elements[i];
        for (int j = 0; j < element->segment_length - 1; j++) {
            float x_center = (element->x[j] + element->x[j + 1]) / 2;
            float y_center = (element->y[j] + element->y[j + 1]) / 2;
            int grid_index = get_grid_index(env, x_center, y_center);
            add_entity_to_grid(env, grid_index, i, j);
        }
    }
}

void init_neighbor_offsets(Drive *env) {
    int vr = env->vision_range;
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

void cache_neighbor_offsets(Drive *env) {
    int vr = env->vision_range;
    int count = 0;
    int cell_count = env->grid_cols * env->grid_rows;

    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_cols;
        int cell_y = i / env->grid_cols;
        env->neighbor_cache_indices[i] = count;
        for (int j = 0; j < vr * vr; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            if (x < 0 || x >= env->grid_cols || y < 0 || y >= env->grid_rows) {
                continue;
            }
            int grid_index = env->grid_cols * y + x;
            count += env->grid_cells[grid_index * SLOTS_PER_CELL] * 2;
        }
    }
    env->neighbor_cache_indices[cell_count] = count;
    env->neighbor_cache_entities = (int *) calloc(count, sizeof(int));

    for (int i = 0; i < cell_count; i++) {
        int neighbor_cache_base_index = 0;
        int cell_x = i % env->grid_cols;
        int cell_y = i / env->grid_cols;
        for (int j = 0; j < vr * vr; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            if (x < 0 || x >= env->grid_cols || y < 0 || y >= env->grid_rows) {
                continue;
            }
            int grid_index = env->grid_cols * y + x;
            int grid_count = env->grid_cells[grid_index * SLOTS_PER_CELL];
            int base_index = env->neighbor_cache_indices[i];
            int src_idx = grid_index * SLOTS_PER_CELL + 1;
            int dst_idx = base_index + neighbor_cache_base_index;
            memcpy(&env->neighbor_cache_entities[dst_idx], &env->grid_cells[src_idx], grid_count * 2 * sizeof(int));
            neighbor_cache_base_index += grid_count * 2;
        }
    }
}

int get_neighbor_cache_entities(Drive *env, int cell_idx, int *entities, int max_entities) {
    if (cell_idx < 0 || cell_idx >= (env->grid_cols * env->grid_rows)) {
        return 0;
    }
    int base_index = env->neighbor_cache_indices[cell_idx];
    int end_index = env->neighbor_cache_indices[cell_idx + 1];
    int count = end_index - base_index;
    int pairs = count / 2;
    if (pairs > max_entities) {
        pairs = max_entities;
        count = pairs * 2;
    }
    memcpy(entities, env->neighbor_cache_entities + base_index, count * sizeof(int));
    return pairs;
}

int get_neighbors_entities(
    Drive *env,
    float x,
    float y,
    int *entity_list,
    int max_size,
    const int (*local_offsets)[2],
    int offset_size) {
    int index = get_grid_index(env, x, y);
    if (index == -1) {
        return 0;
    }
    int cellsX = env->grid_cols;
    int gridX = index % cellsX;
    int gridY = index / cellsX;
    int entity_list_count = 0;

    for (int i = 0; i < offset_size; i++) {
        int nx = gridX + local_offsets[i][0];
        int ny = gridY + local_offsets[i][1];
        if (nx < 0 || nx >= env->grid_cols || ny < 0 || ny >= env->grid_rows) {
            continue;
        }
        int neighborIndex = (ny * env->grid_cols + nx) * SLOTS_PER_CELL;
        int count = env->grid_cells[neighborIndex];
        for (int j = 0; j < count && entity_list_count < max_size; j++) {
            entity_list[entity_list_count] = env->grid_cells[neighborIndex + 1 + j * 2];
            entity_list[entity_list_count + 1] = env->grid_cells[neighborIndex + 2 + j * 2];
            entity_list_count += 2;
        }
    }
    return entity_list_count;
}

// ========================================
// Map Loading Functions
// ========================================

#define READ_OR_FAIL(ptr, elem_size, count)                                                                            \
    do {                                                                                                               \
        size_t _n = (size_t) (count);                                                                                  \
        if (fread((ptr), (elem_size), _n, file) != _n) {                                                               \
            fclose(file);                                                                                              \
            return -1;                                                                                                 \
        }                                                                                                              \
    } while (0)

int load_map_binary(const char *filename, Drive *drive) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        return -1;
    }

    int num_total_agents, num_roads, num_traffic, num_objects;
    READ_OR_FAIL(&num_total_agents, sizeof(int), 1);
    READ_OR_FAIL(&num_roads, sizeof(int), 1);
    READ_OR_FAIL(&num_traffic, sizeof(int), 1);
    READ_OR_FAIL(&num_objects, sizeof(int), 1);

    drive->num_total_agents = num_total_agents;
    drive->num_road_elements = num_roads;
    drive->num_traffic_elements = num_traffic;
    drive->num_objects = num_objects;

    if (num_total_agents > 0) {
        drive->agents = (Agent *) calloc(num_total_agents, sizeof(Agent));
    }
    if (num_roads > 0) {
        drive->road_elements = (RoadMapElement *) calloc(num_roads, sizeof(RoadMapElement));
    }
    if (num_traffic > 0) {
        drive->traffic_elements = (TrafficControlElement *) calloc(num_traffic, sizeof(TrafficControlElement));
    }

    for (int i = 0; i < num_total_agents; i++) {
        Agent *agent = &drive->agents[i];
        int agent_id;

        READ_OR_FAIL(&agent_id, sizeof(int), 1);
        if (agent_id != i) {
            printf("[ERROR] -> Agent id %d != idx %d. Binary must be reindexed (id == idx).\n", agent_id, i);
            fclose(file);
            return -1;
        }
        READ_OR_FAIL(&agent->type, sizeof(int), 1);
        READ_OR_FAIL(&agent->trajectory_length, sizeof(int), 1);

        int tlen = agent->trajectory_length;
        agent->log_trajectory_x = (float *) malloc(tlen * sizeof(float));
        agent->log_trajectory_y = (float *) malloc(tlen * sizeof(float));
        agent->log_trajectory_z = (float *) malloc(tlen * sizeof(float));
        agent->log_heading = (float *) malloc(tlen * sizeof(float));
        agent->log_velocity_x = (float *) malloc(tlen * sizeof(float));
        agent->log_velocity_y = (float *) malloc(tlen * sizeof(float));
        agent->log_length = (float *) malloc(tlen * sizeof(float));
        agent->log_width = (float *) malloc(tlen * sizeof(float));
        agent->log_height = (float *) malloc(tlen * sizeof(float));
        agent->log_valid = (int *) malloc(tlen * sizeof(int));

        READ_OR_FAIL(agent->log_trajectory_x, sizeof(float), tlen);
        READ_OR_FAIL(agent->log_trajectory_y, sizeof(float), tlen);
        READ_OR_FAIL(agent->log_trajectory_z, sizeof(float), tlen);
        READ_OR_FAIL(agent->log_heading, sizeof(float), tlen);
        READ_OR_FAIL(agent->log_velocity_x, sizeof(float), tlen);
        READ_OR_FAIL(agent->log_velocity_y, sizeof(float), tlen);
        READ_OR_FAIL(agent->log_length, sizeof(float), tlen);
        READ_OR_FAIL(agent->log_width, sizeof(float), tlen);
        READ_OR_FAIL(agent->log_height, sizeof(float), tlen);
        READ_OR_FAIL(agent->log_valid, sizeof(int), tlen);

        int route_length;
        READ_OR_FAIL(&route_length, sizeof(int), 1);

        if (route_length > 0) {
            fseek(file, (long) ((size_t) route_length * sizeof(int)), SEEK_CUR);
        }

        int route_gt_len;
        READ_OR_FAIL(&route_gt_len, sizeof(int), 1);

        READ_OR_FAIL(&agent->goal_position_x, sizeof(float), 1);
        READ_OR_FAIL(&agent->goal_position_y, sizeof(float), 1);
        READ_OR_FAIL(&agent->goal_position_z, sizeof(float), 1);
        READ_OR_FAIL(&agent->control_state, sizeof(int), 1);
    }

    for (int i = 0; i < num_roads; i++) {
        RoadMapElement *road = &drive->road_elements[i];
        int road_id;

        READ_OR_FAIL(&road_id, sizeof(int), 1);
        if (road_id != i) {
            printf("[ERROR] -> Road element id %d != idx %d. Binary must be reindexed (id == idx).\n", road_id, i);
            fclose(file);
            return -1;
        }
        READ_OR_FAIL(&road->type, sizeof(int), 1);
        READ_OR_FAIL(&road->segment_length, sizeof(int), 1);

        int slen = road->segment_length;

        road->x = (float *) malloc(slen * sizeof(float));
        road->y = (float *) malloc(slen * sizeof(float));
        road->z = (float *) malloc(slen * sizeof(float));

        READ_OR_FAIL(road->x, sizeof(float), slen);
        READ_OR_FAIL(road->y, sizeof(float), slen);
        READ_OR_FAIL(road->z, sizeof(float), slen);

        road->headings = (float *) malloc(slen * sizeof(float));
        READ_OR_FAIL(road->headings, sizeof(float), slen);

        if (is_road_lane(road->type)) {
            READ_OR_FAIL(&road->num_entries, sizeof(int), 1);
            if (road->num_entries > 0) {
                road->entry_lanes = (int *) malloc(road->num_entries * sizeof(int));
                READ_OR_FAIL(road->entry_lanes, sizeof(int), road->num_entries);
            } else {
                road->entry_lanes = NULL;
            }

            READ_OR_FAIL(&road->num_exits, sizeof(int), 1);
            if (road->num_exits > 0) {
                road->exit_lanes = (int *) malloc(road->num_exits * sizeof(int));
                READ_OR_FAIL(road->exit_lanes, sizeof(int), road->num_exits);
            } else {
                road->exit_lanes = NULL;
            }

            READ_OR_FAIL(&road->speed_limit, sizeof(float), 1);
        } else {
            road->num_entries = 0;
            road->num_exits = 0;
            road->entry_lanes = NULL;
            road->exit_lanes = NULL;
            road->speed_limit = 0.0f;
        }
    }

    for (int i = 0; i < num_traffic; i++) {
        TrafficControlElement *traffic = &drive->traffic_elements[i];
        int traffic_id;

        READ_OR_FAIL(&traffic_id, sizeof(int), 1);
        if (traffic_id != i) {
            printf(
                "[ERROR] -> Traffic element id %d != idx %d. Binary must be reindexed (id == idx).\n",
                traffic_id,
                i);
            fclose(file);
            return -1;
        }
        READ_OR_FAIL(&traffic->type, sizeof(int), 1);
        READ_OR_FAIL(traffic->stop_line, sizeof(float), 6);
        READ_OR_FAIL(&traffic->heading, sizeof(float), 1);
        READ_OR_FAIL(&traffic->state_length, sizeof(int), 1);

        int state_len = traffic->state_length;

        traffic->states = (int *) malloc(state_len * sizeof(int));
        READ_OR_FAIL(traffic->states, sizeof(int), state_len);

        READ_OR_FAIL(&traffic->num_controlled_lanes, sizeof(int), 1);
        if (traffic->num_controlled_lanes > 0) {
            traffic->controlled_lanes = (int *) malloc(traffic->num_controlled_lanes * sizeof(int));
            READ_OR_FAIL(traffic->controlled_lanes, sizeof(int), traffic->num_controlled_lanes);
        } else {
            traffic->controlled_lanes = NULL;
        }
    }

    // Skip objects section
    for (int i = 0; i < num_objects; i++) {
        int obj_id, obj_type, T;
        READ_OR_FAIL(&obj_id, sizeof(int), 1);
        READ_OR_FAIL(&obj_type, sizeof(int), 1);
        READ_OR_FAIL(&T, sizeof(int), 1);
        // Skip: x,y,z,heading,vx,vy,length,width,height (9 float arrays) + valid (1 int array)
        fseek(file, 9 * T * sizeof(float) + T * sizeof(int), SEEK_CUR);
    }

    // Lane graph section
    int n_lanes_graph;
    READ_OR_FAIL(&n_lanes_graph, sizeof(int), 1);
    if (n_lanes_graph > 0) {
        size_t lane_count = (size_t) n_lanes_graph;
        size_t distance_count = lane_count * lane_count;
        if (fseek(file, (long) (lane_count * sizeof(int)), SEEK_CUR) != 0) {
            fclose(file);
            return -1;
        }
        if (fseek(file, (long) (lane_count * sizeof(float)), SEEK_CUR) != 0) {
            fclose(file);
            return -1;
        }
        if (fseek(file, (long) (distance_count * sizeof(float)), SEEK_CUR) != 0) {
            fclose(file);
            return -1;
        }
    }

    // Metadata
    READ_OR_FAIL(drive->scenario_id, sizeof(char), 128);
    READ_OR_FAIL(drive->dataset_name, sizeof(char), 32);
    READ_OR_FAIL(&drive->log_length, sizeof(int), 1);
    READ_OR_FAIL(&drive->log_dt, sizeof(float), 1);

    int num_objects_of_interest;
    READ_OR_FAIL(&num_objects_of_interest, sizeof(int), 1);
    if (num_objects_of_interest > 0) {
        if (fseek(file, (long) ((size_t) num_objects_of_interest * sizeof(int)), SEEK_CUR) != 0) {
            fclose(file);
            return -1;
        }
    }

    int num_tracks_to_predict;
    READ_OR_FAIL(&num_tracks_to_predict, sizeof(int), 1);
    if (num_tracks_to_predict > 0) {
        if (fseek(file, (long) ((size_t) num_tracks_to_predict * sizeof(int)), SEEK_CUR) != 0) {
            fclose(file);
            return -1;
        }
    }

    fclose(file);
    return 0;
}

#undef READ_OR_FAIL

// ========================================
// Metrics/Collision Functions
// ========================================

bool check_line_intersection(float p1[2], float p2[2], float q1[2], float q2[2]) {
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

bool check_obb_collision(Agent *car1, Agent *car2) {
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

int collision_check(Drive *env, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    if (agent->sim_x == INVALID_POSITION) {
        return -1;
    }

    float corners[4][2];
    compute_agent_corners(agent, corners);

    int collided = NO_COLLISION;
    int car_collided_with_index = -1;

    // Check road edge collisions via grid
    int entity_list[MAX_ENTITIES_PER_CELL * 2 * 25];
    int list_size = get_neighbors_entities(
        env,
        agent->sim_x,
        agent->sim_y,
        entity_list,
        MAX_ENTITIES_PER_CELL * 2 * 25,
        collision_offsets,
        25);
    for (int i = 0; i < list_size; i += 2) {
        if (entity_list[i] == -1 || entity_list[i] == agent_idx) {
            continue;
        }
        RoadMapElement *entity = &env->road_elements[entity_list[i]];
        if (!is_road_edge(entity->type)) {
            continue;
        }
        int geometry_idx = entity_list[i + 1];
        float start[2] = {entity->x[geometry_idx], entity->y[geometry_idx]};
        float end[2] = {entity->x[geometry_idx + 1], entity->y[geometry_idx + 1]};
        for (int k = 0; k < 4; k++) {
            int next = (k + 1) % 4;
            if (check_line_intersection(corners[k], corners[next], start, end)) {
                collided = OFFROAD;
                break;
            }
        }
        if (collided == OFFROAD) {
            break;
        }
    }

    // Check vehicle-vehicle collisions
    for (int i = 0; i < MAX_AGENTS; i++) {
        int index = -1;
        if (i < env->active_agent_count) {
            index = env->active_agent_indices[i];
        } else if (i < env->num_actors) {
            index = env->static_agent_indices[i - env->active_agent_count];
        }
        if (index == -1 || index == agent_idx) {
            continue;
        }
        Agent *other_agent = &env->agents[index];
        float dx = other_agent->sim_x - agent->sim_x;
        float dy = other_agent->sim_y - agent->sim_y;
        if ((dx * dx + dy * dy) > COLLISION_DIST_SQ) {
            continue;
        }
        if (check_obb_collision(agent, other_agent)) {
            collided = VEHICLE_COLLISION;
            car_collided_with_index = index;
            break;
        }
    }

    agent->collision_state = collided;

    // Spawn immunity: agent just respawned
    if (collided == VEHICLE_COLLISION && agent->active_agent == 1 && agent->respawn_timestep != -1) {
        agent->collision_state = NO_COLLISION;
    }

    if (collided == OFFROAD) {
        return -1;
    }
    if (car_collided_with_index == -1) {
        return -1;
    }

    // Spawn immunity: collided-with agent just respawned
    if (env->agents[car_collided_with_index].respawn_timestep != -1) {
        agent->collision_state = NO_COLLISION;
    }

    return car_collided_with_index;
}

void add_log(Drive *env) {
    for (int i = 0; i < env->active_agent_count; i++) {
        Agent *e = &env->agents[env->active_agent_indices[i]];
        if (e->reached_goal_this_episode) {
            env->log.completion_rate += 1.0f;
        }
        int offroad = env->logs[i].offroad_rate;
        env->log.offroad_rate += env->logs[i].offroad_rate;
        int collided = env->logs[i].collision_rate;
        env->log.collision_rate += collided;
        int clean_collided = env->logs[i].clean_collision_rate;
        env->log.clean_collision_rate += clean_collided;
        if (e->reached_goal_this_episode && !e->collided_before_goal) {
            env->log.score += 1.0f;
            env->log.perf += 1.0f;
        }
        if (!offroad && !collided && !e->reached_goal_this_episode) {
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

void set_start_position(Drive *env) {
    for (int i = 0; i < env->num_total_agents; i++) {
        if (env->agents[i].log_valid[0] != 1 || env->agents[i].type > CYCLIST || env->agents[i].type == UNKNOWN) {
            continue;
        }
        Agent *agent = &env->agents[i];
        agent->sim_x = agent->log_trajectory_x[0];
        agent->sim_y = agent->log_trajectory_y[0];
        agent->sim_z = agent->log_trajectory_z[0];
        agent->sim_length = agent->log_length[0];
        agent->sim_width = agent->log_width[0];
        agent->sim_height = agent->log_height[0];

        if (agent->control_state == CONTROL_STATE_STATIC) {
            agent->sim_vx = 0.0f;
            agent->sim_vy = 0.0f;
        } else {
            agent->sim_vx = agent->log_velocity_x[0];
            agent->sim_vy = agent->log_velocity_y[0];
        }

        agent->sim_heading = agent->log_heading[0];
        agent->cos_heading = cosf(agent->sim_heading);
        agent->sin_heading = sinf(agent->sim_heading);
        agent->sim_valid = agent->log_valid[0];
        agent->collision_state = NO_COLLISION;
        agent->respawn_timestep = -1;
        agent->reached_goal = 0;
        agent->collided_before_goal = 0;
    }
}

void set_active_agents(Drive *env) {
    env->active_agent_count = 0;
    env->static_agent_count = 0;
    env->num_actors = 0;
    env->expert_static_agent_count = 0;

    int active_agent_indices[MAX_AGENTS];
    int static_agent_indices[MAX_AGENTS];
    int expert_static_agent_indices[MAX_AGENTS];

    if (env->max_agents == 0) {
        env->max_agents = MAX_AGENTS;
    }

    for (int i = 0; i < env->num_total_agents && env->num_actors < MAX_AGENTS; i++) {
        if (env->agents[i].log_valid[0] != 1 || env->agents[i].type > CYCLIST || env->agents[i].type == UNKNOWN) {
            continue;
        }
        env->num_actors++;

        if (env->agents[i].control_state == CONTROL_STATE_ACTIVE && env->active_agent_count < env->max_agents) {
            active_agent_indices[env->active_agent_count++] = i;
        } else {
            static_agent_indices[env->static_agent_count++] = i;
            if (env->agents[i].control_state == CONTROL_STATE_MOVING) {
                expert_static_agent_indices[env->expert_static_agent_count++] = i;
            }
        }
    }

    env->active_agent_indices = (int *) malloc(env->active_agent_count * sizeof(int));
    env->static_agent_indices = (int *) malloc(env->static_agent_count * sizeof(int));
    env->expert_static_agent_indices = (int *) malloc(env->expert_static_agent_count * sizeof(int));
    memcpy(env->active_agent_indices, active_agent_indices, env->active_agent_count * sizeof(int));
    memcpy(env->static_agent_indices, static_agent_indices, env->static_agent_count * sizeof(int));
    memcpy(env->expert_static_agent_indices, expert_static_agent_indices, env->expert_static_agent_count * sizeof(int));
}

void move_expert(Drive *env, float *actions, int agent_idx) {
    Agent *agent = &env->agents[agent_idx];
    int t = env->timestep;
    if (t < 0 || t >= agent->trajectory_length || agent->log_valid[t] != 1) {
        agent->sim_x = INVALID_POSITION;
        agent->sim_y = INVALID_POSITION;
        return;
    }
    agent->sim_x = agent->log_trajectory_x[t];
    agent->sim_y = agent->log_trajectory_y[t];
    agent->sim_z = agent->log_trajectory_z[t];
    agent->sim_heading = agent->log_heading[t];
    agent->cos_heading = cosf(agent->sim_heading);
    agent->sin_heading = sinf(agent->sim_heading);
    agent->sim_valid = agent->log_valid[t];
}

void remove_bad_trajectories(Drive *env) {
    set_start_position(env);
    int collided_agents[env->active_agent_count];
    int collided_with_indices[env->active_agent_count];
    memset(collided_agents, 0, env->active_agent_count * sizeof(int));

    for (int t = 0; t < TRAJECTORY_LENGTH; t++) {
        for (int i = 0; i < env->active_agent_count; i++) {
            move_expert(env, env->actions, env->active_agent_indices[i]);
        }
        for (int i = 0; i < env->expert_static_agent_count; i++) {
            int expert_idx = env->expert_static_agent_indices[i];
            if (env->agents[expert_idx].sim_x == INVALID_POSITION) {
                continue;
            }
            move_expert(env, env->actions, expert_idx);
        }
        for (int i = 0; i < env->active_agent_count; i++) {
            int agent_idx = env->active_agent_indices[i];
            env->agents[agent_idx].collision_state = NO_COLLISION;
            int collided_with = collision_check(env, agent_idx);
            if (env->agents[agent_idx].collision_state > NO_COLLISION && collided_agents[i] == 0) {
                collided_agents[i] = 1;
                collided_with_indices[i] = collided_with;
            }
        }
        env->timestep++;
    }

    for (int i = 0; i < env->active_agent_count; i++) {
        if (collided_with_indices[i] == -1) {
            continue;
        }
        for (int j = 0; j < env->static_agent_count; j++) {
            int static_idx = env->static_agent_indices[j];
            if (static_idx != collided_with_indices[i]) {
                continue;
            }
            env->agents[static_idx].log_trajectory_x[0] = INVALID_POSITION;
            env->agents[static_idx].log_trajectory_y[0] = INVALID_POSITION;
        }
    }
    env->timestep = 0;
}

int init(Drive *env) {
    env->human_agent_idx = 0;
    env->timestep = 0;
    if (load_map_binary(env->map_name, env) != 0) {
        return -1;
    }
    env->dynamics_model = CLASSIC;
    init_grid_map(env);
    env->vision_range = VISION_RANGE;
    init_neighbor_offsets(env);
    env->neighbor_cache_indices = (int *) calloc((env->grid_cols * env->grid_rows) + 1, sizeof(int));
    cache_neighbor_offsets(env);
    set_active_agents(env);
    remove_bad_trajectories(env);
    set_start_position(env);
    env->logs = (Log *) calloc(env->active_agent_count, sizeof(Log));
    return 0;
}

void c_close(Drive *env) {
    for (int i = 0; i < env->num_total_agents; i++) {
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
    free(env->active_agent_indices);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    free(env->logs);
    free(env->map_corners);
    free(env->grid_cells);
    free(env->neighbor_offsets);
    free(env->neighbor_cache_entities);
    free(env->neighbor_cache_indices);
}

int allocate(Drive *env) {
    if (init(env) != 0) {
        return -1;
    }
    env->observations = (float *) calloc(env->active_agent_count * OBS_SIZE, sizeof(float));
    env->actions = (float *) calloc(env->active_agent_count * 2, sizeof(float));
    env->rewards = (float *) calloc(env->active_agent_count, sizeof(float));
    env->terminals = (float *) calloc(env->active_agent_count, sizeof(float));
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
    agent->sim_x = agent->log_trajectory_x[0];
    agent->sim_y = agent->log_trajectory_y[0];
    agent->sim_z = agent->log_trajectory_z[0];
    agent->sim_heading = agent->log_heading[0];
    agent->cos_heading = cosf(agent->sim_heading);
    agent->sin_heading = sinf(agent->sim_heading);
    agent->sim_vx = agent->log_velocity_x[0];
    agent->sim_vy = agent->log_velocity_y[0];
    agent->reached_goal = 0;
    agent->respawn_timestep = env->timestep;
}

void compute_observations(Drive *env) {
    memset(env->observations, 0, OBS_SIZE * env->active_agent_count * sizeof(float));
    float (*observations)[OBS_SIZE] = (float (*)[OBS_SIZE]) env->observations;

    for (int i = 0; i < env->active_agent_count; i++) {
        float *obs = &observations[i][0];
        Agent *ego = &env->agents[env->active_agent_indices[i]];
        if (ego->type > CYCLIST) {
            break;
        }

        if (ego->respawn_timestep != -1) {
            obs[6] = 1;
        }

        float cos_h = ego->cos_heading;
        float sin_h = ego->sin_heading;
        float ego_speed = sqrtf(ego->sim_vx * ego->sim_vx + ego->sim_vy * ego->sim_vy);

        // Goal in ego frame
        float goal_x = ego->goal_position_x - ego->sim_x;
        float goal_y = ego->goal_position_y - ego->sim_y;
        float rel_goal_x = goal_x * cos_h + goal_y * sin_h;
        float rel_goal_y = -goal_x * sin_h + goal_y * cos_h;

        // Ego features
        obs[0] = rel_goal_x * OBS_GOAL_SCALE;
        obs[1] = rel_goal_y * OBS_GOAL_SCALE;
        obs[2] = ego_speed * OBS_SPEED_SCALE;
        obs[3] = ego->sim_width / MAX_VEH_WIDTH;
        obs[4] = ego->sim_length / MAX_VEH_LEN;
        obs[5] = (ego->collision_state > NO_COLLISION) ? 1 : 0;

        // Partner observations
        int obs_idx = EGO_FEATURES;
        int cars_seen = 0;
        for (int j = 0; j < MAX_AGENTS; j++) {
            int index = -1;
            if (j < env->active_agent_count) {
                index = env->active_agent_indices[j];
            } else if (j < env->num_actors) {
                index = env->static_agent_indices[j - env->active_agent_count];
            }
            if (index == -1) {
                continue;
            }
            if (index == env->active_agent_indices[i]) {
                continue;
            }

            Agent *other = &env->agents[index];
            if (ego->respawn_timestep != -1) {
                continue;
            }
            if (other->respawn_timestep != -1) {
                continue;
            }

            float dx = other->sim_x - ego->sim_x;
            float dy = other->sim_y - ego->sim_y;
            if ((dx * dx + dy * dy) > OBS_DIST_SQ) {
                continue;
            }

            float rel_x = dx * cos_h + dy * sin_h;
            float rel_y = -dx * sin_h + dy * cos_h;

            obs[obs_idx + 0] = rel_x * OBS_POSITION_SCALE;
            obs[obs_idx + 1] = rel_y * OBS_POSITION_SCALE;
            obs[obs_idx + 2] = other->sim_width / MAX_VEH_WIDTH;
            obs[obs_idx + 3] = other->sim_length / MAX_VEH_LEN;
            obs[obs_idx + 4] = other->cos_heading * ego->cos_heading + other->sin_heading * ego->sin_heading;
            obs[obs_idx + 5] = other->sin_heading * ego->cos_heading - other->cos_heading * ego->sin_heading;
            float other_speed = sqrtf(other->sim_vx * other->sim_vx + other->sim_vy * other->sim_vy);
            obs[obs_idx + 6] = other_speed / MAX_SPEED;
            cars_seen++;
            obs_idx += PARTNER_FEATURES;
        }
        int remaining_partner_obs = (MAX_AGENTS - 1 - cars_seen) * PARTNER_FEATURES;
        memset(&obs[obs_idx], 0, remaining_partner_obs * sizeof(float));
        obs_idx += remaining_partner_obs;

        // Road observations
        int entity_list[MAX_ROAD_SEGMENT_OBSERVATIONS * 2];
        int grid_idx = get_grid_index(env, ego->sim_x, ego->sim_y);
        int list_size = get_neighbor_cache_entities(env, grid_idx, entity_list, MAX_ROAD_SEGMENT_OBSERVATIONS);

        for (int k = 0; k < list_size; k++) {
            int entity_idx = entity_list[k * 2];
            int geometry_idx = entity_list[k * 2 + 1];
            RoadMapElement *entity = &env->road_elements[entity_idx];

            float start_x = entity->x[geometry_idx];
            float start_y = entity->y[geometry_idx];
            float end_x = entity->x[geometry_idx + 1];
            float end_y = entity->y[geometry_idx + 1];
            float mid_x = (start_x + end_x) / 2.0f;
            float mid_y = (start_y + end_y) / 2.0f;
            float rel_x = mid_x - ego->sim_x;
            float rel_y = mid_y - ego->sim_y;
            float x_obs = rel_x * cos_h + rel_y * sin_h;
            float y_obs = -rel_x * sin_h + rel_y * cos_h;
            float length = compute_euclidean_distance(mid_x, mid_y, end_x, end_y);

            float dx = end_x - mid_x;
            float dy = end_y - mid_y;
            float hypot = sqrtf(dx * dx + dy * dy);
            float dx_norm = dx, dy_norm = dy;
            if (hypot > 0) {
                dx_norm /= hypot;
                dy_norm /= hypot;
            }

            float cos_angle = dx_norm * cos_h + dy_norm * sin_h;
            float sin_angle = -dx_norm * sin_h + dy_norm * cos_h;

            obs[obs_idx + 0] = x_obs * OBS_POSITION_SCALE;
            obs[obs_idx + 1] = y_obs * OBS_POSITION_SCALE;
            obs[obs_idx + 2] = length / MAX_ROAD_SEGMENT_LENGTH;
            obs[obs_idx + 3] = 0.1f / MAX_ROAD_SCALE;
            obs[obs_idx + 4] = cos_angle;
            obs[obs_idx + 5] = sin_angle;
            obs[obs_idx + 6] = normalize_road_type(entity->type);
            obs_idx += ROAD_FEATURES;
        }
        int remaining_obs = (MAX_ROAD_SEGMENT_OBSERVATIONS - list_size) * ROAD_FEATURES;
        memset(&obs[obs_idx], 0, remaining_obs * sizeof(float));
    }
}

void move_dynamics(Drive *env, int action_idx, int agent_idx) {
    if (env->dynamics_model != CLASSIC) {
        return;
    }

    Agent *agent = &env->agents[agent_idx];
    float (*action_array)[2] = (float (*)[2]) env->actions;
    int acceleration_index = action_array[action_idx][0];
    int steering_index = action_array[action_idx][1];
    float acceleration = ACCELERATION_VALUES[acceleration_index];
    float steering = STEERING_VALUES[steering_index];

    float x = agent->sim_x;
    float y = agent->sim_y;
    float heading = agent->sim_heading;
    float speed = sqrtf(agent->sim_vx * agent->sim_vx + agent->sim_vy * agent->sim_vy);

    speed = speed + 0.5f * acceleration * SIM_DT;
    speed = clip(speed, -MAX_SPEED, MAX_SPEED);

    float beta = tanh(0.5 * tanf(steering));
    float yaw_rate = (speed * cosf(beta) * tanf(steering)) / agent->sim_length;
    float new_vx = speed * cosf(heading + beta);
    float new_vy = speed * sinf(heading + beta);

    x += new_vx * SIM_DT;
    y += new_vy * SIM_DT;
    heading += yaw_rate * SIM_DT;

    agent->sim_x = x;
    agent->sim_y = y;
    agent->sim_heading = heading;
    agent->cos_heading = cosf(heading);
    agent->sin_heading = sinf(heading);
    agent->sim_vx = new_vx;
    agent->sim_vy = new_vy;
}

void c_reset(Drive *env) {
    env->timestep = 0;
    set_start_position(env);
    for (int x = 0; x < env->active_agent_count; x++) {
        env->logs[x] = (Log) {0};
        int agent_idx = env->active_agent_indices[x];
        env->agents[agent_idx].respawn_timestep = -1;
        env->agents[agent_idx].reached_goal = 0;
        env->agents[agent_idx].collided_before_goal = 0;
        env->agents[agent_idx].reached_goal_this_episode = 0;

        collision_check(env, agent_idx);
    }
    compute_observations(env);
}

void c_step(Drive *env) {
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    memset(env->terminals, 0, env->active_agent_count * sizeof(float));
    env->timestep++;

    if (env->timestep == TRAJECTORY_LENGTH) {
        add_log(env);
        c_reset(env);
        return;
    }

    // Move expert static agents
    for (int i = 0; i < env->expert_static_agent_count; i++) {
        int expert_idx = env->expert_static_agent_indices[i];
        if (env->agents[expert_idx].sim_x == INVALID_POSITION) {
            continue;
        }
        move_expert(env, env->actions, expert_idx);
    }

    // Apply dynamics for active agents
    for (int i = 0; i < env->active_agent_count; i++) {
        env->logs[i].score = 0.0f;
        env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];
        env->agents[agent_idx].collision_state = NO_COLLISION;
        move_dynamics(env, i, agent_idx);
    }

    // Collision detection and rewards
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        env->agents[agent_idx].collision_state = NO_COLLISION;
        collision_check(env, agent_idx);
        int collision_state = env->agents[agent_idx].collision_state;

        if (collision_state > NO_COLLISION) {
            if (collision_state == VEHICLE_COLLISION && env->agents[agent_idx].respawn_timestep == -1) {
                env->rewards[i] = env->reward_vehicle_collision;
                env->logs[i].episode_return += env->reward_vehicle_collision;
                env->logs[i].clean_collision_rate = 1.0f;
                env->logs[i].collision_rate = 1.0f;
            } else if (collision_state == OFFROAD) {
                env->rewards[i] = env->reward_offroad_collision;
                env->logs[i].offroad_rate = 1.0f;
                env->logs[i].episode_return += env->reward_offroad_collision;
            }
            if (!env->agents[agent_idx].reached_goal_this_episode) {
                env->agents[agent_idx].collided_before_goal = 1;
            }
        }

        float distance_to_goal = compute_euclidean_distance(
            env->agents[agent_idx].sim_x,
            env->agents[agent_idx].sim_y,
            env->agents[agent_idx].goal_position_x,
            env->agents[agent_idx].goal_position_y);

        if (distance_to_goal < MIN_DISTANCE_TO_GOAL) {
            if (env->agents[agent_idx].respawn_timestep != -1) {
                env->rewards[i] += env->reward_goal_post_respawn;
                env->logs[i].episode_return += env->reward_goal_post_respawn;
            } else {
                env->rewards[i] += 1.0f;
                env->logs[i].episode_return += 1.0f;
            }
            env->agents[agent_idx].reached_goal = 1;
            env->agents[agent_idx].reached_goal_this_episode = 1;
        }
    }

    // Respawn agents that reached goal
    for (int i = 0; i < env->active_agent_count; i++) {
        int agent_idx = env->active_agent_indices[i];
        if (env->agents[agent_idx].reached_goal) {
            respawn_agent(env, agent_idx);
        }
    }

    compute_observations(env);
}

#include "render.h"

#endif // SIM_DRIVE_H
