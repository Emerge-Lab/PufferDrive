#ifndef SIM_DATALOADER_H
#define SIM_DATALOADER_H

#include "datatypes.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline int has_bin_suffix(const char *name) {
    int len = strlen(name);
    return len >= 4 && strcmp(name + len - 4, ".bin") == 0;
}

static inline int compare_map_paths(const void *lhs, const void *rhs) {
    const char *const *left = lhs;
    const char *const *right = rhs;
    return strcmp(*left, *right);
}

static inline void free_map_files(char **map_files, int num_map_files) {
    for (int i = 0; i < num_map_files; i++) {
        free(map_files[i]);
    }
    free(map_files);
}

static inline char **discover_map_files(const char *map_binary_dir, int *num_map_files_out) {
    DIR *dir = opendir(map_binary_dir);
    if (!dir) {
        *num_map_files_out = 0;
        return NULL;
    }

    int capacity = 16;
    int count = 0;
    char **map_files = (char **) malloc(capacity * sizeof(char *));
    struct dirent *entry = NULL;

    while ((entry = readdir(dir)) != NULL) {
        if (!has_bin_suffix(entry->d_name)) {
            continue;
        }

        if (count == capacity) {
            capacity *= 2;
            map_files = (char **) realloc(map_files, capacity * sizeof(char *));
        }

        int path_len = snprintf(NULL, 0, "%s/%s", map_binary_dir, entry->d_name);
        char *map_path = (char *) malloc((path_len + 1) * sizeof(char));
        snprintf(map_path, path_len + 1, "%s/%s", map_binary_dir, entry->d_name);
        map_files[count++] = map_path;
    }

    closedir(dir);

    if (count == 0) {
        free(map_files);
        *num_map_files_out = 0;
        return NULL;
    }

    qsort(map_files, count, sizeof(char *), compare_map_paths);
    *num_map_files_out = count;
    return map_files;
}

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

    drive->num_sim_agents = num_total_agents;
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
        READ_OR_FAIL(&agent->id, sizeof(int), 1);
        if (agent->id != i) {
            printf("[ERROR] -> Agent id %d != idx %d. Binary must be reindexed (id == idx).\n", agent->id, i);
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

        READ_OR_FAIL(&agent->route_length, sizeof(int), 1);

        if (agent->route_length > 0) {
            agent->route = (int *) malloc(agent->route_length * sizeof(int));
            READ_OR_FAIL(agent->route, sizeof(int), agent->route_length);
        } else {
            agent->route = NULL;
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

#endif // SIM_DATALOADER_H
