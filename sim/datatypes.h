#ifndef SIM_DATATYPES_H
#define SIM_DATATYPES_H

#include <stdlib.h>

// -- AGENT TYPE
#define UNKNOWN 0
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3

// -- ROAD TYPE
#define LANE_UNKNOWN 0
#define LANE_FREEWAY 1
#define LANE_SURFACE_STREET 2
#define LANE_BIKE_LANE 3
#define LANE_BUS_LANE 4

#define ROAD_LINE_UNKNOWN 10
#define ROAD_LINE_BROKEN_SINGLE_WHITE 11
#define ROAD_LINE_SOLID_SINGLE_WHITE 12
#define ROAD_LINE_SOLID_DOUBLE_WHITE 13
#define ROAD_LINE_BROKEN_SINGLE_YELLOW 14
#define ROAD_LINE_BROKEN_DOUBLE_YELLOW 15
#define ROAD_LINE_SOLID_SINGLE_YELLOW 16
#define ROAD_LINE_SOLID_DOUBLE_YELLOW 17
#define ROAD_LINE_PASSING_DOUBLE_YELLOW 18

#define ROAD_EDGE_UNKNOWN 20
#define ROAD_EDGE_BOUNDARY 21
#define ROAD_EDGE_MEDIAN 22

#define MISC_UNKNOWN 30
#define MISC_CROSSWALK 31
#define MISC_SPEED_BUMP 32

// -- NORMALIZED ROAD TYPE (returned by normalize_road_type)
#define NORMALIZED_ROAD_NONE 0
#define NORMALIZED_ROAD_LANE 1
#define NORMALIZED_ROAD_LINE 2
#define NORMALIZED_ROAD_EDGE 3

// -- AGENT CONTROL STATE
#define CONTROL_STATE_ACTIVE 0
#define CONTROL_STATE_MOVING 1
#define CONTROL_STATE_STATIC 2

// -- TRAFFIC CONTROL TYPE
#define TRAFFIC_CONTROL_TYPE_NONE 0
#define TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT 1
#define TRAFFIC_CONTROL_TYPE_STOP_SIGN 2
#define TRAFFIC_CONTROL_TYPE_YIELD_SIGN 3
#define NUM_TRAFFIC_CONTROL_TYPES 4
// -- TRAFFIC CONTROL STATE
#define TRAFFIC_CONTROL_STATE_UNKNOWN 0
#define TRAFFIC_CONTROL_STATE_RED 1
#define TRAFFIC_CONTROL_STATE_YELLOW 2
#define TRAFFIC_CONTROL_STATE_GREEN 3
#define TRAFFIC_CONTROL_STATE_OFF 4
#define NUM_TRAFFIC_CONTROL_STATES 5

static inline int is_road_lane(int type) {
    return (type >= 0 && type <= 9);
}

static inline int is_drivable_road_lane(int type) {
    return (type == LANE_FREEWAY || type == LANE_SURFACE_STREET);
}

static inline int is_road_line(int type) {
    return (type >= 10 && type <= 19);
}

static inline int is_road_edge(int type) {
    return (type >= 20 && type <= 29);
}

static inline int is_misc_road(int type) {
    return type >= MISC_UNKNOWN;
}

static inline int is_road(int type) {
    return is_road_lane(type) || is_road_line(type) || is_road_edge(type);
}

static inline int is_controllable_agent(int type) {
    return (type == VEHICLE || type == PEDESTRIAN || type == CYCLIST);
}

static inline int normalize_road_type(int type) {
    if (is_road_lane(type)) {
        return NORMALIZED_ROAD_LANE;
    } else if (is_road_line(type)) {
        return NORMALIZED_ROAD_LINE;
    } else if (is_road_edge(type)) {
        return NORMALIZED_ROAD_EDGE;
    } else {
        return NORMALIZED_ROAD_NONE;
    }
}

static inline int unnormalize_road_type(int norm_type) {
    if (norm_type == NORMALIZED_ROAD_LANE) {
        return LANE_SURFACE_STREET;
    } else if (norm_type == NORMALIZED_ROAD_LINE) {
        return ROAD_LINE_SOLID_SINGLE_WHITE;
    } else if (norm_type == NORMALIZED_ROAD_EDGE) {
        return ROAD_EDGE_BOUNDARY;
    } else {
        return 0;
    }
}

// Metrics array indices
#define NUM_METRICS 5
#define COLLISION_IDX 0
#define OFFROAD_IDX 1
#define REACH_GOAL_IDX 2
#define LANE_DIST_IDX 3
#define LANE_ANGLE_IDX 4

struct Agent {
    int type;

    // Log trajectory
    int trajectory_length;
    float log_trajectory_distance;
    float *log_trajectory_x;
    float *log_trajectory_y;
    float *log_trajectory_z;
    float *log_heading;
    float *log_velocity_x;
    float *log_velocity_y;
    float *log_length;
    float *log_width;
    float *log_height;
    int *log_valid;

    // Simulation state
    float sim_x;
    float sim_y;
    float sim_z;
    float sim_heading;
    float cos_heading;
    float sin_heading;
    float sim_vx;
    float sim_vy;
    float sim_length;
    float sim_width;
    float sim_height;
    int sim_valid;
    int collision_state;

    // Metrics and status tracking
    float metrics_array[NUM_METRICS];
    int current_lane_idx;
    int previous_lane_idx;
    int respawn_timestep;
    int reached_goal_this_episode;
    int collided_before_goal;
    int reached_goal;
    int active_agent;
    int control_state;
    int stopped;
    int removed;

    // Goal position
    float goal_position_x;
    float goal_position_y;
    float goal_position_z;
};

struct RoadMapElement {
    int type;

    int segment_length;
    float *x;
    float *y;
    float *z;
    float *headings;

    // Lane specific info
    int num_entries;
    int *entry_lanes;
    int num_exits;
    int *exit_lanes;
    float speed_limit;
};

struct TrafficControlElement {
    int type;

    int state_length;
    int *states;
    float stop_line[6]; // Two 3D endpoints: [x1,y1,z1, x2,y2,z2]
    float heading;
    int num_controlled_lanes;
    int *controlled_lanes;
};

static inline void free_agent(struct Agent *agent) {
    free(agent->log_trajectory_x);
    free(agent->log_trajectory_y);
    free(agent->log_trajectory_z);
    free(agent->log_heading);
    free(agent->log_velocity_x);
    free(agent->log_velocity_y);
    free(agent->log_length);
    free(agent->log_width);
    free(agent->log_height);
    free(agent->log_valid);
}

static inline void free_road_element(struct RoadMapElement *element) {
    free(element->x);
    free(element->y);
    free(element->z);
    free(element->headings);
    free(element->entry_lanes);
    free(element->exit_lanes);
}

static inline void free_traffic_element(struct TrafficControlElement *element) {
    free(element->states);
    free(element->controlled_lanes);
}

#endif // SIM_DATATYPES_H
