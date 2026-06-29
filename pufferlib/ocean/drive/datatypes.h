#include <stdlib.h>

// -- REWARD CONDITIONING COEFFICIENTS
#define REWARD_COEF_GOAL_RADIUS 0
#define REWARD_COEF_GOAL_SPEED 1
#define REWARD_COEF_COLLISION 2
#define REWARD_COEF_OFFROAD 3
#define REWARD_COEF_COMFORT 4
#define REWARD_COEF_LANE_ALIGN 5
#define REWARD_COEF_VEL_ALIGN 6
#define REWARD_COEF_LANE_CENTER 7
#define REWARD_COEF_CENTER_BIAS 8
#define REWARD_COEF_VELOCITY 9
#define REWARD_COEF_REVERSE 10
#define REWARD_COEF_STOP_LINE 11
#define REWARD_COEF_TIMESTEP 12
#define REWARD_COEF_OVERSPEED 13
// Dynamic conditioning coefficients
#define REWARD_COEF_THROTTLE 14
#define REWARD_COEF_STEER 15
#define REWARD_COEF_ACC 16
#define NUM_REWARD_COEFS 17

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

#define TRAFFIC_CONTROL_SCOPE_TRAFFIC_LIGHTS 0
#define TRAFFIC_CONTROL_SCOPE_TRAFFIC_LIGHTS_STOP_SIGN 1
#define TRAFFIC_CONTROL_SCOPE_ALL 2

// Metrics array indices
#define NUM_METRICS 18
#define COLLISION_IDX 0
#define OFFROAD_IDX 1
#define RED_LIGHT_IDX 2
#define STOP_SIGN_IDX 3
#define REACHED_GOAL_IDX 4
#define LANE_DIST_IDX 5
#define LANE_ANGLE_IDX 6
#define COMFORT_VIOLATION_IDX 7
#define VELOCITY_PROGRESS_IDX 8
#define SPEED_LIMIT_IDX 9
#define AVG_DISPLACEMENT_ERROR_IDX 10
#define PROGRESSION_IDX 11
// Evaluation metrics
#define AT_FAULT_COLLISION_IDX 12
#define TTC_IDX 13
#define DISTANCE_TO_COLLISION_IDX 14
#define PROGRESS_RATIO_IDX 15
#define MULTI_LANE_TIME_IDX 16
#define MULTI_LANE_SCORE_IDX 17

// Path
#define MAX_NUM_WP_PATH 200
#define MAX_TARGET_WAYPOINTS 20

// obs_html_frame array field counts
#define AGENT_F32_FIELDS                                                                                               \
    12 // sim_x/y/z, heading, length, width, speed, steering, accel_long, accel_lat, jerk_long, jerk_lat
#define AGENT_I32_FIELDS 8             // id, type, sim_valid, active_agent, stopped, removed, lane_idx, active_idx
#define METRICS_F32_FIELDS NUM_METRICS // must equal NUM_METRICS
#define SCORE_F32_FIELDS 15            // Log struct fields: puffer_score .. weighted_average
#define TRAFFIC_I16_FIELDS 3           // is_valid, type, state

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

static inline int is_road_grid_candidate(int type) {
    return is_road_lane(type) || is_road_edge(type);
}

static inline int is_controllable_agent(int type) {
    return (type == VEHICLE || type == PEDESTRIAN || type == CYCLIST);
}

static inline int traffic_control_in_scope(int type, int scope) {
    if (scope == TRAFFIC_CONTROL_SCOPE_TRAFFIC_LIGHTS) {
        return type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
    } else if (scope == TRAFFIC_CONTROL_SCOPE_TRAFFIC_LIGHTS_STOP_SIGN) {
        return type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT || type == TRAFFIC_CONTROL_TYPE_STOP_SIGN;
    } else if (scope == TRAFFIC_CONTROL_SCOPE_ALL) {
        return type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT || type == TRAFFIC_CONTROL_TYPE_STOP_SIGN
            || type == TRAFFIC_CONTROL_TYPE_YIELD_SIGN;
    } else {
        return type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
    }
}

struct Waypoint {
    float s;           // Arc length (cumulative distance from the start) - init position
    float x;           // Global x-coordinate
    float y;           // Global y-coordinate
    float z;           // Global z-coordinate
    float heading;     // Global heading (tangent angle) in radians
    float cos_heading; // Cached cosf(heading) - set in build_path
    float sin_heading; // Cached sinf(heading) - set in build_path
    int lane_idx;      // Index of the lane this waypoint
};
struct Path {
    struct Waypoint waypoints[MAX_NUM_WP_PATH];
    int num_waypoints;
};

struct Agent {
    int id;
    int type;

    // Log trajectory
    int trajectory_size;
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
    float sim_x;       // Bounding box center x
    float sim_y;       // Bounding box center y
    float sim_z;       // Bounding box center z
    float sim_heading; // Bounding box heading
    float cos_heading;
    float sin_heading;
    float sim_vx;           // Bounding box velocity x
    float sim_vy;           // Bounding box velocity y
    float yaw_rate;         // Angular velocity used to convert between rear and center velocity
    float sim_speed;        // Bounding box center speed magnitude
    float sim_speed_signed; // Bounding box center signed longitudinal speed
    float sim_length;       // Bounding box length
    float sim_width;        // Bounding box width
    float sim_height;       // Bounding box height
    float radius;           // Circumradius (smallest enclosing circle) -> 0.5*sqrt(L^2+W^2)
    float prev_x;
    float prev_y;
    float prev_cos_heading;
    float prev_sin_heading;
    int sim_valid;

    // Route information
    int route_length;
    int *route;
    int route_gt_len;        // Number of leading route lanes supported by GT before extension
    int current_route_index; // Tracks progress through route array

    // Path
    struct Path *path;
    int closest_path_idx_wp;

    // Metrics and status tracking (size must match NUM_METRICS in drive.h)
    float metrics_array[NUM_METRICS]; // [collision, offroad, red_light, stop_sign, reached_goal, lane_dist, lane_angle,
                                      // comfort_violation, velocity_progress, speed_limit, avg_displacement_error,
                                      // progression, at_fault_collision, ttc, distance_to_collision, progress_ratio,
                                      // multi_lane_time, multi_lane_score]
    int current_lane_idx;
    int previous_lane_idx;
    int current_lane_geometry_idx;
    int reached_goal_this_episode;
    int num_waypoints_reached;
    int num_goals_reached;
    int active_agent;
    int mark_as_expert;
    int controller;
    float cumulative_displacement;
    int displacement_sample_count;
    float path_progression;
    float distance_since_spawn;
    float seconds_stopped;

    // Goal positions (N sequential waypoints)
    float goal_positions_x[MAX_TARGET_WAYPOINTS];
    float goal_positions_y[MAX_TARGET_WAYPOINTS];
    float goal_positions_z[MAX_TARGET_WAYPOINTS];
    float goal_position_x; // alias = goal_positions_x[current_goal_idx]
    float goal_position_y; // alias = goal_positions_y[current_goal_idx]
    float goal_position_z; // alias = goal_positions_z[current_goal_idx]
    int current_goal_idx;  // index of next goal to reach (0..N-1)

    int stopped; // 0/1 -> freeze if set
    int removed; // 0/1 -> remove from sim if set

    // Jerk dynamics
    float accel_long;
    float accel_lat;
    float jerk_long;
    float jerk_lat;
    float steering_angle;
    float wheelbase;

    // Reward conditioning coefficients (per-agent, randomized at spawn)
    float reward_coefs[NUM_REWARD_COEFS];

    int phantom_braking_counter;     // >0 means currently phantom braking
    unsigned char is_blind_partner;  // episode-level flag: agent sees no other agents
    unsigned char is_phantom_braker; // episode-level flag: agent may phantom-brake
};

struct RoadMapElement {
    int type;

    int segment_size;
    float *x;
    float *y;
    float *z;
    float *headings; // Pre-computed heading for each segment

    // Lane specific info
    int num_entries;
    int *entry_lanes;
    int num_exits;
    int *exit_lanes;
    float speed_limit;
    float length;
    float *cum_lengths;
};

struct TrafficControlElement {
    int type;

    int state_size;
    int *states;
    float stop_line[6]; // Two 3D endpoints: [x1,y1,z1, x2,y2,z2]
    float heading;
    int num_controlled_lanes;
    int *controlled_lanes;
};

struct LaneGraph {
    int n_lanes;
    int *lane_ids;
    float *distances; // n_lanes * n_lanes row-major
};

void free_agent(struct Agent *agent) {
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
    free(agent->route);
    free(agent->path);
}

void free_road_element(struct RoadMapElement *element) {
    free(element->x);
    free(element->y);
    free(element->z);
    free(element->headings);
    free(element->entry_lanes);
    free(element->exit_lanes);
    free(element->cum_lengths);
}

void free_traffic_element(struct TrafficControlElement *element) {
    free(element->states);
    free(element->controlled_lanes);
}

void free_lane_graph(struct LaneGraph *graph) {
    free(graph->lane_ids);
    free(graph->distances);
}
