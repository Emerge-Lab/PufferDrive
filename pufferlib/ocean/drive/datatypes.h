#define UNKNOWN 0

// -- AGENT TYPE
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3
#define OTHER 4

// -- ROAD TYPE
#define LANE_FREEWAY 1
#define LANE_SURFACE_STREET 2
#define LANE_BIKE_LANE 3

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
#define ROAD_EDGE_SIDEWALK 23

#define CROSSWALK 31
#define SPEED_BUMP 32
#define DRIVEWAY 33

// -- TRAFFIC CONTROL TYPE
#define TRAFFIC_LIGHT 1
#define STOP_SIGN 2
#define YIELD_SIGN 3
#define SPEED_LIMIT_SIGN 4


int is_road_lane(int type){
    return (type >= 0 && type <= 9);
}

int is_drivable_road_lane(int type){
    return (type == LANE_FREEWAY || type == LANE_SURFACE_STREET);
}

int is_road_line(int type){
    return (type >= 10 && type <= 19);
}

int is_road_edge(int type){
    return (type >= 20 && type <= 29);
}

int is_road(int type){
    return is_road_lane(type) || is_road_line(type) || is_road_edge(type);
}

int is_controllable_agent(int type){
    return (type == VEHICLE || type == PEDESTRIAN || type == CYCLIST);
}

int normalize_road_type(int type){
    if(is_road_lane(type)){
        return 0;
    } else if(is_road_line(type)){
        return 1;
    } else if(is_road_edge(type)){
        return 2;
    } else {
        return -1;
    }
}

int unnormalize_road_type(int norm_type){
    if(norm_type == 0){
        return LANE_SURFACE_STREET;
    } else if(norm_type == 1){
        return ROAD_LINE_BROKEN_SINGLE_WHITE;
    } else if(norm_type == 2){
        return ROAD_EDGE_BOUNDARY;
    } else {
        return -1; // Invalid
    }
}



struct Agent {
    int id;
    int type;

    // Log trajectory
    int trajectory_length;
    float* log_trajectory_x;
    float* log_trajectory_y;
    float* log_trajectory_z;
    float* log_heading;
    float* log_velocity_x;
    float* log_velocity_y;
    float* log_length;
    float* log_width;
    float* log_height;
    int* log_valid;

    // Simulation state
    float sim_x;
    float sim_y;
    float sim_z;
    float sim_heading;
    float sim_vx;
    float sim_vy;
    float sim_length;
    float sim_width;
    float sim_height;
    int sim_valid;

    // Route information
    int route_length;
    int* route;

    // Metrics and status tracking
    int collision_state;
    float metrics_array[5]; // [collision, offroad, reached_goal, lane_aligned, avg_displacement_error]
    int current_lane_idx;
    int collided_before_goal;
    int sampled_new_goal;
    int reached_goal_this_episode;
    int num_goals_reached;
    int active_agent;
    int mark_as_expert;
    float cumulative_displacement;
    int displacement_sample_count;

    // Goal positions
    float goal_position_x;
    float goal_position_y;
    float goal_position_z;
    float init_goal_x;    // Initialized from goal_position
    float init_goal_y;    // Initialized from goal_position

    // Respawn tracking
    int respawn_timestep;
    int respawn_count;

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

struct RoadMapElement {
    int id;
    int type;

    int segment_length;
    float* x;
    float* y;
    float* z;

    // Lane specific info
    int num_entries;
    int* entry_lanes;
    int num_exits;
    int* exit_lanes;
    float speed_limit;
};

struct TrafficControlElement {
    int id;
    int type;

    int state_length;
    int* states;
    float x;
    float y;
    float z;
    int controlled_lane;
};

void free_agent(struct Agent* agent){
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
}

void free_road_element(struct RoadMapElement* element){
    free(element->x);
    free(element->y);
    free(element->z);
    free(element->entry_lanes);
    free(element->exit_lanes);
}

void free_traffic_element(struct TrafficControlElement* element){
    free(element->states);
}
