struct DynamicAgent {
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
    int* log_valid;
    float* length;
    float* width;
    float* height;

    // Simulation state (current timestep only - scalars, not arrays)
    float sim_x;
    float sim_y;
    float sim_z;
    float sim_heading;
    float sim_vx;
    float sim_vy;
    int sim_valid;

    // Route information
    int* routes;

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
    float goal_radius;
    float init_goal_x;    // Initialized from goal_position
    float init_goal_y;    // Initialized from goal_position

    // Respawn tracking
    int respawn_timestep;
    int respawn_count;
};

struct RoadMapElement {
    int id;
    int type;
    int segment_length;
    float* x;
    float* y;
    float* z;
    float* dir_x;
    float* dir_y;
    float* dir_z;

    // Lane specific info
    int entry;
    int exit;
    float speed_limit;
};

struct TrafficControlElement {
    int id;
    int type;
    int state_length;
    float x;
    float y;
    float z;
    int* states;
    int controlled_lane;
};

void free_dynamic_agent(struct DynamicAgent* agent){
    free(agent->log_trajectory_x);
    free(agent->log_trajectory_y);
    free(agent->log_trajectory_z);
    free(agent->log_heading);
    free(agent->log_velocity_x);
    free(agent->log_velocity_y);
    free(agent->log_valid);
    free(agent->length);
    free(agent->width);
    free(agent->height);
    free(agent->routes);
}

void free_road_element(struct RoadMapElement* element){
    free(element->x);
    free(element->y);
    free(element->z);
    free(element->dir_x);
    free(element->dir_y);
    free(element->dir_z);
}

void free_traffic_element(struct TrafficControlElement* element){
    free(element->states);
}
