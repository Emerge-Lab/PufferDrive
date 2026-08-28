#ifndef PUFFERLIB_OCEAN_DRIVE_MAP_DATA_H
#define PUFFERLIB_OCEAN_DRIVE_MAP_DATA_H

// ========================================
// Grid Map Functions
// ========================================

static int get_grid_index(Drive *env, float x1, float y1) {
    if (env->grid_map->top_left_x >= env->grid_map->bottom_right_x
        || env->grid_map->bottom_right_y >= env->grid_map->top_left_y) {
        return -1;
    }
    float rel_x = x1 - env->grid_map->top_left_x;
    float rel_y = y1 - env->grid_map->bottom_right_y;
    int grid_x = (int) (rel_x / GRID_CELL_SIZE);
    int grid_y = (int) (rel_y / GRID_CELL_SIZE);
    if (grid_x < 0 || grid_x >= env->grid_map->grid_cols || grid_y < 0 || grid_y >= env->grid_map->grid_rows) {
        return -1;
    }
    return (grid_y * env->grid_map->grid_cols) + grid_x;
}

static void add_entity_to_grid(
    Drive *env,
    int grid_index,
    int entity_idx,
    int geometry_idx,
    int valid_for_obs,
    int *cell_entities_insert_index) {
    if (grid_index == -1) {
        return;
    }

    int count = cell_entities_insert_index[grid_index];
    if (count >= env->grid_map->cell_entities_count[grid_index]) {
        printf(
            "Error: Exceeded precomputed entity count for grid cell %d. Current count: %d, Max count(Precomputed): "
            "%d\n",
            grid_index,
            count,
            env->grid_map->cell_entities_count[grid_index]);
        return;
    }

    env->grid_map->cells[grid_index][count].entity_idx = entity_idx;
    env->grid_map->cells[grid_index][count].geometry_idx = geometry_idx;
    env->grid_map->cells[grid_index][count].valid_for_obs = valid_for_obs;
    cell_entities_insert_index[grid_index] = count + 1;
}

static int init_grid_map(Drive *env) {
    env->grid_map = (GridMap *) calloc(1, sizeof(GridMap));

    float top_left_x = 0.0f, top_left_y = 0.0f, bottom_right_x = 0.0f, bottom_right_y = 0.0f;
    bool first_valid_point = false;
    for (int i = 0; i < env->num_road_elements; i++) {
        if (!is_road_grid_candidate(env->road_elements[i].type)) {
            continue;
        }
        RoadMapElement *element = &env->road_elements[i];
        for (int j = 0; j < element->segment_size; j++) {
            if (element->x[j] == INVALID_POSITION || element->y[j] == INVALID_POSITION) {
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
    if (!first_valid_point || !isfinite(grid_width) || !isfinite(grid_height)) {
        fprintf(stderr, "[ERROR] -> Map has no valid road geometry extent for grid\n");
        return 1;
    }
    env->grid_map->grid_cols = ceil(grid_width / GRID_CELL_SIZE);
    env->grid_map->grid_rows = ceil(grid_height / GRID_CELL_SIZE);
    long long grid_cell_count_wide = (long long) env->grid_map->grid_cols * (long long) env->grid_map->grid_rows;
    if (env->grid_map->grid_cols < 1 || env->grid_map->grid_rows < 1 || grid_cell_count_wide > MAX_GRID_CELL_COUNT) {
        fprintf(
            stderr,
            "[ERROR] -> Invalid grid dimensions %d x %d from map extent %.1f x %.1f\n",
            env->grid_map->grid_cols,
            env->grid_map->grid_rows,
            grid_width,
            grid_height);
        return 1;
    }
    int grid_cell_count = (int) grid_cell_count_wide;
    env->grid_map->cells = (GridMapEntity **) calloc(grid_cell_count, sizeof(GridMapEntity *));
    env->grid_map->cell_entities_count = (int *) calloc(grid_cell_count, sizeof(int));
    // First pass to count entities in each grid cell
    for (int i = 0; i < env->num_road_elements; i++) {
        if (!is_road_grid_candidate(env->road_elements[i].type)) {
            continue;
        }
        RoadMapElement *element = &env->road_elements[i];
        for (int j = 0; j < element->segment_size - 1; j++) {
            float x_center = (element->x[j] + element->x[j + 1]) / 2;
            float y_center = (element->y[j] + element->y[j + 1]) / 2;
            int grid_index = get_grid_index(env, x_center, y_center);
            if (grid_index == -1) {
                continue;
            }
            env->grid_map->cell_entities_count[grid_index]++;
        }
    }
    // Allocate grid cells based on counts
    int *cell_entities_insert_index = (int *) calloc(grid_cell_count, sizeof(int));
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        int count = env->grid_map->cell_entities_count[grid_index];
        env->grid_map->total_entities += count;
        env->grid_map->cells[grid_index] = (GridMapEntity *) calloc(count, sizeof(GridMapEntity));
    }
    // Track which grid cells have drivable lanes
    bool *drivable_grid_seen = (bool *) calloc(grid_cell_count, sizeof(bool));
    for (int i = 0; i < env->num_road_elements; i++) {
        if (!is_road_grid_candidate(env->road_elements[i].type)) {
            continue;
        }
        RoadMapElement *element = &env->road_elements[i];
        int obs_stride = 1;
        if (is_road_lane(element->type)) {
            obs_stride = env->obs_lane_stride;
        } else if (is_road_edge(element->type)) {
            obs_stride = env->obs_boundary_stride;
        }
        int last_kept_idx = 0;
        for (int j = 0; j < element->segment_size - 1; j++) {
            // Keep a point every obs_stride points, plus wherever heading deviates enough
            // since the last kept point (densifies curves/intersections)
            int valid_for_obs = 1;
            if (obs_stride > 1 && j > 0) {
                float heading_dev = fabsf(normalize_heading(element->headings[j] - element->headings[last_kept_idx]));
                valid_for_obs = j - last_kept_idx >= obs_stride || heading_dev > OBS_STRIDE_HEADING_THRESHOLD;
            }
            if (valid_for_obs) {
                last_kept_idx = j;
            }
            float x_center = (element->x[j] + element->x[j + 1]) / 2;
            float y_center = (element->y[j] + element->y[j + 1]) / 2;
            int grid_index = get_grid_index(env, x_center, y_center);
            if (grid_index == -1) {
                continue;
            }
            add_entity_to_grid(env, grid_index, i, j, valid_for_obs, cell_entities_insert_index);
            if (is_drivable_road_lane(element->type) && !drivable_grid_seen[grid_index]) {
                drivable_grid_seen[grid_index] = true;
                env->grid_map->num_drivable_grid_cell++;
            }
        }
    }
    // Create a compact array of drivable grid cell indices for quick access
    env->grid_map->grid_index_drivable = (int *) malloc(env->grid_map->num_drivable_grid_cell * sizeof(int));
    int drivable_idx = 0;
    for (int i = 0; i < grid_cell_count; i++) {
        if (drivable_grid_seen[i]) {
            env->grid_map->grid_index_drivable[drivable_idx++] = i;
        }
    }
    free(drivable_grid_seen);
    free(cell_entities_insert_index);
    return 0;
}

static void init_neighbor_offsets(Drive *env) {
    int vr = env->grid_map->vision_range;
    env->neighbor_offsets = (int *) calloc(vr * vr * 2, sizeof(int));
    // Spiral pattern generation
    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0, y = 0, dir = 0, steps_taken = 0, segments_completed = 0, total = 0, curr_idx = 0;
    int steps_to_take = 1;
    int max_offsets = vr * vr;
    env->neighbor_offsets[curr_idx++] = 0;
    env->neighbor_offsets[curr_idx++] = 0;
    total++;
    // Generate spiral pattern
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
        dir = (dir + 1) % 4; // Change direction (clockwise: right->up->left->down)
        segments_completed++;
        if (segments_completed % 2 == 0) {
            steps_to_take++;
        }
    }
}

static void cache_neighbor_offsets(Drive *env) {
    int count = 0;
    int cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->neighbor_cache_entities = (GridMapEntity **) calloc(cell_count, sizeof(GridMapEntity *));
    env->grid_map->neighbor_cache_count = (int *) calloc(cell_count + 1, sizeof(int));
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int current_cell_neighbor_count = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows) {
                continue;
            }
            int grid_count = env->grid_map->cell_entities_count[grid_index];
            current_cell_neighbor_count += grid_count;
        }
        env->grid_map->neighbor_cache_count[i] = current_cell_neighbor_count;
        count += current_cell_neighbor_count;
        if (current_cell_neighbor_count == 0) {
            env->grid_map->neighbor_cache_entities[i] = NULL;
            continue;
        }
        env->grid_map->neighbor_cache_entities[i]
            = (GridMapEntity *) calloc(current_cell_neighbor_count, sizeof(GridMapEntity));
    }

    env->grid_map->neighbor_cache_count[cell_count] = count;
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols;
        int cell_y = i / env->grid_map->grid_cols;
        int base_index = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows) {
                continue;
            }
            int grid_count = env->grid_map->cell_entities_count[grid_index];
            // Skip if no entities or source is NULL
            if (grid_count == 0 || env->grid_map->cells[grid_index] == NULL) {
                continue;
            }
            // Copy grid_count pairs (entity_idx, geometry_idx) at once
            memcpy(
                &env->grid_map->neighbor_cache_entities[i][base_index],
                env->grid_map->cells[grid_index],
                grid_count * sizeof(GridMapEntity));
            base_index += grid_count;
        }
    }
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
    // Calculate 2D grid coordinates
    int cols = env->grid_map->grid_cols;
    int cell_x = index % cols;
    int cell_y = index / cols;
    int entity_list_count = 0;
    // Fill the provided array
    for (int i = 0; i < offset_size; i++) {
        int nx = cell_x + local_offsets[i][0];
        int ny = cell_y + local_offsets[i][1];
        // Ensure the neighbor is within grid bounds
        if (nx < 0 || nx >= env->grid_map->grid_cols || ny < 0 || ny >= env->grid_map->grid_rows) {
            continue;
        }
        int neighbor_idx = ny * env->grid_map->grid_cols + nx;
        int count = env->grid_map->cell_entities_count[neighbor_idx];
        // Add entities from this cell to the list
        for (int j = 0; j < count && entity_list_count < max_size; j++) {
            entity_list[entity_list_count++] = env->grid_map->cells[neighbor_idx][j];
        }
    }
    return entity_list_count;
}

// ========================================
// Map Loading Functions
// ========================================

int load_map_binary(const char *filename, Drive *drive) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        return -1;
    }

    int loaded_agent_count, num_roads, num_traffic, num_objects;
    if (fread(&loaded_agent_count, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    if (fread(&num_roads, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    if (fread(&num_traffic, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    if (fread(&num_objects, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    drive->num_total_agents = loaded_agent_count;
    drive->num_road_elements = num_roads;
    drive->num_traffic_elements = num_traffic;
    drive->num_objects = num_objects;

    if (loaded_agent_count > 0) {
        drive->agents = (Agent *) calloc(loaded_agent_count, sizeof(Agent));
    }
    if (num_roads > 0) {
        drive->road_elements = (RoadMapElement *) calloc(num_roads, sizeof(RoadMapElement));
    }
    if (num_traffic > 0) {
        drive->traffic_elements = (TrafficControlElement *) calloc(num_traffic, sizeof(TrafficControlElement));
    }

    for (int i = 0; i < loaded_agent_count; i++) {
        Agent *agent = &drive->agents[i];

        if (fread(&agent->id, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (agent->id != i) {
            printf("[ERROR] -> Agent id %d != idx %d. Binary must be reindexed (id == idx).\n", agent->id, i);
            fclose(file);
            return -1;
        }
        if (fread(&agent->type, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fread(&agent->trajectory_size, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }

        int tlen = agent->trajectory_size;
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

        if ((size_t) tlen > 0 && fread(agent->log_trajectory_x, sizeof(float), tlen, file) != (size_t) tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t) tlen > 0 && fread(agent->log_trajectory_y, sizeof(float), tlen, file) != (size_t) tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t) tlen > 0 && fread(agent->log_trajectory_z, sizeof(float), tlen, file) != (size_t) tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t) tlen > 0 && fread(agent->log_heading, sizeof(float), tlen, file) != (size_t) tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t) tlen > 0 && fread(agent->log_velocity_x, sizeof(float), tlen, file) != (size_t) tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t) tlen > 0 && fread(agent->log_velocity_y, sizeof(float), tlen, file) != (size_t) tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t) tlen > 0 && fread(agent->log_length, sizeof(float), tlen, file) != (size_t) tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t) tlen > 0 && fread(agent->log_width, sizeof(float), tlen, file) != (size_t) tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t) tlen > 0 && fread(agent->log_height, sizeof(float), tlen, file) != (size_t) tlen) {
            fclose(file);
            return -1;
        }
        if ((size_t) tlen > 0 && fread(agent->log_valid, sizeof(int), tlen, file) != (size_t) tlen) {
            fclose(file);
            return -1;
        }

        if (fread(&agent->route_length, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }

        if (agent->route_length > 0) {
            agent->route = (int *) malloc(agent->route_length * sizeof(int));
            if (fread(agent->route, sizeof(int), agent->route_length, file) != (size_t) agent->route_length) {
                fclose(file);
                return -1;
            }
        } else {
            agent->route = NULL;
        }

        if (fread(&agent->route_gt_len, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }

        if (fread(&agent->gt_goal_x, sizeof(float), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fread(&agent->gt_goal_y, sizeof(float), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fread(&agent->gt_goal_z, sizeof(float), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fread(&agent->mark_as_expert, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
    }

    for (int i = 0; i < num_roads; i++) {
        RoadMapElement *road = &drive->road_elements[i];
        int road_id;

        if (fread(&road_id, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (road_id != i) {
            printf("[ERROR] -> Road element id %d != idx %d. Binary must be reindexed (id == idx).\n", road_id, i);
            fclose(file);
            return -1;
        }
        if (fread(&road->type, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fread(&road->segment_size, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }

        int slen = road->segment_size;

        road->x = (float *) malloc(slen * sizeof(float));
        road->y = (float *) malloc(slen * sizeof(float));
        road->z = (float *) malloc(slen * sizeof(float));

        if ((size_t) slen > 0 && fread(road->x, sizeof(float), slen, file) != (size_t) slen) {
            fclose(file);
            return -1;
        }
        if ((size_t) slen > 0 && fread(road->y, sizeof(float), slen, file) != (size_t) slen) {
            fclose(file);
            return -1;
        }
        if ((size_t) slen > 0 && fread(road->z, sizeof(float), slen, file) != (size_t) slen) {
            fclose(file);
            return -1;
        }

        road->headings = (float *) malloc(slen * sizeof(float));
        if ((size_t) slen > 0 && fread(road->headings, sizeof(float), slen, file) != (size_t) slen) {
            fclose(file);
            return -1;
        }

        if (is_road_lane(road->type)) {
            if (fread(&road->num_entries, sizeof(int), 1, file) != 1) {
                fclose(file);
                return -1;
            }
            if (road->num_entries > 0) {
                road->entry_lanes = (int *) malloc(road->num_entries * sizeof(int));
                if (fread(road->entry_lanes, sizeof(int), road->num_entries, file) != (size_t) road->num_entries) {
                    fclose(file);
                    return -1;
                }
            } else {
                road->entry_lanes = NULL;
            }

            if (fread(&road->num_exits, sizeof(int), 1, file) != 1) {
                fclose(file);
                return -1;
            }
            if (road->num_exits > 0) {
                road->exit_lanes = (int *) malloc(road->num_exits * sizeof(int));
                if (fread(road->exit_lanes, sizeof(int), road->num_exits, file) != (size_t) road->num_exits) {
                    fclose(file);
                    return -1;
                }
            } else {
                road->exit_lanes = NULL;
            }

            if (fread(&road->speed_limit, sizeof(float), 1, file) != 1) {
                fclose(file);
                return -1;
            }
            if (fread(&road->length, sizeof(float), 1, file) != 1) {
                fclose(file);
                return -1;
            }
            road->cum_lengths = (float *) malloc(slen * sizeof(float));
            if ((size_t) slen > 0 && fread(road->cum_lengths, sizeof(float), slen, file) != (size_t) slen) {
                fclose(file);
                return -1;
            }
        } else {
            road->num_entries = 0;
            road->num_exits = 0;
            road->entry_lanes = NULL;
            road->exit_lanes = NULL;
            road->speed_limit = 0.0f;
            road->length = 0.0f;
            road->cum_lengths = NULL;
        }
    }

    for (int i = 0; i < num_traffic; i++) {
        TrafficControlElement *tc = &drive->traffic_elements[i];
        int traffic_id;

        if (fread(&traffic_id, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (traffic_id != i) {
            printf(
                "[ERROR] -> Traffic element id %d != idx %d. Binary must be reindexed (id == idx).\n",
                traffic_id,
                i);
            fclose(file);
            return -1;
        }
        if (fread(&tc->type, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fread(tc->stop_line, sizeof(float), 6, file) != 6) {
            fclose(file);
            return -1;
        }
        if (fread(&tc->heading, sizeof(float), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (fread(&tc->state_size, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }

        int state_len = tc->state_size;

        tc->states = (int *) malloc(state_len * sizeof(int));
        if ((size_t) state_len > 0 && fread(tc->states, sizeof(int), state_len, file) != (size_t) state_len) {
            fclose(file);
            return -1;
        }

        if (fread(&tc->num_controlled_lanes, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        if (tc->num_controlled_lanes > 0) {
            tc->controlled_lanes = (int *) malloc(tc->num_controlled_lanes * sizeof(int));
            if (fread(tc->controlled_lanes, sizeof(int), tc->num_controlled_lanes, file)
                != (size_t) tc->num_controlled_lanes) {
                fclose(file);
                return -1;
            }
        } else {
            tc->controlled_lanes = NULL;
        }
    }

    // Skip objects section
    int num_objects_features = 9; // x,y,z,heading,vx,vy,length,width,height (9 float arrays) + valid (1 int array)
    for (int i = 0; i < num_objects; i++) {
        int obj_id, obj_type, traj_len;
        if (fread(&obj_id, sizeof(int), 1, file) != 1 || fread(&obj_type, sizeof(int), 1, file) != 1
            || fread(&traj_len, sizeof(int), 1, file) != 1) {
            fclose(file);
            return -1;
        }
        // Skip: x,y,z,heading,vx,vy,length,width,height (9 float arrays) + valid (1 int array)
        fseek(file, num_objects_features * traj_len * sizeof(float) + traj_len * sizeof(int), SEEK_CUR);
    }

    // Lane graph section
    int n_lanes_graph;
    if (fread(&n_lanes_graph, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    drive->lane_graph.n_lanes = n_lanes_graph;
    drive->lane_graph.lane_ids = NULL;
    drive->lane_graph.distances = NULL;
    drive->lane_graph.lane_to_graph_idx = NULL;
    if (n_lanes_graph > 0) {
        drive->lane_graph.lane_ids = (int *) malloc(n_lanes_graph * sizeof(int));
        if (fread(drive->lane_graph.lane_ids, sizeof(int), n_lanes_graph, file) != (size_t) n_lanes_graph) {
            fclose(file);
            return -1;
        }
        drive->lane_graph.distances = (float *) malloc(n_lanes_graph * n_lanes_graph * sizeof(float));
        if (fread(drive->lane_graph.distances, sizeof(float), n_lanes_graph * n_lanes_graph, file)
            != (size_t) (n_lanes_graph * n_lanes_graph)) {
            fclose(file);
            return -1;
        }

        // Build reverse lookup road-element idx -> graph idx
        int num_roads = drive->num_road_elements;
        drive->lane_graph.lane_to_graph_idx = (int *) malloc(num_roads * sizeof(int));
        for (int r = 0; r < num_roads; r++) {
            drive->lane_graph.lane_to_graph_idx[r] = -1;
        }
        for (int g = 0; g < n_lanes_graph; g++) {
            int lane_idx = drive->lane_graph.lane_ids[g];
            if (lane_idx < 0 || lane_idx >= num_roads) {
                printf("[ERROR] -> lane_graph lane_id %d out of range [0,%d).\n", lane_idx, num_roads);
                fclose(file);
                return -1;
            }
            drive->lane_graph.lane_to_graph_idx[lane_idx] = g;
        }
    }

    // Metadata
    if (fread(drive->scenario_id, sizeof(char), 128, file) != 128) {
        fclose(file);
        return -1;
    }
    if (fread(drive->dataset_name, sizeof(char), 32, file) != 32) {
        fclose(file);
        return -1;
    }
    if (fread(&drive->log_length, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    if (fread(&drive->log_dt, sizeof(float), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    if (fread(&drive->num_objects_of_interest, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    if (drive->num_objects_of_interest > 0) {
        drive->objects_of_interest = (int *) malloc(drive->num_objects_of_interest * sizeof(int));
        if (fread(drive->objects_of_interest, sizeof(int), drive->num_objects_of_interest, file)
            != (size_t) drive->num_objects_of_interest) {
            fclose(file);
            return -1;
        }
    } else {
        drive->objects_of_interest = NULL;
    }

    if (fread(&drive->num_tracks_to_predict, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    if (drive->num_tracks_to_predict > 0) {
        drive->tracks_to_predict = (int *) malloc(drive->num_tracks_to_predict * sizeof(int));
        if (fread(drive->tracks_to_predict, sizeof(int), drive->num_tracks_to_predict, file)
            != (size_t) drive->num_tracks_to_predict) {
            fclose(file);
            return -1;
        }
    } else {
        drive->tracks_to_predict = NULL;
    }

    fclose(file);
    return 0;
}

static struct SharedMapData *map_cache_lookup(Drive *env) {
    for (int i = 0; i < g_map_cache_count; i++) {
        if (g_map_cache[i] != NULL && strcmp(g_map_cache[i]->map_name, env->map_name) == 0
            && g_map_cache[i]->obs_lane_stride == env->obs_lane_stride
            && g_map_cache[i]->obs_boundary_stride == env->obs_boundary_stride) {
            return g_map_cache[i];
        }
    }
    return NULL;
}

static void map_cache_insert(struct SharedMapData *entry) {
    // Reuse a NULL slot left by free_shared_map_data before growing the array.
    for (int i = 0; i < g_map_cache_count; i++) {
        if (g_map_cache[i] == NULL) {
            g_map_cache[i] = entry;
            return;
        }
    }
    g_map_cache
        = (struct SharedMapData **) realloc(g_map_cache, (g_map_cache_count + 1) * sizeof(struct SharedMapData *));
    g_map_cache[g_map_cache_count++] = entry;
}

static void free_grid_map(GridMap *grid_map) {
    if (grid_map == NULL) {
        return;
    }
    int grid_cell_count = grid_map->grid_cols * grid_map->grid_rows;
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        free(grid_map->cells[grid_index]);
    }
    free(grid_map->cells);
    free(grid_map->cell_entities_count);
    free(grid_map->grid_index_drivable);
    if (grid_map->neighbor_cache_entities != NULL) {
        for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
            free(grid_map->neighbor_cache_entities[grid_index]);
        }
    }
    free(grid_map->neighbor_cache_entities);
    free(grid_map->neighbor_cache_count);
    free(grid_map);
}

// Free a shared geometry entry and everything it owns, and clear its cache slot.
// Only call in the building process.
static void free_shared_map_data(struct SharedMapData *shared) {
    if (shared == NULL) {
        return;
    }
    for (int i = 0; i < shared->num_road_elements; i++) {
        free_road_element(&shared->road_elements[i]);
    }
    free(shared->road_elements);
    free_grid_map(shared->grid_map);
    free(shared->neighbor_offsets);
    free_lane_graph(&shared->lane_graph);
    free(shared->map_name);
    for (int i = 0; i < g_map_cache_count; i++) {
        if (g_map_cache[i] == shared) {
            g_map_cache[i] = NULL;
            break;
        }
    }
    free(shared);
}

#endif
