#ifndef PUFFERLIB_OCEAN_DRIVE_IDM_H
#define PUFFERLIB_OCEAN_DRIVE_IDM_H

#define IDM_EXTENSION_MAGIC "PDRV_IDM_EXT_V1"
#define IDM_EXTENSION_MAGIC_BYTES 16

#define IDM_MINIMUM_LEAD_DISTANCE 0.1f
#define IDM_MIN_SPACING 2.0f
#define IDM_SAFE_TIME_HEADWAY 2.0f
#define IDM_DELTA 4.0f
#define IDM_LOOKAHEAD_TIME 5.0f
#define IDM_MIN_LOOKAHEAD 20.0f
#define IDM_MAX_LOOKAHEAD 80.0f
#define IDM_BBOX_MARGIN 0.05f
#define IDM_DEFAULT_DESIRED_SPEED 9.0f
#define IDM_ROUTE_SAMPLE_DS 1.0f

struct IDMRoadElement {
    int type;
    int segment_length;
    float *x;
    float *y;
    float *z;
    float *headings;
    int num_entries;
    int *entry_lanes;
    int num_exits;
    int *exit_lanes;
    float speed_limit;
};

struct IDMMap {
    int loaded;
    int num_agents;
    int *route_lengths;
    int **routes;
    int *route_gt_lens;
    int num_roads;
    IDMRoadElement *roads;
};

struct IDMAgentState {
    int initialized;
    int route_idx;
    int lane_idx;
    int segment_idx;
    float t;
};

typedef struct {
    int has_leader;
    int leader_agent_idx;
    float gap;
    float leader_speed;
} IDMLeader;

typedef struct {
    int valid;
    int route_idx;
    int lane_idx;
    int segment_idx;
    float t;
    float dist_sq;
} IDMLaneProjection;

static void idm_die(const char *message) {
    fprintf(stderr, "[drive/idm] %s\n", message);
    abort();
}

static int idm_is_lane_type(int type) { return type >= 0 && type <= 9; }

static void idm_reset_agent_states(Drive *env) {
    free(env->idm_agent_states);
    env->idm_agent_states = NULL;
}

static void idm_free_map(IDMMap *map) {
    if (map == NULL) {
        return;
    }
    if (map->routes != NULL) {
        for (int i = 0; i < map->num_agents; i++) {
            free(map->routes[i]);
        }
    }
    free(map->routes);
    free(map->route_lengths);
    free(map->route_gt_lens);

    if (map->roads != NULL) {
        for (int i = 0; i < map->num_roads; i++) {
            free(map->roads[i].x);
            free(map->roads[i].y);
            free(map->roads[i].z);
            free(map->roads[i].headings);
            free(map->roads[i].entry_lanes);
            free(map->roads[i].exit_lanes);
        }
    }
    free(map->roads);
    free(map);
}

static void idm_free(Drive *env) {
    idm_free_map(env->idm_map);
    env->idm_map = NULL;
    idm_reset_agent_states(env);
}

static void idm_read_or_die(FILE *file, void *ptr, size_t size, size_t count, const char *field) {
    if (count == 0) {
        return;
    }
    if (fread(ptr, size, count, file) != count) {
        char msg[256];
        snprintf(msg, sizeof(msg), "failed to read IDM extension field '%s'", field);
        idm_die(msg);
    }
}

static void idm_skip_or_die(FILE *file, long bytes, const char *field) {
    if (bytes <= 0) {
        return;
    }
    if (fseek(file, bytes, SEEK_CUR) != 0) {
        char msg[256];
        snprintf(msg, sizeof(msg), "failed to skip IDM extension field '%s'", field);
        idm_die(msg);
    }
}

static void idm_load_extension(FILE *file, Drive *env) {
    idm_free(env);

    char magic[IDM_EXTENSION_MAGIC_BYTES];
    size_t n = fread(magic, 1, IDM_EXTENSION_MAGIC_BYTES, file);
    if (n == 0) {
        return;
    }
    if (n != IDM_EXTENSION_MAGIC_BYTES || memcmp(magic, IDM_EXTENSION_MAGIC, strlen(IDM_EXTENSION_MAGIC) + 1) != 0) {
        return;
    }

    uint64_t payload_size = 0;
    idm_read_or_die(file, &payload_size, sizeof(uint64_t), 1, "payload_size");

    int num_agents = 0;
    int num_roads = 0;
    int num_traffic = 0;
    int num_objects = 0;
    idm_read_or_die(file, &num_agents, sizeof(int), 1, "num_agents");
    idm_read_or_die(file, &num_roads, sizeof(int), 1, "num_roads");
    idm_read_or_die(file, &num_traffic, sizeof(int), 1, "num_traffic");
    idm_read_or_die(file, &num_objects, sizeof(int), 1, "num_objects");

    if (num_agents <= 0 || num_roads <= 0) {
        idm_die("IDM extension has no agents or roads");
    }

    IDMMap *map = (IDMMap *)calloc(1, sizeof(IDMMap));
    if (map == NULL) {
        idm_die("failed to allocate IDM extension map");
    }
    env->idm_map = map;
    map->loaded = 1;
    map->num_agents = num_agents;
    map->route_lengths = (int *)calloc(num_agents, sizeof(int));
    map->routes = (int **)calloc(num_agents, sizeof(int *));
    map->route_gt_lens = (int *)calloc(num_agents, sizeof(int));
    map->num_roads = num_roads;
    map->roads = (IDMRoadElement *)calloc(num_roads, sizeof(IDMRoadElement));
    if (map->route_lengths == NULL || map->routes == NULL || map->route_gt_lens == NULL || map->roads == NULL) {
        idm_die("failed to allocate IDM extension map fields");
    }

    for (int i = 0; i < num_agents; i++) {
        int agent_id = -1;
        int agent_type = 0;
        int tlen = 0;
        idm_read_or_die(file, &agent_id, sizeof(int), 1, "agent_id");
        idm_read_or_die(file, &agent_type, sizeof(int), 1, "agent_type");
        idm_read_or_die(file, &tlen, sizeof(int), 1, "agent_tlen");
        if (agent_id != i || tlen < 0) {
            idm_die("IDM extension agents must be reindexed and have non-negative trajectory length");
        }
        idm_skip_or_die(file, (long)(9 * tlen * sizeof(float) + tlen * sizeof(int)), "agent_log_arrays");

        int route_length = 0;
        idm_read_or_die(file, &route_length, sizeof(int), 1, "agent_route_length");
        if (route_length < 0) {
            idm_die("IDM extension has negative route length");
        }
        map->route_lengths[i] = route_length;
        if (route_length > 0) {
            map->routes[i] = (int *)malloc(route_length * sizeof(int));
            if (map->routes[i] == NULL) {
                idm_die("failed to allocate IDM route");
            }
            idm_read_or_die(file, map->routes[i], sizeof(int), route_length, "agent_route");
        }
        idm_read_or_die(file, &map->route_gt_lens[i], sizeof(int), 1, "agent_route_gt_len");
        idm_skip_or_die(file, (long)(3 * sizeof(float) + sizeof(int)), "agent_goal_and_expert_flag");
    }

    for (int i = 0; i < num_roads; i++) {
        int road_id = -1;
        IDMRoadElement *road = &map->roads[i];
        idm_read_or_die(file, &road_id, sizeof(int), 1, "road_id");
        if (road_id != i) {
            idm_die("IDM extension roads must be reindexed");
        }
        idm_read_or_die(file, &road->type, sizeof(int), 1, "road_type");
        idm_read_or_die(file, &road->segment_length, sizeof(int), 1, "road_segment_length");
        if (road->segment_length < 0) {
            idm_die("IDM extension has negative road segment length");
        }

        int npts = road->segment_length;
        road->x = (float *)malloc(npts * sizeof(float));
        road->y = (float *)malloc(npts * sizeof(float));
        road->z = (float *)malloc(npts * sizeof(float));
        road->headings = (float *)malloc(npts * sizeof(float));
        if ((npts > 0) && (road->x == NULL || road->y == NULL || road->z == NULL || road->headings == NULL)) {
            idm_die("failed to allocate IDM road geometry");
        }
        idm_read_or_die(file, road->x, sizeof(float), npts, "road_x");
        idm_read_or_die(file, road->y, sizeof(float), npts, "road_y");
        idm_read_or_die(file, road->z, sizeof(float), npts, "road_z");
        idm_read_or_die(file, road->headings, sizeof(float), npts, "road_headings");

        if (idm_is_lane_type(road->type)) {
            idm_read_or_die(file, &road->num_entries, sizeof(int), 1, "road_num_entries");
            if (road->num_entries > 0) {
                road->entry_lanes = (int *)malloc(road->num_entries * sizeof(int));
                if (road->entry_lanes == NULL) {
                    idm_die("failed to allocate IDM road entries");
                }
                idm_read_or_die(file, road->entry_lanes, sizeof(int), road->num_entries, "road_entries");
            }
            idm_read_or_die(file, &road->num_exits, sizeof(int), 1, "road_num_exits");
            if (road->num_exits > 0) {
                road->exit_lanes = (int *)malloc(road->num_exits * sizeof(int));
                if (road->exit_lanes == NULL) {
                    idm_die("failed to allocate IDM road exits");
                }
                idm_read_or_die(file, road->exit_lanes, sizeof(int), road->num_exits, "road_exits");
            }
            idm_read_or_die(file, &road->speed_limit, sizeof(float), 1, "road_speed_limit");
        }
    }

    for (int i = 0; i < num_traffic; i++) {
        int traffic_id = 0;
        int traffic_type = 0;
        int state_length = 0;
        int controlled_lanes = 0;
        idm_read_or_die(file, &traffic_id, sizeof(int), 1, "traffic_id");
        idm_read_or_die(file, &traffic_type, sizeof(int), 1, "traffic_type");
        idm_skip_or_die(file, (long)(6 * sizeof(float) + sizeof(float)), "traffic_stop_line_heading");
        idm_read_or_die(file, &state_length, sizeof(int), 1, "traffic_state_length");
        idm_skip_or_die(file, (long)(state_length * sizeof(int)), "traffic_states");
        idm_read_or_die(file, &controlled_lanes, sizeof(int), 1, "traffic_num_controlled_lanes");
        idm_skip_or_die(file, (long)(controlled_lanes * sizeof(int)), "traffic_controlled_lanes");
    }

    for (int i = 0; i < num_objects; i++) {
        int obj_id = 0;
        int obj_type = 0;
        int tlen = 0;
        idm_read_or_die(file, &obj_id, sizeof(int), 1, "object_id");
        idm_read_or_die(file, &obj_type, sizeof(int), 1, "object_type");
        idm_read_or_die(file, &tlen, sizeof(int), 1, "object_tlen");
        idm_skip_or_die(file, (long)(9 * tlen * sizeof(float) + tlen * sizeof(int)), "object_arrays");
    }

    int graph_lanes = 0;
    idm_read_or_die(file, &graph_lanes, sizeof(int), 1, "lane_graph_count");
    idm_skip_or_die(
        file,
        (long)(graph_lanes * sizeof(int) + graph_lanes * sizeof(float) + graph_lanes * graph_lanes * sizeof(float)),
        "lane_graph");

    (void)payload_size;
}

static void idm_shift_map(Drive *env, float mean_x, float mean_y) {
    if (env->idm_map == NULL || !env->idm_map->loaded) {
        return;
    }
    for (int i = 0; i < env->idm_map->num_roads; i++) {
        IDMRoadElement *road = &env->idm_map->roads[i];
        for (int j = 0; j < road->segment_length; j++) {
            road->x[j] -= mean_x;
            road->y[j] -= mean_y;
        }
    }
}

static float idm_heading_diff(float new_heading, float old_heading) {
    return normalize_heading(new_heading - old_heading);
}

static float idm_signed_speed(Entity *agent) {
    float speed = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
    float dot = agent->vx * agent->heading_x + agent->vy * agent->heading_y;
    return copysignf(speed, dot);
}

static IDMLeader idm_no_leader(void) {
    IDMLeader leader = {0};
    leader.leader_agent_idx = -1;
    leader.gap = INFINITY;
    return leader;
}

static void idm_update_best_leader(IDMLeader *best, int leader_agent_idx, float gap, float leader_speed) {
    if (gap < 0.0f) {
        gap = IDM_MINIMUM_LEAD_DISTANCE;
    }
    if (gap >= best->gap) {
        return;
    }
    best->has_leader = 1;
    best->leader_agent_idx = leader_agent_idx;
    best->gap = fmaxf(gap, IDM_MINIMUM_LEAD_DISTANCE);
    best->leader_speed = fmaxf(0.0f, leader_speed);
}

static void idm_point_to_ego_frame(Entity *ego, float x, float y, float *out_x, float *out_y) {
    float dx = x - ego->x;
    float dy = y - ego->y;
    *out_x = dx * ego->heading_x + dy * ego->heading_y;
    *out_y = -dx * ego->heading_y + dy * ego->heading_x;
}

static void idm_require_map(Drive *env, int agent_idx) {
    if (env->idm_map == NULL || !env->idm_map->loaded) {
        idm_die("IDM controller requested but map has no IDM extension");
    }
    if (agent_idx < 0 || agent_idx >= env->idm_map->num_agents) {
        idm_die("IDM controller requested for an agent missing from the IDM extension");
    }
    if (env->idm_map->route_lengths[agent_idx] <= 0 || env->idm_map->routes[agent_idx] == NULL) {
        char msg[256];
        snprintf(msg, sizeof(msg), "IDM controller requested for agent %d with no route in the IDM extension",
                 agent_idx);
        idm_die(msg);
    }
}

static IDMAgentState *idm_get_agent_state(Drive *env, int agent_idx) {
    idm_require_map(env, agent_idx);
    if (env->idm_agent_states == NULL) {
        env->idm_agent_states = (IDMAgentState *)calloc(env->num_entities, sizeof(IDMAgentState));
        if (env->idm_agent_states == NULL) {
            idm_die("failed to allocate IDM agent states");
        }
    }
    return &env->idm_agent_states[agent_idx];
}

static float idm_lane_segment_length(IDMRoadElement *lane, int seg_idx) {
    float dx = lane->x[seg_idx + 1] - lane->x[seg_idx];
    float dy = lane->y[seg_idx + 1] - lane->y[seg_idx];
    float dz = lane->z[seg_idx + 1] - lane->z[seg_idx];
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

static IDMLaneProjection idm_project_to_route(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    IDMLaneProjection best = {0};
    best.lane_idx = -1;
    best.dist_sq = INFINITY;

    idm_require_map(env, agent_idx);
    int route_length = env->idm_map->route_lengths[agent_idx];
    int *route = env->idm_map->routes[agent_idx];

    for (int route_idx = 0; route_idx < route_length; route_idx++) {
        int lane_idx = route[route_idx];
        if (lane_idx < 0 || lane_idx >= env->idm_map->num_roads) {
            idm_die("IDM route references an invalid lane index");
        }
        IDMRoadElement *lane = &env->idm_map->roads[lane_idx];
        if (!idm_is_lane_type(lane->type) || lane->segment_length < 2) {
            idm_die("IDM route references a non-lane or degenerate lane");
        }

        for (int seg_idx = 0; seg_idx < lane->segment_length - 1; seg_idx++) {
            float dx = lane->x[seg_idx + 1] - lane->x[seg_idx];
            float dy = lane->y[seg_idx + 1] - lane->y[seg_idx];
            float dz = lane->z[seg_idx + 1] - lane->z[seg_idx];
            float seg_len_sq = dx * dx + dy * dy + dz * dz;
            if (seg_len_sq < 1e-6f) {
                continue;
            }
            float ax = agent->x - lane->x[seg_idx];
            float ay = agent->y - lane->y[seg_idx];
            float az = agent->z - lane->z[seg_idx];
            float t = (ax * dx + ay * dy + az * dz) / seg_len_sq;
            t = clip(t, 0.0f, 1.0f);
            float px = lane->x[seg_idx] + t * dx;
            float py = lane->y[seg_idx] + t * dy;
            float pz = lane->z[seg_idx] + t * dz;
            float ex = agent->x - px;
            float ey = agent->y - py;
            float ez = agent->z - pz;
            float dist_sq = ex * ex + ey * ey + ez * ez;
            if (dist_sq < best.dist_sq) {
                best.valid = 1;
                best.route_idx = route_idx;
                best.lane_idx = lane_idx;
                best.segment_idx = seg_idx;
                best.t = t;
                best.dist_sq = dist_sq;
            }
        }
    }

    return best;
}

static void idm_set_pose_on_lane(Drive *env, int agent_idx, IDMAgentState *state, int route_idx, int lane_idx,
                                 int seg_idx, float t, float *old_heading_out) {
    Entity *agent = &env->entities[agent_idx];
    IDMRoadElement *lane = &env->idm_map->roads[lane_idx];
    if (old_heading_out != NULL) {
        *old_heading_out = agent->heading;
    }
    t = clip(t, 0.0f, 1.0f);
    agent->x = lane->x[seg_idx] + t * (lane->x[seg_idx + 1] - lane->x[seg_idx]);
    agent->y = lane->y[seg_idx] + t * (lane->y[seg_idx + 1] - lane->y[seg_idx]);
    agent->z = lane->z[seg_idx] + t * (lane->z[seg_idx + 1] - lane->z[seg_idx]);
    agent->heading = normalize_heading(lane->headings[seg_idx]);
    agent->heading_x = cosf(agent->heading);
    agent->heading_y = sinf(agent->heading);

    state->initialized = 1;
    state->route_idx = route_idx;
    state->lane_idx = lane_idx;
    state->segment_idx = seg_idx;
    state->t = t;
}

static void idm_initialize_agent_state(Drive *env, int agent_idx) {
    IDMAgentState *state = idm_get_agent_state(env, agent_idx);
    if (state->initialized) {
        return;
    }
    IDMLaneProjection projection = idm_project_to_route(env, agent_idx);
    if (!projection.valid) {
        idm_die("failed to project IDM agent onto its route");
    }
    float old_heading = 0.0f;
    idm_set_pose_on_lane(env, agent_idx, state, projection.route_idx, projection.lane_idx, projection.segment_idx,
                         projection.t, &old_heading);
}

static int idm_advance_along_route(Drive *env, int agent_idx, float distance, float *old_heading_out) {
    idm_initialize_agent_state(env, agent_idx);
    IDMAgentState *state = &env->idm_agent_states[agent_idx];
    int route_length = env->idm_map->route_lengths[agent_idx];
    int *route = env->idm_map->routes[agent_idx];

    if (distance <= 0.0f) {
        if (old_heading_out != NULL) {
            *old_heading_out = env->entities[agent_idx].heading;
        }
        return 1;
    }

    int route_idx = state->route_idx;
    int seg_idx = state->segment_idx;
    float t = state->t;

    while (route_idx < route_length) {
        int lane_idx = route[route_idx];
        if (lane_idx < 0 || lane_idx >= env->idm_map->num_roads) {
            idm_die("IDM route references an invalid lane while advancing");
        }
        IDMRoadElement *lane = &env->idm_map->roads[lane_idx];
        if (lane->segment_length < 2) {
            idm_die("IDM route references a degenerate lane while advancing");
        }

        while (seg_idx < lane->segment_length - 1) {
            float seg_len = idm_lane_segment_length(lane, seg_idx);
            if (seg_len < 1e-6f) {
                seg_idx++;
                t = 0.0f;
                continue;
            }
            float remaining = (1.0f - t) * seg_len;
            if (distance <= remaining) {
                float next_t = t + distance / seg_len;
                idm_set_pose_on_lane(env, agent_idx, state, route_idx, lane_idx, seg_idx, next_t, old_heading_out);
                return 1;
            }
            distance -= remaining;
            seg_idx++;
            t = 0.0f;
        }
        route_idx++;
        seg_idx = 0;
        t = 0.0f;
    }

    return 0;
}

static void idm_consider_agent_leader(Drive *env, int ego_idx, int other_idx, float corridor_start, float corridor_end,
                                      float corridor_half_width, IDMLeader *best) {
    if (ego_idx == other_idx) {
        return;
    }
    Entity *ego = &env->entities[ego_idx];
    Entity *other = &env->entities[other_idx];
    if (other->removed || other->x == INVALID_POSITION || other->valid == 0) {
        return;
    }
    float max_z_gap = 0.5f * ego->height + 0.5f * other->height + 1.0f;
    if (fabsf(other->z - ego->z) > max_z_gap) {
        return;
    }

    float half_length = 0.5f * other->length + IDM_BBOX_MARGIN;
    float half_width = 0.5f * other->width + IDM_BBOX_MARGIN;
    float min_x = INFINITY;
    float max_x = -INFINITY;
    float min_y = INFINITY;
    float max_y = -INFINITY;
    for (int i = 0; i < 4; i++) {
        float corner_x =
            other->x + offsets[i][0] * half_length * other->heading_x - offsets[i][1] * half_width * other->heading_y;
        float corner_y =
            other->y + offsets[i][0] * half_length * other->heading_y + offsets[i][1] * half_width * other->heading_x;
        float rel_x;
        float rel_y;
        idm_point_to_ego_frame(ego, corner_x, corner_y, &rel_x, &rel_y);
        min_x = fminf(min_x, rel_x);
        max_x = fmaxf(max_x, rel_x);
        min_y = fminf(min_y, rel_y);
        max_y = fmaxf(max_y, rel_y);
    }
    if (max_x < corridor_start || min_x > corridor_end) {
        return;
    }
    if (max_y < -corridor_half_width || min_y > corridor_half_width) {
        return;
    }

    float gap = min_x - corridor_start;
    float leader_speed = other->vx * ego->heading_x + other->vy * ego->heading_y;
    idm_update_best_leader(best, other_idx, gap, leader_speed);
}

static IDMLeader idm_find_leader_by_corridor(Drive *env, int ego_idx) {
    Entity *ego = &env->entities[ego_idx];
    IDMLeader best = idm_no_leader();

    float speed = fmaxf(0.0f, idm_signed_speed(ego));
    float lookahead = clip(speed * IDM_LOOKAHEAD_TIME, IDM_MIN_LOOKAHEAD, IDM_MAX_LOOKAHEAD);
    float corridor_start = 0.5f * ego->length + IDM_BBOX_MARGIN;
    float corridor_end = corridor_start + lookahead;
    float corridor_half_width = 0.5f * ego->width + IDM_BBOX_MARGIN;

    for (int i = 0; i < env->active_agent_count; i++) {
        idm_consider_agent_leader(env, ego_idx, env->active_agent_indices[i], corridor_start, corridor_end,
                                  corridor_half_width, &best);
    }
    for (int i = 0; i < env->static_agent_count; i++) {
        idm_consider_agent_leader(env, ego_idx, env->static_agent_indices[i], corridor_start, corridor_end,
                                  corridor_half_width, &best);
    }
    return best;
}

static void idm_set_sample_pose(Entity *sample, IDMRoadElement *lane, int seg_idx, float t) {
    t = clip(t, 0.0f, 1.0f);
    sample->x = lane->x[seg_idx] + t * (lane->x[seg_idx + 1] - lane->x[seg_idx]);
    sample->y = lane->y[seg_idx] + t * (lane->y[seg_idx + 1] - lane->y[seg_idx]);
    sample->z = lane->z[seg_idx] + t * (lane->z[seg_idx + 1] - lane->z[seg_idx]);
    sample->heading = normalize_heading(lane->headings[seg_idx]);
    sample->heading_x = cosf(sample->heading);
    sample->heading_y = sinf(sample->heading);
}

static int idm_sample_hits_agent(Entity *sample, Entity *other) {
    if (other->removed || other->x == INVALID_POSITION || other->valid == 0) {
        return 0;
    }
    float max_z_gap = 0.5f * sample->height + 0.5f * other->height + 1.0f;
    if (fabsf(other->z - sample->z) > max_z_gap) {
        return 0;
    }

    float dx = other->x - sample->x;
    float dy = other->y - sample->y;
    float local_radius =
        0.5f * sample->length + 0.5f * other->length + sample->width + other->width + 1.0f + 2.0f * IDM_BBOX_MARGIN;
    if (dx * dx + dy * dy > local_radius * local_radius) {
        return 0;
    }

    Entity expanded_other = *other;
    expanded_other.length = other->length + 2.0f * IDM_BBOX_MARGIN;
    expanded_other.width = other->width + 2.0f * IDM_BBOX_MARGIN;
    return check_aabb_collision(sample, &expanded_other);
}

static void idm_consider_route_sample_leader(Drive *env, int ego_idx, int other_idx, Entity *sample, float gap,
                                             IDMLeader *best) {
    if (other_idx == ego_idx) {
        return;
    }
    Entity *other = &env->entities[other_idx];
    if (!idm_sample_hits_agent(sample, other)) {
        return;
    }
    float leader_speed = other->vx * sample->heading_x + other->vy * sample->heading_y;
    idm_update_best_leader(best, other_idx, gap, leader_speed);
}

static IDMLeader idm_find_leader_by_route(Drive *env, int ego_idx) {
    Entity *ego = &env->entities[ego_idx];
    IDMLeader best = idm_no_leader();

    idm_initialize_agent_state(env, ego_idx);
    IDMAgentState state = env->idm_agent_states[ego_idx];
    int route_length = env->idm_map->route_lengths[ego_idx];
    int *route = env->idm_map->routes[ego_idx];

    float speed = fmaxf(0.0f, idm_signed_speed(ego));
    float lookahead = clip(speed * IDM_LOOKAHEAD_TIME, IDM_MIN_LOOKAHEAD, IDM_MAX_LOOKAHEAD);
    float next_sample_s = IDM_ROUTE_SAMPLE_DS;
    float traveled_s = 0.0f;
    int route_idx = state.route_idx;
    int seg_idx = state.segment_idx;
    float t = state.t;

    Entity sample = *ego;
    sample.length = ego->length + 2.0f * IDM_BBOX_MARGIN;
    sample.width = ego->width + 2.0f * IDM_BBOX_MARGIN;
    sample.removed = 0;
    sample.valid = 1;

    while (route_idx < route_length && next_sample_s <= lookahead + 1e-4f) {
        int lane_idx = route[route_idx];
        if (lane_idx < 0 || lane_idx >= env->idm_map->num_roads) {
            idm_die("IDM route references an invalid lane while finding route leader");
        }
        IDMRoadElement *lane = &env->idm_map->roads[lane_idx];
        if (!idm_is_lane_type(lane->type) || lane->segment_length < 2) {
            idm_die("IDM route references a non-lane or degenerate lane while finding route leader");
        }

        while (seg_idx < lane->segment_length - 1 && next_sample_s <= lookahead + 1e-4f) {
            float seg_len = idm_lane_segment_length(lane, seg_idx);
            if (seg_len < 1e-6f) {
                seg_idx++;
                t = 0.0f;
                continue;
            }

            float remaining = (1.0f - t) * seg_len;
            if (traveled_s + remaining + 1e-4f < next_sample_s) {
                traveled_s += remaining;
                seg_idx++;
                t = 0.0f;
                continue;
            }

            float sample_t = t + (next_sample_s - traveled_s) / seg_len;
            idm_set_sample_pose(&sample, lane, seg_idx, sample_t);

            for (int i = 0; i < env->active_agent_count; i++) {
                idm_consider_route_sample_leader(env, ego_idx, env->active_agent_indices[i], &sample, next_sample_s,
                                                 &best);
            }
            for (int i = 0; i < env->static_agent_count; i++) {
                idm_consider_route_sample_leader(env, ego_idx, env->static_agent_indices[i], &sample, next_sample_s,
                                                 &best);
            }
            if (best.has_leader) {
                return best;
            }

            next_sample_s += IDM_ROUTE_SAMPLE_DS;
        }

        route_idx++;
        seg_idx = 0;
        t = 0.0f;
    }

    return best;
}

static float idm_desired_speed(Drive *env, int agent_idx) {
    idm_initialize_agent_state(env, agent_idx);
    IDMAgentState *state = &env->idm_agent_states[agent_idx];
    if (state->lane_idx >= 0 && state->lane_idx < env->idm_map->num_roads) {
        float speed_limit = env->idm_map->roads[state->lane_idx].speed_limit;
        if (speed_limit > 0.0f && isfinite(speed_limit)) {
            return clip(speed_limit, 1.0f, MAX_SPEED);
        }
    }
    return IDM_DEFAULT_DESIRED_SPEED;
}

static float idm_compute_acceleration(Drive *env, int agent_idx, IDMLeader leader) {
    Entity *agent = &env->entities[agent_idx];
    float current_speed = fmaxf(0.0f, idm_signed_speed(agent));
    float desired_speed = idm_desired_speed(env, agent_idx);
    float speed_ratio = current_speed / desired_speed;
    float free_road_term = powf(speed_ratio, IDM_DELTA);
    float leader_term = 0.0f;

    if (leader.has_leader) {
        float s_star = IDM_MIN_SPACING + fmaxf(0.0f, current_speed * IDM_SAFE_TIME_HEADWAY +
                                                         current_speed * (current_speed - leader.leader_speed) /
                                                             (2.0f * sqrtf(ACCEL_MAX * (-ACCEL_MIN))));
        float lead_dist = fmaxf(leader.gap, IDM_MINIMUM_LEAD_DISTANCE);
        leader_term = (s_star / lead_dist) * (s_star / lead_dist);
    }

    return ACCEL_MAX * (1.0f - free_road_term - leader_term);
}

static void move_idm(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    if (agent->removed || agent->x == INVALID_POSITION) {
        return;
    }
    if (agent->stopped) {
        agent->vx = 0.0f;
        agent->vy = 0.0f;
        agent->vz = 0.0f;
        return;
    }

    idm_initialize_agent_state(env, agent_idx);
    IDMLeader leader = idm_find_leader_by_route(env, agent_idx);
    float old_a_long = agent->a_long;
    float current_speed = fmaxf(0.0f, idm_signed_speed(agent));
    float accel = clip(idm_compute_acceleration(env, agent_idx, leader), ACCEL_MIN, ACCEL_MAX);
    float new_speed = current_speed + accel * env->dt;
    if (new_speed < 0.0f) {
        new_speed = 0.0f;
    }
    accel = (new_speed - current_speed) / env->dt;

    float old_heading = agent->heading;
    int advanced = idm_advance_along_route(env, agent_idx, new_speed * env->dt, &old_heading);
    if (!advanced) {
        agent->stopped = 1;
        new_speed = 0.0f;
        accel = (new_speed - current_speed) / env->dt;
    }

    agent->vx = new_speed * agent->heading_x;
    agent->vy = new_speed * agent->heading_y;
    agent->vz = 0.0f;
    float yaw_rate = idm_heading_diff(agent->heading, old_heading) / env->dt;
    float new_a_lat = new_speed * yaw_rate;
    agent->jerk_long = (accel - old_a_long) / env->dt;
    agent->jerk_lat = (new_a_lat - agent->a_lat) / env->dt;
    agent->a_long = accel;
    agent->a_lat = new_a_lat;
    agent->steering_angle = 0.0f;
}

static void move_corridor_idm(Drive *env, int agent_idx) {
    Entity *agent = &env->entities[agent_idx];
    if (agent->removed || agent->x == INVALID_POSITION) {
        return;
    }
    if (agent->stopped) {
        agent->vx = 0.0f;
        agent->vy = 0.0f;
        agent->vz = 0.0f;
        return;
    }

    idm_initialize_agent_state(env, agent_idx);
    IDMLeader leader = idm_find_leader_by_corridor(env, agent_idx);
    float old_a_long = agent->a_long;
    float current_speed = fmaxf(0.0f, idm_signed_speed(agent));
    float accel = clip(idm_compute_acceleration(env, agent_idx, leader), ACCEL_MIN, ACCEL_MAX);
    float new_speed = current_speed + accel * env->dt;
    if (new_speed < 0.0f) {
        new_speed = 0.0f;
    }
    accel = (new_speed - current_speed) / env->dt;

    float old_heading = agent->heading;
    int advanced = idm_advance_along_route(env, agent_idx, new_speed * env->dt, &old_heading);
    if (!advanced) {
        agent->stopped = 1;
        new_speed = 0.0f;
        accel = (new_speed - current_speed) / env->dt;
    }

    agent->vx = new_speed * agent->heading_x;
    agent->vy = new_speed * agent->heading_y;
    agent->vz = 0.0f;
    float yaw_rate = idm_heading_diff(agent->heading, old_heading) / env->dt;
    float new_a_lat = new_speed * yaw_rate;
    agent->jerk_long = (accel - old_a_long) / env->dt;
    agent->jerk_lat = (new_a_lat - agent->a_lat) / env->dt;
    agent->a_long = accel;
    agent->a_lat = new_a_lat;
    agent->steering_angle = 0.0f;
}

#endif
