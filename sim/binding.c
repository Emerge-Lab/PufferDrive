#include "drive.h"

#define OBS_TENSOR_T FloatTensor

#define STATIC_VEC_IMPL
#define MY_VEC_INIT
#define Env Drive
#include "env_fields.h"
#include "vecenv.h"

static inline void apply_env_kwargs(Env *env, Dict *kwargs) {
#define APPLY(type, name) env->name = (type) dict_get(kwargs, #name)->value;
    ENV_FIELDS(APPLY)
#undef APPLY
}

Env *my_vec_init(
    int *num_envs_out,
    int *buffer_env_starts,
    int *buffer_env_counts,
    Dict *vec_kwargs,
    Dict *env_kwargs) {
    int total_agents = (int) dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int) dict_get(vec_kwargs, "num_buffers")->value;
    int num_maps = (int) dict_get(env_kwargs, "num_maps")->value;
    int agents_per_buffer = total_agents / num_buffers;
    const char *map_binary_dir = dict_get_str(env_kwargs, "map_binary_dir");

    int discovered_maps = 0;
    char **map_files = discover_map_files(map_binary_dir, &discovered_maps);
    if (!map_files) {
        printf("ERROR: Cannot find .bin files at %s/\n", map_binary_dir);
        *num_envs_out = 0;
        return NULL;
    }

    int map_count = num_maps < discovered_maps ? num_maps : discovered_maps;
    if (map_count <= 0) {
        printf(
            "ERROR: No map files selected from %s/ (discovered %d, num_maps %d)\n",
            map_binary_dir,
            discovered_maps,
            num_maps);
        free_map_files(map_files, discovered_maps);
        *num_envs_out = 0;
        return NULL;
    }

    // Scan all maps for agent counts; collect valid (>0) ones
    int *agents_per_map = (int *) malloc(map_count * sizeof(int));
    int *valid_map_ids = (int *) malloc(map_count * sizeof(int));
    int num_valid_maps = 0;
    for (int m = 0; m < map_count; m++) {
        Env temp_env = {0};
        temp_env.map_name = map_files[m];
        apply_env_kwargs(&temp_env, env_kwargs);
        if (init(&temp_env) != 0) {
            agents_per_map[m] = 0;
            c_close(&temp_env);
            continue;
        }
        agents_per_map[m] = temp_env.num_agents;
        c_close(&temp_env);
        if (agents_per_map[m] > 0) {
            valid_map_ids[num_valid_maps++] = m;
        }
    }
    printf(
        "Scanned %d map binaries from %s/ (%d discovered), %d valid\n",
        map_count,
        map_binary_dir,
        discovered_maps,
        num_valid_maps);

    if (num_valid_maps == 0) {
        printf("ERROR: No valid maps found\n");
        free(agents_per_map);
        free(valid_map_ids);
        free_map_files(map_files, discovered_maps);
        *num_envs_out = 0;
        return NULL;
    }

    // Build per-env layout. Each buffer advances the global cursor so different
    // buffers get different maps. If the next full map would overflow a buffer,
    // pack remaining slots with 1-agent envs advancing the cursor each time.
    int max_envs = agents_per_buffer * num_buffers; // upper bound (all 1-agent)
    int *env_map_ids = (int *) malloc(max_envs * sizeof(int));
    int *env_max_agents = (int *) malloc(max_envs * sizeof(int));
    int total_envs = 0;
    int cursor = 0; // advances across buffers

    for (int b = 0; b < num_buffers; b++) {
        buffer_env_starts[b] = total_envs;
        int buffer_agents = 0;
        while (buffer_agents < agents_per_buffer) {
            int m = valid_map_ids[cursor % num_valid_maps];
            int cap = agents_per_map[m];
            int remaining = agents_per_buffer - buffer_agents;
            if (cap <= remaining) {
                // Full map fits
                env_map_ids[total_envs] = m;
                env_max_agents[total_envs] = cap;
                buffer_agents += cap;
                total_envs++;
                cursor++;
            } else {
                // Pack remaining slots as 1-agent envs, one map each
                while (buffer_agents < agents_per_buffer) {
                    int mm = valid_map_ids[cursor % num_valid_maps];
                    env_map_ids[total_envs] = mm;
                    env_max_agents[total_envs] = 1;
                    buffer_agents++;
                    total_envs++;
                    cursor++;
                }
            }
        }
        buffer_env_counts[b] = total_envs - buffer_env_starts[b];
    }

    printf("total envs: %d (%d maps cycled)\n", total_envs, cursor);

    // Initialize all envs
    Env *envs = (Env *) calloc(total_envs, sizeof(Env));
    for (int i = 0; i < total_envs; i++) {
        Env *env = &envs[i];
        memset(env, 0, sizeof(Env));
        env->map_name = strdup(map_files[env_map_ids[i]]);
        apply_env_kwargs(env, env_kwargs);
        env->num_max_agents = env_max_agents[i];
        if (init(env) != 0) {
            printf("ERROR: Failed to initialize map %s\n", map_files[env_map_ids[i]]);
            for (int j = 0; j < i; j++) {
                c_close(&envs[j]);
            }
            free(envs);
            free(env_map_ids);
            free(env_max_agents);
            free(agents_per_map);
            free(valid_map_ids);
            free_map_files(map_files, discovered_maps);
            *num_envs_out = 0;
            return NULL;
        }
    }

    free(env_map_ids);
    free(env_max_agents);
    free(agents_per_map);
    free(valid_map_ids);
    free_map_files(map_files, discovered_maps);

    printf("Created %d envs, %d total agents (target %d)\n", total_envs, total_agents, total_agents);

    *num_envs_out = total_envs;
    return envs;
}

void my_init(Env *env, Dict *kwargs) {
    apply_env_kwargs(env, kwargs);
    int map_id = dict_get(kwargs, "map_id")->value;
    int num_max_active_agents = dict_get(kwargs, "num_max_active_agents")->value;
    int num_maps = (int) dict_get(kwargs, "num_maps")->value;

    const char *map_binary_dir = dict_get_str(kwargs, "map_binary_dir");

    int discovered_maps = 0;
    char **map_files = discover_map_files(map_binary_dir, &discovered_maps);
    int map_count = num_maps < discovered_maps ? num_maps : discovered_maps;
    if (!map_files || map_id < 0 || map_id >= map_count) {
        printf("ERROR: Invalid map_id %d for %d selected .bin files in %s/\n", map_id, map_count, map_binary_dir);
        free_map_files(map_files, discovered_maps);
        return;
    }

    env->num_max_agents = num_max_active_agents;
    env->map_name = strdup(map_files[map_id]);
    free_map_files(map_files, discovered_maps);
    if (init(env) != 0) {
        printf("ERROR: Failed to initialize map_id %d\n", map_id);
    }
}

void my_log(Log *log, Dict *out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "offroad_rate", log->offroad_rate);
    dict_set(out, "collision_rate", log->collision_rate);
    dict_set(out, "num_goals_reached", log->num_goals_reached);
    dict_set(out, "avg_speed_per_agent", log->avg_speed_per_agent);
    dict_set(out, "dnf_rate", log->dnf_rate);
    dict_set(out, "n", log->n);
}

void my_env_constants(void *env_ptr, Dict *out) {
    Drive *env = (Drive *) env_ptr;
    dict_set(out, "ego_features", EGO_FEATURES);
    dict_set(out, "target_features", STATIC_TARGET_FEATURES);
    dict_set(out, "partner_features", PARTNER_FEATURES);
    dict_set(out, "road_features", ROAD_FEATURES);
    dict_set(out, "traffic_control_features", TRAFFIC_CONTROL_FEATURES);
    dict_set(out, "obs_partner_slots", env->obs_slots_partners);
    dict_set(out, "obs_lane_slots", env->obs_slots_lane);
    dict_set(out, "obs_boundary_slots", env->obs_slots_boundary);
    dict_set(out, "obs_traffic_control_slots", env->obs_slots_traffic_controls);
    dict_set(out, "obs_count_features", OBS_COUNT_FEATURES);
    dict_set(out, "num_target_waypoints", env->num_target_waypoints);
    dict_set(out, "num_reward_coefs", 0);
}
