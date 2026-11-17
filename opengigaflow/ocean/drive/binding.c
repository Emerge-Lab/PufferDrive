#include "drive.h"

#include <Python.h>

static PyObject *map_cache_preload_py(PyObject *self __attribute__((unused)), PyObject *args, PyObject *kwargs);
static PyObject *map_cache_release_py(PyObject *self __attribute__((unused)), PyObject *args, PyObject *kwargs);

// clang-format off
#define MY_METHODS \
    {"map_cache_preload", (PyCFunction) map_cache_preload_py, METH_VARARGS | METH_KEYWORDS, \
     "Build and retain configured map geometry for forked workers."}, \
    {"map_cache_release", (PyCFunction) map_cache_release_py, METH_VARARGS | METH_KEYWORDS, \
     "Release map geometry retained for forked workers."}
// clang-format on

#include "../env_binding.h"

// Seeds are 63-bit non-negative so they survive int64 round-trips (numpy, pandas, CSV).
static int unpack_seed(PyObject *kwargs, uint64_t *seed_out) {
    // a failed unpack() leaves its exception pending: report that one instead of masking it
    if (PyErr_Occurred()) {
        return -1;
    }
    PyObject *seed_obj = PyDict_GetItemString(kwargs, "seed");
    if (seed_obj == NULL || !PyLong_Check(seed_obj)) {
        PyErr_SetString(PyExc_TypeError, "Missing or non-integer keyword argument 'seed'");
        return -1;
    }
    long long seed = PyLong_AsLongLong(seed_obj);
    if (seed < 0) {
        PyErr_SetString(PyExc_ValueError, "seed must be a non-negative 63-bit integer");
        return -1;
    }
    *seed_out = (uint64_t) seed;
    return 0;
}

static void my_reseed(Drive *env, uint64_t seed) {
    env->init_seed = seed;
    rng_seed(&env->seed_stream_rng, seed);
    // Defeat c_reset's timestep==0 fast path so the new stream regenerates the episode
    env->timestep = -1;
}

// get_state() exports arrays at or above these lengths as empty
#define STATE_TRAJECTORY_LEN_LIMIT 10000
#define STATE_SEGMENT_LEN_LIMIT 100000

// Steals value; a NULL value from a failed constructor passes its pending error on.
static int set_dict_item(PyObject *dict, const char *key, PyObject *value) {
    if (value == NULL) {
        return -1;
    }
    int status = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    return status;
}

// None for a missing or empty array.
static int set_float_list(PyObject *dict, const char *key, const float *values, int count) {
    if (values == NULL || count <= 0) {
        return PyDict_SetItemString(dict, key, Py_None);
    }
    PyObject *list = PyList_New(count);
    if (list == NULL) {
        return -1;
    }
    for (int i = 0; i < count; i++) {
        PyObject *item = PyFloat_FromDouble((double) values[i]);
        if (item == NULL) {
            Py_DECREF(list);
            return -1;
        }
        PyList_SET_ITEM(list, i, item);
    }
    return set_dict_item(dict, key, list);
}

static int set_int_list(PyObject *dict, const char *key, const int *values, int count) {
    if (values == NULL || count <= 0) {
        return PyDict_SetItemString(dict, key, Py_None);
    }
    PyObject *list = PyList_New(count);
    if (list == NULL) {
        return -1;
    }
    for (int i = 0; i < count; i++) {
        PyObject *item = PyLong_FromLong(values[i]);
        if (item == NULL) {
            Py_DECREF(list);
            return -1;
        }
        PyList_SET_ITEM(list, i, item);
    }
    return set_dict_item(dict, key, list);
}

static PyObject *agent_state_list(const Drive *env) {
    if (env->agents == NULL || env->num_total_agents <= 0) {
        Py_RETURN_NONE;
    }
    PyObject *agent_list = PyList_New(env->num_total_agents);
    if (agent_list == NULL) {
        return NULL;
    }
    for (int i = 0; i < env->num_total_agents; i++) {
        const Agent *agent = &env->agents[i];
        int trajectory_size = agent->trajectory_size;
        trajectory_size = trajectory_size > 0 && trajectory_size < STATE_TRAJECTORY_LEN_LIMIT ? trajectory_size : 0;
        PyObject *agent_dict = PyDict_New();
        PyList_SET_ITEM(agent_list, i, agent_dict);
        if (agent_dict == NULL || set_dict_item(agent_dict, "trajectory_size", PyLong_FromLong(trajectory_size)) < 0
            || set_float_list(agent_dict, "log_trajectory_x", agent->log_trajectory_x, trajectory_size) < 0
            || set_float_list(agent_dict, "log_trajectory_y", agent->log_trajectory_y, trajectory_size) < 0
            || set_float_list(agent_dict, "log_heading", agent->log_heading, trajectory_size) < 0
            || set_int_list(agent_dict, "log_valid", agent->log_valid, trajectory_size) < 0
            || set_dict_item(agent_dict, "sim_length", PyFloat_FromDouble((double) agent->sim_length)) < 0
            || set_dict_item(agent_dict, "sim_width", PyFloat_FromDouble((double) agent->sim_width)) < 0
            || set_dict_item(agent_dict, "current_goal_x", PyFloat_FromDouble((double) agent->current_goal_x)) < 0
            || set_dict_item(agent_dict, "current_goal_y", PyFloat_FromDouble((double) agent->current_goal_y)) < 0
            || set_dict_item(agent_dict, "mark_as_expert", PyLong_FromLong(agent->mark_as_expert)) < 0
            || set_float_list(agent_dict, "metrics_array", agent->metrics_array, NUM_METRICS) < 0) {
            Py_DECREF(agent_list);
            return NULL;
        }
    }
    return agent_list;
}

static PyObject *road_state_list(const Drive *env) {
    if (env->road_elements == NULL || env->num_road_elements <= 0) {
        Py_RETURN_NONE;
    }
    PyObject *road_list = PyList_New(env->num_road_elements);
    if (road_list == NULL) {
        return NULL;
    }
    for (int i = 0; i < env->num_road_elements; i++) {
        const RoadMapElement *road = &env->road_elements[i];
        int point_count = road->segment_size;
        point_count = point_count > 0 && point_count < STATE_SEGMENT_LEN_LIMIT ? point_count : 0;
        PyObject *road_dict = PyDict_New();
        PyList_SET_ITEM(road_list, i, road_dict);
        if (road_dict == NULL || set_dict_item(road_dict, "id", PyLong_FromLong(i)) < 0
            || set_dict_item(road_dict, "type", PyLong_FromLong(road->type)) < 0
            || set_float_list(road_dict, "x", road->x, point_count) < 0
            || set_float_list(road_dict, "y", road->y, point_count) < 0
            || set_float_list(road_dict, "z", road->z, point_count) < 0) {
            Py_DECREF(road_list);
            return NULL;
        }
    }
    return road_list;
}

static PyObject *traffic_state_list(const Drive *env) {
    if (env->traffic_elements == NULL || env->num_traffic_elements <= 0) {
        Py_RETURN_NONE;
    }
    PyObject *traffic_list = PyList_New(env->num_traffic_elements);
    if (traffic_list == NULL) {
        return NULL;
    }
    for (int i = 0; i < env->num_traffic_elements; i++) {
        const TrafficControlElement *traffic = &env->traffic_elements[i];
        int stop_line_float_count = (int) (sizeof(traffic->stop_line) / sizeof(float));
        PyObject *traffic_dict = PyDict_New();
        PyList_SET_ITEM(traffic_list, i, traffic_dict);
        if (traffic_dict == NULL || set_dict_item(traffic_dict, "type", PyLong_FromLong(traffic->type)) < 0
            || set_float_list(traffic_dict, "stop_line", traffic->stop_line, stop_line_float_count) < 0) {
            Py_DECREF(traffic_list);
            return NULL;
        }
    }
    return traffic_list;
}

// Only the keys some reader uses: the replay viewer, the co-sim obs replay and the tests.
static PyObject *my_get(Drive *env) {
    const char *scenario_id = env->scenario_id[0] != '\0' ? env->scenario_id : NULL;
    PyObject *state = PyDict_New();
    if (state == NULL || set_dict_item(state, "active_agent_count", PyLong_FromLong(env->active_agent_count)) < 0
        || set_dict_item(state, "num_total_agents", PyLong_FromLong(env->num_total_agents)) < 0
        || set_dict_item(state, "num_traffic_elements", PyLong_FromLong(env->num_traffic_elements)) < 0
        || set_dict_item(state, "map_name", Py_BuildValue("z", env->map_name)) < 0
        || set_dict_item(state, "scenario_id", Py_BuildValue("z", scenario_id)) < 0
        || set_int_list(state, "active_agent_indices", env->active_agent_indices, env->active_agent_count) < 0
        || set_dict_item(state, "agents", agent_state_list(env)) < 0
        || set_dict_item(state, "road_elements", road_state_list(env)) < 0
        || set_dict_item(state, "traffic_elements", traffic_state_list(env)) < 0) {
        Py_XDECREF(state);
        return NULL;
    }
    return state;
}

// Releases a REPLAY probe env of my_shared: its loaded map data and agent index lists.
static void free_probe_env(Drive *env) {
    free_loaded_map_data(env);
    free(env->active_agent_indices);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    free(env);
}

static PyObject *my_shared(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *map_files = PyDict_GetItemString(kwargs, "map_files");
    if (!map_files || !PyList_Check(map_files)) {
        PyErr_SetString(PyExc_TypeError, "map_files must be a list of strings");
        return NULL;
    }
    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");
    int starting_map_counter = unpack(kwargs, "starting_map_counter");
    int eval_mode = unpack(kwargs, "eval_mode");
    int eval_training_render = unpack(kwargs, "eval_training_render");
    int s_map_counter = starting_map_counter;
    int init_mode = unpack(kwargs, "init_mode");
    int control_mode = unpack(kwargs, "control_mode");
    int replay_expert_agents = unpack(kwargs, "replay_expert_agents");
    int simulation_mode = unpack(kwargs, "simulation_mode");
    int init_step = unpack(kwargs, "init_step");
    int goal_source = unpack(kwargs, "goal_source");
    float goal_radius = (float) unpack(kwargs, "goal_radius");
    uint64_t seed;
    if (unpack_seed(kwargs, &seed) != 0) {
        return NULL;
    }
    int min_agents_per_env = unpack(kwargs, "min_agents_per_env");
    int max_agents_per_env = unpack(kwargs, "max_agents_per_env");
    int num_eval_scenarios = unpack(kwargs, "num_eval_scenarios");
    if (PyErr_Occurred()) {
        return NULL;
    }
    PyObject *eval_map_indices = PyDict_GetItemString(kwargs, "eval_map_indices");
    int use_eval_map_indices = eval_map_indices != NULL && eval_map_indices != Py_None;
    int eval_target_count = num_eval_scenarios;
    if (use_eval_map_indices) {
        if (!PyList_Check(eval_map_indices)) {
            PyErr_SetString(PyExc_TypeError, "eval_map_indices must be a list of integers");
            return NULL;
        }
        int eval_map_count = (int) PyList_Size(eval_map_indices);
        for (int i = 0; i < eval_map_count; i++) {
            long map_idx = PyLong_AsLong(PyList_GetItem(eval_map_indices, i));
            if (map_idx < 0 || map_idx >= num_maps) {
                PyErr_Format(PyExc_ValueError, "eval_map_indices[%d]=%ld out of range [0, %d)", i, map_idx, num_maps);
                return NULL;
            }
        }
        eval_target_count = eval_map_count < eval_target_count ? eval_map_count : eval_target_count;
    }
    if (min_agents_per_env <= 0 || max_agents_per_env <= 0) {
        PyErr_SetString(PyExc_ValueError, "min_agents_per_env and max_agents_per_env must be > 0");
        return NULL;
    }
    if (min_agents_per_env > max_agents_per_env) {
        PyErr_SetString(PyExc_ValueError, "min_agents_per_env must be <= max_agents_per_env");
        return NULL;
    }
    if (simulation_mode == SIMULATION_MODE_GIGAFLOW && num_agents < min_agents_per_env) {
        PyErr_SetString(PyExc_ValueError, "num_agents must be >= min_agents_per_env");
        return NULL;
    }
    if (eval_training_render && num_agents < max_agents_per_env) {
        PyErr_SetString(PyExc_ValueError, "eval_training_render requires num_agents >= max_agents_per_env");
        return NULL;
    }

    Rng shared_rng;
    rng_seed(&shared_rng, seed);

    if (simulation_mode == SIMULATION_MODE_GIGAFLOW) {
        int environment_limit = eval_mode || eval_training_render ? eval_target_count : num_agents;
        int *agent_counts = malloc((num_agents / min_agents_per_env + 1) * sizeof(int));
        int remaining = num_agents;
        int env_count = 0;
        while (remaining > 0 && env_count < environment_limit) {
            int count;
            if (eval_mode && !eval_training_render) {
                count = max_agents_per_env;
            } else if (eval_training_render) {
                count = min_agents_per_env + rng_below(&shared_rng, max_agents_per_env - min_agents_per_env + 1);
            } else if (remaining <= max_agents_per_env) {
                count = remaining;
            } else {
                // leaves at least min_agents_per_env agents for the envs that follow
                int count_max = remaining - min_agents_per_env < max_agents_per_env ? remaining - min_agents_per_env
                                                                                    : max_agents_per_env;
                count = count_max <= min_agents_per_env
                    ? min_agents_per_env
                    : min_agents_per_env + rng_below(&shared_rng, count_max - min_agents_per_env + 1);
            }
            if (count > remaining) {
                break;
            }
            agent_counts[env_count++] = count;
            remaining -= count;
        }

        PyObject *agent_offsets = PyList_New(env_count + 1);
        PyObject *map_ids = PyList_New(env_count);
        int offset = 0;
        for (int i = 0; i < env_count; i++) {
            int map_id;
            if (eval_mode) {
                map_id = use_eval_map_indices ? (int) PyLong_AsLong(PyList_GetItem(eval_map_indices, i))
                                              : (s_map_counter + i) % num_maps;
            } else {
                map_id = rng_below(&shared_rng, num_maps);
            }
            PyList_SetItem(agent_offsets, i, PyLong_FromLong(offset));
            PyList_SetItem(map_ids, i, PyLong_FromLong(map_id));
            offset += agent_counts[i];
        }
        PyList_SetItem(agent_offsets, env_count, PyLong_FromLong(offset));
        free(agent_counts);
        return Py_BuildValue("(NNii)", agent_offsets, map_ids, env_count, env_count);
    }

    int total_agent_count = 0;
    int env_count = 0;
    int max_envs = eval_mode ? eval_target_count : num_agents;
    int end_map_index = starting_map_counter + eval_target_count;
    PyObject *agent_offsets = PyList_New(max_envs + 1);
    PyObject *map_ids = PyList_New(max_envs);
    // Maps taken or skipped (no controllable agents); a deferred scene is not consumed
    int maps_consumed = 0;
    int eval_map_index_count = use_eval_map_indices ? (int) PyList_Size(eval_map_indices) : 0;

    while (total_agent_count < num_agents && env_count < max_envs
           && (!eval_mode
               || (use_eval_map_indices ? maps_consumed < eval_map_index_count : s_map_counter < end_map_index))) {
        int map_id;
        if (!eval_mode) {
            map_id = rng_below(&shared_rng, num_maps);
        } else if (use_eval_map_indices) {
            map_id = (int) PyLong_AsLong(PyList_GetItem(eval_map_indices, maps_consumed));
        } else {
            map_id = s_map_counter % num_maps;
            s_map_counter += 1;
        }

        const char *map_file = PyUnicode_AsUTF8(PyList_GetItem(map_files, map_id));

        Drive *env = calloc(1, sizeof(Drive));
        env->init_mode = init_mode;
        env->control_mode = control_mode;
        env->replay_expert_agents = replay_expert_agents;
        env->simulation_mode = simulation_mode;
        env->init_step = init_step;
        env->goal_source = goal_source;
        env->goal_radius = goal_radius;
        env->num_max_agents = max_agents_per_env;
        env->eval_mode = eval_mode;
        if (load_map_binary(map_file, env) != 0) {
            PyErr_Format(PyExc_RuntimeError, "Failed to load map binary: %s", map_file);
            free_probe_env(env);
            Py_DECREF(agent_offsets);
            Py_DECREF(map_ids);
            return NULL;
        }
        if (init_grid_map(env) != 0) {
            PyErr_Format(PyExc_RuntimeError, "Failed to build grid map for map: %s", map_file);
            free_probe_env(env);
            Py_DECREF(agent_offsets);
            Py_DECREF(map_ids);
            return NULL;
        }

        set_active_agents(env);
        int active_agent_count = env->active_agent_count;
        free_probe_env(env);

        // Eval leaves a scene that would overflow the buffer to the next batch; training packs greedily.
        if (eval_mode && env_count > 0 && total_agent_count + active_agent_count > num_agents) {
            break;
        }
        maps_consumed++;
        if (active_agent_count == 0) {
            continue;
        }
        PyList_SetItem(map_ids, env_count, PyLong_FromLong(map_id));
        PyList_SetItem(agent_offsets, env_count, PyLong_FromLong(total_agent_count));
        total_agent_count += active_agent_count;
        env_count++;
    }
    if (total_agent_count >= num_agents) {
        total_agent_count = num_agents;
    }
    PyList_SetItem(agent_offsets, env_count, PyLong_FromLong(total_agent_count));
    PyObject *resized_agent_offsets = PyList_GetSlice(agent_offsets, 0, env_count + 1);
    PyObject *resized_map_ids = PyList_GetSlice(map_ids, 0, env_count);
    Py_DECREF(agent_offsets);
    Py_DECREF(map_ids);
    return Py_BuildValue("(NNii)", resized_agent_offsets, resized_map_ids, env_count, maps_consumed);
}

static int unpack_map_cache_args(PyObject *kwargs, Drive *config, const char ***paths, int *num_map_files) {
    PyObject *map_files = PyDict_GetItemString(kwargs, "map_files");
    if (map_files == NULL || !PyList_Check(map_files)) {
        PyErr_SetString(PyExc_TypeError, "map_files must be a list of strings");
        return -1;
    }
    Py_ssize_t map_file_count = PyList_Size(map_files);
    if (map_file_count <= 0 || map_file_count > INT_MAX) {
        PyErr_SetString(PyExc_ValueError, "map_files must contain between 1 and INT_MAX paths");
        return -1;
    }
    *num_map_files = (int) map_file_count;
    config->use_neighbor_cache = (int) unpack(kwargs, "use_neighbor_cache");
    config->obs_lane_stride = (int) unpack(kwargs, "obs_lane_stride");
    config->obs_boundary_stride = (int) unpack(kwargs, "obs_boundary_stride");
    config->obs_stride_merge_segments = (int) unpack(kwargs, "obs_stride_merge_segments");
    config->obs_range_road_front_m = (float) unpack(kwargs, "obs_range_road_front_m");
    config->obs_range_road_behind_m = (float) unpack(kwargs, "obs_range_road_behind_m");
    config->obs_range_road_side_m = (float) unpack(kwargs, "obs_range_road_side_m");
    if (PyErr_Occurred()) {
        return -1;
    }
    if (config->use_neighbor_cache < 0 || config->use_neighbor_cache > 1 || config->obs_lane_stride < 1
        || config->obs_boundary_stride < 1 || config->obs_stride_merge_segments < 0
        || config->obs_stride_merge_segments > 1 || !isfinite(config->obs_range_road_front_m)
        || !isfinite(config->obs_range_road_behind_m) || !isfinite(config->obs_range_road_side_m)
        || config->obs_range_road_front_m < 0.0f || config->obs_range_road_behind_m < 0.0f
        || config->obs_range_road_side_m < 0.0f) {
        PyErr_SetString(PyExc_ValueError, "invalid map cache configuration");
        return -1;
    }
    *paths = (const char **) malloc(*num_map_files * sizeof(char *));
    if (*paths == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    for (int i = 0; i < *num_map_files; i++) {
        (*paths)[i] = PyUnicode_AsUTF8(PyList_GetItem(map_files, i));
        if ((*paths)[i] == NULL) {
            free(*paths);
            return -1;
        }
    }
    return 0;
}

static PyObject *map_cache_preload_py(
    PyObject *self __attribute__((unused)),
    PyObject *args __attribute__((unused)),
    PyObject *kwargs) {
    Drive config = {0};
    const char **paths;
    int num_map_files;
    if (unpack_map_cache_args(kwargs, &config, &paths, &num_map_files) != 0) {
        return NULL;
    }
    int cached = preload_map_cache(&config, paths, num_map_files);
    free(paths);
    if (cached < 0) {
        PyErr_SetString(PyExc_RuntimeError, "map_cache_preload failed to load or build a map binary");
        return NULL;
    }
    return PyLong_FromLong((long) cached);
}

static PyObject *map_cache_release_py(
    PyObject *self __attribute__((unused)),
    PyObject *args __attribute__((unused)),
    PyObject *kwargs) {
    Drive config = {0};
    const char **paths;
    int num_map_files;
    if (unpack_map_cache_args(kwargs, &config, &paths, &num_map_files) != 0) {
        return NULL;
    }
    int released = release_preloaded_map_cache(&config, paths, num_map_files);
    free(paths);
    if (released != 0) {
        PyErr_SetString(PyExc_RuntimeError, "map_cache_release failed: preload configuration drifted");
        return NULL;
    }
    Py_RETURN_NONE;
}

static int my_init(Drive *env, PyObject *kwargs) {
    env->reward_goal = (float) unpack(kwargs, "reward_goal");
    env->reward_collision = (float) unpack(kwargs, "reward_collision");
    env->reward_offroad = (float) unpack(kwargs, "reward_offroad");
    env->reward_comfort = (float) unpack(kwargs, "reward_comfort");
    env->reward_lane_align = (float) unpack(kwargs, "reward_lane_align");
    env->reward_vel_align = (float) unpack(kwargs, "reward_vel_align");
    env->reward_lane_center = (float) unpack(kwargs, "reward_lane_center");
    env->reward_center_bias = (float) unpack(kwargs, "reward_center_bias");
    env->reward_velocity = (float) unpack(kwargs, "reward_velocity");
    env->reward_reverse = (float) unpack(kwargs, "reward_reverse");
    env->reward_stop_line = (float) unpack(kwargs, "reward_stop_line");
    env->reward_timestep = (float) unpack(kwargs, "reward_timestep");
    env->reward_overspeed = (float) unpack(kwargs, "reward_overspeed");
    env->collision_behavior = (int) unpack(kwargs, "collision_behavior");
    env->offroad_behavior = (int) unpack(kwargs, "offroad_behavior");
    env->disable_red_light_infractions = (int) unpack(kwargs, "disable_red_light_infractions");
    env->disable_stop_sign_infractions = (int) unpack(kwargs, "disable_stop_sign_infractions");
    env->traffic_light_green_min_s = (float) unpack(kwargs, "traffic_light_green_min_s");
    env->traffic_light_yellow_min_s = (float) unpack(kwargs, "traffic_light_yellow_min_s");
    env->traffic_light_yellow_max_s = (float) unpack(kwargs, "traffic_light_yellow_max_s");
    if (!(env->traffic_light_green_min_s > 0.0f) || env->traffic_light_green_min_s > TL_DEFAULT_GREEN_DURATION
        || !(env->traffic_light_yellow_min_s > 0.0f)
        || !(env->traffic_light_yellow_max_s >= env->traffic_light_yellow_min_s)
        || !isfinite(env->traffic_light_yellow_max_s)) {
        PyErr_SetString(
            PyExc_ValueError,
            "traffic light cycle: need 0 < green_min <= 10 and 0 < yellow_min <= yellow_max");
        return -1;
    }
    env->traffic_light_behavior = (int) unpack(kwargs, "traffic_light_behavior");
    env->stop_sign_behavior = (int) unpack(kwargs, "stop_sign_behavior");
    env->use_map_cache = (int) unpack(kwargs, "use_map_cache");
    env->use_neighbor_cache = (int) unpack(kwargs, "use_neighbor_cache");
    uint64_t env_seed;
    if (unpack_seed(kwargs, &env_seed) != 0) {
        return -1;
    }
    my_reseed(env, env_seed);
    env->use_exact_episode_seed = (int) unpack(kwargs, "use_exact_episode_seed");
    env->goal_radius = (float) unpack(kwargs, "goal_radius");
    env->min_goal_spacing = (float) unpack(kwargs, "min_goal_spacing");
    env->max_goal_spacing = (float) unpack(kwargs, "max_goal_spacing");
    env->goal_heading_max_deg = (float) unpack(kwargs, "goal_heading_max_deg");
    if (env->goal_heading_max_deg < 0.0f || env->goal_heading_max_deg > 180.0f) {
        PyErr_Format(
            PyExc_ValueError,
            "goal_heading_max_deg must be in [0, 180]. Got: %d",
            (int) env->goal_heading_max_deg);
        return -1;
    }
    env->num_goals = (int) unpack(kwargs, "num_goals");
    if (env->num_goals < 1 || env->num_goals > MAX_GOALS) {
        PyErr_Format(PyExc_ValueError, "num_goals must be in [1, %d]. Got: %d", MAX_GOALS, env->num_goals);
        return -1;
    }
    env->goal_regen_mode = (int) unpack(kwargs, "goal_regen_mode");
    env->goal_source = (int) unpack(kwargs, "goal_source");
    env->obs_slots_boundary_n = (int) unpack(kwargs, "obs_slots_boundary_n");
    env->obs_slots_lane_n = (int) unpack(kwargs, "obs_slots_lane_n");
    env->obs_slots_partners_n = (int) unpack(kwargs, "obs_slots_partners_n");
    env->obs_slots_traffic_controls_n = (int) unpack(kwargs, "obs_slots_traffic_controls_n");
    env->obs_traffic_light_yellow_remaining = (int) unpack(kwargs, "obs_traffic_light_yellow_remaining");
    env->traffic_lights_enabled = (bool) unpack(kwargs, "traffic_lights_enabled");
    env->stop_signs_enabled = (bool) unpack(kwargs, "stop_signs_enabled");
    env->obs_lane_stride = (int) unpack(kwargs, "obs_lane_stride");
    env->obs_boundary_stride = (int) unpack(kwargs, "obs_boundary_stride");
    env->obs_stride_merge_segments = (int) unpack(kwargs, "obs_stride_merge_segments");
    env->dt = (float) unpack(kwargs, "dt");
    env->base_max_speed_mps = (float) unpack(kwargs, "base_max_speed_mps");
    env->max_speed_mps = (float) unpack(kwargs, "max_speed_mps");
    if (!(env->max_speed_mps > 0.0f) || !isfinite(env->max_speed_mps)) {
        PyErr_SetString(PyExc_ValueError, "max_speed_mps must be a positive finite float");
        return -1;
    }
    env->spawn_lateral_offset_max_frac = (float) unpack(kwargs, "spawn_lateral_offset_max_frac");
    env->spawn_heading_max_deg = (float) unpack(kwargs, "spawn_heading_max_deg");
    env->pose_noise_xy_m = (float) unpack(kwargs, "pose_noise_xy_m");
    env->pose_noise_yaw_rad = (float) unpack(kwargs, "pose_noise_yaw_deg") * (float) M_PI / 180.0f;
    env->speed_limit_random_prob = (float) unpack(kwargs, "speed_limit_random_prob");
    env->speed_limit_random_delta_mps = (float) unpack(kwargs, "speed_limit_random_delta_mps");
    env->speed_limit_random_min_mps = (float) unpack(kwargs, "speed_limit_random_min_mps");
    env->speed_limit_random_max_mps = (float) unpack(kwargs, "speed_limit_random_max_mps");
    if (!(env->speed_limit_random_prob >= 0.0f && env->speed_limit_random_prob <= 1.0f)) {
        PyErr_SetString(PyExc_ValueError, "speed_limit_random_prob must be in [0, 1]");
        return -1;
    }
    if (!(env->speed_limit_random_delta_mps >= 0.0f) || !isfinite(env->speed_limit_random_delta_mps)) {
        PyErr_SetString(PyExc_ValueError, "speed_limit_random_delta_mps must be finite and >= 0");
        return -1;
    }
    if (!(env->speed_limit_random_min_mps > 0.0f && env->speed_limit_random_min_mps <= env->speed_limit_random_max_mps)
        || !isfinite(env->speed_limit_random_max_mps)) {
        PyErr_SetString(PyExc_ValueError, "speed limit clip must satisfy 0 < min <= max (finite)");
        return -1;
    }
    env->goal_speed = (float) unpack(kwargs, "goal_speed");
    env->conditioning_accel_scale = (float) unpack(kwargs, "conditioning_accel_scale");
    env->conditioning_speed_scale = (float) unpack(kwargs, "conditioning_speed_scale");
    if (!(env->conditioning_speed_scale >= 1.0f) || !isfinite(env->conditioning_speed_scale)) {
        PyErr_SetString(PyExc_ValueError, "conditioning_speed_scale must be finite and >= 1");
        return -1;
    }
    env->goal_reach_requires_speed = (int) unpack(kwargs, "goal_reach_requires_speed");
    env->scenario_length = (int) unpack(kwargs, "scenario_length");
    env->termination_mode = (int) unpack(kwargs, "termination_mode");
    env->inactive_agent_threshold = (float) unpack(kwargs, "inactive_agent_threshold");
    env->map_name = unpack_str(kwargs, "map_file");
    env->num_controllable_agents = (int) unpack(kwargs, "max_agents");
    env->num_max_agents = (int) unpack(kwargs, "max_agents_per_env");
    env->cosim_partner_slots = (int) unpack(kwargs, "cosim_partner_slots");
    env->cosim_eval_semantics = (int) unpack(kwargs, "cosim_eval_semantics");
    env->scripted_pedestrian_slots = (int) unpack(kwargs, "scripted_pedestrian_slots");
    if (env->scripted_pedestrian_slots < 0) {
        PyErr_SetString(PyExc_ValueError, "scripted_pedestrian_slots must be >= 0");
        return -1;
    }
    env->init_step = (int) unpack(kwargs, "init_step");
    env->init_step_min_horizon = (int) unpack(kwargs, "init_step_min_horizon");
    env->stagger_first_episode = (int) unpack(kwargs, "stagger_first_episode");
    if (env->stagger_first_episode
        && (env->init_step_min_horizon < 1 || env->init_step + env->init_step_min_horizon > env->scenario_length)) {
        PyErr_SetString(
            PyExc_ValueError,
            "stagger_first_episode needs 1 <= init_step_min_horizon <= scenario_length - init_step");
        return -1;
    }
    env->init_mode = (int) unpack(kwargs, "init_mode");
    env->control_mode = (int) unpack(kwargs, "control_mode");
    env->sdc_controller = (int) unpack(kwargs, "sdc_controller");
    env->non_sdc_controller = (int) unpack(kwargs, "non_sdc_controller");
    env->non_vehicle_controller = (int) unpack(kwargs, "non_vehicle_controller");
    env->replay_expert_agents = (int) unpack(kwargs, "replay_expert_agents");
    env->simulation_mode = (int) unpack(kwargs, "simulation_mode");
    if (env->scripted_pedestrian_slots > 0
        && (env->simulation_mode != SIMULATION_MODE_GIGAFLOW || env->cosim_partner_slots > 0)) {
        PyErr_SetString(
            PyExc_ValueError,
            "scripted_pedestrian_slots needs gigaflow simulation_mode without cosim_partner_slots");
        return -1;
    }
    env->reward_conditioning = (bool) unpack(kwargs, "reward_conditioning");
    env->reward_randomization = (bool) unpack(kwargs, "reward_randomization");
    env->compute_eval_metrics = (bool) unpack(kwargs, "compute_eval_metrics");
    env->eval_mode = (int) unpack(kwargs, "eval_mode");
    env->obs_norm_speed_mps = (float) unpack(kwargs, "obs_norm_speed_mps");
    env->eval_training_render = (int) unpack(kwargs, "eval_training_render");
    env->obs_norm_goal_offset_m = (float) unpack(kwargs, "obs_norm_goal_offset_m");
    env->obs_norm_xy_offset_m = (float) unpack(kwargs, "obs_norm_xy_offset_m");
    env->obs_norm_veh_length_m = (float) unpack(kwargs, "obs_norm_veh_length_m");
    env->obs_norm_veh_width_m = (float) unpack(kwargs, "obs_norm_veh_width_m");
    env->obs_norm_road_seg_length_m = (float) unpack(kwargs, "obs_norm_road_seg_length_m");
    env->obs_norm_road_seg_width_m = (float) unpack(kwargs, "obs_norm_road_seg_width_m");
    env->obs_norm_z_m = (float) unpack(kwargs, "obs_norm_z_m");
    env->eval_perceived_size_margin_m = (float) unpack(kwargs, "eval_perceived_size_margin_m");
    env->red_light_train_front_margin_m = (float) unpack(kwargs, "red_light_train_front_margin_m");
    if (!(env->red_light_train_front_margin_m >= 0.0f) || !isfinite(env->red_light_train_front_margin_m)) {
        PyErr_SetString(PyExc_ValueError, "red_light_train_front_margin_m must be finite and >= 0");
        return -1;
    }
    env->obs_partner_split_length_m = (float) unpack(kwargs, "obs_partner_split_length_m");
    if (!(env->obs_partner_split_length_m >= 0.0f) || !isfinite(env->obs_partner_split_length_m)) {
        PyErr_SetString(PyExc_ValueError, "obs_partner_split_length_m must be finite and >= 0");
        return -1;
    }
    env->obs_lane_width_clip_min_m = (float) unpack(kwargs, "obs_lane_width_clip_min_m");
    env->obs_lane_width_clip_max_m = (float) unpack(kwargs, "obs_lane_width_clip_max_m");
    if (!(env->obs_lane_width_clip_min_m >= 0.0f) || !isfinite(env->obs_lane_width_clip_min_m)
        || !(env->obs_lane_width_clip_max_m >= 0.0f) || !isfinite(env->obs_lane_width_clip_max_m)
        || (env->obs_lane_width_clip_max_m > 0.0f && env->obs_lane_width_clip_min_m > env->obs_lane_width_clip_max_m)) {
        PyErr_SetString(PyExc_ValueError, "obs_lane_width_clip_min_m/max_m must be finite, >= 0 and min <= max");
        return -1;
    }
    env->eval_standstill_jerk_deadband_mps3 = (float) unpack(kwargs, "eval_standstill_jerk_deadband_mps3");
    env->obs_range_traffic_control_m = (float) unpack(kwargs, "obs_range_traffic_control_m");
    env->obs_range_partner_m = (float) unpack(kwargs, "obs_range_partner_m");
    env->obs_range_road_front_m = (float) unpack(kwargs, "obs_range_road_front_m");
    env->obs_range_road_behind_m = (float) unpack(kwargs, "obs_range_road_behind_m");
    env->obs_range_road_side_m = (float) unpack(kwargs, "obs_range_road_side_m");
    env->obs_slots_lane_kept = (int) unpack(kwargs, "obs_slots_lane_kept");
    env->obs_slots_boundary_kept = (int) unpack(kwargs, "obs_slots_boundary_kept");
    env->partner_blindness_prob = (float) unpack(kwargs, "partner_blindness_prob");
    env->partner_blindness_prob_min = (float) unpack(kwargs, "partner_blindness_prob_min");
    if (!(env->partner_blindness_prob_min >= 0.0f && env->partner_blindness_prob_min <= env->partner_blindness_prob)) {
        PyErr_SetString(PyExc_ValueError, "partner_blindness_prob_min must be within [0, partner_blindness_prob]");
        return -1;
    }
    env->partner_blindness_trigger_prob = (float) unpack(kwargs, "partner_blindness_trigger_prob");
    env->partner_blindness_duration
        = (int) ceilf((float) unpack(kwargs, "partner_blindness_duration_seconds") / env->dt);
    env->phantom_braking_prob = (float) unpack(kwargs, "phantom_braking_prob");
    env->phantom_braking_trigger_prob = (float) unpack(kwargs, "phantom_braking_trigger_prob");
    env->phantom_braking_duration = (int) ceilf((float) unpack(kwargs, "phantom_braking_duration_seconds") / env->dt);

    if (PyErr_Occurred()) {
        return -1;
    }
    if (init(env) != 0) {
        PyErr_Format(PyExc_RuntimeError, "Failed to initialise the env for map %s (reason on stderr)", env->map_name);
        return -1;
    }
    // forces c_reset's full path, so a logged episode_seed replays the same whichever reset made the episode
    env->timestep = -1;
    return 0;
}

// One row per frozen eval env, whose log holds exactly its finished episode.
static int my_episode_to_dict(PyObject *dict, Drive *env) {
    float n = env->log.n;
    Log normalized = env->log;
    int num_log_fields = sizeof(Log) / sizeof(float);
    for (int field_idx = 0; field_idx < num_log_fields; field_idx++) {
        ((float *) &normalized)[field_idx] /= n;
    }
    my_log(dict, env, &normalized, n);
    assign_to_dict(dict, "active_agent_count", (float) env->active_agent_count);
    assign_to_dict(dict, "episode_timestep", (float) env->timestep);

    PyObject *seed_obj = PyLong_FromUnsignedLongLong(env->log_episode_seed);
    if (seed_obj == NULL) {
        return -1;
    }
    if (PyDict_SetItemString(dict, "seed", seed_obj) < 0) {
        Py_DECREF(seed_obj);
        return -1;
    }
    Py_DECREF(seed_obj);

    if (env->map_name != NULL && env->map_name[0] != '\0') {
        PyObject *map_name = PyUnicode_FromString(env->map_name);
        if (map_name == NULL) {
            return -1;
        }
        if (PyDict_SetItemString(dict, "map_name", map_name) < 0) {
            Py_DECREF(map_name);
            return -1;
        }
        Py_DECREF(map_name);
    }
    if (env->scenario_id[0] != '\0') {
        PyObject *scenario_id = PyUnicode_FromString(env->scenario_id);
        if (scenario_id == NULL) {
            return -1;
        }
        if (PyDict_SetItemString(dict, "scenario_id", scenario_id) < 0) {
            Py_DECREF(scenario_id);
            return -1;
        }
        Py_DECREF(scenario_id);
    }
    return 0;
}

static void my_log(PyObject *dict, Drive *env, Log *log, float n) {
    float total_distance_travelled = log->total_distance_travelled * n;
    float total_infractions = roundf(log->total_infractions * n);
    float avg_distance_per_infraction = total_distance_travelled / fmaxf(1.0f, total_infractions);

    assign_to_dict(dict, "n", n);
    assign_to_dict(dict, "pedestrian_collision_rate", log->pedestrian_collision_rate);
    assign_to_dict(dict, "pedestrian_crossings", log->pedestrian_crossings);
    assign_to_dict(dict, "pedestrian_bystanders", log->pedestrian_bystanders);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "red_light_violation_rate", log->red_light_violation_rate);
    assign_to_dict(dict, "stop_sign_violation_rate", log->stop_sign_violation_rate);
    assign_to_dict(dict, "comfort_violation_count", log->comfort_violation_count);
    assign_to_dict(dict, "velocity_progress_sum", log->velocity_progress_sum);
    assign_to_dict(dict, "num_goals_reached", log->num_goals_reached);
    assign_to_dict(dict, "lane_center_rate", log->lane_center_rate);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "avg_speed_per_agent", log->avg_speed_per_agent);
    assign_to_dict(dict, "avg_distance_per_infraction", avg_distance_per_infraction);
    assign_to_dict(dict, "total_distance_travelled_sum", total_distance_travelled);
    assign_to_dict(dict, "total_infraction_count", total_infractions);
    assign_to_dict(dict, "reward_components/collision", log->reward_collision);
    assign_to_dict(dict, "reward_components/offroad", log->reward_offroad);
    assign_to_dict(dict, "reward_components/red_light", log->reward_red_light);
    assign_to_dict(dict, "reward_components/stop_sign", log->reward_stop_sign);
    assign_to_dict(dict, "reward_components/goal", log->reward_goal);
    assign_to_dict(dict, "reward_components/lane_align", log->reward_lane_align);
    assign_to_dict(dict, "reward_components/lane_center", log->reward_lane_center);
    assign_to_dict(dict, "reward_components/comfort", log->reward_comfort);
    assign_to_dict(dict, "reward_components/velocity", log->reward_velocity);
    assign_to_dict(dict, "reward_components/timestep", log->reward_timestep);
    assign_to_dict(dict, "reward_components/reverse", log->reward_reverse);
    assign_to_dict(dict, "reward_components/overspeed", log->reward_overspeed);

    if (env->compute_eval_metrics) {
        assign_to_dict(dict, "at_fault_collision_rate", log->at_fault_collision_rate);
        assign_to_dict(dict, "driving_direction_score", log->driving_direction_score);
        assign_to_dict(dict, "speed_limit_compliance", log->speed_limit_compliance);
        assign_to_dict(dict, "making_progress_rate", log->making_progress_rate);
        assign_to_dict(dict, "progress_ratio", log->progress_ratio);
        assign_to_dict(dict, "comfort_score", log->comfort_score);
        assign_to_dict(dict, "multi_lane_time", log->multi_lane_time);
        assign_to_dict(dict, "multi_lane_score", log->multi_lane_score);
    }
}
