#include "drive.h"
#define Env Drive
#define MY_SHARED
#define MY_PUT
#include "../env_binding.h"

static int my_put(Env *env, PyObject *args, PyObject *kwargs) {
    PyObject *obs = PyDict_GetItemString(kwargs, "observations");
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return 1;
    }
    PyArrayObject *observations = (PyArrayObject *)obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return 1;
    }
    env->observations = PyArray_DATA(observations);

    PyObject *act = PyDict_GetItemString(kwargs, "actions");
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return 1;
    }
    PyArrayObject *actions = (PyArrayObject *)act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return 1;
    }
    env->actions = PyArray_DATA(actions);
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return 1;
    }

    PyObject *rew = PyDict_GetItemString(kwargs, "rewards");
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return 1;
    }
    PyArrayObject *rewards = (PyArrayObject *)rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return 1;
    }
    env->rewards = PyArray_DATA(rewards);

    PyObject *term = PyDict_GetItemString(kwargs, "terminals");
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return 1;
    }
    PyArrayObject *terminals = (PyArrayObject *)term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return 1;
    }
    env->terminals = PyArray_DATA(terminals);
    return 0;
}

static PyObject *my_shared(PyObject *self, PyObject *args, PyObject *kwargs) {
    char *map_dir = unpack_str(kwargs, "map_dir");
    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");
    int init_mode = unpack(kwargs, "init_mode");
    int control_mode = unpack(kwargs, "control_mode");
    int init_steps = unpack(kwargs, "init_steps");
    int goal_behavior = unpack(kwargs, "goal_behavior");
    float goal_target_distance = unpack(kwargs, "goal_target_distance");
    int sequential_map_sampling = unpack(kwargs, "sequential_map_sampling");
    clock_gettime(CLOCK_REALTIME, &ts);
    srand(ts.tv_nsec);
    int total_agent_count = 0;
    int env_count = 0;
    int max_envs = sequential_map_sampling ? num_maps : num_agents;
    int map_idx = 0;
    int maps_checked = 0;
    PyObject *agent_offsets = PyList_New(max_envs + 1);
    PyObject *map_ids = PyList_New(max_envs);
    // getting env count
    while (sequential_map_sampling ? map_idx < max_envs : total_agent_count < num_agents && env_count < max_envs) {
        char map_file[512];
        // Take the next map in sequence or a random map
        int map_id = sequential_map_sampling ? map_idx++ : rand() % num_maps;
        Drive *env = calloc(1, sizeof(Drive));
        env->init_mode = init_mode;
        env->control_mode = control_mode;
        env->init_steps = init_steps;
        env->goal_behavior = goal_behavior;
        env->goal_target_distance = goal_target_distance;
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", map_dir, map_id);
        env->entities = load_map_binary(map_file, env);
        set_active_agents(env);

        // Skip map if it doesn't contain any controllable agents
        if (env->active_agent_count == 0) {
            if (!sequential_map_sampling) {
                maps_checked++;

                // Safeguard: if we've checked all available maps and found no active agents, raise an error
                if (maps_checked >= num_maps) {
                    for (int j = 0; j < env->num_entities; j++) {
                        free_entity(&env->entities[j]);
                    }
                    free(env->entities);
                    free(env->active_agent_indices);
                    free(env->static_agent_indices);
                    free(env->expert_static_agent_indices);
                    free(env);
                    Py_DECREF(agent_offsets);
                    Py_DECREF(map_ids);
                    char error_msg[256];
                    sprintf(error_msg, "No controllable agents found in any of the %d available maps", num_maps);
                    PyErr_SetString(PyExc_ValueError, error_msg);
                    return NULL;
                }
            }

            for (int j = 0; j < env->num_entities; j++) {
                free_entity(&env->entities[j]);
            }
            free(env->entities);
            free(env->active_agent_indices);
            free(env->static_agent_indices);
            free(env->expert_static_agent_indices);
            free(env);
            continue;
        }

        // Store map_id
        PyObject *map_id_obj = PyLong_FromLong(map_id);
        PyList_SetItem(map_ids, env_count, map_id_obj);
        // Store agent offset
        PyObject *offset = PyLong_FromLong(total_agent_count);
        PyList_SetItem(agent_offsets, env_count, offset);
        total_agent_count += env->active_agent_count;
        env_count++;
        for (int j = 0; j < env->num_entities; j++) {
            free_entity(&env->entities[j]);
        }
        free(env->entities);
        free(env->active_agent_indices);
        free(env->static_agent_indices);
        free(env->expert_static_agent_indices);
        free(env);
    }
    // printf("Generated %d environments to cover %d agents (requested %d agents)\n", env_count, total_agent_count,
    // num_agents);
    if (!sequential_map_sampling && total_agent_count >= num_agents) {
        total_agent_count = num_agents;
    }
    PyObject *final_total_agent_count = PyLong_FromLong(total_agent_count);
    PyList_SetItem(agent_offsets, env_count, final_total_agent_count);
    PyObject *final_env_count = PyLong_FromLong(env_count);
    // resize lists
    PyObject *resized_agent_offsets = PyList_GetSlice(agent_offsets, 0, env_count + 1);
    PyObject *resized_map_ids = PyList_GetSlice(map_ids, 0, env_count);
    PyObject *tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, resized_agent_offsets);
    PyTuple_SetItem(tuple, 1, resized_map_ids);
    PyTuple_SetItem(tuple, 2, final_env_count);
    return tuple;
}

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->ini_file = unpack_str(kwargs, "ini_file");
    env_init_config conf = {0};
    if (ini_parse(env->ini_file, handler, &conf) < 0) {
        printf("Error while loading %s", env->ini_file);
    }
    if (kwargs && PyDict_GetItemString(kwargs, "episode_length")) {
        conf.episode_length = (int)unpack(kwargs, "episode_length");
    }
    if (conf.episode_length <= 0) {
        PyErr_SetString(PyExc_ValueError, "episode_length must be > 0 (set in INI or kwargs)");
        return -1;
    }
    env->action_type = conf.action_type;
    env->dynamics_model = conf.dynamics_model;
    env->reward_vehicle_collision = conf.reward_vehicle_collision;
    env->reward_offroad_collision = conf.reward_offroad_collision;
    env->reward_goal = conf.reward_goal;
    env->reward_goal_post_respawn = conf.reward_goal_post_respawn;
    env->episode_length = conf.episode_length;
    env->termination_mode = conf.termination_mode;
    env->collision_behavior = conf.collision_behavior;
    env->offroad_behavior = conf.offroad_behavior;
    env->max_controlled_agents = unpack(kwargs, "max_controlled_agents");
    env->dt = conf.dt;
    env->init_mode = (int)unpack(kwargs, "init_mode");
    env->control_mode = (int)unpack(kwargs, "control_mode");
    env->goal_behavior = (int)unpack(kwargs, "goal_behavior");
    env->goal_target_distance = (float)unpack(kwargs, "goal_target_distance");
    env->goal_radius = (float)unpack(kwargs, "goal_radius");
    env->goal_speed = (float)unpack(kwargs, "goal_speed");
    char *map_dir = unpack_str(kwargs, "map_dir");
    int map_id = unpack(kwargs, "map_id");
    int max_agents = unpack(kwargs, "max_agents");
    int init_steps = unpack(kwargs, "init_steps");
    char map_file[512];
    snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", map_dir, map_id);
    env->num_agents = max_agents;
    env->map_name = strdup(map_file);
    env->init_steps = init_steps;
    env->timestep = init_steps;
    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    assign_to_dict(dict, "lane_alignment_rate", log->lane_alignment_rate);
    assign_to_dict(dict, "offroad_per_agent", log->offroad_per_agent);
    assign_to_dict(dict, "collisions_per_agent", log->collisions_per_agent);
    assign_to_dict(dict, "goals_sampled_this_episode", log->goals_sampled_this_episode);
    assign_to_dict(dict, "goals_reached_this_episode", log->goals_reached_this_episode);
    assign_to_dict(dict, "speed_at_goal", log->speed_at_goal);
    // assign_to_dict(dict, "avg_displacement_error", log->avg_displacement_error);
    return 0;
}
