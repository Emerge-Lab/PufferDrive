#include "drive.h"
#define Env Drive
#define MY_SHARED
#define MY_PUT
#include "../env_binding.h"

static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    PyObject* obs = PyDict_GetItemString(kwargs, "observations");
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return 1;
    }
    PyArrayObject* observations = (PyArrayObject*)obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return 1;
    }
    env->observations = PyArray_DATA(observations);

    PyObject* act = PyDict_GetItemString(kwargs, "actions");
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return 1;
    }
    PyArrayObject* actions = (PyArrayObject*)act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return 1;
    }
    env->actions = PyArray_DATA(actions);
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return 1;
    }

    PyObject* rew = PyDict_GetItemString(kwargs, "rewards");
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return 1;
    }
    PyArrayObject* rewards = (PyArrayObject*)rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return 1;
    }
    env->rewards = PyArray_DATA(rewards);

    PyObject* term = PyDict_GetItemString(kwargs, "terminals");
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return 1;
    }
    PyArrayObject* terminals = (PyArrayObject*)term;
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

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");
    int control_mode = unpack(kwargs, "control_mode");
    int goal_sampling_mode = unpack(kwargs, "goal_sampling_mode");
    float max_distance_to_goal = unpack(kwargs, "max_distance_to_goal");
    int init_steps = unpack(kwargs, "init_steps");
    int goal_behavior = unpack(kwargs, "goal_behavior");
    int init_mode = unpack(kwargs, "init_mode");
    int num_agents_per_world = unpack(kwargs, "num_agents_per_world");

    // Get configs
    char* ini_file = unpack_str(kwargs, "ini_file");
    env_init_config conf = {0};
    if(ini_parse(ini_file, handler, &conf) < 0) {
        raise_error_with_message(ERROR_UNKNOWN, "Error while loading %s", ini_file);
    }

    PyObject* map_files_list = PyDict_GetItemString(kwargs, "map_files");
    Py_ssize_t map_files_len = 0;
    if (map_files_list && PyList_Check(map_files_list)) {
        map_files_len = PyList_Size(map_files_list);
    }

    int map_count = map_files_len > 0 ? (int)map_files_len : num_maps;
    if (map_count <= 0) {
        PyErr_SetString(PyExc_ValueError, "No map files available to initialize environments");
        return NULL;
    }
    clock_gettime(CLOCK_REALTIME, &ts);
    srand(ts.tv_nsec);
    int total_agent_count = 0;
    int env_count = 0;
    int max_envs = num_agents;
    int maps_checked = 0;
    int* visited_maps = calloc(map_count, sizeof(int));
    PyObject* agent_offsets = PyList_New(max_envs+1);
    PyObject* map_ids = PyList_New(max_envs);
    // getting env count
    while(total_agent_count < num_agents && env_count < max_envs){
        char map_file[100];
        int map_id = rand() % map_count;
        int already_checked = visited_maps ? visited_maps[map_id] : 0;
        if (init_mode == INIT_RANDOM_AGENTS) {
            // Store map_id
            PyObject* map_id_obj = PyLong_FromLong(map_id);
            PyList_SetItem(map_ids, env_count, map_id_obj);
            // Store agent offset
            PyObject* offset = PyLong_FromLong(total_agent_count);
            PyList_SetItem(agent_offsets, env_count, offset);
            total_agent_count += num_agents_per_world;
            env_count++;
        }
        else {
            Drive* env = calloc(1, sizeof(Drive));
            env->init_mode = init_mode;
            env->control_mode = control_mode;
            env->init_steps = init_steps;
            env->goal_behavior = goal_behavior;
            env->goal_sampling_mode = goal_sampling_mode;
            env->max_distance_to_goal = max_distance_to_goal;
            const char* map_file_path = NULL;
            if (map_files_len > 0 && map_id < map_files_len) {
                PyObject* path_obj = PyList_GetItem(map_files_list, map_id);
                if (PyUnicode_Check(path_obj)) {
                    map_file_path = PyUnicode_AsUTF8(path_obj);
                }
            }
            if (map_file_path == NULL) {
                sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);
                    map_file_path = map_file;
            }
            env->entities = load_map_binary(map_file_path, env);
        
            set_active_agents(env);

            // Skip map if it doesn't contain any controllable agents
            if(env->active_agent_count == 0) {
                    if (visited_maps) {
                    visited_maps[map_id] = 1;
                }
                if (!already_checked) {
                    maps_checked++;
                }

                // Safeguard: if we've checked all available maps and found no active agents, raise an error
                if(maps_checked >= map_count) {
                    for(int j=0;j<env->num_entities;j++) {
                        free_entity(&env->entities[j]);
                    }
                    free(env->entities);
                    free(env->active_agent_indices);
                    free(env->static_agent_indices);
                    free(env->expert_static_agent_indices);
                    free(env);
                    Py_DECREF(agent_offsets);
                    Py_DECREF(map_ids);
                    free(visited_maps);
                    char error_msg[256];
                    sprintf(error_msg, "No controllable agents found in any of the %d available maps", map_count);
                    PyErr_SetString(PyExc_ValueError, error_msg);
                    return NULL;
                }

                for(int j=0;j<env->num_entities;j++) {
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
            PyObject* map_id_obj = PyLong_FromLong(map_id);
            PyList_SetItem(map_ids, env_count, map_id_obj);
            // Store agent offset
            PyObject* offset = PyLong_FromLong(total_agent_count);
            PyList_SetItem(agent_offsets, env_count, offset);
            total_agent_count += env->active_agent_count;
            env_count++;
            for(int j=0;j<env->num_entities;j++) {
                free_entity(&env->entities[j]);
            }
            free(env->entities);
            free(env->active_agent_indices);
            free(env->static_agent_indices);
            free(env->expert_static_agent_indices);
            free(env);
        }
    }
    //printf("Generated %d environments to cover %d agents (requested %d agents)\n", env_count, total_agent_count, num_agents);
    if(total_agent_count >= num_agents){
        total_agent_count = num_agents;
    }
    PyObject* final_total_agent_count = PyLong_FromLong(total_agent_count);
    PyList_SetItem(agent_offsets, env_count, final_total_agent_count);
    PyObject* final_env_count = PyLong_FromLong(env_count);
    // resize lists
    PyObject* resized_agent_offsets = PyList_GetSlice(agent_offsets, 0, env_count + 1);
    PyObject* resized_map_ids = PyList_GetSlice(map_ids, 0, env_count);
    PyObject* tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, resized_agent_offsets);
    PyTuple_SetItem(tuple, 1, resized_map_ids);
    PyTuple_SetItem(tuple, 2, final_env_count);
    free(visited_maps);
    return tuple;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->ini_file = unpack_str(kwargs, "ini_file");
    env_init_config conf = {0};
    if(ini_parse(env->ini_file, handler, &conf) < 0) {
        printf("Error while loading %s", env->ini_file);
    }
    if (kwargs && PyDict_GetItemString(kwargs, "scenario_length")) {
        conf.scenario_length = (int)unpack(kwargs, "scenario_length");
    }
    if (conf.scenario_length <= 0) {
        PyErr_SetString(PyExc_ValueError, "scenario_length must be > 0 (set in INI or kwargs)");
        return -1;
    }
    env->action_type = conf.action_type;
    env->dynamics_model = conf.dynamics_model;
    env->reward_vehicle_collision = conf.reward_vehicle_collision;
    env->reward_offroad_collision = conf.reward_offroad_collision;
    env->reward_goal = conf.reward_goal;
    env->reward_goal_post_respawn = conf.reward_goal_post_respawn;
    env->reward_ade = conf.reward_ade;
    env->scenario_length = conf.scenario_length;
    env->collision_behavior = conf.collision_behavior;
    env->offroad_behavior = conf.offroad_behavior;
    env->max_controlled_agents = unpack(kwargs, "max_controlled_agents");
    env->dt = conf.dt;
    env->init_mode = (int)unpack(kwargs, "init_mode");
    env->control_mode = (int)unpack(kwargs, "control_mode");
    env->num_agents_per_world = (int)unpack(kwargs, "num_agents_per_world");
    env->vehicle_width = (float)unpack(kwargs, "vehicle_width");
    env->vehicle_length = (float)unpack(kwargs, "vehicle_length");
    env->vehicle_height = (float)unpack(kwargs, "vehicle_height");
    env->goal_behavior = (int)unpack(kwargs, "goal_behavior");
    env->goal_radius = (float)unpack(kwargs, "goal_radius");
    env->goal_sampling_mode = (int)unpack(kwargs, "goal_sampling_mode");
    env->max_distance_to_goal = (float)unpack(kwargs, "max_distance_to_goal");
    env->goal_curriculum_end_distance = (float)unpack(kwargs, "goal_curriculum_end_distance");
    int map_id = unpack(kwargs, "map_id");
    int max_agents = unpack(kwargs, "max_agents");
    int init_steps = unpack(kwargs, "init_steps");
    PyObject* map_files_list = PyDict_GetItemString(kwargs, "map_files");
    Py_ssize_t map_files_len = 0;
    if (map_files_list && PyList_Check(map_files_list)) {
        map_files_len = PyList_Size(map_files_list);
    }
    char map_file[100];
    const char* map_file_path = NULL;
    if (map_files_len > 0 && map_id < map_files_len) {
        PyObject* path_obj = PyList_GetItem(map_files_list, map_id);
        if (PyUnicode_Check(path_obj)) {
            map_file_path = PyUnicode_AsUTF8(path_obj);
        }
    }
    if (map_file_path == NULL) {
        sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);
        map_file_path = map_file;
    }
    env->max_active_agents = max_agents;
    env->map_name = strdup(map_file_path);
    env->init_steps = init_steps;
    env->timestep = init_steps;
    init(env);

    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "avg_displacement_error", log->avg_displacement_error);
    assign_to_dict(dict, "avg_spawn_distance_to_goal", log->avg_spawn_distance_to_goal);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    assign_to_dict(dict, "lane_alignment_rate", log->lane_alignment_rate);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "avg_offroad_per_agent", log->avg_offroad_per_agent);
    assign_to_dict(dict, "avg_collisions_per_agent", log->avg_collisions_per_agent);
    assign_to_dict(dict, "no_current_lane_rate", log->no_current_lane_rate);
    assign_to_dict(dict, "num_goals_reached", log->num_goals_reached);
    assign_to_dict(dict, "avg_initial_distance_to_goal", log->avg_initial_distance_to_goal);
    assign_to_dict(dict, "map_observation_ratio", log->map_observation_ratio);
    assign_to_dict(dict, "map_observation_ratio_variance", log->map_observation_ratio_variance);
    return 0;
}
