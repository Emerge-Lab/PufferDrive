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

static PyObject* my_shared_self_play(PyObject* self, PyObject* args, PyObject* kwargs) {
    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");
    clock_gettime(CLOCK_REALTIME, &ts);
    srand(ts.tv_nsec);
    int total_agent_count = 0;
    int env_count = 0;
    int max_envs = num_agents;
    PyObject* agent_offsets = PyList_New(max_envs+1);
    PyObject* map_ids = PyList_New(max_envs);
    // getting env count
    while(total_agent_count < num_agents && env_count < max_envs){
        char map_file[100];
        int map_id = rand() % num_maps;
        Drive* env = calloc(1, sizeof(Drive));
        sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);
        env->entities = load_map_binary(map_file, env);
        PyObject* obj = NULL;
        obj = kwargs ? PyDict_GetItemString(kwargs, "num_policy_controlled_agents") : NULL;
        if (obj && PyLong_Check(obj)) {
            env->policy_agents_per_env = (int)PyLong_AsLong(obj);
        } else {
            env->policy_agents_per_env = -1;
        }
        obj = kwargs ? PyDict_GetItemString(kwargs, "control_all_agents") : NULL;
        if (obj && PyLong_Check(obj)) {
            env->control_all_agents = (int)PyLong_AsLong(obj);
        } else {
            env->control_all_agents = 0;
        }
        obj = kwargs ? PyDict_GetItemString(kwargs, "deterministic_agent_selection") : NULL;
        if (obj && PyLong_Check(obj)) {
            env->deterministic_agent_selection = (int)PyLong_AsLong(obj);
        } else {
            env->deterministic_agent_selection = 0;
        }
        set_active_agents(env);
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
        free(env->static_car_indices);
        free(env->expert_static_car_indices);
        free(env);
    }
    if(total_agent_count >= num_agents){
        total_agent_count = num_agents;
    }
    PyObject* final_total_agent_count = PyLong_FromLong(total_agent_count);
    PyList_SetItem(agent_offsets, env_count, final_total_agent_count);
    PyObject* final_env_count = PyLong_FromLong(env_count);
    // resize lists
    PyObject* resized_agent_offsets = PyList_GetSlice(agent_offsets, 0, env_count + 1);
    PyObject* resized_map_ids = PyList_GetSlice(map_ids, 0, env_count);
    //
    //Py_DECREF(agent_offsets);
    //Py_DECREF(map_ids);
    // create a tuple
    PyObject* tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, resized_agent_offsets);
    PyTuple_SetItem(tuple, 1, resized_map_ids);
    PyTuple_SetItem(tuple, 2, final_env_count);
    return tuple;

    //Py_DECREF(num);
    /*
    for(int i = 0;i<num_envs; i++) {
        for(int j=0;j<temp_envs[i].num_entities;j++) {
            free_entity(&temp_envs[i].entities[j]);
        }
        free(temp_envs[i].entities);
        free(temp_envs[i].active_agent_indices);
        free(temp_envs[i].static_car_indices);
    }
    free(temp_envs);
    */
    // return agent_offsets;
}

static double* unpack_float_array(PyObject* kwargs, char* key, Py_ssize_t* out_size) {
    PyObject* val = PyDict_GetItemString(kwargs, key);
    if (val == NULL) {
        char error_msg[100];
        snprintf(error_msg, sizeof(error_msg), "Missing required keyword argument '%s'", key);
        PyErr_SetString(PyExc_TypeError, error_msg);
        return NULL;
    }

    if (!PySequence_Check(val)) {
        char error_msg[100];
        snprintf(error_msg, sizeof(error_msg), "Argument '%s' must be a sequence", key);
        PyErr_SetString(PyExc_TypeError, error_msg);
        return NULL;
    }

    Py_ssize_t size = PySequence_Size(val);
    if (size < 0) {
        return NULL;
    }

    if (size == 0) {
        *out_size = 0;
        return NULL;
    }

    double* array = (double*)malloc(size * sizeof(double));
    if (array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for float array");
        return NULL;
    }


    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PySequence_GetItem(val, i);
        if (item == NULL) {
            free(array);
            return NULL;
        }

        double value;
        if (PyLong_Check(item)) {
            long long_val = PyLong_AsLong(item);
            if (long_val == -1 && PyErr_Occurred()) {
                Py_DECREF(item);
                free(array);
                return NULL;
            }
            value = (double)long_val;
        } else if (PyFloat_Check(item)) {
            value = PyFloat_AsDouble(item);
            if (value == -1.0 && PyErr_Occurred()) {
                Py_DECREF(item);
                free(array);
                return NULL;
            }
        } else {
            char error_msg[100];
            snprintf(error_msg, sizeof(error_msg), "Element %zd in '%s' is not a number", i, key);
            PyErr_SetString(PyExc_TypeError, error_msg);
            Py_DECREF(item);
            free(array);
            return NULL;
        }

        array[i] = value;
        Py_DECREF(item);
    }

    return array;
}

static PyObject* my_shared_population_play(PyObject* self, PyObject* args, PyObject* kwargs) {
    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");
    int num_ego_agents = unpack(kwargs, "num_ego_agents");
    int population_play = unpack(kwargs, "population_play");

    int max_scenes_per_process = 0;
    PyObject* max_envs_obj = PyDict_GetItemString(kwargs, "max_scenes_per_process");
    if (max_envs_obj && PyLong_Check(max_envs_obj)) {
        long v = PyLong_AsLong(max_envs_obj);
        if (v > 0 && v <= INT_MAX) {
            max_scenes_per_process = (int)v;
        }
    }

    // Use current time for randomness
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    srand(ts.tv_nsec);

    int num_coplayers = num_agents - num_ego_agents;
    printf("Creating worlds for %d total agents (%d egos, %d co-players)\n",
           num_agents, num_ego_agents, num_coplayers);

    // Create shuffled agent role array (0 = coplayer, 1 = ego)
    int* agent_roles = malloc(num_agents * sizeof(int));
    for (int i = 0; i < num_ego_agents; i++) {
        agent_roles[i] = 1; // ego
    }
    for (int i = num_ego_agents; i < num_agents; i++) {
        agent_roles[i] = 0; // coplayer
    }

    // Fisher-Yates shuffle to randomize agent roles
    for (int i = num_agents - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = agent_roles[i];
        agent_roles[i] = agent_roles[j];
        agent_roles[j] = temp;
    }

    int total_agent_count = 0;
    int env_count = 0;
    int total_egos_assigned = 0;
    int total_coplayers_assigned = 0;

    int max_envs = num_agents;
    if (max_scenes_per_process > 0 && max_scenes_per_process < max_envs) {
        max_envs = max_scenes_per_process;
    }

    PyObject* agent_offsets = PyList_New(max_envs + 1);
    PyObject* map_ids = PyList_New(max_envs);
    PyObject* ego_agent_ids = PyList_New(max_envs);
    PyObject* coplayer_ids = PyList_New(max_envs);

    // Create worlds by randomly sampling maps
    while (total_agent_count < num_agents && env_count < max_envs) {
        char map_file[100];
        int map_id = rand() % num_maps;
        Drive* env = calloc(1, sizeof(Drive));
        sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);
        env->entities = load_map_binary(map_file, env);

        int remaining_capacity = num_agents - total_agent_count;
        if (remaining_capacity < 0) {
            remaining_capacity = 0;
        }
        env->num_agents = remaining_capacity;
        env->population_play = population_play;

        set_active_agents(env);

        int next_total = total_agent_count + env->active_agent_count;
        if (next_total > num_agents) {
            int remaining = num_agents - total_agent_count;
            fprintf(stderr,
                    "[shared_population_play] ERROR oversubscribed agents: requested=%d remaining=%d map=%d\n",
                    env->active_agent_count,
                    remaining,
                    map_id);
            for(int j=0; j<env->num_entities; j++) {
                free_entity(&env->entities[j]);
            }
            free(env->entities);
            free(env->active_agent_indices);
            free(env->static_car_indices);
            free(env->expert_static_car_indices);
            free(env);
            Py_DECREF(agent_offsets);
            Py_DECREF(map_ids);
            Py_DECREF(ego_agent_ids);
            Py_DECREF(coplayer_ids);
            free(agent_roles);
            PyErr_Format(PyExc_RuntimeError,
                         "shared_population_play oversubscribed: total=%d target=%d map=%d active=%d",
                         next_total,
                         num_agents,
                         map_id,
                         env->active_agent_count);
            return NULL;
        }

        // Store map_id
        PyObject* map_id_obj = PyLong_FromLong(map_id);
        PyList_SetItem(map_ids, env_count, map_id_obj);

        // Store agent offset
        PyObject* offset = PyLong_FromLong(total_agent_count);
        PyList_SetItem(agent_offsets, env_count, offset);

        // Create ego and coplayer lists for this world
        PyObject* ego_list = PyList_New(0);
        PyObject* coplayer_list = PyList_New(0);

        int world_egos = 0;
        int world_coplayers = 0;
        int remaining_egos = num_ego_agents - total_egos_assigned;

        // Assign agents from the shuffled roles
        for (int a = 0; a < env->active_agent_count; a++) {
            PyObject* agent_id = PyLong_FromLong(total_agent_count);

            if (agent_roles[total_agent_count] == 1) {
                // This agent is an ego
                PyList_Append(ego_list, agent_id);
                world_egos++;
                total_egos_assigned++;
            } else {
                // This agent is a coplayer
                PyList_Append(coplayer_list, agent_id);
                world_coplayers++;
                total_coplayers_assigned++;
            }

            Py_DECREF(agent_id);
            total_agent_count++;
        }

        // Enforce constraint: must have at least 1 ego per world (if egos remain)
        if (world_egos == 0 && remaining_egos > 0) {
            fprintf(stderr,
                    "[shared_population_play] WARNING: World %d has no ego agents but %d egos remain. Skipping world.\n",
                    env_count, remaining_egos);

            // Rollback the agent assignments for this world
            total_agent_count -= env->active_agent_count;
            total_coplayers_assigned -= world_coplayers;

            Py_DECREF(ego_list);
            Py_DECREF(coplayer_list);

            for(int j=0; j<env->num_entities; j++) {
                free_entity(&env->entities[j]);
            }
            free(env->entities);
            free(env->active_agent_indices);
            free(env->static_car_indices);
            free(env->expert_static_car_indices);
            free(env);
            continue; // Try another map
        }

        PyList_SetItem(ego_agent_ids, env_count, ego_list);
        PyList_SetItem(coplayer_ids, env_count, coplayer_list);

        printf("World %d (map %d): %d agents (%d egos, %d co-players)\n",
               env_count, map_id, env->active_agent_count, world_egos, world_coplayers);

        env_count++;

        for(int j=0; j<env->num_entities; j++) {
            free_entity(&env->entities[j]);
        }
        free(env->entities);
        free(env->active_agent_indices);
        free(env->static_car_indices);
        free(env->expert_static_car_indices);
        free(env);
    }

    if (total_agent_count >= num_agents) {
        total_agent_count = num_agents;
    }

    PyObject* final_total_agent_count = PyLong_FromLong(total_agent_count);
    PyList_SetItem(agent_offsets, env_count, final_total_agent_count);
    PyObject* final_env_count = PyLong_FromLong(env_count);

    // Resize lists
    PyObject* resized_agent_offsets = PyList_GetSlice(agent_offsets, 0, env_count + 1);
    PyObject* resized_map_ids = PyList_GetSlice(map_ids, 0, env_count);
    PyObject* resized_ego_ids = PyList_GetSlice(ego_agent_ids, 0, env_count);
    PyObject* resized_coplayer_ids = PyList_GetSlice(coplayer_ids, 0, env_count);

    Py_DECREF(agent_offsets);
    Py_DECREF(map_ids);
    Py_DECREF(ego_agent_ids);
    Py_DECREF(coplayer_ids);

    // Free the shuffled roles array
    free(agent_roles);

    // Create a tuple
    PyObject* tuple = PyTuple_New(5);
    PyTuple_SetItem(tuple, 0, resized_agent_offsets);
    PyTuple_SetItem(tuple, 1, resized_map_ids);
    PyTuple_SetItem(tuple, 2, final_env_count);
    PyTuple_SetItem(tuple, 3, resized_ego_ids);
    PyTuple_SetItem(tuple, 4, resized_coplayer_ids);

    printf("Total: %d agents across %d worlds (egos: %d, co-players: %d)\n",
           total_agent_count, env_count, total_egos_assigned, total_coplayers_assigned);

    return tuple;
}

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {

    int population_play = unpack(kwargs, "population_play");
    if (population_play){
        return my_shared_population_play(self, args,  kwargs);
    }
    else{
        return my_shared_self_play( self, args,  kwargs);
    }

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
    env->goal_radius = conf.goal_radius;
    env->scenario_length = conf.scenario_length;
    env->use_goal_generation = conf.use_goal_generation;
    env->policy_agents_per_env = unpack(kwargs, "num_policy_controlled_agents");
    env->control_all_agents = unpack(kwargs, "control_all_agents");
    env->deterministic_agent_selection = unpack(kwargs, "deterministic_agent_selection");
    env->dt = conf.dt;
    env->control_non_vehicles = (int)unpack(kwargs, "control_non_vehicles");

    // Conditioning parameters
    env->use_rc = (bool)unpack(kwargs, "use_rc");
    env->use_ec = (bool)unpack(kwargs, "use_ec");
    env->use_dc = (bool)unpack(kwargs, "use_dc");
    env->collision_weight_lb = (float)unpack(kwargs, "collision_weight_lb");
    env->collision_weight_ub = (float)unpack(kwargs, "collision_weight_ub");
    env->offroad_weight_lb = (float)unpack(kwargs, "offroad_weight_lb");
    env->offroad_weight_ub = (float)unpack(kwargs, "offroad_weight_ub");
    env->goal_weight_lb = (float)unpack(kwargs, "goal_weight_lb");
    env->goal_weight_ub = (float)unpack(kwargs, "goal_weight_ub");
    env->entropy_weight_lb = (float)unpack(kwargs, "entropy_weight_lb");
    env->entropy_weight_ub = (float)unpack(kwargs, "entropy_weight_ub");
    env->discount_weight_lb = (float)unpack(kwargs, "discount_weight_lb");
    env->discount_weight_ub = (float)unpack(kwargs, "discount_weight_ub");

    int map_id = unpack(kwargs, "map_id");
    int max_agents = unpack(kwargs, "max_agents");
    int population_play = unpack(kwargs, "population_play");
    int adaptive_driving = unpack(kwargs, "adaptive_driving");
    int k_scenarios = unpack(kwargs, "k_scenarios");


    env->adaptive_driving_agent = adaptive_driving;



    if (env->adaptive_driving_agent) {
        env->k_scenarios = k_scenarios;
        env->current_scenario = 0;
    } else {
        env->k_scenarios = 0;
    }


    env->population_play = population_play;


    if (env->population_play) {
        env->num_co_players = unpack(kwargs, "num_co_players");
        double* co_player_ids_d = unpack_float_array(kwargs, "co_player_ids", &env->num_co_players);
        if (co_player_ids_d != NULL && env->num_co_players > 0) {
            env->co_player_ids = (int*)malloc(env->num_co_players * sizeof(int));
            for (int i = 0; i < env->num_co_players; i++) {
                env->co_player_ids[i] = (int)co_player_ids_d[i];
            }
            free(co_player_ids_d);
        } else {
            env->co_player_ids = NULL;
            env->num_co_players = 0;
        }

        // Handle ego agents - always as an array
        env->num_ego_agents = unpack(kwargs, "num_ego_agents");
        if (env->num_ego_agents > 0) {
            double* ego_agent_ids_d = unpack_float_array(kwargs, "ego_agent_ids", &env->num_ego_agents);
            if (ego_agent_ids_d != NULL) {
                env->ego_agent_ids = (int*)malloc(env->num_ego_agents * sizeof(int));
                for (int i = 0; i < env->num_ego_agents; i++) {
                    env->ego_agent_ids[i] = (int)ego_agent_ids_d[i];
                }
                free(ego_agent_ids_d);
            } else {
                env->ego_agent_ids = NULL;
                env->num_ego_agents = 0;
            }
        } else {
            env->ego_agent_ids = NULL;
            env->num_ego_agents = 0;
        }
    } else {
        // Non-population play mode - set defaults
        env->num_ego_agents = 0;
        env->ego_agent_ids = NULL;
    }

    int init_steps = unpack(kwargs, "init_steps");
    char map_file[100];
    sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);
    env->num_agents = max_agents;
    env->map_name = strdup(map_file);
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
    //assign_to_dict(dict, "num_goals_reached", log->num_goals_reached);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    assign_to_dict(dict, "lane_alignment_rate", log->lane_alignment_rate);
    assign_to_dict(dict, "score", log->score);
    // assign_to_dict(dict, "active_agent_count", log->active_agent_count);
    // assign_to_dict(dict, "expert_static_car_count", log->expert_static_car_count);
    // assign_to_dict(dict, "static_car_count", log->static_car_count);
    assign_to_dict(dict, "avg_offroad_per_agent", log->avg_offroad_per_agent);
    assign_to_dict(dict, "avg_collisions_per_agent", log->avg_collisions_per_agent);
    return 0;
}
