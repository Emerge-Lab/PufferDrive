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

static PyObject* my_shared_self_play(PyObject* self, PyObject* args, PyObject* kwargs){
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

static PyObject* my_shared_population_play(PyObject* self, PyObject* args, PyObject* kwargs) {
    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");
    
    clock_gettime(CLOCK_REALTIME, &ts);
    srand(ts.tv_nsec);
    
    // Create one world per agent (each agent gets their own world as ego)
    int num_worlds = num_agents;
    
    PyObject* agent_offsets = PyList_New(num_worlds + 1);
    PyObject* map_ids = PyList_New(num_worlds);
    PyObject* ego_agent_ids = PyList_New(num_worlds);
    PyObject* coplayer_offsets = PyList_New(num_worlds);
    
    int total_agent_count = 0;
    
    for(int world_idx = 0; world_idx < num_worlds; world_idx++) {
        char map_file[100];
        int map_id = rand() % num_maps;
        
        Drive* env = calloc(1, sizeof(Drive));
        sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);
        env->entities = load_map_binary(map_file, env);
        set_active_agents(env);
        
        // Skip maps with 0 active agents
        if (env->active_agent_count <= 0) {
            printf("Warning: Map %d has 0 active agents, skipping\n", map_id);
            
            // Cleanup
            if (env->entities) {
                for(int j = 0; j < env->num_entities; j++) {
                    free_entity(&env->entities[j]);
                }
                free(env->entities);
            }
            if (env->active_agent_indices) free(env->active_agent_indices);
            if (env->static_car_indices) free(env->static_car_indices);
            if (env->expert_static_car_indices) free(env->expert_static_car_indices);
            free(env);
            
            // Try again with a different map
            world_idx--; // Decrement to retry this world slot
            continue;
        }
        
        // Store map_id for this world
        PyObject* map_id_obj = PyLong_FromLong(map_id);
        PyList_SetItem(map_ids, world_idx, map_id_obj);
        
        // Store agent offset for this world
        PyObject* offset = PyLong_FromLong(total_agent_count);
        PyList_SetItem(agent_offsets, world_idx, offset);
        
        // Randomly choose which agent position will be the ego (0 to active_agent_count-1)
        int ego_position = rand() % env->active_agent_count;
        
        // Create lists for ego and co-players
        PyObject* coplayers = PyList_New(env->active_agent_count - 1);
        int ego_agent_id = -1;
        int coplayer_idx = 0;
        
        // Assign agent IDs, with one randomly chosen as ego
        for(int agent_pos = 0; agent_pos < env->active_agent_count; agent_pos++) {
            if(agent_pos == ego_position) {
                // This agent is the ego
                ego_agent_id = total_agent_count;
            } else {
                // This agent is a co-player
                PyObject* coplayer_id = PyLong_FromLong(total_agent_count);
                PyList_SetItem(coplayers, coplayer_idx, coplayer_id);
                coplayer_idx++;
            }
            total_agent_count++;
        }
        
        // Store the ego agent ID for this world
        PyObject* ego_id = PyLong_FromLong(ego_agent_id);
        PyList_SetItem(ego_agent_ids, world_idx, ego_id);
        
        // Store co-players for this world
        PyList_SetItem(coplayer_offsets, world_idx, coplayers);
        
        // Cleanup
        if (env->entities) {
            for(int j = 0; j < env->num_entities; j++) {
                free_entity(&env->entities[j]);
            }
            free(env->entities);
        }
        if (env->active_agent_indices) free(env->active_agent_indices);
        if (env->static_car_indices) free(env->static_car_indices);
        if (env->expert_static_car_indices) free(env->expert_static_car_indices);
        free(env);
    }
    
    // Store final total agent count
    PyObject* final_total_agent_count = PyLong_FromLong(total_agent_count);
    PyList_SetItem(agent_offsets, num_worlds, final_total_agent_count);
    
    PyObject* final_world_count = PyLong_FromLong(num_worlds);
    
    // Create return tuple with all the information
    PyObject* tuple = PyTuple_New(5);
    PyTuple_SetItem(tuple, 0, agent_offsets);
    PyTuple_SetItem(tuple, 1, map_ids);
    PyTuple_SetItem(tuple, 2, final_world_count);
    PyTuple_SetItem(tuple, 3, ego_agent_ids);
    PyTuple_SetItem(tuple, 4, coplayer_offsets);
    
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

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->reward_vehicle_collision = unpack(kwargs, "reward_vehicle_collision");
    env->reward_offroad_collision = unpack(kwargs, "reward_offroad_collision");
    env->reward_goal_post_respawn = unpack(kwargs, "reward_goal_post_respawn");
    env->reward_vehicle_collision_post_respawn = unpack(kwargs, "reward_vehicle_collision_post_respawn");
    env->spawn_immunity_timer = unpack(kwargs, "spawn_immunity_timer");
    env->use_rc = unpack(kwargs, "use_rc");
    if (env->use_rc){
        env->collision_weight_lb = unpack(kwargs, "collision_weight_lb");
        env->collision_weight_ub = unpack(kwargs, "collision_weight_ub");
        env->offroad_weight_lb = unpack(kwargs, "offroad_weight_lb");
        env->offroad_weight_ub = unpack(kwargs, "offroad_weight_ub");
        env->goal_weight_lb = unpack(kwargs, "goal_weight_lb");
        env->goal_weight_ub = unpack(kwargs, "goal_weight_ub");
    }
    env->use_ec = unpack(kwargs, "use_ec");
    if (env->use_ec){
        env->entropy_weight_lb = unpack(kwargs, "entropy_weight_lb");
        env->entropy_weight_ub = unpack(kwargs, "entropy_weight_ub");
    }
    env->oracle_mode = unpack(kwargs, "oracle_mode");
    int map_id = unpack(kwargs, "map_id");
    int max_agents = unpack(kwargs, "max_agents");
    int population_play = unpack(kwargs, "population_play");

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
        
        env->ego_agent_id = unpack(kwargs, "ego_agent_id");
    }
    
    char map_file[100];
    sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);
    env->num_agents = max_agents;
    env->map_name = strdup(map_file);
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    return 0;
}
int my_co_player_log(PyObject* dict, Co_Player_Log* log) {
    assign_to_dict(dict, "co_player_completion_rate", log->co_player_completion_rate);
    assign_to_dict(dict, "co_player_collision_rate", log->co_player_collision_rate);
    assign_to_dict(dict, "co_player_off_road_rate", log->co_player_off_road_rate);
    assign_to_dict(dict, "co_player_clean_collision_rate", log->co_player_clean_collision_rate);
    assign_to_dict(dict, "co_player_score", log->co_player_score);
    assign_to_dict(dict, "co_player_perf", log->co_player_perf);
    assign_to_dict(dict, "co_player_dnf_rate", log->co_player_dnf_rate);
    assign_to_dict(dict, "co_player_episode_length", log->co_player_episode_length);
    assign_to_dict(dict, "co_player_episode_return", log->co_player_episode_return);
    assign_to_dict(dict, "co_player_n", log->co_player_n);
    return 0;
}
