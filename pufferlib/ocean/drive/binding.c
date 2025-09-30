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
    int num_agents = unpack(kwargs, "num_agents");  // Number of EGO agents desired
    int num_maps = unpack(kwargs, "num_maps");
    double ego_probability = unpack(kwargs, "ego_probability");
    
    clock_gettime(CLOCK_REALTIME, &ts);
    srand(ts.tv_nsec);
    
    // Pick one random map to use
    int map_id = rand() % num_maps;
    char map_file[100];
    sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);
    
    // Load the map once to see how many agents it has
    Drive* env = calloc(1, sizeof(Drive));
    env->entities = load_map_binary(map_file, env);
    set_active_agents(env);
    
    int agents_per_world = env->active_agent_count;
    
    // Clean up the temporary env
    for(int j = 0; j < env->num_entities; j++) {
        free_entity(&env->entities[j]);
    }
    free(env->entities);
    free(env->active_agent_indices);
    free(env->static_car_indices);
    free(env->expert_static_car_indices);
    free(env);
    
    if(agents_per_world == 0) {
        PyErr_SetString(PyExc_RuntimeError, "Selected map has no agents");
        return NULL;
    }
    
    // Calculate how many worlds we need to get enough agents
    // We need more total agents than num_agents to have co-players
    int estimated_total_agents = (int)(num_agents / ego_probability) + agents_per_world;
    int num_worlds = (estimated_total_agents + agents_per_world - 1) / agents_per_world;  // Ceiling division
    
    int total_agents = num_worlds * agents_per_world;
    
    printf("Using map %d with %d agents per world\n", map_id, agents_per_world);
    printf("Creating %d worlds for %d total agents\n", num_worlds, total_agents);
    
    // Assign ego/co-player status to each agent
    int* is_ego = calloc(total_agents, sizeof(int));
    int ego_count = 0;
    
    // First pass: assign based on probability
    for (int i = 0; i < total_agents; i++) {
        double rand_val = (double)rand() / RAND_MAX;
        if (rand_val < ego_probability) {
            is_ego[i] = 1;
            ego_count++;
        }
    }
    
    // Adjust to get exactly num_agents egos
    while (ego_count < num_agents) {
        int idx = rand() % total_agents;
        if (!is_ego[idx]) {
            is_ego[idx] = 1;
            ego_count++;
        }
    }
    
    while (ego_count > num_agents) {
        int idx = rand() % total_agents;
        if (is_ego[idx]) {
            is_ego[idx] = 0;
            ego_count--;
        }
    }
    
    // Apply per-world constraints
    for (int w = 0; w < num_worlds; w++) {
        int world_start = w * agents_per_world;
        int world_end = world_start + agents_per_world;
        
        int num_egos = 0;
        int num_coplayers = 0;
        
        // Count egos and co-players in this world
        for (int i = world_start; i < world_end; i++) {
            if (is_ego[i]) {
                num_egos++;
            } else {
                num_coplayers++;
            }
        }
        
        // Constraint 1: At least 1 ego per world
        if (num_egos == 0) {
            // Promote a co-player to ego
            for (int i = world_start; i < world_end; i++) {
                if (!is_ego[i]) {
                    is_ego[i] = 1;
                    ego_count++;
                    num_egos++;
                    num_coplayers--;
                    break;
                }
            }
        }
        
        // Constraint 2: If 2+ agents, at least 1 co-player
        if (agents_per_world >= 2 && num_coplayers == 0) {
            // Demote an ego to co-player
            for (int i = world_start; i < world_end; i++) {
                if (is_ego[i]) {
                    is_ego[i] = 0;
                    ego_count--;
                    num_egos--;
                    num_coplayers++;
                    break;
                }
            }
        }
    }
    
    // Final rebalance to exactly num_agents egos while respecting constraints
    while (ego_count < num_agents) {
        int found = 0;
        for (int w = 0; w < num_worlds && !found; w++) {
            int world_start = w * agents_per_world;
            int world_end = world_start + agents_per_world;
            
            // Count co-players in this world
            int coplayer_count = 0;
            for (int i = world_start; i < world_end; i++) {
                if (!is_ego[i]) coplayer_count++;
            }
            
            // Can we promote a co-player? (need to keep at least 1 if world has 2+ agents)
            if (agents_per_world < 2 || coplayer_count > 1) {
                for (int i = world_start; i < world_end; i++) {
                    if (!is_ego[i]) {
                        is_ego[i] = 1;
                        ego_count++;
                        found = 1;
                        break;
                    }
                }
            }
        }
        if (!found) break;
    }
    
    while (ego_count > num_agents) {
        int found = 0;
        for (int w = 0; w < num_worlds && !found; w++) {
            int world_start = w * agents_per_world;
            int world_end = world_start + agents_per_world;
            
            // Count egos in this world
            int ego_count_in_world = 0;
            for (int i = world_start; i < world_end; i++) {
                if (is_ego[i]) ego_count_in_world++;
            }
            
            // Can we demote an ego? (need to keep at least 1)
            if (ego_count_in_world > 1) {
                for (int i = world_start; i < world_end; i++) {
                    if (is_ego[i]) {
                        is_ego[i] = 0;
                        ego_count--;
                        found = 1;
                        break;
                    }
                }
            }
        }
        if (!found) break;
    }
    
    // Build output lists (similar to my_shared_self_play)
    PyObject* agent_offsets = PyList_New(num_worlds + 1);
    PyObject* map_ids = PyList_New(num_worlds);
    PyObject* ego_agent_ids = PyList_New(num_worlds);
    PyObject* coplayer_offsets = PyList_New(num_worlds);
    
    int current_agent_id = 0;
    int total_egos = 0;
    int total_coplayers = 0;
    
    for (int w = 0; w < num_worlds; w++) {
        // Store agent offset for this world
        PyList_SetItem(agent_offsets, w, PyLong_FromLong(current_agent_id));
        
        // Store map_id (same map for all worlds)
        PyList_SetItem(map_ids, w, PyLong_FromLong(map_id));
        
        // Build ego and co-player lists for this world
        PyObject* ego_list = PyList_New(0);
        PyObject* coplayer_list = PyList_New(0);
        
        int world_start = w * agents_per_world;
        int world_end = world_start + agents_per_world;
        
        for (int i = world_start; i < world_end; i++) {
            PyObject* agent_id = PyLong_FromLong(current_agent_id);
            
            if (is_ego[i]) {
                PyList_Append(ego_list, agent_id);
                total_egos++;
            } else {
                PyList_Append(coplayer_list, agent_id);
                total_coplayers++;
            }
            
            Py_DECREF(agent_id);
            current_agent_id++;
        }
        
        PyList_SetItem(ego_agent_ids, w, ego_list);
        PyList_SetItem(coplayer_offsets, w, coplayer_list);
    }
    
    // Store final total agent count
    PyList_SetItem(agent_offsets, num_worlds, PyLong_FromLong(current_agent_id));
    
    // Create return tuple
    PyObject* tuple = PyTuple_New(5);
    PyTuple_SetItem(tuple, 0, agent_offsets);
    PyTuple_SetItem(tuple, 1, map_ids);
    PyTuple_SetItem(tuple, 2, PyLong_FromLong(num_worlds));
    PyTuple_SetItem(tuple, 3, ego_agent_ids);
    PyTuple_SetItem(tuple, 4, coplayer_offsets);
    
    // Cleanup
    free(is_ego);
    
    printf("Total worlds: %d, Total agents: %d (egos: %d, co-players: %d)\n", 
           num_worlds, current_agent_id, total_egos, total_coplayers);
    
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
    env->ini_file = unpack_str(kwargs, "ini_file");
    env_init_config conf;
    if(ini_parse(env->ini_file, handler, &conf) < 0) {
        printf("Error while loading %s", env->ini_file);
    }
    env->action_type = conf.action_type;
    env->reward_vehicle_collision = conf.reward_vehicle_collision;
    env->reward_offroad_collision = conf.reward_offroad_collision;
    env->reward_goal_post_respawn = conf.reward_goal_post_respawn;
    env->reward_vehicle_collision_post_respawn = conf.reward_vehicle_collision_post_respawn;
    env->reward_ade = conf.reward_ade;
    env->goal_radius = conf.goal_radius;
    env->spawn_immunity_timer = conf.spawn_immunity_timer;
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
    assign_to_dict(dict, "lane_alignment_rate", log->lane_alignment_rate);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    assign_to_dict(dict, "clean_collision_rate", log->clean_collision_rate);
    assign_to_dict(dict, "avg_displacement_error", log->avg_displacement_error);
    return 0;
}
