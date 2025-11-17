#include "drive.h"
#include "../env_binding.h"

static PyObject* my_shared_self_play(PyObject* self, PyObject* args, PyObject* kwargs) {
    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");
    int init_mode = unpack(kwargs, "init_mode");
    int control_mode = unpack(kwargs, "control_mode");
    int init_steps = unpack(kwargs, "init_steps");
    int max_controlled_agents = unpack(kwargs, "max_controlled_agents");
    clock_gettime(CLOCK_REALTIME, &ts);
    srand(ts.tv_nsec);
    int total_agent_count = 0;
    int env_count = 0;
    int max_envs = num_agents;
    int maps_checked = 0;
    PyObject* agent_offsets = PyList_New(max_envs+1);
    PyObject* map_ids = PyList_New(max_envs);
    // getting env count
    while(total_agent_count < num_agents && env_count < max_envs){
        char map_file[100];
        int map_id = rand() % num_maps;
        Drive* env = calloc(1, sizeof(Drive));
        env->init_mode = init_mode;
        env->control_mode = control_mode;
        env->init_steps = init_steps;
        env->max_controlled_agents = max_controlled_agents;
        sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);
        env->entities = load_map_binary(map_file, env);
        set_active_agents(env);

        // Skip map if it doesn't contain any controllable agents
        if(env->active_agent_count == 0) {
            maps_checked++;

            // Safeguard: if we've checked all available maps and found no active agents, raise an error
            if(maps_checked >= num_maps) {
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
                char error_msg[256];
                sprintf(error_msg, "No controllable agents found in any of the %d available maps", num_maps);
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
    return tuple;
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
     int init_mode = unpack(kwargs, "init_mode");
    int population_play = unpack(kwargs, "population_play");
    int control_mode = unpack(kwargs, "control_mode");
    int init_steps = unpack(kwargs, "init_steps");
    int max_controlled_agents = unpack(kwargs, "max_controlled_agents");

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
        env->init_mode = init_mode;
        env->control_mode = control_mode;
        env->init_steps = init_steps;
        env->max_controlled_agents = max_controlled_agents;

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
            free(env->static_agent_indices);
            free(env->expert_static_agent_indices);
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
            free(env->static_agent_indices);
            free(env->expert_static_agent_indices);
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
        free(env->static_agent_indices);
        free(env->expert_static_agent_indices);
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
