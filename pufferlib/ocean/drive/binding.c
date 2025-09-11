#include "drive.h"
#define Env Drive
#define MY_SHARED
#define MY_PUT
#define MY_GET
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

static PyObject* my_get(PyObject* dict, Env* env) {
    PyObject* v;
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "env is NULL");
        return NULL;
    }

    v = PyLong_FromLong(env->active_agent_count);
    if (!v) return NULL;
    if (PyDict_SetItemString(dict, "active_agent_count", v) < 0) { Py_DECREF(v); return NULL; }
    Py_DECREF(v);

    v = PyLong_FromLong(env->num_entities);
    if (!v) return NULL;
    if (PyDict_SetItemString(dict, "num_entities", v) < 0) { Py_DECREF(v); return NULL; }
    Py_DECREF(v);

    /* Map name / string fields */
    if (env->map_name) {
        PyObject* s = PyUnicode_FromString(env->map_name);
        if (!s) return NULL;
        if (PyDict_SetItemString(dict, "map_name", s) < 0) { Py_DECREF(s); return NULL; }
        Py_DECREF(s);
    } else {
        if (PyDict_SetItemString(dict, "map_name", Py_None) < 0) return NULL;
    }

    /* Lists (active agent indices) */
    if (env->active_agent_indices && env->active_agent_count > 0) {
        PyObject* lst = PyList_New(env->active_agent_count);
        if (!lst) return NULL;
        for (int i = 0; i < env->active_agent_count; i++) {
            PyObject* it = PyLong_FromLong(env->active_agent_indices[i]);
            if (!it) { Py_DECREF(lst); return NULL; }
            /* PyList_SetItem steals reference */
            PyList_SetItem(lst, i, it);
        }
        if (PyDict_SetItemString(dict, "active_agent_indices", lst) < 0) { Py_DECREF(lst); return NULL; }
        Py_DECREF(lst);
    } else {
        if (PyDict_SetItemString(dict, "active_agent_indices", Py_None) < 0) return NULL;
    }

    /* Optionally expose static car indices if present */
    if (env->static_car_indices && env->static_car_count > 0) {
        PyObject* lst = PyList_New(env->static_car_count);
        if (!lst) return NULL;
        for (int i = 0; i < env->static_car_count; i++) {
            PyObject* it = PyLong_FromLong(env->static_car_indices[i]);
            if (!it) { Py_DECREF(lst); return NULL; }
            PyList_SetItem(lst, i, it);
        }
        if (PyDict_SetItemString(dict, "static_car_indices", lst) < 0) { Py_DECREF(lst); return NULL; }
        Py_DECREF(lst);
    } else {
        if (PyDict_SetItemString(dict, "static_car_indices", Py_None) < 0) return NULL;
    }

    /* Expose entities array as a list of dicts */
    if (env->entities && env->num_entities > 0) {
        PyObject* ent_list = PyList_New(env->num_entities);
        if (!ent_list) return NULL;
        for (int i = 0; i < env->num_entities; i++) {
            Entity* e = &env->entities[i];
            PyObject* ent = PyDict_New();
            if (!ent) { Py_DECREF(ent_list); return NULL; }

            /* scalar ints */
            PyObject* tmp = PyLong_FromLong(e->type);
            if (!tmp) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
            PyDict_SetItemString(ent, "type", tmp); Py_DECREF(tmp);

            tmp = PyLong_FromLong(e->array_size);
            if (!tmp) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
            PyDict_SetItemString(ent, "array_size", tmp); Py_DECREF(tmp);

            /* trajectory float arrays: traj_x, traj_y, traj_z */
            if (e->traj_x && e->array_size > 0) {
                PyObject* lx = PyList_New(e->array_size);
                if (!lx) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* fv = PyFloat_FromDouble((double)e->traj_x[j]);
                    if (!fv) { Py_DECREF(lx); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(lx, j, fv); /* steals ref */
                }
                PyDict_SetItemString(ent, "traj_x", lx); Py_DECREF(lx);
            } else {
                PyDict_SetItemString(ent, "traj_x", Py_None);
            }
            if (e->traj_y && e->array_size > 0) {
                PyObject* ly = PyList_New(e->array_size);
                if (!ly) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* fv = PyFloat_FromDouble((double)e->traj_y[j]);
                    if (!fv) { Py_DECREF(ly); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(ly, j, fv);
                }
                PyDict_SetItemString(ent, "traj_y", ly); Py_DECREF(ly);
            } else {
                PyDict_SetItemString(ent, "traj_y", Py_None);
            }
            if (e->traj_z && e->array_size > 0) {
                PyObject* lz = PyList_New(e->array_size);
                if (!lz) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* fv = PyFloat_FromDouble((double)e->traj_z[j]);
                    if (!fv) { Py_DECREF(lz); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(lz, j, fv);
                }
                PyDict_SetItemString(ent, "traj_z", lz); Py_DECREF(lz);
            } else {
                PyDict_SetItemString(ent, "traj_z", Py_None);
            }

            /* optional velocity / heading / valid arrays for objects */
            if (e->traj_vx && e->array_size > 0) {
                PyObject* lvx = PyList_New(e->array_size);
                if (!lvx) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* fv = PyFloat_FromDouble((double)e->traj_vx[j]);
                    if (!fv) { Py_DECREF(lvx); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(lvx, j, fv);
                }
                PyDict_SetItemString(ent, "traj_vx", lvx); Py_DECREF(lvx);
            } else {
                PyDict_SetItemString(ent, "traj_vx", Py_None);
            }
            if (e->traj_vy && e->array_size > 0) {
                PyObject* lvy = PyList_New(e->array_size);
                if (!lvy) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* fv = PyFloat_FromDouble((double)e->traj_vy[j]);
                    if (!fv) { Py_DECREF(lvy); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(lvy, j, fv);
                }
                PyDict_SetItemString(ent, "traj_vy", lvy); Py_DECREF(lvy);
            } else {
                PyDict_SetItemString(ent, "traj_vy", Py_None);
            }
            if (e->traj_vz && e->array_size > 0) {
                PyObject* lvz = PyList_New(e->array_size);
                if (!lvz) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* fv = PyFloat_FromDouble((double)e->traj_vz[j]);
                    if (!fv) { Py_DECREF(lvz); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(lvz, j, fv);
                }
                PyDict_SetItemString(ent, "traj_vz", lvz); Py_DECREF(lvz);
            } else {
                PyDict_SetItemString(ent, "traj_vz", Py_None);
            }
            if (e->traj_heading && e->array_size > 0) {
                PyObject* lhd = PyList_New(e->array_size);
                if (!lhd) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* fv = PyFloat_FromDouble((double)e->traj_heading[j]);
                    if (!fv) { Py_DECREF(lhd); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(lhd, j, fv);
                }
                PyDict_SetItemString(ent, "traj_heading", lhd); Py_DECREF(lhd);
            } else {
                PyDict_SetItemString(ent, "traj_heading", Py_None);
            }
            if (e->traj_valid && e->array_size > 0) {
                PyObject* lval = PyList_New(e->array_size);
                if (!lval) { Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                for (int j = 0; j < e->array_size; j++) {
                    PyObject* iv = PyLong_FromLong(e->traj_valid[j]);
                    if (!iv) { Py_DECREF(lval); Py_DECREF(ent); Py_DECREF(ent_list); return NULL; }
                    PyList_SetItem(lval, j, iv);
                }
                PyDict_SetItemString(ent, "traj_valid", lval); Py_DECREF(lval);
            } else {
                PyDict_SetItemString(ent, "traj_valid", Py_None);
            }

            /* scalar floats */
            PyObject* pf = PyFloat_FromDouble((double)e->width);
            PyDict_SetItemString(ent, "width", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->length);
            PyDict_SetItemString(ent, "length", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->height);
            PyDict_SetItemString(ent, "height", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->goal_position_x);
            PyDict_SetItemString(ent, "goal_position_x", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->goal_position_y);
            PyDict_SetItemString(ent, "goal_position_y", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->goal_position_z);
            PyDict_SetItemString(ent, "goal_position_z", pf); Py_DECREF(pf);

            /* other scalar int/float fields */
            tmp = PyLong_FromLong(e->mark_as_expert);
            PyDict_SetItemString(ent, "mark_as_expert", tmp); Py_DECREF(tmp);

            tmp = PyLong_FromLong(e->collision_state);
            PyDict_SetItemString(ent, "collision_state", tmp); Py_DECREF(tmp);

            pf = PyFloat_FromDouble((double)e->x); PyDict_SetItemString(ent, "x", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->y); PyDict_SetItemString(ent, "y", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->z); PyDict_SetItemString(ent, "z", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->vx); PyDict_SetItemString(ent, "vx", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->vy); PyDict_SetItemString(ent, "vy", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->vz); PyDict_SetItemString(ent, "vz", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->heading); PyDict_SetItemString(ent, "heading", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->heading_x); PyDict_SetItemString(ent, "heading_x", pf); Py_DECREF(pf);
            pf = PyFloat_FromDouble((double)e->heading_y); PyDict_SetItemString(ent, "heading_y", pf); Py_DECREF(pf);

            tmp = PyLong_FromLong(e->valid); PyDict_SetItemString(ent, "valid", tmp); Py_DECREF(tmp);
            tmp = PyLong_FromLong(e->collided_before_goal); PyDict_SetItemString(ent, "collided_before_goal", tmp); Py_DECREF(tmp);
            tmp = PyLong_FromLong(e->reached_goal_this_episode); PyDict_SetItemString(ent, "reached_goal_this_episode", tmp); Py_DECREF(tmp);
            tmp = PyLong_FromLong(e->active_agent); PyDict_SetItemString(ent, "active_agent", tmp); Py_DECREF(tmp);

            /* Steal reference into list */
            PyList_SetItem(ent_list, i, ent);
        }
        if (PyDict_SetItemString(dict, "entities", ent_list) < 0) { Py_DECREF(ent_list); return NULL; }
        Py_DECREF(ent_list);
    } else {
        if (PyDict_SetItemString(dict, "entities", Py_None) < 0) return NULL;
    }

    /* Grid information */
    v = PyLong_FromLong(env->grid_cols);
    if (!v) return NULL;
    if (PyDict_SetItemString(dict, "grid_cols", v) < 0) { Py_DECREF(v); return NULL; }
    Py_DECREF(v);

    v = PyLong_FromLong(env->grid_rows);
    if (!v) return NULL;
    if (PyDict_SetItemString(dict, "grid_rows", v) < 0) { Py_DECREF(v); return NULL; }
    Py_DECREF(v);

    /* Map corners (bounding box) */
    if (env->map_corners) {
        PyObject* corners_list = PyList_New(4);
        if (!corners_list) return NULL;
        for (int i = 0; i < 4; i++) {
            PyObject* corner = PyFloat_FromDouble((double)env->map_corners[i]);
            if (!corner) { Py_DECREF(corners_list); return NULL; }
            PyList_SetItem(corners_list, i, corner);
        }
        if (PyDict_SetItemString(dict, "map_corners", corners_list) < 0) { Py_DECREF(corners_list); return NULL; }
        Py_DECREF(corners_list);
    } else {
        if (PyDict_SetItemString(dict, "map_corners", Py_None) < 0) return NULL;
    }

    /* Grid cells data */
    if (env->grid_cells && env->grid_cols > 0 && env->grid_rows > 0) {
        int total_grid_cells = env->grid_cols * env->grid_rows;
        PyObject* grid_data = PyList_New(total_grid_cells);
        if (!grid_data) return NULL;

        for (int i = 0; i < total_grid_cells; i++) {
            int base_index = i * 21; // SLOTS_PER_CELL = 21 (MAX_ENTITIES_PER_CELL*2 + 1)
            int count = env->grid_cells[base_index];

            PyObject* cell_data = PyDict_New();
            if (!cell_data) { Py_DECREF(grid_data); return NULL; }

            PyObject* cell_count = PyLong_FromLong(count);
            if (!cell_count) { Py_DECREF(cell_data); Py_DECREF(grid_data); return NULL; }
            PyDict_SetItemString(cell_data, "count", cell_count); Py_DECREF(cell_count);

            if (count > 0) {
                PyObject* entities = PyList_New(count);
                PyObject* geometry_indices = PyList_New(count);
                if (!entities || !geometry_indices) {
                    Py_XDECREF(entities); Py_XDECREF(geometry_indices);
                    Py_DECREF(cell_data); Py_DECREF(grid_data); return NULL;
                }

                for (int j = 0; j < count; j++) {
                    int entity_idx = env->grid_cells[base_index + j*2 + 1];
                    int geometry_idx = env->grid_cells[base_index + j*2 + 2];

                    PyObject* ent_id = PyLong_FromLong(entity_idx);
                    PyObject* geom_id = PyLong_FromLong(geometry_idx);
                    if (!ent_id || !geom_id) {
                        Py_XDECREF(ent_id); Py_XDECREF(geom_id);
                        Py_DECREF(entities); Py_DECREF(geometry_indices);
                        Py_DECREF(cell_data); Py_DECREF(grid_data); return NULL;
                    }

                    PyList_SetItem(entities, j, ent_id);
                    PyList_SetItem(geometry_indices, j, geom_id);
                }

                PyDict_SetItemString(cell_data, "entities", entities); Py_DECREF(entities);
                PyDict_SetItemString(cell_data, "geometry_indices", geometry_indices); Py_DECREF(geometry_indices);
            } else {
                PyDict_SetItemString(cell_data, "entities", Py_None);
                PyDict_SetItemString(cell_data, "geometry_indices", Py_None);
            }

            PyList_SetItem(grid_data, i, cell_data);
        }

        if (PyDict_SetItemString(dict, "grid_cells", grid_data) < 0) { Py_DECREF(grid_data); return NULL; }
        Py_DECREF(grid_data);
    } else {
        if (PyDict_SetItemString(dict, "grid_cells", Py_None) < 0) return NULL;
    }

    /* Agent observations */
    if (env->observations && env->active_agent_count > 0) {
        int max_obs = 7 + 7*(64 - 1) + 7*200; // 7 + 7*(MAX_CARS - 1) + 7*MAX_ROAD_SEGMENT_OBSERVATIONS
        PyObject* obs_data = PyList_New(env->active_agent_count);
        if (!obs_data) return NULL;

        float (*observations)[max_obs] = (float(*)[max_obs])env->observations;

        for (int i = 0; i < env->active_agent_count; i++) {
            PyObject* agent_obs = PyList_New(max_obs);
            if (!agent_obs) { Py_DECREF(obs_data); return NULL; }

            for (int j = 0; j < max_obs; j++) {
                PyObject* obs_val = PyFloat_FromDouble((double)observations[i][j]);
                if (!obs_val) { Py_DECREF(agent_obs); Py_DECREF(obs_data); return NULL; }
                PyList_SetItem(agent_obs, j, obs_val);
            }

            PyList_SetItem(obs_data, i, agent_obs);
        }

        if (PyDict_SetItemString(dict, "agent_observations", obs_data) < 0) { Py_DECREF(obs_data); return NULL; }
        Py_DECREF(obs_data);
    } else {
        if (PyDict_SetItemString(dict, "agent_observations", Py_None) < 0) return NULL;
    }

    return dict;
}

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
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

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->reward_vehicle_collision = unpack(kwargs, "reward_vehicle_collision");
    env->reward_offroad_collision = unpack(kwargs, "reward_offroad_collision");
    env->reward_goal_post_respawn = unpack(kwargs, "reward_goal_post_respawn");
    env->reward_vehicle_collision_post_respawn = unpack(kwargs, "reward_vehicle_collision_post_respawn");
    env->spawn_immunity_timer = unpack(kwargs, "spawn_immunity_timer");
    int map_id = unpack(kwargs, "map_id");
    int max_agents = unpack(kwargs, "max_agents");

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
    assign_to_dict(dict, "clean_collision_rate", log->clean_collision_rate);
    return 0;
}
