#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>

static void my_log(PyObject *dict, Drive *env, Log *log, float n);
static int my_init(Drive *env, PyObject *kwargs);
static int my_episode_to_dict(PyObject *dict, Drive *env);
static int assign_to_dict(PyObject *dict, char *key, float value);

static PyObject *my_shared(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *my_get(Drive *env);
static void my_reseed(Drive *env, uint64_t seed);

// env_init's positional NumPy buffers, in argument order; the seed follows them
#define ENV_BUFFER_COUNT 6

static PyObject *env_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    if (PyTuple_Size(args) != 7) {
        PyErr_SetString(PyExc_TypeError, "Environment requires 7 positional arguments");
        return NULL;
    }

    const struct {
        const char *name;
        int one_dimensional;
        int reject_float64;
    } buffer_specs[ENV_BUFFER_COUNT] = {
        {"Observations", 0, 0},
        {"Actions", 0, 1},
        {"Rewards", 1, 0},
        {"Terminals", 1, 0},
        {"Truncations", 1, 0},
        {"Masks", 0, 0},
    };
    void *buffer_data[ENV_BUFFER_COUNT];
    for (int buffer_idx = 0; buffer_idx < ENV_BUFFER_COUNT; buffer_idx++) {
        PyObject *buffer = PyTuple_GetItem(args, buffer_idx);
        const char *buffer_name = buffer_specs[buffer_idx].name;
        if (!PyObject_TypeCheck(buffer, &PyArray_Type)) {
            PyErr_Format(PyExc_TypeError, "%s must be a NumPy array", buffer_name);
            return NULL;
        }
        PyArrayObject *array = (PyArrayObject *) buffer;
        if (!PyArray_ISCONTIGUOUS(array)) {
            PyErr_Format(PyExc_ValueError, "%s must be contiguous", buffer_name);
            return NULL;
        }
        if (buffer_specs[buffer_idx].one_dimensional && PyArray_NDIM(array) != 1) {
            PyErr_Format(PyExc_ValueError, "%s must be 1D", buffer_name);
            return NULL;
        }
        if (buffer_specs[buffer_idx].reject_float64 && PyArray_ITEMSIZE(array) == sizeof(double)) {
            PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
            return NULL;
        }
        buffer_data[buffer_idx] = PyArray_DATA(array);
    }
    PyObject *seed_arg = PyTuple_GetItem(args, ENV_BUFFER_COUNT);
    if (!PyObject_TypeCheck(seed_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "seed must be an integer");
        return NULL;
    }
    long long seed = PyLong_AsLongLong(seed_arg);
    if (PyErr_Occurred()) {
        return NULL;
    }

    if (kwargs == NULL) {
        kwargs = PyDict_New();
    } else {
        Py_INCREF(kwargs);
    }

    PyObject *py_seed = PyLong_FromLongLong(seed);
    if (PyDict_SetItemString(kwargs, "seed", py_seed) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to set seed in kwargs");
        Py_DECREF(py_seed);
        Py_DECREF(kwargs);
        return NULL;
    }
    Py_DECREF(py_seed);

    Drive *env = (Drive *) calloc(1, sizeof(Drive));
    if (!env) {
        Py_DECREF(kwargs);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate environment");
        return NULL;
    }
    env->observations = buffer_data[0];
    env->actions = buffer_data[1];
    env->rewards = buffer_data[2];
    env->terminals = buffer_data[3];
    env->truncations = buffer_data[4];
    env->masks = buffer_data[5];

    int init_status = my_init(env, kwargs);
    Py_DECREF(kwargs);
    if (init_status != 0 || PyErr_Occurred()) {
        c_close(env);
        free(env);
        return NULL;
    }

    return PyLong_FromVoidPtr(env);
}

typedef struct {
    Drive **envs;
    int num_envs;
} VecEnv;

static VecEnv *unpack_vecenv(PyObject *args) {
    PyObject *handle_obj = PyTuple_GetItem(args, 0);
    if (handle_obj == NULL) {
        return NULL;
    }
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_handle must be an integer");
        return NULL;
    }

    VecEnv *vec = (VecEnv *) PyLong_AsVoidPtr(handle_obj);
    if (!vec || vec->num_envs <= 0) {
        PyErr_SetString(PyExc_ValueError, "Missing or invalid vec env handle");
        return NULL;
    }
    return vec;
}

static PyObject *vectorize(PyObject *self, PyObject *args) {
    int num_envs = PyTuple_Size(args);
    if (num_envs == 0) {
        PyErr_SetString(PyExc_TypeError, "make_vec requires at least 1 env id");
        return NULL;
    }

    VecEnv *vec = (VecEnv *) calloc(1, sizeof(VecEnv));
    if (!vec) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }

    vec->envs = (Drive **) calloc(num_envs, sizeof(Drive *));
    if (!vec->envs) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }

    vec->num_envs = num_envs;
    for (int i = 0; i < num_envs; i++) {
        PyObject *handle_obj = PyTuple_GetItem(args, i);
        if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
            PyErr_SetString(
                PyExc_TypeError,
                "Env ids must be integers. Pass them as separate args with *env_ids, not as a list.");
            return NULL;
        }
        vec->envs[i] = (Drive *) PyLong_AsVoidPtr(handle_obj);
    }

    return PyLong_FromVoidPtr(vec);
}

static PyObject *vec_reset(PyObject *self, PyObject *args) {
    Py_ssize_t num_args = PyTuple_Size(args);
    if (num_args != 1 && num_args != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_reset requires 1 or 2 arguments");
        return NULL;
    }

    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject *seeds = num_args == 2 ? PyTuple_GetItem(args, 1) : NULL;
    if (seeds != NULL && seeds != Py_None) {
        if (!PyList_Check(seeds) || PyList_Size(seeds) != vec->num_envs) {
            PyErr_Format(PyExc_ValueError, "vec_reset seeds must be a list with one entry per env (%d)", vec->num_envs);
            return NULL;
        }
        for (int i = 0; i < vec->num_envs; i++) {
            long long seed = PyLong_AsLongLong(PyList_GetItem(seeds, i));
            if (PyErr_Occurred() || seed < 0) {
                PyErr_SetString(PyExc_ValueError, "vec_reset seeds must be non-negative 63-bit integers");
                return NULL;
            }
            my_reseed(vec->envs[i], (uint64_t) seed);
        }
    }

    for (int i = 0; i < vec->num_envs; i++) {
        c_reset(vec->envs[i]);
    }
    Py_RETURN_NONE;
}

// One row per env whose single frozen eval episode has finished; envs without one (log.n == 0) are skipped.
static PyObject *vec_per_episode_log(PyObject *self, PyObject *args) {
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject *list = PyList_New(0);
    if (!list) {
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        Drive *env = vec->envs[i];
        if (env->log.n <= 0.0f) {
            continue;
        }
        PyObject *dict = PyDict_New();
        if (!dict) {
            Py_DECREF(list);
            return NULL;
        }
        if (my_episode_to_dict(dict, env) != 0) {
            Py_DECREF(dict);
            Py_DECREF(list);
            return NULL;
        }
        assign_to_dict(dict, "env_slot", (float) i);
        if (PyList_Append(list, dict) < 0) {
            Py_DECREF(dict);
            Py_DECREF(list);
            return NULL;
        }
        Py_DECREF(dict);
    }
    return list;
}

static PyObject *vec_step(PyObject *self, PyObject *arg) {
    int num_args = PyTuple_Size(arg);
    if (num_args != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_step requires 1 argument");
        return NULL;
    }

    VecEnv *vec = unpack_vecenv(arg);
    if (!vec) {
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        c_step(vec->envs[i]);
    }
    Py_RETURN_NONE;
}

static int assign_to_dict(PyObject *dict, char *key, float value) {
    PyObject *v = PyFloat_FromDouble(value);
    if (v == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert log value");
        return 1;
    }
    if (PyDict_SetItemString(dict, key, v) < 0) {
        PyErr_SetString(PyExc_TypeError, "Failed to set log value");
        return 1;
    }
    Py_DECREF(v);
    return 0;
}

static PyObject *vec_log(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_log requires 2 arguments");
        return NULL;
    }

    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    // Log must hold only floats: it is summed field by field
    PyObject *num_agents_arg = PyTuple_GetItem(args, 1);
    float num_agents = (float) PyLong_AsLong(num_agents_arg);
    int num_keys = sizeof(Log) / sizeof(float);

    Log aggregate = {0};
    for (int i = 0; i < vec->num_envs; i++) {
        Drive *env = vec->envs[i];
        for (int j = 0; j < num_keys; j++) {
            ((float *) &aggregate)[j] += ((float *) &env->log)[j];
        }
    }

    PyObject *dict = PyDict_New();

    if (aggregate.n < num_agents) {
        return dict;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        Drive *env = vec->envs[i];
        for (int j = 0; j < num_keys; j++) {
            ((float *) &env->log)[j] = 0.0f;
        }
    }

    float n = aggregate.n;

    for (int i = 0; i < num_keys; i++) {
        ((float *) &aggregate)[i] /= n;
    }
    my_log(dict, vec->envs[0], &aggregate, n);
    return dict;
}

static PyObject *vec_get(PyObject *self, PyObject *args) {
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    PyObject *list = PyList_New(vec->num_envs);
    if (!list) {
        return NULL;
    }
    for (int i = 0; i < vec->num_envs; i++) {
        PyObject *state = my_get(vec->envs[i]);
        if (state == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, state);
    }
    return list;
}

static PyObject *vec_get_obs_html_frame(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 8) {
        PyErr_SetString(PyExc_TypeError, "vec_get_obs_html_frame requires 8 arguments");
        return NULL;
    }

    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    for (Py_ssize_t arg_idx = 1; arg_idx < PyTuple_GET_SIZE(args); arg_idx++) {
        if (!PyArray_Check(PyTuple_GET_ITEM(args, arg_idx))) {
            PyErr_SetString(PyExc_TypeError, "All output arrays must be NumPy arrays");
            return NULL;
        }
    }
    for (Py_ssize_t arg_idx = 1; arg_idx < PyTuple_GET_SIZE(args); arg_idx++) {
        PyArrayObject *frame_array = (PyArrayObject *) PyTuple_GET_ITEM(args, arg_idx);
        memset(PyArray_DATA(frame_array), 0, PyArray_NBYTES(frame_array));
    }
    PyArrayObject *agent_f32_array = (PyArrayObject *) PyTuple_GetItem(args, 1);
    PyArrayObject *agent_i32_array = (PyArrayObject *) PyTuple_GetItem(args, 2);
    PyArrayObject *metrics_f32_array = (PyArrayObject *) PyTuple_GetItem(args, 3);
    PyArrayObject *traffic_i16_array = (PyArrayObject *) PyTuple_GetItem(args, 4);
    PyArrayObject *goals_f32_array = (PyArrayObject *) PyTuple_GetItem(args, 5);
    PyArrayObject *rewards_f32_array = (PyArrayObject *) PyTuple_GetItem(args, 6);
    PyArrayObject *coefs_f32_array = (PyArrayObject *) PyTuple_GetItem(args, 7);

    float *agent_f32 = (float *) PyArray_DATA(agent_f32_array);
    int *agent_i32 = (int *) PyArray_DATA(agent_i32_array);
    float *metrics_f32 = (float *) PyArray_DATA(metrics_f32_array);
    short *traffic_i16 = (short *) PyArray_DATA(traffic_i16_array);
    float *goals_f32 = (float *) PyArray_DATA(goals_f32_array);
    float *rewards_f32 = (float *) PyArray_DATA(rewards_f32_array);
    float *coefs_f32 = (float *) PyArray_DATA(coefs_f32_array);

    int env_cap = (int) PyArray_DIM(agent_f32_array, 0);
    int env_count = vec->num_envs < env_cap ? vec->num_envs : env_cap;
    int agent_cap = (int) PyArray_DIM(agent_f32_array, 1);
    int agent_f32_fields = (int) PyArray_DIM(agent_f32_array, 2);
    int agent_i32_fields = (int) PyArray_DIM(agent_i32_array, 2);
    int metric_fields = (int) PyArray_DIM(metrics_f32_array, 2);
    int traffic_cap = (int) PyArray_DIM(traffic_i16_array, 1);
    int traffic_fields = (int) PyArray_DIM(traffic_i16_array, 2);
    int goal_fields = (int) PyArray_DIM(goals_f32_array, 2);
    int goal_slots = goal_fields / GOAL_XY_FIELDS;
    int reward_fields = (int) PyArray_DIM(rewards_f32_array, 2);
    int coef_fields = (int) PyArray_DIM(coefs_f32_array, 2);

    if (coef_fields != NUM_REWARD_COEFS) {
        PyErr_SetString(PyExc_ValueError, "coefs_f32 must have NUM_REWARD_COEFS fields");
        return NULL;
    }

    for (int e = 0; e < env_count; e++) {
        Drive *drive = vec->envs[e];
        int agent_count = drive->num_total_agents < agent_cap ? drive->num_total_agents : agent_cap;
        int traffic_count = drive->num_traffic_elements < traffic_cap ? drive->num_traffic_elements : traffic_cap;

        for (int i = 0; i < agent_count; i++) {
            Agent *a = &drive->agents[i];
            int f32_base = (e * agent_cap + i) * agent_f32_fields;
            int i32_base = (e * agent_cap + i) * agent_i32_fields;
            int metrics_base = (e * agent_cap + i) * metric_fields;

            agent_f32[f32_base + 0] = a->sim_x;
            agent_f32[f32_base + 1] = a->sim_y;
            agent_f32[f32_base + 2] = a->sim_z;
            agent_f32[f32_base + 3] = a->sim_heading;
            agent_f32[f32_base + 4] = a->sim_length;
            agent_f32[f32_base + 5] = a->sim_width;
            agent_f32[f32_base + 6] = a->sim_speed;
            agent_f32[f32_base + 7] = a->steering_angle;
            agent_f32[f32_base + 8] = a->accel_long;
            agent_f32[f32_base + 9] = a->accel_lat;
            agent_f32[f32_base + 10] = a->jerk_long;
            agent_f32[f32_base + 11] = a->jerk_lat;
            agent_f32[f32_base + AGENT_F32_GOAL_RADIUS_IDX] = a->reward_coefs[REWARD_COEF_GOAL_RADIUS];

            agent_i32[i32_base + 0] = i;
            agent_i32[i32_base + 1] = a->type;
            agent_i32[i32_base + 2] = a->sim_valid;
            agent_i32[i32_base + 3] = a->active_agent;
            agent_i32[i32_base + 4] = a->stopped;
            agent_i32[i32_base + 5] = a->removed;
            agent_i32[i32_base + 6] = a->current_lane_idx;
            agent_i32[i32_base + 7] = -1;
            agent_i32[i32_base + 8] = a->partner_blindness_counter > 0;
            agent_i32[i32_base + 9] = a->phantom_braking_counter > 0;

            memcpy(&metrics_f32[metrics_base], a->metrics_array, sizeof(float) * NUM_METRICS);
            memcpy(&coefs_f32[(e * agent_cap + i) * coef_fields], a->reward_coefs, sizeof(float) * NUM_REWARD_COEFS);
        }

        for (int j = 0; j < drive->active_agent_count; j++) {
            int agent_idx = drive->active_agent_indices[j];
            if (agent_idx >= agent_count) {
                continue;
            }
            int i32_base = (e * agent_cap + agent_idx) * agent_i32_fields;
            int reward_base = (e * agent_cap + agent_idx) * reward_fields;
            agent_i32[i32_base + 7] = j;

            Agent *active = &drive->agents[agent_idx];
            int goal_count = active->goal_count < goal_slots ? active->goal_count : goal_slots;
            int goal_base = (e * agent_cap + agent_idx) * goal_fields;
            for (int goal_idx = active->current_goal_idx; goal_idx < goal_count; goal_idx++) {
                goals_f32[goal_base + goal_idx * GOAL_XY_FIELDS] = active->list_goal_x[goal_idx];
                goals_f32[goal_base + goal_idx * GOAL_XY_FIELDS + 1] = active->list_goal_y[goal_idx];
            }

            if (!drive->compute_eval_metrics || !drive->logs || j >= drive->logs_capacity) {
                continue;
            }
            Log *log = &drive->logs[j];
            rewards_f32[reward_base + REWARD_F32_EPISODE_RETURN_IDX] = log->episode_return;
            rewards_f32[reward_base + REWARD_F32_COLLISION_IDX] = log->reward_collision;
            rewards_f32[reward_base + REWARD_F32_OFFROAD_IDX] = log->reward_offroad;
            rewards_f32[reward_base + REWARD_F32_RED_LIGHT_IDX] = log->reward_red_light;
            rewards_f32[reward_base + REWARD_F32_STOP_SIGN_IDX] = log->reward_stop_sign;
            rewards_f32[reward_base + REWARD_F32_GOAL_IDX] = log->reward_goal;
            rewards_f32[reward_base + REWARD_F32_LANE_ALIGN_IDX] = log->reward_lane_align;
            rewards_f32[reward_base + REWARD_F32_LANE_CENTER_IDX] = log->reward_lane_center;
            rewards_f32[reward_base + REWARD_F32_COMFORT_IDX] = log->reward_comfort;
            rewards_f32[reward_base + REWARD_F32_VELOCITY_IDX] = log->reward_velocity;
            rewards_f32[reward_base + REWARD_F32_TIMESTEP_IDX] = log->reward_timestep;
            rewards_f32[reward_base + REWARD_F32_REVERSE_IDX] = log->reward_reverse;
            rewards_f32[reward_base + REWARD_F32_OVERSPEED_IDX] = log->reward_overspeed;
        }

        for (int i = 0; i < traffic_count; i++) {
            TrafficControlElement *t = &drive->traffic_elements[i];
            int base = (e * traffic_cap + i) * traffic_fields;
            traffic_i16[base + 0] = 1;
            traffic_i16[base + 1] = (short) t->type;
            if (t->states && drive->timestep >= 0 && drive->timestep < t->state_size) {
                traffic_i16[base + 2] = (short) t->states[drive->timestep];
            }
        }
    }

    Py_RETURN_NONE;
}

static PyObject *vec_close(PyObject *self, PyObject *args) {
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        c_close(vec->envs[i]);
        free(vec->envs[i]);
    }
    free(vec->envs);
    free(vec);
    Py_RETURN_NONE;
}

static PyObject *vec_get_global_agent_state(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 9) {
        PyErr_SetString(PyExc_TypeError, "vec_get_global_agent_state requires 9 arguments");
        return NULL;
    }
    int include_static = (int) PyLong_AsLong(PyTuple_GetItem(args, 8));

    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject *x_arr = PyTuple_GetItem(args, 1);
    PyObject *y_arr = PyTuple_GetItem(args, 2);
    PyObject *z_arr = PyTuple_GetItem(args, 3);
    PyObject *heading_arr = PyTuple_GetItem(args, 4);
    PyObject *id_arr = PyTuple_GetItem(args, 5);
    PyObject *length_arr = PyTuple_GetItem(args, 6);
    PyObject *width_arr = PyTuple_GetItem(args, 7);
    if (!PyArray_Check(x_arr) || !PyArray_Check(y_arr) || !PyArray_Check(z_arr) || !PyArray_Check(heading_arr)
        || !PyArray_Check(id_arr) || !PyArray_Check(length_arr) || !PyArray_Check(width_arr)) {
        PyErr_SetString(PyExc_TypeError, "All output arrays must be NumPy arrays");
        return NULL;
    }
    npy_intp total_agent_count = 0;
    for (int i = 0; i < vec->num_envs; i++) {
        Drive *drive = vec->envs[i];
        total_agent_count += include_static ? drive->num_agents : drive->active_agent_count;
    }
    const struct {
        PyObject *array;
        int type_num;
    } output_specs[] = {
        {x_arr, NPY_FLOAT32},
        {y_arr, NPY_FLOAT32},
        {z_arr, NPY_FLOAT32},
        {heading_arr, NPY_FLOAT32},
        {id_arr, NPY_INT32},
        {length_arr, NPY_FLOAT32},
        {width_arr, NPY_FLOAT32},
    };
    int output_count = (int) (sizeof(output_specs) / sizeof(output_specs[0]));
    for (int output_idx = 0; output_idx < output_count; output_idx++) {
        PyArrayObject *array = (PyArrayObject *) output_specs[output_idx].array;
        if (PyArray_NDIM(array) != 1 || !PyArray_ISCARRAY(array)
            || PyArray_TYPE(array) != output_specs[output_idx].type_num || PyArray_SIZE(array) < total_agent_count) {
            PyErr_Format(
                PyExc_ValueError,
                "vec_get_global_agent_state: output arrays must be 1D writeable contiguous float32 (id: int32) "
                "arrays of at least %zd entries",
                (Py_ssize_t) total_agent_count);
            return NULL;
        }
    }
    float *x = (float *) PyArray_DATA((PyArrayObject *) x_arr);
    float *y = (float *) PyArray_DATA((PyArrayObject *) y_arr);
    float *z = (float *) PyArray_DATA((PyArrayObject *) z_arr);
    float *heading = (float *) PyArray_DATA((PyArrayObject *) heading_arr);
    int *id = (int *) PyArray_DATA((PyArrayObject *) id_arr);
    float *length = (float *) PyArray_DATA((PyArrayObject *) length_arr);
    float *width = (float *) PyArray_DATA((PyArrayObject *) width_arr);

    int offset = 0;
    for (int i = 0; i < vec->num_envs; i++) {
        Drive *drive = vec->envs[i];
        c_get_global_agent_state(
            drive,
            include_static,
            &x[offset],
            &y[offset],
            &z[offset],
            &heading[offset],
            &id[offset],
            &length[offset],
            &width[offset]);
        offset += include_static ? drive->num_agents : drive->active_agent_count;
    }

    Py_RETURN_NONE;
}

static PyObject *vec_get_static_agent_count(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_get_static_agent_count requires 1 argument");
        return NULL;
    }
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    long static_agent_count = 0;
    for (int i = 0; i < vec->num_envs; i++) {
        static_agent_count += vec->envs[i]->static_agent_count;
    }
    return PyLong_FromLong(static_agent_count);
}

// Co-sim runs a single env (num_envs == 1); these setters operate on vec->envs[0].
static PyObject *vec_set_agent_states(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 12) {
        PyErr_SetString(PyExc_TypeError, "vec_set_agent_states requires 12 arguments");
        return NULL;
    }
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    PyObject *idx_arr = PyTuple_GetItem(args, 1);
    PyObject *x_arr = PyTuple_GetItem(args, 2);
    PyObject *y_arr = PyTuple_GetItem(args, 3);
    PyObject *z_arr = PyTuple_GetItem(args, 4);
    PyObject *heading_arr = PyTuple_GetItem(args, 5);
    PyObject *vx_arr = PyTuple_GetItem(args, 6);
    PyObject *vy_arr = PyTuple_GetItem(args, 7);
    PyObject *yaw_rate_arr = PyTuple_GetItem(args, 8);
    PyObject *accel_long_arr = PyTuple_GetItem(args, 9);
    PyObject *seconds_stopped_arr = PyTuple_GetItem(args, 10); // None keeps c_step's accumulation
    PyObject *steering_arr = PyTuple_GetItem(args, 11);        // None derives steering from yaw_rate
    if (!PyArray_Check(idx_arr) || !PyArray_Check(x_arr) || !PyArray_Check(y_arr) || !PyArray_Check(z_arr)
        || !PyArray_Check(heading_arr) || !PyArray_Check(vx_arr) || !PyArray_Check(vy_arr)
        || !PyArray_Check(yaw_rate_arr) || !PyArray_Check(accel_long_arr)) {
        PyErr_SetString(PyExc_TypeError, "All arrays must be NumPy arrays");
        return NULL;
    }
    if (seconds_stopped_arr != Py_None && !PyArray_Check(seconds_stopped_arr)) {
        PyErr_SetString(PyExc_TypeError, "seconds_stopped must be a NumPy array or None");
        return NULL;
    }
    if (steering_arr != Py_None && !PyArray_Check(steering_arr)) {
        PyErr_SetString(PyExc_TypeError, "steering must be a NumPy array or None");
        return NULL;
    }
    int *idx = (int *) PyArray_DATA((PyArrayObject *) idx_arr);
    float *x = (float *) PyArray_DATA((PyArrayObject *) x_arr);
    float *y = (float *) PyArray_DATA((PyArrayObject *) y_arr);
    float *z = (float *) PyArray_DATA((PyArrayObject *) z_arr);
    float *heading = (float *) PyArray_DATA((PyArrayObject *) heading_arr);
    float *vx = (float *) PyArray_DATA((PyArrayObject *) vx_arr);
    float *vy = (float *) PyArray_DATA((PyArrayObject *) vy_arr);
    float *yaw_rate = (float *) PyArray_DATA((PyArrayObject *) yaw_rate_arr);
    float *accel_long = (float *) PyArray_DATA((PyArrayObject *) accel_long_arr);
    float *seconds_stopped
        = seconds_stopped_arr == Py_None ? NULL : (float *) PyArray_DATA((PyArrayObject *) seconds_stopped_arr);
    float *steering = steering_arr == Py_None ? NULL : (float *) PyArray_DATA((PyArrayObject *) steering_arr);
    int count = (int) PyArray_SIZE((PyArrayObject *) idx_arr);
    PyObject *value_arrs[10]
        = {x_arr,
           y_arr,
           z_arr,
           heading_arr,
           vx_arr,
           vy_arr,
           yaw_rate_arr,
           accel_long_arr,
           seconds_stopped_arr,
           steering_arr};
    for (int k = 0; k < 10; k++) {
        if (value_arrs[k] == Py_None) {
            continue;
        }
        if ((int) PyArray_SIZE((PyArrayObject *) value_arrs[k]) != count) {
            PyErr_SetString(PyExc_ValueError, "vec_set_agent_states: every array must have the same length as idx");
            return NULL;
        }
    }
    if (c_set_agent_states(
            vec->envs[0],
            count,
            idx,
            x,
            y,
            z,
            heading,
            vx,
            vy,
            yaw_rate,
            accel_long,
            seconds_stopped,
            steering)
        != 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "vec_set_agent_states: agent index out of range, non-finite state, negative seconds_stopped or "
            "non-finite steering");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *vec_set_agent_sizes(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 4) {
        PyErr_SetString(PyExc_TypeError, "vec_set_agent_sizes requires 4 arguments");
        return NULL;
    }
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    PyObject *idx_arr = PyTuple_GetItem(args, 1);
    PyObject *length_arr = PyTuple_GetItem(args, 2);
    PyObject *width_arr = PyTuple_GetItem(args, 3);
    if (!PyArray_Check(idx_arr) || !PyArray_Check(length_arr) || !PyArray_Check(width_arr)) {
        PyErr_SetString(PyExc_TypeError, "All arrays must be NumPy arrays");
        return NULL;
    }
    int *idx = (int *) PyArray_DATA((PyArrayObject *) idx_arr);
    float *length = (float *) PyArray_DATA((PyArrayObject *) length_arr);
    float *width = (float *) PyArray_DATA((PyArrayObject *) width_arr);
    int count = (int) PyArray_SIZE((PyArrayObject *) idx_arr);
    if ((int) PyArray_SIZE((PyArrayObject *) length_arr) != count
        || (int) PyArray_SIZE((PyArrayObject *) width_arr) != count) {
        PyErr_SetString(PyExc_ValueError, "vec_set_agent_sizes: length/width must have the same length as idx");
        return NULL;
    }
    if (c_set_agent_sizes(vec->envs[0], count, idx, length, width) != 0) {
        PyErr_SetString(PyExc_ValueError, "vec_set_agent_sizes: agent index out of range or non-positive size");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *vec_recompute_observations(PyObject *self, PyObject *args) {
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    for (int i = 0; i < vec->num_envs; i++) {
        compute_observations(vec->envs[i]);
    }
    Py_RETURN_NONE;
}

static PyObject *vec_set_traffic_light_states(PyObject *self, PyObject *args) {
    Py_ssize_t arg_count = PyTuple_Size(args);
    if (arg_count != 2 && arg_count != 3) {
        PyErr_SetString(
            PyExc_TypeError,
            "vec_set_traffic_light_states requires 2 or 3 arguments (vec_env, states[, yellow_remaining_s])");
        return NULL;
    }
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    PyObject *states_arr = PyTuple_GetItem(args, 1);
    if (!PyArray_Check(states_arr)) {
        PyErr_SetString(PyExc_TypeError, "states must be a NumPy array");
        return NULL;
    }
    int *states = (int *) PyArray_DATA((PyArrayObject *) states_arr);
    Drive *drive = vec->envs[0];
    if ((int) PyArray_SIZE((PyArrayObject *) states_arr) != drive->num_traffic_elements) {
        PyErr_Format(
            PyExc_ValueError,
            "vec_set_traffic_light_states: expected %d states (one per traffic element), got %d",
            drive->num_traffic_elements,
            (int) PyArray_SIZE((PyArrayObject *) states_arr));
        return NULL;
    }
    float *yellow_remaining_s = NULL;
    if (arg_count == 3 && PyTuple_GetItem(args, 2) != Py_None) {
        PyObject *remaining_arr = PyTuple_GetItem(args, 2);
        if (!PyArray_Check(remaining_arr) || PyArray_TYPE((PyArrayObject *) remaining_arr) != NPY_FLOAT32
            || (int) PyArray_SIZE((PyArrayObject *) remaining_arr) != drive->num_traffic_elements) {
            PyErr_Format(
                PyExc_ValueError,
                "vec_set_traffic_light_states: yellow_remaining_s must be a float32 NumPy array of %d entries",
                drive->num_traffic_elements);
            return NULL;
        }
        yellow_remaining_s = (float *) PyArray_DATA((PyArrayObject *) remaining_arr);
    }
    if (c_set_traffic_light_states(drive, states, yellow_remaining_s) != 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "vec_set_traffic_light_states: timestep outside the light state buffer, invalid state, or "
            "yellow_remaining_s missing (obs_traffic_light_yellow_remaining needs it) or outside "
            "[0, traffic_light_yellow_max_s]");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *vec_set_stop_signs(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "vec_set_stop_signs requires 3 arguments (vec_env, lines, headings)");
        return NULL;
    }
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    PyObject *lines_arr = PyTuple_GetItem(args, 1);
    PyObject *headings_arr = PyTuple_GetItem(args, 2);
    if (!PyArray_Check(lines_arr) || !PyArray_Check(headings_arr)) {
        PyErr_SetString(PyExc_TypeError, "vec_set_stop_signs: lines and headings must be NumPy arrays");
        return NULL;
    }
    int count = (int) PyArray_SIZE((PyArrayObject *) headings_arr);
    if ((int) PyArray_SIZE((PyArrayObject *) lines_arr) != count * 6) {
        PyErr_SetString(PyExc_ValueError, "vec_set_stop_signs: lines must hold 6 floats per heading");
        return NULL;
    }
    Drive *drive = vec->envs[0];
    const float *lines = (const float *) PyArray_DATA((PyArrayObject *) lines_arr);
    const float *headings = (const float *) PyArray_DATA((PyArrayObject *) headings_arr);
    if (c_set_stop_signs(drive, count, lines, headings) != 0) {
        PyErr_SetString(PyExc_ValueError, "vec_set_stop_signs: non-finite value or zero-length stop line");
        return NULL;
    }
    return PyLong_FromLong(drive->num_traffic_elements);
}

static PyObject *vec_set_traffic_light_lines(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "vec_set_traffic_light_lines requires 3 arguments (vec_env, indices, lines)");
        return NULL;
    }
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    PyObject *indices_arr = PyTuple_GetItem(args, 1);
    PyObject *lines_arr = PyTuple_GetItem(args, 2);
    if (!PyArray_Check(indices_arr) || !PyArray_Check(lines_arr)) {
        PyErr_SetString(PyExc_TypeError, "vec_set_traffic_light_lines: indices and lines must be NumPy arrays");
        return NULL;
    }
    int count = (int) PyArray_SIZE((PyArrayObject *) indices_arr);
    if ((int) PyArray_SIZE((PyArrayObject *) lines_arr) != count * 6) {
        PyErr_SetString(PyExc_ValueError, "vec_set_traffic_light_lines: lines must hold 6 floats per index");
        return NULL;
    }
    Drive *drive = vec->envs[0];
    const int *indices = (const int *) PyArray_DATA((PyArrayObject *) indices_arr);
    const float *lines = (const float *) PyArray_DATA((PyArrayObject *) lines_arr);
    if (c_set_traffic_light_lines(drive, count, indices, lines) != 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "vec_set_traffic_light_lines: index is not a traffic light, non-finite value or zero-length line");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *vec_get_agent_goal_progress(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_get_agent_goal_progress requires 2 arguments");
        return NULL;
    }
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    Drive *drive = vec->envs[0];
    int agent_idx = (int) PyLong_AsLong(PyTuple_GetItem(args, 1));
    if (agent_idx < 0 || agent_idx >= drive->num_total_agents) {
        PyErr_SetString(PyExc_ValueError, "vec_get_agent_goal_progress: agent index out of range");
        return NULL;
    }
    Agent *agent = &drive->agents[agent_idx];
    return Py_BuildValue("(ii)", agent->current_goal_idx, agent->goal_count);
}

static PyObject *vec_set_agent_goals(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 7) {
        PyErr_SetString(PyExc_TypeError, "vec_set_agent_goals requires 7 arguments");
        return NULL;
    }
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    int agent_idx = (int) PyLong_AsLong(PyTuple_GetItem(args, 1));
    PyObject *gx_arr = PyTuple_GetItem(args, 2);
    PyObject *gy_arr = PyTuple_GetItem(args, 3);
    PyObject *gz_arr = PyTuple_GetItem(args, 4);
    PyObject *gdx_arr = PyTuple_GetItem(args, 5);
    PyObject *gdy_arr = PyTuple_GetItem(args, 6);
    if (!PyArray_Check(gx_arr) || !PyArray_Check(gy_arr) || !PyArray_Check(gz_arr) || !PyArray_Check(gdx_arr)
        || !PyArray_Check(gdy_arr)) {
        PyErr_SetString(PyExc_TypeError, "goal arrays must be NumPy arrays");
        return NULL;
    }
    float *gx = (float *) PyArray_DATA((PyArrayObject *) gx_arr);
    float *gy = (float *) PyArray_DATA((PyArrayObject *) gy_arr);
    float *gz = (float *) PyArray_DATA((PyArrayObject *) gz_arr);
    float *gdx = (float *) PyArray_DATA((PyArrayObject *) gdx_arr);
    float *gdy = (float *) PyArray_DATA((PyArrayObject *) gdy_arr);
    int num_wp = (int) PyArray_SIZE((PyArrayObject *) gx_arr);
    if ((int) PyArray_SIZE((PyArrayObject *) gy_arr) != num_wp || (int) PyArray_SIZE((PyArrayObject *) gz_arr) != num_wp
        || (int) PyArray_SIZE((PyArrayObject *) gdx_arr) != num_wp
        || (int) PyArray_SIZE((PyArrayObject *) gdy_arr) != num_wp) {
        PyErr_SetString(PyExc_ValueError, "vec_set_agent_goals: goal arrays must all have the same length");
        return NULL;
    }
    if (c_set_agent_goals(vec->envs[0], agent_idx, num_wp, gx, gy, gz, gdx, gdy) != 0) {
        PyErr_Format(
            PyExc_ValueError,
            "vec_set_agent_goals: agent index out of range, waypoint count outside 1..%d, or non-finite goal",
            MAX_GOALS);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *vec_get_global_ground_truth_trajectories(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 7) {
        PyErr_SetString(PyExc_TypeError, "vec_get_global_ground_truth_trajectories requires 7 arguments");
        return NULL;
    }

    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject *x_arr = PyTuple_GetItem(args, 1);
    PyObject *y_arr = PyTuple_GetItem(args, 2);
    PyObject *z_arr = PyTuple_GetItem(args, 3);
    PyObject *heading_arr = PyTuple_GetItem(args, 4);
    PyObject *valid_arr = PyTuple_GetItem(args, 5);
    PyObject *id_arr = PyTuple_GetItem(args, 6);
    if (!PyArray_Check(x_arr) || !PyArray_Check(y_arr) || !PyArray_Check(z_arr) || !PyArray_Check(heading_arr)
        || !PyArray_Check(valid_arr) || !PyArray_Check(id_arr)) {
        PyErr_SetString(PyExc_TypeError, "All output arrays must be NumPy arrays");
        return NULL;
    }

    PyArrayObject *trajectory_arrays[5]
        = {(PyArrayObject *) x_arr,
           (PyArrayObject *) y_arr,
           (PyArrayObject *) z_arr,
           (PyArrayObject *) heading_arr,
           (PyArrayObject *) valid_arr};
    npy_intp total_agent_count = 0;
    for (int i = 0; i < vec->num_envs; i++) {
        total_agent_count += vec->envs[i]->active_agent_count;
    }
    for (int k = 0; k < 5; k++) {
        PyArrayObject *array = trajectory_arrays[k];
        if (PyArray_NDIM(array) != 2 || PyArray_DIM(array, 0) < total_agent_count
            || PyArray_DIM(array, 1) != PyArray_DIM(trajectory_arrays[0], 1)) {
            PyErr_SetString(PyExc_ValueError, "trajectory arrays must share one 2D shape [num_agents, num_timesteps]");
            return NULL;
        }
    }
    int num_timesteps = (int) PyArray_DIM(trajectory_arrays[0], 1);
    float *x = (float *) PyArray_DATA((PyArrayObject *) x_arr);
    float *y = (float *) PyArray_DATA((PyArrayObject *) y_arr);
    float *z = (float *) PyArray_DATA((PyArrayObject *) z_arr);
    float *heading = (float *) PyArray_DATA((PyArrayObject *) heading_arr);
    int *valid = (int *) PyArray_DATA((PyArrayObject *) valid_arr);
    int *id = (int *) PyArray_DATA((PyArrayObject *) id_arr);

    int agent_offset = 0;
    int traj_offset = 0;
    for (int i = 0; i < vec->num_envs; i++) {
        Drive *drive = vec->envs[i];
        c_get_global_ground_truth_trajectories(
            drive,
            num_timesteps,
            &x[traj_offset],
            &y[traj_offset],
            &z[traj_offset],
            &heading[traj_offset],
            &valid[traj_offset],
            &id[agent_offset]);
        agent_offset += drive->active_agent_count;
        traj_offset += drive->active_agent_count * num_timesteps;
    }

    Py_RETURN_NONE;
}

static double unpack(PyObject *kwargs, char *key) {
    PyObject *val = PyDict_GetItemString(kwargs, key);
    if (val == NULL) {
        PyErr_Format(PyExc_TypeError, "Missing required keyword argument '%s'", key);
        return 1;
    }
    if (PyLong_Check(val)) {
        long out = PyLong_AsLong(val);
        if (out > INT_MAX || out < INT_MIN) {
            PyErr_Format(PyExc_TypeError, "Value %ld of integer argument %s is out of range", out, key);
            return 1;
        }
        // double holds every 32-bit int exactly
        return out;
    }
    if (PyFloat_Check(val)) {
        return PyFloat_AsDouble(val);
    }
    PyErr_Format(PyExc_TypeError, "Failed to unpack keyword %s as int", key);
    return 1;
}

static char *unpack_str(PyObject *kwargs, char *key) {
    PyObject *val = PyDict_GetItemString(kwargs, key);
    if (val == NULL) {
        PyErr_Format(PyExc_TypeError, "Missing required keyword argument '%s'", key);
        return NULL;
    }
    if (!PyUnicode_Check(val)) {
        PyErr_Format(PyExc_TypeError, "Keyword argument '%s' must be a string", key);
        return NULL;
    }
    const char *str_val = PyUnicode_AsUTF8(val);
    if (str_val == NULL) {
        // PyUnicode_AsUTF8 sets an error on failure
        return NULL;
    }
    char *ret = strdup(str_val);
    if (ret == NULL) {
        PyErr_SetString(PyExc_MemoryError, "strdup failed in unpack_str");
    }
    return ret;
}

static PyMethodDef methods[]
    = {{"env_init",
        (PyCFunction) env_init,
        METH_VARARGS | METH_KEYWORDS,
        "Init environment with observation, action, reward, terminal, truncation arrays"},
       {"vectorize", vectorize, METH_VARARGS, "Make a vector of environment handles"},
       {"vec_reset", vec_reset, METH_VARARGS, "Reset the vector of environments"},
       {"vec_step", vec_step, METH_VARARGS, "Step the vector of environments"},
       {"vec_per_episode_log",
        vec_per_episode_log,
        METH_VARARGS,
        "Return one per-episode metrics dict per env whose eval episode has finished"},
       {"vec_log", vec_log, METH_VARARGS, "Log the vector of environments"},
       {"vec_close", vec_close, METH_VARARGS, "Close the vector of environments"},
       {"vec_get", vec_get, METH_VARARGS, "Get attributes from each env in a VecEnv"},
       {"vec_get_obs_html_frame",
        vec_get_obs_html_frame,
        METH_VARARGS,
        "Fill compact obs_html frame arrays from a VecEnv"},
       {"shared", (PyCFunction) my_shared, METH_VARARGS | METH_KEYWORDS, "Shared state"},
       {"vec_get_global_agent_state", vec_get_global_agent_state, METH_VARARGS, "Get agent state from vectorized env"},
       {"vec_get_static_agent_count",
        vec_get_static_agent_count,
        METH_VARARGS,
        "Static agents summed over the envs (the include_static rows of vec_get_global_agent_state)"},
       {"vec_set_agent_states",
        vec_set_agent_states,
        METH_VARARGS,
        "Overwrite agent states from an external source (co-sim)"},
       {"vec_set_agent_sizes",
        vec_set_agent_sizes,
        METH_VARARGS,
        "Overwrite agent bounding-box sizes from an external source (co-sim)"},
       {"vec_recompute_observations",
        vec_recompute_observations,
        METH_VARARGS,
        "Recompute observations without stepping (co-sim)"},
       {"vec_set_traffic_light_states",
        vec_set_traffic_light_states,
        METH_VARARGS,
        "Override traffic light states (co-sim)"},
       {"vec_set_stop_signs",
        vec_set_stop_signs,
        METH_VARARGS,
        "Replace the stop signs with the external sim's own (co-sim)"},
       {"vec_set_traffic_light_lines",
        vec_set_traffic_light_lines,
        METH_VARARGS,
        "Move traffic-light stop lines onto the external sim's own (co-sim)"},
       {"vec_set_agent_goals", vec_set_agent_goals, METH_VARARGS, "Set an agent's goal waypoints (co-sim)"},
       {"vec_get_agent_goal_progress",
        vec_get_agent_goal_progress,
        METH_VARARGS,
        "(current_goal_idx, goal_count) of one agent (co-sim)"},
       {"vec_get_global_ground_truth_trajectories",
        vec_get_global_ground_truth_trajectories,
        METH_VARARGS,
        "Get ground truth trajectories from vectorized env"},
       MY_METHODS,
       {NULL, NULL, 0, NULL}};

static PyModuleDef module = {PyModuleDef_HEAD_INIT, "binding", NULL, -1, methods};

// Immutable tuple of a C float table, so policy code reads the sim's own values; 0 on success, -1 on failure.
static int add_float_table_constant(PyObject *module_obj, const char *name, const float *values, int value_count) {
    PyObject *table = PyTuple_New(value_count);
    if (table == NULL) {
        return -1;
    }
    for (int i = 0; i < value_count; i++) {
        PyObject *value = PyFloat_FromDouble((double) values[i]);
        if (value == NULL) {
            Py_DECREF(table);
            return -1;
        }
        PyTuple_SET_ITEM(table, i, value);
    }
    if (PyModule_AddObject(module_obj, name, table) < 0) {
        Py_DECREF(table);
        return -1;
    }
    return 0;
}

PyMODINIT_FUNC PyInit_binding(void) {
    import_array();
    PyObject *m = PyModule_Create(&module);

    if (m == NULL) {
        return NULL;
    }

    PyModule_AddIntConstant(m, "LANE_FEATURES", LANE_FEATURES);
    PyModule_AddIntConstant(m, "LANE_SPEED_LIMIT_FEATURES", LANE_SPEED_LIMIT_FEATURES);
    PyModule_AddIntConstant(m, "BOUNDARY_FEATURES", BOUNDARY_FEATURES);
    PyModule_AddIntConstant(m, "PARTNER_FEATURES", PARTNER_FEATURES);
    PyModule_AddIntConstant(m, "PARTNER_RELATIVE_VELOCITY_FEATURES", PARTNER_RELATIVE_VELOCITY_FEATURES);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_FEATURES", TRAFFIC_CONTROL_FEATURES);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_YELLOW_REMAINING_FEATURES", TRAFFIC_CONTROL_YELLOW_REMAINING_FEATURES);
    PyModule_AddIntConstant(m, "OBS_VALID_COUNT_FEATURES", OBS_VALID_COUNT_FEATURES);
    PyModule_AddIntConstant(m, "NUM_TRAFFIC_CONTROL_TYPES", NUM_TRAFFIC_CONTROL_TYPES);
    PyModule_AddIntConstant(m, "NUM_TRAFFIC_CONTROL_STATES", NUM_TRAFFIC_CONTROL_STATES);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT", TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_STATE_RED", TRAFFIC_CONTROL_STATE_RED);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_STATE_YELLOW", TRAFFIC_CONTROL_STATE_YELLOW);
    PyModule_AddIntConstant(m, "EGO_FEATURES", EGO_FEATURES);
    PyModule_AddIntConstant(m, "GOAL_FEATURES", GOAL_FEATURES);
    PyModule_AddIntConstant(m, "MAX_GOALS", MAX_GOALS);
    PyModule_AddIntConstant(m, "AGENT_F32_FIELDS", AGENT_F32_FIELDS);
    PyModule_AddIntConstant(m, "AGENT_F32_GOAL_RADIUS_IDX", AGENT_F32_GOAL_RADIUS_IDX);
    PyModule_AddIntConstant(m, "AGENT_I32_FIELDS", AGENT_I32_FIELDS);
    PyModule_AddIntConstant(m, "GOAL_XY_FIELDS", GOAL_XY_FIELDS);
    PyModule_AddIntConstant(m, "METRICS_F32_FIELDS", METRICS_F32_FIELDS);
    PyModule_AddIntConstant(m, "REWARD_F32_FIELDS", REWARD_F32_FIELDS);
    PyModule_AddIntConstant(m, "TRAFFIC_I16_FIELDS", TRAFFIC_I16_FIELDS);
    PyModule_AddIntConstant(m, "NUM_REWARD_COEFS", NUM_REWARD_COEFS);
    PyModule_AddIntConstant(m, "GOAL_REGEN_FINITE", GOAL_REGEN_FINITE);
    PyModule_AddIntConstant(m, "GOAL_REGEN_ROLLING", GOAL_REGEN_ROLLING);
    PyModule_AddIntConstant(m, "GOAL_SOURCE_ROUTE", GOAL_SOURCE_ROUTE);
    PyModule_AddIntConstant(m, "GOAL_SOURCE_MAP", GOAL_SOURCE_MAP);
    PyModule_AddIntConstant(m, "GOAL_SOURCE_GT", GOAL_SOURCE_GT);
    PyModule_AddIntConstant(m, "GOAL_SOURCE_EXTERNAL", GOAL_SOURCE_EXTERNAL);
    PyModule_AddIntConstant(m, "GOAL_SOURCE_GT_MAP", GOAL_SOURCE_GT_MAP);
    PyModule_AddIntConstant(m, "CONTROLLER_STATIC", CONTROLLER_STATIC);
    PyModule_AddIntConstant(m, "CONTROLLER_POLICY", CONTROLLER_POLICY);
    PyModule_AddIntConstant(m, "CONTROLLER_REPLAY", CONTROLLER_REPLAY);
    PyModule_AddIntConstant(m, "CONTROLLER_IDM", CONTROLLER_IDM);
    PyModule_AddIntConstant(m, "INFRACTION_BEHAVIOR_IGNORE", INFRACTION_BEHAVIOR_IGNORE);
    PyModule_AddIntConstant(m, "INFRACTION_BEHAVIOR_STOP", INFRACTION_BEHAVIOR_STOP);
    PyModule_AddIntConstant(m, "SIMULATION_MODE_GIGAFLOW", SIMULATION_MODE_GIGAFLOW);
    PyModule_AddIntConstant(m, "SIMULATION_MODE_REPLAY", SIMULATION_MODE_REPLAY);
    // obs[4] * ACCEL_LONG_NORM = accel_long in m/s^2: co-sim reads the shadow ego's accel intent back from the obs
    PyModule_AddObject(m, "ACCEL_LONG_NORM", PyFloat_FromDouble(fabsf(ACCEL_LONG_LIMIT[0])));
    PyModule_AddObject(m, "ACCEL_LONG_MAX", PyFloat_FromDouble(ACCEL_LONG_LIMIT[1]));
    PyModule_AddObject(m, "STEERING_LIMIT", PyFloat_FromDouble(STEERING_LIMIT));
    PyModule_AddObject(m, "MAX_BACKWARD_SPEED", PyFloat_FromDouble(MAX_BACKWARD_SPEED));
    PyModule_AddObject(m, "AGENT_STOPPED_SPEED_THRESHOLD", PyFloat_FromDouble(AGENT_STOPPED_SPEED_THRESHOLD));
    PyModule_AddObject(m, "CONDITIONING_ACC_MIN", PyFloat_FromDouble(REWARD_BOUNDS[REWARD_COEF_ACC].min_val));
    PyModule_AddObject(m, "CONDITIONING_ACC_MAX", PyFloat_FromDouble(REWARD_BOUNDS[REWARD_COEF_ACC].max_val));
    PyModule_AddIntConstant(m, "CONTROL_MODE_VEHICLES", CONTROL_MODE_VEHICLES);
    PyModule_AddIntConstant(m, "CONTROL_MODE_SDC_ONLY", CONTROL_MODE_SDC_ONLY);
    PyModule_AddIntConstant(m, "INIT_MODE_CREATE_ALL_VALID", INIT_MODE_CREATE_ALL_VALID);
    PyModule_AddIntConstant(m, "INIT_MODE_CREATE_CONTROLLABLE_TYPES", INIT_MODE_CREATE_CONTROLLABLE_TYPES);

    // The policy decodes discrete actions with the sim's own tables, so the two cannot drift apart
    if (add_float_table_constant(m, "JERK_LONG", JERK_LONG, NUM_JERK_LONG_ACTIONS) < 0
        || add_float_table_constant(m, "JERK_LAT", JERK_LAT, NUM_JERK_LAT_ACTIONS) < 0
        || add_float_table_constant(m, "STEERING_VALUES", STEERING_VALUES, NUM_STEERING_ACTIONS) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
