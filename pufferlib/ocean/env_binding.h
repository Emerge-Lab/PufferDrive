#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>

// Forward declarations for env-specific functions supplied by user
static int my_log(PyObject *dict, Env *env, Log *log, float n);
static int my_init(Env *env, PyObject *args, PyObject *kwargs);
static int my_episode_to_dict(PyObject *dict, Env *env);
static int assign_to_dict(PyObject *dict, char *key, float value);

static PyObject *my_shared(PyObject *self, PyObject *args, PyObject *kwargs);
#ifndef MY_SHARED
static PyObject *my_shared(PyObject *self, PyObject *args, PyObject *kwargs) {
    return NULL;
}
#endif

static PyObject *my_get(PyObject *dict, Env *env);
#ifndef MY_GET
static PyObject *my_get(PyObject *dict, Env *env) {
    return NULL;
}
#endif

static int my_put(Env *env, PyObject *args, PyObject *kwargs);
#ifndef MY_PUT
static int my_put(Env *env, PyObject *args, PyObject *kwargs) {
    return 0;
}
#endif

// Re-derive an env's RNG streams from a fresh seed before reset
static void my_reseed(Env *env, uint64_t seed);
#ifndef MY_RESEED
static void my_reseed(Env *env, uint64_t seed) {
    (void) env;
    (void) seed;
}
#endif

#ifndef MY_METHODS
#define MY_METHODS {NULL, NULL, 0, NULL}
#endif

static Env *unpack_env(PyObject *args) {
    PyObject *handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_handle must be an integer");
        return NULL;
    }

    Env *env = (Env *) PyLong_AsVoidPtr(handle_obj);
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle");
        return NULL;
    }

    return env;
}

// Python function to initialize the environment
static PyObject *env_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    if (PyTuple_Size(args) != 7) {
        PyErr_SetString(PyExc_TypeError, "Environment requires 7 positional arguments");
        return NULL;
    }

    Env *env = (Env *) calloc(1, sizeof(Env));
    if (!env) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate environment");
        return NULL;
    }

    PyObject *obs = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return NULL;
    }
    PyArrayObject *observations = (PyArrayObject *) obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return NULL;
    }
    env->observations = PyArray_DATA(observations);

    PyObject *act = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return NULL;
    }
    PyArrayObject *actions = (PyArrayObject *) act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return NULL;
    }
    env->actions = PyArray_DATA(actions);
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return NULL;
    }

    PyObject *rew = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return NULL;
    }
    PyArrayObject *rewards = (PyArrayObject *) rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return NULL;
    }
    env->rewards = PyArray_DATA(rewards);

    PyObject *term = PyTuple_GetItem(args, 3);
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return NULL;
    }
    PyArrayObject *terminals = (PyArrayObject *) term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return NULL;
    }
    env->terminals = PyArray_DATA(terminals);

    PyObject *trunc = PyTuple_GetItem(args, 4);
    if (!PyObject_TypeCheck(trunc, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Truncations must be a NumPy array");
        return NULL;
    }
    PyArrayObject *truncations = (PyArrayObject *) trunc;
    if (!PyArray_ISCONTIGUOUS(truncations)) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(truncations) != 1) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be 1D");
        return NULL;
    }
    env->truncations = PyArray_DATA(truncations);

    PyObject *msk = PyTuple_GetItem(args, 5);
    if (!PyObject_TypeCheck(msk, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Masks must be a NumPy array");
        return NULL;
    }
    PyArrayObject *masks_array = (PyArrayObject *) msk;
    if (!PyArray_ISCONTIGUOUS(masks_array)) {
        PyErr_SetString(PyExc_ValueError, "Masks must be contiguous");
        return NULL;
    }
    env->masks = PyArray_DATA(masks_array);

    PyObject *seed_arg = PyTuple_GetItem(args, 6);
    if (!PyObject_TypeCheck(seed_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "seed must be an integer");
        return NULL;
    }
    long long seed = PyLong_AsLongLong(seed_arg);
    if (PyErr_Occurred()) {
        return NULL;
    }

    // If kwargs is NULL, create a new dictionary
    if (kwargs == NULL) {
        kwargs = PyDict_New();
    } else {
        Py_INCREF(kwargs); // We need to increment the reference since we'll be modifying it
    }

    // Add the seed to kwargs
    PyObject *py_seed = PyLong_FromLongLong(seed);
    if (PyDict_SetItemString(kwargs, "seed", py_seed) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to set seed in kwargs");
        Py_DECREF(py_seed);
        Py_DECREF(kwargs);
        return NULL;
    }
    Py_DECREF(py_seed);

    PyObject *empty_args = PyTuple_New(0);
    my_init(env, empty_args, kwargs);
    Py_DECREF(kwargs);
    if (PyErr_Occurred()) {
        return NULL;
    }

    return PyLong_FromVoidPtr(env);
}

// Python function to reset the environment
static PyObject *env_reset(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "env_reset requires 2 arguments");
        return NULL;
    }

    Env *env = unpack_env(args);
    if (!env) {
        return NULL;
    }
    c_reset(env);
    Py_RETURN_NONE;
}

// Python function to step the environment
static PyObject *env_step(PyObject *self, PyObject *args) {
    int num_args = PyTuple_Size(args);
    if (num_args != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_render requires 1 argument");
        return NULL;
    }

    Env *env = unpack_env(args);
    if (!env) {
        return NULL;
    }
    c_step(env);
    Py_RETURN_NONE;
}

// Python function to step the environment
static PyObject *env_render(PyObject *self, PyObject *args) {
    Env *env = unpack_env(args);
    if (!env) {
        return NULL;
    }
    c_render((Drive *) env, 0); // single-env binding: VIEW_MODE_DEFAULT
    Py_RETURN_NONE;
}

// Python function to close the environment
static PyObject *env_close(PyObject *self, PyObject *args) {
    Env *env = unpack_env(args);
    if (!env) {
        return NULL;
    }
    c_close(env);
    free(env);
    Py_RETURN_NONE;
}

static PyObject *env_get(PyObject *self, PyObject *args) {
    Env *env = unpack_env(args);
    if (!env) {
        return NULL;
    }
    PyObject *dict = PyDict_New();
    my_get(dict, env);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return dict;
}

static PyObject *env_put(PyObject *self, PyObject *args, PyObject *kwargs) {
    int num_args = PyTuple_Size(args);
    if (num_args != 1) {
        PyErr_SetString(PyExc_TypeError, "env_put requires 1 positional argument");
        return NULL;
    }

    Env *env = unpack_env(args);
    if (!env) {
        return NULL;
    }

    PyObject *empty_args = PyTuple_New(0);
    my_put(env, empty_args, kwargs);
    if (PyErr_Occurred()) {
        return NULL;
    }

    Py_RETURN_NONE;
}

typedef struct {
    Env **envs;
    int num_envs;
} VecEnv;

static VecEnv *unpack_vecenv(PyObject *args) {
    PyObject *handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_handle must be an integer");
        return NULL;
    }

    VecEnv *vec = (VecEnv *) PyLong_AsVoidPtr(handle_obj);
    if (!vec) {
        PyErr_SetString(PyExc_ValueError, "Missing or invalid vec env handle");
        return NULL;
    }

    if (vec->num_envs <= 0) {
        PyErr_SetString(PyExc_ValueError, "Missing or invalid vec env handle");
        return NULL;
    }

    return vec;
}

static PyObject *vec_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    if (PyTuple_Size(args) != 7) {
        PyErr_SetString(PyExc_TypeError, "vec_init requires 6 arguments");
        return NULL;
    }

    VecEnv *vec = (VecEnv *) calloc(1, sizeof(VecEnv));
    if (!vec) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }
    PyObject *num_envs_arg = PyTuple_GetItem(args, 5);
    if (!PyObject_TypeCheck(num_envs_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "num_envs must be an integer");
        return NULL;
    }
    int num_envs = PyLong_AsLong(num_envs_arg);
    if (num_envs <= 0) {
        PyErr_SetString(PyExc_TypeError, "num_envs must be greater than 0");
        return NULL;
    }
    vec->num_envs = num_envs;
    vec->envs = (Env **) calloc(num_envs, sizeof(Env *));
    if (!vec->envs) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }

    PyObject *seed_obj = PyTuple_GetItem(args, 6);
    if (!PyObject_TypeCheck(seed_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "seed must be an integer");
        return NULL;
    }
    int seed = PyLong_AsLong(seed_obj);

    PyObject *obs = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return NULL;
    }
    PyArrayObject *observations = (PyArrayObject *) obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(observations) < 2) {
        PyErr_SetString(PyExc_ValueError, "Batched Observations must be at least 2D");
        return NULL;
    }

    PyObject *act = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return NULL;
    }
    PyArrayObject *actions = (PyArrayObject *) act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return NULL;
    }
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return NULL;
    }

    PyObject *rew = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return NULL;
    }
    PyArrayObject *rewards = (PyArrayObject *) rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return NULL;
    }

    PyObject *term = PyTuple_GetItem(args, 3);
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return NULL;
    }
    PyArrayObject *terminals = (PyArrayObject *) term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return NULL;
    }

    PyObject *trunc = PyTuple_GetItem(args, 4);
    if (!PyObject_TypeCheck(trunc, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Truncations must be a NumPy array");
        return NULL;
    }
    PyArrayObject *truncations = (PyArrayObject *) trunc;
    if (!PyArray_ISCONTIGUOUS(truncations)) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(truncations) != 1) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be 1D");
        return NULL;
    }

    // If kwargs is NULL, create a new dictionary
    if (kwargs == NULL) {
        kwargs = PyDict_New();
    } else {
        Py_INCREF(kwargs); // We need to increment the reference since we'll be modifying it
    }

    for (int i = 0; i < num_envs; i++) {
        Env *env = (Env *) calloc(1, sizeof(Env));
        if (!env) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate environment");
            Py_DECREF(kwargs);
            return NULL;
        }
        vec->envs[i] = env;

        // // Make sure the log is initialized to 0
        memset(&env->log, 0, sizeof(Log));

        env->observations = (void *) ((char *) PyArray_DATA(observations) + i * PyArray_STRIDE(observations, 0));
        env->actions = (void *) ((char *) PyArray_DATA(actions) + i * PyArray_STRIDE(actions, 0));
        env->rewards = (void *) ((char *) PyArray_DATA(rewards) + i * PyArray_STRIDE(rewards, 0));
        env->terminals = (void *) ((char *) PyArray_DATA(terminals) + i * PyArray_STRIDE(terminals, 0));
        env->truncations = (void *) ((char *) PyArray_DATA(truncations) + i * PyArray_STRIDE(truncations, 0));

        // Assumes each process has the same number of environments
        int env_seed = i + seed * vec->num_envs;

        // Add the seed to kwargs for this environment
        PyObject *py_seed = PyLong_FromLong(env_seed);
        if (PyDict_SetItemString(kwargs, "seed", py_seed) < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set seed in kwargs");
            Py_DECREF(py_seed);
            Py_DECREF(kwargs);
            return NULL;
        }
        Py_DECREF(py_seed);

        PyObject *empty_args = PyTuple_New(0);
        my_init(env, empty_args, kwargs);
        if (PyErr_Occurred()) {
            return NULL;
        }
    }

    Py_DECREF(kwargs);
    return PyLong_FromVoidPtr(vec);
}

// Python function to close the environment
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

    vec->envs = (Env **) calloc(num_envs, sizeof(Env *));
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
        vec->envs[i] = (Env *) PyLong_AsVoidPtr(handle_obj);
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

// One row per env whose (single, frozen) eval episode has finished. Envs with no
// completed episode (log.n == 0) are skipped, so no division by zero.
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
        Env *env = vec->envs[i];
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

static PyObject *vec_render(PyObject *self, PyObject *args) {
    int num_args = PyTuple_Size(args);
    if (num_args != 3) {
        PyErr_SetString(PyExc_TypeError, "vec_render requires 3 arguments (vec_env, view_mode, env_id)");
        return NULL;
    }

    VecEnv *vec = (VecEnv *) PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
    if (!vec) {
        PyErr_SetString(PyExc_ValueError, "Invalid vec_env handle");
        return NULL;
    }

    PyObject *view_mode_arg = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(view_mode_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "view_mode must be an integer");
        return NULL;
    }
    int view_mode = PyLong_AsLong(view_mode_arg);

    PyObject *env_id_arg = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
        return NULL;
    }
    int env_id = PyLong_AsLong(env_id_arg);

    c_render(vec->envs[env_id], view_mode);
    Py_RETURN_NONE;
}

// Set the per-env video suffix BEFORE the first vec_render of a rollout.
// make_client reads env->video_suffix when constructing the ffmpeg output
// filename, so multi-view rollouts (sim_state + bev) can produce distinct
// {scenario_id}.mp4 vs {scenario_id}_bev.mp4 without overwrite.
static PyObject *vec_set_video_suffix(PyObject *self, PyObject *args) {
    int num_args = PyTuple_Size(args);
    if (num_args != 3) {
        PyErr_SetString(PyExc_TypeError, "vec_set_video_suffix requires 3 arguments (vec_env, suffix, env_id)");
        return NULL;
    }
    VecEnv *vec = (VecEnv *) PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
    if (!vec) {
        PyErr_SetString(PyExc_ValueError, "Invalid vec_env handle");
        return NULL;
    }
    PyObject *suffix_arg = PyTuple_GetItem(args, 1);
    if (!PyUnicode_Check(suffix_arg)) {
        PyErr_SetString(PyExc_TypeError, "suffix must be a string");
        return NULL;
    }
    const char *suffix = PyUnicode_AsUTF8(suffix_arg);
    if (!suffix) {
        return NULL;
    }
    PyObject *env_id_arg = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
        return NULL;
    }
    int env_id = PyLong_AsLong(env_id_arg);
    if (env_id < 0 || env_id >= vec->num_envs) {
        PyErr_SetString(PyExc_IndexError, "vec_set_video_suffix env_id out of range");
        return NULL;
    }
    Drive *drive = (Drive *) vec->envs[env_id];
    strncpy(drive->video_suffix, suffix, sizeof(drive->video_suffix) - 1);
    drive->video_suffix[sizeof(drive->video_suffix) - 1] = '\0';
    Py_RETURN_NONE;
}

// Explicit per-env client teardown. Distinct from c_close (which tears the
// whole Env down) — this just releases the render Client so ffmpeg/ PBOs
// are flushed without destroying the env. Used by eval renderers that want
// to close out one scenario's mp4 and then reset the env for the next one.
static PyObject *vec_close_client(PyObject *self, PyObject *args) {
    int num_args = PyTuple_Size(args);
    if (num_args != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_close_client requires 2 arguments");
        return NULL;
    }

    VecEnv *vec = (VecEnv *) PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
    if (!vec) {
        PyErr_SetString(PyExc_ValueError, "Invalid vec_env handle");
        return NULL;
    }

    PyObject *env_id_arg = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
        return NULL;
    }
    int env_id = PyLong_AsLong(env_id_arg);

    Env *env = vec->envs[env_id];
    if (env && env->client) {
        close_client(env->client);
        env->client = NULL;
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

    // Iterates over logs one float at a time. Will break
    // horribly if Log has non-float data.
    PyObject *num_agents_arg = PyTuple_GetItem(args, 1);
    float num_agents = (float) PyLong_AsLong(num_agents_arg);
    int num_keys = sizeof(Log) / sizeof(float);

    Log aggregate = {0};
    for (int i = 0; i < vec->num_envs; i++) {
        Env *env = vec->envs[i];
        for (int j = 0; j < num_keys; j++) {
            ((float *) &aggregate)[j] += ((float *) &env->log)[j];
        }
    }

    PyObject *dict = PyDict_New();

    // Only log if we have at least num_agents worth of data
    if (aggregate.n < num_agents) {
        return dict;
    }

    // Got enough data. Reset logs and return metrics
    for (int i = 0; i < vec->num_envs; i++) {
        Env *env = vec->envs[i];
        for (int j = 0; j < num_keys; j++) {
            ((float *) &env->log)[j] = 0.0f;
        }
    }

    float n = aggregate.n;

    // Average across agents
    for (int i = 0; i < num_keys; i++) {
        ((float *) &aggregate)[i] /= n;
    }
    // User populates dict
    Env *env = vec->envs[0];
    my_log(dict, env, &aggregate, n);
    assign_to_dict(dict, "n", n);
    return dict;
}

static PyObject *vec_get(PyObject *self, PyObject *args) {
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        PyErr_SetString(PyExc_ValueError, "Invalid VecEnv handle");
        return NULL;
    }

    PyObject *list = PyList_New(vec->num_envs);
    if (!list) {
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        Env *env = vec->envs[i];
        if (!env) {
            Py_INCREF(Py_None);
            PyList_SetItem(list, i, Py_None);
            continue;
        }
        PyObject *dict = PyDict_New();
        if (!dict) {
            Py_DECREF(list);
            return NULL;
        }
        PyObject *res = my_get(dict, env);
        if (res == NULL) {
            Py_DECREF(dict);
            Py_DECREF(list);
            return NULL;
        }
        /* my_get returns the dict (or NULL on error) */
        PyList_SetItem(list, i, dict);
    }

    return list;
}

static PyObject *vec_get_obs_html_frame(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 7) {
        PyErr_SetString(PyExc_TypeError, "vec_get_obs_html_frame requires 7 arguments");
        return NULL;
    }

    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyArrayObject *agent_f32_array = (PyArrayObject *) PyTuple_GetItem(args, 1);
    PyArrayObject *agent_i32_array = (PyArrayObject *) PyTuple_GetItem(args, 2);
    PyArrayObject *metrics_f32_array = (PyArrayObject *) PyTuple_GetItem(args, 3);
    PyArrayObject *puffer_f32_array = (PyArrayObject *) PyTuple_GetItem(args, 4);
    PyArrayObject *traffic_i16_array = (PyArrayObject *) PyTuple_GetItem(args, 5);
    PyArrayObject *rewards_f32_array = (PyArrayObject *) PyTuple_GetItem(args, 6);

    if (!PyArray_Check(agent_f32_array) || !PyArray_Check(agent_i32_array) || !PyArray_Check(metrics_f32_array)
        || !PyArray_Check(puffer_f32_array) || !PyArray_Check(traffic_i16_array) || !PyArray_Check(rewards_f32_array)) {
        PyErr_SetString(PyExc_TypeError, "All output arrays must be NumPy arrays");
        return NULL;
    }

    memset(PyArray_DATA(agent_f32_array), 0, PyArray_NBYTES(agent_f32_array));
    memset(PyArray_DATA(agent_i32_array), 0, PyArray_NBYTES(agent_i32_array));
    memset(PyArray_DATA(metrics_f32_array), 0, PyArray_NBYTES(metrics_f32_array));
    memset(PyArray_DATA(puffer_f32_array), 0, PyArray_NBYTES(puffer_f32_array));
    memset(PyArray_DATA(traffic_i16_array), 0, PyArray_NBYTES(traffic_i16_array));
    memset(PyArray_DATA(rewards_f32_array), 0, PyArray_NBYTES(rewards_f32_array));

    float *agent_f32 = (float *) PyArray_DATA(agent_f32_array);
    int *agent_i32 = (int *) PyArray_DATA(agent_i32_array);
    float *metrics_f32 = (float *) PyArray_DATA(metrics_f32_array);
    float *puffer_f32 = (float *) PyArray_DATA(puffer_f32_array);
    short *traffic_i16 = (short *) PyArray_DATA(traffic_i16_array);
    float *rewards_f32 = (float *) PyArray_DATA(rewards_f32_array);

    int env_cap = (int) PyArray_DIM(agent_f32_array, 0);
    int env_count = vec->num_envs < env_cap ? vec->num_envs : env_cap;
    int agent_cap = (int) PyArray_DIM(agent_f32_array, 1);
    int agent_f32_fields = (int) PyArray_DIM(agent_f32_array, 2);
    int agent_i32_fields = (int) PyArray_DIM(agent_i32_array, 2);
    int metric_fields = (int) PyArray_DIM(metrics_f32_array, 2);
    int puffer_fields = (int) PyArray_DIM(puffer_f32_array, 2);
    int traffic_cap = (int) PyArray_DIM(traffic_i16_array, 1);
    int traffic_fields = (int) PyArray_DIM(traffic_i16_array, 2);
    int reward_fields = (int) PyArray_DIM(rewards_f32_array, 2);

    for (int e = 0; e < env_count; e++) {
        Drive *drive = (Drive *) vec->envs[e];
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
        }

        if (drive->active_agent_indices) {
            for (int j = 0; j < drive->active_agent_count; j++) {
                int agent_idx = drive->active_agent_indices[j];
                if (agent_idx < 0 || agent_idx >= agent_count) {
                    continue;
                }
                int i32_base = (e * agent_cap + agent_idx) * agent_i32_fields;
                int puffer_base = (e * agent_cap + agent_idx) * puffer_fields;
                agent_i32[i32_base + 7] = j;

                if (!drive->compute_eval_metrics || !drive->logs || j >= drive->logs_capacity) {
                    continue;
                }
                Log *log = &drive->logs[j];
                puffer_f32[puffer_base + 0] = log->puffer_score;
                puffer_f32[puffer_base + 1] = log->no_at_fault;
                puffer_f32[puffer_base + 2] = log->no_offroad;
                puffer_f32[puffer_base + 3] = log->no_red_light;
                puffer_f32[puffer_base + 4] = log->making_progress;
                puffer_f32[puffer_base + 5] = log->driving_direction_score;
                puffer_f32[puffer_base + 6] = log->ttc_puffer_rate;
                puffer_f32[puffer_base + 7] = log->progress_ratio;
                puffer_f32[puffer_base + 8] = log->speed_limit_compliance;
                puffer_f32[puffer_base + 9] = log->comfort_score;
                puffer_f32[puffer_base + 10] = log->multi_lane_score;
                puffer_f32[puffer_base + 11] = log->wrong_way_distance;
                puffer_f32[puffer_base + 12] = log->speed_violation_sum;
                puffer_f32[puffer_base + 13] = log->multiplier;
                puffer_f32[puffer_base + 14] = log->weighted_average;

                int reward_base = (e * agent_cap + agent_idx) * reward_fields;
                rewards_f32[reward_base + 0] = log->episode_return;
                rewards_f32[reward_base + 1] = log->reward_collision;
                rewards_f32[reward_base + 2] = log->reward_offroad;
                rewards_f32[reward_base + 3] = log->reward_red_light;
                rewards_f32[reward_base + 4] = log->reward_goal;
                rewards_f32[reward_base + 5] = log->reward_lane_align;
                rewards_f32[reward_base + 6] = log->reward_lane_center;
                rewards_f32[reward_base + 7] = log->reward_comfort;
                rewards_f32[reward_base + 8] = log->reward_velocity;
                rewards_f32[reward_base + 9] = log->reward_timestep;
                rewards_f32[reward_base + 10] = log->reward_reverse;
                rewards_f32[reward_base + 11] = log->reward_overspeed;
                rewards_f32[reward_base + 12] = log->reward_ade;
            }
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

static PyObject *get_global_agent_state(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 7) {
        PyErr_SetString(PyExc_TypeError, "get_global_agent_state requires 7 arguments");
        return NULL;
    }

    Env *env = unpack_env(args);
    if (!env) {
        return NULL;
    }

    Drive *drive = (Drive *) env; // Cast to Drive*

    // Get the numpy arrays from arguments
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

    float *x_data = (float *) PyArray_DATA((PyArrayObject *) x_arr);
    float *y_data = (float *) PyArray_DATA((PyArrayObject *) y_arr);
    float *z_data = (float *) PyArray_DATA((PyArrayObject *) z_arr);
    float *heading_data = (float *) PyArray_DATA((PyArrayObject *) heading_arr);
    int *id_data = (int *) PyArray_DATA((PyArrayObject *) id_arr);
    float *length_data = (float *) PyArray_DATA((PyArrayObject *) length_arr);
    float *width_data = (float *) PyArray_DATA((PyArrayObject *) width_arr);

    c_get_global_agent_state(drive, 0, x_data, y_data, z_data, heading_data, id_data, length_data, width_data);

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

    // Get the numpy arrays from arguments
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

    PyArrayObject *x_array = (PyArrayObject *) x_arr;
    PyArrayObject *y_array = (PyArrayObject *) y_arr;
    PyArrayObject *z_array = (PyArrayObject *) z_arr;
    PyArrayObject *heading_array = (PyArrayObject *) heading_arr;
    PyArrayObject *id_array = (PyArrayObject *) id_arr;
    PyArrayObject *length_array = (PyArrayObject *) length_arr;
    PyArrayObject *width_array = (PyArrayObject *) width_arr;

    // Get base pointers to the arrays
    float *x_base = (float *) PyArray_DATA(x_array);
    float *y_base = (float *) PyArray_DATA(y_array);
    float *z_base = (float *) PyArray_DATA(z_array);
    float *heading_base = (float *) PyArray_DATA(heading_array);
    int *id_base = (int *) PyArray_DATA(id_array);
    float *length_base = (float *) PyArray_DATA(length_array);
    float *width_base = (float *) PyArray_DATA(width_array);

    // Iterate through environments and write to correct offsets
    int offset = 0;
    for (int i = 0; i < vec->num_envs; i++) {
        Drive *drive = (Drive *) vec->envs[i];

        // Write to the arrays at the current offset
        c_get_global_agent_state(
            drive,
            include_static,
            &x_base[offset],
            &y_base[offset],
            &z_base[offset],
            &heading_base[offset],
            &id_base[offset],
            &length_base[offset],
            &width_base[offset]);

        // Move offset forward by the number of agents in this environment
        offset += include_static ? drive->num_agents : drive->active_agent_count;
    }

    Py_RETURN_NONE;
}

// ── Co-simulation external-state setters (mirror vec_get_global_agent_state) ──
// Co-sim runs a single env (num_envs == 1); these operate on vec->envs[0].
static PyObject *vec_set_agent_states(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 10) {
        PyErr_SetString(PyExc_TypeError, "vec_set_agent_states requires 10 arguments");
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
    if (!PyArray_Check(idx_arr) || !PyArray_Check(x_arr) || !PyArray_Check(y_arr) || !PyArray_Check(z_arr)
        || !PyArray_Check(heading_arr) || !PyArray_Check(vx_arr) || !PyArray_Check(vy_arr)
        || !PyArray_Check(yaw_rate_arr) || !PyArray_Check(accel_long_arr)) {
        PyErr_SetString(PyExc_TypeError, "All arrays must be NumPy arrays");
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
    int count = (int) PyArray_SIZE((PyArrayObject *) idx_arr);
    PyObject *value_arrs[8] = {x_arr, y_arr, z_arr, heading_arr, vx_arr, vy_arr, yaw_rate_arr, accel_long_arr};
    for (int k = 0; k < 8; k++) {
        if ((int) PyArray_SIZE((PyArrayObject *) value_arrs[k]) != count) {
            PyErr_SetString(PyExc_ValueError, "vec_set_agent_states: every array must have the same length as idx");
            return NULL;
        }
    }
    if (c_set_agent_states((Drive *) vec->envs[0], count, idx, x, y, z, heading, vx, vy, yaw_rate, accel_long) != 0) {
        PyErr_SetString(PyExc_ValueError, "vec_set_agent_states: agent index out of range or non-finite state");
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
    if (c_set_agent_sizes((Drive *) vec->envs[0], count, idx, length, width) != 0) {
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
        compute_observations((Drive *) vec->envs[i]);
    }
    Py_RETURN_NONE;
}

static PyObject *vec_set_traffic_light_states(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_set_traffic_light_states requires 2 arguments");
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
    Drive *drive = (Drive *) vec->envs[0];
    if ((int) PyArray_SIZE((PyArrayObject *) states_arr) != drive->num_traffic_elements) {
        PyErr_Format(
            PyExc_ValueError,
            "vec_set_traffic_light_states: expected %d states (one per traffic element), got %d",
            drive->num_traffic_elements,
            (int) PyArray_SIZE((PyArrayObject *) states_arr));
        return NULL;
    }
    if (c_set_traffic_light_states(drive, states) != 0) {
        PyErr_SetString(
            PyExc_ValueError, "vec_set_traffic_light_states: timestep outside the light state buffer or invalid state");
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
    Drive *drive = (Drive *) vec->envs[0];
    int agent_idx = (int) PyLong_AsLong(PyTuple_GetItem(args, 1));
    if (agent_idx < 0 || agent_idx >= drive->num_total_agents) {
        PyErr_SetString(PyExc_ValueError, "vec_get_agent_goal_progress: agent index out of range");
        return NULL;
    }
    int current_goal_idx, goal_count;
    c_get_agent_goal_progress(drive, agent_idx, &current_goal_idx, &goal_count);
    return Py_BuildValue("(ii)", current_goal_idx, goal_count);
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
    if (c_set_agent_goals((Drive *) vec->envs[0], agent_idx, num_wp, gx, gy, gz, gdx, gdy) != 0) {
        PyErr_Format(
            PyExc_ValueError,
            "vec_set_agent_goals: agent index out of range, waypoint count outside 1..%d, or non-finite goal",
            MAX_GOALS);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *get_ground_truth_trajectories(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 7) {
        PyErr_SetString(PyExc_TypeError, "get_ground_truth_trajectories requires 7 arguments");
        return NULL;
    }

    Env *env = unpack_env(args);
    if (!env) {
        return NULL;
    }

    Drive *drive = (Drive *) env;

    // Get the numpy arrays from arguments
    PyObject *x_arr = PyTuple_GetItem(args, 1);
    PyObject *y_arr = PyTuple_GetItem(args, 2);
    PyObject *z_arr = PyTuple_GetItem(args, 3);
    PyObject *heading_arr = PyTuple_GetItem(args, 4);
    PyObject *valid_arr = PyTuple_GetItem(args, 5);
    PyObject *id_arr = PyTuple_GetItem(args, 6);
    PyObject *scenario_id_arr = PyTuple_GetItem(args, 7);

    if (!PyArray_Check(x_arr) || !PyArray_Check(y_arr) || !PyArray_Check(z_arr) || !PyArray_Check(heading_arr)
        || !PyArray_Check(valid_arr) || !PyArray_Check(id_arr) || !PyArray_Check(scenario_id_arr)) {
        PyErr_SetString(PyExc_TypeError, "All output arrays must be NumPy arrays");
        return NULL;
    }

    float *x_data = (float *) PyArray_DATA((PyArrayObject *) x_arr);
    float *y_data = (float *) PyArray_DATA((PyArrayObject *) y_arr);
    float *z_data = (float *) PyArray_DATA((PyArrayObject *) z_arr);
    float *heading_data = (float *) PyArray_DATA((PyArrayObject *) heading_arr);
    int *valid_data = (int *) PyArray_DATA((PyArrayObject *) valid_arr);
    int *id_data = (int *) PyArray_DATA((PyArrayObject *) id_arr);
    int *scenario_id_data = (int *) PyArray_DATA((PyArrayObject *) scenario_id_arr);

    c_get_global_ground_truth_trajectories(
        drive,
        x_data,
        y_data,
        z_data,
        heading_data,
        valid_data,
        id_data,
        scenario_id_data);

    Py_RETURN_NONE;
}

static PyObject *vec_get_global_ground_truth_trajectories(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 8) {
        PyErr_SetString(PyExc_TypeError, "vec_get_global_ground_truth_trajectories requires 8 arguments");
        return NULL;
    }

    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    // Get the numpy arrays from arguments
    PyObject *x_arr = PyTuple_GetItem(args, 1);
    PyObject *y_arr = PyTuple_GetItem(args, 2);
    PyObject *z_arr = PyTuple_GetItem(args, 3);
    PyObject *heading_arr = PyTuple_GetItem(args, 4);
    PyObject *valid_arr = PyTuple_GetItem(args, 5);
    PyObject *id_arr = PyTuple_GetItem(args, 6);
    PyObject *scenario_id_arr = PyTuple_GetItem(args, 7);

    if (!PyArray_Check(x_arr) || !PyArray_Check(y_arr) || !PyArray_Check(z_arr) || !PyArray_Check(heading_arr)
        || !PyArray_Check(valid_arr) || !PyArray_Check(id_arr) || !PyArray_Check(scenario_id_arr)) {
        PyErr_SetString(PyExc_TypeError, "All output arrays must be NumPy arrays");
        return NULL;
    }

    PyArrayObject *x_array = (PyArrayObject *) x_arr;
    PyArrayObject *y_array = (PyArrayObject *) y_arr;
    PyArrayObject *z_array = (PyArrayObject *) z_arr;
    PyArrayObject *heading_array = (PyArrayObject *) heading_arr;
    PyArrayObject *valid_array = (PyArrayObject *) valid_arr;
    PyArrayObject *id_array = (PyArrayObject *) id_arr;
    PyArrayObject *scenario_id_array = (PyArrayObject *) scenario_id_arr;

    // Get base pointers to the arrays
    float *x_base = (float *) PyArray_DATA(x_array);
    float *y_base = (float *) PyArray_DATA(y_array);
    float *z_base = (float *) PyArray_DATA(z_array);
    float *heading_base = (float *) PyArray_DATA(heading_array);
    int *valid_base = (int *) PyArray_DATA(valid_array);
    int *id_base = (int *) PyArray_DATA(id_array);
    int *scenario_id_base = (int *) PyArray_DATA(scenario_id_array);

    // Get number of timesteps from array shape
    npy_intp *x_shape = PyArray_DIMS(x_array);
    int num_timesteps = x_shape[1]; // Second dimension for 2D arrays

    // Iterate through environments and write to correct offsets
    int agent_offset = 0; // Offset for 1D arrays (id, scenario_id)
    int traj_offset = 0;  // Offset for 2D arrays (x, y, z, heading, valid)

    for (int i = 0; i < vec->num_envs; i++) {
        Drive *drive = (Drive *) vec->envs[i];

        c_get_global_ground_truth_trajectories(
            drive,
            &x_base[traj_offset],
            &y_base[traj_offset],
            &z_base[traj_offset],
            &heading_base[traj_offset],
            &valid_base[traj_offset],
            &id_base[agent_offset],
            &scenario_id_base[agent_offset]);

        // Move offsets forward
        agent_offset += drive->active_agent_count;
        traj_offset += drive->active_agent_count * num_timesteps;
    }

    Py_RETURN_NONE;
}

static PyObject *vec_get_road_edge_counts(PyObject *self, PyObject *args) {
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    int total_polylines = 0, total_points = 0;
    for (int i = 0; i < vec->num_envs; i++) {
        Drive *drive = (Drive *) vec->envs[i];
        int np, tp;
        c_get_road_edge_counts(drive, &np, &tp);
        total_polylines += np;
        total_points += tp;
    }
    return Py_BuildValue("(ii)", total_polylines, total_points);
}

static PyObject *vec_get_road_edge_polylines(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 5) {
        PyErr_SetString(PyExc_TypeError, "vec_get_road_edge_polylines requires 5 arguments");
        return NULL;
    }

    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject *x_arr = PyTuple_GetItem(args, 1);
    PyObject *y_arr = PyTuple_GetItem(args, 2);
    PyObject *lengths_arr = PyTuple_GetItem(args, 3);
    PyObject *scenario_ids_arr = PyTuple_GetItem(args, 4);

    if (!PyArray_Check(x_arr) || !PyArray_Check(y_arr) || !PyArray_Check(lengths_arr)
        || !PyArray_Check(scenario_ids_arr)) {
        PyErr_SetString(PyExc_TypeError, "All output arrays must be NumPy arrays");
        return NULL;
    }

    float *x_base = (float *) PyArray_DATA((PyArrayObject *) x_arr);
    float *y_base = (float *) PyArray_DATA((PyArrayObject *) y_arr);
    int *lengths_base = (int *) PyArray_DATA((PyArrayObject *) lengths_arr);
    int *scenario_ids_base = (int *) PyArray_DATA((PyArrayObject *) scenario_ids_arr);

    int poly_offset = 0, pt_offset = 0;
    for (int i = 0; i < vec->num_envs; i++) {
        Drive *drive = (Drive *) vec->envs[i];
        int np, tp;
        c_get_road_edge_counts(drive, &np, &tp);
        c_get_road_edge_polylines(
            drive,
            &x_base[pt_offset],
            &y_base[pt_offset],
            &lengths_base[poly_offset],
            &scenario_ids_base[poly_offset]);
        poly_offset += np;
        pt_offset += tp;
    }
    Py_RETURN_NONE;
}

static double unpack(PyObject *kwargs, char *key) {
    PyObject *val = PyDict_GetItemString(kwargs, key);
    if (val == NULL) {
        char error_msg[100];
        snprintf(error_msg, sizeof(error_msg), "Missing required keyword argument '%s'", key);
        PyErr_SetString(PyExc_TypeError, error_msg);
        return 1;
    }
    if (PyLong_Check(val)) {
        long out = PyLong_AsLong(val);
        if (out > INT_MAX || out < INT_MIN) {
            char error_msg[100];
            snprintf(error_msg, sizeof(error_msg), "Value %ld of integer argument %s is out of range", out, key);
            PyErr_SetString(PyExc_TypeError, error_msg);
            return 1;
        }
        // Cast on return. Safe because double can represent all 32-bit ints exactly
        return out;
    }
    if (PyFloat_Check(val)) {
        return PyFloat_AsDouble(val);
    }
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Failed to unpack keyword %s as int", key);
    PyErr_SetString(PyExc_TypeError, error_msg);
    return 1;
}

static char *unpack_str(PyObject *kwargs, char *key) {
    PyObject *val = PyDict_GetItemString(kwargs, key);
    if (val == NULL) {
        char error_msg[100];
        snprintf(error_msg, sizeof(error_msg), "Missing required keyword argument '%s'", key);
        PyErr_SetString(PyExc_TypeError, error_msg);
        return NULL;
    }
    if (!PyUnicode_Check(val)) {
        char error_msg[100];
        snprintf(error_msg, sizeof(error_msg), "Keyword argument '%s' must be a string", key);
        PyErr_SetString(PyExc_TypeError, error_msg);
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

// Method table
static PyMethodDef methods[]
    = {{"env_init",
        (PyCFunction) env_init,
        METH_VARARGS | METH_KEYWORDS,
        "Init environment with observation, action, reward, terminal, truncation arrays"},
       {"env_reset", env_reset, METH_VARARGS, "Reset the environment"},
       {"env_step", env_step, METH_VARARGS, "Step the environment"},
       {"env_render", env_render, METH_VARARGS, "Render the environment"},
       {"env_close", env_close, METH_VARARGS, "Close the environment"},
       {"env_get", env_get, METH_VARARGS, "Get the environment state"},
       {"env_put", (PyCFunction) env_put, METH_VARARGS | METH_KEYWORDS, "Put stuff into env"},
       {"vectorize", vectorize, METH_VARARGS, "Make a vector of environment handles"},
       {"vec_init", (PyCFunction) vec_init, METH_VARARGS | METH_KEYWORDS, "Initialize a vector of environments"},
       {"vec_reset", vec_reset, METH_VARARGS, "Reset the vector of environments"},
       {"vec_step", vec_step, METH_VARARGS, "Step the vector of environments"},
       {"vec_per_episode_log",
        vec_per_episode_log,
        METH_VARARGS,
        "Return one per-episode metrics dict per env whose eval episode has finished"},
       {"vec_log", vec_log, METH_VARARGS, "Log the vector of environments"},
       {"vec_render", vec_render, METH_VARARGS, "Render the vector of environments"},
       {"vec_set_video_suffix", vec_set_video_suffix, METH_VARARGS, "Set the mp4 filename suffix for an env"},
       {"vec_close_client",
        vec_close_client,
        METH_VARARGS,
        "Release a single env's render client without destroying the env"},
       {"vec_close", vec_close, METH_VARARGS, "Close the vector of environments"},
       {"vec_get", vec_get, METH_VARARGS, "Get attributes from each env in a VecEnv"},
       {"vec_get_obs_html_frame",
        vec_get_obs_html_frame,
        METH_VARARGS,
        "Fill compact obs_html frame arrays from a VecEnv"},
       {"shared", (PyCFunction) my_shared, METH_VARARGS | METH_KEYWORDS, "Shared state"},
       {"get_global_agent_state", get_global_agent_state, METH_VARARGS, "Get global agent state"},
       {"vec_get_global_agent_state", vec_get_global_agent_state, METH_VARARGS, "Get agent state from vectorized env"},
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
       {"vec_set_agent_goals", vec_set_agent_goals, METH_VARARGS, "Set an agent's goal waypoints (co-sim)"},
       {"vec_get_agent_goal_progress",
        vec_get_agent_goal_progress,
        METH_VARARGS,
        "(current_goal_idx, goal_count) of one agent (co-sim)"},
       {"get_ground_truth_trajectories", get_ground_truth_trajectories, METH_VARARGS, "Get ground truth trajectories"},
       {"vec_get_global_ground_truth_trajectories",
        vec_get_global_ground_truth_trajectories,
        METH_VARARGS,
        "Get ground truth trajectories from vectorized env"},
       {"vec_get_road_edge_counts",
        vec_get_road_edge_counts,
        METH_VARARGS,
        "Get road edge polyline counts from vectorized env"},
       {"vec_get_road_edge_polylines",
        vec_get_road_edge_polylines,
        METH_VARARGS,
        "Get road edge polylines from vectorized env"},
       MY_METHODS,
       {NULL, NULL, 0, NULL}};

// Module definition
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "binding", NULL, -1, methods};

// Publish a C float table as an immutable Python tuple so policy code reads the
// sim's values instead of copying them. Returns 0 on success, -1 on failure.
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
    PyObject *m = PyModule_Create(&module); // Changed variable name from 'module' to 'm'

    if (m == NULL) {
        return NULL;
    }

    // Make constants accessible from Python
    PyModule_AddIntConstant(m, "MAX_ENTITIES_PER_CELL", MAX_ENTITIES_PER_CELL);
    PyModule_AddIntConstant(m, "LANE_FEATURES", LANE_FEATURES);
    PyModule_AddObject(m, "LANE_CURVATURE_NORM", PyFloat_FromDouble(LANE_CURVATURE_NORM));
    PyModule_AddIntConstant(m, "BOUNDARY_FEATURES", BOUNDARY_FEATURES);
    PyModule_AddIntConstant(m, "PARTNER_FEATURES", PARTNER_FEATURES);
    PyModule_AddIntConstant(m, "PARTNER_RELATIVE_VELOCITY_FEATURES", PARTNER_RELATIVE_VELOCITY_FEATURES);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_FEATURES", TRAFFIC_CONTROL_FEATURES);
    PyModule_AddIntConstant(m, "OBS_VALID_COUNT_FEATURES", OBS_VALID_COUNT_FEATURES);
    PyModule_AddIntConstant(m, "NUM_TRAFFIC_CONTROL_TYPES", NUM_TRAFFIC_CONTROL_TYPES);
    PyModule_AddIntConstant(m, "NUM_TRAFFIC_CONTROL_STATES", NUM_TRAFFIC_CONTROL_STATES);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_TYPE_NONE", TRAFFIC_CONTROL_TYPE_NONE);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT", TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_TYPE_STOP_SIGN", TRAFFIC_CONTROL_TYPE_STOP_SIGN);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_TYPE_YIELD_SIGN", TRAFFIC_CONTROL_TYPE_YIELD_SIGN);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_STATE_UNKNOWN", TRAFFIC_CONTROL_STATE_UNKNOWN);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_STATE_RED", TRAFFIC_CONTROL_STATE_RED);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_STATE_YELLOW", TRAFFIC_CONTROL_STATE_YELLOW);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_STATE_GREEN", TRAFFIC_CONTROL_STATE_GREEN);
    PyModule_AddIntConstant(m, "TRAFFIC_CONTROL_STATE_OFF", TRAFFIC_CONTROL_STATE_OFF);
    PyModule_AddIntConstant(m, "EGO_FEATURES", EGO_FEATURES);
    PyModule_AddIntConstant(m, "GOAL_FEATURES", GOAL_FEATURES);
    PyModule_AddIntConstant(m, "MAX_GOALS", MAX_GOALS);
    PyModule_AddIntConstant(m, "AGENT_F32_FIELDS", AGENT_F32_FIELDS);
    PyModule_AddIntConstant(m, "AGENT_I32_FIELDS", AGENT_I32_FIELDS);
    PyModule_AddIntConstant(m, "METRICS_F32_FIELDS", METRICS_F32_FIELDS);
    PyModule_AddIntConstant(m, "SCORE_F32_FIELDS", SCORE_F32_FIELDS);
    PyModule_AddIntConstant(m, "REWARD_F32_FIELDS", REWARD_F32_FIELDS);
    PyModule_AddIntConstant(m, "TRAFFIC_I16_FIELDS", TRAFFIC_I16_FIELDS);
    PyModule_AddIntConstant(m, "NUM_REWARD_COEFS", NUM_REWARD_COEFS);
    PyModule_AddIntConstant(m, "GOAL_REGEN_FINITE", GOAL_REGEN_FINITE);
    PyModule_AddIntConstant(m, "GOAL_REGEN_ROLLING", GOAL_REGEN_ROLLING);
    PyModule_AddIntConstant(m, "GOAL_SOURCE_ROUTE", GOAL_SOURCE_ROUTE);
    PyModule_AddIntConstant(m, "GOAL_SOURCE_MAP", GOAL_SOURCE_MAP);
    PyModule_AddIntConstant(m, "GOAL_SOURCE_GT", GOAL_SOURCE_GT);
    PyModule_AddIntConstant(m, "GOAL_SOURCE_EXTERNAL", GOAL_SOURCE_EXTERNAL);
    PyModule_AddIntConstant(m, "CONTROLLER_STATIC", CONTROLLER_STATIC);
    PyModule_AddIntConstant(m, "CONTROLLER_POLICY", CONTROLLER_POLICY);
    PyModule_AddIntConstant(m, "CONTROLLER_REPLAY", CONTROLLER_REPLAY);
    PyModule_AddIntConstant(m, "CONTROLLER_IDM", CONTROLLER_IDM);
    PyModule_AddIntConstant(m, "INFRACTION_BEHAVIOR_IGNORE", INFRACTION_BEHAVIOR_IGNORE);
    PyModule_AddIntConstant(m, "INFRACTION_BEHAVIOR_STOP", INFRACTION_BEHAVIOR_STOP);
    PyModule_AddIntConstant(m, "INFRACTION_BEHAVIOR_REMOVE", INFRACTION_BEHAVIOR_REMOVE);
    PyModule_AddIntConstant(m, "SIMULATION_MODE_GIGAFLOW", SIMULATION_MODE_GIGAFLOW);
    PyModule_AddIntConstant(m, "SIMULATION_MODE_REPLAY", SIMULATION_MODE_REPLAY);
    // Ego obs normalization (obs[4] * ACCEL_LONG_NORM = accel_long m/s^2); co-sim reads
    // the shadow ego's post-step accel intent back out of the observation row. The speed
    // norm (obs[0]) is per-env config: env.obs_norm_speed_mps.
    PyModule_AddObject(m, "ACCEL_LONG_NORM", PyFloat_FromDouble(fabsf(ACCEL_LONG_LIMIT[0])));
    PyModule_AddObject(m, "ACCEL_LONG_MAX", PyFloat_FromDouble(ACCEL_LONG_LIMIT[1]));
    PyModule_AddObject(m, "CONDITIONING_ACC_MIN", PyFloat_FromDouble(REWARD_BOUNDS[REWARD_COEF_ACC].min_val));
    PyModule_AddObject(m, "CONDITIONING_ACC_MAX", PyFloat_FromDouble(REWARD_BOUNDS[REWARD_COEF_ACC].max_val));
    PyModule_AddIntConstant(m, "ACTION_TYPE_DISCRETE", ACTION_TYPE_DISCRETE);
    PyModule_AddIntConstant(m, "ACTION_TYPE_CONTINUOUS", ACTION_TYPE_CONTINUOUS);
    PyModule_AddIntConstant(m, "DYNAMICS_MODEL_CLASSIC", DYNAMICS_MODEL_CLASSIC);
    PyModule_AddIntConstant(m, "DYNAMICS_MODEL_JERK", DYNAMICS_MODEL_JERK);
    PyModule_AddIntConstant(m, "CONTROL_MODE_VEHICLES", CONTROL_MODE_VEHICLES);
    PyModule_AddIntConstant(m, "CONTROL_MODE_AGENTS", CONTROL_MODE_AGENTS);
    PyModule_AddIntConstant(m, "CONTROL_MODE_WOSAC", CONTROL_MODE_WOSAC);
    PyModule_AddIntConstant(m, "CONTROL_MODE_SDC_ONLY", CONTROL_MODE_SDC_ONLY);
    PyModule_AddIntConstant(m, "INIT_MODE_CREATE_ALL_VALID", INIT_MODE_CREATE_ALL_VALID);
    PyModule_AddIntConstant(m, "INIT_MODE_CREATE_ONLY_CONTROLLED", INIT_MODE_CREATE_ONLY_CONTROLLED);
    PyModule_AddIntConstant(m, "INIT_MODE_CREATE_CONTROLLABLE_TYPES", INIT_MODE_CREATE_CONTROLLABLE_TYPES);
    PyObject_SetAttrString(m, "MULTI_LANE_FULL_SCORE_TIME", PyFloat_FromDouble(MULTI_LANE_FULL_SCORE_TIME));
    PyObject_SetAttrString(m, "MULTI_LANE_HALF_SCORE_TIME", PyFloat_FromDouble(MULTI_LANE_HALF_SCORE_TIME));

    // Action discretization tables: the policy decodes discrete actions with the
    // exact values the sim uses, so the two can never drift apart.
    if (add_float_table_constant(m, "JERK_LONG", JERK_LONG, NUM_JERK_LONG_ACTIONS) < 0
        || add_float_table_constant(m, "JERK_LAT", JERK_LAT, NUM_JERK_LAT_ACTIONS) < 0
        || add_float_table_constant(m, "ACCELERATION_VALUES", ACCELERATION_VALUES, NUM_ACCELERATION_ACTIONS) < 0
        || add_float_table_constant(m, "STEERING_VALUES", STEERING_VALUES, NUM_STEERING_ACTIONS) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
