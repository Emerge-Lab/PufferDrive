#include "drive.h"

#include <Python.h>

#define Env Drive
#define MY_SHARED
#define MY_PUT
#define MY_GET

// Total slot count of g_map_cache (live entries plus NULL holes from freed entries).
static PyObject *map_cache_size_py(PyObject *self __attribute__((unused)), PyObject *args __attribute__((unused))) {
    return PyLong_FromLong((long) g_map_cache_count);
}

// Count of slots currently holding a SharedMapData (non-NULL slots only).
static PyObject *map_cache_live_count_py(
    PyObject *self __attribute__((unused)),
    PyObject *args __attribute__((unused))) {
    long live = 0;
    for (int i = 0; i < g_map_cache_count; i++) {
        if (g_map_cache[i] != NULL) {
            live++;
        }
    }
    return PyLong_FromLong(live);
}

// clang-format off
#define MY_METHODS \
    {"map_cache_size", map_cache_size_py, METH_NOARGS, "Map cache slot count."}, \
    {"map_cache_live_count", map_cache_live_count_py, METH_NOARGS, "Map cache live count."}
// clang-format on

#include "../env_binding.h"

static int my_put(Env *env, PyObject *args, PyObject *kwargs) {
    PyObject *obs = PyDict_GetItemString(kwargs, "observations");
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return 1;
    }
    PyArrayObject *observations = (PyArrayObject *) obs;
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
    PyArrayObject *actions = (PyArrayObject *) act;
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
    PyArrayObject *rewards = (PyArrayObject *) rew;
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
    PyArrayObject *terminals = (PyArrayObject *) term;
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

static PyObject *my_get(PyObject *dict, Env *env) {
    PyObject *v;
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "env is NULL");
        return NULL;
    }

    /* Validate main array pointers before accessing */
    if (env->num_total_agents > 0 && !env->agents) {
        PyErr_SetString(PyExc_ValueError, "agents is NULL but count > 0");
        return NULL;
    }
    if (env->num_road_elements > 0 && !env->road_elements) {
        PyErr_SetString(PyExc_ValueError, "road_elements is NULL but count > 0");
        return NULL;
    }
    if (env->num_traffic_elements > 0 && !env->traffic_elements) {
        PyErr_SetString(PyExc_ValueError, "traffic_elements is NULL but count > 0");
        return NULL;
    }

    v = PyLong_FromLong(env->active_agent_count);
    if (!v) {
        return NULL;
    }
    if (PyDict_SetItemString(dict, "active_agent_count", v) < 0) {
        Py_DECREF(v);
        return NULL;
    }
    Py_DECREF(v);

    v = PyLong_FromLong(env->num_total_agents);
    if (!v) {
        return NULL;
    }
    if (PyDict_SetItemString(dict, "num_total_agents", v) < 0) {
        Py_DECREF(v);
        return NULL;
    }
    Py_DECREF(v);

    v = PyLong_FromLong(env->num_road_elements);
    if (!v) {
        return NULL;
    }
    if (PyDict_SetItemString(dict, "num_road_elements", v) < 0) {
        Py_DECREF(v);
        return NULL;
    }
    Py_DECREF(v);

    v = PyLong_FromLong(env->num_traffic_elements);
    if (!v) {
        return NULL;
    }
    if (PyDict_SetItemString(dict, "num_traffic_elements", v) < 0) {
        Py_DECREF(v);
        return NULL;
    }
    Py_DECREF(v);

    /* Map name / string fields */
    if (env->map_name) {
        PyObject *s = PyUnicode_FromString(env->map_name);
        if (!s) {
            return NULL;
        }
        if (PyDict_SetItemString(dict, "map_name", s) < 0) {
            Py_DECREF(s);
            return NULL;
        }
        Py_DECREF(s);
    } else {
        if (PyDict_SetItemString(dict, "map_name", Py_None) < 0) {
            return NULL;
        }
    }

    /* Metadata fields */
    if (env->scenario_id[0] != '\0') {
        PyObject *s = PyUnicode_FromString(env->scenario_id);
        if (!s) {
            return NULL;
        }
        if (PyDict_SetItemString(dict, "scenario_id", s) < 0) {
            Py_DECREF(s);
            return NULL;
        }
        Py_DECREF(s);
    } else {
        if (PyDict_SetItemString(dict, "scenario_id", Py_None) < 0) {
            return NULL;
        }
    }

    if (env->dataset_name[0] != '\0') {
        PyObject *s = PyUnicode_FromString(env->dataset_name);
        if (!s) {
            return NULL;
        }
        if (PyDict_SetItemString(dict, "dataset_name", s) < 0) {
            Py_DECREF(s);
            return NULL;
        }
        Py_DECREF(s);
    } else {
        if (PyDict_SetItemString(dict, "dataset_name", Py_None) < 0) {
            return NULL;
        }
    }

    v = PyLong_FromLong(env->log_length);
    if (!v) {
        return NULL;
    }
    if (PyDict_SetItemString(dict, "length", v) < 0) {
        Py_DECREF(v);
        return NULL;
    }
    Py_DECREF(v);

    v = PyLong_FromLong(env->dynamics_model);
    if (!v) {
        return NULL;
    }
    if (PyDict_SetItemString(dict, "dynamics_model", v) < 0) {
        Py_DECREF(v);
        return NULL;
    }
    Py_DECREF(v);

    /* objects_of_interest array */
    if (env->objects_of_interest && env->num_objects_of_interest > 0) {
        PyObject *lst = PyList_New(env->num_objects_of_interest);
        if (!lst) {
            return NULL;
        }
        for (int i = 0; i < env->num_objects_of_interest; i++) {
            PyObject *it = PyLong_FromLong(env->objects_of_interest[i]);
            if (!it) {
                Py_DECREF(lst);
                return NULL;
            }
            PyList_SetItem(lst, i, it);
        }
        if (PyDict_SetItemString(dict, "objects_of_interest", lst) < 0) {
            Py_DECREF(lst);
            return NULL;
        }
        Py_DECREF(lst);
    } else {
        if (PyDict_SetItemString(dict, "objects_of_interest", Py_None) < 0) {
            return NULL;
        }
    }

    /* tracks_to_predict array */
    if (env->tracks_to_predict && env->num_tracks_to_predict > 0) {
        PyObject *lst = PyList_New(env->num_tracks_to_predict);
        if (!lst) {
            return NULL;
        }
        for (int i = 0; i < env->num_tracks_to_predict; i++) {
            PyObject *it = PyLong_FromLong(env->tracks_to_predict[i]);
            if (!it) {
                Py_DECREF(lst);
                return NULL;
            }
            PyList_SetItem(lst, i, it);
        }
        if (PyDict_SetItemString(dict, "tracks_to_predict", lst) < 0) {
            Py_DECREF(lst);
            return NULL;
        }
        Py_DECREF(lst);
    } else {
        if (PyDict_SetItemString(dict, "tracks_to_predict", Py_None) < 0) {
            return NULL;
        }
    }

    /* Lists (active agent indices) */
    if (env->active_agent_indices && env->active_agent_count > 0) {
        PyObject *lst = PyList_New(env->active_agent_count);
        if (!lst) {
            return NULL;
        }
        for (int i = 0; i < env->active_agent_count; i++) {
            PyObject *it = PyLong_FromLong(env->active_agent_indices[i]);
            if (!it) {
                Py_DECREF(lst);
                return NULL;
            }
            /* PyList_SetItem steals reference */
            PyList_SetItem(lst, i, it);
        }
        if (PyDict_SetItemString(dict, "active_agent_indices", lst) < 0) {
            Py_DECREF(lst);
            return NULL;
        }
        Py_DECREF(lst);
    } else {
        if (PyDict_SetItemString(dict, "active_agent_indices", Py_None) < 0) {
            return NULL;
        }
    }

    /* Optionally expose static car indices if present */
    if (env->static_agent_indices && env->static_agent_count > 0) {
        PyObject *lst = PyList_New(env->static_agent_count);
        if (!lst) {
            return NULL;
        }
        for (int i = 0; i < env->static_agent_count; i++) {
            PyObject *it = PyLong_FromLong(env->static_agent_indices[i]);
            if (!it) {
                Py_DECREF(lst);
                return NULL;
            }
            PyList_SetItem(lst, i, it);
        }
        if (PyDict_SetItemString(dict, "static_agent_indices", lst) < 0) {
            Py_DECREF(lst);
            return NULL;
        }
        Py_DECREF(lst);
    } else {
        if (PyDict_SetItemString(dict, "static_agent_indices", Py_None) < 0) {
            return NULL;
        }
    }

    /* Expose agents array as a list of dicts */
    if (env->agents && env->num_total_agents > 0) {
        PyObject *agents_list = PyList_New(env->num_total_agents);
        if (!agents_list) {
            return NULL;
        }
        int next_active_log_idx = 0;
        for (int i = 0; i < env->num_total_agents; i++) {
            Agent *a = &env->agents[i];
            int active_log_idx = -1;
            if (env->active_agent_indices && next_active_log_idx < env->active_agent_count
                && env->active_agent_indices[next_active_log_idx] == i) {
                active_log_idx = next_active_log_idx;
                next_active_log_idx++;
            }

            PyObject *agent = PyDict_New();
            if (!agent) {
                Py_DECREF(agents_list);
                return NULL;
            }

            /* ID and type */
            PyObject *tmp = PyLong_FromLong(i);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            PyDict_SetItemString(agent, "id", tmp);
            Py_DECREF(tmp);

            tmp = PyLong_FromLong(a->type);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            PyDict_SetItemString(agent, "type", tmp);
            Py_DECREF(tmp);

            /* Validate trajectory_size before using it */
            int traj_len = (a->trajectory_size > 0 && a->trajectory_size < 10000) ? a->trajectory_size : 0;
            tmp = PyLong_FromLong(traj_len);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            PyDict_SetItemString(agent, "trajectory_size", tmp);
            Py_DECREF(tmp);

            /* Log trajectory arrays - validate each pointer individually before access */
            if (a->log_trajectory_x && traj_len > 0) {
                PyObject *lx = PyList_New(traj_len);
                if (!lx) {
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
                for (int j = 0; j < traj_len; j++) {
                    PyObject *fv = PyFloat_FromDouble((double) a->log_trajectory_x[j]);
                    if (!fv) {
                        Py_DECREF(lx);
                        Py_DECREF(agent);
                        Py_DECREF(agents_list);
                        return NULL;
                    }
                    PyList_SetItem(lx, j, fv);
                }
                PyDict_SetItemString(agent, "log_trajectory_x", lx);
                Py_DECREF(lx);
            } else {
                PyDict_SetItemString(agent, "log_trajectory_x", Py_None);
            }
            if (a->log_trajectory_y && traj_len > 0) {
                PyObject *ly = PyList_New(traj_len);
                if (!ly) {
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
                for (int j = 0; j < traj_len; j++) {
                    PyObject *fv = PyFloat_FromDouble((double) a->log_trajectory_y[j]);
                    if (!fv) {
                        Py_DECREF(ly);
                        Py_DECREF(agent);
                        Py_DECREF(agents_list);
                        return NULL;
                    }
                    PyList_SetItem(ly, j, fv);
                }
                PyDict_SetItemString(agent, "log_trajectory_y", ly);
                Py_DECREF(ly);
            } else {
                PyDict_SetItemString(agent, "log_trajectory_y", Py_None);
            }
            if (a->log_trajectory_z && traj_len > 0) {
                PyObject *lz = PyList_New(traj_len);
                if (!lz) {
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
                for (int j = 0; j < traj_len; j++) {
                    PyObject *fv = PyFloat_FromDouble((double) a->log_trajectory_z[j]);
                    if (!fv) {
                        Py_DECREF(lz);
                        Py_DECREF(agent);
                        Py_DECREF(agents_list);
                        return NULL;
                    }
                    PyList_SetItem(lz, j, fv);
                }
                PyDict_SetItemString(agent, "log_trajectory_z", lz);
                Py_DECREF(lz);
            } else {
                PyDict_SetItemString(agent, "log_trajectory_z", Py_None);
            }
            if (a->log_heading && traj_len > 0) {
                PyObject *lh = PyList_New(traj_len);
                if (!lh) {
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
                for (int j = 0; j < traj_len; j++) {
                    PyObject *fv = PyFloat_FromDouble((double) a->log_heading[j]);
                    if (!fv) {
                        Py_DECREF(lh);
                        Py_DECREF(agent);
                        Py_DECREF(agents_list);
                        return NULL;
                    }
                    PyList_SetItem(lh, j, fv);
                }
                PyDict_SetItemString(agent, "log_heading", lh);
                Py_DECREF(lh);
            } else {
                PyDict_SetItemString(agent, "log_heading", Py_None);
            }
            if (a->log_velocity_x && traj_len > 0) {
                PyObject *lvx = PyList_New(traj_len);
                if (!lvx) {
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
                for (int j = 0; j < traj_len; j++) {
                    PyObject *fv = PyFloat_FromDouble((double) a->log_velocity_x[j]);
                    if (!fv) {
                        Py_DECREF(lvx);
                        Py_DECREF(agent);
                        Py_DECREF(agents_list);
                        return NULL;
                    }
                    PyList_SetItem(lvx, j, fv);
                }
                PyDict_SetItemString(agent, "log_velocity_x", lvx);
                Py_DECREF(lvx);
            } else {
                PyDict_SetItemString(agent, "log_velocity_x", Py_None);
            }
            if (a->log_velocity_y && traj_len > 0) {
                PyObject *lvy = PyList_New(traj_len);
                if (!lvy) {
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
                for (int j = 0; j < traj_len; j++) {
                    PyObject *fv = PyFloat_FromDouble((double) a->log_velocity_y[j]);
                    if (!fv) {
                        Py_DECREF(lvy);
                        Py_DECREF(agent);
                        Py_DECREF(agents_list);
                        return NULL;
                    }
                    PyList_SetItem(lvy, j, fv);
                }
                PyDict_SetItemString(agent, "log_velocity_y", lvy);
                Py_DECREF(lvy);
            } else {
                PyDict_SetItemString(agent, "log_velocity_y", Py_None);
            }
            if (a->log_valid && traj_len > 0) {
                PyObject *lv = PyList_New(traj_len);
                if (!lv) {
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
                for (int j = 0; j < traj_len; j++) {
                    PyObject *iv = PyLong_FromLong(a->log_valid[j]);
                    if (!iv) {
                        Py_DECREF(lv);
                        Py_DECREF(agent);
                        Py_DECREF(agents_list);
                        return NULL;
                    }
                    PyList_SetItem(lv, j, iv);
                }
                PyDict_SetItemString(agent, "log_valid", lv);
                Py_DECREF(lv);
            } else {
                PyDict_SetItemString(agent, "log_valid", Py_None);
            }

            /* Simulation state (current) */
            PyObject *pf = PyFloat_FromDouble((double) a->sim_x);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "sim_x", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->sim_y);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "sim_y", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->sim_z);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "sim_z", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->sim_heading);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "sim_heading", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->sim_vx);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "sim_vx", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->sim_vy);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "sim_vy", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->sim_speed);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "sim_speed", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->steering_angle);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "sim_steering", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->accel_long);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "accel_long", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->accel_lat);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "accel_lat", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->jerk_long);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "jerk_long", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->jerk_lat);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "jerk_lat", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->sim_length);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "sim_length", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->sim_width);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "sim_width", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->sim_height);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "sim_height", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->wheelbase);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "wheelbase", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            tmp = PyLong_FromLong(a->sim_valid);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "sim_valid", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(tmp);

            /* Goal position */
            pf = PyFloat_FromDouble((double) a->current_goal_x);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "current_goal_x", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->current_goal_y);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "current_goal_y", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) a->current_goal_z);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "current_goal_z", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            /* Status flags */
            tmp = PyLong_FromLong(a->stopped);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "stopped", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(tmp);

            tmp = PyLong_FromLong(a->current_lane_idx);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "current_lane_idx", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(tmp);

            tmp = PyLong_FromLong(a->reached_goal_this_episode);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "reached_goal_this_episode", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(tmp);

            tmp = PyLong_FromLong(a->current_route_idx);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "current_route_idx", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(tmp);

            pf = PyFloat_FromDouble((double) a->cumulative_displacement);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "cumulative_displacement", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            tmp = PyLong_FromLong(a->displacement_sample_count);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "displacement_sample_count", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(tmp);

            /* Metrics array: [collision, offroad, reached_goal, lane_aligned, avg_displacement_error,
             * red_light_violation] */
            PyObject *metrics = PyList_New(NUM_METRICS);
            if (!metrics) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            for (int j = 0; j < NUM_METRICS; j++) {
                PyObject *metric_val = PyFloat_FromDouble((double) a->metrics_array[j]);
                if (!metric_val) {
                    Py_DECREF(metrics);
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
                PyList_SetItem(metrics, j, metric_val);
            }
            if (PyDict_SetItemString(agent, "metrics_array", metrics) < 0) {
                Py_DECREF(metrics);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(metrics);

            if (env->compute_eval_metrics && env->logs && active_log_idx >= 0 && active_log_idx < env->logs_capacity) {
                Log *log = &env->logs[active_log_idx];

                pf = PyFloat_FromDouble((double) log->puffer_score);
                if (!pf) {
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
                if (PyDict_SetItemString(agent, "puffer_score", pf) < 0) {
                    Py_DECREF(pf);
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
                Py_DECREF(pf);

                PyObject *puffer_metrics = PyDict_New();
                if (!puffer_metrics) {
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
                if (assign_to_dict(puffer_metrics, "score", log->puffer_score)
                    || assign_to_dict(puffer_metrics, "no_at_fault", log->no_at_fault)
                    || assign_to_dict(puffer_metrics, "no_offroad", log->no_offroad)
                    || assign_to_dict(puffer_metrics, "no_red_light", log->no_red_light)
                    || assign_to_dict(puffer_metrics, "making_progress", log->making_progress)
                    || assign_to_dict(puffer_metrics, "direction_score", log->driving_direction_score)
                    || assign_to_dict(puffer_metrics, "ttc_puffer_rate", log->ttc_puffer_rate)
                    || assign_to_dict(puffer_metrics, "progress_ratio", log->progress_ratio)
                    || assign_to_dict(puffer_metrics, "speed_limit_compliance", log->speed_limit_compliance)
                    || assign_to_dict(puffer_metrics, "comfort_score", log->comfort_score)
                    || assign_to_dict(puffer_metrics, "multi_lane_score", log->multi_lane_score)
                    || assign_to_dict(puffer_metrics, "wrong_way_distance", log->wrong_way_distance)
                    || assign_to_dict(puffer_metrics, "speed_violation_sum", log->speed_violation_sum)
                    || assign_to_dict(puffer_metrics, "multiplier", log->multiplier)
                    || assign_to_dict(puffer_metrics, "weighted_average", log->weighted_average)) {
                    Py_DECREF(puffer_metrics);
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
                if (PyDict_SetItemString(agent, "puffer_metrics", puffer_metrics) < 0) {
                    Py_DECREF(puffer_metrics);
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
                Py_DECREF(puffer_metrics);
            }

            /* Export route information */
            tmp = PyLong_FromLong(a->route_length);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "route_length", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(tmp);

            if (a->route && a->route_length > 0) {
                PyObject *route_list = PyList_New(a->route_length);
                if (!route_list) {
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }

                for (int j = 0; j < a->route_length; j++) {
                    PyObject *lane_id = PyLong_FromLong(a->route[j]);
                    if (!lane_id) {
                        Py_DECREF(route_list);
                        Py_DECREF(agent);
                        Py_DECREF(agents_list);
                        return NULL;
                    }
                    PyList_SetItem(route_list, j, lane_id);
                }

                if (PyDict_SetItemString(agent, "route", route_list) < 0) {
                    Py_DECREF(route_list);
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
                Py_DECREF(route_list);
            } else if (PyDict_SetItemString(agent, "route", Py_None) < 0) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }

            PyList_SetItem(agents_list, i, agent);
        }

        if (PyDict_SetItemString(dict, "agents", agents_list) < 0) {
            Py_DECREF(agents_list);
            return NULL;
        }
        Py_DECREF(agents_list);
    } else {
        if (PyDict_SetItemString(dict, "agents", Py_None) < 0) {
            return NULL;
        }
    }

    /* Expose road_elements array as a list of dicts */
    if (env->road_elements && env->num_road_elements > 0) {
        PyObject *road_list = PyList_New(env->num_road_elements);
        if (!road_list) {
            return NULL;
        }
        for (int i = 0; i < env->num_road_elements; i++) {
            RoadMapElement *r = &env->road_elements[i];
            PyObject *road = PyDict_New();
            if (!road) {
                Py_DECREF(road_list);
                return NULL;
            }

            PyObject *tmp = PyLong_FromLong(i);
            if (!tmp) {
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            if (PyDict_SetItemString(road, "id", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            Py_DECREF(tmp);

            tmp = PyLong_FromLong(r->type);
            if (!tmp) {
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            if (PyDict_SetItemString(road, "type", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            Py_DECREF(tmp);

            /* Validate segment_size before using it */
            int seg_len = (r->segment_size > 0 && r->segment_size < 100000) ? r->segment_size : 0;
            tmp = PyLong_FromLong(seg_len);
            if (!tmp) {
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            if (PyDict_SetItemString(road, "segment_size", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            Py_DECREF(tmp);

            /* Geometry arrays - validate each pointer individually before access */
            if (r->x && seg_len > 0) {
                PyObject *lx = PyList_New(seg_len);
                if (!lx) {
                    Py_DECREF(road);
                    Py_DECREF(road_list);
                    return NULL;
                }
                for (int j = 0; j < seg_len; j++) {
                    PyObject *fv = PyFloat_FromDouble((double) r->x[j]);
                    if (!fv) {
                        Py_DECREF(lx);
                        Py_DECREF(road);
                        Py_DECREF(road_list);
                        return NULL;
                    }
                    PyList_SetItem(lx, j, fv);
                }
                if (PyDict_SetItemString(road, "x", lx) < 0) {
                    Py_DECREF(lx);
                    Py_DECREF(road);
                    Py_DECREF(road_list);
                    return NULL;
                }
                Py_DECREF(lx);
            } else if (PyDict_SetItemString(road, "x", Py_None) < 0) {
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            if (r->y && seg_len > 0) {
                PyObject *ly = PyList_New(seg_len);
                if (!ly) {
                    Py_DECREF(road);
                    Py_DECREF(road_list);
                    return NULL;
                }
                for (int j = 0; j < seg_len; j++) {
                    PyObject *fv = PyFloat_FromDouble((double) r->y[j]);
                    if (!fv) {
                        Py_DECREF(ly);
                        Py_DECREF(road);
                        Py_DECREF(road_list);
                        return NULL;
                    }
                    PyList_SetItem(ly, j, fv);
                }
                if (PyDict_SetItemString(road, "y", ly) < 0) {
                    Py_DECREF(ly);
                    Py_DECREF(road);
                    Py_DECREF(road_list);
                    return NULL;
                }
                Py_DECREF(ly);
            } else if (PyDict_SetItemString(road, "y", Py_None) < 0) {
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            if (r->z && seg_len > 0) {
                PyObject *lz = PyList_New(seg_len);
                if (!lz) {
                    Py_DECREF(road);
                    Py_DECREF(road_list);
                    return NULL;
                }
                for (int j = 0; j < seg_len; j++) {
                    PyObject *fv = PyFloat_FromDouble((double) r->z[j]);
                    if (!fv) {
                        Py_DECREF(lz);
                        Py_DECREF(road);
                        Py_DECREF(road_list);
                        return NULL;
                    }
                    PyList_SetItem(lz, j, fv);
                }
                if (PyDict_SetItemString(road, "z", lz) < 0) {
                    Py_DECREF(lz);
                    Py_DECREF(road);
                    Py_DECREF(road_list);
                    return NULL;
                }
                Py_DECREF(lz);
            } else if (PyDict_SetItemString(road, "z", Py_None) < 0) {
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }

            /* Lane-specific fields */
            if (is_road_lane(r->type) && r->entry_lanes != NULL && r->num_entries > 0) {
                tmp = PyList_New(r->num_entries);
            } else {
                tmp = PyList_New(0);
            }
            if (!tmp) {
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            if (is_road_lane(r->type) && r->entry_lanes != NULL && r->num_entries > 0) {
                for (int k = 0; k < r->num_entries; k++) {
                    PyList_SET_ITEM(tmp, k, PyLong_FromLong(r->entry_lanes[k]));
                }
            }
            if (PyDict_SetItemString(road, "entry_lanes", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            Py_DECREF(tmp);

            if (is_road_lane(r->type) && r->exit_lanes != NULL && r->num_exits > 0) {
                tmp = PyList_New(r->num_exits);
            } else {
                tmp = PyList_New(0);
            }
            if (!tmp) {
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            if (is_road_lane(r->type) && r->exit_lanes != NULL && r->num_exits > 0) {
                for (int k = 0; k < r->num_exits; k++) {
                    PyList_SET_ITEM(tmp, k, PyLong_FromLong(r->exit_lanes[k]));
                }
            }
            if (PyDict_SetItemString(road, "exit_lanes", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            Py_DECREF(tmp);

            PyObject *pf = PyFloat_FromDouble((double) r->speed_limit);
            if (!pf) {
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            if (PyDict_SetItemString(road, "speed_limit", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) r->length);
            if (!pf) {
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            if (PyDict_SetItemString(road, "length", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            Py_DECREF(pf);

            if (is_road_lane(r->type) && r->cum_lengths != NULL && seg_len > 0) {
                tmp = PyList_New(seg_len);
                if (!tmp) {
                    Py_DECREF(road);
                    Py_DECREF(road_list);
                    return NULL;
                }
                for (int j = 0; j < seg_len; j++) {
                    PyObject *fv = PyFloat_FromDouble((double) r->cum_lengths[j]);
                    if (!fv) {
                        Py_DECREF(tmp);
                        Py_DECREF(road);
                        Py_DECREF(road_list);
                        return NULL;
                    }
                    PyList_SetItem(tmp, j, fv);
                }
            } else {
                tmp = PyList_New(0);
                if (!tmp) {
                    Py_DECREF(road);
                    Py_DECREF(road_list);
                    return NULL;
                }
            }
            if (PyDict_SetItemString(road, "cum_lengths", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            Py_DECREF(tmp);

            PyList_SetItem(road_list, i, road);
        }
        if (PyDict_SetItemString(dict, "road_elements", road_list) < 0) {
            Py_DECREF(road_list);
            return NULL;
        }
        Py_DECREF(road_list);
    } else {
        if (PyDict_SetItemString(dict, "road_elements", Py_None) < 0) {
            return NULL;
        }
    }

    /* Expose traffic_elements array as a list of dicts */
    if (env->traffic_elements && env->num_traffic_elements > 0) {
        PyObject *traffic_list = PyList_New(env->num_traffic_elements);
        if (!traffic_list) {
            return NULL;
        }
        for (int i = 0; i < env->num_traffic_elements; i++) {
            TrafficControlElement *t = &env->traffic_elements[i];
            PyObject *traffic = PyDict_New();
            if (!traffic) {
                Py_DECREF(traffic_list);
                return NULL;
            }

            PyObject *tmp = PyLong_FromLong(i);
            if (!tmp) {
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            if (PyDict_SetItemString(traffic, "id", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            Py_DECREF(tmp);

            tmp = PyLong_FromLong(t->type);
            if (!tmp) {
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            if (PyDict_SetItemString(traffic, "type", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            Py_DECREF(tmp);

            /* Validate state_size before using it */
            int state_len = (t->state_size > 0 && t->state_size < 10000) ? t->state_size : 0;
            tmp = PyLong_FromLong(state_len);
            if (!tmp) {
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            if (PyDict_SetItemString(traffic, "state_size", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            Py_DECREF(tmp);

            /* States array - validate pointer before access */
            if (t->states && state_len > 0) {
                PyObject *ls = PyList_New(state_len);
                if (!ls) {
                    Py_DECREF(traffic);
                    Py_DECREF(traffic_list);
                    return NULL;
                }
                for (int j = 0; j < state_len; j++) {
                    PyObject *iv = PyLong_FromLong(t->states[j]);
                    if (!iv) {
                        Py_DECREF(ls);
                        Py_DECREF(traffic);
                        Py_DECREF(traffic_list);
                        return NULL;
                    }
                    PyList_SetItem(ls, j, iv);
                }
                if (PyDict_SetItemString(traffic, "states", ls) < 0) {
                    Py_DECREF(ls);
                    Py_DECREF(traffic);
                    Py_DECREF(traffic_list);
                    return NULL;
                }
                Py_DECREF(ls);
            } else if (PyDict_SetItemString(traffic, "states", Py_None) < 0) {
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }

            /* Stop line endpoints */
            PyObject *sl = PyList_New(6);
            if (!sl) {
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            for (int k = 0; k < 6; k++) {
                PyList_SetItem(sl, k, PyFloat_FromDouble((double) t->stop_line[k]));
            }
            if (PyDict_SetItemString(traffic, "stop_line", sl) < 0) {
                Py_DECREF(sl);
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            Py_DECREF(sl);

            /* Position (stop_line midpoint for backward compat) */
            float mid_x = (t->stop_line[0] + t->stop_line[3]) * 0.5f;
            float mid_y = (t->stop_line[1] + t->stop_line[4]) * 0.5f;
            PyObject *pf = PyFloat_FromDouble((double) mid_x);
            if (!pf) {
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            if (PyDict_SetItemString(traffic, "x", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double) mid_y);
            if (!pf) {
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            if (PyDict_SetItemString(traffic, "y", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            Py_DECREF(pf);

            /* Controlled lanes array - validate pointer before access */
            if (t->controlled_lanes && t->num_controlled_lanes > 0) {
                PyObject *ll = PyList_New(t->num_controlled_lanes);
                if (!ll) {
                    Py_DECREF(traffic);
                    Py_DECREF(traffic_list);
                    return NULL;
                }
                for (int j = 0; j < t->num_controlled_lanes; j++) {
                    PyObject *lane_id = PyLong_FromLong(t->controlled_lanes[j]);
                    if (!lane_id) {
                        Py_DECREF(ll);
                        Py_DECREF(traffic);
                        Py_DECREF(traffic_list);
                        return NULL;
                    }
                    PyList_SetItem(ll, j, lane_id);
                }
                if (PyDict_SetItemString(traffic, "controlled_lanes", ll) < 0) {
                    Py_DECREF(ll);
                    Py_DECREF(traffic);
                    Py_DECREF(traffic_list);
                    return NULL;
                }
                Py_DECREF(ll);
            } else if (PyDict_SetItemString(traffic, "controlled_lanes", Py_None) < 0) {
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }

            PyList_SetItem(traffic_list, i, traffic);
        }
        if (PyDict_SetItemString(dict, "traffic_elements", traffic_list) < 0) {
            Py_DECREF(traffic_list);
            return NULL;
        }
        Py_DECREF(traffic_list);
    } else {
        if (PyDict_SetItemString(dict, "traffic_elements", Py_None) < 0) {
            return NULL;
        }
    }

    /* Map corners (bounding box) from GridMap */
    if (env->grid_map) {
        PyObject *corners_list = PyList_New(4);
        if (!corners_list) {
            return NULL;
        }

        PyObject *brx = PyFloat_FromDouble((double) env->grid_map->bottom_right_x);
        PyObject *bry = PyFloat_FromDouble((double) env->grid_map->bottom_right_y);
        PyObject *tlx = PyFloat_FromDouble((double) env->grid_map->top_left_x);
        PyObject *tly = PyFloat_FromDouble((double) env->grid_map->top_left_y);

        if (!tlx || !tly || !brx || !bry) {
            Py_XDECREF(tlx);
            Py_XDECREF(tly);
            Py_XDECREF(brx);
            Py_XDECREF(bry);
            Py_DECREF(corners_list);
            return NULL;
        }

        PyList_SetItem(corners_list, 0, tlx); // min_x = top_left_x
        PyList_SetItem(corners_list, 1, bry); // min_y = bottom_right_y
        PyList_SetItem(corners_list, 2, brx); // max_x = bottom_right_x
        PyList_SetItem(corners_list, 3, tly); // max_y = top_left_y

        if (PyDict_SetItemString(dict, "map_corners", corners_list) < 0) {
            Py_DECREF(corners_list);
            return NULL;
        }
        Py_DECREF(corners_list);
    } else {
        if (PyDict_SetItemString(dict, "map_corners", Py_None) < 0) {
            return NULL;
        }
    }
    if (env->observations && env->active_agent_count > 0) {
        /* Agent observations */
        int max_obs = compute_observation_size(env);
        PyObject *obs_data = PyList_New(env->active_agent_count);
        if (!obs_data) {
            return NULL;
        }

        float (*observations)[max_obs] = (float (*)[max_obs]) env->observations;

        for (int i = 0; i < env->active_agent_count; i++) {
            PyObject *agent_obs = PyList_New(max_obs);
            if (!agent_obs) {
                Py_DECREF(obs_data);
                return NULL;
            }

            for (int j = 0; j < max_obs; j++) {
                PyObject *obs_val = PyFloat_FromDouble((double) observations[i][j]);
                if (!obs_val) {
                    Py_DECREF(agent_obs);
                    Py_DECREF(obs_data);
                    return NULL;
                }
                PyList_SetItem(agent_obs, j, obs_val);
            }

            PyList_SetItem(obs_data, i, agent_obs);
        }

        if (PyDict_SetItemString(dict, "agent_observations", obs_data) < 0) {
            Py_DECREF(obs_data);
            return NULL;
        }
        Py_DECREF(obs_data);
    } else {
        if (PyDict_SetItemString(dict, "agent_observations", Py_None) < 0) {
            return NULL;
        }
    }

    return dict;
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
    int s_map_counter = starting_map_counter;
    int init_mode = unpack(kwargs, "init_mode");
    int control_mode = unpack(kwargs, "control_mode");
    int simulation_mode = unpack(kwargs, "simulation_mode");
    int init_step = unpack(kwargs, "init_step");
    int seed = unpack(kwargs, "seed");
    int min_agents_per_env = unpack(kwargs, "min_agents_per_env");
    int max_agents_per_env = unpack(kwargs, "max_agents_per_env");
    int num_eval_scenarios = unpack(kwargs, "num_eval_scenarios");
    if (min_agents_per_env <= 0 || max_agents_per_env <= 0) {
        PyErr_SetString(PyExc_ValueError, "min_agents_per_env and max_agents_per_env must be > 0");
        return NULL;
    }
    if (min_agents_per_env > max_agents_per_env) {
        PyErr_SetString(PyExc_ValueError, "min_agents_per_env must be <= max_agents_per_env");
        return NULL;
    }
    if (num_agents < min_agents_per_env) {
        PyErr_SetString(PyExc_ValueError, "num_agents must be >= min_agents_per_env");
        return NULL;
    }

    srand(seed);

    // GIGAFLOW mode: use random sampling for agent counts per env
    if (simulation_mode == SIMULATION_GIGAFLOW) {
        if (eval_mode) {
            // Eval mode: fixed agent count, sequential map cycling
            int agents_per_env = max_agents_per_env;
            int env_count = (num_agents + agents_per_env - 1) / agents_per_env;
            env_count--;

            env_count = env_count > num_eval_scenarios ? num_eval_scenarios : env_count;

            PyObject *agent_offsets = PyList_New(env_count + 1);
            PyObject *map_ids_list = PyList_New(env_count);

            int offset = 0;
            for (int i = 0; i < env_count; i++) {
                PyList_SetItem(agent_offsets, i, PyLong_FromLong(offset));
                PyList_SetItem(map_ids_list, i, PyLong_FromLong((s_map_counter + i) % num_maps));
                offset += agents_per_env;
            }
            PyList_SetItem(agent_offsets, env_count, PyLong_FromLong(offset));

            PyObject *tuple = PyTuple_New(3);
            PyTuple_SetItem(tuple, 0, agent_offsets);
            PyTuple_SetItem(tuple, 1, map_ids_list);
            PyTuple_SetItem(tuple, 2, PyLong_FromLong(env_count));
            return tuple;
        }

        // Training mode: random agent counts per env
        int *agent_counts = malloc((num_agents / min_agents_per_env + 1) * sizeof(int));
        int remaining = num_agents;
        int env_count = 0;

        while (remaining > 0) {
            int count;
            if (remaining <= max_agents_per_env) {
                count = remaining;
            } else {
                // 1. We must leave at least min_agents_per_env for the future.
                int absolute_max_allowed = remaining - min_agents_per_env;

                // 2. We cannot take more than max_agents_per_env right now.
                int current_upper_bound
                    = (absolute_max_allowed < max_agents_per_env) ? absolute_max_allowed : max_agents_per_env;

                // 3. We must take at least min_agents_per_env right now.
                int current_lower_bound = min_agents_per_env;

                // Safety check: if constraints are tight, lower might equal upper.
                // If absolute_max_allowed < min_lower_bound for example leading to
                // current_upper_bound < current_lower_bound
                if (current_upper_bound <= current_lower_bound) {
                    count = current_lower_bound;
                } else {
                    // Now the range is guaranteed to be positive.
                    int range = current_upper_bound - current_lower_bound + 1;
                    count = current_lower_bound + (rand() % range);
                }
            }
            agent_counts[env_count++] = count;
            remaining -= count;
        }

        // Build Python return lists
        PyObject *agent_offsets = PyList_New(env_count + 1);
        PyObject *map_ids_list = PyList_New(env_count);

        int offset = 0;
        for (int i = 0; i < env_count; i++) {
            PyList_SetItem(agent_offsets, i, PyLong_FromLong(offset));
            PyList_SetItem(map_ids_list, i, PyLong_FromLong(rand() % num_maps));
            offset += agent_counts[i];
        }
        PyList_SetItem(agent_offsets, env_count, PyLong_FromLong(num_agents));

        free(agent_counts);

        PyObject *tuple = PyTuple_New(3);
        PyTuple_SetItem(tuple, 0, agent_offsets);
        PyTuple_SetItem(tuple, 1, map_ids_list);
        PyTuple_SetItem(tuple, 2, PyLong_FromLong(env_count));
        return tuple;
    }

    // REPLAY mode - existing logic with max_agents_per_env cap
    int total_agent_count = 0;
    int map_id = 0;
    int env_count = 0;
    int max_envs = num_agents;
    int maps_checked = 0;

    if (eval_mode) {
        max_envs = num_eval_scenarios;
    }
    // Define the upper boundary for the map window
    int end_map_index = starting_map_counter + num_eval_scenarios;

    PyObject *agent_offsets = PyList_New(max_envs + 1);
    PyObject *map_ids = PyList_New(max_envs);

    // Added condition: s_map_counter < end_map_index
    while (total_agent_count < num_agents && env_count < max_envs && (!eval_mode || s_map_counter < end_map_index)) {
        if (eval_mode) {
            map_id = s_map_counter % num_maps;
            s_map_counter += 1; // This increments towards end_map_index
        } else {
            map_id = rand() % num_maps;
        }

        const char *map_file = PyUnicode_AsUTF8(PyList_GetItem(map_files, map_id));

        Drive *env = calloc(1, sizeof(Drive));
        env->init_mode = init_mode;
        env->control_mode = control_mode;
        env->simulation_mode = simulation_mode;
        env->init_step = init_step;
        env->num_max_agents = max_agents_per_env;
        load_map_binary(map_file, env);
        env->logs_capacity = 0;

        set_active_agents(env);

        // Skip map (no controllable agents) or scenario overflow (eval budget)
        bool skip = (env->active_agent_count == 0);
        bool overflow = (eval_mode && total_agent_count + env->active_agent_count > num_agents);
        if (skip || overflow) {
            if (skip) {
                maps_checked++;
            }
            free_env(env);
            free(env);
            if (overflow) {
                break;
            }
            continue;
        }

        // Store map_id
        PyList_SetItem(map_ids, env_count, PyLong_FromLong(map_id));
        // Store agent offset
        PyList_SetItem(agent_offsets, env_count, PyLong_FromLong(total_agent_count));
        total_agent_count += env->active_agent_count;
        env_count++;
        free_env(env);
        free(env);
    }
    if (total_agent_count >= num_agents) {
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
    env->action_type = (int) unpack(kwargs, "action_type");
    env->dynamics_model = (int) unpack(kwargs, "dynamics_model");
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
    env->reward_ade = (float) unpack(kwargs, "reward_ade");
    env->collision_behavior = (int) unpack(kwargs, "collision_behavior");
    env->offroad_behavior = (int) unpack(kwargs, "offroad_behavior");
    env->traffic_light_behavior = (int) unpack(kwargs, "traffic_light_behavior");
    env->use_map_cache = (int) unpack(kwargs, "use_map_cache");
    env->use_neighbor_cache = (int) unpack(kwargs, "use_neighbor_cache");
    env->goal_radius = (float) unpack(kwargs, "goal_radius");
    env->min_goal_spacing = (float) unpack(kwargs, "min_goal_spacing");
    env->max_goal_spacing = (float) unpack(kwargs, "max_goal_spacing");
    env->num_goals = (int) unpack(kwargs, "num_goals");
    env->goal_regen_mode = (int) unpack(kwargs, "goal_regen_mode");
    env->goal_source = (int) unpack(kwargs, "goal_source");
    env->obs_goal_lane_distance = (int) unpack(kwargs, "obs_goal_lane_distance");
    env->obs_slots_boundary_n = (int) unpack(kwargs, "obs_slots_boundary_n");
    env->obs_slots_lane_n = (int) unpack(kwargs, "obs_slots_lane_n");
    env->obs_slots_partners_n = (int) unpack(kwargs, "obs_slots_partners_n");
    env->obs_slots_traffic_controls_n = (int) unpack(kwargs, "obs_slots_traffic_controls_n");
    env->traffic_control_scope = (int) unpack(kwargs, "traffic_control_scope");
    env->obs_lane_stride = (int) unpack(kwargs, "obs_lane_stride");
    env->obs_boundary_stride = (int) unpack(kwargs, "obs_boundary_stride");
    env->dt = (float) unpack(kwargs, "dt");
    env->spawn_initial_speed = (float) unpack(kwargs, "spawn_initial_speed");
    env->goal_speed = (float) unpack(kwargs, "goal_speed");
    env->scenario_length = (int) unpack(kwargs, "scenario_length");
    env->termination_mode = (int) unpack(kwargs, "termination_mode");
    env->inactive_agent_threshold = (float) unpack(kwargs, "inactive_agent_threshold");
    char *map_file = unpack_str(kwargs, "map_file");
    env->map_name = map_file;
    env->num_controllable_agents = (int) unpack(kwargs, "max_agents");
    env->num_max_agents = (int) unpack(kwargs, "max_agents_per_env");
    int init_step = (int) unpack(kwargs, "init_step");
    env->init_step = init_step;
    env->timestep = init_step;
    env->init_mode = (int) unpack(kwargs, "init_mode");
    env->control_mode = (int) unpack(kwargs, "control_mode");
    env->simulation_mode = (int) unpack(kwargs, "simulation_mode");
    env->replay_expert_actions = (bool) unpack(kwargs, "replay_expert_actions");
    env->reward_conditioning = (bool) unpack(kwargs, "reward_conditioning");
    env->reward_randomization = (bool) unpack(kwargs, "reward_randomization");
    env->compute_eval_metrics = (bool) unpack(kwargs, "compute_eval_metrics");
    env->eval_mode = (int) unpack(kwargs, "eval_mode");
    env->obs_norm_goal_offset_m = (float) unpack(kwargs, "obs_norm_goal_offset_m");
    env->obs_norm_xy_offset_m = (float) unpack(kwargs, "obs_norm_xy_offset_m");
    env->obs_norm_veh_length_m = (float) unpack(kwargs, "obs_norm_veh_length_m");
    env->obs_norm_veh_width_m = (float) unpack(kwargs, "obs_norm_veh_width_m");
    env->obs_norm_road_seg_length_m = (float) unpack(kwargs, "obs_norm_road_seg_length_m");
    env->obs_norm_road_seg_width_m = (float) unpack(kwargs, "obs_norm_road_seg_width_m");
    env->obs_range_traffic_control_m = (float) unpack(kwargs, "obs_range_traffic_control_m");
    env->obs_range_partner_m = (float) unpack(kwargs, "obs_range_partner_m");
    env->obs_range_road_front_m = (float) unpack(kwargs, "obs_range_road_front_m");
    env->obs_range_road_behind_m = (float) unpack(kwargs, "obs_range_road_behind_m");
    env->obs_range_road_side_m = (float) unpack(kwargs, "obs_range_road_side_m");
    env->obs_slots_lane_kept = (int) unpack(kwargs, "obs_slots_lane_kept");
    env->obs_slots_boundary_kept = (int) unpack(kwargs, "obs_slots_boundary_kept");
    env->partner_blindness_prob = (float) unpack(kwargs, "partner_blindness_prob");
    env->partner_blindness_trigger_prob = (float) unpack(kwargs, "partner_blindness_trigger_prob");
    env->phantom_braking_prob = (float) unpack(kwargs, "phantom_braking_prob");
    env->phantom_braking_trigger_prob = (float) unpack(kwargs, "phantom_braking_trigger_prob");
    env->phantom_braking_duration = (int) unpack(kwargs, "phantom_braking_duration");
    env->dynamics_noise_long_std = (float) unpack(kwargs, "dynamics_noise_long_std");
    env->dynamics_noise_lat_std = (float) unpack(kwargs, "dynamics_noise_lat_std");
    env->dynamics_noise_speed_std = (float) unpack(kwargs, "dynamics_noise_speed_std");
    env->dynamics_noise_heading_std = (float) unpack(kwargs, "dynamics_noise_heading_std");

    init(env);
    return 0;
}

static int my_log(PyObject *dict, Env *env, Log *log, float n) {
    float total_distance_travelled = log->total_distance_travelled * n;
    float total_infractions = log->total_infractions * n;

    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "red_light_violation_rate", log->red_light_violation_rate);
    assign_to_dict(dict, "comfort_violation_count", log->comfort_violation_count);
    // assign_to_dict(dict, "avg_displacement_error", log->avg_displacement_error);
    assign_to_dict(dict, "velocity_progress_sum", log->velocity_progress_sum);
    assign_to_dict(dict, "num_goals_reached", log->num_goals_reached);
    assign_to_dict(dict, "lane_center_rate", log->lane_center_rate);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "avg_speed_per_agent", log->avg_speed_per_agent);
    assign_to_dict(dict, "total_distance_travelled_sum", total_distance_travelled);
    assign_to_dict(dict, "total_infraction_count", total_infractions);

    if (env->compute_eval_metrics) {
        // Puffer score components
        assign_to_dict(dict, "at_fault_collision_rate", log->at_fault_collision_rate);
        assign_to_dict(dict, "puffer_score", log->puffer_score);
        assign_to_dict(dict, "driving_direction_score", log->driving_direction_score);
        assign_to_dict(dict, "speed_limit_compliance", log->speed_limit_compliance);
        assign_to_dict(dict, "making_progress_rate", log->making_progress_rate);
        assign_to_dict(dict, "progress_ratio", log->progress_ratio);
        assign_to_dict(dict, "comfort_score", log->comfort_score);
        assign_to_dict(dict, "multi_lane_time", log->multi_lane_time);
        assign_to_dict(dict, "multi_lane_score", log->multi_lane_score);
    }

    return 0;
}
