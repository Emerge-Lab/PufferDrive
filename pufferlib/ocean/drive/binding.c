#include "drive.h"
#define Env Drive
#define MY_SHARED
#define MY_PUT
#define MY_GET
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
    if (!v)
        return NULL;
    if (PyDict_SetItemString(dict, "active_agent_count", v) < 0) {
        Py_DECREF(v);
        return NULL;
    }
    Py_DECREF(v);

    v = PyLong_FromLong(env->num_total_agents);
    if (!v)
        return NULL;
    if (PyDict_SetItemString(dict, "num_total_agents", v) < 0) {
        Py_DECREF(v);
        return NULL;
    }
    Py_DECREF(v);

    v = PyLong_FromLong(env->num_road_elements);
    if (!v)
        return NULL;
    if (PyDict_SetItemString(dict, "num_road_elements", v) < 0) {
        Py_DECREF(v);
        return NULL;
    }
    Py_DECREF(v);

    v = PyLong_FromLong(env->num_traffic_elements);
    if (!v)
        return NULL;
    if (PyDict_SetItemString(dict, "num_traffic_elements", v) < 0) {
        Py_DECREF(v);
        return NULL;
    }
    Py_DECREF(v);

    /* Map name / string fields */
    if (env->map_name) {
        PyObject *s = PyUnicode_FromString(env->map_name);
        if (!s)
            return NULL;
        if (PyDict_SetItemString(dict, "map_name", s) < 0) {
            Py_DECREF(s);
            return NULL;
        }
        Py_DECREF(s);
    } else {
        if (PyDict_SetItemString(dict, "map_name", Py_None) < 0)
            return NULL;
    }

    /* Metadata fields */
    if (env->scenario_id[0] != '\0') {
        PyObject *s = PyUnicode_FromString(env->scenario_id);
        if (!s)
            return NULL;
        if (PyDict_SetItemString(dict, "scenario_id", s) < 0) {
            Py_DECREF(s);
            return NULL;
        }
        Py_DECREF(s);
    } else {
        if (PyDict_SetItemString(dict, "scenario_id", Py_None) < 0)
            return NULL;
    }

    if (env->dataset_name[0] != '\0') {
        PyObject *s = PyUnicode_FromString(env->dataset_name);
        if (!s)
            return NULL;
        if (PyDict_SetItemString(dict, "dataset_name", s) < 0) {
            Py_DECREF(s);
            return NULL;
        }
        Py_DECREF(s);
    } else {
        if (PyDict_SetItemString(dict, "dataset_name", Py_None) < 0)
            return NULL;
    }

    v = PyLong_FromLong(env->log_length);
    if (!v)
        return NULL;
    if (PyDict_SetItemString(dict, "length", v) < 0) {
        Py_DECREF(v);
        return NULL;
    }
    Py_DECREF(v);

    v = PyLong_FromLong(env->sdc_index);
    if (!v)
        return NULL;
    if (PyDict_SetItemString(dict, "ego_id", v) < 0) {
        Py_DECREF(v);
        return NULL;
    }
    Py_DECREF(v);

    v = PyLong_FromLong(env->dynamics_model);
    if (!v)
        return NULL;
    if (PyDict_SetItemString(dict, "dynamics_model", v) < 0) {
        Py_DECREF(v);
        return NULL;
    }
    Py_DECREF(v);

    /* objects_of_interest array */
    if (env->objects_of_interest && env->num_objects_of_interest > 0) {
        PyObject *lst = PyList_New(env->num_objects_of_interest);
        if (!lst)
            return NULL;
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
        if (PyDict_SetItemString(dict, "objects_of_interest", Py_None) < 0)
            return NULL;
    }

    /* tracks_to_predict array */
    if (env->tracks_to_predict && env->num_tracks_to_predict > 0) {
        PyObject *lst = PyList_New(env->num_tracks_to_predict);
        if (!lst)
            return NULL;
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
        if (PyDict_SetItemString(dict, "tracks_to_predict", Py_None) < 0)
            return NULL;
    }

    /* Lists (active agent indices) */
    if (env->active_agent_indices && env->active_agent_count > 0) {
        PyObject *lst = PyList_New(env->active_agent_count);
        if (!lst)
            return NULL;
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
        if (PyDict_SetItemString(dict, "active_agent_indices", Py_None) < 0)
            return NULL;
    }

    /* Optionally expose static car indices if present */
    if (env->static_agent_indices && env->static_agent_count > 0) {
        PyObject *lst = PyList_New(env->static_agent_count);
        if (!lst)
            return NULL;
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
        if (PyDict_SetItemString(dict, "static_agent_indices", Py_None) < 0)
            return NULL;
    }

    /* Expose agents array as a list of dicts */
    if (env->agents && env->num_total_agents > 0) {
        PyObject *agents_list = PyList_New(env->num_total_agents);
        if (!agents_list)
            return NULL;
        for (int i = 0; i < env->num_total_agents; i++) {
            Agent *a = &env->agents[i];

            PyObject *agent = PyDict_New();
            if (!agent) {
                Py_DECREF(agents_list);
                return NULL;
            }

            /* ID and type */
            PyObject *tmp = PyLong_FromLong(a->id);
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

            /* Validate trajectory_length before using it */
            int traj_len = (a->trajectory_length > 0 && a->trajectory_length < 10000) ? a->trajectory_length : 0;
            tmp = PyLong_FromLong(traj_len);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            PyDict_SetItemString(agent, "trajectory_length", tmp);
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
                    PyObject *fv = PyFloat_FromDouble((double)a->log_trajectory_x[j]);
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
                    PyObject *fv = PyFloat_FromDouble((double)a->log_trajectory_y[j]);
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
                    PyObject *fv = PyFloat_FromDouble((double)a->log_trajectory_z[j]);
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
                    PyObject *fv = PyFloat_FromDouble((double)a->log_heading[j]);
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
                    PyObject *fv = PyFloat_FromDouble((double)a->log_velocity_x[j]);
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
                    PyObject *fv = PyFloat_FromDouble((double)a->log_velocity_y[j]);
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
            PyObject *pf = PyFloat_FromDouble((double)a->sim_x);
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

            pf = PyFloat_FromDouble((double)a->sim_y);
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

            pf = PyFloat_FromDouble((double)a->sim_z);
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

            pf = PyFloat_FromDouble((double)a->sim_heading);
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

            pf = PyFloat_FromDouble((double)a->sim_vx);
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

            pf = PyFloat_FromDouble((double)a->sim_vy);
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

            pf = PyFloat_FromDouble((double)a->sim_speed);
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

            pf = PyFloat_FromDouble((double)a->sim_length);
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

            pf = PyFloat_FromDouble((double)a->sim_width);
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

            pf = PyFloat_FromDouble((double)a->sim_height);
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

            tmp = PyLong_FromLong(a->closest_path_idx_wp);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "closest_path_idx_wp", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(tmp);

            /* Goal position */
            pf = PyFloat_FromDouble((double)a->goal_position_x);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "goal_position_x", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double)a->goal_position_y);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "goal_position_y", pf) < 0) {
                Py_DECREF(pf);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(pf);

            pf = PyFloat_FromDouble((double)a->goal_position_z);
            if (!pf) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "goal_position_z", pf) < 0) {
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

            tmp = PyLong_FromLong(a->current_lane_index);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "current_lane_index", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(tmp);

            tmp = PyLong_FromLong(a->active_agent);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "active_agent", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(tmp);

            tmp = PyLong_FromLong(a->mark_as_expert);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "mark_as_expert", tmp) < 0) {
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

            /* Debug metrics */
            tmp = PyLong_FromLong(a->num_goals_reached);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "num_goals_reached", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(tmp);

            tmp = PyLong_FromLong(a->respawn_count);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "respawn_count", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(tmp);

            tmp = PyLong_FromLong(a->current_route_index);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "current_route_index", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            Py_DECREF(tmp);

            pf = PyFloat_FromDouble((double)a->cumulative_displacement);
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
            PyObject *metrics = PyList_New(8);
            if (!metrics) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            for (int j = 0; j < 8; j++) {
                PyObject *metric_val = PyFloat_FromDouble((double)a->metrics_array[j]);
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
            } else {
                if (PyDict_SetItemString(agent, "route", Py_None) < 0) {
                    Py_DECREF(agent);
                    Py_DECREF(agents_list);
                    return NULL;
                }
            }

            PyList_SetItem(agents_list, i, agent);
        }

        if (PyDict_SetItemString(dict, "agents", agents_list) < 0) {
            Py_DECREF(agents_list);
            return NULL;
        }
        Py_DECREF(agents_list);
    } else {
        if (PyDict_SetItemString(dict, "agents", Py_None) < 0)
            return NULL;
    }

    /* SDC Paths */
    if (env->agents && env->active_agent_count > 0) {
        PyObject *sdc_list = PyList_New(env->active_agent_count);
        if (!sdc_list)
            return NULL;

        for (int i = 0; i < env->active_agent_count; i++) {
            Agent *a = &env->agents[env->active_agent_indices[i]];
            if (a->path) {
                struct Path *path = a->path;
                PyObject *path_dict = PyDict_New();
                if (!path_dict) {
                    Py_DECREF(sdc_list);
                    return NULL;
                }

                PyObject *tmp_val = PyLong_FromLong(path->num_waypoints);
                if (!tmp_val) {
                    Py_DECREF(path_dict);
                    Py_DECREF(sdc_list);
                    return NULL;
                }
                if (PyDict_SetItemString(path_dict, "num_waypoints", tmp_val) < 0) {
                    Py_DECREF(tmp_val);
                    Py_DECREF(path_dict);
                    Py_DECREF(sdc_list);
                    return NULL;
                }
                Py_DECREF(tmp_val);

                PyObject *wp_list = PyList_New(path->num_waypoints);
                if (!wp_list) {
                    Py_DECREF(path_dict);
                    Py_DECREF(sdc_list);
                    return NULL;
                }

                for (int j = 0; j < path->num_waypoints; j++) {
                    struct Waypoint *wp = &path->waypoints[j];
                    PyObject *wp_dict = PyDict_New();
                    if (!wp_dict) {
                        Py_DECREF(wp_list);
                        Py_DECREF(path_dict);
                        Py_DECREF(sdc_list);
                        return NULL;
                    }

#define SET_WAYPOINT_FLOAT(key, val)                                                                                   \
    tmp_val = PyFloat_FromDouble((double)val);                                                                         \
    if (!tmp_val) {                                                                                                    \
        Py_DECREF(wp_dict);                                                                                            \
        Py_DECREF(wp_list);                                                                                            \
        Py_DECREF(path_dict);                                                                                          \
        Py_DECREF(sdc_list);                                                                                           \
        return NULL;                                                                                                   \
    }                                                                                                                  \
    if (PyDict_SetItemString(wp_dict, key, tmp_val) < 0) {                                                             \
        Py_DECREF(tmp_val);                                                                                            \
        Py_DECREF(wp_dict);                                                                                            \
        Py_DECREF(wp_list);                                                                                            \
        Py_DECREF(path_dict);                                                                                          \
        Py_DECREF(sdc_list);                                                                                           \
        return NULL;                                                                                                   \
    }                                                                                                                  \
    Py_DECREF(tmp_val)

                    SET_WAYPOINT_FLOAT("s", wp->s);
                    SET_WAYPOINT_FLOAT("x", wp->x);
                    SET_WAYPOINT_FLOAT("y", wp->y);
                    SET_WAYPOINT_FLOAT("heading", wp->heading);
                    SET_WAYPOINT_FLOAT("kappa", wp->kappa);

                    tmp_val = PyLong_FromLong(wp->lane_id);
                    if (!tmp_val) {
                        Py_DECREF(wp_dict);
                        Py_DECREF(wp_list);
                        Py_DECREF(path_dict);
                        Py_DECREF(sdc_list);
                        return NULL;
                    }
                    if (PyDict_SetItemString(wp_dict, "lane_id", tmp_val) < 0) {
                        Py_DECREF(tmp_val);
                        Py_DECREF(wp_dict);
                        Py_DECREF(wp_list);
                        Py_DECREF(path_dict);
                        Py_DECREF(sdc_list);
                        return NULL;
                    }
                    Py_DECREF(tmp_val);

                    PyList_SetItem(wp_list, j, wp_dict);
                }

                if (PyDict_SetItemString(path_dict, "waypoints", wp_list) < 0) {
                    Py_DECREF(wp_list);
                    Py_DECREF(path_dict);
                    Py_DECREF(sdc_list);
                    return NULL;
                }
                Py_DECREF(wp_list);
                PyList_SetItem(sdc_list, i, path_dict);
            } else {
                Py_INCREF(Py_None);
                PyList_SetItem(sdc_list, i, Py_None);
            }
        }

        if (PyDict_SetItemString(dict, "sdc_paths", sdc_list) < 0) {
            Py_DECREF(sdc_list);
            return NULL;
        }
        Py_DECREF(sdc_list);
    } else {
        if (PyDict_SetItemString(dict, "sdc_paths", Py_None) < 0)
            return NULL;
    }

    /* Expose road_elements array as a list of dicts */
    if (env->road_elements && env->num_road_elements > 0) {
        PyObject *road_list = PyList_New(env->num_road_elements);
        if (!road_list)
            return NULL;
        for (int i = 0; i < env->num_road_elements; i++) {
            RoadMapElement *r = &env->road_elements[i];
            PyObject *road = PyDict_New();
            if (!road) {
                Py_DECREF(road_list);
                return NULL;
            }

            PyObject *tmp = PyLong_FromLong(r->id);
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

            /* Validate segment_length before using it */
            int seg_len = (r->segment_length > 0 && r->segment_length < 100000) ? r->segment_length : 0;
            tmp = PyLong_FromLong(seg_len);
            if (!tmp) {
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            if (PyDict_SetItemString(road, "segment_length", tmp) < 0) {
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
                    PyObject *fv = PyFloat_FromDouble((double)r->x[j]);
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
            } else {
                if (PyDict_SetItemString(road, "x", Py_None) < 0) {
                    Py_DECREF(road);
                    Py_DECREF(road_list);
                    return NULL;
                }
            }
            if (r->y && seg_len > 0) {
                PyObject *ly = PyList_New(seg_len);
                if (!ly) {
                    Py_DECREF(road);
                    Py_DECREF(road_list);
                    return NULL;
                }
                for (int j = 0; j < seg_len; j++) {
                    PyObject *fv = PyFloat_FromDouble((double)r->y[j]);
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
            } else {
                if (PyDict_SetItemString(road, "y", Py_None) < 0) {
                    Py_DECREF(road);
                    Py_DECREF(road_list);
                    return NULL;
                }
            }
            if (r->z && seg_len > 0) {
                PyObject *lz = PyList_New(seg_len);
                if (!lz) {
                    Py_DECREF(road);
                    Py_DECREF(road_list);
                    return NULL;
                }
                for (int j = 0; j < seg_len; j++) {
                    PyObject *fv = PyFloat_FromDouble((double)r->z[j]);
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
            } else {
                if (PyDict_SetItemString(road, "z", Py_None) < 0) {
                    Py_DECREF(road);
                    Py_DECREF(road_list);
                    return NULL;
                }
            }

            /* Lane-specific fields */
            tmp = PyLong_FromLong(r->entry_lanes);
            if (!tmp) {
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            if (PyDict_SetItemString(road, "entry_lanes", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            Py_DECREF(tmp);

            tmp = PyLong_FromLong(r->exit_lanes);
            if (!tmp) {
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            if (PyDict_SetItemString(road, "exit_lanes", tmp) < 0) {
                Py_DECREF(tmp);
                Py_DECREF(road);
                Py_DECREF(road_list);
                return NULL;
            }
            Py_DECREF(tmp);

            PyObject *pf = PyFloat_FromDouble((double)r->speed_limit);
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

            PyList_SetItem(road_list, i, road);
        }
        if (PyDict_SetItemString(dict, "road_elements", road_list) < 0) {
            Py_DECREF(road_list);
            return NULL;
        }
        Py_DECREF(road_list);
    } else {
        if (PyDict_SetItemString(dict, "road_elements", Py_None) < 0)
            return NULL;
    }

    /* Expose traffic_elements array as a list of dicts */
    if (env->traffic_elements && env->num_traffic_elements > 0) {
        PyObject *traffic_list = PyList_New(env->num_traffic_elements);
        if (!traffic_list)
            return NULL;
        for (int i = 0; i < env->num_traffic_elements; i++) {
            TrafficControlElement *t = &env->traffic_elements[i];
            PyObject *traffic = PyDict_New();
            if (!traffic) {
                Py_DECREF(traffic_list);
                return NULL;
            }

            PyObject *tmp = PyLong_FromLong(t->id);
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

            /* Validate state_length before using it */
            int state_len = (t->state_length > 0 && t->state_length < 10000) ? t->state_length : 0;
            tmp = PyLong_FromLong(state_len);
            if (!tmp) {
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            if (PyDict_SetItemString(traffic, "state_length", tmp) < 0) {
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
            } else {
                if (PyDict_SetItemString(traffic, "states", Py_None) < 0) {
                    Py_DECREF(traffic);
                    Py_DECREF(traffic_list);
                    return NULL;
                }
            }

            /* Position */
            PyObject *pf = PyFloat_FromDouble((double)t->x);
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

            pf = PyFloat_FromDouble((double)t->y);
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

            pf = PyFloat_FromDouble((double)t->z);
            if (!pf) {
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            if (PyDict_SetItemString(traffic, "z", pf) < 0) {
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
            } else {
                if (PyDict_SetItemString(traffic, "controlled_lanes", Py_None) < 0) {
                    Py_DECREF(traffic);
                    Py_DECREF(traffic_list);
                    return NULL;
                }
            }

            PyList_SetItem(traffic_list, i, traffic);
        }
        if (PyDict_SetItemString(dict, "traffic_elements", traffic_list) < 0) {
            Py_DECREF(traffic_list);
            return NULL;
        }
        Py_DECREF(traffic_list);
    } else {
        if (PyDict_SetItemString(dict, "traffic_elements", Py_None) < 0)
            return NULL;
    }

    /* Map corners (bounding box) from GridMap */
    if (env->grid_map) {
        PyObject *corners_list = PyList_New(4);
        if (!corners_list)
            return NULL;

        PyObject *brx = PyFloat_FromDouble((double)env->grid_map->bottom_right_x);
        PyObject *bry = PyFloat_FromDouble((double)env->grid_map->bottom_right_y);
        PyObject *tlx = PyFloat_FromDouble((double)env->grid_map->top_left_x);
        PyObject *tly = PyFloat_FromDouble((double)env->grid_map->top_left_y);

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
        if (PyDict_SetItemString(dict, "map_corners", Py_None) < 0)
            return NULL;
    }
    if (env->observations && env->active_agent_count > 0) {
        /* Agent observations */
        int include_goal = (env->target_type == TARGET_GOAL || env->target_type == TARGET_BOTH);
        int include_waypoints = (env->target_type == TARGET_WAYPOINTS || env->target_type == TARGET_BOTH);
        int ego_dim;
        if (include_goal) {
            ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
        } else {
            ego_dim = (env->dynamics_model == JERK) ? EGO_FEATURES_JERK_NO_GOAL : EGO_FEATURES_CLASSIC_NO_GOAL;
        }
        int max_obs = ego_dim + PARTNER_FEATURES * MAX_AGENTS_OBSERVATIONS +
                      ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS + MAX_TRAFFIC_CONTROLS * TRAFFIC_CONTROL_FEATURES;
        if (include_waypoints) {
            max_obs += GPS_FEATURES * MAX_GPS_OBSERVATIONS;
        }
        if (env->reward_conditioning) {
            max_obs += NUM_REWARD_COEFS;
        }
        PyObject *obs_data = PyList_New(env->active_agent_count);
        if (!obs_data)
            return NULL;

        float (*observations)[max_obs] = (float (*)[max_obs])env->observations;

        for (int i = 0; i < env->active_agent_count; i++) {
            PyObject *agent_obs = PyList_New(max_obs);
            if (!agent_obs) {
                Py_DECREF(obs_data);
                return NULL;
            }

            for (int j = 0; j < max_obs; j++) {
                PyObject *obs_val = PyFloat_FromDouble((double)observations[i][j]);
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
        if (PyDict_SetItemString(dict, "agent_observations", Py_None) < 0)
            return NULL;
    }

    return dict;
}

static PyObject *my_shared(PyObject *self, PyObject *args, PyObject *kwargs) {
    char *map_dir = unpack_str(kwargs, "map_dir");
    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");
    int starting_map_counter = unpack(kwargs, "starting_map_counter");
    int eval_mode = unpack(kwargs, "eval_mode");
    int s_map_counter = starting_map_counter;
    int init_mode = unpack(kwargs, "init_mode");
    int control_mode = unpack(kwargs, "control_mode");
    int simulation_mode = unpack(kwargs, "simulation_mode");
    int init_steps = unpack(kwargs, "init_steps");
    int reach_goal_behavior = unpack(kwargs, "reach_goal_behavior");
    int seed = unpack(kwargs, "seed");
    int min_agents_per_env = unpack(kwargs, "min_agents_per_env");
    int max_agents_per_env = unpack(kwargs, "max_agents_per_env");

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

            PyObject *agent_offsets = PyList_New(env_count + 1);
            PyObject *map_ids_list = PyList_New(env_count);

            int offset = 0;
            for (int i = 0; i < env_count; i++) {
                PyList_SetItem(agent_offsets, i, PyLong_FromLong(offset));
                PyList_SetItem(map_ids_list, i, PyLong_FromLong((s_map_counter + i) % num_maps));
                int count = (i < env_count - 1) ? agents_per_env : (num_agents - offset);
                offset += count;
            }
            PyList_SetItem(agent_offsets, env_count, PyLong_FromLong(num_agents));

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
                int upper = (remaining - min_agents_per_env < max_agents_per_env) ? remaining - min_agents_per_env
                                                                                  : max_agents_per_env;
                count = min_agents_per_env + rand() % (upper - min_agents_per_env + 1);
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
        max_envs = 1;
    }
    PyObject *agent_offsets = PyList_New(max_envs + 1);
    PyObject *map_ids = PyList_New(max_envs);
    // getting env count
    while (total_agent_count < num_agents && env_count < max_envs) {
        char map_file[100];
        if (eval_mode) {
            map_id = s_map_counter % num_maps;
            s_map_counter += 1;
        } else {
            map_id = rand() % num_maps;
        }

        Drive *env = calloc(1, sizeof(Drive));
        env->init_mode = init_mode;
        env->control_mode = control_mode;
        env->simulation_mode = simulation_mode;
        env->init_steps = init_steps;
        env->reach_goal_behavior = reach_goal_behavior;
        env->num_max_agents = max_agents_per_env;
        sprintf(map_file, "%s/map_%03d.bin", map_dir, map_id);
        load_map_binary(map_file, env);

        set_active_agents(env);

        // Skip map if it doesn't contain any controllable agents
        if (env->active_agent_count == 0) {
            maps_checked++;
            // Fix the safeguard, the maps_checked is increased even when the same map is picked.
            // Safeguard: if we've checked all available maps and found no active agents, raise an error
            // if(maps_checked >= num_maps) {
            //     for(int j=0;j<env->num_total_agents;j++) free_agent(&env->agents[j]);
            //     for (int j=0;j<env->num_road_elements;j++) free_road_element(&env->road_elements[j]);
            //     for (int j=0;j<env->num_traffic_elements;j++) free_traffic_element(&env->traffic_elements[j]);
            //     free(env->agents);
            //     free(env->road_elements);
            //     free(env->traffic_elements);
            //     free(env->active_agent_indices);
            //     free(env->static_agent_indices);
            //     free(env->expert_static_agent_indices);
            //     free(env);
            //     Py_DECREF(agent_offsets);
            //     Py_DECREF(map_ids);
            //     char error_msg[256];
            //     sprintf(error_msg, "No controllable agents found in any of the %d available maps", num_maps);
            //     PyErr_SetString(PyExc_ValueError, error_msg);
            //     return NULL;
            // }

            for (int j = 0; j < env->num_total_agents; j++)
                free_agent(&env->agents[j]);
            for (int j = 0; j < env->num_road_elements; j++)
                free_road_element(&env->road_elements[j]);
            for (int j = 0; j < env->num_traffic_elements; j++)
                free_traffic_element(&env->traffic_elements[j]);
            free(env->agents);
            free(env->road_elements);
            free(env->traffic_elements);
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
        for (int j = 0; j < env->num_total_agents; j++)
            free_agent(&env->agents[j]);
        for (int j = 0; j < env->num_road_elements; j++)
            free_road_element(&env->road_elements[j]);
        for (int j = 0; j < env->num_traffic_elements; j++)
            free_traffic_element(&env->traffic_elements[j]);
        free(env->agents);
        free(env->road_elements);
        free(env->traffic_elements);
        free(env->active_agent_indices);
        free(env->static_agent_indices);
        free(env->expert_static_agent_indices);
        free(env);
    }
    // printf("Generated %d environments to cover %d agents (requested %d agents)\n", env_count, total_agent_count,
    // num_agents);
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
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->ini_file = unpack_str(kwargs, "ini_file");

    env_init_config conf = {0};
    if (ini_parse(env->ini_file, handler, &conf) < 0) {
        printf("Error while loading %s", env->ini_file);
    }
    if (kwargs && PyDict_GetItemString(kwargs, "scenario_length")) {
        conf.scenario_length = (int)unpack(kwargs, "scenario_length");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "termination_mode")) {
        conf.termination_mode = (int)unpack(kwargs, "termination_mode");
    }
    if (conf.scenario_length <= 0) {
        PyErr_SetString(PyExc_ValueError, "scenario_length must be > 0 (set in INI or kwargs)");
        return -1;
    }
    env->action_type = conf.action_type;
    env->dynamics_model = conf.dynamics_model;
    env->reward_vehicle_collision = conf.reward_vehicle_collision;
    env->reward_offroad_collision = conf.reward_offroad_collision;
    env->reward_traffic_light_violation = conf.reward_traffic_light_violation;
    env->reward_goal = conf.reward_goal;
    env->reward_goal_post_respawn = conf.reward_goal_post_respawn;
    env->reward_ade = conf.reward_ade;
    env->reward_progression = conf.reward_progression;
    env->reward_offroute = conf.reward_offroute;
    env->reward_speed = conf.reward_speed;
    env->reward_comfort = conf.reward_comfort;
    env->reward_velocity = conf.reward_velocity;
    env->reward_lane_align = conf.reward_lane_align;
    env->reward_lane_center = conf.reward_lane_center;
    env->reward_timestep = conf.reward_timestep;
    env->termination_mode = conf.termination_mode;
    env->goal_radius = conf.goal_radius;
    env->scenario_length = conf.scenario_length;
    env->reach_goal_behavior = conf.reach_goal_behavior;
    env->target_type = conf.target_type;
    env->collision_behavior = conf.collision_behavior;
    env->offroad_behavior = conf.offroad_behavior;
    env->traffic_light_behavior = conf.traffic_light_behavior;
    env->end_sdc_path_behavior = conf.end_sdc_path_behavior;
    env->goal_distance = conf.goal_distance;
    env->waypoints_spacing = conf.waypoints_spacing;
    env->reward_conditioning = conf.reward_conditioning;
    env->reward_randomization = conf.reward_randomization;
    // env->policy_agents_per_env = unpack(kwargs, "num_policy_controlled_agents");
    // env->control_all_agents = unpack(kwargs, "control_all_agents");
    // env->deterministic_agent_selection = unpack(kwargs, "deterministic_agent_selection");
    env->dt = conf.dt;
    env->init_mode = (int)unpack(kwargs, "init_mode");
    env->control_mode = (int)unpack(kwargs, "control_mode");
    env->simulation_mode = (int)unpack(kwargs, "simulation_mode");
    env->reach_goal_behavior = (int)unpack(kwargs, "reach_goal_behavior");
    env->eval_mode = (int)unpack(kwargs, "eval_mode");
    if (kwargs && PyDict_GetItemString(kwargs, "target_type")) {
        env->target_type = (int)unpack(kwargs, "target_type");
    }
    env->goal_radius = (float)unpack(kwargs, "goal_radius");
    char *map_dir = unpack_str(kwargs, "map_dir");
    int map_id = unpack(kwargs, "map_id");
    int max_agents = unpack(kwargs, "max_agents");
    int init_steps = unpack(kwargs, "init_steps");
    char map_file[100];
    sprintf(map_file, "%s/map_%03d.bin", map_dir, map_id);
    env->num_controllable_agents = max_agents;
    env->num_max_agents = conf.max_agents_per_env;
    env->map_name = strdup(map_file);
    env->init_steps = init_steps;
    env->timestep = init_steps;
    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "red_light_violation_rate", log->red_light_violation_rate);
    // assign_to_dict(dict, "avg_displacement_error", log->avg_displacement_error);
    assign_to_dict(dict, "comfort_violation_count", log->comfort_violation_count);
    assign_to_dict(dict, "velocity_progress_sum", log->velocity_progress_sum);
    assign_to_dict(dict, "num_goals_reached", log->num_goals_reached);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    assign_to_dict(dict, "progression_rate", log->progression_rate);
    assign_to_dict(dict, "lane_align_rate", log->lane_align_rate);
    assign_to_dict(dict, "lane_center_rate", log->lane_center_rate);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "score", log->score);
    // assign_to_dict(dict, "active_agent_count", log->active_agent_count);
    // assign_to_dict(dict, "expert_static_car_count", log->expert_static_car_count);
    // assign_to_dict(dict, "static_car_count", log->static_car_count);
    // assign_to_dict(dict, "avg_offroad_per_agent", log->avg_offroad_per_agent);
    // assign_to_dict(dict, "avg_collisions_per_agent", log->avg_collisions_per_agent);
    assign_to_dict(dict, "avg_speed_per_agent", log->avg_speed_per_agent);

    return 0;
}
