#include "drive.h"
#define Env Drive
#define MY_SHARED
#define MY_PUT
#define MY_GET
#define MY_COMPLETED_EPISODES
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

    v = PyLong_FromLong(env->timestep);
    if (!v)
        return NULL;
    if (PyDict_SetItemString(dict, "episode_timestep", v) < 0) {
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

    v = PyLong_FromLong(env->dynamics_model);
    if (!v)
        return NULL;
    if (PyDict_SetItemString(dict, "dynamics_model", v) < 0) {
        Py_DECREF(v);
        return NULL;
    }
    Py_DECREF(v);

    float target_episode_length = 0.0f;
    float target_episode_return = 0.0f;
    float did_target_collide = 0.0f;
    float did_target_offroad = 0.0f;
    float did_target_run_light = 0.0f;
    float did_target_fail = 0.0f;
    if (env->active_agent_count > 0) {
        target_episode_length = env->logs[0].episode_length;
        target_episode_return = env->logs[0].episode_return;
        did_target_collide = env->logs[0].collision_rate > 0.0f ? 1.0f : 0.0f;
        did_target_offroad = env->logs[0].offroad_rate > 0.0f ? 1.0f : 0.0f;
        did_target_run_light = env->logs[0].red_light_violation_rate > 0.0f ? 1.0f : 0.0f;
        did_target_fail = did_target_collide || did_target_offroad || did_target_run_light ? 1.0f : 0.0f;
    }

    assign_to_dict(dict, "target_episode_length", target_episode_length);
    assign_to_dict(dict, "target_episode_return", target_episode_return);
    assign_to_dict(dict, "did_target_collide", did_target_collide);
    assign_to_dict(dict, "did_target_offroad", did_target_offroad);
    assign_to_dict(dict, "did_target_run_light", did_target_run_light);
    assign_to_dict(dict, "did_target_fail", did_target_fail);

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

            pf = PyFloat_FromDouble((double)a->wheelbase);
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

            tmp = PyLong_FromLong(a->controller);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "controller", tmp) < 0) {
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
            tmp = PyLong_FromLong(a->num_waypoints_reached);
            if (!tmp) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            if (PyDict_SetItemString(agent, "num_waypoints_reached", tmp) < 0) {
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
            PyObject *metrics = PyList_New(NUM_METRICS);
            if (!metrics) {
                Py_DECREF(agent);
                Py_DECREF(agents_list);
                return NULL;
            }
            for (int j = 0; j < NUM_METRICS; j++) {
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

                    tmp_val = PyLong_FromLong(wp->lane_idx);
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

            /* Stop line endpoints */
            PyObject *sl = PyList_New(6);
            if (!sl) {
                Py_DECREF(traffic);
                Py_DECREF(traffic_list);
                return NULL;
            }
            for (int k = 0; k < 6; k++) {
                PyList_SetItem(sl, k, PyFloat_FromDouble((double)t->stop_line[k]));
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
            PyObject *pf = PyFloat_FromDouble((double)mid_x);
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

            pf = PyFloat_FromDouble((double)mid_y);
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
        int max_obs = compute_observation_size(env);
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

static int my_completed_episode_to_dict(PyObject *dict, Env *env, CompletedEpisodeSummary *summary) {
    if (my_log(dict, env, &summary->log, summary->n) != 0) {
        return -1;
    }

    if (summary->avoidability_debug != NULL && summary->avoidability_debug->valid) {
        AvoidabilityDebug *debug = summary->avoidability_debug;
        PyObject *trace = PyDict_New();
        PyObject *collision =
            Py_BuildValue("{s:i,s:i,s:i}", "target_agent_index", debug->target_agent_index, "collision_adversary_index",
                          debug->collision_adversary_index, "collision_timestep", debug->collision_timestep);
        PyObject *constants = Py_BuildValue(
            "{s:f,s:f,s:f,s:f,s:i,s:i,s:f,s:i,s:f,s:f,s:f,s:f,s:f}", "dt", debug->dt, "braking_deceleration",
            debug->braking_deceleration, "reaction_time_seconds", debug->reaction_time_seconds,
            "braking_rollout_delay_seconds", debug->braking_rollout_delay_seconds, "max_extension_steps",
            debug->max_extension_steps, "max_rollout_steps", debug->max_rollout_steps, "ttc_margin_seconds",
            debug->ttc_margin_seconds, "ttc_max_projection_steps", debug->ttc_max_projection_steps,
            "lateral_buffer_base_distance", debug->lateral_buffer_base_distance, "lateral_buffer_response_time_seconds",
            debug->lateral_buffer_response_time_seconds, "lateral_buffer_deceleration",
            debug->lateral_buffer_deceleration, "lateral_buffer_max_distance", debug->lateral_buffer_max_distance,
            "route_sample_spacing", debug->route_sample_spacing);
        if (constants != NULL) {
            assign_to_dict(constants, "reaction_window_half_width_seconds", debug->reaction_window_half_width_seconds);
            assign_to_dict(constants, "rollout_until_adversary_stop", (float)debug->rollout_until_adversary_stop);
        }
        PyObject *classification =
            Py_BuildValue("{s:f,s:i,s:i,s:i}", "t_brake", debug->last_avoidable_braking_seconds_before_collision,
                          "genuine_target_failure", debug->genuine_target_failure, "adversary_forced",
                          debug->adversary_forced, "unavoidable", debug->unavoidable);
        PyObject *target_route = PyList_New(debug->target_route_length);
        if (target_route != NULL) {
            for (int i = 0; i < debug->target_route_length; i++) {
                PyObject *lane_index = PyLong_FromLong(debug->target_route[i]);
                if (lane_index == NULL || PyList_SetItem(target_route, i, lane_index) < 0) {
                    Py_XDECREF(lane_index);
                    Py_CLEAR(target_route);
                    break;
                }
            }
        }
        PyObject *target_route_length = PyLong_FromLong(debug->target_route_length);

#define SNAPSHOT_DICT(snapshot)                                                                                        \
    Py_BuildValue("{s:i,s:i,s:i,s:i,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:i,s:i}", "valid", (snapshot).valid, "index", \
                  (snapshot).index, "id", (snapshot).id, "type", (snapshot).type, "x", (snapshot).x, "y",              \
                  (snapshot).y, "z", (snapshot).z, "heading", (snapshot).heading, "length", (snapshot).length,         \
                  "width", (snapshot).width, "height", (snapshot).height, "vx", (snapshot).vx, "vy", (snapshot).vy,    \
                  "active", (snapshot).active, "stopped", (snapshot).stopped)

        PyObject *target_snapshot = SNAPSHOT_DICT(debug->target_at_collision);
        PyObject *adversary_snapshot = SNAPSHOT_DICT(debug->adversary_at_collision);
        if (collision != NULL && target_snapshot != NULL && adversary_snapshot != NULL) {
            PyDict_SetItemString(collision, "target", target_snapshot);
            PyDict_SetItemString(collision, "adversary", adversary_snapshot);
        }
        Py_XDECREF(target_snapshot);
        Py_XDECREF(adversary_snapshot);

        const char *candidate_keys[] = {
            "steps_back",
            "avoided",
            "collision_with_original_adversary",
            "at_fault_collision_with_other_adversary",
            "blocking_agent_index",
            "blocking_rollout_step",
            "blocking_agent",
            "ignored_overlap_agent_index",
            "ignored_overlap_rollout_step",
            "ignored_overlap_agent",
        };
        PyObject *candidate_arrays = PyDict_New();
        PyObject *candidate_lists[10] = {0};
        int trace_ok = trace != NULL && collision != NULL && constants != NULL && classification != NULL &&
                       target_route != NULL && target_route_length != NULL && candidate_arrays != NULL;
        for (int j = 0; j < 10 && trace_ok; j++) {
            candidate_lists[j] = PyList_New(0);
            if (candidate_lists[j] == NULL ||
                PyDict_SetItemString(candidate_arrays, candidate_keys[j], candidate_lists[j]) < 0) {
                trace_ok = 0;
            }
        }
        for (int i = 0; i < debug->candidate_count && trace_ok; i++) {
            AvoidabilityCandidateDebug *candidate = &debug->candidates[i];
            PyObject *values[10] = {
                PyLong_FromLong(candidate->steps_back),
                PyBool_FromLong(candidate->avoided),
                PyBool_FromLong(candidate->collision_with_original_adversary),
                PyBool_FromLong(candidate->at_fault_collision_with_other_adversary),
                PyLong_FromLong(candidate->blocking_agent_index),
                PyLong_FromLong(candidate->blocking_rollout_step),
                SNAPSHOT_DICT(candidate->blocking_agent),
                PyLong_FromLong(candidate->ignored_overlap_agent_index),
                PyLong_FromLong(candidate->ignored_overlap_rollout_step),
                SNAPSHOT_DICT(candidate->ignored_overlap_agent),
            };
            for (int j = 0; j < 10; j++) {
                if (values[j] == NULL || PyList_Append(candidate_lists[j], values[j]) < 0) {
                    trace_ok = 0;
                }
                Py_XDECREF(values[j]);
            }
        }
#undef SNAPSHOT_DICT

        if (trace_ok && PyDict_SetItemString(trace, "collision", collision) == 0 &&
            PyDict_SetItemString(trace, "constants", constants) == 0 &&
            PyDict_SetItemString(trace, "classification", classification) == 0 &&
            PyDict_SetItemString(trace, "target_route_lane_indices", target_route) == 0 &&
            PyDict_SetItemString(trace, "target_route_length", target_route_length) == 0 &&
            PyDict_SetItemString(trace, "candidate_arrays", candidate_arrays) == 0 &&
            PyDict_SetItemString(dict, "avoidability_debug", trace) == 0) {
            // Added successfully.
        } else if (trace_ok) {
            trace_ok = 0;
        }
        for (int j = 0; j < 10; j++) {
            Py_XDECREF(candidate_lists[j]);
        }
        Py_XDECREF(candidate_arrays);
        Py_XDECREF(collision);
        Py_XDECREF(constants);
        Py_XDECREF(classification);
        Py_XDECREF(target_route);
        Py_XDECREF(target_route_length);
        Py_XDECREF(trace);
        if (!trace_ok) {
            return -1;
        }
    }

    assign_to_dict(dict, "n", summary->n);
    assign_to_dict(dict, "target_n", summary->target_n);
    assign_to_dict(dict, "active_agent_count", summary->active_agent_count);
    assign_to_dict(dict, "episode_timestep", summary->timestep);

    if (summary->map_name[0] != '\0') {
        PyObject *s = PyUnicode_FromString(summary->map_name);
        if (!s)
            return -1;
        if (PyDict_SetItemString(dict, "map_name", s) < 0) {
            Py_DECREF(s);
            return -1;
        }
        Py_DECREF(s);
    } else if (PyDict_SetItemString(dict, "map_name", Py_None) < 0) {
        return -1;
    }

    if (summary->scenario_id[0] != '\0') {
        PyObject *s = PyUnicode_FromString(summary->scenario_id);
        if (!s)
            return -1;
        if (PyDict_SetItemString(dict, "scenario_id", s) < 0) {
            Py_DECREF(s);
            return -1;
        }
        Py_DECREF(s);
    } else if (PyDict_SetItemString(dict, "scenario_id", Py_None) < 0) {
        return -1;
    }

    if (summary->dataset_name[0] != '\0') {
        PyObject *s = PyUnicode_FromString(summary->dataset_name);
        if (!s)
            return -1;
        if (PyDict_SetItemString(dict, "dataset_name", s) < 0) {
            Py_DECREF(s);
            return -1;
        }
        Py_DECREF(s);
    } else if (PyDict_SetItemString(dict, "dataset_name", Py_None) < 0) {
        return -1;
    }

    return 0;
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
    int init_steps = unpack(kwargs, "init_steps");
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

            env_count = env_count > num_eval_scenarios ? num_eval_scenarios : env_count;

            PyObject *agent_offsets = PyList_New(env_count + 1);
            PyObject *map_ids_list = PyList_New(env_count);

            int offset = 0;
            for (int i = 0; i < env_count; i++) {
                PyList_SetItem(agent_offsets, i, PyLong_FromLong(offset));
                PyList_SetItem(map_ids_list, i, PyLong_FromLong((s_map_counter + i) % num_maps));
                int remaining = num_agents - offset;
                offset += (remaining < agents_per_env) ? remaining : agents_per_env;
            }
            PyList_SetItem(agent_offsets, env_count, PyLong_FromLong(offset));

            PyObject *tuple = PyTuple_New(3);
            PyTuple_SetItem(tuple, 0, agent_offsets);
            PyTuple_SetItem(tuple, 1, map_ids_list);
            PyTuple_SetItem(tuple, 2, PyLong_FromLong(env_count));
            return tuple;
        }

        // Training mode: random agent counts per env
        int max_env_count = (num_agents / min_agents_per_env) + 1;
        int *agent_counts = malloc(max_env_count * sizeof(int));
        int *map_id_values = malloc(max_env_count * sizeof(int));
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
                int current_upper_bound =
                    (absolute_max_allowed < max_agents_per_env) ? absolute_max_allowed : max_agents_per_env;

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
            agent_counts[env_count] = count;
            map_id_values[env_count] = rand() % num_maps;
            env_count++;
            remaining -= count;
        }

        // Build Python return lists
        PyObject *agent_offsets = PyList_New(env_count + 1);
        PyObject *map_ids_list = PyList_New(env_count);

        int offset = 0;
        for (int i = 0; i < env_count; i++) {
            PyList_SetItem(agent_offsets, i, PyLong_FromLong(offset));
            PyList_SetItem(map_ids_list, i, PyLong_FromLong(map_id_values[i]));
            offset += agent_counts[i];
        }
        PyList_SetItem(agent_offsets, env_count, PyLong_FromLong(num_agents));

        free(agent_counts);
        free(map_id_values);

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
        env->init_steps = init_steps;
        env->num_max_agents = max_agents_per_env;
        load_map_binary(map_file, env);

        set_active_agents(env);

        // Skip map if it doesn't contain any controllable agents
        if (env->active_agent_count == 0) {
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
        PyList_SetItem(map_ids, env_count, PyLong_FromLong(map_id));
        // Store agent offset
        PyList_SetItem(agent_offsets, env_count, PyLong_FromLong(total_agent_count));
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
    PyObject *map_cache_handle = PyDict_GetItemString(kwargs, "map_cache_handle");
    if (map_cache_handle && map_cache_handle != Py_None) {
        if (!PyObject_TypeCheck(map_cache_handle, &PyLong_Type)) {
            PyErr_SetString(PyExc_TypeError, "map_cache_handle must be an integer");
            return 1;
        }
        env->map_cache = (DriveMapCache *)PyLong_AsVoidPtr(map_cache_handle);
    }

    env->action_type = (int)unpack(kwargs, "action_type");
    env->dynamics_model = (int)unpack(kwargs, "dynamics_model");
    env->reward_goal = (float)unpack(kwargs, "reward_goal");
    env->reward_vehicle_collision = (float)unpack(kwargs, "reward_vehicle_collision");
    env->reward_offroad_collision = (float)unpack(kwargs, "reward_offroad_collision");
    env->reward_comfort = (float)unpack(kwargs, "reward_comfort");
    env->reward_lane_align = (float)unpack(kwargs, "reward_lane_align");
    env->reward_vel_align = (float)unpack(kwargs, "reward_vel_align");
    env->reward_lane_center = (float)unpack(kwargs, "reward_lane_center");
    env->reward_center_bias = (float)unpack(kwargs, "reward_center_bias");
    env->reward_velocity = (float)unpack(kwargs, "reward_velocity");
    env->reward_reverse = (float)unpack(kwargs, "reward_reverse");
    env->reward_stop_line = (float)unpack(kwargs, "reward_stop_line");
    env->reward_timestep = (float)unpack(kwargs, "reward_timestep");
    env->reward_overspeed = (float)unpack(kwargs, "reward_overspeed");
    env->reward_ade = (float)unpack(kwargs, "reward_ade");
    env->adv_reward_weight_collision = (float)unpack(kwargs, "adv_reward_weight_collision");
    env->adv_reward_weight_offroad = (float)unpack(kwargs, "adv_reward_weight_offroad");
    env->adv_reward_weight_traffic_light = (float)unpack(kwargs, "adv_reward_weight_traffic_light");
    env->adv_reward_weight_drive = (float)unpack(kwargs, "adv_reward_weight_drive");
    env->adv_target_offroad_reward = (float)unpack(kwargs, "adv_target_offroad_reward");
    env->adv_target_collision_reward = (float)unpack(kwargs, "adv_target_collision_reward");
    env->adv_target_collision_reward_use_responsibility =
        (bool)unpack(kwargs, "adv_target_collision_reward_use_responsibility");
    env->adv_target_failure_reward = (float)unpack(kwargs, "adv_target_failure_reward");
    env->adv_target_avoidability_reward = (float)unpack(kwargs, "adv_target_avoidability_reward");
    env->adv_target_detection_reward = (float)unpack(kwargs, "adv_target_detection_reward");
    env->adv_target_time_reward_tau = (float)unpack(kwargs, "adv_target_time_reward_tau");
    env->adv_target_hit_at_fault_bonus = (float)unpack(kwargs, "adv_target_hit_at_fault_bonus");
    env->adv_target_hit_low_responsibility_threshold =
        (float)unpack(kwargs, "adv_target_hit_low_responsibility_threshold");
    env->adv_target_hit_unavoidable_penalty = (float)unpack(kwargs, "adv_target_hit_unavoidable_penalty");
    env->adv_target_hit_unavoidable_behavior = (int)unpack(kwargs, "adv_target_hit_unavoidable_behavior");
    env->collision_behavior = (int)unpack(kwargs, "collision_behavior");
    env->ignore_target_collision_behavior = (bool)unpack(kwargs, "ignore_target_collision_behavior");
    env->remove_target_on_collision_or_offroad = (bool)unpack(kwargs, "remove_target_on_collision_or_offroad");
    env->offroad_behavior = (int)unpack(kwargs, "offroad_behavior");
    env->traffic_light_behavior = (int)unpack(kwargs, "traffic_light_behavior");
    env->deterministic_traffic_lights = (bool)unpack(kwargs, "deterministic_traffic_lights");
    env->goal_radius = (float)unpack(kwargs, "goal_radius");
    env->min_waypoint_spacing = (float)unpack(kwargs, "min_waypoint_spacing");
    env->max_waypoint_spacing = (float)unpack(kwargs, "max_waypoint_spacing");
    env->num_target_waypoints = (int)unpack(kwargs, "num_target_waypoints");
    if (env->num_target_waypoints > MAX_TARGET_WAYPOINTS) {
        env->num_target_waypoints = MAX_TARGET_WAYPOINTS;
    }
    env->target_type = (int)unpack(kwargs, "target_type");
    env->max_boundary_segment_observations = (int)unpack(kwargs, "max_boundary_segment_observations");
    env->max_lane_segment_observations = (int)unpack(kwargs, "max_lane_segment_observations");
    env->max_partner_observations = (int)unpack(kwargs, "max_partner_observations");
    env->target_max_partner_observations = (int)unpack(kwargs, "target_max_partner_observations");
    env->adversary_max_partner_observations = (int)unpack(kwargs, "adversary_max_partner_observations");
    env->max_traffic_control_observations = (int)unpack(kwargs, "max_traffic_control_observations");
    env->traffic_control_scope = (int)unpack(kwargs, "traffic_control_scope");
    env->dt = (float)unpack(kwargs, "dt");
    env->pdm_horizon = (float)unpack(kwargs, "pdm_horizon");
    env->pdm_planning_dt = (float)unpack(kwargs, "pdm_planning_dt");
    env->spawn_initial_speed = (float)unpack(kwargs, "spawn_initial_speed");
    env->goal_speed = (float)unpack(kwargs, "goal_speed");
    env->scenario_length = (int)unpack(kwargs, "scenario_length");
    env->termination_mode = (int)unpack(kwargs, "termination_mode");
    env->inactive_agent_threshold = (float)unpack(kwargs, "inactive_agent_threshold");
    env->adversarial_termination_mode = (int)unpack(kwargs, "adversarial_termination_mode");
    env->terminate_ep_on_target_failure = (bool)unpack(kwargs, "terminate_ep_on_target_failure");
    char *map_file = unpack_str(kwargs, "map_file");
    env->map_name = map_file;
    env->num_controllable_agents = (int)unpack(kwargs, "max_agents");
    env->num_max_agents = (int)unpack(kwargs, "max_agents_per_env");
    env->targeted_spawn_mode = (int)unpack(kwargs, "targeted_spawn_mode");
    env->targeted_spawn_radius = (float)unpack(kwargs, "targeted_spawn_radius");
    env->targeted_spawn_attempts = (int)unpack(kwargs, "targeted_spawn_attempts");
    env->spawn_min_separation = (float)unpack(kwargs, "spawn_min_separation");
    env->targeted_spawn_close_probability = (float)unpack(kwargs, "targeted_spawn_close_probability");
    int init_steps = (int)unpack(kwargs, "init_steps");
    env->init_steps = init_steps;
    env->timestep = init_steps;
    env->init_mode = (int)unpack(kwargs, "init_mode");
    env->control_mode = (int)unpack(kwargs, "control_mode");
    env->sdc_controller = (int)unpack(kwargs, "sdc_controller");
    env->non_sdc_controller = (int)unpack(kwargs, "non_sdc_controller");
    env->simulation_mode = (int)unpack(kwargs, "simulation_mode");
    env->reward_conditioning = (bool)unpack(kwargs, "reward_conditioning");
    env->reward_randomization = (bool)unpack(kwargs, "reward_randomization");
    env->compute_eval_metrics = (bool)unpack(kwargs, "compute_eval_metrics");
    env->capture_avoidability_debug = (bool)unpack(kwargs, "capture_avoidability_debug");
    env->eval_mode = (int)unpack(kwargs, "eval_mode");
    env->max_goal_position = (float)unpack(kwargs, "max_goal_position");
    env->max_position = (float)unpack(kwargs, "max_position");
    env->max_veh_len = (float)unpack(kwargs, "max_veh_len");
    env->max_veh_width = (float)unpack(kwargs, "max_veh_width");
    env->max_road_segment_length = (float)unpack(kwargs, "max_road_segment_length");
    env->max_road_segment_width = (float)unpack(kwargs, "max_road_segment_width");
    env->max_traffic_control_distance = (float)unpack(kwargs, "max_traffic_control_distance");
    env->agent_obs_max_dist = (float)unpack(kwargs, "agent_obs_max_dist");
    env->road_obs_front_dist = (float)unpack(kwargs, "road_obs_front_dist");
    env->road_obs_behind_dist = (float)unpack(kwargs, "road_obs_behind_dist");
    env->road_obs_side_dist = (float)unpack(kwargs, "road_obs_side_dist");
    env->obs_lane_segment_count = (int)unpack(kwargs, "obs_lane_segment_count");
    env->obs_boundary_segment_count = (int)unpack(kwargs, "obs_boundary_segment_count");
    env->partner_blindness_prob = (float)unpack(kwargs, "partner_blindness_prob");
    env->phantom_braking_prob = (float)unpack(kwargs, "phantom_braking_prob");
    env->phantom_braking_trigger_prob = (float)unpack(kwargs, "phantom_braking_trigger_prob");
    env->phantom_braking_duration = (int)unpack(kwargs, "phantom_braking_duration");

    init(env);
    return 0;
}

static int my_log(PyObject *dict, Env *env, Log *log, float n) {
    float total_distance_travelled = log->total_distance_travelled * n;
    float total_infractions = log->total_infractions * n;
    float avg_distance_per_infraction = total_distance_travelled / fmaxf(1.0f, total_infractions);
    float safe_episode_length = fmaxf(log->episode_length, 1.0f);
    float safe_target_episode_length = fmaxf(log->target_episode_length, 1.0f);

    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "target_episode_length", log->target_episode_length);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_return_collision", log->episode_return_collision);
    assign_to_dict(dict, "episode_return_offroad", log->episode_return_offroad);
    assign_to_dict(dict, "episode_return_drive", log->episode_return_drive);
    assign_to_dict(dict, "episode_return_adversarial", log->episode_return_adversarial);
    assign_to_dict(dict, "mean_reward", log->episode_return / safe_episode_length);
    assign_to_dict(dict, "mean_reward_collision", log->episode_return_collision / safe_episode_length);
    assign_to_dict(dict, "mean_reward_offroad", log->episode_return_offroad / safe_episode_length);
    assign_to_dict(dict, "mean_reward_drive", log->episode_return_drive / safe_episode_length);
    assign_to_dict(dict, "mean_reward_adversarial", log->episode_return_adversarial / safe_episode_length);
    assign_to_dict(dict, "adversaries_episode_length", log->episode_length);
    assign_to_dict(dict, "adversaries_episode_return", log->episode_return);
    assign_to_dict(dict, "adversaries_episode_return_collision", log->episode_return_collision);
    assign_to_dict(dict, "adversaries_episode_return_offroad", log->episode_return_offroad);
    assign_to_dict(dict, "adversaries_episode_return_drive", log->episode_return_drive);
    assign_to_dict(dict, "adversaries_episode_return_adversarial", log->episode_return_adversarial);
    assign_to_dict(dict, "adversaries_mean_reward", log->episode_return / safe_episode_length);
    assign_to_dict(dict, "adversaries_mean_reward_collision", log->episode_return_collision / safe_episode_length);
    assign_to_dict(dict, "adversaries_mean_reward_offroad", log->episode_return_offroad / safe_episode_length);
    assign_to_dict(dict, "adversaries_mean_reward_drive", log->episode_return_drive / safe_episode_length);
    assign_to_dict(dict, "adversaries_mean_reward_adversarial", log->episode_return_adversarial / safe_episode_length);
    assign_to_dict(dict, "target_episode_return", log->target_episode_return);
    assign_to_dict(dict, "target_episode_return_collision", log->target_episode_return_collision);
    assign_to_dict(dict, "target_episode_return_offroad", log->target_episode_return_offroad);
    assign_to_dict(dict, "target_episode_return_drive", log->target_episode_return_drive);
    assign_to_dict(dict, "target_mean_reward", log->target_episode_return / safe_target_episode_length);
    assign_to_dict(dict, "target_mean_reward_collision",
                   log->target_episode_return_collision / safe_target_episode_length);
    assign_to_dict(dict, "target_mean_reward_offroad", log->target_episode_return_offroad / safe_target_episode_length);
    assign_to_dict(dict, "target_mean_reward_drive", log->target_episode_return_drive / safe_target_episode_length);
    assign_to_dict(dict, "did_target_collide", log->did_target_collide);
    assign_to_dict(dict, "did_target_offroad", log->did_target_offroad);
    assign_to_dict(dict, "did_target_run_light", log->did_target_run_light);
    assign_to_dict(dict, "did_target_fail", log->did_target_fail);
    assign_to_dict(dict, "did_target_make_progress", log->did_target_make_progress);
    assign_to_dict(dict, "did_target_have_at_fault_collision", log->did_target_have_at_fault_collision);
    assign_to_dict(dict, "did_target_have_unavoidable_collision", log->did_target_have_unavoidable_collision);
    assign_to_dict(dict, "adversaries_offroad_rate", log->adversaries_offroad_rate);
    assign_to_dict(dict, "adversaries_collision_rate", log->adversaries_collision_rate);
    assign_to_dict(dict, "adversaries_target_collision_rate", log->adversaries_target_collision_rate);
    assign_to_dict(dict, "adversaries_adversary_collision_rate", log->adversaries_adversary_collision_rate);
    assign_to_dict(dict, "adversaries_red_light_violation_rate", log->adversaries_red_light_violation_rate);
    assign_to_dict(dict, "adversaries_at_fault_collision_rate", log->adversaries_at_fault_collision_rate);
    assign_to_dict(dict, "adversaries_making_progress_rate", log->adversaries_making_progress_rate);
    assign_to_dict(dict, "adversaries_comfort_score", log->adversaries_comfort_score);
    assign_to_dict(dict, "adversaries_comfort_violations_per_agent_timestep", log->adversaries_comfort_violation_rate);
    assign_to_dict(dict, "adversaries_comfort_longitudinal_accel_violations_per_timestep",
                   log->adversaries_comfort_longitudinal_accel_violations_per_timestep);
    assign_to_dict(dict, "adversaries_comfort_lateral_accel_violations_per_timestep",
                   log->adversaries_comfort_lateral_accel_violations_per_timestep);
    assign_to_dict(dict, "adversaries_comfort_jerk_violations_per_timestep",
                   log->adversaries_comfort_jerk_violations_per_timestep);
    assign_to_dict(dict, "adversaries_uncomfortable_timestep_rate", log->adversaries_uncomfortable_timestep_rate);
    assign_to_dict(dict, "adversaries_avg_speed", log->adversaries_avg_speed);
    assign_to_dict(dict, "adversaries_lane_center_rate", log->adversaries_lane_center_rate);
    assign_to_dict(dict, "adversaries_lane_heading_aligned_rate", log->adversaries_lane_heading_aligned_rate);
    assign_to_dict(dict, "adversaries_velocity_progress", log->adversaries_velocity_progress);
    assign_to_dict(dict, "adversaries_speed_limit_compliance", log->adversaries_speed_limit_compliance);
    assign_to_dict(dict, "adversaries_driving_direction_score", log->adversaries_driving_direction_score);
    assign_to_dict(dict, "adversaries_multi_lane_time", log->adversaries_multi_lane_time);
    assign_to_dict(dict, "adversaries_multi_lane_score", log->adversaries_multi_lane_score);
    assign_to_dict(dict, "adversaries_dnf_rate", log->adversaries_dnf_rate);
    assign_to_dict(dict, "adversaries_score", log->adversaries_score);
    assign_to_dict(dict, "adversaries_num_waypoints_reached", log->adversaries_num_waypoints_reached);
    assign_to_dict(dict, "target_num_goals_reached", log->target_num_goals_reached);
    assign_to_dict(dict, "target_ttc_within_bound_rate", log->target_ttc_within_bound_rate);
    assign_to_dict(dict, "target_progress_ratio", log->target_progress_ratio);
    assign_to_dict(dict, "target_puffer_score", log->target_puffer_score);
    assign_to_dict(dict, "target_comfort_violations_per_timestep", log->target_comfort_violations_per_timestep);
    assign_to_dict(dict, "target_comfort_longitudinal_accel_violations_per_timestep",
                   log->target_comfort_longitudinal_accel_violations_per_timestep);
    assign_to_dict(dict, "target_comfort_lateral_accel_violations_per_timestep",
                   log->target_comfort_lateral_accel_violations_per_timestep);
    assign_to_dict(dict, "target_comfort_jerk_violations_per_timestep",
                   log->target_comfort_jerk_violations_per_timestep);
    assign_to_dict(dict, "target_uncomfortable_timestep_rate", log->target_uncomfortable_timestep_rate);
    assign_to_dict(dict, "target_collision_severity", log->target_collision_severity);
    assign_to_dict(dict, "target_collision_responsibility", log->target_collision_responsibility);
    assign_to_dict(dict, "target_collision_impact_zone", log->target_collision_impact_zone);
    assign_to_dict(dict, "target_collision_type", log->target_collision_type);
    assign_to_dict(dict, "target_collision_target_speed", log->target_collision_target_speed);
    assign_to_dict(dict, "target_collision_other_speed", log->target_collision_other_speed);
    assign_to_dict(dict, "target_collision_relative_speed", log->target_collision_relative_speed);
    assign_to_dict(dict, "target_collision_other_active", log->target_collision_other_active);
    assign_to_dict(dict, "target_collision_other_stopped", log->target_collision_other_stopped);
    assign_to_dict(dict, "target_collision_other_removed", log->target_collision_other_removed);
    assign_to_dict(dict, "target_collision_unavoidable_rate", log->target_collision_unavoidable_rate);
    assign_to_dict(dict, "target_collision_adversary_forced_rate", log->target_collision_adversary_forced_rate);
    assign_to_dict(dict, "target_collision_target_failure_rate", log->target_collision_target_failure_rate);
    assign_to_dict(dict, "target_mean_last_avoidable_braking_seconds_before_collision",
                   log->target_last_avoidable_braking_seconds_before_collision);
    // Concise per-episode aliases used by paper tables and compact replay indexes.
    assign_to_dict(dict, "t_brake", log->target_last_avoidable_braking_seconds_before_collision);
    assign_to_dict(dict, "genuine_target_failure", log->target_collision_target_failure_rate);
    assign_to_dict(dict, "adversary_forced", log->target_collision_adversary_forced_rate);
    assign_to_dict(dict, "unavoidable", log->target_collision_unavoidable_rate);
    assign_to_dict(dict, "adversaries_collision_severity", log->adversaries_collision_severity);
    assign_to_dict(dict, "adversaries_collision_responsibility", log->adversaries_collision_responsibility);
    assign_to_dict(dict, "adversaries_collision_impact_zone", log->adversaries_collision_impact_zone);
    assign_to_dict(dict, "target_hit_responsibility", log->target_hit_responsibility);
    assign_to_dict(dict, "target_hit_low_responsibility_rate", log->target_hit_low_responsibility_rate);
    assign_to_dict(dict, "target_hit_at_fault_rate", log->target_hit_at_fault_rate);
    assign_to_dict(dict, "red_light_violation_rate", log->red_light_violation_rate);
    assign_to_dict(dict, "comfort_violation_count", log->comfort_violation_count);
    // assign_to_dict(dict, "avg_displacement_error", log->avg_displacement_error);
    assign_to_dict(dict, "velocity_progress_sum", log->velocity_progress_sum);
    assign_to_dict(dict, "num_goals_reached", log->num_goals_reached);
    assign_to_dict(dict, "adversaries_num_goals_reached", log->num_goals_reached);
    assign_to_dict(dict, "lane_center_rate", log->lane_center_rate);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "avg_speed_per_agent", log->avg_speed_per_agent);
    assign_to_dict(dict, "avg_distance_per_infraction", avg_distance_per_infraction);

    if (env->compute_eval_metrics) {
        // Puffer score components
        assign_to_dict(dict, "at_fault_collision_rate", log->at_fault_collision_rate);
        assign_to_dict(dict, "puffer_score", log->puffer_score);
        assign_to_dict(dict, "ttc_within_bound_rate", log->ttc_within_bound_rate);
        assign_to_dict(dict, "adversaries_ttc_within_bound_rate", log->ttc_within_bound_rate);
        assign_to_dict(dict, "driving_direction_score", log->driving_direction_score);
        assign_to_dict(dict, "speed_limit_compliance", log->speed_limit_compliance);
        assign_to_dict(dict, "making_progress_rate", log->making_progress_rate);
        assign_to_dict(dict, "progress_ratio", log->progress_ratio);
        assign_to_dict(dict, "adversaries_progress_ratio", log->progress_ratio);
        assign_to_dict(dict, "comfort_score", log->comfort_score);
        assign_to_dict(dict, "adversaries_puffer_score", log->puffer_score);
        assign_to_dict(dict, "multi_lane_time", log->multi_lane_time);
        assign_to_dict(dict, "multi_lane_score", log->multi_lane_score);
    }

    return 0;
}
