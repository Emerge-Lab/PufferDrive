// Python module `drive_render`: the wheel demo's window, kept in its own shared object so that
// pufferlib.ocean.drive.binding never links raylib. It takes the SAME integer env handle the sim
// binding hands out (PyLong_FromVoidPtr), so no state is duplicated between the two modules.
#define PY_SSIZE_T_CLEAN
#include "pufferlib/ocean/drive/render/render.h"

#include <Python.h>
#include <stdlib.h>

// Mirrors VecEnv in env_binding.h. Two fields, so there is no layout ambiguity to get wrong.
typedef struct {
    Drive **envs;
    int num_envs;
} RenderVecEnv;

static float *g_pose_x = NULL;
static float *g_pose_y = NULL;
static float *g_pose_heading = NULL;
static int g_pose_capacity = 0;

static Drive *unpack_first_env(PyObject *handle_obj) {
    if (!PyLong_Check(handle_obj)) {
        PyErr_SetString(PyExc_TypeError, "env handle must be the integer returned by binding.vectorize");
        return NULL;
    }
    RenderVecEnv *vec = (RenderVecEnv *) PyLong_AsVoidPtr(handle_obj);
    if (vec == NULL || vec->num_envs <= 0) {
        PyErr_SetString(PyExc_ValueError, "invalid env handle");
        return NULL;
    }
    return vec->envs[0];
}

static int ensure_pose_capacity(int count) {
    if (count <= g_pose_capacity) {
        return 0;
    }
    // Each pointer is adopted as soon as its realloc succeeds: realloc frees the old block on
    // success, so deferring the assignment would lose it if a later one failed.
    size_t bytes = (size_t) count * sizeof(float);
    float *grown = (float *) realloc(g_pose_x, bytes);
    if (grown == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    g_pose_x = grown;
    grown = (float *) realloc(g_pose_y, bytes);
    if (grown == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    g_pose_y = grown;
    grown = (float *) realloc(g_pose_heading, bytes);
    if (grown == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    g_pose_heading = grown;
    g_pose_capacity = count;
    return 0;
}

static PyObject *render_init(PyObject *self, PyObject *args) {
    const char *resource_dir;
    int width, height, camera_mode, night_mode;
    if (!PyArg_ParseTuple(args, "siiii", &resource_dir, &width, &height, &camera_mode, &night_mode)) {
        return NULL;
    }
    if (demo_render_init(resource_dir, width, height, camera_mode, night_mode) != 0) {
        PyErr_Format(PyExc_RuntimeError, "could not open the demo window or load car models from %s", resource_dir);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *render_frame(PyObject *self, PyObject *args) {
    PyObject *handle_obj;
    double elapsed_seconds;
    if (!PyArg_ParseTuple(args, "Od", &handle_obj, &elapsed_seconds)) {
        return NULL;
    }
    Drive *env = unpack_first_env(handle_obj);
    if (env == NULL) {
        return NULL;
    }
    if (ensure_pose_capacity(env->num_agents) != 0) {
        return NULL;
    }
    c_get_render_poses(env, (float) elapsed_seconds, g_pose_x, g_pose_y, g_pose_heading);
    demo_render_frame(env, g_pose_x, g_pose_y, g_pose_heading);
    Py_RETURN_NONE;
}

static PyObject *render_set_camera_mode(PyObject *self, PyObject *args) {
    int camera_mode;
    if (!PyArg_ParseTuple(args, "i", &camera_mode)) {
        return NULL;
    }
    if (camera_mode < 0 || camera_mode >= CAMERA_MODE_COUNT) {
        PyErr_Format(PyExc_ValueError, "camera_mode %d outside [0, %d)", camera_mode, CAMERA_MODE_COUNT);
        return NULL;
    }
    demo_render_set_camera_mode(camera_mode);
    Py_RETURN_NONE;
}

static PyObject *render_set_night_mode(PyObject *self, PyObject *args) {
    int night_mode;
    if (!PyArg_ParseTuple(args, "i", &night_mode)) {
        return NULL;
    }
    demo_render_set_night_mode(night_mode);
    Py_RETURN_NONE;
}

static PyObject *render_should_close(PyObject *self, PyObject *args) {
    return PyBool_FromLong(demo_render_should_close());
}

// Edge-triggered keys, so the host sees one event per press rather than one per frame.
static PyObject *render_poll_keys(PyObject *self, PyObject *args) {
    if (!g_renderer.window_open) {
        return Py_BuildValue("{}");
    }
    return Py_BuildValue(
        "{s:O,s:O,s:O,s:O,s:O,s:O}",
        "camera_chase",
        IsKeyPressed(KEY_ONE) ? Py_True : Py_False,
        "camera_driver",
        IsKeyPressed(KEY_TWO) ? Py_True : Py_False,
        "camera_agent",
        IsKeyPressed(KEY_THREE) ? Py_True : Py_False,
        "toggle_night",
        IsKeyPressed(KEY_N) ? Py_True : Py_False,
        "next_agent",
        IsKeyPressed(KEY_TAB) ? Py_True : Py_False,
        "restart_engine",
        IsKeyPressed(KEY_R) ? Py_True : Py_False);
}

static PyObject *render_close(PyObject *self, PyObject *args) {
    demo_render_close();
    free(g_pose_x);
    free(g_pose_y);
    free(g_pose_heading);
    g_pose_x = g_pose_y = g_pose_heading = NULL;
    g_pose_capacity = 0;
    Py_RETURN_NONE;
}

static PyMethodDef render_methods[]
    = {{"init", render_init, METH_VARARGS, "Open the demo window and load car models"},
       {"frame", render_frame, METH_VARARGS, "Draw one frame for an env handle"},
       {"set_camera_mode", render_set_camera_mode, METH_VARARGS, "0 chase, 1 driver, 2 agent"},
       {"set_night_mode", render_set_night_mode, METH_VARARGS, "Switch between day and night"},
       {"should_close", render_should_close, METH_VARARGS, "True once the window wants to close"},
       {"poll_keys", render_poll_keys, METH_VARARGS, "Edge-triggered key events as a dict"},
       {"close", render_close, METH_VARARGS, "Close the window and release models"},
       {NULL, NULL, 0, NULL}};

static struct PyModuleDef render_module
    = {PyModuleDef_HEAD_INIT,
       "drive_render",
       "Wheel demo renderer (raylib)",
       -1,
       render_methods,
       NULL,
       NULL,
       NULL,
       NULL};

PyMODINIT_FUNC PyInit_drive_render(void) {
    PyObject *module = PyModule_Create(&render_module);
    if (module == NULL) {
        return NULL;
    }
    PyModule_AddIntConstant(module, "CAMERA_MODE_CHASE", CAMERA_MODE_CHASE);
    PyModule_AddIntConstant(module, "CAMERA_MODE_DRIVER", CAMERA_MODE_DRIVER);
    PyModule_AddIntConstant(module, "CAMERA_MODE_AGENT", CAMERA_MODE_AGENT);
    PyModule_AddIntConstant(module, "CAMERA_MODE_COUNT", CAMERA_MODE_COUNT);
    return module;
}
