/*
 * _native.c — CPython extension shell for the trajviz Vulkan renderer.
 *
 * Exposes three Python functions:
 *
 *   init(width: int, height: int) -> capsule
 *       Creates a TrajvizCtx and returns it wrapped in a PyCapsule with
 *       a destructor that calls trajviz_close. The capsule can be passed
 *       to render_episode any number of times.
 *
 *   render_episode(ctx, road_xy, road_offsets, road_types, traj_xyh,
 *                  agent_dims, agent_lengths, ego_idx, fps,
 *                  out_topdown, out_bev) -> int
 *       Validates the numpy arrays, releases the GIL, calls
 *       trajviz_render_episode, reacquires the GIL. Raises RuntimeError
 *       on a non-zero return code with the underlying error message.
 *
 *   close(ctx) -> None
 *       Manually destroy the ctx. Optional — the capsule destructor
 *       will also do it on garbage collection.
 *
 * Numpy arrays are validated for dtype, ndim, and contiguity. Each is
 * coerced to its expected dtype and C-contiguous layout via
 * PyArray_FROMANY (which is a no-op for already-conforming arrays). The
 * resulting reference is held for the duration of the call so the data
 * pointer stays valid.
 *
 * The GIL release pattern: numpy unwrapping happens with the GIL held
 * (necessary), then we release for the trajviz_render_episode call,
 * which is the long-running part. Other Python threads can run during
 * the GPU + ffmpeg work.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "trajviz.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* PyCapsule name — used by PyCapsule_GetPointer to validate the type. */
static const char CAPSULE_NAME[] = "trajviz._native.TrajvizCtx";

/* We track "already closed" state in the capsule's context slot (not in
 * the wrapped pointer, because PyCapsule_SetPointer rejects NULL and
 * because the pointer points at freed memory after close). A non-NULL
 * context = closed, NULL context = open. */
#define CLOSED_SENTINEL ((void *)(uintptr_t)1)

static void capsule_destructor(PyObject *capsule) {
    if (PyCapsule_GetContext(capsule) == CLOSED_SENTINEL)
        return;
    TrajvizCtx *ctx = (TrajvizCtx *)PyCapsule_GetPointer(capsule, CAPSULE_NAME);
    if (ctx)
        trajviz_close(ctx);
}

/* ---------------------------- init / close ---------------------------- */

static PyObject *py_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"width", "height", NULL};
    int width = 0, height = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:init", kwlist, &width, &height)) {
        return NULL;
    }
    TrajvizCtx *ctx = trajviz_init(width, height);
    if (!ctx) {
        PyErr_Format(PyExc_RuntimeError, "trajviz_init failed: %s", trajviz_last_error(NULL));
        return NULL;
    }
    PyObject *capsule = PyCapsule_New(ctx, CAPSULE_NAME, capsule_destructor);
    if (!capsule) {
        trajviz_close(ctx);
        return NULL;
    }
    return capsule;
}

static PyObject *py_close(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O:close", &capsule))
        return NULL;
    if (!PyCapsule_CheckExact(capsule)) {
        PyErr_SetString(PyExc_TypeError, "expected a TrajvizCtx capsule");
        return NULL;
    }
    if (PyCapsule_GetContext(capsule) == CLOSED_SENTINEL) {
        Py_RETURN_NONE; /* already closed */
    }
    TrajvizCtx *ctx = (TrajvizCtx *)PyCapsule_GetPointer(capsule, CAPSULE_NAME);
    if (!ctx)
        Py_RETURN_NONE;
    trajviz_close(ctx);
    PyCapsule_SetContext(capsule, CLOSED_SENTINEL);
    Py_RETURN_NONE;
}

/* ----------------------- numpy validation helpers ----------------------- */

/* Coerce a Python object to a contiguous numpy array of the given dtype
 * and exact ndim. Returns a NEW reference (caller must DECREF). On
 * failure raises a Python exception and returns NULL.
 *
 * If allow_none is non-zero and obj is Py_None, returns NULL with no
 * exception set (caller checks).
 */
static PyArrayObject *as_array(PyObject *obj, int dtype, int ndim, int allow_none, const char *name) {
    if (allow_none && obj == Py_None)
        return NULL;
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROMANY(obj, dtype, ndim, ndim, NPY_ARRAY_C_CONTIGUOUS);
    if (!arr) {
        if (PyErr_Occurred()) {
            /* PyArray_FROMANY already set a reasonable error message; we
             * just prepend the argument name for clarity. */
            PyObject *type, *value, *tb;
            PyErr_Fetch(&type, &value, &tb);
            PyErr_Format(PyExc_TypeError, "%s: %s", name,
                         value ? PyUnicode_AsUTF8(PyObject_Str(value)) : "type/shape mismatch");
            Py_XDECREF(type);
            Py_XDECREF(value);
            Py_XDECREF(tb);
        }
        return NULL;
    }
    return arr;
}

/* --------------------------- render_episode --------------------------- */

static PyObject *py_render_episode(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"ctx",           "road_xy", "road_offsets", "road_types",  "traj_xyh", "agent_dims",
                             "agent_lengths", "ego_idx", "fps",          "out_topdown", "out_bev",  NULL};

    PyObject *capsule = NULL;
    PyObject *o_road_xy, *o_road_off, *o_road_types;
    PyObject *o_traj, *o_dims, *o_lens;
    int ego_idx = -1;
    int fps = 30;
    const char *out_topdown = NULL;
    const char *out_bev = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOOiizz:render_episode", kwlist, &capsule, &o_road_xy,
                                     &o_road_off, &o_road_types, &o_traj, &o_dims, &o_lens, &ego_idx, &fps,
                                     &out_topdown, &out_bev)) {
        return NULL;
    }

    if (!PyCapsule_CheckExact(capsule)) {
        PyErr_SetString(PyExc_TypeError, "ctx: expected a TrajvizCtx capsule");
        return NULL;
    }
    if (PyCapsule_GetContext(capsule) == CLOSED_SENTINEL) {
        PyErr_SetString(PyExc_RuntimeError, "ctx has been closed");
        return NULL;
    }
    TrajvizCtx *ctx = (TrajvizCtx *)PyCapsule_GetPointer(capsule, CAPSULE_NAME);
    if (!ctx) {
        PyErr_SetString(PyExc_RuntimeError, "ctx is null");
        return NULL;
    }

    PyArrayObject *a_xy = as_array(o_road_xy, NPY_FLOAT32, 2, 0, "road_xy");
    PyArrayObject *a_off = as_array(o_road_off, NPY_UINT32, 1, 0, "road_offsets");
    PyArrayObject *a_typ = as_array(o_road_types, NPY_UINT32, 1, 0, "road_types");
    PyArrayObject *a_traj = as_array(o_traj, NPY_FLOAT32, 3, 0, "traj_xyh");
    PyArrayObject *a_dims = as_array(o_dims, NPY_FLOAT32, 2, 1, "agent_dims");
    PyArrayObject *a_lens = as_array(o_lens, NPY_INT32, 1, 1, "agent_lengths");

    if (!a_xy || !a_off || !a_typ || !a_traj)
        goto fail;

    /* Shape checks. */
    if (PyArray_DIM(a_xy, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "road_xy must have shape (N, 2)");
        goto fail;
    }
    if (PyArray_DIM(a_traj, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "traj_xyh must have shape (T, A, 3)");
        goto fail;
    }
    npy_intp num_steps = PyArray_DIM(a_traj, 0);
    npy_intp num_agents = PyArray_DIM(a_traj, 1);

    if (a_dims && (PyArray_DIM(a_dims, 0) != num_agents || PyArray_DIM(a_dims, 1) != 2)) {
        PyErr_Format(PyExc_ValueError, "agent_dims must have shape (%ld, 2)", (long)num_agents);
        goto fail;
    }
    if (a_lens && PyArray_DIM(a_lens, 0) != num_agents) {
        PyErr_Format(PyExc_ValueError, "agent_lengths must have shape (%ld,)", (long)num_agents);
        goto fail;
    }

    npy_intp num_polys = PyArray_DIM(a_typ, 0);
    if (PyArray_DIM(a_off, 0) != num_polys + 1) {
        PyErr_Format(PyExc_ValueError, "road_offsets must have shape (num_polys+1=%ld,), got (%ld,)",
                     (long)(num_polys + 1), (long)PyArray_DIM(a_off, 0));
        goto fail;
    }

    /* Pull raw pointers. */
    const float *road_xy = (const float *)PyArray_DATA(a_xy);
    const uint32_t *road_offsets = (const uint32_t *)PyArray_DATA(a_off);
    const uint32_t *road_types = (const uint32_t *)PyArray_DATA(a_typ);
    const float *traj_xyh = (const float *)PyArray_DATA(a_traj);
    const float *agent_dims_p = a_dims ? (const float *)PyArray_DATA(a_dims) : NULL;
    const int32_t *agent_lens_p = a_lens ? (const int32_t *)PyArray_DATA(a_lens) : NULL;

    int rc;
    Py_BEGIN_ALLOW_THREADS rc = trajviz_render_episode(
        ctx, road_xy, road_offsets, road_types, (uint32_t)num_polys, traj_xyh, (uint32_t)num_steps,
        (uint32_t)num_agents, agent_dims_p, agent_lens_p, (int32_t)ego_idx, fps, out_topdown, out_bev);
    Py_END_ALLOW_THREADS

        Py_XDECREF(a_xy);
    Py_XDECREF(a_off);
    Py_XDECREF(a_typ);
    Py_XDECREF(a_traj);
    Py_XDECREF(a_dims);
    Py_XDECREF(a_lens);

    if (rc != TRAJVIZ_OK) {
        PyErr_Format(PyExc_RuntimeError, "trajviz_render_episode failed (%d): %s", rc, trajviz_last_error(ctx));
        return NULL;
    }
    Py_RETURN_NONE;

fail:
    Py_XDECREF(a_xy);
    Py_XDECREF(a_off);
    Py_XDECREF(a_typ);
    Py_XDECREF(a_traj);
    Py_XDECREF(a_dims);
    Py_XDECREF(a_lens);
    return NULL;
}

/* --------------------------- render_episodes_batch --------------------------- */

/* Python signature:
 *
 *   render_episodes_batch(
 *       ctx,
 *       all_road_xy,         # (V_total, 2)            float32
 *       vert_offsets,        # (batch_size+1,)         uint32
 *       all_road_offsets,    # (P_meta_total,)         uint32
 *       poly_meta_offsets,   # (batch_size+1,)         uint32
 *       all_road_types,      # (P_total,)              uint32
 *       poly_type_offsets,   # (batch_size+1,)         uint32
 *       traj_xyh,            # (batch, T, A, 3)        float32
 *       agent_lengths,       # (batch, A)              int32
 *       ego_idx_per_ep,      # (batch,)                int32
 *       fps,                 # int
 *       out_topdown_paths,   # list of str or None, len batch
 *       out_bev_paths,       # list of str or None, len batch
 *   ) -> None
 *
 * Returns None on success; raises RuntimeError with the C-side error
 * message on failure.
 */
static PyObject *py_render_episodes_batch(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"ctx",
                             "all_road_xy",
                             "vert_offsets",
                             "all_road_offsets",
                             "poly_meta_offsets",
                             "all_road_types",
                             "poly_type_offsets",
                             "traj_xyh",
                             "agent_lengths",
                             "ego_idx_per_ep",
                             "fps",
                             "out_topdown_paths",
                             "out_bev_paths",
                             NULL};
    PyObject *capsule = NULL;
    PyObject *o_xy, *o_voff, *o_roff, *o_pmoff, *o_rtypes, *o_ptoff;
    PyObject *o_traj, *o_lens, *o_egos;
    int fps = 30;
    PyObject *o_td_paths = NULL, *o_bev_paths = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOOOOOiOO:render_episodes_batch", kwlist, &capsule, &o_xy,
                                     &o_voff, &o_roff, &o_pmoff, &o_rtypes, &o_ptoff, &o_traj, &o_lens, &o_egos, &fps,
                                     &o_td_paths, &o_bev_paths)) {
        return NULL;
    }

    if (!PyCapsule_CheckExact(capsule)) {
        PyErr_SetString(PyExc_TypeError, "ctx: expected a TrajvizCtx capsule");
        return NULL;
    }
    if (PyCapsule_GetContext(capsule) == CLOSED_SENTINEL) {
        PyErr_SetString(PyExc_RuntimeError, "ctx has been closed");
        return NULL;
    }
    TrajvizCtx *ctx = (TrajvizCtx *)PyCapsule_GetPointer(capsule, CAPSULE_NAME);
    if (!ctx) {
        PyErr_SetString(PyExc_RuntimeError, "ctx is null");
        return NULL;
    }

    PyArrayObject *a_xy = as_array(o_xy, NPY_FLOAT32, 2, 0, "all_road_xy");
    PyArrayObject *a_voff = as_array(o_voff, NPY_UINT32, 1, 0, "vert_offsets");
    PyArrayObject *a_roff = as_array(o_roff, NPY_UINT32, 1, 0, "all_road_offsets");
    PyArrayObject *a_pmoff = as_array(o_pmoff, NPY_UINT32, 1, 0, "poly_meta_offsets");
    PyArrayObject *a_rtyp = as_array(o_rtypes, NPY_UINT32, 1, 0, "all_road_types");
    PyArrayObject *a_ptoff = as_array(o_ptoff, NPY_UINT32, 1, 0, "poly_type_offsets");
    PyArrayObject *a_traj = as_array(o_traj, NPY_FLOAT32, 4, 0, "traj_xyh");
    PyArrayObject *a_lens = as_array(o_lens, NPY_INT32, 2, 0, "agent_lengths");
    PyArrayObject *a_egos = as_array(o_egos, NPY_INT32, 1, 0, "ego_idx_per_ep");

    if (!a_xy || !a_voff || !a_roff || !a_pmoff || !a_rtyp || !a_ptoff || !a_traj || !a_lens || !a_egos)
        goto fail;

    if (PyArray_DIM(a_xy, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "all_road_xy must have shape (V, 2)");
        goto fail;
    }
    if (PyArray_DIM(a_traj, 3) != 3) {
        PyErr_SetString(PyExc_ValueError, "traj_xyh must have shape (batch, T, A, 3)");
        goto fail;
    }

    npy_intp batch_size = PyArray_DIM(a_traj, 0);
    npy_intp num_steps = PyArray_DIM(a_traj, 1);
    npy_intp max_agents = PyArray_DIM(a_traj, 2);

    if (PyArray_DIM(a_lens, 0) != batch_size || PyArray_DIM(a_lens, 1) != max_agents) {
        PyErr_Format(PyExc_ValueError, "agent_lengths must have shape (%ld, %ld)", (long)batch_size, (long)max_agents);
        goto fail;
    }
    if (PyArray_DIM(a_egos, 0) != batch_size) {
        PyErr_Format(PyExc_ValueError, "ego_idx_per_ep must have shape (%ld,)", (long)batch_size);
        goto fail;
    }
    if (PyArray_DIM(a_voff, 0) != batch_size + 1 || PyArray_DIM(a_pmoff, 0) != batch_size + 1 ||
        PyArray_DIM(a_ptoff, 0) != batch_size + 1) {
        PyErr_Format(PyExc_ValueError,
                     "vert_offsets / poly_meta_offsets / poly_type_offsets must have "
                     "shape (batch_size+1=%ld,)",
                     (long)(batch_size + 1));
        goto fail;
    }

    /* Output path arrays. Each list (or None) must have length == batch_size.
     * We allocate a C array of const char* per list and copy the strings'
     * pointers from PyUnicode_AsUTF8. Strings stay valid for the duration
     * of this call because we hold the Python lists. */
    if (o_td_paths != Py_None && !PyList_Check(o_td_paths)) {
        PyErr_SetString(PyExc_TypeError, "out_topdown_paths must be a list or None");
        goto fail;
    }
    if (o_bev_paths != Py_None && !PyList_Check(o_bev_paths)) {
        PyErr_SetString(PyExc_TypeError, "out_bev_paths must be a list or None");
        goto fail;
    }
    if (o_td_paths != Py_None && PyList_GET_SIZE(o_td_paths) != batch_size) {
        PyErr_Format(PyExc_ValueError, "out_topdown_paths length %zd != batch_size %ld", PyList_GET_SIZE(o_td_paths),
                     (long)batch_size);
        goto fail;
    }
    if (o_bev_paths != Py_None && PyList_GET_SIZE(o_bev_paths) != batch_size) {
        PyErr_Format(PyExc_ValueError, "out_bev_paths length %zd != batch_size %ld", PyList_GET_SIZE(o_bev_paths),
                     (long)batch_size);
        goto fail;
    }

    const char **td_arr = NULL;
    const char **bev_arr = NULL;
    td_arr = (const char **)calloc((size_t)batch_size, sizeof(const char *));
    bev_arr = (const char **)calloc((size_t)batch_size, sizeof(const char *));
    if (!td_arr || !bev_arr) {
        PyErr_NoMemory();
        free(td_arr);
        free(bev_arr);
        goto fail;
    }
    if (o_td_paths != Py_None) {
        for (npy_intp i = 0; i < batch_size; ++i) {
            PyObject *item = PyList_GET_ITEM(o_td_paths, i);
            if (item == Py_None) {
                td_arr[i] = NULL;
            } else if (PyUnicode_Check(item)) {
                td_arr[i] = PyUnicode_AsUTF8(item);
                if (!td_arr[i]) {
                    free(td_arr);
                    free(bev_arr);
                    goto fail;
                }
            } else {
                PyErr_Format(PyExc_TypeError, "out_topdown_paths[%zd] must be str or None", i);
                free(td_arr);
                free(bev_arr);
                goto fail;
            }
        }
    }
    if (o_bev_paths != Py_None) {
        for (npy_intp i = 0; i < batch_size; ++i) {
            PyObject *item = PyList_GET_ITEM(o_bev_paths, i);
            if (item == Py_None) {
                bev_arr[i] = NULL;
            } else if (PyUnicode_Check(item)) {
                bev_arr[i] = PyUnicode_AsUTF8(item);
                if (!bev_arr[i]) {
                    free(td_arr);
                    free(bev_arr);
                    goto fail;
                }
            } else {
                PyErr_Format(PyExc_TypeError, "out_bev_paths[%zd] must be str or None", i);
                free(td_arr);
                free(bev_arr);
                goto fail;
            }
        }
    }

    int rc;
    Py_BEGIN_ALLOW_THREADS rc = trajviz_render_episodes_batch(
        ctx, (int)batch_size, (uint32_t)num_steps, (uint32_t)max_agents, (const float *)PyArray_DATA(a_xy),
        (const uint32_t *)PyArray_DATA(a_voff), (const uint32_t *)PyArray_DATA(a_roff),
        (const uint32_t *)PyArray_DATA(a_pmoff), (const uint32_t *)PyArray_DATA(a_rtyp),
        (const uint32_t *)PyArray_DATA(a_ptoff), (const float *)PyArray_DATA(a_traj),
        (const int32_t *)PyArray_DATA(a_lens), (const int32_t *)PyArray_DATA(a_egos), fps, td_arr, bev_arr);
    Py_END_ALLOW_THREADS

        free(td_arr);
    free(bev_arr);

    Py_XDECREF(a_xy);
    Py_XDECREF(a_voff);
    Py_XDECREF(a_roff);
    Py_XDECREF(a_pmoff);
    Py_XDECREF(a_rtyp);
    Py_XDECREF(a_ptoff);
    Py_XDECREF(a_traj);
    Py_XDECREF(a_lens);
    Py_XDECREF(a_egos);

    if (rc != TRAJVIZ_OK) {
        PyErr_Format(PyExc_RuntimeError, "trajviz_render_episodes_batch failed (%d): %s", rc, trajviz_last_error(ctx));
        return NULL;
    }
    Py_RETURN_NONE;

fail:
    Py_XDECREF(a_xy);
    Py_XDECREF(a_voff);
    Py_XDECREF(a_roff);
    Py_XDECREF(a_pmoff);
    Py_XDECREF(a_rtyp);
    Py_XDECREF(a_ptoff);
    Py_XDECREF(a_traj);
    Py_XDECREF(a_lens);
    Py_XDECREF(a_egos);
    return NULL;
}

/* ----------------------------- module def ----------------------------- */

static PyMethodDef methods[] = {
    {"init", (PyCFunction)py_init, METH_VARARGS | METH_KEYWORDS, "init(width, height) -> capsule"},
    {"render_episode", (PyCFunction)py_render_episode, METH_VARARGS | METH_KEYWORDS,
     "render_episode(ctx, road_xy, road_offsets, road_types, traj_xyh, "
     "agent_dims, agent_lengths, ego_idx, fps, out_topdown, out_bev) -> None"},
    {"render_episodes_batch", (PyCFunction)py_render_episodes_batch, METH_VARARGS | METH_KEYWORDS,
     "render_episodes_batch(ctx, all_road_xy, vert_offsets, all_road_offsets, "
     "poly_meta_offsets, all_road_types, poly_type_offsets, traj_xyh, "
     "agent_lengths, ego_idx_per_ep, fps, out_topdown_paths, out_bev_paths) -> None"},
    {"close", (PyCFunction)py_close, METH_VARARGS, "close(ctx) -> None"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT,
                                       "trajviz._native",
                                       "Vulkan-backed renderer for saved Drive trajectories.",
                                       -1,
                                       methods,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL};

PyMODINIT_FUNC PyInit__native(void) {
    import_array(); /* numpy C API initialization — REQUIRED before any PyArray_* */
    if (PyErr_Occurred())
        return NULL;
    return PyModule_Create(&moduledef);
}
