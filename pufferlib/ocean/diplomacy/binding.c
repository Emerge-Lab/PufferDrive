#include <Python.h>
#include <numpy/arrayobject.h>
#include "diplomacy.h"

// Define MY_INIT macro before including env_binding.h
#define MY_INIT

// Implementation of environment-specific init function
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    c_init(env);
    return 0;
}

// Stub for log function (required by env_binding.h)
static int my_log(PyObject* dict, Log* log) {
    // TODO: Implement logging
    return 0;
}

// Include the PufferLib environment binding template
#include "../env_binding.h"
