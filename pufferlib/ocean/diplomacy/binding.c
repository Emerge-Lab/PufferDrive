#include <Python.h>
#include <numpy/arrayobject.h>
#include "diplomacy.h"

// Define MY_INIT macro before including env_binding.h
#define MY_INIT

// Implementation of environment-specific init function
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"welfare_mode", "max_years", NULL};
    int welfare_mode = 1;
    int max_years = 10;

    if (kwargs) {
        PyObject* wm = PyDict_GetItemString(kwargs, "welfare_mode");
        PyObject* my = PyDict_GetItemString(kwargs, "max_years");
        if (wm && PyLong_Check(wm)) {
            welfare_mode = (int)PyLong_AsLong(wm);
        }
        if (my && PyLong_Check(my)) {
            max_years = (int)PyLong_AsLong(my);
        }
    }

    c_init(env);
    c_configure(env, welfare_mode, max_years);
    return 0;
}

// Stub for log function (required by env_binding.h)
static int my_log(PyObject* dict, Log* log) {
    // TODO: Implement logging
    return 0;
}

// ============================================================================
// Custom Python Methods for Game State Queries
// ============================================================================

static PyObject* query_game_state(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }

    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or game not initialized");
        return NULL;
    }

    GameState* game = env->game;

    // Return dict with game state info
    PyObject* state_dict = PyDict_New();
    PyDict_SetItemString(state_dict, "year", PyLong_FromLong(get_current_year(game)));
    PyDict_SetItemString(state_dict, "phase", PyLong_FromLong((long)get_current_phase(game)));

    // Add power info
    PyObject* powers_list = PyList_New(MAX_POWERS);
    for (int p = 0; p < MAX_POWERS; p++) {
        PyObject* power_dict = PyDict_New();
        PyDict_SetItemString(power_dict, "num_units", PyLong_FromLong(get_num_units(game, p)));
        PyDict_SetItemString(power_dict, "num_centers", PyLong_FromLong(get_num_centers(game, p)));
        PyDict_SetItemString(power_dict, "welfare_points", PyLong_FromLong(get_welfare_points(game, p)));

        // Add unit locations
        PyObject* units_list = PyList_New(get_num_units(game, p));
        for (int u = 0; u < get_num_units(game, p); u++) {
            UnitType type;
            int location;
            get_unit_info(game, p, u, &type, &location);
            PyObject* unit_dict = PyDict_New();
            PyDict_SetItemString(unit_dict, "type", PyLong_FromLong((long)type));
            PyDict_SetItemString(unit_dict, "location", PyLong_FromLong(location));
            PyList_SetItem(units_list, u, unit_dict);
        }
        PyDict_SetItemString(power_dict, "units", units_list);

        // Add center locations
        int centers[MAX_UNITS];
        int num_centers;
        get_center_locations(game, p, centers, &num_centers);
        PyObject* centers_list = PyList_New(num_centers);
        for (int c = 0; c < num_centers; c++) {
            PyList_SetItem(centers_list, c, PyLong_FromLong(centers[c]));
        }
        PyDict_SetItemString(power_dict, "centers", centers_list);

        PyList_SetItem(powers_list, p, power_dict);
    }
    PyDict_SetItemString(state_dict, "powers", powers_list);

    return state_dict;
}

static PyObject* query_map_info(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }

    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game || !env->game->map) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or map not initialized");
        return NULL;
    }

    Map* map = env->game->map;

    PyObject* map_dict = PyDict_New();
    PyDict_SetItemString(map_dict, "num_locations", PyLong_FromLong(get_num_locations(map)));

    // Add location info
    PyObject* locations_list = PyList_New(get_num_locations(map));
    for (int i = 0; i < get_num_locations(map); i++) {
        PyObject* loc_dict = PyDict_New();
        PyDict_SetItemString(loc_dict, "name", PyUnicode_FromString(get_location_name(map, i)));
        PyDict_SetItemString(loc_dict, "type", PyLong_FromLong((long)get_location_type(map, i)));
        PyDict_SetItemString(loc_dict, "is_supply_center", PyLong_FromLong(is_supply_center(map, i)));
        PyList_SetItem(locations_list, i, loc_dict);
    }
    PyDict_SetItemString(map_dict, "locations", locations_list);

    return map_dict;
}

static PyObject* test_parse_order(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    const char* order_str;

    if (!PyArg_ParseTuple(args, "Os", &handle_obj, &order_str)) {
        return NULL;
    }

    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or game not initialized");
        return NULL;
    }

    Order order;
    int result = parse_order(order_str, &order, env->game);

    if (result < 0) {
        Py_RETURN_NONE;  // Parse failed
    }

    // Return parsed order as dict
    PyObject* order_dict = PyDict_New();
    PyDict_SetItemString(order_dict, "type", PyLong_FromLong((long)order.type));
    PyDict_SetItemString(order_dict, "unit_type", PyLong_FromLong((long)order.unit_type));
    PyDict_SetItemString(order_dict, "unit_location", PyLong_FromLong(order.unit_location));
    PyDict_SetItemString(order_dict, "target_location", PyLong_FromLong(order.target_location));
    PyDict_SetItemString(order_dict, "target_unit_location", PyLong_FromLong(order.target_unit_location));
    PyDict_SetItemString(order_dict, "dest_location", PyLong_FromLong(order.dest_location));

    return order_dict;
}

static PyObject* test_validate_order(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int power_id;
    const char* order_str;

    if (!PyArg_ParseTuple(args, "Ois", &handle_obj, &power_id, &order_str)) {
        return NULL;
    }

    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or game not initialized");
        return NULL;
    }

    Order order;
    int parse_result = parse_order(order_str, &order, env->game);
    if (parse_result < 0) {
        return PyLong_FromLong(-1);  // Parse failed
    }

    order.power_id = power_id;
    int validate_result = validate_order(env->game, power_id, &order);

    return PyLong_FromLong(validate_result);
}

static PyObject* env_configure(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int welfare_mode;
    int max_years;
    if (!PyArg_ParseTuple(args, "Oii", &handle_obj, &welfare_mode, &max_years)) {
        return NULL;
    }

    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle");
        return NULL;
    }
    c_configure(env, welfare_mode, max_years);
    Py_RETURN_NONE;
}

static PyObject* get_location_index(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    const char* name;
    if (!PyArg_ParseTuple(args, "Os", &handle_obj, &name)) {
        return NULL;
    }

    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game || !env->game->map) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or map not initialized");
        return NULL;
    }

    int idx = find_location_by_name(env->game->map, name);
    return PyLong_FromLong(idx);
}

static PyObject* can_move_names(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int unit_type;
    const char* from_name;
    const char* to_name;
    if (!PyArg_ParseTuple(args, "Oiss", &handle_obj, &unit_type, &from_name, &to_name)) {
        return NULL;
    }

    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game || !env->game->map) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or map not initialized");
        return NULL;
    }
    int from_idx = find_location_by_name(env->game->map, from_name);
    int to_idx = find_location_by_name(env->game->map, to_name);
    if (from_idx < 0 || to_idx < 0) {
        return PyLong_FromLong(0);
    }
    int res = can_move(env->game->map, (UnitType)unit_type, from_idx, to_idx);
    return PyLong_FromLong(res);
}

static PyObject* is_home_center_index(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int location_idx;
    if (!PyArg_ParseTuple(args, "Oi", &handle_obj, &location_idx)) {
        return NULL;
    }
    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game || !env->game->map) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or map not initialized");
        return NULL;
    }
    if (location_idx < 0 || location_idx >= env->game->map->num_locations) {
        return PyLong_FromLong(-1);
    }
    return PyLong_FromLong(env->game->map->locations[location_idx].is_home_center);
}

static PyObject* get_home_centers(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int power_id;
    if (!PyArg_ParseTuple(args, "Oi", &handle_obj, &power_id)) {
        return NULL;
    }
    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game || !env->game->map) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or map not initialized");
        return NULL;
    }
    if (power_id < 0 || power_id >= MAX_POWERS) {
        PyErr_SetString(PyExc_ValueError, "Invalid power id");
        return NULL;
    }
    int n = env->game->map->num_homes[power_id];
    PyObject* list = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyList_SetItem(list, i, PyLong_FromLong(env->game->map->home_centers[power_id][i]));
    }
    return list;
}

// =============================================================================
// Game state mutation helpers for adapters
// =============================================================================

static PyObject* game_clear_units(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int power_id;
    if (!PyArg_ParseTuple(args, "Oi", &handle_obj, &power_id)) {
        return NULL;
    }
    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or game not initialized");
        return NULL;
    }
    if (power_id < 0 || power_id >= MAX_POWERS) {
        PyErr_SetString(PyExc_ValueError, "Invalid power id");
        return NULL;
    }
    env->game->powers[power_id].num_units = 0;
    Py_RETURN_NONE;
}

static int parse_unit_type(PyObject* obj, UnitType* out) {
    if (PyLong_Check(obj)) {
        long v = PyLong_AsLong(obj);
        if (v == 1) { *out = UNIT_ARMY; return 1; }
        if (v == 2) { *out = UNIT_FLEET; return 1; }
        return 0;
    } else if (PyUnicode_Check(obj)) {
        const char* s = PyUnicode_AsUTF8(obj);
        if (!s) return 0;
        if (s[0] == 'A' || s[0] == 'a') { *out = UNIT_ARMY; return 1; }
        if (s[0] == 'F' || s[0] == 'f') { *out = UNIT_FLEET; return 1; }
        return 0;
    }
    return 0;
}

static PyObject* game_set_units(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int power_id;
    PyObject* units_list;
    if (!PyArg_ParseTuple(args, "OiO", &handle_obj, &power_id, &units_list)) {
        return NULL;
    }
    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game || !env->game->map) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or map not initialized");
        return NULL;
    }
    if (power_id < 0 || power_id >= MAX_POWERS) {
        PyErr_SetString(PyExc_ValueError, "Invalid power id");
        return NULL;
    }
    if (!PyList_Check(units_list)) {
        PyErr_SetString(PyExc_TypeError, "units must be a list");
        return NULL;
    }
    Power* power = &env->game->powers[power_id];
    int count = (int)PyList_Size(units_list);
    if (count > MAX_UNITS) count = MAX_UNITS;
    power->num_units = 0;
    for (int i = 0; i < count; i++) {
        PyObject* item = PyList_GetItem(units_list, i);
        if (!PyTuple_Check(item) || PyTuple_Size(item) != 2) continue;
        PyObject* ty = PyTuple_GetItem(item, 0);
        PyObject* loc = PyTuple_GetItem(item, 1);
        UnitType ut;
        if (!parse_unit_type(ty, &ut)) continue;
        if (!PyUnicode_Check(loc)) continue;
        const char* loc_name = PyUnicode_AsUTF8(loc);
        int idx = find_location_by_name(env->game->map, loc_name);
        if (idx < 0) continue;
        // Ensure uniqueness: remove any existing unit occupying this location across all powers
        for (int p = 0; p < MAX_POWERS; p++) {
            Power* op = &env->game->powers[p];
            for (int u = 0; u < op->num_units; ) {
                if (op->units[u].location == idx) {
                    // remove by shifting tail
                    for (int k = u + 1; k < op->num_units; k++) {
                        op->units[k-1] = op->units[k];
                    }
                    op->num_units--;
                    // do not increment u, check new item at u index
                } else {
                    u++;
                }
            }
        }
        power->units[power->num_units].type = ut;
        power->units[power->num_units].location = idx;
        power->units[power->num_units].power_id = power_id;
        power->units[power->num_units].can_retreat = 0;
        power->num_units++;
    }
    Py_RETURN_NONE;
}

static PyObject* game_set_centers(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int power_id;
    PyObject* centers_list;
    if (!PyArg_ParseTuple(args, "OiO", &handle_obj, &power_id, &centers_list)) {
        return NULL;
    }
    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game || !env->game->map) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or map not initialized");
        return NULL;
    }
    if (power_id < 0 || power_id >= MAX_POWERS) {
        PyErr_SetString(PyExc_ValueError, "Invalid power id");
        return NULL;
    }
    if (!PyList_Check(centers_list)) {
        PyErr_SetString(PyExc_TypeError, "centers must be a list of names");
        return NULL;
    }

    // Clear previous ownership for this power
    for (int i = 0; i < env->game->map->num_locations; i++) {
        if (env->game->map->locations[i].owner_power == power_id) {
            env->game->map->locations[i].owner_power = -1;
        }
    }

    Power* power = &env->game->powers[power_id];
    power->num_centers = 0;
    int count = (int)PyList_Size(centers_list);
    if (count > MAX_UNITS) count = MAX_UNITS;
    for (int i = 0; i < count; i++) {
        PyObject* loc = PyList_GetItem(centers_list, i);
        if (!PyUnicode_Check(loc)) continue;
        const char* loc_name = PyUnicode_AsUTF8(loc);
        int idx = find_location_by_name(env->game->map, loc_name);
        if (idx < 0) continue;
        power->centers[power->num_centers++] = idx;
        env->game->map->locations[idx].has_supply_center = 1;
        env->game->map->locations[idx].owner_power = power_id;
    }
    Py_RETURN_NONE;
}

// Define custom methods macro
#define MY_METHODS \
    {"query_game_state", query_game_state, METH_VARARGS, "Query game state"}, \
    {"query_map_info", query_map_info, METH_VARARGS, "Query map information"}, \
    {"test_parse_order", test_parse_order, METH_VARARGS, "Parse an order string"}, \
    {"test_validate_order", test_validate_order, METH_VARARGS, "Validate an order"}, \
    {"env_configure", env_configure, METH_VARARGS, "Configure welfare mode and max years"}, \
    {"get_location_index", get_location_index, METH_VARARGS, "Get location index by name"}, \
    {"can_move_names", can_move_names, METH_VARARGS, "Check adjacency/movement between locations by names"}, \
    {"is_home_center_index", is_home_center_index, METH_VARARGS, "Return home-center power id or -1"}, \
    {"get_home_centers", get_home_centers, METH_VARARGS, "List of home center indices for a power"}, \
    {"game_clear_units", game_clear_units, METH_VARARGS, "Clear all units for a power"}, \
    {"game_set_units", game_set_units, METH_VARARGS, "Set units for a power from list[(type, name)]"}, \
    {"game_set_centers", game_set_centers, METH_VARARGS, "Set centers for a power from list[name]"}, \
    {"game_clear_centers", game_clear_centers, METH_VARARGS, "Clear all owned centers for all powers"}, \
    {"game_submit_orders", game_submit_orders, METH_VARARGS, "Submit orders for a power from list of order strings"}, \
    {"get_dislodged_units", get_dislodged_units, METH_VARARGS, "Get list of dislodged units with dislodger info"}, \
    {"list_possible_orders", list_possible_orders, METH_VARARGS, "List HOLD/MOVE orders for a unit"}

static PyObject* game_submit_orders(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int power_id;
    PyObject* orders_list;
    if (!PyArg_ParseTuple(args, "OiO", &handle_obj, &power_id, &orders_list)) {
        return NULL;
    }
    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game || !env->game->map) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or game not initialized");
        return NULL;
    }
    if (power_id < 0 || power_id >= MAX_POWERS) {
        PyErr_SetString(PyExc_ValueError, "Invalid power id");
        return NULL;
    }
    if (!PyList_Check(orders_list)) {
        PyErr_SetString(PyExc_TypeError, "orders must be a list");
        return NULL;
    }
    
    Power* power = &env->game->powers[power_id];
    power->num_orders = 0;  // Clear existing orders
    
    int count = (int)PyList_Size(orders_list);
    if (count > MAX_UNITS) count = MAX_UNITS;
    
    for (int i = 0; i < count; i++) {
        PyObject* order_str_obj = PyList_GetItem(orders_list, i);
        if (!PyUnicode_Check(order_str_obj)) continue;
        
        const char* order_str = PyUnicode_AsUTF8(order_str_obj);
        if (!order_str) continue;
        
        // Parse order
        Order order;
        if (parse_order(order_str, &order, env->game) == 0) {
            order.power_id = power_id;
            power->orders[power->num_orders++] = order;
        }
        // If parsing fails, just skip this order
    }
    
    Py_RETURN_NONE;
}

static PyObject* get_dislodged_units(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }
    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game || !env->game->map) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or game not initialized");
        return NULL;
    }
    
    GameState* game = env->game;
    PyObject* result_list = PyList_New(0);
    
    for (int i = 0; i < game->num_dislodged; i++) {
        DislodgedUnit* d = &game->dislodged[i];
        PyObject* d_dict = PyDict_New();
        
        PyDict_SetItemString(d_dict, "power_id", PyLong_FromLong(d->power_id));
        PyDict_SetItemString(d_dict, "type", PyLong_FromLong(d->type));
        PyDict_SetItemString(d_dict, "from_location", PyLong_FromLong(d->from_location));
        PyDict_SetItemString(d_dict, "dislodged_by_location", PyLong_FromLong(d->dislodged_by_location));
        
        PyList_Append(result_list, d_dict);
        Py_DECREF(d_dict);
    }
    
    return result_list;
}

static PyObject* game_clear_centers(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }
    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game || !env->game->map) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or map not initialized");
        return NULL;
    }
    // Clear ownership on all locations
    for (int i = 0; i < env->game->map->num_locations; i++) {
        env->game->map->locations[i].owner_power = -1;
    }
    // Clear each power's centers list
    for (int p = 0; p < MAX_POWERS; p++) {
        env->game->powers[p].num_centers = 0;
    }
    Py_RETURN_NONE;
}

static PyObject* list_possible_orders(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int power_id;
    int location;
    if (!PyArg_ParseTuple(args, "Oii", &handle_obj, &power_id, &location)) {
        return NULL;
    }
    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env || !env->game) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle or game not initialized");
        return NULL;
    }
    char orders_buf[MAX_UNITS][MAX_ORDER_LENGTH];
    int n = 0;
    get_possible_orders(env->game, power_id, location, orders_buf, &n);
    PyObject* list = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyList_SetItem(list, i, PyUnicode_FromString(orders_buf[i]));
    }
    return list;
}

// Include the PufferLib environment binding template
#include "../env_binding.h"
