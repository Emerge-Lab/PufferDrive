#define Env Drive
#define MY_SHARED
#define MY_PUT
#include "binding.h"

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

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {

    int population_play = unpack(kwargs, "population_play");
    if (population_play){
        return my_shared_population_play(self, args,  kwargs);
    }
    else{
        return my_shared_self_play( self, args,  kwargs);
    }

}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->ini_file = unpack_str(kwargs, "ini_file");
    env_init_config conf = {0};
    if(ini_parse(env->ini_file, handler, &conf) < 0) {
        printf("Error while loading %s", env->ini_file);
    }
    if (kwargs && PyDict_GetItemString(kwargs, "scenario_length")) {
        conf.scenario_length = (int)unpack(kwargs, "scenario_length");
    }
    if (conf.scenario_length <= 0) {
        PyErr_SetString(PyExc_ValueError, "scenario_length must be > 0 (set in INI or kwargs)");
        return -1;
    }
    env->action_type = conf.action_type;
    env->dynamics_model = conf.dynamics_model;
    if (PyDict_GetItemString(kwargs, "dynamics_model")) {
        char* dynamics_str = unpack_str(kwargs, "dynamics_model");
        env->dynamics_model = (strcmp(dynamics_str, "jerk") == 0) ? JERK : CLASSIC;
    }
    env->reward_vehicle_collision = conf.reward_vehicle_collision;
    env->reward_offroad_collision = conf.reward_offroad_collision;
    env->reward_goal = conf.reward_goal;
    env->reward_goal_post_respawn = conf.reward_goal_post_respawn;
    env->reward_ade = conf.reward_ade;
    env->scenario_length = conf.scenario_length;
    env->collision_behavior = conf.collision_behavior;
    env->offroad_behavior = conf.offroad_behavior;
    env->max_controlled_agents = unpack(kwargs, "max_controlled_agents");
    env->dt = conf.dt;

    // Conditioning parameters
    env->use_rc = (bool)unpack(kwargs, "use_rc");
    env->use_ec = (bool)unpack(kwargs, "use_ec");
    env->use_dc = (bool)unpack(kwargs, "use_dc");
    env->collision_weight_lb = (float)unpack(kwargs, "collision_weight_lb");
    env->collision_weight_ub = (float)unpack(kwargs, "collision_weight_ub");
    env->offroad_weight_lb = (float)unpack(kwargs, "offroad_weight_lb");
    env->offroad_weight_ub = (float)unpack(kwargs, "offroad_weight_ub");
    env->goal_weight_lb = (float)unpack(kwargs, "goal_weight_lb");
    env->goal_weight_ub = (float)unpack(kwargs, "goal_weight_ub");
    env->entropy_weight_lb = (float)unpack(kwargs, "entropy_weight_lb");
    env->entropy_weight_ub = (float)unpack(kwargs, "entropy_weight_ub");
    env->discount_weight_lb = (float)unpack(kwargs, "discount_weight_lb");
    env->discount_weight_ub = (float)unpack(kwargs, "discount_weight_ub");

    env->adaptive_driving_agent = unpack(kwargs, "adaptive_driving");

    if (env->adaptive_driving_agent) {
        env->k_scenarios = unpack(kwargs, "k_scenarios");
        env->current_scenario = 0;
    } else {
        env->k_scenarios = 0;
    }

    env->population_play = unpack(kwargs, "population_play");

    if (env->population_play) {
        env->num_co_players = unpack(kwargs, "num_co_players");
        double* co_player_ids_d = unpack_float_array(kwargs, "co_player_ids", &env->num_co_players);

        if (co_player_ids_d != NULL && env->num_co_players > 0) {
            env->co_player_ids = (int*)malloc(env->num_co_players * sizeof(int));
            if (env->co_player_ids == NULL) {
                fprintf(stderr, "Error: Failed to allocate memory for co_player_ids\n");
                free(co_player_ids_d);
                env->num_co_players = 0;
            } else {
                for (int i = 0; i < env->num_co_players; i++) {
                    env->co_player_ids[i] = (int)co_player_ids_d[i];
                }
                free(co_player_ids_d);
            }
        } else {
            if (co_player_ids_d != NULL) {
                free(co_player_ids_d);
            }
            env->co_player_ids = NULL;
            env->num_co_players = 0;
        }

        // Handle ego agents - always as an array
        env->num_ego_agents = unpack(kwargs, "num_ego_agents");
        if (env->num_ego_agents > 0) {
            double* ego_agent_ids_d = unpack_float_array(kwargs, "ego_agent_ids", &env->num_ego_agents);
            if (ego_agent_ids_d != NULL) {
                env->ego_agent_ids = (int*)malloc(env->num_ego_agents * sizeof(int));
                for (int i = 0; i < env->num_ego_agents; i++) {
                    env->ego_agent_ids[i] = (int)ego_agent_ids_d[i];
                }
                free(ego_agent_ids_d);
            } else {
                env->ego_agent_ids = NULL;
                env->num_ego_agents = 0;
            }
        } else {
            env->ego_agent_ids = NULL;
            env->num_ego_agents = 0;
        }
    } else {
        // Non-population play mode - set defaults
        env->num_ego_agents = 0;
        env->ego_agent_ids = NULL;
    }


    env->init_mode = (int)unpack(kwargs, "init_mode");
    env->control_mode = (int)unpack(kwargs, "control_mode");
    env->goal_behavior = (int)unpack(kwargs, "goal_behavior");
    env->goal_radius = (float)unpack(kwargs, "goal_radius");
    int map_id = unpack(kwargs, "map_id");
    int max_agents = unpack(kwargs, "max_agents");
    int init_steps = unpack(kwargs, "init_steps");
    char map_file[100];
    sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);
    env->num_agents = max_agents;
    env->map_name = strdup(map_file);
    env->init_steps = init_steps;
    env->timestep = init_steps;
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "avg_displacement_error", log->avg_displacement_error);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    assign_to_dict(dict, "lane_alignment_rate", log->lane_alignment_rate);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "avg_offroad_per_agent", log->avg_offroad_per_agent);
    assign_to_dict(dict, "avg_collisions_per_agent", log->avg_collisions_per_agent);
    return 0;
}
