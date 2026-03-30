#include <time.h>
#include "drive.h"
#include "puffernet.h"
#include <math.h>
#include <raylib.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define NN_INPUT_SIZE 64
#define NN_HIDDEN_SIZE 256

typedef struct DriveNet DriveNet;
struct DriveNet {
    int num_agents;
    int ego_dim;
    float *obs_self;
    float *obs_partner;
    float *obs_road;
    float *partner_linear_output;
    float *road_linear_output;
    float *partner_layernorm_output;
    float *road_layernorm_output;
    float *partner_linear_output_two;
    float *road_linear_output_two;
    Linear *ego_encoder;
    Linear *road_encoder;
    Linear *partner_encoder;
    LayerNorm *ego_layernorm;
    LayerNorm *road_layernorm;
    LayerNorm *partner_layernorm;
    Linear *ego_encoder_two;
    Linear *road_encoder_two;
    Linear *partner_encoder_two;
    MaxDim1 *partner_max;
    MaxDim1 *road_max;
    CatDim1 *cat1;
    CatDim1 *cat2;
    GELU *gelu;
    Linear *shared_embedding;
    ReLU *relu;
    LSTM *lstm;
    Linear *actor;
    Linear *value_fn;
    Multidiscrete *multidiscrete;
    // Predicted trajectory (rolled out from actor output)
    int action_size;
    int traj_steps;
    float *predicted_traj_x;
    float *predicted_traj_y;
};

DriveNet *init_drivenet(Weights *weights, int num_agents, int dynamics_model, int reward_conditioning) {
    DriveNet *net = calloc(1, sizeof(DriveNet));
    // Use constants directly from drive.h
    int ego_dim = (dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    if (reward_conditioning) {
        ego_dim += NUM_REWARD_COEFS; // Add 16 conditioning features
    }
    int max_partners = MAX_AGENTS - 1;
    int max_road_obs = MAX_ROAD_SEGMENT_OBSERVATIONS;
    int partner_features = PARTNER_FEATURES;
    int road_features = ROAD_FEATURES;
    int input_size = NN_INPUT_SIZE;
    int hidden_size = NN_HIDDEN_SIZE;
    int road_feat_onehot = road_features + 6; // one-hot extra 6 features for road

    // Determine action space size based on dynamics model
    int action_size, logit_sizes[2];
    int action_dim;
    if (dynamics_model == CLASSIC) {
        action_size = 7 * 13; // Joint action space
        logit_sizes[0] = 7 * 13;
        action_dim = 1;
    } else {                 // JERK
        action_size = 4 * 3; // Joint action space (4 longitudinal × 3 lateral = 12)
        logit_sizes[0] = 4 * 3;
        action_dim = 1;
    }

    net->num_agents = num_agents;
    net->ego_dim = ego_dim;
    net->obs_self = calloc(num_agents * ego_dim, sizeof(float));
    net->obs_partner = calloc(num_agents * max_partners * partner_features, sizeof(float));
    net->obs_road = calloc(num_agents * max_road_obs * road_feat_onehot, sizeof(float));
    net->partner_linear_output = calloc(num_agents * max_partners * input_size, sizeof(float));
    net->road_linear_output = calloc(num_agents * max_road_obs * input_size, sizeof(float));
    net->partner_linear_output_two = calloc(num_agents * max_partners * input_size, sizeof(float));
    net->road_linear_output_two = calloc(num_agents * max_road_obs * input_size, sizeof(float));
    net->partner_layernorm_output = calloc(num_agents * max_partners * input_size, sizeof(float));
    net->road_layernorm_output = calloc(num_agents * max_road_obs * input_size, sizeof(float));

    net->ego_encoder = make_linear(weights, num_agents, ego_dim, input_size);
    net->ego_layernorm = make_layernorm(weights, num_agents, input_size);
    net->ego_encoder_two = make_linear(weights, num_agents, input_size, input_size);
    net->road_encoder = make_linear(weights, num_agents, road_feat_onehot, input_size);
    net->road_layernorm = make_layernorm(weights, num_agents, input_size);
    net->road_encoder_two = make_linear(weights, num_agents, input_size, input_size);
    net->partner_encoder = make_linear(weights, num_agents, partner_features, input_size);
    net->partner_layernorm = make_layernorm(weights, num_agents, input_size);
    net->partner_encoder_two = make_linear(weights, num_agents, input_size, input_size);
    net->partner_max = make_max_dim1(num_agents, max_partners, input_size);
    net->road_max = make_max_dim1(num_agents, max_road_obs, input_size);
    net->cat1 = make_cat_dim1(num_agents, input_size, input_size);
    net->cat2 = make_cat_dim1(num_agents, input_size + input_size, input_size);
    net->gelu = make_gelu(num_agents, 3 * input_size);
    net->shared_embedding = make_linear(weights, num_agents, input_size * 3, hidden_size);
    net->relu = make_relu(num_agents, hidden_size);
    net->actor = make_linear(weights, num_agents, hidden_size, action_size);
    net->value_fn = make_linear(weights, num_agents, hidden_size, 1);
    net->lstm = make_lstm(weights, num_agents, hidden_size, NN_HIDDEN_SIZE);
    memset(net->lstm->state_h, 0, num_agents * NN_HIDDEN_SIZE * sizeof(float));
    memset(net->lstm->state_c, 0, num_agents * NN_HIDDEN_SIZE * sizeof(float));
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, action_dim);
    net->action_size = action_size;
    net->traj_steps = 80;
    net->predicted_traj_x = calloc(num_agents * net->traj_steps, sizeof(float));
    net->predicted_traj_y = calloc(num_agents * net->traj_steps, sizeof(float));
    return net;
}

void free_drivenet(DriveNet *net) {
    free(net->obs_self);
    free(net->obs_partner);
    free(net->obs_road);
    free(net->partner_linear_output);
    free(net->road_linear_output);
    free(net->partner_linear_output_two);
    free(net->road_linear_output_two);
    free(net->partner_layernorm_output);
    free(net->road_layernorm_output);
    free(net->ego_encoder);
    free(net->road_encoder);
    free(net->partner_encoder);
    free(net->ego_layernorm);
    free(net->road_layernorm);
    free(net->partner_layernorm);
    free(net->ego_encoder_two);
    free(net->road_encoder_two);
    free(net->partner_encoder_two);
    free(net->partner_max);
    free(net->road_max);
    free(net->cat1);
    free(net->cat2);
    free(net->gelu);
    free(net->shared_embedding);
    free(net->relu);
    free(net->multidiscrete);
    free(net->predicted_traj_x);
    free(net->predicted_traj_y);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net);
}

void forward(DriveNet *net, Drive *env, float *observations, int *actions) {
    int ego_dim = net->ego_dim;
    int max_partners = MAX_AGENTS - 1;
    int max_road_obs = MAX_ROAD_SEGMENT_OBSERVATIONS;
    int partner_features = PARTNER_FEATURES;
    int road_features = ROAD_FEATURES;
    int road_feat_onehot = road_features + 6; // one-hot extra 6 features for road

    // Clear previous observations
    memset(net->obs_self, 0, net->num_agents * ego_dim * sizeof(float));
    memset(net->obs_partner, 0, net->num_agents * max_partners * partner_features * sizeof(float));
    memset(net->obs_road, 0, net->num_agents * max_road_obs * road_feat_onehot * sizeof(float));

    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b * (ego_dim + max_partners * partner_features + max_road_obs * road_features);
        int partner_offset = b_offset + ego_dim;
        int road_offset = b_offset + ego_dim + max_partners * partner_features;

        // Process self observation
        for (int i = 0; i < ego_dim; i++) {
            net->obs_self[b * ego_dim + i] = observations[b_offset + i];
        }

        // Process partner observation
        for (int i = 0; i < max_partners; i++) {
            for (int j = 0; j < partner_features; j++) {
                net->obs_partner[b * max_partners * partner_features + i * partner_features + j] =
                    observations[partner_offset + i * partner_features + j];
            }
        }

        // Process road observation
        for (int i = 0; i < MAX_ROAD_SEGMENT_OBSERVATIONS; i++) {
            for (int j = 0; j < 8; j++) {
                net->obs_road[b * MAX_ROAD_SEGMENT_OBSERVATIONS * ROAD_FEATURES_ONEHOT + i * ROAD_FEATURES_ONEHOT + j] =
                    observations[road_offset + i * 8 + j];
            }
            for (int j = 0; j < 7; j++) {
                if (j == observations[road_offset + i * 8 + 7]) {
                    net->obs_road[b * MAX_ROAD_SEGMENT_OBSERVATIONS * ROAD_FEATURES_ONEHOT + i * ROAD_FEATURES_ONEHOT +
                                  7 + j] = 1.0f;
                } else {
                    net->obs_road[b * MAX_ROAD_SEGMENT_OBSERVATIONS * ROAD_FEATURES_ONEHOT + i * ROAD_FEATURES_ONEHOT +
                                  7 + j] = 0.0f;
                }
            }
        }
    }

    // Forward pass through the network
    linear(net->ego_encoder, net->obs_self);
    layernorm(net->ego_layernorm, net->ego_encoder->output);
    linear(net->ego_encoder_two, net->ego_layernorm->output);
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_partners; obj++) {
            // Get the 7 features for this object
            float *obj_features = &net->obs_partner[b * max_partners * partner_features + obj * partner_features];
            // Apply linear layer to this object
            _linear(obj_features, net->partner_encoder->weights, net->partner_encoder->bias,
                    &net->partner_linear_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    partner_features, NN_INPUT_SIZE);
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_partners; obj++) {
            float *after_first = &net->partner_linear_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE];
            _layernorm(after_first, net->partner_layernorm->weights, net->partner_layernorm->bias,
                       &net->partner_layernorm_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                       NN_INPUT_SIZE);
        }
    }
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_partners; obj++) {
            // Get the 7 features for this object
            float *obj_features =
                &net->partner_layernorm_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE];
            // Apply linear layer to this object
            _linear(obj_features, net->partner_encoder_two->weights, net->partner_encoder_two->bias,
                    &net->partner_linear_output_two[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    NN_INPUT_SIZE, NN_INPUT_SIZE);
        }
    }

    // Process road objects: apply linear to each object individually
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_road_obs; obj++) {
            // Get the 13 features for this object
            float *obj_features = &net->obs_road[b * max_road_obs * ROAD_FEATURES_ONEHOT + obj * ROAD_FEATURES_ONEHOT];
            // Apply linear layer to this object
            _linear(obj_features, net->road_encoder->weights, net->road_encoder->bias,
                    &net->road_linear_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    ROAD_FEATURES_ONEHOT, NN_INPUT_SIZE);
        }
    }

    // Apply layer norm and second linear to each road object
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_road_obs; obj++) {
            float *after_first = &net->road_linear_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE];
            _layernorm(after_first, net->road_layernorm->weights, net->road_layernorm->bias,
                       &net->road_layernorm_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                       NN_INPUT_SIZE);
        }
    }
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_road_obs; obj++) {
            float *after_first = &net->road_layernorm_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE];
            _linear(after_first, net->road_encoder_two->weights, net->road_encoder_two->bias,
                    &net->road_linear_output_two[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    NN_INPUT_SIZE, NN_INPUT_SIZE);
        }
    }

    max_dim1(net->partner_max, net->partner_linear_output_two);
    max_dim1(net->road_max, net->road_linear_output_two);
    cat_dim1(net->cat1, net->ego_encoder_two->output, net->road_max->output);
    cat_dim1(net->cat2, net->cat1->output, net->partner_max->output);
    gelu(net->gelu, net->cat2->output);
    linear(net->shared_embedding, net->gelu->output);
    relu(net->relu, net->shared_embedding->output);
    lstm(net->lstm, net->relu->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);

    // Get action by taking argmax of actor output
    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);

    // Roll out predicted trajectories using the reusable dynamics step functions
    if (env != NULL) {
        int traj_steps = net->traj_steps;
        int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);
        int num_lat = sizeof(JERK_LAT) / sizeof(JERK_LAT[0]);

        for (int b = 0; b < net->num_agents; b++) {
            int agent_idx = env->active_agent_indices[b];
            Agent *agent = &env->agents[agent_idx];
            int action_val = actions[b];

            DynState s = {agent->sim_x, agent->sim_y, agent->sim_heading,
                          agent->sim_vx, agent->sim_vy,
                          agent->a_long, agent->a_lat, agent->steering_angle};

            for (int t = 0; t < traj_steps; t++) {
                // Apply action on first step, zero-jerk/zero-accel after
                if (t == 0) {
                    if (env->dynamics_model == CLASSIC) {
                        int accel_idx = action_val / num_steer;
                        int steer_idx = action_val % num_steer;
                        s = classic_dynamics_step(s, ACCELERATION_VALUES[accel_idx],
                                                  STEERING_VALUES[steer_idx], agent->sim_length, env->dt);
                    } else {
                        int al_idx = action_val / num_lat;
                        int at_idx = action_val % num_lat;
                        s = jerk_dynamics_step(s, JERK_LONG[al_idx], JERK_LAT[at_idx],
                                              agent->wheelbase, env->dt);
                    }
                } else {
                    if (env->dynamics_model == CLASSIC) {
                        s = classic_dynamics_step(s, 0.0f, 0.0f, agent->sim_length, env->dt);
                    } else {
                        s = jerk_dynamics_step(s, 0.0f, 0.0f, agent->wheelbase, env->dt);
                    }
                }
                net->predicted_traj_x[b * traj_steps + t] = s.x;
                net->predicted_traj_y[b * traj_steps + t] = s.y;
            }
        }
    }
}
