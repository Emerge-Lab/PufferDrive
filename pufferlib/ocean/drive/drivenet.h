#include <time.h>
#include "puffernet.h"
#include <math.h>
#include <raylib.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define MAX_ROAD_OBJECTS 128
#define MAX_PARTNER_OBJECTS 31
#define ROAD_FEATURES 7
#define ROAD_FEATURES_AFTER_ONEHOT 13
#define PARTNER_FEATURES 7

typedef struct DriveNet DriveNet;
struct DriveNet {
    int num_agents;
    int ego_dim;
    float* obs_self;
    float* obs_partner;
    float* obs_road;
    float* partner_linear_output;
    float* road_linear_output;
    float* partner_layernorm_output;
    float* road_layernorm_output;
    Linear* ego_encoder;
    Linear* road_encoder;
    Linear* partner_encoder;
    LayerNorm* ego_layernorm;
    LayerNorm* road_layernorm;
    LayerNorm* partner_layernorm;
    Linear* ego_encoder_two;
    MaxDim1* partner_max;
    MaxDim1* road_max;
    CatDim1* cat1;
    CatDim1* cat2;
    GELU* gelu;
    Linear* shared_embedding;
    ReLU* relu;
    LSTM* lstm;
    Linear* actor;
    Linear* value_fn;
    Multidiscrete* multidiscrete;
};

DriveNet* init_drivenet(Weights* weights, int num_agents, int dynamics_model) {
    DriveNet* net = calloc(1, sizeof(DriveNet));
    int hidden_size = 256;
    int input_size = 128;

    int ego_dim = (dynamics_model == JERK) ? 10 : 7;

    // Determine action space size based on dynamics model
    int action_size, logit_sizes[2];
    int action_dim;
    if (dynamics_model == CLASSIC) {
        action_size = 7 * 13; // Joint action space
        logit_sizes[0] = 7 * 13;
        action_dim = 1;
    } else {  // JERK
        action_size = 7;   // 4 + 3
        logit_sizes[0] = 4;
        logit_sizes[1] = 3;
        action_dim = 2;
    }

    net->num_agents = num_agents;
    net->ego_dim = ego_dim;
    net->obs_self = calloc(num_agents*ego_dim, sizeof(float));
    net->obs_partner = calloc(num_agents*MAX_PARTNER_OBJECTS*PARTNER_FEATURES, sizeof(float));
    net->obs_road = calloc(num_agents*MAX_ROAD_OBJECTS*ROAD_FEATURES_AFTER_ONEHOT, sizeof(float));
    net->partner_linear_output = calloc(num_agents*input_size, sizeof(float));
    net->road_linear_output = calloc(num_agents*input_size, sizeof(float));
    net->partner_layernorm_output = calloc(num_agents*input_size, sizeof(float));
    net->road_layernorm_output = calloc(num_agents*input_size, sizeof(float));
    net->ego_encoder = make_linear(weights, num_agents, ego_dim, input_size);
    net->ego_layernorm = make_layernorm(weights, num_agents, input_size);
    net->ego_encoder_two = make_linear(weights, num_agents, input_size, input_size);
    net->road_encoder = make_linear(weights, num_agents, ROAD_FEATURES_AFTER_ONEHOT, input_size);
    net->road_layernorm = make_layernorm(weights, num_agents, input_size);
    net->partner_encoder = make_linear(weights, num_agents, PARTNER_FEATURES, input_size);
    net->partner_layernorm = make_layernorm(weights, num_agents, input_size);
    net->partner_max = make_max_dim1(num_agents, MAX_PARTNER_OBJECTS, input_size);
    net->road_max = make_max_dim1(num_agents, MAX_ROAD_OBJECTS, input_size);
    net->cat1 = make_cat_dim1(num_agents, input_size, input_size);
    net->cat2 = make_cat_dim1(num_agents, input_size + input_size, input_size);
    net->gelu = make_gelu(num_agents, 3*input_size);
    net->shared_embedding = make_linear(weights, num_agents, input_size*3, hidden_size);
    net->relu = make_relu(num_agents, hidden_size);
    net->actor = make_linear(weights, num_agents, hidden_size, action_size);
    net->value_fn = make_linear(weights, num_agents, hidden_size, 1);
    net->lstm = make_lstm(weights, num_agents, hidden_size, 256);
    memset(net->lstm->state_h, 0, num_agents*256*sizeof(float));
    memset(net->lstm->state_c, 0, num_agents*256*sizeof(float));
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, action_dim);
    return net;
}

void free_drivenet(DriveNet* net) {
    free(net->obs_self);
    free(net->obs_partner);
    free(net->obs_road);
    free(net->partner_linear_output);
    free(net->road_linear_output);
    free(net->partner_layernorm_output);
    free(net->road_layernorm_output);
    free(net->ego_encoder);
    free(net->road_encoder);
    free(net->partner_encoder);
    free(net->ego_layernorm);
    free(net->road_layernorm);
    free(net->partner_layernorm);
    free(net->ego_encoder_two);
    free(net->partner_max);
    free(net->road_max);
    free(net->cat1);
    free(net->cat2);
    free(net->gelu);
    free(net->shared_embedding);
    free(net->relu);
    free(net->multidiscrete);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net);
}

void forward(DriveNet* net, float* observations, int* actions) {
    int ego_dim = net->ego_dim;
    int input_size = 128;

    // Clear previous observations
    memset(net->obs_self, 0, net->num_agents * ego_dim * sizeof(float));
    memset(net->obs_partner, 0, net->num_agents * MAX_PARTNER_OBJECTS * PARTNER_FEATURES * sizeof(float));
    memset(net->obs_road, 0, net->num_agents * MAX_ROAD_OBJECTS * ROAD_FEATURES_AFTER_ONEHOT * sizeof(float));

    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b * (ego_dim + MAX_PARTNER_OBJECTS*PARTNER_FEATURES + MAX_ROAD_OBJECTS*ROAD_FEATURES);
        int partner_offset = b_offset + ego_dim;
        int road_offset = b_offset + ego_dim + MAX_PARTNER_OBJECTS*PARTNER_FEATURES;

        // Process self observation
        for(int i = 0; i < ego_dim; i++) {
            net->obs_self[b * ego_dim + i] = observations[b_offset + i];
        }

        // Process partner observation
        for(int i = 0; i < MAX_PARTNER_OBJECTS; i++) {
            for(int j = 0; j < PARTNER_FEATURES; j++) {
                net->obs_partner[b*MAX_PARTNER_OBJECTS*PARTNER_FEATURES + i*PARTNER_FEATURES + j] = observations[partner_offset + i*PARTNER_FEATURES + j];
            }
        }

        // Process road observation
        for(int i = 0; i < MAX_ROAD_OBJECTS; i++) {
            for(int j = 0; j < ROAD_FEATURES - 1; j++) {
                net->obs_road[b*MAX_ROAD_OBJECTS*ROAD_FEATURES_AFTER_ONEHOT + i*ROAD_FEATURES_AFTER_ONEHOT + j] = observations[road_offset + i*ROAD_FEATURES + j];
            }
            // One-hot encode the categorical feature
            int category = (int)observations[road_offset + i*ROAD_FEATURES + (ROAD_FEATURES - 1)];
            for(int j = 0; j < 7; j++) {
                if(j == category) {
                    net->obs_road[b*MAX_ROAD_OBJECTS*ROAD_FEATURES_AFTER_ONEHOT + i*ROAD_FEATURES_AFTER_ONEHOT + 6 + j] = 1.0f;
                } else {
                    net->obs_road[b*MAX_ROAD_OBJECTS*ROAD_FEATURES_AFTER_ONEHOT + i*ROAD_FEATURES_AFTER_ONEHOT + 6 + j] = 0.0f;
                }
            }
        }
    }

    // Forward pass through the network
    linear(net->ego_encoder, net->obs_self);
    layernorm(net->ego_layernorm, net->ego_encoder->output);
    relu(net->relu, net->ego_layernorm->output);
    linear(net->ego_encoder_two, net->relu->output);

    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < MAX_PARTNER_OBJECTS; obj++) {
            float* obj_features = &net->obs_partner[b*MAX_PARTNER_OBJECTS*PARTNER_FEATURES + obj*PARTNER_FEATURES];
            _linear(obj_features, net->partner_encoder->weights, net->partner_encoder->bias,
                   &net->partner_linear_output[b*MAX_PARTNER_OBJECTS*input_size + obj*input_size], 1, PARTNER_FEATURES, input_size);
        }
    }
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < MAX_PARTNER_OBJECTS; obj++) {
            float* after_linear = &net->partner_linear_output[b*MAX_PARTNER_OBJECTS*input_size + obj*input_size];
            _layernorm(after_linear, net->partner_layernorm->weights, net->partner_layernorm->bias,
                        &net->partner_layernorm_output[b*MAX_PARTNER_OBJECTS*input_size + obj*input_size], 1, input_size);
        }
    }
    max_dim1(net->partner_max, net->partner_layernorm_output);

    // Process road objects: apply linear to each object individually, then max pool
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < MAX_ROAD_OBJECTS; obj++) {
            float* obj_features = &net->obs_road[b*MAX_ROAD_OBJECTS*ROAD_FEATURES_AFTER_ONEHOT + obj*ROAD_FEATURES_AFTER_ONEHOT];
            _linear(obj_features, net->road_encoder->weights, net->road_encoder->bias,
                   &net->road_linear_output[b*MAX_ROAD_OBJECTS*input_size + obj*input_size], 1, ROAD_FEATURES_AFTER_ONEHOT, input_size);
        }
    }
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < MAX_ROAD_OBJECTS; obj++) {
            float* after_linear = &net->road_linear_output[b*MAX_ROAD_OBJECTS*input_size + obj*input_size];
            _layernorm(after_linear, net->road_layernorm->weights, net->road_layernorm->bias,
                        &net->road_layernorm_output[b*MAX_ROAD_OBJECTS*input_size + obj*input_size], 1, input_size);
        }
    }
    max_dim1(net->road_max, net->road_layernorm_output);

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
}
