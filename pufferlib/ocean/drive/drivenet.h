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
#define INPUT_SIZE 64
#define HIDDEN_SIZE 256

typedef struct DriveNet DriveNet;
struct DriveNet {
    int num_agents;
    int ego_dim;
    int obs_size;
    float* obs_self;
    float* obs_partner;
    float* obs_road;
    float* obs_full;
    float* ego_relu_output;
    float* partner_linear_output;
    float* road_linear_output;
    float* partner_layernorm_output;
    float* road_layernorm_output;
    float* partner_relu_output;
    float* partner_linear_output_two;
    float* road_linear_output_two;
    Linear* ego_encoder;
    Linear* road_encoder;
    Linear* partner_encoder;
    LayerNorm* ego_layernorm;
    LayerNorm* road_layernorm;
    LayerNorm* partner_layernorm;
    ReLU* ego_relu;
    ReLU* partner_relu;
    Linear* ego_encoder_two;
    Linear* road_encoder_two;
    Linear* partner_encoder_two;
    Linear* full_scene_encoder;
    LayerNorm* full_scene_layernorm;
    MaxDim1* partner_max;
    MaxDim1* road_max;
    CatDim1* cat1;
    CatDim1* cat2;
    CatDim1* cat3;
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
    int hidden_size = HIDDEN_SIZE;
    int input_size = INPUT_SIZE;

    int ego_dim = (dynamics_model == JERK) ? 10 : 7;
    int obs_size = ego_dim + MAX_PARTNER_OBJECTS*PARTNER_FEATURES + MAX_ROAD_OBJECTS*ROAD_FEATURES;

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
    net->obs_size = obs_size;
    net->obs_self = calloc(num_agents*ego_dim, sizeof(float));
    net->obs_partner = calloc(num_agents*MAX_PARTNER_OBJECTS*PARTNER_FEATURES, sizeof(float));
    net->obs_road = calloc(num_agents*MAX_ROAD_OBJECTS*ROAD_FEATURES_AFTER_ONEHOT, sizeof(float));
    net->obs_full = calloc(num_agents*obs_size, sizeof(float));
    net->ego_relu_output = calloc(num_agents*input_size, sizeof(float));
    net->partner_linear_output = calloc(num_agents*MAX_PARTNER_OBJECTS*input_size, sizeof(float));
    net->road_linear_output = calloc(num_agents*MAX_ROAD_OBJECTS*input_size, sizeof(float));
    net->partner_linear_output_two = calloc(num_agents*MAX_PARTNER_OBJECTS*input_size, sizeof(float));
    net->road_linear_output_two = calloc(num_agents*MAX_ROAD_OBJECTS*input_size, sizeof(float));
    net->partner_layernorm_output = calloc(num_agents*MAX_PARTNER_OBJECTS*input_size, sizeof(float));
    net->road_layernorm_output = calloc(num_agents*MAX_ROAD_OBJECTS*input_size, sizeof(float));
    net->partner_relu_output = calloc(num_agents*MAX_PARTNER_OBJECTS*input_size, sizeof(float));

    // Ego encoder: Linear -> LayerNorm -> ReLU -> Linear
    net->ego_encoder = make_linear(weights, num_agents, ego_dim, input_size);
    net->ego_layernorm = make_layernorm(weights, num_agents, input_size);
    net->ego_relu = make_relu(num_agents, input_size);
    net->ego_encoder_two = make_linear(weights, num_agents, input_size, input_size);

    // Road encoder: Linear -> LayerNorm -> Linear (no ReLU in between)
    net->road_encoder = make_linear(weights, num_agents, ROAD_FEATURES_AFTER_ONEHOT, input_size);
    net->road_layernorm = make_layernorm(weights, num_agents, input_size);
    net->road_encoder_two = make_linear(weights, num_agents, input_size, input_size);

    // Partner encoder: Linear -> LayerNorm -> ReLU -> Linear
    net->partner_encoder = make_linear(weights, num_agents, PARTNER_FEATURES, input_size);
    net->partner_layernorm = make_layernorm(weights, num_agents, input_size);
    net->partner_relu = make_relu(num_agents, input_size);
    net->partner_encoder_two = make_linear(weights, num_agents, input_size, input_size);

    // Full scene encoder: Linear -> LayerNorm
    net->full_scene_encoder = make_linear(weights, num_agents, obs_size, input_size);
    net->full_scene_layernorm = make_layernorm(weights, num_agents, input_size);

    net->partner_max = make_max_dim1(num_agents, MAX_PARTNER_OBJECTS, input_size);
    net->road_max = make_max_dim1(num_agents, MAX_ROAD_OBJECTS, input_size);

    // Concatenation: ego + road + partner + full_scene = 4 * input_size
    net->cat1 = make_cat_dim1(num_agents, input_size, input_size);
    net->cat2 = make_cat_dim1(num_agents, input_size*2, input_size);
    net->cat3 = make_cat_dim1(num_agents, input_size*3, input_size);

    net->gelu = make_gelu(num_agents, input_size*4);

    // Shared embedding includes GELU internally, followed by Linear
    net->shared_embedding = make_linear(weights, num_agents, input_size*4, hidden_size);
    net->relu = make_relu(num_agents, hidden_size);

    net->actor = make_linear(weights, num_agents, hidden_size, action_size);
    net->value_fn = make_linear(weights, num_agents, hidden_size, 1);
    net->lstm = make_lstm(weights, num_agents, hidden_size, HIDDEN_SIZE);
    memset(net->lstm->state_h, 0, num_agents*HIDDEN_SIZE*sizeof(float));
    memset(net->lstm->state_c, 0, num_agents*HIDDEN_SIZE*sizeof(float));
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, action_dim);
    return net;
}

void free_drivenet(DriveNet* net) {
    free(net->obs_self);
    free(net->obs_partner);
    free(net->obs_road);
    free(net->obs_full);
    free(net->ego_relu_output);
    free(net->partner_linear_output);
    free(net->road_linear_output);
    free(net->partner_linear_output_two);
    free(net->road_linear_output_two);
    free(net->partner_layernorm_output);
    free(net->road_layernorm_output);
    free(net->partner_relu_output);
    free(net->ego_encoder);
    free(net->road_encoder);
    free(net->partner_encoder);
    free(net->ego_layernorm);
    free(net->road_layernorm);
    free(net->partner_layernorm);
    free(net->ego_relu);
    free(net->partner_relu);
    free(net->ego_encoder_two);
    free(net->road_encoder_two);
    free(net->partner_encoder_two);
    free(net->full_scene_encoder);
    free(net->full_scene_layernorm);
    free(net->partner_max);
    free(net->road_max);
    free(net->cat1);
    free(net->cat2);
    free(net->cat3);
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

    // Clear previous observations
    memset(net->obs_self, 0, net->num_agents * ego_dim * sizeof(float));
    memset(net->obs_partner, 0, net->num_agents * MAX_PARTNER_OBJECTS * PARTNER_FEATURES * sizeof(float));
    memset(net->obs_road, 0, net->num_agents * MAX_ROAD_OBJECTS * ROAD_FEATURES_AFTER_ONEHOT * sizeof(float));
    memset(net->obs_full, 0, net->num_agents * net->obs_size * sizeof(float));

    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b * (ego_dim + MAX_PARTNER_OBJECTS*PARTNER_FEATURES + MAX_ROAD_OBJECTS*ROAD_FEATURES);
        int partner_offset = b_offset + ego_dim;
        int road_offset = b_offset + ego_dim + MAX_PARTNER_OBJECTS*PARTNER_FEATURES;

        // Process self observation
        for(int i = 0; i < ego_dim; i++) {
            net->obs_self[b * ego_dim + i] = observations[b_offset + i];
            net->obs_full[b * net->obs_size + i] = observations[b_offset + i];
        }

        // Process partner observation
        for(int i = 0; i < MAX_PARTNER_OBJECTS; i++) {
            for(int j = 0; j < PARTNER_FEATURES; j++) {
                net->obs_partner[b*MAX_PARTNER_OBJECTS*PARTNER_FEATURES + i*PARTNER_FEATURES + j] = observations[partner_offset + i*PARTNER_FEATURES + j];
                net->obs_full[b * net->obs_size + ego_dim + i*PARTNER_FEATURES + j] = observations[partner_offset + i*PARTNER_FEATURES + j];
            }
        }

        // Process road observation (convert to one-hot)
        for(int i = 0; i < MAX_ROAD_OBJECTS; i++) {
            for(int j = 0; j < ROAD_FEATURES - 1; j++) {
                net->obs_road[b*MAX_ROAD_OBJECTS*ROAD_FEATURES_AFTER_ONEHOT + i*ROAD_FEATURES_AFTER_ONEHOT + j] = observations[road_offset + i*ROAD_FEATURES + j];
            }
            // One-hot encode the categorical feature (last feature)
            int categorical = (int)observations[road_offset + i*ROAD_FEATURES + (ROAD_FEATURES - 1)];
            for(int j = 0; j < 7; j++) {
                if(j == categorical) {
                    net->obs_road[b*MAX_ROAD_OBJECTS*ROAD_FEATURES_AFTER_ONEHOT + i*ROAD_FEATURES_AFTER_ONEHOT + 6 + j] = 1.0f;
                } else {
                    net->obs_road[b*MAX_ROAD_OBJECTS*ROAD_FEATURES_AFTER_ONEHOT + i*ROAD_FEATURES_AFTER_ONEHOT + 6 + j] = 0.0f;
                }
            }

            // Store original features in obs_full (not one-hot)
            for(int j = 0; j < ROAD_FEATURES; j++) {
                net->obs_full[b * net->obs_size + ego_dim + MAX_PARTNER_OBJECTS*PARTNER_FEATURES + i*ROAD_FEATURES + j] = observations[road_offset + i*ROAD_FEATURES + j];
            }
        }
    }

    // Ego encoder: Linear -> LayerNorm -> ReLU -> Linear
    linear(net->ego_encoder, net->obs_self);
    layernorm(net->ego_layernorm, net->ego_encoder->output);
    relu(net->ego_relu, net->ego_layernorm->output);
    linear(net->ego_encoder_two, net->ego_relu->output);

    // Partner encoder: Linear -> LayerNorm -> ReLU -> Linear (per object)
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < MAX_PARTNER_OBJECTS; obj++) {
            float* obj_features = &net->obs_partner[b*MAX_PARTNER_OBJECTS*PARTNER_FEATURES + obj*PARTNER_FEATURES];
            _linear(obj_features, net->partner_encoder->weights, net->partner_encoder->bias,
                   &net->partner_linear_output[b*MAX_PARTNER_OBJECTS*INPUT_SIZE + obj*INPUT_SIZE],
                   1, PARTNER_FEATURES, INPUT_SIZE);
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < MAX_PARTNER_OBJECTS; obj++) {
            float* after_first = &net->partner_linear_output[b*MAX_PARTNER_OBJECTS*INPUT_SIZE + obj*INPUT_SIZE];
            _layernorm(after_first, net->partner_layernorm->weights, net->partner_layernorm->bias,
                        &net->partner_layernorm_output[b*MAX_PARTNER_OBJECTS*INPUT_SIZE + obj*INPUT_SIZE],
                        1, INPUT_SIZE);
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < MAX_PARTNER_OBJECTS; obj++) {
            float* after_norm = &net->partner_layernorm_output[b*MAX_PARTNER_OBJECTS*INPUT_SIZE + obj*INPUT_SIZE];
            _relu(after_norm, &net->partner_relu_output[b*MAX_PARTNER_OBJECTS*INPUT_SIZE + obj*INPUT_SIZE],
                INPUT_SIZE);
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < MAX_PARTNER_OBJECTS; obj++) {
            float* after_relu = &net->partner_relu_output[b*MAX_PARTNER_OBJECTS*INPUT_SIZE + obj*INPUT_SIZE];
            _linear(after_relu, net->partner_encoder_two->weights, net->partner_encoder_two->bias,
                   &net->partner_linear_output_two[b*MAX_PARTNER_OBJECTS*INPUT_SIZE + obj*INPUT_SIZE],
                   1, INPUT_SIZE, INPUT_SIZE);
        }
    }

    // Road encoder: Linear -> LayerNorm -> Linear (per object, NO ReLU)
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < MAX_ROAD_OBJECTS; obj++) {
            float* obj_features = &net->obs_road[b*MAX_ROAD_OBJECTS*ROAD_FEATURES_AFTER_ONEHOT + obj*ROAD_FEATURES_AFTER_ONEHOT];
            _linear(obj_features, net->road_encoder->weights, net->road_encoder->bias,
                   &net->road_linear_output[b*MAX_ROAD_OBJECTS*INPUT_SIZE + obj*INPUT_SIZE],
                   1, ROAD_FEATURES_AFTER_ONEHOT, INPUT_SIZE);
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < MAX_ROAD_OBJECTS; obj++) {
            float* after_first = &net->road_linear_output[b*MAX_ROAD_OBJECTS*INPUT_SIZE + obj*INPUT_SIZE];
            _layernorm(after_first, net->road_layernorm->weights, net->road_layernorm->bias,
                        &net->road_layernorm_output[b*MAX_ROAD_OBJECTS*INPUT_SIZE + obj*INPUT_SIZE],
                        1, INPUT_SIZE);
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < MAX_ROAD_OBJECTS; obj++) {
            float* after_norm = &net->road_layernorm_output[b*MAX_ROAD_OBJECTS*INPUT_SIZE + obj*INPUT_SIZE];
            _linear(after_norm, net->road_encoder_two->weights, net->road_encoder_two->bias,
                    &net->road_linear_output_two[b*MAX_ROAD_OBJECTS*INPUT_SIZE + obj*INPUT_SIZE],
                    1, INPUT_SIZE, INPUT_SIZE);
        }
    }

    // Full scene encoder: Linear -> LayerNorm
    linear(net->full_scene_encoder, net->obs_full);
    layernorm(net->full_scene_layernorm, net->full_scene_encoder->output);

    // Max pooling over objects
    max_dim1(net->partner_max, net->partner_linear_output_two);
    max_dim1(net->road_max, net->road_linear_output_two);

    // Concatenate: ego + road + partner + full_scene
    cat_dim1(net->cat1, net->ego_encoder_two->output, net->road_max->output);
    cat_dim1(net->cat2, net->cat1->output, net->partner_max->output);
    cat_dim1(net->cat3, net->cat2->output, net->full_scene_layernorm->output);

    // Shared embedding: GELU -> Linear
    gelu(net->gelu, net->cat3->output);
    linear(net->shared_embedding, net->gelu->output);
    relu(net->relu, net->shared_embedding->output);

    // LSTM and output
    lstm(net->lstm, net->relu->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);

    // Get action
    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);
}
