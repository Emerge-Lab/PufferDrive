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
#define ACTIONS_TRAJECTORY_LENGTH 10
#define PAST_ACTIONS_DIM (ACTIONS_TRAJECTORY_LENGTH * 2) // steps × 2 action dims

typedef struct DriveNet DriveNet;
struct DriveNet {
    int num_agents;
    int ego_dim;
    int action_size;     // per-step action size (e.g. 91 for classic)
    int trajectory_len;  // number of trajectory steps
    float *obs_self;
    float *obs_partner;
    float *obs_road;
    float *obs_past_actions;
    float *partner_linear_output;
    float *road_linear_output;
    float *partner_layernorm_output;
    float *road_layernorm_output;
    float *partner_linear_output_two;
    float *road_linear_output_two;
    Linear *ego_encoder;
    Linear *road_encoder;
    Linear *partner_encoder;
    Linear *past_actions_encoder;
    LayerNorm *ego_layernorm;
    LayerNorm *road_layernorm;
    LayerNorm *partner_layernorm;
    Linear *ego_encoder_two;
    Linear *road_encoder_two;
    Linear *partner_encoder_two;
    Linear *past_actions_encoder_two;
    ReLU *past_actions_relu;
    MaxDim1 *partner_max;
    MaxDim1 *road_max;
    CatDim1 *cat1;
    CatDim1 *cat2;
    CatDim1 *cat3;
    GELU *gelu;
    Linear *shared_embedding;
    ReLU *relu;
    LSTM *lstm;
    Linear *actor;
    Linear *value_fn;
    Multidiscrete *multidiscrete;
    // Predicted trajectory (rolled out from actor output)
    float *predicted_trajectory_x;
    float *predicted_trajectory_y;
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

    int traj_len = ACTIONS_TRAJECTORY_LENGTH;
    int total_actor_output = traj_len * action_size;

    net->num_agents = num_agents;
    net->ego_dim = ego_dim;
    net->action_size = action_size;
    net->trajectory_len = traj_len;
    net->obs_self = calloc(num_agents * ego_dim, sizeof(float));
    net->obs_partner = calloc(num_agents * max_partners * partner_features, sizeof(float));
    net->obs_road = calloc(num_agents * max_road_obs * road_feat_onehot, sizeof(float));
    net->obs_past_actions = calloc(num_agents * PAST_ACTIONS_DIM, sizeof(float));
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
    net->cat3 = make_cat_dim1(num_agents, input_size * 3, input_size);
    net->gelu = make_gelu(num_agents, 4 * input_size);
    // shared_embedding must be consumed before past_actions_encoder to match Python weight order
    net->shared_embedding = make_linear(weights, num_agents, input_size * 4, hidden_size);
    // Past actions encoder: Linear(PAST_ACTIONS_DIM → input_size) + ReLU + Linear(input_size → input_size)
    net->past_actions_encoder = make_linear(weights, num_agents, PAST_ACTIONS_DIM, input_size);
    net->past_actions_relu = make_relu(num_agents, input_size);
    net->past_actions_encoder_two = make_linear(weights, num_agents, input_size, input_size);
    net->relu = make_relu(num_agents, hidden_size);
    net->actor = make_linear(weights, num_agents, hidden_size, total_actor_output);
    net->value_fn = make_linear(weights, num_agents, hidden_size, 1);
    net->lstm = make_lstm(weights, num_agents, hidden_size, NN_HIDDEN_SIZE);
    memset(net->lstm->state_h, 0, num_agents * NN_HIDDEN_SIZE * sizeof(float));
    memset(net->lstm->state_c, 0, num_agents * NN_HIDDEN_SIZE * sizeof(float));
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, action_dim);
    // Trajectory rollout buffers
    net->predicted_trajectory_x = calloc(num_agents * traj_len, sizeof(float));
    net->predicted_trajectory_y = calloc(num_agents * traj_len, sizeof(float));
    return net;
}

void free_drivenet(DriveNet *net) {
    free(net->obs_self);
    free(net->obs_partner);
    free(net->obs_road);
    free(net->obs_past_actions);
    free(net->partner_linear_output);
    free(net->road_linear_output);
    free(net->partner_linear_output_two);
    free(net->road_linear_output_two);
    free(net->partner_layernorm_output);
    free(net->road_layernorm_output);
    free(net->ego_encoder);
    free(net->road_encoder);
    free(net->partner_encoder);
    free(net->past_actions_encoder);
    free(net->ego_layernorm);
    free(net->road_layernorm);
    free(net->partner_layernorm);
    free(net->ego_encoder_two);
    free(net->road_encoder_two);
    free(net->partner_encoder_two);
    free(net->past_actions_encoder_two);
    free(net->past_actions_relu);
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
    free(net->predicted_trajectory_x);
    free(net->predicted_trajectory_y);
    free(net);
}

void forward(DriveNet *net, Drive *env, float *observations, int *actions) {
    int ego_dim = net->ego_dim;
    int max_partners = MAX_AGENTS - 1;
    int max_road_obs = MAX_ROAD_SEGMENT_OBSERVATIONS;
    int partner_features = PARTNER_FEATURES;
    int road_features = ROAD_FEATURES;
    int road_feat_onehot = road_features + 6; // one-hot extra 6 features for road
    int obs_stride = ego_dim + max_partners * partner_features + max_road_obs * road_features + PAST_ACTIONS_DIM;

    // Clear previous observations
    memset(net->obs_self, 0, net->num_agents * ego_dim * sizeof(float));
    memset(net->obs_partner, 0, net->num_agents * max_partners * partner_features * sizeof(float));
    memset(net->obs_road, 0, net->num_agents * max_road_obs * road_feat_onehot * sizeof(float));
    memset(net->obs_past_actions, 0, net->num_agents * PAST_ACTIONS_DIM * sizeof(float));

    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b * obs_stride;
        int partner_offset = b_offset + ego_dim;
        int road_offset = b_offset + ego_dim + max_partners * partner_features;
        int past_actions_offset = road_offset + max_road_obs * road_features;

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

        // Process past actions trajectory
        for (int i = 0; i < PAST_ACTIONS_DIM; i++) {
            net->obs_past_actions[b * PAST_ACTIONS_DIM + i] = observations[past_actions_offset + i];
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

    // Past actions encoder: Linear → ReLU → Linear
    linear(net->past_actions_encoder, net->obs_past_actions);
    relu(net->past_actions_relu, net->past_actions_encoder->output);
    linear(net->past_actions_encoder_two, net->past_actions_relu->output);

    max_dim1(net->partner_max, net->partner_linear_output_two);
    max_dim1(net->road_max, net->road_linear_output_two);
    // Concatenate: [ego, road, partner, past_actions] (4 × input_size)
    cat_dim1(net->cat1, net->ego_encoder_two->output, net->road_max->output);
    cat_dim1(net->cat2, net->cat1->output, net->partner_max->output);
    cat_dim1(net->cat3, net->cat2->output, net->past_actions_encoder_two->output);
    gelu(net->gelu, net->cat3->output);
    linear(net->shared_embedding, net->gelu->output);
    relu(net->relu, net->shared_embedding->output);
    lstm(net->lstm, net->relu->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);

    // Actor output is [num_agents × (trajectory_len × action_size)]
    // Take first timestep's action via softmax_multidiscrete
    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);

    // Roll out predicted trajectory using classic dynamics
    if (env != NULL) {
        int traj_len = net->trajectory_len;
        int action_size = net->action_size;
        int num_steer = sizeof(STEERING_VALUES) / sizeof(STEERING_VALUES[0]);

        for (int b = 0; b < net->num_agents; b++) {
            int agent_idx = env->active_agent_indices[b];
            Agent *agent = &env->agents[agent_idx];

            // Start from current agent state
            float x = agent->sim_x;
            float y = agent->sim_y;
            float heading = agent->sim_heading;
            float vx = agent->sim_vx;
            float vy = agent->sim_vy;

            float *actor_out = &net->actor->output[b * traj_len * action_size];

            for (int t = 0; t < traj_len; t++) {
                // Find argmax action for this timestep
                float *logits = &actor_out[t * action_size];
                int best_action = 0;
                float best_val = logits[0];
                for (int a = 1; a < action_size; a++) {
                    if (logits[a] > best_val) {
                        best_val = logits[a];
                        best_action = a;
                    }
                }

                // Decode action into acceleration + steering (classic model)
                int accel_idx = best_action / num_steer;
                int steer_idx = best_action % num_steer;
                float acceleration = ACCELERATION_VALUES[accel_idx];
                float steering = STEERING_VALUES[steer_idx];

                // Classic dynamics step
                float speed_mag = sqrtf(vx * vx + vy * vy);
                float v_dot = vx * cosf(heading) + vy * sinf(heading);
                float signed_speed = copysignf(speed_mag, v_dot);
                signed_speed = signed_speed + acceleration * env->dt;
                signed_speed = clipSpeed(signed_speed);
                float beta = tanhf(0.5f * tanf(steering));
                float new_vx = signed_speed * cosf(heading + beta);
                float new_vy = signed_speed * sinf(heading + beta);
                float yaw_rate = (signed_speed * cosf(beta) * tanf(steering)) / agent->sim_length;

                x += new_vx * env->dt;
                y += new_vy * env->dt;
                heading += yaw_rate * env->dt;
                vx = new_vx;
                vy = new_vy;

                net->predicted_trajectory_x[b * traj_len + t] = x;
                net->predicted_trajectory_y[b * traj_len + t] = y;
            }
        }
    }
}
