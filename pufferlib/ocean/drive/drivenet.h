#include <time.h>
#include "drive.h"
#include "puffernet.h"
#include <math.h>
#include <raylib.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

// Must match Python Drive(input_size=64, hidden_size=256) + LSTMWrapper(input_size=256, hidden_size=256)
#define NN_INPUT_SIZE 64
#define NN_HIDDEN_SIZE 256

typedef struct DriveNet DriveNet;
struct DriveNet {
    int num_agents;
    int ego_dim;
    int action_size;            // logits per step (91 for CLASSIC, 12 for JERK)
    int traj_len;               // actions_trajectory_length (from INI config)
    int past_actions_input_dim; // traj_len * 2 (accel_idx + steer_idx per step)
    int actor_output_dim;       // traj_len * action_size

    // Observation parse buffers
    float *obs_self;
    float *obs_partner;
    float *obs_road;

    // Per-object intermediate buffers (partner / road)
    float *partner_linear_output;
    float *road_linear_output;
    float *partner_layernorm_output;
    float *road_layernorm_output;
    float *partner_linear_output_two;
    float *road_linear_output_two;

    // Past-actions state: [num_agents * past_actions_input_dim]
    // Stores (accel_idx, steer_idx) pairs for each of traj_len steps.
    float *past_actions_traj;

    // Per-step argmax action indices for trajectory visualisation.
    // Shape [num_agents * traj_len].
    int *pred_actions_traj;

    // ----- Layers: weight-loading order MUST match Python named_parameters() -----
    // Weight export order: LSTMWrapper.named_parameters() yields policy.* then lstm.*

    // 1. ego_encoder (Linear -> LayerNorm -> Linear)
    Linear *ego_encoder;
    LayerNorm *ego_layernorm;
    Linear *ego_encoder_two;

    // 2. road_encoder (Linear -> LayerNorm -> Linear)
    Linear *road_encoder;
    LayerNorm *road_layernorm;
    Linear *road_encoder_two;

    // 3. partner_encoder (Linear -> LayerNorm -> Linear)
    Linear *partner_encoder;
    LayerNorm *partner_layernorm;
    Linear *partner_encoder_two;

    // 4. shared_embedding: Python nn.Sequential(GELU(), Linear(4*64=256, 256))
    Linear *shared_embedding;

    // 5. past_actions_encoder (Linear -> LayerNorm -> Linear)
    Linear *past_act_encoder;
    LayerNorm *past_act_layernorm;
    Linear *past_act_encoder_two;

    // 6. actor -- output is full trajectory: [num_agents, actor_output_dim]
    Linear *actor;
    // 7. value_fn
    Linear *value_fn;

    // 8. LSTM
    LSTM *lstm;

    // Aggregators / activations (no learnable weights)
    MaxDim1 *partner_max;
    MaxDim1 *road_max;
    CatDim1 *cat1; // ego(64)  + road(64)     -> 128
    CatDim1 *cat2; // 128      + partner(64)   -> 192
    CatDim1 *cat3; // 192      + past_act(64)  -> 256
    GELU *gelu;
    ReLU *relu;

    Multidiscrete *multidiscrete;
};

DriveNet *init_drivenet(Weights *weights, int num_agents, int dynamics_model, int reward_conditioning, int traj_len) {
    DriveNet *net = calloc(1, sizeof(DriveNet));

    int ego_dim = (dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    if (reward_conditioning) {
        ego_dim += NUM_REWARD_COEFS;
    }

    int max_partners = MAX_AGENTS - 1;
    int max_road_obs = MAX_ROAD_SEGMENT_OBSERVATIONS;
    int partner_features = PARTNER_FEATURES;
    int road_feat_onehot = ROAD_FEATURES + 6;
    int input_size = NN_INPUT_SIZE;   // 64
    int hidden_size = NN_HIDDEN_SIZE; // 256

    int action_size;
    int logit_sizes[2];
    int action_dim = 1;
    if (dynamics_model == CLASSIC) {
        action_size = 7 * 13;
        logit_sizes[0] = 7 * 13;
    } else {
        action_size = 4 * 3;
        logit_sizes[0] = 4 * 3;
    }
    int past_actions_input_dim = traj_len * 2;
    int actor_output_dim = traj_len * action_size;

    net->num_agents = num_agents;
    net->ego_dim = ego_dim;
    net->action_size = action_size;
    net->traj_len = traj_len;
    net->past_actions_input_dim = past_actions_input_dim;
    net->actor_output_dim = actor_output_dim;

    // Observation parse buffers
    net->obs_self = calloc(num_agents * ego_dim, sizeof(float));
    net->obs_partner = calloc(num_agents * max_partners * partner_features, sizeof(float));
    net->obs_road = calloc(num_agents * max_road_obs * road_feat_onehot, sizeof(float));

    net->partner_linear_output = calloc(num_agents * max_partners * input_size, sizeof(float));
    net->road_linear_output = calloc(num_agents * max_road_obs * input_size, sizeof(float));
    net->partner_linear_output_two = calloc(num_agents * max_partners * input_size, sizeof(float));
    net->road_linear_output_two = calloc(num_agents * max_road_obs * input_size, sizeof(float));
    net->partner_layernorm_output = calloc(num_agents * max_partners * input_size, sizeof(float));
    net->road_layernorm_output = calloc(num_agents * max_road_obs * input_size, sizeof(float));

    // Zero-init: "no previous trajectory" on first step
    net->past_actions_traj = calloc(num_agents * past_actions_input_dim, sizeof(float));
    net->pred_actions_traj = calloc(num_agents * traj_len, sizeof(int));

    // ----------------------------------------------------------------
    // Weight loading -- order MUST match Python LSTMWrapper.named_parameters()
    // which yields policy.* parameters first, then lstm.* parameters.
    // ----------------------------------------------------------------

    // 1. ego_encoder
    net->ego_encoder = make_linear(weights, num_agents, ego_dim, input_size);
    net->ego_layernorm = make_layernorm(weights, num_agents, input_size);
    net->ego_encoder_two = make_linear(weights, num_agents, input_size, input_size);

    // 2. road_encoder
    net->road_encoder = make_linear(weights, num_agents, road_feat_onehot, input_size);
    net->road_layernorm = make_layernorm(weights, num_agents, input_size);
    net->road_encoder_two = make_linear(weights, num_agents, input_size, input_size);

    // 3. partner_encoder
    net->partner_encoder = make_linear(weights, num_agents, partner_features, input_size);
    net->partner_layernorm = make_layernorm(weights, num_agents, input_size);
    net->partner_encoder_two = make_linear(weights, num_agents, input_size, input_size);

    // 4. shared_embedding (GELU at index 0 has no weights, Linear at index 1)
    net->shared_embedding = make_linear(weights, num_agents, 4 * input_size, hidden_size);

    // 5. past_actions_encoder
    net->past_act_encoder = make_linear(weights, num_agents, past_actions_input_dim, input_size);
    net->past_act_layernorm = make_layernorm(weights, num_agents, input_size);
    net->past_act_encoder_two = make_linear(weights, num_agents, input_size, input_size);

    // 6. actor -- output is full trajectory
    net->actor = make_linear(weights, num_agents, hidden_size, actor_output_dim);
    // 7. value_fn
    net->value_fn = make_linear(weights, num_agents, hidden_size, 1);

    // 8. LSTM
    net->lstm = make_lstm(weights, num_agents, hidden_size, hidden_size);

    // Aggregators / activations (no weight consumption)
    net->partner_max = make_max_dim1(num_agents, max_partners, input_size);
    net->road_max = make_max_dim1(num_agents, max_road_obs, input_size);
    net->cat1 = make_cat_dim1(num_agents, input_size, input_size);     // -> 128
    net->cat2 = make_cat_dim1(num_agents, 2 * input_size, input_size); // -> 192
    net->cat3 = make_cat_dim1(num_agents, 3 * input_size, input_size); // -> 256
    net->gelu = make_gelu(num_agents, 4 * input_size);
    net->relu = make_relu(num_agents, hidden_size);

    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, action_dim);
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
    free(net->past_actions_traj);
    free(net->pred_actions_traj);
    free(net->ego_encoder);
    free(net->ego_layernorm);
    free(net->ego_encoder_two);
    free(net->road_encoder);
    free(net->road_layernorm);
    free(net->road_encoder_two);
    free(net->partner_encoder);
    free(net->partner_layernorm);
    free(net->partner_encoder_two);
    free(net->shared_embedding);
    free(net->past_act_encoder);
    free(net->past_act_layernorm);
    free(net->past_act_encoder_two);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net->partner_max);
    free(net->road_max);
    free(net->cat1);
    free(net->cat2);
    free(net->cat3);
    free(net->gelu);
    free(net->relu);
    free(net->multidiscrete);
    free(net);
}

void forward(DriveNet *net, float *observations, int *actions) {
    int ego_dim = net->ego_dim;
    int max_partners = MAX_AGENTS - 1;
    int max_road_obs = MAX_ROAD_SEGMENT_OBSERVATIONS;
    int partner_features = PARTNER_FEATURES;
    int road_feat_onehot = ROAD_FEATURES + 6;
    int action_size = net->action_size;
    int actor_output_dim = net->actor_output_dim;
    int num_steer = (action_size == 7 * 13) ? 13 : 3; // CLASSIC: 13 steering, JERK: 3 lateral

    // Clear observation parse buffers
    memset(net->obs_self, 0, net->num_agents * ego_dim * sizeof(float));
    memset(net->obs_partner, 0, net->num_agents * max_partners * partner_features * sizeof(float));
    memset(net->obs_road, 0, net->num_agents * max_road_obs * road_feat_onehot * sizeof(float));

    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b * (ego_dim + max_partners * partner_features + max_road_obs * ROAD_FEATURES);
        int partner_offset = b_offset + ego_dim;
        int road_offset = b_offset + ego_dim + max_partners * partner_features;

        // Ego
        for (int i = 0; i < ego_dim; i++) {
            net->obs_self[b * ego_dim + i] = observations[b_offset + i];
        }

        // Partners
        for (int i = 0; i < max_partners; i++) {
            for (int j = 0; j < partner_features; j++) {
                net->obs_partner[b * max_partners * partner_features + i * partner_features + j] =
                    observations[partner_offset + i * partner_features + j];
            }
        }

        // Road: copy raw features then overwrite last feature with 7-way one-hot
        for (int i = 0; i < max_road_obs; i++) {
            for (int j = 0; j < ROAD_FEATURES; j++) {
                net->obs_road[b * max_road_obs * road_feat_onehot + i * road_feat_onehot + j] =
                    observations[road_offset + i * ROAD_FEATURES + j];
            }
            int cat_class = (int)observations[road_offset + i * ROAD_FEATURES + (ROAD_FEATURES - 1)];
            for (int j = 0; j < 7; j++) {
                net->obs_road[b * max_road_obs * road_feat_onehot + i * road_feat_onehot + (ROAD_FEATURES - 1) + j] =
                    (j == cat_class) ? 1.0f : 0.0f;
            }
        }
    }

    // ---- Ego encoder ----
    linear(net->ego_encoder, net->obs_self);
    layernorm(net->ego_layernorm, net->ego_encoder->output);
    linear(net->ego_encoder_two, net->ego_layernorm->output);

    // ---- Partner encoder (per object, then max-pool) ----
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_partners; obj++) {
            float *feat = &net->obs_partner[b * max_partners * partner_features + obj * partner_features];
            _linear(feat, net->partner_encoder->weights, net->partner_encoder->bias,
                    &net->partner_linear_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    partner_features, NN_INPUT_SIZE);
        }
    }
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_partners; obj++) {
            _layernorm(&net->partner_linear_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE],
                       net->partner_layernorm->weights, net->partner_layernorm->bias,
                       &net->partner_layernorm_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                       NN_INPUT_SIZE);
        }
    }
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_partners; obj++) {
            float *feat = &net->partner_layernorm_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE];
            _linear(feat, net->partner_encoder_two->weights, net->partner_encoder_two->bias,
                    &net->partner_linear_output_two[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    NN_INPUT_SIZE, NN_INPUT_SIZE);
        }
    }

    // ---- Road encoder (per object, then max-pool) ----
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_road_obs; obj++) {
            float *feat = &net->obs_road[b * max_road_obs * road_feat_onehot + obj * road_feat_onehot];
            _linear(feat, net->road_encoder->weights, net->road_encoder->bias,
                    &net->road_linear_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    road_feat_onehot, NN_INPUT_SIZE);
        }
    }
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_road_obs; obj++) {
            _layernorm(&net->road_linear_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE],
                       net->road_layernorm->weights, net->road_layernorm->bias,
                       &net->road_layernorm_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                       NN_INPUT_SIZE);
        }
    }
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_road_obs; obj++) {
            float *feat = &net->road_layernorm_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE];
            _linear(feat, net->road_encoder_two->weights, net->road_encoder_two->bias,
                    &net->road_linear_output_two[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    NN_INPUT_SIZE, NN_INPUT_SIZE);
        }
    }

    // ---- Max-pool over objects ----
    max_dim1(net->partner_max, net->partner_linear_output_two);
    max_dim1(net->road_max, net->road_linear_output_two);

    // ---- Past-actions encoder ----
    linear(net->past_act_encoder, net->past_actions_traj);
    layernorm(net->past_act_layernorm, net->past_act_encoder->output);
    linear(net->past_act_encoder_two, net->past_act_layernorm->output);

    // ---- 4-way concat: ego + road + partner + past_act -> 256 ----
    cat_dim1(net->cat1, net->ego_encoder_two->output, net->road_max->output);
    cat_dim1(net->cat2, net->cat1->output, net->partner_max->output);
    cat_dim1(net->cat3, net->cat2->output, net->past_act_encoder_two->output);

    // ---- GELU -> shared_embedding(256->256) -> ReLU ----
    gelu(net->gelu, net->cat3->output);
    linear(net->shared_embedding, net->gelu->output);
    relu(net->relu, net->shared_embedding->output);

    // ---- LSTM(256->256) ----
    lstm(net->lstm, net->relu->output);

    // ---- Actor: [num_agents, actor_output_dim] ----
    linear(net->actor, net->lstm->state_h);
    // ---- Value ----
    linear(net->value_fn, net->lstm->state_h);

    // ---- Extract step-0 action + per-step argmax for trajectory ----
    for (int b = 0; b < net->num_agents; b++) {
        float *logits_b = &net->actor->output[b * actor_output_dim];

        // Step 0 argmax -> actual env action
        int best = 0;
        float best_val = logits_b[0];
        for (int a = 1; a < action_size; a++) {
            if (logits_b[a] > best_val) {
                best_val = logits_b[a];
                best = a;
            }
        }
        actions[b] = best;

        // Per-step argmax for trajectory visualisation + update past_actions_traj
        int tl = net->traj_len;
        for (int t = 0; t < tl; t++) {
            float *step_logits = &logits_b[t * action_size];
            int step_best = 0;
            float step_best_val = step_logits[0];
            for (int a = 1; a < action_size; a++) {
                if (step_logits[a] > step_best_val) {
                    step_best_val = step_logits[a];
                    step_best = a;
                }
            }
            net->pred_actions_traj[b * tl + t] = step_best;

            // Update past_actions_traj with (accel_idx, steer_idx) for next forward call
            // Matches Python: convert_action_integers_to_r2(action) -> (action // num_steer, action % num_steer)
            net->past_actions_traj[b * net->past_actions_input_dim + t * 2 + 0] = (float)(step_best / num_steer);
            net->past_actions_traj[b * net->past_actions_input_dim + t * 2 + 1] = (float)(step_best % num_steer);
        }
    }
}
