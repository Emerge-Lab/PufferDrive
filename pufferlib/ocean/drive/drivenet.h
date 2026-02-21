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
// These come from [policy] and [rnn] sections of drive.ini respectively.
#define NN_INPUT_SIZE 64   // Drive encoder output size (policy.input_size)
#define NN_HIDDEN_SIZE 256 // Drive shared_embedding output = LSTM input/hidden (policy.hidden_size)

typedef struct DriveNet DriveNet;
struct DriveNet {
    int num_agents;
    int ego_dim;
    int action_size;      // logits per step (91 for CLASSIC, 12 for JERK)
    int past_actions_dim; // action_size * PREDICTED_TRAJ_LEN

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

    // Past-actions state: raw actor logits from the previous step.
    // Shape [num_agents * past_actions_dim]. Zero-initialised = "no history".
    float *past_actions_traj;

    // Per-step argmax action indices for trajectory visualisation.
    // Shape [num_agents * PREDICTED_TRAJ_LEN].
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
    // GELU has no learnable params -> only the Linear weight chunk is consumed.
    Linear *shared_embedding;

    // 5. past_actions_encoder (Linear -> LayerNorm -> Linear)
    Linear *past_act_encoder;
    LayerNorm *past_act_layernorm;
    Linear *past_act_encoder_two;

    // 6. actor  -- output is full trajectory: [num_agents, past_actions_dim]
    Linear *actor;
    // 7. value_fn
    Linear *value_fn;

    // 8. LSTM: LSTMWrapper.lstm = nn.LSTM(input_size=256, hidden_size=256)
    LSTM *lstm;

    // Aggregators / activations (no learnable weights)
    MaxDim1 *partner_max;
    MaxDim1 *road_max;
    CatDim1 *cat1; // ego(64)  + road(64)    -> 128
    CatDim1 *cat2; // 128      + partner(64)  -> 192
    CatDim1 *cat3; // 192      + past_act(64) -> 256
    GELU *gelu;    // GELU(256)
    ReLU *relu;    // ReLU(256)

    Multidiscrete *multidiscrete;
};

DriveNet *init_drivenet(Weights *weights, int num_agents, int dynamics_model, int reward_conditioning) {
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
    int past_actions_dim = action_size * PREDICTED_TRAJ_LEN;

    net->num_agents = num_agents;
    net->ego_dim = ego_dim;
    net->action_size = action_size;
    net->past_actions_dim = past_actions_dim;

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
    net->past_actions_traj = calloc(num_agents * past_actions_dim, sizeof(float));
    net->pred_actions_traj = calloc(num_agents * PREDICTED_TRAJ_LEN, sizeof(int));

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

    // 4. shared_embedding (index 1 in Sequential; GELU at index 0 has no weights)
    //    Input: 4 * input_size = 4 * 64 = 256; Output: hidden_size = 256
    net->shared_embedding = make_linear(weights, num_agents, 4 * input_size, hidden_size);

    // 5. past_actions_encoder
    net->past_act_encoder = make_linear(weights, num_agents, past_actions_dim, input_size);
    net->past_act_layernorm = make_layernorm(weights, num_agents, input_size);
    net->past_act_encoder_two = make_linear(weights, num_agents, input_size, input_size);

    // 6. actor -- output is full trajectory: [num_agents, past_actions_dim]
    net->actor = make_linear(weights, num_agents, hidden_size, past_actions_dim);
    // 7. value_fn
    net->value_fn = make_linear(weights, num_agents, hidden_size, 1);

    // 8. LSTM: nn.LSTM(input_size=256, hidden_size=256)
    //    Loads: weight_ih_l0 [4*256, 256], weight_hh_l0 [4*256, 256],
    //           bias_ih_l0 [4*256], bias_hh_l0 [4*256]
    net->lstm = make_lstm(weights, num_agents, hidden_size, hidden_size);

    // Aggregators / activations (no weight consumption)
    net->partner_max = make_max_dim1(num_agents, max_partners, input_size);
    net->road_max = make_max_dim1(num_agents, max_road_obs, input_size);
    net->cat1 = make_cat_dim1(num_agents, input_size, input_size);     // -> 2*64 = 128
    net->cat2 = make_cat_dim1(num_agents, 2 * input_size, input_size); // -> 3*64 = 192
    net->cat3 = make_cat_dim1(num_agents, 3 * input_size, input_size); // -> 4*64 = 256
    net->gelu = make_gelu(num_agents, 4 * input_size);                 // GELU(256)
    net->relu = make_relu(num_agents, hidden_size);                    // ReLU(256)

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
    int past_actions_dim = net->past_actions_dim;

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
        // raw layout per object: [f0..f6, cat_class]  (ROAD_FEATURES = 8 values)
        // output layout per object: [f0..f6, oh0..oh6] (road_feat_onehot = 14 values)
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

    // ---- LSTM(256->256): takes encode_observations output, outputs hidden ----
    lstm(net->lstm, net->relu->output);
    // net->lstm->state_h is the 256-dim recurrent output

    // ---- Actor: takes LSTM hidden state, outputs [num_agents, past_actions_dim] ----
    linear(net->actor, net->lstm->state_h);

    // ---- Extract step-0 action + per-step argmax for trajectory ----
    for (int b = 0; b < net->num_agents; b++) {
        float *logits_b = &net->actor->output[b * past_actions_dim];

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

        // All steps argmax -> stored for rollout_trajectory()
        for (int t = 0; t < PREDICTED_TRAJ_LEN; t++) {
            float *logits_bt = logits_b + t * action_size;
            int best_t = 0;
            float best_val_t = logits_bt[0];
            for (int a = 1; a < action_size; a++) {
                if (logits_bt[a] > best_val_t) {
                    best_val_t = logits_bt[a];
                    best_t = a;
                }
            }
            net->pred_actions_traj[b * PREDICTED_TRAJ_LEN + t] = best_t;
        }
    }

    // ---- Update past_actions_traj for the next step ----
    memcpy(net->past_actions_traj, net->actor->output, net->num_agents * past_actions_dim * sizeof(float));
}
