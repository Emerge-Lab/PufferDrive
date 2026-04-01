#include <time.h>
#include "drive.h"
#include "puffernet.h"
#include <math.h>
#include <raylib.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

// Tanh activation (not in puffernet.h)
static inline void tanh_activation(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = tanhf(data[i]);
    }
}

// Gigaflow-style encoder: Linear → LayerNorm → Tanh → Linear
// Weights order matches PyTorch: encoder.0 (Linear), encoder.1 (LayerNorm), encoder.4 (Linear)
typedef struct {
    Linear *linear1;
    LayerNorm *layernorm;
    Linear *linear2;
    float *ln_output;
    int batch_size;
    int input_size;
} Encoder;

Encoder *make_encoder(Weights *weights, int batch_size, int in_features, int input_size) {
    Encoder *enc = calloc(1, sizeof(Encoder));
    enc->batch_size = batch_size;
    enc->input_size = input_size;
    enc->linear1 = make_linear(weights, batch_size, in_features, input_size);
    enc->layernorm = make_layernorm(weights, batch_size, input_size);
    enc->ln_output = calloc(batch_size * input_size, sizeof(float));
    enc->linear2 = make_linear(weights, batch_size, input_size, input_size);
    return enc;
}

// Apply encoder to flat input (batch_size x in_features)
void encoder_forward(Encoder *enc, float *input) {
    linear(enc->linear1, input);
    layernorm(enc->layernorm, enc->linear1->output);
    // Tanh after layernorm
    memcpy(enc->ln_output, enc->layernorm->output, enc->batch_size * enc->input_size * sizeof(float));
    tanh_activation(enc->ln_output, enc->batch_size * enc->input_size);
    // Second linear
    linear(enc->linear2, enc->ln_output);
}

// Apply encoder per-object: input is [batch_size * num_objects * features]
// Output is max-pooled: [batch_size * input_size]
void encoder_forward_objects(Encoder *enc, float *input, float *output,
                             int batch_size, int num_objects, int in_features, int input_size) {
    float *obj_encoded = calloc(batch_size * num_objects * input_size, sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        for (int obj = 0; obj < num_objects; obj++) {
            float *obj_in = &input[b * num_objects * in_features + obj * in_features];
            float *lin1_out = &obj_encoded[b * num_objects * input_size + obj * input_size];

            // Linear1
            _linear(obj_in, enc->linear1->weights, enc->linear1->bias, lin1_out, 1, in_features, input_size);

            // LayerNorm
            float ln_out[input_size];
            _layernorm(lin1_out, enc->layernorm->weights, enc->layernorm->bias, ln_out, 1, input_size);

            // Tanh
            tanh_activation(ln_out, input_size);

            // Linear2
            float *final_out = &obj_encoded[b * num_objects * input_size + obj * input_size];
            _linear(ln_out, enc->linear2->weights, enc->linear2->bias, final_out, 1, input_size, input_size);
        }
    }

    // Max pool over objects dimension
    for (int b = 0; b < batch_size; b++) {
        for (int f = 0; f < input_size; f++) {
            float max_val = -1e30f;
            for (int obj = 0; obj < num_objects; obj++) {
                float val = obj_encoded[b * num_objects * input_size + obj * input_size + f];
                if (val > max_val) max_val = val;
            }
            output[b * input_size + f] = max_val;
        }
    }

    free(obj_encoded);
}

void free_encoder(Encoder *enc) {
    free(enc->linear1);
    free(enc->layernorm);
    free(enc->linear2);
    free(enc->ln_output);
    free(enc);
}

typedef struct DriveNet DriveNet;
struct DriveNet {
    int num_agents;
    int ego_dim;
    int input_size;
    int backbone_hidden_size;

    // Observation dimensions
    int max_partner_obs;
    int partner_features;
    int max_lane_obs;
    int max_boundary_obs;
    int road_features;
    int max_traffic_light_obs;
    int traffic_light_features_raw;
    int traffic_light_features_onehot;
    int conditioning_dim;

    // Encoders
    Encoder *ego_encoder;
    Encoder *lane_encoder;
    Encoder *boundary_encoder;
    Encoder *partner_encoder;
    Encoder *traffic_light_encoder;
    Encoder *conditioning_encoder;

    // Temp buffers for per-object encoding + max pooling
    float *lane_pooled;
    float *boundary_pooled;
    float *partner_pooled;
    float *traffic_light_pooled;
    float *traffic_light_onehot_buf;

    // Backbone layers: GELU → Linear → GELU → Linear → ... → GELU
    int num_backbone_layers;
    GELU **backbone_gelus;
    Linear **backbone_linears;

    // Heads
    Linear *actor;
    Linear *value_fn;

    // Final GELU before heads
    GELU *final_gelu;

    // Action selection
    Multidiscrete *multidiscrete;

    // Concatenation buffer
    float *concat_buf;
};

DriveNet *init_drivenet(Weights *weights, int num_agents, int dynamics_model) {
    DriveNet *net = calloc(1, sizeof(DriveNet));

    // Architecture config (must match torch.py / drive.ini)
    int input_size = 64;
    int backbone_hidden_size = 512;
    int backbone_num_layers = 4;

    int ego_dim = (dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int partner_features = PARTNER_FEATURES;         // 8
    int road_features = ROAD_FEATURES;               // 7
    int traffic_light_features_raw = TRAFFIC_LIGHT_FEATURES; // 7
    int num_tl_states = 4; // NUM_TRAFFIC_LIGHT_STATES from datatypes.h
    int traffic_light_features_onehot = traffic_light_features_raw - 1 + num_tl_states; // 6+4=10

    // Observation counts (from drive.ini defaults)
    int max_partner_obs = 20;
    int max_lane_obs = 64;
    int max_boundary_obs = 32;
    int max_traffic_light_obs = 4;

    // Conditioning: target waypoints (static: 3 waypoints * 3 features = 9)
    int conditioning_dim = 9;

    // Action space
    int action_size, logit_sizes[1];
    if (dynamics_model == CLASSIC) {
        action_size = 7 * 9; // 7 accel * 9 steer
        logit_sizes[0] = action_size;
    } else { // JERK
        action_size = 4 * 3; // 4 longitudinal * 3 lateral = 12
        logit_sizes[0] = action_size;
    }

    net->num_agents = num_agents;
    net->ego_dim = ego_dim;
    net->input_size = input_size;
    net->backbone_hidden_size = backbone_hidden_size;
    net->max_partner_obs = max_partner_obs;
    net->partner_features = partner_features;
    net->max_lane_obs = max_lane_obs;
    net->max_boundary_obs = max_boundary_obs;
    net->road_features = road_features;
    net->max_traffic_light_obs = max_traffic_light_obs;
    net->traffic_light_features_raw = traffic_light_features_raw;
    net->traffic_light_features_onehot = traffic_light_features_onehot;
    net->conditioning_dim = conditioning_dim;

    // Create encoders (weight order must match PyTorch export)
    // actor_backbone.ego_encoder.{0,1,4}
    net->ego_encoder = make_encoder(weights, num_agents, ego_dim, input_size);
    // actor_backbone.lane_encoder.{0,1,4}
    net->lane_encoder = make_encoder(weights, num_agents, road_features, input_size);
    // actor_backbone.boundary_encoder.{0,1,4}
    net->boundary_encoder = make_encoder(weights, num_agents, road_features, input_size);
    // actor_backbone.partner_encoder.{0,1,4}
    net->partner_encoder = make_encoder(weights, num_agents, partner_features, input_size);
    // actor_backbone.traffic_light_encoder.{0,1,4}
    net->traffic_light_encoder = make_encoder(weights, num_agents, traffic_light_features_onehot, input_size);
    // actor_backbone.conditioning_encoder.{0,1,4}
    net->conditioning_encoder = make_encoder(weights, num_agents, conditioning_dim, input_size);

    // Pooling output buffers
    net->lane_pooled = calloc(num_agents * input_size, sizeof(float));
    net->boundary_pooled = calloc(num_agents * input_size, sizeof(float));
    net->partner_pooled = calloc(num_agents * input_size, sizeof(float));
    net->traffic_light_pooled = calloc(num_agents * input_size, sizeof(float));
    net->traffic_light_onehot_buf = calloc(num_agents * max_traffic_light_obs * traffic_light_features_onehot, sizeof(float));

    // Backbone: GELU → Linear → GELU → Linear → ... → GELU
    // actor_backbone.backbone.{1,3,5,7} are the Linear layers (0,2,4,6,8 are GELUs)
    net->num_backbone_layers = backbone_num_layers;
    net->backbone_gelus = calloc(backbone_num_layers + 1, sizeof(GELU *));
    net->backbone_linears = calloc(backbone_num_layers, sizeof(Linear *));

    int bb_in = 6 * input_size; // 6 feature sets * 64 = 384
    for (int i = 0; i < backbone_num_layers; i++) {
        net->backbone_gelus[i] = make_gelu(num_agents, bb_in);
        net->backbone_linears[i] = make_linear(weights, num_agents, bb_in, backbone_hidden_size);
        bb_in = backbone_hidden_size;
    }
    net->final_gelu = make_gelu(num_agents, backbone_hidden_size);

    // Actor and critic heads
    // actor_head.0, critic_head.0
    net->actor = make_linear(weights, num_agents, backbone_hidden_size, action_size);
    net->value_fn = make_linear(weights, num_agents, backbone_hidden_size, 1);

    // Concat buffer
    net->concat_buf = calloc(num_agents * 6 * input_size, sizeof(float));

    // Action selection
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 1);

    return net;
}

void free_drivenet(DriveNet *net) {
    free_encoder(net->ego_encoder);
    free_encoder(net->lane_encoder);
    free_encoder(net->boundary_encoder);
    free_encoder(net->partner_encoder);
    free_encoder(net->traffic_light_encoder);
    free_encoder(net->conditioning_encoder);
    free(net->lane_pooled);
    free(net->boundary_pooled);
    free(net->partner_pooled);
    free(net->traffic_light_pooled);
    free(net->traffic_light_onehot_buf);
    for (int i = 0; i < net->num_backbone_layers; i++) {
        free(net->backbone_gelus[i]);
        free(net->backbone_linears[i]);
    }
    free(net->backbone_gelus);
    free(net->backbone_linears);
    free(net->final_gelu);
    free(net->concat_buf);
    free(net->multidiscrete);
    free(net->actor);
    free(net->value_fn);
    free(net);
}

void forward(DriveNet *net, float *observations, int *actions) {
    int n = net->num_agents;
    int ego_dim = net->ego_dim;
    int input_size = net->input_size;

    // Observation layout (from torch.py forward):
    // [ego | conditioning | partners | lanes | boundaries | traffic_lights | stop_signs]
    int cond_dim = net->conditioning_dim;
    int partner_dim = net->max_partner_obs * net->partner_features;
    int lane_dim = net->max_lane_obs * net->road_features;
    int boundary_dim = net->max_boundary_obs * net->road_features;
    int tl_dim = net->max_traffic_light_obs * net->traffic_light_features_raw;

    int obs_stride = ego_dim + cond_dim + partner_dim + lane_dim + boundary_dim + tl_dim;

    // 1. Ego encoder (flat, no pooling)
    // Extract ego observations into contiguous buffer
    float *ego_buf = calloc(n * ego_dim, sizeof(float));
    for (int b = 0; b < n; b++) {
        memcpy(&ego_buf[b * ego_dim], &observations[b * obs_stride], ego_dim * sizeof(float));
    }
    encoder_forward(net->ego_encoder, ego_buf);

    // 2. Conditioning encoder (flat, no pooling)
    float *cond_buf = calloc(n * cond_dim, sizeof(float));
    for (int b = 0; b < n; b++) {
        memcpy(&cond_buf[b * cond_dim], &observations[b * obs_stride + ego_dim], cond_dim * sizeof(float));
    }
    encoder_forward(net->conditioning_encoder, cond_buf);

    // 3. Lane encoder (per-object + max pool)
    float *lane_buf = calloc(n * lane_dim, sizeof(float));
    for (int b = 0; b < n; b++) {
        int offset = ego_dim + cond_dim + partner_dim;
        memcpy(&lane_buf[b * lane_dim], &observations[b * obs_stride + offset], lane_dim * sizeof(float));
    }
    encoder_forward_objects(net->lane_encoder, lane_buf, net->lane_pooled,
                           n, net->max_lane_obs, net->road_features, input_size);

    // 4. Boundary encoder (per-object + max pool)
    float *boundary_buf = calloc(n * boundary_dim, sizeof(float));
    for (int b = 0; b < n; b++) {
        int offset = ego_dim + cond_dim + partner_dim + lane_dim;
        memcpy(&boundary_buf[b * boundary_dim], &observations[b * obs_stride + offset], boundary_dim * sizeof(float));
    }
    encoder_forward_objects(net->boundary_encoder, boundary_buf, net->boundary_pooled,
                           n, net->max_boundary_obs, net->road_features, input_size);

    // 5. Partner encoder (per-object + max pool)
    float *partner_buf = calloc(n * partner_dim, sizeof(float));
    for (int b = 0; b < n; b++) {
        int offset = ego_dim + cond_dim;
        memcpy(&partner_buf[b * partner_dim], &observations[b * obs_stride + offset], partner_dim * sizeof(float));
    }
    encoder_forward_objects(net->partner_encoder, partner_buf, net->partner_pooled,
                           n, net->max_partner_obs, net->partner_features, input_size);

    // 6. Traffic light encoder (per-object + max pool, with one-hot encoding)
    // Raw: [n * max_tl * 6], need to convert last feature to one-hot → [n * max_tl * 9]
    memset(net->traffic_light_onehot_buf, 0, n * net->max_traffic_light_obs * net->traffic_light_features_onehot * sizeof(float));
    for (int b = 0; b < n; b++) {
        int raw_offset = ego_dim + cond_dim + partner_dim + lane_dim + boundary_dim;
        for (int obj = 0; obj < net->max_traffic_light_obs; obj++) {
            float *raw = &observations[b * obs_stride + raw_offset + obj * net->traffic_light_features_raw];
            float *out = &net->traffic_light_onehot_buf[(b * net->max_traffic_light_obs + obj) * net->traffic_light_features_onehot];
            // Copy continuous features (first 6: rel_x1, rel_y1, rel_x2, rel_y2, rel_z, elapsed_time)
            for (int f = 0; f < net->traffic_light_features_raw - 1; f++) {
                out[f] = raw[f];
            }
            // One-hot encode the last feature (traffic light state)
            int state = (int)raw[net->traffic_light_features_raw - 1];
            if (state >= 0 && state < 4) {
                out[net->traffic_light_features_raw - 1 + state] = 1.0f;
            }
        }
    }
    encoder_forward_objects(net->traffic_light_encoder, net->traffic_light_onehot_buf, net->traffic_light_pooled,
                           n, net->max_traffic_light_obs, net->traffic_light_features_onehot, input_size);

    // Concatenate all features: [ego | lane | boundary | partner | traffic_light | conditioning]
    for (int b = 0; b < n; b++) {
        int off = 0;
        memcpy(&net->concat_buf[b * 6 * input_size + off], &net->ego_encoder->linear2->output[b * input_size], input_size * sizeof(float));
        off += input_size;
        memcpy(&net->concat_buf[b * 6 * input_size + off], &net->lane_pooled[b * input_size], input_size * sizeof(float));
        off += input_size;
        memcpy(&net->concat_buf[b * 6 * input_size + off], &net->boundary_pooled[b * input_size], input_size * sizeof(float));
        off += input_size;
        memcpy(&net->concat_buf[b * 6 * input_size + off], &net->partner_pooled[b * input_size], input_size * sizeof(float));
        off += input_size;
        memcpy(&net->concat_buf[b * 6 * input_size + off], &net->traffic_light_pooled[b * input_size], input_size * sizeof(float));
        off += input_size;
        memcpy(&net->concat_buf[b * 6 * input_size + off], &net->conditioning_encoder->linear2->output[b * input_size], input_size * sizeof(float));
    }

    // Backbone: GELU → Linear → GELU → Linear → ... → GELU
    float *bb_input = net->concat_buf;
    for (int i = 0; i < net->num_backbone_layers; i++) {
        gelu(net->backbone_gelus[i], bb_input);
        linear(net->backbone_linears[i], net->backbone_gelus[i]->output);
        bb_input = net->backbone_linears[i]->output;
    }
    gelu(net->final_gelu, bb_input);

    // Actor and critic heads
    linear(net->actor, net->final_gelu->output);
    linear(net->value_fn, net->final_gelu->output);

    // Get action by taking argmax of actor output
    softmax_multidiscrete(net->multidiscrete, net->actor->output, actions);

    // Cleanup temp buffers
    free(ego_buf);
    free(cond_buf);
    free(lane_buf);
    free(boundary_buf);
    free(partner_buf);
}
