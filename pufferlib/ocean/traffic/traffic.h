// PufferTraffic signal-control functions.
// Included by pufferlib/ocean/drive/binding.c after drive.h — do NOT include drive.h here.
#pragma once

// --------------------------------------------------------------------------
// Observation layout (SIGNAL_OBS_DIM = 61)
//   [ego(5) | lane_stats(8*4=32) | partner_signals(4*6=24)]
// --------------------------------------------------------------------------
#define SIGNAL_EGO_DIM 5
#define SIGNAL_LANE_FEATURES 4
#define SIGNAL_PARTNER_FEATURES 6
#define MAX_SIGNAL_LANES 8
#define MAX_SIGNAL_PARTNERS 4
#define SIGNAL_OBS_DIM 61  // 5 + 8*4 + 4*6
#define SIGNAL_N_ACTIONS 3 // 0=RED, 1=YELLOW, 2=GREEN

// Reward / observation thresholds
#define SIGNAL_STOPPED_SPEED 0.5f            // m/s below which agent counts as queued
#define SIGNAL_QUEUE_RADIUS 40.0f            // metres behind stop-line for queue count
#define SIGNAL_LANE_CHECK_DIST 5.0f          // metres from lane-point to count agent on lane
#define SIGNAL_THROUGHPUT_DIST 12.0f         // metres from stop-line to credit throughput
#define SIGNAL_MAX_QUEUE 10.0f               // normalisation cap for queue count
#define SIGNAL_THROUGHPUT_MIN_SPD_RATIO 0.3f // fraction of goal_speed for throughput credit

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

// RL action {0,1,2} → TC state constant
static inline int signal_action_to_state(int a) {
    if (a == 0)
        return TRAFFIC_CONTROL_STATE_RED;
    if (a == 1)
        return TRAFFIC_CONTROL_STATE_YELLOW;
    return TRAFFIC_CONTROL_STATE_GREEN;
}

// TC state → float in [0, 1]
static inline float signal_state_to_float(int s) {
    if (s == TRAFFIC_CONTROL_STATE_RED)
        return 0.0f;
    if (s == TRAFFIC_CONTROL_STATE_YELLOW)
        return 0.5f;
    if (s == TRAFFIC_CONTROL_STATE_GREEN)
        return 1.0f;
    return 0.0f;
}

// Absolute index of the n-th traffic-light in env->traffic_elements (-1 if absent)
static inline int signal_abs_idx(Drive *env, int sig_idx) {
    int count = 0;
    for (int i = 0; i < env->num_traffic_elements; i++) {
        if (env->traffic_elements[i].type == TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
            if (count == sig_idx)
                return i;
            count++;
        }
    }
    return -1;
}

// --------------------------------------------------------------------------
// init_signal_states_green
// Set every TL state to GREEN for all timesteps.
// Call after vec_reset to override the random cycles from generate_traffic_light_states.
// --------------------------------------------------------------------------
static void init_signal_states_green(Drive *env) {
    int steps = env->scenario_length;
    for (int i = 0; i < env->num_traffic_elements; i++) {
        TrafficControlElement *tc = &env->traffic_elements[i];
        if (tc->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT)
            continue;
        if (tc->states == NULL || tc->state_length < steps)
            continue;
        for (int t = 0; t < steps; t++)
            tc->states[t] = TRAFFIC_CONTROL_STATE_GREEN;
    }
}

// --------------------------------------------------------------------------
// apply_signal_actions_next
// Write RL actions into states[timestep+1] for all TLs.
// Must be called BEFORE vec_step: C increments timestep then reads states[timestep].
// actions[j] in {0=RED, 1=YELLOW, 2=GREEN}, length = n_signals
// --------------------------------------------------------------------------
static void apply_signal_actions_next(Drive *env, int *actions, int n_signals) {
    int t_next = env->timestep + 1;
    if (t_next >= env->scenario_length)
        return;
    int sig = 0;
    for (int i = 0; i < env->num_traffic_elements && sig < n_signals; i++) {
        TrafficControlElement *tc = &env->traffic_elements[i];
        if (tc->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT)
            continue;
        if (tc->states != NULL && t_next < tc->state_length)
            tc->states[t_next] = signal_action_to_state(actions[sig]);
        sig++;
    }
}

// Consecutive steps the TC has been in its current state at time t
static int signal_steps_in_phase(TrafficControlElement *tc, int t) {
    if (tc->states == NULL || t < 0)
        return 0;
    int cur = tc->states[t];
    int count = 1;
    for (int s = t - 1; s >= 0 && tc->states[s] == cur; s--)
        count++;
    return count;
}

// Steps since the TC was last GREEN at time t (0 if currently GREEN)
static int signal_steps_since_green(TrafficControlElement *tc, int t) {
    if (tc->states == NULL)
        return 0;
    if (tc->states[t] == TRAFFIC_CONTROL_STATE_GREEN)
        return 0;
    for (int s = t - 1; s >= 0; s--)
        if (tc->states[s] == TRAFFIC_CONTROL_STATE_GREEN)
            return t - s;
    return t + 1;
}

// --------------------------------------------------------------------------
// compute_signal_observation
// Fill obs[0..SIGNAL_OBS_DIM-1] for traffic light sig_idx.
// Layout: [ego(5) | lane_stats(8×4) | partner_signals(4×6)]
// --------------------------------------------------------------------------
static void compute_signal_observation(Drive *env, int sig_idx, float *obs) {
    for (int i = 0; i < SIGNAL_OBS_DIM; i++)
        obs[i] = 0.0f;

    int tc_abs = signal_abs_idx(env, sig_idx);
    if (tc_abs < 0)
        return;

    TrafficControlElement *tc = &env->traffic_elements[tc_abs];
    int t = env->timestep;
    float max_t = (float)(env->scenario_length > 1 ? env->scenario_length : 1);
    float max_pos = env->max_position > 0.0f ? env->max_position : 100.0f;

    float sl_x = (tc->stop_line[0] + tc->stop_line[3]) * 0.5f;
    float sl_y = (tc->stop_line[1] + tc->stop_line[4]) * 0.5f;

    int cur_state = (tc->states && t < tc->state_length) ? tc->states[t] : TRAFFIC_CONTROL_STATE_GREEN;

    // --- Ego (5 features) ---
    int off = 0;
    obs[off++] = signal_state_to_float(cur_state);
    obs[off++] = fminf((float)signal_steps_in_phase(tc, t) / max_t, 1.0f);
    obs[off++] = fminf((float)signal_steps_since_green(tc, t) / max_t, 1.0f);
    obs[off++] = sl_x / max_pos;
    obs[off++] = sl_y / max_pos;

    // --- Lane stats (MAX_SIGNAL_LANES × SIGNAL_LANE_FEATURES) ---
    float max_speed = env->goal_speed > 0.0f ? env->goal_speed : 10.0f;
    int n_lanes = tc->num_controlled_lanes < MAX_SIGNAL_LANES ? tc->num_controlled_lanes : MAX_SIGNAL_LANES;
    float dist_sq_thresh = SIGNAL_LANE_CHECK_DIST * SIGNAL_LANE_CHECK_DIST;

    for (int li = 0; li < n_lanes; li++) {
        int lane_idx = tc->controlled_lanes[li];
        if (lane_idx < 0 || lane_idx >= env->num_road_elements) {
            off += 4;
            continue;
        }
        RoadMapElement *lane = &env->road_elements[lane_idx];

        int count = 0;
        float speed_sum = 0.0f;
        int queue = 0;

        for (int ai = 0; ai < env->num_total_agents; ai++) {
            Agent *a = &env->agents[ai];
            if (!a->sim_valid)
                continue;
            int on_lane = 0;
            for (int si = 0; si < lane->segment_length && !on_lane; si++) {
                float dx = a->sim_x - lane->x[si];
                float dy = a->sim_y - lane->y[si];
                if (dx * dx + dy * dy < dist_sq_thresh)
                    on_lane = 1;
            }
            if (!on_lane)
                continue;
            count++;
            speed_sum += a->sim_speed;
            float dx_sl = a->sim_x - sl_x, dy_sl = a->sim_y - sl_y;
            if (a->sim_speed < SIGNAL_STOPPED_SPEED && sqrtf(dx_sl * dx_sl + dy_sl * dy_sl) < SIGNAL_QUEUE_RADIUS)
                queue++;
        }

        obs[off++] = fminf((float)count / SIGNAL_MAX_QUEUE, 1.0f);
        obs[off++] = count > 0 ? fminf(speed_sum / (float)count / max_speed, 1.0f) : 0.0f;
        obs[off++] = fminf((float)queue / SIGNAL_MAX_QUEUE, 1.0f);
        obs[off++] = 1.0f; // lane-slot valid
    }
    off += (MAX_SIGNAL_LANES - n_lanes) * SIGNAL_LANE_FEATURES;

    // --- Partner signals (MAX_SIGNAL_PARTNERS × SIGNAL_PARTNER_FEATURES) ---
    float partner_dist[MAX_SIGNAL_PARTNERS];
    int partner_abs[MAX_SIGNAL_PARTNERS];
    for (int p = 0; p < MAX_SIGNAL_PARTNERS; p++) {
        partner_dist[p] = 1e9f;
        partner_abs[p] = -1;
    }

    for (int i = 0; i < env->num_traffic_elements; i++) {
        if (i == tc_abs)
            continue;
        TrafficControlElement *other = &env->traffic_elements[i];
        if (other->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT)
            continue;
        float ox = (other->stop_line[0] + other->stop_line[3]) * 0.5f;
        float oy = (other->stop_line[1] + other->stop_line[4]) * 0.5f;
        float dist = sqrtf((ox - sl_x) * (ox - sl_x) + (oy - sl_y) * (oy - sl_y));
        for (int p = 0; p < MAX_SIGNAL_PARTNERS; p++) {
            if (dist < partner_dist[p]) {
                for (int q = MAX_SIGNAL_PARTNERS - 1; q > p; q--) {
                    partner_dist[q] = partner_dist[q - 1];
                    partner_abs[q] = partner_abs[q - 1];
                }
                partner_dist[p] = dist;
                partner_abs[p] = i;
                break;
            }
        }
    }

    for (int p = 0; p < MAX_SIGNAL_PARTNERS; p++) {
        if (partner_abs[p] < 0) {
            off += SIGNAL_PARTNER_FEATURES;
            continue;
        }
        TrafficControlElement *other = &env->traffic_elements[partner_abs[p]];
        float ox = (other->stop_line[0] + other->stop_line[3]) * 0.5f;
        float oy = (other->stop_line[1] + other->stop_line[4]) * 0.5f;
        int o_state = (other->states && t < other->state_length) ? other->states[t] : TRAFFIC_CONTROL_STATE_GREEN;
        obs[off++] = (ox - sl_x) / max_pos;
        obs[off++] = (oy - sl_y) / max_pos;
        obs[off++] = fminf(partner_dist[p] / max_pos, 1.0f);
        obs[off++] = signal_state_to_float(o_state);
        obs[off++] = fminf((float)signal_steps_in_phase(other, t) / max_t, 1.0f);
        obs[off++] = fminf((float)signal_steps_since_green(other, t) / max_t, 1.0f);
    }
}

// --------------------------------------------------------------------------
// compute_signal_reward
// IntelliLight-style reward for signal sig_idx:
//   + weight_throughput * vehicles crossing stop-line with decent speed
//   - weight_queue      * vehicles stopped within SIGNAL_QUEUE_RADIUS
//   - weight_flicker    * 1 if state changed from previous step
// --------------------------------------------------------------------------
static float compute_signal_reward(Drive *env, int sig_idx, float weight_throughput, float weight_queue,
                                   float weight_flicker) {
    int tc_abs = signal_abs_idx(env, sig_idx);
    if (tc_abs < 0)
        return 0.0f;

    TrafficControlElement *tc = &env->traffic_elements[tc_abs];
    int t = env->timestep;
    float sl_x = (tc->stop_line[0] + tc->stop_line[3]) * 0.5f;
    float sl_y = (tc->stop_line[1] + tc->stop_line[4]) * 0.5f;
    float max_speed = env->goal_speed > 0.0f ? env->goal_speed : 10.0f;
    float min_tp_spd = max_speed * SIGNAL_THROUGHPUT_MIN_SPD_RATIO;
    float dist_sq_thresh = SIGNAL_LANE_CHECK_DIST * SIGNAL_LANE_CHECK_DIST;

    int queue = 0, throughput = 0;

    for (int ai = 0; ai < env->num_total_agents; ai++) {
        Agent *a = &env->agents[ai];
        if (!a->sim_valid)
            continue;

        int near = 0;
        for (int li = 0; li < tc->num_controlled_lanes && !near; li++) {
            int lane_idx = tc->controlled_lanes[li];
            if (lane_idx < 0 || lane_idx >= env->num_road_elements)
                continue;
            RoadMapElement *lane = &env->road_elements[lane_idx];
            for (int si = 0; si < lane->segment_length && !near; si++) {
                float dx = a->sim_x - lane->x[si];
                float dy = a->sim_y - lane->y[si];
                if (dx * dx + dy * dy < dist_sq_thresh)
                    near = 1;
            }
        }
        if (!near)
            continue;

        float dx_sl = a->sim_x - sl_x, dy_sl = a->sim_y - sl_y;
        float dist_sl = sqrtf(dx_sl * dx_sl + dy_sl * dy_sl);

        if (a->sim_speed < SIGNAL_STOPPED_SPEED && dist_sl < SIGNAL_QUEUE_RADIUS)
            queue++;
        if (a->sim_speed > min_tp_spd && dist_sl < SIGNAL_THROUGHPUT_DIST)
            throughput++;
    }

    int flicker = 0;
    if (tc->states && t > 0 && t < tc->state_length)
        flicker = (tc->states[t] != tc->states[t - 1]) ? 1 : 0;

    return weight_throughput * (float)throughput - weight_queue * (float)queue - weight_flicker * (float)flicker;
}
