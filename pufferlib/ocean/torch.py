from torch import nn
import torch
import torch.nn.functional as F

import numpy as np

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401


Recurrent = pufferlib.models.LSTMWrapper


class Drive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6  # 6 is the number of one-hot encoded categories
        self.actions_trajectory_length = kwargs.get("actions_trajectory_length", 80)
        # Determine ego dimension from environment's feature layout
        self.ego_dim = env.ego_features

        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(4 * input_size, hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.action_tensor_shape = [sum(self.atn_dim)] + [self.actions_trajectory_length]

        assert all(type(i) == int for i in self.action_tensor_shape)  # check that all entries are integers

        self.previous_actions_trajectory = torch.zeros(
            np.prod(self.action_tensor_shape)
        )  # TODO: remove, this is a temporary input

        self.past_actions_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(np.prod(self.action_tensor_shape), input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, np.prod(self.action_tensor_shape)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        past_actions_dim = np.prod(self.action_tensor_shape)
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]
        past_actions_traj = observations[
            :, ego_dim + partner_dim + road_dim : ego_dim + partner_dim + road_dim + past_actions_dim
        ]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)

        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)  # Shape: [batch, ROAD_MAX_OBJECTS, 7]
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)
        past_actions_features = self.past_actions_encoder(past_actions_traj)

        concat_features = torch.cat([ego_features, road_features, partner_features, past_actions_features], dim=1)

        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        # embedding = self.shared_embedding(concat_features)
        return embedding

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden).reshape(
                (flat_hidden.shape[0], self.actions_trajectory_length, sum(self.atn_dim))
            )
            loc, scale = torch.split(parameters, self.atn_dim, dim=2)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden).reshape(
                flat_hidden.shape[0] * self.actions_trajectory_length, sum(self.atn_dim)
            )  # push the sequence length in the batch
            # repack the actions into n tuples where n is the number of kind of actions you can have (MultiDiscrete)
            action = torch.split(action, self.atn_dim, dim=1)

        value = self.value_fn(
            flat_hidden
        )  # NOTE: there shouldn't be any difference if we pass the hidden state that is used for future actions

        return action, value

    def observation_spec(self):
        """Return structured observation specification.

        Documents every feature by name, index, normalization, and expected range.
        This is the single source of truth for building input adapters.

        The flat observation vector is laid out as:
            [ego_features | reward_conditioning (optional) | partner_features | road_features | past_actions_trajectory]

        The past_actions_trajectory section (appended by LSTMWrapper) contains
        the flattened previous action logits: shape (sum(atn_dim) * actions_trajectory_length,).
        At the first timestep this is all zeros.
        """
        import math

        # --- Constants matching C defines in drive.h / datatypes.h ---
        MAX_SPEED = 100.0
        MAX_VEH_WIDTH = 15.0
        MAX_VEH_LEN = 30.0
        LANE_DIST_NORM = 4.0
        SPEED_LIMIT = 20.0
        JERK_LONG_MIN = -15.0  # braking
        JERK_LONG_MAX = 4.0  # acceleration
        JERK_LAT_MAX = 4.0
        MAX_ROAD_SEGMENT_LENGTH = 100.0
        MAX_ROAD_SCALE = 100.0
        NUM_REWARD_COEFS = 16
        GOAL_POSITION_SCALE = 0.005  # 1/200m — goal relative position normalization
        RELATIVE_POSITION_SCALE = 0.02  # 1/50m  — partner & road relative position normalization

        # Determine dynamics model from ego_dim
        # Jerk: 16 base, Classic: 13 base
        base_ego = self.ego_dim
        has_conditioning = base_ego > 16  # if > 16, conditioning is appended
        if has_conditioning:
            base_ego_no_cond = base_ego - NUM_REWARD_COEFS
        else:
            base_ego_no_cond = base_ego
        is_jerk = base_ego_no_cond == 16

        # --- Ego features ---
        ego_spec = [
            {
                "name": "goal_rel_x",
                "norm": f"* {GOAL_POSITION_SCALE}",
                "range": [-1, 1],
                "desc": "Goal X in ego frame (1/200m scale)",
            },
            {
                "name": "goal_rel_y",
                "norm": f"* {GOAL_POSITION_SCALE}",
                "range": [-1, 1],
                "desc": "Goal Y in ego frame (1/200m scale)",
            },
            {
                "name": "goal_rel_z",
                "norm": f"* {GOAL_POSITION_SCALE}",
                "range": [-1, 1],
                "desc": "Goal Z in ego frame (1/200m scale)",
            },
            {"name": "signed_speed", "norm": f"/ {MAX_SPEED}", "range": [-1, 1], "desc": "Signed speed along heading"},
            {"name": "vehicle_width", "norm": f"/ {MAX_VEH_WIDTH}", "range": [0, 1], "desc": "Ego vehicle width"},
            {"name": "vehicle_length", "norm": f"/ {MAX_VEH_LEN}", "range": [0, 1], "desc": "Ego vehicle length"},
            {"name": "collision_flag", "norm": "binary", "range": [0, 1], "desc": "1 if currently collided"},
        ]
        if is_jerk:
            # Jerk dynamics state variable limits (from c_step clipping in drive.h):
            #   steering_angle: clipped to [-0.55, 0.55] rad (~[-31.5°, 31.5°])
            #   a_long:         clipped to [-5.0, 2.5] m/s²
            #   a_lat:          clipped to [-4.0, 4.0] m/s²
            # Jerk action space:
            #   JERK_LONG = [-15.0, -4.0, 0.0, 4.0] m/s³
            #   JERK_LAT  = [-4.0, 0.0, 4.0] m/s³
            ego_spec += [
                {
                    "name": "steering_angle",
                    "norm": f"/ π ({math.pi:.4f})",
                    "range": [-0.175, 0.175],
                    "physical_range": [-0.55, 0.55],
                    "unit": "rad",
                    "desc": "Current steering angle (clipped ±0.55 rad ≈ ±31.5°)",
                },
                {
                    "name": "a_long",
                    "norm": f"asymmetric: /{-JERK_LONG_MIN} if neg, /{JERK_LONG_MAX} if pos",
                    "range": [-0.333, 0.625],
                    "physical_range": [-5.0, 2.5],
                    "unit": "m/s²",
                    "desc": "Longitudinal acceleration (clipped [-5.0, 2.5] m/s², asymmetric norm)",
                },
                {
                    "name": "a_lat",
                    "norm": f"/ {JERK_LAT_MAX}",
                    "range": [-1, 1],
                    "physical_range": [-4.0, 4.0],
                    "unit": "m/s²",
                    "desc": "Lateral acceleration (clipped ±4.0 m/s²)",
                },
                {
                    "name": "respawned_flag",
                    "norm": "binary",
                    "range": [0, 1],
                    "desc": "1 while agent is in respawn transit (respawn_timestep != -1). "
                    "GOAL_RESPAWN(0): set to 1 on respawn, cleared to 0 once agent resumes. "
                    "GOAL_GENERATE_NEW(1): always 0 (agent keeps driving, never respawns). "
                    "GOAL_STOP(2): always 0 (agent stops at goal, never respawns).",
                },
                {
                    "name": "goal_speed_min",
                    "norm": f"/ {MAX_SPEED}",
                    "range": [0, 1],
                    "desc": "Min goal speed (0 if disabled)",
                },
                {
                    "name": "goal_speed_max",
                    "norm": f"/ {MAX_SPEED}",
                    "range": [0, 1],
                    "desc": "Max goal speed (0 if disabled)",
                },
                {"name": "speed_limit", "norm": f"/ {MAX_SPEED}, clamped", "range": [0, 1], "desc": "Road speed limit"},
                {
                    "name": "lane_center_dist",
                    "norm": f"/ {LANE_DIST_NORM}, clamped",
                    "range": [-1, 1],
                    "desc": "Signed distance from lane center",
                },
                {"name": "lane_angle_cos", "norm": "raw", "range": [-1, 1], "desc": "cos(heading diff from lane)"},
            ]
        else:  # Classic
            ego_spec += [
                {
                    "name": "respawned_flag",
                    "norm": "binary",
                    "range": [0, 1],
                    "desc": "1 while agent is in respawn transit (respawn_timestep != -1). "
                    "GOAL_RESPAWN(0): set to 1 on respawn, cleared to 0 once agent resumes. "
                    "GOAL_GENERATE_NEW(1): always 0 (agent keeps driving, never respawns). "
                    "GOAL_STOP(2): always 0 (agent stops at goal, never respawns).",
                },
                {
                    "name": "goal_speed_min",
                    "norm": f"/ {MAX_SPEED}",
                    "range": [0, 1],
                    "desc": "Min goal speed (0 if disabled)",
                },
                {
                    "name": "goal_speed_max",
                    "norm": f"/ {MAX_SPEED}",
                    "range": [0, 1],
                    "desc": "Max goal speed (0 if disabled)",
                },
                {"name": "speed_limit", "norm": f"/ {MAX_SPEED}, clamped", "range": [0, 1], "desc": "Road speed limit"},
                {
                    "name": "lane_center_dist",
                    "norm": f"/ {LANE_DIST_NORM}, clamped",
                    "range": [-1, 1],
                    "desc": "Signed distance from lane center",
                },
                {"name": "lane_angle_cos", "norm": "raw", "range": [-1, 1], "desc": "cos(heading diff from lane)"},
            ]

        # --- Reward conditioning (optional) ---
        reward_coef_names = [
            "goal_radius",
            "collision",
            "offroad",
            "comfort",
            "lane_align",
            "lane_center",
            "velocity",
            "traffic_light",
            "center_bias",
            "vel_align",
            "overspeed",
            "timestep",
            "reverse",
            "throttle",
            "steer",
            "acc",
        ]
        conditioning_spec = None
        if has_conditioning:
            conditioning_spec = [
                {
                    "name": f"reward_coef_{name}",
                    "norm": "tanh-normalized",
                    "range": [-1, 1],
                    "desc": f"Reward conditioning coef for {name}",
                }
                for name in reward_coef_names
            ]

        # --- Partner features (per object, 8 features) ---
        partner_spec = [
            {
                "name": "rel_x",
                "norm": f"* {RELATIVE_POSITION_SCALE}",
                "range": [-1, 1],
                "desc": "Partner X in ego frame (1/50m scale)",
            },
            {
                "name": "rel_y",
                "norm": f"* {RELATIVE_POSITION_SCALE}",
                "range": [-1, 1],
                "desc": "Partner Y in ego frame (1/50m scale)",
            },
            {
                "name": "rel_z",
                "norm": f"* {RELATIVE_POSITION_SCALE}",
                "range": [-1, 1],
                "desc": "Partner Z in ego frame (1/50m scale)",
            },
            {"name": "width", "norm": f"/ {MAX_VEH_WIDTH}", "range": [0, 1], "desc": "Partner vehicle width"},
            {"name": "length", "norm": f"/ {MAX_VEH_LEN}", "range": [0, 1], "desc": "Partner vehicle length"},
            {"name": "rel_heading_cos", "norm": "raw", "range": [-1, 1], "desc": "cos(partner_heading - ego_heading)"},
            {"name": "rel_heading_sin", "norm": "raw", "range": [-1, 1], "desc": "sin(partner_heading - ego_heading)"},
            {"name": "signed_speed", "norm": f"/ {MAX_SPEED}", "range": [-1, 1], "desc": "Partner signed speed"},
        ]

        # --- Road features (per segment, 8 features) ---
        road_spec = [
            {
                "name": "rel_x",
                "norm": f"* {RELATIVE_POSITION_SCALE}",
                "range": [-1, 1],
                "desc": "Segment midpoint X in ego frame (1/50m scale)",
            },
            {
                "name": "rel_y",
                "norm": f"* {RELATIVE_POSITION_SCALE}",
                "range": [-1, 1],
                "desc": "Segment midpoint Y in ego frame (1/50m scale)",
            },
            {
                "name": "rel_z",
                "norm": f"* {RELATIVE_POSITION_SCALE}",
                "range": [-1, 1],
                "desc": "Segment midpoint Z in ego frame (1/50m scale)",
            },
            {"name": "length", "norm": f"/ {MAX_ROAD_SEGMENT_LENGTH}", "range": [0, 1], "desc": "Segment half-length"},
            {"name": "width", "norm": f"/ {MAX_ROAD_SCALE}", "range": [0, 1], "desc": "Segment width (hardcoded 0.1)"},
            {"name": "cos_angle", "norm": "raw", "range": [-1, 1], "desc": "cos(segment direction - ego heading)"},
            {"name": "sin_angle", "norm": "raw", "range": [-1, 1], "desc": "sin(segment direction - ego heading)"},
            {
                "name": "road_type",
                "norm": "categorical: type - 4",
                "range": [0, 2],
                "desc": "Road element type: 0=LANE (drivable surface center-line), "
                "1=LINE (painted lane marking/divider between lanes), "
                "2=EDGE (road boundary/curb)",
            },
        ]

        past_actions_dim = np.prod(self.action_tensor_shape)

        return {
            "layout": "[ego | reward_conditioning? | partners | road_segments | past_actions_trajectory]",
            "base_obs_dim": self.observation_size,
            "total_dim": self.observation_size + past_actions_dim,
            "ego": {
                "offset": 0,
                "count": 1,
                "features_per_object": base_ego_no_cond,
                "total_dim": base_ego_no_cond,
                "features": ego_spec,
            },
            "reward_conditioning": {
                "offset": base_ego_no_cond,
                "count": NUM_REWARD_COEFS if has_conditioning else 0,
                "total_dim": NUM_REWARD_COEFS if has_conditioning else 0,
                "features": conditioning_spec,
            }
            if has_conditioning
            else None,
            "partners": {
                "offset": self.ego_dim,
                "count": self.max_partner_objects,
                "features_per_object": self.partner_features,
                "total_dim": self.max_partner_objects * self.partner_features,
                "features": partner_spec,
            },
            "road_segments": {
                "offset": self.ego_dim + self.max_partner_objects * self.partner_features,
                "count": self.max_road_objects,
                "features_per_object": self.road_features,
                "total_dim": self.max_road_objects * self.road_features,
                "features": road_spec,
            },
            "past_actions_trajectory": {
                "offset": self.ego_dim
                + self.max_partner_objects * self.partner_features
                + self.max_road_objects * self.road_features,
                "trajectory_length": self.actions_trajectory_length,
                "actions_per_step": sum(self.atn_dim),
                "total_dim": past_actions_dim,
                "desc": f"Flattened previous action logits: ({sum(self.atn_dim)} actions × "
                f"{self.actions_trajectory_length} timesteps). Zeros at first timestep.",
            },
        }

    def action_spec(self):
        """Return structured action/output specification for the discrete joint action space.

        Documents the network output layout, how to decompose the joint action integer
        into sub-action indices, and the physical values (with units) each index maps to.
        This is the single source of truth for interpreting model outputs on the deployment side.

        The network produces an action trajectory of T timesteps:
            logits:   tuple of (batch * T, joint_action_size) tensors (multi-discrete from torch.split)
                      — T = actions_trajectory_length (default 80)
                      — during training, only timestep t=0 is sampled for the env step;
                        future timesteps provide an action-prediction training signal.
                      — for ONNX export, these are flattened to (batch, joint_action_size * T).
            value:    (batch, 1)                  — critic estimate (unbounded)
            lstm_h:   (batch, hidden_size)        — LSTM hidden state
            lstm_c:   (batch, hidden_size)        — LSTM cell state

        Deployment post-processing (discrete, deterministic):
            # Reshape ONNX flat logits back to trajectory:
            logits = logits.reshape(batch, T, joint_action_size)
            joint_action = argmax(logits[:, 0, :])   # use t=0 for env step
            primary_idx  = joint_action // num_secondary
            secondary_idx = joint_action %  num_secondary
            physical_primary   = PRIMARY_VALUES[primary_idx]
            physical_secondary = SECONDARY_VALUES[secondary_idx]
        """
        from pufferlib.ocean.drive import binding

        dynamics_model = (
            "jerk"
            if (self.ego_dim == binding.EGO_FEATURES_JERK or self.ego_dim == binding.EGO_FEATURES_JERK_CONDITIONING)
            else "classic"
        )
        is_jerk = dynamics_model == "jerk"

        # --- Action value arrays matching C defines in drive.h ---
        if is_jerk:
            # JERK_LONG[4] and JERK_LAT[3]
            primary_values = [-15.0, -4.0, 0.0, 4.0]
            primary_unit = "m/s³"
            primary_name = "longitudinal_jerk"
            secondary_values = [-4.0, 0.0, 4.0]
            secondary_unit = "m/s³"
            secondary_name = "lateral_jerk"
        else:
            # ACCELERATION_VALUES[7] and STEERING_VALUES[13]
            primary_values = [-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0]
            primary_unit = "m/s²"
            primary_name = "acceleration"
            secondary_values = [-1.0, -0.833, -0.667, -0.5, -0.333, -0.167, 0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0]
            secondary_unit = "rad"
            secondary_name = "steering"

        num_primary = len(primary_values)
        num_secondary = len(secondary_values)
        joint_size = num_primary * num_secondary

        # Verify consistency with the model's atn_dim
        assert sum(self.atn_dim) == joint_size, (
            f"action_spec joint_size={joint_size} != model atn_dim sum={sum(self.atn_dim)}"
        )

        T = self.actions_trajectory_length

        return {
            "mode": "discrete",
            "dynamics_model": dynamics_model,
            "joint_action_size": joint_size,
            "trajectory_length": T,
            "flat_logits_dim": joint_size * T,
            "decomposition": f"joint_action = {primary_name}_idx * {num_secondary} + {secondary_name}_idx",
            "primary": {
                "name": primary_name,
                "num_actions": num_primary,
                "values": primary_values,
                "unit": primary_unit,
                "index_formula": f"joint_action // {num_secondary}",
                "desc": f"{primary_name} sub-action index (row of the joint grid)",
            },
            "secondary": {
                "name": secondary_name,
                "num_actions": num_secondary,
                "values": secondary_values,
                "unit": secondary_unit,
                "index_formula": f"joint_action % {num_secondary}",
                "desc": f"{secondary_name} sub-action index (column of the joint grid)",
            },
            "outputs": {
                "logits": {
                    "shape": f"(batch, {joint_size * T})  — flatten of (batch, {T}, {joint_size})",
                    "desc": f"Unnormalized log-probabilities for {T} future timesteps. "
                    f"Reshape to (batch, {T}, {joint_size}). "
                    "During training only t=0 is sampled; remaining timesteps "
                    "provide an action-prediction training signal.",
                },
                "value": {
                    "shape": "(batch, 1)",
                    "desc": "State-value estimate V(s). Unbounded scalar, no activation.",
                },
                "lstm_h": {
                    "shape": f"(batch, {self.hidden_size})",
                    "desc": "LSTM hidden state to feed back at next timestep.",
                },
                "lstm_c": {
                    "shape": f"(batch, {self.hidden_size})",
                    "desc": "LSTM cell state to feed back at next timestep.",
                },
            },
            "post_processing": {
                "reshape": f"logits = logits.reshape(batch, {T}, {joint_size})",
                "deterministic": "joint_action = argmax(logits[:, 0, :], dim=-1)  # t=0 for env step",
                "stochastic": "joint_action = Categorical(logits=logits[:, 0, :]).sample()  # t=0",
                "all_timesteps": f"joint_actions = argmax(logits, dim=-1)  # (batch, {T}) trajectory",
                "decompose": (
                    f"{primary_name}_idx = joint_action // {num_secondary}; "
                    f"{secondary_name}_idx = joint_action % {num_secondary}"
                ),
                "lookup": (
                    f"physical_{primary_name} = {primary_name}_values[{primary_name}_idx]; "
                    f"physical_{secondary_name} = {secondary_name}_values[{secondary_name}_idx]"
                ),
            },
        }

    @staticmethod
    def construct_action_output(logits, dynamics_model="classic", actions_trajectory_length=80):
        """Decode raw network logits into physical action values for the discrete joint action space.

        Uses pufferlib.pytorch.sample_logits — the same function used during training —
        to ensure identical sampling behaviour.  Handles action trajectories:
        the network outputs logits for T future timesteps, but only t=0 is
        sampled for the current env step (matching training).

        Also decodes *all* T timesteps via argmax so the deployment side can
        inspect the full planned trajectory.

        Mirrors the C-side action decoding in move_dynamics() (drive.h).
        This is the output-side counterpart of build_structured_observation().

        Args:
            logits: Raw logits as returned by decode_actions() — a tuple of tensors
                    each of shape (batch * T, joint_size) for multi-discrete.
            dynamics_model: "classic" or "jerk"
            actions_trajectory_length: number of future timesteps T (default 80)

        Returns:
            dict with keys:
                # --- t=0 sampled action (matches training) ---
                joint_action:       (batch,) int  — sampled joint action at t=0
                primary_idx:        (batch,) int  — primary sub-action index at t=0
                secondary_idx:      (batch,) int  — secondary sub-action index at t=0
                primary_physical:   (batch,) float — physical primary value at t=0
                secondary_physical: (batch,) float — physical secondary value at t=0
                log_prob:           (batch,) float — log-probability of sampled t=0 action
                entropy:            (batch,) float — categorical entropy at t=0
                # --- full trajectory (all T timesteps, argmax) ---
                trajectory_logits:    (batch, T, joint_size) — reshaped logits
                trajectory_joint:     (batch, T) int — argmax joint action per timestep
                trajectory_primary:   (batch, T) float — physical primary per timestep
                trajectory_secondary: (batch, T) float — physical secondary per timestep
                # --- metadata ---
                metadata:           dict — action names, units, value tables, trajectory_length
        """
        import pufferlib.pytorch

        T = actions_trajectory_length

        # --- Sample t=0 using the exact same function as training ---
        sample_result = pufferlib.pytorch.sample_logits(logits, actions_trajectory_length=T)
        # sample_logits returns (action, log_prob, entropy, logits_full) for multi-discrete
        action, log_prob, entropy_val = sample_result[0], sample_result[1], sample_result[2]
        joint_action = action.squeeze(-1).int()

        # --- Action value tables matching C defines ---
        if dynamics_model == "jerk":
            primary_values = torch.tensor([-15.0, -4.0, 0.0, 4.0])
            primary_name = "longitudinal_jerk"
            primary_unit = "m/s³"
            secondary_values = torch.tensor([-4.0, 0.0, 4.0])
            secondary_name = "lateral_jerk"
            secondary_unit = "m/s³"
        else:
            primary_values = torch.tensor([-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0])
            primary_name = "acceleration"
            primary_unit = "m/s²"
            secondary_values = torch.tensor(
                [-1.0, -0.833, -0.667, -0.5, -0.333, -0.167, 0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0]
            )
            secondary_name = "steering"
            secondary_unit = "rad"

        num_secondary = len(secondary_values)
        joint_size = len(primary_values) * num_secondary

        # --- Decompose t=0 sampled action ---
        primary_idx = joint_action // num_secondary
        secondary_idx = joint_action % num_secondary
        primary_physical = primary_values[primary_idx.long()]
        secondary_physical = secondary_values[secondary_idx.long()]

        # --- Decode full trajectory (all T timesteps) via argmax ---
        # logits_full from sample_logits: (1, num_splits, T, joint_size)
        logits_full = sample_result[3]  # 4th return for multi-discrete
        # Squeeze num_splits=1 → (batch, T, joint_size)
        traj_logits = logits_full.squeeze(1)
        traj_joint = traj_logits.argmax(dim=-1).int()  # (batch, T)
        traj_primary_idx = traj_joint // num_secondary
        traj_secondary_idx = traj_joint % num_secondary
        traj_primary_phys = primary_values[traj_primary_idx.long()]
        traj_secondary_phys = secondary_values[traj_secondary_idx.long()]

        return {
            # t=0 sampled action (matches training)
            "joint_action": joint_action,
            "primary_idx": primary_idx,
            "secondary_idx": secondary_idx,
            f"{primary_name}": primary_physical,
            f"{secondary_name}": secondary_physical,
            "log_prob": log_prob,
            "entropy": entropy_val,
            # Full trajectory (all T timesteps, argmax)
            "trajectory_logits": traj_logits,
            "trajectory_joint": traj_joint,
            f"trajectory_{primary_name}": traj_primary_phys,
            f"trajectory_{secondary_name}": traj_secondary_phys,
            "metadata": {
                "dynamics_model": dynamics_model,
                "primary_name": primary_name,
                "primary_unit": primary_unit,
                "primary_values": primary_values.tolist(),
                "secondary_name": secondary_name,
                "secondary_unit": secondary_unit,
                "secondary_values": secondary_values.tolist(),
                "decomposition": f"joint = {primary_name}_idx * {num_secondary} + {secondary_name}_idx",
                "trajectory_length": T,
                "joint_action_size": joint_size,
            },
        }

    @staticmethod
    def build_structured_observation(dynamics_model="classic", reward_conditioning=False, batch_size=1):
        """Build a physically valid dummy observation tensor for export/testing.

        Reads observation dimensions directly from the C binding constants.
        All values are within the ranges that compute_observations() in C would produce.

        Args:
            dynamics_model: "classic" or "jerk"
            reward_conditioning: whether reward conditioning coefficients are appended to ego
            batch_size: batch dimension
        """
        import math
        from pufferlib.ocean.drive import binding

        # --- Dimensions from C binding ---
        max_road_objects = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
        max_partner_objects = binding.MAX_AGENTS - 1
        partner_features = binding.PARTNER_FEATURES
        road_features = binding.ROAD_FEATURES

        if dynamics_model == "jerk":
            ego_dim = binding.EGO_FEATURES_JERK_CONDITIONING if reward_conditioning else binding.EGO_FEATURES_JERK
        else:
            ego_dim = binding.EGO_FEATURES_CLASSIC_CONDITIONING if reward_conditioning else binding.EGO_FEATURES_CLASSIC

        is_jerk = dynamics_model == "jerk"
        has_conditioning = reward_conditioning
        base_ego = binding.EGO_FEATURES_JERK if is_jerk else binding.EGO_FEATURES_CLASSIC

        # --- Constants matching C normalization ---
        MAX_SPEED = 100.0
        MAX_VEH_WIDTH = 15.0
        MAX_VEH_LEN = 30.0
        SPEED_LIMIT = 20.0
        NUM_REWARD_COEFS = 16
        GOAL_POSITION_SCALE = 0.005  # 1/200m — goal relative position normalization
        RELATIVE_POSITION_SCALE = 0.02  # 1/50m  — partner & road relative position normalization

        # --- Ego features ---
        ego = torch.zeros(batch_size, ego_dim)
        # Goal relative position (normalized by *GOAL_POSITION_SCALE, so raw ~[-200, 200] → [-1, 1])
        ego[:, 0] = 30.0 * GOAL_POSITION_SCALE  # goal_rel_x: ~30m ahead
        ego[:, 1] = 2.0 * GOAL_POSITION_SCALE  # goal_rel_y: ~2m lateral
        ego[:, 2] = 0.0  # goal_rel_z
        ego[:, 3] = 5.0 / MAX_SPEED  # signed_speed: 5 m/s
        ego[:, 4] = 2.0 / MAX_VEH_WIDTH  # vehicle_width: 2m
        ego[:, 5] = 4.5 / MAX_VEH_LEN  # vehicle_length: 4.5m
        ego[:, 6] = 0.0  # collision_flag: no collision

        if is_jerk:
            ego[:, 7] = 0.0  # steering_angle: straight
            ego[:, 8] = 0.0  # a_long: no acceleration
            ego[:, 9] = 0.0  # a_lat: no lateral accel
            ego[:, 10] = 0.0  # respawned_flag
            ego[:, 11] = 0.0  # goal_speed_min (disabled)
            ego[:, 12] = 10.0 / MAX_SPEED  # goal_speed_max
            ego[:, 13] = min(SPEED_LIMIT / MAX_SPEED, 1.0)  # speed_limit
            ego[:, 14] = 0.05  # lane_center_dist: slightly off-center
            ego[:, 15] = 0.98  # lane_angle_cos: well-aligned
        else:
            ego[:, 7] = 0.0  # respawned_flag
            ego[:, 8] = 0.0  # goal_speed_min
            ego[:, 9] = 10.0 / MAX_SPEED  # goal_speed_max
            ego[:, 10] = min(SPEED_LIMIT / MAX_SPEED, 1.0)  # speed_limit
            ego[:, 11] = 0.05  # lane_center_dist
            ego[:, 12] = 0.98  # lane_angle_cos

        # Reward conditioning: tanh-normalized values in [-1, 1]
        if has_conditioning:
            cond_offset = base_ego
            for c in range(NUM_REWARD_COEFS):
                ego[:, cond_offset + c] = 0.0  # neutral conditioning

        # --- Partner features (mostly empty = no visible partners) ---
        partner_dim = max_partner_objects * partner_features
        partners = torch.zeros(batch_size, partner_dim)
        # Place one visible partner ~10m ahead, 3m to the right
        partners[:, 0] = 10.0 * RELATIVE_POSITION_SCALE  # rel_x
        partners[:, 1] = 3.0 * RELATIVE_POSITION_SCALE  # rel_y
        partners[:, 2] = 0.0  # rel_z
        partners[:, 3] = 2.0 / MAX_VEH_WIDTH  # width
        partners[:, 4] = 4.5 / MAX_VEH_LEN  # length
        partners[:, 5] = 1.0  # rel_heading_cos (same direction)
        partners[:, 6] = 0.0  # rel_heading_sin
        partners[:, 7] = 8.0 / MAX_SPEED  # signed_speed

        # --- Road features ---
        road_dim = max_road_objects * road_features
        roads = torch.zeros(batch_size, road_dim)
        # Place a few road segments nearby
        for seg in range(min(5, max_road_objects)):
            base = seg * road_features
            dist = 3.0 + seg * 4.0  # stagger segments ahead
            roads[:, base + 0] = dist * RELATIVE_POSITION_SCALE  # rel_x
            roads[:, base + 1] = 0.5 * RELATIVE_POSITION_SCALE  # rel_y: slightly off-center
            roads[:, base + 2] = 0.0  # rel_z
            roads[:, base + 3] = 5.0 / 100.0  # length
            roads[:, base + 4] = 0.1 / 100.0  # width (hardcoded 0.1 in C)
            angle = 0.05 * seg
            roads[:, base + 5] = math.cos(angle)  # cos_angle
            roads[:, base + 6] = math.sin(angle)  # sin_angle
            roads[:, base + 7] = 0.0  # road_type: lane (type 4 - 4 = 0)

        # --- Past actions trajectory (appended by LSTMWrapper) ---
        # At initial state (first timestep) these are all zeros.
        # Shape: (batch, sum(atn_dim) * actions_trajectory_length)
        # For classic: 91 * 80 = 7280, for jerk: 12 * 80 = 960
        num_accel = 7 if not is_jerk else 4
        num_steer = 13 if not is_jerk else 3
        atn_dim_sum = num_accel * num_steer  # joint action size
        past_actions_flat = atn_dim_sum * 80  # default trajectory length
        past_actions = torch.zeros(batch_size, past_actions_flat)

        obs = torch.cat([ego, partners, roads, past_actions], dim=1)
        return obs
