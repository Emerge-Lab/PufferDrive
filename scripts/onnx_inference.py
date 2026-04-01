"""Utility for running inference on an exported PufferDrive ONNX model.

Handles observation construction, reward conditioning insertion, and action decoding.

Usage (library):
    from scripts.onnx_inference import DrivePolicyONNX

    policy = DrivePolicyONNX("path/to/model.onnx")
    action, value, trajectory = policy.step(ego_state, partners, road_segments)

Usage (CLI — single inference step + optional video render):
    python -m scripts.onnx_inference \
        --model experiments/puffer_drive_177505208138.onnx \
        --bin experiments/puffer_drive_177505208138.bin \
        --map resources/drive/binaries/carla_2D/map_000.bin

    --model       Path to .onnx model file
    --bin         Path to .bin policy weights (required with --map)
    --map         Path to .bin map file for video rendering
    --dynamics    Dynamics model: jerk (default) or classic
    --ini-config  Path to INI config (default: pufferlib/config/ocean/drive.ini)
                  The [safe_eval] reward conditioning values are used for rendering.
    --verify      Path to .pt checkpoint to verify ONNX output against PyTorch
    --steps       Number of verification steps (default: 5)
"""

import numpy as np
import onnxruntime as ort


# Observation layout constants (must match drive.h)
MAX_AGENTS = 128
MAX_ROAD_SEGMENT_OBSERVATIONS = 256
PARTNER_FEATURES = 8
ROAD_FEATURES = 8
EGO_FEATURES_JERK = 16
EGO_FEATURES_CLASSIC = 13
NUM_REWARD_COEFS = 16
MAX_SPEED = 100.0
MAX_VEH_WIDTH = 3.0
MAX_VEH_LEN = 6.0
SPEED_LIMIT = 20.0

# Reward coefficient indices (must match datatypes.h)
COEF_NAMES = [
    "goal_radius",  # 0
    "collision",  # 1
    "offroad",  # 2
    "comfort",  # 3
    "lane_align",  # 4
    "lane_center",  # 5
    "velocity",  # 6
    "traffic_light",  # 7
    "center_bias",  # 8
    "vel_align",  # 9
    "overspeed",  # 10
    "timestep",  # 11
    "reverse",  # 12
    "throttle",  # 13
    "steer",  # 14
    "acc",  # 15
]

# Gigaflow-matching defaults
GIGAFLOW_COEFS = {
    "goal_radius": 10.0,
    "collision": -3.0,
    "offroad": -3.0,
    "comfort": -0.05,
    "lane_align": 0.025,
    "lane_center": -0.0038,
    "velocity": 0.0025,
    "traffic_light": -1.0,
    "center_bias": 0.0,
    "vel_align": 1.0,
    "overspeed": -1.0,
    "timestep": -0.000025,
    "reverse": -0.005,
    "throttle": 1.0,
    "steer": 1.0,
    "acc": 1.0,
}

# Jerk action decoding
JERK_LONG = np.array([-15.0, -4.0, 0.0, 4.0])
JERK_LAT = np.array([-4.0, 0.0, 4.0])


class DrivePolicyONNX:
    """Wrapper for running inference on an exported PufferDrive ONNX model.

    Args:
        onnx_path: Path to the .onnx model file.
        dynamics_model: "jerk" or "classic".
        reward_coefs: Dict of reward coefficient name → value.
            Defaults to gigaflow-matching values.
        reward_bounds: Dict of reward coefficient name → (min, max) for normalization.
            If None, uses gigaflow bounds (min=max, normalized to 0).
        normalize_coefs: If True, normalize reward coefs to [0, 1] using bounds.
            If False, pass raw values (matches turn_off_normalization=1).
    """

    def __init__(
        self,
        onnx_path,
        dynamics_model="jerk",
        reward_coefs=None,
        reward_bounds=None,
        normalize_coefs=False,
    ):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, sess_options)

        self.dynamics_model = dynamics_model
        self.ego_dim = EGO_FEATURES_JERK if dynamics_model == "jerk" else EGO_FEATURES_CLASSIC
        self.obs_dim = (
            self.ego_dim
            + NUM_REWARD_COEFS
            + (MAX_AGENTS - 1) * PARTNER_FEATURES
            + MAX_ROAD_SEGMENT_OBSERVATIONS * ROAD_FEATURES
        )

        # Reward conditioning
        self.reward_coefs = reward_coefs or GIGAFLOW_COEFS
        self.normalize_coefs = normalize_coefs
        self.reward_bounds = reward_bounds

        # Build the conditioning vector
        self._coef_vector = self._build_coef_vector()

        # LSTM state
        output_names = [o.name for o in self.session.get_outputs()]
        self.has_trajectory = "trajectory" in output_names
        hidden_size = self.session.get_inputs()[1].shape[1]
        self.hidden_size = hidden_size
        self.lstm_h = np.zeros((1, hidden_size), dtype=np.float32)
        self.lstm_c = np.zeros((1, hidden_size), dtype=np.float32)

    def _build_coef_vector(self):
        """Build the 16-element reward conditioning vector."""
        raw = np.zeros(NUM_REWARD_COEFS, dtype=np.float32)
        for i, name in enumerate(COEF_NAMES):
            raw[i] = self.reward_coefs.get(name, 0.0)

        if self.normalize_coefs and self.reward_bounds:
            normalized = np.zeros_like(raw)
            for i, name in enumerate(COEF_NAMES):
                lo, hi = self.reward_bounds.get(name, (raw[i], raw[i]))
                rng = hi - lo
                if rng < 1e-9:
                    normalized[i] = 0.0
                else:
                    normalized[i] = np.clip((raw[i] - lo) / rng, 0.0, 1.0)
            return normalized
        else:
            return raw

    def build_observation(self, ego_state, partners=None, road_segments=None):
        """Build a flattened observation vector.

        Args:
            ego_state: Dict with keys matching the ego observation layout:
                For jerk dynamics:
                    rel_goal_x, rel_goal_y, rel_goal_z: relative goal position
                    speed: signed speed (m/s)
                    width: vehicle width (m)
                    length: vehicle length (m)
                    collision: bool, currently colliding
                    steering_angle: current steering angle (rad)
                    a_long: longitudinal acceleration
                    a_lat: lateral acceleration
                    respawned: bool, has been respawned
                    goal_speed_min, goal_speed_max: normalized goal speed bounds
                    speed_limit: speed limit (m/s)
                    lane_center_dist: distance from lane center
                    lane_angle: cos(heading diff from lane)

            partners: np.array of shape [num_partners, 8] or None.
                Each partner: [rel_x, rel_y, rel_z, speed, width, length, heading_x, heading_y]
                Padded to MAX_AGENTS-1 with zeros.

            road_segments: np.array of shape [num_segments, 8] or None.
                Each segment: [rel_x, rel_y, rel_z, length, width, cos_angle, sin_angle, type]
                Padded to MAX_ROAD_SEGMENT_OBSERVATIONS with zeros.

        Returns:
            obs: np.array of shape [1, obs_dim], ready for ONNX inference.
        """
        obs = np.zeros((1, self.obs_dim), dtype=np.float32)
        idx = 0

        # Ego features
        if self.dynamics_model == "jerk":
            obs[0, 0] = ego_state.get("rel_goal_x", 0) * 0.005
            obs[0, 1] = ego_state.get("rel_goal_y", 0) * 0.005
            obs[0, 2] = ego_state.get("rel_goal_z", 0) * 0.005
            obs[0, 3] = ego_state.get("speed", 0) / MAX_SPEED
            obs[0, 4] = ego_state.get("width", 2.0) / MAX_VEH_WIDTH
            obs[0, 5] = ego_state.get("length", 4.5) / MAX_VEH_LEN
            obs[0, 6] = 1.0 if ego_state.get("collision", False) else 0.0
            obs[0, 7] = ego_state.get("steering_angle", 0) / np.pi
            a_long = ego_state.get("a_long", 0)
            obs[0, 8] = a_long / 15.0 if a_long < 0 else a_long / 4.0
            obs[0, 9] = ego_state.get("a_lat", 0) / 4.0
            obs[0, 10] = 1.0 if ego_state.get("respawned", False) else 0.0
            obs[0, 11] = ego_state.get("goal_speed_min", 0)
            obs[0, 12] = ego_state.get("goal_speed_max", 0)
            obs[0, 13] = min(SPEED_LIMIT / MAX_SPEED, 1.0)
            obs[0, 14] = ego_state.get("lane_center_dist", 0)
            obs[0, 15] = ego_state.get("lane_angle", 1.0)
            idx = EGO_FEATURES_JERK
        else:
            obs[0, 0] = ego_state.get("rel_goal_x", 0) * 0.005
            obs[0, 1] = ego_state.get("rel_goal_y", 0) * 0.005
            obs[0, 2] = ego_state.get("rel_goal_z", 0) * 0.005
            obs[0, 3] = ego_state.get("speed", 0) / MAX_SPEED
            obs[0, 4] = ego_state.get("width", 2.0) / MAX_VEH_WIDTH
            obs[0, 5] = ego_state.get("length", 4.5) / MAX_VEH_LEN
            obs[0, 6] = 1.0 if ego_state.get("collision", False) else 0.0
            obs[0, 7] = 1.0 if ego_state.get("respawned", False) else 0.0
            obs[0, 8] = ego_state.get("goal_speed_min", 0)
            obs[0, 9] = ego_state.get("goal_speed_max", 0)
            obs[0, 10] = min(SPEED_LIMIT / MAX_SPEED, 1.0)
            obs[0, 11] = ego_state.get("lane_center_dist", 0)
            obs[0, 12] = ego_state.get("lane_angle", 1.0)
            idx = EGO_FEATURES_CLASSIC

        # Reward conditioning (16 values)
        obs[0, idx : idx + NUM_REWARD_COEFS] = self._coef_vector
        idx += NUM_REWARD_COEFS

        # Partner features
        max_partners = MAX_AGENTS - 1
        if partners is not None:
            n = min(len(partners), max_partners)
            obs[0, idx : idx + n * PARTNER_FEATURES] = partners[:n].flatten()
        idx += max_partners * PARTNER_FEATURES

        # Road segment features
        if road_segments is not None:
            n = min(len(road_segments), MAX_ROAD_SEGMENT_OBSERVATIONS)
            obs[0, idx : idx + n * ROAD_FEATURES] = road_segments[:n].flatten()

        return obs

    def step(self, ego_state, partners=None, road_segments=None):
        """Run one inference step.

        Args:
            ego_state: Dict of ego features (see build_observation).
            partners: [N, 8] array of partner observations, or None.
            road_segments: [N, 8] array of road segment observations, or None.

        Returns:
            action: int, the discrete action index.
            value: float, value estimate.
            trajectory: np.array [traj_steps, 3] of (x, y, heading), or None.
        """
        obs = self.build_observation(ego_state, partners, road_segments)

        inputs = {
            "observation": obs,
            "lstm_h_in": self.lstm_h,
            "lstm_c_in": self.lstm_c,
        }

        outputs = self.session.run(None, inputs)

        logits = outputs[0]  # [1, num_actions]
        value = outputs[1]  # [1, 1]
        self.lstm_h = outputs[2]
        self.lstm_c = outputs[3]

        action = int(np.argmax(logits[0]))
        trajectory = None
        if self.has_trajectory and len(outputs) > 4:
            trajectory = outputs[4][0]  # [traj_steps, 3]

        return action, float(value[0, 0]), trajectory

    def decode_action(self, action):
        """Decode a discrete action index into human-readable values.

        Returns:
            Dict with jerk_long, jerk_lat (for jerk model) or
            acceleration, steering (for classic model).
        """
        if self.dynamics_model == "jerk":
            num_lat = len(JERK_LAT)
            a_long_idx = action // num_lat
            a_lat_idx = action % num_lat
            return {
                "jerk_long": float(JERK_LONG[a_long_idx]),
                "jerk_lat": float(JERK_LAT[a_lat_idx]),
                "a_long_idx": a_long_idx,
                "a_lat_idx": a_lat_idx,
            }
        else:
            from pufferlib.ocean.drive.drive import ACCELERATION_VALUES, STEERING_VALUES

            num_steer = 13
            accel_idx = action // num_steer
            steer_idx = action % num_steer
            return {
                "acceleration": float(ACCELERATION_VALUES[accel_idx]),
                "steering": float(STEERING_VALUES[steer_idx]),
                "accel_idx": accel_idx,
                "steer_idx": steer_idx,
            }

    def reset(self):
        """Reset LSTM state."""
        self.lstm_h = np.zeros((1, self.hidden_size), dtype=np.float32)
        self.lstm_c = np.zeros((1, self.hidden_size), dtype=np.float32)


def parse_env_obs_to_ego_state(obs_row, dynamics_model="jerk"):
    """Reverse-parse a single agent's env observation into an ego_state dict.

    This undoes the normalization in compute_observations so we can feed
    the raw values into build_observation and check the round-trip.
    """
    ego = {}
    ego["rel_goal_x"] = obs_row[0] / 0.005
    ego["rel_goal_y"] = obs_row[1] / 0.005
    ego["rel_goal_z"] = obs_row[2] / 0.005
    ego["speed"] = obs_row[3] * MAX_SPEED
    ego["width"] = obs_row[4] * MAX_VEH_WIDTH
    ego["length"] = obs_row[5] * MAX_VEH_LEN
    ego["collision"] = obs_row[6] > 0.5

    if dynamics_model == "jerk":
        ego["steering_angle"] = obs_row[7] * np.pi
        # a_long: asymmetric normalization
        a_long_norm = obs_row[8]
        ego["a_long"] = a_long_norm * 15.0 if a_long_norm < 0 else a_long_norm * 4.0
        ego["a_lat"] = obs_row[9] * 4.0
        ego["respawned"] = obs_row[10] > 0.5
        ego["goal_speed_min"] = obs_row[11]
        ego["goal_speed_max"] = obs_row[12]
        ego["speed_limit"] = obs_row[13] * MAX_SPEED
        ego["lane_center_dist"] = obs_row[14]
        ego["lane_angle"] = obs_row[15]
        ego_dim = EGO_FEATURES_JERK
    else:
        ego["respawned"] = obs_row[7] > 0.5
        ego["goal_speed_min"] = obs_row[8]
        ego["goal_speed_max"] = obs_row[9]
        ego["speed_limit"] = obs_row[10] * MAX_SPEED
        ego["lane_center_dist"] = obs_row[11]
        ego["lane_angle"] = obs_row[12]
        ego_dim = EGO_FEATURES_CLASSIC

    # Extract reward conditioning (16 values after ego features)
    coef_start = ego_dim
    reward_coefs = obs_row[coef_start : coef_start + NUM_REWARD_COEFS]

    # Extract partners
    partner_start = ego_dim + NUM_REWARD_COEFS
    max_partners = MAX_AGENTS - 1
    partners = obs_row[partner_start : partner_start + max_partners * PARTNER_FEATURES]
    partners = partners.reshape(max_partners, PARTNER_FEATURES)

    # Extract road segments
    road_start = partner_start + max_partners * PARTNER_FEATURES
    roads = obs_row[road_start : road_start + MAX_ROAD_SEGMENT_OBSERVATIONS * ROAD_FEATURES]
    roads = roads.reshape(MAX_ROAD_SEGMENT_OBSERVATIONS, ROAD_FEATURES)

    return ego, reward_coefs, partners, roads


def verify_onnx_vs_pytorch(onnx_path, checkpoint_path, env_name="puffer_drive", num_steps=5):
    """Verify ONNX model matches PyTorch model on real env observations.

    Two checks:
    1. Export correctness: same env obs → same outputs from ONNX and PyTorch.
    2. Observation construction: env obs round-trips through parse → build_observation.
    """
    import torch
    import importlib
    import pufferlib.vector
    import pufferlib.models
    from pufferlib.ocean.torch import Drive as DrivePolicy
    from scripts.export_model_bin import load_config

    config = load_config(env_name)
    package = config["base"]["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)

    vecenv = pufferlib.vector.make(
        make_env,
        env_kwargs=config["env"],
        backend=pufferlib.vector.Serial,
        num_envs=1,
    )

    # Load PyTorch policy
    policy = DrivePolicy(vecenv.driver_env, **config["policy"])
    if config["base"]["rnn_name"]:
        policy = pufferlib.models.LSTMWrapper(vecenv.driver_env, policy, **config["rnn"])

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
        state_dict = checkpoint["agent_state_dict"]
    else:
        state_dict = checkpoint
    state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    policy.eval()

    # Load ONNX model
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    ort_session = ort.InferenceSession(onnx_path, sess_options)

    # Run both on real env observations
    ob, _ = vecenv.reset()
    num_agents = ob.shape[0]
    hidden_size = config["rnn"]["hidden_size"]
    pt_h = torch.zeros(num_agents, hidden_size)
    pt_c = torch.zeros(num_agents, hidden_size)
    ort_h = np.zeros((num_agents, hidden_size), dtype=np.float32)
    ort_c = np.zeros((num_agents, hidden_size), dtype=np.float32)

    # Determine dynamics model
    dynamics = config["env"].get("dynamics_model", "jerk")

    # Build an inference wrapper to get build_observation
    onnx_policy = DrivePolicyONNX(onnx_path, dynamics_model=dynamics, normalize_coefs=False)

    all_pass = True
    for step in range(num_steps):
        ob_t = torch.as_tensor(ob).float()

        # --- Check 1: Observation round-trip ---
        # Parse env obs → raw fields → rebuild via build_observation → compare
        agent_0_obs = ob[0].astype(np.float32)
        ego, coefs, partners, roads = parse_env_obs_to_ego_state(agent_0_obs, dynamics)

        # Override the inference wrapper's coef vector with what the env actually produced
        onnx_policy._coef_vector = coefs
        rebuilt_obs = onnx_policy.build_observation(ego, partners, roads)
        rebuilt_obs_flat = rebuilt_obs[0]

        obs_match = np.allclose(agent_0_obs, rebuilt_obs_flat, atol=1e-4)
        max_obs_diff = np.abs(agent_0_obs - rebuilt_obs_flat).max()
        if not obs_match:
            # Find which indices differ
            diffs = np.where(np.abs(agent_0_obs - rebuilt_obs_flat) > 1e-4)[0]
            print(f"Step {step}: OBS MISMATCH at indices {diffs[:10]}  max_diff={max_obs_diff:.6f}")
            all_pass = False
        else:
            print(f"Step {step}: OBS MATCH  max_diff={max_obs_diff:.6f}")

        # --- Check 2: Model output match ---
        # PyTorch gets env obs, ONNX gets rebuilt obs
        with torch.no_grad():
            state = {"lstm_h": pt_h, "lstm_c": pt_c}
            pt_logits, pt_value = policy.forward_eval(ob_t, state)
            pt_h = state["lstm_h"]
            pt_c = state["lstm_c"]

        if isinstance(pt_logits, (tuple, list)):
            pt_logits_np = pt_logits[0].numpy()
        else:
            pt_logits_np = pt_logits.numpy()

        # ONNX gets the REBUILT observation (tests full pipeline)
        ort_inputs = {
            "observation": rebuilt_obs.astype(np.float32),
            "lstm_h_in": ort_h[0:1],
            "lstm_c_in": ort_c[0:1],
        }
        ort_outs = ort_session.run(None, ort_inputs)
        ort_logits_np = ort_outs[0]
        ort_value_np = ort_outs[1]
        ort_h[0:1] = ort_outs[2]
        ort_c[0:1] = ort_outs[3]

        logits_match = np.allclose(pt_logits_np[0:1], ort_logits_np, atol=1e-3)
        value_match = np.allclose(pt_value[0:1].numpy(), ort_value_np, atol=1e-3)
        pt_action = pt_logits_np[0].argmax()
        ort_action = ort_logits_np[0].argmax()
        action_match = pt_action == ort_action

        status = "PASS" if (logits_match and value_match and action_match) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"         MODEL {status}"
            f"  logits_close={logits_match}"
            f"  value_close={value_match}"
            f"  action_match={action_match} (pt={pt_action}, ort={ort_action})"
            f"  max_logit_diff={np.abs(pt_logits_np[0:1] - ort_logits_np).max():.6f}"
        )

        # Step env
        import pufferlib.pytorch

        action, _, _ = pufferlib.pytorch.sample_logits(pt_logits)
        action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)
        ob = vecenv.step(action_np)[0]

    vecenv.close()
    if all_pass:
        print(f"\nAll {num_steps} steps PASSED — observation normalization and ONNX output verified.")
    else:
        print(f"\nSome steps FAILED.")
    return all_pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ONNX inference utility")
    parser.add_argument("--model", type=str, required=True, help="Path to .onnx model")
    parser.add_argument("--dynamics", type=str, default="jerk", choices=["jerk", "classic"])
    parser.add_argument("--verify", type=str, default=None, help="Path to .pt checkpoint to verify against")
    parser.add_argument("--steps", type=int, default=5, help="Number of verification steps")
    parser.add_argument("--map", type=str, default=None, help="Path to .bin map file for video rendering")
    parser.add_argument("--bin", type=str, default=None, help="Path to .bin policy weights for the visualize binary")
    parser.add_argument(
        "--ini-config",
        type=str,
        default="pufferlib/config/ocean/drive.ini",
        help="Path to INI config for the visualize binary",
    )
    args = parser.parse_args()

    if args.verify:
        verify_onnx_vs_pytorch(args.model, args.verify, num_steps=args.steps)
    else:
        policy = DrivePolicyONNX(args.model, dynamics_model=args.dynamics)
        ego = {
            "rel_goal_x": 50.0,
            "rel_goal_y": 0.0,
            "rel_goal_z": 0.0,
            "speed": 10.0,
            "width": 2.0,
            "length": 4.5,
            "steering_angle": 0.0,
            "a_long": 0.0,
            "a_lat": 0.0,
            "lane_center_dist": 0.0,
            "lane_angle": 1.0,
        }
        action, value, trajectory = policy.step(ego)
        decoded = policy.decode_action(action)
        print(f"Action: {action} → {decoded}")
        print(f"Value: {value:.4f}")
        if trajectory is not None:
            print(f"Trajectory: {trajectory.shape}, endpoint: ({trajectory[-1, 0]:.1f}, {trajectory[-1, 1]:.1f})")

    if args.map:
        if not args.bin:
            parser.error("--bin is required when --map is provided")
        import configparser
        import os
        from pufferlib.utils import generate_safe_eval_ini, render_videos

        # Read [safe_eval] section and generate a temp INI with pinned reward bounds
        cfg = configparser.ConfigParser()
        cfg.read(args.ini_config)
        safe_eval_config = dict(cfg["safe_eval"]) if cfg.has_section("safe_eval") else {}
        tmp_ini = generate_safe_eval_ini(safe_eval_config, base_ini_path=args.ini_config)
        print(f"Generated temporary INI for rendering with safe eval config: {tmp_ini}")
        try:
            render_videos(
                config={"data_dir": ".", "env": "puffer_drive", "render_map": args.map},
                run_id="onnx_inference",
                wandb_log=False,
                epoch=0,
                global_step=0,
                bin_path=args.bin,
                render_async=False,
                config_path=tmp_ini,
                num_maps=1,
            )
        finally:
            if os.path.exists(tmp_ini):
                os.remove(tmp_ini)
