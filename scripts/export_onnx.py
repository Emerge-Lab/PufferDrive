"""Export a PufferDrive policy checkpoint (.pt) to ONNX format.

Produces a single .onnx file with external weights (.onnx.data),
plus example_input.pt and example_output.pt for verification.

Args:
    --env             Environment name used to look up the ini config (default: puffer_drive)
    --checkpoint      Path to the .pt checkpoint file
    --output_dir      Directory where .onnx, .onnx.data, and example .pt files are written
    --fake_trajectory Wrap model to also output a (n_step, 3) x/y/heading trajectory.
                      Step 0 uses the argmax action; remaining steps use zero jerk.
    --traj_steps      Number of trajectory rollout steps (default: 80)
    --traj_dt         Timestep in seconds for the trajectory rollout (default: 0.1)

Usage:
    # Standard export
    python -m scripts.export_onnx \
        --env puffer_drive \
        --checkpoint pufferlib/resources/drive/onnx_files/model_puffer_drive_011500.pt \
        --output_dir pufferlib/resources/drive/onnx_files

    # Export with fake trajectory output
    python -m scripts.export_onnx \
        --env puffer_drive \
        --checkpoint pufferlib/resources/drive/onnx_files/model_puffer_drive_011500.pt \
        --output_dir pufferlib/resources/drive/onnx_files \
        --fake_trajectory \
        --traj_steps 80 \
        --traj_dt 0.1
"""

import argparse
import ast
import configparser
import glob
import os
import types
import numpy as np
import torch
import onnxruntime as ort

import gymnasium
import pufferlib

from pufferlib.ocean.torch import Drive as DrivePolicy
from pufferlib.ocean.drive import binding


JERK_LONG = torch.tensor([-15.0, -4.0, 0.0, 4.0])
JERK_LAT = torch.tensor([-4.0, 0.0, 4.0])
NUM_LAT = 3


class JerkDynamicsRollout(torch.nn.Module):
    """Roll out the argmax action through jerk bicycle dynamics for N steps.

    Step 0 applies the chosen jerk; steps 1..N-1 apply zero jerk (constant
    acceleration coast), producing the "fake trajectory" in ego frame.
    """

    def __init__(self, num_steps=80, dt=0.1):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.register_buffer("jerk_long_table", JERK_LONG)
        self.register_buffer("jerk_lat_table", JERK_LAT)

    def forward(self, logits, speed, a_long, a_lat, steering_angle, wheelbase):
        """
        Args:
            logits:         [B, num_actions]
            speed:          [B]
            a_long:         [B]
            a_lat:          [B]
            steering_angle: [B]
            wheelbase:      [B]
        Returns:
            trajectory: [B, num_steps, 3]  — (x, y, heading) in ego frame
        """
        B = logits.shape[0]
        action = logits.argmax(dim=-1)
        jerk_long_val = self.jerk_long_table[action // NUM_LAT]
        jerk_lat_val = self.jerk_lat_table[action % NUM_LAT]

        x = torch.zeros(B, device=logits.device)
        y = torch.zeros(B, device=logits.device)
        heading = torch.zeros(B, device=logits.device)
        v = speed.clone()
        al = a_long.clone()
        at = a_lat.clone()
        steer = steering_angle.clone()

        dt = self.dt
        trajectory = torch.zeros(B, self.num_steps, 3, device=logits.device)

        # Apply chosen jerk only on step 0; zero jerk thereafter
        jl_schedule = torch.zeros(self.num_steps, B, device=logits.device)
        jt_schedule = torch.zeros(self.num_steps, B, device=logits.device)
        jl_schedule[0] = jerk_long_val
        jt_schedule[0] = jerk_lat_val

        for t in range(self.num_steps):
            al_new = al + jl_schedule[t] * dt
            at_new = at + jt_schedule[t] * dt

            al_new = torch.clamp(al_new, -5.0, 2.5)
            at_new = torch.clamp(at_new, -4.0, 4.0)

            al_new = torch.where(al * al_new < 0, torch.zeros_like(al_new), al_new)
            at_new = torch.where(at * at_new < 0, torch.zeros_like(at_new), at_new)

            v_new = v + 0.5 * (al_new + al) * dt
            v_new = torch.where(v * v_new < 0, torch.zeros_like(v_new), v_new)
            v_new = torch.clamp(v_new, -2.0, 20.0)

            signed_curvature = at_new / torch.clamp(v_new * v_new, min=1e-5)
            steer_target = torch.atan(signed_curvature * wheelbase)
            delta_steer = torch.clamp(steer_target - steer, -0.6 * dt, 0.6 * dt)
            steer_new = torch.clamp(steer + delta_steer, -0.55, 0.55)

            signed_curvature = torch.tan(steer_new) / wheelbase
            at_new = v_new * v_new * signed_curvature

            d = 0.5 * (v_new + v) * dt
            theta = d * signed_curvature

            small = signed_curvature.abs() < 1e-5
            dx_local = torch.where(small, d, torch.sin(theta) / (signed_curvature + 1e-10))
            dy_local = torch.where(small, torch.zeros_like(d), (1 - torch.cos(theta)) / (signed_curvature + 1e-10))

            cos_h = torch.cos(heading)
            sin_h = torch.sin(heading)
            x = x + dx_local * cos_h - dy_local * sin_h
            y = y + dx_local * sin_h + dy_local * cos_h
            heading = heading + theta
            v, al, at, steer = v_new, al_new, at_new, steer_new

            trajectory[:, t, 0] = x
            trajectory[:, t, 1] = y
            trajectory[:, t, 2] = heading

        return trajectory


class OnnxWrapper(torch.nn.Module):
    """Thin wrapper around the Drive policy for ONNX export.

    Without --fake_trajectory: outputs (logits, value).
    With    --fake_trajectory: outputs (logits, value, trajectory),
            where trajectory is [B, traj_steps, 3] (x, y, heading in ego frame).
    """

    def __init__(self, policy, fake_trajectory=False, traj_steps=80, dt=0.1):
        super().__init__()
        self.policy = policy
        self.fake_trajectory = fake_trajectory
        if fake_trajectory:
            self.rollout = JerkDynamicsRollout(num_steps=traj_steps, dt=dt)

    def forward(self, observation):
        actions, value = self.policy(observation)
        # actions is a tuple of per-head logit tensors; cat for a single tensor
        logits = torch.cat(actions, dim=1) if isinstance(actions, tuple) else actions

        if self.fake_trajectory:
            # Ego features are the first slice of the observation (jerk model layout):
            #   obs[:, 3] = signed_speed / MAX_SPEED  (MAX_SPEED = 100)
            #   obs[:, 5] = vehicle_length / MAX_VEH_LEN  (proxy for wheelbase, ~5.5 m)
            #   obs[:, 7] = steering_angle / pi
            #   obs[:, 8] = a_long (asymmetrically normalised: neg→/15, pos→/4)
            #   obs[:, 9] = a_lat / 4.0
            speed = observation[:, 3] * 100.0
            wheelbase = observation[:, 5] * 5.5
            steering_angle = observation[:, 7] * 3.14159265
            a_long_norm = observation[:, 8]
            a_long = torch.where(a_long_norm < 0, a_long_norm * 15.0, a_long_norm * 4.0)
            a_lat = observation[:, 9] * 4.0

            trajectory = self.rollout(logits, speed, a_long, a_lat, steering_angle, wheelbase)
            return logits, value, trajectory

        return logits, value


def _read_ini(env_name):
    """Parse the ini config for env_name and return {section: {key: value}} dicts."""
    puffer_dir = os.path.dirname(os.path.realpath(pufferlib.__file__))
    default_ini = os.path.join(puffer_dir, "config/default.ini")
    for path in glob.glob(os.path.join(puffer_dir, "config/**/*.ini"), recursive=True):
        p = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        p.read([default_ini, path])
        if env_name in p["base"]["env_name"].split():
            break
    else:
        raise ValueError(f"No config found for env_name={env_name!r}")

    def _cast(v):
        try:
            return ast.literal_eval(v)
        except Exception:
            return v

    return {section: {k: _cast(v) for k, v in p[section].items()} for section in p.sections()}


# ---------------------------------------------------------------------------
# Minimal mock environment — avoids spinning up a real vecenv
# ---------------------------------------------------------------------------


def _make_mock_env(
    dynamics_model="jerk",
    max_lane_segment_observations=80,
    max_boundary_segment_observations=80,
    max_partner_observations=16,
    max_traffic_control_observations=4,
    lane_segment_dropout=0.0,
    boundary_segment_dropout=0.0,
    num_target_waypoints=3,
    target_type="static",
    reward_conditioning=False,
    split_network=True,
):
    env = types.SimpleNamespace()

    env.ego_features = {
        "classic": binding.EGO_FEATURES_CLASSIC,
        "jerk": binding.EGO_FEATURES_JERK,
    }[dynamics_model]

    env.max_lane_segment_observations = max_lane_segment_observations
    env.max_boundary_segment_observations = max_boundary_segment_observations
    env.max_partner_observations = max_partner_observations
    env.max_traffic_control_observations = max_traffic_control_observations

    def _effective(max_count, dropout):
        return int(max_count * (1.0 - min(max(dropout, 0.0), 1.0))) if max_count > 0 else 0

    env.obs_lane_segment_count = _effective(max_lane_segment_observations, lane_segment_dropout)
    env.obs_boundary_segment_count = _effective(max_boundary_segment_observations, boundary_segment_dropout)

    env.partner_features = binding.PARTNER_FEATURES
    env.road_features = binding.ROAD_FEATURES
    env.traffic_control_features = binding.TRAFFIC_CONTROL_FEATURES

    env.num_reward_coefs = binding.NUM_REWARD_COEFS if reward_conditioning else 0
    target_features = binding.STATIC_TARGET_FEATURES if target_type == "static" else binding.DYNAMIC_TARGET_FEATURES
    env.target_dim = num_target_waypoints * target_features

    # Action space: jerk → 4×3 = 12, classic → 7×9 = 63
    if dynamics_model == "jerk":
        env.single_action_space = gymnasium.spaces.MultiDiscrete([4 * 3])
    else:
        env.single_action_space = gymnasium.spaces.MultiDiscrete([7 * 9])

    env.split_network = split_network

    num_obs = (
        env.ego_features
        + env.num_reward_coefs
        + env.target_dim
        + env.max_partner_observations * env.partner_features
        + env.obs_lane_segment_count * env.road_features
        + env.obs_boundary_segment_count * env.road_features
        + env.max_traffic_control_observations * env.traffic_control_features
    )
    env.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(num_obs,), dtype=np.float32)
    return env


# ---------------------------------------------------------------------------
# Realistic example input construction
# ---------------------------------------------------------------------------


def _make_example_input(env, batch_size=1):
    """Build a plausible (non-trivial) observation tensor."""
    rng = np.random.default_rng(42)
    parts = []

    # 1. Ego state — jerk model: [speed, accel, heading_sin, heading_cos,
    #    yaw_rate, steer, jerk, norm_progress, dist_to_goal] (9 features)
    #    Use physically reasonable normalised values.
    ego = np.zeros((batch_size, env.ego_features), dtype=np.float32)
    ego[:, 0] = 0.3  # speed ~30% of max
    ego[:, 1] = 0.05  # small accel
    ego[:, 2] = 0.5  # heading sin
    ego[:, 3] = 0.866  # heading cos (≈30°)
    ego[:, 4] = 0.02  # slight yaw rate
    if env.ego_features > 5:
        ego[:, 5] = 0.01  # steer
    if env.ego_features > 6:
        ego[:, 6] = 0.0  # jerk
    if env.ego_features > 7:
        ego[:, 7] = 0.2  # progress along route
    if env.ego_features > 8:
        ego[:, 8] = 0.6  # dist to goal
    parts.append(ego)

    # 2. Conditioning (reward coefs + target waypoints)
    conditioning_dim = env.num_reward_coefs + env.target_dim
    if conditioning_dim > 0:
        # Reward coefs (uniform 1.0 if present), target waypoints as relative positions
        cond = np.zeros((batch_size, conditioning_dim), dtype=np.float32)
        coef_dim = env.num_reward_coefs
        if coef_dim > 0:
            cond[:, :coef_dim] = 1.0  # all reward coefs at baseline
        # Waypoints: alternating x/y relative goal positions
        target_start = coef_dim
        target_features_per_wp = env.target_dim // (env.target_dim // max(1, binding.STATIC_TARGET_FEATURES))
        for i in range(env.target_dim // target_features_per_wp):
            base = target_start + i * target_features_per_wp
            cond[:, base] = 0.3 + i * 0.15  # x progress
            cond[:, base + 1] = 0.05 * (i % 2)  # slight lateral offset
            if target_features_per_wp > 2:
                cond[:, base + 2] = 0.0  # heading delta
        parts.append(cond)

    # 3. Partner observations (some agents ahead and to the side)
    partner_dim = env.max_partner_observations * env.partner_features
    partners = rng.uniform(-0.2, 0.2, (batch_size, partner_dim)).astype(np.float32)
    # First 3 partners are real (non-zero), rest are masked/absent
    for i in range(min(3, env.max_partner_observations)):
        base = i * env.partner_features
        partners[:, base + 0] = 0.1 + i * 0.05  # rel x
        partners[:, base + 1] = 0.02 * i  # rel y
        partners[:, base + 2] = 0.3  # speed
        partners[:, base + 3] = 0.5  # heading sin
        partners[:, base + 4] = 0.866  # heading cos
    parts.append(partners)

    # 4. Lane segment observations (road centerline segments ahead)
    lane_dim = env.obs_lane_segment_count * env.road_features
    lanes = rng.uniform(-0.1, 0.1, (batch_size, lane_dim)).astype(np.float32)
    # Add gradient to simulate a curved road ahead
    for i in range(env.obs_lane_segment_count):
        base = i * env.road_features
        lanes[:, base + 0] = 0.05 * i / env.obs_lane_segment_count  # x offset increases
        lanes[:, base + 1] = 0.3 * i / env.obs_lane_segment_count  # y distance ahead
        lanes[:, base + 2] = 0.5  # heading sin
        lanes[:, base + 3] = 0.866  # heading cos
    parts.append(lanes)

    # 5. Boundary segment observations
    boundary_dim = env.obs_boundary_segment_count * env.road_features
    boundaries = rng.uniform(-0.1, 0.1, (batch_size, boundary_dim)).astype(np.float32)
    for i in range(env.obs_boundary_segment_count):
        base = i * env.road_features
        boundaries[:, base + 0] = -0.15 + 0.05 * i / env.obs_boundary_segment_count
        boundaries[:, base + 1] = 0.3 * i / env.obs_boundary_segment_count
        boundaries[:, base + 2] = 0.5
        boundaries[:, base + 3] = 0.866
    parts.append(boundaries)

    # 6. Traffic control observations (1 red light ahead, rest absent)
    tc_dim = env.max_traffic_control_observations * env.traffic_control_features
    tc = np.zeros((batch_size, tc_dim), dtype=np.float32)
    # First traffic control: stop sign at distance 0.4, type=0, state=0 (red)
    tc[:, 0] = 0.4  # distance
    tc[:, 1] = 0.0  # rel x
    tc[:, 2] = 0.4  # rel y
    tc[:, 3] = 0.5  # heading sin
    tc[:, 4] = 0.866  # heading cos
    tc[:, 5] = 0.0  # type (int 0 = traffic_light) — will be one-hot in forward
    tc[:, 6] = 0.0  # state (int 0 = red)
    parts.append(tc)

    obs = np.concatenate(parts, axis=1)
    obs = obs.clip(-1.0, 1.0)
    return torch.tensor(obs, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def export(args):
    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    onnx_path = os.path.join(output_dir, base + ".onnx")

    # Load config from ini file
    config = _read_ini(args.env)
    env_cfg = config["env"]
    policy_kwargs = config["policy"]

    # Build mock env from ini env config
    env = _make_mock_env(
        dynamics_model=env_cfg.get("dynamics_model", "jerk"),
        max_lane_segment_observations=env_cfg.get("max_lane_segment_observations", 80),
        max_boundary_segment_observations=env_cfg.get("max_boundary_segment_observations", 80),
        max_partner_observations=env_cfg.get("max_partner_observations", 16),
        max_traffic_control_observations=env_cfg.get("max_traffic_control_observations", 4),
        lane_segment_dropout=env_cfg.get("lane_segment_dropout", 0.0),
        boundary_segment_dropout=env_cfg.get("boundary_segment_dropout", 0.0),
        num_target_waypoints=env_cfg.get("num_target_waypoints", 3),
        target_type=env_cfg.get("target_type", "static"),
        reward_conditioning=env_cfg.get("reward_conditioning", False),
        split_network=policy_kwargs.get("split_network", False),
    )
    obs_dim = env.single_observation_space.shape[0]
    print(f"Observation dim: {obs_dim}")

    # Build policy from ini policy config
    policy = DrivePolicy(env, **policy_kwargs)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Strip DDP `module.` prefix
    state_dict = {}
    for k, v in ckpt.items():
        key = k[len("module.") :] if k.startswith("module.") else k
        key = key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key
        state_dict[key] = v

    policy.load_state_dict(state_dict)
    policy.eval()

    # Wrap policy for export
    wrapped = OnnxWrapper(
        policy,
        fake_trajectory=args.fake_trajectory,
        traj_steps=args.traj_steps,
        dt=args.traj_dt,
    )
    wrapped.eval()

    # Build example input
    example_input = _make_example_input(env, batch_size=1)
    print(f"Example input shape: {example_input.shape}")
    print(f"Example input (first 12 values): {example_input[0, :12].tolist()}")

    # Run PyTorch forward pass through the wrapper
    with torch.no_grad():
        torch_outs = wrapped(example_input)

    if args.fake_trajectory:
        logits, value, trajectory = torch_outs
        print(f"PyTorch trajectory shape: {trajectory.shape}")
    else:
        logits, value = torch_outs

    print(f"PyTorch logits shape: {logits.shape}, value shape: {value.shape}")
    print(f"PyTorch logits sample: {logits[0].tolist()}")
    print(f"PyTorch value: {value[0].item():.6f}")

    # Save example input/output
    input_path = os.path.join(output_dir, "example_input.pt")
    output_path = os.path.join(output_dir, "example_output.pt")
    torch.save(example_input, input_path)
    saved = {"logits": logits, "value": value}
    if args.fake_trajectory:
        saved["trajectory"] = trajectory
    torch.save(saved, output_path)
    print(f"Saved example_input.pt -> {input_path}")
    print(f"Saved example_output.pt -> {output_path}")

    # Build ONNX export names / dynamic axes
    output_names = ["logits", "value"]
    dynamic_axes = {
        "observation": {0: "batch_size"},
        "logits": {0: "batch_size"},
        "value": {0: "batch_size"},
    }
    if args.fake_trajectory:
        output_names.append("trajectory")
        dynamic_axes["trajectory"] = {0: "batch_size"}

    # Export to ONNX with external data (single .onnx.data file)
    print(f"\nExporting to ONNX: {onnx_path}")
    torch.onnx.export(
        wrapped,
        example_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Convert inline weights to external data file
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    model_proto = onnx.load(onnx_path)
    data_file = base + ".onnx.data"
    convert_model_to_external_data(
        model_proto,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=0,
        convert_attribute=True,
    )
    onnx.save_model(model_proto, onnx_path)
    print(f"Weights saved to: {os.path.join(output_dir, data_file)}")

    # Quick ONNX runtime verification — merge external data into memory first
    print("\nVerifying ONNX output with onnxruntime...")
    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    verify_proto = onnx.load(onnx_path, load_external_data=False)
    load_external_data_for_model(verify_proto, output_dir)
    sess = ort.InferenceSession(verify_proto.SerializeToString())
    ort_outs = sess.run(None, {"observation": example_input.numpy()})

    np.testing.assert_allclose(
        logits.numpy(),
        ort_outs[0],
        rtol=1e-3,
        atol=1e-4,
        err_msg="Logits mismatch between PyTorch and ONNX Runtime",
    )
    np.testing.assert_allclose(
        value.numpy(),
        ort_outs[1],
        rtol=1e-3,
        atol=1e-4,
        err_msg="Value mismatch between PyTorch and ONNX Runtime",
    )
    print("  logits:     MATCH")
    print("  value:      MATCH")
    if args.fake_trajectory:
        np.testing.assert_allclose(
            trajectory.numpy(),
            ort_outs[2],
            rtol=1e-3,
            atol=1e-4,
            err_msg="Trajectory mismatch between PyTorch and ONNX Runtime",
        )
        print("  trajectory: MATCH")
    print("\nExport complete.")
    print(f"  ONNX model:    {onnx_path}")
    print(f"  Weights data:  {os.path.join(output_dir, data_file)}")
    print(f"  Example input: {input_path}")
    print(f"  Example output:{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PufferDrive policy to ONNX")
    parser.add_argument("--env", type=str, default="puffer_drive", help="Environment name (matches ini config)")
    parser.add_argument(
        "--checkpoint",
        default="pufferlib/resources/drive/onnx_files/model_puffer_drive_011500.pt",
    )
    parser.add_argument(
        "--output_dir",
        default="pufferlib/resources/drive/onnx_files",
    )
    parser.add_argument(
        "--fake_trajectory",
        action="store_true",
        help="Add trajectory output: step 0 uses argmax action, remaining steps use zero jerk",
    )
    parser.add_argument("--traj_steps", type=int, default=80, help="Number of trajectory rollout steps")
    parser.add_argument("--traj_dt", type=float, default=0.1, help="Timestep in seconds for trajectory rollout")
    export(parser.parse_args())
