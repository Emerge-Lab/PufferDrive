"""Export a trained PufferDrive policy checkpoint (.pt) to ONNX format.

The exported ONNX model accepts an observation vector (see in torch.py for the exact layout),
plus LSTM hidden states, and produces action logits,
a value estimate, and updated LSTM states.

Usage:
    python -m scripts.export_onnx --checkpoint <path/to/model.pt> [--output <path.onnx>]
"""

import argparse
import os
import torch
import importlib
import numpy as np
import onnxruntime as ort

import pufferlib.utils
import pufferlib.vector
import pufferlib.models

from pufferlib.ocean.torch import Drive
from scripts.export_model_bin import load_config


JERK_LONG = torch.tensor([-15.0, -4.0, 0.0, 4.0])
JERK_LAT = torch.tensor([-4.0, 0.0, 4.0])
NUM_LAT = 3
SPEED_LIMIT = 20.0


class JerkDynamicsRollout(torch.nn.Module):
    """Roll out a single discrete action through jerk dynamics for N steps.

    Takes the argmax action from logits, decodes into jerk_long/jerk_lat,
    and integrates through the bicycle jerk model in ego frame (starting
    at x=0, y=0, heading=0).

    Requires initial state from observation: speed, a_long, a_lat, steering_angle.
    """

    def __init__(self, num_steps=80, dt=0.1):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.register_buffer("jerk_long", JERK_LONG)
        self.register_buffer("jerk_lat", JERK_LAT)

    def forward(self, logits, speed, a_long, a_lat, steering_angle, wheelbase):
        """
        Args:
            logits: [B, num_actions] — policy output
            speed: [B] — signed speed from obs
            a_long: [B] — longitudinal acceleration from obs
            a_lat: [B] — lateral acceleration from obs
            steering_angle: [B] — current steering angle from obs
            wheelbase: [B] — vehicle wheelbase from obs

        Returns:
            trajectory: [B, num_steps, 3] — (x, y, heading) in ego frame
        """
        B = logits.shape[0]
        action = logits.argmax(dim=-1)  # [B]
        a_long_idx = action // NUM_LAT
        a_lat_idx = action % NUM_LAT

        jerk_long_val = self.jerk_long[a_long_idx]  # [B]
        jerk_lat_val = self.jerk_lat[a_lat_idx]  # [B]

        # State in ego frame
        x = torch.zeros(B, device=logits.device)
        y = torch.zeros(B, device=logits.device)
        heading = torch.zeros(B, device=logits.device)
        v = speed.clone()
        al = a_long.clone()
        at = a_lat.clone()
        steer = steering_angle.clone()

        dt = self.dt
        trajectory = torch.zeros(B, self.num_steps, 3, device=logits.device)

        # Build jerk schedule: apply action on step 0, zero-jerk after
        jl_schedule = torch.zeros(self.num_steps, B, device=logits.device)
        jt_schedule = torch.zeros(self.num_steps, B, device=logits.device)
        jl_schedule[0] = jerk_long_val
        jt_schedule[0] = jerk_lat_val

        for t in range(self.num_steps):
            # Update acceleration
            al_new = al + jl_schedule[t] * dt
            at_new = at + jt_schedule[t] * dt

            # Clamp acceleration
            al_new = torch.clamp(al_new, -5.0, 2.5)
            at_new = torch.clamp(at_new, -4.0, 4.0)

            # Zero-crossing: easy stop
            al_new = torch.where(al * al_new < 0, torch.zeros_like(al_new), al_new)
            at_new = torch.where(at * at_new < 0, torch.zeros_like(at_new), at_new)

            # Update velocity
            v_new = v + 0.5 * (al_new + al) * dt
            v_new = torch.where(v * v_new < 0, torch.zeros_like(v_new), v_new)
            v_new = torch.clamp(v_new, -2.0, SPEED_LIMIT)

            # Steering from lateral acceleration
            signed_curvature = at_new / torch.clamp(v_new * v_new, min=1e-5)
            steer_target = torch.atan(signed_curvature * wheelbase)
            delta_steer = torch.clamp(steer_target - steer, -0.6 * dt, 0.6 * dt)
            steer_new = torch.clamp(steer + delta_steer, -0.55, 0.55)

            # Recompute curvature from clamped steering
            signed_curvature = torch.tan(steer_new) / wheelbase
            at_new = v_new * v_new * signed_curvature

            # Displacement (bicycle model)
            d = 0.5 * (v_new + v) * dt
            theta = d * signed_curvature

            # Local displacement
            small = signed_curvature.abs() < 1e-5
            dx_local = torch.where(small, d, torch.sin(theta) / (signed_curvature + 1e-10))
            dy_local = torch.where(small, torch.zeros_like(d), (1 - torch.cos(theta)) / (signed_curvature + 1e-10))

            # Rotate to world frame
            cos_h = torch.cos(heading)
            sin_h = torch.sin(heading)
            dx = dx_local * cos_h - dy_local * sin_h
            dy = dx_local * sin_h + dy_local * cos_h

            # Update state
            x = x + dx
            y = y + dy
            heading = heading + theta
            v = v_new
            al = al_new
            at = at_new
            steer = steer_new

            trajectory[:, t, 0] = x
            trajectory[:, t, 1] = y
            trajectory[:, t, 2] = heading

        return trajectory


class OnnxWrapper(torch.nn.Module):
    def __init__(self, policy, fake_trajectory=False, traj_steps=80, dt=0.1):
        super().__init__()
        self.policy = policy
        self.fake_trajectory = fake_trajectory
        if fake_trajectory:
            self.rollout = JerkDynamicsRollout(num_steps=traj_steps, dt=dt)

    def forward(self, observation, h, c):
        state = {"lstm_h": h, "lstm_c": c}
        logits, value = self.policy.forward_eval(observation, state)
        new_h = state["lstm_h"]
        new_c = state["lstm_c"]

        if self.fake_trajectory:
            # Extract ego state from observation (jerk model layout)
            # obs[3] = signed_speed / MAX_SPEED
            speed = observation[:, 3] * 100.0  # MAX_SPEED = 100
            # obs[7] = steering_angle / pi
            steering_angle = observation[:, 7] * 3.14159265
            # obs[8] = a_long (normalized asymmetrically)
            a_long_norm = observation[:, 8]
            a_long = torch.where(
                a_long_norm < 0,
                a_long_norm * 15.0,  # JERK_LONG[0] = -15
                a_long_norm * 4.0,  # JERK_LONG[3] = 4
            )
            # obs[9] = a_lat / JERK_LAT[2]
            a_lat = observation[:, 9] * 4.0  # JERK_LAT[2] = 4
            # obs[5] = sim_length / MAX_VEH_LEN
            wheelbase = observation[:, 5] * 5.5  # MAX_VEH_LEN approximate

            # logits[0] for single-head multi-discrete (jerk: MultiDiscrete([12]))
            logits_tensor = logits[0] if isinstance(logits, (tuple, list)) else logits
            trajectory = self.rollout(logits_tensor, speed, a_long, a_lat, steering_angle, wheelbase)
            return logits, value, new_h, new_c, trajectory

        return logits, value, new_h, new_c


def export_to_onnx(verify=True):
    parser = argparse.ArgumentParser(description="Export PufferDrive model to ONNX")
    parser.add_argument("--env", type=str, default="puffer_drive", help="Environment name")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/puffer_drive_73kbtsi5/model_puffer_drive_000200.pt",
        help="Path to .pt checkpoint",
    )
    parser.add_argument("--output", type=str, help="Output .onnx file path")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument(
        "--fake-trajectory",
        action="store_true",
        help="Add trajectory output by repeating the argmax action through jerk dynamics",
    )
    parser.add_argument("--traj-steps", type=int, default=80, help="Number of trajectory rollout steps")
    parser.add_argument("--traj-dt", type=float, default=0.1, help="Timestep for trajectory rollout")
    parser.add_argument("--render", action="store_true", help="Render eval video after export")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.env)

    # Load environment to get observation/action spaces
    package = config["base"]["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(args.env)

    # Ensure env args/kwargs are correctly passed
    env_kwargs = config["env"]

    vecenv = pufferlib.vector.make(make_env, env_kwargs=env_kwargs, backend=pufferlib.vector.Serial, num_envs=1)

    # Initialize Policy
    print("Initializing Policy...")
    policy = Drive(vecenv.driver_env, **config["policy"])

    if config["base"]["rnn_name"]:
        print("Wrapping with LSTM...")
        policy = pufferlib.models.LSTMWrapper(vecenv.driver_env, policy, **config["rnn"])

    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Handle both full checkpoint dict and raw state dict
    if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
        state_dict = checkpoint["agent_state_dict"]
    else:
        state_dict = checkpoint

    # Strip DDP and compile prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("_orig_mod."):
            key = key[10:]
        new_state_dict[key] = v

    policy.load_state_dict(new_state_dict)
    policy.eval()

    # Prepare inputs for ONNX export
    print("Preparing sample inputs...")
    batch_size = 1

    obs_space = vecenv.single_observation_space
    # Flatten observation if needed, Drive policy handles flattening internally usually but check vecenv
    # The LSTMWrapper expects (B, ObsDim)
    obs_dim = np.prod(obs_space.shape)

    # Create Dummy Observation
    if config["base"]["rnn_name"]:
        # If wrapped, access the internal Drive policy
        drive_policy = policy.policy
    else:
        drive_policy = policy

    if hasattr(drive_policy, "ego_dim"):
        # Construct valid dummy observation for Drive policy
        # Retrieve needed dimensions
        ego_dim = drive_policy.ego_dim
        max_partner_objects = drive_policy.max_partner_objects
        partner_features = drive_policy.partner_features
        max_road_objects = drive_policy.max_road_objects
        road_features = drive_policy.road_features

        partner_dim = max_partner_objects * partner_features
        road_dim = max_road_objects * road_features

        # Random parts
        dummy_ego = torch.randn(batch_size, ego_dim)
        dummy_partner = torch.randn(batch_size, partner_dim)

        # Road part: continuous features + categorical feature
        road_cont_dim = road_features - 1

        # (Batch, MaxObjects, Feats-1)
        dummy_road_cont = torch.randn(batch_size, max_road_objects, road_cont_dim)

        # (Batch, MaxObjects, 1) - valid categorical values [0, 6]
        # Ensure it's 0-6 range. 7 is num_classes.
        dummy_road_cat = torch.randint(0, 7, (batch_size, max_road_objects, 1)).float()

        # Concatenate and flatten
        dummy_road_objs = torch.cat([dummy_road_cont, dummy_road_cat], dim=2)
        dummy_road = dummy_road_objs.view(batch_size, -1)

        dummy_obs = torch.cat([dummy_ego, dummy_partner, dummy_road], dim=1)
    else:
        print("Warning: Could not determine Drive policy structure. Using random observation.")
        dummy_obs = torch.randn(batch_size, obs_dim)

    # Dummy LSTM States
    hidden_size = config["rnn"]["hidden_size"]
    # LSTMCell expects (Batch, Hidden) not (NumLayers, Batch, Hidden)
    dummy_h = torch.zeros(batch_size, hidden_size)
    dummy_c = torch.zeros(batch_size, hidden_size)

    # Wrap policy for export
    onnx_policy = OnnxWrapper(
        policy,
        fake_trajectory=args.fake_trajectory,
        traj_steps=args.traj_steps,
        dt=args.traj_dt,
    )
    onnx_policy.eval()

    # Determine output path
    if not args.output:
        args.output = os.path.splitext(args.checkpoint)[0] + ".onnx"
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting to {args.output}...")

    # Dynamic axes for batch size flexibility
    output_names = ["logits", "value", "lstm_h_out", "lstm_c_out"]
    dynamic_axes = {
        "observation": {0: "batch_size"},
        "lstm_h_in": {0: "batch_size"},
        "lstm_c_in": {0: "batch_size"},
        "logits": {0: "batch_size"},
        "value": {0: "batch_size"},
        "lstm_h_out": {0: "batch_size"},
        "lstm_c_out": {0: "batch_size"},
    }
    if args.fake_trajectory:
        output_names.append("trajectory")
        dynamic_axes["trajectory"] = {0: "batch_size"}

    dummy_inputs = (dummy_obs, dummy_h, dummy_c)
    with torch.no_grad():
        torch.onnx.export(
            onnx_policy,
            dummy_inputs,
            args.output,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["observation", "lstm_h_in", "lstm_c_in"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    # Inlined weights into single onnx file
    import onnx

    model_proto = onnx.load(args.output, load_external_data=True)
    data_file = args.output + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    # Patch dim[0] of all inputs and outputs to be symbolic "batch_size"
    for tensor in list(model_proto.graph.input) + list(model_proto.graph.output):
        dim = tensor.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "batch_size"
    onnx.save(model_proto, args.output, save_as_external_data=False)
    print(f"Inlined weights into single file: {args.output}")

    print("Export complete!")
    print("\nSample Inputs shapes:")
    print(f"Observation: {dummy_obs.shape}")
    print(f"LSTM h: {dummy_h.shape}")
    print(f"LSTM c: {dummy_c.shape}")

    # Verify ONNX model
    if verify:
        print("\nVerifying ONNX model...")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        ort_session = ort.InferenceSession(args.output, sess_options)

        # PyTorch output
        with torch.no_grad():
            torch_outs = onnx_policy(*dummy_inputs)
        if args.fake_trajectory:
            torch_logits, torch_value, torch_h, torch_c, torch_traj = torch_outs
        else:
            torch_logits, torch_value, torch_h, torch_c = torch_outs

        # Output .pt files for testing
        print(f"Saving test inputs/outputs to {output_dir}")
        torch.save(dummy_inputs, os.path.join(output_dir, "test_inputs.pt"))
        torch.save(torch_outs, os.path.join(output_dir, "test_outputs.pt"))

        # ONNX Runtime output
        ort_inputs = {"observation": dummy_obs.numpy(), "lstm_h_in": dummy_h.numpy(), "lstm_c_in": dummy_c.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        # Compare outputs
        def compare(name, torch_out, ort_out, atol=1e-5):
            if isinstance(torch_out, tuple):
                for i, (t_out, o_out) in enumerate(zip(torch_out, ort_out)):
                    compare(f"{name}_{i}", t_out, o_out, atol)
                return

            try:
                np.testing.assert_allclose(torch_out.detach().numpy(), ort_out, rtol=1e-03, atol=atol)
                print(f"✔ {name} match")
            except AssertionError as e:
                print(f"✘ {name} mismatch")
                print(e)

        # Unpack ONNX outputs if logits was a tuple
        if isinstance(torch_logits, tuple):
            num_logits = len(torch_logits)
            ort_logits = ort_outs[:num_logits]
            ort_value = ort_outs[num_logits]
            ort_h = ort_outs[num_logits + 1]
            ort_c = ort_outs[num_logits + 2]
        else:
            ort_logits = ort_outs[0]
            ort_value = ort_outs[1]
            ort_h = ort_outs[2]
            ort_c = ort_outs[3]

        compare("Logits", torch_logits, ort_logits)
        compare("Value", torch_value, ort_value)
        compare("LSTM h", torch_h, ort_h)
        compare("LSTM c", torch_c, ort_c)

        if args.fake_trajectory:
            ort_traj = ort_outs[4]
            compare("Trajectory", torch_traj, ort_traj)

    # Optionally render video using the visualize binary
    if args.render:
        print("\nExporting weights to .bin for rendering...")
        from pufferlib.pufferl import export as export_bin

        bin_path = os.path.splitext(args.output)[0] + ".bin"
        export_bin(
            env_name=args.env,
            path=bin_path,
            args={"env_name": config["base"]["env_name"], "load_model_path": args.checkpoint, **config},
            vecenv=vecenv,
            policy=policy,
            silent=True,
        )
        print(f"Saved .bin weights to {bin_path}")

        # Generate INI with gigaflow-matching coefficients pinned (min=max)
        import configparser
        import tempfile

        ini_config = configparser.ConfigParser()
        ini_config.read("pufferlib/config/ocean/drive.ini")
        # Gigaflow-matching coefficients (sign matches INI convention)
        gigaflow_bounds = {
            "goal_radius": (10.0, 10.0),
            "collision": (-3.0, -3.0),
            "offroad": (-3.0, -3.0),
            "comfort": (-0.05, -0.05),
            "lane_align": (0.025, 0.025),
            "vel_align": (1.0, 1.0),
            "lane_center": (-0.0038, -0.0038),
            "center_bias": (0.0, 0.0),
            "velocity": (0.0025, 0.0025),
            "reverse": (-0.005, -0.005),
            "traffic_light": (-1.0, -1.0),
            "timestep": (-0.000025, -0.000025),
            "overspeed": (-1.0, -1.0),
            "throttle": (1.0, 1.0),
            "steer": (1.0, 1.0),
            "acc": (1.0, 1.0),
        }
        for key, (lo, hi) in gigaflow_bounds.items():
            ini_config.set("env", f"reward_bound_{key}_min", str(lo))
            ini_config.set("env", f"reward_bound_{key}_max", str(hi))
        ini_config.set("env", "goal_radius", "10.0")
        ini_config.set("env", "max_goal_speed", "3.0")
        ini_config.set("env", "episode_length", "1000")
        ini_config.set("env", "resample_frequency", "0")
        ini_config.set("env", "termination_mode", "0")
        ini_config.set("env", "goal_behavior", "1")
        fd, ini_path = tempfile.mkstemp(suffix=".ini", prefix="gigaflow_")
        with os.fdopen(fd, "w") as f:
            ini_config.write(f)

        # Build visualize binary
        import subprocess

        subprocess.run(["bash", "scripts/build_ocean.sh", "visualize", "local"], check=True)

        # Pick a map
        map_dir = config["env"].get("map_dir", "resources/drive/binaries/carla_2D")
        map_files = sorted([f for f in os.listdir(map_dir) if f.endswith(".bin")])
        if not map_files:
            print(f"No maps in {map_dir}")
        else:
            map_path = os.path.join(map_dir, map_files[0])
            output_video = os.path.splitext(args.output)[0] + "_video.mp4"
            cmd = [
                "./visualize",
                "--config",
                ini_path,
                "--policy-name",
                bin_path,
                "--map-name",
                map_path,
                "--view",
                "both",
                "--output-topdown",
                output_video,
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)
            print(f"Video saved to {output_video}")
            print(f"INI config used: {ini_path}")
            # os.remove(ini_path)  # keep for debugging


if __name__ == "__main__":
    export_to_onnx(verify=True)
