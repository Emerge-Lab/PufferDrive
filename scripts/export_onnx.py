"""Export a trained PufferDrive policy checkpoint (.pt) to ONNX format.

The exported ONNX model accepts an observation vector (see
observation_spec() in torch.py for the exact layout and feature
descriptions), plus LSTM hidden states, and produces action logits,
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


class OnnxWrapper(torch.nn.Module):
    """Wraps the LSTM policy for ONNX export.

    The network's decode_actions returns a tuple of tensors for multi-discrete,
    each of shape (batch * T, joint_size).  ONNX needs a single flat tensor,
    so we concatenate and reshape to (batch, joint_size * T).
    """

    def __init__(self, policy, actions_trajectory_length=80):
        super().__init__()
        self.policy = policy
        self.T = actions_trajectory_length

    def forward(self, observation, h, c):
        # Reconstruct the state dictionary expected by LSTMWrapper
        # state must be mutable as forward_eval updates it
        state = {"lstm_h": h, "lstm_c": c}

        # Call forward_eval
        logits, value = self.policy.forward_eval(observation, state)

        # Extract updated states
        new_h = state["lstm_h"]
        new_c = state["lstm_c"]

        # Flatten tuple logits for ONNX: (batch*T, joint_size) → (batch, joint_size*T)
        if isinstance(logits, tuple):
            # Multi-discrete: single element tuple of (batch*T, joint_size)
            flat = logits[0]  # (batch*T, joint_size)
            batch = observation.shape[0]
            joint_size = flat.shape[-1]
            # Reshape: (batch*T, joint_size) → (batch, T, joint_size) → (batch, T*joint_size)
            flat = flat.reshape(batch, self.T, joint_size).reshape(batch, self.T * joint_size)
            logits = flat

        return logits, value, new_h, new_c


def export_to_onnx(verify=True):
    parser = argparse.ArgumentParser(description="Export PufferDrive model to ONNX")
    parser.add_argument("--env", type=str, default="puffer_drive", help="Environment name")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model_puffer_drive_000250.pt",
        help="Path to .pt checkpoint",
    )
    parser.add_argument("--output", type=str, help="Output .onnx file path")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")

    args = parser.parse_args()

    # Load environment
    config = load_config(args.env)
    package = config["base"]["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(args.env)
    env_kwargs = config["env"]
    vecenv = pufferlib.vector.make(make_env, env_kwargs=env_kwargs, backend=pufferlib.vector.Serial, num_envs=1)

    # Initialize Policy
    print("Initializing Policy...")
    actions_trajectory_length = int(config["train"].get("actions_trajectory_length", 80))
    policy = Drive(vecenv.driver_env, actions_trajectory_length=actions_trajectory_length, **config["policy"])
    if config["base"]["rnn_name"]:
        print("Wrapping with LSTM...")
        policy = pufferlib.models.LSTMWrapper(
            vecenv.driver_env,
            policy,
            actions_trajectory_length=actions_trajectory_length,
            **config["rnn"],
        )

    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Handle both full checkpoint dict and raw state dict
    if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
        state_dict = checkpoint["agent_state_dict"]
    else:
        state_dict = checkpoint

    # Strip compile prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v

    policy.load_state_dict(new_state_dict)
    policy.eval()

    # Prepare inputs for ONNX export
    print("Preparing sample inputs...")
    batch_size = 1

    obs_space = vecenv.single_observation_space
    # The LSTMWrapper expects (B, ObsDim)
    obs_dim = np.prod(obs_space.shape)

    # Create Dummy Observation
    if config["base"]["rnn_name"]:
        drive_policy = policy.policy
    else:
        drive_policy = policy

    if hasattr(drive_policy, "ego_dim"):
        # Build a physically valid structured observation using binding constants
        dummy_obs = drive_policy.build_structured_observation(
            dynamics_model=config["env"].get("dynamics_model", "classic"),
            reward_conditioning=bool(config["env"].get("reward_conditioning", 0)),
            batch_size=batch_size,
        )

        # Print observation spec for reference
        spec = drive_policy.observation_spec()
        print(f"\nObservation layout: {spec['layout']}")
        print(f"  Ego:      offset={spec['ego']['offset']}, dim={spec['ego']['total_dim']}")
        if spec.get("reward_conditioning"):
            rc = spec["reward_conditioning"]
            print(f"  Conditioning: offset={rc['offset']}, dim={rc['total_dim']}")
        print(
            f"  Partners: offset={spec['partners']['offset']}, dim={spec['partners']['total_dim']} ({spec['partners']['count']}x{spec['partners']['features_per_object']})"
        )
        print(
            f"  Road:     offset={spec['road_segments']['offset']}, dim={spec['road_segments']['total_dim']} ({spec['road_segments']['count']}x{spec['road_segments']['features_per_object']})"
        )
        pa = spec.get("past_actions_trajectory", {})
        if pa:
            print(
                f"  PastAct:  offset={pa['offset']}, dim={pa['total_dim']} "
                f"({pa['actions_per_step']}x{pa['trajectory_length']})"
            )
        print(f"  Base obs: {spec['base_obs_dim']}")
        print(f"  Total:    {spec['total_dim']}  (base_obs + past_actions_trajectory)")
    else:
        print("Warning: Could not determine Drive policy structure. Using random observation.")
        dummy_obs = torch.randn(batch_size, obs_dim)

    # Dummy LSTM States
    hidden_size = config["rnn"]["hidden_size"]
    # LSTMCell expects (Batch, Hidden) not (NumLayers, Batch, Hidden)
    dummy_h = torch.zeros(batch_size, hidden_size)
    dummy_c = torch.zeros(batch_size, hidden_size)

    # Wrap policy for export
    onnx_policy = OnnxWrapper(policy, actions_trajectory_length=actions_trajectory_length)
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
    dynamic_axes = {
        "observation": {0: "batch_size"},
        "lstm_h_in": {0: "batch_size"},
        "lstm_c_in": {0: "batch_size"},
        "logits": {0: "batch_size"},
        "value": {0: "batch_size"},
        "lstm_h_out": {0: "batch_size"},
        "lstm_c_out": {0: "batch_size"},
    }

    dummy_inputs = (dummy_obs, dummy_h, dummy_c)
    torch.onnx.export(
        onnx_policy,
        dummy_inputs,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["observation", "lstm_h_in", "lstm_c_in"],
        output_names=["logits", "value", "lstm_h_out", "lstm_c_out"],
        dynamic_axes=dynamic_axes,
    )

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

        with torch.no_grad():
            torch_logits, torch_value, torch_h, torch_c = onnx_policy(*dummy_inputs)

        # OnnxWrapper already flattens tuple → (batch, joint_size*T)
        T = actions_trajectory_length
        joint_size = sum(drive_policy.atn_dim)
        print(f"\nLogits shape (flat): {torch_logits.shape}  (batch, {joint_size}×{T} = {joint_size * T})")

        # Output .pt files for testing
        print(f"Saving test inputs/outputs to {output_dir}")
        torch.save(dummy_inputs, os.path.join(output_dir, "test_inputs.pt"))
        torch.save((torch_logits, torch_value, torch_h, torch_c), os.path.join(output_dir, "test_outputs_raw.pt"))

        # ONNX Runtime output
        ort_inputs = {"observation": dummy_obs.numpy(), "lstm_h_in": dummy_h.numpy(), "lstm_c_in": dummy_c.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        # Compare outputs (logits is now a single flat tensor, not a tuple)
        def compare(name, torch_out, ort_out, atol=1e-5):
            try:
                np.testing.assert_allclose(torch_out.detach().numpy(), ort_out, rtol=1e-03, atol=atol)
                print(f"✔ {name} match")
            except AssertionError as e:
                print(f"✘ {name} mismatch")
                print(e)

        ort_logits = ort_outs[0]
        ort_value = ort_outs[1]
        ort_h = ort_outs[2]
        ort_c = ort_outs[3]

        compare("Logits", torch_logits, ort_logits)
        compare("Value", torch_value, ort_value)
        compare("LSTM h", torch_h, ort_h)
        compare("LSTM c", torch_c, ort_c)

        # --- Construct and save decoded action outputs ---
        dynamics_model = config["env"].get("dynamics_model", "classic")

        # Reconstruct the tuple logits format that construct_action_output expects
        # OnnxWrapper flattened (batch*T, joint_size) → (batch, joint_size*T)
        # Undo: (batch, joint_size*T) → (batch, T, joint_size) → (batch*T, joint_size) → tuple
        logits_for_decode = torch_logits.reshape(batch_size, T, joint_size)
        logits_for_decode = logits_for_decode.reshape(batch_size * T, joint_size)
        logits_tuple = (logits_for_decode,)  # multi-discrete single-element tuple

        action_output = Drive.construct_action_output(
            logits_tuple,
            dynamics_model=dynamics_model,
            actions_trajectory_length=T,
        )

        # Print action spec for reference
        atn_spec = drive_policy.action_spec()
        print(f"\nAction spec ({atn_spec['dynamics_model']}, {atn_spec['mode']}):")
        print(f"  Joint action size: {atn_spec['joint_action_size']}")
        print(f"  Trajectory length: {atn_spec['trajectory_length']}")
        print(f"  Flat logits dim:   {atn_spec['flat_logits_dim']}")
        print(f"  Decomposition: {atn_spec['decomposition']}")
        p = atn_spec["primary"]
        s = atn_spec["secondary"]
        print(f"  Primary:   {p['name']} ({p['unit']}), {p['num_actions']} values: {p['values']}")
        print(f"  Secondary: {s['name']} ({s['unit']}), {s['num_actions']} values: {s['values']}")

        # Print decoded action for the dummy input
        meta = action_output["metadata"]
        print(f"\nDecoded action (categorical sample at t=0) for test observation:")
        print(f"  Joint action index: {action_output['joint_action'].item()}")
        print(
            f"  {meta['primary_name']}_idx: {action_output['primary_idx'].item()}"
            f"  → {action_output[meta['primary_name']].item():.3f} {meta['primary_unit']}"
        )
        print(
            f"  {meta['secondary_name']}_idx: {action_output['secondary_idx'].item()}"
            f"  → {action_output[meta['secondary_name']].item():.3f} {meta['secondary_unit']}"
        )
        print(f"\nFull trajectory ({T} timesteps, argmax):")
        print(f"  trajectory_joint shape: {action_output['trajectory_joint'].shape}")
        print(f"  trajectory_joint[0,:5]: {action_output['trajectory_joint'][0, :5].tolist()} ...")

        # Save complete output checkpoint: raw network outputs + decoded actions + trajectory
        output_checkpoint = {
            # Raw network outputs
            "logits_flat": torch_logits,  # (batch, joint_size*T)
            "logits_trajectory": action_output["trajectory_logits"],  # (batch, T, joint_size)
            "value": torch_value,
            "lstm_h": torch_h,
            "lstm_c": torch_c,
            # Decoded discrete actions at t=0 (categorical sampling, matches training)
            "joint_action": action_output["joint_action"],
            "primary_idx": action_output["primary_idx"],
            "secondary_idx": action_output["secondary_idx"],
            f"{meta['primary_name']}": action_output[meta["primary_name"]],
            f"{meta['secondary_name']}": action_output[meta["secondary_name"]],
            "log_prob": action_output["log_prob"],
            "entropy": action_output["entropy"],
            # Full trajectory (all T timesteps, argmax)
            "trajectory_joint": action_output["trajectory_joint"],
            f"trajectory_{meta['primary_name']}": action_output[f"trajectory_{meta['primary_name']}"],
            f"trajectory_{meta['secondary_name']}": action_output[f"trajectory_{meta['secondary_name']}"],
            # Metadata for the deployment side to reconstruct decoding
            "action_metadata": action_output["metadata"],
        }
        output_path = os.path.join(output_dir, "test_outputs.pt")
        torch.save(output_checkpoint, output_path)
        print(f"\n✔ Saved output checkpoint (raw + decoded + trajectory) to {output_path}")


if __name__ == "__main__":
    export_to_onnx(verify=True)
