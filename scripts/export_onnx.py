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


class OnnxWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation, h, c):
        # Reconstruct the state dictionary expected by LSTMWrapper
        # state must be mutable as forward_eval updates it
        state = {"lstm_h": h, "lstm_c": c}

        # Call forward_eval
        logits, value = self.policy.forward_eval(observation, state)

        # Extract updated states
        new_h = state["lstm_h"]
        new_c = state["lstm_c"]

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
    onnx_policy = OnnxWrapper(policy)
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

        # PyTorch output
        with torch.no_grad():
            torch_logits, torch_value, torch_h, torch_c = onnx_policy(*dummy_inputs)

        # Output .pt files for testing
        print(f"Saving test inputs/outputs to {output_dir}")
        torch.save(dummy_inputs, os.path.join(output_dir, "test_inputs.pt"))
        torch.save((torch_logits, torch_value, torch_h, torch_c), os.path.join(output_dir, "test_outputs.pt"))

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

        # Export example input and output to .pt files


if __name__ == "__main__":
    export_to_onnx(verify=True)
