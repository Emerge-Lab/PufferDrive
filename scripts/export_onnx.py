"""Export a trained puffer-4 policy checkpoint (.bin / .pt state dict) to ONNX.

puffer-4's policy is a generic `pufferlib.models.Policy(encoder, decoder, network)`
composition. The network is usually a recurrent module (MinGRU, LSTM, or GRU) whose
`initial_state` returns a tuple of hidden-state tensors. This script exports a wrapper
that takes `(observation, *state_in)` and returns `(logits..., value, *state_out)`.

Usage:
    python scripts/export_onnx.py --checkpoint path/to/model.bin [--output path.onnx]

Notes:
    * Checkpoints on puffer-4 are saved via `torch.save(policy.state_dict(), path)`
      with a `.bin` extension (see `pufferlib/pufferl.py` and
      `pufferlib/torch_pufferl.py::save_weights`). We load with `torch.load` just
      like a `.pt` file.
    * This script patches `sys.argv` before calling `load_config(env_name)` so
      puffer-4's argparse-based config loader doesn't try to interpret our CLI flags.
    * MinGRU / GRU / LSTM networks expose `initial_state(batch, device)` returning
      a tuple; the wrapper unpacks that tuple into individual ONNX inputs/outputs.
"""

import argparse
import os
import sys

import numpy as np
import torch


class OnnxWrapper(torch.nn.Module):
    """Wraps a `pufferlib.models.Policy` for ONNX export.

    Args:
        policy: a Policy instance whose `forward_eval(x, state)` returns
            `(logits, values, new_state)` where `state` and `new_state` are tuples
            of tensors (one per hidden state component).
        num_state: number of state tensors produced by `policy.initial_state`.
    """

    def __init__(self, policy, num_state: int):
        super().__init__()
        self.policy = policy
        self.num_state = num_state

    def forward(self, observation, *state_in):
        state = tuple(state_in)
        logits, values, new_state = self.policy.forward_eval(observation, state)
        if isinstance(logits, (list, tuple)):
            return (*logits, values, *new_state)
        return (logits, values, *new_state)


def _strip_compile_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state[k[len("_orig_mod.") :]] = v
        else:
            new_state[k.replace("module.", "")] = v
    return new_state


def main():
    parser = argparse.ArgumentParser(description="Export puffer-4 policy to ONNX")
    parser.add_argument("--env", type=str, default="drive", help="Environment name (default: drive)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .bin / .pt checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output .onnx file path")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size for export")
    parser.add_argument("--verify", action="store_true", help="Verify the exported model against PyTorch outputs")
    args = parser.parse_args()

    # Prevent puffer-4's load_config from seeing our CLI flags — it uses argparse
    # globally and will reject unknown flags.
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        from pufferlib.pufferl import load_config
        from pufferlib.torch_pufferl import load_policy

        config = load_config(args.env)
    finally:
        sys.argv = saved_argv

    # Create a minimal CPU vec env so `load_policy` can read obs_size / act_sizes.
    # `_C.create_vec` expects the full args dict and the gpu flag.
    from pufferlib import _C

    vec = _C.create_vec(config, False)  # force CPU

    print("Instantiating policy...")
    policy = load_policy(config, vec)
    policy.eval()

    # Load checkpoint weights on top of the freshly initialized policy.
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "agent_state_dict" in state_dict:
        state_dict = state_dict["agent_state_dict"]
    policy.load_state_dict(_strip_compile_prefix(state_dict))
    policy.to("cpu")
    policy.eval()

    # Prepare dummy inputs
    batch_size = args.batch_size
    obs_size = vec.obs_size
    dummy_obs = torch.randn(batch_size, obs_size)
    initial_state = policy.initial_state(batch_size, "cpu")
    if not isinstance(initial_state, tuple):
        initial_state = (initial_state,)
    num_state = len(initial_state)

    # Build wrapper
    wrapper = OnnxWrapper(policy, num_state=num_state)
    wrapper.eval()

    # Determine output path
    if not args.output:
        args.output = os.path.splitext(args.checkpoint)[0] + ".onnx"
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Figure out how many logit tensors the decoder produces by running once.
    with torch.no_grad():
        sample_out = wrapper(dummy_obs, *initial_state)
    num_logit_tensors = len(sample_out) - 1 - num_state
    if num_logit_tensors < 1:
        raise RuntimeError(
            f"Unexpected wrapper output length {len(sample_out)} with num_state={num_state}"
        )

    # Name the outputs explicitly so ONNX consumers can pick them apart.
    logit_names = (
        ["logits"] if num_logit_tensors == 1 else [f"logits_{i}" for i in range(num_logit_tensors)]
    )
    state_in_names = [f"state_in_{i}" for i in range(num_state)]
    state_out_names = [f"state_out_{i}" for i in range(num_state)]
    input_names = ["observation"] + state_in_names
    output_names = logit_names + ["value"] + state_out_names

    dynamic_axes = {"observation": {0: "batch_size"}, "value": {0: "batch_size"}}
    for n in logit_names:
        dynamic_axes[n] = {0: "batch_size"}
    for n in state_in_names + state_out_names:
        # Hidden-state tensors are typically (num_layers, batch, hidden), so batch is dim 1.
        dynamic_axes[n] = {1: "batch_size"}

    print(f"Exporting to {args.output}")
    torch.onnx.export(
        wrapper,
        (dummy_obs, *initial_state),
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("Export complete")
    print(f"  observation: {tuple(dummy_obs.shape)}")
    for i, s in enumerate(initial_state):
        print(f"  state_in_{i}: {tuple(s.shape)}")

    if args.verify:
        import onnxruntime as ort

        print("Verifying ONNX model with onnxruntime...")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        session = ort.InferenceSession(args.output, sess_options)

        with torch.no_grad():
            torch_outs = wrapper(dummy_obs, *initial_state)

        ort_inputs = {"observation": dummy_obs.numpy()}
        for name, tensor in zip(state_in_names, initial_state):
            ort_inputs[name] = tensor.numpy()
        ort_outs = session.run(None, ort_inputs)

        def compare(name, torch_out, ort_out, atol=1e-5):
            try:
                np.testing.assert_allclose(torch_out.detach().cpu().numpy(), ort_out, rtol=1e-3, atol=atol)
                print(f"  ok  {name}")
            except AssertionError as exc:
                print(f"  FAIL {name}: {exc}")

        for name, t_out, o_out in zip(output_names, torch_outs, ort_outs):
            compare(name, t_out, o_out)


if __name__ == "__main__":
    main()
