import argparse
import os
import torch
import importlib
import numpy as np

import pufferlib.utils
import pufferlib.vector


def load_config(env_name, config_dir=None):
    # Minimal config loader based on pufferl.py
    import configparser
    import glob
    from collections import defaultdict
    import ast

    if config_dir is None:
        puffer_dir = os.path.dirname(os.path.realpath(pufferlib.__file__))
    else:
        puffer_dir = config_dir

    puffer_config_dir = os.path.join(puffer_dir, "config/**/*.ini")
    puffer_default_config = os.path.join(puffer_dir, "config/default.ini")

    found = False
    for path in glob.glob(puffer_config_dir, recursive=True):
        p = configparser.ConfigParser()
        p.read([puffer_default_config, path])
        if env_name in p["base"]["env_name"].split():
            found = True
            break

    if not found:
        raise ValueError(f"No config for env_name {env_name}")

    def puffer_type(value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    args = defaultdict(dict)
    for section in p.sections():
        for key in p[section]:
            value = puffer_type(p[section][key])
            args[section][key] = value

    return args


# Export PufferDrive model weights to .bin
def export_weights():
    parser = argparse.ArgumentParser(description="Export PufferDrive model weights to .bin")
    parser.add_argument("--env", type=str, default="puffer_drive", help="Environment name")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pufferlib/resources/drive/model_puffer_drive_000100.bin",
        help="Output .bin file path",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.env)

    # Load environment to get observation/action spaces
    package = config["base"]["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(args.env)

    # Use valid dummy env to initialize policy
    # Ensure env args/kwargs are correctly passed as expected by make()
    env_kwargs = config["env"]

    vecenv = pufferlib.vector.make(make_env, env_kwargs=env_kwargs, backend=pufferlib.vector.Serial, num_envs=1)

    # Initialize Policy
    policy_name = config["base"]["policy_name"]
    policy_cls = getattr(__import__("pufferlib.ocean.policies", fromlist=[policy_name]), policy_name)
    print(f"Initializing {policy_name}...")
    policy = policy_cls(vecenv.driver_env, **config["policy"])

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

    # Export Weights
    print(f"Exporting weights to {args.output}...")
    weights = []
    total_params = 0
    for name, param in policy.named_parameters():
        param_flat = param.data.cpu().numpy().flatten()
        weights.append(param_flat)
        count = param_flat.size
        print(f"  {name}: {param.shape} -> {count} params")
        total_params += count

    weights = np.concatenate(weights)
    weights.tofile(args.output)
    print(f"Success! Saved {len(weights)} weights ({total_params} params) to {args.output}")


if __name__ == "__main__":
    export_weights()
