import numbers
import os
import random
import shutil
import subprocess

import numpy as np
import torch


def torch_device(device):
    if isinstance(device, int):
        return torch.device("cuda", device) if torch.cuda.is_available() else torch.device("cpu")
    return device


def is_cuda_device(device):
    if isinstance(device, int):
        return torch.cuda.is_available()
    device = torch.device(device)
    return device.type == "cuda"


def base_policy(policy):
    return policy.module if hasattr(policy, "module") else policy


def clean_state_key(key):
    prefixes = ("module.", "_orig_mod.")
    while key.startswith(prefixes):
        key = key.split(".", 1)[1]
    return key


def clean_policy_state_dict(state_dict):
    return {clean_state_key(k): v for k, v in state_dict.items()}


def logits_to_float(logits):
    if isinstance(logits, torch.distributions.Normal):
        return torch.distributions.Normal(logits.loc.float(), logits.scale.float())
    return logits.float()


def abbreviate(num, b2, c2):
    if num < 1e3:
        return f"{b2}{num}{c2}"
    elif num < 1e6:
        return f"{b2}{num / 1e3:.1f}{c2}K"
    elif num < 1e9:
        return f"{b2}{num / 1e6:.1f}{c2}M"
    elif num < 1e12:
        return f"{b2}{num / 1e9:.1f}{c2}B"
    else:
        return f"{b2}{num / 1e12:.2f}{c2}T"


def duration(seconds, b2, c2):
    if seconds < 0:
        return f"{b2}0{c2}s"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"


def fmt_perf(name, color, delta_ref, prof, b2, c2):
    percent = 0 if delta_ref == 0 else int(100 * prof["buffer"] / delta_ref - 1e-5)
    return f"{color}{name}", duration(prof["elapsed"], b2, c2), f"{b2}{percent:2d}{c2}%"


def dist_sum(value, device):
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()


def dist_mean(value, device):
    if not torch.distributed.is_initialized():
        return value

    return dist_sum(value, device) / torch.distributed.get_world_size()


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    rng_state = state.get("rng_state")
    if not rng_state:
        return

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in rng_state:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["cuda"]])


def downsample(data_list, num_points):
    if not data_list or num_points <= 0:
        return []
    if num_points == 1:
        return [data_list[-1]]
    if len(data_list) <= num_points:
        return data_list

    last = data_list[-1]
    data_list = data_list[:-1]

    data_np = np.array(data_list)
    num_points -= 1  # one down for the last one

    n = (len(data_np) // num_points) * num_points
    data_np = data_np[-n:] if n > 0 else data_np
    downsampled = data_np.reshape(num_points, -1).mean(axis=1)

    return downsampled.tolist() + [last]


def get_git_metadata():
    git_metadata = {
        "commit_hash": os.environ.get("GITHUB_SHA") or os.environ.get("COMMIT_SHA"),
    }

    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if shutil.which("git") is None:
            return git_metadata

        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        if git_metadata["commit_hash"] is None:
            git_metadata["commit_hash"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
    except (OSError, subprocess.SubprocessError):
        pass

    return git_metadata


def save_experiment_config(args, path):
    import yaml
    import json

    experiment_dir = path
    os.makedirs(experiment_dir, exist_ok=True)

    # Save config as yaml
    config_yaml_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_yaml_path, "w") as f:
        # Convert defaultdict to dict for cleaner output
        config = json.loads(json.dumps(args))
        yaml.dump(config, f)


def global_agent_steps(pufferl):
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return int(pufferl.global_step * world_size)


def derive_rank_seeds(vec_seed, train_seed, world_size, global_rank):
    """Deterministic per-rank (torch_seed, env_seed): DDP ranks share weights, so identical seeds
    would collect duplicate experience. global_rank is torchrun's global RANK, not LOCAL_RANK."""
    torch_seed = train_seed * world_size + global_rank
    env_seed = vec_seed
    if env_seed is not None:
        env_seed = int(np.random.SeedSequence([env_seed, train_seed, global_rank]).generate_state(1)[0])
    return torch_seed, env_seed


def reduce_environment_metrics(metric_lists):
    # Preserve raw sums and sample counts so log batches with different numbers
    # of completed episodes contribute with the correct weight.
    local_metrics = {}
    for key, values in metric_lists.items():
        if not values or not isinstance(values[0], numbers.Number):
            continue
        local_metrics[key] = (float(np.sum(values)), len(values))

    reduced_metrics = {}
    total_distance = local_metrics.get("total_distance_travelled_sum")
    total_infractions = local_metrics.get("total_infraction_count")
    for key, (value_sum, value_count) in local_metrics.items():
        if key in ("total_distance_travelled_sum", "total_infraction_count"):
            continue
        reduced_metrics[key] = value_sum / max(value_count, 1)

    if total_distance is not None and total_infractions is not None:
        reduced_metrics["total_distance_travelled"] = total_distance[0]
        reduced_metrics["total_infractions"] = total_infractions[0]
        reduced_metrics["avg_distance_per_infraction"] = total_distance[0] / max(total_infractions[0], 1.0)

    return reduced_metrics
