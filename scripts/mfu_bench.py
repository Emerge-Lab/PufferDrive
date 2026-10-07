"""Model FLOP utilization of the real PPO loop: rollout, update and whole epoch, plus agent steps per second.

Builds the trainer exactly as `puffer train puffer_drive` does, runs warm-up epochs (torch.compile), then measures
steady-state epochs. Model FLOPs are counted analytically from the policy's linear layers, so no profiler touches
the compiled policy (a TorchDispatchMode around it makes dynamo skip the frame for the rest of the process).

Usage:
    python scripts/mfu_bench.py --num-envs 16 --minibatch 131072 [--fused] [-- train.seed=3 ...]
"""

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict


os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
import torch._dynamo
from profile_policy_step import PEAK_BF16_TFLOPS_BY_CAPABILITY
from torch import nn


ENV_NAME = "puffer_drive"
RECOMPILE_EPOCH_FACTOR = 1.25
SLOT_ENCODER_COUNTS = {
    "lane_encoder": "obs_slots_lane_kept",
    "boundary_encoder": "obs_slots_boundary_kept",
    "partner_encoder": "obs_slots_partners_n",
    "traffic_control_encoder": "obs_slots_traffic_controls_n",
}


def model_flops_per_transition(policy):
    """Forward and forward+backward FLOPs of one transition through the policy (matmuls only, the MFU convention)."""
    forward = 0.0
    backward = 0.0
    for name, module in policy.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        rows = 1
        backbone_name = name.split(".")[0]
        backbone = getattr(policy, backbone_name, None)
        for encoder_name, count_attribute in SLOT_ENCODER_COUNTS.items():
            if encoder_name in name and backbone is not None:
                rows = getattr(backbone, count_attribute)
        layer_flops = 2.0 * rows * module.in_features * module.out_features
        forward += layer_flops
        # the first layer of every encoder reads observations, which need no input gradient
        first_encoder_layer = "encoder" in name and name.endswith(".0")
        backward += layer_flops if first_encoder_layer else 2.0 * layer_flops
    return forward, forward + backward


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start, result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--minibatch", type=int, default=131072)
    parser.add_argument("--warmup", type=int, default=2, help="epochs before measuring (compile, recompiles)")
    parser.add_argument("--epochs", type=int, default=3, help="measured epochs")
    parser.add_argument("--fused", action="store_true", help="policy.fused_slot_encoder=true")
    parser.add_argument("--rollout-only", action="store_true", help="time only the rollout (data collection)")
    parser.add_argument("overrides", nargs="*", help="extra hydra overrides, e.g. train.seed=3")
    args = parser.parse_args()

    data_dir = tempfile.mkdtemp(prefix="mfu_bench_")
    sys.argv = [
        sys.argv[0],
        f"vec.num_envs={args.num_envs}",
        "env.preload_map_cache=True",
        f"train.minibatch_size={args.minibatch}",
        f"train.max_minibatch_size={args.minibatch}",
        f"train.data_dir={data_dir}",
        f"policy.fused_slot_encoder={str(args.fused).lower()}",
        "wandb=False",
        "tb=False",
        "neptune=False",
        *args.overrides,
    ]
    import pufferlib.pufferl as P

    config = P.load_config(ENV_NAME)
    config = P.normalize_puffer_drive_config(config, "training")
    P.validate_puffer_drive_config(config, "training")
    P.validate_puffer_drive_resources(config, "training")
    torch_seed, env_seed = P.derive_rank_seeds(config["vec"]["seed"], config["train"]["seed"], 1, 0)
    torch.manual_seed(torch_seed)
    vecenv = P.load_env(ENV_NAME, config, seed=env_seed)
    policy = P.load_policy(config, vecenv, ENV_NAME)
    train_config = dict(**config["train"], env=ENV_NAME, eval=config.get("eval", {}), run_name=config["run_name"])
    pufferl = P.PuffeRL(train_config, vecenv, policy)
    pufferl.profile.frequency = 10**9

    forward_flops, train_flops = model_flops_per_transition(pufferl.uncompiled_policy)
    counters = defaultdict(int)
    original_ppo_loss = pufferl._ppo_loss

    def counting_ppo_loss(mb_obs, *rest, **kwargs):
        counters["samples"] += mb_obs.shape[0]
        counters["minibatches"] += 1
        return original_ppo_loss(mb_obs, *rest, **kwargs)

    pufferl._ppo_loss = counting_ppo_loss
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    peak_tflops = PEAK_BF16_TFLOPS_BY_CAPABILITY.get(capability)
    agent_steps = train_config["batch_size"]
    print(
        f"{properties.name} capability {capability}, {args.num_envs} envs, {vecenv.num_agents} agents, "
        f"minibatch {pufferl.minibatch_size}, fused_slot_encoder={args.fused}, "
        f"model FLOP per transition: forward {forward_flops / 1e6:.2f} MFLOP, update {train_flops / 1e6:.2f} MFLOP",
        flush=True,
    )

    try:

        def update_or_skip():
            if args.rollout_only:
                pufferl.epoch += 1  # keep the trainer's epoch bookkeeping moving without an update
                return 0.0, None
            return timed(pufferl.train)

        for epoch in range(args.warmup):
            rollout_seconds, _ = timed(pufferl.evaluate)
            update_seconds, _ = update_or_skip()
            print(f"warmup epoch {epoch}: rollout {rollout_seconds:.2f} s, update {update_seconds:.2f} s", flush=True)
        frames = dict(torch._dynamo.utils.counters["frames"])
        epochs = []
        for epoch in range(args.epochs):
            counters.clear()
            rollout_seconds, _ = timed(pufferl.evaluate)
            update_seconds, _ = update_or_skip()
            epochs.append((rollout_seconds, update_seconds, counters["samples"]))
            print(
                f"epoch {epoch}: rollout {rollout_seconds:.2f} s, update {update_seconds:.2f} s, "
                f"kept transitions {counters['samples']}",
                flush=True,
            )
    finally:
        pufferl.utilization.stop()
        vecenv.close()

    if args.rollout_only:
        rollout = sum(e[0] for e in epochs) / len(epochs)
        print(
            f"ROLLOUT_RESULT envs={args.num_envs} agents={vecenv.num_agents} rollout_s={rollout:.3f} agent_steps_per_s={agent_steps / rollout:.0f}"
        )
        return
    fastest_update = min(update for _, update, _ in epochs)
    steady = [e for e in epochs if e[1] < RECOMPILE_EPOCH_FACTOR * fastest_update]
    rollout = sum(e[0] for e in steady) / len(steady)
    update = sum(e[1] for e in steady) / len(steady)
    kept = sum(e[2] for e in steady) / len(steady)
    phases = {
        "update": (kept * train_flops, update),
        "rollout": (agent_steps * forward_flops, rollout),
        "full_loop": (kept * train_flops + agent_steps * forward_flops, update + rollout),
    }
    result = {
        "gpu": properties.name,
        "capability": list(capability),
        "fused": args.fused,
        "num_envs": args.num_envs,
        "minibatch": pufferl.minibatch_size,
        "steady_epochs": len(steady),
        "kept_transitions_per_epoch": kept,
        "agent_steps_per_epoch": agent_steps,
        "sps": agent_steps / (update + rollout),
        "peak_tflops": peak_tflops,
        "dynamo_frames": frames,
        "max_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    for phase, (flops, seconds) in phases.items():
        result[f"{phase}_seconds"] = seconds
        result[f"{phase}_tflops"] = flops / seconds / 1e12
        if peak_tflops:
            result[f"{phase}_mfu"] = 100.0 * flops / seconds / (peak_tflops * 1e12)
    print("MFU_RESULT " + json.dumps(result))
    label = "fused" if args.fused else "reference"
    for phase in phases:
        mfu = f", MFU {result[f'{phase}_mfu']:.1f}%" if peak_tflops else ""
        print(
            f"{label} {phase:9s}: {result[f'{phase}_seconds']:6.2f} s/epoch, {result[f'{phase}_tflops']:6.1f} TFLOP/s{mfu}"
        )
    print(f"{label} SPS {result['sps'] / 1e3:.0f}k agent-steps/s, peak GPU memory {result['max_memory_gb']:.1f} GB")


if __name__ == "__main__":
    main()
