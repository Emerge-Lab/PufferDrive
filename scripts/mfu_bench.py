"""Model FLOP utilization of the real PPO loop: rollout, update and whole epoch, plus agent steps per second.

Builds the trainer exactly as `puffer train puffer_drive` does, runs warm-up epochs (torch.compile), then measures
steady-state epochs. Model FLOPs are counted analytically from the policy's linear layers, so no profiler touches
the compiled policy (a TorchDispatchMode around it makes dynamo skip the frame for the rest of the process).

Usage:
    python scripts/mfu_bench.py --num-envs 16 --minibatch 131072 [--fused] [--profile] [-- train.seed=3 ...]
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
from profile_policy_step import peak_bf16_tflops
from torch import nn


ENV_NAME = "puffer_drive"
RECOMPILE_EPOCH_FACTOR = 1.25
PROFILE_SECTIONS = (
    "env",
    "eval_copy",
    "eval_forward",
    "eval_misc",
    "train_copy",
    "train_forward",
    "learn",
    "train_misc",
)
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


def print_kernel_table(prof, label, wall_seconds, top=18):
    """Per-kernel self CUDA time of one profiled phase, the share of all kernel time, and the GPU busy fraction."""
    events = [e for e in prof.key_averages() if getattr(e, "device_type", None) is not None and e.self_device_time_total > 0]
    events = [e for e in events if str(e.device_type).endswith("CUDA")]
    total_us = sum(e.self_device_time_total for e in events)
    print(f"kernel table, {label}: {total_us / 1e6:.3f} s of CUDA kernel time in {wall_seconds:.3f} s wall "
          f"(GPU busy {100 * total_us / 1e6 / max(wall_seconds, 1e-9):.0f}%), {len(events)} distinct kernels")
    print(f"  {'share':>6s} {'ms':>9s} {'calls':>7s}  kernel")
    for e in sorted(events, key=lambda e: -e.self_device_time_total)[:top]:
        print(f"  {100 * e.self_device_time_total / total_us:5.1f}% {e.self_device_time_total / 1e3:9.1f} {e.count:7d}  {e.key[:110]}")


def measure_bf16_peak(seconds=3.0, size=8192):
    """Achieved dense bf16 matmul TFLOP/s with fp32 accumulation on this device (cuBLAS, large square shapes)."""
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    start = time.perf_counter()
    iterations = 0
    while time.perf_counter() - start < seconds:
        a @ b
        iterations += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return 2.0 * size**3 * iterations / elapsed / 1e12


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
    parser.add_argument(
        "--profile", action="store_true", help="print the trainer's section split (env wait, copy, forward, learn)"
    )
    parser.add_argument("--load-model", help="policy checkpoint (.pt) so the rollout shows trained behavior")
    parser.add_argument(
        "--measure-peak",
        action="store_true",
        help="time large bf16 matmuls (fp32 accumulate, as the model uses) and print the achieved TFLOP/s next to the "
        "table peak the MFU figures divide by",
    )
    parser.add_argument(
        "--kernel-table",
        action="store_true",
        help="after warmup, profile one update and one rollout with torch.profiler and print CUDA kernel time by kernel "
        "(timeline only, needs no GPU performance-counter permission)",
    )
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
    pufferl.profile.frequency = 1 if args.profile else 10**9
    if args.load_model:
        state = torch.load(args.load_model, map_location=config["train"]["device"], weights_only=False)
        state = state.get("policy_state_dict", state)
        pufferl.uncompiled_policy.load_state_dict(P.clean_policy_state_dict(state))

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
    peak_tflops = peak_bf16_tflops(properties.name)
    if args.measure_peak:
        measured = measure_bf16_peak()
        print(f"measured bf16 matmul peak (fp32 accumulate, {8192}^3 cuBLAS): {measured:.0f} TFLOP/s; "
              f"table peak used for MFU: {peak_tflops} TFLOP/s", flush=True)
    agent_steps = train_config["batch_size"]
    print(
        f"{properties.name} capability {capability}, {args.num_envs} envs, {vecenv.num_agents} agents, "
        f"minibatch {pufferl.minibatch_size}, fused_slot_encoder={args.fused}, "
        f"model FLOP per transition: forward {forward_flops / 1e6:.2f} MFLOP, update {train_flops / 1e6:.2f} MFLOP, dense bf16 peak {peak_tflops} TFLOP/s",
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
        if args.kernel_table:
            from torch.profiler import ProfilerActivity, profile

            for label, fn in (("rollout", pufferl.evaluate), ("update", update_or_skip)):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    wall_seconds, _ = timed(fn)
                print_kernel_table(prof, label, wall_seconds if label == "rollout" else wall_seconds)
        profile_baseline = {name: pufferl.profile.profiles[name]["elapsed"] for name in PROFILE_SECTIONS}
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

    if args.profile:
        print("section split per epoch (trainer profiler, CUDA-synchronized at every boundary):")
        for name in PROFILE_SECTIONS:
            seconds = (pufferl.profile.profiles[name]["elapsed"] - profile_baseline[name]) / len(epochs)
            if seconds > 0:
                print(f"  {name:14s} {seconds:6.2f} s")
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
