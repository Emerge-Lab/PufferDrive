"""Run one PPO update minibatch and one rollout forward of the real Drive policy inside NVTX ranges for Nsight Compute.

Usage:
    ncu --nvtx --nvtx-include "update/" --nvtx-include "rollout/" --nvtx-include "calibrate/" \\
        --metrics gpu__time_duration.sum,sm__ops_path_tensor_src_bf16_dst_fp32.sum,sm__ops_path_tensor_src_fp16_dst_fp32.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes.sum \\
        --csv --page raw python scripts/profile_policy_step.py --rows 65536 [--fused] > ncu_out.csv
    python scripts/profile_policy_step.py --parse ncu_out.csv --rows 65536
The calibrate range runs a matmul of known size so the tensor-op metric is converted to FLOPs without guessing.
"""

import argparse
import copy
import csv
import json
import os
import sys


os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch


TRAIN_FLOP_PER_TRANSITION = 112145920.0
ROLLOUT_FLOP_PER_TRANSITION = 37654113.0
CALIBRATE_N = 8192
CALIBRATE_FLOPS = 2.0 * CALIBRATE_N**3
PEAK_BF16_TFLOPS = 209.5
# dense bf16 tensor peak with fp32 accumulation, by compute capability
PEAK_BF16_TFLOPS_BY_CAPABILITY = {(12, 0): 209.5, (9, 0): 989.0, (10, 0): 2250.0, (10, 3): 2250.0}
ROLLOUT_ROWS = 16384
WARMUP_STEPS = 4
TIMED_STEPS = 5
ENV_AGENTS = 512
ROLLOUT_STEPS = 40
TRAIN_OVERRIDES = [
    "vec.backend=Serial",
    "vec.num_envs=1",
    "vec.num_workers=1",
    "vec.batch_size=1",
    f"env.num_agents={ENV_AGENTS}",
    "train.minibatch_size=8192",
    "train.max_minibatch_size=8192",
    "wandb=False",
    "tb=False",
    "neptune=False",
]


def build_policy_and_obs(rows, fused):
    sys.argv = ["profile_policy_step"] + TRAIN_OVERRIDES
    import pufferlib.pufferl as P

    args = P.load_config("puffer_drive")
    args = P.normalize_puffer_drive_config(args, "training")
    P.validate_puffer_drive_config(args, "training")
    args = copy.deepcopy(args)
    args["policy"]["fused_slot_encoder"] = fused
    vecenv = P.load_env("puffer_drive", args, seed=3)
    torch.manual_seed(0)
    policy = P.load_policy(args, vecenv, "puffer_drive")
    vecenv.async_reset(3)
    obs_rows = []
    for _ in range(ROLLOUT_STEPS):
        o, *_ = vecenv.recv()
        obs_rows.append(np.array(o, copy=True))
        vecenv.send(np.random.uniform(-1, 1, size=vecenv.action_space.shape).astype(vecenv.action_space.dtype))
    vecenv.close()
    obs = torch.as_tensor(np.concatenate(obs_rows)).cuda()
    repeats = -(-max(rows, ROLLOUT_ROWS) // obs.shape[0])
    return policy, obs.repeat(repeats, 1)


def fse_config_name():
    import pufferlib.ocean.fused_slot_encoder as fse

    if fse.CONFIG_OVERRIDE is not None:
        return "override"
    return "table"


def apply_kernel_config(path):
    import pufferlib.ocean.fused_slot_encoder as fse

    with open(path) as handle:
        fse.CONFIG_OVERRIDE = fse.KernelConfig(**json.load(handle))
    print(f"kernel config override: {fse.CONFIG_OVERRIDE}")


def run_phases(rows, fused):
    import pufferlib.pytorch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    policy, obs = build_policy_and_obs(rows, fused)
    compiled = torch.compile(policy)
    compiled_eval = torch.compile(policy.forward_eval)
    sample_logits = torch.compile(pufferlib.pytorch.sample_logits)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=0.01)
    update_obs = obs[:rows]
    actions = torch.randint(0, policy.action_dim, (rows,), device="cuda", dtype=torch.int32)
    rollout_obs = obs[:ROLLOUT_ROWS]
    amp = torch.autocast("cuda", dtype=torch.bfloat16)

    def update_step():
        optimizer.zero_grad(set_to_none=True)
        with amp:
            logits, value = compiled(update_obs, {"action": actions, "lstm_h": None, "lstm_c": None})
        logits = logits.float()
        _, logprob, entropy, _ = sample_logits(logits, action=actions)
        loss = -logprob.float().mean() + 0.5 * value.float().square().mean() - 0.01 * entropy.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

    def rollout_step():
        with torch.no_grad(), amp:
            logits, _ = compiled_eval(rollout_obs, {})
            sample_logits(logits.float(), env_continuous=True, policy=policy)

    for _ in range(WARMUP_STEPS):
        update_step()
        rollout_step()
    wall_ms = {}
    for name, step in (("update", update_step), ("rollout", rollout_step)):
        samples = []
        for _ in range(TIMED_STEPS):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        wall_ms[name] = sorted(samples)[len(samples) // 2]
    print(
        f"unprofiled wall time (median of {TIMED_STEPS}): update {wall_ms['update']:.2f} ms, rollout {wall_ms['rollout']:.2f} ms"
    )
    peak = PEAK_BF16_TFLOPS_BY_CAPABILITY.get(torch.cuda.get_device_capability(0))
    update_rate = rows * TRAIN_FLOP_PER_TRANSITION / (wall_ms["update"] * 1e-3) / 1e12
    rollout_rate = ROLLOUT_ROWS * ROLLOUT_FLOP_PER_TRANSITION / (wall_ms["rollout"] * 1e-3) / 1e12
    mfu = f" update_mfu={100 * update_rate / peak:.1f}% rollout_mfu={100 * rollout_rate / peak:.1f}%" if peak else ""
    print(
        f"RESULT fused={int(fused)} rows={rows} update_ms={wall_ms['update']:.2f} rollout_ms={wall_ms['rollout']:.2f}"
        f" update_tflops={update_rate:.1f} rollout_tflops={rollout_rate:.1f}{mfu} config={fse_config_name()}"
    )
    a = torch.randn(CALIBRATE_N, CALIBRATE_N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a @ b
    torch.cuda.synchronize()
    # Start/end ranges are process-wide and also cover the autograd thread's kernels; push/pop ranges are per thread.
    for name, step in (("calibrate", lambda: a @ b), ("update", update_step), ("rollout", rollout_step)):
        range_id = torch.cuda.nvtx.range_start(name)
        step()
        torch.cuda.nvtx.range_end(range_id)
        torch.cuda.synchronize()
    print(
        f"analytic model FLOPs: update {rows * TRAIN_FLOP_PER_TRANSITION:.4e}, rollout {ROLLOUT_ROWS * ROLLOUT_FLOP_PER_TRANSITION:.4e}, calibrate {CALIBRATE_FLOPS:.4e}"
    )


METRIC_COLUMNS = {
    "time": "gpu__time_duration.sum",
    "bf16": "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32.sum",
    "fp16": "sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32.sum",
    "tf32": "sm__ops_path_tensor_op_hmma_src_tf32_dst_fp32.sum",
    "ffma": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "fadd": "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "fmul": "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "dram": "dram__bytes.sum",
    "local_ld": "sm__sass_inst_executed_op_local_ld.sum",
    "local_st": "sm__sass_inst_executed_op_local_st.sum",
}
HEAD_FLOP_PER_TRANSITION = 3166208.0
ROLLOUT_HEAD_FLOP_PER_TRANSITION = 1055232.0


RANGE_NAMES = ("calibrate", "update", "rollout")


def _phase_of(record, kernel_col):
    kernel = record[kernel_col]
    if "/" in kernel:
        return kernel.split("/")[0]
    text = " ".join(record)
    for name in RANGE_NAMES:
        if f":{name}:" in text:
            return name
    return "unlabeled"


def parse(path, rows):
    with open(path) as handle:
        table = list(csv.reader(line for line in handle if line.startswith('"')))
    header, data_rows = table[0], table[2:]
    kernel_col = header.index("Kernel Name")
    columns = {key: header.index(name) for key, name in METRIC_COLUMNS.items() if name in header}
    sums = {}
    for record in data_rows:
        phase = _phase_of(record, kernel_col)
        phase_sums = sums.setdefault(phase, {key: 0.0 for key in columns})
        phase_sums["kernels"] = phase_sums.get("kernels", 0) + 1
        for key, col in columns.items():
            phase_sums[key] += float(record[col].replace(",", "") or 0.0)
    calibrate = sums.get("calibrate")
    if calibrate:
        print(f"calibration: counter {calibrate['bf16']:.6e} FLOP vs exact {CALIBRATE_FLOPS:.6e} for the 8192^3 matmul")
    model_tensor = {
        "update": rows * (TRAIN_FLOP_PER_TRANSITION - HEAD_FLOP_PER_TRANSITION),
        "rollout": ROLLOUT_ROWS * (ROLLOUT_FLOP_PER_TRANSITION - ROLLOUT_HEAD_FLOP_PER_TRANSITION),
        "calibrate": CALIBRATE_FLOPS,
    }
    model_fp32 = {"update": rows * HEAD_FLOP_PER_TRANSITION, "rollout": ROLLOUT_ROWS * ROLLOUT_HEAD_FLOP_PER_TRANSITION}
    peak = PEAK_BF16_TFLOPS * 1e12
    for phase, m in sums.items():
        seconds = m["time"] * 1e-9
        tensor = m["bf16"] + m.get("fp16", 0.0) + m.get("tf32", 0.0)
        fp32 = 2.0 * m.get("ffma", 0.0) + m.get("fadd", 0.0) + m.get("fmul", 0.0)
        print(f"== {phase}: {m['kernels']} kernels, summed kernel time {seconds * 1e3:.2f} ms")
        print(
            f"   tensor FLOP counted {tensor:.4e} (bf16 {m['bf16']:.3e}, tf32 {m.get('tf32', 0.0):.2e}) vs model GEMM FLOP {model_tensor.get(phase, 0.0):.4e}"
        )
        print(f"   fp32 CUDA-core FLOP counted {fp32:.4e} vs model fp32 head FLOP {model_fp32.get(phase, 0.0):.3e}")
        print(
            f"   tensor-FLOP rate over kernel time {tensor / seconds / 1e12:.1f} TFLOP/s = {100 * tensor / seconds / peak:.1f}% of bf16 peak; model-FLOP rate {model_tensor.get(phase, 0.0) / seconds / 1e12:.1f} TFLOP/s = {100 * model_tensor.get(phase, 0.0) / seconds / peak:.1f}%"
        )
        print(
            f"   DRAM {m['dram'] / 1e9:.2f} GB -> {m['dram'] / seconds / 1e9:.0f} GB/s; local (spill) ld/st instructions {m.get('local_ld', 0.0):.3e} / {m.get('local_st', 0.0):.3e}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=65536, help="update minibatch rows")
    parser.add_argument("--fused", action="store_true", help="enable policy.fused_slot_encoder")
    parser.add_argument("--parse", help="ncu raw-page CSV to summarize instead of running")
    parser.add_argument("--kernel-config", help="JSON KernelConfig (from tune_fused_slot_encoder.py --write-config)")
    args = parser.parse_args()
    if args.parse:
        parse(args.parse, args.rows)
        return
    if args.kernel_config:
        apply_kernel_config(args.kernel_config)
    run_phases(args.rows, args.fused)


if __name__ == "__main__":
    main()
