"""Shard one PufferDrive eval benchmark across GPUs or nodes and aggregate the results.

Each shard runs an independent `puffer eval` process over a disjoint scenario
window (eval.scenario_offset), then the per-shard episode_metrics.csv files are
merged into a single report and optionally attached to the training wandb run.
--num-gpus runs one shard per local GPU; --num-nodes runs one shard per node of
the surrounding SLURM allocation (each claiming a full node's CPUs and one GPU
via srun), so every shard gets its own node's cores.

Usage (from the repo root, venv active):
    python scripts/parallel_eval.py carla \
        --total-scenarios 40000 --num-gpus 8 \
        load_model_path=experiments/run/final_model.pt \
        eval.output_name=run vec.num_envs=16 wandb=True
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

RESERVED_OVERRIDE_KEYS = ("num_scenarios", "eval.num_scenarios", "eval.scenario_offset", "eval.output_subdir")
FAILED_LOG_TAIL_LINES = 30


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", help="Benchmark name from the benchmark config, e.g. carla")
    parser.add_argument("--env-name", default="puffer_drive")
    parser.add_argument("--total-scenarios", type=int, required=True)
    shard_group = parser.add_mutually_exclusive_group(required=True)
    shard_group.add_argument("--num-gpus", type=int, help="Run one shard per local GPU")
    shard_group.add_argument("--num-nodes", type=int, help="Run one shard per SLURM node via srun, 1 GPU each")
    arguments, overrides = parser.parse_known_args()
    for override in overrides:
        if override.startswith("--"):
            parser.error(f"Unknown flag {override}; Hydra overrides use key=value syntax")
        if "=" not in override:
            parser.error(f"Override '{override}' must use key=value syntax")
    return arguments, overrides


def override_value(overrides, key):
    value = None
    for override in overrides:
        override_key, _, override_val = override.partition("=")
        if override_key == key:
            value = override_val
    return value


def resolve_gpu_ids(num_gpus):
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return [str(gpu_idx) for gpu_idx in range(num_gpus)]
    gpu_ids = [device.strip() for device in visible_devices.split(",") if device.strip()]
    if len(gpu_ids) < num_gpus:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES exposes {len(gpu_ids)} GPUs but --num-gpus is {num_gpus}")
    return gpu_ids[:num_gpus]


def plan_shards(total_scenarios, num_shards):
    scenarios_per_shard, remainder = divmod(total_scenarios, num_shards)
    shards = []
    scenario_offset = 0
    for shard_idx in range(num_shards):
        shard_scenario_count = scenarios_per_shard + (1 if shard_idx < remainder else 0)
        shards.append((scenario_offset, shard_scenario_count))
        scenario_offset += shard_scenario_count
    return shards


def print_log_tail(log_path):
    with open(log_path, "r", errors="replace") as log_file:
        tail = log_file.readlines()[-FAILED_LOG_TAIL_LINES:]
    print(f"--- last {len(tail)} lines of {log_path} ---")
    print("".join(tail))


def main():
    arguments, overrides = parse_arguments()
    use_srun = arguments.num_nodes is not None
    num_shards = arguments.num_nodes if use_srun else arguments.num_gpus
    if num_shards < 1:
        raise SystemExit("--num-gpus/--num-nodes must be at least 1")
    if arguments.total_scenarios < num_shards:
        raise SystemExit("--total-scenarios must be at least the shard count")
    if use_srun:
        if "SLURM_JOB_ID" not in os.environ:
            raise SystemExit("--num-nodes requires running inside a SLURM allocation")
        cpus_per_shard = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpus_per_shard is None:
            raise SystemExit("--num-nodes requires SLURM_CPUS_PER_TASK (set #SBATCH --cpus-per-task)")
    for override in overrides:
        if override.partition("=")[0] in RESERVED_OVERRIDE_KEYS:
            raise SystemExit(f"Override '{override}' is managed by parallel_eval.py; remove it")

    model_path = override_value(overrides, "load_model_path")
    if model_path is None:
        raise SystemExit("load_model_path=<checkpoint.pt> override is required")

    report_wandb = str(override_value(overrides, "wandb")).lower() in ("true", "1")
    shard_overrides = [o for o in overrides if o.partition("=")[0] != "wandb"] + ["wandb=False"]

    from pufferlib.ocean.evaluation_utils import evaluation_utils

    run_dir = evaluation_utils.resolve_run_dir(model_path)
    output_dir_name = override_value(overrides, "eval.output_dir_name") or "eval"
    output_name = override_value(overrides, "eval.output_name")
    benchmark_dir_name = arguments.benchmark if output_name is None else f"{arguments.benchmark}_{output_name}"
    benchmark_dir = os.path.join(run_dir, output_dir_name, benchmark_dir_name)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    aggregate_dir = os.path.join(benchmark_dir, stamp)
    os.makedirs(aggregate_dir)

    gpu_ids = None if use_srun else resolve_gpu_ids(num_shards)
    shards = plan_shards(arguments.total_scenarios, num_shards)
    processes = []
    for shard_idx, (scenario_offset, shard_scenario_count) in enumerate(shards):
        shard_subdir = f"{stamp}_shard{shard_idx:02d}"
        command = [
            sys.executable,
            "-m",
            "pufferlib.pufferl",
            "eval",
            arguments.env_name,
            arguments.benchmark,
            *shard_overrides,
            f"num_scenarios={shard_scenario_count}",
            f"eval.scenario_offset={scenario_offset}",
            f"eval.output_subdir={shard_subdir}",
        ]
        if use_srun:
            # Full-node CPU request forces SLURM to place each shard step on its own node.
            command = [
                "srun",
                "--nodes=1",
                "--ntasks=1",
                f"--cpus-per-task={cpus_per_shard}",
                "--gres=gpu:1",
                *command,
            ]
            shard_env = dict(os.environ)
            placement = "own node via srun"
        else:
            shard_env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_ids[shard_idx])
            placement = f"GPU {gpu_ids[shard_idx]}"
        log_path = os.path.join(aggregate_dir, f"shard{shard_idx:02d}.log")
        log_file = open(log_path, "w")
        process = subprocess.Popen(command, env=shard_env, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((shard_idx, process, log_file, log_path, shard_subdir))
        print(
            f"Shard {shard_idx}: {placement}, scenarios [{scenario_offset}, "
            f"{scenario_offset + shard_scenario_count}), log {log_path}"
        )

    failed_shards = []
    for shard_idx, process, log_file, log_path, _ in processes:
        exit_code = process.wait()
        log_file.close()
        status = "done" if exit_code == 0 else f"FAILED (exit {exit_code})"
        print(f"Shard {shard_idx}: {status}")
        if exit_code != 0:
            failed_shards.append((shard_idx, log_path))
    if failed_shards:
        for _, log_path in failed_shards:
            print_log_tail(log_path)
        raise SystemExit(f"{len(failed_shards)} of {len(processes)} eval shards failed; not aggregating")

    shard_frames = []
    for shard_idx, _, _, _, shard_subdir in processes:
        csv_path = os.path.join(benchmark_dir, shard_subdir, "episode_metrics.csv")
        if not os.path.isfile(csv_path):
            raise SystemExit(f"Shard {shard_idx} finished but wrote no {csv_path}")
        shard_frame = pd.read_csv(csv_path)
        if shard_frame.empty:
            raise SystemExit(f"Shard {shard_idx} recorded zero episodes in {csv_path}")
        shard_frames.append(shard_frame)

    episode_summaries = pd.concat(shard_frames, ignore_index=True).to_dict("records")
    summary = evaluation_utils._write_eval_reports(episode_summaries, aggregate_dir, arguments.total_scenarios)
    print(f"Aggregated {summary['num_episodes']} episodes from {len(processes)} shards into {aggregate_dir}")

    if report_wandb:
        import pufferlib.pufferl as pufferl

        sys.argv = [sys.argv[0]] + [o for o in overrides if o.partition("=")[0] != "wandb"]
        config_args = pufferl.load_config(arguments.env_name)
        config_args["wandb"] = True
        checkpoint_config_path = os.path.join(run_dir, "config.yaml")
        wandb_run_identity = evaluation_utils.load_checkpoint_run_identity(checkpoint_config_path)
        pufferl.report_eval_to_wandb(
            config_args,
            {arguments.benchmark: {"summary": summary}},
            wandb_run_identity,
            output_dir_name,
        )


if __name__ == "__main__":
    main()
