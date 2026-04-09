from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime
import os

import yaml

from pufferlib.pufferl import load_config

from .backend_pufferl import PufferLTrainerBackend
from .config import load_mfpbt_config
from .controller import run_mfpbt
from .scheduler import WorkerPoolScheduler


def _prepare_run_directory(env_name: str, config_path: str, mfpbt_config):
    run_name = mfpbt_config.run_name or f"{env_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(mfpbt_config.experiment_root, run_name)
    os.makedirs(run_dir, exist_ok=False)

    checkpoint_name = os.path.basename(mfpbt_config.checkpoint_path or "checkpoint.pt")
    mfpbt_config.checkpoint_path = os.path.join(run_dir, checkpoint_name)

    if mfpbt_config.log_dir is None:
        mfpbt_config.log_dir = os.path.join(run_dir, "logs")
    else:
        log_dir_name = os.path.basename(os.path.normpath(mfpbt_config.log_dir))
        mfpbt_config.log_dir = os.path.join(run_dir, log_dir_name)

    resolved_config_path = os.path.join(run_dir, "mfpbt_config_resolved.yaml")
    with open(resolved_config_path, "w") as handle:
        yaml.safe_dump(asdict(mfpbt_config), handle, sort_keys=False)

    original_config_copy = os.path.join(run_dir, "mfpbt_config_input.yaml")
    with open(config_path, "r") as src, open(original_config_copy, "w") as dst:
        dst.write(src.read())

    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str)
    parser.add_argument("--config", type=str, required=True, help="Path to MF-PBT yaml config")
    parser.add_argument("--rounds", type=int, default=None, help="Optional override for yaml num_rounds")
    parser.add_argument("--num-devices", type=int, default=None, help="Optional override for yaml num_devices")
    parser.add_argument(
        "--num-agents-per-device",
        type=int,
        default=None,
        help="Optional override for yaml num_agents_per_device",
    )
    parser.add_argument("--num-agents", type=int, default=None, help="Optional override for yaml num_agents")
    parser.add_argument(
        "--frequencies",
        type=int,
        nargs="+",
        default=None,
        help="Optional override for yaml frequencies",
    )
    args = parser.parse_args()

    mfpbt_config = load_mfpbt_config(args.config)
    if args.rounds is not None:
        mfpbt_config.num_rounds = args.rounds
    if args.num_devices is not None:
        mfpbt_config.num_devices = args.num_devices
    if args.num_agents_per_device is not None:
        mfpbt_config.num_agents_per_device = args.num_agents_per_device
    if args.num_agents is not None:
        mfpbt_config.num_agents = args.num_agents
    if args.frequencies is not None:
        mfpbt_config.frequencies = args.frequencies
    if any(
        value is not None
        for value in (
            args.rounds,
            args.num_devices,
            args.num_agents_per_device,
            args.num_agents,
            args.frequencies,
        )
    ):
        mfpbt_config.validate()
    base_args = load_config(args.env_name, argv=[])
    base_args = copy.deepcopy(base_args)
    base_args["wandb"] = False
    base_args["neptune"] = False
    base_args["tb"] = False

    run_dir = _prepare_run_directory(args.env_name, args.config, mfpbt_config)
    print(f"MF-PBT run directory: {run_dir}")

    scheduler = WorkerPoolScheduler(
        PufferLTrainerBackend,
        num_devices=mfpbt_config.num_devices,
        num_agents_per_device=mfpbt_config.num_agents_per_device,
        start_method=mfpbt_config.start_method,
        env_name=args.env_name,
        base_args=base_args,
        selection_metric=mfpbt_config.selection_metric,
        selection_source=mfpbt_config.selection_source,
        eval_simulation_mode=mfpbt_config.eval_simulation_mode,
        eval_map_dir=mfpbt_config.eval_map_dir,
        eval_num_scenarios=mfpbt_config.eval_num_scenarios,
        eval_num_agents=mfpbt_config.eval_num_agents,
        eval_num_carla_maps=mfpbt_config.eval_num_carla_maps,
        trainer_state_dir=os.path.join(run_dir, "trainer_states"),
    )
    run_mfpbt(mfpbt_config, scheduler, num_rounds=mfpbt_config.num_rounds)


if __name__ == "__main__":
    main()
