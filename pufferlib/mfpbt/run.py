from __future__ import annotations

import argparse
import copy

from pufferlib.pufferl import load_config

from .backend_pufferl import PufferLTrainerBackend
from .config import load_mfpbt_config
from .controller import run_mfpbt
from .scheduler import WorkerPoolScheduler


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

    scheduler = WorkerPoolScheduler(
        PufferLTrainerBackend,
        num_devices=mfpbt_config.num_devices,
        num_agents_per_device=mfpbt_config.num_agents_per_device,
        start_method=mfpbt_config.start_method,
        env_name=args.env_name,
        base_args=base_args,
        selection_metric=mfpbt_config.selection_metric,
    )
    run_mfpbt(mfpbt_config, scheduler, num_rounds=mfpbt_config.num_rounds)


if __name__ == "__main__":
    main()
