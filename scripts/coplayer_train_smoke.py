"""~30 s PuffeRL run that trains the SDC against a frozen co-player checkpoint on one map, then checks
that the co-player never changed, the learner did, only learner rows were trained on, metrics counted
only SDCs, and the checkpoint holds just the learner.

    python scripts/coplayer_train_smoke.py --coplayer weights/mimolette/models/model_puffer_drive_003815.pt \
        --map <nuplan map .bin> --learner-reward-conditioning 0

The env's obs setup is taken from the co-player's config.yaml so both policies read the same layout. The
learner is a fresh policy (or --learner <ckpt>, which must share that layout); it may differ from the
co-player only in reward conditioning.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

from pufferlib.ocean.drive.drive import REWARD_COEF_NAMES, reward_coef_to_normalized
from pufferlib.ocean.evaluation_utils.evaluation_utils import resolve_run_dir
from pufferlib.pufferl import (
    COPLAYER_MATCHED_ENV_KEYS,
    PuffeRL,
    apply_coplayer_env_setup,
    load_config,
    load_coplayer_config,
    load_env,
    load_policy,
)

SDC_COLLISION = 4.0
COPLAYER_COLLISION = -1.0
OBS_SEMANTIC_PREFIXES = ("obs_norm_", "obs_range_")


def build_args(cli):
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv
    coplayer_env = load_coplayer_config(cli.coplayer).get("env") or {}
    args["env"].update(
        {
            key: value
            for key, value in coplayer_env.items()
            if key in COPLAYER_MATCHED_ENV_KEYS or key.startswith(OBS_SEMANTIC_PREFIXES)
        }
    )
    learner_reward_conditioning = bool(cli.learner_reward_conditioning)
    if cli.learner:
        with open(os.path.join(resolve_run_dir(cli.learner), "config.yaml")) as config_file:
            learner_config = yaml.safe_load(config_file)
        for section in ("policy", "rnn"):
            args[section].update(learner_config.get(section) or {})
        args["policy_name"] = learner_config["policy_name"]
        args["rnn_name"] = learner_config.get("rnn_name")
        learner_reward_conditioning = bool(learner_config["env"].get("reward_conditioning", False))
        args["load_model_path"] = cli.learner
    args["train"]["use_rnn"] = args["rnn_name"] is not None
    args["env"].update(
        map_dir=cli.map,
        num_maps=1,
        simulation_mode="replay",
        scenario_length=91,
        resample_frequency=0,
        num_agents=256,
        min_agents_per_env=1,
        max_agents_per_env=128,
        init_step=0,
        init_step_spread=False,
        terminate_on_goal=0,
        collision_behavior="ignore",
        offroad_behavior="ignore",
        traffic_light_behavior="ignore",
        goal_reached_behavior="ignore",
        reward_conditioning=learner_reward_conditioning,
        reward_randomization=False,
    )
    apply_coplayer_env_setup(args["env"], cli.coplayer)
    args["env"]["reward_coef_overrides"] = {
        "sdc": {"collision": SDC_COLLISION},
        "others": {"collision": COPLAYER_COLLISION},
    }
    args["vec"].update(backend="Multiprocessing", num_envs=2, num_workers=2, batch_size=2, seed=7)
    args["train"].update(
        device=cli.device,
        compile=False,
        bptt_horizon=64,
        minibatch_size=16384,
        max_minibatch_size=16384,
        total_timesteps=10**12,
        checkpoint_interval=10**9,
        data_dir=cli.data_dir,
        coplayer_policy_path=cli.coplayer,
        coplayer_action_selection=cli.coplayer_action_selection,
    )
    args["wandb"] = False
    return args, learner_reward_conditioning


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--coplayer", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--learner")
    parser.add_argument("--learner-reward-conditioning", type=int, default=0)
    parser.add_argument("--coplayer-action-selection", default="sample")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", default="experiments/coplayer_smoke")
    cli = parser.parse_args()

    args, learner_reward_conditioning = build_args(cli)
    vecenv = load_env("puffer_drive", args)
    learner = load_policy(args, vecenv, "puffer_drive", reward_conditioning=learner_reward_conditioning)
    train_config = dict(**args["train"], env="puffer_drive", eval={}, run_name="coplayer_smoke")
    try:
        trainer = PuffeRL(train_config, vecenv, learner, logger=None)
    except Exception:
        vecenv.close()
        raise
    coplayer_conditioned = trainer.coplayer.obs_columns is None
    print(
        f"learner reward_conditioning={learner_reward_conditioning}, co-player reward_conditioning="
        f"{coplayer_conditioned}, env reward_conditioning={vecenv.driver_env.reward_conditioning}"
    )

    column = vecenv.driver_env.ego_features + REWARD_COEF_NAMES.index("collision")
    split_errors = []
    original_recv = vecenv.recv

    def checked_recv():
        batch = original_recv()
        observations, env_id = batch[0], batch[5]
        learner_rows = np.array(vecenv.learner_masks[env_id[0] : env_id[-1] + 1])
        sdc_ok = np.allclose(observations[learner_rows, column], reward_coef_to_normalized("collision", SDC_COLLISION))
        others = reward_coef_to_normalized("collision", COPLAYER_COLLISION)
        coplayer_ok = np.allclose(observations[~learner_rows, column], others)
        if not (sdc_ok and coplayer_ok):
            split_errors.append(int(env_id[0]))
        return batch

    vecenv.recv = checked_recv
    coplayer_before = {k: v.clone() for k, v in trainer.coplayer.policy.state_dict().items()}
    learner_before = {k: v.clone() for k, v in trainer.uncompiled_policy.state_dict().items()}
    masked_fraction_errors = []
    logged_counts = []
    epochs = 0
    deadline = time.time() + cli.seconds
    try:
        while time.time() < deadline:
            trainer.evaluate()
            logged_counts.extend(trainer.stats.get("n", []))
            trainer.last_log_time = 0.0
            trainer.train()
            epochs += 1
            coplayer_fraction = 1.0 - float(np.mean(vecenv.learner_masks))
            masked_fraction = trainer.losses.get("masked_fraction")
            if masked_fraction is None or abs(masked_fraction - coplayer_fraction) > 1e-6:
                masked_fraction_errors.append((masked_fraction, coplayer_fraction))
        model_path = trainer.save_checkpoint()
    finally:
        vecenv.recv = original_recv
        vecenv.close()
        trainer.utilization.stop()

    saved = torch.load(model_path, map_location="cpu")
    learner_state = {k: v.cpu() for k, v in trainer.uncompiled_policy.state_dict().items()}
    rows_per_worker = vecenv.num_agents // vecenv.num_workers
    sdc_rows_per_worker = np.asarray(vecenv.learner_masks).reshape(-1, rows_per_worker).sum(axis=1)
    checks = {
        "co-player parameters unchanged": all(
            torch.equal(v, coplayer_before[k]) for k, v in trainer.coplayer.policy.state_dict().items()
        ),
        "learner parameters changed": any(
            not torch.equal(v, learner_before[k]) for k, v in trainer.uncompiled_policy.state_dict().items()
        ),
        "learner rows are exactly the SDC rows on every step": not split_errors,
        "masked_fraction == co-player rows / total every epoch": not masked_fraction_errors,
        "metrics logged and counted only SDCs": bool(logged_counts) and max(logged_counts) <= max(sdc_rows_per_worker),
        "checkpoint holds exactly the learner": saved.keys() == learner_state.keys()
        and all(torch.equal(saved[k].cpu(), learner_state[k]) for k in saved),
    }
    print(
        f"{epochs} epochs in {cli.seconds:.0f}s; co-player rows {1.0 - float(np.mean(vecenv.learner_masks)):.3f} "
        f"of {vecenv.num_agents}; SDC rows per worker {sdc_rows_per_worker.tolist()}; "
        f"logged n per worker emission {sorted(set(logged_counts))}"
    )
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
