import argparse
import time

import numpy as np

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive


CONFIGS = [
    {
        "name": "corridor_idm",
        "controller": "corridor_idm",
        "pdm_horizon": 4.0,
        "pdm_planning_dt": 0.5,
    },
    {
        "name": "idm",
        "controller": "idm",
        "pdm_horizon": 4.0,
        "pdm_planning_dt": 0.5,
    },
    {
        "name": "pdm_fast_4s_0.5",
        "controller": "pdm",
        "pdm_horizon": 4.0,
        "pdm_planning_dt": 0.5,
    },
    {
        "name": "pdm_killer_8s_0.1",
        "controller": "pdm",
        "pdm_horizon": 8.0,
        "pdm_planning_dt": 0.1,
    },
]


def extract_log(env):
    payload = binding.vec_log(env.c_envs, env.num_agents)
    summaries = env._normalize_log_summaries(payload)
    if not summaries:
        return {}
    return summaries[0]


def run_config(config, args):
    env = Drive(
        report_interval=args.steps + 1,
        num_agents=args.agents,
        min_agents_per_env=args.agents,
        max_agents_per_env=args.agents,
        num_maps=1,
        maps=args.map,
        map_dir=args.map_dir,
        simulation_mode="gigaflow",
        sdc_controller=config["controller"],
        non_sdc_controller=config["controller"],
        control_mode="control_vehicles",
        scenario_length=args.steps,
        resample_frequency=0,
        termination_mode=0,
        pdm_horizon=config["pdm_horizon"],
        pdm_planning_dt=config["pdm_planning_dt"],
        compute_eval_metrics=True,
        seed=args.seed,
    )

    try:
        env.reset()
        actions = np.zeros(env.actions.shape, dtype=env.actions.dtype)
        start = time.perf_counter()
        for _ in range(args.steps):
            env.step(actions)
        elapsed = time.perf_counter() - start
        log = extract_log(env)
    finally:
        env.close()

    agent_steps = args.agents * args.steps
    return {
        "name": config["name"],
        "controller": config["controller"],
        "pdm_horizon": config["pdm_horizon"],
        "pdm_planning_dt": config["pdm_planning_dt"],
        "elapsed_s": elapsed,
        "sps": agent_steps / elapsed if elapsed > 0 else float("inf"),
        "collision_rate": float(log.get("collision_rate", float("nan"))),
        "offroad_rate": float(log.get("offroad_rate", float("nan"))),
        "red_light_violation_rate": float(log.get("red_light_violation_rate", float("nan"))),
        "comfort_violation_count": float(log.get("comfort_violation_count", float("nan"))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--agents", type=int, default=64)
    parser.add_argument("--map", default="Town02")
    parser.add_argument("--map-dir", default="pufferlib/resources/drive/binaries/carla")
    parser.add_argument("--seed", type=int, default=2904)
    args = parser.parse_args()

    print(
        f"{'config':<22} {'ctrl':<12} {'horizon':>7} {'dt':>5} "
        f"{'elapsed_s':>10} {'sps':>12} {'collision':>10} {'offroad':>10} "
        f"{'red_light':>10} {'comfort':>10}"
    )
    for config in CONFIGS:
        row = run_config(config, args)
        print(
            f"{row['name']:<22} {row['controller']:<12} {row['pdm_horizon']:>7.1f} "
            f"{row['pdm_planning_dt']:>5.1f} {row['elapsed_s']:>10.3f} "
            f"{row['sps']:>12.0f} {row['collision_rate']:>10.4f} {row['offroad_rate']:>10.4f} "
            f"{row['red_light_violation_rate']:>10.4f} {row['comfort_violation_count']:>10.4f}"
        )


if __name__ == "__main__":
    main()
