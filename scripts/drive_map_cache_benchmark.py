#!/usr/bin/env python
import argparse
import gc
import json
import time

import numpy as np
import psutil

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive


def rss_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)


def parse_ints(value):
    return [int(part) for part in str(value).split(",") if part]


def benchmark_config(args, num_agents, agents_per_env, cache_enabled):
    gc.collect()
    rss_before = rss_mb()
    start = time.perf_counter()
    env = Drive(
        num_agents=num_agents,
        min_agents_per_env=agents_per_env,
        max_agents_per_env=agents_per_env,
        num_maps=args.num_maps,
        maps=args.maps,
        map_dir=args.map_dir,
        simulation_mode="gigaflow",
        scenario_length=args.scenario_length,
        resample_frequency=0,
        render_mode=None,
        enable_map_cache=cache_enabled,
    )
    init_s = time.perf_counter() - start
    rss_after_init = rss_mb()

    stats = None
    if getattr(env, "_map_cache", None) is not None:
        stats = binding.map_cache_stats(env._map_cache)

    start = time.perf_counter()
    obs, _ = env.reset(seed=args.seed)
    reset_s = time.perf_counter() - start
    finite = bool(np.isfinite(obs).all())

    start = time.perf_counter()
    for _ in range(args.steps):
        actions = np.zeros_like(env.actions)
        obs, rewards, _, _, _ = env.step(actions)
        finite = finite and bool(np.isfinite(obs).all()) and bool(np.isfinite(rewards).all())
    step_s = time.perf_counter() - start
    rss_after_steps = rss_mb()

    result = {
        "cache": cache_enabled,
        "num_agents": num_agents,
        "agents_per_env": agents_per_env,
        "num_envs": env.num_envs,
        "num_maps": args.num_maps,
        "cache_count": stats["count"] if stats else 0,
        "cache_hits": stats["cache_hits"] if stats else 0,
        "cache_misses": stats["cache_misses"] if stats else 0,
        "init_s": init_s,
        "reset_s": reset_s,
        "step_s": step_s,
        "steps": args.steps,
        "agent_steps_per_s": (num_agents * args.steps / step_s) if step_s > 0 else None,
        "rss_before_mb": rss_before,
        "rss_after_init_mb": rss_after_init,
        "rss_after_steps_mb": rss_after_steps,
        "rss_init_delta_mb": rss_after_init - rss_before,
        "finite": finite,
    }

    env.close()
    del env
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-dir", default="pufferlib/resources/drive/binaries/carla")
    parser.add_argument("--maps", default=2)
    parser.add_argument("--num-maps", type=int, default=1)
    parser.add_argument("--num-agents", default="64,128,256")
    parser.add_argument("--agents-per-env", default="4,8,16,64")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--scenario-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cache", choices=["on", "off", "both"], default="both")
    parser.add_argument("--max-rss-mb", type=float, default=0.0)
    args = parser.parse_args()

    cache_modes = [True, False] if args.cache == "both" else [args.cache == "on"]
    for num_agents in parse_ints(args.num_agents):
        for agents_per_env in parse_ints(args.agents_per_env):
            if agents_per_env > num_agents:
                continue
            for cache_enabled in cache_modes:
                try:
                    result = benchmark_config(args, num_agents, agents_per_env, cache_enabled)
                    print(json.dumps(result, sort_keys=True), flush=True)
                    if args.max_rss_mb > 0 and result["rss_after_steps_mb"] > args.max_rss_mb:
                        print(
                            json.dumps(
                                {
                                    "stopped": True,
                                    "reason": "max_rss_mb",
                                    "max_rss_mb": args.max_rss_mb,
                                    "rss_after_steps_mb": result["rss_after_steps_mb"],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        return
                except Exception as err:
                    print(
                        json.dumps(
                            {
                                "cache": cache_enabled,
                                "num_agents": num_agents,
                                "agents_per_env": agents_per_env,
                                "error": repr(err),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
