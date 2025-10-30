import time
import json
import numpy as np
import warnings
from pathlib import Path
from pufferlib.ocean.drive.drive import Drive

# CONFIGURATION


def test_simulator_raw():
    timeout = 5  # seconds (short for CI)
    atn_cache = 16  # batched action cache
    num_agents = 32

    # ---- Run simulation ----
    env = Drive(num_agents=num_agents, num_maps=1, scenario_length=91)
    obs, _ = env.reset()
    tick = 0

    actions = np.stack(
        [np.random.randint(0, space.n, (atn_cache, num_agents)) for space in env.single_action_space], axis=-1
    )

    step_times = []
    start_time = time.time()

    while time.time() - start_time < timeout:
        atn = actions[tick % atn_cache]
        t0 = time.time()
        env.step(atn)
        step_times.append(time.time() - t0)
        tick += 1

    env.close()

    total_time = time.time() - start_time
    sps = num_agents * tick / total_time
    print(f"Steps per second (SPS): {sps:.1f}")
    print(f"Step time mean: {np.mean(step_times):.6f}s, std: {np.std(step_times):.6f}s")

    mean = 24690  # hardcoded baseline mean SPS - via multiple tests
    threshold = 0.8 * mean
    if sps < threshold:
        raise RuntimeError(
            f"\033[91m❌ Performance regression detected: {sps:.1f} < {threshold:.1f}.\033[0m\n"
            f"Expected mean: {mean:.1f}, got {sps:.1f}"
        )
    return


if __name__ == "__main__":
    test_simulator_raw()
