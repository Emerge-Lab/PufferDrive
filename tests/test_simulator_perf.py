import time
import json
import numpy as np
import warnings
from pathlib import Path
from pufferlib.ocean.drive.drive import Drive

# CONFIGURATION

# Path where the history JSON will be stored
# (relative to project root, regardless of where the test runs from)
ROOT_DIR = Path(__file__).resolve().parents[2]  # goes up from tests/ to PufferDrive/
DATA_FILE = ROOT_DIR / "pufferlib" / "resources" / "drive" / "simulator_perf_history.json"


def load_history():
    if DATA_FILE.exists():
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"sps_values": []}


def save_history(history):
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(history, f, indent=4)


def compute_baseline(history):
    """Compute mean and std from stored SPS values."""
    sps_values = np.array(history["sps_values"], dtype=float)
    return np.mean(sps_values), np.std(sps_values)


def test_simulator_raw():
    timeout = 5  # seconds (short for CI)
    atn_cache = 16  # batched action cache
    num_agents = 32

    # ---- Run simulation ----
    env = Drive(num_agents=num_agents, num_maps=1)
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
        warnings.warn(
            f"\033[91m⚠️ Performance regression detected: {sps:.1f} < {threshold:.1f} .\033[0m\nCurrent mean: {mean:.1f}"
        )
    return


if __name__ == "__main__":
    test_simulator_raw()
