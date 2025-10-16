# tests/test_simulator_perf.py
import time
import numpy as np
import warnings
import argparse
from pufferlib.ocean.drive.drive import Drive


def test_simulator_raw(expected_sps, warning_threshold):
    """
    Measure raw simulator performance and warn if it regresses.

    Args:
        expected_sps (float): Expected baseline steps per second (SPS).
        warning_threshold (float): Fraction of expected SPS below which a warning is issued.
    """
    timeout = 5  # seconds (short for CI)
    atn_cache = 16  # batched action cache
    num_agents = 32

    env = Drive(num_agents=num_agents, num_maps=1)
    obs, _ = env.reset()
    tick = 0

    # Pre-generate random batched actions
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

    # Calculate performance
    total_time = time.time() - start_time
    sps = num_agents * tick / total_time
    print(f"Steps per second (SPS): {sps:.1f}")
    print(f"Step time mean: {np.mean(step_times):.6f}s, std: {np.std(step_times):.6f}s")

    # Compare to expected baseline
    performance_pct = sps / expected_sps
    if performance_pct < warning_threshold:
        warnings.warn(
            f"⚠️ Simulator performance is {performance_pct * 100:.1f}% of expected baseline "
            f"({expected_sps} SPS). Threshold for warning: {warning_threshold * 100:.0f}%."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test simulator performance")
    parser.add_argument("--expected_sps", type=float, default=100000, help="Expected baseline steps per second (SPS)")
    parser.add_argument(
        "--warning_threshold", type=float, default=0.9, help="Fraction of expected SPS below which a warning is issued"
    )

    args = parser.parse_args()
    test_simulator_raw(expected_sps=args.expected_sps, warning_threshold=args.warning_threshold)
