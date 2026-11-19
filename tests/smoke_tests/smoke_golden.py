"""Config, infraction check and golden record/compare shared by the deterministic CPU smoke tests."""

import json
import os

import numpy as np

from opengigaflow.rl import load_config


SEED = 42
BPTT_HORIZON = 64  # env steps per epoch; == scenario length so episodes complete every epoch
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "data")
# qemu-user runs FP on the host CPU: float sums and EMAs keep ~5% host residue, the other metrics are bit-stable
RTOL = float(os.environ.get("SMOKE_RTOL", "2e-2"))
LOOSE_RTOL = float(os.environ.get("SMOKE_LOOSE_RTOL", "8e-2"))
ATOL = float(os.environ.get("SMOKE_ATOL", "1e-3"))
LOOSE_ATOL = float(os.environ.get("SMOKE_LOOSE_ATOL", "5e-3"))
LOOSE_KEYS = frozenset({"value_loss", "ema_max", "episode_return"})
INFRACTION_KEYS = ("collision_rate", "offroad_rate")


def build_config(num_envs):
    args = load_config("opengigaflow", overrides=[])
    # Serial, as Multiprocessing's recv order is host-dependent; it steps batch_size (auto: num_envs // 2) envs
    args["vec"].update({"backend": "Serial", "num_envs": num_envs, "seed": SEED})
    args["env"].update(
        {
            "num_agents": 16,
            "min_agents_per_env": 16,
            "max_agents_per_env": 16,
            "num_maps": 2,
            "use_map_cache": True,
            "map_dir": "opengigaflow/resources/carla",
            "scenario_length": BPTT_HORIZON,
            "stagger_first_episode": False,  # the goldens expect full-length first episodes
        }
    )
    return args


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def assert_infractions_occur(env_means):
    present = [key for key in INFRACTION_KEYS if key in env_means]
    if not present:
        raise AssertionError(f"no infraction metrics captured; got keys: {sorted(env_means)}")
    if not any(env_means[key] > 0 for key in present):
        raise AssertionError(
            f"expected nonzero {present} (collisions/offroad), got { {key: env_means[key] for key in present} }"
        )


def _mismatches(section, actual, expected, loose_atol, equal_nan):
    mismatches = []
    for key, expected_value in expected.items():
        rtol, atol = (LOOSE_RTOL, loose_atol) if key in LOOSE_KEYS else (RTOL, ATOL)
        if key not in actual:
            mismatches.append(f"  {section}/{key}: MISSING (expected {expected_value})")
        elif not np.isclose(actual[key], expected_value, rtol=rtol, atol=atol, equal_nan=equal_nan):
            mismatches.append(f"  {section}/{key}: {actual[key]!r} != expected {expected_value!r}")
    return mismatches


def record_or_compare_golden(golden_name, total_agents, epochs, sections, loose_atol=LOOSE_ATOL, equal_nan=True):
    """Writes the golden when SMOKE_UPDATE_GOLDEN=1 or it is missing, else asserts every section matches it."""
    golden_path = os.path.join(GOLDEN_DIR, golden_name)
    if os.environ.get("SMOKE_UPDATE_GOLDEN") == "1" or not os.path.exists(golden_path):
        meta = {"total_agents": int(total_agents), "bptt_horizon": BPTT_HORIZON, "epochs": epochs, "seed": SEED}
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(golden_path, "w") as golden_file:
            json.dump({"meta": meta, **sections}, golden_file, indent=2, sort_keys=True)
        print(f"wrote golden -> {golden_path}")
        return
    with open(golden_path) as golden_file:
        golden = json.load(golden_file)
    mismatches = [
        line
        for name, values in sections.items()
        for line in _mismatches(name, values, golden[name], loose_atol, equal_nan)
    ]
    if mismatches:
        raise AssertionError(f"{golden_name}: metrics drifted from the golden:\n" + "\n".join(mismatches))
