"""Co-sim config contract: the shadow env rebuilt from a checkpoint config keeps every key the policy relies on."""

import inspect
import unittest
from pathlib import Path

import yaml

from opengigaflow.ocean.drive.drive import Drive


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "opengigaflow" / "config" / "opengigaflow.yaml"

# env keys the co-sim reads straight from checkpoint configs, plus keys whose Drive default would break parity
COSIM_ENV_KEYS = (
    "dt",
    "max_agents_per_env",
    "min_goal_spacing",
    "max_goal_spacing",
    "goal_radius",
    "num_goals",
    "goal_speed",
    "goal_source",
    "reward_conditioning",
)

# obs_* keys need no pin: one a checkpoint config lacks falls back to the Drive default its training run used too

# the co-sim policy loaders build getattr(drive_torch, cfg["policy_name"])(env, **cfg["policy"])
COSIM_TOP_LEVEL_KEYS = ("policy", "policy_name")


class TestCosimConfigContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONFIG_PATH) as f:
            cls.cfg = yaml.safe_load(f)

    def test_env_section_has_cosim_keys(self):
        env = self.cfg.get("env")
        self.assertIsInstance(env, dict, f"{CONFIG_PATH} lost its env section")
        for key in COSIM_ENV_KEYS:
            self.assertIn(
                key,
                env,
                f"env.{key} was removed from opengigaflow.yaml; the co-sim shadow env "
                f"(opengigaflow/ocean/cosim) reads it from checkpoint configs and would "
                f"either crash or silently fall back to a Drive default, breaking "
                f"parity with the trained policy.",
            )

    def test_top_level_has_policy_keys(self):
        for key in COSIM_TOP_LEVEL_KEYS:
            self.assertIn(
                key,
                self.cfg,
                f"'{key}' was removed from opengigaflow.yaml; the co-sim loads the "
                f"policy class/kwargs from it when restoring a checkpoint.",
            )

    def test_drive_still_accepts_pinned_keys(self):
        # shadow_env_kwargs filters adopted keys by Drive.__init__'s signature: a renamed kwarg drops silently
        accepted = set(inspect.signature(Drive.__init__).parameters)
        for key in COSIM_ENV_KEYS:
            self.assertIn(
                key,
                accepted,
                f"Drive.__init__ no longer accepts '{key}'; shadow_env_kwargs would "
                f"silently drop it and the shadow env would diverge from the training env.",
            )


class TestShadowEnvKwargs(unittest.TestCase):
    def test_clean_eval_matches_benchmark_profile(self):
        """cosim/arch.py mirrors benchmark.yaml's noise/light keys: co-sim venvs never import the training stack."""
        from opengigaflow.ocean.cosim.arch import CLEAN_EVAL_OVERRIDES as cosim_overrides

        benchmark_yaml = REPO_ROOT / "opengigaflow" / "config" / "evaluation" / "benchmark.yaml"
        bench_env = yaml.safe_load(benchmark_yaml.read_text())["env"]
        for key, value in cosim_overrides.items():
            self.assertEqual(
                bench_env.get(key),
                value,
                f"cosim/arch.py CLEAN_EVAL_OVERRIDES['{key}'] drifted from "
                "config/evaluation/benchmark.yaml; the shadow env must run "
                "the same clean-eval profile as the repo's own evaluations.",
            )

    def test_every_benchmark_randomization_pin_is_mirrored(self):
        """A randomization benchmark.yaml pins off is pinned off in the co-sim mirror too: no training value leaks."""
        from opengigaflow.ocean.cosim.arch import CLEAN_EVAL_OVERRIDES as cosim_overrides

        benchmark_yaml = REPO_ROOT / "opengigaflow" / "config" / "evaluation" / "benchmark.yaml"
        bench_env = yaml.safe_load(benchmark_yaml.read_text())["env"]
        randomization_markers = ("random", "noise", "dropout", "blindness", "phantom", "spawn_")
        for key in bench_env:
            if not any(marker in key for marker in randomization_markers):
                continue
            self.assertIn(
                key,
                cosim_overrides,
                f"benchmark.yaml pins '{key}' but cosim/arch.py CLEAN_EVAL_OVERRIDES does not mirror it; "
                "the co-sim shadow env would adopt the checkpoint's training value.",
            )

    def test_clean_eval_overrides_beat_checkpoint_config(self):
        """A checkpoint trained with observation noise still evaluates clean: CLEAN_EVAL_OVERRIDES beat its config."""
        from opengigaflow.ocean.cosim.arch import CLEAN_EVAL_OVERRIDES, shadow_env_kwargs

        cfg = {"env": {key: 0.9 for key in CLEAN_EVAL_OVERRIDES if key != "traffic_light_behavior"}}
        cfg["env"]["traffic_light_behavior"] = "remove"
        kwargs = shadow_env_kwargs(cfg)
        for key, value in CLEAN_EVAL_OVERRIDES.items():
            self.assertEqual(
                kwargs.get(key),
                value,
                f"'{key}' from the checkpoint config leaked past the clean-eval profile into the co-sim shadow env.",
            )

    def test_adopts_every_drive_accepted_key_and_filters_the_rest(self):
        from opengigaflow.ocean.cosim.arch import shadow_env_kwargs

        cfg = {"env": {"obs_slots_partners_n": 63, "min_goal_spacing": 11.0, "not_a_drive_kwarg": 1}}
        kwargs = shadow_env_kwargs(cfg)
        self.assertEqual(kwargs["obs_slots_partners_n"], 63)
        self.assertEqual(kwargs["min_goal_spacing"], 11.0)
        self.assertNotIn("not_a_drive_kwarg", kwargs)

    def test_structural_overrides_beat_everything(self):
        """The co-sim's map/pool wiring keys win over the training config's training maps and multi-env pools."""
        from opengigaflow.ocean.cosim.arch import shadow_env_kwargs

        cfg = {"env": {"num_agents": 512, "num_maps": 100, "resample_frequency": 91}}
        kwargs = shadow_env_kwargs(
            cfg, overrides={"num_agents": 64, "num_maps": 1, "resample_frequency": 0, "map_dir": "town.bin"}
        )
        self.assertEqual(kwargs["num_agents"], 64)
        self.assertEqual(kwargs["num_maps"], 1)
        self.assertEqual(kwargs["resample_frequency"], 0)
        self.assertEqual(kwargs["map_dir"], "town.bin")


if __name__ == "__main__":
    unittest.main()
