#!/usr/bin/env python3
"""Config contract for the co-simulation integrations (pufferlib/ocean/cosim).

The CARLA leaderboard agent, the standalone CARLA co-sim, and the nuPlan
planner all rebuild the checkpoint's shadow Drive env FROM ITS config.yaml
(saved next to the checkpoint from pufferlib/config/puffer_drive.yaml) via
cosim/arch.py's shadow_env_kwargs: every Drive-accepted env key is adopted,
then CLEAN_EVAL_OVERRIDES is applied, then the co-sim's structural keys win.
A key silently dropped (renamed in Drive.__init__ or removed from the yaml)
flips the shadow env to a Drive default and breaks observation-encoding
parity with the trained policy WITHOUT any error.

This test pins the keys the co-sim reads, the adopt-then-override precedence,
and equality between cosim's CLEAN_EVAL_OVERRIDES and the benchmark's.

Run: python -m unittest tests.unit_tests.test_cosim_config_contract
"""

import inspect
import os
import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "pufferlib" / "config" / "puffer_drive.yaml"

# Env keys the co-sim reads DIRECTLY from the checkpoint config (KeyError if
# missing): leaderboard_agent.setup() for the policy-step interval, the shadow
# agent pool, and the route-goal placement; plus keys whose silent fallback to
# a Drive default would break parity for every consumer.
COSIM_ENV_KEYS = (
    "dt",
    "max_agents_per_env",
    "min_goal_spacing",
    "max_goal_spacing",
    "goal_radius",
    "num_goals",
    "dynamics_model",
    "goal_speed",
    "goal_source",
    "reward_conditioning",
)

# obs_* keys are deliberately NOT pinned: shadow_env_kwargs adopts whatever
# obs_* keys the checkpoint config carries, and a key absent from the config
# falls back to the same Drive default the training run used -- consistent.

# Top-level keys the cosim policy loaders read (leaderboard_agent.py,
# carla_cosim.py, nuplan/planner.py): getattr(drive_torch, cfg["policy_name"])
# and policy_cls(env, **cfg["policy"]).
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
                f"env.{key} was removed from puffer_drive.yaml; the co-sim shadow env "
                f"(pufferlib/ocean/cosim) reads it from checkpoint configs and would "
                f"either crash or silently fall back to a Drive default, breaking "
                f"parity with the trained policy.",
            )

    def test_top_level_has_policy_keys(self):
        for key in COSIM_TOP_LEVEL_KEYS:
            self.assertIn(
                key,
                self.cfg,
                f"'{key}' was removed from puffer_drive.yaml; the co-sim loads the "
                f"policy class/kwargs from it when restoring a checkpoint.",
            )

    def test_drive_still_accepts_pinned_keys(self):
        # shadow_env_kwargs filters adopted keys through Drive.__init__'s
        # signature: a renamed Drive kwarg silently drops the key instead of
        # erroring.
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
        """cosim/arch.py duplicates the benchmark profile's noise/light keys so
        the co-sim venvs never import the training stack; each mirrored key
        must match pufferlib/config/evaluation/benchmark.yaml."""
        from pufferlib.ocean.cosim.arch import CLEAN_EVAL_OVERRIDES as cosim_overrides

        benchmark_yaml = REPO_ROOT / "pufferlib" / "config" / "evaluation" / "benchmark.yaml"
        bench_env = yaml.safe_load(benchmark_yaml.read_text())["env"]
        for key, value in cosim_overrides.items():
            self.assertEqual(
                bench_env.get(key),
                value,
                f"cosim/arch.py CLEAN_EVAL_OVERRIDES['{key}'] drifted from "
                "config/evaluation/benchmark.yaml; the shadow env must run "
                "the same clean-eval profile as the repo's own evaluations.",
            )

    def test_clean_eval_overrides_beat_checkpoint_config(self):
        """A checkpoint trained WITH observation noise must still evaluate
        clean: every CLEAN_EVAL_OVERRIDES key wins over the config's value."""
        from pufferlib.ocean.cosim.arch import CLEAN_EVAL_OVERRIDES, shadow_env_kwargs

        cfg = {"env": {key: 0.9 for key in CLEAN_EVAL_OVERRIDES if key != "traffic_light_behavior"}}
        cfg["env"]["traffic_light_behavior"] = "remove"
        kwargs = shadow_env_kwargs(cfg)
        for key, value in CLEAN_EVAL_OVERRIDES.items():
            self.assertEqual(
                kwargs.get(key),
                value,
                f"'{key}' from the checkpoint config leaked past the clean-eval profile into the co-sim shadow env.",
            )

    def test_int_infraction_behaviors_normalize_to_strings(self):
        """Saved checkpoint configs may carry resolved enum ints for the
        infraction behaviors (weights/mimolette: collision_behavior: 1);
        Drive.__init__ only accepts the strings."""
        from pufferlib.ocean.cosim.arch import shadow_env_kwargs

        kwargs = shadow_env_kwargs({"env": {"collision_behavior": 1, "offroad_behavior": 0}})
        self.assertEqual(kwargs["collision_behavior"], "stop")
        self.assertEqual(kwargs["offroad_behavior"], "ignore")

    def test_adopts_every_drive_accepted_key_and_filters_the_rest(self):
        from pufferlib.ocean.cosim.arch import shadow_env_kwargs

        cfg = {"env": {"obs_slots_partners_n": 63, "min_goal_spacing": 11.0, "not_a_drive_kwarg": 1}}
        kwargs = shadow_env_kwargs(cfg)
        self.assertEqual(kwargs["obs_slots_partners_n"], 63)
        self.assertEqual(kwargs["min_goal_spacing"], 11.0)
        self.assertNotIn("not_a_drive_kwarg", kwargs)

    def test_structural_overrides_beat_everything(self):
        """The co-sim's structural keys (map/pool wiring) must win over the
        training config, which points at training maps and multi-env pools."""
        from pufferlib.ocean.cosim.arch import shadow_env_kwargs

        cfg = {"env": {"num_agents": 512, "num_maps": 100, "resample_frequency": 91}}
        kwargs = shadow_env_kwargs(
            cfg, overrides={"num_agents": 64, "num_maps": 1, "resample_frequency": 0, "map_dir": "town.bin"}
        )
        self.assertEqual(kwargs["num_agents"], 64)
        self.assertEqual(kwargs["num_maps"], 1)
        self.assertEqual(kwargs["resample_frequency"], 0)
        self.assertEqual(kwargs["map_dir"], "town.bin")

    def test_defaults_only_fill_gaps(self):
        """`defaults` (DEFAULT_ARCH / dummy-run arch) must lose to a real
        checkpoint config."""
        from pufferlib.ocean.cosim.arch import shadow_env_kwargs

        kwargs = shadow_env_kwargs({"env": {"goal_source": "route"}}, defaults={"goal_source": "map", "num_goals": 3})
        self.assertEqual(kwargs["goal_source"], "route")
        self.assertEqual(kwargs["num_goals"], 3)


if __name__ == "__main__":
    unittest.main()
