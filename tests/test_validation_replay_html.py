"""Integration test for the validation_replay HTML render pipeline.

The single thing we care about: calling the evaluator with render_backend=html
produces non-empty HTML files at the expected output path.

Uses a stub policy (zero actions) so no trained checkpoint is needed.
"""

import os
import signal
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_ROOT = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries")
WOMD_MAP_DIR = os.path.join(BIN_ROOT, "obstacles")

SCENARIO_LENGTH = 91
NUM_SCENARIOS = 1


@contextmanager
def _watchdog(seconds, what):
    def _handler(signum, frame):
        raise TimeoutError(f"{what} hung for >{seconds}s")

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


class _ZeroPolicy:
    """Stub policy that always outputs zero logits. Infers action dimension
    from the single_action_space of the vecenv."""

    def __init__(self, single_action_space):
        import gymnasium

        if isinstance(single_action_space, gymnasium.spaces.Box):
            self._n = int(np.prod(single_action_space.shape))
            self._continuous = True
        else:
            nvec = getattr(single_action_space, "nvec", None)
            self._n = int(np.prod(nvec)) if nvec is not None else int(single_action_space.n)
            self._continuous = False

    def forward_eval(self, obs, state):
        batch = obs.shape[0]
        logits = torch.zeros(batch, self._n)
        if self._continuous:
            return torch.distributions.Normal(logits, torch.ones_like(logits)), state
        return logits, state

    def eval(self):
        return self

    def train(self, **_):
        return self

    def parameters(self):
        return iter([])


def test_validation_replay_produces_html(tmp_path):
    """Calling the evaluator with render_backend=html must produce non-empty HTML files."""
    assert os.path.isdir(WOMD_MAP_DIR), f"Test fixture missing: {WOMD_MAP_DIR}"

    from pufferlib.ocean.benchmark.evaluators import MultiScenarioEvaluator
    from pufferlib.ocean.drive.drive import Drive

    config = {
        "type": "multi_scenario",
        "render": True,
        "render_backend": "html",
        "env": {
            "simulation_mode": "replay",
            "control_mode": "control_sdc_only",
            "map_dir": WOMD_MAP_DIR,
            "num_maps": NUM_SCENARIOS,
            "num_agents": 1,
            "min_agents_per_env": 1,
            "max_agents_per_env": 1,
            "scenario_length": SCENARIO_LENGTH,
            "resample_frequency": 0,
        },
        "eval": {
            "num_scenarios": NUM_SCENARIOS,
            "render_num_scenarios": NUM_SCENARIOS,
            "render_max_steps": SCENARIO_LENGTH,
        },
    }
    train_config = {
        "package": "ocean",
        "env_name": "puffer_drive",
        "env": config["env"],
        "vec": {"backend": "PufferEnv", "num_envs": 1},
        "train": {"device": "cpu", "seed": 0},
        "render_results_dir": str(tmp_path),
    }

    ev = MultiScenarioEvaluator(name="validation_replay", config=config, train_config=train_config)

    import pufferlib.vector as pvec

    make_env = lambda: Drive(**config["env"])
    with _watchdog(30, "vecenv init"):
        vecenv = pvec.make(make_env, env_kwargs={}, backend="PufferEnv", num_envs=1)

    try:
        policy = _ZeroPolicy(vecenv.single_action_space)
        args = dict(train_config)
        args["env_name"] = "puffer_drive"
        args["render_backend"] = "html"

        with _watchdog(120, "evaluator rollout + render"):
            ev.rollout(vecenv, policy, args)
    finally:
        vecenv.close()

    html_files = list(Path(tmp_path).rglob("*.html"))
    assert html_files, (
        f"No HTML files produced under {tmp_path}. render_backend=html should write one HTML per rendered scenario."
    )
    for html_path in html_files:
        size = html_path.stat().st_size
        assert size > 50_000, f"{html_path.name} is only {size} bytes — likely a blank replay"
