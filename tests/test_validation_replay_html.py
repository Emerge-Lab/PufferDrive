"""Integration test for the HTML render backends (triage_html, obs_html).

The single thing we care about: calling the evaluator with each HTML render
backend produces non-empty HTML files at the expected output path. Both are
CPU-only (no EGL/ffmpeg), so they run everywhere.

Uses a stub policy (zero actions) so no trained checkpoint is needed.
"""

import os
import signal
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
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
        value = torch.zeros(batch, 1)
        if self._continuous:
            return torch.distributions.Normal(logits, torch.ones_like(logits)), value
        return logits, value

    def eval(self):
        return self

    def train(self, **_):
        return self

    def parameters(self):
        return iter([])


@pytest.mark.parametrize("backend", ["triage_html", "obs_html"])
def test_html_render_backend_produces_html(tmp_path, backend):
    """Each HTML render backend must produce non-empty HTML files."""
    assert os.path.isdir(WOMD_MAP_DIR), f"Test fixture missing: {WOMD_MAP_DIR}"

    from pufferlib.ocean.benchmark.evaluators import MultiScenarioEvaluator
    from pufferlib.ocean.drive.drive import Drive

    config = {
        "type": "multi_scenario",
        "render": True,
        "render_backend": backend,
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
            # Observation shape, needed by the obs_html viewer to unpack the NN
            # input; the env is built with these same values so the two agree.
            "action_type": "discrete",
            "dynamics_model": "jerk",
            "target_type": "static",
            "reward_conditioning": False,
            "num_target_waypoints": 3,
            "obs_slots_partners_n": 16,
            "obs_slots_lane_n": 80,
            "obs_slots_boundary_n": 80,
            "obs_slots_traffic_controls_n": 4,
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
        args["render_backend"] = backend

        with _watchdog(120, "evaluator rollout + render"):
            ev.rollout(vecenv, policy, args)
    finally:
        vecenv.close()

    html_files = list(Path(tmp_path).rglob("*.html"))
    assert html_files, (
        f"No HTML files produced under {tmp_path}. render_backend={backend} should write one HTML per rendered scenario."
    )
    # Each backend writes one substantial scene HTML per scenario (plus, for
    # obs_html, a small index.html). Require the largest to be non-trivial so a
    # blank/empty render fails.
    largest = max(p.stat().st_size for p in html_files)
    assert largest > 50_000, (
        f"Largest HTML under {tmp_path} is only {largest} bytes — render_backend={backend} likely produced a blank replay."
    )
