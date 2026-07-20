import os
import pickle
from types import SimpleNamespace
import zlib

import numpy as np
import torch

import pufferlib
from pufferlib import pufferl


class _ZeroPolicy:
    def eval(self):
        return self

    def forward_eval(self, observations):
        return (torch.zeros((observations.shape[0], 1)),), torch.zeros((observations.shape[0], 1))


class _RecordingPolicy(_ZeroPolicy):
    def forward_eval(self, observations):
        self.observations = observations.clone()
        return super().forward_eval(observations)


class _CompletedReplayVec:
    agents_per_batch = 1
    action_space = SimpleNamespace(shape=(1, 1))

    def reset(self, seed):
        return np.ones((1, 1), dtype=np.float32), {}

    def step(self, action):
        self.last_action = action
        summary = {
            "summary_type": "completed_episode",
            "replay_environment_bundle": zlib.compress(
                pickle.dumps(
                    {
                        "schema": "interactive_replay_environment_v1",
                        "metadata": {
                            "episode_length": 1,
                            "worker_idx": 0,
                            "active_agent_offset": 0,
                            "active_agent_count": 1,
                        },
                        "scenario": {},
                        "frames": {},
                    }
                )
            ),
        }
        empty = np.zeros(1, dtype=np.float32)
        return np.zeros((1, 1), dtype=np.float32), empty, empty, empty, [[summary]]

    def close(self):
        pass


def test_eval_rollout_writes_replay_bundle_and_keeps_bytes_out_of_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(pufferlib.vector, "make", lambda *args, **kwargs: _CompletedReplayVec())

    def fake_save(scenario, replay, path):
        assert replay["obs"].shape == (1, 1, 1)
        assert replay["raw_action"].shape == (1, 1, 1)
        assert replay["clipped_action"].shape == (1, 1, 1)
        assert replay["value"].shape == (1, 1)
        assert replay["entropy"].shape == (1, 1)
        assert replay["policy_probs"].shape == (1, 1, 1)
        with open(path, "wb") as replay_file:
            replay_file.write(b"standard replay")

    monkeypatch.setattr(pufferlib.viz, "save_interactive_replay_zlib", fake_save)
    args = {
        "package": "ocean",
        "env": {},
        "train": {"seed": 42, "device": "cpu"},
    }

    summaries = pufferl._run_eval_rollout(
        args=args,
        env_name="puffer_drive",
        worker_env_kwargs=[{}],
        total_steps=1,
        desc="test replay",
        expected_episodes=1,
        policy=_ZeroPolicy(),
        replay_output_dir=tmp_path,
    )

    replay_path = tmp_path / "episode_000000.replay.zlib"
    assert replay_path.read_bytes() == b"standard replay"
    assert summaries[0]["has_replay"] == 1
    assert summaries[0]["replay_path"] == str(replay_path.resolve())
    assert "replay_environment_bundle" not in summaries[0]


def test_direct_eval_save_html_enables_zlib_capture_and_save_zlib_alone_does_not_render(monkeypatch):
    def fake_rollout(args, env_name, worker_env_kwargs, *unused_args):
        assert args["save_zlib"] is True
        assert worker_env_kwargs[0]["capture_replay"] is True
        return []

    render_calls = []
    monkeypatch.setattr(pufferl, "_run_eval_rollout", fake_rollout)
    monkeypatch.setattr(pufferl, "_write_eval_reports", lambda *args: None)
    monkeypatch.setattr(pufferl, "_render_eval_replays", lambda summaries, out_dir: render_calls.append(out_dir))
    args = {
        "package": "ocean",
        "train": {"use_rnn": False},
        "env": {
            "scenario_length": 128,
            "num_agents": 1,
        },
        "vec": {"num_envs": 1},
        "num_scenarios": 1,
        "save_html": True,
        "save_zlib": False,
        "replay_csv": None,
        "replay_filter": None,
        "replay_rows": None,
        "load_model_path": None,
    }

    pufferl.eval("puffer_drive", args=args)

    assert render_calls == ["eval_results/puffer_drive/eval"]

    args["save_html"] = False
    render_calls.clear()
    pufferl.eval("puffer_drive", args=args)

    assert render_calls == []


def test_eval_rollout_pads_policy_batch_and_slices_environment_actions(monkeypatch):
    vecenv = _CompletedReplayVec()
    policy = _RecordingPolicy()
    monkeypatch.setattr(pufferlib.vector, "make", lambda *args, **kwargs: vecenv)
    args = {
        "package": "ocean",
        "train": {"seed": 42, "device": "cpu"},
    }

    summaries = pufferl._run_eval_rollout(
        args=args,
        env_name="puffer_drive",
        worker_env_kwargs=[{}],
        total_steps=1,
        desc="test padded replay",
        expected_episodes=1,
        policy=policy,
        expected_agents_per_batch=4,
    )

    assert policy.observations.shape == (4, 1)
    assert policy.observations[0].item() == 1
    assert torch.count_nonzero(policy.observations[1:]).item() == 0
    assert vecenv.last_action.shape == (1, 1)
    assert summaries[0]["agents_per_batch"] == 4


def test_replay_worker_kwargs_balances_without_fillers():
    args = {
        "env": {"num_agents": 1024},
        "save_zlib": False,
    }
    pairs = [(map_idx, 100 + map_idx) for map_idx in range(5)]

    worker_kwargs, total_steps = pufferl._replay_worker_kwargs(
        args=args,
        pairs=pairs,
        num_workers=4,
        scenario_length=128,
    )

    assert total_steps == 256
    assert [kwargs["num_eval_scenarios"] for kwargs in worker_kwargs] == [2, 1, 1, 1]
    assert [map_idx for kwargs in worker_kwargs for map_idx in kwargs["eval_map_indices"]] == list(range(5))


def test_render_eval_replays_writes_pages_with_navigation_and_index(monkeypatch, tmp_path):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay_paths = [replay_dir / f"episode_{episode_id:06d}.replay.zlib" for episode_id in range(2)]
    for replay_path in replay_paths:
        replay_path.write_bytes(b"replay")

    render_calls = []

    def fake_render(replay_path, output_path):
        render_calls.append((replay_path, output_path))
        with open(output_path, "w") as html_file:
            html_file.write("rendered")

    def fake_index(output_path, file_metrics):
        assert sorted(file_metrics) == ["episode_000000.html", "episode_000001.html"]
        with open(os.path.join(output_path, "index.html"), "w") as html_file:
            html_file.write("index")

    monkeypatch.setattr(pufferlib.viz, "render_interactive_replay_zlib", fake_render)
    monkeypatch.setattr(pufferlib.viz, "build_gallery_index", fake_index)
    summaries = [
        {
            "map_name": f"map_{episode_id}",
            "scenario_id": f"scenario_{episode_id}",
            "has_replay": 1,
            "replay_path": str(replay_path),
        }
        for episode_id, replay_path in enumerate(replay_paths)
    ]

    render_dir = pufferl._render_eval_replays(summaries, str(tmp_path))

    assert render_dir == str(tmp_path / "rendered_replays")
    assert (tmp_path / "rendered_replays" / "episode_000000.html").read_text() == "rendered"
    assert (tmp_path / "rendered_replays" / "episode_000001.html").read_text() == "rendered"
    assert (tmp_path / "rendered_replays" / "index.html").read_text() == "index"
    assert render_calls[0][0] == str(replay_paths[0])
    assert render_calls[1][0] == str(replay_paths[1])
