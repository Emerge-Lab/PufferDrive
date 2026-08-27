import numbers
import os
import pickle
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

import pufferlib
import pufferlib.viz


def _eval_replay_stem(summary, episode_id):
    map_stem = os.path.splitext(os.path.basename(summary["map_name"]))[0]
    return f"{map_stem}__seed_{summary['seed']}__episode_{episode_id:06d}"


class EvalReplayCapture:
    """Capture policy history and combine it with completed environment replay bundles."""

    def __init__(
        self,
        args,
        policy,
        replay_output_dir,
        capture_observations,
        num_workers,
        agents_per_batch,
        capture_batch_steps,
        episode_id_offset,
    ):
        if capture_batch_steps <= 0:
            raise RuntimeError("Replay capture requires a positive resample frequency")
        self.capture_observations = capture_observations
        self.replay_writer_count = num_workers
        if self.capture_observations:
            observation_replay_writer_count = args["eval"]["observation_replay_writer_count"]
            if (
                isinstance(observation_replay_writer_count, bool)
                or not isinstance(observation_replay_writer_count, int)
                or observation_replay_writer_count <= 0
            ):
                raise pufferlib.APIUsageError(
                    "eval.observation_replay_writer_count must be a positive integer when rendering observations"
                )
            self.replay_writer_count = min(num_workers, observation_replay_writer_count)

        self.env_config = dict(args["env"])
        self.replay_output_dir = replay_output_dir
        self.num_workers = num_workers
        self.agents_per_batch = agents_per_batch
        self.agents_per_worker = agents_per_batch // num_workers
        self.capture_batch_steps = capture_batch_steps
        self.episode_id_offset = episode_id_offset
        self.pool_slot_counts = None
        if self.capture_observations:
            self.pool_slot_counts = getattr(policy, "pool_slot_counts", None)
            if self.pool_slot_counts is None and getattr(policy, "policy", None) is not None:
                self.pool_slot_counts = getattr(policy.policy, "pool_slot_counts", None)
        self.policy_history = {}
        self.policy_history_frame_count = 0
        self.pending_replays = []
        os.makedirs(replay_output_dir, exist_ok=True)

    @property
    def pending_count(self):
        return len(self.pending_replays)

    def capture_frame(self, obs, policy_obs_tensor, raw_action, action, logits, value, logprob, entropy):
        if self.policy_history_frame_count == self.capture_batch_steps:
            self.policy_history = {}
            self.policy_history_frame_count = 0
        replay_frame = {
            "raw_action": np.asarray(raw_action, dtype=np.float32),
            "clipped_action": np.asarray(action, dtype=np.float32),
            "value": value[: self.agents_per_batch].detach().reshape(-1).float().cpu().numpy(),
            "entropy": entropy[: self.agents_per_batch].detach().reshape(-1).float().cpu().numpy(),
        }
        if self.capture_observations:
            replay_frame["obs"] = np.asarray(obs, dtype=np.float16)
        if isinstance(logits, torch.distributions.Normal):
            replay_frame["policy_mean"] = logits.loc[: self.agents_per_batch].detach().float().cpu().numpy()
            replay_frame["policy_std"] = logits.scale[: self.agents_per_batch].detach().float().cpu().numpy()
            replay_frame["policy_log_prob"] = (
                logprob[: self.agents_per_batch].detach().reshape(-1).float().cpu().numpy()
            )
        else:
            discrete_logits = logits if isinstance(logits, torch.Tensor) else logits[0]
            replay_frame["policy_probs"] = (
                torch.softmax(discrete_logits[: self.agents_per_batch], dim=-1).detach().float().cpu().numpy()
            )
        if self.pool_slot_counts is not None:
            for pool_name, pool_values in self.pool_slot_counts(policy_obs_tensor).items():
                replay_frame[pool_name] = (
                    pool_values[: self.agents_per_batch].detach().cpu().numpy().astype(np.int16, copy=False)
                )

        if not self.policy_history:
            self.policy_history = {
                replay_key: np.empty(
                    (self.capture_batch_steps, *frame_values.shape),
                    dtype=frame_values.dtype,
                )
                for replay_key, frame_values in replay_frame.items()
            }
        for replay_key, frame_values in replay_frame.items():
            self.policy_history[replay_key][self.policy_history_frame_count] = frame_values
        self.policy_history_frame_count += 1

    def queue_replay(self, summary, episode_id):
        replay_environment_bytes = summary.pop("replay_environment_bundle", None)
        if not isinstance(replay_environment_bytes, bytes):
            raise RuntimeError(
                "Replay capture was requested, but an evaluation episode did not include environment bytes"
            )
        replay_environment = pickle.loads(zlib.decompress(replay_environment_bytes))
        if replay_environment.get("schema") != "interactive_replay_environment_v1":
            raise RuntimeError("Replay environment bundle has an unsupported schema")

        metadata = replay_environment["metadata"]
        episode_length = metadata["episode_length"]
        worker_idx = metadata["worker_idx"]
        active_agent_offset = metadata["active_agent_offset"]
        num_agents = metadata["num_agents"]
        global_agent_start = worker_idx * self.agents_per_worker + active_agent_offset
        global_agent_end = global_agent_start + num_agents
        if (
            worker_idx < 0
            or worker_idx >= self.num_workers
            or active_agent_offset < 0
            or num_agents <= 0
            or global_agent_end > self.agents_per_batch
            or episode_length > self.policy_history_frame_count
        ):
            raise RuntimeError("Replay environment metadata is incompatible with the policy history")

        replay = {"env": self.env_config, **replay_environment["frames"]}
        for replay_key, history_values in self.policy_history.items():
            replay[replay_key] = history_values[:episode_length, global_agent_start:global_agent_end]
        replay_stem = _eval_replay_stem(summary, self.episode_id_offset + episode_id)
        replay_path = os.path.abspath(os.path.join(self.replay_output_dir, f"{replay_stem}.replay.zlib"))
        self.pending_replays.append((replay_environment["scenario"], replay, replay_path))
        summary["has_replay"] = 1
        summary["replay_path"] = replay_path

    def write_pending(self):
        scenarios, replays, replay_paths = zip(*self.pending_replays)
        writer_count = min(self.replay_writer_count, len(self.pending_replays))
        with ThreadPoolExecutor(max_workers=writer_count) as replay_writer:
            for _ in replay_writer.map(
                pufferlib.viz.save_interactive_replay_zlib,
                scenarios,
                replays,
                replay_paths,
            ):
                pass
        self.pending_replays = []


def _render_eval_replays(episode_summaries, out_dir):
    """Render captured eval replays as navigable HTML pages plus an index."""
    render_dir = os.path.join(out_dir, "rendered_replays")
    os.makedirs(render_dir, exist_ok=True)

    file_metrics = {}
    replay_paths = []
    output_paths = []
    for episode_id, summary in enumerate(episode_summaries):
        replay_path = summary.get("replay_path")
        if not replay_path or not os.path.isfile(replay_path):
            raise RuntimeError(f"Cannot render episode {episode_id}: replay file is missing: {replay_path}")
        replay_filename = os.path.basename(replay_path)
        replay_suffix = ".replay.zlib"
        if not replay_filename.endswith(replay_suffix):
            raise RuntimeError(f"Cannot render episode {episode_id}: unexpected replay filename: {replay_filename}")
        html_filename = f"{replay_filename[: -len(replay_suffix)]}.html"
        output_path = os.path.join(render_dir, html_filename)
        replay_paths.append(replay_path)
        output_paths.append(output_path)
        file_metrics[html_filename] = {
            key: value for key, value in summary.items() if isinstance(value, numbers.Real) and np.isfinite(value)
        }

    if replay_paths:
        html_renderer_count = min(os.cpu_count() or 1, len(replay_paths))
        with ThreadPoolExecutor(max_workers=html_renderer_count) as html_renderer:
            rendered_replays = html_renderer.map(
                pufferlib.viz.render_interactive_replay_zlib,
                replay_paths,
                output_paths,
            )
            for _ in tqdm(rendered_replays, total=len(replay_paths), desc="Rendering replay HTML"):
                pass

    pufferlib.viz.build_gallery_index(render_dir, file_metrics=file_metrics)
    print(f"Rendered {len(episode_summaries)} replay pages into {render_dir}")
    print(f"Wrote replay index to {os.path.join(render_dir, 'index.html')}")
    return render_dir
