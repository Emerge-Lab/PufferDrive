import os
import pickle
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import pufferlib
import pufferlib.viz


def _eval_replay_stem(summary, episode_id):
    """Build a safe, unique replay stem from externally supplied scenario metadata."""

    def safe_part(value, fallback):
        if value is None:
            return fallback
        basename = os.path.basename(str(value)).rsplit(".", 1)[0]
        sanitized = "".join(character if character.isalnum() or character in "-_" else "_" for character in basename)
        return sanitized.strip("_-")[:64] or fallback

    map_name = safe_part(summary.get("map_name"), "unknown_map")
    scenario_id = safe_part(summary.get("scenario_id"), "")
    seed = safe_part(summary.get("seed"), "unknown")
    parts = [map_name]
    if scenario_id and scenario_id.lower() not in map_name.lower():
        parts.append(scenario_id)
    parts.extend((f"seed_{seed}", f"episode_{episode_id:06d}"))
    return "__".join(parts)


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
        replay_episode_offset,
    ):
        if capture_batch_steps <= 0:
            raise RuntimeError("Replay capture requires a positive resample frequency")
        self.capture_observations = bool(capture_observations)
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
        self.replay_episode_offset = replay_episode_offset
        self.pool_method = None
        if self.capture_observations:
            self.pool_method = getattr(policy, "pool_slot_counts", None)
            if self.pool_method is None and getattr(policy, "policy", None) is not None:
                self.pool_method = getattr(policy.policy, "pool_slot_counts", None)
        self.history = {}
        self.history_frame_count = 0
        self.pending_replays = []
        os.makedirs(replay_output_dir, exist_ok=True)

    @property
    def pending_count(self):
        return len(self.pending_replays)

    def capture_frame(self, obs, policy_obs_tensor, raw_action, action, logits, value, logprob, entropy):
        if self.history_frame_count == self.capture_batch_steps:
            self.reset_history()
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
        if self.pool_method is not None:
            for pool_name, pool_values in self.pool_method(policy_obs_tensor).items():
                replay_frame[pool_name] = (
                    pool_values[: self.agents_per_batch].detach().cpu().numpy().astype(np.int16, copy=False)
                )

        if not self.history:
            self.history = {
                replay_key: np.empty(
                    (self.capture_batch_steps, *frame_values.shape),
                    dtype=frame_values.dtype,
                )
                for replay_key, frame_values in replay_frame.items()
            }
        for replay_key, frame_values in replay_frame.items():
            self.history[replay_key][self.history_frame_count] = frame_values
        self.history_frame_count += 1

    def queue_episode(self, summary, episode_idx):
        replay_environment_bytes = summary.pop("replay_environment_bundle", None)
        if not isinstance(replay_environment_bytes, bytes):
            raise RuntimeError(
                "Replay capture was requested, but an evaluation episode did not include environment bytes"
            )
        replay_environment = pickle.loads(zlib.decompress(replay_environment_bytes))
        if replay_environment.get("schema") != "interactive_replay_environment_v1":
            raise RuntimeError("Replay environment bundle has an unsupported schema")

        metadata = replay_environment["metadata"]
        episode_length = int(metadata["episode_length"])
        worker_idx = int(metadata["worker_idx"])
        active_agent_offset = int(metadata["active_agent_offset"])
        active_agent_count = int(metadata["active_agent_count"])
        global_agent_start = worker_idx * self.agents_per_worker + active_agent_offset
        global_agent_end = global_agent_start + active_agent_count
        if (
            worker_idx < 0
            or worker_idx >= self.num_workers
            or active_agent_offset < 0
            or active_agent_count <= 0
            or global_agent_end > self.agents_per_batch
            or episode_length > self.history_frame_count
        ):
            raise RuntimeError("Replay environment metadata is incompatible with the policy history")

        replay = {"env": self.env_config, **replay_environment["frames"]}
        for replay_key, history_values in self.history.items():
            replay[replay_key] = history_values[:episode_length, global_agent_start:global_agent_end]
        replay_stem = _eval_replay_stem(summary, self.replay_episode_offset + episode_idx)
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

    def reset_history(self):
        self.history = {}
        self.history_frame_count = 0
