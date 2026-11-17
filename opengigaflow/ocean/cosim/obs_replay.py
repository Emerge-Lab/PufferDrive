"""Co-sim observation replay pages (opengigaflow.viz): per-step policy obs, outputs and encoder pool winners."""

from pathlib import Path

import numpy as np

from opengigaflow.ocean.drive import binding


OBS_CLIP = 100.0  # parked FAR_AWAY slots carry ~5e3 goal offsets that would blow up the viewer's int16 quantization
ROAD_CROP_MARGIN_M = 250.0  # whole-city bins: keep only road elements near the driven trajectory


class ObsReplayCapture:
    def __init__(self, env, policy, out_stem, max_steps=800, ghost_trajectory=None):
        """ghost_trajectory: (N, 5) logged ego boxes [x, y, heading, length, width] in the bin frame (row = frame)."""
        self.env = env
        self.policy = policy
        self.out_stem = Path(out_stem)
        self.max_steps = int(max_steps)
        self.ghost_trajectory = (
            None if ghost_trajectory is None else np.asarray(ghost_trajectory, np.float32).reshape(-1, 5)
        )
        state = env.get_state()
        self.scenario = state[0] if isinstance(state, list) else state
        self.agent_cap = int(self.scenario["num_total_agents"])
        self.traffic_cap = max(int(self.scenario["num_traffic_elements"]), 1)
        self.frame_buffer_layouts = {
            "agent_f32": ((1, self.agent_cap, binding.AGENT_F32_FIELDS), np.float32),
            "agent_i32": ((1, self.agent_cap, binding.AGENT_I32_FIELDS), np.int32),
            "metrics_f32": ((1, self.agent_cap, binding.METRICS_F32_FIELDS), np.float32),
            "traffic_i16": ((1, self.traffic_cap, binding.TRAFFIC_I16_FIELDS), np.int16),
            "goals_f32": ((1, self.agent_cap, int(env.num_goals) * binding.GOAL_XY_FIELDS), np.float32),
            "rewards_f32": ((1, self.agent_cap, binding.REWARD_F32_FIELDS), np.float32),
            "coefs_f32": ((1, self.agent_cap, binding.NUM_REWARD_COEFS), np.float32),
        }
        self.frames = {key: [] for key in self.frame_buffer_layouts}
        self.frames.update(obs=[], raw_action=[], action_index=[], value=[], entropy=[], policy_probs=[])

    def __len__(self):
        return len(self.frames["obs"])

    def policy_outputs(self, obs_tensor, logits, value, entropy):
        """-> aux dict for capture(): value, entropy, action probabilities and encoder pool winners."""
        import torch

        probs = torch.softmax(logits if isinstance(logits, torch.Tensor) else logits[0], dim=-1)
        aux = {
            "value": value.detach().cpu().numpy().reshape(-1).astype(np.float32),
            "entropy": entropy.detach().cpu().numpy().reshape(-1).astype(np.float32),
            "policy_probs": probs.detach().cpu().numpy().astype(np.float32),
        }
        with torch.no_grad():
            pool = self.policy.pool_slot_counts(obs_tensor)
        aux["pool"] = {name: counts.cpu().numpy().astype(np.int16) for name, counts in pool.items()}
        return aux

    def capture(self, obs, actions, aux, action_index=None):
        """action_index: discrete class per agent behind `actions` (argmax for a mean action); None or -1: none."""
        if len(self) >= self.max_steps:
            return
        frame_buffers = {name: np.zeros(shape, dtype) for name, (shape, dtype) in self.frame_buffer_layouts.items()}
        self.env.get_obs_html_frame(*frame_buffers.values())
        frames = self.frames
        for name, frame_buffer in frame_buffers.items():
            frames[name].append(frame_buffer[0])
        obs = np.asarray(obs, dtype=np.float32)
        frames["obs"].append(np.clip(obs, -OBS_CLIP, OBS_CLIP))
        frames["raw_action"].append(np.asarray(actions, dtype=np.float32).reshape(obs.shape[0], -1))
        index = (
            np.full(obs.shape[0], -1, np.int32)
            if action_index is None
            else np.asarray(action_index, np.int32).reshape(obs.shape[0])
        )
        frames["action_index"].append(index)
        frames["value"].append(aux.get("value", np.zeros(obs.shape[0], np.float32)))
        frames["entropy"].append(aux.get("entropy", np.zeros(obs.shape[0], np.float32)))
        if "policy_probs" in aux:
            frames["policy_probs"].append(aux["policy_probs"])
        for pool_name, counts in aux.get("pool", {}).items():
            frames.setdefault(pool_name, []).append(counts)

    def _env_cfg(self):
        env = self.env
        return {
            "action_type": "discrete",
            "num_goals": int(env.num_goals),
            "goal_radius": float(env.goal_radius),
            "reward_conditioning": bool(env.num_reward_coefs),
            "obs_slots_partners_n": int(env.obs_slots_partners_n),
            "obs_slots_lane_n": int(env.obs_slots_lane_n),
            "obs_slots_boundary_n": int(env.obs_slots_boundary_n),
            "obs_slots_traffic_controls_n": int(env.obs_slots_traffic_controls_n),
            "obs_dropout_lane": float(env.obs_dropout_lane),
            "obs_dropout_boundary": float(env.obs_dropout_boundary),
            "obs_norm_goal_offset_m": float(env.obs_norm_goal_offset_m),
            "obs_norm_xy_offset_m": float(env.obs_norm_xy_offset_m),
            "obs_norm_veh_width_m": float(env.obs_norm_veh_width_m),
            "obs_norm_veh_length_m": float(env.obs_norm_veh_length_m),
            "obs_norm_road_seg_length_m": float(env.obs_norm_road_seg_length_m),
            "obs_norm_road_seg_width_m": float(env.obs_norm_road_seg_width_m),
            "obs_norm_speed_mps": float(env.obs_norm_speed_mps),
            "obs_traffic_light_yellow_remaining": bool(env.obs_traffic_light_yellow_remaining),
        }

    def _check_layout(self, obs_dim):
        """The viewer decodes the obs from binding constants + env_cfg; refuse to write a replay it would misread."""
        env = self.env
        viewer_dim = (
            env.ego_features
            + env.num_reward_coefs
            + env.goal_dim
            + env.obs_slots_partners_n * env.partner_features
            + env.obs_slots_lane_kept * env.lane_features
            + env.obs_slots_boundary_kept * env.boundary_features
            + env.obs_slots_traffic_controls_n * env.traffic_control_features
        )
        if obs_dim < viewer_dim or obs_dim - viewer_dim > binding.OBS_VALID_COUNT_FEATURES:
            raise RuntimeError(
                f"obs replay layout mismatch: obs has {obs_dim} columns, viewer layout expects {viewer_dim}"
            )

    def _cropped_scenario(self, agent_f32):
        """Road elements within ROAD_CROP_MARGIN_M of the ego's driven positions (agent_f32 x/y are the road frame)."""
        ego_xy = agent_f32[:, 0, :2]
        lo = ego_xy.min(axis=0) - ROAD_CROP_MARGIN_M
        hi = ego_xy.max(axis=0) + ROAD_CROP_MARGIN_M
        roads = []
        for elem in self.scenario.get("road_elements", []) or []:
            xs, ys = np.asarray(elem.get("x") or [], np.float32), np.asarray(elem.get("y") or [], np.float32)
            if len(xs) and xs.max() >= lo[0] and xs.min() <= hi[0] and ys.max() >= lo[1] and ys.min() <= hi[1]:
                roads.append(elem)
        scenario = dict(self.scenario)
        scenario["road_elements"] = roads
        return scenario

    def write(self, render_html=True):
        """Save <stem>.replay.zlib (+ <stem>.html if render_html) -> the last path written, None if nothing captured."""
        from opengigaflow import viz

        frames = self.frames
        if not frames["obs"]:
            return None
        self._check_layout(frames["obs"][0].shape[-1])
        replay = {name: np.stack(values) for name, values in frames.items() if values}
        replay["env"] = self._env_cfg()
        replay["clipped_action"] = replay["raw_action"]
        if self.ghost_trajectory is not None:
            frame_count = replay["agent_f32"].shape[0]
            ghost = np.zeros((frame_count, replay["raw_action"].shape[1], 5), np.float32)
            ghost_count = min(frame_count, len(self.ghost_trajectory))
            ghost[:ghost_count, 0] = self.ghost_trajectory[:ghost_count]
            replay["ghost_f32"] = ghost
        self.out_stem.parent.mkdir(parents=True, exist_ok=True)
        zlib_path = str(self.out_stem) + ".replay.zlib"
        viz.save_interactive_replay_zlib(self._cropped_scenario(replay["agent_f32"]), replay, zlib_path)
        if not render_html:
            return zlib_path
        html_path = str(self.out_stem) + ".html"
        viz.render_interactive_replay_zlib(zlib_path, html_path)
        return html_path
