"""Interactive observation replay (pufferlib.viz) for a co-sim shadow env: the exact obs vector the
policy received each step, its outputs (action, value, entropy, action probabilities) and the
encoder max-pool winners, rendered as one self-contained HTML per scenario/route."""

import json
from pathlib import Path

import numpy as np

from pufferlib.ocean.drive import binding

OBS_CLIP = 100.0  # parked FAR_AWAY slots carry ~5e3 goal offsets that would blow up the viewer's int16 quantization
ROAD_CROP_MARGIN_M = 250.0  # whole-city bins: keep only road elements near the driven trajectory
POOL_NAMES = ("pool_partner", "pool_lane", "pool_boundary", "pool_traffic")
OBS_TRAILING_COUNT_FEATURES = 4  # lane/boundary/partner/traffic-control counts appended after the slots


class ObsReplayCapture:
    def __init__(self, env, policy, out_stem, max_steps=800):
        self.env = env
        self.policy = policy
        self.out_stem = Path(out_stem)
        self.max_steps = int(max_steps)
        state = env.get_state()
        self.scenario = state[0] if isinstance(state, list) else state
        self.agent_cap = int(self.scenario["num_total_agents"])
        self.traffic_cap = max(int(self.scenario["num_traffic_elements"]), 1)
        self.frames = {
            key: []
            for key in (
                "agent_f32",
                "agent_i32",
                "metrics_f32",
                "puffer_f32",
                "traffic_i16",
                "obs",
                "raw_action",
                "action_index",
                "value",
                "entropy",
                "policy_probs",
            )
        }

    def __len__(self):
        return len(self.frames["obs"])

    def policy_outputs(self, obs_tensor, logits, value, entropy):
        """-> aux dict for capture(): value/entropy/action probabilities (+ pool winners when the policy exposes them)."""
        import torch

        probs = torch.softmax(logits if isinstance(logits, torch.Tensor) else logits[0], dim=-1)
        aux = {
            "value": value.detach().cpu().numpy().reshape(-1).astype(np.float32),
            "entropy": entropy.detach().cpu().numpy().reshape(-1).astype(np.float32),
            "policy_probs": probs.detach().cpu().numpy().astype(np.float32),
        }
        pool_method = getattr(self.policy, "pool_slot_counts", None)
        if pool_method is not None:
            with torch.no_grad():
                pool = pool_method(obs_tensor)
            aux["pool"] = {name: counts.cpu().numpy().astype(np.int16) for name, counts in pool.items()}
        return aux

    def capture(self, obs, actions, aux, action_index=None):
        """action_index: discrete class per agent behind `actions` (argmax when the executed action is the
        distribution mean, the sampled/mode class otherwise); None or -1 when there is no class."""
        if len(self) >= self.max_steps:
            return
        cap, tcap = self.agent_cap, self.traffic_cap
        agent_f32 = np.zeros((1, cap, binding.AGENT_F32_FIELDS), np.float32)
        agent_i32 = np.zeros((1, cap, binding.AGENT_I32_FIELDS), np.int32)
        metrics_f32 = np.zeros((1, cap, binding.METRICS_F32_FIELDS), np.float32)
        puffer_f32 = np.zeros((1, cap, binding.SCORE_F32_FIELDS), np.float32)
        traffic_i16 = np.zeros((1, tcap, binding.TRAFFIC_I16_FIELDS), np.int16)
        rewards_f32 = np.zeros((1, cap, binding.REWARD_F32_FIELDS), np.float32)
        self.env.get_obs_html_frame(agent_f32, agent_i32, metrics_f32, puffer_f32, traffic_i16, rewards_f32)
        frames = self.frames
        frames["agent_f32"].append(agent_f32[0])
        frames["agent_i32"].append(agent_i32[0])
        frames["metrics_f32"].append(metrics_f32[0])
        frames["puffer_f32"].append(puffer_f32[0])
        frames["traffic_i16"].append(traffic_i16[0])
        obs = np.asarray(obs, dtype=np.float32)
        frames["obs"].append(np.clip(obs, -OBS_CLIP, OBS_CLIP))
        frames["raw_action"].append(np.asarray(actions, dtype=np.float32).reshape(obs.shape[0], -1))
        index = np.full(obs.shape[0], -1, np.int32) if action_index is None else np.asarray(action_index, np.int32).reshape(obs.shape[0])
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
            "init_step": 0,
            "goal_regen_mode": "finite",
            "action_type": "discrete",
            "dynamics_model": env.dynamics_model,
            "num_goals": int(env.num_goals),
            "reward_conditioning": bool(env.num_reward_coefs),
            "obs_slots_partners_n": int(env.obs_slots_partners_n),
            "obs_slots_lane_n": int(env.obs_slots_lane_n),
            "obs_slots_boundary_n": int(env.obs_slots_boundary_n),
            "obs_lane_stride": int(env.obs_lane_stride),
            "obs_boundary_stride": int(env.obs_boundary_stride),
            "obs_slots_traffic_controls_n": int(env.obs_slots_traffic_controls_n),
            "obs_dropout_lane": float(env.obs_dropout_lane),
            "obs_dropout_boundary": float(env.obs_dropout_boundary),
            "obs_norm_goal_offset_m": float(env.obs_norm_goal_offset_m),
            "obs_norm_xy_offset_m": float(env.obs_norm_xy_offset_m),
            "obs_norm_veh_width_m": float(env.obs_norm_veh_width_m),
            "obs_norm_veh_length_m": float(env.obs_norm_veh_length_m),
            "obs_norm_road_seg_length_m": float(env.obs_norm_road_seg_length_m),
            "obs_norm_road_seg_width_m": float(env.obs_norm_road_seg_width_m),
            "obs_partner_relative_velocity": bool(env.obs_partner_relative_velocity),
            "obs_goal_lane_distance": bool(env.obs_goal_lane_distance),
            "ego_features": int(env.ego_features),
            "num_reward_coefs": int(env.num_reward_coefs),
            "goal_dim": int(env.goal_dim),
            "partner_features": int(env.partner_features),
            "lane_features": int(env.lane_features),
            "boundary_features": int(env.boundary_features),
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
            + env.obs_slots_traffic_controls_n * binding.TRAFFIC_CONTROL_FEATURES
        )
        if obs_dim < viewer_dim or obs_dim - viewer_dim > OBS_TRAILING_COUNT_FEATURES:
            raise RuntimeError(f"obs replay layout mismatch: obs has {obs_dim} columns, viewer layout expects {viewer_dim}")

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
        scenario["map_corners"] = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
        return scenario

    def write(self, render_html=True, save_npz=False):
        """Save <stem>.replay.zlib (+ .html when render_html, + .npz raw arrays when save_npz).
        -> path of the html (or the zlib when not rendered), None when nothing was captured.
        Deferred rendering: `render_replay_html(zlib_path)` turns a saved replay into the page later,
        e.g. only for scenarios that scored badly."""
        from pufferlib import viz

        frames = self.frames
        if not frames["obs"]:
            return None
        env_cfg = self._env_cfg()
        self._check_layout(frames["obs"][0].shape[-1])
        agent_f32 = np.stack(frames["agent_f32"])
        replay = {
            "schema": "obs_html_compact_v1",
            "env": env_cfg,
            "agent_f32": agent_f32,
            "agent_i32": np.stack(frames["agent_i32"]),
            "metrics_f32": np.stack(frames["metrics_f32"]),
            "puffer_f32": np.stack(frames["puffer_f32"]),
            "traffic_i16": np.stack(frames["traffic_i16"]),
            "obs": np.stack(frames["obs"]),
            "raw_action": np.stack(frames["raw_action"]),
            "clipped_action": np.stack(frames["raw_action"]),
            "action_index": np.stack(frames["action_index"]),
            "value": np.stack(frames["value"]),
            "entropy": np.stack(frames["entropy"]),
            "policy_probs": np.stack(frames["policy_probs"]) if frames["policy_probs"] else None,
            "policy_mean": None,
            "policy_std": None,
            "policy_log_prob": None,
        }
        for pool_name in POOL_NAMES:
            if frames.get(pool_name):
                replay[pool_name] = np.stack(frames[pool_name])
        self.out_stem.parent.mkdir(parents=True, exist_ok=True)
        zlib_path = str(self.out_stem) + ".replay.zlib"
        html_path = str(self.out_stem) + ".html"
        viz.save_interactive_replay_zlib(self._cropped_scenario(agent_f32), replay, zlib_path)
        if save_npz:
            np.savez(str(self.out_stem) + ".npz", obs=replay["obs"], agent_f32=agent_f32, env_cfg_json=np.array(json.dumps(env_cfg)))
        if not render_html:
            return zlib_path
        render_replay_html(zlib_path)
        return html_path


def render_replay_html(zlib_path):
    """<stem>.replay.zlib -> <stem>.html"""
    from pufferlib import viz

    html_path = str(zlib_path)[: -len(".replay.zlib")] + ".html"
    viz.render_interactive_replay_zlib(str(zlib_path), html_path)
    return html_path
