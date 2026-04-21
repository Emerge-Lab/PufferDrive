import os
import numpy as np
import torch
import gymnasium
import pufferlib

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive

# Observation and action constants (mirrored from traffic.h)
SIGNAL_OBS_DIM = 61  # per_lane=True:  5 + 8*4 + 4*6
SIGNAL_OBS_DIM_AGG = 33  # per_lane=False: 5 + 4   + 4*6
SIGNAL_N_ACTIONS = 3  # 0=RED, 1=YELLOW, 2=GREEN


class Traffic(pufferlib.PufferEnv):
    """PufferTraffic — RL environment for traffic signal control.

    One RL agent per traffic light; shared policy parameters across all signals.
    Background vehicles are driven by a frozen pretrained Drive policy (or random
    if no checkpoint path is provided).

    Reward (IntelliLight-inspired):
        + reward_throughput * vehicles crossing the stop-line at decent speed
        - reward_queue      * vehicles stopped upstream of the stop-line
        - reward_flicker    * 1 if the signal state changed this step
    """

    def __init__(
        self,
        # --- Map / simulation ---
        map_dir="pufferlib/resources/drive/binaries/carla_py123d",
        num_maps=1,
        scenario_length=910,
        resample_frequency=910,
        traffic_light_behavior=1,
        simulation_mode="gigaflow",
        # --- Background vehicles ---
        num_bg_agents=256,
        min_agents_per_env=16,
        max_agents_per_env=64,
        bg_policy_path=None,
        # --- Signal agents ---
        n_tls_per_env=36,  # Town01 default; auto-corrected after first reset
        per_lane_obs=True,  # True=61-dim per-lane; False=33-dim aggregated
        # --- Rewards ---
        reward_throughput=0.5,
        reward_queue=0.25,
        reward_flicker=1.0,
        # --- Misc ---
        render_mode=None,
        report_interval=1,
        seed=1,
        buf=None,
        width=1280,
        height=1024,
    ):
        self.reward_throughput = float(reward_throughput)
        self.reward_queue = float(reward_queue)
        self.reward_flicker = float(reward_flicker)
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.n_tls_per_env = n_tls_per_env
        self.per_lane_obs = bool(per_lane_obs)
        self.rng = np.random.default_rng(seed)

        # ---- Inner Drive env (background vehicle simulation) ----
        self.drive = Drive(
            render_mode=render_mode,
            map_dir=map_dir,
            num_agents=num_bg_agents,
            num_maps=num_maps,
            min_agents_per_env=min_agents_per_env,
            max_agents_per_env=max_agents_per_env,
            resample_frequency=resample_frequency,
            scenario_length=scenario_length,
            traffic_light_behavior=traffic_light_behavior,
            simulation_mode=simulation_mode,
            seed=seed,
            width=width,
            height=height,
        )
        self.n_drive_envs = self.drive.num_envs

        # Total RL agents = n_drive_envs × n_tls_per_env
        total_signals = self.n_drive_envs * n_tls_per_env
        self.num_agents = total_signals

        # PufferEnv spaces (signal agents)
        obs_dim = SIGNAL_OBS_DIM if self.per_lane_obs else SIGNAL_OBS_DIM_AGG
        self.single_observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        # MultiDiscrete([3]) matches Drive's convention and gives nvec=[3]
        self.single_action_space = gymnasium.spaces.MultiDiscrete([SIGNAL_N_ACTIONS])

        # Working buffers for signal obs/rew/actions
        self._sig_obs = np.zeros((total_signals, obs_dim), dtype=np.float32)
        self._sig_rew = np.zeros(total_signals, dtype=np.float32)
        self._sig_act = np.zeros(total_signals, dtype=np.int32)

        # Frozen background policy (optional)
        self._bg_policy = None
        self._bg_hidden = None
        if bg_policy_path is not None:
            self._load_bg_policy(bg_policy_path)

        self.tick = 0
        super().__init__(buf=buf)

    # ------------------------------------------------------------------
    # Background policy
    # ------------------------------------------------------------------

    def _get_obs(self):
        if self.per_lane_obs:
            binding.traffic_get_signal_observations(self.drive.c_envs, self._sig_obs, self.n_tls_per_env)
        else:
            binding.traffic_get_signal_observations_agg(self.drive.c_envs, self._sig_obs, self.n_tls_per_env)

    def _load_bg_policy(self, path):
        from pufferlib.ocean.torch import Drive as DriveNet

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("policy_state_dict", ckpt.get("model_state_dict", ckpt))
        # Build a minimal Drive net matching the stored checkpoint dims
        self._bg_policy = DriveNet(
            env=self.drive,
            input_size=64,
            backbone_hidden_size=512,
            backbone_num_layers=4,
            actor_hidden_size=512,
            actor_num_layers=0,
            critic_hidden_size=512,
            critic_num_layers=0,
            encoder_gigaflow=True,
            dropout=0.0,
            split_network=False,
        )
        self._bg_policy.load_state_dict(state, strict=False)
        self._bg_policy.eval()

    # ------------------------------------------------------------------
    # PufferEnv interface
    # ------------------------------------------------------------------

    @property
    def random_seed(self):
        return int(self.rng.integers(0, 2**24))

    def reset(self, seed=0):
        self.drive.reset(seed=seed)

        # Override the random TL cycles with all-GREEN so RL starts clean
        binding.traffic_init_signal_states(self.drive.c_envs)

        # Auto-detect actual TL count from the loaded map
        detected = binding.traffic_count_signals(self.drive.c_envs)
        if detected != self.n_tls_per_env:
            self.n_tls_per_env = detected
            total = self.n_drive_envs * detected
            self.num_agents = total
            self._sig_obs = np.zeros((total, SIGNAL_OBS_DIM), dtype=np.float32)
            self._sig_rew = np.zeros(total, dtype=np.float32)
            self._sig_act = np.zeros(total, dtype=np.int32)

        self._bg_hidden = None
        self.tick = 0
        self.truncations[:] = 0

        # Populate initial signal observations
        self._get_obs()
        self.observations[:] = self._sig_obs
        return self.observations, []

    def step(self, actions):
        # 1. Background vehicle actions ----------------------------------
        if self._bg_policy is not None:
            drive_obs_t = torch.from_numpy(self.drive.observations).float()
            with torch.no_grad():
                bg_atn_tuple, _ = self._bg_policy(drive_obs_t, self._bg_hidden)
            # bg_atn_tuple is a tuple of logit tensors; take argmax for each
            bg_actions = (
                torch.cat([t.argmax(dim=-1, keepdim=True) for t in bg_atn_tuple], dim=-1)
                .squeeze(-1)
                .numpy()
                .astype(np.float32)
            )
            self.drive.actions[:] = bg_actions
        else:
            # Random background driving
            space = self.drive.single_action_space
            self.drive.actions[:] = np.random.randint(0, space.nvec[0], size=self.drive.actions.shape).astype(
                np.float32
            )

        # 2. Apply RL signal actions BEFORE vec_step ---------------------
        #    C reads states[timestep] after incrementing timestep, so we
        #    write to states[timestep+1] here, then vec_step increments.
        self._sig_act[:] = np.asarray(actions, dtype=np.int32).reshape(-1)
        binding.traffic_set_signal_actions(self.drive.c_envs, self._sig_act, self.n_tls_per_env)

        # 3. Advance the Drive simulation --------------------------------
        binding.vec_step(self.drive.c_envs)
        self.drive.tick += 1
        self.tick += 1

        # 4. Collect signal observations and rewards ---------------------
        self._get_obs()
        binding.traffic_get_signal_rewards(
            self.drive.c_envs,
            self._sig_rew,
            self.n_tls_per_env,
            self.reward_throughput,
            self.reward_queue,
            self.reward_flicker,
        )
        self.observations[:] = self._sig_obs
        self.rewards[:] = self._sig_rew

        # 5. Terminals / truncations ------------------------------------
        #    Signal agents are never individually terminal; they truncate
        #    together with Drive's episode boundary (map resample).
        self.terminals[:] = 0
        self.truncations[:] = 0

        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.drive.c_envs, self.drive.num_agents)
            if log:
                info.append(log)

        # Handle Drive map resample (same logic as Drive.step)
        if self.drive.resample_frequency > 0 and self.drive.tick % self.drive.resample_frequency == 0:
            self.drive.tick = 0
            self._resample_drive()
            self.truncations[:] = 1

        return self.observations, self.rewards, self.terminals, self.truncations, info

    # ------------------------------------------------------------------
    # Map resample (mirrors Drive.step resample block)
    # ------------------------------------------------------------------

    def _resample_drive(self):
        binding.vec_close(self.drive.c_envs)

        agent_offsets, map_ids, num_envs = binding.shared(
            num_agents=self.drive.num_agents,
            num_maps=self.drive.num_maps,
            starting_map_counter=self.drive.starting_map_counter,
            eval_mode=self.drive.eval_mode,
            init_mode=self.drive.init_mode,
            control_mode=self.drive.control_mode,
            simulation_mode=self.drive.simulation_mode,
            init_steps=self.drive.init_steps,
            map_files=self.drive.map_files,
            seed=self.random_seed,
            min_agents_per_env=self.drive.min_agents_per_env,
            max_agents_per_env=self.drive.max_agents_per_env,
            num_eval_scenarios=self.drive.current_num_eval_scenarios,
            road_obs_front_dist=self.drive.road_obs_front_dist,
            road_obs_behind_dist=self.drive.road_obs_behind_dist,
            road_obs_side_dist=self.drive.road_obs_side_dist,
        )
        self.drive.starting_map_counter += num_envs

        env_ids = []
        for i in range(num_envs):
            cur, nxt = agent_offsets[i], agent_offsets[i + 1]
            eid = binding.env_init(
                self.drive.observations[cur:nxt],
                self.drive.actions[cur:nxt],
                self.drive.rewards[cur:nxt],
                self.drive.terminals[cur:nxt],
                self.drive.truncations[cur:nxt],
                self.drive.masks[cur:nxt],
                self.random_seed,
                map_id=map_ids[i],
                **self.drive._env_init_kwargs(self.drive.map_files[map_ids[i]], nxt - cur),
            )
            env_ids.append(eid)

        self.drive.c_envs = binding.vectorize(*env_ids)
        binding.vec_reset(self.drive.c_envs, self.random_seed)
        binding.traffic_init_signal_states(self.drive.c_envs)
        self._bg_hidden = None

    # ------------------------------------------------------------------
    # Render / close
    # ------------------------------------------------------------------

    def render(self, view_mode=0, env_id=0):
        self.drive.render(view_mode=view_mode, env_id=env_id)

    def set_video_suffix(self, suffix, env_id=0):
        self.drive.set_video_suffix(suffix, env_id=env_id)

    def close_client(self, env_id=0):
        self.drive.close_client(env_id=env_id)

    def get_state(self):
        return self.drive.get_state()

    def close(self):
        self.drive.close()
