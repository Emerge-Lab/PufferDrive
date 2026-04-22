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
        _bg_policy_state_dict=None,  # preloaded state dict to avoid repeated disk I/O in workers
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

        obs_dim_expected = SIGNAL_OBS_DIM if per_lane_obs else SIGNAL_OBS_DIM_AGG
        print(
            f"[Traffic.__init__] map_dir={map_dir}  num_maps={num_maps}  "
            f"simulation_mode={simulation_mode}  n_tls_per_env={n_tls_per_env}  "
            f"obs_dim={obs_dim_expected} (per_lane={per_lane_obs})  "
            f"bg_agents={num_bg_agents}  bg_policy={'set' if bg_policy_path else 'None'}"
        )

        # ---- Inner Drive env (background vehicle simulation) ----
        # NOTE: Drive is always initialised with seed=0 so that binding.shared() produces
        # the same number of sub-envs on every worker.  Per-instance variety comes from the
        # seed passed to reset(), not from the init-time packing.
        print(f"[Traffic.__init__] Building inner Drive env ...")
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
            seed=0,   # fixed — keeps n_drive_envs identical across all workers
            width=width,
            height=height,
        )
        self.n_drive_envs = self.drive.num_envs
        # Capture the exact seed Drive used for its initial binding.shared() call so
        # every _resample_drive() uses the SAME seed → same num_envs / agent_offsets.
        # Drive.random_seed is consumed once by the constructor rng; save it now.
        self._drive_init_seed = self.drive.random_seed
        print(f"[Traffic.__init__] Drive env ready: num_envs={self.n_drive_envs}  drive_init_seed={self._drive_init_seed}")

        # Total RL agents = n_drive_envs × n_tls_per_env
        total_signals = self.n_drive_envs * n_tls_per_env
        self.num_agents = total_signals
        print(f"[Traffic.__init__] Total signal agents = {self.n_drive_envs} envs × {n_tls_per_env} TLs = {total_signals}")

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
            if _bg_policy_state_dict is not None:
                print(f"[Traffic.__init__] Using preloaded bg policy state dict (skipping disk I/O)")
                self._load_bg_policy(bg_policy_path, _state_dict=_bg_policy_state_dict)
            else:
                print(f"[Traffic.__init__] Loading background Drive policy from {bg_policy_path}")
                self._load_bg_policy(bg_policy_path)
            print(f"[Traffic.__init__] Background policy loaded OK")
        else:
            print(f"[Traffic.__init__] No bg_policy_path — background vehicles will act randomly")

        self.tick = 0
        print(f"[Traffic.__init__] Calling super().__init__ (allocates shared buffers) ...")
        super().__init__(buf=buf)
        print(f"[Traffic.__init__] Done. observations.shape={self.observations.shape}")

        # Episode-level stat accumulators (reset on truncation)
        self._ep_steps = 0
        self._ep_reward_sum = 0.0
        self._ep_queue_sum = 0.0
        self._ep_count_sum = 0.0
        self._ep_flicker_sum = 0.0
        self._ep_green_sum = 0.0
        self._prev_sig_act = None  # for flicker detection

    # ------------------------------------------------------------------
    # Background policy
    # ------------------------------------------------------------------

    def _get_obs(self):
        if self.per_lane_obs:
            binding.traffic_get_signal_observations(self.drive.c_envs, self._sig_obs, self.n_tls_per_env)
        else:
            binding.traffic_get_signal_observations_agg(self.drive.c_envs, self._sig_obs, self.n_tls_per_env)

    def _load_bg_policy(self, path, _state_dict=None):
        from pufferlib.ocean.torch import Drive as DriveNet

        if _state_dict is not None:
            state = _state_dict
        else:
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
        import time as _time
        _t = _time.time()
        print(f"[Traffic.reset] seed={seed}  n_tls_per_env={self.n_tls_per_env}")
        self.drive.reset(seed=seed)
        print(f"[Traffic.reset] drive.reset() done in {_time.time()-_t:.2f}s")

        # Override the random TL cycles with all-GREEN so RL starts clean
        binding.traffic_init_signal_states(self.drive.c_envs)

        # Auto-detect actual TL count from the loaded map
        detected = binding.traffic_count_signals(self.drive.c_envs)
        print(f"[Traffic.reset] traffic_count_signals → {detected} (expected {self.n_tls_per_env})")
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

        # Reset episode stat accumulators
        self._ep_steps = 0
        self._ep_reward_sum = 0.0
        self._ep_queue_sum = 0.0
        self._ep_count_sum = 0.0
        self._ep_flicker_sum = 0.0
        self._ep_green_sum = 0.0
        self._prev_sig_act = None

        # Populate initial signal observations
        self._get_obs()
        self.observations[:] = self._sig_obs
        print(
            f"[Traffic.reset] Done. num_agents={self.num_agents}  "
            f"obs shape={self.observations.shape}  obs range=[{self._sig_obs.min():.3f}, {self._sig_obs.max():.3f}]"
        )
        return self.observations, []

    def step(self, actions):
        if self.tick == 0:
            print(
                f"[Traffic.step] First step — actions.shape={np.asarray(actions).shape}  "
                f"n_tls={self.n_tls_per_env}  bg_policy={'loaded' if self._bg_policy else 'random'}"
            )
        # 1. Background vehicle actions ----------------------------------
        if self._bg_policy is not None:
            drive_obs_t = torch.from_numpy(self.drive.observations).float()
            with torch.no_grad():
                bg_atn_tuple, _ = self._bg_policy(drive_obs_t, self._bg_hidden)
            # bg_atn_tuple is a tuple of logit tensors; take argmax for each
            bg_actions = (
                torch.cat([t.argmax(dim=-1, keepdim=True) for t in bg_atn_tuple], dim=-1)
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

        # 5. Accumulate per-episode metrics ----------------------------
        #
        # Obs layout (per_lane=True, 61-dim):
        #   ego[0:5] | 8×lane[5:37] | 4×partner[37:61]
        #   lane slot: [count/10, avg_speed, queue/10, valid]
        #   → queue at indices 7, 11, 15, …, 35  (slice [7:36:4])
        #   → count at indices 5,  9, 13, …, 33  (slice [5:34:4])
        #   ego[2] = current state (0=RED, 0.5=YELLOW, 1=GREEN)
        _SIGNAL_MAX_QUEUE = 10.0
        if self.per_lane_obs:
            lane_queue = self._sig_obs[:, 7:36:4]   # (n_tls, 8) normalised
            lane_count = self._sig_obs[:, 5:34:4]   # (n_tls, 8) normalised
        else:
            # Aggregated layout (33-dim): ego[0:5] | agg[5:9] | partners[9:33]
            # agg slot: [total_count/10, overall_avg_speed, total_queue/10, n_valid/8]
            lane_queue = self._sig_obs[:, 7:8]      # (n_tls, 1) normalised
            lane_count = self._sig_obs[:, 5:6]      # (n_tls, 1) normalised

        queue_per_signal = lane_queue.sum(axis=1) * _SIGNAL_MAX_QUEUE   # approx queued vehicles
        count_per_signal = lane_count.sum(axis=1) * _SIGNAL_MAX_QUEUE   # approx total vehicles

        # Current state: 0=RED, 0.5=YELLOW, 1=GREEN
        green_frac = float((self._sig_obs[:, 2] >= 1.0).mean())

        # Flicker: fraction of signals whose action changed vs last step
        if self._prev_sig_act is not None:
            flicker = float((self._sig_act.reshape(-1) != self._prev_sig_act.reshape(-1)).mean())
        else:
            flicker = 0.0
        self._prev_sig_act = self._sig_act.copy()

        self._ep_reward_sum  += float(self._sig_rew.mean())
        self._ep_queue_sum   += float(queue_per_signal.mean())
        self._ep_count_sum   += float(count_per_signal.mean())
        self._ep_flicker_sum += flicker
        self._ep_green_sum   += green_frac
        self._ep_steps       += 1

        # 6. Terminals / truncations ------------------------------------
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

            # Emit episode stats — picked up by pufferl stats collector → wandb
            n = max(self._ep_steps, 1)
            info.append({
                "traffic/mean_reward":    self._ep_reward_sum  / n,
                "traffic/mean_queue":     self._ep_queue_sum   / n,   # avg queued vehicles/signal
                "traffic/mean_vehicles":  self._ep_count_sum   / n,   # avg vehicles/signal
                "traffic/flicker_rate":   self._ep_flicker_sum / n,   # fraction changing state/step
                "traffic/green_fraction": self._ep_green_sum   / n,   # fraction of signals green
                "traffic/episode_length": float(self._ep_steps),
            })

            # Reset accumulators for next episode
            self._ep_steps = 0
            self._ep_reward_sum = self._ep_queue_sum = 0.0
            self._ep_count_sum  = self._ep_flicker_sum = self._ep_green_sum = 0.0
            self._prev_sig_act = None

        return self.observations, self.rewards, self.terminals, self.truncations, info

    # ------------------------------------------------------------------
    # Map resample (mirrors Drive.step resample block)
    # ------------------------------------------------------------------

    def _resample_drive(self):
        binding.vec_close(self.drive.c_envs)

        # Use the SAME seed that Drive.__init__ passed to binding.shared() so the
        # packing (num_envs, agent_offsets) is always identical to what was used
        # when the numpy observation/action buffers were first allocated.
        # A different seed here can give a different num_envs, making _sig_act
        # too small and causing traffic_set_signal_actions to write past its end
        # → glibc heap corruption ("free(): invalid next size").
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
            seed=self._drive_init_seed,  # must match Drive.__init__'s binding.shared() seed
            min_agents_per_env=self.drive.min_agents_per_env,
            max_agents_per_env=self.drive.max_agents_per_env,
            num_eval_scenarios=self.drive.current_num_eval_scenarios,
            goal_radius=self.drive.goal_radius,
            road_obs_front_dist=self.drive.road_obs_front_dist,
            road_obs_behind_dist=self.drive.road_obs_behind_dist,
            road_obs_side_dist=self.drive.road_obs_side_dist,
        )
        self.drive.starting_map_counter += num_envs

        # Safety: if num_envs changed (different seed → different packing),
        # reallocate signal buffers so traffic_set_signal_actions never writes
        # past the end of _sig_act → no heap corruption.
        if num_envs != self.n_drive_envs:
            print(
                f"[Traffic._resample_drive] WARNING: num_envs changed "
                f"{self.n_drive_envs} → {num_envs}; reallocating signal buffers"
            )
            self.n_drive_envs = num_envs
            total = num_envs * self.n_tls_per_env
            self.num_agents = total
            obs_dim = SIGNAL_OBS_DIM if self.per_lane_obs else SIGNAL_OBS_DIM_AGG
            self._sig_obs = np.zeros((total, obs_dim), dtype=np.float32)
            self._sig_rew = np.zeros(total, dtype=np.float32)
            self._sig_act = np.zeros(total, dtype=np.int32)
            # Resize PufferEnv shared buffers too
            self.observations = np.zeros((total, obs_dim), dtype=np.float32)
            self.rewards       = np.zeros(total, dtype=np.float32)
            self.terminals     = np.zeros(total, dtype=bool)
            self.truncations   = np.zeros(total, dtype=bool)
            self.masks         = np.ones(total,  dtype=bool)

        # Also guard against agent_offsets overrunning the Drive observation buffer.
        drive_buf_size = self.drive.observations.shape[0]  # always num_bg_agents (e.g. 1024)
        assert agent_offsets[-1] <= drive_buf_size, (
            f"_resample_drive: agent_offsets[-1]={agent_offsets[-1]} > "
            f"drive buffer size={drive_buf_size}. "
            f"Increase num_bg_agents or fix seed."
        )

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
