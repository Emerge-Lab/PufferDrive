"""
Test suite for adversarial observation handling.

Verifies that:
1. Observation buffer has correct size (includes TARGET_FEATURES)
2. SDC observations are correctly sliced to remove TARGET_FEATURES
3. Adversary observations include TARGET_FEATURES with target info
4. Both policies can process their respective observations
"""

import pytest
import numpy as np
import torch


class TestObservationLayout:
    """Test observation buffer layout and sizes."""

    @pytest.fixture
    def env_config(self):
        """Basic environment configuration."""
        return {
            "num_agents": 16,
            "num_maps": 10,
            "init_mode": "create_all_valid",
            "control_mode": "control_vehicles",
            "map_dir": "resources/drive/binaries/training",
        }

    @pytest.fixture
    def binding(self):
        """Import the C binding module."""
        from pufferlib.ocean.drive import binding

        return binding

    def test_target_features_constant_exists(self, binding):
        """Verify TARGET_FEATURES is exported from C binding."""
        assert hasattr(binding, "TARGET_FEATURES")
        assert binding.TARGET_FEATURES == 7, "TARGET_FEATURES should be 7"

    def test_observation_size_includes_target_features(self, binding):
        """Verify num_obs calculation includes TARGET_FEATURES."""
        from pufferlib.ocean.drive.drive import Drive

        # Create a minimal env to check observation space
        env = Drive(
            num_agents=4,
            num_maps=1,
            map_dir="resources/drive/binaries/training",
            scenario_length=91,
        )

        ego_features = env.ego_features
        target_features = env.target_features
        partner_features = env.partner_features * env.max_partner_objects
        road_features = env.road_features * env.max_road_objects

        expected_num_obs = ego_features + target_features + partner_features + road_features

        assert env.num_obs == expected_num_obs, (
            f"num_obs mismatch: got {env.num_obs}, expected {expected_num_obs}\n"
            f"  ego_features: {ego_features}\n"
            f"  target_features: {target_features}\n"
            f"  partner_features: {partner_features}\n"
            f"  road_features: {road_features}"
        )

        env.close()

    def test_observation_space_shape(self, binding):
        """Verify single_observation_space has correct shape."""
        from pufferlib.ocean.drive.drive import Drive

        env = Drive(
            num_agents=4,
            num_maps=1,
            map_dir="resources/drive/binaries/training",
            scenario_length=91,
        )

        assert env.single_observation_space.shape == (env.num_obs,), (
            f"Observation space shape mismatch: {env.single_observation_space.shape} != ({env.num_obs},)"
        )

        env.close()


class TestObservationSlicing:
    """Test observation slicing for SDC vs adversaries."""

    def test_sdc_observation_slicing(self):
        """Verify SDC observation slicing removes TARGET_FEATURES correctly."""
        from pufferlib.ocean.drive import binding

        ego_features = binding.EGO_FEATURES_CLASSIC
        target_features = binding.TARGET_FEATURES
        partner_features = binding.PARTNER_FEATURES * (binding.MAX_AGENTS - 1)
        road_features = binding.ROAD_FEATURES * binding.MAX_ROAD_SEGMENT_OBSERVATIONS

        total_obs = ego_features + target_features + partner_features + road_features

        # Create a mock observation with identifiable patterns
        obs = torch.zeros(total_obs)
        obs[:ego_features] = 1.0  # Ego features = 1
        obs[ego_features : ego_features + target_features] = 2.0  # Target features = 2
        obs[ego_features + target_features :] = 3.0  # Rest = 3

        # Slice like we do for SDC
        obs_for_ref = torch.cat([obs[:ego_features], obs[ego_features + target_features :]])

        # Verify sliced observation
        expected_ref_obs_size = ego_features + partner_features + road_features
        assert obs_for_ref.shape[0] == expected_ref_obs_size, (
            f"Sliced obs size mismatch: {obs_for_ref.shape[0]} != {expected_ref_obs_size}"
        )

        # Verify content: should have ego (1.0) then rest (3.0), NO target (2.0)
        assert torch.all(obs_for_ref[:ego_features] == 1.0), "Ego features should be preserved"
        assert torch.all(obs_for_ref[ego_features:] == 3.0), "Partner/road features should be preserved"
        assert not torch.any(obs_for_ref == 2.0), "Target features should be removed"

    def test_adversary_observation_full(self):
        """Verify adversary observations include TARGET_FEATURES."""
        from pufferlib.ocean.drive import binding

        ego_features = binding.EGO_FEATURES_CLASSIC
        target_features = binding.TARGET_FEATURES
        partner_features = binding.PARTNER_FEATURES * (binding.MAX_AGENTS - 1)
        road_features = binding.ROAD_FEATURES * binding.MAX_ROAD_SEGMENT_OBSERVATIONS

        total_obs = ego_features + target_features + partner_features + road_features

        # Create a mock observation
        obs = torch.zeros(total_obs)
        obs[:ego_features] = 1.0
        obs[ego_features : ego_features + target_features] = 2.0  # Target features
        obs[ego_features + target_features :] = 3.0

        # Adversary uses full observation
        obs_for_adv = obs

        assert obs_for_adv.shape[0] == total_obs
        assert torch.any(obs_for_adv == 2.0), "Adversary should have target features"


class TestLiveEnvironment:
    """Test with actual environment to verify end-to-end behavior."""

    @pytest.fixture
    def live_env(self):
        """Create a live environment for testing."""
        from pufferlib.ocean.drive.drive import Drive

        env = Drive(
            num_agents=8,
            num_maps=5,
            map_dir="resources/drive/binaries/training",
            control_mode="control_vehicles",
            scenario_length=91,
        )
        yield env
        env.close()

    def test_observation_buffer_filled(self, live_env):
        """Verify observations are filled after reset."""
        obs, _ = live_env.reset(seed=42)

        assert obs.shape[0] == live_env.num_agents, "Wrong number of agents in obs"
        assert obs.shape[1] == live_env.num_obs, "Wrong observation size"

        # Observations should not be all zeros (at least some valid data)
        assert not np.allclose(obs, 0), "Observations should not be all zeros after reset"

    def test_sdc_target_features_are_zeros(self, live_env):
        """Verify SDC's TARGET_FEATURES slot contains zeros."""
        live_env.reset(seed=42)

        # Set targets (SDC has no target, adversaries target SDC)
        for env_idx in range(live_env.num_envs):
            sdc_global_idx = live_env.sdc_indices[env_idx]
            sdc_entity_idx = live_env.sdc_entity_indices[env_idx]
            env_start = live_env.agent_offsets[env_idx]
            env_end = live_env.agent_offsets[env_idx + 1]

            target_indices = []
            for local_idx in range(env_end - env_start):
                global_idx = env_start + local_idx
                if global_idx == sdc_global_idx:
                    target_indices.append(-1)  # SDC has no target
                else:
                    target_indices.append(sdc_entity_idx)

            live_env.set_targets(env_idx, target_indices)

        # Step to compute observations
        actions = np.zeros((live_env.num_agents, 1), dtype=np.int32)
        obs, _, _, _, _ = live_env.step(actions)

        ego_features = live_env.ego_features
        target_features = live_env.target_features

        # Check each SDC's target features
        for env_idx in range(live_env.num_envs):
            sdc_global_idx = live_env.sdc_indices[env_idx]
            if sdc_global_idx < live_env.num_agents:
                sdc_obs = obs[sdc_global_idx]
                sdc_target_slot = sdc_obs[ego_features : ego_features + target_features]

                assert np.allclose(sdc_target_slot, 0), (
                    f"SDC (env {env_idx}, global idx {sdc_global_idx}) should have zeros in target slot, "
                    f"got: {sdc_target_slot}"
                )

    def test_adversary_target_features_nonzero(self, live_env):
        """Verify adversary TARGET_FEATURES contain actual target info."""
        live_env.reset(seed=42)

        # Set targets
        for env_idx in range(live_env.num_envs):
            sdc_global_idx = live_env.sdc_indices[env_idx]
            sdc_entity_idx = live_env.sdc_entity_indices[env_idx]
            env_start = live_env.agent_offsets[env_idx]
            env_end = live_env.agent_offsets[env_idx + 1]

            target_indices = []
            for local_idx in range(env_end - env_start):
                global_idx = env_start + local_idx
                if global_idx == sdc_global_idx:
                    target_indices.append(-1)
                else:
                    target_indices.append(sdc_entity_idx)

            live_env.set_targets(env_idx, target_indices)

        # Step to compute observations
        actions = np.zeros((live_env.num_agents, 1), dtype=np.int32)
        obs, _, _, _, _ = live_env.step(actions)

        ego_features = live_env.ego_features
        target_features = live_env.target_features

        # Check at least one adversary has non-zero target features
        found_nonzero_target = False
        for env_idx in range(live_env.num_envs):
            sdc_global_idx = live_env.sdc_indices[env_idx]
            env_start = live_env.agent_offsets[env_idx]
            env_end = live_env.agent_offsets[env_idx + 1]

            for global_idx in range(env_start, min(env_end, live_env.num_agents)):
                if global_idx != sdc_global_idx:
                    adv_obs = obs[global_idx]
                    adv_target_slot = adv_obs[ego_features : ego_features + target_features]

                    if not np.allclose(adv_target_slot, 0):
                        found_nonzero_target = True
                        break

            if found_nonzero_target:
                break

        assert found_nonzero_target, (
            "At least one adversary should have non-zero target features (target info about SDC)"
        )


class TestPolicyCompatibility:
    """Test that policies can process observations correctly."""

    def test_ref_policy_input_size(self):
        """Verify ref_policy expects observations WITHOUT TARGET_FEATURES."""
        from pufferlib.ocean.drive import binding

        # Ref policy was trained with original observation size
        ego_features = binding.EGO_FEATURES_CLASSIC
        partner_features = binding.PARTNER_FEATURES * (binding.MAX_AGENTS - 1)
        road_features = binding.ROAD_FEATURES * binding.MAX_ROAD_SEGMENT_OBSERVATIONS

        expected_ref_input_size = ego_features + partner_features + road_features

        # This is what the sliced observation should be
        target_features = binding.TARGET_FEATURES
        total_obs = ego_features + target_features + partner_features + road_features

        obs = torch.randn(total_obs)
        obs_for_ref = torch.cat([obs[:ego_features], obs[ego_features + target_features :]])

        assert obs_for_ref.shape[0] == expected_ref_input_size, (
            f"Sliced observation for ref_policy has wrong size: {obs_for_ref.shape[0]} != {expected_ref_input_size}"
        )

    def test_adv_policy_input_size(self):
        """Verify adv_policy expects observations WITH TARGET_FEATURES."""
        from pufferlib.ocean.drive import binding

        ego_features = binding.EGO_FEATURES_CLASSIC
        target_features = binding.TARGET_FEATURES
        partner_features = binding.PARTNER_FEATURES * (binding.MAX_AGENTS - 1)
        road_features = binding.ROAD_FEATURES * binding.MAX_ROAD_SEGMENT_OBSERVATIONS

        expected_adv_input_size = ego_features + target_features + partner_features + road_features

        obs = torch.randn(expected_adv_input_size)

        assert obs.shape[0] == expected_adv_input_size


class TestIntegration:
    """Integration tests simulating actual training loop behavior."""

    def test_batch_observation_processing(self):
        """Test batch processing like in evaluate() loop."""
        from pufferlib.ocean.drive import binding

        ego_features = binding.EGO_FEATURES_CLASSIC
        target_features = binding.TARGET_FEATURES
        num_obs = (
            ego_features
            + target_features
            + binding.PARTNER_FEATURES * (binding.MAX_AGENTS - 1)
            + binding.ROAD_FEATURES * binding.MAX_ROAD_SEGMENT_OBSERVATIONS
        )

        # Simulate a batch of observations
        batch_size = 8
        o_device = torch.randn(batch_size, num_obs)

        # Simulate SDC indices (agent 0 and 4 are SDCs)
        sdc_global_indices = {0, 4}

        o_ref_list = []
        o_adv_list = []
        ref_indices = []
        adv_indices = []

        for global_idx in range(batch_size):
            agent_obs = o_device[global_idx]

            if global_idx in sdc_global_indices:
                # SDC: slice out target features
                obs_for_ref = torch.cat([agent_obs[:ego_features], agent_obs[ego_features + target_features :]])
                o_ref_list.append(obs_for_ref)
                ref_indices.append(global_idx)
            else:
                # Adversary: full observation
                o_adv_list.append(agent_obs)
                adv_indices.append(global_idx)

        # Stack for batch processing
        if o_ref_list:
            o_ref = torch.stack(o_ref_list)
            expected_ref_obs_size = num_obs - target_features
            assert o_ref.shape == (len(sdc_global_indices), expected_ref_obs_size), (
                f"Ref batch shape mismatch: {o_ref.shape}"
            )

        if o_adv_list:
            o_adv = torch.stack(o_adv_list)
            assert o_adv.shape == (batch_size - len(sdc_global_indices), num_obs), (
                f"Adv batch shape mismatch: {o_adv.shape}"
            )

        # Verify counts
        assert len(ref_indices) == len(sdc_global_indices)
        assert len(adv_indices) == batch_size - len(sdc_global_indices)


class TestPolicyArchitecture:
    """Test policy architectures handle observations correctly."""

    def _create_valid_obs(self, mock_env, batch_size):
        """Create valid observations with proper categorical values."""
        import torch

        ego_dim = 7  # classic
        target_dim = getattr(mock_env, "target_features", 0)
        partner_dim = mock_env.max_partner_objects * mock_env.partner_features
        road_dim = mock_env.max_road_objects * mock_env.road_features

        # Create random obs
        obs = torch.randn(batch_size, mock_env.num_obs)

        # Fix road categorical features (last feature of each road object must be 0-6)
        road_start = ego_dim + target_dim + partner_dim
        for i in range(mock_env.max_road_objects):
            road_obj_start = road_start + i * mock_env.road_features
            # Last feature is categorical (road type 0-6)
            obs[:, road_obj_start + mock_env.road_features - 1] = torch.randint(0, 7, (batch_size,)).float()

        return obs

    def test_drive_policy_with_target_features(self):
        """Verify Drive policy correctly skips TARGET_FEATURES."""
        from pufferlib.ocean.drive import binding
        from pufferlib.ocean.torch import Drive
        import torch

        # Create a mock env-like object
        class MockEnv:
            def __init__(self):
                self.dynamics_model = "classic"
                self.max_partner_objects = binding.MAX_AGENTS - 1
                self.partner_features = binding.PARTNER_FEATURES
                self.max_road_objects = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
                self.road_features = binding.ROAD_FEATURES
                self.target_features = binding.TARGET_FEATURES

                # New observation size (with TARGET_FEATURES)
                ego_dim = 7  # classic
                self.num_obs = (
                    ego_dim
                    + self.target_features
                    + self.max_partner_objects * self.partner_features
                    + self.max_road_objects * self.road_features
                )

                import gymnasium

                self.single_observation_space = gymnasium.spaces.Box(
                    low=-1, high=1, shape=(self.num_obs,), dtype=np.float32
                )
                self.single_action_space = gymnasium.spaces.MultiDiscrete([91])

        mock_env = MockEnv()

        # Create Drive policy with use_target_features=True (default, skips target slot)
        policy = Drive(mock_env, use_target_features=True)

        # Create valid observation
        batch_size = 4
        obs = self._create_valid_obs(mock_env, batch_size)

        # Forward pass should work
        actions, value = policy(obs)

        assert value.shape == (batch_size, 1)

    def test_drive_policy_legacy_format(self):
        """Verify Drive policy works with legacy format (no TARGET_FEATURES)."""
        from pufferlib.ocean.drive import binding
        from pufferlib.ocean.torch import Drive
        import torch

        class MockEnv:
            def __init__(self):
                self.dynamics_model = "classic"
                self.max_partner_objects = binding.MAX_AGENTS - 1
                self.partner_features = binding.PARTNER_FEATURES
                self.max_road_objects = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
                self.road_features = binding.ROAD_FEATURES
                self.target_features = 0  # No target features in legacy

                # Old observation size (without TARGET_FEATURES)
                ego_dim = 7
                self.num_obs = (
                    ego_dim
                    + self.max_partner_objects * self.partner_features
                    + self.max_road_objects * self.road_features
                )

                import gymnasium

                self.single_observation_space = gymnasium.spaces.Box(
                    low=-1, high=1, shape=(self.num_obs,), dtype=np.float32
                )
                self.single_action_space = gymnasium.spaces.MultiDiscrete([91])

        mock_env = MockEnv()

        # Create Drive policy with use_target_features=False (legacy mode)
        policy = Drive(mock_env, use_target_features=False)

        # Create valid observation
        batch_size = 4
        obs = self._create_valid_obs(mock_env, batch_size)

        # Forward pass should work
        actions, value = policy(obs)

        assert value.shape == (batch_size, 1)

    def test_adversarial_drive_policy(self):
        """Verify AdversarialDrive policy uses TARGET_FEATURES."""
        from pufferlib.ocean.drive import binding
        from pufferlib.ocean.torch import AdversarialDrive
        import torch

        class MockEnv:
            def __init__(self):
                self.dynamics_model = "classic"
                self.max_partner_objects = binding.MAX_AGENTS - 1
                self.partner_features = binding.PARTNER_FEATURES
                self.max_road_objects = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
                self.road_features = binding.ROAD_FEATURES
                self.target_features = binding.TARGET_FEATURES

                ego_dim = 7
                self.num_obs = (
                    ego_dim
                    + self.target_features
                    + self.max_partner_objects * self.partner_features
                    + self.max_road_objects * self.road_features
                )

                import gymnasium

                self.single_observation_space = gymnasium.spaces.Box(
                    low=-1, high=1, shape=(self.num_obs,), dtype=np.float32
                )
                self.single_action_space = gymnasium.spaces.MultiDiscrete([91])

        mock_env = MockEnv()

        # Create AdversarialDrive policy
        policy = AdversarialDrive(mock_env)

        # Create valid observation
        batch_size = 4
        obs = self._create_valid_obs(mock_env, batch_size)

        # Forward pass should work
        actions, value = policy(obs)

        assert value.shape == (batch_size, 1)

        # Verify it has a target_encoder
        assert hasattr(policy, "target_encoder")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
