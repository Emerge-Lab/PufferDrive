"""Test C/Python binding for the Diplomacy environment.

This module tests that the C implementation can be successfully loaded,
initialized, and basic operations work correctly.
"""

import pytest
import numpy as np
from pufferlib.ocean.diplomacy import Diplomacy


class TestCBinding:
    """Tests for C/Python binding functionality."""

    def test_environment_creation(self):
        """Test that the environment can be created."""
        env = Diplomacy()
        assert env is not None

    def test_environment_reset(self):
        """Test that the environment can be reset."""
        env = Diplomacy()
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (7, 175)  # 7 agents, 175 features each

    def test_environment_step(self):
        """Test that the environment can take a step."""
        env = Diplomacy()
        env.reset()

        # Take a step with all hold orders (action 0)
        actions = np.zeros(7, dtype=np.int32)
        obs, rewards, dones, truncated, info = env.step(actions)

        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert isinstance(rewards, np.ndarray)
        assert isinstance(dones, np.ndarray)
        assert isinstance(truncated, np.ndarray)

    def test_observation_space(self):
        """Test that observation space is correctly defined."""
        env = Diplomacy()
        assert env.single_observation_space.shape == (175,)
        assert env.single_observation_space.dtype == np.float32

    def test_action_space(self):
        """Test that action space is correctly defined."""
        env = Diplomacy()
        assert env.single_action_space.n == 1000

    def test_num_agents(self):
        """Test that the number of agents is correct."""
        env = Diplomacy()
        assert env.num_agents == 7

    def test_multiple_resets(self):
        """Test that the environment can be reset multiple times."""
        env = Diplomacy()

        for _ in range(3):
            obs, info = env.reset()
            assert obs is not None
            assert obs.shape == (7, 175)

    def test_multiple_steps(self):
        """Test that the environment can take multiple steps."""
        env = Diplomacy()
        env.reset()

        actions = np.zeros(7, dtype=np.int32)
        for _ in range(5):
            obs, rewards, dones, truncated, info = env.step(actions)
            assert obs is not None

    @pytest.mark.skip(reason="Rendering not yet implemented")
    def test_render(self):
        """Test that the environment can be rendered."""
        env = Diplomacy()
        env.reset()
        env.render()
