#!/usr/bin/env python3
"""Comprehensive pytest test suite for wosac_num_scenarios feature.

Tests scenario-count mode, edge cases, and backward compatibility.
"""

import pytest
import sys

sys.path.insert(0, ".")

from pufferlib.ocean.drive import drive


class TestScenarioCountMode:
    """Tests for num_scenarios feature."""

    def test_basic_scenario_count(self):
        """Test loading exactly N scenarios."""
        env = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=5,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            init_steps=10,
            scenario_length=91,
        )

        assert env.num_envs == 5, f"Expected 5 scenarios, got {env.num_envs}"
        assert env.num_agents > 0, "Should have at least some agents"
        assert len(env.map_ids) == 5, "Should have 5 map IDs"
        assert len(env.agent_offsets) == 6, "Offsets should have N+1 elements"

        env.close()

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same maps."""
        env1 = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=3,
            scenario_seed=42,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        env2 = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=3,
            scenario_seed=42,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        assert env1.map_ids == env2.map_ids, "Same seed should produce same maps"
        assert env1.num_agents == env2.num_agents, "Same maps should have same agent count"

        env1.close()
        env2.close()

    def test_different_seeds_different_maps(self):
        """Test that different seeds produce different maps."""
        env1 = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=5,
            scenario_seed=42,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        env2 = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=5,
            scenario_seed=99,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        # Very unlikely (but possible) to be equal
        assert env1.map_ids != env2.map_ids, "Different seeds should produce different maps (probabilistically)"

        env1.close()
        env2.close()

    def test_no_duplicate_scenarios(self):
        """Test that sampling without replacement produces unique scenarios."""
        env = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=50,
            scenario_seed=789,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        # Check all map_ids are unique
        unique_maps = set(env.map_ids)
        assert len(unique_maps) == len(env.map_ids), (
            f"Expected all unique maps, got {len(unique_maps)} unique out of {len(env.map_ids)}"
        )
        assert len(env.map_ids) == 50, "Should have loaded 50 scenarios"

        env.close()

    def test_all_scenarios_unique_large_request(self):
        """Test uniqueness with larger scenario count."""
        env = drive.Drive(
            num_maps=1000,
            num_agents=999999,
            num_scenarios=500,
            scenario_seed=12345,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        # Verify no duplicates
        map_ids_list = list(env.map_ids)
        unique_maps = set(map_ids_list)
        assert len(unique_maps) == 500, f"Expected 500 unique maps, got {len(unique_maps)}"
        assert len(unique_maps) == len(map_ids_list), "All scenarios should be unique"

        env.close()

    def test_agent_count_determined_from_scenarios(self):
        """Test that num_agents is correctly determined from loaded scenarios."""
        env = drive.Drive(
            num_maps=100,
            num_agents=999999,  # This should be ignored
            num_scenarios=3,
            scenario_seed=123,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        # Verify agent count matches offset calculation
        expected_agent_count = env.agent_offsets[-1]
        assert env.num_agents == expected_agent_count

        # Verify observations buffer has correct size
        assert env.observations.shape[0] == env.num_agents

        env.close()

    def test_single_scenario(self):
        """Edge case: Load exactly 1 scenario."""
        env = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=1,
            scenario_seed=1,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        assert env.num_envs == 1
        assert env.num_agents > 0

        env.close()

    def test_many_scenarios(self):
        """Test loading many scenarios (e.g., 20)."""
        env = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=20,
            scenario_seed=456,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        assert env.num_envs == 20
        assert len(env.map_ids) == 20

        env.close()


class TestBackwardCompatibility:
    """Tests to ensure existing agent-count mode still works."""

    def test_default_agent_count_mode(self):
        """Test that not specifying num_scenarios uses agent-count mode."""
        env = drive.Drive(
            num_maps=100,
            num_agents=100,
            control_mode="control_vehicles",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        # Should load 100 agents
        assert env.num_agents == 100, f"Expected 100 agents, got {env.num_agents}"
        assert env.num_envs > 0

        env.close()

    def test_agent_count_with_none_scenarios(self):
        """Test explicitly setting num_scenarios=None uses agent-count mode."""
        env = drive.Drive(
            num_maps=100,
            num_agents=50,
            num_scenarios=None,
            control_mode="control_vehicles",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        assert env.num_agents == 50, f"Expected 50 agents, got {env.num_agents}"

        env.close()

    def test_different_control_modes(self):
        """Test that scenario mode works with different control modes."""
        for control_mode in ["control_wosac", "control_vehicles", "control_agents"]:
            env = drive.Drive(
                num_maps=100,
                num_agents=999999,
                num_scenarios=2,
                scenario_seed=789,
                control_mode=control_mode,
                init_mode="create_all_valid",
                scenario_length=91,
            )

            assert env.num_envs == 2
            env.close()


class TestOffsetConsistency:
    """Tests for agent_offsets correctness."""

    def test_offsets_are_monotonic(self):
        """Verify agent_offsets are strictly increasing."""
        env = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=10,
            scenario_seed=111,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        offsets = env.agent_offsets
        for i in range(len(offsets) - 1):
            assert offsets[i] < offsets[i + 1], f"Offsets should be monotonic: {offsets}"

        env.close()

    def test_offsets_start_at_zero(self):
        """Verify first offset is 0."""
        env = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=5,
            scenario_seed=222,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        assert env.agent_offsets[0] == 0, "First offset should be 0"

        env.close()

    def test_last_offset_equals_num_agents(self):
        """Verify last offset equals total agent count."""
        env = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=7,
            scenario_seed=333,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        assert env.agent_offsets[-1] == env.num_agents

        env.close()


class TestSeedBehavior:
    """Tests for scenario_seed parameter."""

    def test_no_seed_random_maps(self):
        """Test that without seed, maps are random (probabilistic test)."""
        env1 = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=5,
            # No scenario_seed
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        env2 = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=5,
            # No scenario_seed
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        # Very likely to be different (not guaranteed, but high probability)
        # If this fails occasionally, it's expected (random chance)
        different = env1.map_ids != env2.map_ids

        env1.close()
        env2.close()

        # Note: This test might occasionally fail due to random chance
        assert different, "Random maps should likely be different (can fail due to chance)"

    def test_seed_zero_is_valid(self):
        """Test that seed=0 is valid and reproducible."""
        env1 = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=3,
            scenario_seed=0,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        env2 = drive.Drive(
            num_maps=100,
            num_agents=999999,
            num_scenarios=3,
            scenario_seed=0,
            control_mode="control_wosac",
            init_mode="create_all_valid",
            scenario_length=91,
        )

        assert env1.map_ids == env2.map_ids, "Seed=0 should be reproducible"

        env1.close()
        env2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
