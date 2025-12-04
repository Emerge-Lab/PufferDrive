#!/usr/bin/env python3
"""
Test script for SDC (Self-Driving Car) indices exposure from C to Python.

This tests that:
1. binding.shared() returns sdc_indices as the 4th element
2. SDC indices correctly map to global agent positions
3. SDC indices are correctly updated on environment resample
4. The adversarial mode correctly uses SDC indices instead of agent0

Running the test: python -m pytest tests/test_sdc_indices.py -v
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSDCIndices(unittest.TestCase):
    """Test SDC index functionality in the Drive environment."""

    def setUp(self):
        """Set up test fixtures."""
        self.map_dir = "resources/drive/binaries/training"

        # Check if map directory exists
        if not os.path.exists(self.map_dir):
            self.skipTest(f"Map directory {self.map_dir} not found. Skipping SDC index tests.")

    def test_binding_shared_returns_sdc_indices(self):
        """Test that binding.shared() returns a 4-tuple including sdc_indices."""
        from pufferlib.ocean.drive import binding

        result = binding.shared(
            map_dir=self.map_dir,
            num_agents=1024,
            num_maps=10,
            init_mode=0,  # INIT_ALL_VALID
            control_mode=0,  # CONTROL_VEHICLES
            init_steps=0,
            max_controlled_agents=-1,
            goal_behavior=0,
        )

        # Should return 4 elements: (agent_offsets, map_ids, num_envs, sdc_indices)
        self.assertEqual(len(result), 4, "binding.shared() should return 4 elements")

        agent_offsets, map_ids, num_envs, sdc_indices = result

        # Basic type checks
        self.assertIsInstance(agent_offsets, list)
        self.assertIsInstance(map_ids, list)
        self.assertIsInstance(num_envs, int)
        self.assertIsInstance(sdc_indices, list)

        # sdc_indices should have one entry per environment
        self.assertEqual(
            len(sdc_indices), num_envs, f"sdc_indices should have {num_envs} entries, got {len(sdc_indices)}"
        )

        print(f"\n  num_envs: {num_envs}")
        print(f"  agent_offsets: {agent_offsets}")
        print(f"  sdc_indices: {sdc_indices}")

    def test_sdc_indices_valid_range(self):
        """Test that SDC indices are within valid agent ranges or -1."""
        from pufferlib.ocean.drive import binding

        result = binding.shared(
            map_dir=self.map_dir,
            num_agents=1024,
            num_maps=20,
            init_mode=0,
            control_mode=0,
            init_steps=0,
            max_controlled_agents=-1,
            goal_behavior=0,
        )

        agent_offsets, map_ids, num_envs, sdc_indices = result

        for env_idx in range(num_envs):
            sdc_idx = sdc_indices[env_idx]
            env_start = agent_offsets[env_idx]
            env_end = agent_offsets[env_idx + 1]

            if sdc_idx == -1:
                # SDC not in active agents for this env (valid)
                print(f"  Env {env_idx}: SDC not active (index=-1)")
            else:
                # SDC should be within this environment's agent range
                self.assertGreaterEqual(
                    sdc_idx, env_start, f"Env {env_idx}: SDC index {sdc_idx} < env_start {env_start}"
                )
                self.assertLess(sdc_idx, env_end, f"Env {env_idx}: SDC index {sdc_idx} >= env_end {env_end}")
                print(f"  Env {env_idx}: SDC at global index {sdc_idx} (local: {sdc_idx - env_start})")

    def test_drive_env_exposes_sdc_indices(self):
        """Test that the Drive environment class exposes sdc_indices."""
        from pufferlib.ocean.drive.drive import Drive

        env = Drive(
            num_agents=1024,
            num_maps=10,
            map_dir=self.map_dir,
            init_mode="create_all_valid",
            control_mode="control_vehicles",
            scenario_length=91,  # Required to avoid None handling issue
        )

        # Check that sdc_indices attribute exists
        self.assertTrue(hasattr(env, "sdc_indices"), "Drive env should have 'sdc_indices' attribute")

        # Check that it's a list with correct length
        self.assertIsInstance(env.sdc_indices, list)
        self.assertEqual(len(env.sdc_indices), env.num_envs, f"sdc_indices should have {env.num_envs} entries")

        print(f"\n  Drive env created with {env.num_envs} environments")
        print(f"  sdc_indices: {env.sdc_indices[:5]}...")  # First 5 entries

        env.close()

    def test_sdc_indices_vs_agent0(self):
        """Test that SDC indices differ from agent0 indices in some cases."""
        from pufferlib.ocean.drive import binding

        result = binding.shared(
            map_dir=self.map_dir,
            num_agents=256,
            num_maps=50,
            init_mode=0,
            control_mode=0,
            init_steps=0,
            max_controlled_agents=-1,
            goal_behavior=0,
        )

        agent_offsets, map_ids, num_envs, sdc_indices = result

        # Compute agent0 indices (what we used before)
        agent0_indices = [agent_offsets[i] for i in range(num_envs)]

        # Count how many differ
        differ_count = 0
        same_count = 0
        inactive_count = 0

        for env_idx in range(num_envs):
            sdc_idx = sdc_indices[env_idx]
            agent0_idx = agent0_indices[env_idx]

            if sdc_idx == -1:
                inactive_count += 1
            elif sdc_idx != agent0_idx:
                differ_count += 1
            else:
                same_count += 1

        print(f"\n  Total environments: {num_envs}")
        print(f"  SDC == agent0: {same_count}")
        print(f"  SDC != agent0: {differ_count}")
        print(f"  SDC inactive: {inactive_count}")

        # This is informational - we don't assert anything here since
        # the SDC could legitimately be agent0 in many scenarios

    def test_sdc_indices_after_resample(self):
        """Test that sdc_indices update correctly after environment resampling."""
        from pufferlib.ocean.drive.drive import Drive

        env = Drive(
            num_agents=64,
            num_maps=20,
            map_dir=self.map_dir,
            init_mode="create_all_valid",
            control_mode="control_vehicles",
            resample_frequency=10,  # Resample every 10 steps
            scenario_length=91,  # Required to avoid None handling issue
        )

        initial_sdc_indices = list(env.sdc_indices)

        env.reset()

        # Step through until resample triggers
        # Action shape is (num_agents, 1) for MultiDiscrete([91]) action space
        for _ in range(15):  # Should trigger resample
            actions = np.zeros((env.num_agents, 1), dtype=np.int32)
            env.step(actions)

        # Check that sdc_indices is still valid after resample
        self.assertTrue(hasattr(env, "sdc_indices"))
        self.assertEqual(len(env.sdc_indices), env.num_envs)

        # Validate new indices are in valid ranges
        for env_idx in range(env.num_envs):
            sdc_idx = env.sdc_indices[env_idx]
            if sdc_idx != -1:
                env_start = env.agent_offsets[env_idx]
                env_end = env.agent_offsets[env_idx + 1]
                self.assertGreaterEqual(sdc_idx, env_start)
                self.assertLess(sdc_idx, env_end)

        print(f"\n  Before resample - sdc_indices[:3]: {initial_sdc_indices[:3]}")
        print(f"  After resample - sdc_indices[:3]: {env.sdc_indices[:3]}")

        env.close()


class TestSDCIndicesInPuffeRL(unittest.TestCase):
    """Test SDC index usage in PuffeRL adversarial mode."""

    def setUp(self):
        """Set up test fixtures."""
        self.map_dir = "resources/drive/binaries/training"

        if not os.path.exists(self.map_dir):
            self.skipTest(f"Map directory {self.map_dir} not found.")

    def test_sdc_indices_tensor_creation(self):
        """Test that sdc_indices can be converted to PyTorch tensor correctly."""
        import torch
        from pufferlib.ocean.drive import binding

        result = binding.shared(
            map_dir=self.map_dir,
            num_agents=64,
            num_maps=10,
            init_mode=0,
            control_mode=0,
            init_steps=0,
            max_controlled_agents=-1,
            goal_behavior=0,
        )

        agent_offsets, map_ids, num_envs, sdc_indices = result

        # This is how pufferl.py creates the tensor
        device = "cpu"
        sdc_tensor = torch.tensor(sdc_indices, device=device, dtype=torch.int64)

        self.assertEqual(sdc_tensor.shape, (num_envs,))
        self.assertEqual(sdc_tensor.dtype, torch.int64)

        print(f"\n  Created SDC tensor with shape {sdc_tensor.shape}")
        print(f"  Values: {sdc_tensor[:5].tolist()}...")

    def test_sdc_mask_creation(self):
        """Test that SDC mask can be created for batch processing."""
        import torch
        from pufferlib.ocean.drive import binding

        result = binding.shared(
            map_dir=self.map_dir,
            num_agents=64,
            num_maps=10,
            init_mode=0,
            control_mode=0,
            init_steps=0,
            max_controlled_agents=-1,
            goal_behavior=0,
        )

        agent_offsets, map_ids, num_envs, sdc_indices = result
        num_agents = agent_offsets[-1]  # Total agents

        device = "cpu"
        sdc_tensor = torch.tensor(sdc_indices, device=device, dtype=torch.int64)

        # Simulate batch processing (as in evaluate())
        batch_start = 0
        batch_end = min(32, num_agents)
        batch_indices = torch.arange(batch_start, batch_end, device=device)

        # Create SDC mask - which agents in this batch are the SDC
        sdc_mask = torch.isin(batch_indices, sdc_tensor)

        self.assertEqual(sdc_mask.shape, (batch_end - batch_start,))
        self.assertEqual(sdc_mask.dtype, torch.bool)

        sdc_count = sdc_mask.sum().item()
        adv_count = (~sdc_mask).sum().item()

        print(f"\n  Batch [{batch_start}, {batch_end})")
        print(f"  SDC agents in batch: {sdc_count}")
        print(f"  Adversarial agents in batch: {adv_count}")

        # At least some agents should be SDC (if any SDC indices fall in this batch)
        # This depends on the actual data, so we just verify the logic works


class TestSDCIndicesEdgeCases(unittest.TestCase):
    """Test edge cases for SDC index handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.map_dir = "resources/drive/binaries/training"

        if not os.path.exists(self.map_dir):
            self.skipTest(f"Map directory {self.map_dir} not found.")

    def test_sdc_index_negative_one_handling(self):
        """Test that SDC index of -1 (inactive) is handled correctly."""
        import torch

        # Simulate a case where some SDCs are inactive
        sdc_indices = [0, -1, 5, -1, 10]
        sdc_tensor = torch.tensor(sdc_indices, dtype=torch.int64)

        # When checking if an agent is SDC, -1 should never match
        batch_indices = torch.arange(0, 15)
        sdc_mask = torch.isin(batch_indices, sdc_tensor)

        # -1 should not be in range [0, 15), so it shouldn't match anything
        # Agents 0, 5, 10 should be marked as SDC
        expected_sdc_positions = [0, 5, 10]
        actual_sdc_positions = batch_indices[sdc_mask].tolist()

        self.assertEqual(
            actual_sdc_positions,
            expected_sdc_positions,
            f"Expected SDC at {expected_sdc_positions}, got {actual_sdc_positions}",
        )

    def test_empty_environment_handling(self):
        """Test handling when no environments are created."""
        from pufferlib.ocean.drive import binding

        # Request very few agents - might result in fewer envs
        result = binding.shared(
            map_dir=self.map_dir,
            num_agents=8,
            num_maps=5,
            init_mode=0,
            control_mode=0,
            init_steps=0,
            max_controlled_agents=-1,
            goal_behavior=0,
        )

        agent_offsets, map_ids, num_envs, sdc_indices = result

        # Even with few agents, we should have consistent data
        self.assertEqual(len(sdc_indices), num_envs)
        self.assertEqual(len(map_ids), num_envs)
        self.assertEqual(len(agent_offsets), num_envs + 1)

        print(f"\n  Small env test: {num_envs} environments created")


if __name__ == "__main__":
    unittest.main(verbosity=2)
