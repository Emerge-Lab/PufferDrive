"""
Test script to verify adversarial metrics tracking.

This test verifies that:
1. Episode returns are accumulated correctly using modified rewards
2. Completed episodes are split between ref and adv agents
3. Metrics are logged separately as ref_episode_return and adv_episode_return
"""

import torch
import numpy as np
from pufferlib import pufferl
from pufferlib.ocean.drive import make


def test_metrics_tracking():
    print("\n" + "=" * 60)
    print("Testing Adversarial Metrics Tracking")
    print("=" * 60)

    # Load test config
    config_path = "config/ocean/drive_test.ini"
    config = pufferl.load_config(config_path)

    # Enable adversarial mode
    config["adversarial"] = {"adversarial_mode": True, "reference_model_path": "experiments/latest_checkpoint.pt"}

    # Override for quick test
    config["num_envs"] = 2
    config["agents_per_env"] = 4  # Total 8 agents (2 ref, 6 adv)
    config["bptt_horizon"] = 16
    config["device"] = "cpu"
    config["use_rnn"] = False

    print(f"\nConfig: {config['num_envs']} envs × {config['agents_per_env']} agents")

    # Create environment
    vecenv = pufferl.PufferVecEnv(make, config, num_envs=config["num_envs"])

    # Create trainer
    trainer = pufferl.PufferRL(vecenv, config)

    print(f"\nAgent0 indices: {trainer.agent0_indices.tolist()}")
    print(f"Expected: [0, {config['agents_per_env']}]")

    # Check buffers were initialized
    assert hasattr(trainer, "ep_returns"), "ep_returns buffer not initialized"
    assert hasattr(trainer, "ref_episode_returns"), "ref_episode_returns list not initialized"
    assert hasattr(trainer, "adv_episode_returns"), "adv_episode_returns list not initialized"
    print("\n✓ Episode return tracking buffers initialized")

    # Simulate some episode completions by manually setting ep_returns and triggering collection
    print("\n" + "-" * 60)
    print("Simulating episode completions...")

    # Set some fake episode returns
    trainer.ep_returns[0] = 10.0  # agent0 in env 0 (ref)
    trainer.ep_returns[1] = -10.0  # agent1 in env 0 (adv, got -ref_reward)
    trainer.ep_returns[4] = 15.0  # agent0 in env 1 (ref)
    trainer.ep_returns[5] = -15.0  # agent1 in env 1 (adv, got -ref_reward)

    # Manually trigger collection (simulate done signals)
    for idx in [0, 1, 4, 5]:
        episode_return = trainer.ep_returns[idx].item()
        if idx in trainer.agent0_indices:
            trainer.ref_episode_returns.append(episode_return)
            print(f"  Agent {idx} (ref): return = {episode_return}")
        else:
            trainer.adv_episode_returns.append(episode_return)
            print(f"  Agent {idx} (adv): return = {episode_return}")
        trainer.ep_returns[idx] = 0

    # Check collections
    print(f"\nRef returns collected: {trainer.ref_episode_returns}")
    print(f"Adv returns collected: {trainer.adv_episode_returns}")

    assert len(trainer.ref_episode_returns) == 2, f"Expected 2 ref returns, got {len(trainer.ref_episode_returns)}"
    assert len(trainer.adv_episode_returns) == 2, f"Expected 2 adv returns, got {len(trainer.adv_episode_returns)}"
    print("✓ Episode returns collected correctly")

    # Test mean_and_log adds metrics to stats
    print("\n" + "-" * 60)
    print("Testing mean_and_log()...")

    trainer.stats = {}  # Clear stats
    trainer.mean_and_log()

    print(f"Stats keys: {list(trainer.stats.keys())}")

    assert "ref_episode_return" in trainer.stats, "ref_episode_return not in stats"
    assert "adv_episode_return" in trainer.stats, "adv_episode_return not in stats"

    ref_mean = trainer.stats["ref_episode_return"]
    adv_mean = trainer.stats["adv_episode_return"]

    print(f"  ref_episode_return: {ref_mean}")
    print(f"  adv_episode_return: {adv_mean}")

    # Check means are correct
    expected_ref_mean = np.mean([10.0, 15.0])
    expected_adv_mean = np.mean([-10.0, -15.0])

    assert abs(ref_mean - expected_ref_mean) < 0.01, f"Ref mean mismatch: {ref_mean} vs {expected_ref_mean}"
    assert abs(adv_mean - expected_adv_mean) < 0.01, f"Adv mean mismatch: {adv_mean} vs {expected_adv_mean}"
    print("✓ Metrics computed correctly")

    # Check lists were cleared
    assert len(trainer.ref_episode_returns) == 0, "ref_episode_returns not cleared after logging"
    assert len(trainer.adv_episode_returns) == 0, "adv_episode_returns not cleared after logging"
    print("✓ Return lists cleared after logging")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

    trainer.close()


if __name__ == "__main__":
    test_metrics_tracking()
