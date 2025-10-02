"""Test that the network can be initialized and run with different conditioning configs."""
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.torch import Recurrent, Drive as DriveNet
import numpy as np
import torch

def test_baseline_network():
    """Test network with condition_type='none', oracle_mode=False."""
    print("\n=== Testing baseline (none, False) ===")
    env = Drive(num_agents=64, condition_type="none", oracle_mode=False, num_maps=1)

    encoder_size = 64
    base_policy = DriveNet(env, input_size=encoder_size, hidden_size=256)
    policy = Recurrent(env, base_policy, input_size=base_policy.input_size, hidden_size=256)

    obs, _ = env.reset()
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

    # Initialize hidden state
    state = {"lstm_h": None, "lstm_c": None}

    # Forward pass
    logits, value = policy(obs_tensor, state)

    print(f"Obs shape: {obs.shape}")
    if isinstance(logits, tuple):
        print(f"Logits: tuple of {len(logits)} tensors, shapes: {[l.shape for l in logits]}")
        logits_tensor = torch.cat(logits, dim=-1)
    else:
        logits_tensor = logits
        print(f"Logits shape: {logits.shape}")
    
    print(f"Value shape: {value.shape}")
    print(f"Value mean: {value.mean().item():.4f}, std: {value.std().item():.4f}")

    # Check for NaN/Inf
    assert not torch.isnan(logits_tensor).any(), "Logits contain NaN"
    assert not torch.isnan(value).any(), "Value contains NaN"
    assert not torch.isinf(logits_tensor).any(), "Logits contain Inf"
    assert not torch.isinf(value).any(), "Value contains Inf"

    env.close()
    print("✓ Baseline network test passed")

def test_reward_conditioning_network():
    """Test network with reward conditioning."""
    print("\n=== Testing reward conditioning (reward, False) ===")
    env = Drive(
        num_agents=64,
        condition_type="reward",
        oracle_mode=False,
        collision_weight_lb=-1.0,
        collision_weight_ub=0.0,
        offroad_weight_lb=-1.0,
        offroad_weight_ub=0.0,
        goal_weight_lb=0.0,
        goal_weight_ub=1.0,
        num_maps=1
    )

    encoder_size = 64
    base_policy = DriveNet(env, input_size=encoder_size, hidden_size=256)
    policy = Recurrent(env, base_policy, input_size=base_policy.input_size, hidden_size=256)

    obs, _ = env.reset()
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
    state = {"lstm_h": None, "lstm_c": None}

    logits, value = policy(obs_tensor, state)

    print(f"Obs shape: {obs.shape}")
    print(f"Expected obs shape: (64, {7 + 3 + 63*7 + 200*7})")
    if isinstance(logits, tuple):
        print(f"Logits: tuple of {len(logits)} tensors")
        logits_tensor = torch.cat(logits, dim=-1)
    else:
        logits_tensor = logits
    print(f"Value shape: {value.shape}")
    print(f"Value mean: {value.mean().item():.4f}, std: {value.std().item():.4f}")

    assert not torch.isnan(logits_tensor).any(), "Logits contain NaN"
    assert not torch.isnan(value).any(), "Value contains NaN"

    env.close()
    print("✓ Reward conditioning network test passed")

def test_oracle_mode_network():
    """Test network with oracle mode."""
    print("\n=== Testing oracle mode (all, True) ===")
    env = Drive(
        num_agents=64,
        condition_type="all",
        oracle_mode=True,
        collision_weight_lb=-1.0,
        collision_weight_ub=0.0,
        offroad_weight_lb=-1.0,
        offroad_weight_ub=0.0,
        goal_weight_lb=0.0,
        goal_weight_ub=1.0,
        entropy_weight_lb=0.0,
        entropy_weight_ub=0.1,
        num_maps=1
    )

    encoder_size = 64
    base_policy = DriveNet(env, input_size=encoder_size, hidden_size=256)
    policy = Recurrent(env, base_policy, input_size=base_policy.input_size, hidden_size=256)

    obs, _ = env.reset()
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
    state = {"lstm_h": None, "lstm_c": None}

    logits, value = policy(obs_tensor, state)

    conditioning_dims = 4  # 3 reward + 1 entropy
    oracle_dims = 64 * conditioning_dims
    expected_obs_size = 7 + 4 + 63*7 + 200*7 + oracle_dims

    print(f"Obs shape: {obs.shape}")
    print(f"Expected obs shape: (64, {expected_obs_size})")
    print(f"Oracle dims: {oracle_dims}")
    if isinstance(logits, tuple):
        print(f"Logits: tuple of {len(logits)} tensors")
        logits_tensor = torch.cat(logits, dim=-1)
    else:
        logits_tensor = logits
    print(f"Value mean: {value.mean().item():.4f}, std: {value.std().item():.4f}")

    assert not torch.isnan(logits_tensor).any(), "Logits contain NaN"
    assert not torch.isnan(value).any(), "Value contains NaN"

    env.close()
    print("✓ Oracle mode network test passed")

if __name__ == '__main__':
    test_baseline_network()
    test_reward_conditioning_network()
    test_oracle_mode_network()
    print("\n✓ All network tests passed!")