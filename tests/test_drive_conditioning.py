from pufferlib.ocean.drive.drive import Drive
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dynamics_model,base_dim,total_dim",
    [
        ("classic", 7, 1848),
        ("jerk", 10, 1851),
    ],
)
def test_no_conditioning(dynamics_model, base_dim, total_dim):
    """Test that condition_type='none' works for both dynamics models."""
    env = Drive(num_agents=4, condition_type="none", num_maps=1, dynamics_model=dynamics_model, scenario_length=91)
    assert env.single_observation_space.shape[0] == base_dim + 63 * 7 + 200 * 7
    assert not env.reward_conditioned
    assert not env.entropy_conditioned
    obs, _ = env.reset()
    assert obs.shape == (4, total_dim)
    env.close()


@pytest.mark.parametrize(
    "dynamics_model,base_dim",
    [
        ("classic", 7),
        ("jerk", 10),
    ],
)
def test_reward_conditioning(dynamics_model, base_dim):
    """Test that RC adds 3 dimensions and weights are in range for both dynamics models."""
    env = Drive(
        num_agents=4,
        condition_type="reward",
        collision_weight_lb=-1.0,
        collision_weight_ub=0.0,
        offroad_weight_lb=-1.0,
        offroad_weight_ub=0.0,
        goal_weight_lb=0.0,
        goal_weight_ub=1.0,
        num_maps=1,
        dynamics_model=dynamics_model,
        scenario_length=91,
    )
    assert env.single_observation_space.shape[0] == base_dim + 3 + 63 * 7 + 200 * 7  # base + 3
    assert env.reward_conditioned
    obs, _ = env.reset()
    rc_weights = obs[:, base_dim : base_dim + 3]
    assert np.all((rc_weights[:, 0] >= -1.0) & (rc_weights[:, 0] <= 0.0))  # collision
    assert np.all((rc_weights[:, 1] >= -1.0) & (rc_weights[:, 1] <= 0.0))  # offroad
    assert np.all((rc_weights[:, 2] >= 0.0) & (rc_weights[:, 2] <= 1.0))  # goal
    env.close()


@pytest.mark.parametrize(
    "dynamics_model,base_dim",
    [
        ("classic", 7),
        ("jerk", 10),
    ],
)
def test_entropy_conditioning(dynamics_model, base_dim):
    """Test that EC adds 1 dimension and weight is in range for both dynamics models."""
    env = Drive(
        num_agents=4,
        condition_type="entropy",
        entropy_weight_lb=0.0,
        entropy_weight_ub=0.1,
        num_maps=1,
        dynamics_model=dynamics_model,
        scenario_length=91,
    )
    assert env.single_observation_space.shape[0] == base_dim + 1 + 63 * 7 + 200 * 7  # base + 1
    assert env.entropy_conditioned
    obs, _ = env.reset()
    ec_weight = obs[:, base_dim]
    assert np.all((ec_weight >= 0.0) & (ec_weight <= 0.1))
    env.close()


@pytest.mark.parametrize(
    "dynamics_model,base_dim",
    [
        ("classic", 7),
        ("jerk", 10),
    ],
)
def test_discount_conditioning(dynamics_model, base_dim):
    """Test that DC adds 1 dimension and weight is in range for both dynamics models."""
    env = Drive(
        num_agents=4,
        condition_type="discount",
        discount_weight_lb=0.9,
        discount_weight_ub=0.99,
        num_maps=1,
        dynamics_model=dynamics_model,
        scenario_length=91,
    )
    assert env.single_observation_space.shape[0] == base_dim + 1 + 63 * 7 + 200 * 7  # base + 1
    assert env.discount_conditioned
    obs, _ = env.reset()
    dc_weight = obs[:, base_dim]
    assert np.all((dc_weight >= 0.9) & (dc_weight <= 0.99))
    env.close()


@pytest.mark.parametrize(
    "dynamics_model,base_dim",
    [
        ("classic", 7),
        ("jerk", 10),
    ],
)
def test_combined_conditioning(dynamics_model, base_dim):
    """Test that RC + EC + DC work together for both dynamics models."""
    env = Drive(
        num_agents=4,
        condition_type="all",
        collision_weight_lb=-1.0,
        collision_weight_ub=0.0,
        offroad_weight_lb=-1.0,
        offroad_weight_ub=0.0,
        goal_weight_lb=0.0,
        goal_weight_ub=1.0,
        entropy_weight_lb=0.0,
        entropy_weight_ub=0.1,
        discount_weight_lb=0.9,
        discount_weight_ub=0.99,
        num_maps=1,
        dynamics_model=dynamics_model,
        scenario_length=91,
    )
    assert env.single_observation_space.shape[0] == base_dim + 5 + 63 * 7 + 200 * 7  # base + 3 + 1 + 1
    assert env.reward_conditioned
    assert env.entropy_conditioned
    assert env.discount_conditioned
    obs, _ = env.reset()
    weights = obs[:, base_dim : base_dim + 5]
    assert np.all((weights[:, 0] >= -1.0) & (weights[:, 0] <= 0.0))  # collision
    assert np.all((weights[:, 3] >= 0.0) & (weights[:, 3] <= 0.1))  # entropy
    assert np.all((weights[:, 4] >= 0.9) & (weights[:, 4] <= 0.99))  # discount
    env.close()


if __name__ == "__main__":
    # Run tests for both dynamics models
    for dynamics_model, base_dim, total_dim in [("classic", 7, 1848), ("jerk", 10, 1851)]:
        print(f"\nTesting {dynamics_model} dynamics model...")
        test_no_conditioning(dynamics_model, base_dim, total_dim)
        test_reward_conditioning(dynamics_model, base_dim)
        test_entropy_conditioning(dynamics_model, base_dim)
        test_discount_conditioning(dynamics_model, base_dim)
        test_combined_conditioning(dynamics_model, base_dim)
        print(f"✓ All tests passed for {dynamics_model} dynamics!")
