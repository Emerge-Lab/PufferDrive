from pufferlib.ocean.drive.drive import Drive
import numpy as np

def test_no_compatibility():
    """Test that condition_type='none' works."""
    env = Drive(num_agents=4, condition_type="none", oracle_mode=False, num_maps=1)
    assert env.single_observation_space.shape[0] == 7 + 63*7 + 200*7
    assert not env.reward_conditioned
    assert not env.entropy_conditioned
    obs, _ = env.reset()
    assert obs.shape == (4, 1848)
    env.close()

def test_reward_conditioning():
    """Test that RC adds 3 dimensions and weights are in range."""
    env = Drive(
        num_agents=4,
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
    assert env.single_observation_space.shape[0] == 7 + 3 + 63*7 + 200*7  # base + 3
    assert env.reward_conditioned
    obs, _ = env.reset()
    rc_weights = obs[:, 7:10]
    assert np.all((rc_weights[:, 0] >= -1.0) & (rc_weights[:, 0] <= 0.0))  # collision
    assert np.all((rc_weights[:, 1] >= -1.0) & (rc_weights[:, 1] <= 0.0))  # offroad
    assert np.all((rc_weights[:, 2] >= 0.0) & (rc_weights[:, 2] <= 1.0))   # goal
    env.close()

def test_entropy_conditioning():
    """Test that EC adds 1 dimension and weight is in range."""
    env = Drive(
        num_agents=4,
        condition_type="entropy",
        oracle_mode=False,
        entropy_weight_lb=0.0,
        entropy_weight_ub=0.1,
        num_maps=1
    )
    assert env.single_observation_space.shape[0] == 7 + 1 + 63*7 + 200*7  # base + 1
    assert env.entropy_conditioned
    obs, _ = env.reset()
    ec_weight = obs[:, 7]
    assert np.all((ec_weight >= 0.0) & (ec_weight <= 0.1))
    env.close()

def test_combined_conditioning():
    """Test that RC + EC work together."""
    env = Drive(
        num_agents=4,
        condition_type="all",
        oracle_mode=False,
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
    assert env.single_observation_space.shape[0] == 7 + 4 + 63*7 + 200*7  # base + 3 + 1
    assert env.reward_conditioned
    assert env.entropy_conditioned
    obs, _ = env.reset()
    weights = obs[:, 7:11]
    assert np.all((weights[:, 0] >= -1.0) & (weights[:, 0] <= 0.0))
    assert np.all((weights[:, 3] >= 0.0) & (weights[:, 3] <= 0.1))
    env.close()

if __name__ == '__main__':
    test_no_compatibility()
    test_reward_conditioning()
    test_entropy_conditioning()
    test_combined_conditioning()
