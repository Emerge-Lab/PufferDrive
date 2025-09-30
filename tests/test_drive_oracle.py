from pufferlib.ocean.drive.drive import Drive
import numpy as np

def test_oracle_mode():
    """Test that oracle mode correctly shares conditioning values."""
    num_agents = 4
    env = Drive(
        num_agents=num_agents,
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

    # Check observation space size
    conditioning_dims = 4  # 3 RC + 1 EC
    oracle_dims = num_agents * conditioning_dims
    expected_size = 7 + conditioning_dims + 63*7 + 200*7 + oracle_dims
    assert env.single_observation_space.shape[0] == expected_size
    assert env.oracle_mode

    obs, _ = env.reset()
    oracle_start = 7 + conditioning_dims + 63*7 + 200*7

    for i in range(num_agents):
        own_weights = obs[i, 7:11]

        oracle_obs_i = obs[i, oracle_start:oracle_start + oracle_dims].reshape(num_agents, conditioning_dims)
        assert np.allclose(oracle_obs_i[i], 0.0, atol=1e-5), \
            f"Agent {i} should see zeros at its own position {i}"

        for j in range(num_agents):
            if i == j:
                continue
            oracle_obs_j = obs[j, oracle_start:oracle_start + oracle_dims].reshape(num_agents, conditioning_dims)
            seen_weights = oracle_obs_j[i]
            assert np.allclose(own_weights, seen_weights, atol=1e-5), \
                f"Agent {j} should see agent {i}'s weights at oracle position {i}"

    env.close()

if __name__ == '__main__':
    test_oracle_mode()
