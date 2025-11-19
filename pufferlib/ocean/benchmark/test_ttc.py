import math
import numpy as np
import pytest

from pufferlib.ocean.benchmark import interaction_features

MAX_HEADING_DIFF = interaction_features.MAX_HEADING_DIFF
SMALL_OVERLAP_THRESHOLD = interaction_features.SMALL_OVERLAP_THRESHOLD
MAX_HEADING_DIFF_FOR_SMALL_OVERLAP = interaction_features.MAX_HEADING_DIFF_FOR_SMALL_OVERLAP
MAX_TTC_SEC = interaction_features.MAXIMUM_TIME_TO_COLLISION


def test_time_to_collision_output_shape():
    """Test that TTC returns correct shape."""
    num_agents = 4
    num_rollouts = 2
    num_steps = 10

    center_x = np.random.randn(num_agents, num_rollouts, num_steps).astype(np.float32)
    center_y = np.random.randn(num_agents, num_rollouts, num_steps).astype(np.float32)
    length = np.ones((num_agents, num_rollouts), dtype=np.float32) * 4.0
    width = np.ones((num_agents, num_rollouts), dtype=np.float32) * 2.0
    heading = np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32)
    valid = np.ones((num_agents, num_rollouts, num_steps), dtype=bool)

    eval_mask = np.ones(num_agents, dtype=bool)
    ttc = interaction_features.compute_time_to_collision(
        center_x=center_x,
        center_y=center_y,
        length=length,
        width=width,
        heading=heading,
        valid=valid,
        evaluated_object_mask=eval_mask,
        seconds_per_step=0.1,
    )

    assert ttc.shape == (num_agents, num_rollouts, num_steps)


@pytest.mark.parametrize(
    "center_xys,headings,boxes_sizes,speeds,expected_ttc_sec",
    [
        # 9 square boxes in a 3x3 grid
        (
            [[-3, 3], [0, 3], [3, 3], [-3, 0], [0, 0], [3, 0], [-3, -3], [0, -3], [3, -3]],
            [0] * 9,
            [[1, 1]] * 9,
            [10, 6, 1] * 3,
            [2 / 4, 2 / 5, MAX_TTC_SEC] * 3,
        ),
        # Rectangles in a line
        (
            [[0, 0], [5, 0], [10, 0], [15, 0]],
            [0, 0, 0, 0],
            [[4, 2]] * 4,
            [6, 10, 3, 1],
            [MAX_TTC_SEC, 1 / 7, 1 / 2, MAX_TTC_SEC],
        ),
        # Test ignore misaligned
        (
            [[0, 0], [5, 0], [10, 0], [15, 0]],
            [0, MAX_HEADING_DIFF + 0.01, 0, MAX_HEADING_DIFF - 0.01],
            [[4, 2]] * 4,
            [10, 6, 3, 1],
            [
                6 / 7,
                MAX_TTC_SEC,
                (3 - math.cos(MAX_HEADING_DIFF - 0.01) * 2 - math.sin(MAX_HEADING_DIFF - 0.01)) / 2,
                MAX_TTC_SEC,
            ],
        ),
        # Test ignore no overlap
        (
            [[0, 0], [5, 2.1], [10, 1.1], [15, 0]],
            [0, 0, 0, 0],
            [[4, 2]] * 4,
            [10, 6, 3, 1],
            [6 / 7, 1 / 3, 1 / 2, MAX_TTC_SEC],
        ),
        # Test ignore small misalignment with low overlap
        (
            [
                [0, 0],
                [5, 2.5 - SMALL_OVERLAP_THRESHOLD],
                [10, -2.5 + SMALL_OVERLAP_THRESHOLD],
            ],
            [0, MAX_HEADING_DIFF_FOR_SMALL_OVERLAP + 0.01, -MAX_HEADING_DIFF_FOR_SMALL_OVERLAP + 0.01],
            [[4, 2]] * 3,
            [6, 3, 1],
            [
                (8 - math.cos(MAX_HEADING_DIFF_FOR_SMALL_OVERLAP - 0.01) * 2 - math.sin(MAX_HEADING_DIFF_FOR_SMALL_OVERLAP - 0.01)) / 5,
                MAX_TTC_SEC,
                MAX_TTC_SEC,
            ],
        ),
    ],
)
def test_time_to_collision_values(center_xys, headings, boxes_sizes, speeds, expected_ttc_sec):
    """Test TTC computation with various configurations."""
    center_xys = np.array(center_xys, dtype=np.float32)
    headings = np.array(headings, dtype=np.float32)
    boxes_sizes = np.array(boxes_sizes, dtype=np.float32)
    speeds = np.array(speeds, dtype=np.float32)
    expected_ttc_sec = np.array(expected_ttc_sec, dtype=np.float32)

    num_agents = len(center_xys)
    num_rollouts = 1
    seconds_per_step = 0.1

    # Simulate 3 timesteps (t-1, t, t+1) to get proper speeds with central difference
    center_x_1 = center_xys[:, 0]
    center_x_2 = center_x_1 + speeds * np.cos(headings) * seconds_per_step
    center_x_0 = center_x_1 - speeds * np.cos(headings) * seconds_per_step
    center_x = np.stack([center_x_0, center_x_1, center_x_2], axis=-1)

    center_y_1 = center_xys[:, 1]
    center_y_2 = center_y_1 - speeds * np.sin(headings) * seconds_per_step
    center_y_0 = center_y_1 + speeds * np.sin(headings) * seconds_per_step
    center_y = np.stack([center_y_0, center_y_1, center_y_2], axis=-1)

    # Reshape to (num_agents, num_rollouts, num_steps)
    center_x = center_x[:, np.newaxis, :]
    center_y = center_y[:, np.newaxis, :]

    length = np.broadcast_to(boxes_sizes[:, 0:1], (num_agents, num_rollouts))
    width = np.broadcast_to(boxes_sizes[:, 1:2], (num_agents, num_rollouts))
    heading = np.broadcast_to(headings[:, np.newaxis, np.newaxis], (num_agents, num_rollouts, 3))
    valid = np.ones((num_agents, num_rollouts, 3), dtype=bool)

    eval_mask = np.ones(num_agents, dtype=bool)
    ttc = interaction_features.compute_time_to_collision(
        center_x=center_x,
        center_y=center_y,
        length=length,
        width=width,
        heading=heading,
        valid=valid,
        evaluated_object_mask=eval_mask,
        seconds_per_step=seconds_per_step,
    )

    # Check TTC at timestep 1 (middle) where speeds are valid
    np.testing.assert_allclose(ttc[:, 0, 1], expected_ttc_sec, rtol=1e-5, atol=1e-5)


def test_time_to_collision_invalid_objects():
    """Test that invalid objects are ignored in TTC computation."""
    num_agents = 3
    num_rollouts = 1
    num_steps = 3

    # Create 3 agents in a line: agent 0 at x=0, agent 1 at x=5, agent 2 at x=10
    # All moving forward with speeds [6, 3, 1]
    center_xys = np.array([[0, 0], [5, 0], [10, 0]], dtype=np.float32)
    speeds = np.array([6, 3, 1], dtype=np.float32)
    headings = np.zeros(num_agents, dtype=np.float32)
    seconds_per_step = 0.1

    # Create 3 timesteps
    center_x_1 = center_xys[:, 0]
    center_x_2 = center_x_1 + speeds * np.cos(headings) * seconds_per_step
    center_x_0 = center_x_1 - speeds * np.cos(headings) * seconds_per_step
    center_x = np.stack([center_x_0, center_x_1, center_x_2], axis=-1)[:, np.newaxis, :]

    center_y_1 = center_xys[:, 1]
    center_y_2 = center_y_1 - speeds * np.sin(headings) * seconds_per_step
    center_y_0 = center_y_1 + speeds * np.sin(headings) * seconds_per_step
    center_y = np.stack([center_y_0, center_y_1, center_y_2], axis=-1)[:, np.newaxis, :]

    length = np.ones((num_agents, num_rollouts), dtype=np.float32) * 4.0
    width = np.ones((num_agents, num_rollouts), dtype=np.float32) * 2.0
    heading = np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32)

    # Test with agent 1 invalid - agent 0 should see agent 2 as nearest
    valid = np.ones((num_agents, num_rollouts, num_steps), dtype=bool)
    valid[1, 0, 1] = False

    eval_mask = np.ones(num_agents, dtype=bool)
    ttc = interaction_features.compute_time_to_collision(
        center_x=center_x,
        center_y=center_y,
        length=length,
        width=width,
        heading=heading,
        valid=valid,
        evaluated_object_mask=eval_mask,
        seconds_per_step=seconds_per_step,
    )

    # Agent 0 should now see agent 2 at distance 10 with relative speed 6-1=5
    # TTC = (10 - 2 - 2) / 5 = 6/5 = 1.2
    expected_ttc_agent0 = (10 - 2 - 2) / (6 - 1)

    np.testing.assert_allclose(ttc[0, 0, 1], expected_ttc_agent0, rtol=1e-5, atol=1e-5)


def test_time_to_collision_no_object_ahead():
    """Test TTC returns max value when no object is ahead."""
    num_agents = 2
    num_rollouts = 1
    num_steps = 3

    # Create 2 agents moving away from each other
    center_xys = np.array([[0, 0], [10, 0]], dtype=np.float32)
    speeds = np.array([5, 5], dtype=np.float32)
    headings = np.array([0, math.pi], dtype=np.float32)  # Opposite directions
    seconds_per_step = 0.1

    center_x_1 = center_xys[:, 0]
    center_x_2 = center_x_1 + speeds * np.cos(headings) * seconds_per_step
    center_x_0 = center_x_1 - speeds * np.cos(headings) * seconds_per_step
    center_x = np.stack([center_x_0, center_x_1, center_x_2], axis=-1)[:, np.newaxis, :]

    center_y_1 = center_xys[:, 1]
    center_y_2 = center_y_1 - speeds * np.sin(headings) * seconds_per_step
    center_y_0 = center_y_1 + speeds * np.sin(headings) * seconds_per_step
    center_y = np.stack([center_y_0, center_y_1, center_y_2], axis=-1)[:, np.newaxis, :]

    length = np.ones((num_agents, num_rollouts), dtype=np.float32) * 4.0
    width = np.ones((num_agents, num_rollouts), dtype=np.float32) * 2.0
    heading = np.broadcast_to(headings[:, np.newaxis, np.newaxis], (num_agents, num_rollouts, num_steps))
    valid = np.ones((num_agents, num_rollouts, num_steps), dtype=bool)

    eval_mask = np.ones(num_agents, dtype=bool)
    ttc = interaction_features.compute_time_to_collision(
        center_x=center_x,
        center_y=center_y,
        length=length,
        width=width,
        heading=heading,
        valid=valid,
        evaluated_object_mask=eval_mask,
        seconds_per_step=seconds_per_step,
    )

    # Both agents should have max TTC since they're moving away
    np.testing.assert_allclose(ttc[:, 0, 1], MAX_TTC_SEC, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])