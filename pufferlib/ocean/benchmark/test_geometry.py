import numpy as np
import torch

from pufferlib.ocean.benchmark import interaction_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor(value, *, dtype=torch.float32):
    return torch.tensor(value, dtype=dtype, device=DEVICE)


def test_box_distance_calculations():
    """Test with manually designed box configurations.

    Box 0: x=2, y=1, heading=0, length=2, width=1
    Box 1: x=4.5, y=2.5, heading=0, l=2, w=1
    Box 2: x=1.5, y=2, heading=pi/2, l=2, w=1
    Box 3: x=3.5, y=0, heading=pi/4, l=sqrt(2), w=sqrt(2)

    Expected distances from Box 0 to others:
    - Box 0 to Box 0: EXTREMELY_LARGE_DISTANCE (self-distance masked)
    - Box 0 to Box 1: 1/sqrt(2) ≈ 0.707
    - Box 0 to Box 2: -0.5 (overlapping)
    - Box 0 to Box 3: 0.0 (touching)
    """

    center_x_np = np.array([[[2.0]], [[4.5]], [[1.5]], [[3.5]]])
    center_y_np = np.array([[[1.0]], [[2.5]], [[2.0]], [[0.0]]])
    length_np = np.array([[2.0], [2.0], [2.0], [np.sqrt(2)]])
    width_np = np.array([[1.0], [1.0], [1.0], [np.sqrt(2)]])
    heading_np = np.array([[[0.0]], [[0.0]], [[np.pi / 2]], [[np.pi / 4]]])
    valid_np = np.ones((4, 1, 1), dtype=bool)

    print("Test: Handpicked box configurations")
    print("=" * 60)
    print(f"\nArray shapes:")
    print(f"  center_x: {center_x_np.shape} = (num_agents=4, num_rollouts=1, num_timesteps=1)")
    print(f"  length:   {length_np.shape} = (num_agents=4, num_rollouts=1)")
    print(f"  This test uses: 4 agents, 1 rollout, 1 timestep")

    center_x = _tensor(center_x_np)
    center_y = _tensor(center_y_np)
    length = _tensor(length_np)
    width = _tensor(width_np)
    heading = _tensor(heading_np)
    valid = _tensor(valid_np, dtype=torch.bool)
    eval_mask = _tensor(np.ones(4, dtype=bool), dtype=torch.bool)

    signed_distances = interaction_features.compute_signed_distances(
        center_x=center_x,
        center_y=center_y,
        length=length,
        width=width,
        heading=heading,
        valid=valid,
        evaluated_object_mask=eval_mask,
        corner_rounding_factor=0.0,
    )
    signed_distances = signed_distances.cpu().numpy()

    expected_distances = np.array([interaction_features.EXTREMELY_LARGE_DISTANCE, 1.0 / np.sqrt(2), -0.5, 0.0])

    print("\nBox configurations:")
    print("  Box 0: center=(2.0, 1.0), heading=0°, length=2.0, width=1.0")
    print("  Box 1: center=(4.5, 2.5), heading=0°, length=2.0, width=1.0")
    print("  Box 2: center=(1.5, 2.0), heading=90°, length=2.0, width=1.0")
    print(f"  Box 3: center=(3.5, 0.0), heading=45°, length={np.sqrt(2):.3f}, width={np.sqrt(2):.3f}")

    print("\nDistances from Box 0:")
    print(f"  To itself (Box 0): {signed_distances[0, 0, 0, 0]:.6e} (expected: {expected_distances[0]:.6e})")
    print(f"  To Box 1: {signed_distances[0, 1, 0, 0]:.6f} (expected: {expected_distances[1]:.6f})")
    print(f"  To Box 2: {signed_distances[0, 2, 0, 0]:.6f} (expected: {expected_distances[2]:.6f})")
    print(f"  To Box 3: {signed_distances[0, 3, 0, 0]:.6f} (expected: {expected_distances[3]:.6f})")

    atol = 0.01
    assert signed_distances[0, 0, 0, 0] >= interaction_features.EXTREMELY_LARGE_DISTANCE - 1, (
        f"Self-distance should be EXTREMELY_LARGE_DISTANCE, got {signed_distances[0, 0, 0, 0]}"
    )

    assert np.abs(signed_distances[0, 1, 0, 0] - expected_distances[1]) < atol, (
        f"Distance to Box 1 should be {expected_distances[1]:.6f}, got {signed_distances[0, 1, 0, 0]:.6f}"
    )

    assert np.abs(signed_distances[0, 2, 0, 0] - expected_distances[2]) < atol, (
        f"Distance to Box 2 should be {expected_distances[2]:.6f}, got {signed_distances[0, 2, 0, 0]:.6f}"
    )

    assert np.abs(signed_distances[0, 3, 0, 0] - expected_distances[3]) < atol, (
        f"Distance to Box 3 should be {expected_distances[3]:.6f}, got {signed_distances[0, 3, 0, 0]:.6f}"
    )

    print("  ✓ Test passed!")


def test_invalid_objects():
    """Test invalid objects using handpicked box 1, but marked invalid."""

    center_x_np = np.array([[[2.0]], [[4.5]]])
    center_y_np = np.array([[[1.0]], [[2.5]]])
    length_np = np.array([[2.0], [2.0]])
    width_np = np.array([[1.0], [1.0]])
    heading_np = np.array([[[0.0]], [[0.0]]])
    valid_np = np.array([[True], [False]], dtype=bool)[:, :, np.newaxis]

    print("\nTest: Invalid objects")
    print(f"Array shapes: {center_x_np.shape} = (num_agents=2, num_rollouts=1, num_timesteps=1)")
    print("This test uses: 2 agents (Box 0 and Box 1), 1 rollout, 1 timestep")
    print("Box 1 is marked as invalid")

    center_x = _tensor(center_x_np)
    center_y = _tensor(center_y_np)
    length = _tensor(length_np)
    width = _tensor(width_np)
    heading = _tensor(heading_np)
    valid = _tensor(valid_np, dtype=torch.bool)
    eval_mask = _tensor(np.ones(2, dtype=bool), dtype=torch.bool)

    distances = interaction_features.compute_distance_to_nearest_object(
        center_x=center_x,
        center_y=center_y,
        length=length,
        width=width,
        heading=heading,
        valid=valid,
        evaluated_object_mask=eval_mask,
    )
    distances = distances.cpu().numpy()
    print(f"  Agent 0 distance (box 1 invalid): {distances[0, 0, 0]}")
    print(f"  Expected: {interaction_features.EXTREMELY_LARGE_DISTANCE}")

    assert distances[0, 0, 0] >= interaction_features.EXTREMELY_LARGE_DISTANCE - 1, (
        f"Invalid object not handled correctly! Got {distances[0, 0, 0]}"
    )

    print("  ✓ Test passed!")


def test_multiple_rollouts():
    """Test with handpicked boxes across multiple rollouts."""

    center_x_np = np.array(
        [
            [[2.0], [2.0]],
            [[4.5], [5.0]],
        ]
    )
    center_y_np = np.array(
        [
            [[1.0], [1.0]],
            [[2.5], [3.0]],
        ]
    )
    length_np = np.ones((2, 2)) * 2.0
    width_np = np.ones((2, 2)) * 1.0
    heading_np = np.zeros_like(center_x_np)
    valid_np = np.ones((2, 2, 1), dtype=bool)

    print("\nTest: Multiple rollouts")
    print(f"Array shapes: {center_x_np.shape} = (num_agents=2, num_rollouts=2, num_timesteps=1)")
    print("This test uses: 2 agents (Box 0 and Box 1), 2 rollouts, 1 timestep")
    print("Rollout 0: Box 1 at (4.5, 2.5) - closer to Box 0")
    print("Rollout 1: Box 1 at (5.0, 3.0) - further from Box 0")

    center_x = _tensor(center_x_np)
    center_y = _tensor(center_y_np)
    length = _tensor(length_np)
    width = _tensor(width_np)
    heading = _tensor(heading_np)
    valid = _tensor(valid_np, dtype=torch.bool)
    eval_mask = _tensor(np.ones(2, dtype=bool), dtype=torch.bool)

    distances = interaction_features.compute_distance_to_nearest_object(
        center_x=center_x,
        center_y=center_y,
        length=length,
        width=width,
        heading=heading,
        valid=valid,
        evaluated_object_mask=eval_mask,
        corner_rounding_factor=0.0,
    )
    distances = distances.cpu().numpy()
    print(f"  Agent 0, rollout 0: {distances[0, 0, 0]:.2f}m")
    print(f"  Agent 0, rollout 1: {distances[0, 1, 0]:.2f}m")
    print("  Distances should vary across rollouts")

    assert distances[0, 0, 0] != distances[0, 1, 0], "Distances should differ across rollouts"

    print("  ✓ Test passed!")


def test_multiple_timesteps():
    """Test with handpicked boxes across multiple timesteps."""

    center_x_np = np.array(
        [
            [[2.0, 2.0, 2.0]],
            [[4.5, 5.0, 6.0]],
        ]
    )
    center_y_np = np.array(
        [
            [[1.0, 1.0, 1.0]],
            [[2.5, 2.5, 2.5]],
        ]
    )
    length_np = np.ones((2, 1)) * 2.0
    width_np = np.ones((2, 1)) * 1.0
    heading_np = np.zeros_like(center_x_np)
    valid_np = np.ones((2, 1, 3), dtype=bool)

    print("\nTest: Multiple timesteps")
    print(f"Array shapes: {center_x_np.shape} = (num_agents=2, num_rollouts=1, num_timesteps=3)")
    print("This test uses: 2 agents (Box 0 and Box 1), 1 rollout, 3 timesteps")
    print("Box 0 stays at (2.0, 1.0) across all timesteps")
    print("Box 1 moves away: t=0 at (4.5, 2.5), t=1 at (5.0, 2.5), t=2 at (6.0, 2.5)")

    center_x = _tensor(center_x_np)
    center_y = _tensor(center_y_np)
    length = _tensor(length_np)
    width = _tensor(width_np)
    heading = _tensor(heading_np)
    valid = _tensor(valid_np, dtype=torch.bool)
    eval_mask = _tensor(np.ones(2, dtype=bool), dtype=torch.bool)

    distances = interaction_features.compute_distance_to_nearest_object(
        center_x=center_x,
        center_y=center_y,
        length=length,
        width=width,
        heading=heading,
        valid=valid,
        evaluated_object_mask=eval_mask,
        corner_rounding_factor=0.0,
    )
    distances = distances.cpu().numpy()

    print(f"  Agent 0, timestep 0: {distances[0, 0, 0]:.2f}m")
    print(f"  Agent 0, timestep 1: {distances[0, 0, 1]:.2f}m")
    print(f"  Agent 0, timestep 2: {distances[0, 0, 2]:.2f}m")
    print("  Distances should increase as Box 1 moves away")

    assert distances[0, 0, 0] < distances[0, 0, 1] < distances[0, 0, 2], (
        "Distances should increase over time as Box 1 moves away"
    )

    print("  ✓ Test passed!")


def test_rollout_isolation():
    """Test that agents from different rollouts never interact."""

    center_x_np = np.array([[[2.0], [100.0]], [[4.5], [2.0]]])
    center_y_np = np.array([[[1.0], [1.0]], [[2.5], [100.0]]])
    length_np = np.ones((2, 2)) * 2.0
    width_np = np.ones((2, 2)) * 1.0
    heading_np = np.zeros_like(center_x_np)
    valid_np = np.ones((2, 2, 1), dtype=bool)

    print("\nTest: Rollout isolation")
    print(f"Array shapes: {center_x_np.shape} = (num_agents=2, num_rollouts=2, num_timesteps=1)")
    print("This test uses: 2 agents (Box 0 and Box 1), 2 rollouts, 1 timestep")
    print("Rollout 0: Box 0 at (2.0, 1.0), Box 1 at (4.5, 2.5) - close together")
    print("Rollout 1: Box 0 at (100.0, 1.0), Box 1 at (2.0, 100.0) - far apart")
    print("Each agent should only see the other agent in its own rollout")

    center_x = _tensor(center_x_np)
    center_y = _tensor(center_y_np)
    length = _tensor(length_np)
    width = _tensor(width_np)
    heading = _tensor(heading_np)
    valid = _tensor(valid_np, dtype=torch.bool)
    eval_mask = _tensor(np.ones(2, dtype=bool), dtype=torch.bool)

    distances = interaction_features.compute_distance_to_nearest_object(
        center_x=center_x,
        center_y=center_y,
        length=length,
        width=width,
        heading=heading,
        valid=valid,
        evaluated_object_mask=eval_mask,
        corner_rounding_factor=0.0,
    )
    distances = distances.cpu().numpy()
    print(f"  Agent 0, rollout 0: {distances[0, 0, 0]:.2f}m (should see Agent 1 rollout 0)")
    print(f"  Agent 0, rollout 1: {distances[0, 1, 0]:.2f}m (should see Agent 1 rollout 1)")

    assert distances[0, 0, 0] < 10.0, "Agent 0 rollout 0 should see nearby Agent 1 rollout 0"
    assert distances[0, 1, 0] > 90.0, "Agent 0 rollout 1 should see distant Agent 1 rollout 1"

    print("  ✓ Test passed! Rollouts are properly isolated.")


if __name__ == "__main__":
    print("Running geometry and distance computation tests...\n")
    print("=" * 60)

    try:
        test_box_distance_calculations()
        test_invalid_objects()
        test_multiple_timesteps()
        test_multiple_rollouts()
        test_rollout_isolation()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
