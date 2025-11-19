"""Tests for map metric features (distance to road edge)."""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from pufferlib.ocean.benchmark import map_metric_features


def plot_test_cases():
    """Visualize all test cases."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Test 1: Sign correctness
    ax = axes[0, 0]
    ax.plot([0, 0], [0, 2], 'b-', linewidth=2, label='Road edge')
    ax.arrow(0, 0.5, 0, 0.8, head_width=0.1, head_length=0.1, fc='b', ec='b')
    ax.plot(-1, 1, 'go', markersize=10, label='P (left, neg)')
    ax.plot(2, 1, 'ro', markersize=10, label='Q (right, pos)')
    ax.set_xlim(-2, 3)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title('Test 1: Sign convention')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Test 2: Magnitude
    ax = axes[0, 1]
    ax.plot([0, 2], [0, 0], 'b-', linewidth=2)
    ax.arrow(0.5, 0, 0.8, 0, head_width=0.1, head_length=0.1, fc='b', ec='b')
    ax.plot(0, 1, 'go', markersize=10, label='P (d=1)')
    ax.plot(3, -1, 'ro', markersize=10, label=f'Q (d={math.sqrt(2):.2f})')
    ax.plot([0, 0], [0, 1], 'g--', alpha=0.5)
    ax.plot([2, 3], [0, -1], 'r--', alpha=0.5)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Test 2: Distance magnitude')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Test 3: Two parallel lines
    ax = axes[0, 2]
    ax.plot([0, 0], [3, -3], 'b-', linewidth=2, label='Left edge')
    ax.plot([2, 2], [-3, 3], 'b-', linewidth=2, label='Right edge')
    ax.arrow(0, 0, 0, -1.5, head_width=0.1, head_length=0.1, fc='b', ec='b')
    ax.arrow(2, 0, 0, 1.5, head_width=0.1, head_length=0.1, fc='b', ec='b')
    ax.axvspan(0, 2, alpha=0.2, color='green', label='On-road')
    ax.set_xlim(-1.5, 4.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.set_title('Test 3: Road corridor')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Test 4: Padded polylines
    ax = axes[1, 0]
    ax.plot([0, 0, 0, 0], [4, 1.5, -1.5, -4], 'b-', linewidth=2, marker='o', markersize=4, label='4-pt line')
    ax.plot([2, 2], [-4, 4], 'r-', linewidth=2, marker='o', markersize=4, label='2-pt line (padded)')
    ax.axvspan(0, 2, alpha=0.2, color='green')
    ax.arrow(0, 0, 0, -1.5, head_width=0.1, head_length=0.1, fc='b', ec='b')
    ax.arrow(2, 0, 0, 1.5, head_width=0.1, head_length=0.1, fc='r', ec='r')
    ax.text(1, 0, 'on-road', ha='center', va='center', fontsize=8)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title('Test 4: Polylines padded to same length')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Test 5: Agent boxes
    ax = axes[1, 1]
    ax.plot([0, 0], [5, -5], 'b-', linewidth=2)
    ax.plot([2, 2], [-5, 5], 'b-', linewidth=2)
    ax.axvspan(0, 2, alpha=0.2, color='green')

    # A0: Fully on-road - center (1, 0), 1m x 0.5m
    rect0 = Rectangle((0.5, -0.25), 1, 0.5, fill=False, edgecolor='green', linewidth=2)
    ax.add_patch(rect0)
    ax.text(1, 0, 'A0', ha='center', va='center', fontsize=8)

    # A1: At boundary - center (1, 0), 2m x 2m (offset in y for visibility)
    rect1 = Rectangle((0, 1.5), 2, 2, fill=False, edgecolor='orange', linewidth=2)
    ax.add_patch(rect1)
    ax.text(1, 2.5, 'A1', ha='center', va='center', fontsize=8)

    # A2: One side off - center (1.75, 0), 1m x 0.5m (offset in y)
    rect2 = Rectangle((1.25, -1.5), 1, 0.5, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect2)
    ax.text(1.75, -1.25, 'A2', ha='center', va='center', fontsize=8)

    # A3: Fully off-road - center (5, 0), 1m x 0.5m
    rect3 = Rectangle((4.5, -0.25), 1, 0.5, fill=False, edgecolor='darkred', linewidth=2)
    ax.add_patch(rect3)
    ax.text(5, 0, 'A3', ha='center', va='center', fontsize=8)

    # A4: Rotated - center (1.5, 0), sqrt(2) x sqrt(2), heading=pi/4
    # Corners at (2.5, 0), (1.5, 1), (0.5, 0), (1.5, -1)
    diamond_x = [2.5, 1.5, 0.5, 1.5, 2.5]
    diamond_y = [-3.5, -2.5, -3.5, -4.5, -3.5]
    ax.plot(diamond_x, diamond_y, 'purple', linewidth=2)
    ax.plot(2.5, -3.5, 'ro', markersize=6)  # off-road corner
    ax.text(1.5, -3.5, 'A4', ha='center', va='center', fontsize=8)

    ax.set_xlim(-1, 6)
    ax.set_ylim(-6, 4)
    ax.set_aspect('equal')
    ax.set_title('Test 5: Agent boxes')
    ax.grid(True, alpha=0.3)

    # Distance field visualization
    ax = axes[1, 2]
    x = np.linspace(-1, 4, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    mesh_xys = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)

    polyline_x = np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32)
    polyline_y = np.array([10.0, -10.0, -10.0, 10.0], dtype=np.float32)
    polyline_lengths = np.array([2, 2], dtype=np.int32)

    polylines, valid = map_metric_features._pad_polylines(
        polyline_x, polyline_y, polyline_lengths
    )
    distances = map_metric_features._compute_signed_distance_to_polylines(
        mesh_xys, polylines, valid
    )
    Z = distances.reshape(X.shape)

    contour = ax.contourf(X, Y, Z, levels=20, cmap='RdYlGn_r')
    ax.contour(X, Y, Z, levels=[0], colors='black', linewidths=2)
    plt.colorbar(contour, ax=ax, label='Signed distance')
    ax.plot([0, 0], [3, -3], 'b-', linewidth=2)
    ax.plot([2, 2], [-3, 3], 'b-', linewidth=2)
    ax.set_title('Distance field (0 = boundary)')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('test_map_metrics.png', dpi=150)
    print(f"Plot saved to test_map_metrics.png")
    plt.close()


def test_signed_distance_correct_sign():
    """Test sign convention: negative = left (port), positive = right (starboard).

         R2
         ^
    P    |     Q
         R1

    P at (-1, 1) should be negative (left of upward line)
    Q at (2, 1) should be positive (right of upward line)
    """
    query_points = np.array([[-1.0, 1.0], [2.0, 1.0]], dtype=np.float32)

    polyline_x = np.array([0.0, 0.0], dtype=np.float32)
    polyline_y = np.array([0.0, 2.0], dtype=np.float32)
    polyline_lengths = np.array([2], dtype=np.int32)

    polylines, valid = map_metric_features._pad_polylines(
        polyline_x, polyline_y, polyline_lengths
    )

    distances = map_metric_features._compute_signed_distance_to_polylines(
        query_points, polylines, valid
    )

    expected = np.array([-1.0, 2.0])
    np.testing.assert_allclose(distances, expected, rtol=1e-5, atol=1e-5)
    print("✓ test_signed_distance_correct_sign passed")


def test_signed_distance_correct_magnitude():
    """Test distance magnitude for points projecting onto and beyond segment.

         P

    R1----->R2

              Q

    P at (0, 1) projects onto segment -> distance = 1.0
    Q at (3, -1) projects beyond R2 -> distance = sqrt(2) to corner
    """
    query_points = np.array([[0.0, 1.0], [3.0, -1.0]], dtype=np.float32)

    polyline_x = np.array([0.0, 2.0], dtype=np.float32)
    polyline_y = np.array([0.0, 0.0], dtype=np.float32)
    polyline_lengths = np.array([2], dtype=np.int32)

    polylines, valid = map_metric_features._pad_polylines(
        polyline_x, polyline_y, polyline_lengths
    )

    distances = map_metric_features._compute_signed_distance_to_polylines(
        query_points, polylines, valid
    )

    expected_abs = np.array([1.0, math.sqrt(2)])
    np.testing.assert_allclose(np.abs(distances), expected_abs, rtol=1e-5, atol=1e-5)
    print("✓ test_signed_distance_correct_magnitude passed")


def test_signed_distance_two_parallel_lines():
    """Test with two parallel lines forming a road corridor.

    Query grid from -1 to 4, two lines at x=0 and x=2.
    Points between lines should be negative (on-road).
    Points outside should be positive (off-road).
    Expected: |x - 1| - 1 (distance to center minus half-width)
    """
    x = np.linspace(-1.0, 4.0, 10, dtype=np.float32)
    mesh_xys = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)

    # Line 1: x=0, pointing down (y: 10 to -10)
    # Line 2: x=2, pointing up (y: -10 to 10)
    polyline_x = np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32)
    polyline_y = np.array([10.0, -10.0, -10.0, 10.0], dtype=np.float32)
    polyline_lengths = np.array([2, 2], dtype=np.int32)

    polylines, valid = map_metric_features._pad_polylines(
        polyline_x, polyline_y, polyline_lengths
    )

    distances = map_metric_features._compute_signed_distance_to_polylines(
        mesh_xys, polylines, valid
    )

    expected = np.abs(mesh_xys[:, 0] - 1.0) - 1.0
    np.testing.assert_allclose(distances, expected, rtol=1e-5, atol=1e-5)
    print("✓ test_signed_distance_two_parallel_lines passed")


def test_signed_distance_with_padding():
    """Test with polylines of different lengths (padded)."""
    x = np.linspace(-1.0, 4.0, 10, dtype=np.float32)
    mesh_xys = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)

    # Line 1: 4 points, Line 2: 2 points (will be padded)
    polyline_x = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 2.0], dtype=np.float32)
    polyline_y = np.array([10.0, 3.0, -3.0, -10.0, -10.0, 10.0], dtype=np.float32)
    polyline_lengths = np.array([4, 2], dtype=np.int32)

    polylines, valid = map_metric_features._pad_polylines(
        polyline_x, polyline_y, polyline_lengths
    )

    distances = map_metric_features._compute_signed_distance_to_polylines(
        mesh_xys, polylines, valid
    )

    expected = np.abs(mesh_xys[:, 0] - 1.0) - 1.0
    np.testing.assert_allclose(distances, expected, rtol=1e-5, atol=1e-5)
    print("✓ test_signed_distance_with_padding passed")


def test_compute_distance_to_road_edge():
    """Test full pipeline with agent boxes."""
    num_agents = 5
    num_steps = 1

    # Road corridor from x=0 to x=2
    # A0: Fully on-road - center (1, 0), 1m x 0.5m, heading=0
    #     Corners x ∈ [0.5, 1.5] → all inside, nearest edge 0.5m away, expected ~ -0.5
    # A1: At boundary - center (1, 0), 2m x 2m, heading=0
    #     Corners x ∈ [0, 2] → exactly at edges, expected ~ 0
    # A2: One side off - center (1.75, 0), 1m x 0.5m, heading=0
    #     Corners x ∈ [1.25, 2.25] → right off by 0.25m, expected ~ 0.25
    # A3: Fully off-road - center (5, 0), 1m x 0.5m, heading=0
    #     Corners x ∈ [4.5, 5.5] → far off, expected ~ 3.5
    # A4: Rotated with one corner off - center (1.5, 0), sqrt(2) x sqrt(2), heading=pi/4
    #     Corners at (2.5, 0), (1.5, 1), (0.5, 0), (1.5, -1)
    #     Corner at (2.5, 0) is 0.5m outside, expected ~ +0.5

    center_x = np.array([[1.0], [1.0], [1.75], [5.0], [1.5]], dtype=np.float32)
    center_y = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    length = np.array([1.0, 2.0, 1.0, 1.0, np.sqrt(2)], dtype=np.float32)
    width = np.array([0.5, 2.0, 0.5, 0.5, np.sqrt(2)], dtype=np.float32)
    heading = np.array([[0.0], [0.0], [0.0], [0.0], [np.pi/4]], dtype=np.float32)
    valid = np.ones((num_agents, num_steps), dtype=bool)

    # Two parallel lines at x=0 and x=2
    polyline_x = np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32)
    polyline_y = np.array([10.0, -10.0, -10.0, 10.0], dtype=np.float32)
    polyline_lengths = np.array([2, 2], dtype=np.int32)

    distances = map_metric_features.compute_distance_to_road_edge(
        center_x=center_x,
        center_y=center_y,
        length=length,
        width=width,
        heading=heading,
        valid=valid,
        polyline_x=polyline_x,
        polyline_y=polyline_y,
        polyline_lengths=polyline_lengths,
    )

    assert distances.shape == (num_agents, num_steps)

    print(f"A0 (fully on-road): {distances[0, 0]:.3f} (expected ~ -0.5)")
    print(f"A1 (at boundary): {distances[1, 0]:.3f} (expected ~ 0)")
    print(f"A2 (one side off): {distances[2, 0]:.3f} (expected ~ 0.25)")
    print(f"A3 (fully off-road): {distances[3, 0]:.3f} (expected ~ 3.5)")
    print(f"A4 (rotated, one corner off): {distances[4, 0]:.3f} (expected ~ 0.5)")

    # A0: fully on-road, corners at x=0.5 and x=1.5, both 0.5m inside road
    assert distances[0, 0] < 0, f"A0 should be on-road (negative), got {distances[0, 0]}"
    np.testing.assert_allclose(distances[0, 0], -0.5, atol=0.1)

    # A1: at boundary, distance ~ 0
    np.testing.assert_allclose(distances[1, 0], 0.0, atol=0.1)

    # A2: one side off by 0.25m
    np.testing.assert_allclose(distances[2, 0], 0.25, atol=0.1)

    # A3: fully off-road
    np.testing.assert_allclose(distances[3, 0], 3.5, atol=0.1)

    # A4: rotated, corner at (2.5, 0) is 0.5m outside
    np.testing.assert_allclose(distances[4, 0], 0.5, atol=0.1)

    print("✓ test_compute_distance_to_road_edge passed")


if __name__ == "__main__":
    print("Running map metric feature tests...\n")
    print("=" * 60)

    try:
        test_signed_distance_correct_sign()
        test_signed_distance_correct_magnitude()
        test_signed_distance_two_parallel_lines()
        test_signed_distance_with_padding()
        test_compute_distance_to_road_edge()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")

        print("\nGenerating visualization...")
        plot_test_cases()

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed:")
        import traceback
        traceback.print_exc()
