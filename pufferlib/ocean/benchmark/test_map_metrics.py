"""Tests for map metric features (distance to road edge)."""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import torch

from pufferlib.ocean.benchmark import map_metric_features


def _tensor(data, dtype=torch.float32):
    """Convenience helper to create torch tensors for the map feature API."""
    return torch.as_tensor(data, dtype=dtype)


def plot_test_cases():
    """Visualize all test cases."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Test 1: Sign correctness
    ax = axes[0, 0]
    ax.plot([0, 0], [0, 2], "b-", linewidth=2, label="Road edge")
    ax.arrow(0, 0.5, 0, 0.8, head_width=0.1, head_length=0.1, fc="b", ec="b")
    ax.plot(-1, 1, "go", markersize=10, label="P (left, neg)")
    ax.plot(2, 1, "ro", markersize=10, label="Q (right, pos)")
    ax.set_xlim(-2, 3)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect("equal")
    ax.set_title("Test 1: Sign convention")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Test 2: Magnitude
    ax = axes[0, 1]
    ax.plot([0, 2], [0, 0], "b-", linewidth=2)
    ax.arrow(0.5, 0, 0.8, 0, head_width=0.1, head_length=0.1, fc="b", ec="b")
    ax.plot(0, 1, "go", markersize=10, label="P (d=1)")
    ax.plot(3, -1, "ro", markersize=10, label=f"Q (d={math.sqrt(2):.2f})")
    ax.plot([0, 0], [0, 1], "g--", alpha=0.5)
    ax.plot([2, 3], [0, -1], "r--", alpha=0.5)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_title("Test 2: Distance magnitude")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Test 3: Two parallel lines
    ax = axes[0, 2]
    ax.plot([0, 0], [3, -3], "b-", linewidth=2, label="Left edge")
    ax.plot([2, 2], [-3, 3], "b-", linewidth=2, label="Right edge")
    ax.arrow(0, 0, 0, -1.5, head_width=0.1, head_length=0.1, fc="b", ec="b")
    ax.arrow(2, 0, 0, 1.5, head_width=0.1, head_length=0.1, fc="b", ec="b")
    ax.axvspan(0, 2, alpha=0.2, color="green", label="On-road")
    ax.set_xlim(-1.5, 4.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_title("Test 3: Road corridor")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Test 4: Padded polylines
    ax = axes[1, 0]
    ax.plot([0, 0, 0, 0], [4, 1.5, -1.5, -4], "b-", linewidth=2, marker="o", markersize=4, label="4-pt line")
    ax.plot([2, 2], [-4, 4], "r-", linewidth=2, marker="o", markersize=4, label="2-pt line (padded)")
    ax.axvspan(0, 2, alpha=0.2, color="green")
    ax.arrow(0, 0, 0, -1.5, head_width=0.1, head_length=0.1, fc="b", ec="b")
    ax.arrow(2, 0, 0, 1.5, head_width=0.1, head_length=0.1, fc="r", ec="r")
    ax.text(1, 0, "on-road", ha="center", va="center", fontsize=8)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_title("Test 4: Polylines padded to same length")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Test 5: Agent boxes
    ax = axes[1, 1]
    ax.plot([0, 0], [5, -5], "b-", linewidth=2)
    ax.plot([2, 2], [-5, 5], "b-", linewidth=2)
    ax.axvspan(0, 2, alpha=0.2, color="green")

    # A0: Fully on-road - center (1, 0), 1m x 0.5m
    rect0 = Rectangle((0.5, -0.25), 1, 0.5, fill=False, edgecolor="green", linewidth=2)
    ax.add_patch(rect0)
    ax.text(1, 0, "A0", ha="center", va="center", fontsize=8)

    # A1: At boundary - center (1, 0), 2m x 2m (offset in y for visibility)
    rect1 = Rectangle((0, 1.5), 2, 2, fill=False, edgecolor="orange", linewidth=2)
    ax.add_patch(rect1)
    ax.text(1, 2.5, "A1", ha="center", va="center", fontsize=8)

    # A2: One side off - center (1.75, 0), 1m x 0.5m (offset in y)
    rect2 = Rectangle((1.25, -1.5), 1, 0.5, fill=False, edgecolor="red", linewidth=2)
    ax.add_patch(rect2)
    ax.text(1.75, -1.25, "A2", ha="center", va="center", fontsize=8)

    # A3: Fully off-road - center (5, 0), 1m x 0.5m
    rect3 = Rectangle((4.5, -0.25), 1, 0.5, fill=False, edgecolor="darkred", linewidth=2)
    ax.add_patch(rect3)
    ax.text(5, 0, "A3", ha="center", va="center", fontsize=8)

    # A4: Rotated - center (1.5, 0), sqrt(2) x sqrt(2), heading=pi/4
    # Corners at (2.5, 0), (1.5, 1), (0.5, 0), (1.5, -1)
    diamond_x = [2.5, 1.5, 0.5, 1.5, 2.5]
    diamond_y = [-3.5, -2.5, -3.5, -4.5, -3.5]
    ax.plot(diamond_x, diamond_y, "purple", linewidth=2)
    ax.plot(2.5, -3.5, "ro", markersize=6)  # off-road corner
    ax.text(1.5, -3.5, "A4", ha="center", va="center", fontsize=8)

    ax.set_xlim(-1, 6)
    ax.set_ylim(-6, 4)
    ax.set_aspect("equal")
    ax.set_title("Test 5: Agent boxes")
    ax.grid(True, alpha=0.3)

    # Hide unused subplot
    axes[1, 2].axis("off")

    # Test 6: Donut road (outer CCW, inner CW)
    ax = axes[2, 0]
    # Outer square
    outer_x = [0, 4, 4, 0, 0]
    outer_y = [0, 0, 4, 4, 0]
    ax.plot(outer_x, outer_y, "b-", linewidth=2)
    # Inner square
    inner_x = [1, 1, 3, 3, 1]
    inner_y = [1, 3, 3, 1, 1]
    ax.plot(inner_x, inner_y, "b-", linewidth=2)
    # Fill road area
    from matplotlib.patches import Polygon

    outer_poly = np.array([[0, 0], [4, 0], [4, 4], [0, 4]])
    inner_poly = np.array([[1, 1], [1, 3], [3, 3], [3, 1]])
    ax.fill(outer_x, outer_y, alpha=0.2, color="green")
    ax.fill(inner_x, inner_y, alpha=1.0, color="white")

    # Direction arrows - outer CCW, inner CW
    ax.arrow(1.5, 0, 1, 0, head_width=0.15, head_length=0.1, fc="b", ec="b")  # outer bottom: right
    ax.arrow(2.5, 1, -1, 0, head_width=0.15, head_length=0.1, fc="b", ec="b")  # inner bottom: left (CW)

    # Test points
    ax.plot(0.5, 2, "go", markersize=10, label="P1 (on road)")
    ax.plot(2, 2, "ro", markersize=10, label="P2 (inside inner)")
    ax.plot(5, 2, "ro", markersize=8)
    ax.plot(2, 0.5, "go", markersize=8)

    ax.text(0.5, 2.3, "P1", ha="center", fontsize=8)
    ax.text(2, 2.3, "P2", ha="center", fontsize=8)
    ax.text(5.2, 2, "P3", ha="left", fontsize=8)
    ax.text(2, 0.2, "P4", ha="center", fontsize=8)

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.set_title("Test 6: Donut road")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Test 7: Triangle with acute corner (cyclic test)
    ax = axes[2, 1]
    # Triangle counterclockwise
    shape_x = [0, 10, 10, 0]
    shape_y = [0, 0, 10, 0]
    ax.plot(shape_x, shape_y, "b-", linewidth=2)
    ax.fill(shape_x, shape_y, alpha=0.2, color="green")

    # Mark the acute corner at (0, 0)
    ax.plot(0, 0, "ko", markersize=10, markerfacecolor="yellow", label="Acute corner")

    # Test point P at (-2, 1) - outside but Seg 0 thinks inside
    ax.plot(-2.0, 1.0, "ro", markersize=10, label="P (-2, 1)")

    # Direction arrows
    ax.arrow(3, 0, 3, 0, head_width=0.4, head_length=0.3, fc="b", ec="b")  # Seg 0: right
    ax.arrow(6, 6, -1.5, -1.5, head_width=0.4, head_length=0.3, fc="b", ec="b")  # Seg 2: down-left (on hypotenuse)

    # Labels
    ax.text(5, -1, "Seg 0", ha="center", fontsize=7)
    ax.text(1.5, 5, "Seg 2", ha="center", fontsize=7)

    ax.set_xlim(-4, 12)
    ax.set_ylim(-2, 12)
    ax.set_aspect("equal")
    ax.set_title("Test 7: Triangle (cyclic corner)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Hide unused subplot
    axes[2, 2].axis("off")

    plt.tight_layout()
    plt.savefig("test_map_metrics.png", dpi=150)
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
    query_points = _tensor([[-1.0, 1.0], [2.0, 1.0]])

    polyline_x = _tensor([0.0, 0.0])
    polyline_y = _tensor([0.0, 2.0])
    polyline_lengths = _tensor([2], dtype=torch.int64)

    polylines, valid = map_metric_features._pad_polylines(polyline_x, polyline_y, polyline_lengths)

    distances = map_metric_features._compute_signed_distance_to_polylines(query_points, polylines, valid).cpu().numpy()

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
    query_points = _tensor([[0.0, 1.0], [3.0, -1.0]])

    polyline_x = _tensor([0.0, 2.0])
    polyline_y = _tensor([0.0, 0.0])
    polyline_lengths = _tensor([2], dtype=torch.int64)

    polylines, valid = map_metric_features._pad_polylines(polyline_x, polyline_y, polyline_lengths)

    distances = map_metric_features._compute_signed_distance_to_polylines(query_points, polylines, valid).cpu().numpy()

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
    mesh_xys_np = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)
    mesh_xys = _tensor(mesh_xys_np)

    # Line 1: x=0, pointing down (y: 10 to -10)
    # Line 2: x=2, pointing up (y: -10 to 10)
    polyline_x = _tensor([0.0, 0.0, 2.0, 2.0])
    polyline_y = _tensor([10.0, -10.0, -10.0, 10.0])
    polyline_lengths = _tensor([2, 2], dtype=torch.int64)

    polylines, valid = map_metric_features._pad_polylines(polyline_x, polyline_y, polyline_lengths)

    distances = map_metric_features._compute_signed_distance_to_polylines(mesh_xys, polylines, valid).cpu().numpy()

    expected = np.abs(mesh_xys_np[:, 0] - 1.0) - 1.0
    np.testing.assert_allclose(distances, expected, rtol=1e-5, atol=1e-5)
    print("✓ test_signed_distance_two_parallel_lines passed")


def test_signed_distance_with_padding():
    """Test with polylines of different lengths (padded)."""
    x = np.linspace(-1.0, 4.0, 10, dtype=np.float32)
    mesh_xys_np = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)
    mesh_xys = _tensor(mesh_xys_np)

    # Line 1: 4 points, Line 2: 2 points (will be padded)
    polyline_x = _tensor([0.0, 0.0, 0.0, 0.0, 2.0, 2.0])
    polyline_y = _tensor([10.0, 3.0, -3.0, -10.0, -10.0, 10.0])
    polyline_lengths = _tensor([4, 2], dtype=torch.int64)

    polylines, valid = map_metric_features._pad_polylines(polyline_x, polyline_y, polyline_lengths)

    distances = map_metric_features._compute_signed_distance_to_polylines(mesh_xys, polylines, valid).cpu().numpy()

    expected = np.abs(mesh_xys_np[:, 0] - 1.0) - 1.0
    np.testing.assert_allclose(distances, expected, rtol=1e-5, atol=1e-5)
    print("✓ test_signed_distance_with_padding passed")


def test_cyclic_polyline():
    """Test with a cyclic polyline (closed square boundary).

    Square road boundary: corners at (0,0), (2,0), (2,2), (0,2), back to (0,0)
    Winding order: counterclockwise (inside = on-road)

    Points:
    - P1 at (1, 1): center of square, should be negative (on-road)
    - P2 at (3, 1): outside right edge, should be positive (off-road)
    - P3 at (1, 3): outside top edge, should be positive (off-road)
    - P4 at (-1, 1): outside left edge, should be positive (off-road)
    - P5 at (2, 2): exactly at corner, should be ~0
    """
    query_points = _tensor(
        [
            [1.0, 1.0],  # P1: center
            [3.0, 1.0],  # P2: outside right
            [1.0, 3.0],  # P3: outside top
            [-1.0, 1.0],  # P4: outside left
            [2.0, 2.0],  # P5: at corner
        ]
    )

    # Counterclockwise square: (0,0) -> (2,0) -> (2,2) -> (0,2) -> (0,0)
    polyline_x = _tensor([0.0, 2.0, 2.0, 0.0, 0.0])
    polyline_y = _tensor([0.0, 0.0, 2.0, 2.0, 0.0])
    polyline_lengths = _tensor([5], dtype=torch.int64)

    polylines, valid = map_metric_features._pad_polylines(polyline_x, polyline_y, polyline_lengths)

    distances = map_metric_features._compute_signed_distance_to_polylines(query_points, polylines, valid).cpu().numpy()

    print(f"P1 (center): {distances[0]:.3f} (expected ~ -1.0)")
    print(f"P2 (outside right): {distances[1]:.3f} (expected ~ +1.0)")
    print(f"P3 (outside top): {distances[2]:.3f} (expected ~ +1.0)")
    print(f"P4 (outside left): {distances[3]:.3f} (expected ~ +1.0)")
    print(f"P5 (at corner): {distances[4]:.3f} (expected ~ 0.0)")

    # P1: inside square, distance to nearest edge is 1.0
    assert distances[0] < 0, f"P1 should be on-road (negative), got {distances[0]}"
    np.testing.assert_allclose(distances[0], -1.0, atol=0.1)

    # P2, P3, P4: outside square by 1.0
    assert distances[1] > 0, f"P2 should be off-road (positive), got {distances[1]}"
    assert distances[2] > 0, f"P3 should be off-road (positive), got {distances[2]}"
    assert distances[3] > 0, f"P4 should be off-road (positive), got {distances[3]}"
    np.testing.assert_allclose(distances[1], 1.0, atol=0.1)
    np.testing.assert_allclose(distances[2], 1.0, atol=0.1)
    np.testing.assert_allclose(distances[3], 1.0, atol=0.1)

    # P5: at corner
    np.testing.assert_allclose(distances[4], 0.0, atol=0.1)

    print("✓ test_cyclic_polyline passed")


# NOTE: I made this test to understand why it is needed to handle cyclic polylines specially.
def test_cyclic_seam():
    """Test acute corner tie-breaking using a clean Triangle.

    Shape: Triangle (0,0) -> (10,0) -> (10,10) -> (0,0)
    Winding: Counter-Clockwise (Left is Inside).
    Corner at (0,0) is Acute (45 degrees).
    """
    # Query Point P (-2, 1)
    # - Physically: To the left of the diagonal hypotenuse -> OUTSIDE.
    # - To Segment 0 (Bottom): Above y=0 -> INSIDE (The Blind Spot).
    query_points = _tensor([[-2.0, 1.0]])

    # The Triangle
    polyline_x = _tensor([0.0, 10.0, 10.0, 0.0])
    polyline_y = _tensor([0.0, 0.0, 10.0, 0.0])
    polyline_lengths = _tensor([4], dtype=torch.int64)

    polylines, valid = map_metric_features._pad_polylines(polyline_x, polyline_y, polyline_lengths)

    distances = map_metric_features._compute_signed_distance_to_polylines(query_points, polylines, valid).cpu().numpy()

    # Distance to vertex (0,0) is sqrt(2^2 + 1^2) = sqrt(5)
    expected = np.sqrt(5)
    print(f"P at (-2, 1): {distances[0]:.3f} (expected ~ +{expected:.3f})")

    assert distances[0] > 0, f"P should be OFF-ROAD (Positive), but got {distances[0]}"
    np.testing.assert_allclose(distances[0], expected, atol=0.01)

    print("✓ test_cyclic_seam passed (Clean Triangle)")


def test_donut_road():
    """Test with a donut-shaped road (outer CCW, inner CW).

    Outer square: (0,0) -> (4,0) -> (4,4) -> (0,4) -> (0,0) [counterclockwise]
    Inner square: (1,1) -> (1,3) -> (3,3) -> (3,1) -> (1,1) [clockwise]

    Road is the region between the two squares (width=1 on each side).

    Points:
    - P1 at (0.5, 2): on road (between outer and inner), should be negative
    - P2 at (2, 2): inside inner square (off-road), should be positive
    - P3 at (5, 2): outside outer square (off-road), should be positive
    - P4 at (2, 0.5): on road (between outer and inner), should be negative
    """
    query_points = _tensor(
        [
            [0.5, 2.0],  # P1: on road (left side)
            [2.0, 2.0],  # P2: inside inner square
            [5.0, 2.0],  # P3: outside outer square
            [2.0, 0.5],  # P4: on road (bottom side)
        ]
    )

    # Outer square: counterclockwise
    outer_x = _tensor([0.0, 4.0, 4.0, 0.0, 0.0])
    outer_y = _tensor([0.0, 0.0, 4.0, 4.0, 0.0])

    # Inner square: clockwise (opposite winding)
    inner_x = _tensor([1.0, 1.0, 3.0, 3.0, 1.0])
    inner_y = _tensor([1.0, 3.0, 3.0, 1.0, 1.0])

    polyline_x = torch.cat([outer_x, inner_x])
    polyline_y = torch.cat([outer_y, inner_y])
    polyline_lengths = _tensor([5, 5], dtype=torch.int64)

    polylines, valid = map_metric_features._pad_polylines(polyline_x, polyline_y, polyline_lengths)

    distances = map_metric_features._compute_signed_distance_to_polylines(query_points, polylines, valid).cpu().numpy()

    print(f"P1 (on road, left): {distances[0]:.3f} (expected ~ -0.5)")
    print(f"P2 (inside inner): {distances[1]:.3f} (expected ~ +1.0)")
    print(f"P3 (outside outer): {distances[2]:.3f} (expected ~ +1.0)")
    print(f"P4 (on road, bottom): {distances[3]:.3f} (expected ~ -0.5)")

    # P1: on road, 0.5m from outer edge
    assert distances[0] < 0, f"P1 should be on-road (negative), got {distances[0]}"
    np.testing.assert_allclose(distances[0], -0.5, atol=0.1)

    # P2: inside inner square, 1m from inner edge
    assert distances[1] > 0, f"P2 should be off-road (positive), got {distances[1]}"
    np.testing.assert_allclose(distances[1], 1.0, atol=0.1)

    # P3: outside outer square, 1m from outer edge
    assert distances[2] > 0, f"P3 should be off-road (positive), got {distances[2]}"
    np.testing.assert_allclose(distances[2], 1.0, atol=0.1)

    # P4: on road, 0.5m from outer edge
    assert distances[3] < 0, f"P4 should be on-road (negative), got {distances[3]}"
    np.testing.assert_allclose(distances[3], -0.5, atol=0.1)

    print("✓ test_donut_road passed")


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

    center_x = _tensor([[1.0], [1.0], [1.75], [5.0], [1.5]])
    center_y = _tensor([[0.0], [0.0], [0.0], [0.0], [0.0]])
    length = _tensor([1.0, 2.0, 1.0, 1.0, np.sqrt(2)])
    width = _tensor([0.5, 2.0, 0.5, 0.5, np.sqrt(2)])
    heading = _tensor([[0.0], [0.0], [0.0], [0.0], [np.pi / 4]])
    valid = torch.ones((num_agents, num_steps), dtype=torch.bool)

    # Two parallel lines at x=0 and x=2
    polyline_x = _tensor([0.0, 0.0, 2.0, 2.0])
    polyline_y = _tensor([10.0, -10.0, -10.0, 10.0])
    polyline_lengths = _tensor([2, 2], dtype=torch.int64)

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

    distances_np = distances.cpu().numpy()

    assert distances.shape == (num_agents, num_steps)

    print(f"A0 (fully on-road): {distances_np[0, 0]:.3f} (expected ~ -0.5)")
    print(f"A1 (at boundary): {distances_np[1, 0]:.3f} (expected ~ 0)")
    print(f"A2 (one side off): {distances_np[2, 0]:.3f} (expected ~ 0.25)")
    print(f"A3 (fully off-road): {distances_np[3, 0]:.3f} (expected ~ 3.5)")
    print(f"A4 (rotated, one corner off): {distances_np[4, 0]:.3f} (expected ~ 0.5)")

    # A0: fully on-road, corners at x=0.5 and x=1.5, both 0.5m inside road
    assert distances_np[0, 0] < 0, f"A0 should be on-road (negative), got {distances_np[0, 0]}"
    np.testing.assert_allclose(distances_np[0, 0], -0.5, atol=0.1)

    # A1: at boundary, distance ~ 0
    np.testing.assert_allclose(distances_np[1, 0], 0.0, atol=0.1)

    # A2: one side off by 0.25m
    np.testing.assert_allclose(distances_np[2, 0], 0.25, atol=0.1)

    # A3: fully off-road
    np.testing.assert_allclose(distances_np[3, 0], 3.5, atol=0.1)

    # A4: rotated, corner at (2.5, 0) is 0.5m outside
    np.testing.assert_allclose(distances_np[4, 0], 0.5, atol=0.1)

    print("✓ test_compute_distance_to_road_edge passed")


if __name__ == "__main__":
    print("Running map metric feature tests...\n")
    print("=" * 60)

    try:
        print("Generating visualization first...")
        plot_test_cases()
        print()

        test_signed_distance_correct_sign()
        test_signed_distance_correct_magnitude()
        test_signed_distance_two_parallel_lines()
        test_signed_distance_with_padding()
        test_cyclic_polyline()
        test_cyclic_seam()
        test_donut_road()
        test_compute_distance_to_road_edge()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed:")
        import traceback

        traceback.print_exc()
