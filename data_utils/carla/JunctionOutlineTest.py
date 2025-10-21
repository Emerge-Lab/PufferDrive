import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.spatial import Delaunay

from shapely.geometry import Point, LineString, MultiPolygon, Polygon
from shapely.ops import unary_union, polygonize, snap


def create_y_junction_polylines():
    """
    Defines and returns a list of numpy arrays representing the polylines
    for the edges of a Y-junction road.

    NOTE: The 'bottom_edge' is commented out to demonstrate the fallback logic.
    Uncomment it to see the precise polygonization.
    """
    # 1. Outer edge of the left arm
    y_left_outer = np.linspace(10, 20, 20)
    x_left_outer = -2 - 8 * ((y_left_outer - 10) / 10) ** 2
    left_arm_outer = np.vstack([x_left_outer, y_left_outer]).T
    left_stem = np.array([[-2, 10], [-2, 0]])
    outer_left_edge = np.vstack([left_stem, left_arm_outer])

    # 2. Outer edge of the right arm
    y_right_outer = np.linspace(10, 20, 20)
    x_right_outer = 2 + 8 * ((y_right_outer - 10) / 10) ** 2
    right_arm_outer = np.vstack([x_right_outer, y_right_outer]).T
    right_stem = np.array([[2, 10], [2, 0]])
    outer_right_edge = np.vstack([right_stem, right_arm_outer])

    # 3. The central island (this will become a hole in the polygon)
    # This must be a closed loop.
    t = np.linspace(0, 2 * np.pi, 20)
    center_island = np.array([1.5 * np.sin(t) + 0, 1.5 * np.cos(t) + 12.5]).T

    # 4. Closing polylines to ensure the shape can be polygonized
    bottom_edge = np.array([[-2, 0], [2, 0]])
    left_top_edge = np.array([left_arm_outer[-1], [-2, 20]])
    right_top_edge = np.array([right_arm_outer[-1], [2, 20]])

    # Stopping line between left and right stems
    stopping_line = np.array([[-1.91, 10], [1.91, 10]])

    # An inner edge to connect the central island to the top openings
    inner_divider = np.array([[-2, 20], [0, 14], [2, 20]])

    # --- NEW: Add extra polylines inside the final shape ---
    # These lines will be merged but won't change the final polygon outline
    # because they don't create new closed boundaries.
    center_dashed_line = np.array([[0, 0], [0, 10]])
    left_lane_divider = np.array([[-1, 10], [-4, 18]])
    right_lane_divider = np.array([[1, 10], [4, 18]])

    return [
        outer_left_edge,
        outer_right_edge,
        center_island,
        bottom_edge,  # <-- Intentionally commented out to force an open shape
        left_top_edge,
        right_top_edge,
        inner_divider,
        # Add the new lines to the list
        stopping_line,
        center_dashed_line,
        left_lane_divider,
        right_lane_divider,
    ]


def create_alpha_shape(merged_lines, alpha):
    """
    Creates a concave hull (alpha shape) from a set of lines.
    This is a much more accurate fallback than a convex hull.
    """
    # 1. Collect all unique points from the lines
    all_coords = np.vstack([np.array(line.coords) for line in getattr(merged_lines, "geoms", [merged_lines])])
    unique_coords = np.unique(all_coords, axis=0)

    if len(unique_coords) < 4:
        # Not enough points for a meaningful shape, return convex hull as last resort
        return merged_lines.convex_hull, "convex_hull"

    # 2. Create Delaunay triangulation
    try:
        tri = Delaunay(unique_coords)
    except Exception:
        # Delaunay can fail on co-linear points
        return merged_lines.convex_hull, "convex_hull"

    # 3. Filter edges based on the alpha parameter
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            p1_idx, p2_idx = simplex[i], simplex[(i + 1) % 3]
            p1 = unique_coords[p1_idx]
            p2 = unique_coords[p2_idx]
            dist = np.linalg.norm(p1 - p2)
            if dist < alpha:
                edges.add(tuple(sorted((p1_idx, p2_idx))))

    # 4. Create LineStrings from the valid edges and polygonize
    boundary_lines = [LineString([unique_coords[edge[0]], unique_coords[edge[1]]]) for edge in edges]

    alpha_polygons = list(polygonize(unary_union(boundary_lines)))

    if not alpha_polygons:
        return merged_lines.convex_hull, "convex_hull"

    return MultiPolygon(alpha_polygons), "alpha_shape"


def snap_lines_to_grid(lines, tolerance=0.01):
    """
    Snap line endpoints to a grid to ensure proper connections.
    """
    snapped_lines = []
    for line in lines:
        coords = np.array(line.coords)
        # Round coordinates to grid
        coords = np.round(coords / tolerance) * tolerance
        snapped_lines.append(LineString(coords))
    return snapped_lines


def snap_lines_together(lines, tolerance=0.1):
    """
    Snap lines to each other to ensure connectivity.
    """
    if not lines:
        return lines

    # Start with first line as reference
    snapped = [lines[0]]

    for line in lines[1:]:
        # Snap to all previously processed lines
        for ref_line in snapped:
            line = snap(line, ref_line, tolerance)
        snapped.append(line)

    return snapped


def fuse_polygons_remove_internal_edges(list_of_polylines, snap_tolerance=0.001):
    """
    Improved version with snapping and cleaning.
    """
    if not list_of_polylines:
        return None, "empty"

    # Step 1: Create LineStrings
    lines = [LineString(poly) for poly in list_of_polylines if len(poly) >= 2]

    if not lines:
        return None, "empty"

    # Step 2: Snap to grid
    snapped_lines = []
    for line in lines:
        coords = np.array(line.coords)
        if snap_tolerance > 0:
            coords = np.round(coords / snap_tolerance) * snap_tolerance
        snapped_lines.append(LineString(coords))

    # Step 3: Merge and polygonize
    merged_lines = unary_union(snapped_lines)
    polygons = list(polygonize(merged_lines))

    if polygons:
        # Step 4: Fuse and clean
        fused = unary_union(polygons).buffer(0)
        return fused, "precise"

    # Fallback to alpha shape
    return create_alpha_shape(merged_lines, alpha=snap_tolerance * 30)[0], "alpha_shape"


def fuse_existing_polygons(polygons):
    """
    Fuses a list of Shapely Polygon objects into a single polygon/multipolygon.
    Removes internal edges by taking the union.

    Args:
        polygons (list): List of Shapely Polygon objects

    Returns:
        Polygon or MultiPolygon: The fused geometry
    """
    if not polygons:
        return None

    # Simply take the union and clean
    fused = unary_union(polygons).buffer(0)
    return fused


def plot_junction(polylines, final_shape, test_points, results, method="precise", junction_id="Y-Junction"):
    """
    Visualizes the Y-junction using Matplotlib.

    Args:
        polylines (list): List of original np.array polylines.
        final_shape (Polygon/MultiPolygon): The final Shapely geometry.
        test_points (list): List of (x, y) tuples for query points.
        results (list): List of booleans indicating if a point is inside.
        method (str): The method used to generate the final_shape.
    """
    fig, ax = plt.subplots(figsize=(10, 12))

    # 1. Plot the original polylines as dashed lines
    for poly in polylines:
        ax.plot(
            poly[:, 0],
            poly[:, 1],
            "k--",
            lw=1.0,
            alpha=0.7,
            label="Original Polylines" if "Original" not in [l.get_label() for l in ax.get_lines()] else "",
        )

    # 2. Plot the final polygon shape (handling holes correctly)
    if final_shape and not final_shape.is_empty:
        # Set color and label based on the creation method
        if method == "precise":
            shape_label = "Polygonized Road Surface"
            face_color, edge_color = "lightblue", "blue"
        elif method == "alpha_shape":
            shape_label = "Concave Hull (Alpha Shape)"
            face_color, edge_color = "mediumpurple", "purple"
        else:  # convex_hull or other fallbacks
            shape_label = "Force-Closed Hull"
            face_color, edge_color = "lightcoral", "red"

        for geom in getattr(final_shape, "geoms", [final_shape]):
            exterior_coords = np.array(geom.exterior.coords)
            interior_coords = [np.array(interior.coords) for interior in geom.interiors]

            vertices = np.vstack([exterior_coords] + interior_coords)
            codes = [Path.MOVETO] + [Path.LINETO] * (len(exterior_coords) - 1)
            for interior in interior_coords:
                codes += [Path.MOVETO] + [Path.LINETO] * (len(interior) - 1)

            path = Path(vertices, codes)
            patch = PathPatch(path, facecolor=face_color, edgecolor=edge_color, alpha=0.6, lw=1.5, label=shape_label)
            ax.add_patch(patch)

    # 3. Plot the test points
    points_array = np.array(test_points)
    colors = ["green" if r else "red" for r in results]
    markers = ["o" if r else "x" for r in results]

    for point, color, marker in zip(points_array, colors, markers):
        label = "Point Inside" if marker == "o" else "Point Outside"
        ax.scatter(
            point[0],
            point[1],
            c=color,
            marker=marker,
            s=100,
            zorder=5,
            label=label if label not in [sc.get_label() for sc in ax.collections] else "",
        )

    ax.set_title(f"Junction {junction_id} Outline Analysis", fontsize=16)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.show()


def main():
    """
    Main function to run the analysis and visualization.
    """
    list_of_polylines = create_y_junction_polylines()

    lines = [LineString(poly) for poly in list_of_polylines]
    lines = snap_lines_together(lines, tolerance=0.1)
    snapped_polylines = [np.array(line.coords) for line in lines]
    merged_lines = unary_union(lines)
    polygons = list(polygonize(merged_lines))

    road_surface = None
    method_used = "precise"

    if polygons:
        print("Success: The polylines formed a closed shape. Creating precise polygon.")
        # CHANGED: Fuse all polygons to remove internal edges
        road_surface = fuse_existing_polygons(polygons)
        print(f"Number of polygons before fusion: {len(polygons)}")
        print(f"Number of polygons after fusion: {len(getattr(road_surface, 'geoms', [road_surface]))}")
    else:
        print("Warning: Polylines do not form a closed shape.")
        print("--> Fallback: Creating a force-closed polygon using an alpha shape (concave hull).")

        # Auto-calculate a reasonable alpha value (e.g., 3x the average segment length)
        total_length, num_segments = 0, 0
        for line in lines:
            coords = np.array(line.coords)
            if len(coords) > 1:
                total_length += np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
                num_segments += len(coords) - 1

        avg_segment_length = total_length / num_segments if num_segments > 0 else 1.0
        alpha = avg_segment_length * 3.0  # This is a key tuning parameter
        print(f"Using alpha = {alpha:.2f} for concave hull generation.")

        road_surface, method_used = create_alpha_shape(merged_lines, alpha=alpha)

    if not road_surface or road_surface.is_empty:
        print("Error: Could not create any valid polygon from the input lines.")
        return

    print(f"Final shape created using '{method_used}' method.")
    print(f"Final shape is valid: {road_surface.is_valid}")
    print(f"Area: {road_surface.area:.2f} square units")

    test_points = [
        (-2, 10),  # On the stopping line (should now be INSIDE)
        (2, 10),  # On the stopping line
        (0, 10),  # On the stopping line
        (2, 5),  # Inside
        (1.5, 5.0),  # Inside the stem
        (-5, 15.0),  # Inside the left arm
        (8, 17.0),  # Inside the right arm
        (0, 12.5),  # Inside the central island (should be FALSE - it's a hole)
        (0, 1),  # Inside
        (0, 18),  # Outside (between arms)
        (-10, 10),  # Outside, to the left
        (10, 0),  # Outside, to the lower right
    ]

    results = [road_surface.contains(Point(p)) for p in test_points]

    for point, result in zip(test_points, results):
        status = "INSIDE" if result else "OUTSIDE"
        print(f"Query: Is point {point} in the road surface? -> {status}")

    plot_junction(snapped_polylines, road_surface, test_points, results, method=method_used)


if __name__ == "__main__":
    main()
