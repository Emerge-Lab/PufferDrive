import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any


def plot_road_data(
    data: List[Dict[str, Any]],
    title: str = "Road Data Visualization",
    figsize: tuple = (12, 8),
    save_path: str = None,
    show_id: bool = False,
    plot_as_points: bool = False,
):
    """
    Plot 2D road data with different colors for different road element types.

    Args:
        data: List of road elements, each containing:
            - type: string ("road_edge", "road_line", "lane", "crosswalk", "speed_bump", "stop_sign", "driveway")
            - geometry: Array of points with x, y, z coordinates
            - id: Unique road ID
            - map_element_id: Map type identifier (optional)
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
        show_id: If True, show IDs for all road elements.
        plot_as_points: If True, plot as scatter points instead of lines.
    """

    # Define colors for different road element types
    color_map = {
        "road_edge": "black",
        "road_line": "yellow",
        "lane": "blue",
        "crosswalk": "orange",
        "speed_bump": "red",
        "stop_sign": "red",
    }

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Group elements by type for legend
    type_counts = {}

    # Plot each road element
    for element in data:
        element_type = element.get("type", "unknown")
        geometry = element.get("geometry", [])
        element_id = element.get("id", "unknown")
        road_id = element.get("road_id", "-1")
        junction_id = element.get("junction_id", "-1")

        if not geometry:
            continue

        # Extract x and y coordinates
        x_coords = [point["x"] for point in geometry]
        y_coords = [point["y"] for point in geometry]

        # Get color for this element type
        color = color_map.get(element_type, "gray")

        # Plot the geometry as a line
        line_width = 2 if element_type == "road_edge" else 1
        alpha = 0.8 if element_type == "lane" else 1.0

        # Only add to legend if it's the first occurrence of this type
        label = element_type if element_type not in type_counts else ""
        if element_type not in type_counts:
            type_counts[element_type] = 0
        type_counts[element_type] += 1

        # Plot as points or lines based on mode
        if plot_as_points:
            marker_size = 3 if element_type == "road_edge" else 2
            ax.scatter(x_coords, y_coords, color=color, s=marker_size, alpha=alpha, label=label)
        else:
            ax.plot(x_coords, y_coords, color=color, linewidth=line_width, alpha=alpha, label=label)

        # Add ID annotations for all elements if show_id is True
        if show_id and len(x_coords) > 0:
            # Place ID at multiple positions for road_edge, middle for others
            if element_type == "road_edge":
                positions_to_annotate = []
                if len(x_coords) <= 3:
                    positions_to_annotate = [len(x_coords) // 2]
                elif len(x_coords) <= 10:
                    positions_to_annotate = [len(x_coords) // 2]
                else:
                    positions_to_annotate = [len(x_coords) // 3, 2 * len(x_coords) // 3]
                for pos_idx in positions_to_annotate:
                    if pos_idx < len(x_coords):
                        offset_x = 0.5
                        offset_y = 0.5
                        ax.annotate(
                            f"ID:{element_id} | Road:{road_id} | Junc:{junction_id}",
                            (x_coords[pos_idx] + offset_x, y_coords[pos_idx] + offset_y),
                            fontsize=6,
                            ha="left",
                            va="bottom",
                            color="darkred",
                            weight="normal",
                            alpha=0.8,
                            bbox=dict(
                                boxstyle="round,pad=0.1",
                                facecolor="lightyellow",
                                alpha=0.6,
                                edgecolor="darkred",
                                linewidth=0.3,
                            ),
                        )
            else:
                mid_idx = len(x_coords) // 2
                ax.annotate(
                    f"ID:{element_id}", (x_coords[mid_idx], y_coords[mid_idx]), fontsize=8, ha="center", alpha=0.7
                )

    # Set equal aspect ratio to maintain proper geometry
    ax.set_aspect("equal", adjustable="box")

    # Add labels and title
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(title)

    # Add legend if we have multiple types
    if len(type_counts) > 1:
        ax.legend(loc="best")

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = "\n".join([f"{type_name}: {count}" for type_name, count in type_counts.items()])
    ax.text(
        0.02,
        0.98,
        f"Elements:\n{stats_text}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()

    return fig, ax


def plot_road_edges_with_ids(
    data: List[Dict[str, Any]],
    title: str = "Road Edges with IDs",
    figsize: tuple = (15, 10),
    save_path: str = None,
    show_ids_for: List[int] = None,
):
    """
    Plot only road_edge elements with enhanced ID visibility.

    Args:
        data: List of road elements
        title: Plot title
        figsize: Figure size (larger default for better ID visibility)
        save_path: Optional path to save the plot
        show_ids_for: List of IDs to show labels for. If None, show all road_edge IDs.
                     If empty list [], show no IDs. If list with IDs, show only those IDs.
    """

    # Filter for road_edge elements only
    road_edges = [element for element in data if element.get("type") == "road_edge"]

    if not road_edges:
        print("No road_edge elements found in the data.")
        return None, None

    # Create figure and axis with larger size for better ID visibility
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    print(f"Plotting {len(road_edges)} road edge elements with IDs...")

    # Plot each road edge
    for element in road_edges:
        geometry = element.get("geometry", [])
        element_id = element.get("id", "unknown")

        if not geometry:
            continue

        # Extract x and y coordinates
        x_coords = [point["x"] for point in geometry]
        y_coords = [point["y"] for point in geometry]

        # Plot the road edge
        ax.plot(x_coords, y_coords, color="black", linewidth=2, alpha=0.8)

        # Add ID annotations at multiple positions for better visibility
        if len(x_coords) > 0:
            # Check if we should show ID for this element
            should_show_id = False
            if int(element_id) in show_ids_for:
                # Show only if ID is in the specified list
                should_show_id = True

            if should_show_id:
                # Calculate optimal number of ID labels based on road length
                road_length = len(x_coords)
                if road_length <= 5:
                    num_labels = 1
                    positions = [road_length // 2]
                elif road_length <= 15:
                    num_labels = 1
                    positions = [road_length // 2]
                else:
                    # For very long roads, place only 2 labels maximum
                    num_labels = 2
                    positions = [road_length // 3, 2 * road_length // 3]

                for pos in positions:
                    if pos < len(x_coords):
                        # Add ID with enhanced styling for zoom visibility
                        ax.annotate(
                            f"ID:{element_id}",
                            (x_coords[pos], y_coords[pos]),
                            fontsize=7,  # Reduced from 10
                            ha="center",
                            va="bottom",
                            color="darkred",  # Darker color
                            weight="normal",  # Changed from bold
                            alpha=0.8,  # Reduced transparency
                            bbox=dict(
                                boxstyle="round,pad=0.15",  # Smaller padding
                                facecolor="lightyellow",  # Lighter background
                                alpha=0.7,  # More transparent
                                edgecolor="darkred",
                                linewidth=0.5,
                            ),
                        )  # Thinner border

    # Set equal aspect ratio
    ax.set_aspect("equal", adjustable="box")

    # Add labels and title
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.set_title(f"{title} ({len(road_edges)} elements)", fontsize=14)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Add instructions for zooming
    instruction_text = "Tip: Use zoom tool to see individual road edge IDs clearly"
    ax.text(
        0.02,
        0.02,
        instruction_text,
        transform=ax.transAxes,
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()

    return fig, ax


def plot_single_element_type(
    data: List[Dict[str, Any]],
    element_type: str,
    title: str = None,
    figsize: tuple = (12, 8),
    show_ids_for: List[int] = None,
):
    """
    Plot only elements of a specific type.

    Args:
        data: List of road elements
        element_type: Type to filter and plot
        title: Plot title (auto-generated if None)
        figsize: Figure size
        show_ids_for: List of IDs to show labels for. If None, show IDs based on default rules.
    """

    # Filter data for the specific type
    filtered_data = [element for element in data if element.get("type") == element_type]

    if not filtered_data:
        print(f"No elements of type '{element_type}' found in the data.")
        return None, None

    # Generate title if not provided
    if title is None:
        title = f"{element_type.replace('_', ' ').title()} Elements ({len(filtered_data)} items)"

    return plot_road_data(filtered_data, title=title, figsize=figsize, show_ids_for=show_ids_for)


def load_and_plot_json(
    json_file_path: str, plot_objects: bool = True, show_id: bool = True, plot_as_points: bool = False, **kwargs
):
    """
    Load road data and object trajectories from a JSON file and plot them.

    Args:
        json_file_path: Path to the JSON file.
        plot_objects: Whether to plot object trajectories.
        show_id: If True, annotate object IDs on the plot.
        plot_as_points: If True, plot road elements as scatter points instead of lines.
        **kwargs: Additional arguments for plot_road_data.
    """
    import json

    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)

        # Handle different JSON structures
        road_data = []
        if isinstance(data, dict):
            if "roads" in data:
                road_data = data["roads"]
            else:
                raise ValueError("JSON structure not recognized: missing 'roads' key")

        if not isinstance(road_data, list):
            raise ValueError("Expected road data to be a list")

        # Add filename to title if not provided
        if "title" not in kwargs:
            import os

            filename = os.path.basename(json_file_path)
            kwargs["title"] = f"Road Data from {filename}"

        # Create a single figure for both roads and objects
        fig, ax = plt.subplots(1, 1, figsize=kwargs.get("figsize", (12, 8)))

        # Plot road data first (manually inline instead of calling plot_road_data)
        color_map = {
            "road_edge": "black",
            "road_line": "yellow",
            "lane": "blue",
            "crosswalk": "orange",
            "speed_bump": "red",
            "stop_sign": "red",
        }

        type_counts = {}

        for element in road_data:
            element_type = element.get("type", "unknown")
            geometry = element.get("geometry", [])
            element_id = element.get("id", "unknown")
            road_id = element.get("road_id", "-1")
            junction_id = element.get("junction_id", "-1")

            if not geometry:
                continue

            x_coords = [point["x"] for point in geometry]
            y_coords = [point["y"] for point in geometry]
            color = color_map.get(element_type, "gray")
            line_width = 2 if element_type == "road_edge" else 1
            alpha = 0.8 if element_type == "lane" else 1.0
            label = element_type if element_type not in type_counts else ""

            if element_type not in type_counts:
                type_counts[element_type] = 0
            type_counts[element_type] += 1

            # Plot as points or lines based on mode
            if plot_as_points:
                marker_size = 3 if element_type == "road_edge" else 2
                ax.scatter(x_coords, y_coords, color=color, s=marker_size, alpha=alpha, label=label)
            else:
                ax.plot(x_coords, y_coords, color=color, linewidth=line_width, alpha=alpha, label=label)

            if show_id and len(x_coords) > 0:
                if element_type == "road_edge":
                    positions_to_annotate = []
                    if len(x_coords) <= 3:
                        positions_to_annotate = [len(x_coords) // 2]
                    elif len(x_coords) <= 10:
                        positions_to_annotate = [len(x_coords) // 2]
                    else:
                        positions_to_annotate = [len(x_coords) // 3, 2 * len(x_coords) // 3]
                    for pos_idx in positions_to_annotate:
                        if pos_idx < len(x_coords):
                            offset_x = 0.5
                            offset_y = 0.5
                            ax.annotate(
                                f"ID:{element_id} | Road:{road_id} | Junc:{junction_id}",
                                (x_coords[pos_idx] + offset_x, y_coords[pos_idx] + offset_y),
                                fontsize=6,
                                ha="left",
                                va="bottom",
                                color="darkred",
                                weight="normal",
                                alpha=0.8,
                                bbox=dict(
                                    boxstyle="round,pad=0.1",
                                    facecolor="lightyellow",
                                    alpha=0.6,
                                    edgecolor="darkred",
                                    linewidth=0.3,
                                ),
                            )
                else:
                    mid_idx = len(x_coords) // 2
                    ax.annotate(
                        f"ID:{element_id}", (x_coords[mid_idx], y_coords[mid_idx]), fontsize=8, ha="center", alpha=0.7
                    )

        # Plot object trajectories if requested
        if plot_objects and ("objects" in data) and (isinstance(data["objects"], list)):
            print(f"Plotting {len(data['objects'])} objects...")

            # Define colors for different object types
            object_colors = {"vehicle": "green"}

            # Track unique object types for legend
            type_handles = {}

            for idx, obj in enumerate(data["objects"]):
                if "position" not in obj or not obj["position"]:
                    continue

                obj_id = obj.get("id", "unknown")
                obj_type = obj.get("type", "vehicle")  # Default to vehicle if type not specified
                positions = obj["position"]
                print(f"Object ID: {obj_id}, Type: {obj_type}, Positions: {len(positions)}")
                valid = obj.get("valid", [True] * len(positions))

                # Extract x and y coordinates for valid positions
                x_coords = [pos["x"] for pos, is_valid in zip(positions, valid) if is_valid]
                y_coords = [pos["y"] for pos, is_valid in zip(positions, valid) if is_valid]

                if not x_coords:  # Skip if no valid positions
                    continue

                # Get color based on object type
                color = object_colors.get(obj_type, "gray")

                # Plot trajectory
                line = ax.plot(
                    x_coords,
                    y_coords,
                    linestyle="--",
                    color=color,
                    alpha=0.8,
                    linewidth=1.5,
                    label=obj_type if obj_type not in type_handles else "",
                )[0]

                if obj_type not in type_handles:
                    type_handles[obj_type] = line

                # Mark start point (blue square) and goal point (green circle) — reduced size
                ax.plot(
                    x_coords[0],
                    y_coords[0],
                    "s",
                    color="blue",
                    markersize=5,
                    markeredgecolor="darkblue",
                    markeredgewidth=0.3,
                    zorder=10,
                )
                ax.plot(
                    x_coords[-1],
                    y_coords[-1],
                    "o",
                    color="green",
                    markersize=5,
                    markeredgecolor="darkgreen",
                    markeredgewidth=0.3,
                    zorder=10,
                )

                # Add object ID label at start point if show_id is True
                if show_id:
                    ax.annotate(
                        f"ID:{obj_id}",
                        (x_coords[0], y_coords[0]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        color=color,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                    )

            # Combine road element types and object types for legend
            all_handles = []
            all_labels = []

            # Add road element types
            for element_type in type_counts.keys():
                color = color_map.get(element_type, "gray")
                line_width = 2 if element_type == "road_edge" else 1
                alpha = 0.8 if element_type == "lane" else 1.0
                all_handles.append(plt.Line2D([0], [0], color=color, linewidth=line_width, alpha=alpha))
                all_labels.append(element_type)

            # Add object types
            for obj_type, handle in type_handles.items():
                all_handles.append(handle)
                all_labels.append(f"{obj_type} (trajectory)")

            # Add start and goal point markers to legend
            all_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor="blue",
                    markeredgecolor="darkblue",
                    markeredgewidth=1.5,
                    markersize=8,
                    linestyle="None",
                )
            )
            all_labels.append("Start Point")

            all_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="green",
                    markeredgecolor="darkgreen",
                    markeredgewidth=1.5,
                    markersize=8,
                    linestyle="None",
                )
            )
            all_labels.append("Goal Point")

            if all_handles:
                ax.legend(all_handles, all_labels, loc="center left", bbox_to_anchor=(1, 0.5))
        else:
            # Only road elements, add legend for them
            if len(type_counts) > 1:
                ax.legend(loc="best")

        # Set equal aspect ratio
        ax.set_aspect("equal", adjustable="box")

        # Add labels and title
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(kwargs.get("title", "Road Data Visualization"))

        # Add grid for better visibility
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = "\n".join([f"{type_name}: {count}" for type_name, count in type_counts.items()])
        ax.text(
            0.02,
            0.98,
            f"Elements:\n{stats_text}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Adjust layout to prevent legend from being cut off
        plt.tight_layout()

        # Show the plot
        plt.show()

        return fig, ax

    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return None, None
    except Exception as e:
        print(f"Error loading and plotting data: {e}")
        return None, None


# Example usage
if __name__ == "__main__":
    # json_file_path = "data_utils\carla\carla\Town04.json"
    # json_file_path = r"data\processed\carla_data\Town01_0.json"  # Use raw string to avoid escape sequence warning
    # json_file_paths = [
    #     r"data\processed\carla_data_small_vel=5\Town01_0.json",
    #     r"data\processed\carla_data\Town01_1.json",
    #     r"data\processed\carla_data\Town01_2.json",
    #     r"data\processed\carla_data\Town01_3.json",
    #     r"data\processed\carla_data\Town01_4.json",
    #     r"data\processed\carla_data\Town01_5.json",
    #     r"data\processed\carla_data\Town01_6.json",
    #     r"data\processed\carla_data\Town01_7.json",
    #     r"data\processed\carla_data\Town01_8.json",
    #     r"data\processed\carla_data\Town01_9.json",
    #     r"data\processed\carla_data\Town01_10.json",
    #     r"data\processed\carla_data\Town01_11.json",
    #     r"data\processed\carla_data\Town01_12.json",
    #     r"data\processed\carla_data\Town01_13.json",
    #     r"data\processed\carla_data\Town01_14.json",
    #     r"data\processed\carla_data\Town01_15.json"
    # ]
    json_file_paths = [
        # "data_utils\\carla\\carla\\Town01.json",
        # "data_utils\\carla\\carla\\Town02.json",
        # "data_utils\\carla\\carla\\Town03.json",
        # "data_utils\\carla\\carla\\Town04.json",
        # "data_utils\\carla\\carla\\Town05.json",
        # "data_utils\\carla\\carla\\Town06.json",
        "data_utils\\carla\\carla_before_manual\\Town07.json",
        # "data_utils\\carla\\carla\\Town10HD.json"
    ]
    show_id = False  # Set to show IDs on the plot
    plot_as_points = False  # Set to True to plot as scatter points instead of lines
    plot_objects = False  # Set to True to plot object trajectories
    # load_and_plot_json(json_file_path, plot_objects=True, show_id=show_id, plot_as_points=plot_as_points)
    for json_file_path in json_file_paths:
        print(f"Processing file: {json_file_path}")
        # Load data and plot all elements
        load_and_plot_json(json_file_path, show_id=show_id, plot_as_points=plot_as_points, plot_objects=plot_objects)

    # To specifically visualize road edges with enhanced ID visibility:
    # First load the data
    # try:
    #     import json
    #     with open(json_file_path, 'r') as f:
    #         data = json.load(f)
    #     if isinstance(data, dict) and 'roads' in data:
    #         road_data = data['roads']

    #         # Example: Show only specific IDs (replace with your desired IDs)
    #         specific_ids = [6, 10, 11, 15, 26, 30, 31, 35, 46, 50, 51, 55, 66, 70, 71, 75, 86, 90, 91, 95, 105, 106, 111, 115, 116, 120, 121, 126, 130, 131, 133, 134, 136, 137, 139, 140, 142, 145, 146, 148, 149, 151, 152, 154, 155, 157, 158, 160, 163, 164, 166, 167, 169, 170, 172, 173, 175, 176, 178, 179, 181, 184, 185, 187, 188, 190, 191, 193, 196, 197, 199, 200, 202, 203, 205, 206, 208, 209, 211, 212, 214, 215, 217, 218, 220, 221, 223, 224, 226, 229, 230, 232, 233, 235, 236, 238, 241, 242, 244, 245, 247, 248, 250, 251, 253, 256, 257, 259, 260, 262, 263, 265, 266, 268, 269, 271, 274, 275, 277, 278, 280, 281, 283, 284, 286, 287, 289, 290, 292, 293, 295, 296, 298, 301, 302, 304, 305, 307, 308, 310, 311, 313, 314, 316, 319, 320, 322, 323, 325, 326, 328, 329, 331, 332, 334, 335, 337, 338, 340, 341, 343, 344]

    #         if show_id:
    #             # Plot all road edges but show IDs only for specific ones
    #             plot_road_edges_with_ids(road_data,
    #                                      title="Road Edges with Selected IDs",
    #                                      show_ids_for=specific_ids)
    #         else:
    #             # Alternative: Show no IDs at all
    #             plot_road_edges_with_ids(road_data, title="Road Edges (No IDs)", show_ids_for=[])

    #         # Alternative: Show all IDs (default behavior)
    #         # plot_road_edges_with_ids(road_data, title="Road Edges with All IDs", show_ids_for=None)

    # except Exception as e:
    #     print(f"Could not load data for road edge visualization: {e}")
