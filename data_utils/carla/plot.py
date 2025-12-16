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

    color_map = {
        "road_edge": "black",
        "road_line": "yellow",
        "lane": "blue",
        "crosswalk": "orange",
        "speed_bump": "red",
        "stop_sign": "red",
    }

    fig, ax = plt.subplots(1, 1, figsize=figsize)

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

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(title)

    if len(type_counts) > 1:
        ax.legend(loc="best")

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

    road_edges = [element for element in data if element.get("type") == "road_edge"]

    if not road_edges:
        print("No road_edge elements found in the data.")
        return None, None

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    print(f"Plotting {len(road_edges)} road edge elements with IDs...")

    for element in road_edges:
        geometry = element.get("geometry", [])
        element_id = element.get("id", "unknown")

        if not geometry:
            continue

        x_coords = [point["x"] for point in geometry]
        y_coords = [point["y"] for point in geometry]

        # Plot the road edge
        ax.plot(x_coords, y_coords, color="black", linewidth=2, alpha=0.8)
        if len(x_coords) > 0:
            should_show_id = False
            if int(element_id) in show_ids_for:
                should_show_id = True

            if should_show_id:
                road_length = len(x_coords)
                if road_length <= 5:
                    num_labels = 1
                    positions = [road_length // 2]
                elif road_length <= 15:
                    num_labels = 1
                    positions = [road_length // 2]
                else:
                    num_labels = 2
                    positions = [road_length // 3, 2 * road_length // 3]

                for pos in positions:
                    if pos < len(x_coords):
                        ax.annotate(
                            f"ID:{element_id}",
                            (x_coords[pos], y_coords[pos]),
                            fontsize=7,
                            ha="center",
                            va="bottom",
                            color="darkred",
                            weight="normal",
                            alpha=0.8,
                            bbox=dict(
                                boxstyle="round,pad=0.15",
                                facecolor="lightyellow",
                                alpha=0.7,
                                edgecolor="darkred",
                                linewidth=0.5,
                            ),
                        )

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.set_title(f"{title} ({len(road_edges)} elements)", fontsize=14)

    ax.grid(True, alpha=0.3)

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

        road_data = []
        if isinstance(data, dict):
            if "roads" in data:
                road_data = data["roads"]
            else:
                raise ValueError("JSON structure not recognized: missing 'roads' key")

        if not isinstance(road_data, list):
            raise ValueError("Expected road data to be a list")

        if "title" not in kwargs:
            import os

            filename = os.path.basename(json_file_path)
            kwargs["title"] = f"Road Data from {filename}"

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

            object_colors = {"vehicle": "green"}

            type_handles = {}

            for idx, obj in enumerate(data["objects"]):
                if "position" not in obj or not obj["position"]:
                    continue

                obj_id = obj.get("id", "unknown")
                obj_type = obj.get("type", "vehicle")  # Default to vehicle if type not specified
                positions = obj["position"]
                print(f"Object ID: {obj_id}, Type: {obj_type}, Positions: {len(positions)}")
                valid = obj.get("valid", [True] * len(positions))

                x_coords = [pos["x"] for pos, is_valid in zip(positions, valid) if is_valid]
                y_coords = [pos["y"] for pos, is_valid in zip(positions, valid) if is_valid]

                if not x_coords:  # Skip if no valid positions
                    continue

                color = object_colors.get(obj_type, "gray")

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
            if len(type_counts) > 1:
                ax.legend(loc="best")

        # Set equal aspect ratio
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(kwargs.get("title", "Road Data Visualization"))

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
    json_file_paths = [
        "data_utils/carla/carla_py123d/Town01.json",
        "data_utils/carla/carla_py123d/Town02.json",
        "data_utils/carla/carla_py123d/Town03.json",
        "data_utils/carla/carla_py123d/Town04.json",
        "data_utils/carla/carla_py123d/Town05.json",
        "data_utils/carla/carla_py123d/Town06.json",
        "data_utils/carla/carla_py123d/Town07.json",
        "data_utils/carla/carla_py123d/Town10HD.json",
    ]
    show_id = False  # Set to show IDs on the plot
    plot_as_points = False  # Set to True to plot as scatter points instead of lines
    plot_objects = True  # Set to True to plot object trajectories
    for json_file_path in json_file_paths:
        print(f"Processing file: {json_file_path}")
        load_and_plot_json(json_file_path, show_id=show_id, plot_as_points=plot_as_points, plot_objects=plot_objects)
