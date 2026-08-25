"""PufferDrive binary map preview.

- Render lanes, road lines, and road edges to PNG.
- Show segment and connected-group counts beside the map.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


if __package__:
    from .mirror_map_bin import read_bin
else:
    from mirror_map_bin import read_bin


DEFAULT_WIDTH_PX = 900
DEFAULT_HEIGHT_PX = 900
ROAD_LINE_COLOR = "#a9b1b8"
ROAD_LANE_COLOR = "#2f78a8"
ROAD_EDGE_COLOR = "#20262d"
LEGEND_PANEL_FRACTION = 0.30
ELEMENT_LABELS = ("Road lane", "Road line", "Road edge")
ELEMENT_STYLES = {
    "Road lane": (ROAD_LANE_COLOR, 1.2, 0.85),
    "Road line": (ROAD_LINE_COLOR, 0.65, 0.65),
    "Road edge": (ROAD_EDGE_COLOR, 1.0, 0.9),
}


def _road_style(road_type):
    """Map a PufferDrive road type to its preview style."""

    if road_type <= 9:
        label = "Road lane"
    elif road_type <= 19:
        label = "Road line"
    else:
        label = "Road edge"
    return label, *ELEMENT_STYLES[label]


def _connected_group_count(roads):
    """Count road groups joined by identical endpoint coordinates."""

    if not roads:
        return 0
    parents = list(range(len(roads)))

    def find(index):
        """Return the canonical index for one connected component."""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    endpoints = {}
    for index, road in enumerate(roads):
        if not road["x"] or not road["y"]:
            continue
        for point in ((road["x"][0], road["y"][0]), (road["x"][-1], road["y"][-1])):
            other = endpoints.setdefault(point, index)
            first = find(index)
            second = find(other)
            if first != second:
                parents[first] = second
    return len({find(index) for index in range(len(roads))})


def _element_summary(roads):
    """Summarize segment and group counts by map-element type."""

    grouped = {label: [] for label in ELEMENT_LABELS}
    for road in roads:
        label, _, _, _ = _road_style(road["type"])
        grouped[label].append(road)
    return {
        label: {"segments": len(grouped[label]), "groups": _connected_group_count(grouped[label])}
        for label in ELEMENT_LABELS
    }


def render_frame(path, width_px=DEFAULT_WIDTH_PX, height_px=DEFAULT_HEIGHT_PX):
    """Render one binary map as an RGB image array."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = read_bin(path)
    roads = data["roads"]
    if not roads:
        raise ValueError("binary contains no roads")
    figure = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    axes = figure.add_axes((0.02, 0.02, 1.0 - LEGEND_PANEL_FRACTION - 0.04, 0.96))
    for road in roads:
        _, color, width, alpha = _road_style(road["type"])
        axes.plot(road["x"], road["y"], color=color, linewidth=width, alpha=alpha)
    axes.set_aspect("equal", adjustable="box")
    axes.margins(0.02)
    axes.axis("off")
    summary = _element_summary(roads)
    handles = []
    labels = []
    for label in ELEMENT_LABELS:
        color, width, alpha = ELEMENT_STYLES[label]
        handles.append(Line2D((0,), (0,), color=color, linewidth=max(width, 1.5), alpha=alpha))
        counts = summary[label]
        group_word = "group" if counts["groups"] == 1 else "groups"
        labels.append(f"{label}: {counts['segments']} seg. ({counts['groups']} {group_word})")
    figure.legend(
        handles,
        labels,
        title="Map elements",
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        handlelength=1.5,
        handletextpad=0.6,
    )
    figure.canvas.draw()
    frame = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(figure)
    return frame


def write_png(input_path, output_path, width_px=DEFAULT_WIDTH_PX, height_px=DEFAULT_HEIGHT_PX):
    """Render a binary map and write the resulting PNG file."""

    if width_px <= 0 or height_px <= 0:
        raise ValueError("width and height must be positive")
    frame = render_frame(input_path, width_px=width_px, height_px=height_px)
    import imageio.v3 as imageio_v3

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio_v3.imwrite(output_path, frame)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
    }


def parse_args():
    """Parse input, output, and image-size arguments."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH_PX)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT_PX)
    return parser.parse_args()


def main():
    """Run PNG rendering and print the generated image size."""

    args = parse_args()
    result = write_png(
        args.input,
        args.png,
        width_px=args.width,
        height_px=args.height,
    )
    print(f"rendered {result['width']}x{result['height']} PNG from {result['input']}")


if __name__ == "__main__":
    main()
