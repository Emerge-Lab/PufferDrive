import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROAD_STYLES = {
    "lane": ((150, 156, 162), 1),
    "road_edge": ((55, 64, 72), 2),
    "road_line": ((206, 165, 48), 2),
    "crosswalk": ((124, 111, 176), 2),
    "speed_bump": ((214, 111, 44), 2),
    "stop_sign": ((191, 63, 47), 4),
    "driveway": ((123, 139, 92), 1),
}

OBJECT_TYPE_MAP = {
    "TYPE_VEHICLE": "vehicle",
    "TYPE_PEDESTRIAN": "pedestrian",
    "TYPE_CYCLIST": "cyclist",
}


def point_dict(point):
    return {
        "x": float(point.get("x", point.get("center_x", 0.0))),
        "y": float(point.get("y", point.get("center_y", 0.0))),
        "z": float(point.get("z", point.get("center_z", 0.0))),
    }


def convert_raw_waymo(data):
    if "objects" in data and "roads" in data:
        return data
    if "tracks" not in data or "map_features" not in data:
        raise ValueError("Unsupported JSON schema: expected objects/roads or tracks/map_features")

    objects = []
    for track in data["tracks"]:
        states = track.get("states", [])
        valid_states = [state for state in states if state.get("valid")]
        dimensions = valid_states[0] if valid_states else (states[0] if states else {})
        objects.append(
            {
                "id": track.get("id"),
                "type": OBJECT_TYPE_MAP.get(track.get("object_type"), "vehicle"),
                "position": [point_dict(state) for state in states],
                "heading": [float(state.get("heading", 0.0)) for state in states],
                "velocity": [
                    {
                        "x": float(state.get("velocity_x", 0.0)),
                        "y": float(state.get("velocity_y", 0.0)),
                        "z": 0.0,
                    }
                    for state in states
                ],
                "valid": [bool(state.get("valid")) for state in states],
                "length": float(dimensions.get("length", 4.0)),
                "width": float(dimensions.get("width", 2.0)),
                "height": float(dimensions.get("height", 1.5)),
                "goalPosition": point_dict(valid_states[-1]) if valid_states else {"x": 0, "y": 0, "z": 0},
                "mark_as_expert": False,
            }
        )

    roads = []
    for feature_index, feature in enumerate(data["map_features"]):
        feature_id = feature.get("id", feature_index)
        for road_type in ROAD_STYLES:
            if road_type not in feature:
                continue
            feature_data = feature[road_type]
            geometry = feature_data.get("polyline") or feature_data.get("polygon") or []
            if not geometry and feature_data.get("position"):
                geometry = [feature_data["position"]]
            roads.append(
                {
                    "id": feature_id,
                    "map_element_id": feature_id,
                    "type": road_type,
                    "geometry": [point_dict(point) for point in geometry],
                }
            )
            break

    return {
        "name": data.get("scenario_id", "waymo_scenario"),
        "scenario_id": data.get("scenario_id"),
        "objects": objects,
        "roads": roads,
        "timestamps_seconds": data.get("timestamps_seconds", []),
        "metadata": {
            "sdc_track_index": data.get("sdc_track_index"),
            "tracks_to_predict": data.get("tracks_to_predict", []),
        },
    }


def valid_at(obj, frame):
    valid = obj.get("valid")
    return valid is None or frame < len(valid) and bool(valid[frame])


def point_at(obj, frame):
    pos = obj.get("position", [])
    if frame >= len(pos):
        return None
    return pos[frame]


def bounds_for(data):
    xs = []
    ys = []
    for road in data.get("roads", []):
        for point in road.get("geometry", []):
            xs.append(point["x"])
            ys.append(point["y"])

    for obj in data.get("objects", []):
        for frame, point in enumerate(obj.get("position", [])):
            if valid_at(obj, frame):
                xs.append(point["x"])
                ys.append(point["y"])

    if not xs or not ys:
        raise ValueError("Scenario has no drawable x/y coordinates")

    return min(xs), max(xs), min(ys), max(ys)


def make_transform(bounds, width, height, pad):
    min_x, max_x, min_y, max_y = bounds
    world_w = max(max_x - min_x, 1.0)
    world_h = max(max_y - min_y, 1.0)
    scale = min((width - 2 * pad) / world_w, (height - 2 * pad) / world_h)
    extra_x = (width - 2 * pad - world_w * scale) / 2
    extra_y = (height - 2 * pad - world_h * scale) / 2

    def transform(x, y):
        px = pad + extra_x + (x - min_x) * scale
        py = height - pad - extra_y - (y - min_y) * scale
        return px, py

    return transform, scale


def object_color(data, index, obj):
    metadata = data.get("metadata", {})
    predicted = {track.get("track_index") for track in metadata.get("tracks_to_predict", [])}
    if index == metadata.get("sdc_track_index"):
        return 25, 103, 210
    if index in predicted or obj.get("mark_as_expert"):
        return 191, 63, 47
    if obj.get("type") in {"pedestrian", "cyclist"}:
        return 141, 90, 194
    return 44, 139, 87


def draw_polyline(draw, points, transform, fill, width):
    coords = []
    for point in points:
        x = point.get("x")
        y = point.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            coords.append(transform(x, y))
    if len(coords) >= 2:
        draw.line(coords, fill=fill, width=width, joint="curve")


def draw_object(draw, data, obj, index, frame, transform, scale, trails, highlight_index=None):
    point = point_at(obj, frame)
    if not point or not valid_at(obj, frame):
        return

    highlighted = index == highlight_index
    color = (0, 173, 239) if highlighted else object_color(data, index, obj)
    if trails:
        trail = []
        for i in range(frame + 1):
            trail_point = point_at(obj, i)
            if trail_point and valid_at(obj, i):
                trail.append(trail_point)
        if len(trail) > 1:
            draw_polyline(draw, trail, transform, color + ((220 if highlighted else 95),), 4 if highlighted else 2)

    cx, cy = transform(point["x"], point["y"])
    length = max(5, obj.get("length", 4.0) * scale)
    width = max(3, obj.get("width", 2.0) * scale)
    heading_values = obj.get("heading", [])
    heading = heading_values[frame] if frame < len(heading_values) else 0

    forward = np.array([np.cos(-heading), np.sin(-heading)])
    side = np.array([-forward[1], forward[0]])
    center = np.array([cx, cy])
    corners = [
        center + forward * length / 2 + side * width / 2,
        center + forward * length / 2 - side * width / 2,
        center - forward * length / 2 - side * width / 2,
        center - forward * length / 2 + side * width / 2,
    ]
    polygon = [tuple(corner) for corner in corners]
    if highlighted:
        halo = max(8, int(max(length, width) * 0.8))
        draw.ellipse((cx - halo, cy - halo, cx + halo, cy + halo), outline=(0, 173, 239, 210), width=3)
    draw.polygon(polygon, fill=color, outline=(255, 255, 255), width=2 if highlighted else 1)

    nose = [
        tuple(center + forward * length / 2),
        tuple(center + forward * length * 0.22 + side * width * 0.28),
        tuple(center + forward * length * 0.22 - side * width * 0.28),
    ]
    draw.polygon(nose, fill=(245, 247, 249))


def render_frame(data, frame, transform, scale, width, height, trails, highlight_index=None):
    image = Image.new("RGB", (width, height), (235, 231, 223))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for road in data.get("roads", []):
        color, line_width = ROAD_STYLES.get(road.get("type"), ((110, 110, 110), 1))
        alpha = 125 if road.get("type") == "lane" else 185
        draw_polyline(draw, road.get("geometry", []), transform, color + (alpha,), line_width)

    for index, obj in enumerate(data.get("objects", [])):
        draw_object(draw, data, obj, index, frame, transform, scale, trails, highlight_index)

    timestamp_values = data.get("timestamps_seconds", [])
    timestamp = timestamp_values[frame] if frame < len(timestamp_values) else frame / 10
    label = f"{data.get('scenario_id', data.get('name', 'scenario'))} | frame {frame:02d} | t={timestamp:.1f}s"
    if highlight_index is not None and highlight_index < len(data.get("objects", [])):
        target = data["objects"][highlight_index]
        label += f" | target track {highlight_index}, id {target.get('id')}"
    ImageDraw.Draw(overlay).rounded_rectangle((10, 8, min(width - 10, 470), 34), radius=4, fill=(245, 247, 249, 220))
    ImageDraw.Draw(overlay).text(
        (16, 14),
        label,
        fill=(29, 37, 44, 230),
        font=ImageFont.load_default(),
    )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def render_scenario(
    path,
    output_dir,
    width,
    height,
    fps,
    trails,
    focus_center=None,
    focus_radius=None,
    highlight_index=None,
    start_frame=0,
    end_frame=None,
    output_suffix="",
):
    with path.open("r", encoding="utf-8") as f:
        data = convert_raw_waymo(json.load(f))

    max_frame = max(len(obj.get("position", [])) for obj in data.get("objects", [])) - 1
    end_frame = max_frame if end_frame is None else min(end_frame, max_frame)
    start_frame = max(0, min(start_frame, end_frame))
    if focus_center and focus_radius:
        center_x, center_y = focus_center
        bounds = (
            center_x - focus_radius,
            center_x + focus_radius,
            center_y - focus_radius,
            center_y + focus_radius,
        )
    else:
        bounds = bounds_for(data)
    transform, scale = make_transform(bounds, width, height, pad=40)
    scenario_id = str(data.get("scenario_id") or path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / f"{scenario_id}{output_suffix}.mp4"
    png_path = output_dir / f"{scenario_id}{output_suffix}_frame{start_frame:03d}.png"

    frames = []
    for frame in range(start_frame, end_frame + 1):
        image = render_frame(data, frame, transform, scale, width, height, trails, highlight_index)
        if frame == start_frame:
            image.save(png_path)
        frames.append(np.asarray(image))

    imageio.mimsave(mp4_path, frames, fps=fps)
    return mp4_path, png_path, len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--no-trails", action="store_true")
    parser.add_argument("--focus-center", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--focus-radius", type=float)
    parser.add_argument("--highlight-track", type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    for path in args.json_files:
        mp4_path, png_path, frame_count = render_scenario(
            path=path,
            output_dir=args.output_dir,
            width=args.width,
            height=args.height,
            fps=args.fps,
            trails=not args.no_trails,
            focus_center=args.focus_center,
            focus_radius=args.focus_radius,
            highlight_index=args.highlight_track,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            output_suffix=args.output_suffix,
        )
        print(f"Rendered {path.name}: {frame_count} frames")
        print(f"  video: {mp4_path}")
        print(f"  preview: {png_path}")


if __name__ == "__main__":
    main()
