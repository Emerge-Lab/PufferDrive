import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROAD_COLORS = {
    "road_edge": (222, 226, 229),
    "road_line": (244, 198, 63),
    "lane": (91, 98, 105),
    "crosswalk": (201, 205, 209),
    "speed_bump": (218, 135, 63),
    "driveway": (112, 124, 105),
}


def normalize(vector):
    length = np.linalg.norm(vector)
    return vector / max(length, 1e-8)


def feature_geometry(feature):
    for feature_type in ROAD_COLORS:
        if feature_type not in feature:
            continue
        data = feature[feature_type]
        points = data.get("polyline") or data.get("polygon") or []
        if len(points) >= 2:
            return feature_type, points
    return None, None


class ChaseCamera:
    def __init__(self, state, width, height):
        heading = float(state.get("heading", 0.0))
        forward_xy = np.array([math.cos(heading), math.sin(heading), 0.0])
        target = np.array([state["center_x"], state["center_y"], 1.0])
        self.position = target - forward_xy * 15.0 + np.array([0.0, 0.0, 8.0])
        look_at = target + forward_xy * 9.0 + np.array([0.0, 0.0, 0.5])
        self.forward = normalize(look_at - self.position)
        self.right = normalize(np.cross(self.forward, np.array([0.0, 0.0, 1.0])))
        self.up = normalize(np.cross(self.right, self.forward))
        self.width = width
        self.height = height
        self.focal = width / (2 * math.tan(math.radians(68) / 2))

    def project(self, point):
        relative = np.asarray(point, dtype=float) - self.position
        depth = float(np.dot(relative, self.forward))
        if depth < 0.8:
            return None
        x = float(np.dot(relative, self.right))
        y = float(np.dot(relative, self.up))
        return (
            self.width / 2 + self.focal * x / depth,
            self.height * 0.53 - self.focal * y / depth,
            depth,
        )


def car_vertices(state):
    heading = float(state.get("heading", 0.0))
    length = float(state.get("length", 4.5))
    width = float(state.get("width", 1.9))
    height = float(state.get("height", 1.6))
    center = np.array([state["center_x"], state["center_y"]])
    forward = np.array([math.cos(heading), math.sin(heading)])
    side = np.array([-forward[1], forward[0]])
    footprint = [
        center + forward * length / 2 + side * width / 2,
        center + forward * length / 2 - side * width / 2,
        center - forward * length / 2 - side * width / 2,
        center - forward * length / 2 + side * width / 2,
    ]
    return [
        np.array([point[0], point[1], z])
        for z in (0.08, height)
        for point in footprint
    ]


def shade(color, factor):
    return tuple(max(0, min(255, int(channel * factor))) for channel in color)


def car_faces(state, color, camera):
    vertices = car_vertices(state)
    faces = [
        ([4, 5, 6, 7], shade(color, 1.12)),
        ([0, 1, 5, 4], shade(color, 0.92)),
        ([1, 2, 6, 5], shade(color, 0.72)),
        ([2, 3, 7, 6], shade(color, 0.82)),
        ([3, 0, 4, 7], shade(color, 0.68)),
    ]
    projected_faces = []
    for indices, face_color in faces:
        projected = [camera.project(vertices[index]) for index in indices]
        if any(point is None for point in projected):
            continue
        depth = sum(point[2] for point in projected) / len(projected)
        projected_faces.append((depth, [(point[0], point[1]) for point in projected], face_color))
    return projected_faces


def draw_ground(draw, camera, target_state):
    heading = float(target_state.get("heading", 0.0))
    forward = np.array([math.cos(heading), math.sin(heading)])
    side = np.array([-forward[1], forward[0]])
    center = np.array([target_state["center_x"], target_state["center_y"]])

    for lateral in range(-40, 41, 5):
        points = [
            camera.project((*point, 0.0))
            for point in (center + side * lateral + forward * distance for distance in range(-10, 91, 5))
        ]
        points = [(point[0], point[1]) for point in points if point]
        if len(points) >= 2:
            draw.line(points, fill=(48, 54, 58), width=1)

    for distance in range(-5, 91, 5):
        points = [
            camera.project((*point, 0.0))
            for point in (center + forward * distance + side * lateral for lateral in range(-40, 41, 4))
        ]
        points = [(point[0], point[1]) for point in points if point]
        if len(points) >= 2:
            draw.line(points, fill=(48, 54, 58), width=1)


def draw_map(draw, camera, map_features, target_state):
    target_xy = np.array([target_state["center_x"], target_state["center_y"]])
    for feature_type, points in map_features:
        visible = []
        for point in points:
            point_xy = np.array([point["x"], point["y"]])
            if np.linalg.norm(point_xy - target_xy) > 105:
                if len(visible) >= 2:
                    draw.line(visible, fill=ROAD_COLORS[feature_type], width=2)
                visible = []
                continue
            projected = camera.project((point["x"], point["y"], 0.05))
            if projected:
                visible.append((projected[0], projected[1]))
        if len(visible) >= 2:
            width = 3 if feature_type in {"road_edge", "road_line"} else 1
            draw.line(visible, fill=ROAD_COLORS[feature_type], width=width, joint="curve")


def render_frame(data, map_features, target_track_index, frame, width, height):
    target_state = data["tracks"][target_track_index]["states"][frame]
    camera = ChaseCamera(target_state, width, height)
    image = Image.new("RGB", (width, height), (18, 23, 27))
    draw = ImageDraw.Draw(image)

    horizon = int(height * 0.48)
    draw.rectangle((0, horizon, width, height), fill=(31, 36, 39))
    draw_ground(draw, camera, target_state)
    draw_map(draw, camera, map_features, target_state)

    all_faces = []
    target_xy = np.array([target_state["center_x"], target_state["center_y"]])
    for track_index, track in enumerate(data["tracks"]):
        states = track.get("states", [])
        if frame >= len(states):
            continue
        state = states[frame]
        if not state.get("valid"):
            continue
        distance = np.linalg.norm(
            np.array([state["center_x"], state["center_y"]]) - target_xy
        )
        if distance > 75:
            continue
        if track_index == target_track_index:
            color = (0, 190, 242)
        elif track.get("object_type") == "TYPE_PEDESTRIAN":
            color = (178, 102, 220)
        else:
            color = (222, 83, 58)
        all_faces.extend(car_faces(state, color, camera))

    for _, polygon, color in sorted(all_faces, key=lambda item: item[0], reverse=True):
        draw.polygon(polygon, fill=color, outline=(235, 239, 242))

    timestamp = data.get("timestamps_seconds", [])
    seconds = timestamp[frame] if frame < len(timestamp) else frame / 10
    draw.rounded_rectangle((18, 18, 440, 73), radius=6, fill=(9, 13, 16))
    draw.text(
        (32, 29),
        f"Roundabout chase | track {target_track_index} | t={seconds:.1f}s",
        fill=(238, 242, 244),
        font=ImageFont.load_default(),
    )
    draw.text(
        (32, 50),
        "Waymo recorded trajectory",
        fill=(0, 190, 242),
        font=ImageFont.load_default(),
    )
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--track-index", type=int, default=90)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=60)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.json_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    map_features = []
    for feature in data.get("map_features", []):
        feature_type, points = feature_geometry(feature)
        if points:
            map_features.append((feature_type, points))

    scenario_id = data.get("scenario_id", args.json_file.stem)
    output = args.output or Path("visualizations") / f"{scenario_id}_roundabout_follow_3d.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = output.with_suffix(".png")

    frames = []
    skipped_frames = []
    for frame_index in range(args.start_frame, args.end_frame + 1):
        target_states = data["tracks"][args.track_index].get("states", [])
        if frame_index >= len(target_states):
            skipped_frames.append(frame_index)
            continue
        target_state = target_states[frame_index]
        if not target_state.get("valid") or "center_x" not in target_state:
            skipped_frames.append(frame_index)
            continue
        frame = render_frame(
            data,
            map_features,
            args.track_index,
            frame_index,
            args.width,
            args.height,
        )
        if frame_index == args.start_frame:
            frame.save(preview)
        frames.append(np.asarray(frame))
        if frame_index % 10 == 0:
            print(f"rendered frame {frame_index:02d}/{args.end_frame}")

    imageio.mimsave(output, frames, fps=args.fps, macro_block_size=16)
    print(f"video: {output}")
    print(f"preview: {preview}")
    print(f"frames: {len(frames)}, duration: {len(frames) / args.fps:.1f}s")
    if skipped_frames:
        print(f"skipped invalid source frames: {skipped_frames}")


if __name__ == "__main__":
    main()
