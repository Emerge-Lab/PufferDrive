"""Parse map binary files and compute grid dimensions to diagnose memory usage.

Usage: python scripts/inspect_maps.py <map_dir> [num_maps]
"""
import struct
import sys
import os
import math

GRID_CELL_SIZE = 5.0
OBSERVATION_WINDOW_SIZE = 100.0
ROAD_LANE = 4
ROAD_LINE = 5
ROAD_EDGE = 6


def read_int(f):
    data = f.read(4)
    if len(data) < 4:
        return None
    return struct.unpack('i', data)[0]


def read_float(f):
    data = f.read(4)
    if len(data) < 4:
        return None
    return struct.unpack('f', data)[0]


def read_floats(f, n):
    data = f.read(4 * n)
    if len(data) < 4 * n:
        return None
    return struct.unpack(f'{n}f', data)


def read_ints(f, n):
    data = f.read(4 * n)
    if len(data) < 4 * n:
        return None
    return struct.unpack(f'{n}i', data)


def parse_map(filepath):
    with open(filepath, 'rb') as f:
        sdc_track_index = read_int(f)
        num_tracks_to_predict = read_int(f)
        if num_tracks_to_predict > 0:
            tracks = read_ints(f, num_tracks_to_predict)

        num_objects = read_int(f)
        num_roads = read_int(f)

        print(f"  sdc_track_index={sdc_track_index}, num_tracks_to_predict={num_tracks_to_predict}")
        print(f"  num_objects={num_objects}, num_roads={num_roads}")
        print(f"  total_entities={num_objects + num_roads}")

        total_entities = num_objects + num_roads
        road_coords_x = []
        road_coords_y = []
        road_types = []
        total_road_points = 0

        for i in range(total_entities):
            scenario_id = read_int(f)
            entity_type = read_int(f)
            entity_id = read_int(f)
            array_size = read_int(f)

            if i < num_objects:
                # Agent: read trajectories + scalars
                # log_trajectory_x, y, z, velocity_x, y, z, heading (7 float arrays)
                f.read(4 * array_size * 7)
                # log_valid (int array)
                f.read(4 * array_size)
                # width, length, height, goal_x, goal_y, goal_z (6 floats) + mark_as_expert (1 int)
                f.read(4 * 7)
            else:
                # Road: read x, y, z coords
                xs = read_floats(f, array_size)
                ys = read_floats(f, array_size)
                zs = read_floats(f, array_size)
                # width, length, height, goal_x, goal_y, goal_z, mark_as_expert
                f.read(4 * 7)

                road_types.append(entity_type)
                if entity_type in (ROAD_LANE, ROAD_LINE, ROAD_EDGE):
                    road_coords_x.extend(xs)
                    road_coords_y.extend(ys)
                    total_road_points += array_size

        # Compute bounding box (matching init_grid_map logic)
        # Filter out INVALID_POSITION (need to check what that is)
        INVALID_POSITION = -10000.0  # From drive.h
        valid_x = [x for x in road_coords_x if x > INVALID_POSITION]
        valid_y = [y for y in road_coords_y if y > INVALID_POSITION]

        if not valid_x:
            print("  NO VALID ROAD COORDINATES!")
            return

        min_x, max_x = min(valid_x), max(valid_x)
        min_y, max_y = min(valid_y), max(valid_y)

        grid_width = max_x - min_x
        grid_height = max_y - min_y
        grid_cols = math.ceil(grid_width / GRID_CELL_SIZE)
        grid_rows = math.ceil(grid_height / GRID_CELL_SIZE)
        grid_cell_count = grid_cols * grid_rows
        vision_range = math.ceil(OBSERVATION_WINDOW_SIZE / GRID_CELL_SIZE) + 1

        print(f"\n  Road types present: {set(road_types)}")
        print(f"  Total road points (lane/line/edge): {total_road_points}")
        print(f"  X range: [{min_x:.1f}, {max_x:.1f}] (width={grid_width:.1f}m)")
        print(f"  Y range: [{min_y:.1f}, {max_y:.1f}] (height={grid_height:.1f}m)")
        print(f"  Grid: {grid_cols} x {grid_rows} = {grid_cell_count:,} cells")
        print(f"  Vision range: {vision_range} ({vision_range}x{vision_range} = {vision_range**2} cells per lookup)")

        # Memory estimates
        base_grid_mem = grid_cell_count * (8 + 4)  # cells ptr + count
        print(f"\n  Base grid memory: {base_grid_mem / 1024 / 1024:.1f} MB")
        # Neighbor cache: cell_count * (8 + 4) for pointers + counts
        neighbor_cache_overhead = grid_cell_count * (8 + 4)
        print(f"  Neighbor cache overhead (pointers): {neighbor_cache_overhead / 1024 / 1024:.1f} MB")
        # Worst case: every cell's neighbor area has all road segments
        # Each GridMapEntity is 2 ints = 8 bytes
        # If road segments are dense, each cell might have many entities
        avg_entities_per_cell = total_road_points / max(grid_cell_count, 1)
        print(f"  Avg road points per cell: {avg_entities_per_cell:.2f}")
        # Neighbor cache per cell: vision_range^2 * avg_entities * 8 bytes
        est_neighbor_cache = grid_cell_count * vision_range**2 * avg_entities_per_cell * 8
        print(f"  Estimated neighbor cache (if uniform): {est_neighbor_cache / 1024 / 1024 / 1024:.1f} GB")


if __name__ == "__main__":
    map_dir = sys.argv[1] if len(sys.argv) > 1 else "resources/drive/binaries/carla_data"
    num_maps = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    # Also check carla_3D for comparison
    for d in [map_dir, "resources/drive/binaries/carla_3D"]:
        print(f"\n{'='*60}")
        print(f"Map directory: {d}")
        print(f"{'='*60}")
        if not os.path.exists(d):
            print(f"  Directory not found!")
            continue
        # Try both map_NNN.bin and *.bin naming
        files = []
        n = num_maps if d == map_dir else 3
        for i in range(n):
            path = os.path.join(d, f"map_{i:03d}.bin")
            if os.path.exists(path):
                files.append(path)
        if not files:
            # Fall back to listing all .bin files
            files = sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.bin')])
        for path in files:
            size = os.path.getsize(path)
            if size == 0:
                print(f"\n{path}: EMPTY (0 bytes)")
                continue
            print(f"\n{path} ({size:,} bytes):")
            try:
                parse_map(path)
            except Exception as e:
                print(f"  ERROR: {e}")
