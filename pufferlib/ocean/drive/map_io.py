"""map_io.py — read PufferDrive .bin map files into plain numpy arrays.

The binary format is the one written by C-side save_map_binary in
drive.h. It is shared between the live sim and offline tooling, so this
parser must stay in sync with the C side. The notebook
``notebooks/visualize_trajectories.py`` had its own copy of this code;
extracted here so trajviz, the notebook, and any future tools can share
one source of truth.

Layout (little-endian, no padding):
    int32  sdc_track_index
    int32  num_tracks_to_predict
    int32 * num_tracks_to_predict   (track indices, skipped — not needed
                                      for road geometry)
    int32  num_objects
    int32  num_roads
    repeat (num_objects + num_roads) entities:
        int32  scenario_id
        int32  entity_type
        int32  entity_id
        int32  array_size
        if entity is an object:
            float32 * array_size  x
            float32 * array_size  y
            float32 * array_size  z
            float32 * array_size  vx
            float32 * array_size  vy
            float32 * array_size  vz
            float32 * array_size  heading
            int32   * array_size  valid
            float32 * 3           length, width, height
            float32 * 3           goal x, y, z
            int32                 mark_as_expert
        else (road element):
            float32 * array_size  x
            float32 * array_size  y
            float32 * array_size  z
            float32 * 3           tail scalars (we don't need them)
            float32 * 3           more scalars
            int32                 final scalar

The "tail scalars" on road elements are not used by trajviz; the C sim
parses them. We just skip them.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import List, Sequence

import numpy as np

# Road type ids — copied from drive.h. Mirrored as TVZ_ROAD_* in trajviz.h.
ROAD_LANE = 4
ROAD_LINE = 5
ROAD_EDGE = 6
ROAD_DRIVEWAY = 10


class Road:
    """One road polyline."""

    __slots__ = ("type", "x", "y", "z")

    def __init__(self, type: int, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.type = int(type)
        self.x = x
        self.y = y
        self.z = z

    def __len__(self) -> int:
        return int(self.x.shape[0])


def load_map_roads(map_path: Path | str) -> List[Road]:
    """Read all road polylines from a PufferDrive .bin map file.

    Returns a list of Road objects in source-map (un-centered) coordinates.
    Use ``mean_center_roads`` to subtract a world_mean if you need them in
    sim frame.
    """
    map_path = Path(map_path)
    roads: List[Road] = []

    with open(map_path, "rb") as f:
        _sdc = struct.unpack("<i", f.read(4))[0]
        num_tracks_to_predict = struct.unpack("<i", f.read(4))[0]
        if num_tracks_to_predict > 0:
            f.read(num_tracks_to_predict * 4)

        num_objects = struct.unpack("<i", f.read(4))[0]
        num_roads = struct.unpack("<i", f.read(4))[0]

        for i in range(num_objects + num_roads):
            _scenario_id = struct.unpack("<i", f.read(4))[0]
            entity_type = struct.unpack("<i", f.read(4))[0]
            _entity_id = struct.unpack("<i", f.read(4))[0]
            array_size = struct.unpack("<i", f.read(4))[0]

            if i < num_objects:
                # Agent: skip the per-step arrays + scalar tail.
                f.read(array_size * 4 * 6)  # x, y, z, vx, vy, vz
                f.read(array_size * 4)  # heading (float32)
                f.read(array_size * 4)  # valid (int32)
                f.read(4 * 3 + 4 * 3 + 4)  # length/width/height + goal xyz + mark_as_expert
            else:
                x = np.frombuffer(f.read(array_size * 4), dtype=np.float32).copy()
                y = np.frombuffer(f.read(array_size * 4), dtype=np.float32).copy()
                z = np.frombuffer(f.read(array_size * 4), dtype=np.float32).copy()
                f.read(4 * 3 + 4 * 3 + 4)  # tail scalars we don't need
                roads.append(Road(entity_type, x, y, z))

    return roads


def mean_center_roads(roads: Sequence[Road], world_mean: np.ndarray) -> List[Road]:
    """Return a new list of roads with world_mean subtracted from x/y (and z
    if world_mean has 3 components). The input list is not modified."""
    out: List[Road] = []
    for r in roads:
        nx = r.x - world_mean[0]
        ny = r.y - world_mean[1]
        nz = r.z - world_mean[2] if len(world_mean) > 2 else r.z.copy()
        out.append(Road(r.type, nx, ny, nz))
    return out


def roads_to_csr(roads: Sequence[Road]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of Road objects into the CSR layout the trajviz C
    extension expects.

    Returns:
        road_xy:      (N, 2) float32 — concatenated (x, y) of all polylines
        road_offsets: (P+1,) uint32 — start index per polyline
        road_types:   (P,)   uint32 — TVZ_ROAD_* type id per polyline
    """
    if not roads:
        return (np.zeros((0, 2), dtype=np.float32), np.zeros((1,), dtype=np.uint32), np.zeros((0,), dtype=np.uint32))

    lens = np.array([len(r) for r in roads], dtype=np.uint32)
    offsets = np.zeros(len(roads) + 1, dtype=np.uint32)
    np.cumsum(lens, out=offsets[1:])
    total = int(offsets[-1])

    xy = np.empty((total, 2), dtype=np.float32)
    for i, r in enumerate(roads):
        s, e = int(offsets[i]), int(offsets[i + 1])
        xy[s:e, 0] = r.x
        xy[s:e, 1] = r.y
    types = np.array([r.type for r in roads], dtype=np.uint32)
    return xy, offsets, types
