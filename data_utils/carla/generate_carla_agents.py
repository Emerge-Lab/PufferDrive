import sys
import os
import json
import numpy as np
import random
from lxml import etree
import pyxodr
from pyxodr.road_objects.road import Road
from pyxodr.road_objects.lane import Lane, ConnectionPosition, LaneOrientation, TrafficOrientation
from pyxodr.road_objects.junction import Junction
from pyxodr.road_objects.lane_section import LaneSection
from pyxodr.road_objects.network import RoadNetwork
from shapely.geometry import Polygon
from enum import IntEnum
import argparse


class MapType(IntEnum):
    LANE_UNDEFINED = 0
    LANE_FREEWAY = 1
    LANE_SURFACE_STREET = 2
    LANE_BIKE_LANE = 3
    # Original definition skips 4
    ROAD_LINE_UNKNOWN = 5
    ROAD_LINE_BROKEN_SINGLE_WHITE = 6
    ROAD_LINE_SOLID_SINGLE_WHITE = 7
    ROAD_LINE_SOLID_DOUBLE_WHITE = 8
    ROAD_LINE_BROKEN_SINGLE_YELLOW = 9
    ROAD_LINE_BROKEN_DOUBLE_YELLOW = 10
    ROAD_LINE_SOLID_SINGLE_YELLOW = 11
    ROAD_LINE_SOLID_DOUBLE_YELLOW = 12
    ROAD_LINE_PASSING_DOUBLE_YELLOW = 13
    ROAD_EDGE_UNKNOWN = 14
    ROAD_EDGE_BOUNDARY = 15
    ROAD_EDGE_MEDIAN = 16
    STOP_SIGN = 17
    CROSSWALK = 18
    SPEED_BUMP = 19
    DRIVEWAY = 20  # New womd datatype in v1.2.0: Driveway entrances
    UNKNOWN = -1
    NUM_TYPES = 21


def get_lane_data(lane, type="BOUNDARY", check_dir=True):
    if type == "BOUNDARY":
        points = lane.boundary_line
    elif type == "CENTERLINE":
        points = lane.centre_line
    else:
        raise ValueError(f"Unknown lane data type: {type}")

    if not check_dir:
        return points

    # Check traffic direction
    travel_dir = None
    vector_lane = lane.lane_xml.find(".//userData/vectorLane")
    if vector_lane is not None:
        travel_dir = vector_lane.get("travelDir")

    if travel_dir == "backward":
        # Reverse points for backward travel
        points = points[::-1]

    return points


class RoadLinkObject:
    def __init__(self, road_id, pred_lane_section_ids, succ_lane_section_ids):
        self.road_id = road_id
        self.pred_lane_section_ids = pred_lane_section_ids
        self.succ_lane_section_ids = succ_lane_section_ids
        self.lane_links_map = {}  # map from (lane_id, lane_section_index) to LaneLinkObject
        self.predecessor_roads = []
        self.successor_roads = []


class LaneLinkObject:
    def __init__(self, lane_id, lane_section_index, road_id, lane, lane_centerpoints, forward_dir, is_junction):
        self.lane_id = lane_id
        self.lane_section_index = lane_section_index
        self.road_id = road_id
        self.lane = lane
        self.lane_centerpoints = lane_centerpoints
        self.forward_dir = forward_dir
        self.predecessor_lanes = []
        self.successor_lanes = []
        self.outgoing_edges = []
        self.incoming_edges = []
        self.is_sampled = False
        self.is_junction = is_junction


def get_junction(road_network, junction_id):
    for junction in road_network.get_junctions():
        if junction.id == junction_id:
            return junction
    return None


def get_road(road_network, road_id):
    for road in road_network.get_roads():
        if road.id == road_id:
            return road
    return None


def is_forward_dir(lane):
    # Check traffic direction
    travel_dir = None
    vector_lane = lane.lane_xml.find(".//userData/vectorLane")
    if vector_lane is not None:
        travel_dir = vector_lane.get("travelDir")

    if travel_dir == "forward":
        return True
    return False


def create_lane_link_elements(road_network, roads, road_link_map):
    roads_json_cnt = [[], [], []]
    print(f"Network has {len(roads)} roads.")
    for road_obj in roads:
        lane_sections = road_obj.lane_sections

        is_road_junction = False if road_obj.road_xml.attrib["junction"] == "-1" else True
        pred_lane_section_ids = {}
        for predecessor_xml in road_obj.road_xml.find("link").findall("predecessor"):
            if predecessor_xml.attrib["elementType"] == "road":
                pred_road_id = predecessor_xml.attrib["elementId"]
                if predecessor_xml.attrib["contactPoint"] == "start":
                    pred_lane_section_ids[pred_road_id] = 0
                else:
                    pred_road = get_road(road_network, pred_road_id)
                    pred_lane_section_ids[pred_road_id] = len(pred_road.lane_sections) - 1

        succ_lane_section_ids = {}
        for successor_xml in road_obj.road_xml.find("link").findall("successor"):
            if successor_xml.attrib["elementType"] == "road":
                succ_road_id = successor_xml.attrib["elementId"]
                if successor_xml.attrib["contactPoint"] == "start":
                    succ_lane_section_ids[succ_road_id] = 0
                else:
                    succ_road = get_road(road_network, succ_road_id)
                    succ_lane_section_ids[succ_road_id] = len(succ_road.lane_sections) - 1

        road_link_object = RoadLinkObject(
            road_id=road_obj.id,
            pred_lane_section_ids=pred_lane_section_ids,
            succ_lane_section_ids=succ_lane_section_ids,
        )

        for lane_section in lane_sections:
            road_edges = []
            road_lines = []
            lanes = []
            # sidwalks = []

            left_immediate_driveable = False
            right_immediate_driveable = False

            # Left Lanes
            add_lane_data = False
            add_edge_data = False
            previous_lane = None
            for i, left_lane in enumerate(lane_section.left_lanes):
                if left_lane.type == "driving":  # We only deal with driving lanes
                    if i == 0:
                        left_immediate_driveable = True

                    if add_lane_data:
                        road_line_data = get_lane_data(previous_lane, "BOUNDARY")
                        road_line_data = road_line_data[::40]
                        road_lines.append(road_line_data)
                        waypoints = get_lane_data(previous_lane, "CENTERLINE")
                        lanes.append(waypoints)
                        road_link_object.lane_links_map[(str(previous_lane.id), lane_section.lane_section_ordinal)] = (
                            LaneLinkObject(
                                lane_id=previous_lane.id,
                                lane_section_index=lane_section.lane_section_ordinal,
                                road_id=road_obj.id,
                                lane=previous_lane,
                                lane_centerpoints=waypoints,
                                forward_dir=is_forward_dir(previous_lane),
                                is_junction=is_road_junction,
                            )
                        )
                    # Add outer edge as road edge
                    elif add_edge_data:
                        road_edges.append(get_lane_data(previous_lane, "BOUNDARY"))
                    add_lane_data = True
                    add_edge_data = False
                else:
                    # Add inner lane as road edge
                    if add_lane_data and i != 0:
                        waypoints = get_lane_data(previous_lane, "CENTERLINE")
                        lanes.append(waypoints)
                        road_link_object.lane_links_map[(str(previous_lane.id), lane_section.lane_section_ordinal)] = (
                            LaneLinkObject(
                                lane_id=previous_lane.id,
                                lane_section_index=lane_section.lane_section_ordinal,
                                road_id=road_obj.id,
                                lane=previous_lane,
                                lane_centerpoints=waypoints,
                                forward_dir=is_forward_dir(previous_lane),
                                is_junction=is_road_junction,
                            )
                        )
                        road_edges.append(get_lane_data(previous_lane, "BOUNDARY"))
                    add_edge_data = True
                    add_lane_data = False
                previous_lane = left_lane

            if add_lane_data:
                waypoints = get_lane_data(previous_lane, "CENTERLINE")
                lanes.append(waypoints)
                road_link_object.lane_links_map[(str(previous_lane.id), lane_section.lane_section_ordinal)] = (
                    LaneLinkObject(
                        lane_id=previous_lane.id,
                        lane_section_index=lane_section.lane_section_ordinal,
                        road_id=road_obj.id,
                        lane=previous_lane,
                        lane_centerpoints=waypoints,
                        forward_dir=is_forward_dir(previous_lane),
                        is_junction=is_road_junction,
                    )
                )
                road_edges.append(get_lane_data(previous_lane, "BOUNDARY"))

            # Right Lanes
            add_lane_data = False
            add_edge_data = False
            previous_lane = None
            for i, right_lane in enumerate(lane_section.right_lanes):
                if right_lane.type == "driving":
                    if i == 0:
                        right_immediate_driveable = True

                    if add_lane_data:
                        road_line_data = get_lane_data(previous_lane, "BOUNDARY")
                        road_line_data = road_line_data[::40]
                        road_lines.append(road_line_data)
                        waypoints = get_lane_data(previous_lane, "CENTERLINE")
                        lanes.append(waypoints)
                        road_link_object.lane_links_map[(str(previous_lane.id), lane_section.lane_section_ordinal)] = (
                            LaneLinkObject(
                                lane_id=previous_lane.id,
                                lane_section_index=lane_section.lane_section_ordinal,
                                road_id=road_obj.id,
                                lane=previous_lane,
                                lane_centerpoints=waypoints,
                                forward_dir=is_forward_dir(previous_lane),
                                is_junction=is_road_junction,
                            )
                        )
                    # Add outer edge as road edge
                    elif add_edge_data:
                        road_edges.append(get_lane_data(previous_lane, "BOUNDARY"))
                    add_lane_data = True
                    add_edge_data = False
                else:
                    # Add inner lane as road edge
                    if add_lane_data and i != 0:
                        waypoints = get_lane_data(previous_lane, "CENTERLINE")
                        lanes.append(waypoints)
                        road_link_object.lane_links_map[(str(previous_lane.id), lane_section.lane_section_ordinal)] = (
                            LaneLinkObject(
                                lane_id=previous_lane.id,
                                lane_section_index=lane_section.lane_section_ordinal,
                                road_id=road_obj.id,
                                lane=previous_lane,
                                lane_centerpoints=waypoints,
                                forward_dir=is_forward_dir(previous_lane),
                                is_junction=is_road_junction,
                            )
                        )
                        road_edges.append(get_lane_data(previous_lane, "BOUNDARY"))
                    add_edge_data = True
                    add_lane_data = False
                previous_lane = right_lane

            if add_lane_data:
                waypoints = get_lane_data(previous_lane, "CENTERLINE")
                lanes.append(waypoints)
                road_link_object.lane_links_map[(str(previous_lane.id), lane_section.lane_section_ordinal)] = (
                    LaneLinkObject(
                        lane_id=previous_lane.id,
                        lane_section_index=lane_section.lane_section_ordinal,
                        road_id=road_obj.id,
                        lane=previous_lane,
                        lane_centerpoints=waypoints,
                        forward_dir=is_forward_dir(previous_lane),
                        is_junction=is_road_junction,
                    )
                )
                road_edges.append(get_lane_data(previous_lane, "BOUNDARY"))

            road_link_map[road_obj.id] = road_link_object

            roads_json_cnt[0].append(len(road_edges))
            roads_json_cnt[1].append(len(road_lines))
            roads_json_cnt[2].append(len(lanes))

    print(f"Total roads JSON count: {sum(roads_json_cnt[0]) + sum(roads_json_cnt[1]) + sum(roads_json_cnt[2])}")
    print(f"Lanes count: {sum(roads_json_cnt[2])}")
    total_lane_links = sum(len(obj.lane_links_map) for obj in road_link_map.values())
    assert sum(roads_json_cnt[2]) == total_lane_links


def create_successor_predecessor_elements(road_network, roads, road_link_map):
    stopping_points = 0

    for road_obj in roads:
        road_link_object = road_link_map[road_obj.id]
        lane_sections = road_obj.lane_sections

        for lane_link_obj in road_link_object.lane_links_map.values():
            lane_link_obj.predecessor_lanes = []
            lane_link_obj.successor_lanes = []

            lane = lane_link_obj.lane

            link_xml = lane.lane_xml.find("link")

            successor_is_junction = True
            if link_xml is not None and link_xml.findall("successor") != []:
                successor_is_junction = False
                # Process Successor Links
                for successor in link_xml.findall("successor"):
                    successor_id = successor.get("id")

                    if lane_link_obj.lane_section_index + 1 < len(lane_sections):
                        # Lane link in next lane section
                        lane_link_obj.successor_lanes.append(
                            road_link_object.lane_links_map[(successor_id, lane_link_obj.lane_section_index + 1)]
                        )
                    else:
                        # Lane link in successor roads
                        for road_succ in road_obj.road_xml.find("link").findall("successor"):
                            if road_succ.attrib["elementType"] == "road":
                                succ_road_id = road_succ.attrib["elementId"]
                                succ_road = road_link_map[succ_road_id]
                                succ_lane_section_index = road_link_object.succ_lane_section_ids[succ_road_id]
                                lane_link_obj.successor_lanes.append(
                                    succ_road.lane_links_map[(successor_id, succ_lane_section_index)]
                                )
                            else:
                                # Junction case
                                successor_is_junction = True
            elif successor_is_junction:
                # Handle junction case
                for successor in road_obj.road_xml.find("link").findall("successor"):
                    if successor.attrib["elementType"] == "junction":
                        junction_id = successor.attrib["elementId"]
                        junction = get_junction(road_network, junction_id)
                        connected_lanes = junction.get_lane_junction_lanes(str(lane.id), road_id=road_obj.id)
                        if len(connected_lanes) == 0 and lane_link_obj.forward_dir:
                            stopping_points += 1
                            print(
                                f"Non junction road: {road_obj.id}, lane_section: {lane_link_obj.lane_section_index}, lane: {lane_link_obj.lane.id} has no junction or road links, it is an ending lane"
                            )
                        for conn_lane in connected_lanes:
                            succ_road_obj = get_road(road_network, conn_lane["road_id"])
                            succ_road = road_link_map[conn_lane["road_id"]]
                            succ_lane_section_index = conn_lane["lane_section_index"]
                            if succ_lane_section_index == -1:
                                succ_lane_section_index = len(succ_road_obj.lane_sections) - 1
                            succ_lane_id = conn_lane["lane_id"]
                            lane_link_obj.successor_lanes.append(
                                succ_road.lane_links_map[(str(succ_lane_id), succ_lane_section_index)]
                            )
                    else:
                        if lane_link_obj.forward_dir:
                            # Stopping point
                            stopping_points += 1
                            print(
                                f"Non junction road: {road_obj.id}, lane_section: {lane_link_obj.lane_section_index}, lane: {lane_link_obj.lane.id} has no junction or road links, it is an ending lane"
                            )

            predecessor_is_junction = True
            if link_xml is not None and link_xml.findall("predecessor") != []:
                predecessor_is_junction = False
                # Process Predecessor Links
                for predecessor in link_xml.findall("predecessor"):
                    predecessor_id = predecessor.get("id")

                    if lane_link_obj.lane_section_index - 1 >= 0:
                        # Lane link in previous lane section
                        lane_link_obj.predecessor_lanes.append(
                            road_link_object.lane_links_map[(predecessor_id, lane_link_obj.lane_section_index - 1)]
                        )
                    else:
                        # Lane link in predecessor roads
                        for road_pred in road_obj.road_xml.find("link").findall("predecessor"):
                            if road_pred.attrib["elementType"] == "road":
                                pred_id = road_pred.attrib["elementId"]
                                pred_road = road_link_map[pred_id]
                                pred_lane_section_index = road_link_object.pred_lane_section_ids[pred_id]
                                lane_link_obj.predecessor_lanes.append(
                                    pred_road.lane_links_map[(predecessor_id, pred_lane_section_index)]
                                )
                            else:
                                # Junction case
                                predecessor_is_junction = True
            elif predecessor_is_junction:
                # Handle junction case
                for predecessor in road_obj.road_xml.find("link").findall("predecessor"):
                    if predecessor.attrib["elementType"] == "junction":
                        junction_id = predecessor.attrib["elementId"]
                        junction = get_junction(road_network, junction_id)
                        connected_lanes = junction.get_lane_junction_lanes(lane_id=str(lane.id), road_id=road_obj.id)
                        if len(connected_lanes) == 0 and not lane_link_obj.forward_dir:
                            stopping_points += 1
                            print(
                                f"Non junction road: {road_obj.id}, lane_section: {lane_link_obj.lane_section_index}, lane: {lane_link_obj.lane.id} has no junction or road links, it is a starting lane"
                            )
                        for conn_lane in connected_lanes:
                            pred_road_obj = get_road(road_network, conn_lane["road_id"])
                            pred_road = road_link_map[conn_lane["road_id"]]
                            pred_lane_section_index = conn_lane["lane_section_index"]
                            if pred_lane_section_index == -1:
                                pred_lane_section_index = len(pred_road_obj.lane_sections) - 1
                            pred_lane_id = conn_lane["lane_id"]
                            lane_link_obj.predecessor_lanes.append(
                                pred_road.lane_links_map[(str(pred_lane_id), pred_lane_section_index)]
                            )
                    else:
                        if not lane_link_obj.forward_dir:
                            # Stopping case
                            stopping_points += 1
                            print(
                                f"Non junction road: {road_obj.id}, lane_section: {lane_link_obj.lane_section_index}, lane: {lane_link_obj.lane.id} has no junction or road links, it is a starting lane"
                            )

    print(f"Road network has {stopping_points} stopping points (lanes with no predecessors or successors).")


def add_incoming_outgoing_edges(road_network, roads, road_link_map):
    for road_obj in roads:
        road_link_object = road_link_map[road_obj.id]
        lane_sections = road_obj.lane_sections

        for lane_link_obj in road_link_object.lane_links_map.values():
            lane = lane_link_obj.lane
            link_xml = lane.lane_xml.find("link")

            if lane_link_obj.forward_dir:
                # Forward direction, predecessor lanes are incoming edges, successor lanes are outgoing edges
                lane_link_obj.incoming_edges = lane_link_obj.predecessor_lanes
                lane_link_obj.outgoing_edges = lane_link_obj.successor_lanes
            else:
                # Backward direction, predecessor lanes are outgoing edges, successor lanes are incoming edges
                lane_link_obj.outgoing_edges = lane_link_obj.predecessor_lanes
                lane_link_obj.incoming_edges = lane_link_obj.successor_lanes


def test_linkage(road_link_map):
    # Testing linkage

    start_road_id = "0"
    road_link_object = road_link_map[start_road_id]

    lane_link_keys = list(road_link_object.lane_links_map.keys())
    random_lane_link_key = None
    if random_lane_link_key is None:
        random_lane_link_key = random.choice(lane_link_keys)
    lane_link_obj = road_link_object.lane_links_map[random_lane_link_key]

    print(f"Starting at road id: {start_road_id}")
    print(f"Lane link key: {random_lane_link_key}")
    print(f"Lane id: {lane_link_obj.lane_id}, Lane section index: {lane_link_obj.lane_section_index}")

    # Traverse outgoing edges for 10 steps
    current = lane_link_obj
    for step in range(10):
        print(f"\nStep {step}:")
        print(
            f"  Road id: {current.road_id}, Lane id: {current.lane_id}, Lane section index: {current.lane_section_index}"
        )
        if current.outgoing_edges:
            current = random.choice(current.outgoing_edges)
        else:
            print("  No outgoing edges. Stopping traversal.")
            print("Debugging")
            print(
                f"Outgoing Edges: {[(edge.road_id, str(edge.lane.id), edge.lane_section_index) for edge in current.outgoing_edges]}"
            )
            print(
                f"Incoming Edges: {[(edge.road_id, str(edge.lane.id), edge.lane_section_index) for edge in current.incoming_edges]}"
            )
            print(
                f"Successors: {[(edge.road_id, str(edge.lane.id), edge.lane_section_index) for edge in current.successor_lanes]}"
            )
            print(
                f"Predecessors: {[(edge.road_id, str(edge.lane.id), edge.lane_section_index) for edge in current.predecessor_lanes]}"
            )
            break


def print_traj_stats(waypoints_list, num_timestamps, episode_length):
    # Extract positions
    positions = [wp["position"] for wp in waypoints_list if "position" in wp]
    positions = np.array(positions)

    # Compute consecutive distances
    if positions.shape[0] < 2:
        print("Not enough waypoints to compute statistics.")
        return

    diffs = positions[1:] - positions[:-1]
    distances = np.linalg.norm(diffs, axis=1)
    distance_traversed = np.sum(distances)

    # Compute max speed along trajectory
    timestamp_dur = episode_length / num_timestamps
    speeds = distances / timestamp_dur
    max_speed_traj = np.max(speeds)

    print(f"Distance traversed: {distance_traversed} meters")
    print(f"Max speed in trajectory: {max_speed_traj} m/s")


def check_geometry(point, all_geometries):
    for pt in all_geometries:
        dist = np.linalg.norm(np.array(pt) - np.array(point))
        if dist < 1e-9:  # threshold, adjust as needed
            return False
    return True


class AABB:
    def __init__(self, center_point, length, width, height):
        """
        Axis-aligned bounding box defined by center and extents.
        Compute the 8 cuboid corners then derive x/y/z ranges from them.
        """
        self.center_point = np.asarray(center_point, dtype=float)
        if self.center_point.size == 2:
            # assume z = 0 if omitted
            self.center_point = np.append(self.center_point, 0.0)

        # Take max of extents in 2D
        self.length = max(float(length), float(width))
        self.width = self.length
        self.height = float(height)

        # Double the extents to prevent collision with any orientation
        self.length = 2.0 * float(self.length)
        self.width = 2.0 * float(self.width)
        self.height = 2.0 * float(self.height)

        half = np.array([self.length / 2.0, self.width / 2.0, self.height / 2.0], dtype=float)

        signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)], dtype=float)

        self.corners = self.center_point + signs * half

        self.min_point = np.min(self.corners, axis=0)
        self.max_point = np.max(self.corners, axis=0)

        self.x_range = [float(self.min_point[0]), float(self.max_point[0])]
        self.y_range = [float(self.min_point[1]), float(self.max_point[1])]
        self.z_range = [float(self.min_point[2]), float(self.max_point[2])]

    def intersects(self, other):
        overlap_x = (self.x_range[0] <= other.x_range[1]) and (self.x_range[1] >= other.x_range[0])
        overlap_y = (self.y_range[0] <= other.y_range[1]) and (self.y_range[1] >= other.y_range[0])
        overlap_z = (self.z_range[0] <= other.z_range[1]) and (self.z_range[1] >= other.z_range[0])

        return overlap_x and overlap_y and overlap_z


def AABB_unit_test():
    center = [0.0, 0.0, 0.0]
    length = 2.0
    width = 4.0
    height = 6.0
    aabb = AABB(center, length, width, height)
    assert np.allclose(aabb.min_point, [-4.0, -4.0, -6.0])
    assert np.allclose(aabb.max_point, [4.0, 4.0, 6.0])
    assert np.allclose(aabb.corners[0], [-4.0, -4.0, -6.0])
    assert np.allclose(aabb.corners[7], [4.0, 4.0, 6.0])
    print("AABB unit test passed.")


def generate_traj_data(
    start_object_aabbs,  # To prevent init collisions
    road_link_map,
    num_timestamps=90,
    resolution=0.1,
    episode_length=9,
    avg_speed=2,
    random_sampling_variation=1,
    init_resample=True,
    lane_change_resample=True,
    all_geometries=[],
    obj_length=4.5,
    obj_width=2.0,
    obj_height=1.8,
    num_attempts=500,
    initial_velocity=None,  # If None, use calculated velocity; otherwise set to this value (m/s)
):
    avg_cons_pts_dist = resolution
    time_step_dur = episode_length / num_timestamps
    sampling_length = int((avg_speed * time_step_dur) / avg_cons_pts_dist)

    num_samples = 0
    # Pick a random start_lane_key
    while True:
        if num_samples > 1000:
            print("Failed to find a free spot after 1000 samples, returning None")
            return None
        num_samples += 1
        road_link_keys = list(road_link_map.keys())
        start_key = random.choice(road_link_keys)
        lane_link_keys = list(road_link_map[start_key].lane_links_map.keys())
        if lane_link_keys != []:
            if init_resample:
                start_lane_key = random.choice(lane_link_keys)
                lane_link_obj = road_link_map[start_key].lane_links_map[start_lane_key]
                found_free_spot = False
                for attempt in range(num_attempts):
                    idx = random.randint(0, len(lane_link_obj.lane_centerpoints) - 1)
                    check_AABB = AABB(lane_link_obj.lane_centerpoints[idx], obj_length, obj_width, obj_height)
                    if not any(check_AABB.intersects(aabb) for aabb in start_object_aabbs):
                        found_free_spot = True
                        break
                if found_free_spot:
                    break
                # if no free spot found need to choose a different lane link
            else:
                start_lane_key = random.choice(lane_link_keys)
                lane_link_obj = road_link_map[start_key].lane_links_map[start_lane_key]
                if not lane_link_obj.is_sampled:
                    found_free_spot = False
                    for attempt in range(num_attempts):
                        idx = random.randint(0, len(lane_link_obj.lane_centerpoints) - 1)
                        check_AABB = AABB(lane_link_obj.lane_centerpoints[idx], obj_length, obj_width, obj_height)
                        if not any(check_AABB.intersects(aabb) for aabb in start_object_aabbs):
                            found_free_spot = True
                            break
                        else:
                            pass
                    if found_free_spot:
                        break
                    # if no free spot found need to choose a different lane link

    waypoints_list = []
    current_lane_link = lane_link_obj
    current_lane_link.is_sampled = True

    start_object_aabbs.append(AABB(current_lane_link.lane_centerpoints[idx], obj_length, obj_width, obj_height))

    waypoints_list.append(
        {
            "timestamp": 0,
            "position": current_lane_link.lane_centerpoints[idx].tolist()
            if hasattr(current_lane_link.lane_centerpoints[idx], "tolist")
            else list(current_lane_link.lane_centerpoints[idx]),
            "lane_id": current_lane_link.lane_id,
            "lane_section_index": current_lane_link.lane_section_index,
            "road_id": current_lane_link.road_id,
        }
    )
    change_lane = False

    for t in range(num_timestamps):
        if change_lane:
            # Pick a random outgoing edge
            if current_lane_link.outgoing_edges:
                if lane_change_resample:
                    current_lane_link = random.choice(current_lane_link.outgoing_edges)
                else:
                    unsampled_edges = [
                        edge for edge in current_lane_link.outgoing_edges if not getattr(edge, "is_sampled", False)
                    ]
                    if unsampled_edges != []:
                        current_lane_link = random.choice(unsampled_edges)
                    else:
                        current_lane_link = random.choice(current_lane_link.outgoing_edges)
                    current_lane_link.is_sampled = True
                    idx = 0  # Lane connection width is zero so reset at start

            else:
                print("No outgoing edge, stopping trajectory")
                while len(waypoints_list) < num_timestamps + 1:
                    waypoints_list.append(
                        {
                            "position": waypoint.tolist() if hasattr(waypoint, "tolist") else list(waypoint),
                            "velocity": {"x": 0.0, "y": 0.0},
                            "heading": 0.0 if len(waypoints_list) == 0 else waypoints_list[-1]["heading"],
                            "lane_id": current_lane_link.lane_id,
                            "lane_section_index": current_lane_link.lane_section_index,
                            "road_id": current_lane_link.road_id,
                            "change_lane": change_lane,
                        }
                    )
                break

        # Randomize sampling length
        curr_sampling_length = sampling_length + random.randint(-random_sampling_variation, random_sampling_variation)
        next_idx = idx + curr_sampling_length

        if next_idx < len(current_lane_link.lane_centerpoints):
            waypoint = current_lane_link.lane_centerpoints[next_idx]
            idx = next_idx
            change_lane = False
        else:
            waypoint = current_lane_link.lane_centerpoints[-1]
            change_lane = True

        # Velocity and heading calculation
        v_x = (waypoint[0] - waypoints_list[t - 1]["position"][0]) / time_step_dur
        v_y = (waypoint[1] - waypoints_list[t - 1]["position"][1]) / time_step_dur
        heading = np.arctan2(v_y, v_x)

        waypoints_list.append(
            {
                "position": waypoint.tolist() if hasattr(waypoint, "tolist") else list(waypoint),
                "velocity": {"x": v_x, "y": v_y},
                "heading": heading,
                "lane_id": current_lane_link.lane_id,
                "lane_section_index": current_lane_link.lane_section_index,
                "road_id": current_lane_link.road_id,
            }
        )

    # Change first waypoint with velocity and heading from pos t=0 to pos t=1 keeping everything else same
    pos0 = np.array(waypoints_list[0]["position"])
    pos1 = np.array(waypoints_list[1]["position"])
    heading = float(np.arctan2(pos1[1] - pos0[1], pos1[0] - pos0[0]))

    if initial_velocity is None:
        # Use average velocity from rest of trajectory
        mean_vx = np.mean([wp["velocity"]["x"] for wp in waypoints_list[1:]])
        mean_vy = np.mean([wp["velocity"]["y"] for wp in waypoints_list[1:]])
    else:
        # Use configured initial velocity along the heading direction
        mean_vx = initial_velocity * np.cos(heading)
        mean_vy = initial_velocity * np.sin(heading)

    waypoints_list[0] = {
        "position": waypoints_list[0]["position"],
        "velocity": {"x": mean_vx, "y": mean_vy},
        "heading": heading,
        "lane_id": waypoints_list[0]["lane_id"],
        "lane_section_index": waypoints_list[0]["lane_section_index"],
        "road_id": waypoints_list[0]["road_id"],
    }

    return waypoints_list


def save_object_to_json(
    xodr_json,
    road_link_map,
    id,
    start_object_aabbs,
    resolution=0.1,
    object_type="vehicle",
    all_geometries=[],
    initial_velocity=None,  # If None, use mean velocity; if 0.0 start at rest; otherwise set to this value (m/s)
    init_resample=True,
    lane_change_resample=True,
    avg_speed=2,
    obj_length=4.5,
    obj_width=2.0,
    obj_height=1.8,
):
    traj_data = generate_traj_data(
        start_object_aabbs,
        road_link_map=road_link_map,
        resolution=resolution,
        init_resample=init_resample,
        lane_change_resample=lane_change_resample,
        all_geometries=all_geometries,
        obj_length=obj_length,
        obj_width=obj_width,
        obj_height=obj_height,
        initial_velocity=initial_velocity,
        avg_speed=avg_speed,
    )

    headings = []
    positions = []
    velocities = []
    for i, traj in enumerate(traj_data):
        x = traj["position"][0]
        y = traj["position"][1]
        z = traj["position"][2] + obj_height / 2.0  # Adjust for object height
        positions.append({"x": float(x), "y": float(y), "z": float(z)})
        v_x = traj["velocity"]["x"]
        v_y = traj["velocity"]["y"]
        velocities.append({"x": float(v_x), "y": float(v_y)})
        headings.append(float(traj["heading"]))

    object_data = {
        "position": list(positions),
        "width": obj_width,
        "length": obj_length,
        "height": obj_height,
        "id": id,
        "heading": list(headings),
        "velocity": list(velocities),
        "valid": [True] * len(positions),
        "goalPosition": {
            "x": float(positions[-1]["x"]),
            "y": float(positions[-1]["y"]),
            "z": float(positions[-1]["z"]),
        },
        "type": object_type,
        "mark_as_expert": False,
    }

    objects = xodr_json.get("objects", [])
    objects.append(object_data)
    xodr_json["objects"] = objects
    return id + 1


def generate_data_each_map(
    town_names,
    carla_map_dir,
    resolution,
    input_json_base_path,
    output_json_root_dir,
    num_data_per_map,
    num_objects,
    make_only_first_agent_controllable,
    initial_velocity=None,  # If None, use calculated velocity; if 0.0 start at rest; otherwise set to this value (m/s)
    init_resample=True,
    lane_change_resample=True,
    avg_speed=2,
):
    os.makedirs(output_json_root_dir, exist_ok=True)
    for town_name in town_names:
        json_file = town_name + ".json"
        odr_file = os.path.join(carla_map_dir, "T" + town_name[1:] + ".xodr")
        input_json_path = os.path.join(input_json_base_path, json_file)

        for id in range(num_data_per_map):
            output_json_path = os.path.join(output_json_root_dir, f"{town_name}_{id}.json")

            road_network = RoadNetwork(xodr_file_path=odr_file, resolution=resolution)
            roads = road_network.get_roads()
            print(f"Number of roads in the network: {len(roads)}")

            # Create a dictionary to map road_id to RoadLinkObject
            road_link_map = {}

            # Create the lane link elements
            create_lane_link_elements(road_network, roads, road_link_map)

            # Create successor predecessor elements
            create_successor_predecessor_elements(road_network, roads, road_link_map)

            # Add outgoing and incoming edges based on driving direction
            add_incoming_outgoing_edges(road_network, roads, road_link_map)

            # Test linkage
            # test_linkage(road_link_map)

            with open(input_json_path, "r") as f:
                xodr_json = json.load(f)

            roads = xodr_json["roads"]
            all_geometries = []
            for road in roads:
                geometry = [[pt["x"], pt["y"], pt["z"]] for pt in road["geometry"]]
                all_geometries.extend(geometry)

            print(f"Total number of geometry points: {len(all_geometries)}")

            xodr_json["objects"] = []

            # Start AABBs to prevent init collisions or extreme closeness
            start_object_aabbs = []

            for i in range(num_objects):
                id = i + 1
                save_object_to_json(
                    xodr_json,
                    road_link_map,
                    id,
                    start_object_aabbs,
                    resolution=resolution,
                    all_geometries=all_geometries,
                    initial_velocity=initial_velocity,
                    init_resample=init_resample,
                    lane_change_resample=lane_change_resample,
                    avg_speed=avg_speed,
                )

            # Make first agent only controllable
            if make_only_first_agent_controllable:
                for i, obj in enumerate(xodr_json.get("objects", [])):
                    if i == 0:
                        obj["mark_as_expert"] = False
                    else:
                        obj["mark_as_expert"] = True

            with open(output_json_path, "w") as f:
                json.dump(xodr_json, f, indent=2)

            with open(output_json_path, "r") as f:
                xodr_json = json.load(f)
            assert len(xodr_json.get("objects", [])) == num_objects

            print(f"Saved {num_objects} objects to {output_json_path}")


if __name__ == "__main__":
    AABB_unit_test()
    parser = argparse.ArgumentParser(description="Process CARLA XODR and generate data.")

    parser.add_argument(
        "--town_names",
        nargs="+",
        default=["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"],
        help="List of CARLA town names",
    )
    parser.add_argument(
        "--input_json_base_path",
        type=str,
        default="data_utils/carla/carla_py123d",
        help="Base path for input JSON files",
    )
    parser.add_argument(
        "--output_json_root_dir",
        type=str,
        default="data/processed/carla_data",
        help="Root directory for output JSON files",
    )
    parser.add_argument(
        "--carla_map_dir", type=str, default="data/CarlaXODRs", help="Directory containing CARLA XODR files"
    )
    parser.add_argument("--resolution", type=float, default=0.1, help="Resolution for road network processing")
    parser.add_argument("--num_data_per_map", type=int, default=8, help="Number of data samples per map")
    parser.add_argument("--num_objects", type=int, default=32, help="Number of objects per data sample")
    parser.add_argument(
        "--make_only_first_agent_controllable", action="store_true", help="If set, only the first agent is controllable"
    )
    parser.add_argument(
        "--initial_velocity",
        type=float,
        default=0.0,
        help="Initial velocity for objects (set to None for mean velocity)",
    )
    parser.add_argument("--init_resample", action="store_true", help="Enable resampling of initial lane")
    parser.add_argument("--lane_change_resample", action="store_true", help="Enable resampling of lane change lane")
    parser.add_argument("--avg_speed", type=float, default=1.0, help="Average speed of the objects in m/s")

    args = parser.parse_args()

    town_names = args.town_names
    input_json_base_path = args.input_json_base_path
    output_json_root_dir = args.output_json_root_dir
    carla_map_dir = args.carla_map_dir
    resolution = args.resolution
    num_data_per_map = args.num_data_per_map
    num_objects = args.num_objects
    make_only_first_agent_controllable = args.make_only_first_agent_controllable
    initial_velocity = args.initial_velocity
    init_resample = args.init_resample
    lane_change_resample = args.lane_change_resample
    avg_speed = args.avg_speed
    generate_data_each_map(
        town_names,
        carla_map_dir,
        resolution,
        input_json_base_path,
        output_json_root_dir,
        num_data_per_map,
        num_objects=num_objects,
        make_only_first_agent_controllable=make_only_first_agent_controllable,
        initial_velocity=initial_velocity,
        init_resample=init_resample,
        lane_change_resample=lane_change_resample,
        avg_speed=avg_speed,
    )
