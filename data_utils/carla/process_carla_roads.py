import sys
import os
import pyxodr
import json
import numpy as np
from lxml import etree
from pyxodr.road_objects.road import Road
from pyxodr.road_objects.lane import Lane, ConnectionPosition, LaneOrientation, TrafficOrientation
from pyxodr.road_objects.junction import Junction
from pyxodr.road_objects.lane_section import LaneSection
from pyxodr.road_objects.network import RoadNetwork
from shapely.geometry import Point, LineString, MultiPolygon, Polygon
from shapely.ops import unary_union, polygonize
from JunctionOutlineTest import fuse_existing_polygons, plot_junction, snap_lines_together
from enum import IntEnum
import random
import string

# Global Variables
max_z = -1e6
min_z = 1e6


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


def save_lane_section_to_json(xodr_json, id, road_edges, road_lines, lanes, road_id, junction_id):
    roads = xodr_json.get("roads", [])
    for road_edge in road_edges:
        # edge_polygon = Polygon(road_edge)
        edge_data = {
            "id": id,
            "map_element_id": int(MapType.ROAD_EDGE_BOUNDARY),
            "type": "road_edge",
            "road_id": road_id,
            "junction_id": junction_id,
            "geometry": [{"x": float(pt[0]), "y": float(pt[1]), "z": 0.0} for pt in road_edge],
        }
        roads.append(edge_data)
        id += 1
    for road_line in road_lines:
        line_data = {
            "id": id,
            "map_element_id": int(MapType.ROAD_LINE_BROKEN_SINGLE_WHITE),
            "type": "road_line",
            "road_id": road_id,
            "junction_id": junction_id,
            "geometry": [{"x": float(pt[0]), "y": float(pt[1]), "z": 0.0} for pt in road_line],
        }
        roads.append(line_data)
        id += 1
    for lane in lanes:
        lane_data = {
            "id": id,
            "map_element_id": int(MapType.LANE_SURFACE_STREET),
            "type": "lane",
            "road_id": road_id,
            "junction_id": junction_id,
            "geometry": [{"x": float(pt[0]), "y": float(pt[1]), "z": 0.0} for pt in lane],
        }
        roads.append(lane_data)
        id += 1
    xodr_json["roads"] = roads
    return id


def get_lane_data(lane, type="BOUNDARY", check_dir=True):
    global max_z, min_z
    if type == "BOUNDARY":
        points = lane.boundary_line
        points = np.append(points, lane.lane_z_coords[:, np.newaxis], axis=1)
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


def sum_pts(road_elts):
    road_geometries = [len(elt) for elt in road_elts]
    return sum(road_geometries)


def create_empty_json(town_name):
    def random_string(length=8):
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    json_data = {
        "name": town_name,
        "scenario_id": random_string(12),
        "objects": [],
        "roads": [],
        "tl_states": {},
        "metadata": {"sdc_track_index": 0, "tracks_to_predict": [], "objects_of_interest": []},
    }
    return json_data


def interpolate_points(start_pt, end_pt, resolution) -> np.ndarray:
    start_pt = np.array(start_pt)
    end_pt = np.array(end_pt)
    line_vec = end_pt - start_pt
    line_len = np.linalg.norm(line_vec[:2])
    num_points = max(2, int(line_len // resolution) + 1)
    points = np.array([start_pt + (line_vec * (i / (num_points - 1))) for i in range(num_points)])
    return points


def get_listof_polylines_for_junction(junction_data):
    list_of_polylines = []
    if "road_edges" in junction_data:
        for road_edges in junction_data["road_edges"]:
            road_edges_2d = [np.array([[pt[0], pt[1]] for pt in line]) for line in road_edges]
            list_of_polylines.extend(road_edges_2d)
    if "stop_lines" in junction_data:
        list_of_polylines.extend(junction_data["stop_lines"])
    return list_of_polylines


def generate_carla_road(
    town_name,
    source_dir,
    carla_map_dir,
    resolution,
    dest_dir,
    max_samples,
    print_number_of_sample_truncations,
    junction_filter_thresh=0.3,
):
    global max_z, min_z
    src_file_path = os.path.join(source_dir, f"{town_name}.json")
    dst_file_path = os.path.join(dest_dir, f"{town_name}.json")
    if not os.path.isfile(src_file_path):
        print(f"Warning: {src_file_path} does not exist, creating empty file.")
        empty_json = create_empty_json(town_name)
        with open(src_file_path, "w") as f:
            json.dump(empty_json, f, indent=2)

    with open(src_file_path, "r") as f:
        xodr_json = json.load(f)
    xodr_json["roads"] = []

    with open(dst_file_path, "w") as f:
        json.dump(xodr_json, f, indent=2)

    odr_file = os.path.join(carla_map_dir, town_name + ".xodr")

    road_network = RoadNetwork(xodr_file_path=odr_file, resolution=resolution, max_samples=max_samples)
    roads = road_network.get_roads()

    junctions_map = {}

    # Go only till last "driving" lane("parking" NTD)
    # "median" lane means a road edge(add after all of them appear)

    id = 0
    roads_json_cnt = [[], [], []]
    max_z = -1e6
    min_z = 1e6
    for road_obj in roads:
        lane_sections = road_obj.lane_sections
        for lane_section in lane_sections:
            road_edges = []
            road_lines = []
            lanes = []

            left_immediate_driveable = False
            right_immediate_driveable = False

            # Left Lanes
            add_lane_data = False
            add_edge_data = False
            previous_lane = None
            for i, left_lane in enumerate(lane_section.left_lanes):
                if left_lane.type == "driving" or left_lane.type == "parking":
                    if i == 0:
                        left_immediate_driveable = True

                    if add_lane_data:
                        road_line_data = get_lane_data(previous_lane, "BOUNDARY")
                        road_line_data = road_line_data
                        road_lines.append(road_line_data)
                        lanes.append(get_lane_data(previous_lane, "CENTERLINE"))
                    # Add outer edge as road edge
                    elif add_edge_data:
                        road_edges.append(get_lane_data(previous_lane, "BOUNDARY"))
                    add_lane_data = True
                    add_edge_data = False
                else:
                    # Add inner lane as road edge
                    if add_lane_data and i != 0:
                        lanes.append(get_lane_data(previous_lane, "CENTERLINE"))
                        road_edges.append(get_lane_data(previous_lane, "BOUNDARY"))
                    add_edge_data = True
                    add_lane_data = False
                previous_lane = left_lane

            if add_lane_data:
                lanes.append(get_lane_data(previous_lane, "CENTERLINE"))
                road_edges.append(get_lane_data(previous_lane, "BOUNDARY"))

            # Right Lanes
            add_lane_data = False
            add_edge_data = False
            previous_lane = None
            for i, right_lane in enumerate(lane_section.right_lanes):
                if right_lane.type == "driving" or right_lane.type == "parking":
                    if i == 0:
                        right_immediate_driveable = True

                    if add_lane_data:
                        road_line_data = get_lane_data(previous_lane, "BOUNDARY")
                        road_line_data = road_line_data
                        road_lines.append(road_line_data)
                        lanes.append(get_lane_data(previous_lane, "CENTERLINE"))
                    # Add outer edge as road edge
                    elif add_edge_data:
                        road_edges.append(get_lane_data(previous_lane, "BOUNDARY"))
                    add_lane_data = True
                    add_edge_data = False
                else:
                    # Add inner lane as road edge
                    if add_lane_data and i != 0:
                        lanes.append(get_lane_data(previous_lane, "CENTERLINE"))
                        road_edges.append(get_lane_data(previous_lane, "BOUNDARY"))
                    add_edge_data = True
                    add_lane_data = False
                previous_lane = right_lane

            if add_lane_data:
                lanes.append(get_lane_data(previous_lane, "CENTERLINE"))
                road_edges.append(get_lane_data(previous_lane, "BOUNDARY"))

            # If atleast one side has no immediate driveable lane add center as road edge
            if not left_immediate_driveable or not right_immediate_driveable:
                road_edges.append(lane_section.lane_section_reference_line)
            else:
                road_lines.append(lane_section.lane_section_reference_line)

            if len(road_lines) == 0 and len(lanes) == 0:
                road_edges = []
            if road_obj.__getitem__("junction") != "-1":
                junction_id = road_obj.__getitem__("junction")
                if junction_id not in junctions_map:
                    junctions_map[junction_id] = {}
                if "road_edges" not in junctions_map[junction_id]:
                    junctions_map[junction_id]["road_edges"] = []
                junctions_map[junction_id]["road_edges"].append(road_edges)
                road_edges = []

            id = save_lane_section_to_json(
                xodr_json,
                id,
                road_edges,
                road_lines,
                lanes,
                road_obj.__getitem__("id"),
                road_obj.__getitem__("junction"),
            )
            roads_json_cnt[0].append(len(road_edges))
            roads_json_cnt[1].append(len(road_lines))
            roads_json_cnt[2].append(len(lanes))
        #     break
        # break

        # Add stop lines for junctions using predecessor and successor to junction info
        if road_obj.road_xml.get("junction") == "-1":
            if (
                len(road_obj.road_xml.findall("link")) > 0
                and len(road_obj.road_xml.findall("link")[0].findall("predecessor")) > 0
                and road_obj.road_xml.findall("link")[0].findall("predecessor")[0].get("elementType") == "junction"
            ):
                junction_id = road_obj.road_xml.findall("link")[0].findall("predecessor")[0].get("elementId")
                if junction_id not in junctions_map:
                    junctions_map[junction_id] = {}
                if "stop_lines" not in junctions_map[junction_id]:
                    junctions_map[junction_id]["stop_lines"] = []

                # Add stop lines
                if len(road_obj.lane_sections) > 0 and len(road_obj.lane_sections[0].left_lanes) > 0:
                    # Get the last driving lane on the left side
                    for left_lane in road_obj.lane_sections[-1].left_lanes:
                        if left_lane.type == "driving":
                            last_driving_lane = left_lane
                    start_pt = road_obj.reference_line[0]
                    end_pt = last_driving_lane.boundary_line[0]
                    # Interpolate points between start_pt and end_pt at given resolution
                    points = interpolate_points(start_pt, end_pt, resolution)
                    junctions_map[junction_id]["stop_lines"].append(points)

                if len(road_obj.lane_sections) > 0 and len(road_obj.lane_sections[0].right_lanes) > 0:
                    # Get the last driving lane on the right side
                    for right_lane in road_obj.lane_sections[-1].right_lanes:
                        if right_lane.type == "driving":
                            last_driving_lane = right_lane
                    start_pt = road_obj.reference_line[0]
                    end_pt = last_driving_lane.boundary_line[0]
                    # Interpolate points between start_pt and end_pt at given resolution
                    points = interpolate_points(start_pt, end_pt, resolution)
                    junctions_map[junction_id]["stop_lines"].append(points)

            if (
                len(road_obj.road_xml.findall("link")) > 0
                and len(road_obj.road_xml.findall("link")[0].findall("successor")) > 0
                and road_obj.road_xml.findall("link")[0].findall("successor")[0].get("elementType") == "junction"
            ):
                junction_id = road_obj.road_xml.findall("link")[0].findall("successor")[0].get("elementId")
                if junction_id not in junctions_map:
                    junctions_map[junction_id] = {}
                if "stop_lines" not in junctions_map[junction_id]:
                    junctions_map[junction_id]["stop_lines"] = []

                # Add stop lines
                if len(road_obj.lane_sections) > 0 and len(road_obj.lane_sections[-1].left_lanes) > 0:
                    # Get the last driving lane on the left side
                    for left_lane in road_obj.lane_sections[-1].left_lanes:
                        if left_lane.type == "driving":
                            last_driving_lane = left_lane
                    start_pt = road_obj.reference_line[-1]
                    end_pt = last_driving_lane.boundary_line[-1]
                    # Interpolate points between start_pt and end_pt at given resolution
                    points = interpolate_points(start_pt, end_pt, resolution)
                    junctions_map[junction_id]["stop_lines"].append(points)

                if len(road_obj.lane_sections) > 0 and len(road_obj.lane_sections[-1].right_lanes) > 0:
                    # Get the last driving lane on the right side
                    for right_lane in road_obj.lane_sections[-1].right_lanes:
                        if right_lane.type == "driving":
                            last_driving_lane = right_lane
                    start_pt = road_obj.reference_line[-1]
                    end_pt = last_driving_lane.boundary_line[-1]
                    # Interpolate points between start_pt and end_pt at given resolution
                    points = interpolate_points(start_pt, end_pt, resolution)
                    junctions_map[junction_id]["stop_lines"].append(points)

    # Now filter road edges in junction area by polygonizing
    for junction_id in junctions_map:
        list_of_polylines = get_listof_polylines_for_junction(junctions_map[junction_id])
        # Filter out polylines with less than 2 points
        valid_polylines = [poly for poly in list_of_polylines if len(poly) >= 2]
        lines = [LineString(poly) for poly in valid_polylines]
        lines = snap_lines_together(lines, tolerance=0.23)
        snapped_polylines = [np.array(line.coords) for line in lines]
        merged_lines = unary_union(lines)
        polygons = list(polygonize(merged_lines))

        junction_surface = None
        method_used = "precise"

        if polygons:
            # junction_surface = MultiPolygon(polygons)
            junction_surface = fuse_existing_polygons(polygons)
            if junction_id == "703":
                print("Debugging junction 703")
                plot_junction(snapped_polylines, junction_surface, [], [], method_used)
            road_edges_lists = junctions_map[junction_id].get("road_edges", [])
            filtered_road_edges = []
            filtered_road_lines = []
            filtered_lanes = []
            # We only want to filter out road edges
            for road_edges in road_edges_lists:
                for road_edge in road_edges:
                    results = [junction_surface.contains(Point(p)) for p in road_edge]
                    if sum(results) / len(results) <= junction_filter_thresh:
                        filtered_road_edges.append(road_edge)
                    else:
                        print(
                            f"Filtered out road edge of junction {junction_id} with {len(road_edge)} points and {sum(results) / len(results)} inside."
                        )
            id = save_lane_section_to_json(
                xodr_json,
                id,
                filtered_road_edges,
                filtered_road_lines,
                filtered_lanes,
                road_id=-1,
                junction_id=junction_id,
            )
            roads_json_cnt[0].append(sum_pts(filtered_road_edges))
            roads_json_cnt[1].append(sum_pts(filtered_road_lines))
            roads_json_cnt[2].append(sum_pts(filtered_lanes))
        else:
            print(f"Warning: Polylines do not form a closed shape junc_id = {junction_id}. Saving everything.")
            road_edges_lists = junctions_map[junction_id].get("road_edges", [])
            for road_edges in road_edges_lists:
                id = save_lane_section_to_json(xodr_json, id, road_edges, [], [], road_id=-1, junction_id=junction_id)
                roads_json_cnt[0].append(len(road_edges))

    print(f"Total roads JSON count: {sum(roads_json_cnt[0]) + sum(roads_json_cnt[1]) + sum(roads_json_cnt[2])}")

    # Save to file
    with open(dst_file_path, "w") as f:
        json.dump(xodr_json, f, indent=2)

    # Print logs
    if print_number_of_sample_truncations:
        road_network.print_logs_max_samples_hit()


def generate_carla_roads(
    town_names,
    source_dir,
    carla_map_dir,
    resolution,
    dest_dir,
    max_samples,
    juncn_filter_thresholds,
    print_number_of_sample_truncations,
):
    if type(resolution) == float:
        resolution = [resolution] * len(town_names)
    elif type(resolution) != list:
        raise ValueError("Resolution must be a float or a list type")
    elif len(resolution) != len(town_names):
        raise ValueError("Resolution must be of the same length as town_names.")
    for i, town in enumerate(town_names):
        junction_filter_thresh = (
            juncn_filter_thresholds[i] if i < len(juncn_filter_thresholds) else juncn_filter_thresholds[-1]
        )
        print(f"Processing town: {town}")
        generate_carla_road(
            town,
            source_dir,
            carla_map_dir,
            resolution[i],
            dest_dir,
            max_samples,
            print_number_of_sample_truncations=print_number_of_sample_truncations,
            junction_filter_thresh=junction_filter_thresh,
        )


if __name__ == "__main__":
    # town_names = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
    town_names = ["Town07"]
    source_dir = "data_utils/carla/carla"
    dest_dir = "data_utils/carla/carla"
    carla_map_dir = "C:\CarlaMaps"
    resolution = 1.0  # [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # Meters
    max_samples = int(1e5)  # Max points to sample per reference line
    print_number_of_sample_truncations = True  # Enable to see the number of data points lost
    # juncn_filter_thresholds = [0.3, 0.4, 1.0, 0.7, 0.35, 0.5, 0.375, 0.3]     # Final Filtering values, please don't change. Remember to backup towns before enabling for run
    juncn_filter_thresholds = [0.375]
    generate_carla_roads(
        town_names,
        source_dir,
        carla_map_dir,
        resolution,
        dest_dir,
        max_samples,
        juncn_filter_thresholds,
        print_number_of_sample_truncations,
    )
