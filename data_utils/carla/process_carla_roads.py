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
from shapely.geometry import Polygon
from enum import IntEnum
import random
import string


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


def save_lane_section_to_json(xodr_json, id, road_edges, road_lines, lanes, sidewalks=[]):
    roads = xodr_json.get("roads", [])
    for road_edge in road_edges:
        # edge_polygon = Polygon(road_edge)
        edge_data = {
            "id": id,
            "map_element_id": int(MapType.ROAD_EDGE_BOUNDARY),
            "type": "road_edge",
            "geometry": [{"x": float(pt[0]), "y": float(pt[1]), "z": 0.0} for pt in road_edge],
        }
        roads.append(edge_data)
        id += 1
    for road_line in road_lines:
        line_data = {
            "id": id,
            "map_element_id": int(MapType.ROAD_LINE_BROKEN_SINGLE_WHITE),
            "type": "road_line",
            "geometry": [{"x": float(pt[0]), "y": float(pt[1]), "z": 0.0} for pt in road_line],
        }
        roads.append(line_data)
        id += 1
    for lane in lanes:
        lane_data = {
            "id": id,
            "map_element_id": int(MapType.LANE_SURFACE_STREET),
            "type": "lane",
            "geometry": [{"x": float(pt[0]), "y": float(pt[1]), "z": 0.0} for pt in lane],
        }
        roads.append(lane_data)
        id += 1
    # for sidewalk in sidewalks:
    #     sidewalk_data = {
    #         "id": id,
    #         "map_element_id": int(MapType.LANE_BIKE_LANE),
    #         "type": "sidewalk",
    #         "geometry": [{"x": float(pt[0]), "y": float(pt[1]), "z": 0.0} for pt in sidewalk]
    #     }
    #     roads.append(sidewalk_data)
    #     id += 1
    xodr_json["roads"] = roads
    return id


def get_lane_data(lane, type="BOUNDARY", check_dir=True):
    if type == "BOUNDARY":
        points = lane.boundary_line
        # print(f'Number of boundary pts: {len(points)}')
    elif type == "CENTERLINE":
        points = lane.centre_line
        # print(f'Number of centerline pts: {len(points)}')
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


def generate_carla_road(
    town_name, source_dir, carla_map_dir, resolution, dest_dir, max_samples, print_number_of_sample_truncations
):
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
    print(f"Number of roads in the network: {len(roads)}")
    # print(f"Type: {type(roads[0])}\nRoads: {roads}")
    print(f"Number of lanes in the network: {sum([sum([len(ls.lanes) for ls in r.lane_sections]) for r in roads])}")
    print(
        f"Road 0 lane 1 boundary_pts: {len(roads[0].lane_sections[0].lanes[1].boundary_line)}"
    )  # Sanity check to see if resolution is working

    # Go only till last "driving" lane("parking" NTD)
    # "median" lane means a road edge(add after all of them appear)
    # Add "sidewalk" lane as well

    id = 0
    roads_json_cnt = [[], [], []]
    print(f"Network has {len(roads)} roads.")
    for road_obj in roads:
        # print(f"Road ID: {road_obj.id}")
        lane_sections = road_obj.lane_sections
        # print(f"Lane Sections: {lane_sections}")
        for lane_section in lane_sections:
            # print(f"Lane Section ID: {lane_section.lane_section_ordinal}")
            # print(f"Number of Left Lanes: {len(lane_section.left_lanes)}")
            # print(f"Number of Right Lanes: {len(lane_section.right_lanes)}")
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
            # elif add_edge_data:
            # if previous_lane.type == 'sidewalk':
            #     sidwalks.append(get_lane_data(previous_lane, "BOUNDARY"))

            # print("LEFT STATS")
            # print(f"Number of Road edges: {len(road_edges)}")
            # print(f"Road lines: {len(road_lines)}")
            # print(f"Lanes: {len(lanes)}")
            # print(f"Sidewalks: {len(sidwalks)}")

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
            # elif add_edge_data:
            #     if previous_lane.type == 'sidewalk':
            #         sidwalks.append(get_lane_data(previous_lane, "BOUNDARY"))

            # print(f"Number of Road edges in {road_obj.id}: {len(road_edges)}")
            # print(f"Road lines in {road_obj.id}: {len(road_lines)}")
            # print(f"Lanes in {road_obj.id}: {len(lanes)}")
            # print(f"Sidewalks in {road_obj.id}: {len(sidwalks)}")

            # If atleast one side has no immediate driveable lane add center as road edge
            if not left_immediate_driveable or not right_immediate_driveable:
                road_edges.append(lane_section.lane_section_reference_line)
            else:
                road_line_data = get_lane_data(previous_lane, "BOUNDARY")
                road_lines.append(road_line_data)

            if len(road_lines) == 0 and len(lanes) == 0:
                road_edges = []
            id = save_lane_section_to_json(xodr_json, id, road_edges, road_lines, lanes)
            roads_json_cnt[0].append(len(road_edges))
            roads_json_cnt[1].append(len(road_lines))
            roads_json_cnt[2].append(len(lanes))
            # if len(lanes) == 0 and len(road_lines) != 0:
            #     print(f"Road: {road_obj.id}, Lane Section: {lane_section.lane_section_ordinal}")
            #     print(f"Road edges: {len(road_edges)}, Road lines: {len(road_lines)}, Lanes: {len(lanes)}")
        #     break
        # break
    print(f"Total roads JSON count: {sum(roads_json_cnt[0]) + sum(roads_json_cnt[1]) + sum(roads_json_cnt[2])}")

    # Save to file
    with open(dst_file_path, "w") as f:
        json.dump(xodr_json, f, indent=2)

    # Print logs
    if print_number_of_sample_truncations:
        road_network.print_logs_max_samples_hit()


def generate_carla_roads(
    town_names, source_dir, carla_map_dir, resolution, dest_dir, max_samples, print_number_of_sample_truncations
):
    if type(resolution) == float:
        resolution = [resolution] * len(town_names)
    elif type(resolution) != list:
        raise ValueError("Resolution must be a float or a list type")
    elif len(resolution) != len(town_names):
        raise ValueError("Resolution must be of the same length as town_names.")
    for i, town in enumerate(town_names):
        print(f"Processing town: {town}")
        generate_carla_road(
            town,
            source_dir,
            carla_map_dir,
            resolution[i],
            dest_dir,
            max_samples,
            print_number_of_sample_truncations=print_number_of_sample_truncations,
        )


if __name__ == "__main__":
    town_names = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
    source_dir = "data_utils/carla"
    dest_dir = "data_utils/carla"
    carla_map_dir = "/scratch/pm3881/Carla-0.10.0-Linux-Shipping/CarlaUnreal/Content/Carla/Maps/OpenDrive"
    resolution = 1.0  # [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # Meters
    max_samples = int(1e5)  # Max points to sample per reference line
    print_number_of_sample_truncations = True  # Enable to see the number of data points lost
    generate_carla_roads(
        town_names, source_dir, carla_map_dir, resolution, dest_dir, max_samples, print_number_of_sample_truncations
    )
    # resolution = 1.0
    # town_name = 'town06'
    # generate_carla_road(
    #     town_name,
    #     source_dir,
    #     carla_map_dir,
    #     resolution,
    #     dest_dir,
    #     max_samples,
    #     print_number_of_sample_truncations
    # )
