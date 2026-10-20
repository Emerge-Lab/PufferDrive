"""lb1/ground_truth.py: the Leaderboard 1.0 criteria recomputed from plain CARLA-frame geometry."""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pufferlib.ocean.cosim.carla.lb1.ground_truth import LeaderboardOneGroundTruth, segments_intersect


EGO_EXTENT_X = 2.4
LANE_WIDTH = 3.5
# one light whose stop line crosses the +x lane at x = 50 (ego drives along +x, y = 0)
LIGHT = {
    "id": 11,
    "x": 55.0,
    "y": 6.0,
    "trigger_x": 50.0,
    "trigger_y": 0.0,
    "stop_waypoints": [{"x": 50.0, "y": 0.0, "yaw_deg": 0.0, "lane_width": LANE_WIDTH, "backward": [], "forward": []}],
}
STOP_SIGN = {
    "id": 21,
    "center": [120.0, 0.0, 0.0],
    "extent": [2.0, 1.5, 1.0],
    "yaw_deg": 0.0,
    "lane": [120.0, 0.0, 0.0, 0.0],
}
IN_LANE = [0.0, LANE_WIDTH, -1.0, 0.0, 0.0, 0.0]  # on the driving lane centre, no parking lane, lane yaw 0, no junction


def ego(x, speed=5.0, yaw_deg=0.0):
    return [x, 0.0, 0.0, yaw_deg, speed, 0.0, 0.0, 0.0, 0.0, 0.0]


def snapshot(x, light_state="Red", speed=5.0, yaw_deg=0.0):
    return {"ego": ego(x, speed, yaw_deg), "partners": [], "light_states": [[11, light_state]], "ego_light_id": -1}


def test_segments_intersect():
    assert segments_intersect((0, 0), (2, 0), (1, -1), (1, 1))
    assert not segments_intersect((0, 0), (2, 0), (3, -1), (3, 1))
    assert segments_intersect((0, 0), (2, 0), (2, 0), (2, 1))  # touching end point


def test_red_light_fires_once_when_the_tail_crosses_the_stop_line():
    truth = LeaderboardOneGroundTruth([LIGHT], [], EGO_EXTENT_X)
    observation = {"collision": [], "lane": IN_LANE}
    assert truth.update(snapshot(45.0), observation)["red_light"] == 0.0  # nose over, tail not yet
    assert truth.update(snapshot(52.5), observation)["red_light"] == 1.0  # tail line 50.1..49.6 straddles x = 50
    assert truth.update(snapshot(52.6), observation)["red_light"] == 0.0  # same light never fires twice


def test_green_light_and_oncoming_lane_do_not_fire():
    truth = LeaderboardOneGroundTruth([LIGHT], [], EGO_EXTENT_X)
    observation = {"collision": [], "lane": IN_LANE}
    assert truth.update(snapshot(52.5, light_state="Green"), observation)["red_light"] == 0.0
    oncoming = snapshot(47.5, yaw_deg=180.0)  # driving -x: the stop line faces the other lane
    oncoming["ego"][4] = -5.0
    assert truth.update(oncoming, {"collision": [], "lane": IN_LANE})["red_light"] == 0.0


def test_stop_sign_run_vs_stopped():
    observation = {"collision": [], "lane": IN_LANE}
    run = LeaderboardOneGroundTruth([], [STOP_SIGN], EGO_EXTENT_X)
    flags = [run.update(snapshot(x, speed=5.0), observation)["stop_sign"] for x in (100.0, 110.0, 119.5, 123.0, 150.0)]
    assert flags == [0.0, 0.0, 0.0, 1.0, 0.0]  # targeted at 100 m (20 m lookahead), inside at 119.5, fires once past it

    stopped = LeaderboardOneGroundTruth([], [STOP_SIGN], EGO_EXTENT_X)
    for x, speed in ((100.0, 5.0), (119.5, 0.05), (123.0, 3.0), (150.0, 5.0)):
        assert stopped.update(snapshot(x, speed=speed), observation)["stop_sign"] == 0.0


def test_outside_lane_and_wrong_lane():
    truth = LeaderboardOneGroundTruth([], [], EGO_EXTENT_X)
    far = [LANE_WIDTH / 2.0 + 1.4, LANE_WIDTH, -1.0, 0.0, 0.0, 0.0]
    assert truth.update(snapshot(10.0), {"collision": [], "lane": far})["offroad"] == 1.0
    on_parking = [LANE_WIDTH / 2.0 + 1.4, LANE_WIDTH, 0.5, 2.5, 0.0, 0.0]  # closer to a parking lane centre
    assert truth.update(snapshot(10.0), {"collision": [], "lane": on_parking})["offroad"] == 0.0
    wrong_way = snapshot(10.0, yaw_deg=170.0)
    assert truth.update(wrong_way, {"collision": [], "lane": IN_LANE})["offroad"] == 1.0
    junction = [0.0, LANE_WIDTH, -1.0, 0.0, 0.0, 1.0]
    assert truth.update(wrong_way, {"collision": [], "lane": junction})["offroad"] == 0.0


def test_collision_flag_follows_the_sensor_events():
    truth = LeaderboardOneGroundTruth([], [], EGO_EXTENT_X)
    assert truth.update(snapshot(10.0), {"collision": ["vehicle.audi.a2"], "lane": IN_LANE})["collision"] == 1.0
    assert truth.update(snapshot(11.0), {"collision": [], "lane": IN_LANE})["collision"] == 0.0
