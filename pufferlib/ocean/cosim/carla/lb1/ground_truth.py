"""CARLA-side ground truth of the Leaderboard 1.0 criteria, recomputed in the policy server from what the
Python 3.7 client observes (it cannot run CaRL's criteria the Leaderboard 2.0 agent uses). Feeds the
carla_* telemetry columns and the infraction clips; the leaderboard's own result.json stays the score.

Per-tick observation sent by lb1/leaderboard_agent.py (CARLA frame):
  {"collision": [other actor type ids since the last tick],
   "lane": [driving_distance_m, driving_lane_width_m, parking_distance_m (-1: no parking lane),
            parking_lane_width_m, driving_lane_yaw_deg, driving_is_junction]}
plus the snapshot's ego state. Ports of scenario_runner's RunningRedLightTest (tail line crossing the
0.4-lane-width stop line of a red light within 15 m, once per light), RunningStopTest (entered a stop
trigger volume, left its 20 m influence without a standstill) and OutsideRouteLanesTest (farther than
half a lane plus 1.3 m from both the driving and the parking lane, or facing more than 120 deg away from
a non-junction lane).
"""

import math

from pufferlib.ocean.cosim.carla.shadow_ego import STATE_X, STATE_Y, STATE_YAW_DEG, ego_speed_from_state


RED_LIGHT_DISTANCE_M = 15.0  # RunningRedLightTest.DISTANCE_LIGHT
STOP_LINE_HALF_WIDTH_FRACTION = 0.4  # of the lane width, RunningRedLightTest
TAIL_CLOSE_FRACTION = 0.8  # tail line from 0.8 * extent to extent + 1 m behind the ego centre
TAIL_FAR_MARGIN_M = 1.0
STOP_SIGN_SPEED_THRESHOLD_MPS = 0.1  # RunningStopTest.SPEED_THRESHOLD
STOP_SIGN_PROXIMITY_M = 50.0  # RunningStopTest.PROXIMITY_THRESHOLD
STOP_SIGN_LOOKAHEAD_STEPS = 20  # RunningStopTest: 20 waypoints, WAYPOINT_STEP 1 m
STOP_SIGN_LOOKAHEAD_STEP_M = 1.0
OUTSIDE_LANE_ALLOWED_DISTANCE_M = 1.3  # OutsideRouteLanesTest.ALLOWED_OUT_DISTANCE
WRONG_LANE_MAX_ANGLE_DEG = 120.0  # OutsideRouteLanesTest.MAX_ALLOWED_VEHICLE_ANGLE
(
    LANE_DRIVING_DISTANCE,
    LANE_DRIVING_WIDTH,
    LANE_PARKING_DISTANCE,
    LANE_PARKING_WIDTH,
    LANE_YAW_DEG,
    LANE_IS_JUNCTION,
) = range(6)


def segments_intersect(a0, a1, b0, b1):
    """Proper or touching intersection of segments a0-a1 and b0-b1 (2-D)."""

    def orientation(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r):
        return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])

    o1, o2 = orientation(a0, a1, b0), orientation(a0, a1, b1)
    o3, o4 = orientation(b0, b1, a0), orientation(b0, b1, a1)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0) and o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0:
        return True
    if o1 == 0 and on_segment(a0, a1, b0):
        return True
    if o2 == 0 and on_segment(a0, a1, b1):
        return True
    if o3 == 0 and on_segment(b0, b1, a0):
        return True
    return o4 == 0 and on_segment(b0, b1, a1)


def point_inside_box(point, center, extent):
    """RunningStopTest.point_inside_boundingbox: axis-aligned test, the trigger volume's rotation ignored."""
    return abs(point[0] - center[0]) < extent[0] and abs(point[1] - center[1]) < extent[1]


class LeaderboardOneGroundTruth:
    def __init__(self, lights, stop_signs, ego_extent_x):
        """lights / stop_signs: the init_route geometry dicts (carla_bridge.light_geometry_from_carla /
        stop_sign_geometry_from_carla layout), ego_extent_x: half length of the ego's bounding box."""
        self.stop_lines_by_light = {}
        self.trigger_by_light = {}
        for light in lights:
            lines = []
            for stop_wp in light["stop_waypoints"]:
                yaw = math.radians(stop_wp["yaw_deg"])
                half_width = STOP_LINE_HALF_WIDTH_FRACTION * stop_wp["lane_width"]
                across = (-math.sin(yaw) * half_width, math.cos(yaw) * half_width)
                left = (stop_wp["x"] + across[0], stop_wp["y"] + across[1])
                right = (stop_wp["x"] - across[0], stop_wp["y"] - across[1])
                lines.append(((math.cos(yaw), math.sin(yaw)), left, right))
            self.stop_lines_by_light[int(light["id"])] = lines
            self.trigger_by_light[int(light["id"])] = (light["trigger_x"], light["trigger_y"])
        self.stop_signs = [(sign["center"][:2], sign["extent"][:2]) for sign in stop_signs]
        self.ego_extent_x = float(ego_extent_x)
        self.last_red_light_id = None
        self.target_stop_sign = None
        self.stop_completed = False
        self.affected_by_stop = False

    def update(self, snapshot, observation):
        """-> {'collision', 'offroad', 'red_light', 'stop_sign'} flags (1.0 fired this tick) from one snapshot."""
        ego = [float(v) for v in snapshot["ego"]]
        light_states = {int(light_id): state_name for light_id, state_name in snapshot["light_states"]}
        return {
            "collision": float(bool(observation.get("collision"))),
            "offroad": float(self._outside_lanes(ego, observation["lane"])),
            "red_light": float(self._ran_red_light(ego, light_states)),
            "stop_sign": float(self._ran_stop_sign(ego, observation["lane"])),
        }

    def _ran_red_light(self, ego, light_states):
        yaw = math.radians(ego[STATE_YAW_DEG])
        forward = (math.cos(yaw), math.sin(yaw))
        x, y = ego[STATE_X], ego[STATE_Y]
        tail_close = (
            x - TAIL_CLOSE_FRACTION * self.ego_extent_x * forward[0],
            y - TAIL_CLOSE_FRACTION * self.ego_extent_x * forward[1],
        )
        tail_far_m = self.ego_extent_x + TAIL_FAR_MARGIN_M
        tail_far = (x - tail_far_m * forward[0], y - tail_far_m * forward[1])
        for light_id, lines in self.stop_lines_by_light.items():
            if light_states.get(light_id) != "Red" or light_id == self.last_red_light_id:
                continue
            trigger_x, trigger_y = self.trigger_by_light[light_id]
            if math.hypot(trigger_x - x, trigger_y - y) > RED_LIGHT_DISTANCE_M:
                continue
            for lane_forward, left, right in lines:
                if forward[0] * lane_forward[0] + forward[1] * lane_forward[1] <= 0.0:
                    continue  # this stop line does not face our lane
                if segments_intersect(tail_close, tail_far, left, right):
                    self.last_red_light_id = light_id
                    return True
        return False

    def _affected_by_stop_sign(self, ego, sign):
        """RunningStopTest.is_actor_affected_by_stop with straight-ahead probes in place of lane waypoints."""
        center, extent = sign
        x, y = ego[STATE_X], ego[STATE_Y]
        if math.hypot(center[0] - x, center[1] - y) > STOP_SIGN_PROXIMITY_M:
            return False
        yaw = math.radians(ego[STATE_YAW_DEG])
        for k in range(STOP_SIGN_LOOKAHEAD_STEPS + 1):
            probe = (
                x + k * STOP_SIGN_LOOKAHEAD_STEP_M * math.cos(yaw),
                y + k * STOP_SIGN_LOOKAHEAD_STEP_M * math.sin(yaw),
            )
            if point_inside_box(probe, center, extent):
                return True
        return False

    def _ran_stop_sign(self, ego, lane):
        if self.target_stop_sign is None:
            lane_yaw = math.radians(lane[LANE_YAW_DEG])
            yaw = math.radians(ego[STATE_YAW_DEG])
            if math.cos(yaw - lane_yaw) <= 0.0:
                return False  # ignore all when going in a wrong lane
            for sign in self.stop_signs:
                if self._affected_by_stop_sign(ego, sign):
                    self.target_stop_sign = sign
                    break
            return False
        if not self.stop_completed and ego_speed_from_state(ego) < STOP_SIGN_SPEED_THRESHOLD_MPS:
            self.stop_completed = True
        center, extent = self.target_stop_sign
        if not self.affected_by_stop and point_inside_box((ego[STATE_X], ego[STATE_Y]), center, extent):
            self.affected_by_stop = True
        if self._affected_by_stop_sign(ego, self.target_stop_sign):
            return False
        ran = self.affected_by_stop and not self.stop_completed
        self.target_stop_sign = None
        self.stop_completed = False
        self.affected_by_stop = False
        return ran

    def _outside_lanes(self, ego, lane):
        driving_distance, parking_distance = lane[LANE_DRIVING_DISTANCE], lane[LANE_PARKING_DISTANCE]
        if parking_distance >= 0.0 and driving_distance >= parking_distance:
            distance, lane_width = parking_distance, lane[LANE_PARKING_WIDTH]
        else:
            distance, lane_width = driving_distance, lane[LANE_DRIVING_WIDTH]
        outside = distance > lane_width / 2.0 + OUTSIDE_LANE_ALLOWED_DISTANCE_M
        if lane[LANE_IS_JUNCTION] > 0.0:
            return outside  # lanes and roads are too chaotic at junctions
        angle = (lane[LANE_YAW_DEG] - ego[STATE_YAW_DEG]) % 360.0
        wrong_lane = WRONG_LANE_MAX_ANGLE_DEG <= angle <= 360.0 - WRONG_LANE_MAX_ANGLE_DEG
        return outside or wrong_lane
