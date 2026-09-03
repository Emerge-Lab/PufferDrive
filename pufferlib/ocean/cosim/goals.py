"""Route goal windows for a co-sim ego (Drive goal_source="external").

The shadow env consumes the goals of its window itself, exactly like training
with goal_regen_mode=finite (consumed slots are zeroed in the obs, the window
is replaced only once every goal in it was reached). The co-sim only decides
WHICH `num_goals` route goals form the next window. A sliding window instead
refills to `num_goals` goals ahead after every consumed goal, so the ego sees a
window-final (speed-gated) goal only at the true route end.
"""

import math

import numpy as np

ROUTE_GOAL_COLUMNS = 5  # x, y, z, dir_x, dir_y in the bin frame
ROUTE_MAX_LATERAL_M = 8.0  # farther from the goal polyline the ego is off-route: never skip goals then


class RouteGoalWindow:
    def __init__(self, env, route_goals, agent_idx=0, sliding=False):
        """route_goals: (N, 5) float32 (x, y, z, dir_x, dir_y), bin frame, in
        travel order. dir_* is the local route direction at the goal, used by
        set_agent_goals' route-aligned lane snapping. sliding: advance the
        window past every consumed goal instead of replacing it once exhausted."""
        self.env = env
        self.goals = np.ascontiguousarray(route_goals, dtype=np.float32).reshape(-1, ROUTE_GOAL_COLUMNS)
        if len(self.goals) == 0:
            raise ValueError("RouteGoalWindow needs at least one route goal")
        self.agent_idx = int(agent_idx)
        self.sliding = bool(sliding)
        self.num_goals = int(env.num_goals)
        self.goal_radius = float(env.goal_radius)
        segment_lengths = np.hypot(*np.diff(self.goals[:, :2], axis=0).T)
        self.arc_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])  # route progress of each goal
        self.window_start = 0
        self.current_index = 0  # route index of the goal the ego is heading for
        self.pushed = False

    @property
    def window(self):
        return self.goals[self.window_start : self.window_start + self.num_goals]

    @property
    def consumed_count(self):
        """Goals of the current window the shadow env has already consumed (zeroed in the obs)."""
        if not self.pushed:
            return 0
        current_goal_idx, goal_count = self.env.get_agent_goal_progress(self.agent_idx)
        return min(current_goal_idx, goal_count)

    @property
    def exhausted(self):
        return self.pushed and self.window_start + self.num_goals >= len(self.goals) and self._consumed_all()

    def _consumed_all(self):
        current_goal_idx, goal_count = self.env.get_agent_goal_progress(self.agent_idx)
        return current_goal_idx >= goal_count

    def route_progress(self, ego_x, ego_y, first_goal=0, last_goal=None):
        """(arc length of the ego projected onto the goal polyline between two goal indices, lateral
        distance to it). Restricting the goal range keeps a route that loops back near the ego from
        projecting onto its far end."""
        pts = self.goals[:, :2].astype(np.float64)
        last_goal = len(pts) - 1 if last_goal is None else min(last_goal, len(pts) - 1)
        if last_goal <= first_goal:
            return float(self.arc_length[first_goal]), float(np.hypot(pts[first_goal, 0] - ego_x, pts[first_goal, 1] - ego_y))
        a, b = pts[first_goal:last_goal], pts[first_goal + 1 : last_goal + 1]
        d = b - a
        length_sq = np.maximum((d * d).sum(axis=1), 1e-9)
        t = np.clip(((ego_x - a[:, 0]) * d[:, 0] + (ego_y - a[:, 1]) * d[:, 1]) / length_sq, 0.0, 1.0)
        proj = a + t[:, None] * d
        dist = np.hypot(proj[:, 0] - ego_x, proj[:, 1] - ego_y)
        k = int(np.argmin(dist))
        return float(self.arc_length[first_goal + k] + t[k] * math.sqrt(length_sq[k])), float(dist[k])

    def sync(self, ego_x, ego_y, ego_heading):
        """Push the next window when the current one is exhausted (sliding: when any goal was
        consumed), or when the ego has driven past its current goal along the route (e.g. in the
        adjacent lane) without consuming it. An ego far off the route keeps its window: skipping
        goals there cascades through the whole route."""
        if not self.pushed:
            self._push(0)
            return
        current_goal_idx, goal_count = self.env.get_agent_goal_progress(self.agent_idx)
        if current_goal_idx >= goal_count:
            next_start = self.window_start + goal_count
        else:
            self.current_index = self.window_start + current_goal_idx
            progress, lateral = self.route_progress(
                ego_x, ego_y, max(self.current_index - 1, 0), self.current_index + self.num_goals
            )
            passed = lateral <= ROUTE_MAX_LATERAL_M and progress > self.arc_length[self.current_index] + self.goal_radius
            next_start = int(np.searchsorted(self.arc_length, progress, side="right")) if passed else self.window_start
            if self.sliding and not passed:
                next_start = self.current_index
        if next_start == self.window_start or next_start >= len(self.goals):
            return  # window still live, or route exhausted: the last window stays saturated
        self._push(next_start)

    def _push(self, start):
        sel = self.goals[start : start + self.num_goals]
        self.env.set_agent_goals(
            self.agent_idx, sel[:, 0].copy(), sel[:, 1].copy(), sel[:, 2].copy(), sel[:, 3].copy(), sel[:, 4].copy()
        )
        self.window_start = start
        self.current_index = start
        self.pushed = True


def route_goals_from_xy(goals_xy, goals_z=None):
    """(N, 2) goal points -> (N, 5) route goals; dir = previous goal -> goal."""
    xy = np.asarray(goals_xy, dtype=np.float64).reshape(-1, 2)
    z = np.zeros(len(xy)) if goals_z is None else np.asarray(goals_z, dtype=np.float64).reshape(-1)
    prev = np.vstack([xy[:1], xy[:-1]])
    return np.column_stack([xy, z, xy - prev]).astype(np.float32)
