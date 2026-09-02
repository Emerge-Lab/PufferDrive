"""Route goal windows for a co-sim ego (Drive goal_source="external").

The shadow env consumes the goals of its window itself, exactly like training
with goal_regen_mode=finite (consumed slots are zeroed in the obs, the window
is replaced only once every goal in it was reached). The co-sim only decides
WHICH `num_goals` route goals form the next window.
"""

import math

import numpy as np

ROUTE_GOAL_COLUMNS = 5  # x, y, z, dir_x, dir_y in the bin frame


class RouteGoalWindow:
    def __init__(self, env, route_goals, agent_idx=0):
        """route_goals: (N, 5) float32 (x, y, z, dir_x, dir_y), bin frame, in
        travel order. dir_* is the local route direction at the goal, used by
        set_agent_goals' route-aligned lane snapping."""
        self.env = env
        self.goals = np.ascontiguousarray(route_goals, dtype=np.float32).reshape(-1, ROUTE_GOAL_COLUMNS)
        if len(self.goals) == 0:
            raise ValueError("RouteGoalWindow needs at least one route goal")
        self.agent_idx = int(agent_idx)
        self.num_goals = int(env.num_goals)
        self.goal_radius = float(env.goal_radius)
        self.window_start = 0
        self.current_index = 0  # route index of the goal the ego is heading for
        self.pushed = False

    @property
    def window(self):
        return self.goals[self.window_start : self.window_start + self.num_goals]

    @property
    def exhausted(self):
        return self.pushed and self.window_start + self.num_goals >= len(self.goals) and self._consumed_all()

    def _consumed_all(self):
        current_goal_idx, goal_count = self.env.get_agent_goal_progress(self.agent_idx)
        return current_goal_idx >= goal_count

    def sync(self, ego_x, ego_y, ego_heading):
        """Push the next window when the current one is exhausted, or when its
        current goal is clearly behind the ego (never pull the ego backwards)."""
        if not self.pushed:
            self._push(0)
            return
        current_goal_idx, goal_count = self.env.get_agent_goal_progress(self.agent_idx)
        if current_goal_idx >= goal_count:
            next_start = self.window_start + goal_count
        else:
            self.current_index = self.window_start + current_goal_idx
            gx, gy = self.goals[self.current_index, :2]
            behind = (gx - ego_x) * math.cos(ego_heading) + (gy - ego_y) * math.sin(ego_heading) < -self.goal_radius
            next_start = self.current_index + 1 if behind else self.window_start
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
