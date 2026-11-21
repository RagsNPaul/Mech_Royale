"""Navigation: path planning and cover selection on the precomputed grid.

TODO(cursor):
- astar(map, start_xy, goal_xy) -> list of waypoints (2 m grid, 8-connected, avoid slopes > max_slope).
- class Navigator: holds current path; replan at most every PARAMS['replan_every_s'] or when goal moves;
  skip replanning if elapsed > PARAMS['plan_budget_ms']. steer(pos) -> unit move vector.
- best_cover(map, me, threats, zone) -> (x, y): nearest cover point inside the NEXT zone circle
  that breaks LOS to the threat bearings, reachable before we'd die (use their DPS / our HP).
- time_to_reach(path, speed) helper used by the zone logic.
"""


class Navigator:
    def __init__(self, mapdata, rules):
        self.map = mapdata
        self.rules = rules
        self.path = []

    def steer(self, pos, goal, now, t0):
        raise NotImplementedError
