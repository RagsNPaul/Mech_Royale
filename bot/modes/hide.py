"""Mode: HIDE. Cover inside the next circle, then stand and scan."""

from __future__ import annotations

import math

from bot.control import Intent

_ARRIVED_M = 4.0
_COVER_DIST_M = 80.0


def run(brain, obs) -> Intent:
    me = obs["self"]["pos"]
    px, py = float(me[0]), float(me[1])
    threats = brain.threat_points(obs)
    goal = brain.cover_goal(obs, threats, _COVER_DIST_M)
    arrived = goal is None or math.hypot(goal[0] - px, goal[1] - py) <= _ARRIVED_M
    if arrived:
        move, jump = (0.0, 0.0), False
    else:
        move, jump = brain.approach(obs, goal)
    bearing = brain.threat_bearing()
    if bearing is None:
        flip = 1.0 if int(float(obs.get("t") or 0.0) // 3.0) % 2 == 0 else -1.0
        bearing = flip * math.radians(60.0)
    return Intent(
        reason="cover" if arrived else "to-cover",
        move_vec=move,
        jump=jump,
        aim_yaw_pitch=(bearing, 0.0),
        slow=arrived,
        fire=(False, False),
    )
