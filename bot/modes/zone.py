"""Mode: ZONE. Path into the next circle and boost."""

from __future__ import annotations

import math

from bot.control import Intent

_INSIDE_M = 15.0


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def run(brain, obs) -> Intent:
    goal = brain.ring_point(obs, _INSIDE_M)
    move, jump = brain.approach(obs, goal)
    px, py = float(obs["self"]["pos"][0]), float(obs["self"]["pos"][1])
    toward = (goal[0] - px, goal[1] - py)
    if _dot(move, toward) <= 0.0:
        length = math.hypot(toward[0], toward[1])
        move = (0.0, 0.0) if length < 1e-6 else (toward[0] / length, toward[1] / length)
    bearing = brain.threat_bearing()
    if bearing is None:
        bearing = math.atan2(goal[1] - py, goal[0] - px)
    return Intent(
        reason="zone",
        move_vec=move,
        jump=jump,
        aim_yaw_pitch=(bearing, 0.0),
        boost=True,
    )
