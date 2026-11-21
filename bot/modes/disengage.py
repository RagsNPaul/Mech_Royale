"""Mode: DISENGAGE. Break away while facing the nearest attacker."""

from __future__ import annotations

import math

from bot.control import Intent

_COVER_DIST_M = 100.0


def run(brain, obs) -> Intent:
    me = obs["self"]
    px, py = float(me["pos"][0]), float(me["pos"][1])
    squad = not brain.world.ffa
    enemies = brain.world.visible_enemies()
    nearest = None
    if enemies:
        nearest = min(enemies, key=lambda enemy: _flat(px, py, enemy.pos[0], enemy.pos[1]))
    away = _away_vector(px, py, enemies, brain.threat_bearing())
    aim_at = nearest
    if squad and aim_at is None:
        known = brain.world.enemies()
        if known:
            aim_at = min(known, key=lambda enemy: _flat(px, py, enemy.pos[0], enemy.pos[1]))
    threats = [(enemy.pos[0], enemy.pos[1]) for enemy in enemies]
    if not threats:
        threats = brain.threat_points(obs)
    cover = brain.cover_goal(obs, threats, _COVER_DIST_M)
    goal = None
    if cover is not None:
        toward = _unit(cover[0] - px, cover[1] - py)
        if _dot(toward, away) > 0.0:
            goal = cover
    brain._peek = goal if goal is not None else None
    if goal is None:
        goal = (px + away[0] * 40.0, py + away[1] * 40.0)
    move, jump = brain.approach(obs, goal)
    if _dot(move, away) <= 0.0:
        move = away
    if squad and brain._visible_los(obs) and math.hypot(move[0], move[1]) < 0.15:
        move = away
    max_hp = float(me.get("max_hp") or 0.0)
    frac = float(me.get("hp") or 0.0) / max_hp if max_hp else 1.0
    bearing = brain.threat_bearing()
    spotted = aim_at is not None or bearing is not None
    intent = Intent(
        reason="retreat",
        move_vec=move,
        jump=jump,
        boost=frac < 0.40,
        fire=(True, True) if squad and spotted else ((True, False) if nearest is not None else (False, False)),
    )
    if aim_at is not None:
        intent.aim_target = aim_at
    else:
        intent.aim_yaw_pitch = (0.0 if bearing is None else bearing, 0.0)
    return intent


def _flat(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(bx - ax, by - ay)


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, 0.0
    return dx / length, dy / length


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _away_vector(px: float, py: float, enemies: list, bearing: float | None) -> tuple[float, float]:
    dx = dy = 0.0
    for enemy in enemies:
        ox, oy = _unit(px - enemy.pos[0], py - enemy.pos[1])
        dx += ox
        dy += oy
    if dx or dy:
        return _unit(dx, dy)
    if bearing is not None:
        return -math.cos(bearing), -math.sin(bearing)
    return 1.0, 0.0
