"""Hold a roof inside the next circle. Aim and fire the way the brawler does.

A roof is a building whose roof sits 12–40 m above the ground, with at least
300 m² of footprint, and whose centroid is at least 20 m inside the next
circle. The shortest walk wins. Climbing jets up the nearest wall; three
failed climbs hand the bot back to the brawler for 20 s.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bot.control import Intent
from bot.nav import astar, time_to_reach
from bot.strategies import brawler
from bot.threat import zone_safe_s

_HEIGHT_LO = 12.0
_HEIGHT_HI = 40.0
_MIN_AREA = 300.0
_INSIDE_M = 20.0
_EDGE_IN_M = 4.0
_CLEAR_M = 0.5
_HEAL_ENTER = 0.40
_HEAL_EXIT = 0.85
_EDGE_ENERGY = 60.0
_LEAVE_S = 15.0
_CLIMB_S = 8.0
_STUCK_S = 1.5
_FAILS = 3
_BRAWL_S = 20.0
_ARRIVE_M = 3.0


def choose_roof(pos, zone: dict, candidates: list[dict]) -> dict | None:
    """The shortest-path roof that is tall enough and inside the next circle."""
    if not zone or "next_center" not in zone or "next_radius" not in zone:
        return None
    cx, cy = float(zone["next_center"][0]), float(zone["next_center"][1])
    radius = float(zone["next_radius"])
    px, py = float(pos[0]), float(pos[1])
    best = None
    best_key = None
    for roof in candidates:
        height = float(roof.get("height") or 0.0)
        area = float(roof.get("area") or 0.0)
        if height < _HEIGHT_LO or height > _HEIGHT_HI or area < _MIN_AREA:
            continue
        centroid = roof.get("centroid") or (0.0, 0.0)
        dist = math.hypot(float(centroid[0]) - cx, float(centroid[1]) - cy)
        if dist > radius - _INSIDE_M:
            continue
        length = roof.get("path_length")
        if length is None:
            length = math.hypot(float(centroid[0]) - px, float(centroid[1]) - py)
        key = (float(length), height, dist)
        if best_key is None or key < best_key:
            best, best_key = roof, key
    return best


def climb_command(
    grounded: bool,
    z: float,
    roof_top: float,
    into_wall: tuple[float, float],
) -> tuple[tuple[float, float], bool, bool]:
    """Jump off the ground, then hold the jet with the stick pointed into the wall.

    Returns (move, jump, boost). Once the feet clear the roof, boosting stops.
    """
    move = (float(into_wall[0]), float(into_wall[1]))
    if z > float(roof_top) + _CLEAR_M:
        return move, False, False
    if grounded:
        return move, True, False
    return move, False, True


def run(brain, obs) -> Intent:
    now = float(obs.get("t") or 0.0)
    if now < float(getattr(brain, "_roof_brawler_until", 0.0)):
        return brawler.run(brain, obs)
    me = obs["self"]
    roof = _ensure(brain, obs, me)
    if roof is None:
        return brawler.run(brain, obs)
    if _on_roof(me, roof) and _leaves_soon(brain, obs, roof):
        brain._roof = None
        brain._roof_healing = False
        brain._roof_charged = False
        return _drop(brain, obs, me)
    if _on_roof(me, roof):
        return _hold(brain, obs, me, roof, now)
    brain._roof_charged = False
    return _climb(brain, obs, me, roof, now)


def _ensure(brain, obs, me) -> dict | None:
    roof = getattr(brain, "_roof", None)
    zone = obs.get("zone") or {}
    if roof is not None and choose_roof(me["pos"], zone, [roof]) is not None:
        return roof
    brain._roof = _pick_live(brain, obs, me)
    brain._roof_side = 0
    brain._roof_fails = 0
    brain._roof_charged = False
    brain._climb_t = None
    brain._climb_z = None
    brain._climb_rise_t = None
    return brain._roof


def _hold(brain, obs, me, roof, now: float) -> Intent:
    px, py, _pz = (float(v) for v in me["pos"][:3])
    if not _charged(brain, me):
        return _recharge(brain, obs, me, roof, px, py)
    frac = _hp_frac(me)
    healing = bool(getattr(brain, "_roof_healing", False))
    on_roof_enemy = _enemy_on_roof(brain, me, roof)
    if frac < _HEAL_ENTER:
        healing = True
    if frac >= _HEAL_EXIT or on_roof_enemy is not None:
        healing = False
    brain._roof_healing = healing
    if healing and on_roof_enemy is None:
        return _heal(brain, obs, me, roof, px, py)
    post = _edge_post(roof, obs.get("zone") or {})
    foe = _los_enemy(brain, me)
    move, jump = _go(brain, obs, me, post)
    if math.hypot(post[0] - px, post[1] - py) <= _ARRIVE_M:
        move, jump = (0.0, 0.0), False
    intent = Intent(reason="roof", move_vec=move, jump=jump, slow=move == (0.0, 0.0), boost=False)
    if foe is not None:
        intent.aim_target = foe
        intent.fire = brawler._shots(brain, me, foe, now)
        return intent
    bearing = brain.threat_bearing()
    if bearing is None:
        bearing = math.atan2(py - float(roof["centroid"][1]), px - float(roof["centroid"][0]))
    intent.aim_yaw_pitch = (bearing, 0.0)
    intent.fire = (False, False)
    return intent


def _charged(brain, me: dict) -> bool:
    """The climb spends the pool. Stay in the middle until lightning can fire again."""
    if getattr(brain, "_roof_charged", False):
        return True
    energy = me.get("energy")
    if energy is None or float(energy) >= _EDGE_ENERGY:
        brain._roof_charged = True
        return True
    return False


def _recharge(brain, obs, me, roof, px: float, py: float) -> Intent:
    centroid = (float(roof["centroid"][0]), float(roof["centroid"][1]))
    if math.hypot(centroid[0] - px, centroid[1] - py) <= _ARRIVE_M:
        move, jump = (0.0, 0.0), False
    else:
        move, jump = _go(brain, obs, me, centroid)
    intent = Intent(reason="recharge", move_vec=move, jump=jump, slow=move == (0.0, 0.0), boost=False, fire=(False, False))
    foe = _los_enemy(brain, me)
    if foe is not None:
        intent.aim_target = foe
        intent.fire = (True, False)
        intent.aim_yaw_pitch = None
    return intent


def _heal(brain, obs, me, roof, px: float, py: float) -> Intent:
    centroid = (float(roof["centroid"][0]), float(roof["centroid"][1]))
    if math.hypot(centroid[0] - px, centroid[1] - py) <= _ARRIVE_M:
        return Intent(reason="heal", move_vec=(0.0, 0.0), slow=True, fire=(False, False), aim_yaw_pitch=(0.0, 0.0))
    move, jump = _go(brain, obs, me, centroid)
    return Intent(reason="heal", move_vec=move, jump=jump, fire=(False, False))


def _climb(brain, obs, me, roof, now: float) -> Intent:
    px, py, pz = (float(v) for v in me["pos"][:3])
    wall = _wall(roof, px, py, int(getattr(brain, "_roof_side", 0)))
    into = _unit(float(roof["centroid"][0]) - px, float(roof["centroid"][1]) - py)
    dist = math.hypot(wall[0] - px, wall[1] - py)
    grounded = bool(me.get("grounded", True))
    if pz <= float(roof["top"]) + _CLEAR_M and dist > _ARRIVE_M and grounded:
        move, jump = _go(brain, obs, me, wall)
        return Intent(reason="approach", move_vec=move, jump=jump, fire=(False, False))
    need = (float(roof["height"]) / 60.0) * 100.0 + 15.0
    energy = me.get("energy")
    if energy is not None and float(energy) < need and pz < float(roof["top"]):
        return _wait(brain, obs, me, roof, into)
    _note_climb(brain, now, pz)
    if _climb_failed(brain, now, pz):
        return _fail(brain, obs, now)
    move, jump, boost = climb_command(grounded, pz, float(roof["top"]), into)
    if pz > float(roof["top"]) + _CLEAR_M:
        interior = roof.get("interior") or roof["centroid"]
        move = _unit(float(interior[0]) - px, float(interior[1]) - py)
    return Intent(reason="climb", move_vec=move, jump=jump, boost=boost, fire=(False, False))


def _wait(brain, obs, me, roof, into: tuple[float, float]) -> Intent:
    """Out of energy for the jet. Face away from the wall and shoot back if shot at."""
    foe = _los_enemy(brain, me)
    yaw = math.atan2(-into[1], -into[0])
    intent = Intent(reason="wait", move_vec=(0.0, 0.0), slow=True, aim_yaw_pitch=(yaw, 0.0), fire=(False, False))
    if foe is not None and _attacked(brain, obs):
        intent.aim_target = foe
        intent.fire = brawler._triggers(me)
        intent.aim_yaw_pitch = None
    return intent


def _fail(brain, obs, now: float) -> Intent:
    brain._roof_fails = int(getattr(brain, "_roof_fails", 0)) + 1
    brain._roof_side = int(getattr(brain, "_roof_side", 0)) + 1
    brain._climb_t = None
    brain._climb_z = None
    brain._climb_rise_t = None
    if brain._roof_fails >= _FAILS:
        brain._roof = None
        brain._roof_brawler_until = now + _BRAWL_S
        brain._roof_fails = 0
        return brawler.run(brain, obs)
    return Intent(reason="repick", move_vec=(0.0, 0.0), slow=True, fire=(False, False))


def _drop(brain, obs, me) -> Intent:
    intent = brawler.run(brain, obs)
    zone = obs.get("zone") or {}
    center = zone.get("next_center")
    if not center:
        return intent
    px, py = float(me["pos"][0]), float(me["pos"][1])
    move = _unit(float(center[0]) - px, float(center[1]) - py)
    intent.move_vec = move
    intent.reason = "drop"
    intent.boost = True
    return intent


def _note_climb(brain, now: float, z: float) -> None:
    if getattr(brain, "_climb_t", None) is None:
        brain._climb_t = now
        brain._climb_z = z
        brain._climb_rise_t = now
        return
    if z > float(brain._climb_z) + 0.05:
        brain._climb_z = z
        brain._climb_rise_t = now


def _climb_failed(brain, now: float, z: float) -> bool:
    started = getattr(brain, "_climb_t", None)
    if started is None:
        return False
    if now - float(started) > _CLIMB_S:
        return True
    rose = getattr(brain, "_climb_rise_t", None)
    if rose is None:
        return False
    return now - float(rose) > _STUCK_S and z <= float(getattr(brain, "_climb_z", z)) + 0.05


def _leaves_soon(brain, obs, roof) -> bool:
    centroid = roof["centroid"]
    probe = {
        "pos": [float(centroid[0]), float(centroid[1]), float(roof["top"])],
        "in_zone": True,
    }
    safe = zone_safe_s(probe, obs)
    return not math.isinf(safe) and safe <= _LEAVE_S


def warm(brain) -> None:
    """Build the building list once, before the first decision."""
    _catalog(brain)


def _pick_live(brain, obs, me) -> dict | None:
    zone = obs.get("zone") or {}
    px, py = float(me["pos"][0]), float(me["pos"][1])
    near = []
    for roof in _catalog(brain):
        if choose_roof(me["pos"], zone, [{**roof, "path_length": 0.0}]) is None:
            continue
        cx, cy = float(roof["centroid"][0]), float(roof["centroid"][1])
        near.append((math.hypot(cx - px, cy - py), roof))
    near.sort(key=lambda item: item[0])
    scored = []
    for _dist, roof in near[:3]:
        wall = _wall(roof, px, py, 0)
        scored.append({**roof, "path_length": _path_length(brain, (px, py), wall)})
    return choose_roof(me["pos"], zone, scored)


def _catalog(brain) -> list[dict]:
    cached = getattr(brain, "_roof_catalog", None)
    if cached is not None:
        return cached
    roofs: list[dict] = []
    world = _world()
    grid = brain.map
    if world is not None and grid is not None:
        for prism in world.get("prisms") or []:
            if prism.get("kind") != "building":
                continue
            ring = prism.get("ring") or []
            if len(ring) < 6:
                continue
            cx, cy = _centroid_flat(ring)
            holes = prism.get("holes") or []
            area = _shoelace(ring) - sum(_shoelace(hole) for hole in holes)
            ground = float(grid.ground_z(cx, cy))
            top = float(prism.get("top") or ground)
            walls = _sides(grid, ring, holes, cx, cy)
            roofs.append(
                {
                    "centroid": (cx, cy),
                    "top": top,
                    "ground": ground,
                    "height": top - ground,
                    "area": area,
                    "ring": ring,
                    "walls": walls,
                    "interior": (cx, cy),
                }
            )
    brain._roof_catalog = roofs
    return roofs


_WORLD: dict | None = None
_WORLD_TRIED = False


def _world() -> dict | None:
    global _WORLD, _WORLD_TRIED
    if _WORLD_TRIED:
        return _WORLD
    _WORLD_TRIED = True
    for path in (
        Path("/arena/v2/world.json"),
        Path(__file__).resolve().parents[2] / "arena" / "v2" / "world.json",
    ):
        if path.is_file():
            _WORLD = json.loads(path.read_text(encoding="utf-8"))
            break
    return _WORLD


def _wall(roof: dict, px: float, py: float, side: int = 0) -> tuple[float, float]:
    walls = [(float(p[0]), float(p[1])) for p in (roof.get("walls") or [])]
    if not walls and roof.get("wall"):
        point = roof["wall"]
        walls = [(float(point[0]), float(point[1]))]
    if not walls:
        return (float(roof["centroid"][0]), float(roof["centroid"][1]))
    walls.sort(key=lambda point: math.hypot(point[0] - px, point[1] - py))
    return walls[int(side) % len(walls)]


def _edge_post(roof: dict, zone: dict) -> tuple[float, float]:
    centroid = (float(roof["centroid"][0]), float(roof["centroid"][1]))
    ring = _pairs(roof.get("ring"))
    center = zone.get("next_center") or centroid
    if len(ring) < 2:
        return centroid
    point = _closest_on_ring(ring, (float(center[0]), float(center[1])))
    inward = _unit(centroid[0] - point[0], centroid[1] - point[1])
    return (point[0] + inward[0] * _EDGE_IN_M, point[1] + inward[1] * _EDGE_IN_M)


def _on_roof(me: dict, roof: dict) -> bool:
    x, y, z = (float(v) for v in me["pos"][:3])
    if z < float(roof["top"]) - 1.0:
        return False
    ring = _pairs(roof.get("ring"))
    if len(ring) < 3:
        return math.hypot(x - float(roof["centroid"][0]), y - float(roof["centroid"][1])) < 15.0
    return _pip(x, y, ring)


def _enemy_on_roof(brain, me, roof) -> Any:
    visible = getattr(brain.world, "visible_enemies", None)
    if not callable(visible):
        return None
    ring = _pairs(roof.get("ring"))
    top = float(roof["top"])
    for enemy in visible():
        if abs(float(enemy.pos[2]) - top) > 4.0:
            continue
        if len(ring) < 3 or _pip(float(enemy.pos[0]), float(enemy.pos[1]), ring):
            return enemy
    return None


def _los_enemy(brain, me):
    visible = getattr(brain.world, "visible_enemies", None)
    if not callable(visible):
        return None
    px, py, pz = (float(v) for v in me["pos"][:3])
    eye = float(brain.rules.mech.eye_height)
    best = None
    best_dist = math.inf
    for enemy in visible():
        dist = math.hypot(enemy.pos[0] - px, enemy.pos[1] - py)
        if dist >= best_dist:
            continue
        grid = brain.map
        if grid is not None and hasattr(grid, "los"):
            if not grid.los((px, py, pz + eye), (enemy.pos[0], enemy.pos[1], enemy.pos[2] + eye), True):
                continue
        best, best_dist = enemy, dist
    return best


def _attacked(brain, obs) -> bool:
    if brain.world.last_damage_t is None:
        return False
    return float(obs.get("t") or 0.0) - brain.world.last_damage_t <= 3.0


def _go(brain, obs, me, goal) -> tuple[tuple[float, float], bool]:
    px, py = float(me["pos"][0]), float(me["pos"][1])
    if brain.map is None or brain.nav is None:
        return _unit(float(goal[0]) - px, float(goal[1]) - py), False
    return brain.approach(obs, (float(goal[0]), float(goal[1])))


def _path_length(brain, start, goal) -> float:
    if brain.map is None:
        return math.hypot(goal[0] - start[0], goal[1] - start[1])
    path = astar(brain.map, start, goal)
    if not path:
        return math.hypot(goal[0] - start[0], goal[1] - start[1])
    return time_to_reach(path, 1.0)


def _hp_frac(me: dict) -> float:
    max_hp = float(me.get("max_hp") or 0.0)
    if max_hp <= 0.0:
        return 1.0
    return float(me.get("hp") or 0.0) / max_hp


def _sides(grid, ring, holes, cx: float, cy: float) -> list[tuple[float, float]]:
    found = []
    for dx, dy in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        point = _free_beside(grid, ring, holes, cx, cy, dx, dy)
        if point is not None:
            found.append(point)
    return found


def _free_beside(grid, ring, holes, cx: float, cy: float, dx: float, dy: float):
    distance = 2.0
    while distance < 80.0:
        x, y = cx + dx * distance, cy + dy * distance
        if not grid.in_bounds(x, y):
            return None
        if not _contains(x, y, ring, holes) and not grid.is_blocked(x, y):
            return (x, y)
        distance += 2.0
    return None


def _contains(x: float, y: float, ring, holes) -> bool:
    if not _pip(x, y, _pairs(ring)):
        return False
    return not any(_pip(x, y, _pairs(hole)) for hole in holes or [])


def _centroid_flat(ring) -> tuple[float, float]:
    xs, ys = ring[0::2], ring[1::2]
    area = cx = cy = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        cross = xs[i] * ys[j] - xs[j] * ys[i]
        area += cross
        cx += (xs[i] + xs[j]) * cross
        cy += (ys[i] + ys[j]) * cross
    area *= 0.5
    if abs(area) < 1e-6:
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return cx / (6.0 * area), cy / (6.0 * area)


def _shoelace(ring) -> float:
    xs, ys = ring[0::2], ring[1::2]
    if len(xs) < 3:
        return 0.0
    area = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) * 0.5


def _pairs(ring) -> list[tuple[float, float]]:
    if not ring:
        return []
    if isinstance(ring[0], (list, tuple)):
        return [(float(p[0]), float(p[1])) for p in ring]
    return [(float(ring[i]), float(ring[i + 1])) for i in range(0, len(ring) - 1, 2)]


def _pip(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _closest_on_ring(ring: list[tuple[float, float]], point: tuple[float, float]) -> tuple[float, float]:
    best = ring[0]
    best_dist = math.inf
    for i, start in enumerate(ring):
        end = ring[(i + 1) % len(ring)]
        hit = _closest_segment(start, end, point)
        dist = math.hypot(hit[0] - point[0], hit[1] - point[1])
        if dist < best_dist:
            best, best_dist = hit, dist
    return best


def _closest_segment(a, b, p) -> tuple[float, float]:
    abx, aby = b[0] - a[0], b[1] - a[1]
    length = abx * abx + aby * aby
    if length < 1e-12:
        return a
    t = max(0.0, min(1.0, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / length))
    return (a[0] + abx * t, a[1] + aby * t)


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, 0.0
    return dx / length, dy / length
