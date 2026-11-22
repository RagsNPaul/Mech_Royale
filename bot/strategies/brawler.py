"""Rush the nearest enemy. No hiding and no retreat.

Pathfinding is used only when the straight line is blocked. The first 30 s are
for the nearest weapon inside 60 m. After that, and when no weapon is waiting,
the nearest known mech is the target.
"""

from __future__ import annotations

import math

from bot.control import Intent
from bot.threat import zone_safe_s

_RANK = {"lightning": 0, "laser": 1, "mortar": 2}
_LOOT_S = 30.0
_LOOT_M = 60.0
_ZONE_S = 8.0
_LIGHTNING_M = 25.0
_GUN_M = 45.0
_RING_M = 15.0


def run(brain, obs) -> Intent:
    me = obs["self"]
    now = float(obs.get("t") or 0.0)
    target = _nearest(brain, me)
    if _zone_soon(me, obs):
        return _zone(brain, obs, me, target)
    if now < _LOOT_S:
        pickup = _weapon(obs, me)
        if pickup is not None:
            return _take(brain, obs, me, pickup, target)
    if target is not None:
        return _rush(brain, obs, me, target, now)
    return _search(brain, obs, me)


def _zone_soon(me: dict, obs: dict) -> bool:
    """Outside the circle, or the circle reaches us within 8 s."""
    safe = zone_safe_s(me, obs)
    return not math.isinf(safe) and safe <= _ZONE_S


def _rush(brain, obs, me, target, now: float) -> Intent:
    px, py = _xy(me["pos"])
    dist = math.hypot(target.pos[0] - px, target.pos[1] - py)
    if dist > _standoff(me):
        move, jump = _toward(brain, obs, me, target.pos)
    else:
        move, jump = _strafe(px, py, target.pos[0], target.pos[1], now), False
    return Intent(reason="rush", move_vec=move, jump=jump, aim_target=target, fire=(True, True))


def _take(brain, obs, me, pickup, target) -> Intent:
    px, py = _xy(me["pos"])
    pos = pickup["pos"]
    dist = math.hypot(float(pos[0]) - px, float(pos[1]) - py)
    intent = Intent(reason="weapon", fire=(True, True) if target is not None else (False, False))
    if target is not None:
        intent.aim_target = target
    else:
        intent.aim_yaw_pitch = (math.atan2(float(pos[1]) - py, float(pos[0]) - px), 0.0)
    if dist <= float(brain.rules.loot.pickup_radius):
        intent.take = int(pickup["id"])
        intent.move_vec = (0.0, 0.0)
        intent.slow = True
        return intent
    move, jump = _toward(brain, obs, me, pos)
    intent.move_vec = move
    intent.jump = jump
    return intent


def _zone(brain, obs, me, target) -> Intent:
    goal = brain.ring_point(obs, _RING_M)
    move, jump = _toward(brain, obs, me, goal)
    intent = Intent(reason="zone", move_vec=move, jump=jump, boost=me.get("in_zone") is False)
    if target is not None:
        intent.aim_target = target
        intent.fire = (True, True)
        return intent
    px, py = _xy(me["pos"])
    intent.aim_yaw_pitch = (math.atan2(goal[1] - py, goal[0] - px), 0.0)
    intent.fire = (False, False)
    return intent


def _search(brain, obs, me) -> Intent:
    bearing = brain.threat_bearing()
    if bearing is not None:
        return Intent(
            reason="bearing",
            move_vec=(math.cos(bearing), math.sin(bearing)),
            aim_yaw_pitch=(bearing, 0.0),
            fire=(False, False),
        )
    zone = obs.get("zone") or {}
    center = zone.get("next_center")
    if not center:
        return Intent(slow=True, reason="idle", fire=(False, False))
    px, py = _xy(me["pos"])
    goal = (float(center[0]), float(center[1]))
    move, jump = _toward(brain, obs, me, goal)
    return Intent(
        reason="center",
        move_vec=move,
        jump=jump,
        aim_yaw_pitch=(math.atan2(goal[1] - py, goal[0] - px), 0.0),
        fire=(False, False),
    )


def _toward(brain, obs, me, pos) -> tuple[tuple[float, float], bool]:
    """Straight at `pos`, or a path when a building blocks that line."""
    px, py = _xy(me["pos"])
    tx, ty = float(pos[0]), float(pos[1])
    if _blocked(brain, me, pos):
        return brain.approach(obs, (tx, ty))
    return _unit(tx - px, ty - py), False


def _blocked(brain, me, pos) -> bool:
    grid = brain.map
    if grid is None or not hasattr(grid, "los"):
        return False
    ax, ay, az = (float(v) for v in me["pos"][:3])
    eye = float(brain.rules.mech.eye_height)
    bz = float(pos[2]) if len(pos) > 2 else az
    return not grid.los((ax, ay, az + eye), (float(pos[0]), float(pos[1]), bz + eye), False)


def _strafe(px: float, py: float, tx: float, ty: float, now: float) -> tuple[float, float]:
    bearing = math.atan2(ty - py, tx - px)
    side = 1.0 if math.floor(now) % 2 == 0 else -1.0
    yaw = bearing + side * (math.pi / 2.0)
    return math.cos(yaw), math.sin(yaw)


def _standoff(me: dict) -> float:
    return _LIGHTNING_M if "lightning" in _kinds(me) else _GUN_M


def _weapon(obs: dict, me: dict) -> dict | None:
    if not _has_empty_slot(me):
        return None
    held = _kinds(me)
    px, py = _xy(me["pos"])
    best = None
    best_key = None
    for pickup in obs.get("pickups") or []:
        kind = str(pickup.get("kind") or "")
        if kind not in _RANK or kind in held:
            continue
        pos = pickup.get("pos")
        if not pos:
            continue
        dist = math.hypot(float(pos[0]) - px, float(pos[1]) - py)
        if dist > _LOOT_M:
            continue
        key = (_RANK[kind], dist, int(pickup.get("id") or 0))
        if best_key is None or key < best_key:
            best, best_key = pickup, key
    return best


def _nearest(brain, me: dict):
    enemies = getattr(brain.world, "enemies", None)
    if not callable(enemies):
        return None
    px, py = _xy(me["pos"])
    best = None
    best_key = None
    for enemy in enemies():
        dist = math.hypot(enemy.pos[0] - px, enemy.pos[1] - py)
        hp = math.inf if enemy.hp is None else float(enemy.hp)
        key = (dist, hp, enemy.id)
        if best_key is None or key < best_key:
            best, best_key = enemy, key
    return best


def _kinds(me: dict) -> set[str]:
    found = set()
    for weapon in me.get("weapons") or []:
        if isinstance(weapon, dict) and weapon.get("kind"):
            found.add(str(weapon["kind"]))
        elif isinstance(weapon, str):
            found.add(weapon)
    return found


def _has_empty_slot(me: dict) -> bool:
    weapons = list(me.get("weapons") or [])
    if len(weapons) < 2:
        return True
    return any(slot is None for slot in weapons[:2])


def _xy(pos) -> tuple[float, float]:
    return float(pos[0]), float(pos[1])


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, 0.0
    return dx / length, dy / length
