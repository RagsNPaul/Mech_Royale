"""Mode: LOOT. Take a weapon, break nearby crates, and farm drones."""

from __future__ import annotations

import math
from types import SimpleNamespace

from bot.control import Intent
from bot.params import PARAMS
from bot.threat import zone_safe_s

_RANK = {"lightning": 0, "laser": 1, "mortar": 2}
_PHASE_S = 45.0
_CRATE_M = 60.0
_CLOSE_M = 20.0
_STAND_M = 25.0
_CUBE_M = 40.0
_CLEAR_M = 60.0
_FARM_CLEAR_M = 100.0
_NPC_LO = 40.0
_NPC_HI = 60.0


def run(brain, obs) -> Intent | None:
    me = obs["self"]
    goal = choose(obs, me, brain.world, float(obs.get("t") or 0.0))
    if goal is None:
        return None
    if goal["what"] == "weapon":
        return _take(brain, obs, me, goal["pickup"])
    if goal["what"] == "cube":
        return _cube(brain, obs, me, goal["pos"])
    if goal["what"] == "npc":
        return _npc(brain, obs, me, goal["pos"])
    return _shoot(brain, obs, me, goal["pos"], goal["what"])


def choose(obs: dict, me: dict, world, now: float) -> dict | None:
    """The best loot objective, or None when this tick should not loot."""
    px, py = float(me["pos"][0]), float(me["pos"][1])
    if not _hostile(world, _CLEAR_M):
        cube = _nearest(obs.get("cubes") or [], px, py, _CUBE_M)
        if cube is not None:
            return {"what": "cube", "pos": cube["pos"]}
    weapon = _weapon(obs, me, now)
    if weapon is not None:
        return {"what": "weapon", "pickup": weapon}
    if now < _PHASE_S:
        crate = _nearest(obs.get("crates") or [], px, py, _CRATE_M)
        if crate is not None:
            return {"what": "crate", "pos": crate["pos"]}
    if not _farm_ok(me, obs, world):
        return None
    limit = float(PARAMS["npc_farm_max_dist_m"])
    cube = _nearest(obs.get("cubes") or [], px, py, limit)
    if cube is not None:
        return {"what": "cube", "pos": cube["pos"]}
    npc = _nearest_npc(world, px, py, limit)
    if npc is not None:
        return {"what": "npc", "pos": npc.pos}
    return None


def _take(brain, obs, me, pickup) -> Intent:
    px, py = float(me["pos"][0]), float(me["pos"][1])
    pos = pickup["pos"]
    dist = math.hypot(float(pos[0]) - px, float(pos[1]) - py)
    radius = float(brain.rules.loot.pickup_radius)
    yaw = math.atan2(float(pos[1]) - py, float(pos[0]) - px)
    intent = Intent(reason="weapon", aim_yaw_pitch=(yaw, 0.0), fire=(False, False))
    if dist <= radius:
        intent.take = int(pickup["id"])
        intent.drop = _swap_slot(me, str(pickup.get("kind") or ""))
        intent.move_vec = (0.0, 0.0)
        intent.slow = True
        return intent
    move, jump = brain.approach(obs, (float(pos[0]), float(pos[1])))
    intent.move_vec = move
    intent.jump = jump
    return intent


def _cube(brain, obs, me, pos) -> Intent:
    px, py = float(me["pos"][0]), float(me["pos"][1])
    tx, ty = float(pos[0]), float(pos[1])
    dist = math.hypot(tx - px, ty - py)
    yaw = math.atan2(ty - py, tx - px)
    radius = float(brain.rules.cubes.pickup_radius)
    if dist <= radius:
        return Intent(reason="cube", move_vec=(0.0, 0.0), slow=True, aim_yaw_pitch=(yaw, 0.0), fire=(False, False))
    move, jump = brain.approach(obs, (tx, ty))
    return Intent(reason="cube", move_vec=move, jump=jump, aim_yaw_pitch=(yaw, 0.0), fire=(False, False))


def _npc(brain, obs, me, pos) -> Intent:
    """Hold gun range. A drone's cubes are walked once they are on the ground."""
    px, py = float(me["pos"][0]), float(me["pos"][1])
    tx, ty = float(pos[0]), float(pos[1])
    tz = float(pos[2]) if len(pos) > 2 else float(me["pos"][2])
    dist = math.hypot(tx - px, ty - py)
    target = SimpleNamespace(pos=(tx, ty, tz), vel=(0.0, 0.0, 0.0), id=None)
    intent = Intent(reason="npc", aim_target=target, fire=(True, True))
    if dist > _NPC_HI:
        move, jump = brain.approach(obs, (tx, ty))
        intent.move_vec = move
        intent.jump = jump
        return intent
    side = brain.control.strafe((px, py, 0.0), (tx, ty, 0.0), float(obs.get("t") or 0.0))
    if dist < _NPC_LO:
        away = _unit(px - tx, py - ty)
        blended = _unit(away[0] + side[0], away[1] + side[1])
        intent.move_vec = blended if blended != (0.0, 0.0) else away
    else:
        intent.move_vec = side
        intent.slow = True
        intent.strafe = True
    return intent


def _shoot(brain, obs, me, pos, reason: str) -> Intent:
    px, py = float(me["pos"][0]), float(me["pos"][1])
    tx, ty = float(pos[0]), float(pos[1])
    tz = float(pos[2]) if len(pos) > 2 else float(me["pos"][2])
    dist = math.hypot(tx - px, ty - py)
    target = SimpleNamespace(pos=(tx, ty, tz), vel=(0.0, 0.0, 0.0), id=None)
    if dist > _STAND_M:
        move, jump = brain.approach(obs, (tx, ty))
        slow = False
    else:
        move, jump, slow = (0.0, 0.0), False, True
    return Intent(
        reason=reason,
        move_vec=move,
        jump=jump,
        slow=slow,
        aim_target=target,
        fire=(True, True),
    )


def _weapon(obs: dict, me: dict, now: float) -> dict | None:
    held = set(_kinds(me))
    free = _has_empty_slot(me)
    phase = now < _PHASE_S
    px, py = float(me["pos"][0]), float(me["pos"][1])
    best = None
    best_key = None
    for pickup in obs.get("pickups") or []:
        kind = str(pickup.get("kind") or "")
        if kind not in _RANK or kind in held:
            continue
        pos = pickup.get("pos") or (0.0, 0.0)
        dist = math.hypot(float(pos[0]) - px, float(pos[1]) - py)
        swap = _swap_slot(me, kind) is not None and dist <= _CLOSE_M
        if free and (phase or dist <= _CLOSE_M):
            pass
        elif swap:
            pass
        else:
            continue
        key = (_RANK[kind], dist, int(pickup.get("id") or 0))
        if best_key is None or key < best_key:
            best, best_key = pickup, key
    return best


def _swap_slot(me: dict, kind: str) -> int | None:
    """Slot to drop when a lightning or laser should replace a mortar."""
    if kind not in ("lightning", "laser") or _has_empty_slot(me):
        return None
    for index, weapon in enumerate(me.get("weapons") or []):
        held = weapon.get("kind") if isinstance(weapon, dict) else weapon
        if held == "mortar":
            return index
    return None


def _has_empty_slot(me: dict) -> bool:
    weapons = list(me.get("weapons") or [])
    if len(weapons) < 2:
        return True
    return any(slot is None for slot in weapons[:2])


def _kinds(me: dict) -> list[str]:
    found = []
    for weapon in me.get("weapons") or []:
        if isinstance(weapon, dict) and weapon.get("kind"):
            found.append(str(weapon["kind"]))
        elif isinstance(weapon, str):
            found.append(weapon)
    return found


def _nearest(items: list, px: float, py: float, limit: float) -> dict | None:
    best = None
    best_dist = limit
    for item in items:
        pos = item.get("pos") if isinstance(item, dict) else None
        if not pos:
            continue
        dist = math.hypot(float(pos[0]) - px, float(pos[1]) - py)
        if dist <= best_dist:
            best, best_dist = item, dist
    return best


def _farm_ok(me: dict, obs: dict, world) -> bool:
    """No known mech inside 100 m, and the zone will not reach us inside the safety window."""
    if _hostile(world, _FARM_CLEAR_M):
        return False
    safe = zone_safe_s(me, obs)
    return math.isinf(safe) or safe >= float(PARAMS["zone_safe_s"])


def _hostile(world, radius: float) -> bool:
    enemies = getattr(world, "enemies", None)
    if not callable(enemies):
        return False
    return len(enemies(radius)) > 0


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, 0.0
    return dx / length, dy / length


def _nearest_npc(world, px: float, py: float, limit: float):
    best = None
    best_dist = limit
    store = getattr(world, "npcs", None)
    if store is None:
        return None
    tracks = store() if callable(store) else store.values()
    for npc in tracks:
        if getattr(npc, "last_seen_t", None) != getattr(world, "t", None):
            continue
        dist = math.hypot(float(npc.pos[0]) - px, float(npc.pos[1]) - py)
        if dist <= best_dist:
            best, best_dist = npc, dist
    return best
