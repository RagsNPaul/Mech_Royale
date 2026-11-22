"""Rush a clustered enemy. Lightning in slot 1 comes before a distant fight.

Pathfinding is used only when the straight line is blocked. Until slot 1 holds
lightning, the nearest lightning pickup inside 150 m is the goal. Crates inside
40 m are shot on the way, and an enemy is fought only when it is inside 40 m.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from bot.control import Intent
from bot.threat import zone_safe_s

_ZONE_S = 15.0
_BOOST_ZONE_S = 10.0
_NPC_CLEAR_M = 80.0
_NPC_AVOID_M = 50.0
_STUCK_S = 3.0
_STUCK_M = 1.0
_STUCK_STRAFE_S = 1.0
_ALONG_COS = 0.5
_LIGHTNING_M = 20.0
_GUN_M = 45.0
_RING_M = 15.0
_HUNT_M = 150.0
_CRATE_M = 40.0
_FIGHT_M = 60.0
_FIGHT_ON_WAY_M = 40.0
_CHAIN_M = 14.0
_HOLD_M = 2.0
_LIGHTNING_ENERGY = 8.0
_RESERVE = 9.0
_REGEN_ENERGY = 15.0
_REGEN_DIST = 25.0
_REGEN_HOLD_S = 1.0
_CUBE_CLEAR_M = 120.0
_CUBE_FAR_M = 80.0
_CUBE_FIGHT_M = 12.0
_DROP_VISIBLE_M = 80.0
_DROP_S = 10.0
_HP_GRAB = 0.50


def run(brain, obs) -> Intent:
    me = obs["self"]
    now = float(obs.get("t") or 0.0)
    return _finish(brain, obs, me, _choose(brain, obs, me, now), now)


def _choose(brain, obs, me, now: float) -> Intent:
    target = _chain_target(brain)
    if _zone_soon(me, obs):
        return _zone(brain, obs, me, target, now)
    _note_drops(brain, obs, now)
    if _nearest_known(brain, me, _NPC_CLEAR_M) is None:
        npc = _npc_to_kill(brain, obs, me)
        if npc is not None:
            return _kill_npc(brain, obs, me, npc, now)
    bearing = brain.threat_bearing()
    only_bearing = (
        target is None and _nearest_known(brain, me, math.inf) is None and bearing is not None
    )
    if only_bearing:
        return _along_bearing(brain, obs, me, bearing, now)
    cube = _cube_move(brain, obs, me, now)
    if cube is not None:
        return cube
    drop = _drop_move(brain, obs, me, now)
    if drop is not None:
        return drop
    if not _slot1_lightning(me):
        close = _within(brain, me, _FIGHT_ON_WAY_M)
        if close is not None:
            return _rush(brain, obs, me, close, now)
        pickup = _lightning(obs, me)
        if pickup is not None:
            return _hunt(brain, obs, me, pickup)
    if target is not None:
        return _rush(brain, obs, me, target, now)
    return _search(brain, obs, me)


def _knob(brain, key: str, default: float) -> float:
    params = getattr(brain, "params", None)
    if not isinstance(params, dict) or key not in params:
        return default
    try:
        return float(params[key])
    except (TypeError, ValueError):
        return default


def _hp_frac(me: dict) -> float:
    max_hp = float(me.get("max_hp") or 0.0)
    if max_hp <= 0.0:
        return 1.0
    return float(me.get("hp") or 0.0) / max_hp


def _cube_move(brain, obs, me, now: float) -> Intent | None:
    """Walk up to cube_detour_m when the area is clear, or when hp is under heal_hp_frac."""
    px, py = _xy(me["pos"])
    far = _knob(brain, "cube_detour_m", _CUBE_FAR_M)
    healing = _hp_frac(me) < _knob(brain, "heal_hp_frac", 0.0)
    if healing or _nearest_known(brain, me, _CUBE_CLEAR_M) is None:
        cube = _nearest_cube(obs, me, far)
        if cube is not None:
            return _walk_cube(brain, obs, me, cube["pos"], now)
        if not healing:
            return None
    cube = _nearest_cube(obs, me, _CUBE_FIGHT_M)
    if cube is None or not _winning(me):
        return None
    pos = cube["pos"]
    move, jump = _toward(brain, obs, me, pos)
    if math.hypot(float(pos[0]) - px, float(pos[1]) - py) <= float(brain.rules.cubes.pickup_radius):
        move, jump = (0.0, 0.0), False
    intent = Intent(
        reason="cube",
        move_vec=move,
        jump=jump,
        slow=move == (0.0, 0.0),
        boost=_boost(brain, obs, me),
    )
    return _face(intent, brain, me, now)


def _walk_cube(brain, obs, me, pos, now: float) -> Intent:
    px, py = _xy(me["pos"])
    tx, ty = float(pos[0]), float(pos[1])
    dist = math.hypot(tx - px, ty - py)
    if dist <= float(brain.rules.cubes.pickup_radius):
        intent = Intent(reason="cube", move_vec=(0.0, 0.0), slow=True)
    else:
        move, jump = _toward(brain, obs, me, pos)
        intent = Intent(reason="cube", move_vec=move, jump=jump, boost=_boost(brain, obs, me))
    return _face(intent, brain, me, now)


def _drop_move(brain, obs, me, now: float) -> Intent | None:
    """A kill's cubes land on the body. Skip the walk when someone is in sight inside 80 m."""
    if _visible_within(brain, me, _DROP_VISIBLE_M) is not None:
        return None
    drops = getattr(brain, "_cube_drops", ())
    if not drops:
        return None
    px, py = _xy(me["pos"])
    pos = min(drops, key=lambda item: math.hypot(item[1][0] - px, item[1][1] - py))[1]
    weapon = _dropped_weapon(obs, me, pos)
    dist = math.hypot(pos[0] - px, pos[1] - py)
    intent = Intent(reason="drop", boost=_boost(brain, obs, me))
    if weapon is not None and dist <= float(brain.rules.loot.pickup_radius):
        intent.take = int(weapon["id"])
        intent.drop = _swap_slot(me)
        intent.move_vec = (0.0, 0.0)
        intent.slow = True
        return _face(intent, brain, me, now)
    if dist <= float(brain.rules.cubes.pickup_radius):
        intent.move_vec = (0.0, 0.0)
        intent.slow = True
        return _face(intent, brain, me, now)
    move, jump = _toward(brain, obs, me, pos)
    intent.move_vec = move
    intent.jump = jump
    return _face(intent, brain, me, now)


def _face(intent: Intent, brain, me, now: float) -> Intent:
    """Keep the guns on the nearest known enemy while the legs go for the cube."""
    foe = _nearest_known(brain, me, math.inf)
    if foe is None:
        intent.fire = (False, False)
        return intent
    intent.aim_target = foe
    intent.fire = _shots(brain, me, foe, now)
    return intent


def _note_drops(brain, obs, now: float) -> None:
    pending = [item for item in getattr(brain, "_cube_drops", ()) if now <= item[0]]
    stores = []
    for name in ("mechs", "npcs"):
        store = getattr(brain.world, name, None)
        if isinstance(store, dict):
            stores.append(store)
    for event in obs.get("events") or []:
        if event.get("type") != "kill":
            continue
        target = event.get("target")
        if target is None:
            continue
        track = None
        for store in stores:
            track = store.get(int(target))
            if track is not None:
                break
        if track is None:
            continue
        pending.append((now + _DROP_S, (float(track.pos[0]), float(track.pos[1]), float(track.pos[2]))))
    brain._cube_drops = pending


def _nearest_cube(obs, me, radius: float) -> dict | None:
    px, py = _xy(me["pos"])
    best = None
    best_dist = radius
    for cube in obs.get("cubes") or []:
        pos = cube.get("pos") if isinstance(cube, dict) else None
        if not pos:
            continue
        dist = math.hypot(float(pos[0]) - px, float(pos[1]) - py)
        if dist <= best_dist:
            best, best_dist = cube, dist
    return best


def _dropped_weapon(obs, me, pos) -> dict | None:
    if _slot1_lightning(me):
        return None
    best = None
    best_key = None
    for pickup in obs.get("pickups") or []:
        kind = str(pickup.get("kind") or "")
        if kind not in ("lightning", "laser"):
            continue
        spot = pickup.get("pos")
        if not spot:
            continue
        if math.hypot(float(spot[0]) - pos[0], float(spot[1]) - pos[1]) > 15.0:
            continue
        dist = math.hypot(float(spot[0]) - _xy(me["pos"])[0], float(spot[1]) - _xy(me["pos"])[1])
        key = (0 if kind == "lightning" else 1, dist)
        if best_key is None or key < best_key:
            best, best_key = pickup, key
    return best


def _winning(me: dict) -> bool:
    max_hp = float(me.get("max_hp") or 0.0)
    if max_hp <= 0.0:
        return True
    return float(me.get("hp") or 0.0) / max_hp > _HP_GRAB


def _zone_soon(me: dict, obs: dict) -> bool:
    """Outside the circle, or the circle reaches us within 15 s."""
    safe = zone_safe_s(me, obs)
    return not math.isinf(safe) and safe <= _ZONE_S


def _rush(brain, obs, me, target, now: float) -> Intent:
    px, py = _xy(me["pos"])
    tx, ty = float(target.pos[0]), float(target.pos[1])
    dist = math.hypot(tx - px, ty - py)
    hold = _standoff(brain, me)
    if dist > hold + _HOLD_M:
        move, jump = _toward(brain, obs, me, target.pos)
    elif dist < hold - _HOLD_M:
        move, jump = _unit(px - tx, py - ty), False
    else:
        move, jump = _strafe(px, py, tx, ty, now), False
    return Intent(
        reason="rush",
        move_vec=move,
        jump=jump,
        aim_target=target,
        fire=_shots(brain, me, target, now),
        boost=_boost(brain, obs, me),
    )


def _hunt(brain, obs, me, pickup) -> Intent:
    """Walk to a lightning pickup. Shoot a crate that is already within 40 m."""
    px, py = _xy(me["pos"])
    pos = pickup["pos"]
    dist = math.hypot(float(pos[0]) - px, float(pos[1]) - py)
    crate = _crate(obs, me)
    intent = Intent(reason="lightning", fire=(False, False))
    if crate is not None:
        cpos = crate["pos"]
        cz = float(cpos[2]) if len(cpos) > 2 else float(me["pos"][2])
        intent.aim_target = SimpleNamespace(pos=(float(cpos[0]), float(cpos[1]), cz), vel=(0.0, 0.0, 0.0), id=None)
        intent.fire = (True, False)
    else:
        intent.aim_yaw_pitch = (math.atan2(float(pos[1]) - py, float(pos[0]) - px), 0.0)
    if dist <= float(brain.rules.loot.pickup_radius):
        intent.take = int(pickup["id"])
        intent.drop = _swap_slot(me)
        intent.move_vec = (0.0, 0.0)
        intent.slow = True
        return intent
    move, jump = _toward(brain, obs, me, pos)
    intent.move_vec = move
    intent.jump = jump
    intent.boost = _boost(brain, obs, me)
    return intent


def _zone(brain, obs, me, target, now: float) -> Intent:
    goal = brain.ring_point(obs, _RING_M)
    move, jump = _toward(brain, obs, me, goal)
    intent = Intent(reason="zone", move_vec=move, jump=jump, boost=_boost(brain, obs, me))
    foe = target if target is not None else _nearest_known(brain, me, math.inf)
    if foe is not None:
        intent.aim_target = foe
        intent.fire = _shots(brain, me, foe, now)
        return intent
    px, py = _xy(me["pos"])
    intent.aim_yaw_pitch = (math.atan2(goal[1] - py, goal[0] - px), 0.0)
    intent.fire = (False, False)
    return intent


def _npc_tracks(brain) -> list:
    store = getattr(brain.world, "npcs", None)
    if isinstance(store, dict):
        return list(store.values())
    return []


def _npc_shooter(obs) -> int | None:
    for event in obs.get("events") or []:
        if event.get("type") == "damaged" and event.get("weapon") == "npc" and event.get("by") is not None:
            return int(event["by"])
    return None


def _npc_shooting(obs) -> bool:
    for event in obs.get("events") or []:
        if event.get("type") == "damaged" and event.get("weapon") == "npc":
            return True
    return False


def _los_pos(brain, me, pos) -> bool:
    grid = brain.map
    if grid is None or not hasattr(grid, "los"):
        return True
    ax, ay, az = (float(v) for v in me["pos"][:3])
    eye = float(brain.rules.mech.eye_height)
    bz = float(pos[2]) if len(pos) > 2 else az
    return bool(grid.los((ax, ay, az + eye), (float(pos[0]), float(pos[1]), bz + eye), True))


def _npc_to_kill(brain, obs, me):
    """A drone that is shooting us, or one inside aggro range with line of sight."""
    aggro = float(brain.rules.npc.aggro_range)
    leash = float(brain.rules.npc.chase_leash)
    px, py = _xy(me["pos"])
    shooter = _npc_shooter(obs)
    if shooter is not None:
        for npc in _npc_tracks(brain):
            if int(npc.id) != shooter:
                continue
            if math.hypot(float(npc.pos[0]) - px, float(npc.pos[1]) - py) <= leash:
                return npc
    if not _npc_shooting(obs) and shooter is None:
        limit = aggro
        need_los = True
    else:
        limit = leash
        need_los = False
    best = None
    best_dist = limit
    for npc in _npc_tracks(brain):
        if npc.last_seen_t != brain.world.t:
            continue
        dist = math.hypot(float(npc.pos[0]) - px, float(npc.pos[1]) - py)
        if dist > best_dist:
            continue
        if need_los and not _los_pos(brain, me, npc.pos):
            continue
        best, best_dist = npc, dist
    return best


def _kill_npc(brain, obs, me, npc, now: float) -> Intent:
    """Lightning and the gun. A drone has 350 hp; its two cubes are picked up after the kill."""
    px, py = _xy(me["pos"])
    tx, ty, tz = (float(v) for v in npc.pos)
    dist = math.hypot(tx - px, ty - py)
    target = SimpleNamespace(pos=(tx, ty, tz), vel=(0.0, 0.0, 0.0), id=npc.id)
    intent = Intent(reason="npc", aim_target=target, fire=(True, True), boost=False)
    hold = _standoff(brain, me)
    if dist > hold + _HOLD_M:
        move, jump = _toward(brain, obs, me, npc.pos)
    else:
        move, jump = _strafe(px, py, tx, ty, now), False
    intent.move_vec = move
    intent.jump = jump
    return intent


def _in_front(px: float, py: float, bearing: float, pos, reach: float) -> bool:
    dx = float(pos[0]) - px
    dy = float(pos[1]) - py
    dist = math.hypot(dx, dy)
    if dist > reach or dist < 1e-6:
        return False
    return dx / dist * math.cos(bearing) + dy / dist * math.sin(bearing) >= _ALONG_COS


def _along_bearing(brain, obs, me, bearing: float, now: float) -> Intent:
    """Follow the sound, but scoop a cube, crate, or drone that sits on that ray."""
    del now
    px, py = _xy(me["pos"])
    far = _knob(brain, "cube_detour_m", _CUBE_FAR_M)
    cube = None
    cube_dist = far
    for item in obs.get("cubes") or []:
        pos = item.get("pos") if isinstance(item, dict) else None
        if not pos or not _in_front(px, py, bearing, pos, far):
            continue
        dist = math.hypot(float(pos[0]) - px, float(pos[1]) - py)
        if dist <= cube_dist:
            cube, cube_dist = item, dist
    if cube is not None:
        intent = _walk_cube(brain, obs, me, cube["pos"], float(obs.get("t") or 0.0))
        intent.reason = "bearing"
        return intent
    crate = None
    crate_dist = _CRATE_M
    for item in obs.get("crates") or []:
        pos = item.get("pos") if isinstance(item, dict) else None
        if not pos or not _in_front(px, py, bearing, pos, _CRATE_M):
            continue
        dist = math.hypot(float(pos[0]) - px, float(pos[1]) - py)
        if dist <= crate_dist:
            crate, crate_dist = item, dist
    if crate is not None:
        pos = crate["pos"]
        cz = float(pos[2]) if len(pos) > 2 else float(me["pos"][2])
        intent = Intent(
            reason="bearing",
            aim_target=SimpleNamespace(pos=(float(pos[0]), float(pos[1]), cz), vel=(0.0, 0.0, 0.0), id=None),
            fire=(True, False),
            boost=_boost(brain, obs, me),
        )
        move, jump = _toward(brain, obs, me, pos)
        intent.move_vec = move
        intent.jump = jump
        return intent
    return Intent(
        reason="bearing",
        move_vec=(math.cos(bearing), math.sin(bearing)),
        aim_yaw_pitch=(bearing, 0.0),
        fire=(False, False),
        boost=_boost(brain, obs, me),
    )


def _segment_clearance(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> tuple[float, float]:
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-9:
        return math.hypot(px - ax, py - ay), 0.0
    tee = ((px - ax) * abx + (py - ay) * aby) / ab2
    tee = max(0.0, min(1.0, tee))
    return math.hypot(px - (ax + tee * abx), py - (ay + tee * aby)), tee


def _push_off_npcs(brain, me, intent: Intent) -> Intent:
    """While a mech is close, don't walk through a drone's 50 m bubble."""
    tracks = _npc_tracks(brain)
    if not tracks or intent.move_vec is None:
        return intent
    px, py = _xy(me["pos"])
    steered = (float(intent.move_vec[0]), float(intent.move_vec[1]))
    changed = False
    for npc in tracks:
        nx, ny = float(npc.pos[0]), float(npc.pos[1])
        dist = math.hypot(nx - px, ny - py)
        if dist < _NPC_AVOID_M:
            steered = _unit(px - nx, py - ny)
            changed = True
            break
        reach = 40.0
        clear, tee = _segment_clearance(px, py, px + steered[0] * reach, py + steered[1] * reach, nx, ny)
        if clear < _NPC_AVOID_M and tee > 0.02:
            tx, ty = _unit(-(ny - py), nx - px)
            if steered[0] * tx + steered[1] * ty < 0.0:
                tx, ty = -tx, -ty
            away = _unit(px - nx, py - ny)
            steered = _unit(tx + 0.5 * away[0], ty + 0.5 * away[1])
            changed = True
    if changed:
        intent.move_vec = steered
        intent.slow = False
    return intent


def _replan(brain) -> None:
    nav = getattr(brain, "nav", None)
    if nav is None:
        return
    nav.path = []
    nav.goal = None
    nav.plan_time = -1e9


def _apply_stuck(brain, me, intent: Intent, now: float) -> Intent:
    """Under 1 m of progress in 3 s of commanded motion: jump and strafe for 1 s, then replan."""
    px, py = _xy(me["pos"])
    until = float(getattr(brain, "_unstuck_until", 0.0) or 0.0)
    if until and now < until:
        yaw = float(getattr(brain, "_unstuck_heading", 0.0)) + math.pi / 2.0
        intent.move_vec = (math.cos(yaw), math.sin(yaw))
        intent.jump = True
        intent.slow = False
        return intent
    if until and now >= until and not getattr(brain, "_stuck_replanned", True):
        _replan(brain)
        brain._stuck_replanned = True
        brain._stuck_anchor = (now, (px, py))
    move = intent.move_vec or (0.0, 0.0)
    commanding = not intent.slow and math.hypot(float(move[0]), float(move[1])) > 0.2
    if not commanding:
        brain._stuck_anchor = (now, (px, py))
        return intent
    anchor = getattr(brain, "_stuck_anchor", None)
    if anchor is None:
        brain._stuck_anchor = (now, (px, py))
        return intent
    anchor_t, anchor_xy = anchor
    if now - float(anchor_t) < _STUCK_S:
        return intent
    moved = math.hypot(px - float(anchor_xy[0]), py - float(anchor_xy[1]))
    brain._stuck_anchor = (now, (px, py))
    if moved >= _STUCK_M:
        return intent
    brain._unstuck_heading = math.atan2(float(move[1]), float(move[0]))
    brain._unstuck_until = now + _STUCK_STRAFE_S
    brain._stuck_replanned = False
    yaw = brain._unstuck_heading + math.pi / 2.0
    intent.move_vec = (math.cos(yaw), math.sin(yaw))
    intent.jump = True
    intent.slow = False
    return intent


def _finish(brain, obs, me, intent: Intent, now: float) -> Intent:
    del obs
    if intent.reason != "npc" and _nearest_known(brain, me, _NPC_CLEAR_M) is not None:
        intent = _push_off_npcs(brain, me, intent)
    return _apply_stuck(brain, me, intent, now)


def _search(brain, obs, me) -> Intent:
    bearing = brain.threat_bearing()
    if bearing is not None:
        return _along_bearing(brain, obs, me, bearing, float(obs.get("t") or 0.0))
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
        boost=_boost(brain, obs, me),
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


def _standoff(brain, me: dict) -> float:
    if "lightning" in _kinds(me):
        return _knob(brain, "standoff_lightning_m", _LIGHTNING_M)
    return _knob(brain, "standoff_gun_m", _GUN_M)


def _lightning(obs: dict, me: dict) -> dict | None:
    px, py = _xy(me["pos"])
    best = None
    best_dist = _HUNT_M
    for pickup in obs.get("pickups") or []:
        if str(pickup.get("kind") or "") != "lightning":
            continue
        pos = pickup.get("pos")
        if not pos:
            continue
        dist = math.hypot(float(pos[0]) - px, float(pos[1]) - py)
        if dist <= best_dist:
            best, best_dist = pickup, dist
    return best


def _crate(obs: dict, me: dict) -> dict | None:
    px, py = _xy(me["pos"])
    best = None
    best_dist = _CRATE_M
    for crate in obs.get("crates") or []:
        pos = crate.get("pos") if isinstance(crate, dict) else None
        if not pos:
            continue
        dist = math.hypot(float(pos[0]) - px, float(pos[1]) - py)
        if dist <= best_dist:
            best, best_dist = crate, dist
    return best


def _chain_target(brain):
    """Visible enemy with the most neighbors inside 14 m. Ties go to the lowest HP."""
    visible = _visible(brain)
    best = None
    best_key = None
    for enemy in visible:
        others = 0
        for other in visible:
            if other.id == enemy.id:
                continue
            if math.hypot(other.pos[0] - enemy.pos[0], other.pos[1] - enemy.pos[1]) <= _CHAIN_M:
                others += 1
        hp = math.inf if enemy.hp is None else float(enemy.hp)
        key = (-others, hp, enemy.id)
        if best_key is None or key < best_key:
            best, best_key = enemy, key
    return best


def _within(brain, me: dict, radius: float):
    """The chain target when it is close, otherwise the nearest known enemy inside `radius`."""
    px, py = _xy(me["pos"])
    chosen = _chain_target(brain)
    if chosen is not None and math.hypot(chosen.pos[0] - px, chosen.pos[1] - py) <= radius:
        return chosen
    enemies = getattr(brain.world, "enemies", None)
    if not callable(enemies):
        return None
    best = None
    best_dist = radius
    for enemy in enemies():
        dist = math.hypot(enemy.pos[0] - px, enemy.pos[1] - py)
        if dist <= best_dist:
            best, best_dist = enemy, dist
    return best


def _nearest_known(brain, me: dict, radius: float):
    """Nearest visible or remembered enemy inside `radius`."""
    px, py = _xy(me["pos"])
    enemies = getattr(brain.world, "enemies", None)
    if not callable(enemies):
        return None
    best = None
    best_dist = radius
    for enemy in enemies():
        dist = math.hypot(enemy.pos[0] - px, enemy.pos[1] - py)
        if dist <= best_dist:
            best, best_dist = enemy, dist
    return best


def _visible_within(brain, me: dict, radius: float):
    px, py = _xy(me["pos"])
    best = None
    best_dist = radius
    for enemy in _visible(brain):
        dist = math.hypot(enemy.pos[0] - px, enemy.pos[1] - py)
        if dist <= best_dist:
            best, best_dist = enemy, dist
    return best


def _visible(brain) -> list:
    visible = getattr(brain.world, "visible_enemies", None)
    if not callable(visible):
        return []
    return list(visible())


def _slot1_lightning(me: dict) -> bool:
    weapons = list(me.get("weapons") or [])
    if len(weapons) < 2 or weapons[1] is None:
        return False
    weapon = weapons[1]
    kind = weapon.get("kind") if isinstance(weapon, dict) else weapon
    return kind == "lightning"


def _swap_slot(me: dict) -> int | None:
    """Drop slot 1 when a mortar or laser is occupying the lightning slot."""
    weapons = list(me.get("weapons") or [])
    if len(weapons) < 2 or weapons[1] is None:
        return None
    weapon = weapons[1]
    kind = weapon.get("kind") if isinstance(weapon, dict) else weapon
    if kind in ("mortar", "laser"):
        return 1
    return None


def _triggers(me: dict) -> tuple[bool, bool]:
    energy = me.get("energy")
    lightning = energy is None or float(energy) >= _LIGHTNING_ENERGY
    return True, lightning and "lightning" in _kinds(me)


def _shots(brain, me: dict, target, now: float) -> tuple[bool, bool]:
    """Gun stays up. Lightning only inside range and the cone, above the reserve."""
    return True, _lightning_ready(brain, me, target, now)


def _lightning_ready(brain, me: dict, target, now: float) -> bool:
    if target is None or "lightning" not in _kinds(me):
        return False
    px, py = _xy(me["pos"])
    tx, ty = float(target.pos[0]), float(target.pos[1])
    dist = math.hypot(tx - px, ty - py)
    weapon = brain.rules.weapons["lightning"]
    if dist > float(weapon.range):
        return False
    cone = math.radians(float(weapon.cone_deg))
    bearing = math.atan2(ty - py, tx - px)
    aim = me.get("aim") or [0.0, 0.0]
    if abs(math.atan2(math.sin(bearing - float(aim[0])), math.cos(bearing - float(aim[0])))) > cone:
        return False
    energy = me.get("energy")
    cost = float(weapon.energy)
    if energy is not None and float(energy) - cost < _RESERVE:
        return False
    if _regen_pause(brain, energy, dist, now):
        return False
    return True


def _regen_pause(brain, energy, dist: float, now: float) -> bool:
    """Hold lightning long enough for regen to start, then let it fire again."""
    delay = float(brain.rules.mech.energy_regen_delay)
    hold = _REGEN_HOLD_S if _REGEN_HOLD_S >= delay else delay
    until = getattr(brain, "_brawl_lightning_until", None)
    if until is not None and now < float(until):
        return True
    if until is not None:
        brain._brawl_lightning_until = None
        brain._brawl_lightning_resumed = True
    low_far = energy is not None and float(energy) < _REGEN_ENERGY and dist > _REGEN_DIST
    if not low_far:
        brain._brawl_lightning_resumed = False
        return False
    if getattr(brain, "_brawl_lightning_resumed", False):
        return False
    brain._brawl_lightning_until = now + hold
    return True


def _boost(brain, obs, me: dict) -> bool:
    """No boost inside 60 m. Outside a fight, boost only to stay in the circle."""
    if _within(brain, me, _FIGHT_M) is not None:
        return False
    safe = zone_safe_s(me, obs)
    return not math.isinf(safe) and safe <= _BOOST_ZONE_S


def _kinds(me: dict) -> set[str]:
    found = set()
    for weapon in me.get("weapons") or []:
        if isinstance(weapon, dict) and weapon.get("kind"):
            found.add(str(weapon["kind"]))
        elif isinstance(weapon, str):
            found.add(weapon)
    return found


def _xy(pos) -> tuple[float, float]:
    return float(pos[0]), float(pos[1])


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, 0.0
    return dx / length, dy / length
