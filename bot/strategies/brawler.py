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

_ZONE_S = 8.0
_BOOST_ZONE_S = 10.0
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


def run(brain, obs) -> Intent:
    me = obs["self"]
    now = float(obs.get("t") or 0.0)
    target = _chain_target(brain)
    if _zone_soon(me, obs):
        return _zone(brain, obs, me, target)
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


def _zone_soon(me: dict, obs: dict) -> bool:
    """Outside the circle, or the circle reaches us within 8 s."""
    safe = zone_safe_s(me, obs)
    return not math.isinf(safe) and safe <= _ZONE_S


def _rush(brain, obs, me, target, now: float) -> Intent:
    px, py = _xy(me["pos"])
    tx, ty = float(target.pos[0]), float(target.pos[1])
    dist = math.hypot(tx - px, ty - py)
    hold = _standoff(me)
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


def _zone(brain, obs, me, target) -> Intent:
    now = float(obs.get("t") or 0.0)
    goal = brain.ring_point(obs, _RING_M)
    move, jump = _toward(brain, obs, me, goal)
    intent = Intent(reason="zone", move_vec=move, jump=jump, boost=_boost(brain, obs, me))
    if target is not None:
        intent.aim_target = target
        intent.fire = _shots(brain, me, target, now)
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
            boost=_boost(brain, obs, me),
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


def _standoff(me: dict) -> float:
    return _LIGHTNING_M if "lightning" in _kinds(me) else _GUN_M


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
