"""Fight-or-hide math. Pure functions, checked by tests/test_threat.py.

my_dps(target)   = sum over my usable weapons at distance d of
                   damage / cooldown * p_hit(kind, d) * (1 + damage_per_cube * min(cubes, stat_cap))
their_dps(enemy) = same from their weapons list, * triangle modifier,
                   * rear_mult if they are outside MY front arc (they hit my rear)
my_ttk    = target_hp / my_dps(target)
their_ttk = my_hp / sum(their_dps for enemies within threat_radius)

margin (v1) = base_margin. The v2 terms (reward, finish, desperation, crowd) return 0.
FIGHT if my_ttk < margin * their_ttk and hp_frac > min_hp_frac and zone safe >= zone_safe_s.
A target at or under finish_hp is finished when we can shoot it and survive the next 2 s.
No line of sight never fights.

Weapon triangle: laser > mortar, mortar > lightning, lightning > laser, gun neutral.
Winning matchup is 1.25, losing is 0.8.
"""

from __future__ import annotations

import math
from typing import Any

_BEATS = {("laser", "mortar"), ("mortar", "lightning"), ("lightning", "laser")}


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _xyz(obj: Any) -> tuple[float, float, float]:
    pos = _field(obj, "pos") or (0.0, 0.0, 0.0)
    z = float(pos[2]) if len(pos) > 2 else 0.0
    return (float(pos[0]), float(pos[1]), z)


def _dist(a: Any, b: Any) -> float:
    ax, ay, az = _xyz(a)
    bx, by, bz = _xyz(b)
    return math.dist((ax, ay, az), (bx, by, bz))


def _hp(obj: Any) -> float:
    value = _field(obj, "hp")
    return 0.0 if value is None else float(value)


def _max_hp(obj: Any) -> float:
    value = _field(obj, "max_hp")
    return _hp(obj) if value is None else float(value)


def _cubes(obj: Any) -> float:
    value = _field(obj, "cubes")
    return 0.0 if value is None else max(0.0, float(value))


def _kinds(obj: Any) -> list[str]:
    found = []
    for item in _field(obj, "weapons") or []:
        if item is None:
            continue
        if isinstance(item, str):
            found.append(item)
        else:
            kind = item.get("kind") if isinstance(item, dict) else getattr(item, "kind", None)
            if kind:
                found.append(str(kind))
    return found


def _weapon_range(kind: str, stats: Any) -> float:
    ranged = getattr(stats, "range", None)
    if ranged is not None:
        return float(ranged)
    if kind == "mortar":
        speed = float(getattr(stats, "speed", 0.0))
        gravity = float(getattr(stats, "gravity", 0.0))
        if gravity > 0.0:
            return speed * speed / gravity
    return math.inf


def _period(kind: str, stats: Any) -> float:
    cooldown = float(stats.cooldown)
    if kind == "gun":
        mag = float(stats.mag)
        return cooldown + (float(stats.reload) / mag if mag else 0.0)
    if kind == "laser":
        return float(stats.charge) + cooldown
    return cooldown


def p_hit(kind: str, dist: float, rules: Any, params: dict[str, Any]) -> float:
    """Hit chance at dist. Gun falls off linearly; everything else is flat inside its range."""
    stats = rules.weapons.get(kind)
    if stats is None:
        return 0.0
    dist = max(0.0, float(dist))
    if dist > _weapon_range(kind, stats):
        return 0.0
    if kind == "gun":
        far = float(params["gun_far_m"])
        near_p = float(params["p_hit_gun_near"])
        far_p = float(params["p_hit_gun_far"])
        if far <= 0.0:
            return near_p
        blend = min(1.0, dist / far)
        return near_p + (far_p - near_p) * blend
    if kind == "lightning":
        return float(params["p_hit_lightning"])
    if kind == "laser":
        return float(params["p_hit_laser"])
    if kind == "mortar":
        return float(params["p_hit_mortar"])
    return 0.0


def weapon_dps(kind: str, dist: float, cubes: int, rules: Any, params: dict[str, Any]) -> float:
    """Sustained damage per second. Gun reload is spread across the magazine; laser adds charge time."""
    stats = rules.weapons.get(kind)
    if stats is None:
        return 0.0
    hit = p_hit(kind, dist, rules, params)
    if hit <= 0.0:
        return 0.0
    period = _period(kind, stats)
    if period <= 0.0:
        return 0.0
    cap = float(rules.cubes.stat_cap)
    per = float(rules.cubes.damage_per_cube)
    scale = 1.0 + per * min(float(cubes), cap)
    return (float(stats.damage) / period) * hit * scale


def triangle(my_kind: str, their_kind: str) -> float:
    """1.25 when my weapon beats theirs, 0.8 when theirs beats mine, 1 when either side is neutral."""
    if (my_kind, their_kind) in _BEATS:
        return 1.25
    if (their_kind, my_kind) in _BEATS:
        return 0.8
    return 1.0


def _my_kind(me: Any, dist: float, rules: Any, params: dict[str, Any]) -> str | None:
    best_kind = None
    best = -1.0
    cubes = _cubes(me)
    for kind in _kinds(me):
        dealt = weapon_dps(kind, dist, cubes, rules, params)
        if dealt > best:
            best = dealt
            best_kind = kind
    return best_kind


def _outside_front_arc(me: Any, enemy: Any, rules: Any) -> bool:
    mx, my, _mz = _xyz(me)
    ex, ey, _ez = _xyz(enemy)
    bearing = math.atan2(ey - my, ex - mx)
    aim = _field(me, "aim") or (0.0, 0.0)
    yaw = float(aim[0])
    delta = abs(math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw)))
    half = math.radians(float(rules.mech.front_arc_deg)) / 2.0
    return delta > half


def my_dps(me: Any, target: Any, rules: Any, params: dict[str, Any]) -> float:
    dist = _dist(me, target)
    cubes = _cubes(me)
    return sum(weapon_dps(kind, dist, cubes, rules, params) for kind in _kinds(me))


def incoming_dps(me: Any, enemy: Any, rules: Any, params: dict[str, Any]) -> float:
    """Damage per second this enemy deals to me, after the triangle and rear armor."""
    dist = _dist(me, enemy)
    mine = _my_kind(me, dist, rules, params)
    total = 0.0
    cubes = _cubes(enemy)
    for kind in _kinds(enemy):
        dealt = weapon_dps(kind, dist, cubes, rules, params)
        if mine is not None:
            dealt *= triangle(mine, kind)
        total += dealt
    if total > 0.0 and _outside_front_arc(me, enemy, rules):
        total *= float(rules.mech.rear_mult)
    return total


def _nearby(me: Any, target: Any, world: Any, radius: float) -> list[Any]:
    pool: list[Any] = []
    if world is not None and hasattr(world, "enemies"):
        pool.extend(world.enemies(radius))
    if target is not None:
        pool.append(target)
    seen: set[Any] = set()
    chosen = []
    for actor in pool:
        ident = _field(actor, "id")
        key = id(actor) if ident is None else ident
        if key in seen or _dist(me, actor) > radius:
            continue
        seen.add(key)
        chosen.append(actor)
    return chosen


def _threat_dps(me: Any, target: Any, world: Any, rules: Any, params: dict[str, Any]) -> float:
    radius = float(params["threat_radius_m"])
    return sum(incoming_dps(me, enemy, rules, params) for enemy in _nearby(me, target, world, radius))


def _reward_bonus(me: Any, target: Any, rules: Any, params: dict[str, Any]) -> float:
    """TODO(v2): reward_per_cube * cubes the kill drops, when under the stat cap."""
    del me, target, rules, params
    return 0.0


def _finish_bonus(me: Any, target: Any, rules: Any, params: dict[str, Any]) -> float:
    """TODO(v2): bonus when the target dies inside finish_burst_s."""
    del me, target, rules, params
    return 0.0


def _desperation(obs: Any, world: Any, params: dict[str, Any]) -> float:
    """TODO(v2): bonus in the last zone phases."""
    del obs, world, params
    return 0.0


def _crowd_penalty(me: Any, target: Any, world: Any, params: dict[str, Any]) -> float:
    """TODO(v2): penalty per extra hostile inside threat_radius."""
    del me, target, world, params
    return 0.0


def risk_margin(me: Any, target: Any, world: Any, obs: Any, rules: Any, params: dict[str, Any]) -> float:
    """v1 margin is the base. v2 terms are added here and currently return 0."""
    return (
        float(params["base_margin"])
        + _reward_bonus(me, target, rules, params)
        + _finish_bonus(me, target, rules, params)
        + _desperation(obs, world, params)
        - _crowd_penalty(me, target, world, params)
    )


def _has_los(me: Any, target: Any, world: Any, rules: Any) -> bool:
    grid = None if world is None else getattr(world, "map", None)
    if grid is None or not hasattr(grid, "los"):
        return True
    eye = float(rules.mech.eye_height)
    ax, ay, az = _xyz(me)
    bx, by, bz = _xyz(target)
    return bool(grid.los((ax, ay, az + eye), (bx, by, bz + eye), for_vision=True))


def zone_safe_s(me: Any, obs: Any) -> float:
    """Seconds the current and next circles still cover me. 0 when already outside."""
    if _field(me, "in_zone") is False:
        return 0.0
    zone = None if obs is None else _field(obs, "zone")
    if not isinstance(zone, dict):
        return math.inf
    pos = _xyz(me)
    now = float(_field(obs, "t") or 0.0)
    center = (float(zone["center"][0]), float(zone["center"][1]))
    radius = float(zone["radius"])
    nxt = (float(zone["next_center"][0]), float(zone["next_center"][1]))
    next_radius = float(zone["next_radius"])
    shrink_start = float(zone["shrink_start"])
    shrink_end = float(zone["shrink_end"])

    def inside(point: tuple[float, float], ring: float) -> bool:
        return math.hypot(pos[0] - point[0], pos[1] - point[1]) <= ring

    if not inside(center, radius):
        return 0.0
    if inside(nxt, next_radius) or now >= shrink_end:
        return math.inf if inside(nxt, next_radius) else 0.0

    duration = shrink_end - shrink_start
    if duration <= 0.0:
        return 0.0

    def outside_at(fraction: float) -> bool:
        point = (
            center[0] + fraction * (nxt[0] - center[0]),
            center[1] + fraction * (nxt[1] - center[1]),
        )
        ring = radius + fraction * (next_radius - radius)
        return not inside(point, ring)

    if not outside_at(1.0):
        return math.inf
    start = 0.0 if now <= shrink_start else min(1.0, (now - shrink_start) / duration)
    lo, hi = start, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2.0
        if outside_at(mid):
            hi = mid
        else:
            lo = mid
    leaves = shrink_start + hi * duration
    return max(0.0, leaves - now)


def should_fight(
    me: Any, target: Any, world: Any, obs: Any, rules: Any, params: dict[str, Any]
) -> tuple[bool, str]:
    """Return (fight?, reason). Reason goes into the debug label."""
    if target is None:
        return False, "no_target"
    if not _has_los(me, target, world, rules):
        return False, "no_los"
    if _hp(target) <= float(params["finish_hp"]) and my_dps(me, target, rules, params) > 0.0:
        burst = _threat_dps(me, target, world, rules, params) * 2.0
        if _hp(me) >= burst:
            return True, "finish"
    max_hp = _max_hp(me)
    if max_hp <= 0.0 or _hp(me) / max_hp <= float(params["min_hp_frac_to_fight"]):
        return False, "low_hp"
    if zone_safe_s(me, obs) < float(params["zone_safe_s"]):
        return False, "zone"
    dealt = my_dps(me, target, rules, params)
    incoming = _threat_dps(me, target, world, rules, params)
    my_ttk = _hp(target) / dealt if dealt > 0.0 else math.inf
    their_ttk = _hp(me) / incoming if incoming > 0.0 else math.inf
    if my_ttk < risk_margin(me, target, world, obs, rules, params) * their_ttk:
        return True, "favorable"
    return False, "losing"


def pick_target(me: Any, world: Any, rules: Any, params: dict[str, Any]) -> Any:
    """Visible enemy with line of sight and the lowest time-to-kill. Ties go to lower HP."""
    if world is None or not hasattr(world, "visible_enemies"):
        return None
    best = None
    best_key: tuple[float, float, Any] | None = None
    for enemy in world.visible_enemies():
        if not _has_los(me, enemy, world, rules):
            continue
        dealt = my_dps(me, enemy, rules, params)
        hp = _hp(enemy)
        ttk = hp / dealt if dealt > 0.0 else math.inf
        key = (ttk, hp, _field(enemy, "id"))
        if best_key is None or key < best_key:
            best_key = key
            best = enemy
    return best
