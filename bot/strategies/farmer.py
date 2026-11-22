"""Farm gear and drones, then hold a covered spot inside the next circle.

Until 100 s or 8 cubes, mechs are ignored unless one is inside 60 m or shooting
us. After that the bot stands on cover that is 25 m inside the next circle and
shielded from at least 4 of 8 directions, and it does not chase off that spot.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from bot.control import Intent
from bot.nav import best_cover
from bot.strategies import brawler

_FARM_S = 100.0
_FARM_CUBES = 8.0
_WEAPON_M = 150.0
_NPC_M = 150.0
_CRATE_M = 80.0
_HOLD_EDGE_M = 25.0
_HOLD_DIRS = 4
_HOLD_REACH_M = 200.0
_LEASH_M = 25.0
_SPOT_M = 3.0
_COLLECT_M = 30.0
_HEAL_ENTER = 0.50
_HEAL_EXIT = 0.85
_QUIET_S = 2.0
_HEAL_REACH_M = 40.0
_SCAN_S = 1.2
_SCAN_RAD = math.radians(150.0)
_LEAD_NEAR_M = 60.0
_LEAD_M = 100.0
_CELL_M = 100.0
_VISIT_M = 40.0
_SWAP_M = 60.0
_FORGET_M = 25.0
_WEAPON_RANK = {"lightning": 0, "laser": 1, "mortar": 2}


def run(brain, obs) -> Intent:
    me = obs["self"]
    now = float(obs.get("t") or 0.0)
    brawler._note_drops(brain, obs, now)
    if brawler._zone_soon(me, obs):
        intent = brawler._zone(brain, obs, me, None, now)
    elif _farming(me, now):
        intent = _farm(brain, obs, me, now)
    else:
        intent = _hold(brain, obs, me, now)
    return brawler._finish(brain, obs, me, intent, now)


def _farming(me: dict, now: float) -> bool:
    if now >= _FARM_S:
        return False
    return float(me.get("cubes") or 0.0) < _FARM_CUBES


def _farm(brain, obs, me, now: float) -> Intent:
    _remember(brain, obs, me, now)
    _mark_visited(brain, me)
    foe = _must_fight(brain, obs, me)
    if foe is not None:
        return brawler._rush(brain, obs, me, foe, now)
    swap = _lightning_swap(obs, me)
    if swap is not None:
        intent = brawler._hunt(brain, obs, me, swap)
        intent.reason = "weapon"
        return intent
    pickup = _weapon_goal(brain, obs, me)
    if pickup is not None:
        intent = brawler._hunt(brain, obs, me, pickup)
        intent.reason = "weapon"
        return intent
    drop = brawler._drop_move(brain, obs, me, now)
    if drop is not None:
        return drop
    npc = _farm_npc(brain, me)
    if npc is not None:
        return brawler._kill_npc(brain, obs, me, npc, now)
    crate = _remembered_crate(brain, me)
    if crate is not None:
        return _break_crate(brain, obs, me, crate)
    reach = brawler._knob(brain, "cube_detour_m", brawler._CUBE_FAR_M)
    cube = brawler._nearest_cube(obs, me, reach)
    if cube is not None:
        return brawler._walk_cube(brain, obs, me, cube["pos"], now)
    lead = _sound_lead(brain, me)
    if lead is not None:
        return _scan(_walk_to(brain, obs, me, lead, "sound"), me, now)
    cell = _explore_cell(brain, obs, me)
    if cell is not None:
        return _scan(_walk_to(brain, obs, me, cell, "explore"), me, now)
    return _scan(_toward_center(brain, obs, me), me, now)


def _hold(brain, obs, me, now: float) -> Intent:
    spot = _hold_spot(brain, obs, me)
    if spot is None:
        return _toward_center(brain, obs, me)
    if _healing(brain, me, now):
        return _heal(brain, obs, me)
    brain._farmer_heal_at = None
    looted = _after_kill(brain, obs, me, now)
    if looted is not None:
        return looted
    fight = _hold_fight(brain, obs, me, spot, now)
    if fight is not None:
        return fight
    return _stand(brain, obs, me, spot)


def _must_fight(brain, obs, me):
    close = brawler._nearest_known(brain, me, brawler._FIGHT_M)
    if close is not None:
        return close
    store = getattr(brain.world, "mechs", None)
    if not isinstance(store, dict):
        return None
    for event in obs.get("events") or []:
        if event.get("type") != "damaged" or event.get("weapon") == "npc":
            continue
        attacker = event.get("by")
        if attacker is None:
            continue
        track = store.get(int(attacker))
        if track is not None:
            return track
    return None


def _weapon_goal(brain, obs, me) -> dict | None:
    have = _slot_kind(me, 1)
    have_rank = _WEAPON_RANK.get(have, 99)
    px, py = brawler._xy(me["pos"])
    best = None
    best_key = None
    for pickup in _known_weapons(brain, obs):
        kind = str(pickup.get("kind") or "")
        rank = _WEAPON_RANK.get(kind)
        if rank is None or rank >= have_rank:
            continue
        pos = pickup.get("pos")
        if not pos:
            continue
        dist = math.hypot(float(pos[0]) - px, float(pos[1]) - py)
        if dist > _WEAPON_M:
            continue
        key = (rank, dist)
        if best_key is None or key < best_key:
            best, best_key = pickup, key
    return best


def _slot_kind(me: dict, slot: int) -> str | None:
    weapons = list(me.get("weapons") or [])
    if slot >= len(weapons) or weapons[slot] is None:
        return None
    weapon = weapons[slot]
    kind = weapon.get("kind") if isinstance(weapon, dict) else weapon
    return None if kind is None else str(kind)


def _farm_npc(brain, me):
    px, py = brawler._xy(me["pos"])
    best = None
    best_dist = _NPC_M
    for item in _mem(brain)["npc"].values():
        dist = math.hypot(item["pos"][0] - px, item["pos"][1] - py)
        if dist > best_dist:
            continue
        best, best_dist = item, dist
    if best is None:
        return None
    return SimpleNamespace(id=best["id"], pos=best["pos"], hp=best.get("hp"), last_seen_t=best["last_seen"])


def _break_crate(brain, obs, me, crate) -> Intent:
    pos = crate["pos"]
    px, py = brawler._xy(me["pos"])
    dist = math.hypot(float(pos[0]) - px, float(pos[1]) - py)
    cz = float(pos[2]) if len(pos) > 2 else float(me["pos"][2])
    intent = Intent(
        reason="crate",
        aim_target=SimpleNamespace(pos=(float(pos[0]), float(pos[1]), cz), vel=(0.0, 0.0, 0.0), id=crate.get("id")),
        fire=(True, False),
        boost=brawler._boost(brain, obs, me),
    )
    if dist <= 20.0:
        intent.move_vec = (0.0, 0.0)
        intent.slow = True
        return intent
    move, jump = brawler._toward(brain, obs, me, pos)
    intent.move_vec = move
    intent.jump = jump
    return intent


def _lightning_swap(obs, me) -> dict | None:
    """A laser or mortar in slot 1 is dropped for a lightning we can see inside 60 m."""
    if _slot_kind(me, 1) not in ("laser", "mortar"):
        return None
    px, py = brawler._xy(me["pos"])
    best = None
    best_dist = _SWAP_M
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


def _mem(brain) -> dict:
    mem = getattr(brain, "_farmer_seen", None)
    if mem is None:
        mem = {"npc": {}, "crate": {}, "weapon": {}}
        brain._farmer_seen = mem
    return mem


def _remember(brain, obs, me, now: float) -> None:
    """Drones, crates, and weapon pickups stay a goal after they leave the wedge."""
    mem = _mem(brain)
    px, py = brawler._xy(me["pos"])
    for event in obs.get("events") or []:
        if event.get("type") == "kill" and event.get("target") is not None:
            mem["npc"].pop(int(event["target"]), None)
        if event.get("type") == "took" and event.get("id") is not None:
            mem["weapon"].pop(int(event["id"]), None)
    seen_npcs: set[int] = set()
    for actor in obs.get("mechs") or []:
        if str(actor.get("kind") or "") != "npc" or actor.get("id") is None:
            continue
        ident = int(actor["id"])
        seen_npcs.add(ident)
        pos = actor.get("pos") or (0.0, 0.0, 0.0)
        mem["npc"][ident] = {
            "id": ident,
            "pos": (float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0),
            "hp": None if actor.get("hp") is None else float(actor["hp"]),
            "last_seen": now,
        }
    for ident, item in list(mem["npc"].items()):
        if ident in seen_npcs:
            continue
        if math.hypot(item["pos"][0] - px, item["pos"][1] - py) <= _FORGET_M:
            mem["npc"].pop(ident, None)
    seen_crates: set[int] = set()
    for crate in obs.get("crates") or []:
        if not isinstance(crate, dict) or not crate.get("pos"):
            continue
        ident = int(crate["id"]) if crate.get("id") is not None else _pos_key(crate["pos"])
        seen_crates.add(ident)
        pos = crate["pos"]
        mem["crate"][ident] = {
            "id": crate.get("id"),
            "pos": [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else float(me["pos"][2])],
            "hp": None if crate.get("hp") is None else float(crate["hp"]),
            "last_seen": now,
        }
    for ident, item in list(mem["crate"].items()):
        if ident in seen_crates:
            continue
        if math.hypot(item["pos"][0] - px, item["pos"][1] - py) <= 12.0:
            mem["crate"].pop(ident, None)
    seen_weapons: set[int] = set()
    for pickup in obs.get("pickups") or []:
        kind = str(pickup.get("kind") or "")
        if kind not in _WEAPON_RANK or not pickup.get("pos") or pickup.get("id") is None:
            continue
        ident = int(pickup["id"])
        seen_weapons.add(ident)
        pos = pickup["pos"]
        mem["weapon"][ident] = {
            "id": ident,
            "kind": kind,
            "pos": [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else float(me["pos"][2])],
            "hp": None,
            "last_seen": now,
        }
    reach = float(brain.rules.loot.pickup_radius)
    for ident, item in list(mem["weapon"].items()):
        if ident in seen_weapons:
            continue
        if math.hypot(item["pos"][0] - px, item["pos"][1] - py) <= reach:
            mem["weapon"].pop(ident, None)


def _pos_key(pos) -> int:
    return (int(round(float(pos[0]))) << 20) ^ int(round(float(pos[1])))


def _known_weapons(brain, obs) -> list:
    found = [pickup for pickup in (obs.get("pickups") or []) if isinstance(pickup, dict)]
    seen = {int(pickup["id"]) for pickup in found if pickup.get("id") is not None}
    for item in _mem(brain)["weapon"].values():
        if item["id"] in seen:
            continue
        found.append({"id": item["id"], "kind": item["kind"], "pos": item["pos"]})
    return found


def _remembered_crate(brain, me) -> dict | None:
    px, py = brawler._xy(me["pos"])
    best = None
    best_dist = _CRATE_M
    for item in _mem(brain)["crate"].values():
        dist = math.hypot(item["pos"][0] - px, item["pos"][1] - py)
        if dist <= best_dist:
            best, best_dist = item, dist
    if best is None:
        return None
    return {"id": best["id"], "pos": best["pos"]}


def _sound_lead(brain, me):
    hints = [hint for hint in brain.world.sounds if hint.kind in ("npc_gun", "crate")]
    if not hints:
        return None
    hint = max(hints, key=lambda item: item.t)
    dist = _LEAD_NEAR_M if hint.dist_bucket == "near" else _LEAD_M
    px, py = brawler._xy(me["pos"])
    return (px + math.cos(hint.bearing) * dist, py + math.sin(hint.bearing) * dist)


def _visited(brain) -> set:
    cells = getattr(brain, "_farmer_visited", None)
    if cells is None:
        cells = set()
        brain._farmer_visited = cells
    return cells


def _mark_visited(brain, me) -> None:
    px, py = brawler._xy(me["pos"])
    cells = _visited(brain)
    gx = math.floor(px / _CELL_M)
    gy = math.floor(py / _CELL_M)
    for ix in range(gx - 1, gx + 2):
        for iy in range(gy - 1, gy + 2):
            cx = (ix + 0.5) * _CELL_M
            cy = (iy + 0.5) * _CELL_M
            if math.hypot(cx - px, cy - py) <= _VISIT_M:
                cells.add((ix, iy))


def _explore_cell(brain, obs, me):
    zone = obs.get("zone") or {}
    center = zone.get("next_center")
    radius = float(zone.get("next_radius") or 0.0)
    if not center or radius <= 0.0:
        return None
    zx, zy = float(center[0]), float(center[1])
    px, py = brawler._xy(me["pos"])
    visited = _visited(brain)
    gx = math.floor(px / _CELL_M)
    gy = math.floor(py / _CELL_M)
    span = min(int(radius / _CELL_M) + 2, 30)
    best = None
    best_dist = math.inf
    for ix in range(gx - span, gx + span + 1):
        for iy in range(gy - span, gy + span + 1):
            if (ix, iy) in visited:
                continue
            cx = (ix + 0.5) * _CELL_M
            cy = (iy + 0.5) * _CELL_M
            if math.hypot(cx - zx, cy - zy) > radius:
                continue
            dist = math.hypot(cx - px, cy - py)
            if dist < best_dist:
                best, best_dist = (cx, cy), dist
    return best


def _walk_to(brain, obs, me, pos, reason: str) -> Intent:
    move, jump = brawler._toward(brain, obs, me, pos)
    return Intent(
        reason=reason,
        move_vec=move,
        jump=jump,
        fire=(False, False),
        boost=brawler._boost(brain, obs, me),
    )


def _scan(intent: Intent, me, now: float) -> Intent:
    """Sweep the 120° wedge ±150° around the way the legs are walking."""
    move = intent.move_vec or (0.0, 0.0)
    if math.hypot(float(move[0]), float(move[1])) > 0.2:
        travel = math.atan2(float(move[1]), float(move[0]))
    else:
        aim = me.get("aim") or (0.0, 0.0)
        travel = float(aim[0])
    side = 1.0 if int(math.floor(now / _SCAN_S)) % 2 == 0 else -1.0
    yaw = travel + side * _SCAN_RAD
    intent.aim_target = None
    intent.aim_yaw_pitch = (math.atan2(math.sin(yaw), math.cos(yaw)), 0.0)
    return intent


def _toward_center(brain, obs, me) -> Intent:
    zone = obs.get("zone") or {}
    center = zone.get("next_center")
    if not center:
        return Intent(slow=True, reason="farm", fire=(False, False))
    px, py = brawler._xy(me["pos"])
    goal = (float(center[0]), float(center[1]))
    move, jump = brawler._toward(brain, obs, me, goal)
    return Intent(
        reason="farm",
        move_vec=move,
        jump=jump,
        aim_yaw_pitch=(math.atan2(goal[1] - py, goal[0] - px), 0.0),
        fire=(False, False),
        boost=brawler._boost(brain, obs, me),
    )


def _zone_key(obs) -> tuple:
    zone = obs.get("zone") or {}
    center = zone.get("next_center") or (0.0, 0.0)
    return (
        round(float(center[0]), 1),
        round(float(center[1]), 1),
        round(float(zone.get("next_radius") or 0.0), 1),
        round(float(zone.get("shrink_start") or 0.0), 1),
        round(float(zone.get("shrink_end") or 0.0), 1),
    )


def _hold_spot(brain, obs, me):
    key = _zone_key(obs)
    if getattr(brain, "_farmer_phase", None) == key and getattr(brain, "_farmer_spot", None) is not None:
        return brain._farmer_spot
    zone = obs.get("zone") or {}
    spot = None
    if brain.map is not None and zone.get("next_center") is not None and zone.get("next_radius") is not None:
        spot = best_cover(
            brain.map,
            me["pos"],
            [],
            zone["next_center"],
            float(zone["next_radius"]),
            _HOLD_REACH_M,
            edge_margin=_HOLD_EDGE_M,
            min_dirs=_HOLD_DIRS,
        )
    if spot is None and zone.get("next_center") is not None:
        spot = brain.ring_point(obs, _HOLD_EDGE_M)
    brain._farmer_phase = key
    brain._farmer_spot = spot
    return spot


def _healing(brain, me, now: float) -> bool:
    frac = brawler._hp_frac(me)
    if frac >= _HEAL_EXIT:
        brain._farmer_healing = False
        return False
    quiet = _damage_quiet(brain, now)
    if getattr(brain, "_farmer_healing", False):
        return quiet
    if frac < _HEAL_ENTER and quiet:
        brain._farmer_healing = True
        return True
    return False


def _damage_quiet(brain, now: float) -> bool:
    last = brain.world.last_damage_t
    if last is None:
        return True
    return now - float(last) >= _QUIET_S


def _heal(brain, obs, me) -> Intent:
    enemies = brawler._visible(brain)
    goal = _break_los(brain, me, enemies)
    if goal is None:
        return Intent(reason="heal", move_vec=(0.0, 0.0), slow=True, fire=(False, False))
    px, py = brawler._xy(me["pos"])
    if math.hypot(goal[0] - px, goal[1] - py) <= 2.0:
        return Intent(reason="heal", move_vec=(0.0, 0.0), slow=True, fire=(False, False))
    move, jump = brawler._toward(brain, obs, me, goal)
    return Intent(reason="heal", move_vec=move, jump=jump, fire=(False, False), boost=False)


def _break_los(brain, me, enemies):
    cached = getattr(brain, "_farmer_heal_at", None)
    if cached is not None:
        return cached
    px, py, pz = (float(v) for v in me["pos"][:3])
    grid = brain.map
    ranked: list[tuple[int, float, float]] = []
    threat = enemies[0] if enemies else None
    if grid is not None and grid.cover_points.size:
        covers = grid.cover_points
        shields = grid.cover_shield
        d2 = (covers[:, 0] - px) ** 2 + (covers[:, 1] - py) ** 2
        near = np.flatnonzero(d2 <= _HEAL_REACH_M * _HEAL_REACH_M)
        if near.size:
            near = near[np.argsort(d2[near])][:40]
            for index in near:
                x = float(covers[index, 0])
                y = float(covers[index, 1])
                rank = 1
                if threat is not None and getattr(shields, "ndim", 0) == 2:
                    dx = float(threat.pos[0]) - x
                    dy = float(threat.pos[1]) - y
                    sector = int(math.floor((math.atan2(dy, dx) + math.pi / 8.0) / (math.pi / 4.0))) % 8
                    if bool(shields[int(index), sector]):
                        rank = 0
                ranked.append((rank, x, y))
    bearing = None
    if threat is not None:
        bearing = math.atan2(float(threat.pos[1]) - py, float(threat.pos[0]) - px)
    elif brain.threat_bearing() is not None:
        bearing = brain.threat_bearing()
    if bearing is not None:
        away = bearing + math.pi
        for dist in (6.0, 12.0, 18.0):
            ranked.append((1, px + math.cos(away) * dist, py + math.sin(away) * dist))
    for dist in (6.0, 12.0, 18.0):
        for step in range(8):
            ang = step * math.pi / 4.0
            ranked.append((2, px + math.cos(ang) * dist, py + math.sin(ang) * dist))
    ranked.sort(key=lambda item: item[0])
    for _rank, x, y in ranked:
        if grid is not None and not grid.in_bounds(x, y):
            continue
        z = float(grid.ground_z(x, y)) if grid is not None else pz
        pos = (x, y, z)
        if any(brawler._los_pos(brain, {"pos": pos}, enemy.pos) for enemy in enemies):
            continue
        brain._farmer_heal_at = pos
        return pos
    return None


def _after_kill(brain, obs, me, now: float):
    mark = _kill_mark(brain, now)
    if mark is None:
        brain._farmer_loot_mark = None
        brain._farmer_saw_cubes = False
        return None
    if getattr(brain, "_farmer_loot_mark", None) != mark[0]:
        brain._farmer_loot_mark = mark[0]
        brain._farmer_saw_cubes = False
    cube = brawler._nearest_cube(obs, me, _COLLECT_M)
    if cube is not None:
        brain._farmer_saw_cubes = True
        return brawler._walk_cube(brain, obs, me, cube["pos"], now)
    if brain._farmer_saw_cubes:
        return None
    pos = mark[1]
    px, py = brawler._xy(me["pos"])
    dist = math.hypot(pos[0] - px, pos[1] - py)
    if dist > _COLLECT_M or dist <= float(brain.rules.cubes.pickup_radius):
        return None
    move, jump = brawler._toward(brain, obs, me, pos)
    intent = Intent(reason="drop", move_vec=move, jump=jump, boost=brawler._boost(brain, obs, me))
    return brawler._face(intent, brain, me, now)


def _kill_mark(brain, now: float):
    drops = [item for item in getattr(brain, "_cube_drops", ()) if now <= item[0]]
    if not drops:
        return None
    return max(drops, key=lambda item: item[0])


def _hold_fight(brain, obs, me, spot, now: float) -> Intent | None:
    foes = _los_from_spot(brain, me, spot)
    if not foes:
        return None
    px, py = brawler._xy(me["pos"])
    foe = min(foes, key=lambda enemy: math.hypot(float(enemy.pos[0]) - px, float(enemy.pos[1]) - py))
    from_spot = math.hypot(float(foe.pos[0]) - spot[0], float(foe.pos[1]) - spot[1])
    dist_spot = math.hypot(px - spot[0], py - spot[1])
    if from_spot <= _LEASH_M and dist_spot <= _LEASH_M:
        return brawler._rush(brain, obs, me, foe, now)
    intent = Intent(
        reason="hold",
        aim_target=foe,
        fire=brawler._shots(brain, me, foe, now),
        boost=brawler._boost(brain, obs, me),
    )
    if dist_spot > _SPOT_M:
        move, jump = brawler._toward(brain, obs, me, spot)
        intent.move_vec = move
        intent.jump = jump
    else:
        intent.move_vec = (0.0, 0.0)
        intent.slow = True
    return intent


def _los_from_spot(brain, me, spot) -> list:
    z = float(me["pos"][2])
    if brain.map is not None:
        z = float(brain.map.ground_z(spot[0], spot[1]))
    origin = {"pos": (float(spot[0]), float(spot[1]), z)}
    return [enemy for enemy in brawler._visible(brain) if brawler._los_pos(brain, origin, enemy.pos)]


def _stand(brain, obs, me, spot) -> Intent:
    px, py = brawler._xy(me["pos"])
    dist = math.hypot(spot[0] - px, spot[1] - py)
    bearing = brain.threat_bearing()
    if bearing is None:
        bearing = math.atan2(spot[1] - py, spot[0] - px) if dist > 1.0 else 0.0
    if dist <= _SPOT_M:
        return Intent(
            reason="hold",
            move_vec=(0.0, 0.0),
            slow=True,
            aim_yaw_pitch=(bearing, 0.0),
            fire=(False, False),
        )
    move, jump = brawler._toward(brain, obs, me, spot)
    return Intent(
        reason="hold",
        move_vec=move,
        jump=jump,
        aim_yaw_pitch=(bearing, 0.0),
        fire=(False, False),
        boost=brawler._boost(brain, obs, me),
    )
