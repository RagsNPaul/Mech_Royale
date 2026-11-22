"""The farmer: drones and loot first, then a covered hold inside the next circle."""

import math
import time

import numpy as np

from bot.brain import Brain
from tests.fixtures import at_time, make_obs, with_self
from tests.test_scenarios import _actor, _los_point, _origin, _quiet, _safe_zone


def _debug(act) -> str:
    return str((act.get("debug") or {}).get("label") or "")


def _dot(move, dx: float, dy: float) -> float:
    return float(move[0]) * dx + float(move[1]) * dy


def _ang(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def test_search_sweeps_aim_while_it_walks(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    aims = []
    moved = []
    for step in range(5):
        obs = make_obs(
            at_time(10.0 + step * 0.5),
            with_self(hp=1400, pos=origin, cubes=0, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "gun"]),
            _safe_zone(origin),
            **_quiet(),
        )
        act = brain.act(obs, time.perf_counter())
        aims.append(float(act["aim"][0]))
        moved.append(math.hypot(act["move"][0], act["move"][1]) > 0.3)
    assert all(moved)
    assert max(aims) - min(aims) > 1.0


def test_remembered_npc_stays_the_goal_out_of_view(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    npc_pos = _los_point(brain, origin, 90.0)
    npc = _actor(101, npc_pos, 350)
    npc["kind"] = "npc"
    npc["max_hp"] = 350.0
    seen = make_obs(
        at_time(10.0),
        with_self(hp=1400, pos=origin, cubes=0, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "gun"]),
        _safe_zone(origin),
        **_quiet(mechs=[npc]),
    )
    brain.act(seen, time.perf_counter())
    hidden = make_obs(
        at_time(11.0),
        with_self(hp=1400, pos=origin, cubes=0, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "gun"]),
        _safe_zone(origin),
        **_quiet(),
    )
    act = brain.act(hidden, time.perf_counter())
    assert "| npc" in _debug(act)
    assert _dot(act["move"], npc_pos[0] - origin[0], npc_pos[1] - origin[1]) > 0.0


def test_npc_gun_sound_is_walked(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    point = _los_point(brain, origin, 100.0)
    bearing = math.atan2(point[1] - origin[1], point[0] - origin[0])
    obs = make_obs(
        at_time(10.0),
        with_self(hp=1400, pos=origin, cubes=0, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "gun"]),
        _safe_zone(origin),
        **_quiet(sounds=[{"kind": "npc_gun", "bearing": bearing, "dist": "mid", "age": 0.0}]),
    )
    act = brain.act(obs, time.perf_counter())
    travel = math.atan2(act["move"][1], act["move"][0])
    assert _ang(travel, bearing) < 0.4
    assert "| sound" in _debug(act)


def test_farms_the_npc_not_the_distant_mech(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    npc_pos = _los_point(brain, origin, 90.0)
    vx, vy = npc_pos[0] - origin[0], npc_pos[1] - origin[1]
    scale = 150.0 / 90.0
    mech_pos = [origin[0] - vy * scale, origin[1] + vx * scale, origin[2]]
    npc = _actor(101, npc_pos, 350)
    npc["kind"] = "npc"
    npc["max_hp"] = 350.0
    mech = _actor(7, mech_pos, 1400)
    obs = make_obs(
        at_time(10.0),
        with_self(hp=1400, pos=origin, cubes=0, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "gun"]),
        _safe_zone(origin),
        **_quiet(mechs=[npc, mech]),
    )
    act = brain.act(obs, time.perf_counter())
    assert "| npc" in _debug(act)
    assert _dot(act["move"], vx, vy) > _dot(act["move"], mech_pos[0] - origin[0], mech_pos[1] - origin[1])
    assert _dot(act["move"], vx, vy) > 0.0


def test_hold_moves_to_cover_inside_the_next_circle(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    obs = make_obs(
        at_time(120.0),
        with_self(hp=1400, pos=origin, cubes=0, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(),
    )
    obs["self"]["max_hp"] = 1400.0
    act = brain.act(obs, time.perf_counter())
    spot = brain._farmer_spot
    assert spot is not None
    assert math.hypot(spot[0] - origin[0], spot[1] - origin[1]) <= 800.0 - 25.0 + 0.1
    covers = brain.map.cover_points
    nearest = int(np.argmin((covers[:, 0] - spot[0]) ** 2 + (covers[:, 1] - spot[1]) ** 2))
    assert float((covers[nearest, 0] - spot[0]) ** 2 + (covers[nearest, 1] - spot[1]) ** 2) < 1.0
    assert int(np.asarray(brain.map.cover_shield[nearest]).sum()) >= 4
    dx, dy = spot[0] - origin[0], spot[1] - origin[1]
    assert _dot(act["move"], dx, dy) > 0.0
    assert "| hold" in _debug(act)


def test_kill_collects_cubes_then_returns(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    first = make_obs(
        at_time(120.0),
        with_self(hp=1400, pos=origin, cubes=0, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(),
    )
    first["self"]["max_hp"] = 1400.0
    brain.act(first, time.perf_counter())
    spot = brain._farmer_spot
    body = _body(brain, origin)
    victim = _actor(9, body, 400)
    seen = make_obs(
        at_time(120.1),
        with_self(hp=1400, pos=origin, cubes=0, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[victim]),
    )
    seen["self"]["max_hp"] = 1400.0
    brain.act(seen, time.perf_counter())
    killed = make_obs(
        at_time(120.2),
        with_self(hp=1400, pos=origin, cubes=0, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(events=[{"type": "kill", "target": 9}], cubes=[{"id": 5019, "pos": body}]),
    )
    killed["self"]["max_hp"] = 1400.0
    grab = brain.act(killed, time.perf_counter())
    toward_cube = _dot(grab["move"], body[0] - origin[0], body[1] - origin[1])
    toward_spot = _dot(grab["move"], spot[0] - origin[0], spot[1] - origin[1])
    assert toward_cube > toward_spot
    assert toward_cube > 0.0
    back = make_obs(
        at_time(120.3),
        with_self(hp=1400, pos=origin, cubes=0, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(),
    )
    back["self"]["max_hp"] = 1400.0
    returned = brain.act(back, time.perf_counter())
    assert _dot(returned["move"], spot[0] - origin[0], spot[1] - origin[1]) > _dot(
        returned["move"], body[0] - origin[0], body[1] - origin[1]
    )


def test_low_hp_steps_out_of_los(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    enemy_pos = _los_point(brain, origin, 40.0)
    enemy = _actor(4, enemy_pos, 1400)
    obs = make_obs(
        at_time(120.0),
        with_self(hp=560, pos=origin, cubes=0, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy]),
    )
    obs["self"]["max_hp"] = 1400.0
    eye = float(brain.rules.mech.eye_height)
    assert brain.map.los(
        (origin[0], origin[1], origin[2] + eye),
        (enemy_pos[0], enemy_pos[1], enemy_pos[2] + eye),
        True,
    )
    act = brain.act(obs, time.perf_counter())
    goal = brain._farmer_heal_at
    assert goal is not None
    assert not brain.map.los(
        (goal[0], goal[1], goal[2] + eye),
        (enemy_pos[0], enemy_pos[1], enemy_pos[2] + eye),
        True,
    )
    assert "| heal" in _debug(act)
    assert math.hypot(act["move"][0], act["move"][1]) > 0.2


def _body(brain, origin: list[float]) -> list[float]:
    """A point 20 m from us, perpendicular to the hold spot, still inside the map."""
    spot = brain._farmer_spot
    dx, dy = spot[0] - origin[0], spot[1] - origin[1]
    length = math.hypot(dx, dy) or 1.0
    return [origin[0] - dy / length * 20.0, origin[1] + dx / length * 20.0, origin[2]]
