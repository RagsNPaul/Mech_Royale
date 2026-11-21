"""Scenario tests for the v1 mode priority."""

import math
import time

from bot.brain import Brain
from tests.fixtures import at_time, make_obs, with_self, with_zone


def _origin(brain) -> list[float]:
    x, y = 320.21, -280.29
    if brain.map is not None and brain.map.in_bounds(x, y):
        z = float(brain.map.ground_z(x, y))
        return [x, y, z]
    return [0.0, 0.0, 0.0]


def _actor(ident: int, pos: list[float], hp: float) -> dict:
    return {
        "id": ident,
        "kind": "mech",
        "pos": [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0],
        "vel": [0.0, 0.0, 0.0],
        "aim": [0.0, 0.0],
        "hp": hp,
        "max_hp": 1400.0,
        "weapons": ["gun"],
        "cubes": 0,
    }


def _safe_zone(pos: list[float]):
    center = [pos[0], pos[1]]
    return with_zone(center, 900.0, center, 800.0, 1000.0, 1100.0, 5.0)


def _los_point(brain, origin: list[float], dist: float) -> list[float]:
    eye = float(brain.rules.mech.eye_height)
    z = origin[2]
    for step in range(24):
        ang = step * math.tau / 24.0
        x = origin[0] + math.cos(ang) * dist
        y = origin[1] + math.sin(ang) * dist
        if brain.map is None:
            return [x, y, z]
        if not brain.map.in_bounds(x, y):
            continue
        ez = float(brain.map.ground_z(x, y))
        if brain.map.los((origin[0], origin[1], z + eye), (x, y, ez + eye), True):
            return [x, y, ez]
    return [origin[0] + dist, origin[1], z]


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def test_outside_the_zone_moves_toward_the_next_center(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    nxt = [origin[0] + 180.0, origin[1]]
    obs = make_obs(
        at_time(20.0),
        with_self(hp=1400, pos=origin, in_zone=False, aim=[0.0, 0.0]),
        with_zone(origin[:2], 40.0, nxt, 30.0, 1000.0, 1100.0, 20.0),
        mechs=[],
        events=[],
        sounds=[],
    )
    act = brain.act(obs, time.perf_counter())
    assert brain.mode == "ZONE"
    toward = (nxt[0] - origin[0], nxt[1] - origin[1])
    assert act["move"][0] * toward[0] + act["move"][1] * toward[1] > 0.0


def test_lone_low_target_is_fought_once_aligned(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    enemy_pos = _los_point(brain, origin, 50.0)
    obs = make_obs(
        at_time(20.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0]),
        _safe_zone(origin),
        mechs=[_actor(5, enemy_pos, 150)],
        events=[],
        sounds=[],
    )
    act = brain.act(obs, time.perf_counter())
    assert brain.mode == "FIGHT"
    obs["self"]["aim"] = list(act["aim"])
    obs["t"] = 20.1
    obs["tick"] = int(round(obs["t"] * 20))
    aligned = brain.act(obs, time.perf_counter())
    assert brain.mode == "FIGHT"
    assert aligned["fire"][0] is True


def test_low_hp_against_two_enemies_retreats_while_facing_one(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    first = _actor(5, [origin[0] + 40.0, origin[1], origin[2]], 1400)
    second = _actor(6, [origin[0] + 70.0, origin[1] + 8.0, origin[2]], 1400)
    obs = make_obs(
        at_time(20.0),
        with_self(hp=0.14 * 1400.0, pos=origin, in_zone=True, aim=[0.0, 0.0]),
        _safe_zone(origin),
        mechs=[first, second],
        events=[],
        sounds=[],
    )
    act = brain.act(obs, time.perf_counter())
    assert brain.mode == "DISENGAGE"
    aim = act["aim"][0]
    errors = []
    for enemy in (first, second):
        bearing = math.atan2(enemy["pos"][1] - origin[1], enemy["pos"][0] - origin[0])
        errors.append(abs(_wrap(aim - bearing)))
    assert min(errors) < 0.45
    away = (origin[0] - first["pos"][0], origin[1] - first["pos"][1])
    assert act["move"][0] * away[0] + act["move"][1] * away[1] > 0.0


def test_two_enemies_above_the_floor_stay_in_the_fight(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    first = _actor(5, [origin[0] + 40.0, origin[1], origin[2]], 1400)
    second = _actor(6, [origin[0] + 70.0, origin[1] + 8.0, origin[2]], 1400)
    obs = make_obs(
        at_time(20.0),
        with_self(hp=0.30 * 1400.0, pos=origin, in_zone=True, aim=[0.0, 0.0]),
        _safe_zone(origin),
        mechs=[first, second],
        events=[],
        sounds=[],
    )
    brain.act(obs, time.perf_counter())
    assert brain.mode == "FIGHT"


def test_rear_damage_is_not_hide_and_aim_turns_toward_it(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    bearing = 2.2
    obs = make_obs(
        at_time(20.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0]),
        _safe_zone(origin),
        mechs=[],
        sounds=[],
        events=[
            {
                "type": "damaged",
                "amount": 40,
                "weapon": "gun",
                "bearing": bearing,
                "by": 9,
                "rear": True,
            }
        ],
    )
    act = brain.act(obs, time.perf_counter())
    assert brain.mode != "HIDE"
    assert abs(_wrap(act["aim"][0] - bearing)) < 0.35


def test_no_information_hides_and_goes_quiet_at_cover(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    obs = make_obs(
        at_time(40.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0]),
        with_zone(origin[:2], 2000.0, origin[:2], 1800.0, 5000.0, 5100.0, 5.0),
        mechs=[],
        events=[],
        sounds=[],
    )
    walk = float(brain.rules.mech.walk_speed)
    step = float(brain.rules.sound.step_speed)
    quiet = False
    for _ in range(80):
        act = brain.act(obs, time.perf_counter())
        assert brain.mode == "HIDE"
        assert act["fire"] == [False, False]
        if math.hypot(act["move"][0], act["move"][1]) * walk < step:
            quiet = True
            break
        obs["self"]["pos"][0] += act["move"][0] * walk * 0.4
        obs["self"]["pos"][1] += act["move"][1] * walk * 0.4
        obs["t"] += 0.4
        obs["tick"] = int(round(obs["t"] * 20))
    assert quiet


def test_modes_do_not_flip_flop(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    enemy_pos = _los_point(brain, origin, 50.0)
    previous = brain.mode
    switches = 0
    # 2.9 s of alternation sits inside the 3 s hold.
    for step in range(29):
        t = 10.0 + step * 0.1
        mechs = [_actor(5, enemy_pos, 150)] if step % 2 == 0 else []
        obs = make_obs(
            at_time(t),
            with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0]),
            _safe_zone(origin),
            mechs=mechs,
            events=[],
            sounds=[],
        )
        brain.act(obs, time.perf_counter())
        if brain.mode != previous:
            switches += 1
            previous = brain.mode
    assert switches <= 1


def test_missing_map_never_raises(monkeypatch, hello):
    def explode(*_args, **_kwargs):
        raise RuntimeError("map unavailable")

    monkeypatch.setattr("bot.brain.load_map", explode)
    brain = Brain(hello)
    assert brain.map is None
    assert brain.nav is None
    origin = [0.0, 0.0, 0.0]
    enemy = _actor(5, [50.0, 0.0, 0.0], 150)
    scenarios = [
        make_obs(
            at_time(20.0),
            with_self(hp=1400, pos=origin, in_zone=False, aim=[0.0, 0.0]),
            with_zone(origin[:2], 20.0, [180.0, 0.0], 30.0, 1000.0, 1100.0, 20.0),
            mechs=[],
            events=[],
            sounds=[],
        ),
        make_obs(
            at_time(20.0),
            with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0]),
            _safe_zone(origin),
            mechs=[enemy],
            events=[],
            sounds=[],
        ),
        make_obs(
            at_time(20.0),
            with_self(hp=420, pos=origin, in_zone=True, aim=[0.0, 0.0]),
            _safe_zone(origin),
            mechs=[enemy, _actor(6, [70.0, 5.0, 0.0], 1400)],
            events=[],
            sounds=[],
        ),
        make_obs(
            at_time(40.0),
            with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0]),
            _safe_zone(origin),
            mechs=[],
            events=[],
            sounds=[],
        ),
    ]
    for obs in scenarios:
        brain.act(obs, time.perf_counter())
