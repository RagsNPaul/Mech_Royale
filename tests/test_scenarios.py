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


def test_empty_slot_takes_nearby_lightning(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    pickup = [origin[0] + 3.0, origin[1], origin[2]]
    obs = make_obs(
        at_time(10.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", None]),
        _safe_zone(origin),
        mechs=[],
        events=[],
        sounds=[],
        pickups=[{"id": 8013, "kind": "lightning", "pos": pickup}],
        crates=[],
        drops=[],
    )
    act = brain.act(obs, time.perf_counter())
    assert brain.mode == "LOOT"
    assert act["take"] == 8013


def test_lightning_closes_from_100m(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    enemy = _actor(5, [origin[0] + 100.0, origin[1], origin[2]], 1400)
    obs = make_obs(
        at_time(20.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        mechs=[enemy],
        events=[],
        sounds=[],
        pickups=[],
        crates=[],
        drops=[],
    )
    act = brain.act(obs, time.perf_counter())
    assert brain.mode == "FIGHT"
    assert act["move"][0] > 0.0


def test_lightning_fires_inside_the_cone(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    enemy = _actor(5, [origin[0] + 30.0, origin[1], origin[2]], 1400)
    obs = make_obs(
        at_time(20.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        mechs=[enemy],
        events=[],
        sounds=[],
        pickups=[],
        crates=[],
        drops=[],
    )
    aimed = brain.act(obs, time.perf_counter())
    obs["self"]["aim"] = list(aimed["aim"])
    obs["t"] = 20.1
    obs["tick"] = int(round(obs["t"] * 20))
    fired = brain.act(obs, time.perf_counter())
    assert brain.mode == "FIGHT"
    assert fired["fire"][1] is True


def test_npc_is_farmed_when_no_hostiles(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    npc = _actor(101, [origin[0] + 40.0, origin[1], origin[2]], 350)
    npc["kind"] = "npc"
    npc["max_hp"] = 350
    obs = make_obs(
        at_time(15.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", None]),
        _safe_zone(origin),
        mechs=[npc],
        events=[],
        sounds=[],
        pickups=[],
        crates=[],
        drops=[],
    )
    aimed = brain.act(obs, time.perf_counter())
    assert brain.mode == "LOOT"
    obs["self"]["aim"] = list(aimed["aim"])
    obs["t"] = 15.1
    obs["tick"] = int(round(obs["t"] * 20))
    fired = brain.act(obs, time.perf_counter())
    assert brain.mode == "LOOT"
    assert fired["fire"][0] is True


def _quiet(**overrides):
    base = {"mechs": [], "events": [], "sounds": [], "pickups": [], "crates": [], "cubes": [], "drops": []}
    base.update(overrides)
    return base


def test_farms_the_nearest_npc_at_gun_range(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    far = _actor(101, [origin[0] + 120.0, origin[1], origin[2]], 350)
    far["kind"] = "npc"
    far["max_hp"] = 350
    obs = make_obs(
        at_time(80.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", None]),
        _safe_zone(origin),
        **_quiet(mechs=[far]),
    )
    act = brain.act(obs, time.perf_counter())
    assert brain.mode == "LOOT"
    assert act["move"][0] > 0.0

    close_brain = Brain(hello)
    close = _actor(102, [origin[0] + 20.0, origin[1], origin[2]], 350)
    close["kind"] = "npc"
    close["max_hp"] = 350
    close_obs = make_obs(
        at_time(80.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", None]),
        _safe_zone(origin),
        **_quiet(mechs=[close]),
    )
    backed = close_brain.act(close_obs, time.perf_counter())
    assert close_brain.mode == "LOOT"
    assert backed["move"][0] < 0.0

    cubes = Brain(hello)
    dropped = [origin[0] + 55.0, origin[1], origin[2]]
    cube_obs = make_obs(
        at_time(80.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", None]),
        _safe_zone(origin),
        **_quiet(cubes=[{"id": 5012, "pos": dropped}]),
    )
    walked = cubes.act(cube_obs, time.perf_counter())
    assert cubes.mode == "LOOT"
    assert walked["move"][0] > 0.0

    blocked = Brain(hello)
    hostile = _actor(5, [origin[0] + 90.0, origin[1], origin[2]], 1400)
    blocked_obs = make_obs(
        at_time(80.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", None]),
        _safe_zone(origin),
        **_quiet(mechs=[far, hostile]),
    )
    blocked.act(blocked_obs, time.perf_counter())
    assert blocked.mode == "FIGHT"


def test_walks_to_a_visible_cube_inside_40m(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    cube = [origin[0] + 20.0, origin[1], origin[2]]
    enemy = _actor(5, [origin[0] - 80.0, origin[1], origin[2]], 1400)
    obs = make_obs(
        at_time(80.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", None]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy], cubes=[{"id": 5013, "pos": cube}]),
    )
    act = brain.act(obs, time.perf_counter())
    assert brain.mode == "LOOT"
    assert act["move"][0] > 0.0

    near = Brain(hello)
    close_enemy = _actor(6, [origin[0] - 30.0, origin[1], origin[2]], 1400)
    near_obs = make_obs(
        at_time(80.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", None]),
        _safe_zone(origin),
        **_quiet(mechs=[close_enemy], cubes=[{"id": 5013, "pos": cube}]),
    )
    near.act(near_obs, time.perf_counter())
    assert near.mode == "FIGHT"


def test_supply_drop_in_the_next_circle(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    landing = [origin[0] + 40.0, origin[1], origin[2]]
    drop = {"id": 20001, "pos": landing, "radius": 8, "lands_in": 0.0, "capture_s": 4, "progress": 0, "holder": None}
    obs = make_obs(
        at_time(80.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0]),
        _safe_zone(origin),
        **_quiet(drops=[drop]),
    )
    act = brain.act(obs, time.perf_counter())
    assert brain.mode == "DROP"
    assert act["move"][0] > 0.0

    late = Brain(hello)
    far_landing = [origin[0] + 400.0, origin[1], origin[2]]
    late_drop = {"id": 20002, "pos": far_landing, "radius": 8, "lands_in": 0.0, "capture_s": 4, "progress": 0, "holder": None}
    late_obs = make_obs(
        at_time(80.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0]),
        with_zone(origin[:2], 5000.0, origin[:2], 5000.0, 1000.0, 1100.0, 5.0),
        **_quiet(drops=[late_drop]),
    )
    late.act(late_obs, time.perf_counter())
    assert late.mode != "DROP"

    hurt = Brain(hello)
    hurt_obs = make_obs(
        at_time(80.0),
        with_self(hp=0.30 * 1400.0, pos=origin, in_zone=True, aim=[0.0, 0.0]),
        _safe_zone(origin),
        **_quiet(drops=[drop]),
    )
    hurt.act(hurt_obs, time.perf_counter())
    assert hurt.mode != "DROP"

    contest = Brain(hello)
    ring = [origin[0] + 200.0, origin[1], origin[2]]
    contested = {"id": 20003, "pos": ring, "radius": 8, "lands_in": 20.0, "capture_s": 4, "progress": 0, "holder": None}
    foe = _actor(7, ring, 1400)
    contest_obs = make_obs(
        at_time(80.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", None]),
        _safe_zone(origin),
        **_quiet(mechs=[foe], drops=[contested]),
    )
    contested_act = contest.act(contest_obs, time.perf_counter())
    assert contest.mode == "FIGHT"
    assert contest.target is not None and contest.target.id == 7
    assert contested_act["move"][0] > 0.0


def test_phase_two_zone_overrides_fight_and_keeps_firing(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    enemy = _actor(5, [origin[0], origin[1] + 40.0, origin[2]], 1400)
    fought = make_obs(
        at_time(20.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", None]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy]),
    )
    brain.act(fought, time.perf_counter())
    assert brain.mode == "FIGHT"

    nxt = [origin[0] + 250.0, origin[1]]
    closing = make_obs(
        at_time(20.2),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", None]),
        with_zone(origin[:2], 80.0, nxt, 50.0, 20.0, 30.0, 35.0),
        **_quiet(mechs=[enemy]),
    )
    act = brain.act(closing, time.perf_counter())
    assert brain.mode == "ZONE"
    assert act["move"][0] > 0.0
    assert abs(_wrap(act["aim"][0] - math.pi / 2.0)) < 0.45

    closing["self"]["aim"] = list(act["aim"])
    closing["t"] = 20.3
    closing["tick"] = int(round(closing["t"] * 20))
    fired = brain.act(closing, time.perf_counter())
    assert brain.mode == "ZONE"
    assert fired["fire"][0] is True
    assert fired["move"][0] > 0.0


def test_fight_strafes_instead_of_standing_still(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    enemy = _actor(5, [origin[0] + 50.0, origin[1], origin[2]], 1400)
    obs = make_obs(
        at_time(20.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", None]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy]),
    )
    act = brain.act(obs, time.perf_counter())
    assert brain.mode == "FIGHT"
    assert math.hypot(*act["move"]) > 0.5
    assert abs(act["move"][0]) < 0.35
