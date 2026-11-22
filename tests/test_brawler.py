"""The brawler over the recorded match: valid actions, no exceptions, under 20 ms."""

import math
import time

import pytest

from bot.brain import Brain
from bot.params import PARAMS
from tests.fixtures import at_time, make_obs, with_self, with_zone
from tests.test_replay import _validator
from tests.test_scenarios import _actor, _los_point, _origin, _quiet, _safe_zone


@pytest.fixture(autouse=True)
def _brawler_strategy(monkeypatch):
    monkeypatch.setitem(PARAMS, "strategy", "brawler")


def test_brawler_replay_is_fast_and_schema_valid(hello, observations, monkeypatch):
    monkeypatch.setitem(PARAMS, "strategy", "brawler")
    brain = Brain(hello)
    validator = _validator()
    worst = 0.0
    for obs in observations:
        started = time.perf_counter()
        action = brain.act(obs, started)
        worst = max(worst, time.perf_counter() - started)
        validator.validate(action)
    assert worst < 0.020


def test_lightning_is_taken_over_a_closer_laser(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    laser = [origin[0] + 8.0, origin[1], origin[2]]
    lightning = [origin[0] - 40.0, origin[1], origin[2]]
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "laser"]),
        _safe_zone(origin),
        **_quiet(
            pickups=[
                {"id": 8011, "kind": "laser", "pos": laser},
                {"id": 8013, "kind": "lightning", "pos": lightning},
            ]
        ),
    )
    act = brain.act(obs, time.perf_counter())
    assert brain.mode == "BRAWL"
    assert act["move"][0] < 0.0
    obs["self"]["pos"] = lightning
    obs["t"] = 50.1
    obs["tick"] = int(round(obs["t"] * 20))
    taken = brain.act(obs, time.perf_counter())
    assert taken["take"] == 8013
    assert taken["drop"] == 1


def test_clustered_enemy_is_preferred(hello, monkeypatch):
    monkeypatch.setitem(PARAMS, "standoff_lightning_m", 20.0)
    monkeypatch.setitem(PARAMS, "standoff_gun_m", 20.0)
    brain = Brain(hello)
    origin = _origin(brain)
    isolated = _actor(5, [origin[0] + 50.0, origin[1] + 40.0, origin[2]], 100)
    low = _actor(6, [origin[0] + 70.0, origin[1], origin[2]], 500)
    high = _actor(7, [origin[0] + 80.0, origin[1], origin[2]], 1400)
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[isolated, low, high]),
    )
    act = brain.act(obs, time.perf_counter())
    toward_low = (70.0, 0.0)
    toward_isolated = (50.0, 40.0)
    low_dot = act["move"][0] * toward_low[0] + act["move"][1] * toward_low[1]
    isolated_dot = act["move"][0] * toward_isolated[0] + act["move"][1] * toward_isolated[1]
    assert low_dot > isolated_dot


def test_lightning_fires_at_20m_inside_the_cone(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    enemy = _actor(5, [origin[0] + 20.0, origin[1], origin[2]], 1400)
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy]),
    )
    aimed = brain.act(obs, time.perf_counter())
    obs["self"]["aim"] = list(aimed["aim"])
    obs["t"] = 50.1
    obs["tick"] = int(round(obs["t"] * 20))
    fired = brain.act(obs, time.perf_counter())
    assert fired["fire"][0] is True
    assert fired["fire"][1] is True


def test_fight_does_not_boost(hello, monkeypatch):
    """Inside 60 m is not enough. Boost only while backing out from standoff - 20."""
    monkeypatch.setitem(PARAMS, "standoff_lightning_m", 50.0)
    monkeypatch.setitem(PARAMS, "standoff_gun_m", 50.0)
    brain = Brain(hello)
    origin = _origin(brain)
    enemy = _actor(5, [origin[0] + 30.0, origin[1], origin[2]], 1400)
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy]),
    )
    obs["self"]["energy"] = 100.0
    act = brain.act(obs, time.perf_counter())
    assert act["boost"] is False


def test_low_energy_holds_lightning_and_keeps_the_gun(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    enemy = _actor(5, [origin[0] + 35.0, origin[1], origin[2]], 1400)
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy]),
    )
    obs["self"]["energy"] = 12.0
    aimed = brain.act(obs, time.perf_counter())
    obs["self"]["aim"] = list(aimed["aim"])
    obs["self"]["energy"] = 12.0
    obs["t"] = 50.1
    obs["tick"] = int(round(obs["t"] * 20))
    fired = brain.act(obs, time.perf_counter())
    assert fired["fire"][0] is True
    assert fired["fire"][1] is False


def test_walks_to_a_cube_when_nobody_is_close(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    cube = [origin[0] + 30.0, origin[1], origin[2]]
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(cubes=[{"id": 5012, "pos": cube}]),
    )
    act = brain.act(obs, time.perf_counter())
    assert act["move"][0] > 0.5


def test_kill_sets_the_goal_on_the_victim(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    victim = _actor(5, [origin[0] + 50.0, origin[1], origin[2]], 100)
    seen = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[victim]),
    )
    brain.act(seen, time.perf_counter())
    obs = make_obs(
        at_time(50.2),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(events=[{"type": "kill", "target": 5}]),
    )
    act = brain.act(obs, time.perf_counter())
    assert act["move"][0] > 0.5


def test_steps_onto_a_close_cube_while_aiming_at_the_enemy(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    enemy = _actor(5, [origin[0] + 20.0, origin[1], origin[2]], 1400)
    cube = [origin[0], origin[1] + 8.0, origin[2]]
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy], cubes=[{"id": 5013, "pos": cube}]),
    )
    act = brain.act(obs, time.perf_counter())
    assert act["move"][1] > 0.5
    assert abs(act["aim"][0]) < 0.4


def test_enemy_at_70m_is_rushed_instead_of_a_cube(hello, monkeypatch):
    monkeypatch.setitem(PARAMS, "standoff_lightning_m", 20.0)
    monkeypatch.setitem(PARAMS, "standoff_gun_m", 20.0)
    brain = Brain(hello)
    origin = _origin(brain)
    enemy = _actor(5, [origin[0] + 70.0, origin[1], origin[2]], 1400)
    cube = [origin[0], origin[1] + 30.0, origin[2]]
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy], cubes=[{"id": 5014, "pos": cube}]),
    )
    act = brain.act(obs, time.perf_counter())
    assert act["move"][0] > 0.5
    assert act["move"][0] > abs(act["move"][1])


def test_far_enemy_is_aimed_while_walking_to_a_cube(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    enemy = _actor(5, [origin[0], origin[1] + 150.0, origin[2]], 1400)
    cube = [origin[0] + 30.0, origin[1], origin[2]]
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy], cubes=[{"id": 5015, "pos": cube}]),
    )
    aimed = brain.act(obs, time.perf_counter())
    assert aimed["move"][0] > 0.5
    assert abs(aimed["aim"][0] - math.pi / 2.0) < 0.4
    obs["self"]["aim"] = list(aimed["aim"])
    obs["t"] = 50.1
    obs["tick"] = int(round(obs["t"] * 20))
    fired = brain.act(obs, time.perf_counter())
    assert fired["move"][0] > 0.5
    assert fired["fire"][0] is True


def test_too_close_backs_away_with_boost(hello, monkeypatch):
    monkeypatch.setitem(PARAMS, "standoff_gun_m", 100.0)
    monkeypatch.setitem(PARAMS, "standoff_lightning_m", 100.0)
    brain = Brain(hello)
    origin = _origin(brain)
    enemy = _actor(5, [origin[0] + 50.0, origin[1], origin[2]], 1400)
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy]),
    )
    obs["self"]["energy"] = 80.0
    act = brain.act(obs, time.perf_counter())
    assert act["move"][0] < -0.3
    assert act["boost"] is True
    assert abs(act["aim"][0]) < 0.35


def test_lightning_hunt_skips_once_an_enemy_is_known(hello, monkeypatch):
    monkeypatch.setitem(PARAMS, "standoff_lightning_m", 40.0)
    monkeypatch.setitem(PARAMS, "standoff_gun_m", 40.0)
    brain = Brain(hello)
    origin = _origin(brain)
    pickup = [origin[0] - 40.0, origin[1], origin[2]]
    enemy = _actor(5, [origin[0] + 120.0, origin[1], origin[2]], 1400)
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "laser"]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy], pickups=[{"id": 8021, "kind": "lightning", "pos": pickup}]),
    )
    act = brain.act(obs, time.perf_counter())
    assert act["move"][0] > 0.3


def test_standoff_distance_comes_from_params(hello, monkeypatch):
    monkeypatch.setitem(PARAMS, "standoff_gun_m", 20.0)
    monkeypatch.setitem(PARAMS, "standoff_lightning_m", 40.0)
    brain = Brain(hello)
    origin = _origin(brain)
    enemy = _actor(5, [origin[0] + 20.0, origin[1], origin[2]], 1400)
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy]),
    )
    act = brain.act(obs, time.perf_counter())
    assert act["move"][0] < -0.5


def test_low_hp_detours_for_a_cube_when_heal_frac_says_so(hello, monkeypatch):
    monkeypatch.setitem(PARAMS, "heal_hp_frac", 0.75)
    brain = Brain(hello)
    origin = _origin(brain)
    enemy = _actor(5, [origin[0] + 70.0, origin[1], origin[2]], 1400)
    cube = [origin[0], origin[1] + 30.0, origin[2]]
    obs = make_obs(
        at_time(50.0),
        with_self(hp=200, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy], cubes=[{"id": 5016, "pos": cube}]),
    )
    obs["self"]["max_hp"] = 1000.0
    act = brain.act(obs, time.perf_counter())
    assert act["move"][1] > 0.5
    assert abs(act["move"][1]) > abs(act["move"][0])


def _debug(act) -> str:
    return str((act.get("debug") or {}).get("label") or "")


def _npc_at(brain, origin, dist: float, ident: int = 101) -> dict:
    npc = _actor(ident, _los_point(brain, origin, dist), 350)
    npc["kind"] = "npc"
    npc["max_hp"] = 350
    return npc


def _dot(move, origin, pos) -> float:
    dx = float(pos[0]) - float(origin[0])
    dy = float(pos[1]) - float(origin[1])
    return float(move[0]) * dx + float(move[1]) * dy


def test_npc_inside_aggro_is_killed_with_both_weapons(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    npc = _npc_at(brain, origin, 30.0)
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[npc]),
    )
    obs["self"]["energy"] = 100.0
    aimed = brain.act(obs, time.perf_counter())
    assert "| npc" in _debug(aimed)
    assert _dot(aimed["move"], origin, npc["pos"]) > 0.0
    obs["self"]["aim"] = list(aimed["aim"])
    obs["t"] = 50.1
    obs["tick"] = int(round(obs["t"] * 20))
    fired = brain.act(obs, time.perf_counter())
    assert fired["fire"][0] is True
    assert fired["fire"][1] is True


def test_npc_that_shot_us_is_chased_out_to_the_leash(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    npc = _npc_at(brain, origin, 55.0)
    quiet = _quiet(mechs=[npc])
    ignored = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **quiet,
    )
    assert "| npc" not in _debug(brain.act(ignored, time.perf_counter()))
    shot = make_obs(
        at_time(50.2),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(
            mechs=[npc],
            events=[{"type": "damaged", "weapon": "npc", "amount": 25, "by": 101}],
        ),
    )
    act = brain.act(shot, time.perf_counter())
    assert "| npc" in _debug(act)
    assert _dot(act["move"], origin, npc["pos"]) > 0.0


def test_npc_cubes_are_collected_after_the_kill(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    npc = _npc_at(brain, origin, 30.0)
    seen = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[npc]),
    )
    brain.act(seen, time.perf_counter())
    killed = make_obs(
        at_time(50.2),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(events=[{"type": "kill", "target": 101}]),
    )
    act = brain.act(killed, time.perf_counter())
    assert _dot(act["move"], origin, npc["pos"]) > 0.0


def test_engaged_mech_steers_around_a_drone(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    mech_pos = _los_point(brain, origin, 70.0)
    dx = mech_pos[0] - origin[0]
    dy = mech_pos[1] - origin[1]
    npc_pos = [origin[0] + dx * (25.0 / 70.0), origin[1] + dy * (25.0 / 70.0), origin[2]]
    npc = _actor(101, npc_pos, 350)
    npc["kind"] = "npc"
    npc["max_hp"] = 350
    mech = _actor(5, mech_pos, 1400)
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[mech, npc]),
    )
    act = brain.act(obs, time.perf_counter())
    assert "| rush" in _debug(act)
    assert _dot(act["move"], origin, npc_pos) < 0.0


def test_zone_starts_within_15s_and_keeps_firing(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    enemy = _actor(5, _los_point(brain, origin, 25.0), 1400)
    soon = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        with_zone(origin[:2], 40.0, [origin[0] + 180.0, origin[1]], 20.0, 50.0, 110.0, 20.0),
        **_quiet(mechs=[enemy]),
    )
    soon["self"]["energy"] = 100.0
    aimed = brain.act(soon, time.perf_counter())
    assert "| zone" in _debug(aimed)
    soon["self"]["aim"] = list(aimed["aim"])
    soon["t"] = 50.1
    soon["tick"] = int(round(soon["t"] * 20))
    fired = brain.act(soon, time.perf_counter())
    assert fired["fire"][0] is True
    assert fired["fire"][1] is True
    later = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        with_zone(origin[:2], 40.0, [origin[0] + 100.0, origin[1]], 20.0, 50.0, 110.0, 20.0),
        **_quiet(mechs=[enemy]),
    )
    assert "| rush" in _debug(brain.act(later, time.perf_counter()))


def test_stuck_motion_jumps_strafes_then_replans(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    enemy_pos = _los_point(brain, origin, 70.0)
    enemy = _actor(5, enemy_pos, 1400)
    obs = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(mechs=[enemy]),
    )
    first = brain.act(obs, time.perf_counter())
    heading = math.atan2(first["move"][1], first["move"][0])
    brain.nav.path = [(origin[0], origin[1])]
    brain.nav.goal = (origin[0] + 10.0, origin[1])
    obs["t"] = 53.0
    obs["tick"] = int(round(obs["t"] * 20))
    stuck = brain.act(obs, time.perf_counter())
    yaw = math.atan2(stuck["move"][1], stuck["move"][0])
    assert stuck["jump"] is True
    assert abs(math.atan2(math.sin(yaw - (heading + math.pi / 2.0)), math.cos(yaw - (heading + math.pi / 2.0)))) < 0.4
    assert brain.nav.path
    obs["t"] = 54.0
    obs["tick"] = int(round(obs["t"] * 20))
    brain.act(obs, time.perf_counter())
    assert brain.nav.path == []
    assert brain.nav.goal is None


def test_bearing_collects_a_cube_and_shoots_a_crate(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    cube = [origin[0] + 20.0, origin[1] + 15.0, origin[2]]
    sound = [{"bearing": 0.0, "dist": "near", "kind": "gun", "age": 0.0}]
    to_cube = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(sounds=sound, cubes=[{"id": 5017, "pos": cube}]),
    )
    act = brain.act(to_cube, time.perf_counter())
    assert "| bearing" in _debug(act)
    assert act["move"][1] > 0.25
    crate_brain = Brain(hello)
    crate = [origin[0] + 18.0, origin[1] + 10.0, origin[2]]
    to_crate = make_obs(
        at_time(50.0),
        with_self(hp=1400, pos=origin, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
        _safe_zone(origin),
        **_quiet(sounds=sound, crates=[{"id": 3011, "pos": crate, "hp": 100}]),
    )
    aimed = crate_brain.act(to_crate, time.perf_counter())
    assert "| bearing" in _debug(aimed)
    assert aimed["aim"][0] > 0.25
    to_crate["self"]["aim"] = list(aimed["aim"])
    to_crate["t"] = 50.1
    to_crate["tick"] = int(round(to_crate["t"] * 20))
    fired = crate_brain.act(to_crate, time.perf_counter())
    assert fired["fire"][0] is True
