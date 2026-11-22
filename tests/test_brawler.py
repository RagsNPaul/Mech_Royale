"""The brawler over the recorded match: valid actions, no exceptions, under 20 ms."""

import time

from bot.brain import Brain
from bot.params import PARAMS
from tests.fixtures import at_time, make_obs, with_self
from tests.test_replay import _validator
from tests.test_scenarios import _actor, _origin, _quiet, _safe_zone


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


def test_clustered_enemy_is_preferred(hello):
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


def test_fight_does_not_boost(hello):
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


def test_strafe_reversals_are_not_evenly_spaced(hello):
    brain = Brain(hello)
    origin = _origin(brain)
    hold = 12.0
    pos = list(origin)
    enemy = [pos[0] + hold, pos[1], pos[2]]
    flips = []
    last_side = None
    step = 0.05
    for index in range(int(10.0 / step)):
        now = 50.0 + index * step
        obs = make_obs(
            at_time(now),
            with_self(hp=1400, pos=pos, in_zone=True, aim=[0.0, 0.0], weapons=["gun", "lightning"]),
            _safe_zone(pos),
            **_quiet(mechs=[_actor(5, enemy, 1400)]),
        )
        act = brain.act(obs, time.perf_counter())
        assert abs(act["aim"][0]) < 0.35
        side = 1 if act["move"][1] >= 0.0 else -1
        if last_side is not None and side != last_side:
            flips.append(now)
        last_side = side
    gaps = [b - a for a, b in zip(flips, flips[1:])]
    assert len(gaps) >= 8
    assert max(gaps) - min(gaps) > 0.12
