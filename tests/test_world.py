"""Tracks, damage, regen, and FFA evidence."""

import time

import numpy as np

from bot.params import PARAMS
from bot.world import WorldModel
from tests.fixtures import at_time, make_obs


def _actor(ident: int, pos: list[float], kind: str = "mech") -> dict:
    return {
        "id": ident,
        "kind": kind,
        "pos": pos,
        "vel": [0.0, 0.0, 0.0],
        "aim": [0.0, 0.0],
        "hp": 1400,
        "max_hp": 1400,
    }


def test_replay_updates_within_budget(observations, rules):
    world = WorldModel(rules, PARAMS)
    times = []
    for obs in observations:
        started = time.perf_counter()
        world.update(obs)
        times.append(time.perf_counter() - started)
    assert float(np.percentile(times, 99)) < 0.002
    assert world.ffa is False
    assert world.ffa_evidence < 3


def test_track_survives_five_seconds_and_damage_blocks_regen(rules):
    world = WorldModel(rules, PARAMS)
    world.update(make_obs(at_time(10.0), mechs=[_actor(7, [40.0, 0.0, 0.0])]))
    assert [track.id for track in world.visible_enemies()] == [7]

    world.update(make_obs(at_time(16.0), mechs=[], events=[]))
    assert [track.id for track in world.enemies()] == [7]
    assert world.visible_enemies() == []

    world.update(make_obs(at_time(10.0 + 15.01), mechs=[], events=[]))
    assert world.enemies() == []

    fresh = WorldModel(rules, PARAMS)
    fresh.update(
        make_obs(
            at_time(30.0),
            events=[
                {
                    "type": "damaged",
                    "amount": 40,
                    "weapon": "gun",
                    "bearing": 1.25,
                    "by": 5,
                    "rear": True,
                }
            ],
            mechs=[],
        )
    )
    assert 5 in fresh.hostile_ids(10.0)
    assert fresh.hostiles[5].bearing == 1.25
    assert fresh.regen_ready() is False

    delay = float(rules.mech.regen_delay)
    fresh.update(make_obs(at_time(30.0 + delay - 0.1), events=[], mechs=[]))
    assert fresh.regen_ready() is False
    fresh.update(make_obs(at_time(30.0 + delay), events=[], mechs=[]))
    assert fresh.regen_ready() is True


def test_three_infighting_beams_flip_ffa(rules):
    world = WorldModel(rules, PARAMS)
    beams = [
        {
            "by": 2,
            "kind": "laser",
            "path": [[0.0, 30.0, 7.0], [80.0, 30.0, 7.0]],
            "age": 0.0,
        }
        for _ in range(3)
    ]
    world.update(
        make_obs(
            at_time(5.0),
            mechs=[_actor(2, [0.0, 30.0, 0.0]), _actor(4, [80.0, 30.0, 0.0])],
            beams=beams,
            events=[],
            projectiles=[],
        )
    )
    assert world.ffa_evidence >= 3
    assert world.ffa is True
