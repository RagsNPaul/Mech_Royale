"""Rules reads hello.config and falls back to protocol defaults."""

import copy

from bot.config import Rules
from tests.fixtures import load_stream


def test_rules_match_real_hello(hello):
    rules = Rules(hello["config"])
    cfg = hello["config"]

    assert rules.raw is cfg
    assert rules.weapons["gun"].damage == 45
    assert rules.weapons["gun"].cooldown == 0.24
    assert rules.weapons["gun"].speed == 252
    assert rules.mech.rear_mult == 1.5
    assert rules.mech.front_arc_deg == 180

    for key, value in cfg["mech"].items():
        assert getattr(rules.mech, key) == value
    for kind, stats in cfg["weapons"].items():
        for key, value in stats.items():
            assert getattr(rules.weapons[kind], key) == value
    for key, value in cfg["vision"].items():
        assert getattr(rules.vision, key) == value
    for key, value in cfg["sound"].items():
        if key == "range":
            assert rules.sound.range == value
        else:
            assert getattr(rules.sound, key) == value
    assert rules.zone.start_center == tuple(cfg["zone"]["start_center"])
    assert rules.zone.start_radius == cfg["zone"]["start_radius"]
    assert len(rules.zone.phases) == len(cfg["zone"]["phases"])
    for got, src in zip(rules.zone.phases, cfg["zone"]["phases"]):
        assert (got.hold_s, got.shrink_s, got.radius, got.dps) == (
            src["hold_s"],
            src["shrink_s"],
            src["radius"],
            src["dps"],
        )
    for key, value in cfg["cubes"].items():
        got = getattr(rules.cubes, key)
        if key == "upgrade_thresholds":
            assert tuple(got) == tuple(value)
        else:
            assert got == value
    assert rules.loot.pickup_radius == cfg["loot"]["pickup_radius"]
    assert rules.drops.radius == cfg["drops"]["radius"]
    assert rules.drops.capture_s == cfg["drops"]["capture_s"]
    assert tuple(rules.drops.times) == tuple(cfg["drops"]["times"])
    for key, value in cfg["npc"].items():
        assert getattr(rules.npc, key) == value

    changed = copy.deepcopy(cfg)
    changed["weapons"]["gun"]["damage"] = 99
    assert Rules(changed).weapons["gun"].damage == 99


def test_empty_config_uses_defaults_and_logs_once(capsys):
    import bot.config as config_mod

    config_mod._logged.clear()
    rules = Rules({})
    assert rules.mech.rear_mult == 1.5
    assert rules.mech.front_arc_deg == 180
    assert rules.mech.radius == 2.5
    assert rules.mech.height == 9
    assert rules.mech.regen_delay == 5
    assert rules.mech.regen_per_s == 45
    assert rules.weapons["gun"].damage == 45
    assert rules.weapons["gun"].cooldown == 0.24
    assert rules.weapons["gun"].speed == 252
    assert rules.weapons["laser"].charge == 0.75
    assert rules.weapons["mortar"].gravity == 22
    assert rules.weapons["lightning"].cone_deg == 30
    assert rules.sound.step_speed == 5
    assert rules.sound.range["gun"] == 320
    assert rules.cubes.stat_cap == 10
    assert rules.cubes.damage_per_cube == 0.04
    assert rules.loot.pickup_radius == 4
    assert rules.npc.leash == 30
    assert rules.drops.capture_s == 4
    assert rules.zone.start_radius == 560

    Rules({})
    err = capsys.readouterr().err
    assert "rules: missing mech.rear_mult" in err
    assert err.count("mech.rear_mult") == 1


def test_partial_config_fills_only_the_missing_key():
    hello, _obs = load_stream()
    cfg = copy.deepcopy(hello["config"])
    del cfg["mech"]["rear_mult"]
    rules = Rules(cfg)
    assert rules.mech.rear_mult == 1.5
    assert rules.mech.walk_speed == hello["config"]["mech"]["walk_speed"]
