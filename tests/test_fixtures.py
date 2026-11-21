"""Sanity checks for the recorded stream and synthetic observations."""

from bot.config import Rules
from tests.fixtures import (
    at_time,
    load_stream,
    make_obs,
    server_message_validator,
    with_damage_event,
    with_enemy,
    with_self,
    with_zone,
)


def test_stream_loads_300_observations():
    hello, observations = load_stream()
    assert hello["type"] == "hello"
    assert hello["protocol"] == "mechfight/1"
    assert len(observations) == 300
    assert all(obs["type"] == "obs" for obs in observations)


def test_rules_fixture_uses_hello_config(hello, rules):
    assert isinstance(rules, Rules)
    assert rules.raw == hello["config"]
    assert rules.raw["weapons"]["gun"]["damage"] == 45


def test_recorded_messages_match_server_schema(hello, observations):
    validator = server_message_validator()
    validator.validate(hello)
    for obs in observations:
        validator.validate(obs)


def test_make_obs_stays_schema_valid_and_applies_helpers(observations):
    original_mechs = len(observations[0]["mechs"])
    original_hp = observations[0]["self"]["hp"]
    obs = make_obs(
        with_self(
            hp=400,
            pos=[1, 2, 3],
            aim=[0.4, -0.2],
            weapons=["gun", None],
            cubes=3,
            in_zone=False,
        ),
        with_enemy(7, [50, 0, 1], 150, ["lightning"], [1.0, 0.0], 2),
        with_damage_event(7, 40, 1.2, True),
        with_zone([0, 0], 100, [5, -5], 80, 10, 40, 12),
        at_time(12.5),
    )

    server_message_validator().validate(obs)
    assert obs["self"]["hp"] == 400
    assert obs["self"]["pos"] == [1.0, 2.0, 3.0]
    assert obs["self"]["aim"] == [0.4, -0.2]
    assert obs["self"]["weapons"][0]["kind"] == "gun"
    assert obs["self"]["weapons"][1] is None
    assert obs["self"]["cubes"] == 3
    assert obs["self"]["in_zone"] is False
    assert obs["mechs"][-1]["id"] == 7
    assert obs["mechs"][-1]["hp"] == 150
    assert obs["mechs"][-1]["weapons"] == ["lightning"]
    assert obs["events"][-1]["type"] == "damaged"
    assert obs["events"][-1]["rear"] is True
    assert obs["zone"]["next_radius"] == 80
    assert obs["t"] == 12.5
    assert obs["tick"] == 250
    assert observations[0]["self"]["hp"] == original_hp
    assert len(observations[0]["mechs"]) == original_mechs
