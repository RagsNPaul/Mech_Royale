"""Upgrade picks follow PARAMS['upgrade_priority']."""

from bot.params import PARAMS
from bot.upgrades import choose


def _offer(*ids: str) -> dict:
    return {"id": 1, "expires": 30.0, "choices": [{"id": ident} for ident in ids]}


def _me(*kinds: str) -> dict:
    return {"weapons": list(kinds)}


def test_lightning_chain_beats_armor_and_sensors(rules):
    picked = choose(_offer("armor", "lightning_chain", "sensors"), _me("gun"), rules, PARAMS)
    assert picked == 1


def test_held_weapon_rate_beats_armor_and_sensors(rules):
    picked = choose(_offer("sensors", "armor", "gun_rate"), _me("gun"), rules, {})
    assert picked == 2


def test_weapon_we_do_not_hold_loses_to_armor(rules):
    picked = choose(_offer("laser_damage", "sensors", "armor"), _me("gun"), rules, {})
    assert picked == 2


def test_max_hp_beats_speed_and_sensors(rules):
    picked = choose(_offer("sensors", "servos", "plating"), _me("gun"), rules, {})
    assert picked == 2
