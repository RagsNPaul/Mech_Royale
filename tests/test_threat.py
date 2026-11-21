"""Unit tests for bot/threat.py (pure math, no sim needed)."""

import math

import pytest

from bot.params import PARAMS
from bot.threat import incoming_dps, pick_target, risk_margin, should_fight, triangle, weapon_dps


class _World:
    def __init__(self, actors, grid=None):
        self._actors = list(actors)
        self.map = grid
        self.ffa = False

    def enemies(self, max_dist=None):
        del max_dist
        return list(self._actors)

    def visible_enemies(self):
        return list(self._actors)


class _BlockedX:
    """Line of sight fails for an eye whose x is within 1 m of `blocked`."""

    def __init__(self, blocked):
        self.blocked = blocked

    def los(self, a, b, for_vision=False):
        del a, for_vision
        return abs(b[0] - self.blocked) > 1.0


def _me(**overrides):
    body = {
        "id": 1,
        "pos": [0.0, 0.0, 0.0],
        "hp": 1400.0,
        "max_hp": 1400.0,
        "aim": [0.0, 0.0],
        "weapons": ["gun"],
        "cubes": 0,
        "in_zone": True,
    }
    body.update(overrides)
    return body


def _enemy(ident, pos, hp=1400.0):
    return {
        "id": ident,
        "pos": pos,
        "hp": hp,
        "max_hp": 1400.0,
        "aim": [0.0, 0.0],
        "weapons": ["gun"],
        "cubes": 0,
    }


def _zone(safe):
    if safe:
        center, radius, nxt, next_radius = [0.0, 0.0], 500.0, [0.0, 0.0], 400.0
    else:
        center, radius, nxt, next_radius = [0.0, 0.0], 5.0, [0.0, 0.0], 1.0
    return {
        "t": 0.0,
        "zone": {
            "center": center,
            "radius": radius,
            "next_center": nxt,
            "next_radius": next_radius,
            "shrink_start": 1000.0,
            "shrink_end": 1010.0,
            "dps": 20.0,
        },
    }


def test_point_blank_gun_dps_includes_reload(rules):
    gun = rules.weapons["gun"]
    assert gun.damage == 45
    assert gun.cooldown == 0.24
    burst = gun.damage / gun.cooldown * PARAMS["p_hit_gun_near"]
    period = gun.cooldown + gun.reload / gun.mag
    sustained = gun.damage / period * PARAMS["p_hit_gun_near"]
    got = weapon_dps("gun", 0.0, 0, rules, PARAMS)
    assert got == pytest.approx(sustained)
    assert got < burst


def test_lightning_dps_is_zero_beyond_its_range(rules):
    reach = rules.weapons["lightning"].range
    assert reach == 45
    assert weapon_dps("lightning", reach, 0, rules, PARAMS) > 0
    assert weapon_dps("lightning", reach + 0.1, 0, rules, PARAMS) == 0


def test_triangle_matchups():
    assert triangle("laser", "mortar") == 1.25
    assert triangle("mortar", "laser") == 0.8
    assert triangle("gun", "laser") == 1.0
    assert triangle("lightning", "lightning") == 1.0


def test_back_turned_incoming_is_rear_mult(rules):
    enemy = _enemy(2, [30.0, 0.0, 0.0])
    facing = incoming_dps(_me(aim=[0.0, 0.0]), enemy, rules, PARAMS)
    turned = incoming_dps(_me(aim=[math.pi, 0.0]), enemy, rules, PARAMS)
    assert facing > 0
    assert turned / facing == pytest.approx(rules.mech.rear_mult)


def test_cubes_scale_damage_up_to_the_cap(rules):
    base = weapon_dps("gun", 0.0, 0, rules, PARAMS)
    ten = weapon_dps("gun", 0.0, 10, rules, PARAMS)
    fifteen = weapon_dps("gun", 0.0, 15, rules, PARAMS)
    assert ten / base == pytest.approx(1.0 + 10.0 * rules.cubes.damage_per_cube)
    assert fifteen == pytest.approx(ten)


def test_finish_a_lone_low_hp_target(rules):
    me = _me()
    target = _enemy(2, [20.0, 0.0, 0.0], hp=200.0)
    fight, reason = should_fight(me, target, _World([target]), _zone(True), rules, PARAMS)
    assert fight is True
    assert reason == "finish"


def test_low_hp_does_not_fight_two_full_enemies(rules):
    me = _me(hp=0.30 * 1400.0)
    first = _enemy(2, [20.0, 0.0, 0.0])
    second = _enemy(3, [0.0, 20.0, 0.0])
    fight, reason = should_fight(me, first, _World([first, second]), _zone(True), rules, PARAMS)
    assert (fight, reason) == (False, "low_hp")


def test_no_fight_without_line_of_sight(rules):
    me = _me()
    target = _enemy(2, [20.0, 0.0, 0.0], hp=200.0)
    world = _World([target], _BlockedX(20.0))
    assert should_fight(me, target, world, _zone(True), rules, PARAMS) == (False, "no_los")


def test_zone_unsafe_blocks_a_winning_fight(rules):
    me = _me(in_zone=False)
    target = _enemy(2, [20.0, 0.0, 0.0], hp=400.0)
    fight, reason = should_fight(me, target, _World([target]), _zone(False), rules, PARAMS)
    assert (fight, reason) == (False, "zone")


def test_risk_margin_is_the_base_until_v2(rules):
    me = _me()
    target = _enemy(2, [20.0, 0.0, 0.0], hp=200.0)
    assert risk_margin(me, target, _World([target]), _zone(True), rules, PARAMS) == PARAMS["base_margin"]


def test_pick_target_prefers_lower_ttk_then_lower_hp(rules):
    me = _me()
    close_low = _enemy(2, [20.0, 0.0, 0.0], hp=300.0)
    close_high = _enemy(3, [0.0, 20.0, 0.0], hp=1400.0)
    hidden = _enemy(4, [10.0, 0.0, 0.0], hp=1.0)
    world = _World([hidden, close_high, close_low], _BlockedX(10.0))
    assert pick_target(me, world, rules, PARAMS) is close_low
