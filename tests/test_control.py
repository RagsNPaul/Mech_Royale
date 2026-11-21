"""Intent to action: lead, mortar arc, fire discipline, quiet movement, schema."""

import json
import math
from pathlib import Path

import jsonschema
import pytest

from bot.control import Controller, Intent, lead_aim, mortar_pitch
from bot.params import PARAMS

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "mechfight.schema.json"


def _bot_validator():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    wrapped = {
        "$schema": schema["$schema"],
        "definitions": schema["definitions"],
        "$ref": "#/definitions/BotMessage",
    }
    return jsonschema.Draft7Validator(wrapped)


VALIDATOR = _bot_validator()


def _check(action):
    VALIDATOR.validate(action)


def _obs(aim, ammo=12, cooldown=0.0, walk=11.7):
    return {
        "tick": 4,
        "t": 2.0,
        "self": {
            "pos": [0.0, 0.0, 0.0],
            "aim": [float(aim[0]), float(aim[1])],
            "hp": 900,
            "weapons": [
                {
                    "kind": "gun",
                    "cooldown": cooldown,
                    "ammo": ammo,
                    "reload": 0,
                    "stats": {"speed": 252, "mag": 12, "range": 230},
                },
                None,
            ],
            "stats": {"walk_speed": walk, "muzzle_height": 7},
        },
    }


def test_lead_aim_stationary_and_crossing_target():
    muzzle = (0.0, 0.0, 7.0)
    target = (80.0, 30.0, 1.0)
    yaw, pitch = lead_aim(muzzle, target, (0.0, 0.0, 0.0), 252.0)
    dx, dy = target[0] - muzzle[0], target[1] - muzzle[1]
    dz = (target[2] + 4.5) - muzzle[2]
    assert yaw == pytest.approx(math.atan2(dy, dx))
    assert pitch == pytest.approx(math.atan2(dz, math.hypot(dx, dy)))

    muzzle = (0.0, 0.0, 0.0)
    target = (100.0, 0.0, 0.0)
    yaw, _pitch = lead_aim(muzzle, target, (0.0, 10.0, 0.0), 252.0)
    lead = abs(math.atan2(math.sin(yaw), math.cos(yaw)))
    assert lead == pytest.approx(math.atan(10.0 / 252.0), rel=0.1)


def test_mortar_pitch_lands_a_flat_150m_shot(rules):
    mortar = rules.weapons["mortar"]
    speed = float(mortar.speed)
    gravity = float(mortar.gravity)
    pitch = mortar_pitch(150.0, 0.0, speed, gravity)
    assert pitch is not None
    vx = speed * math.cos(pitch)
    vz = speed * math.sin(pitch)
    x = z = 0.0
    dt = 0.002
    for _ in range(20000):
        vz -= gravity * dt
        x += vx * dt
        z += vz * dt
        if z <= 0.0 and x > 1.0:
            break
    assert abs(x - 150.0) < 3.0
    assert mortar_pitch(10_000.0, 0.0, speed, gravity) is None


def test_fire_only_when_aligned_and_ready(rules):
    control = Controller(rules, PARAMS, seed=1)
    target = {"pos": [100.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0]}
    intent = Intent(mode="FIGHT", reason="favorable", aim_target=target, fire=(True, False))
    blocked = control.to_action(intent, _obs((math.pi, 0.0)), None, 0.0)
    assert blocked["fire"] == [False, False]
    _check(blocked)

    aligned = _obs(blocked["aim"])
    shot = control.to_action(intent, aligned, None, 0.0)
    assert shot["fire"] == [True, False]
    _check(shot)

    cooling = _obs(blocked["aim"], cooldown=0.2)
    held = control.to_action(intent, cooling, None, 0.0)
    assert held["fire"] == [True, False]
    _check(held)

    reloading = _obs(blocked["aim"], cooldown=0.0)
    reloading["self"]["weapons"][0]["reload"] = 1.2
    waiting = control.to_action(intent, reloading, None, 0.0)
    assert waiting["fire"] == [False, False]
    _check(waiting)


def test_slow_move_stays_under_step_speed(rules):
    control = Controller(rules, PARAMS, seed=1)
    intent = Intent(mode="HIDE", reason="cover", move_vec=(1.0, 0.0), slow=True)
    obs = _obs((0.0, 0.0), walk=float(rules.mech.walk_speed))
    action = control.to_action(intent, obs, None, 0.0)
    speed = math.hypot(*action["move"]) * float(rules.mech.walk_speed)
    assert speed < float(rules.sound.step_speed)
    assert speed == pytest.approx(float(rules.sound.step_speed) * 0.9)
    assert action["debug"]["label"] == "HIDE | 900 | cover"
    _check(action)


def test_actions_match_the_bot_message_schema(rules):
    control = Controller(rules, PARAMS, seed=7)
    obs = _obs((0.0, 0.0), ammo=2)
    intents = [
        Intent(mode="ZONE", reason="late", goal_xy=(40.0, 10.0), boost=True),
        Intent(mode="FIGHT", reason="favorable", aim_target={"pos": [20.0, 5.0, 0.0], "vel": [1.0, 0.0, 0.0]}, fire=(True, False), strafe=True),
        Intent(mode="HIDE", reason="quiet", move_vec=(1.0, 0.0), slow=True),
        Intent(mode="LOOT", reason="gun", take=8013, drop=1),
    ]
    for intent in intents:
        _check(control.to_action(intent, obs, None, 0.0))
    quiet = control.to_action(Intent(mode="HIDE", reason="reload"), obs, None, 0.0)
    assert quiet["reload"] is True
    _check(quiet)
