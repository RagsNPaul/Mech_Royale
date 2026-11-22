"""Replay the recorded match through the brain."""

import json
import time
from pathlib import Path

import jsonschema
import numpy as np

from bot.brain import Brain
from bot.params import PARAMS

ROOT = Path(__file__).resolve().parents[1]


def _validator():
    schema = json.loads((ROOT / "mechfight.schema.json").read_text(encoding="utf-8"))
    return jsonschema.Draft7Validator(
        {
            "$schema": schema["$schema"],
            "definitions": schema["definitions"],
            "$ref": "#/definitions/BotMessage",
        }
    )


def test_replay_stays_inside_the_budget(hello, observations, monkeypatch):
    monkeypatch.setitem(PARAMS, "strategy", "roof")
    brain = Brain(hello)
    validator = _validator()
    times = []
    for obs in observations:
        started = time.perf_counter()
        action = brain.act(obs, started)
        times.append(time.perf_counter() - started)
        validator.validate(action)
    assert float(np.percentile(times, 50)) < 0.005
    assert float(np.percentile(times, 99)) < 0.015


def test_replay_max_decision_under_20ms(hello, observations, monkeypatch):
    monkeypatch.setitem(PARAMS, "strategy", "roof")
    brain = Brain(hello)
    worst = 0.0
    for obs in observations:
        started = time.perf_counter()
        brain.act(obs, started)
        worst = max(worst, time.perf_counter() - started)
    assert worst < 0.020
