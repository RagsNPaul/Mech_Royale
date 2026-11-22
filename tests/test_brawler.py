"""The brawler over the recorded match: valid actions, no exceptions, under 20 ms."""

import time

from bot.brain import Brain
from bot.params import PARAMS
from tests.test_replay import _validator


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
