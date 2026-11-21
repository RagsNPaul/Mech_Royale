"""Entrypoint. Thin glue: wires hello -> Brain, never crashes, never answers late.

Safety net (keep it):
- Any exception inside the brain returns `fallback_action` (walk toward next zone center).
- Logs go to stderr only. stdout is protocol-only (mechfight.py handles it).
"""
import math
import sys
import time
import traceback

import mechfight as mf
from bot.brain import Brain


def fallback_action(obs: mf.Observation) -> mf.Action:
    """Minimal safe action: head for the next zone center, boost if outside the zone."""
    x, y, _ = obs["self"]["pos"]
    cx, cy = obs["zone"]["next_center"]
    yaw = math.atan2(cy - y, cx - x)
    act: mf.Action = {"tick": obs["tick"], "move": [math.cos(yaw), math.sin(yaw)],
                      "aim": [yaw, 0.0], "boost": not obs["self"]["in_zone"]}
    if obs.get("offer"):
        act["pick"] = 0
    return act


class Bot:
    name = "rags-mech"

    def __init__(self, hello: mf.Hello):
        self.brain = None
        try:
            self.brain = Brain(hello)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    def act(self, obs: mf.Observation) -> mf.Action:
        t0 = time.perf_counter()
        try:
            if self.brain is None:
                return fallback_action(obs)
            act = self.brain.act(obs, t0)
            act["tick"] = obs["tick"]
            return act
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return fallback_action(obs)


if __name__ == "__main__":
    mf.run(Bot)
