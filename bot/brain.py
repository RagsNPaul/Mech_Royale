"""Top-level decision loop: state machine over modes with hysteresis.

Priority each tick (first match wins, subject to mode_hold_s except ZONE/cornered):
  1. ZONE       - zone overrides everything when time-to-safety is tight
  2. FIGHT      - threat.should_fight says yes
  3. DISENGAGE  - under fire / losing -> break LOS, face attacker while backing off
  4. DROP       - supply drop capturable and area clear
  5. LOOT       - early game / no threats: weapons, crates, NPC drones, cubes
  6. HIDE       - default: cover inside next circle, silent (<5 m/s), scan likely threat bearing

Each mode module returns an Intent (move goal or vector, aim, fire flags, take/drop, boost, jump).
control.py turns the Intent into the protocol Action.
"""
from bot.config import Rules
from bot.params import PARAMS


class Brain:
    def __init__(self, hello):
        self.rules = Rules(hello["config"])
        self.params = PARAMS
        self.slot = hello["slot"]
        self.mode = "HIDE"
        self.mode_since = 0.0
        # TODO(cursor): self.map = mapdata.load(...); self.world = WorldModel(...);
        #               self.nav = Navigator(...); self.control = Controller(...)

    def act(self, obs, t0):
        """Return a protocol Action dict (tick is set by main.py)."""
        raise NotImplementedError
