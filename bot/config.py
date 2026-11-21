"""Rules snapshot from hello.config. Read numbers here, never hard-code them (balance changes).

TODO(cursor):
- class Rules: built from hello["config"]; expose typed attributes the rest of the bot uses:
  mech radius/height/eye/muzzle, walk/boost speed, turn_rate, front_arc_deg, rear_mult,
  regen_delay, regen_per_s, weapon stats per kind (damage, cooldown, speed, range, ...),
  vision (near_range, range, fov_deg, memory_s), sound ranges, zone, cubes (stat_cap,
  damage_per_cube, upgrade_thresholds), loot.pickup_radius, drops (radius, capture_s).
- Keep the raw dict as self.raw for anything not mapped yet.
"""
from typing import Any


class Rules:
    def __init__(self, config: dict[str, Any]):
        self.raw = config
        # TODO(cursor): map fields listed above.
