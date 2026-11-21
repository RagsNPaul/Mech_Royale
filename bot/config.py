"""Rules snapshot from hello.config. Read numbers here, never hard-code them (balance changes).

Missing keys fall back to the protocol.md defaults (and, where protocol.md gives no number,
the recorded hello). Each missing path is logged to stderr once per process.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

# Logged missing paths. One line per path for the life of the process.
_logged: set[str] = set()

# protocol.md §5 and §7. Weapon numbers that the prose does not state are the
# recorded hello values (they match every number the prose does state).
_MECH = {
    "radius": 2.5,
    "height": 9.0,
    "eye_height": 8.0,
    "muzzle_height": 7.0,
    "hp": 1400.0,
    "walk_speed": 11.7,
    "boost_speed": 19.5,
    "ground_accel": 58.5,
    "air_accel": 12.0,
    "gravity": 22.0,
    "jump_speed": 11.0,
    "jet_accel": 34.0,
    "max_rise_speed": 10.0,
    "step_up": 1.2,
    "max_slope_deg": 50.0,
    "energy": 100.0,
    "energy_regen": 18.0,
    "energy_regen_delay": 0.6,
    "boost_drain": 17.0,
    "jet_drain": 15.0,
    "turn_rate": 4.0,
    "pitch_rate": 3.0,
    "min_pitch": -0.7,
    "max_pitch": 1.3,
    "regen_delay": 5.0,
    "regen_per_s": 45.0,
    "charge_move_mult": 0.5,
    "self_damage_mult": 0.5,
    "armor": 0.0,
    "rear_mult": 1.5,
    "front_arc_deg": 180.0,
    "near_range_bonus": 0.0,
    "vision_bonus": 0.0,
    "hearing": 1.0,
}

_WEAPONS = {
    "gun": {
        "damage": 45.0,
        "cooldown": 0.24,
        "speed": 252.0,
        "range": 230.0,
        "spread_deg": 0.6,
        "mag": 12.0,
        "reload": 2.4,
    },
    "mortar": {
        "damage": 200.0,
        "splash": 160.0,
        "splash_radius": 7.0,
        "splash_edge": 0.3,
        "cooldown": 2.6,
        "speed": 72.0,
        "gravity": 22.0,
        "ttl": 7.0,
        "knock": 16.0,
        "knock_up": 9.0,
        "drain": 45.0,
    },
    "laser": {
        "damage": 315.0,
        "charge": 0.75,
        "min_charge": 0.15,
        "min_frac": 0.35,
        "hold_s": 2.0,
        "cooldown": 1.65,
        "range": 280.0,
        "pierce": 0.0,
        "charge_turn_mult": 1.0,
        "cookoff": 0.6,
    },
    "lightning": {
        "damage": 60.0,
        "cooldown": 0.25,
        "range": 45.0,
        "cone_deg": 30.0,
        "chain": 2.0,
        "chain_range": 14.0,
        "chain_falloff": 0.7,
        "energy": 3.0,
        "short_s": 2.0,
    },
}

_VISION = {
    "range": 200.0,
    "fov_deg": 120.0,
    "near_range": 45.0,
    "canopy_blocks": True,
    "show_enemy_hp": True,
    "memory_s": 5.0,
}

_SOUND_RANGE = {
    "step": 40.0,
    "boost": 90.0,
    "reload": 60.0,
    "gun": 320.0,
    "mortar": 420.0,
    "explosion": 420.0,
    "laser_charge": 140.0,
    "laser": 380.0,
    "lightning": 160.0,
    "jet": 90.0,
    "death": 600.0,
    "crate": 220.0,
    "npc_gun": 200.0,
    "drop": 600.0,
}

_SOUND = {
    "bearing_noise_deg": 7.0,
    "near": 60.0,
    "mid": 160.0,
    "step_speed": 5.0,
    "step_every": 0.6,
}

_CUBES = {
    "pickup_radius": 3.5,
    "hp_per_cube": 0.04,
    "damage_per_cube": 0.04,
    "stat_cap": 10.0,
    "death_drop_extra": 2.0,
    "size_per_cube": 0.05,
}

# protocol.md §10 starts 1, 3, 6, 10, 15, 21, … (triangular). The recorded hello continues it.
_UPGRADE_THRESHOLDS = (1, 3, 6, 10, 15, 21, 28, 36, 45)

_LOOT = {
    "pickup_radius": 4.0,
    "equip_s": 1.0,
    "floor_count": 40.0,
    "floor_roof_fraction": 0.25,
    "crate_weapon_chance": 0.5,
    "death_drop": 1.0,
}

_LOOT_WEIGHTS = {"gun": 0.0, "mortar": 1.0, "laser": 1.0, "lightning": 1.0}

_NPC = {
    "count": 12.0,
    "hp": 350.0,
    "radius": 1.8,
    "height": 3.5,
    "speed": 5.0,
    "leash": 30.0,
    "chase_leash": 60.0,
    "aggro_range": 45.0,
    "deaggro_range": 90.0,
    "spawn_clear": 0.0,
    "damage": 25.0,
    "cooldown": 1.0,
    "projectile_speed": 90.0,
    "spread_deg": 2.5,
    "cubes": 2.0,
}

_DROPS = {
    "fall_s": 25.0,
    "radius": 8.0,
    "capture_s": 4.0,
    "cubes": 3.0,
    "repair": 1.0,
    "spread": 0.7,
}
_DROP_TIMES = (70, 140, 210)

_ZONE_PHASES = (
    {"hold_s": 60, "shrink_s": 50, "radius": 380, "dps": 20},
    {"hold_s": 45, "shrink_s": 40, "radius": 240, "dps": 35},
    {"hold_s": 35, "shrink_s": 35, "radius": 140, "dps": 60},
    {"hold_s": 30, "shrink_s": 30, "radius": 70, "dps": 100},
    {"hold_s": 25, "shrink_s": 25, "radius": 25, "dps": 160},
    {"hold_s": 15, "shrink_s": 25, "radius": 0, "dps": 260},
)


def _log_missing(path: str, default: Any) -> None:
    if path in _logged:
        return
    _logged.add(path)
    print(f"rules: missing {path}; using default {default!r}", file=sys.stderr)


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if isinstance(value, dict):
        return value
    _log_missing(key, "{}")
    return {}


def _pick(section: dict[str, Any], key: str, default: Any, path: str) -> Any:
    if key in section:
        return section[key]
    _log_missing(path, default)
    return default


def _namespace(section: dict[str, Any], defaults: dict[str, Any], prefix: str) -> SimpleNamespace:
    values = {key: _pick(section, key, default, f"{prefix}.{key}") for key, default in defaults.items()}
    return SimpleNamespace(**values)


def _weapon(kind: str, weapons: dict[str, Any]) -> SimpleNamespace:
    raw = weapons.get(kind)
    if not isinstance(raw, dict):
        raw = {}
        _log_missing(f"weapons.{kind}", "{}")
    defaults = _WEAPONS[kind]
    values = {
        key: _pick(raw, key, default, f"weapons.{kind}.{key}") for key, default in defaults.items()
    }
    return SimpleNamespace(**values)


def _phases(zone: dict[str, Any]) -> list[SimpleNamespace]:
    raw = zone.get("phases")
    if not isinstance(raw, list) or not raw:
        _log_missing("zone.phases", _ZONE_PHASES)
        raw = _ZONE_PHASES
    phases = []
    for index, phase in enumerate(raw):
        src = phase if isinstance(phase, dict) else {}
        fallback = _ZONE_PHASES[index] if index < len(_ZONE_PHASES) else _ZONE_PHASES[-1]
        phases.append(
            SimpleNamespace(
                **{
                    key: _pick(src, key, fallback[key], f"zone.phases[{index}].{key}")
                    for key in ("hold_s", "shrink_s", "radius", "dps")
                }
            )
        )
    return phases


class Rules:
    def __init__(self, config: dict[str, Any] | None):
        if not isinstance(config, dict):
            _log_missing("config", "{}")
            config = {}
        self.raw = config

        mech = _section(config, "mech")
        self.mech = _namespace(mech, _MECH, "mech")

        weapons = _section(config, "weapons")
        self.weapons = {kind: _weapon(kind, weapons) for kind in _WEAPONS}

        self.vision = _namespace(_section(config, "vision"), _VISION, "vision")

        sound = _section(config, "sound")
        sound_values = {
            key: _pick(sound, key, default, f"sound.{key}") for key, default in _SOUND.items()
        }
        raw_ranges = sound.get("range")
        if isinstance(raw_ranges, dict):
            ranges = {
                key: _pick(raw_ranges, key, default, f"sound.range.{key}")
                for key, default in _SOUND_RANGE.items()
            }
        else:
            _log_missing("sound.range", _SOUND_RANGE)
            ranges = dict(_SOUND_RANGE)
        sound_values["range"] = ranges
        self.sound = SimpleNamespace(**sound_values)

        zone = _section(config, "zone")
        center = zone.get("start_center")
        if not (isinstance(center, (list, tuple)) and len(center) == 2):
            center = (41.3, -72.8)
            _log_missing("zone.start_center", center)
        self.zone = SimpleNamespace(
            start_center=(float(center[0]), float(center[1])),
            start_radius=_pick(zone, "start_radius", 560.0, "zone.start_radius"),
            phases=_phases(zone),
        )

        cubes = _section(config, "cubes")
        cube_values = {
            key: _pick(cubes, key, default, f"cubes.{key}") for key, default in _CUBES.items()
        }
        thresholds = cubes.get("upgrade_thresholds")
        if not isinstance(thresholds, (list, tuple)):
            _log_missing("cubes.upgrade_thresholds", _UPGRADE_THRESHOLDS)
            thresholds = _UPGRADE_THRESHOLDS
        cube_values["upgrade_thresholds"] = tuple(thresholds)
        self.cubes = SimpleNamespace(**cube_values)

        loot = _section(config, "loot")
        loot_values = {
            key: _pick(loot, key, default, f"loot.{key}") for key, default in _LOOT.items()
        }
        weights = loot.get("weights")
        if isinstance(weights, dict):
            loot_values["weights"] = {
                key: _pick(weights, key, default, f"loot.weights.{key}")
                for key, default in _LOOT_WEIGHTS.items()
            }
        else:
            _log_missing("loot.weights", _LOOT_WEIGHTS)
            loot_values["weights"] = dict(_LOOT_WEIGHTS)
        self.loot = SimpleNamespace(**loot_values)

        self.drops = _namespace(_section(config, "drops"), _DROPS, "drops")
        drop_times = _section(config, "drops").get("times") if isinstance(config.get("drops"), dict) else None
        if not isinstance(drop_times, (list, tuple)):
            _log_missing("drops.times", _DROP_TIMES)
            drop_times = _DROP_TIMES
        self.drops.times = tuple(drop_times)

        self.npc = _namespace(_section(config, "npc"), _NPC, "npc")
