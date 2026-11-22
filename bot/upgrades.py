"""Upgrade offer handling (protocol §10).

choose() scores each choice from hello.config.upgrades and returns that index.
Damage or fire-rate on a weapon we hold scores 3, armor and max hp 2, speed 1,
sensors 0.5. Ties keep the earlier choice.
"""

from __future__ import annotations

from typing import Any

_RATE_STATS = {"cooldown", "reload", "charge"}
_SPEED_STATS = {"walk_speed", "boost_speed"}
_SENSOR_STATS = {"near_range_bonus", "vision_bonus", "hearing"}


def choose(offer, me, rules, params) -> int:
    del params
    choices = offer.get("choices") if isinstance(offer, dict) else None
    if not choices:
        return 0
    catalog = _catalog(rules)
    held = _held(me)
    best_index = 0
    best_score = -1.0
    for index, choice in enumerate(choices):
        ident = choice.get("id") if isinstance(choice, dict) else choice
        score = _score(catalog.get(str(ident)), held)
        if score > best_score:
            best_index = index
            best_score = score
    return best_index


def _catalog(rules: Any) -> dict[str, dict]:
    raw = getattr(rules, "raw", None)
    upgrades = raw.get("upgrades") if isinstance(raw, dict) else None
    if not isinstance(upgrades, list):
        return {}
    found = {}
    for item in upgrades:
        if isinstance(item, dict) and item.get("id"):
            found[str(item["id"])] = item
    return found


def _held(me: Any) -> set[str]:
    weapons = me.get("weapons") if isinstance(me, dict) else None
    found = set()
    for weapon in weapons or []:
        if isinstance(weapon, dict) and weapon.get("kind"):
            found.add(str(weapon["kind"]))
        elif isinstance(weapon, str):
            found.add(weapon)
    return found


def _score(upgrade: dict | None, held: set[str]) -> float:
    if not isinstance(upgrade, dict):
        return 0.0
    best = 0.0
    for effect in upgrade.get("effects") or []:
        if not isinstance(effect, dict):
            continue
        best = max(best, _effect_score(str(effect.get("path") or ""), effect, held))
    return best


def _effect_score(path: str, effect: dict, held: set[str]) -> float:
    parts = path.split(".")
    if len(parts) == 3 and parts[0] == "weapons":
        kind, stat = parts[1], parts[2]
        if kind not in held:
            return 0.0
        if stat == "damage" or _faster(stat, effect):
            return 3.0
        return 0.0
    if len(parts) == 2 and parts[0] == "mech":
        stat = parts[1]
        if stat in ("armor", "hp"):
            return 2.0
        if stat in _SPEED_STATS:
            return 1.0
        if stat in _SENSOR_STATS:
            return 0.5
    return 0.0


def _faster(stat: str, effect: dict) -> bool:
    """A lower cooldown, reload, or charge time is a fire-rate upgrade."""
    if stat not in _RATE_STATS:
        return False
    mul = effect.get("mul")
    add = effect.get("add")
    if mul is not None and float(mul) < 1.0:
        return True
    if add is not None and float(add) < 0.0:
        return True
    return False
