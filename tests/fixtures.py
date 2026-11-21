"""Load the recorded match stream and build schema-valid synthetic observations."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
STREAM_PATH = ROOT / "check" / "stream.jsonl.gz"
SCHEMA_PATH = ROOT / "mechfight.schema.json"

Patch = Callable[[dict[str, Any]], None]

_stream_cache: tuple[str, dict[str, Any], list[dict[str, Any]]] | None = None
_validator = None


def load_stream(
    path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read check/stream.jsonl.gz.

    Line 0 is hello, lines 1..300 are observations, and the last line is end.
    Returns (hello, observations). The end message is checked and not returned.
    """
    global _stream_cache
    stream_path = Path(path) if path is not None else STREAM_PATH
    key = str(stream_path)
    if _stream_cache is not None and _stream_cache[0] == key:
        return _stream_cache[1], _stream_cache[2]

    with gzip.open(stream_path, "rt", encoding="utf-8") as handle:
        messages = [json.loads(line) for line in handle if line.strip()]
    if len(messages) != 302:
        raise ValueError(f"expected 302 stream messages, got {len(messages)}")
    hello, observations, end = messages[0], messages[1:-1], messages[-1]
    if hello.get("type") != "hello":
        raise ValueError("line 0 must be hello")
    if end.get("type") != "end":
        raise ValueError("last line must be end")
    if len(observations) != 300 or any(obs.get("type") != "obs" for obs in observations):
        raise ValueError("lines 1..300 must be observations")

    _stream_cache = (key, hello, observations)
    return hello, observations


def server_message_validator():
    """Draft-07 validator for the ServerMessage definition."""
    global _validator
    if _validator is None:
        import jsonschema

        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        wrapped = {
            "$schema": schema["$schema"],
            "definitions": schema["definitions"],
            "$ref": "#/definitions/ServerMessage",
        }
        _validator = jsonschema.Draft7Validator(wrapped)
    return _validator


def _vec2(value: Any) -> list[float]:
    coords = [float(v) for v in value]
    if len(coords) != 2:
        raise ValueError(f"expected 2 coordinates, got {len(coords)}")
    return coords


def _vec3(value: Any) -> list[float]:
    coords = [float(v) for v in value]
    if len(coords) == 2:
        coords.append(0.0)
    if len(coords) != 3:
        raise ValueError(f"expected 2 or 3 coordinates, got {len(coords)}")
    return coords


def _weapon_slot(weapon: Any) -> dict[str, Any] | None:
    if weapon is None:
        return None
    if isinstance(weapon, str):
        return {"kind": weapon, "cooldown": 0.0, "stats": {}}
    if isinstance(weapon, dict):
        return copy.deepcopy(weapon)
    raise TypeError(f"weapon slot must be a kind, object, or null, got {type(weapon).__name__}")


def with_self(
    hp: float | None = None,
    pos: Any = None,
    aim: Any = None,
    weapons: list[Any] | None = None,
    cubes: float | None = None,
    in_zone: bool | None = None,
) -> Patch:
    """Patch obs['self']. Omitted arguments leave the copied observation unchanged."""

    def apply(obs: dict[str, Any]) -> None:
        me = obs["self"]
        if hp is not None:
            me["hp"] = hp
        if pos is not None:
            me["pos"] = _vec3(pos)
        if aim is not None:
            me["aim"] = _vec2(aim)
        if weapons is not None:
            slots = [_weapon_slot(weapon) for weapon in weapons]
            if len(slots) < 2:
                slots.extend([None] * (2 - len(slots)))
            me["weapons"] = slots[:2]
        if cubes is not None:
            me["cubes"] = cubes
        if in_zone is not None:
            me["in_zone"] = in_zone

    return apply


def with_enemy(
    id: int,
    pos: Any,
    hp: float,
    weapons: list[str] | None = None,
    aim: Any = None,
    cubes: float | None = None,
) -> Patch:
    """Append one visible mech to obs['mechs']."""

    def apply(obs: dict[str, Any]) -> None:
        actor: dict[str, Any] = {
            "id": id,
            "kind": "mech",
            "pos": _vec3(pos),
            "vel": [0.0, 0.0, 0.0],
            "aim": _vec2(aim if aim is not None else [0.0, 0.0]),
            "hp": hp,
        }
        if weapons is not None:
            actor["weapons"] = list(weapons)
        if cubes is not None:
            actor["cubes"] = cubes
        obs["mechs"].append(actor)

    return apply


def with_damage_event(by: int | None, amount: float, bearing: float | None, rear: bool) -> Patch:
    """Append a damaged event. Weapon defaults to gun; the helper has no weapon argument."""

    def apply(obs: dict[str, Any]) -> None:
        obs["events"].append(
            {
                "type": "damaged",
                "amount": amount,
                "weapon": "gun",
                "bearing": bearing,
                "by": by,
                "rear": rear,
            }
        )

    return apply


def with_zone(
    center: Any,
    radius: float,
    next_center: Any,
    next_radius: float,
    shrink_start: float,
    shrink_end: float,
    dps: float,
) -> Patch:
    """Replace obs['zone']."""

    def apply(obs: dict[str, Any]) -> None:
        obs["zone"] = {
            "center": _vec2(center),
            "radius": radius,
            "next_center": _vec2(next_center),
            "next_radius": next_radius,
            "shrink_start": shrink_start,
            "shrink_end": shrink_end,
            "dps": dps,
        }

    return apply


def at_time(t: float) -> Patch:
    """Set match time. Tick stays on the 20 Hz clock (tick == round(t * 20))."""

    def apply(obs: dict[str, Any]) -> None:
        obs["t"] = t
        obs["tick"] = int(round(t * 20))

    return apply


def make_obs(*patches: Patch, **overrides: Any) -> dict[str, Any]:
    """Deep-copy the first recorded observation and apply helper patches.

    Pass helpers positionally. Keyword arguments replace top-level fields
    (dicts are merged one level). Example:

        make_obs(with_self(hp=400, in_zone=False), with_enemy(5, [10, 0, 0], 150, ["gun"], [0, 0], 0))
    """
    _, observations = load_stream()
    obs = copy.deepcopy(observations[0])
    for patch in patches:
        patch(obs)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(obs.get(key), dict):
            obs[key].update(copy.deepcopy(value))
        else:
            obs[key] = copy.deepcopy(value)
    return obs
