"""Mode: FIGHT. Lead the target, strafe, and hold a favorable range inside the zone."""

from __future__ import annotations

import math

from bot.control import Intent


def run(brain, obs) -> Intent:
    me = obs["self"]
    target = brain.target
    px, py = float(me["pos"][0]), float(me["pos"][1])
    if target is None:
        return Intent(reason="no-target", move_vec=(0.0, 0.0), slow=True, fire=(False, False))
    tx, ty = float(target.pos[0]), float(target.pos[1])
    dist = math.hypot(tx - px, ty - py)
    squad = not brain.world.ffa
    lo, hi = _band(me, squad)
    side = brain.control.strafe((px, py, 0.0), (tx, ty, 0.0), float(obs.get("t") or 0.0))
    peeked = _from_cover(px, py, brain._peek, side) if brain.fight_reason == "reengage" else None
    if peeked is not None:
        move = peeked
    else:
        if dist > hi:
            radial = _unit(tx - px, ty - py)
        elif dist < lo:
            radial = _unit(px - tx, py - ty)
        else:
            radial = (0.0, 0.0)
        move = _unit(radial[0] + side[0], radial[1] + side[1])
        if move == (0.0, 0.0):
            move = side
    move = _stay_inside(px, py, obs.get("zone"), move)
    return Intent(
        reason=brain.fight_reason or "fight",
        move_vec=move,
        aim_target=target,
        fire=(True, True),
        strafe=True,
    )


def _band(me: dict, squad: bool) -> tuple[float, float]:
    kinds = set()
    for weapon in me.get("weapons") or []:
        if isinstance(weapon, dict) and weapon.get("kind"):
            kinds.add(weapon["kind"])
        elif isinstance(weapon, str):
            kinds.add(weapon)
    if squad and kinds and kinds <= {"gun"}:
        return 0.0, 110.0
    if kinds and kinds <= {"lightning"}:
        return 0.0, 40.0
    return 60.0, 150.0


def _from_cover(
    px: float, py: float, peek: tuple[float, float] | None, side: tuple[float, float]
) -> tuple[float, float] | None:
    """Strafe on the cover edge, or step back to it when the peek has drifted off."""
    if peek is None:
        return None
    dx, dy = peek[0] - px, peek[1] - py
    if math.hypot(dx, dy) <= 6.0:
        return side
    toward = _unit(dx, dy)
    move = _unit(toward[0] + side[0], toward[1] + side[1])
    return toward if move == (0.0, 0.0) else move


def _stay_inside(px: float, py: float, zone: dict | None, move: tuple[float, float]) -> tuple[float, float]:
    if not zone:
        return move
    cx, cy = float(zone["center"][0]), float(zone["center"][1])
    radius = float(zone["radius"])
    dx, dy = px - cx, py - cy
    dist = math.hypot(dx, dy)
    if dist < 1.0 or dist < radius - 20.0:
        return move
    inward = _unit(-dx, -dy)
    blended = _unit(move[0] + inward[0], move[1] + inward[1])
    return blended if blended[0] * inward[0] + blended[1] * inward[1] > 0.0 else inward


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, 0.0
    return dx / length, dy / length
