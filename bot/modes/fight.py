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
    lo, hi = _band(me)
    side = brain.control.strafe((px, py, 0.0), (tx, ty, 0.0), float(obs.get("t") or 0.0))
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


def _band(me: dict) -> tuple[float, float]:
    kinds = set()
    for weapon in me.get("weapons") or []:
        if isinstance(weapon, dict) and weapon.get("kind"):
            kinds.add(weapon["kind"])
        elif isinstance(weapon, str):
            kinds.add(weapon)
    if kinds and kinds <= {"lightning"}:
        return 0.0, 40.0
    return 60.0, 150.0


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
