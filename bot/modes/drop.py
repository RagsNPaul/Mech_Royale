"""Mode: DROP. Walk into a supply drop and stand in its ring."""

from __future__ import annotations

import math

from bot.control import Intent


def run(brain, obs) -> Intent | None:
    drop = brain.supply
    if not isinstance(drop, dict):
        return None
    me = obs["self"]
    px, py = float(me["pos"][0]), float(me["pos"][1])
    pos = drop.get("pos") or (px, py, 0.0)
    tx, ty = float(pos[0]), float(pos[1])
    dist = math.hypot(tx - px, ty - py)
    radius = float(drop.get("radius") or 8.0)
    yaw = math.atan2(ty - py, tx - px)
    if dist <= radius:
        return Intent(reason="capture", move_vec=(0.0, 0.0), slow=True, aim_yaw_pitch=(yaw, 0.0), fire=(False, False))
    move, jump = brain.approach(obs, (tx, ty))
    return Intent(reason="supply", move_vec=move, jump=jump, aim_yaw_pitch=(yaw, 0.0), fire=(False, False))
