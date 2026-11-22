"""Low-level control: Intent -> protocol action.

lead_aim solves a 3-step intercept and aims at center of mass.
mortar_pitch picks the low arc when it is clear, otherwise the high arc.
to_action fires only when the torso is on target, reloads a quiet gun, and
strafes on a seeded timer.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable

import mechfight as mf


@dataclass
class Intent:
    """What a mode wants. Every field can be left unset."""

    mode: str | None = None
    reason: str | None = None
    goal_xy: Any = None
    move_vec: Any = None
    aim_target: Any = None
    aim_yaw_pitch: Any = None
    fire: Any = None
    fire_slots: Any = None
    boost: bool | None = None
    jump: bool | None = None
    take: int | None = None
    drop: int | None = None
    pick: int | None = None
    slow: bool = False
    strafe: bool = False


def _xyz(value: Any, default_z: float = 0.0) -> tuple[float, float, float]:
    if value is None:
        return (0.0, 0.0, default_z)
    z = float(value[2]) if len(value) > 2 else default_z
    return (float(value[0]), float(value[1]), z)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _yaw_pitch(
    ax: float, ay: float, az: float, bx: float, by: float, bz: float
) -> tuple[float, float]:
    dx, dy, dz = bx - ax, by - ay, bz - az
    flat = math.hypot(dx, dy)
    return math.atan2(dy, dx), math.atan2(dz, flat)


def lead_aim(
    shooter_muzzle: Any,
    target_pos: Any,
    target_vel: Any,
    speed: float,
    size: float | None = None,
) -> tuple[float, float]:
    """Yaw and pitch of a 3-iteration intercept, aimed at center of mass.

    Center of mass is pos.z + 4.5, multiplied by `size` when the target's scale is known.
    """
    mx, my, mz = _xyz(shooter_muzzle)
    tx, ty, tz = _xyz(target_pos)
    vx, vy, vz = _xyz(target_vel)
    tz += 4.5 * (1.0 if size is None else float(size))
    px, py, pz = tx, ty, tz
    shot = float(speed)
    if shot > 1e-6:
        for _ in range(3):
            flight = math.dist((mx, my, mz), (px, py, pz)) / shot
            px = tx + vx * flight
            py = ty + vy * flight
            pz = tz + vz * flight
    return _yaw_pitch(mx, my, mz, px, py, pz)


def mortar_pitch(
    dist: float,
    dz: float,
    speed: float,
    gravity: float,
    clears_los: bool | Callable[[float], bool] | None = None,
) -> float | None:
    """Low ballistic arc, or the high arc when the low one does not clear. None if unreachable."""
    dist = float(dist)
    speed = float(speed)
    gravity = float(gravity)
    if dist <= 1e-4 or speed <= 0.0 or gravity <= 0.0:
        return None
    v2 = speed * speed
    disc = v2 * v2 - gravity * (gravity * dist * dist + 2.0 * float(dz) * v2)
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    denom = gravity * dist
    low = math.atan2(v2 - root, denom)
    high = math.atan2(v2 + root, denom)
    if clears_los is None or clears_los is True:
        return low
    if clears_los is False:
        return high
    if clears_los(low):
        return low
    return high


class Controller:
    def __init__(self, rules: Any, params: dict[str, Any], seed: int = 0):
        self.rules = rules
        self.params = params
        self._rng = random.Random(int(seed))
        self._strafe_dir = 1
        self._strafe_until = -1.0

    def strafe(self, me_pos: Any, target_pos: Any, now: float) -> tuple[float, float]:
        """Unit step perpendicular to the target. Direction flips every 0.8–1.4 s."""
        mx, my, _mz = _xyz(me_pos)
        tx, ty, _tz = _xyz(target_pos)
        bearing = math.atan2(ty - my, tx - mx)
        if now >= self._strafe_until:
            self._strafe_dir *= -1
            self._strafe_until = now + self._rng.uniform(0.8, 1.4)
        yaw = bearing + self._strafe_dir * (math.pi / 2.0)
        return math.cos(yaw), math.sin(yaw)

    def to_action(self, intent: Intent, obs: dict[str, Any], nav: Any, t0: float) -> dict[str, Any]:
        """Protocol action for this intent. `tick` is echoed from the observation."""
        me = obs["self"]
        pos = _xyz(me.get("pos"))
        now = float(obs.get("t") or 0.0)
        move, steered_jump = self._move(intent, pos, now, nav, t0)
        move = self._limit_speed(move, me, bool(intent.slow))
        desired, dist = self._desired_aim(intent, me, pos, nav)
        fire = self._fire(intent, me, desired, dist)
        jump = steered_jump if intent.jump is None else bool(intent.jump)
        boost = False if intent.slow else bool(intent.boost)
        mode = (intent.mode or "").upper()
        energy = me.get("energy")
        if mode == "FIGHT" and energy is not None and float(energy) < 30.0:
            boost = False
        if mode == "BRAWL" and intent.reason == "rush" and energy is not None and float(energy) < 25.0:
            boost = False
        action: dict[str, Any] = {
            "tick": obs["tick"],
            "move": [float(move[0]), float(move[1])],
            "boost": boost,
            "jump": bool(jump),
            "aim": [float(desired[0]), float(desired[1])],
            "fire": fire,
            "reload": self._reload(intent, me),
            "debug": self._debug(intent, me),
        }
        if intent.pick is not None:
            action["pick"] = int(intent.pick)
        if intent.take is not None:
            action["take"] = int(intent.take)
        if intent.drop is not None:
            action["drop"] = int(intent.drop)
        return action

    def _move(
        self, intent: Intent, pos: tuple[float, float, float], now: float, nav: Any, t0: float
    ) -> tuple[tuple[float, float], bool]:
        if intent.move_vec is not None:
            vec = _xyz(intent.move_vec)
            return (vec[0], vec[1]), False
        target = _target_pos(intent.aim_target)
        if intent.strafe and target is not None:
            return self.strafe(pos, target, now), False
        if intent.goal_xy is not None:
            if nav is not None:
                steered = nav.steer(pos, intent.goal_xy, now, t0)
                return (float(steered.move[0]), float(steered.move[1])), bool(steered.jump)
            return _unit(pos, _xyz(intent.goal_xy)), False
        return (0.0, 0.0), False

    def _limit_speed(self, move: tuple[float, float], me: dict[str, Any], slow: bool) -> tuple[float, float]:
        cap = 1.0
        if slow:
            walk = self._walk_speed(me)
            step = float(self.rules.sound.step_speed)
            cap = 0.0 if walk <= 0.0 else (step * 0.9) / walk
        length = math.hypot(move[0], move[1])
        if length > cap > 0.0:
            scale = cap / length
            return move[0] * scale, move[1] * scale
        if cap <= 0.0:
            return 0.0, 0.0
        return move

    def _desired_aim(
        self, intent: Intent, me: dict[str, Any], pos: tuple[float, float, float], nav: Any
    ) -> tuple[tuple[float, float], float]:
        actual = me.get("aim") or [0.0, 0.0]
        actual_aim = (float(actual[0]), float(actual[1]))
        if intent.aim_yaw_pitch is not None:
            aim = intent.aim_yaw_pitch
            return (float(aim[0]), float(aim[1])), 0.0
        target = intent.aim_target
        if target is None:
            return actual_aim, math.inf
        spot = _target_pos(target)
        vel = _target_vel(target)
        if spot is None:
            return actual_aim, math.inf
        muzzle = (pos[0], pos[1], pos[2] + self._muzzle_height(me))
        dist = math.dist(muzzle, _com(spot, self._size(target)))
        kind = self._aim_kind(intent, me)
        speed = self._projectile_speed(kind, me)
        if kind == "mortar" and speed > 0.0:
            flat = math.hypot(spot[0] - muzzle[0], spot[1] - muzzle[1])
            dz = _com(spot, self._size(target))[2] - muzzle[2]
            gravity = float(self.rules.weapons["mortar"].gravity)
            grid = getattr(nav, "map", None) if nav is not None else None
            pitch = mortar_pitch(
                flat, dz, speed, gravity, _arc_checker(grid, muzzle, spot, speed, gravity)
            )
            yaw = math.atan2(spot[1] - muzzle[1], spot[0] - muzzle[0])
            if pitch is None:
                return lead_aim(muzzle, spot, vel, speed, self._size(target)), dist
            return (yaw, pitch), dist
        return lead_aim(muzzle, spot, vel, speed, self._size(target)), dist

    def _fire(
        self,
        intent: Intent,
        me: dict[str, Any],
        desired: tuple[float, float],
        dist: float,
    ) -> list[bool]:
        if (intent.mode or "").upper() == "HIDE":
            return [False, False]
        want = intent.fire if intent.fire is not None else intent.fire_slots
        slots = [False, False]
        if not want:
            return slots
        actual = me.get("aim") or [0.0, 0.0]
        error = _aim_error((float(actual[0]), float(actual[1])), desired)
        aligned = error < _aim_tolerance(dist)
        weapons = me.get("weapons") or [None, None]
        for index in range(2):
            if index >= len(want) or not want[index]:
                continue
            weapon = weapons[index] if index < len(weapons) else None
            if not isinstance(weapon, dict):
                continue
            kind = str(weapon.get("kind") or "")
            if not aligned and kind not in ("laser", "lightning"):
                continue
            if not self._weapon_ready(kind, weapon, me, desired, dist, aligned):
                continue
            slots[index] = True
        return slots

    def _weapon_ready(
        self,
        kind: str,
        weapon: dict[str, Any],
        me: dict[str, Any],
        desired: tuple[float, float],
        dist: float,
        aligned: bool,
    ) -> bool:
        if kind == "laser":
            return self._laser_trigger(weapon, me, desired, dist)
        if kind == "lightning":
            energy = me.get("energy")
            if energy is not None and float(energy) < 8.0:
                return False
            return self._in_cone(me, desired, dist)
        if not aligned:
            return False
        reach = self._range(kind, weapon)
        if math.isfinite(reach) and dist > reach:
            return False
        ammo = weapon.get("ammo")
        if ammo is not None and float(ammo) <= 0.0:
            return False
        if float(weapon.get("reload") or 0.0) > 0.0:
            return False
        return True

    def _laser_trigger(self, weapon: dict[str, Any], me: dict[str, Any], desired: tuple[float, float], dist: float) -> bool:
        """Hold the trigger to charge; release (False) at 0.6 s or when the target leaves the beam."""
        charge = float(weapon.get("charge") or 0.0)
        leaving = not self._on_target(me, desired, dist)
        if charge >= 0.6 or (leaving and charge > 0.0):
            return False
        if leaving:
            return False
        return True

    def _in_cone(self, me: dict[str, Any], desired: tuple[float, float], dist: float) -> bool:
        if not math.isfinite(dist):
            return False
        reach = float(self.rules.weapons["lightning"].range)
        if dist > reach:
            return False
        cone = math.radians(float(self.rules.weapons["lightning"].cone_deg))
        actual = me.get("aim") or [0.0, 0.0]
        return abs(_wrap(desired[0] - float(actual[0]))) <= cone

    def _on_target(self, me: dict[str, Any], desired: tuple[float, float], dist: float) -> bool:
        actual = me.get("aim") or [0.0, 0.0]
        error = _aim_error((float(actual[0]), float(actual[1])), desired)
        return error < _aim_tolerance(dist)

    def _reload(self, intent: Intent, me: dict[str, Any]) -> bool:
        if intent.aim_target is not None or intent.aim_yaw_pitch is not None:
            return False
        gun = self.rules.weapons["gun"]
        for weapon in me.get("weapons") or []:
            if not isinstance(weapon, dict) or weapon.get("kind") != "gun":
                continue
            ammo = weapon.get("ammo")
            if ammo is None:
                continue
            mag = float((weapon.get("stats") or {}).get("mag", gun.mag))
            if float(weapon.get("reload") or 0.0) <= 0.0 and float(ammo) < mag / 2.0:
                return True
        return False

    def _debug(self, intent: Intent, me: dict[str, Any]) -> dict[str, Any]:
        hp = me.get("hp")
        hp_text = "?" if hp is None else format(float(hp), "g")
        draw = mf.Draw()
        draw.label(f"{intent.mode or ''} | {hp_text} | {intent.reason or ''}")
        drawn = draw.take()
        return drawn or {}

    def _aim_kind(self, intent: Intent, me: dict[str, Any]) -> str:
        want = intent.fire if intent.fire is not None else intent.fire_slots
        weapons = me.get("weapons") or []
        if want:
            for index, flag in enumerate(want):
                if flag and index < len(weapons) and isinstance(weapons[index], dict):
                    return str(weapons[index].get("kind") or "gun")
        for weapon in weapons:
            if isinstance(weapon, dict) and weapon.get("kind"):
                return str(weapon["kind"])
        return "gun"

    def _projectile_speed(self, kind: str, me: dict[str, Any]) -> float:
        if kind in ("laser", "lightning"):
            return math.inf
        for weapon in me.get("weapons") or []:
            if isinstance(weapon, dict) and weapon.get("kind") == kind:
                stats = weapon.get("stats") or {}
                if "speed" in stats:
                    return float(stats["speed"])
        weapon = self.rules.weapons.get(kind)
        return 0.0 if weapon is None else float(getattr(weapon, "speed", 0.0))

    def _range(self, kind: str, weapon: dict[str, Any]) -> float:
        stats = weapon.get("stats") or {}
        if "range" in stats:
            return float(stats["range"])
        base = self.rules.weapons.get(kind)
        if base is None:
            return math.inf
        if getattr(base, "range", None) is not None:
            return float(base.range)
        if kind == "mortar":
            speed = float(base.speed)
            gravity = float(base.gravity)
            return speed * speed / gravity if gravity else math.inf
        return math.inf

    def _size(self, target: Any) -> float | None:
        size = _field(target, "size")
        if size is not None:
            return float(size)
        height = _field(target, "height")
        if height is not None and float(height) > 0.0:
            return float(height) / 9.0
        cubes = _field(target, "cubes")
        if cubes is None:
            return None
        return 1.0 + float(self.rules.cubes.size_per_cube) * float(cubes)

    def _muzzle_height(self, me: dict[str, Any]) -> float:
        stats = me.get("stats") or {}
        if "muzzle_height" in stats:
            return float(stats["muzzle_height"])
        return float(self.rules.mech.muzzle_height)

    def _walk_speed(self, me: dict[str, Any]) -> float:
        stats = me.get("stats") or {}
        if "walk_speed" in stats:
            return float(stats["walk_speed"])
        return float(self.rules.mech.walk_speed)

def _target_pos(target: Any) -> tuple[float, float, float] | None:
    if target is None:
        return None
    pos = _field(target, "pos")
    if pos is None and not isinstance(target, dict):
        pos = target
    if pos is None or isinstance(pos, dict):
        return None
    try:
        return _xyz(pos)
    except (TypeError, IndexError, ValueError):
        return None


def _arc_checker(
    grid: Any,
    muzzle: tuple[float, float, float],
    spot: tuple[float, float, float],
    speed: float,
    gravity: float,
) -> Callable[[float], bool] | None:
    if grid is None or not hasattr(grid, "los"):
        return None

    def clears(pitch: float) -> bool:
        dx, dy = spot[0] - muzzle[0], spot[1] - muzzle[1]
        flat = math.hypot(dx, dy)
        if flat < 1e-6:
            return True
        ux, uy = dx / flat, dy / flat
        vx = speed * math.cos(pitch)
        if vx <= 1e-6:
            return False
        prev = muzzle
        total = flat / vx
        for step in range(1, 7):
            t = total * step / 6.0
            point = (
                muzzle[0] + ux * vx * t,
                muzzle[1] + uy * vx * t,
                muzzle[2] + speed * math.sin(pitch) * t - 0.5 * gravity * t * t,
            )
            if not grid.los(prev, point):
                return False
            prev = point
        return True

    return clears


def _target_vel(target: Any) -> tuple[float, float, float]:
    vel = _field(target, "vel")
    return _xyz(vel)


def _com(pos: tuple[float, float, float], size: float | None) -> tuple[float, float, float]:
    return (pos[0], pos[1], pos[2] + 4.5 * (1.0 if size is None else size))


def _unit(src: tuple[float, float, float], dst: tuple[float, float, float]) -> tuple[float, float]:
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return 0.0, 0.0
    return dx / length, dy / length


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _aim_error(actual: tuple[float, float], desired: tuple[float, float]) -> float:
    return math.hypot(_wrap(desired[0] - actual[0]), desired[1] - actual[1])


def _aim_tolerance(dist: float) -> float:
    """Hold the trigger inside this aim error. The server fires on its own cooldown."""
    if not math.isfinite(dist) or dist <= 1e-6:
        return 0.03
    return max(0.03, math.atan(3.5 / dist) + 0.01)
