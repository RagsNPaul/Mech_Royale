"""Top-level decision loop: state machine over modes with hysteresis.

Priority each tick (first match wins, subject to mode_hold_s except ZONE and
hp under the fight floor while under fire):
  1. ZONE       - out of the zone, or the safe circle will reach us before we do
  2. DISENGAGE  - hp below min_hp_frac_to_fight AND at least two hostiles inside
                  threat_radius
  3. FIGHT      - anyone else we can see, remember, or who just shot us
  4. HIDE       - default: cover inside the next circle, silent, scan

DROP and LOOT are not in v1; those modules return None.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bot import upgrades
from bot.config import Rules
from bot.control import Controller, Intent
from bot.mapdata import load as load_map
from bot.modes import disengage, fight, hide, zone
from bot.nav import Navigator, astar, best_cover, time_to_reach
from bot.params import PARAMS
from bot.threat import pick_target, should_fight, zone_safe_s
from bot.world import WorldModel

_ROOT = Path(__file__).resolve().parents[1]
_MODES = {
    "ZONE": zone.run,
    "FIGHT": fight.run,
    "DISENGAGE": disengage.run,
    "HIDE": hide.run,
}
_WEAPON_SOUNDS = {"gun", "mortar", "laser", "lightning", "laser_charge", "explosion", "npc_gun", "death"}
_DAMAGE_WINDOW_S = 3.0
_HOSTILE_RANGE_M = 120.0
_ZONE_MARGIN_M = 10.0
_ZONE_INSIDE_M = 15.0


def _arena_dirs(hello: dict[str, Any]) -> list[Path]:
    raw = str((hello.get("map") or {}).get("dir") or "")
    found: list[Path] = []
    candidates = [Path("/arena"), _ROOT / "arena"]
    if raw:
        candidates.extend([Path(raw), Path.cwd() / raw, _ROOT / raw])
    for candidate in candidates:
        try:
            path = candidate.resolve()
        except OSError:
            continue
        if path.is_dir() and path not in found:
            found.append(path)
    return found


def _load_map(hello: dict[str, Any]) -> Any:
    """Docker path first, then the repo arena. None if every attempt fails."""
    arenas = _arena_dirs(hello)
    npz_paths = [Path("/bot/data/map.npz"), _ROOT / "data" / "map.npz"]
    errors: list[str] = []
    for arena in arenas:
        for npz in npz_paths:
            try:
                return load_map(str(npz), str(arena))
            except Exception as exc:
                errors.append(f"{npz} @ {arena}: {exc}")
    if errors:
        print(f"map load failed; running without a map ({errors[-1]})", file=sys.stderr)
    return None


def _xy(pos: Any) -> tuple[float, float]:
    return float(pos[0]), float(pos[1])


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(bx - ax, by - ay)


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, 0.0
    return dx / length, dy / length


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


class Brain:
    def __init__(self, hello: dict[str, Any]):
        self.rules = Rules(hello["config"])
        self.params = PARAMS
        self.slot = hello["slot"]
        self.mode = "HIDE"
        self.mode_since = -1e9
        self.map = None
        try:
            self.map = _load_map(hello)
        except Exception as exc:
            print(f"map load failed; running without a map ({exc})", file=sys.stderr)
            self.map = None
        self.world = WorldModel(self.rules, self.params)
        self.world.map = self.map
        self.nav = Navigator(self.map, self.rules) if self.map is not None else None
        if self.map is not None:
            try:
                from bot import nav as _nav

                _nav._blocked_bytes(self.map)
                _nav._jump_tables(self.map)
            except Exception as exc:
                print(f"path cache warm failed ({exc})", file=sys.stderr)
        self.control = Controller(self.rules, self.params, seed=int(hello.get("seed") or 0))
        self.target = None
        self.fight_reason = ""
        self.t0 = 0.0
        self._offer: int | None = None
        self._cover: tuple[float, float] | None = None
        self._cover_key: tuple[Any, ...] | None = None

    def act(self, obs: dict[str, Any], t0: float) -> dict[str, Any]:
        """One decision. Returns a protocol action; tick is also set by main.py."""
        self.t0 = t0
        self.world.map = self.map
        self.world.update(obs)
        offer = obs.get("offer")
        self._offer = None if not offer else int(upgrades.choose(offer, obs["self"], self.rules, self.params))
        now = float(obs.get("t") or 0.0)
        candidate = self._candidate(obs)
        if candidate != self.mode and self._switch_now(candidate, obs, now):
            self.mode = candidate
            self.mode_since = now
        intent = _MODES[self.mode](self, obs)
        if not isinstance(intent, Intent):
            intent = Intent(slow=True, reason="idle")
        intent.mode = self.mode
        if self._offer is not None:
            intent.pick = self._offer
        nav = None if self.map is None else self.nav
        return self.control.to_action(intent, obs, nav, t0)

    def _candidate(self, obs: dict[str, Any]) -> str:
        self.target = None
        self.fight_reason = ""
        if self._zone_urgent(obs):
            return "ZONE"
        if self._outnumbered(obs):
            return "DISENGAGE"
        target = pick_target(obs["self"], self.world, self.rules, self.params)
        reason = "press"
        if target is not None:
            _fight, reason = should_fight(obs["self"], target, self.world, obs, self.rules, self.params)
            if reason in ("low_hp", "losing", "zone", "no_los", "no_target"):
                reason = "press"
        else:
            target = self._fallback_target(obs)
        if target is not None:
            self.target = target
            self.fight_reason = reason
            return "FIGHT"
        return "HIDE"

    def _outnumbered(self, obs: dict[str, Any]) -> bool:
        """Retreat only below the fight floor and only against two or more nearby hostiles."""
        floor = float(self.params["min_hp_frac_to_fight"])
        if self._hp_frac(obs) >= floor:
            return False
        return len(self.world.enemies(float(self.params["threat_radius_m"]))) >= 2

    def _fallback_target(self, obs: dict[str, Any]) -> Any:
        """A remembered mech, or a point along the latest damage bearing."""
        radius = float(self.params["threat_radius_m"])
        known = self.world.enemies(radius)
        if known:
            px, py = _xy(obs["self"]["pos"])
            return min(known, key=lambda enemy: _dist(px, py, enemy.pos[0], enemy.pos[1]))
        if self._damage_age(obs) > _DAMAGE_WINDOW_S:
            return None
        bearing = self.threat_bearing()
        if bearing is None:
            return None
        px, py, pz = (float(v) for v in obs["self"]["pos"][:3])
        return SimpleNamespace(
            id=None,
            pos=(px + math.cos(bearing) * 50.0, py + math.sin(bearing) * 50.0, pz),
            vel=(0.0, 0.0, 0.0),
        )

    def _hp_frac(self, obs: dict[str, Any]) -> float:
        me = obs["self"]
        max_hp = float(me.get("max_hp") or 0.0)
        if max_hp <= 0.0:
            return 1.0
        return float(me.get("hp") or 0.0) / max_hp

    def _switch_now(self, candidate: str, obs: dict[str, Any], now: float) -> bool:
        if candidate == "ZONE":
            return True
        if self._hp_frac(obs) < float(self.params["min_hp_frac_to_fight"]) and self._under_fire(obs):
            return True
        return now - self.mode_since >= float(self.params["mode_hold_s"])

    def _zone_urgent(self, obs: dict[str, Any]) -> bool:
        me = obs["self"]
        if me.get("in_zone") is False:
            return True
        safe = zone_safe_s(me, obs)
        if math.isinf(safe):
            return False
        goal = self.ring_point(obs, _ZONE_MARGIN_M)
        px, py = _xy(me["pos"])
        straight = _dist(px, py, goal[0], goal[1])
        speed = self._walk_speed(me)
        eta = 0.0 if straight < 1.0 or speed <= 0.0 else straight / speed
        if eta * 1.3 >= safe:
            return True
        if self.map is None or eta * 2.5 < safe:
            return False
        path = astar(self.map, (px, py), goal)
        if path:
            eta = time_to_reach(path, speed)
        return eta * 1.3 >= safe

    def _pressed(self, obs: dict[str, Any]) -> bool:
        """Recent damage, or any visible mech inside 120 m."""
        if self._damage_age(obs) <= _DAMAGE_WINDOW_S:
            return True
        px, py = _xy(obs["self"]["pos"])
        for enemy in self.world.visible_enemies():
            if _dist(px, py, enemy.pos[0], enemy.pos[1]) <= _HOSTILE_RANGE_M:
                return True
        return False

    def _under_fire(self, obs: dict[str, Any]) -> bool:
        return self._pressed(obs)

    def _damage_age(self, obs: dict[str, Any]) -> float:
        if self.world.last_damage_t is None:
            return math.inf
        return float(obs.get("t") or 0.0) - self.world.last_damage_t

    def ring_point(self, obs: dict[str, Any], margin: float) -> tuple[float, float]:
        """Point on the next circle, `margin` meters in from the edge, nearest to us."""
        zone = obs["zone"]
        cx, cy = float(zone["next_center"][0]), float(zone["next_center"][1])
        radius = max(0.0, float(zone["next_radius"]) - margin)
        px, py = _xy(obs["self"]["pos"])
        dx, dy = px - cx, py - cy
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return cx + radius, cy
        return cx + dx / length * radius, cy + dy / length * radius

    def threat_bearing(self) -> float | None:
        """Newest damage bearing, else the newest weapon sound."""
        best_t = -1e9
        bearing = None
        for mark in self.world.hostiles.values():
            if mark.bearing is None or mark.t < best_t:
                continue
            best_t = mark.t
            bearing = mark.bearing
        if bearing is not None:
            return bearing
        heard = [hint for hint in self.world.sounds if hint.kind in _WEAPON_SOUNDS]
        if not heard:
            return None
        heard.sort(key=lambda hint: hint.t)
        return heard[-1].bearing

    def threat_points(self, obs: dict[str, Any]) -> list[tuple[float, float]]:
        points = [(enemy.pos[0], enemy.pos[1]) for enemy in self.world.visible_enemies()]
        bearing = self.threat_bearing()
        if bearing is not None and not points:
            px, py = _xy(obs["self"]["pos"])
            points.append((px + math.cos(bearing) * 50.0, py + math.sin(bearing) * 50.0))
        return points

    def cover_goal(self, obs: dict[str, Any], threats: list[tuple[float, float]], max_dist: float) -> tuple[float, float] | None:
        if self.map is None:
            return None
        px, py = _xy(obs["self"]["pos"])
        key = (round(float(obs.get("t") or 0.0)), len(threats), round(px, -1), round(py, -1))
        if key == self._cover_key:
            return self._cover
        zone = obs["zone"]
        self._cover = best_cover(
            self.map,
            (px, py),
            threats,
            (float(zone["next_center"][0]), float(zone["next_center"][1])),
            float(zone["next_radius"]),
            max_dist,
        )
        self._cover_key = key
        return self._cover

    def approach(self, obs: dict[str, Any], goal: tuple[float, float]) -> tuple[tuple[float, float], bool]:
        """Steer along a path, or a straight line when that path is missing."""
        px, py = _xy(obs["self"]["pos"])
        straight = _unit(goal[0] - px, goal[1] - py)
        if self.nav is None or self.map is None:
            return straight, False
        steered = self.nav.steer((px, py), goal, float(obs.get("t") or 0.0), self.t0)
        move = (float(steered.move[0]), float(steered.move[1]))
        if _dot(move, straight) > 0.15:
            return move, bool(steered.jump)
        return straight, False

    def _walk_speed(self, me: dict[str, Any]) -> float:
        stats = me.get("stats") or {}
        if "walk_speed" in stats:
            return float(stats["walk_speed"])
        return float(self.rules.mech.walk_speed)
