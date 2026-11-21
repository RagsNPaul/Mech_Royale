"""World model: tracks, sounds, damage, and squad-vs-FFA.

WorldModel.update merges visible mechs and NPCs into tracks kept for 15 s,
estimates velocity from the position change, and remembers sounds for 5 s.
Damage and hits drive the regen timer. self.ffa flips on, and stays on, once
three pieces of infighting evidence have been seen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

TRACK_TTL_S = 15.0
SOUND_TTL_S = 5.0
FFA_EVIDENCE = 3
ALIVE_WINDOW_S = 10.0
BEAM_NEAR_M = 12.0
PROJ_NEAR_M = 8.0
_VEL_DT_MAX_S = 1.0


@dataclass
class Track:
    id: int
    kind: str
    pos: tuple[float, float, float]
    vel: tuple[float, float, float]
    hp: float | None
    max_hp: float | None
    cubes: float | None
    weapons: tuple[str, ...]
    aim: tuple[float, float]
    charging: bool
    last_seen_t: float


@dataclass(frozen=True)
class SoundHint:
    bearing: float
    dist_bucket: str
    kind: str
    t: float


@dataclass(frozen=True)
class Hostile:
    bearing: float | None
    t: float
    rear: bool


def _xyz3(value: Any) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]) if len(value) > 2 else 0.0)


def _xyz2(value: Any) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))


class WorldModel:
    def __init__(self, rules: Any, params: dict[str, Any]):
        self.rules = rules
        self.params = params
        self.ffa = bool(params["ffa_default"])
        self.ffa_evidence = 0
        self.mechs: dict[int, Track] = {}
        self.npcs: dict[int, Track] = {}
        self.sounds: list[SoundHint] = []
        self.death_sounds = 0
        self.hostiles: dict[int, Hostile] = {}
        self.last_damage_t: float | None = None
        self.last_dealt_t: float | None = None
        self.kills: list[tuple[float, int]] = []
        self.pickups: list[tuple[float, float]] = []
        self.took: list[tuple[float, Any, Any, Any]] = []
        self.dropped: list[tuple[float, Any, Any, Any]] = []
        self.weapons: list[Any] = []
        self.self_id: int | None = None
        self.pos: tuple[float, float, float] | None = None
        self.hp: float | None = None
        self.t = 0.0
        self._prev_alive: int | None = None
        self._death_times: list[float] = []
        self._kill_times: list[float] = []
        self._alive_pairs = 0
        self._seen_projectiles: set[int] = set()

    def update(self, obs: dict[str, Any]) -> None:
        self.t = float(obs["t"])
        me = obs["self"]
        self.self_id = int(me["id"])
        self.pos = _xyz3(me["pos"])
        self.hp = float(me["hp"]) if me.get("hp") is not None else None
        self.weapons = list(me.get("weapons") or [])

        for actor in obs.get("mechs") or []:
            track = self._track_from(actor)
            store = self.mechs if track.kind == "mech" else self.npcs
            store[track.id] = track
        self._drop_stale(self.mechs)
        self._drop_stale(self.npcs)

        for sound in obs.get("sounds") or []:
            hint = SoundHint(
                bearing=float(sound.get("bearing") or 0.0),
                dist_bucket=str(sound.get("dist") or ""),
                kind=str(sound.get("kind") or ""),
                t=self.t - float(sound.get("age") or 0.0),
            )
            self.sounds.append(hint)
            if hint.kind == "death":
                self.death_sounds += 1
        self.sounds = [hint for hint in self.sounds if self.t - hint.t <= SOUND_TTL_S]

        for event in obs.get("events") or []:
            kind = event.get("type")
            if kind == "damaged":
                self.last_damage_t = self.t
                attacker = event.get("by")
                if attacker is not None:
                    self.hostiles[int(attacker)] = Hostile(
                        bearing=None if event.get("bearing") is None else float(event["bearing"]),
                        t=self.t,
                        rear=bool(event.get("rear")),
                    )
            elif kind == "hit":
                self.last_dealt_t = self.t
            elif kind == "kill":
                target = event.get("target")
                self.kills.append((self.t, -1 if target is None else int(target)))
                self._kill_times.append(self.t)
            elif kind == "pickup":
                cubes = event.get("cubes")
                self.pickups.append((self.t, 0.0 if cubes is None else float(cubes)))
            elif kind == "took":
                self.took.append((self.t, event.get("id"), event.get("kind"), event.get("slot")))
            elif kind == "dropped":
                self.dropped.append((self.t, event.get("id"), event.get("kind"), event.get("slot")))

        self._note_infighting(obs)
        if self.ffa_evidence >= FFA_EVIDENCE:
            self.ffa = True

    def regen_ready(self) -> bool:
        latest = self.last_damage_t
        if self.last_dealt_t is not None and (latest is None or self.last_dealt_t > latest):
            latest = self.last_dealt_t
        if latest is None:
            return True
        return self.t - latest >= float(self.rules.mech.regen_delay)

    def enemies(self, max_dist: float | None = None) -> list[Track]:
        tracks = list(self.mechs.values())
        if max_dist is not None:
            tracks = [track for track in tracks if self._dist(track) <= max_dist]
        tracks.sort(key=lambda track: track.id)
        return tracks

    def visible_enemies(self) -> list[Track]:
        tracks = [track for track in self.mechs.values() if track.last_seen_t == self.t]
        tracks.sort(key=lambda track: track.id)
        return tracks

    def npcs(self) -> list[Track]:
        tracks = list(self.npcs.values())
        tracks.sort(key=lambda track: track.id)
        return tracks

    def threats_within(self, radius: float) -> list[Track]:
        return self.enemies(radius)

    def hostile_ids(self, window_s: float) -> set[int]:
        return {ident for ident, mark in self.hostiles.items() if self.t - mark.t <= window_s}

    def _track_from(self, actor: dict[str, Any]) -> Track:
        kind = str(actor.get("kind") or "mech")
        ident = int(actor["id"])
        pos = _xyz3(actor["pos"])
        reported = _xyz3(actor["vel"]) if actor.get("vel") is not None else (0.0, 0.0, 0.0)
        store = self.mechs if kind == "mech" else self.npcs
        prev = store.get(ident)
        vel = reported
        if prev is not None:
            dt = self.t - prev.last_seen_t
            if 1e-4 < dt <= _VEL_DT_MAX_S:
                vel = (
                (pos[0] - prev.pos[0]) / dt,
                (pos[1] - prev.pos[1]) / dt,
                (pos[2] - prev.pos[2]) / dt,
            )
        aim = _xyz2(actor["aim"]) if actor.get("aim") is not None else (0.0, 0.0)
        weapons = tuple(str(kind) for kind in (actor.get("weapons") or []))
        hp = None if actor.get("hp") is None else float(actor["hp"])
        max_hp = None if actor.get("max_hp") is None else float(actor["max_hp"])
        cubes = None if actor.get("cubes") is None else float(actor["cubes"])
        return Track(
            id=ident,
            kind=kind,
            pos=pos,
            vel=vel,
            hp=hp,
            max_hp=max_hp,
            cubes=cubes,
            weapons=weapons,
            aim=aim,
            charging=bool(actor.get("charging")),
            last_seen_t=self.t,
        )

    def _drop_stale(self, store: dict[int, Track]) -> None:
        stale = [ident for ident, track in store.items() if self.t - track.last_seen_t > TRACK_TTL_S]
        for ident in stale:
            del store[ident]

    def _dist(self, track: Track) -> float:
        if self.pos is None:
            return math.inf
        return math.hypot(track.pos[0] - self.pos[0], track.pos[1] - self.pos[1])

    def _note_infighting(self, obs: dict[str, Any]) -> None:
        for beam in obs.get("beams") or []:
            if self._beam_hits_other(beam):
                self.ffa_evidence += 1
        for proj in obs.get("projectiles") or []:
            ident = proj.get("id")
            if ident is None or int(ident) in self._seen_projectiles:
                continue
            if self._projectile_between_others(proj):
                self._seen_projectiles.add(int(ident))
                self.ffa_evidence += 1
        alive = obs.get("alive")
        if alive is not None and self._prev_alive is not None and int(alive) < self._prev_alive:
            for _ in range(self._prev_alive - int(alive)):
                self._death_times.append(self.t)
        if alive is not None:
            self._prev_alive = int(alive)
        self._death_times = [when for when in self._death_times if self.t - when <= ALIVE_WINDOW_S]
        self._kill_times = [when for when in self._kill_times if self.t - when <= ALIVE_WINDOW_S]
        pairs = 0 if self._kill_times else len(self._death_times) // 2
        if pairs > self._alive_pairs:
            self.ffa_evidence += pairs - self._alive_pairs
        self._alive_pairs = pairs

    def _beam_hits_other(self, beam: dict[str, Any]) -> bool:
        shooter = beam.get("by")
        path = beam.get("path") or []
        if shooter is None or not path or self.self_id is None:
            return False
        shooter_id = int(shooter)
        if shooter_id == self.self_id or shooter_id in self.npcs:
            return False
        if shooter_id not in self.mechs and not self._mech_id(shooter_id):
            return False
        end = path[-1]
        ex, ey = float(end[0]), float(end[1])
        limit = BEAM_NEAR_M * BEAM_NEAR_M
        for track in self.mechs.values():
            if track.id == shooter_id or track.id == self.self_id:
                continue
            dx = track.pos[0] - ex
            dy = track.pos[1] - ey
            if dx * dx + dy * dy <= limit:
                return True
        return False

    def _projectile_between_others(self, proj: dict[str, Any]) -> bool:
        others = [track for track in self.mechs.values() if track.id != self.self_id]
        if len(others) < 2:
            return False
        pos = proj.get("pos")
        vel = proj.get("vel")
        if not pos or not vel:
            return False
        px, py = float(pos[0]), float(pos[1])
        vx, vy = float(vel[0]), float(vel[1])
        speed = math.hypot(vx, vy)
        if speed < 1.0:
            return False
        limit = PROJ_NEAR_M * PROJ_NEAR_M
        for i, left in enumerate(others):
            for right in others[i + 1 :]:
                dx = right.pos[0] - left.pos[0]
                dy = right.pos[1] - left.pos[1]
                seg2 = dx * dx + dy * dy
                if seg2 < 1.0:
                    continue
                along = ((px - left.pos[0]) * dx + (py - left.pos[1]) * dy) / seg2
                if along < 0.0 or along > 1.0:
                    continue
                cx = left.pos[0] + along * dx
                cy = left.pos[1] + along * dy
                if (px - cx) ** 2 + (py - cy) ** 2 > limit:
                    continue
                align = abs(vx * dx + vy * dy) / (speed * math.sqrt(seg2))
                if align >= 0.85:
                    return True
        return False

    @staticmethod
    def _mech_id(ident: int) -> bool:
        return 0 <= ident < 100
