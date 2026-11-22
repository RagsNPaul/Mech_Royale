"""Read one relay match directory and print what the bot actually did."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

_MECH_HIT = {"fire", "zap", "laser"}


def _read_text(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8")
    gz = Path(str(path) + ".gz")
    if gz.is_file():
        with gzip.open(gz, "rt", encoding="utf-8") as handle:
            return handle.read()
    return ""


def _load_jsonl(path: Path) -> list[dict]:
    text = _read_text(path)
    if not text:
        return []
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return row if isinstance(row, dict) else {}


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_id(row: list) -> int | None:
    if not row:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


def analyze(directory: Path) -> str:
    replay = _load_jsonl(directory / "replay.jsonl")
    obs = _load_jsonl(directory / "obs.jsonl")
    actions = _load_jsonl(directory / "actions.jsonl")
    end = _load_json(directory / "end.json")

    header = next((row for row in replay if row.get("type") == "header"), {})
    frames = [row for row in replay if isinstance(row.get("m"), list) and "t" in row]
    tick_hz = _num(header.get("tick_hz"), 20.0) or 20.0
    mech_ids = {int(m["id"]) for m in header.get("mechs") or [] if isinstance(m, dict) and "id" in m}
    our_id = header.get("self")
    if our_id is None and obs:
        our_id = (obs[0].get("self") or {}).get("id")
    try:
        our_id = int(our_id) if our_id is not None else None
    except (TypeError, ValueError):
        our_id = None

    by_tick: dict[int, dict] = {}
    series: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for frame in frames:
        tick = int(_num(frame.get("t")))
        by_tick[tick] = frame
        when = tick / tick_hz
        for row in frame["m"]:
            if not isinstance(row, list) or len(row) < 10:
                continue
            mid = _row_id(row)
            if mid is None:
                continue
            mech_ids.add(mid)
            series[mid].append((when, _num(row[1]), _num(row[2]), _num(row[9])))

    events: list[tuple[float, dict]] = []
    for frame in frames:
        for ev in frame.get("ev") or []:
            if isinstance(ev, dict):
                events.append((_num(ev.get("t"), _num(frame.get("t"))) / tick_hz, ev))

    dealt_w: dict[str, float] = defaultdict(float)
    taken_w: dict[str, float] = defaultdict(float)
    taken_by: dict[str, float] = defaultdict(float)
    dealt_by: dict[str, float] = defaultdict(float)
    taken_hits: list[tuple[float, dict]] = []
    our_mech_hits: list[tuple[float, dict]] = []
    shots: Counter[str] = Counter()
    death_at: float | None = None

    for when, ev in events:
        kind = ev.get("e")
        if kind == "death" and _same(ev.get("id"), our_id):
            death_at = when
        if kind in _MECH_HIT and _same(ev.get("id"), our_id):
            shots[str(kind)] += 1
        if kind != "hit":
            continue
        weapon = str(ev.get("w") or "?")
        amount = _num(ev.get("amt"))
        src = ev.get("src")
        dst = ev.get("dst")
        if _same(src, our_id):
            dealt_w[weapon] += amount
            dealt_by[str(dst)] += amount
            if _is_mech(dst, mech_ids):
                our_mech_hits.append((when, ev))
        if _same(dst, our_id):
            taken_w[weapon] += amount
            taken_by[str(src)] += amount
            taken_hits.append((when, ev))

    lines: list[str] = [f"match: {directory}"]
    place = end.get("place", "?")
    kills = end.get("kills", "?")
    if death_at is None:
        death_at = _hp_zero(frames, our_id, tick_hz)
    alive = _time_alive(death_at, frames, tick_hz, obs)
    lines.append(f"place: {place}")
    lines.append(f"kills: {kills}")
    lines.append(f"time alive: {alive}")
    lines.append("")
    lines.append("damage dealt by weapon")
    lines.extend(_amount_lines(dealt_w))
    lines.append("damage dealt by target")
    lines.extend(_amount_lines(dealt_by))
    lines.append("damage taken by weapon")
    lines.extend(_amount_lines(taken_w))
    lines.append("damage taken by attacker")
    lines.extend(_amount_lines(taken_by))
    rear_line, rear_note = _rear(taken_hits, obs)
    lines.append(rear_line)
    if rear_note:
        lines.append(rear_note)
    lines.append("")
    fire_n = sum(shots.values())
    hit_n = len(our_mech_hits)
    rate = f"{100.0 * hit_n / fire_n:.1f}%" if fire_n else "n/a"
    shot_bits = ", ".join(f"{name} {shots[name]}" for name in ("fire", "zap", "laser") if shots[name]) or "0"
    lines.append(f"our shots: {fire_n} ({shot_bits})")
    lines.append(f"our hits on mechs: {hit_n}")
    lines.append(f"hit rate: {rate}")
    lines.append("distance at each hit on a mech (m)")
    if not our_mech_hits:
        lines.append("  none")
    for when, ev in our_mech_hits:
        dist = _hit_distance(when, ev, our_id, by_tick, tick_hz)
        dist_text = "?" if dist is None else f"{dist:.1f}"
        lines.append(f"  t={when:.2f} {ev.get('w')} -> {ev.get('dst')} {dist_text}")
    lines.append("")
    lines.append("cubes over time (m[9])")
    lines.extend(_cubes(series, our_id))
    lines.append("")
    lines.extend(_cause(taken_hits, death_at, place))
    lines.append("")
    lines.append("debug label/reason")
    lines.extend(_debug_hist(actions))
    lines.append("")
    lines.append(_stuck(series.get(our_id, [])))
    return "\n".join(lines)


def _same(value, our_id: int | None) -> bool:
    if our_id is None or value is None:
        return False
    try:
        return int(value) == our_id
    except (TypeError, ValueError):
        return False


def _is_mech(dst, mech_ids: set[int]) -> bool:
    try:
        return int(dst) in mech_ids
    except (TypeError, ValueError):
        return False


def _amount_lines(amounts: dict[str, float]) -> list[str]:
    if not amounts:
        return ["  none"]
    return [f"  {key}: {amounts[key]:.1f}" for key in sorted(amounts, key=lambda k: (-amounts[k], str(k)))]


def _hp_zero(frames: list[dict], our_id: int | None, tick_hz: float) -> float | None:
    if our_id is None:
        return None
    for frame in frames:
        for row in frame.get("m") or []:
            if isinstance(row, list) and len(row) > 6 and _row_id(row) == our_id and _num(row[6]) <= 0:
                return _num(frame.get("t")) / tick_hz
    return None


def _time_alive(death_at, frames, tick_hz: float, obs: list[dict]) -> str:
    start = _num(frames[0].get("t")) / tick_hz if frames else _num((obs[0] if obs else {}).get("t"))
    if death_at is not None:
        return f"{max(0.0, death_at - start):.1f} s"
    if obs:
        return f"{max(0.0, _num(obs[-1].get('t')) - _num(obs[0].get('t'))):.1f} s"
    if frames:
        return f"{max(0.0, _num(frames[-1].get('t')) / tick_hz - start):.1f} s"
    return "unknown"


def _rear(taken_hits: list[tuple[float, dict]], obs: list[dict]) -> tuple[str, str]:
    if taken_hits and any("rear" in ev or "back" in ev for _when, ev in taken_hits):
        rear = sum(1 for _when, ev in taken_hits if ev.get("rear") or ev.get("back"))
        return _rear_line(rear, len(taken_hits)), ""
    damaged = [ev for row in obs for ev in (row.get("events") or []) if isinstance(ev, dict) and ev.get("type") == "damaged"]
    if not damaged:
        return "hits taken in the rear: 0/0", "rear flag was not on replay hit events or obs damaged events"
    rear = sum(1 for ev in damaged if ev.get("rear"))
    return _rear_line(rear, len(damaged)), "rear flag taken from obs damaged events (replay hits had none)"


def _rear_line(rear: int, total: int) -> str:
    pct = 100.0 * rear / total if total else 0.0
    return f"hits taken in the rear: {rear}/{total} ({pct:.1f}%)"


def _hit_distance(when: float, ev: dict, our_id: int | None, by_tick: dict[int, dict], tick_hz: float) -> float | None:
    tick = int(round(_num(ev.get("t"), when * tick_hz)))
    frame = by_tick.get(tick)
    if frame is None and by_tick:
        tick = min(by_tick, key=lambda t: abs(t - tick))
        frame = by_tick[tick]
    if frame is None or our_id is None:
        return None
    ours = _xy(frame, our_id)
    theirs = _xy(frame, ev.get("dst"))
    if ours is None or theirs is None:
        return None
    return math.hypot(ours[0] - theirs[0], ours[1] - theirs[1])


def _xy(frame: dict, mid) -> tuple[float, float] | None:
    try:
        want = int(mid)
    except (TypeError, ValueError):
        return None
    for row in frame.get("m") or []:
        if isinstance(row, list) and len(row) >= 3 and _row_id(row) == want:
            return _num(row[1]), _num(row[2])
    return None


def _cubes(series: dict[int, list[tuple[float, float, float, float]]], our_id: int | None) -> list[str]:
    if not series:
        return ["  none"]
    times = sorted({round(pt[0], 1) for pts in series.values() for pt in pts})
    if not times:
        return ["  none"]
    note = ""
    if len(times) <= 30:
        picked = times
    else:
        step = 10.0
        picked = [t for t in times if abs(t - times[0]) < 1e-6 or abs(t - times[-1]) < 1e-6 or abs(math.fmod(t, step)) < 0.15]
        if len(picked) < 2:
            picked = times[:: max(1, len(times) // 12)]
        note = "  sampled every 10 s"
    ids = sorted(series)
    lines = [line for line in [note] if line]
    lines.append("  t " + " ".join(("us" if mid == our_id else str(mid)) for mid in ids))
    for stamp in picked:
        cells = []
        for mid in ids:
            cubes = _cubes_at(series[mid], stamp)
            cells.append("?" if cubes is None else f"{cubes:.0f}")
        lines.append(f"  {stamp:.1f} " + " ".join(cells))
    return lines


def _cubes_at(points: list[tuple[float, float, float, float]], stamp: float) -> float | None:
    best = None
    for when, _x, _y, cubes in points:
        if when <= stamp + 1e-6 and (best is None or when >= best[0]):
            best = (when, cubes)
    return None if best is None else best[1]


def _cause(taken_hits: list[tuple[float, dict]], death_at: float | None, place) -> list[str]:
    if death_at is None:
        if place in (1, "1"):
            return ["cause of death: survived (place 1)"]
        return ["cause of death: no death event in the replay"]
    window = [(when, ev) for when, ev in taken_hits if death_at - 5.0 <= when <= death_at + 1e-6]
    lines = [f"cause of death: last 5.0 s before t={death_at:.2f}"]
    if not window:
        lines.append("  no hit events")
        return lines
    total = 0.0
    for when, ev in window:
        amount = _num(ev.get("amt"))
        total += amount
        rear = " rear" if ev.get("rear") or ev.get("back") else ""
        lines.append(f"  t={when:.2f} {ev.get('w')} from {ev.get('src')} {amount:.1f}{rear}")
    lines.append(f"  total {total:.1f}")
    return lines


def _debug_hist(actions: list[dict]) -> list[str]:
    counts: Counter[str] = Counter()
    for action in actions:
        debug = action.get("debug")
        if not isinstance(debug, dict):
            continue
        label = str(debug.get("label") or "").strip()
        if not label:
            continue
        parts = [part.strip() for part in label.split("|")]
        reason = parts[-1] if len(parts) >= 3 else label
        counts[reason or label] += 1
    if not counts:
        return ["  none"]
    return [f"  {name}: {counts[name]}" for name, _n in counts.most_common()]


def _stuck(points: list[tuple[float, float, float, float]]) -> str:
    if len(points) < 2:
        return "longest stretch moved < 1 m: 0.0 s"
    best = 0.0
    best_at = points[0][0]
    anchor = 0
    for i in range(1, len(points)):
        dist = math.hypot(points[i][1] - points[anchor][1], points[i][2] - points[anchor][2])
        if dist >= 1.0:
            span = points[i - 1][0] - points[anchor][0]
            if span > best:
                best = span
                best_at = points[anchor][0]
            anchor = i
    span = points[-1][0] - points[anchor][0]
    if span > best:
        best = span
        best_at = points[anchor][0]
    return f"longest stretch moved < 1 m: {best:.1f} s starting at t={best_at:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one runs/<match> directory.")
    parser.add_argument("match", type=Path)
    args = parser.parse_args()
    print(analyze(args.match))


if __name__ == "__main__":
    main()
