"""Dev-only successive-halving search over the public room.

Not part of the match image. Each candidate is a shallow MECH_PARAMS overlay.
A key in the search space that the bot never reads is an error.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.params import PARAMS  # noqa: E402
from tools.analyze_runs import analyze  # noqa: E402

_CSV_FIELDS = ("time", "candidate", "round", "slot", "attempt", "counted", "place", "score", "match", "params")


def unread_keys(keys, root: Path | None = None) -> list[str]:
    """Keys that never appear as a quoted name under bot/, outside params.py."""
    base = Path(root) if root is not None else _ROOT / "bot"
    blobs: list[str] = []
    for path in sorted(base.rglob("*.py")):
        if path.name == "params.py":
            continue
        blobs.append(path.read_text(encoding="utf-8"))
    text = "\n".join(blobs)
    missing = []
    for key in keys:
        if f'"{key}"' not in text and f"'{key}'" not in text:
            missing.append(key)
    return missing


def keep_count(n: int) -> int:
    """Top half, but never cut a field of 3 down to 1. Stop at 2."""
    if n <= 2:
        return n
    half = n // 2
    return half if half >= 2 else 2


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return float(value)


def _sig(params: dict) -> tuple:
    return tuple((key, _num(params[key])) for key in sorted(params))


def sample_candidates(space: dict, defaults: dict, k: int, rng: random.Random) -> list[dict]:
    """k unique overlays. The first is the current defaults, named baseline."""
    keys = list(space)
    baseline = {key: defaults[key] if key in defaults else space[key][0] for key in keys}
    found = [baseline]
    seen = {_sig(baseline)}
    guard = 0
    while len(found) < k and guard < max(1000, k * 50):
        guard += 1
        cand = {key: rng.choice(list(space[key])) for key in keys}
        sig = _sig(cand)
        if sig in seen:
            continue
        seen.add(sig)
        found.append(cand)
    out = []
    for i, params in enumerate(found):
        out.append({"id": "baseline" if i == 0 else f"c{i:02d}", "params": params, "scores": [], "matches": []})
    return out


def _mean_se(scores: list[float]) -> tuple[float, float, int]:
    n = len(scores)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = statistics.fmean(scores)
    se = 0.0 if n < 2 else statistics.stdev(scores) / math.sqrt(n)
    return mean, se, n


def _rank(cand: dict) -> tuple:
    mean, se, n = _mean_se(cand["scores"])
    if n == 0:
        return (1.0, 0.0, 0, cand["id"] != "baseline", cand["id"])
    return (-mean, se, -n, cand["id"] != "baseline", cand["id"])


def top_half(cands: list[dict]) -> list[dict]:
    return sorted(cands, key=_rank)[: keep_count(len(cands))]


def deadline_at(hhmm: str, now: datetime) -> datetime:
    hour, minute = (int(part) for part in hhmm.split(":", 1))
    stop = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if stop <= now:
        stop += timedelta(days=1)
    return stop


def gzip_replay(match_dir: Path) -> None:
    src = match_dir / "replay.jsonl"
    if not src.is_file():
        return
    dest = match_dir / "replay.jsonl.gz"
    with src.open("rb") as raw, gzip.open(dest, "wb") as packed:
        shutil.copyfileobj(raw, packed)
    src.unlink()


def _fmt(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "—"
    return f"{value:.2f}"


def render_report(cands: list[dict]) -> str:
    ranked = sorted(cands, key=_rank)
    lines = ["# Overnight", ""]
    lines.append("| candidate | mean | std err | n |")
    lines.append("| --- | --- | --- | --- |")
    for cand in ranked:
        mean, se, n = _mean_se(cand["scores"])
        lines.append(f"| {cand['id']} | {_fmt(mean)} | {_fmt(se)} | {n} |")
    best = next((cand for cand in ranked if cand["scores"]), None)
    lines.append("")
    lines.append("## best params")
    lines.append("")
    if best is None:
        lines.append("No counted matches yet.")
    else:
        lines.append("Paste over the matching keys in `bot/params.py`:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(best["params"], indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
        lines.append(f"## {best['id']} last matches")
        lines.append("")
        for match in best["matches"][-3:]:
            lines.append(f"### {match}")
            lines.append("")
            try:
                lines.append(analyze(match))
            except Exception as exc:
                lines.append(f"(analyze failed: {exc})")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_report(path: Path, cands: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_report(cands), encoding="utf-8")
    tmp.replace(path)


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


class Search:
    def __init__(self, cands, play, sleep, deadline, now, report_path, csv_path, clock_sleep):
        self.cands = cands
        self.play = play
        self.sleep_s = sleep
        self.deadline = deadline
        self.now = now
        self.report_path = Path(report_path)
        self.csv_path = Path(csv_path)
        self.clock_sleep = clock_sleep
        self._gap = False
        self.stopped = False

    def _past(self) -> bool:
        return self.now() >= self.deadline

    def _pause(self) -> bool:
        if self._past():
            self.stopped = True
            return False
        self.clock_sleep(self.sleep_s)
        if self._past():
            self.stopped = True
            return False
        return True

    def _connect(self, cand, round_name, slot, attempt):
        if self._past():
            self.stopped = True
            return "stop"
        if self._gap and not self._pause():
            return "stop"
        self._gap = True
        dest = self.csv_path.parent / "data" / cand["id"] / round_name / f"s{slot}a{attempt}"
        try:
            place = self.play(cand, dest)
        except Exception as exc:
            print(f"overnight: {cand['id']} relay error: {exc}", file=sys.stderr)
            place = None
        match = dest / "1"
        stamp = self.now().isoformat(timespec="seconds")
        params = json.dumps(cand["params"], sort_keys=True)
        if place is None:
            gzip_replay(match)
            _append_csv(self.csv_path, {
                "time": stamp, "candidate": cand["id"], "round": round_name, "slot": slot,
                "attempt": attempt, "counted": 0, "place": "", "score": "", "match": str(match), "params": params,
            })
            _write_report(self.report_path, self.cands)
            print(f"overnight: {cand['id']} {round_name} slot {slot} attempt {attempt} not counted", file=sys.stderr)
            return None
        place = int(place)
        score = 16 - place
        cand["scores"].append(score)
        cand["matches"].append(match)
        _append_csv(self.csv_path, {
            "time": stamp, "candidate": cand["id"], "round": round_name, "slot": slot,
            "attempt": attempt, "counted": 1, "place": place, "score": score, "match": str(match), "params": params,
        })
        _write_report(self.report_path, self.cands)
        gzip_replay(match)
        print(f"overnight: {cand['id']} place {place} score {score}", file=sys.stderr)
        return place

    def _slots(self, cands, slots, round_name) -> bool:
        for cand in cands:
            for slot in range(slots):
                first = self._connect(cand, round_name, slot, 1)
                if first == "stop":
                    return False
                if first is None:
                    second = self._connect(cand, round_name, slot, 2)
                    if second == "stop":
                        return False
        return True

    def run(self, first_slots: int = 4, final_slots: int = 8) -> list[dict]:
        alive = list(self.cands)
        rnd = 1
        while len(alive) > 2:
            if not self._slots(alive, first_slots, f"r{rnd}"):
                return self.cands
            alive = top_half(alive)
            print("overnight: still in " + ", ".join(cand["id"] for cand in alive), file=sys.stderr)
            rnd += 1
        if alive and not self._past():
            self._slots(alive, final_slots, "final")
        return self.cands


def _load_space(path: Path) -> dict:
    space = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(space, dict) or not space:
        raise SystemExit("search space must be a non-empty object")
    for key, values in space.items():
        if not isinstance(values, list) or not values:
            raise SystemExit(f"search space {key} needs a non-empty list")
    missing = unread_keys(space)
    if missing:
        print("error: search space keys the code does not read:", file=sys.stderr)
        for key in missing:
            print(f"  {key}", file=sys.stderr)
        raise SystemExit(2)
    return space


def _relay(room: int):
    python = sys.executable

    def play(cand, dest: Path):
        dest.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["MECH_PARAMS"] = json.dumps(cand["params"])
        cmd = [
            python, "-u", str(_ROOT / "tools" / "live_relay.py"),
            "--room", str(room),
            "--matches", "1",
            "--name", f"rags-{cand['id']}"[:32],
            "--out", str(dest),
            "--cmd", python, "-u", "main.py",
        ]
        print(f"overnight: connect {cand['id']} {dest}", file=sys.stderr)
        proc = subprocess.run(cmd, cwd=str(_ROOT), env=env)
        if proc.returncode != 0:
            print(f"overnight: relay exit {proc.returncode}", file=sys.stderr)
            return None
        end_path = dest / "1" / "end.json"
        if not end_path.is_file():
            print("overnight: no end.json", file=sys.stderr)
            return None
        try:
            end = json.loads(end_path.read_text(encoding="utf-8"))
            return int(end["place"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            print("overnight: end.json has no place", file=sys.stderr)
            return None

    return play


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Successive-halving search on the public room.")
    parser.add_argument("--until", default="07:00", help="local HH:MM to stop (default 07:00)")
    parser.add_argument("--room", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=30, help="seconds between disconnect and reconnect")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="runs")
    parser.add_argument("--space", type=Path, default=_ROOT / "tools" / "search_space.json")
    args = parser.parse_args(argv)
    space = _load_space(args.space)
    defaults = {key: PARAMS[key] for key in space if key in PARAMS}
    rng = random.Random(args.seed)
    cands = sample_candidates(space, defaults, 12, rng)
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    now = datetime.now()
    search = Search(
        cands,
        _relay(args.room),
        args.sleep,
        deadline_at(args.until, now),
        datetime.now,
        out / "REPORT.md",
        out / "results.csv",
        time.sleep,
    )
    print(
        f"overnight: {len(cands)} candidates until {search.deadline.strftime('%Y-%m-%d %H:%M')} local",
        file=sys.stderr,
    )
    try:
        search.run()
    except KeyboardInterrupt:
        print("overnight: stopped", file=sys.stderr)
    print(f"overnight: report {out / 'REPORT.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
