"""Successive halving, unread keys, and the report. No public-room connection."""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bot.params import PARAMS
from tools.analyze_runs import analyze
from tools.overnight import (
    Search,
    deadline_at,
    gzip_replay,
    keep_count,
    sample_candidates,
    unread_keys,
    _load_space,
)
import random


def test_shipped_search_space_is_read_by_the_bot():
    space = json.loads(Path("tools/search_space.json").read_text(encoding="utf-8"))
    assert unread_keys(space) == []
    assert "heal_hp_frac" in space
    assert "base_margin" in space


def test_unread_key_is_an_error(tmp_path, capsys):
    path = tmp_path / "space.json"
    path.write_text(json.dumps({"not_a_real_knob": [1], "strategy": ["brawler"]}), encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        _load_space(path)
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "not_a_real_knob" in err
    assert "strategy" not in err.split("does not read:", 1)[-1]


def test_keep_count_halts_at_two():
    assert keep_count(12) == 6
    assert keep_count(6) == 3
    assert keep_count(3) == 2
    assert keep_count(2) == 2


def test_sample_always_includes_the_defaults():
    space = json.loads(Path("tools/search_space.json").read_text(encoding="utf-8"))
    defaults = {key: PARAMS[key] for key in space}
    cands = sample_candidates(space, defaults, 12, random.Random(0))
    assert len(cands) == 12
    assert cands[0]["id"] == "baseline"
    assert cands[0]["params"]["standoff_lightning_m"] == pytest.approx(20.0)
    assert cands[0]["params"]["base_margin"] == pytest.approx(5.0)
    sigs = {json.dumps(c["params"], sort_keys=True, default=float) for c in cands}
    assert len(sigs) == 12


def _stamp(dest: Path, place: int) -> None:
    match = dest / "1"
    match.mkdir(parents=True)
    (match / "end.json").write_text(json.dumps({"type": "end", "place": place, "kills": 0}), encoding="utf-8")
    (match / "obs.jsonl").write_text("", encoding="utf-8")
    (match / "actions.jsonl").write_text("", encoding="utf-8")
    (match / "replay.jsonl").write_text(
        json.dumps({"type": "header", "tick_hz": 20, "self": 1, "mechs": [{"id": 1}]}) + "\n",
        encoding="utf-8",
    )


def test_halving_plays_four_then_eight_and_writes_the_report(tmp_path):
    calls = []

    def play(cand, dest):
        calls.append(cand["id"])
        place = 1 if cand["id"] == "baseline" else 8
        _stamp(dest, place)
        return place

    cands = [
        {"id": "baseline", "params": {"base_margin": 5}, "scores": [], "matches": []},
        {"id": "c01", "params": {"base_margin": 2}, "scores": [], "matches": []},
        {"id": "c02", "params": {"base_margin": 2}, "scores": [], "matches": []},
        {"id": "c03", "params": {"base_margin": 2}, "scores": [], "matches": []},
    ]
    now = datetime(2026, 11, 21, 22, 0, 0)
    search = Search(
        cands, play, 0, deadline_at("07:00", now), lambda: now, tmp_path / "REPORT.md", tmp_path / "results.csv",
        lambda _s: None,
    )
    search.run(first_slots=1, final_slots=1)
    # 4 candidates once, keep 2, those two once more.
    assert calls == ["baseline", "c01", "c02", "c03", "baseline", "c01"]
    assert len(cands[0]["scores"]) == 2
    assert cands[2]["scores"] == [8]
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "| baseline |" in text
    assert "std err" in text
    assert '"base_margin": 5' in text
    assert "place: 1" in text
    rows = list(csv.DictReader((tmp_path / "results.csv").open(encoding="utf-8")))
    assert len(rows) == 6
    assert all(row["counted"] == "1" for row in rows)
    assert not list(tmp_path.rglob("replay.jsonl"))
    gz = list(tmp_path.rglob("replay.jsonl.gz"))
    assert len(gz) == 6
    assert "place:" in analyze(gz[0].parent)


def test_failed_match_is_retried_once_and_not_counted(tmp_path):
    calls = {"n": 0}

    def play(cand, dest):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        _stamp(dest, 4)
        return 4

    cand = {"id": "baseline", "params": {"base_margin": 5}, "scores": [], "matches": []}
    now = datetime(2026, 11, 21, 22, 0, 0)
    slept = []
    search = Search(
        [cand], play, 30, deadline_at("07:00", now), lambda: now, tmp_path / "REPORT.md", tmp_path / "results.csv",
        slept.append,
    )
    search.run(first_slots=4, final_slots=1)
    assert calls["n"] == 2
    assert cand["scores"] == [12]
    assert slept == [30]
    rows = list(csv.DictReader((tmp_path / "results.csv").open(encoding="utf-8")))
    assert [row["counted"] for row in rows] == ["0", "1"]


def test_stops_at_the_local_deadline(tmp_path):
    clock = {"t": datetime(2026, 11, 21, 6, 59, 40)}
    calls = []

    def play(cand, dest):
        calls.append(clock["t"])
        _stamp(dest, 3)
        return 3

    def sleep(seconds):
        clock["t"] += timedelta(seconds=seconds)

    cand = {"id": "baseline", "params": {"base_margin": 5}, "scores": [], "matches": []}
    search = Search(
        [cand], play, 30, deadline_at("07:00", clock["t"]), lambda: clock["t"],
        tmp_path / "REPORT.md", tmp_path / "results.csv", sleep,
    )
    search.run(first_slots=4, final_slots=2)
    assert calls == [datetime(2026, 11, 21, 6, 59, 40)]
    assert search.stopped is True
    assert cand["scores"] == [13]


def test_gzip_leaves_the_replay_readable(tmp_path):
    match = tmp_path / "1"
    _stamp(tmp_path, 2)
    frame = {
        "t": 0,
        "m": [[1, 0, 0, 0, 0, 0, 100, 0, 0, 1, 100, 0, -1]],
        "ev": [{"t": 0, "e": "hit", "src": 1, "dst": 1, "amt": 40, "w": "lightning"}],
    }
    header = {"type": "header", "tick_hz": 20, "self": 1, "mechs": [{"id": 1}]}
    (match / "replay.jsonl").write_text(json.dumps(header) + "\n" + json.dumps(frame) + "\n", encoding="utf-8")
    gzip_replay(match)
    assert not (match / "replay.jsonl").exists()
    text = analyze(match)
    assert "place: 2" in text
    assert "lightning: 40.0" in text
