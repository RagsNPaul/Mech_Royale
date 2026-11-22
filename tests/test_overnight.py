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
    _DEFAULT_CANDIDATES,
    deadline_at,
    gzip_replay,
    keep_count,
    match_points,
    sample_candidates,
    unread_keys,
    _end_result,
    _load_space,
)
import random


def test_shipped_search_space_is_read_by_the_bot():
    space = json.loads(Path("tools/search_space.json").read_text(encoding="utf-8"))
    assert unread_keys(space) == []
    assert set(space) == {
        "strategy", "cube_detour_m", "heal_hp_frac", "standoff_lightning_m", "standoff_gun_m",
    }
    assert space["strategy"] == ["brawler"]
    assert space["cube_detour_m"] == [0, 40, 80]
    assert space["heal_hp_frac"] == [0.0, 0.4, 0.6]
    assert space["standoff_lightning_m"] == [15, 20, 30]
    assert space["standoff_gun_m"] == [35, 45, 60]


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
    cands = sample_candidates(space, defaults, _DEFAULT_CANDIDATES, random.Random(0))
    assert _DEFAULT_CANDIDATES == 8
    assert len(cands) == 8
    assert cands[0]["id"] == "baseline"
    assert cands[0]["params"]["strategy"] == PARAMS["strategy"]
    assert cands[0]["params"]["standoff_lightning_m"] == pytest.approx(PARAMS["standoff_lightning_m"])
    assert cands[0]["params"]["cube_detour_m"] == pytest.approx(80.0)
    assert cands[0]["params"]["heal_hp_frac"] == pytest.approx(0.0)
    sigs = {json.dumps(c["params"], sort_keys=True, default=float) for c in cands}
    assert len(sigs) == 8


def test_kills_outrank_placement():
    """A kill ties a win. Any kill short of last place beats a win with none."""
    assert match_points(15, 1) > match_points(1, 0)
    assert match_points(16, 1) == pytest.approx(match_points(1, 0))
    assert match_points(1, 0) == pytest.approx(1.0)
    assert match_points(1, 0) > match_points(2, 0)
    assert match_points(16, 0) == pytest.approx(0.0)


def _stamp(dest: Path, place: int, kills: int = 0) -> None:
    match = dest / "1"
    match.mkdir(parents=True)
    (match / "end.json").write_text(json.dumps({"type": "end", "place": place, "kills": kills}), encoding="utf-8")
    (match / "obs.jsonl").write_text("", encoding="utf-8")
    (match / "actions.jsonl").write_text("", encoding="utf-8")
    (match / "replay.jsonl").write_text(
        json.dumps({"type": "header", "tick_hz": 20, "self": 1, "mechs": [{"id": 1}]}) + "\n",
        encoding="utf-8",
    )


def _played(dest: Path, place: int, kills: int = 0) -> dict:
    _stamp(dest, place, kills)
    return {"place": place, "kills": kills}


def test_halving_plays_four_then_eight_and_writes_the_report(tmp_path):
    calls = []

    def play(cand, dest):
        calls.append(cand["id"])
        place = 1 if cand["id"] == "baseline" else 8
        return _played(dest, place, kills=2 if cand["id"] == "baseline" else 0)

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
    assert cands[0]["scores"] == pytest.approx([match_points(1, 2), match_points(1, 2)])
    assert cands[2]["scores"] == pytest.approx([match_points(8, 0)])
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "| baseline |" in text
    assert "| place | kills |" in text
    assert "std err" in text
    assert '"base_margin": 5' in text
    assert "place: 1" in text
    rows = list(csv.DictReader((tmp_path / "results.csv").open(encoding="utf-8")))
    assert len(rows) == 6
    assert all(row["counted"] == "1" for row in rows)
    assert rows[0]["place"] == "1"
    assert rows[0]["kills"] == "2"
    assert float(rows[0]["score"]) == pytest.approx(match_points(1, 2))
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
        return _played(dest, 4, kills=1)

    cand = {"id": "baseline", "params": {"base_margin": 5}, "scores": [], "matches": []}
    now = datetime(2026, 11, 21, 22, 0, 0)
    slept = []
    search = Search(
        [cand], play, 30, deadline_at("07:00", now), lambda: now, tmp_path / "REPORT.md", tmp_path / "results.csv",
        slept.append,
    )
    search.run(first_slots=4, final_slots=1)
    assert calls["n"] == 2
    assert cand["scores"] == pytest.approx([match_points(4, 1)])
    assert slept == [30]
    rows = list(csv.DictReader((tmp_path / "results.csv").open(encoding="utf-8")))
    assert [row["counted"] for row in rows] == ["0", "1"]
    assert rows[0]["place"] == ""
    assert rows[0]["kills"] == ""
    assert rows[1]["kills"] == "1"


def test_stops_at_the_local_deadline(tmp_path):
    clock = {"t": datetime(2026, 11, 21, 6, 59, 40)}
    calls = []

    def play(cand, dest):
        calls.append(clock["t"])
        return _played(dest, 3, kills=0)

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
    assert cand["scores"] == pytest.approx([match_points(3, 0)])


def test_end_json_supplies_place_and_kills(tmp_path):
    _stamp(tmp_path, 6, kills=3)
    got = _end_result(tmp_path / "1" / "end.json")
    assert got == {"place": 6, "kills": 3}
    assert match_points(6, 3) == pytest.approx(3 + 10 / 15)
    (tmp_path / "1" / "end.json").write_text(json.dumps({"place": 6}), encoding="utf-8")
    assert _end_result(tmp_path / "1" / "end.json") is None


def test_round_interleaves_candidates(tmp_path):
    calls = []

    def play(cand, dest):
        calls.append(cand["id"])
        return _played(dest, 5, kills=1)

    cands = [
        {"id": name, "params": {"heal_hp_frac": 0}, "scores": [], "matches": []}
        for name in ("baseline", "c01", "c02")
    ]
    now = datetime(2026, 11, 21, 22, 0, 0)
    search = Search(
        cands, play, 0, deadline_at("07:00", now), lambda: now, tmp_path / "REPORT.md", tmp_path / "results.csv",
        lambda _s: None,
    )
    search.run(first_slots=2, final_slots=1)
    assert calls == ["baseline", "c01", "c02", "baseline", "c01", "c02", "baseline", "c01"]


def test_eight_candidates_play_sixty_four_matches(tmp_path):
    calls = []

    def play(cand, dest):
        calls.append(cand["id"])
        return _played(dest, 8, kills=2)

    names = ["baseline"] + [f"c{i:02d}" for i in range(1, 8)]
    cands = [{"id": name, "params": {"heal_hp_frac": 0}, "scores": [], "matches": []} for name in names]
    now = datetime(2026, 11, 21, 22, 0, 0)
    search = Search(
        cands, play, 0, deadline_at("07:00", now), lambda: now, tmp_path / "REPORT.md", tmp_path / "results.csv",
        lambda _s: None,
    )
    search.run()
    assert len(calls) == 64
    assert calls[:32] == names * 4
    assert calls[32:48] == ["baseline", "c01", "c02", "c03"] * 4
    assert calls[48:] == ["baseline", "c01"] * 8
    assert cands[0]["scores"] == pytest.approx([match_points(8, 2)] * (4 + 4 + 8))


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
