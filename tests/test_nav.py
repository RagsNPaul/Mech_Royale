"""Path planning, steering, and cover selection."""

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

import bot.nav as nav
from bot.nav import Navigator, astar, best_cover, time_to_reach
from tools.precompute_map import build_map

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "arena"
WORLD = ARENA / "v2" / "world.json"


def _pip(x: float, y: float, ring: list[float]) -> bool:
    px = ring[0::2]
    py = ring[1::2]
    inside = False
    j = len(px) - 1
    for i in range(len(px)):
        yi, yj = py[i], py[j]
        xi, xj = px[i], px[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _contains(x: float, y: float, prism: dict) -> bool:
    if not _pip(x, y, prism["ring"]):
        return False
    return not any(_pip(x, y, hole) for hole in prism.get("holes") or [])


def _centroid(ring: list[float]) -> tuple[float, float]:
    xs = ring[0::2]
    ys = ring[1::2]
    area = cx = cy = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        cross = xs[i] * ys[j] - xs[j] * ys[i]
        area += cross
        cx += (xs[i] + xs[j]) * cross
        cy += (ys[i] + ys[j]) * cross
    area *= 0.5
    if abs(area) < 1e-6:
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return cx / (6.0 * area), cy / (6.0 * area)


def _segment_clear(mapdata, start, end) -> bool:
    a = mapdata.cell_of(start[0], start[1])
    b = mapdata.cell_of(end[0], end[1])
    assert a is not None and b is not None
    return nav._line_clear(mapdata, a, b)


def _open_pair(mapdata, gap_cells: int):
    free = ~mapdata.blocked
    for iy in range(0, mapdata.ny, 4):
        row = free[iy]
        run = 0
        for ix, ok in enumerate(row):
            run = run + 1 if ok else 0
            if run > gap_cells + 2:
                a = mapdata.cell_center(ix - gap_cells, iy)
                b = mapdata.cell_center(ix, iy)
                return a, b
    raise AssertionError("no open straight cells")


@pytest.fixture(scope="module")
def mapdata(tmp_path_factory):
    out = tmp_path_factory.mktemp("navmap") / "map.npz"
    build_map(ARENA, out)
    from bot.mapdata import load

    return load(str(out), str(ARENA))


@pytest.fixture(scope="module")
def world():
    return json.loads(WORLD.read_text(encoding="utf-8"))


def test_open_path_stays_on_free_cells(mapdata):
    start, goal = _open_pair(mapdata, 20)
    path = astar(mapdata, start, goal)
    assert path is not None and len(path) >= 2
    for point in path:
        assert mapdata.is_blocked(point[0], point[1]) is False
    for src, dst in zip(path, path[1:]):
        assert _segment_clear(mapdata, src, dst)


def test_path_around_building_is_longer_than_the_straight_line(mapdata, world):
    picked = None
    for prism in world["prisms"]:
        if prism["kind"] not in ("building", "stadium"):
            continue
        ring = prism["ring"]
        xs, ys = ring[0::2], ring[1::2]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        cx, cy = _centroid(ring)
        if not _contains(cx, cy, prism):
            continue
        if picked is None or area > picked[0]:
            picked = (area, prism, cx, cy)
    assert picked is not None
    _area, prism, cx, cy = picked
    sides = []
    for sign in (-1.0, 1.0):
        distance = 2.0
        found = None
        while distance < 500:
            x = cx + sign * distance
            if mapdata.in_bounds(x, cy) and not _contains(x, cy, prism) and not mapdata.is_blocked(x, cy):
                found = (x + sign * 6.0, cy)
                break
            distance += 2.0
        assert found is not None and not mapdata.is_blocked(found[0], found[1])
        sides.append(found)
    path = astar(mapdata, sides[0], sides[1])
    assert path is not None and len(path) >= 2
    straight = math.hypot(sides[1][0] - sides[0][0], sides[1][1] - sides[0][1])
    assert time_to_reach(path, 1.0) > straight


def test_astar_and_cached_steer_budgets(mapdata):
    rng = np.random.default_rng(1)
    free = np.argwhere(~mapdata.blocked)
    pairs = []
    tries = 0
    while len(pairs) < 100 and tries < 4000:
        tries += 1
        iy, ix = free[int(rng.integers(0, len(free)))]
        yaw = float(rng.random() * 2.0 * math.pi)
        dist = 300.0
        x2 = mapdata.min_x + (ix + 0.5) * mapdata.cell + math.cos(yaw) * dist
        y2 = mapdata.min_y + (iy + 0.5) * mapdata.cell + math.sin(yaw) * dist
        cell = mapdata.cell_of(x2, y2)
        if cell is None or not mapdata.walkable(cell[0], cell[1]):
            continue
        start = mapdata.cell_center(int(ix), int(iy))
        goal = mapdata.cell_center(cell[0], cell[1])
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) < 250.0:
            continue
        pairs.append((start, goal))
    assert len(pairs) == 100
    times = []
    for start, goal in pairs:
        t0 = time.perf_counter()
        astar(mapdata, start, goal)
        times.append(time.perf_counter() - t0)
    assert float(np.percentile(times, 95)) < 0.025

    navbot = Navigator(mapdata, None)
    start, goal = pairs[0]
    t0 = time.perf_counter()
    navbot.steer(start, goal, now=0.0, t0=t0)
    t0 = time.perf_counter()
    started = time.perf_counter()
    navbot.steer(start, goal, now=0.05, t0=t0)
    assert time.perf_counter() - started < 0.001


def test_best_cover_faces_an_eastern_threat(mapdata):
    east = np.flatnonzero(mapdata.cover_shield[:, 0])
    assert east.size > 0
    index = int(east[len(east) // 2])
    cx, cy = (float(mapdata.cover_points[index, 0]), float(mapdata.cover_points[index, 1]))
    me = (cx - 12.0, cy)
    threat = (cx + 4000.0, cy)
    point = best_cover(mapdata, me, [threat], (cx, cy), 220.0, 80.0)
    assert point is not None
    covers = mapdata.cover_points
    nearest = int(np.hypot(covers[:, 0] - point[0], covers[:, 1] - point[1]).argmin())
    assert mapdata.cover_shield[nearest, 0]
    assert math.hypot(point[0] - cx, point[1] - cy) <= 220.0 - 15.0


def test_unreachable_goal_returns_partial_or_none(mapdata):
    assert astar(mapdata, (0.0, 0.0), (1.0e6, 1.0e6)) is None
    start, goal = _open_pair(mapdata, 40)
    previous = nav.MAX_EXPANSIONS
    nav.MAX_EXPANSIONS = 1
    try:
        path = astar(mapdata, start, goal)
    finally:
        nav.MAX_EXPANSIONS = previous
    assert path is None or isinstance(path, list)
    if path:
        assert math.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]) > 20.0
