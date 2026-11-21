"""Map precompute and line-of-sight queries."""

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from bot.mapdata import load
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
        if (yi > y) != (yj > y):
            x_hit = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_hit:
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
    area = 0.0
    cx = 0.0
    cy = 0.0
    count = len(xs)
    for i in range(count):
        j = (i + 1) % count
        cross = xs[i] * ys[j] - xs[j] * ys[i]
        area += cross
        cx += (xs[i] + xs[j]) * cross
        cy += (ys[i] + ys[j]) * cross
    area *= 0.5
    if abs(area) < 1e-6:
        return sum(xs) / count, sum(ys) / count
    return cx / (6.0 * area), cy / (6.0 * area)


def _large_building(world: dict) -> dict:
    best = None
    for prism in world["prisms"]:
        if prism["kind"] not in ("building", "stadium"):
            continue
        ring = prism["ring"]
        xs, ys = ring[0::2], ring[1::2]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        cx, cy = _centroid(ring)
        if not _contains(cx, cy, prism):
            continue
        if best is None or area > best[0]:
            best = (area, prism, cx, cy)
    assert best is not None
    return {"prism": best[1], "cx": best[2], "cy": best[3]}


@pytest.fixture(scope="module")
def world():
    return json.loads(WORLD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mapdata(tmp_path_factory):
    out = tmp_path_factory.mktemp("map") / "map.npz"
    started = time.perf_counter()
    build_map(ARENA, out)
    assert time.perf_counter() - started < 120
    assert out.stat().st_size < 15 * 1024 * 1024
    started = time.perf_counter()
    loaded = load(str(out), str(ARENA))
    assert time.perf_counter() - started < 1.5
    return loaded


def test_ground_is_finite_and_origin_is_near_zero(mapdata):
    assert abs(mapdata.ground_z(0.0, 0.0)) < 1.0
    assert np.isfinite(mapdata.ground).all()
    xs = np.linspace(mapdata.min_x + 1, mapdata.max_x - 1, 40)
    ys = np.linspace(mapdata.min_y + 1, mapdata.max_y - 1, 30)
    for y in ys:
        for x in xs:
            assert math.isfinite(mapdata.ground_z(float(x), float(y)))


def test_building_blocks_its_centroid_and_not_a_point_outside(mapdata, world):
    picked = _large_building(world)
    cx, cy = picked["cx"], picked["cy"]
    assert mapdata.is_blocked(cx, cy)

    outside = None
    for angle in np.linspace(0.0, 2.0 * math.pi, 36, endpoint=False):
        dx, dy = math.cos(float(angle)), math.sin(float(angle))
        distance = 1.0
        while distance < 500 and _contains(cx + dx * distance, cy + dy * distance, picked["prism"]):
            distance += 1.0
        px = cx + dx * (distance + 30.0)
        py = cy + dy * (distance + 30.0)
        if mapdata.in_bounds(px, py) and not mapdata.is_blocked(px, py):
            outside = (px, py)
            break
    assert outside is not None
    assert not mapdata.is_blocked(*outside)


def test_los_through_building_and_across_open_ground(mapdata, world):
    picked = _large_building(world)
    cx, cy = picked["cx"], picked["cy"]
    z = mapdata.ground_z(cx, cy) + 8.0
    left = right = None
    for sign in (-1.0, 1.0):
        distance = 1.0
        while distance < 500 and _contains(cx + sign * distance, cy, picked["prism"]):
            distance += 1.0
        point = (cx + sign * (distance + 3.0), cy, z)
        if sign < 0:
            left = point
        else:
            right = point
    assert left is not None and right is not None
    assert mapdata.los(left, right) is False

    free = ~mapdata.blocked & ~np.isfinite(mapdata.solid_top)
    found = None
    step = 10  # 20 m at 2 m cells
    for iy in range(0, mapdata.ny, 5):
        row = free[iy]
        run = 0
        for ix, ok in enumerate(row):
            run = run + 1 if ok else 0
            if run > step:
                found = (ix - step, iy, ix, iy)
                break
        if found:
            break
    assert found is not None
    x0, y0 = mapdata.cell_center(found[0], found[1])
    x1, y1 = mapdata.cell_center(found[2], found[3])
    z0 = mapdata.ground_z(x0, y0) + 2.0
    z1 = mapdata.ground_z(x1, y1) + 2.0
    assert math.hypot(x1 - x0, y1 - y0) == pytest.approx(20.0, abs=0.1)
    assert mapdata.los((x0, y0, z0), (x1, y1, z1)) is True


def test_los_average_under_budget(mapdata):
    rng = np.random.default_rng(0)
    span_x = mapdata.max_x - mapdata.min_x
    span_y = mapdata.max_y - mapdata.min_y
    pairs = []
    while len(pairs) < 1000:
        x = mapdata.min_x + float(rng.random()) * span_x
        y = mapdata.min_y + float(rng.random()) * span_y
        yaw = float(rng.random()) * 2.0 * math.pi
        dist = float(rng.random()) * 200.0
        x2 = min(max(x + math.cos(yaw) * dist, mapdata.min_x), mapdata.max_x)
        y2 = min(max(y + math.sin(yaw) * dist, mapdata.min_y), mapdata.max_y)
        if math.hypot(x2 - x, y2 - y) > 200.0:
            continue
        z = mapdata.ground_z(x, y) + 8.0
        z2 = mapdata.ground_z(x2, y2) + 8.0
        pairs.append(((x, y, z), (x2, y2, z2)))
    started = time.perf_counter()
    for a, b in pairs:
        mapdata.los(a, b)
    average = (time.perf_counter() - started) / len(pairs)
    assert average < 0.0002
