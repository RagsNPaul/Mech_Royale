"""Static map queries. Loads the grids precomputed at docker build (data/map.npz).

Frame: x east, y north, z up, meters (protocol.md §5, §11).
Shots are blocked by building, stadium, and bridge prisms. Trees block vision only.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_COMPASS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


class MapData:
    def __init__(self, arrays: dict[str, Any]):
        self.ground = arrays["ground"]
        self.solid_base = arrays["solid_base"]
        self.solid_top = arrays["solid_top"]
        self.blocked = arrays["blocked"]
        self.steep = arrays["steep"]
        self.cover_points = arrays["cover_xy"]
        self.cover_shield = arrays["cover_shield"]
        self.min_x = float(arrays["min_x"])
        self.min_y = float(arrays["min_y"])
        self.max_x = float(arrays["max_x"])
        self.max_y = float(arrays["max_y"])
        self.cell = float(arrays["cell"])
        self.nx = int(arrays["nx"])
        self.ny = int(arrays["ny"])
        self.mech_height = float(arrays["mech_height"])
        self.max_slope_deg = float(arrays["max_slope_deg"])
        self.canopy_density = float(arrays["canopy_density"])
        self.canopy_block_t = float(arrays["canopy_block_t"])
        self.tree_x = arrays["tree_x"]
        self.tree_y = arrays["tree_y"]
        self.tree_z = arrays["tree_z"]
        self.tree_height = arrays["tree_height"]
        self.tree_crown_r = arrays["tree_crown_r"]
        self.tree_crown_base = arrays["tree_crown_base"]
        self.tree_trunk_r = arrays["tree_trunk_r"]
        self.tree_crown_x = arrays["tree_crown_x"]
        self.tree_crown_y = arrays["tree_crown_y"]
        self._index_trees()

    def _index_trees(self) -> None:
        self.bucket = 32.0
        crowns = self.tree_crown_r
        self.tree_reach = float(crowns.max()) + 1.0 if crowns.size else 1.0
        buckets: dict[tuple[int, int], list[int]] = {}
        for index, (x, y) in enumerate(zip(self.tree_x, self.tree_y, strict=True)):
            key = (math.floor(float(x) / self.bucket), math.floor(float(y) / self.bucket))
            buckets.setdefault(key, []).append(index)
        self.tree_buckets = buckets

    def in_bounds(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def cell_of(self, x: float, y: float) -> tuple[int, int] | None:
        ix = math.floor((x - self.min_x) / self.cell)
        iy = math.floor((y - self.min_y) / self.cell)
        if ix < 0 or iy < 0 or ix >= self.nx or iy >= self.ny:
            return None
        return ix, iy

    def cell_center(self, ix: int, iy: int) -> tuple[float, float]:
        return (self.min_x + (ix + 0.5) * self.cell, self.min_y + (iy + 0.5) * self.cell)

    def walkable(self, ix: int, iy: int) -> bool:
        if ix < 0 or iy < 0 or ix >= self.nx or iy >= self.ny:
            return False
        return not bool(self.blocked[iy, ix])

    def _sample(self, grid: np.ndarray, x: float, y: float) -> float:
        """Bilinear sample of a grid stored at cell centers."""
        gx = (x - self.min_x) / self.cell - 0.5
        gy = (y - self.min_y) / self.cell - 0.5
        gx = min(max(gx, 0.0), self.nx - 1)
        gy = min(max(gy, 0.0), self.ny - 1)
        x0 = int(math.floor(gx))
        y0 = int(math.floor(gy))
        x1 = min(x0 + 1, self.nx - 1)
        y1 = min(y0 + 1, self.ny - 1)
        tx = gx - x0
        ty = gy - y0
        g00 = float(grid[y0, x0])
        g10 = float(grid[y0, x1])
        g01 = float(grid[y1, x0])
        g11 = float(grid[y1, x1])
        return (1.0 - tx) * (1.0 - ty) * g00 + tx * (1.0 - ty) * g10 + (1.0 - tx) * ty * g01 + tx * ty * g11

    def ground_z(self, x: float, y: float) -> float:
        return self._sample(self.ground, x, y)

    def solid_top_at(self, x: float, y: float) -> float | None:
        cell = self.cell_of(x, y)
        if cell is None:
            return None
        ix, iy = cell
        top = float(self.solid_top[iy, ix])
        if top != top:
            return None
        return top

    def solid_base_at(self, x: float, y: float) -> float | None:
        cell = self.cell_of(x, y)
        if cell is None:
            return None
        ix, iy = cell
        base = float(self.solid_base[iy, ix])
        if base != base:
            return None
        return base

    def surface_z(self, x: float, y: float) -> float:
        top = self.solid_top_at(x, y)
        if top is None:
            return self.ground_z(x, y)
        return top

    def is_blocked(self, x: float, y: float) -> bool:
        cell = self.cell_of(x, y)
        if cell is None:
            return True
        ix, iy = cell
        return bool(self.blocked[iy, ix])

    def los(self, a: Any, b: Any, for_vision: bool = False) -> bool:
        """True when the segment does not hit a solid (and, for vision, a trunk or dense crown)."""
        ax, ay, az = float(a[0]), float(a[1]), float(a[2])
        bx, by, bz = float(b[0]), float(b[1]), float(b[2])
        if self._hits_solid(ax, ay, az, bx, by, bz):
            return False
        if for_vision and self._hits_foliage(ax, ay, az, bx, by, bz):
            return False
        return True

    def _hits_solid(self, ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> bool:
        cell = self.cell
        gx0 = (ax - self.min_x) / cell
        gy0 = (ay - self.min_y) / cell
        gx1 = (bx - self.min_x) / cell
        gy1 = (by - self.min_y) / cell
        dx = gx1 - gx0
        dy = gy1 - gy0
        ix = math.floor(gx0)
        iy = math.floor(gy0)
        sx = 1 if dx > 0.0 else -1 if dx < 0.0 else 0
        sy = 1 if dy > 0.0 else -1 if dy < 0.0 else 0
        if sx == 0:
            t_delta_x = math.inf
            t_max_x = math.inf
        else:
            t_delta_x = abs(1.0 / dx)
            boundary = (math.floor(gx0) + 1) if sx > 0 else math.floor(gx0)
            t_max_x = (boundary - gx0) / dx
        if sy == 0:
            t_delta_y = math.inf
            t_max_y = math.inf
        else:
            t_delta_y = abs(1.0 / dy)
            boundary = (math.floor(gy0) + 1) if sy > 0 else math.floor(gy0)
            t_max_y = (boundary - gy0) / dy

        base = self.solid_base
        top = self.solid_top
        nx = self.nx
        ny = self.ny
        t_enter = 0.0
        limit = abs(math.floor(gx1) - ix) + abs(math.floor(gy1) - iy) + 3
        for _ in range(limit):
            if 0 <= ix < nx and 0 <= iy < ny:
                sb = float(base[iy, ix])
                st = float(top[iy, ix])
                if sb == sb and st == st:
                    t_exit = t_max_x if t_max_x < t_max_y else t_max_y
                    if t_exit > 1.0:
                        t_exit = 1.0
                    if t_exit > t_enter:
                        z0 = az + (bz - az) * t_enter
                        z1 = az + (bz - az) * t_exit
                        z_lo = z0 if z0 < z1 else z1
                        z_hi = z1 if z1 > z0 else z0
                        if z_hi > sb and z_lo < st:
                            return True
            if t_max_x > 1.0 and t_max_y > 1.0:
                break
            if t_max_x < t_max_y:
                ix += sx
                t_enter = t_max_x
                t_max_x += t_delta_x
            else:
                iy += sy
                t_enter = t_max_y
                t_max_y += t_delta_y
        return False

    def _tree_ids(self, ax: float, ay: float, bx: float, by: float) -> list[int]:
        reach = self.tree_reach
        ix0 = math.floor((min(ax, bx) - reach) / self.bucket)
        ix1 = math.floor((max(ax, bx) + reach) / self.bucket)
        iy0 = math.floor((min(ay, by) - reach) / self.bucket)
        iy1 = math.floor((max(ay, by) + reach) / self.bucket)
        found: list[int] = []
        buckets = self.tree_buckets
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                found.extend(buckets.get((ix, iy), ()))
        return found

    def _hits_foliage(self, ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> bool:
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        length = math.hypot(dx, dy, dz)
        if length < 1e-8:
            return False
        tau = 0.0
        density = self.canopy_density
        limit = self.canopy_block_t
        xs = self.tree_x
        ys = self.tree_y
        zs = self.tree_z
        heights = self.tree_height
        crowns = self.tree_crown_r
        crown_bases = self.tree_crown_base
        trunks = self.tree_trunk_r
        cxs = self.tree_crown_x
        cys = self.tree_crown_y
        for index in self._tree_ids(ax, ay, bx, by):
            tx = float(xs[index])
            ty = float(ys[index])
            tz = float(zs[index])
            trunk_r = float(trunks[index])
            crown_base = float(crown_bases[index])
            if trunk_r > 0.0 and _hits_cylinder(ax, ay, az, dx, dy, dz, tx, ty, tz, tz + max(crown_base, trunk_r), trunk_r):
                return True
            crown_r = float(crowns[index])
            height = float(heights[index])
            if crown_r <= 0.0 or height <= crown_base:
                continue
            rz = max((height - crown_base) * 0.5, 0.3)
            chord = _ellipsoid_chord(
                ax, ay, az, dx, dy, dz,
                float(cxs[index]), float(cys[index]), tz + crown_base + rz,
                crown_r, crown_r, rz,
            )
            if chord <= 0.0:
                continue
            tau += density * chord / max(2.0 * crown_r, 1e-3)
            if tau >= limit:
                return True
        return False


def _hits_cylinder(
    ax: float, ay: float, az: float,
    dx: float, dy: float, dz: float,
    cx: float, cy: float, z0: float, z1: float, radius: float,
) -> bool:
    ox = ax - cx
    oy = ay - cy
    a = dx * dx + dy * dy
    if a < 1e-12:
        return False
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    root = math.sqrt(disc)
    inv = 0.5 / a
    t0 = max(0.0, (-b - root) * inv)
    t1 = min(1.0, (-b + root) * inv)
    if t1 <= t0:
        return False
    za = az + dz * t0
    zb = az + dz * t1
    z_lo = za if za < zb else zb
    z_hi = zb if zb > za else za
    return z_hi > z0 and z_lo < z1


def _ellipsoid_chord(
    ax: float, ay: float, az: float,
    dx: float, dy: float, dz: float,
    cx: float, cy: float, cz: float,
    rx: float, ry: float, rz: float,
) -> float:
    ox = (ax - cx) / rx
    oy = (ay - cy) / ry
    oz = (az - cz) / rz
    ddx = dx / rx
    ddy = dy / ry
    ddz = dz / rz
    a = ddx * ddx + ddy * ddy + ddz * ddz
    if a < 1e-12:
        return 0.0
    b = 2.0 * (ox * ddx + oy * ddy + oz * ddz)
    c = ox * ox + oy * oy + oz * oz - 1.0
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return 0.0
    root = math.sqrt(disc)
    inv = 0.5 / a
    t0 = max(0.0, (-b - root) * inv)
    t1 = min(1.0, (-b + root) * inv)
    if t1 <= t0:
        return 0.0
    return (t1 - t0) * math.hypot(dx, dy, dz)


def _run_precompute(arena_dir: str, out_path: str) -> None:
    script = Path(__file__).resolve().parents[1] / "tools" / "precompute_map.py"
    spec = importlib.util.spec_from_file_location("precompute_map", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build_map(arena_dir, out_path)


def load(npz_path: str, arena_dir: str) -> MapData:
    path = Path(npz_path)
    if not path.is_file():
        fallback = Path("/tmp") / "mech-map.npz"
        print(f"mapdata: {path} missing; precomputing into {fallback}", file=sys.stderr)
        _run_precompute(arena_dir, str(fallback))
        path = fallback
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    return MapData(arrays)
