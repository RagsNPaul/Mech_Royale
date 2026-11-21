"""Build-time map precompute (runs in `docker build`, network on, 10 min budget).

Inputs: arena/v2/world.json, arena/v2/terrain.i16 (protocol.md §5, §7, §11).
Output: data/map.npz (compressed) with a 2 m grid:
  ground, solid_base, solid_top, blocked, steep,
  cover_xy (N x 2) and cover_shield (N x 8),
  tree columns, and grid metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

CELL = 2.0
MECH_HEIGHT = 9.0
# E, NE, N, NW, W, SW, S, SE. Index i is +x (east), index j is +y (north).
COMPASS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
TREE_FIELDS = ("x", "y", "z", "height", "crownR", "crownBase", "trunkR", "crownX", "crownY")


def _bilinear(height_m: np.ndarray, origin_x: float, origin_y: float, step: float, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample a 1 m terrain grid at cell centers xs (nx,), ys (ny,)."""
    max_x = height_m.shape[1] - 1
    max_y = height_m.shape[0] - 1
    fx = np.clip((xs - origin_x) / step, 0.0, max_x)
    fy = np.clip((ys - origin_y) / step, 0.0, max_y)
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.minimum(x0 + 1, max_x)
    y1 = np.minimum(y0 + 1, max_y)
    tx = (fx - x0).astype(np.float32)[None, :]
    ty = (fy - y0).astype(np.float32)[:, None]
    g00 = height_m[y0[:, None], x0[None, :]]
    g10 = height_m[y0[:, None], x1[None, :]]
    g01 = height_m[y1[:, None], x0[None, :]]
    g11 = height_m[y1[:, None], x1[None, :]]
    return ((1.0 - tx) * (1.0 - ty) * g00 + tx * (1.0 - ty) * g10 + (1.0 - tx) * ty * g01 + tx * ty * g11).astype(np.float32)


def _ring_mask(xs: np.ndarray, ys: np.ndarray, ring: list[float]) -> np.ndarray:
    """Even-odd fill. xs, ys are 1d coordinates of the bbox cell centers; returns (ny, nx) bool."""
    px = np.asarray(ring[0::2], dtype=np.float64)
    py = np.asarray(ring[1::2], dtype=np.float64)
    x = xs[None, :]
    y = ys[:, None]
    inside = np.zeros((ys.shape[0], xs.shape[0]), dtype=bool)
    j = len(px) - 1
    for i in range(len(px)):
        yi = py[i]
        yj = py[j]
        xi = px[i]
        xj = px[j]
        dy = yj - yi
        # Horizontal edges never cross; keep the denominator nonzero for the other branch.
        x_hit = (xj - xi) * (y - yi) / (dy if dy != 0.0 else 1.0) + xi
        crossed = ((yi > y) != (yj > y)) & (x < x_hit)
        inside ^= crossed
        j = i
    return inside


def _paint(mask: np.ndarray, i0: int, j0: int, inside: np.ndarray) -> None:
    j1 = j0 + inside.shape[0]
    i1 = i0 + inside.shape[1]
    mask[j0:j1, i0:i1] |= inside


def _steep(ground: np.ndarray, cell: float, max_slope_deg: float) -> np.ndarray:
    limit = math.tan(math.radians(max_slope_deg)) * cell
    steep = np.zeros(ground.shape, dtype=bool)
    dz = np.abs(ground[:, 1:] - ground[:, :-1])
    hit = dz > limit
    steep[:, 1:] |= hit
    steep[:, :-1] |= hit
    dz = np.abs(ground[1:, :] - ground[:-1, :])
    hit = dz > limit
    steep[1:, :] |= hit
    steep[:-1, :] |= hit
    diag = limit * math.sqrt(2.0)
    dz = np.abs(ground[1:, 1:] - ground[:-1, :-1])
    hit = dz > diag
    steep[1:, 1:] |= hit
    steep[:-1, :-1] |= hit
    dz = np.abs(ground[1:, :-1] - ground[:-1, 1:])
    hit = dz > diag
    steep[1:, :-1] |= hit
    steep[:-1, 1:] |= hit
    return steep


def build_map(arena_dir: str | Path, out_path: str | Path) -> None:
    started = time.perf_counter()
    arena = Path(arena_dir)
    world_path = arena / "v2" / "world.json"
    with world_path.open(encoding="utf-8") as handle:
        world = json.load(handle)

    bounds = world["bounds"]
    min_x = float(bounds["minX"])
    min_y = float(bounds["minY"])
    max_x = float(bounds["maxX"])
    max_y = float(bounds["maxY"])
    nx = int(math.ceil((max_x - min_x) / CELL))
    ny = int(math.ceil((max_y - min_y) / CELL))

    terrain = world["terrain"]
    t_nx = int(terrain["nx"])
    t_ny = int(terrain["ny"])
    raw = np.fromfile(arena / "v2" / terrain["file"], dtype="<i2")
    if raw.size != t_nx * t_ny:
        raise ValueError(f"terrain size {raw.size} != {t_nx}*{t_ny}")
    height_m = raw.reshape(t_ny, t_nx).astype(np.float32) * float(terrain.get("scale", 0.01))
    xs = min_x + (np.arange(nx, dtype=np.float64) + 0.5) * CELL
    ys = min_y + (np.arange(ny, dtype=np.float64) + 0.5) * CELL
    ground = _bilinear(height_m, float(terrain["minX"]), float(terrain["minY"]), float(terrain["step"]), xs, ys)

    solid_base = np.full((ny, nx), np.nan, dtype=np.float32)
    solid_top = np.full((ny, nx), np.nan, dtype=np.float32)
    has_building = np.zeros((ny, nx), dtype=bool)
    geom_blocked = np.zeros((ny, nx), dtype=bool)
    tall = np.zeros((ny, nx), dtype=bool)

    for prism in world["prisms"]:
        kind = prism["kind"]
        if kind not in ("building", "stadium", "bridge"):
            continue
        ring = prism["ring"]
        poly_x = ring[0::2]
        poly_y = ring[1::2]
        i0 = max(0, math.floor((min(poly_x) - min_x) / CELL) - 1)
        i1 = min(nx - 1, math.floor((max(poly_x) - min_x) / CELL) + 1)
        j0 = max(0, math.floor((min(poly_y) - min_y) / CELL) - 1)
        j1 = min(ny - 1, math.floor((max(poly_y) - min_y) / CELL) + 1)
        if i1 < i0 or j1 < j0:
            continue
        inside = _ring_mask(xs[i0 : i1 + 1], ys[j0 : j1 + 1], ring)
        for hole in prism.get("holes") or []:
            inside &= ~_ring_mask(xs[i0 : i1 + 1], ys[j0 : j1 + 1], hole)
        if not inside.any():
            continue

        base = float(prism["base"])
        top = float(prism["top"])
        sub_ground = ground[j0 : j1 + 1, i0 : i1 + 1]
        clearance = base - sub_ground
        buried = top <= sub_ground + 0.05
        blocks_walk = inside & ~buried & (clearance < MECH_HEIGHT)
        geom_blocked[j0 : j1 + 1, i0 : i1 + 1] |= blocks_walk

        sub_base = solid_base[j0 : j1 + 1, i0 : i1 + 1]
        sub_top = solid_top[j0 : j1 + 1, i0 : i1 + 1]
        sub_building = has_building[j0 : j1 + 1, i0 : i1 + 1]
        if kind == "bridge":
            open_bridge = inside & ~sub_building & (np.isnan(sub_top) | (top > sub_top))
            sub_base[open_bridge] = base
            sub_top[open_bridge] = top
            stacked = inside & sub_building & (base <= sub_top + 1.0) & (top > sub_top)
            sub_top[stacked] = top
        else:
            fresh = inside & ~sub_building
            sub_base[fresh] = base
            sub_top[fresh] = top
            sub_building[fresh] = True
            higher = inside & sub_building & (top > sub_top)
            sub_top[higher] = top
            lower = inside & sub_building & (base < sub_base)
            sub_base[lower] = base
            tall[j0 : j1 + 1, i0 : i1 + 1] |= inside & ((top - sub_ground) >= MECH_HEIGHT)

    rules = world.get("rules") or {}
    max_slope = float(rules.get("maxSlopeDeg", 50.0))
    steep = _steep(ground, CELL, max_slope)
    blocked = geom_blocked | steep

    pad = np.pad(tall, 1, constant_values=False)
    shield = np.zeros((ny, nx, 8), dtype=bool)
    for index, (di, dj) in enumerate(COMPASS):
        shield[:, :, index] = pad[1 + dj : 1 + dj + ny, 1 + di : 1 + di + nx]
    cover = (~blocked) & shield.any(axis=2)
    cy, cx = np.nonzero(cover)
    cover_xy = np.column_stack(
        (min_x + (cx.astype(np.float64) + 0.5) * CELL, min_y + (cy.astype(np.float64) + 0.5) * CELL)
    ).astype(np.float32)
    cover_shield = shield[cy, cx]

    trees = world.get("trees") or {}
    fields = list(trees.get("fields") or [])
    flat = trees.get("data") or []
    columns = {name: np.zeros(0, dtype=np.float32) for name in TREE_FIELDS}
    if fields and flat:
        table = np.asarray(flat, dtype=np.float64).reshape(-1, len(fields))
        for name in TREE_FIELDS:
            columns[name] = table[:, fields.index(name)].astype(np.float32)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ground=ground,
        solid_base=solid_base,
        solid_top=solid_top,
        blocked=blocked,
        steep=steep,
        cover_xy=cover_xy,
        cover_shield=cover_shield,
        tree_x=columns["x"],
        tree_y=columns["y"],
        tree_z=columns["z"],
        tree_height=columns["height"],
        tree_crown_r=columns["crownR"],
        tree_crown_base=columns["crownBase"],
        tree_trunk_r=columns["trunkR"],
        tree_crown_x=columns["crownX"],
        tree_crown_y=columns["crownY"],
        min_x=np.float64(min_x),
        min_y=np.float64(min_y),
        max_x=np.float64(max_x),
        max_y=np.float64(max_y),
        cell=np.float64(CELL),
        nx=np.int32(nx),
        ny=np.int32(ny),
        mech_height=np.float64(MECH_HEIGHT),
        max_slope_deg=np.float64(max_slope),
        canopy_density=np.float64(float(rules.get("canopyDensity", 0.35))),
        canopy_block_t=np.float64(float(rules.get("canopyBlockT", 0.25))),
    )
    elapsed = time.perf_counter() - started
    size_mb = out.stat().st_size / (1024 * 1024)
    print(
        f"precompute_map: wrote {out} ({size_mb:.2f} MB, {elapsed:.1f}s, {nx}x{ny})",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_map(args.arena, args.out)


if __name__ == "__main__":
    main()
