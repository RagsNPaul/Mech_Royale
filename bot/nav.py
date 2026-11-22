"""Navigation on the precomputed 2 m walk grid.

astar is 8-connected with an octile heuristic and a hard expansion cap.
Navigator keeps the current path and only replans inside the tick budget.
best_cover picks a shielded point inside the next zone circle.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Any, NamedTuple

import numpy as np

from bot.params import PARAMS

MAX_EXPANSIONS = 20_000
GOAL_MOVE_M = 6.0
WAYPOINT_REACH_M = 2.0
STUCK_WINDOW_S = 1.5
STUCK_MOVE_M = 0.5
ZONE_MARGIN_M = 15.0
SNAP_RADIUS = 48
_ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBORS = (
    (1, 0, 10),
    (-1, 0, 10),
    (0, 1, 10),
    (0, -1, 10),
    (1, 1, 14),
    (1, -1, 14),
    (-1, 1, 14),
    (-1, -1, 14),
)


class Steer(NamedTuple):
    move: tuple[float, float]
    jump: bool


class _Scratch:
    def __init__(self, size: int):
        self.stamp = np.zeros(size, np.int32)
        self.closed = np.zeros(size, np.int32)
        self.g = np.zeros(size, np.int32)
        self.parent = np.full(size, -1, np.int32)
        self.gen = 1

    def next_gen(self) -> int:
        self.gen += 1
        if self.gen >= 2_000_000_000:
            self.stamp.fill(0)
            self.closed.fill(0)
            self.gen = 1
        return self.gen


_SCRATCH: dict[int, _Scratch] = {}


def _scratch(size: int, key: int) -> _Scratch:
    found = _SCRATCH.get(key)
    if found is None or found.stamp.size != size:
        found = _Scratch(size)
        _SCRATCH[key] = found
    return found


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _unit(src: tuple[float, float], dst: tuple[float, float]) -> tuple[float, float]:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (0.0, 0.0)
    return (dx / length, dy / length)


def _snap(mapdata: Any, xy: tuple[float, float]) -> tuple[int, int] | None:
    origin = mapdata.cell_of(xy[0], xy[1])
    if origin is None:
        return None
    if mapdata.walkable(origin[0], origin[1]):
        return origin
    blocked = mapdata.blocked
    nx = mapdata.nx
    ny = mapdata.ny
    queue = [origin]
    seen = {origin}
    head = 0
    while head < len(queue):
        ix, iy = queue[head]
        head += 1
        if max(abs(ix - origin[0]), abs(iy - origin[1])) > SNAP_RADIUS:
            continue
        for dx, dy in _ORTHO:
            jx, jy = ix + dx, iy + dy
            if jx < 0 or jy < 0 or jx >= nx or jy >= ny or (jx, jy) in seen:
                continue
            if max(abs(jx - origin[0]), abs(jy - origin[1])) > SNAP_RADIUS:
                continue
            seen.add((jx, jy))
            if not blocked[jy, jx]:
                return (jx, jy)
            queue.append((jx, jy))
    return None


def _line_clear(mapdata: Any, start: tuple[int, int], end: tuple[int, int]) -> bool:
    """True when every grid cell the center-to-center segment crosses is walkable."""
    blocked = mapdata.blocked
    nx = mapdata.nx
    ny = mapdata.ny
    x0 = start[0] + 0.5
    y0 = start[1] + 0.5
    x1 = end[0] + 0.5
    y1 = end[1] + 0.5
    dx = x1 - x0
    dy = y1 - y0
    ix = start[0]
    iy = start[1]
    sx = 1 if dx > 0.0 else -1 if dx < 0.0 else 0
    sy = 1 if dy > 0.0 else -1 if dy < 0.0 else 0
    if sx == 0:
        t_delta_x = math.inf
        t_max_x = math.inf
    else:
        t_delta_x = abs(1.0 / dx)
        boundary = (ix + 1) if sx > 0 else ix
        t_max_x = (boundary - x0) / dx
    if sy == 0:
        t_delta_y = math.inf
        t_max_y = math.inf
    else:
        t_delta_y = abs(1.0 / dy)
        boundary = (iy + 1) if sy > 0 else iy
        t_max_y = (boundary - y0) / dy
    limit = abs(end[0] - start[0]) + abs(end[1] - start[1]) + 3
    for _ in range(limit):
        if ix < 0 or iy < 0 or ix >= nx or iy >= ny or blocked[iy, ix]:
            return False
        if ix == end[0] and iy == end[1]:
            return True
        if abs(t_max_x - t_max_y) <= 1e-9:
            if t_max_x > 1.0:
                break
            side_x = ix + sx
            side_y = iy + sy
            if (
                side_x < 0
                or side_y < 0
                or side_x >= nx
                or side_y >= ny
                or blocked[iy, side_x]
                or blocked[side_y, ix]
            ):
                return False
            ix = side_x
            iy = side_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            if t_max_x > 1.0:
                break
            ix += sx
            t_max_x += t_delta_x
        else:
            if t_max_y > 1.0:
                break
            iy += sy
            t_max_y += t_delta_y
    return False


def _smooth(mapdata: Any, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(cells) <= 2:
        return cells
    kept = [cells[0]]
    index = 0
    last = len(cells) - 1
    while index < last:
        farthest = index + 1
        for candidate in range(last, index + 1, -1):
            if _line_clear(mapdata, cells[index], cells[candidate]):
                farthest = candidate
                break
        kept.append(cells[farthest])
        index = farthest
    return kept


def _reconstruct(parent: np.ndarray, end: int, nx: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    idx = end
    guard = 0
    while idx >= 0 and guard <= MAX_EXPANSIONS + 2:
        iy, ix = divmod(idx, nx)
        cells.append((ix, iy))
        idx = int(parent[idx])
        guard += 1
    cells.reverse()
    return cells


def _octile(dx: int, dy: int) -> int:
    if dx < 0:
        dx = -dx
    if dy < 0:
        dy = -dy
    if dx > dy:
        return 10 * dx + 4 * dy
    return 10 * dy + 4 * dx


def _blocked_bytes(mapdata: Any) -> bytes:
    cached = getattr(mapdata, "_blocked_bytes", None)
    if cached is None:
        cached = mapdata.blocked.ravel().tobytes()
        mapdata._blocked_bytes = cached
    return cached


def _jump_tables(mapdata: Any) -> tuple[np.ndarray, ...]:
    """Next cardinal jump point and blocking cell, built once per grid.

    A jump point is where a classic orthogonal scan would stop: a forced neighbor.
    """
    cached = getattr(mapdata, "_jps_tables", None)
    if cached is not None:
        return cached
    nx = mapdata.nx
    ny = mapdata.ny
    blocked_b = _blocked_bytes(mapdata)

    def free(ix: int, iy: int) -> bool:
        return 0 <= ix < nx and 0 <= iy < ny and blocked_b[iy * nx + ix] == 0

    def stop_east(ix: int, iy: int) -> bool:
        return (free(ix + 1, iy + 1) and not free(ix, iy + 1)) or (
            free(ix + 1, iy - 1) and not free(ix, iy - 1)
        )

    def stop_west(ix: int, iy: int) -> bool:
        return (free(ix - 1, iy + 1) and not free(ix, iy + 1)) or (
            free(ix - 1, iy - 1) and not free(ix, iy - 1)
        )

    def stop_north(ix: int, iy: int) -> bool:
        return (free(ix + 1, iy + 1) and not free(ix + 1, iy)) or (
            free(ix - 1, iy + 1) and not free(ix - 1, iy)
        )

    def stop_south(ix: int, iy: int) -> bool:
        return (free(ix + 1, iy - 1) and not free(ix + 1, iy)) or (
            free(ix - 1, iy - 1) and not free(ix - 1, iy)
        )

    east = np.full(nx * ny, -1, np.int16)
    west = np.full(nx * ny, -1, np.int16)
    north = np.full(nx * ny, -1, np.int16)
    south = np.full(nx * ny, -1, np.int16)
    wall_e = np.full(nx * ny, nx, np.int16)
    wall_w = np.full(nx * ny, -1, np.int16)
    wall_n = np.full(nx * ny, ny, np.int16)
    wall_s = np.full(nx * ny, -1, np.int16)

    for iy in range(ny):
        nxt = -1
        wall = nx
        for ix in range(nx - 1, -1, -1):
            i = iy * nx + ix
            if not free(ix, iy):
                wall = ix
                nxt = -1
                wall_e[i] = ix
                continue
            wall_e[i] = wall
            if ix + 1 >= nx or not free(ix + 1, iy):
                east[i] = -1
            elif stop_east(ix + 1, iy):
                east[i] = ix + 1
            else:
                east[i] = nxt
            nxt = ix if stop_east(ix, iy) else int(east[i])
        nxt = -1
        wall = -1
        for ix in range(nx):
            i = iy * nx + ix
            if not free(ix, iy):
                wall = ix
                nxt = -1
                wall_w[i] = ix
                continue
            wall_w[i] = wall
            if ix - 1 < 0 or not free(ix - 1, iy):
                west[i] = -1
            elif stop_west(ix - 1, iy):
                west[i] = ix - 1
            else:
                west[i] = nxt
            nxt = ix if stop_west(ix, iy) else int(west[i])

    for ix in range(nx):
        nxt = -1
        wall = ny
        for iy in range(ny - 1, -1, -1):
            i = iy * nx + ix
            if not free(ix, iy):
                wall = iy
                nxt = -1
                wall_n[i] = iy
                continue
            wall_n[i] = wall
            if iy + 1 >= ny or not free(ix, iy + 1):
                north[i] = -1
            elif stop_north(ix, iy + 1):
                north[i] = iy + 1
            else:
                north[i] = nxt
            nxt = iy if stop_north(ix, iy) else int(north[i])
        nxt = -1
        wall = -1
        for iy in range(ny):
            i = iy * nx + ix
            if not free(ix, iy):
                wall = iy
                nxt = -1
                wall_s[i] = iy
                continue
            wall_s[i] = wall
            if iy - 1 < 0 or not free(ix, iy - 1):
                south[i] = -1
            elif stop_south(ix, iy - 1):
                south[i] = iy - 1
            else:
                south[i] = nxt
            nxt = iy if stop_south(ix, iy) else int(south[i])

    cached = (east, west, north, south, wall_e, wall_w, wall_n, wall_s)
    mapdata._jps_tables = cached
    return cached


def _jps(
    mapdata: Any,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Jump-point search. Corner-cutting diagonals are repaired when the path is expanded."""
    nx = mapdata.nx
    ny = mapdata.ny
    blocked_b = _blocked_bytes(mapdata)
    east, west, north, south, wall_e, wall_w, wall_n, wall_s = _jump_tables(mapdata)
    gx, gy = goal
    sx, sy = start

    def free(ix: int, iy: int) -> bool:
        return 0 <= ix < nx and 0 <= iy < ny and blocked_b[iy * nx + ix] == 0

    def has_ortho(ix: int, iy: int, dx: int, dy: int) -> bool:
        i = iy * nx + ix
        if dx == 1:
            return (gy == iy and ix < gx < int(wall_e[i])) or east[i] >= 0
        if dx == -1:
            return (gy == iy and int(wall_w[i]) < gx < ix) or west[i] >= 0
        if dy == 1:
            return (gx == ix and iy < gy < int(wall_n[i])) or north[i] >= 0
        return (gx == ix and int(wall_s[i]) < gy < iy) or south[i] >= 0

    def jump_ortho(ix: int, iy: int, dx: int, dy: int) -> tuple[int, int] | None:
        i = iy * nx + ix
        if dx == 1:
            if gy == iy and ix < gx < int(wall_e[i]) and (east[i] < 0 or gx <= int(east[i])):
                return gx, iy
            jx = int(east[i])
            return None if jx < 0 else (jx, iy)
        if dx == -1:
            if gy == iy and int(wall_w[i]) < gx < ix and (west[i] < 0 or gx >= int(west[i])):
                return gx, iy
            jx = int(west[i])
            return None if jx < 0 else (jx, iy)
        if dy == 1:
            if gx == ix and iy < gy < int(wall_n[i]) and (north[i] < 0 or gy <= int(north[i])):
                return ix, gy
            jy = int(north[i])
            return None if jy < 0 else (ix, jy)
        if gx == ix and int(wall_s[i]) < gy < iy and (south[i] < 0 or gy >= int(south[i])):
            return ix, gy
        jy = int(south[i])
        return None if jy < 0 else (ix, jy)

    def jump_diag(ix: int, iy: int, dx: int, dy: int) -> tuple[int, int] | None:
        while True:
            ix += dx
            iy += dy
            if not free(ix, iy):
                return None
            if ix == gx and iy == gy:
                return ix, iy
            if (free(ix - dx, iy + dy) and not free(ix - dx, iy)) or (
                free(ix + dx, iy - dy) and not free(ix, iy - dy)
            ):
                return ix, iy
            if has_ortho(ix, iy, dx, 0) or has_ortho(ix, iy, 0, dy):
                return ix, iy

    def jump(ix: int, iy: int, dx: int, dy: int) -> tuple[int, int] | None:
        if dx != 0 and dy != 0:
            return jump_diag(ix, iy, dx, dy)
        return jump_ortho(ix, iy, dx, dy)

    def directions(ix: int, iy: int, pdx: int, pdy: int) -> tuple[tuple[int, int], ...]:
        if pdx == 0 and pdy == 0:
            found = []
            for dx, dy, _step in _NEIGHBORS:
                if free(ix + dx, iy + dy):
                    found.append((dx, dy))
            return tuple(found)
        found = []
        if pdx != 0 and pdy != 0:
            if free(ix, iy + pdy):
                found.append((0, pdy))
            if free(ix + pdx, iy):
                found.append((pdx, 0))
            if free(ix + pdx, iy + pdy):
                found.append((pdx, pdy))
            if not free(ix - pdx, iy) and free(ix - pdx, iy + pdy):
                found.append((-pdx, pdy))
            if not free(ix, iy - pdy) and free(ix + pdx, iy - pdy):
                found.append((pdx, -pdy))
        else:
            if free(ix + pdx, iy + pdy):
                found.append((pdx, pdy))
            if pdx != 0:
                if not free(ix, iy + 1) and free(ix + pdx, iy + 1):
                    found.append((pdx, 1))
                if not free(ix, iy - 1) and free(ix + pdx, iy - 1):
                    found.append((pdx, -1))
            else:
                if not free(ix + 1, iy) and free(ix + 1, iy + pdy):
                    found.append((1, pdy))
                if not free(ix - 1, iy) and free(ix - 1, iy + pdy):
                    found.append((-1, pdy))
        return tuple(found)

    scratch = _scratch(nx * ny, id(mapdata.blocked))
    gen = scratch.next_gen()
    stamp = scratch.stamp
    closed = scratch.closed
    gscore = scratch.g
    parent = scratch.parent
    start_idx = sy * nx + sx
    goal_idx = gy * nx + gx
    stamp[start_idx] = gen
    gscore[start_idx] = 0
    parent[start_idx] = -1
    h0 = _octile(sx - gx, sy - gy)
    heap: list[tuple[int, int, int]] = [(h0, 0, start_idx)]
    best_idx = start_idx
    best_h = h0
    expansions = 0

    while heap:
        _f, g, idx = heapq.heappop(heap)
        if closed[idx] == gen or stamp[idx] != gen or g != gscore[idx]:
            continue
        closed[idx] = gen
        expansions += 1
        iy, ix = divmod(idx, nx)
        h = _octile(ix - gx, iy - gy)
        if h < best_h:
            best_h = h
            best_idx = idx
        if idx == goal_idx:
            return _reconstruct(parent, idx, nx)
        if expansions >= MAX_EXPANSIONS:
            return _reconstruct(parent, best_idx, nx)
        prev = int(parent[idx])
        if prev < 0:
            pdx = pdy = 0
        else:
            py, px = divmod(prev, nx)
            pdx = (ix > px) - (ix < px)
            pdy = (iy > py) - (iy < py)
        for dx, dy in directions(ix, iy, pdx, pdy):
            found = jump(ix, iy, dx, dy)
            if found is None:
                continue
            jx, jy = found
            nidx = jy * nx + jx
            if closed[nidx] == gen:
                continue
            ng = g + _octile(jx - ix, jy - iy)
            if stamp[nidx] == gen and ng >= gscore[nidx]:
                continue
            stamp[nidx] = gen
            gscore[nidx] = ng
            parent[nidx] = idx
            nh = _octile(jx - gx, jy - gy)
            heapq.heappush(heap, (ng + nh, ng, nidx))
    return _reconstruct(parent, best_idx, nx)


def _search(
    mapdata: Any,
    start: tuple[int, int],
    goal: tuple[int, int] | None,
    targets: dict[int, tuple[float, float]] | None,
    max_cost: float | None,
) -> list[tuple[int, int]] | tuple[float, float] | None:
    """A* on the walk grid. Costs are integer octile units (10 per cell)."""
    nx = mapdata.nx
    ny = mapdata.ny
    blocked_b = _blocked_bytes(mapdata)
    scratch = _scratch(nx * ny, id(mapdata.blocked))
    gen = scratch.next_gen()
    stamp = scratch.stamp
    closed = scratch.closed
    gscore = scratch.g
    parent = scratch.parent
    sx, sy = start
    start_idx = sy * nx + sx
    gx = goal[0] if goal is not None else sx
    gy = goal[1] if goal is not None else sy
    goal_idx = (gy * nx + gx) if goal is not None else -1
    use_targets = targets is not None
    max_g = None if max_cost is None else int(max_cost * 10.0 / mapdata.cell)

    stamp[start_idx] = gen
    gscore[start_idx] = 0
    parent[start_idx] = -1

    def heuristic(ix: int, iy: int) -> int:
        adx = ix - gx
        ady = iy - gy
        if adx < 0:
            adx = -adx
        if ady < 0:
            ady = -ady
        base = 10 * adx + 4 * ady if adx > ady else 10 * ady + 4 * adx
        return base + base

    h0 = 0 if use_targets else heuristic(sx, sy)
    heap: list[tuple[int, int, int]] = [(h0, 0, start_idx)]
    best_idx = start_idx
    best_h = _octile(sx - gx, sy - gy)
    expansions = 0

    while heap:
        _f, g, idx = heapq.heappop(heap)
        if closed[idx] == gen or stamp[idx] != gen or g != gscore[idx]:
            continue
        closed[idx] = gen
        expansions += 1
        iy, ix = divmod(idx, nx)
        if use_targets:
            found = targets.get(idx)
            if found is not None:
                return None if max_g is not None and g > max_g else found
        else:
            h = _octile(ix - gx, iy - gy)
            if h < best_h:
                best_h = h
                best_idx = idx
            if idx == goal_idx:
                return _reconstruct(parent, idx, nx)
        if max_g is not None and g > max_g:
            return None
        if expansions >= MAX_EXPANSIONS:
            if use_targets:
                return None
            return _reconstruct(parent, best_idx, nx)

        for dx, dy, step in _NEIGHBORS:
            jx = ix + dx
            jy = iy + dy
            if jx < 0 or jy < 0 or jx >= nx or jy >= ny:
                continue
            nidx = jy * nx + jx
            if blocked_b[nidx]:
                continue
            if dx != 0 and dy != 0 and (blocked_b[iy * nx + jx] or blocked_b[jy * nx + ix]):
                continue
            if closed[nidx] == gen:
                continue
            ng = g + step
            if stamp[nidx] == gen and ng >= gscore[nidx]:
                continue
            stamp[nidx] = gen
            gscore[nidx] = ng
            parent[nidx] = idx
            nf = ng if use_targets else ng + heuristic(jx, jy)
            heapq.heappush(heap, (nf, ng, nidx))
    if use_targets:
        return None
    return _reconstruct(parent, best_idx, nx)


def _expand_jumps(mapdata: Any, jumps: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Fill cells between jump points, replacing a corner-cutting diagonal with two orthogonal steps."""
    blocked_b = _blocked_bytes(mapdata)
    nx = mapdata.nx

    def free(ix: int, iy: int) -> bool:
        return 0 <= ix < mapdata.nx and 0 <= iy < mapdata.ny and blocked_b[iy * nx + ix] == 0

    full = [jumps[0]]
    for (x0, y0), (x1, y1) in zip(jumps, jumps[1:]):
        dx = (x1 > x0) - (x1 < x0)
        dy = (y1 > y0) - (y1 < y0)
        x, y = x0, y0
        guard = 0
        while (x, y) != (x1, y1):
            guard += 1
            if guard > mapdata.nx + mapdata.ny:
                return full
            nxt_x, nxt_y = x + dx, y + dy
            if dx != 0 and dy != 0 and (not free(nxt_x, y) or not free(x, nxt_y)):
                if not free(nxt_x, nxt_y):
                    return full
                if free(nxt_x, y):
                    full.append((nxt_x, y))
                    full.append((nxt_x, nxt_y))
                    x, y = nxt_x, nxt_y
                    continue
                if free(x, nxt_y):
                    full.append((x, nxt_y))
                    full.append((nxt_x, nxt_y))
                    x, y = nxt_x, nxt_y
                    continue
                return full
            if not free(nxt_x, nxt_y):
                return full
            x, y = nxt_x, nxt_y
            full.append((x, y))
    return full


def astar(mapdata: Any, start_xy: Any, goal_xy: Any) -> list[tuple[float, float]] | None:
    """Waypoints from start to goal. None if either point cannot be snapped onto a free cell.

    A search that hits the expansion cap returns a partial path toward the goal.
    """
    start = _snap(mapdata, (float(start_xy[0]), float(start_xy[1])))
    goal = _snap(mapdata, (float(goal_xy[0]), float(goal_xy[1])))
    if start is None or goal is None:
        return None
    if start == goal:
        return [mapdata.cell_center(start[0], start[1])]
    cells = _jps(mapdata, start, goal)
    expanded = _expand_jumps(mapdata, cells) if cells else None
    if expanded and expanded[-1] == goal:
        cells = expanded
    elif cells and cells[-1] == goal:
        cells = _search(mapdata, start, goal, None, None)
    elif expanded:
        cells = expanded
    if not isinstance(cells, list) or not cells:
        return None
    smoothed = _smooth(mapdata, cells)
    return [mapdata.cell_center(ix, iy) for ix, iy in smoothed]


def time_to_reach(path: list[tuple[float, float]] | None, speed: float) -> float:
    if not path or len(path) < 2:
        return 0.0
    if speed <= 0.0:
        return math.inf
    length = 0.0
    for src, dst in zip(path, path[1:]):
        length += _dist(src, dst)
    return length / speed


def best_cover(
    mapdata: Any,
    me_pos: Any,
    threat_xys: list[Any],
    zone_next_center: Any,
    zone_next_radius: float,
    max_dist: float,
    edge_margin: float | None = None,
    min_dirs: int = 0,
) -> tuple[float, float] | None:
    """Cover inside the next circle. The shortest walk within max_dist wins.

    The default keeps 15 m of margin and requires every threat to be shielded.
    ``edge_margin`` and ``min_dirs`` (walls blocking that many of 8 directions)
    are the farmer's hold spot.
    """
    covers = mapdata.cover_points
    if covers.size == 0 or max_dist <= 0.0:
        return None
    margin = ZONE_MARGIN_M if edge_margin is None else float(edge_margin)
    limit = float(zone_next_radius) - margin
    if limit < 0.0:
        return None
    cx = covers[:, 0].astype(np.float64)
    cy = covers[:, 1].astype(np.float64)
    zx, zy = float(zone_next_center[0]), float(zone_next_center[1])
    mx, my = float(me_pos[0]), float(me_pos[1])
    ok = ((cx - zx) ** 2 + (cy - zy) ** 2 <= limit * limit) & (
        (cx - mx) ** 2 + (cy - my) ** 2 <= max_dist * max_dist
    )
    shields = mapdata.cover_shield
    if int(min_dirs) > 0:
        ok &= shields.sum(axis=1) >= int(min_dirs)
    for threat in threat_xys:
        dx = float(threat[0]) - cx
        dy = float(threat[1]) - cy
        ang = np.arctan2(dy, dx)
        sector = np.floor((ang + math.pi / 8.0) / (math.pi / 4.0)).astype(np.int32) % 8
        covered = shields[np.arange(covers.shape[0]), sector]
        covered = covered | ((np.abs(dx) < 1e-3) & (np.abs(dy) < 1e-3))
        ok &= covered
    chosen = np.flatnonzero(ok)
    if chosen.size == 0:
        return None
    nx = mapdata.nx
    targets: dict[int, tuple[float, float]] = {}
    for index in chosen:
        cell = mapdata.cell_of(float(cx[index]), float(cy[index]))
        if cell is None or not mapdata.walkable(cell[0], cell[1]):
            continue
        targets[cell[1] * nx + cell[0]] = (float(cx[index]), float(cy[index]))
    if not targets:
        return None
    start = _snap(mapdata, (mx, my))
    if start is None:
        return None
    found = _search(mapdata, start, None, targets, max_dist)
    if isinstance(found, tuple):
        return found
    return None


class Navigator:
    def __init__(self, mapdata: Any, rules: Any):
        self.map = mapdata
        self.rules = rules
        self.path: list[tuple[float, float]] = []
        self.goal: tuple[float, float] | None = None
        self.plan_time = -1e9
        self.anchor: tuple[float, float] | None = None
        self.anchor_time = 0.0
        self.commanding = False
        self.jump_next = False

    def _path_ok(self) -> bool:
        if not self.path:
            return False
        for point in self.path:
            cell = self.map.cell_of(point[0], point[1])
            if cell is None or not self.map.walkable(cell[0], cell[1]):
                return False
        return True

    def _stuck(self, pos: tuple[float, float], now: float) -> bool:
        if not self.commanding or self.anchor is None:
            self.anchor = pos
            self.anchor_time = now
            return False
        if now - self.anchor_time < STUCK_WINDOW_S:
            return False
        moved = _dist(pos, self.anchor)
        self.anchor = pos
        self.anchor_time = now
        return moved < STUCK_MOVE_M

    def steer(self, pos: Any, goal: Any, now: float, t0: float) -> Steer:
        """Unit move toward the next waypoint, plus jump on the tick after getting stuck.

        Replans when the goal shifts, the path is blocked, the replan interval elapses,
        or the mech is stuck. Skips the replan once the tick is past plan_budget_ms.
        """
        jump = self.jump_next
        self.jump_next = False
        here = (float(pos[0]), float(pos[1]))
        target = (float(goal[0]), float(goal[1]))
        over_budget = (time.perf_counter() - t0) * 1000.0 > PARAMS["plan_budget_ms"]
        stuck = self._stuck(here, now)
        replan = self.goal is None or _dist(target, self.goal) > GOAL_MOVE_M
        if not replan and now - self.plan_time >= PARAMS["replan_every_s"]:
            replan = True
        if not replan and not self._path_ok():
            replan = True
        if stuck:
            # Jump is returned on the following tick; replan on this one.
            self.jump_next = True
            replan = True
        if replan and not over_budget:
            found = astar(self.map, here, target)
            self.path = list(found) if found else []
            self.goal = target
            self.plan_time = now

        while self.path and _dist(here, self.path[0]) <= WAYPOINT_REACH_M:
            self.path.pop(0)
        if self.path:
            move = _unit(here, self.path[0])
        elif self.goal is None:
            move = _unit(here, target)
        else:
            move = (0.0, 0.0)
        self.commanding = move[0] != 0.0 or move[1] != 0.0
        if not self.commanding:
            self.anchor = here
            self.anchor_time = now
        return Steer(move, jump)
