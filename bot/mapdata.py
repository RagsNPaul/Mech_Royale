"""Static map queries. Loads the grids precomputed at docker build (data/map.npz).

Frame: x east, y north, z up, meters (protocol.md §5, §11).

TODO(cursor):
- load(path) -> MapData. If the npz is missing, fall back to parsing /arena directly (slow, log to stderr).
- ground_z(x, y): bilinear terrain height.
- solid_top(x, y): highest prism top covering (x, y) or None (buildings/stadium; bridges have base > ground).
- walkable(ix, iy) / blocked grid for nav.
- los(a, b, for_vision: bool) -> bool: grid DDA raycast against prisms;
  for_vision also attenuates through tree canopy (trees never block shots).
- cover_points: precomputed array of (x, y) cells next to buildings, with which directions they shield.
- Keep each call cheap: per tick we can afford ~10-30 raycasts in Python.
"""
from dataclasses import dataclass


@dataclass
class MapData:
    pass  # TODO(cursor): grids + metadata (origin, cell size, nx, ny)


def load(npz_path: str, arena_dir: str) -> MapData:
    raise NotImplementedError
