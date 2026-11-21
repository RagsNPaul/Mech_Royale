"""Build-time map precompute (runs in `docker build`, network on, 10 min budget).

Inputs: arena/v2/world.json, arena/v2/terrain.i16 (protocol.md §11).
Output: data/map.npz with
  - ground (float32 grid, 2 m cells), solid_top (float32, NaN = open), blocked (bool for walking)
  - slope mask (> max_slope_deg), cover_points (N x 2 float32) + shield directions (N x 8 bool)
  - grid metadata (minX, minY, cell, nx, ny)
Keep output well under the 20 MB repo/image comfort zone (it is generated at build, not committed).

TODO(cursor): implement. Until then this stub exits 0 so the image still builds.
"""
import argparse
import sys

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(f"precompute_map: not implemented yet ({a.arena} -> {a.out})", file=sys.stderr)
