# Mech Royale bot

Deterministic, survival-first bot for the mechfight/1 battle royale. Score = mean(16 - place).
Rules: `docs/protocol.md`. Every tunable number: `bot/params.py`.

## Module map

| File | Role |
|---|---|
| `main.py` | Entrypoint. Wraps the brain in try/except; on any error returns a safe "walk to next zone" action. |
| `mechfight.schema.json` | Message schema (check.py reads it from the repo root). |
| `mechfight.py` | Kit wrapper (stdin/stdout loop, types, debug draw). Don't edit. |
| `bot/config.py` | `Rules`: every game number read from `hello.config` (never hard-code). |
| `bot/params.py` | `PARAMS`: our thresholds. One submission = one change. |
| `bot/mapdata.py` | Loads `data/map.npz`; `ground_z`, `los`, cover points. |
| `bot/nav.py` | A* on the 2 m grid, path following, `best_cover`. |
| `bot/world.py` | Enemy tracks, sound hints, damage events, regen timer, squad-vs-FFA flag. |
| `bot/threat.py` | DPS / TTK math, risk margin, `should_fight`, `pick_target`. Pure functions. |
| `bot/brain.py` | State machine: ZONE > FIGHT > DISENGAGE > DROP > LOOT > HIDE, with 1.5 s hold. |
| `bot/modes/*.py` | One file per mode, each returns an `Intent`. |
| `bot/control.py` | `Intent` -> protocol action: lead aim, weapon choice, strafing, reload. |
| `bot/upgrades.py` | Upgrade offer picks. |
| `tools/precompute_map.py` | Runs during `docker build`: map grids + cover points -> `data/map.npz`. |
| `tools/run_check.sh` | Conformance check, plain and in the real sandbox. |
| `tests/` | Unit tests for the pure math. |

## Modes

| Mode | When | Does |
|---|---|---|
| ZONE | Time-to-safety is tight | Path into the next circle; boost if needed |
| FIGHT | `should_fight` true | Face target, lead shots, strafe, lowest-HP target |
| DISENGAGE | Losing or outnumbered | Break LOS behind buildings while facing attacker, then HIDE |
| DROP | Drop landed, area clear | Hold the ring alone 4 s: +3 cubes, full HP/energy |
| LOOT | Early / no threats | 2nd weapon, crates, NPC drones (cheap cubes) |
| HIDE | Default | Cover inside next circle, move < 5 m/s (silent), no firing, heal |

Fight rule: `my_ttk < margin * their_ttk`, margin = base + reward + finish + desperation - crowd.
Overrides: finish low-HP targets, fight when cornered, avoid HP trades in the final minute.

## Build order

1. **v1 (get on the board):** config, mapdata + precompute, nav, world, ZONE/HIDE/DISENGAGE, basic FIGHT.
2. **v2:** risk-budget margin, finish/cornered/final-minute overrides, LOOT with NPC farming.
3. **v3:** DROP, late-zone desperation, FFA detection, mortar lobbing, upgrade tuning.

## Run

```sh
./tools/run_check.sh   # must PASS before every push
```
Submit: push to a public repo (Dockerfile at top level, < 20 MB), submit URL on the track page. Log each submission's score + the one change in a sheet.
