# Cursor Prompts: Mech Royale bot, from scaffold to submission

How to use this file: do the steps in order. Each step has a **prompt** to paste into Cursor (Agent mode, with the repo open) and a **test gate**. Don't start the next step until the gate passes. Every prompt assumes Cursor can read `docs/protocol.md`, `README.md` and the docstrings in the stub files. Tell it to read them first if it doesn't.

---

## The plan (what we're building)

**Goal:** maximize the mean of `16 − place`, and beat the 1.15 baseline by a wide margin. Survival matters most. Kills are worth taking when the risk is acceptable.

**Algorithm:** a deterministic state machine. No RL, because there's no simulator and the only feedback is the score from each submission. Every threshold lives in `bot/params.py` and is tuned by submitting.

1. **Fight-or-hide rule (`bot/threat.py`)**
   - `my_dps` = Σ over my weapons: damage ÷ cooldown × p_hit(kind, distance) × cube multiplier
   - `their_dps` = the same from their weapons list, × the weapon-triangle modifier, × rear_mult (1.5) if they're outside my front arc
   - `my_ttk = target_hp / my_dps`, `their_ttk = my_hp / Σ their_dps` (all enemies within 120 m)
   - **FIGHT if** `my_ttk < margin × their_ttk` AND hp > 40% AND the zone is safe for 10 s or more
2. **Risk budget (v2):** `margin = base + reward(cubes the kill would drop, if I'm under the cap) + finish(target dies in ≤1.5 s) + desperation(late zone) − crowd(extra enemies)`
3. **Override rules:** always finish low-HP targets; fight when cornered; in the final 60 s, avoid trading HP (the tiebreak is HP).
4. **Modes, in priority order:** ZONE > FIGHT > DISENGAGE > DROP > LOOT > HIDE, with a minimum 1.5 s hold before switching modes.
5. **What the rulebook rewards:** hiding heals 45 HP/s after 5 s without combat. Walking under 5 m/s makes no sound. Retreat while facing the attacker so hits land on your front. NPC drones are cheap cubes. Cubes add max HP, which decides the endgame in the final zone.
6. **Practice vs final:** practice is 1 vs a 15-bot squad (kills move you up places). The final is a free-for-all (patience pays). The bot detects which one it's in from what it observes and switches `margin` base.

**Releases:**
| Release | Contents | Submit after step |
|---|---|---|
| v1 | map, nav, world model, ZONE / HIDE / DISENGAGE / basic FIGHT | Step 9 |
| v2 | risk budget, overrides, LOOT + NPC farming | Step 11 |
| v3 | DROP, late-zone desperation, FFA detection, mortar lobbing | Step 13 |

**Test layers (used at every gate):**
- **Unit tests** (`pytest tests/`): the math and the map, checked against known answers.
- **Replay tests:** feed the 300 real recorded observations from `check/stream.jsonl.gz` through the modules. No exceptions allowed, plus timing limits.
- **Scenario tests:** hand-built observations ("I'm outside the zone", "two enemies behind me at 30% HP") with an assertion on the chosen mode.
- **Conformance:** `python3 check/check.py -- python3 -u main.py`, and `./tools/run_check.sh` (the sandboxed docker run, which is exactly what the scorer runs) before every submission.

**Timing budget:** 50 ms per decision on 1 CPU. Target: **p99 < 15 ms locally** (the match container is slower). Tests enforce this.

---

## Step 0: Environment and baseline

**Prompt:**
```
Set up the dev environment for this repo:
1. Create .venv, install numpy==2.1.3 and pytest (add a requirements-dev.txt with pytest pinned; do NOT add it to requirements.txt).
2. chmod +x tools/run_check.sh.
3. git init, first commit of the scaffold.
Do not change any bot logic.
```

**Test gate:**
```sh
python3 check/check.py -- python3 -u main.py     # PASS (fallback mode)
./tools/run_check.sh                              # PASS inside the docker sandbox
```
Optional: push to a public repo and submit now. The fallback only walks to the zone, but this proves the submission pipeline works end to end.

---

## Step 1: Test fixtures (do this first, every later gate uses them)

**Prompt:**
```
Create tests/fixtures.py and tests/conftest.py:
- load_stream(): read check/stream.jsonl.gz (line 0 = hello, lines 1..300 = obs, last = end) and return (hello, [obs...]).
- pytest fixtures: `hello`, `observations`, `rules` (bot.config.Rules(hello['config'])).
- make_obs(**overrides): build a minimal valid synthetic observation by deep-copying observations[0] and
  applying overrides with helpers:
    with_self(hp=, pos=, aim=, weapons=, cubes=, in_zone=)
    with_enemy(id, pos, hp, weapons, aim, cubes)          # appends to obs['mechs']
    with_damage_event(by, amount, bearing, rear)
    with_zone(center, radius, next_center, next_radius, shrink_start, shrink_end, dps)
    at_time(t)
- tests/test_fixtures.py: sanity tests that the stream loads, 300 observations, make_obs output still validates
  against mechfight.schema.json (ServerMessage) using a small hand-written validator or jsonschema if
  available in dev deps (add jsonschema pinned to requirements-dev.txt).
```

**Test gate:** `pytest tests/test_fixtures.py -q` is all green.

---

## Step 2: `bot/config.py` (Rules)

**Prompt:**
```
Implement bot/config.py per its docstring. Rules(config) exposes typed attributes for everything the bot
uses: mech (radius, height, eye/muzzle heights, walk_speed, boost_speed, turn_rate, front_arc_deg, rear_mult,
regen_delay, regen_per_s, step_up, max_slope_deg), weapons dict keyed by kind (damage, cooldown, speed, range,
mag, reload, spread_deg, charge/min_frac/min_charge for laser, gravity/splash for mortar,
cone_deg/chain/energy for lightning), vision, sound ranges + step_speed, zone, cubes (stat_cap,
damage_per_cube, hp_per_cube, size_per_cube, upgrade_thresholds, pickup_radius), loot.pickup_radius,
drops, npc. Inspect the real hello in check/stream.jsonl.gz to get exact key names; never guess.
Missing keys -> fall back to the protocol.md defaults and log once to stderr. Keep .raw.
Add tests/test_config.py: every attribute is populated from the real hello; gun damage 45, cooldown 0.24,
speed 252; rear_mult 1.5; front_arc_deg 180; Rules({}) does not crash (defaults used).
```

**Test gate:** `pytest tests/test_config.py -q` passes, and `check.py` still PASSES.

---

## Step 3: Map precompute + `bot/mapdata.py`

**Prompt:**
```
Implement tools/precompute_map.py and bot/mapdata.py per their docstrings and protocol.md §5, §7, §11.
precompute (numpy only):
- 2 m grid over world.json bounds. ground = bilinear-sampled terrain.i16 (int16 cm, nx*ny row-major from minX,minY, 1 m step).
- solid_top / solid_base per cell from prisms (point-in-polygon with holes, kinds building/stadium/bridge).
  Bridges: base above ground -> walkable underneath if clearance >= mech height (9 m).
- blocked[cell] for walking: solid from ground up to head height, or slope > max_slope_deg vs neighbors.
- cover_points: free walkable cells adjacent to a building whose top is >= 9 m above ground there;
  store for each the 8 compass directions in which the building shields it.
- trees: store x, y, z, height, crownR, crownBase, trunkR arrays for vision checks.
- save data/map.npz (compressed) + metadata. Must run in < 2 min and output < 15 MB.
mapdata.load(npz, arena_dir): load npz (if missing, run the precompute in-process into /tmp and log).
Queries: ground_z(x,y), surface_z(x,y) (roof if any), is_blocked(x,y), cell helpers,
los(a, b, for_vision=False): DDA over the 2 m grid; blocked if the segment height at a cell is between
solid_base and solid_top; for_vision also fails on tree trunks and on dense canopy per world.json rules.
Add tests/test_mapdata.py:
- npz builds, size < 15 MB, loads in < 1.5 s
- ground_z at map origin ~ 0; finite everywhere inside bounds
- pick a large building prism from world.json: its centroid is_blocked; a point 30 m outside its ring is not
- los between two points on opposite sides of that building at z=ground+8 is False; los between two
  open-ground points 20 m apart is True
- 1000 random los calls of length <= 200 m average < 0.2 ms each
```

**Test gate:** `pytest tests/test_mapdata.py -q` passes. Also confirm `docker build .` runs the precompute (look for `data/map.npz` inside the image) and the build stays well under 10 min.

---

## Step 4: `bot/nav.py` (paths + cover)

**Prompt:**
```
Implement bot/nav.py per its docstring.
- astar(map, start, goal): 8-connected on the 2 m grid, octile heuristic, uses blocked; snap start/goal to
  nearest free cell; cap expansions (e.g. 20k) and return best-effort partial path if capped.
  Smooth the path by skipping waypoints with clear walkable straight lines.
- Navigator.steer(pos, goal, now, t0): keep current path; replan if goal moved > 6 m, path invalid, or
  replan_every_s elapsed, but never if (time.perf_counter()-t0)*1000 > PARAMS['plan_budget_ms'].
  Return unit move vector toward next waypoint; pop waypoints within 2 m. If stuck (moved < 0.5 m in 1.5 s
  while commanding motion) set jump=True next tick and replan.
- best_cover(map, me_pos, threat_xys, zone_next_center, zone_next_radius, max_dist): choose the cover
  point inside the next circle (with 15 m margin), whose shield directions cover the bearings to the
  threats, minimizing path length; return None if none within max_dist.
- time_to_reach(path, speed).
Add tests/test_nav.py:
- path between two open points exists and never enters a blocked cell
- path around the big building from step 3 is longer than the straight line
- astar on 300 m query < 25 ms p95 (100 random queries), Navigator.steer cached call < 1 ms
- best_cover with a threat to the east returns a point whose shield set includes east, inside next circle
- unreachable goal returns partial/None without exceptions
```

**Test gate:** `pytest tests/test_nav.py -q` passes, and all earlier tests still pass.

---

## Step 5: `bot/world.py` (world model)

**Prompt:**
```
Implement bot/world.py per its docstring.
- Track dataclass; WorldModel.update(obs) merges obs['mechs'] (mechs and npcs separately), keeps
  last_seen_t, estimates velocity, drops tracks older than 15 s.
- Sounds -> SoundHint(bearing, dist_bucket, kind, t) kept 5 s; 'death' sounds counted.
- Events: damaged -> attacker marked hostile with bearing/time, last_damage_t; hit -> last_dealt_t;
  kill/pickup/took/dropped bookkeeping; my weapons from obs['self'].
- regen_ready(): now - max(last_damage_t, last_dealt_t) >= regen_delay.
- FFA detection: self.ffa starts at PARAMS['ffa_default']. Set True (sticky) if we see a beam whose
  shooter is another mech and whose end is near a third mech, or projectiles traveling between two
  other mechs, or 'alive' drops by >= 2 within 10 s while we dealt no kill. Keep a counter of evidence;
  flip only at >= 3 pieces of evidence.
- Queries: enemies(max_dist=None), visible_enemies(), npcs(), threats_within(r), hostile_ids(window_s).
Add tests/test_world.py:
- replay all 300 real observations: no exceptions, update p99 < 2 ms
- synthetic: enemy seen then gone from obs keeps a track for > 5 s; a damaged event registers attacker
  and bearing; regen_ready false right after damage, true after regen_delay
- ffa stays False on the real stream unless evidence is present; becomes True with 3 synthetic infighting beams
```

**Test gate:** `pytest tests/test_world.py -q` passes, and all earlier tests still pass.

---

## Step 6: `bot/threat.py` (the decision math)

**Prompt:**
```
Implement bot/threat.py per its docstring, v1 only (risk_margin returns base margin for now; leave
v2 terms as TODO stubs returning 0). Use Rules for all weapon numbers, PARAMS for p_hit and thresholds.
- p_hit: gun linear from p_hit_gun_near at 0 m to p_hit_gun_far at gun_far_m, 0 beyond weapon range;
  lightning p_hit_lightning inside range else 0 (LOS is caller's job); laser/mortar constants inside range.
- weapon_dps(kind, dist, cubes): damage/cooldown*p_hit*(1+damage_per_cube*min(cubes,stat_cap)).
  Laser effective period = charge + cooldown. Gun includes reload time amortized over the mag.
- triangle(my_kind, their_kind) multiplier: winning matchup x1.25, losing x0.8, gun neutral.
- incoming_dps(me, enemy): their best weapons at the distance * triangle * rear_mult if enemy bearing
  is outside my front arc (use my ACTUAL aim yaw).
- should_fight(...) -> (bool, reason): the v1 rule from README; plus "no LOS to target -> False".
- pick_target: among visible enemies with LOS, minimize my_ttk, ties -> lower hp.
Fill in tests/test_threat.py with the cases in its docstring plus:
- facing vs back-turned enemy: incoming dps ratio == rear_mult
- 10 cubes vs 0 cubes: dps ratio == 1 + 10*damage_per_cube; 15 cubes same as 10 (cap)
- should_fight False when zone unsafe even if winning
```

**Test gate:** `pytest tests/test_threat.py -q` passes, and all earlier tests still pass.

---

## Step 7: `bot/control.py` (Intent → Action)

**Prompt:**
```
Implement bot/control.py per its docstring.
- Intent dataclass (all optional fields) and Controller.to_action(intent, obs, nav, t0) -> protocol action dict.
- lead_aim(shooter_muzzle, target_pos, target_vel, speed): iterative intercept (3 iterations),
  aim at center of mass (pos.z + 4.5 scaled by target size if known). Return (yaw, pitch).
- mortar_pitch(dist, dz, speed, gravity): low-arc solution if it clears LOS else high arc; None if out of range.
- Fire discipline: only fire a slot when |actual aim - desired aim| < max(0.03 rad, atan(2.5/dist)),
  weapon cooldown == 0, and intent.fire allows it. Laser: hold fire to charge, release when charge >=
  0.6 s (or target leaving cone); never charge while hiding. Lightning only if dist <= range and in cone.
- Movement: goal -> nav.steer; slow=True clamps move length so speed < step_speed*0.9 (silent).
  strafe(): perpendicular to target bearing, flip direction every 0.8-1.4 s (seeded RNG from hello.seed).
- Auto reload when no target visible and ammo < mag/2. Always echo tick in main.py (already done).
- Debug label "<MODE> | hp | reason" via mechfight draw helpers.
Add tests/test_control.py:
- stationary target: lead == direct bearing; moving target 100 m away at 10 m/s perpendicular:
  lead angle ~ atan(10/252) within 10%
- mortar_pitch for 150 m flat returns a solution that lands within 3 m (simulate the arc in the test)
- no fire when aim error is large; fire when aligned and cooldown 0
- slow intent produces move vector speed below step_speed
- every action produced validates against the BotMessage schema
```

**Test gate:** `pytest tests/test_control.py -q` passes, and all earlier tests still pass.

---

## Step 8: v1 modes + brain (ZONE, HIDE, DISENGAGE, FIGHT)

**Prompt:**
```
Implement bot/brain.py and bot/modes/{zone,hide,disengage,fight}.py (loot/drop return None for now).
Brain.__init__: Rules, PARAMS, mapdata.load('/bot/data/map.npz', hello['map']['dir']) with fallback
to repo paths when running locally, WorldModel, Navigator, Controller. Wrap map load in try/except:
if it fails, run with map=None and modes must degrade to straight-line movement (never crash).
Brain.act(obs, t0):
 1. world.update(obs); handle offer via upgrades.choose
 2. candidate mode by priority: ZONE (time_to_reach(next circle edge + 10 m margin) * 1.3 >= time until
    the circle passes us, or out of zone) > FIGHT (threat.should_fight on pick_target) > DISENGAGE
    (took damage in last 3 s or hostile visible within 120 m and not fighting) > HIDE.
 3. hysteresis: keep current mode unless mode_hold_s elapsed, except ZONE and "hp < 25% under fire"
    which switch immediately.
 4. run the mode -> Intent -> control.to_action.
Modes:
 - ZONE: path to the point on the next circle nearest us, pulled 15 m inside; boost when out of zone
   or late; aim toward the most recent threat bearing (else along travel).
 - HIDE: best_cover vs known threats/sound bearings; once there, stand (slow), aim at the most likely
   threat bearing, rotate scan +-60 deg every 3 s if nothing known; no firing.
 - DISENGAGE: best_cover away from attackers; move there while AIMING at the nearest attacker (legs
   and torso independent); fire only if the target is in arc and aligned (free damage), never stop to
   shoot; boost if hp < 40%.
 - FIGHT: aim with lead at target, fire per control rules, strafe; keep distance in the band where
   my weapons outperform theirs (lightning wants < 40 m, laser/gun prefer 60-150 m); stay inside zone.
Add tests/test_scenarios.py (scenario tests using make_obs):
 - outside zone, no enemies -> ZONE, move vector points toward next center
 - full HP, lone 150-HP enemy at 50 m in LOS, zone safe -> FIGHT, fire on slot 0 once aligned
 - 30% HP, two full-HP enemies within 80 m -> DISENGAGE, aim points at an enemy while move points away
 - damaged from behind (rear=True) -> not HIDE; aim swings toward attacker bearing
 - no info at all, mid-game -> HIDE, no fire, move speed below step_speed once at cover
 - mode does not flip-flop: alternate fight-favorable / unfavorable obs every tick for 1 s -> <= 1 switch
 - Brain with map=None never raises on any scenario
Plus tests/test_replay.py: run Brain over all 300 real observations; zero exceptions caught by main's
fallback (assert by calling brain.act directly), p50 < 5 ms, p99 < 15 ms, every action schema-valid.
```

**Test gate:**
```sh
pytest -q                                         # everything green
python3 check/check.py -- python3 -u main.py      # PASS, look at p99 time
```

---

## Step 9: v1 hardening → **SUBMISSION #1**

**Prompt:**
```
Prepare v1 for submission:
- In main.py add a per-tick time guard: if brain.act takes > 35 ms on 3 consecutive ticks, set
  brain.lite_mode = True for 5 s (skip replans and extra raycasts). Log to stderr at most once per 10 s.
- Make sure nothing writes outside /tmp and nothing imports dev-only packages at runtime.
- Make sure stderr spam is rate-limited (exceptions logged once per type per 30 s).
- Set Bot.name to "rags-mech v1".
- Update README "Run" section if anything changed.
Then run tools/run_check.sh and paste the output.
```

**Test gate (all required before pushing):**
```sh
pytest -q                     # green
./tools/run_check.sh          # PASS in the docker sandbox; p99 well under 50 ms
du -sh --exclude=.git .       # < 20 MB
```
Commit, push, submit the repo URL. Write the score in a log sheet (`date | commit | change | score`). **Target: > 3.0.**

---

## Step 10: v2 risk budget + overrides

**Prompt:**
```
Implement the v2 terms in bot/threat.py:
- reward_bonus = reward_per_cube * (target.cubes + 2) if my cubes < stat_cap else 0 (cap the bonus at 0.4)
- finish_bonus: if target_hp <= my burst damage over finish_burst_s (or target_hp < finish_hp) -> +0.5
- desperation: +desperation_late when the zone is in its last 2 phases (next_radius <= 70) or
  alive <= 4 in squad mode
- crowd_penalty = crowd_penalty_per_enemy * (number of OTHER hostiles within threat_radius)
- base = ffa_base_margin if world.ffa else base_margin
Overrides in should_fight (return reason strings):
- "finish": target hp < finish_hp and in LOS and in range -> True unless my_hp < incoming over 2 s
- "cornered": no cover reachable before death (path time to best_cover > their_ttk) -> True
- "final_minute": t > match_length - final_minute_no_trade_s and no one is damaging me -> False
  (avoid trades; HP decides the timeout ranking)
Add tests to tests/test_threat.py and tests/test_scenarios.py for each term and each override,
including: reward bonus 0 at cap; desperation only in late phases; final-minute avoids a favorable fight;
cornered fights back.
```

**Test gate:** `pytest -q` green (every earlier test included), and `check.py` PASSES.

---

## Step 11: LOOT mode + upgrades → **SUBMISSION #2**

**Prompt:**
```
Implement bot/modes/loot.py and bot/upgrades.py.
LOOT (only when no hostile seen/heard within 120 m in the last 5 s, and zone-safe):
- priority: weapon pickup if slot 1 empty (prefer by triangle vs most-seen enemy weapon, else laser >
  lightning > mortar), then cubes on the ground, then NPC drones within npc_farm_max_dist_m (engage from
  > 50 m with gun while it's leashed; it deals only 25 dmg / 1 s), then crates within 60 m.
- before t < loot_phase_end_s LOOT outranks HIDE; after it, only opportunistic (target within 40 m).
- take only when within pickup radius; swap with drop only if new weapon is strictly better per triangle.
Upgrades: choose by PARAMS['upgrade_priority'] then weapon upgrades for held weapons, else index 0.
Hook LOOT into brain priority (between DROP and HIDE). Keep v1 behavior otherwise.
Tests (tests/test_loot.py + scenarios):
- empty slot 1 + laser pickup 3 m away -> take set to its id
- enemy heard nearby -> not LOOT
- NPC at 60 m, no hostiles -> LOOT targets it and fires once aligned
- offer containing 'armor' -> picks its index
Replay test still p99 < 15 ms.
```

**Test gate:** `pytest -q`, `./tools/run_check.sh`, size check. Then submit #2 and log the score. **Keep it only if it beats v1.** Otherwise revert LOOT to opportunistic-only and resubmit.

---

## Step 12: DROP mode + mortar + late game

**Prompt:**
```
Implement bot/modes/drop.py and the v3 extras:
- DROP: consider drops with lands_in <= travel time + 2 s, inside the zone. Go only if no hostile
  within drop_contest_clear_m (seen or heard) and hp >= 50% or the drop is the only way to heal fast.
  On the ring: stand still in the center, aim at the most likely approach bearing, abort if
  progress resets twice or a hostile appears within 60 m (switch to FIGHT/DISENGAGE by normal rules).
- Mortar in FIGHT/DISENGAGE: if holding mortar and a remembered (not visible) enemy behind cover is
  within range, lob at its last tracked position (mortar_pitch high arc). Never mortar within 12 m
  (self splash 50% + knockback).
- Late game (next_radius <= 70): HIDE picks cover nearest the final circle center early; ZONE
  margin grows; avoid jumping/boosting noise unless needed.
Tests: drop taken when clear; aborted when enemy appears; mortar never fires at < 12 m; late-game HIDE
point lies inside final circle.
```

**Test gate:** `pytest -q` green, and `check.py` PASSES.

---

## Step 13: FFA tuning → **SUBMISSION #3**

**Prompt:**
```
Final v3 pass:
- Verify world.ffa switches margin base and that in FFA mode the bot prefers HIDE and third-partying:
  when two other mechs are fighting (beams/projectiles between them) within 160 m, approach to
  cover within 100 m, wait, then FIGHT the one whose hp < 40%.
- Add scenario test for third-partying and for squad mode NOT doing it.
- Audit: every number in modes/threat comes from Rules or PARAMS (grep for stray literals; move them).
- Bot.name "rags-mech v3". Run full tests + run_check.sh and paste results.
```

**Test gate:** full `pytest -q` and `./tools/run_check.sh`. Submit #3 and log it. Choose your final submission (the best score is the default).

---

## Step 14: Tuning loop (as much time as you have left)

You get 4 submissions per hour. Change **one** `PARAMS` value per submission, in this order:
1. `base_margin` 0.7 → 0.6 / 0.85
2. `min_hp_frac_to_fight` 0.40 → 0.30 / 0.55
3. `loot_phase_end_s` 60 → 90
4. `crowd_penalty_per_enemy`, `reward_per_cube`
5. `upgrade_priority` order

Before every push: `pytest -q && ./tools/run_check.sh`. Keep the repo public, and **never rewrite git history**, because every submitted commit has to stay fetchable for the final.

---

## Stop-the-line rules (apply at every step)

- A gate fails → fix it before moving on. Don't stack steps on a red test suite.
- p99 decision time locally > 15 ms → profile (`python -m cProfile`) before adding features.
- `check.py` fails in docker but passes locally → it's almost always a write outside `/tmp`, a missing system-wide install, or the map path. Check those first.
- Never let an exception reach `main.py`'s fallback silently in tests. The replay test calls `brain.act` directly so errors surface.
