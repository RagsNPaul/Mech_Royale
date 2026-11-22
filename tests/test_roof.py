"""Roof choice, the jet climb, and the heal walk back to the middle of the roof."""

import math
import time

from bot.brain import Brain
from bot.params import PARAMS
from bot.strategies.roof import choose_roof, climb_command
from tests.fixtures import at_time, make_obs, with_self, with_zone
from tests.test_scenarios import _quiet


def test_chosen_roof_is_inside_the_next_circle_and_in_height_range():
    zone = {"next_center": [0.0, 0.0], "next_radius": 200.0}
    short = {"centroid": (0.0, 40.0), "height": 8.0, "area": 800.0, "path_length": 1.0}
    outside = {"centroid": (0.0, 190.0), "height": 20.0, "area": 800.0, "path_length": 2.0}
    small = {"centroid": (0.0, 50.0), "height": 16.0, "area": 100.0, "path_length": 3.0}
    farther = {"centroid": (0.0, 150.0), "height": 30.0, "area": 500.0, "path_length": 400.0}
    nearer = {"centroid": (0.0, 100.0), "height": 18.0, "area": 400.0, "path_length": 80.0}
    chosen = choose_roof((0.0, 0.0), zone, [short, outside, small, farther, nearer])
    assert chosen is nearer
    dist = math.hypot(chosen["centroid"][0], chosen["centroid"][1])
    assert dist + 20.0 <= zone["next_radius"]
    assert 12.0 <= chosen["height"] <= 40.0


def test_climb_reaches_the_roof_from_a_standing_start(rules):
    mech = rules.mech
    gravity = float(mech.gravity)
    jet_accel = float(mech.jet_accel)
    max_rise = float(mech.max_rise_speed)
    jump_speed = float(mech.jump_speed)
    ground = 0.0
    roof_top = 24.0
    into = (1.0, 0.0)
    z = ground
    vz = 0.0
    grounded = True
    dt = 0.05
    reached = False
    for _ in range(int(12.0 / dt)):
        move, do_jump, boost = climb_command(grounded, z, roof_top, into)
        assert move[0] > 0.0
        if grounded and do_jump:
            vz = jump_speed
            grounded = False
        if not grounded:
            if boost:
                vz = min(max_rise, vz + jet_accel * dt)
            vz -= gravity * dt
            z += vz * dt
            if z <= ground:
                z = ground
                vz = 0.0
                grounded = True
        if z > roof_top + 0.5:
            reached = True
            break
    assert reached


def test_heal_cycle_walks_to_the_centroid_at_35_percent(hello, monkeypatch):
    monkeypatch.setitem(PARAMS, "strategy", "roof")
    brain = Brain(hello)
    brain.map = None
    brain.nav = None
    cx, cy = 0.0, 0.0
    half = 20.0
    top = 30.0
    roof = {
        "centroid": (cx, cy),
        "top": top,
        "ground": top - 20.0,
        "height": 20.0,
        "area": 1600.0,
        "path_length": 10.0,
        "ring": [cx - half, cy - half, cx + half, cy - half, cx + half, cy + half, cx - half, cy + half],
        "wall": (cx - half - 4.0, cy),
        "interior": (cx, cy),
    }
    brain._roof = roof
    px = cx - half + 5.0
    obs = make_obs(
        at_time(40.0),
        with_self(pos=[px, cy, top + 1.0], in_zone=True, aim=[0.0, 0.0], weapons=["gun", None]),
        with_zone([cx, cy], 900.0, [cx, cy], 800.0, 1000.0, 1100.0, 5.0),
        **_quiet(),
    )
    obs["self"]["hp"] = 0.35 * float(obs["self"]["max_hp"])
    obs["self"]["grounded"] = True
    act = brain.act(obs, time.perf_counter())
    assert act["move"][0] > 0.5
    assert math.hypot(act["move"][0], act["move"][1]) > 0.5
