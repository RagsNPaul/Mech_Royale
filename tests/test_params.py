"""MECH_PARAMS is a shallow overlay applied when the bot process starts."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mech_params_overrides_shallow():
    env = os.environ.copy()
    env["MECH_PARAMS"] = json.dumps({"base_margin": 2, "upgrade_priority": ["armor"]})
    code = (
        "import json; from bot.params import PARAMS; "
        "print(json.dumps({"
        "'base_margin': PARAMS['base_margin'], "
        "'strategy': PARAMS['strategy'], "
        "'finish_hp': PARAMS['finish_hp'], "
        "'standoff_lightning_m': PARAMS['standoff_lightning_m'], "
        "'upgrade_priority': PARAMS['upgrade_priority']}))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, env=env, text=True)
    got = json.loads(out)
    assert got["base_margin"] == 2
    assert got["strategy"] == "brawler"
    assert got["finish_hp"] == 800.0
    assert got["standoff_lightning_m"] == 20.0
    assert got["upgrade_priority"] == ["armor"]


def test_mech_params_defaults_when_unset():
    env = os.environ.copy()
    env.pop("MECH_PARAMS", None)
    code = "from bot.params import PARAMS; print(PARAMS['base_margin']); print(PARAMS['heal_hp_frac'])"
    out = subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, env=env, text=True)
    margin, heal = out.split()
    assert float(margin) == 5.0
    assert float(heal) == 0.0
