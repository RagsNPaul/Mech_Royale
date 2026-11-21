"""Fight-or-hide math. Pure functions -> easy to unit test (tests/test_threat.py).

my_dps(target)   = sum over my usable weapons at distance d of
                   damage / cooldown * p_hit(kind, d) * (1 + damage_per_cube * min(cubes, stat_cap))
their_dps(enemy) = same from their weapons list, * triangle modifier,
                   * rear_mult if they are outside MY front arc (they hit my rear)
my_ttk    = target_hp / my_dps(target)
their_ttk = my_hp / sum(their_dps for enemies within threat_radius)

margin (v2 risk budget) = base (or ffa_base) + reward_bonus + finish_bonus + desperation - crowd_penalty
FIGHT if my_ttk < margin * their_ttk and hp_frac > min_hp_frac and zone safe >= zone_safe_s.
Overrides: finish low-HP target; cornered -> fight; final minute -> avoid trades (HP tiebreak).

Weapon triangle: laser > mortar, mortar > lightning, lightning > laser, gun neutral.
"""


def p_hit(kind: str, dist: float, rules, params) -> float:
    raise NotImplementedError


def weapon_dps(kind: str, dist: float, cubes: int, rules, params) -> float:
    raise NotImplementedError


def risk_margin(me, target, world, obs, rules, params) -> float:
    raise NotImplementedError


def should_fight(me, target, world, obs, rules, params) -> tuple[bool, str]:
    """Return (fight?, reason). Reason goes into the debug label."""
    raise NotImplementedError


def pick_target(me, world, rules, params):
    """Lowest effective TTK target among visible enemies (prefer low HP, in LOS, in front arc)."""
    raise NotImplementedError
