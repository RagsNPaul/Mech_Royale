"""Every tunable number lives here. One submission = change one thing, log the score.

Rules numbers (weapon damage, ranges, zone) are NOT here: read them from hello.config via bot.config.
"""
PARAMS = {
    # --- fight / hide decision (threat.py) ---
    "base_margin": 5.0,          # FIGHT if my_ttk < margin * their_ttk
    "min_hp_frac_to_fight": 0.0,
    "zone_safe_s": 10.0,         # need this many seconds of zone safety to start a fight
    "threat_radius_m": 120.0,    # enemies within this count toward total incoming DPS
    "mode_hold_s": 4.0,          # hysteresis: min time in a mode before switching

    # --- risk budget (v2) ---
    "reward_per_cube": 0.03,     # margin bonus per cube the kill would give (if under cap)
    "finish_hp": 800.0,          # target HP under which we commit to finishing
    "finish_burst_s": 1.5,
    "desperation_late": 0.4,     # margin bonus in the last zone phases
    "crowd_penalty_per_enemy": 0.25,
    "final_minute_no_trade_s": 60.0,  # avoid HP trades this close to match end (HP tiebreak)

    # --- hit probability model (threat.py) ---
    "p_hit_gun_near": 0.8, "p_hit_gun_far": 0.25, "gun_far_m": 200.0,
    "p_hit_laser": 0.7, "p_hit_mortar": 0.4, "p_hit_lightning": 1.0,

    # --- squad vs FFA (world.py) ---
    "ffa_default": False,        # practice is squad; flip when infighting is observed
    "ffa_base_margin": 0.6,      # more patient in FFA

    # --- loot / drops ---
    "loot_phase_end_s": 60.0,
    "npc_farm_max_dist_m": 150.0,
    "drop_contest_clear_m": 60.0,  # contest a drop only if nobody seen/heard this close

    # --- timing ---
    "plan_budget_ms": 30.0,      # skip path replans past this elapsed time
    "replan_every_s": 1.0,

    # --- upgrades (priority order; unknown ids fall to the end) ---
    "upgrade_priority": [
        "gun_rate",
        "gun_damage",
        "laser_damage",
        "lightning_efficiency",
        "mortar_reload",
        "laser_charge",
        "armor",
        "sensors",
    ],
}
