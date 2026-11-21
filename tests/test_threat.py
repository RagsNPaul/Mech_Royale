"""Unit tests for bot/threat.py (pure math, no sim needed). Run: python3 -m pytest tests

TODO(cursor):
- gun DPS at point blank ~= 45/0.24 * p_hit_gun_near
- lightning DPS is 0 beyond 45 m
- rear hit multiplies incoming DPS by rear_mult
- should_fight: full-HP me vs 200-HP lone target -> True (finish override)
- should_fight: 30%-HP me vs two full-HP enemies -> False
"""
