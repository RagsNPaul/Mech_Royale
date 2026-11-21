"""Upgrade offer handling (protocol §10).

TODO(cursor): choose(offer, me, rules, params) -> index into offer['choices'].
Use PARAMS['upgrade_priority']; prefer weapon upgrades for weapons we actually hold;
answer before offer['expires'] (default pick 0 otherwise).
"""


def choose(offer, me, rules, params) -> int:
    return 0
