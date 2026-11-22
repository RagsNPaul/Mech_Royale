"""Upgrade offer handling (protocol §10).

choose() returns the index of the highest entry in PARAMS["upgrade_priority"].
Ids that are not on the list come after every listed id. Equal rank keeps the
earlier choice.
"""

from __future__ import annotations

from typing import Any

from bot.params import PARAMS


def choose(offer, me, rules, params) -> int:
    del me, rules
    choices = offer.get("choices") if isinstance(offer, dict) else None
    if not choices:
        return 0
    order = _order(params)
    rank = {str(ident): index for index, ident in enumerate(order)}
    last = len(order)
    best_index = 0
    best_rank = last + 1
    for index, choice in enumerate(choices):
        ident = choice.get("id") if isinstance(choice, dict) else choice
        score = rank.get(str(ident), last)
        if score < best_rank:
            best_index = index
            best_rank = score
    return best_index


def _order(params: Any) -> list:
    raw = params.get("upgrade_priority") if isinstance(params, dict) else None
    if raw:
        return list(raw)
    return list(PARAMS["upgrade_priority"])
