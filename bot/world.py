"""World model: everything we know beyond the current observation.

TODO(cursor):
- class Track: id, kind, pos, vel, hp, max_hp, cubes, weapons, aim, charging, last_seen_t.
- class WorldModel.update(obs):
  * merge obs['mechs'] into tracks (server already remembers 5 s; we keep longer, decaying confidence)
  * sounds -> list of (bearing, dist bucket, kind, t) threat hints; 'gun' etc. tell weapon held
  * events: 'damaged' -> mark attacker as active threat (bearing, by, rear);
           'hit'/'kill' -> bookkeeping; 'took'/'dropped' -> our weapons
  * last_damage_t / last_dealt_t -> regen timer (regen after regen_delay without either)
  * ffa flag: set True if we observe enemies damaging each other (beams/projectiles between others,
    or 'death' sounds we didn't cause while no house-squad pattern). Default PARAMS['ffa_default'].
- Queries: enemies(within_m), visible_enemies(), threats_near(), time_since_damage().
"""


class WorldModel:
    def __init__(self, rules, params):
        self.rules = rules
        self.params = params
        self.ffa = params["ffa_default"]

    def update(self, obs):
        raise NotImplementedError
