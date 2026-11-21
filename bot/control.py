"""Low-level control: Intent -> Action.

TODO(cursor):
- class Intent (dataclass): goal_xy | move_vec, aim_target (pos+vel) | aim_yaw_pitch, fire_slots,
  boost, jump, take, drop, pick, slow (stay under step_speed to stay silent).
- lead_aim(me, target, projectile_speed): solve intercept for moving target; pitch to center of mass (pos.z+4.5).
  Mortar: ballistic pitch for range (gravity from rules), lob over buildings.
- weapon_select(me, target, dist): which slot(s) to fire. Laser: hold to charge, release when aim error small.
  Don't fire when HIDE (sound gives us away at 320 m).
- strafe(me, target): perpendicular dodge pattern in FIGHT; back-off-while-facing in DISENGAGE.
- reload when out of combat and ammo < mag/2.
- Respect self.aim being the ACTUAL torso aim (slews at turn_rate): only fire when aim error < tolerance.
"""


class Controller:
    def __init__(self, rules, params):
        self.rules = rules
        self.params = params

    def to_action(self, intent, obs) -> dict:
        raise NotImplementedError
