"""Direction and minimum standstill interlocks, independent of the SDK.

Raw GPS gear has no verified command-enum mapping. A direction change waits
for observed standstill in Neutral, then holds the requested command gear
before permitting propulsion. This is not a gearbox acknowledgement.
"""

import math
from core.geometry import normalize_angle


class DirectionInterlock(object):
    def __init__(self):
        # Preserve legacy forward startup. Explicit reverse must be armed.
        self.active_direction = 1
        self.last_direction = 1
        self.pending_direction = None
        self.phase = None
        self.since = None

    def disarm(self):
        self.active_direction = None
        self.pending_direction = None
        self.phase = None
        self.since = None

    def update(self, direction, speed, now, settings):
        if self.pending_direction is None and direction == self.active_direction:
            return None
        if self.pending_direction != direction:
            self.pending_direction = direction
            self.phase, self.since = "GEAR_STOP", None
        if speed > settings.gear_standstill_speed_mps:
            self.phase, self.since = "GEAR_STOP", None
            return self.phase, 0
        if self.since is None:
            self.since = now
        if self.phase == "GEAR_STOP":
            if now - self.since < settings.gear_standstill_s:
                return self.phase, 0
            self.phase, self.since = "GEAR_ENGAGE", now
        gear = 1 if direction == 1 else 2
        if now - self.since < settings.gear_engage_s:
            return self.phase, gear
        self.active_direction = direction
        self.last_direction = direction
        self.pending_direction = self.phase = self.since = None
        return None


class StandstillGuard(object):
    """Latch a requested dwell at the stop; early replans cannot shorten it.

    A failed/stale input or drift interrupts the continuous standstill and
    restarts the countdown. Only a case reset clears an unfinished obligation.
    """

    def __init__(self):
        self.obligation_id = None
        self.duration = 0.0
        self.elapsed = 0.0
        self.anchor = None
        self.direction = 1
        self.parking_brake = False

    @property
    def active(self):
        return self.anchor is not None and self.elapsed < self.duration

    def interrupt(self):
        if self.active:
            self.elapsed = 0.0

    def begin(self, trajectory, ego):
        duration = getattr(trajectory, "hold_duration_s", 0.0)
        if duration <= 0 or self.active:
            return
        identity=getattr(trajectory,'stop_obligation_id',None)
        if identity is not None and identity==self.obligation_id and self.anchor is not None and self.elapsed>=self.duration:
            return
        # An unchanged completed stop must not start a new ten-second dwell
        # every frame. A genuinely different stop can request a new dwell.
        if (identity is None and self.anchor is not None and self.elapsed >= self.duration and
                math.hypot(ego.x - self.anchor[0], ego.y - self.anchor[1]) < 0.3):
            return
        self.obligation_id=identity
        self.duration, self.elapsed = duration, 0.0
        self.anchor = (ego.x, ego.y, ego.heading)
        self.direction = getattr(trajectory, "motion_direction", 1)
        self.parking_brake = getattr(trajectory, "parking_brake_at_stop", False)

    def update(self, ego, dt, settings):
        if not self.active:
            if (self.anchor is not None and
                    math.hypot(ego.x - self.anchor[0], ego.y - self.anchor[1]) > 0.3):
                # After leaving a completed stop the same bay may be visited
                # again in this case, with a new minimum dwell obligation.
                self.anchor = None
            return False
        if (ego.speed > settings.gear_standstill_speed_mps or
                math.hypot(ego.x - self.anchor[0], ego.y - self.anchor[1]) >
                settings.precision_arrival_tolerance_m or
                abs(normalize_angle(ego.heading - self.anchor[2])) >
                settings.precision_heading_tolerance_rad):
            self.elapsed = 0.0
        else:
            self.elapsed = min(self.duration, self.elapsed + dt)
        return self.active
