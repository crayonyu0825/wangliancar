"""D02 bounded target history and conservative constraint handoff.

Local arrival seconds are used, never guessed SDK timestamp units. History is
constraint memory, not a fabricated fresh Sensor API target or acceleration.
"""

import math
from collections import OrderedDict, deque

from core.validation import number
from members.decision import protocol


class TargetSample(object):
    __slots__ = ("frame_id", "observed_at_s", "x", "y", "vx", "vy", "route_s",
                 "lateral_m", "forward_speed", "lateral_speed", "gap", "conflict",
                 "dangerous", "motion_supported", "static")

    def __init__(self, frame_id, observed_at_s, candidate, settings):
        target = candidate.target
        self.frame_id, self.observed_at_s = frame_id, observed_at_s
        self.x, self.y, self.vx, self.vy = target.x, target.y, target.vx, target.vy
        self.route_s, self.lateral_m = candidate.route_s, candidate.lateral_offset_m
        self.forward_speed, self.lateral_speed, self.gap = candidate.lead_speed, candidate.lateral_speed, candidate.gap
        self.conflict, self.motion_supported, self.static = candidate.conflict, candidate.motion_supported, candidate.static
        self.dangerous = bool(candidate.conflict and (candidate.reason or candidate.static
                              or not candidate.motion_supported or candidate.gap is None
                              or candidate.gap <= settings.min_gap
                              or 0 <= candidate.ttc <= settings.emergency_ttc))


class TargetTrack(object):
    def __init__(self, identifier, sample_budget):
        self.identifier = identifier
        self.samples = deque(maxlen=sample_budget)
        self.was_followed = False
        self.last_speed_demand = None
        self.quality = "FIRST_OBSERVATION"
        self.relation = "UNKNOWN"
        self.handoff_state = "OBSERVED"


class TargetHistory(object):
    def __init__(self, settings, clock=None):
        self.settings = settings
        self._clock = clock
        self.reset()

    def reset(self):
        self.tracks = OrderedDict()
        self._danger_memory = OrderedDict()
        self._danger_clear_counts = {}
        self._danger_clear_source = {}
        self._danger_overflow = False
        self._context, self._source, self._last_frame, self._last_time = None, None, None, None
        self._now, self._usable, self._record, self._seen = 0.0, False, False, set()
        self.events = deque(maxlen=128)
        self.reset_reason = ""

    def begin(self, perception, now_s, distinct):
        if not number(now_s):
            raise ValueError("target history needs finite local monotonic seconds")
        ctx = (perception.case_id, perception.task_id, perception.scene_id)
        source = perception.target_source
        rolled = self._last_frame is not None and perception.frame_id < self._last_frame
        clock_back = self._last_time is not None and now_s < self._last_time
        source_changed = self._source is not None and source != self._source
        context_changed = self._context is not None and ctx != self._context
        reason = ("SOURCE_CHANGED" if source_changed else "CONTEXT_CHANGED" if context_changed
                  else "GPS_ROLLBACK" if rolled else "CLOCK_ROLLBACK" if clock_back else "")
        self.reset_reason = reason
        if reason:
            retained = (self._danger_memory, self._danger_overflow)
            self.reset()
            if reason in ('SOURCE_CHANGED', 'GPS_ROLLBACK', 'CLOCK_ROLLBACK', 'CONTEXT_CHANGED'):
                self._danger_memory, self._danger_overflow = retained
            self.reset_reason = reason
            self.events.append(dict(frame_id=perception.frame_id, reason=reason))
        gap = self._last_time is not None and distinct and now_s - self._last_time > self.settings.history_max_gap_s
        if gap:
            self.tracks.clear()
            self.events.append(dict(frame_id=perception.frame_id, reason="OBSERVATION_GAP"))
        self._context, self._source = ctx, source
        self._now, self._usable, self._seen = now_s, protocol.targets_usable(perception), set()
        self._record = bool(distinct and self._usable)
        self._last_frame = perception.frame_id
        if distinct or self._last_time is None:
            self._last_time = now_s
        for identifier in list(self.tracks):
            if not self.tracks[identifier].samples or now_s - self.tracks[identifier].samples[-1].observed_at_s > self.settings.history_age_s:
                self.tracks.pop(identifier, None)

    def observe(self, perception, candidate):
        target = candidate.target
        if candidate.reason == "mapped traffic-light fixture":
            return
        identifier = getattr(target, "id", -1)
        if (type(identifier) is not int or identifier < 0 or target.valid is not True
                or not all(number(v) for v in (target.x, target.y, target.vx, target.vy))):
            return
        self._seen.add(identifier)
        if not self._record:
            return
        sample = TargetSample(perception.frame_id, self._now, candidate, self.settings)
        if (getattr(perception, 'behavior_channel', None)
                and getattr(getattr(perception, 'maneuver_environment', None), 'valid', False)):
            if sample.dangerous:
                from core.maneuver_facts import ManeuverFacts
                try:
                    digest = ManeuverFacts(perception, self._clock or (lambda: self._now)).map_digest()
                except (ValueError, TypeError, AttributeError, KeyError):
                    digest = None
                if identifier in self._danger_memory or len(self._danger_memory) < self.settings.history_max_tracks:
                    self._danger_memory[identifier] = (target.x, target.y, target.z,
                        target.length, target.width, target.heading, digest)
                else:
                    self._danger_overflow = True
                self._danger_clear_counts[identifier] = 0
                self._danger_clear_source.pop(identifier, None)
            else:
                self._danger_memory.pop(identifier, None)
                self._danger_clear_counts.pop(identifier, None)
                self._danger_clear_source.pop(identifier, None)
        track = self.tracks.get(identifier)
        if track is None:
            track = TargetTrack(identifier, self.settings.history_samples)
        elif track.samples:
            previous = track.samples[-1]
            dt = self._now - previous.observed_at_s
            predicted_x, predicted_y = previous.x + previous.vx * dt, previous.y + previous.vy * dt
            if dt > 0 and math.hypot(sample.x - predicted_x, sample.y - predicted_y) > self.settings.history_position_jump_m:
                track = TargetTrack(identifier, self.settings.history_samples)
                track.quality = "ID_DISCONTINUITY"
                self.events.append(dict(frame_id=perception.frame_id, target_id=identifier,
                                        reason="ID_DISCONTINUITY"))
            else:
                track.quality = "CONTINUOUS"
        track.samples.append(sample)
        track.relation = candidate.relation
        if candidate.lateral_offset_m is not None and candidate.lateral_speed is not None:
            product = candidate.lateral_offset_m * candidate.lateral_speed
            track.handoff_state = ("ENTERING" if product < -0.01 else
                                   "EXITING" if product > 0.01 else "ALONG_ROUTE")
        else:
            track.handoff_state = "UNRESOLVED_ROUTE"
        self.tracks.pop(identifier, None)
        self.tracks[identifier] = track
        while len(self.tracks) > self.settings.history_max_tracks:
            self.tracks.popitem(last=False)

    def followed(self, identifier, speed_demand):
        track = self.tracks.get(identifier)
        if track is not None and number(speed_demand) and speed_demand >= 0:
            track.was_followed = True
            track.last_speed_demand = speed_demand

    def discontinuous(self, identifier):
        track = self.tracks.get(identifier)
        return bool(track is not None and track.quality == "ID_DISCONTINUITY")

    def was_followed(self, identifier):
        track = self.tracks.get(identifier)
        return bool(track is not None and track.was_followed and track.samples
                    and self._now - track.samples[-1].observed_at_s <= self.settings.history_age_s)

    def retained_constraints(self, ego_speed, cruise):
        """Keep a safe vanished lead's cap briefly; never create a stale target.

        Unsafe or unknown disappearance stays with the existing hazard/source
        guards. Confirming hazard clearance needs the R06/R09 coverage contract.
        """
        result = []
        if not self._usable:
            return result
        for identifier, track in self.tracks.items():
            if identifier in self._seen or not track.was_followed or not track.samples:
                continue
            last = track.samples[-1]
            age = self._now - last.observed_at_s
            if not 0 <= age <= self.settings.history_handoff_s:
                continue
            if last.dangerous or last.gap is None or last.forward_speed is None or last.static:
                continue
            if track.last_speed_demand is None or not last.motion_supported:
                continue
            braking = self.settings.history_braking_uncertainty_mps2
            lower_gap = last.gap + (last.forward_speed - ego_speed) * age - 0.5 * braking * age * age
            if lower_gap <= self.settings.min_gap:
                result.append(("PROTECT", identifier, None, "HANDOFF_GAP_UNCERTAIN"))
            else:
                demand = min(cruise, max(0.0, track.last_speed_demand - braking * age))
                result.append(("FOLLOW", identifier, demand, "HANDOFF_PREVIOUS_LEAD_CAP"))
        return result

    def missing_danger_constraints(self, perception):
        """History expiry never proves that a vanished hazard is clear.

        Use main's Sensor/map facts and the original complete footprint. Only
        distinct original Sensor packets can discharge the retained duty.
        """
        if not self._danger_memory and not self._danger_overflow:
            return []
        from core.maneuver_facts import ManeuverFacts
        from core.behavior_channel import read_channel
        from core.region_geometry import rectangle, intersects
        result = []
        if self._danger_overflow:
            result.append(('PROTECT', -1, None, 'DANGER_MEMORY_LIMIT_REQUIRES_TASK_RESET'))
        for identifier, shape in list(self._danger_memory.items()):
            if identifier in self._seen:
                continue
            clear = False
            source = None
            try:
                read_channel(perception, self._now)
                facts = ManeuverFacts(perception, self._clock or (lambda: self._now))
                objects = facts.objects()
                x, y, z, length, width, heading, digest = shape
                if digest is None or facts.map_digest() != digest:
                    raise ValueError('retained hazard map changed')
                if not all(number(v) and v > 0 for v in (length, width)):
                    raise ValueError('retained hazard dimensions unavailable')
                if heading is None:
                    length = width = math.hypot(length, width)
                    heading = 0.
                body = rectangle(x, y, heading, length / 2, length / 2, width / 2)
                clear = facts.region_visible([(a, b, z) for a, b in body])
                age = self._now - facts.env.dynamic_observed_at_s
                margin = .5 * self.settings.history_braking_uncertainty_mps2 * age * age
                for obj in objects:
                    size = math.hypot(obj['length'], obj['width'])
                    yaw = obj['heading']
                    front = (obj['length'] if yaw is not None else size) / 2 + margin
                    half_width = (obj['width'] if yaw is not None else size) / 2 + margin
                    occupied = rectangle(obj['x'] + obj['vx'] * age, obj['y'] + obj['vy'] * age,
                                         yaw if yaw is not None else 0., front, front, half_width)
                    if intersects(body, occupied):
                        clear = False
                        break
                facts.check_current(True)
                source = (perception.target_source, perception.targets_frame_id,
                          facts.env.dynamic_observed_at_s)
            except (ValueError, TypeError, AttributeError, KeyError, OverflowError):
                clear = False
            prior = self._danger_clear_source.get(identifier)
            distinct = (source is not None and (prior is None or
                        source[0] == prior[0] and source[1] > prior[1] and source[2] > prior[2]))
            if self._record:
                if clear and distinct:
                    self._danger_clear_counts[identifier] = self._danger_clear_counts.get(identifier, 0) + 1
                    self._danger_clear_source[identifier] = source
                elif not clear or source is not None and prior is not None and source != prior and not distinct:
                    self._danger_clear_counts[identifier] = 0
                    self._danger_clear_source.pop(identifier, None)
            if clear and self._danger_clear_counts.get(identifier, 0) >= self.settings.release_frames:
                self._danger_memory.pop(identifier, None)
                self._danger_clear_counts.pop(identifier, None)
                self._danger_clear_source.pop(identifier, None)
            else:
                result.append(('PROTECT', identifier, None, 'DANGER_CLEARANCE_COVERAGE_UNVERIFIED'))
        return result

    def snapshot(self):
        return dict(time_basis="local_monotonic_arrival_s", source=self._source,
                    reset_reason=self.reset_reason,
                    track_count=len(self.tracks), unresolved_danger_ids=list(self._danger_memory),
                    danger_memory_overflow=self._danger_overflow, tracks=[dict(
                        target_id=identifier, quality=track.quality, relation=track.relation,
                        handoff_state=track.handoff_state, was_followed=track.was_followed,
                        sample_count=len(track.samples), last_frame_id=track.samples[-1].frame_id,
                        age_s=max(0.0, self._now - track.samples[-1].observed_at_s),
                        route_s=track.samples[-1].route_s, lateral_m=track.samples[-1].lateral_m)
                        for identifier, track in self.tracks.items() if track.samples], events=list(self.events))
