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
    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def reset(self):
        self.tracks = OrderedDict()
        self._danger_memory = OrderedDict()
        self._danger_clear_counts = {}
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
            danger=dict(self._danger_memory)
            self.reset()
            if reason in ('SOURCE_CHANGED','GPS_ROLLBACK','CLOCK_ROLLBACK'):
                self._danger_memory=OrderedDict(danger)
                self._danger_clear_counts={identifier:0 for identifier in danger}
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
        if sample.dangerous:
            self._danger_memory[identifier]=(target.x,target.y,target.length,target.width,target.heading)
            self._danger_clear_counts[identifier]=0
            while len(self._danger_memory)>self.settings.history_max_tracks:
                oldest=next(iter(self._danger_memory));self._danger_memory.pop(oldest,None)
                self._danger_clear_counts.pop(oldest,None)
        else:
            self._danger_memory.pop(identifier,None);self._danger_clear_counts.pop(identifier,None)
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

    def missing_danger_constraints(self,perception):
        """Aging an ID is not coverage proof. Preserve a separate bounded duty."""
        if not perception.session_id:
            return []  # Legacy protocol keeps its established release gate.
        from core.behavior_environment import BehaviorEnvironment,vehicle_footprint
        from core.behavior_protocol import behavior_frame,GoalPose
        env=perception.behavior_environment
        frame=behavior_frame(perception,self._now)
        covered=(isinstance(env,BehaviorEnvironment) and env.coverage is not None
                 and env.coverage.verified(frame,perception.observation_mode))
        result=[]
        for identifier,shape in list(self._danger_memory.items()):
            if identifier in self._seen:continue
            x,y,length,width,heading=shape
            if heading is None:
                length=width=math.hypot(length,width);heading=0.0
            footprint=vehicle_footprint(GoalPose(x,y,heading),length/2,length/2,width/2)
            clear=covered and env.coverage.contains(footprint)
            if self._record:
                self._danger_clear_counts[identifier]=(self._danger_clear_counts.get(identifier,0)+1) if clear else 0
            if clear and self._danger_clear_counts.get(identifier,0)>=self.settings.release_frames:
                self._danger_memory.pop(identifier,None);self._danger_clear_counts.pop(identifier,None)
            else:
                result.append(('PROTECT',identifier,None,'DANGER_CLEARANCE_COVERAGE_UNVERIFIED'))
        return result

    def snapshot(self):
        return dict(time_basis="local_monotonic_arrival_s", source=self._source,
                    reset_reason=self.reset_reason,
                    track_count=len(self.tracks), unresolved_danger_ids=list(self._danger_memory),tracks=[dict(
                        target_id=identifier, quality=track.quality, relation=track.relation,
                        handoff_state=track.handoff_state, was_followed=track.was_followed,
                        sample_count=len(track.samples), last_frame_id=track.samples[-1].frame_id,
                        age_s=max(0.0, self._now - track.samples[-1].observed_at_s),
                        route_s=track.samples[-1].route_s, lateral_m=track.samples[-1].lateral_m)
                        for identifier, track in self.tracks.items() if track.samples], events=list(self.events))
