"""D01 blocked-road lights and signal minimum-stop obligations.

The minimum dwell is enforced from validated controller evidence. This module
never runs a second standstill countdown or equates planned/sent with arrived.
"""

from members.decision.behaviors.contract import (
    BehaviorGoal, LightIntent, GoalPose, finite, require,
)
from members.decision.behaviors.session import BehaviorSession


class PolicyResult(object):
    def __init__(self, request=None, hold_required=False, reason="", phase="IDLE"):
        self.request, self.hold_required, self.reason, self.phase = request, hold_required, reason, phase


class SignalStopObservation(object):
    def __init__(self, stop_id, stop_distance_m, red_required, usable=True,
                 speed_cap_mps=0.0, goal_pose=None):
        require(isinstance(stop_id, str) and bool(stop_id), "signal stop identity missing")
        require(stop_distance_m is None or finite(stop_distance_m) and stop_distance_m >= 0,
                "invalid signal stop distance")
        require(type(red_required) is bool and type(usable) is bool
                and finite(speed_cap_mps) and speed_cap_mps >= 0, "invalid signal stop evidence")
        require(goal_pose is None or isinstance(goal_pose, GoalPose), "invalid signal goal")
        self.stop_id, self.stop_distance_m, self.red_required = stop_id, stop_distance_m, red_required
        self.usable, self.speed_cap_mps, self.goal_pose = usable, speed_cap_mps, goal_pose


class SignalDwellPolicy(object):
    def __init__(self, minimum_duration_s=5.1, session=None):
        require(finite(minimum_duration_s) and minimum_duration_s > 5.0,
                "signal policy requires strictly more than five seconds")
        self.minimum_duration_s = minimum_duration_s
        self.session = session or BehaviorSession()
        self.reset()

    def reset(self):
        self.session.reset()
        self._context = None
        self._completed = set()
        self._stop_id, self._arrived, self._phase = None, False, "IDLE"
        self._last_frame = None

    def evaluate(self, frame, observation, capabilities=None, feedback=None,
                 dwell_required=True, safety_override=False):
        if not frame.current(frame.observed_at_s):
            request = self.session.tick(frame, capabilities, safety_override=True) if self.session.intent_id else None
            return PolicyResult(request, bool(request), "OBSERVATION_UNUSABLE", "SUSPENDED")
        if self._context != frame.context.key():
            self.reset()
            self._context = frame.context.key()
        if self._last_frame is not None and frame.frame_id < self._last_frame:
            old = self.session.finish(frame, False, "GPS_FRAME_ROLLBACK") if self.session.intent_id else None
            self.reset()
            self._context, self._last_frame = frame.context.key(), frame.frame_id
            return PolicyResult(old, True, "GPS_FRAME_ROLLBACK", "RECOVER")
        self._last_frame = frame.frame_id
        if not dwell_required:
            if self.session.intent_id is not None and self.session.status not in ("COMPLETED", "CANCELLED"):
                request = self.session.finish(frame, False, "DWELL_POLICY_NOT_APPLICABLE")
                return PolicyResult(request, False, request.reason_code, "CANCELLED")
            return PolicyResult(reason="NO_MINIMUM_DWELL_POLICY")
        if observation is None or not isinstance(observation, SignalStopObservation) or not observation.usable:
            if self.session.intent_id is not None:
                request = self.session.tick(frame, capabilities, safety_override=True)
                return PolicyResult(request, self._arrived, "SIGNAL_OBSERVATION_UNAVAILABLE", self._phase)
            return PolicyResult(reason="SIGNAL_OBSERVATION_UNAVAILABLE")
        if observation.stop_id in self._completed:
            self._phase = "WAIT_SIGNAL" if observation.red_required else "COMPLETE"
            return PolicyResult(self.session.snapshot(frame), observation.red_required,
                                "DWELL_ALREADY_COMPLETED", self._phase)
        if self._stop_id is not None and observation.stop_id != self._stop_id and self._arrived:
            request = self.session.block(frame, "UNFINISHED_STOP_OBLIGATION_CHANGED")
            return PolicyResult(request, True, request.reason_code, "BLOCKED")
        if observation.stop_distance_m is None:
            if self.session.intent_id is not None:
                request = self.session.block(frame, "STOP_BOUNDARY_UNKNOWN")
                return PolicyResult(request, self._arrived, request.reason_code, "BLOCKED")
            return PolicyResult(reason="STOP_BOUNDARY_UNKNOWN")
        if self._stop_id != observation.stop_id:
            self._stop_id, self._arrived, self._phase = observation.stop_id, False, "APPROACH"
        goal = BehaviorGoal(goal_pose=observation.goal_pose,
                            speed_cap_mps=0.0 if self._arrived else observation.speed_cap_mps,
                            stop_distance_m=0.0 if self._arrived else observation.stop_distance_m,
                            precision_stop=True,
                            minimum_standstill_duration_s=self.minimum_duration_s,
                            stop_obligation_id="signal:" + observation.stop_id)
        # One combined stop+dwell segment keeps the controller's same-stop
        # dwell identity; arrival does not reissue a new timed obligation.
        self.session.start(frame, "signal:" + observation.stop_id, "SIGNAL_STOP",
                           "STOP_AND_DWELL", goal, ("PATH_STOP", "DWELL", "FEEDBACK"))
        request = self.session.tick(frame, capabilities, feedback, safety_override)
        reply = self.session.last_feedback
        if (reply is not None and self.session.feedback_distinct
                and reply.producer in ("control", "runtime")):
            if reply.goal_pose_arrived and reply.actual_standstill_confirmed:
                self._arrived, self._phase = True, "DWELL"
            elif self._arrived:
                self._phase = "DWELL_REVALIDATE"
            complete = (reply.status == "COMPLETED" and reply.goal_pose_arrived
                        and reply.actual_standstill_confirmed and reply.hold_completed
                        and reply.standstill_duration_s >= self.minimum_duration_s
                        and reply.standstill_duration_s > 5.0)
            if complete:
                self._completed.add(observation.stop_id)
                if len(self._completed) > 64:
                    self._completed = {observation.stop_id}
                request = self.session.finish(frame, True, "SIGNAL_DWELL_COMPLETED")
                self._phase = "WAIT_SIGNAL" if observation.red_required else "COMPLETE"
                return PolicyResult(request, observation.red_required, request.reason_code, self._phase)
        if request.status in ("BLOCKED", "SUSPENDED", "CANCELLED"):
            return PolicyResult(request, self._arrived or observation.stop_distance_m <= 0.05,
                                request.reason_code, self._phase)
        return PolicyResult(request, self._arrived, "DWELL_PENDING" if self._arrived else "STOP_APPROACH",
                            self._phase)


class BlockingObservation(object):
    def __init__(self, blockage_id, stop_distance_m, verified_blockage,
                 alternative_status="unknown", ordinary_follow=False,
                 signal_only=False, verified_clear=False, usable=True,
                 speed_cap_mps=0.0, source=None, source_frame_id=None, source_observed_at_s=None):
        require(isinstance(blockage_id, str) and bool(blockage_id), "invalid blockage ID")
        require(stop_distance_m is None or finite(stop_distance_m) and stop_distance_m >= 0,
                "invalid blockage stop")
        require(alternative_status in ("unknown", "unavailable", "feasible"), "invalid alternative status")
        require(all(type(v) is bool for v in (verified_blockage, ordinary_follow, signal_only,
                                              verified_clear, usable)), "invalid blockage flags")
        require(finite(speed_cap_mps) and speed_cap_mps >= 0, "invalid blockage cap")
        require(all(v is None for v in (source, source_frame_id, source_observed_at_s)) or
                isinstance(source, str) and bool(source) and type(source_frame_id) is int and source_frame_id >= 0
                and finite(source_observed_at_s) and source_observed_at_s >= 0, 'invalid blockage source identity')
        self.blockage_id, self.stop_distance_m = blockage_id, stop_distance_m
        self.verified_blockage, self.alternative_status = verified_blockage, alternative_status
        self.ordinary_follow, self.signal_only = ordinary_follow, signal_only
        self.verified_clear, self.usable, self.speed_cap_mps = verified_clear, usable, speed_cap_mps
        self.source, self.source_frame_id, self.source_observed_at_s = source, source_frame_id, source_observed_at_s


class BlockedRoadPolicy(object):
    def __init__(self, clear_frames=2, session=None):
        require(type(clear_frames) is int and clear_frames > 0, "invalid blockage release count")
        self.clear_frames, self.session = clear_frames, session or BehaviorSession()
        self.reset()

    def reset(self):
        self.session.reset()
        self._context, self._last_frame, self._clear = None, None, 0
        self._clear_source = None

    def evaluate(self, frame, observation, capabilities=None, feedback=None, safety_override=False):
        if not frame.current(frame.observed_at_s):
            self._clear, self._clear_source = 0, None
            request = self.session.tick(frame, capabilities, safety_override=True) if self.session.intent_id else None
            return PolicyResult(request, bool(request), "OBSERVATION_UNUSABLE", "SUSPENDED")
        if self._context != frame.context.key():
            self.reset()
            self._context = frame.context.key()
        if self._last_frame is not None and frame.frame_id < self._last_frame:
            old = self.session.finish(frame, False, "GPS_FRAME_ROLLBACK") if self.session.intent_id else None
            self.reset()
            self._context, self._last_frame = frame.context.key(), frame.frame_id
            return PolicyResult(old, True, "GPS_FRAME_ROLLBACK", "RECOVER")
        distinct = self._last_frame is None or frame.frame_id > self._last_frame
        self._last_frame = frame.frame_id
        if observation is None or not isinstance(observation, BlockingObservation) or not observation.usable:
            self._clear, self._clear_source = 0, None
            if self.session.intent_id:
                request = self.session.tick(frame, capabilities, safety_override=True)
                return PolicyResult(request, True, "BLOCKAGE_EVIDENCE_UNKNOWN", "BLOCKED")
            return PolicyResult(reason="BLOCKAGE_EVIDENCE_UNKNOWN")
        excluded = (observation.ordinary_follow or observation.signal_only
                    or observation.alternative_status == "feasible")
        if excluded:
            self._clear = 0
            self._clear_source = None
            if self.session.intent_id:
                request = self.session.finish(frame, False, "NOT_UNRESOLVED_BLOCKAGE")
                return PolicyResult(request, False, request.reason_code, "CANCELLED")
            return PolicyResult(reason="NOT_UNRESOLVED_BLOCKAGE")
        if observation.verified_clear:
            source = ((observation.source, observation.source_frame_id, observation.source_observed_at_s)
                      if observation.source is not None else ('legacy_explicit_frame', frame.frame_id, frame.observed_at_s))
            prior = self._clear_source
            fresh = prior is None or (source[0] == prior[0] and source[1] > prior[1] and source[2] > prior[2])
            repeated = source == prior
            if prior is not None and not fresh and not repeated:
                self._clear = 0
                fresh = source[0] != prior[0]
            if (not frame.current(frame.observed_at_s) or frame.paused or safety_override
                    or source[1] > frame.frame_id or source[2] > frame.observed_at_s):
                self._clear, self._clear_source = 0, None
            elif distinct and fresh:
                self._clear += 1
                self._clear_source = source
            if self.session.intent_id and self._clear >= self.clear_frames:
                request = self.session.finish(frame, True, "BLOCKAGE_CLEARED")
                return PolicyResult(request, False, request.reason_code, "COMPLETE")
            request = self.session.tick(frame, capabilities, feedback, safety_override) if self.session.intent_id else None
            return PolicyResult(request, bool(request), "BLOCKAGE_CLEAR_CONFIRMING", "HOLD")
        self._clear = 0
        self._clear_source = None
        if not observation.verified_blockage or observation.stop_distance_m is None:
            return PolicyResult(reason="BLOCKAGE_BOUNDARY_UNVERIFIED")
        goal = BehaviorGoal(stop_distance_m=observation.stop_distance_m,
                            speed_cap_mps=observation.speed_cap_mps, precision_stop=True,
                            light_intent=LightIntent(hazard_signal=True))
        self.session.start(frame, "blocked:" + observation.blockage_id, "BLOCKED_ROAD",
                           "STOP_AND_WARN", goal, ("PATH_STOP", "LIGHTS", "FEEDBACK"))
        request = self.session.tick(frame, capabilities, feedback, safety_override)
        arrived = (self.session.last_feedback is not None
                   and self.session.last_feedback.producer in ("control", "runtime")
                   and self.session.last_feedback.goal_pose_arrived
                   and self.session.last_feedback.actual_standstill_confirmed)
        return PolicyResult(request, arrived, request.reason_code, "BLOCKED_HOLD" if arrived else "APPROACH")
