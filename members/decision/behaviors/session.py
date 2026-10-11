"""Stable intent identity, bounded retries and feedback-driven stage lifecycle."""

from members.decision.behaviors.contract import (
    BehaviorFrame, BehaviorGoal, BehaviorRequest, Capabilities, FeedbackValidator,
    LightIntent, finite, require,
)


class BehaviorSession(object):
    def __init__(self, feedback_max_age_s=0.5, acknowledgement_timeout_s=2.0,
                 progress_timeout_s=5.0, retry_delay_s=0.5, max_retries=2,
                 max_observation_gap_s=0.5):
        require(all(finite(v) and v > 0 for v in (acknowledgement_timeout_s,
                progress_timeout_s, retry_delay_s, max_observation_gap_s)), "invalid session timeouts")
        require(type(max_retries) is int and max_retries >= 0, "invalid retry limit")
        self.validator = FeedbackValidator(feedback_max_age_s)
        self.acknowledgement_timeout_s = acknowledgement_timeout_s
        self.progress_timeout_s, self.retry_delay_s = progress_timeout_s, retry_delay_s
        self.max_retries, self.max_observation_gap_s = max_retries, max_observation_gap_s
        self._counter = 0
        self.reset()

    def reset(self):
        self.context = None
        self.key = self.intent_id = self.maneuver = self.stage = None
        self.revision, self.source_frame_id = 0, -1
        self._issued_at_s = 0.0
        self.goal, self.required = None, ()
        self.status, self.reason_code = "IDLE", "NO_INTENT"
        self.attempt, self._elapsed, self._no_progress = 0, 0.0, 0.0
        self._total_retries = 0
        self._last_frame, self._last_time, self._progress = None, None, 0.0
        self._hold_progress = 0.0
        self._actual_progress_m = 0.0
        self._suspended, self._dispatched = False, False
        self.last_feedback, self.feedback_distinct = None, False
        self.events = []
        self.validator.reset()

    def _event(self, frame, reason):
        reply = self.last_feedback
        self.events.append(dict(frame_id=frame.frame_id, observed_at_s=frame.observed_at_s,
                                task_context=None if self.context is None else self.context.to_dict(),
                                intent_id=self.intent_id, stage=self.stage, revision=self.revision,
                                status=self.status, reason_code=reason, attempt=self._total_retries,
                                goal=None if self.goal is None else self.goal.to_dict(),
                                required_capabilities=list(self.required),
                                feedback=None if reply is None else dict(producer=reply.producer,
                                    status=reply.status, source_frame_id=reply.source_frame_id,
                                    produced_frame_id=reply.produced_frame_id, progress=reply.progress,
                                    standstill_duration_s=reply.standstill_duration_s,
                                    reason_code=reply.reason_code)))
        del self.events[:-128]

    @staticmethod
    def _goal_key(goal):
        pose = None if goal.goal_pose is None else (goal.goal_pose.x, goal.goal_pose.y,
                                                   goal.goal_pose.body_heading_rad)
        return (goal.target_lane_id, goal.parking_space_id, pose, goal.motion_direction,
                goal.precision_stop, goal.minimum_standstill_duration_s,
                goal.parking_brake_at_stop, tuple(sorted(goal.light_intent.to_dict().items())),
                goal.stop_obligation_id)

    def start(self, frame, key, maneuver, stage, goal, required=()):
        require(isinstance(frame, BehaviorFrame) and isinstance(goal, BehaviorGoal), "invalid intent start")
        if not frame.current(frame.observed_at_s) or frame.paused:
            require(self.intent_id is not None, "unusable frame cannot create an intent")
            if self.status in ("BLOCKED", "COMPLETED", "CANCELLED"):
                return self.snapshot(frame)
            self._suspended, self._dispatched = True, False
            self.status, self.reason_code = "SUSPENDED", "OBSERVATION_UNUSABLE"
            return self.snapshot(frame)
        require(isinstance(key, str) and bool(key) and isinstance(maneuver, str)
                and isinstance(stage, str) and all(isinstance(v, str) for v in required), "invalid intent identity")
        if (self.context is not None and self.context.key() == frame.context.key()
                and self.key == key and self.maneuver == maneuver
                and self.status not in ("CANCELLED", "COMPLETED")):
            if self.status == "BLOCKED" and self.reason_code != "CAPABILITY_OR_CONTRACT_UNAVAILABLE":
                return self.snapshot(frame)
            if self._goal_key(self.goal) != self._goal_key(goal):
                self.goal = goal
                self._revise(frame, "SEMANTIC_GOAL_CHANGED")
            else:
                self.goal = goal
            return self.snapshot(frame)
        self.reset()
        self._counter += 1
        self.context, self.key, self.maneuver, self.stage = frame.context, key, maneuver, stage
        self.intent_id = "decision:{0}:{1}".format(frame.context.token(), self._counter)
        self.revision, self.source_frame_id = 1, frame.frame_id
        self._issued_at_s = frame.observed_at_s
        self.goal, self.required = goal, tuple(required)
        self.status, self.reason_code = "REQUESTED", "INTENT_CREATED"
        self._last_frame, self._last_time = frame.frame_id, frame.observed_at_s
        self._event(frame, self.reason_code)
        return self.snapshot(frame)

    def advance(self, frame, stage, goal=None, required=None, reason="STAGE_ADVANCED",
                force_revision=False):
        require(self.intent_id is not None and self.context.key() == frame.context.key()
                and frame.current(frame.observed_at_s) and not frame.paused, "no current matching active intent")
        require(isinstance(stage, str) and bool(stage), "invalid stage")
        require(type(force_revision) is bool, "invalid revision flag")
        if self.status in ("COMPLETED", "CANCELLED") or (self.status == "BLOCKED"
                and self.reason_code != "CAPABILITY_OR_CONTRACT_UNAVAILABLE"):
            return self.snapshot(frame)
        if stage == self.stage and not force_revision:
            if goal is not None:
                self.goal = goal
            return self.snapshot(frame)
        self.stage, self.revision, self.source_frame_id = stage, self.revision + 1, frame.frame_id
        self._issued_at_s = frame.observed_at_s
        if goal is not None:
            require(isinstance(goal, BehaviorGoal), "invalid stage goal")
            self.goal = goal
        if required is not None:
            self.required = tuple(required)
        self.status, self.reason_code = "REQUESTED", reason
        self.attempt, self._elapsed, self._no_progress, self._progress = 0, 0.0, 0.0, 0.0
        self.last_feedback, self.feedback_distinct = None, False
        self._dispatched = False
        self._hold_progress = 0.0
        self._actual_progress_m = 0.0
        self._event(frame, reason)
        return self.snapshot(frame)

    def finish(self, frame, completed=True, reason="INTENT_COMPLETED"):
        if completed:
            require(self.context is not None and self.context.key() == frame.context.key()
                    and frame.current(frame.observed_at_s) and not frame.paused,
                    "completion needs current matching observation")
        self.status = "COMPLETED" if completed else "CANCELLED"
        self.reason_code = reason
        self._dispatched = False
        self.goal = BehaviorGoal(motion_direction=self.goal.motion_direction if self.goal else 1,
                                 light_intent=LightIntent())
        self._event(frame, reason)
        return self.snapshot(frame)

    def block(self, frame, reason):
        if self.status in ("COMPLETED", "CANCELLED") or (self.status == "BLOCKED"
                and self.reason_code != "CAPABILITY_OR_CONTRACT_UNAVAILABLE"):
            return self.snapshot(frame)
        if self.goal is not None:
            old = self.goal
            self.goal = BehaviorGoal(target_lane_id=old.target_lane_id, parking_space_id=old.parking_space_id,
                goal_pose=old.goal_pose, motion_direction=old.motion_direction, speed_cap_mps=0.0,
                stop_distance_m=old.stop_distance_m, precision_stop=old.precision_stop,
                minimum_standstill_duration_s=old.minimum_standstill_duration_s,
                parking_brake_at_stop=old.parking_brake_at_stop, stop_obligation_id=old.stop_obligation_id,
                light_intent=LightIntent(hazard_signal=old.light_intent.hazard_signal))
        self.status, self.reason_code, self._dispatched = "BLOCKED", reason, False
        self._event(frame, reason)
        return self.snapshot(frame)

    def retry_after_evidence(self, frame, reason):
        if (self.status != "BLOCKED" or not frame.current(frame.observed_at_s)
                or frame.paused or self._total_retries >= self.max_retries
                or self.context.key() != frame.context.key()):
            return False
        self.attempt += 1
        self._total_retries += 1
        self._revise(frame, reason)
        return True

    def _revise(self, frame, reason):
        self.revision += 1
        self.source_frame_id = frame.frame_id
        self._issued_at_s = frame.observed_at_s
        self._elapsed = self._no_progress = self._progress = self._hold_progress = 0.0
        self._actual_progress_m = 0.0
        self.last_feedback, self.feedback_distinct = None, False
        self.status, self.reason_code = "REQUESTED", reason
        self._dispatched = False
        self._event(frame, reason)

    def tick(self, frame, capabilities=None, feedback=None, safety_override=False):
        if self.intent_id is None:
            return None
        require(isinstance(frame, BehaviorFrame) and type(safety_override) is bool, "invalid behavior tick")
        self.last_feedback, self.feedback_distinct = None, False
        if self.status in ("COMPLETED", "CANCELLED"):
            return self.snapshot(frame)
        if self.status == "BLOCKED" and self.reason_code != "CAPABILITY_OR_CONTRACT_UNAVAILABLE":
            if frame.current(frame.observed_at_s) and self.context.key() != frame.context.key():
                return self.finish(frame, False, "TASK_CONTEXT_CHANGED")
            return self.snapshot(frame)
        if not frame.current(frame.observed_at_s):
            self._suspended, self._dispatched = True, False
            self.status, self.reason_code = "SUSPENDED", "OBSERVATION_UNUSABLE"
            return self.snapshot(frame)
        if self.context.key() != frame.context.key():
            return self.finish(frame, False, "TASK_CONTEXT_CHANGED")
        if self._last_frame is not None and frame.frame_id < self._last_frame:
            self._last_frame,self._last_time=frame.frame_id,frame.observed_at_s
            self._suspended,self._dispatched=True,False
            self.status,self.reason_code="SUSPENDED","GPS_FRAME_ROLLBACK_REVALIDATE"
            self._event(frame,self.reason_code)
            return self.snapshot(frame)
        distinct = self._last_frame is None or frame.frame_id > self._last_frame
        dt = 0.0 if self._last_time is None else frame.observed_at_s - self._last_time
        if dt < 0:
            return self.block(frame, "MONOTONIC_CLOCK_ROLLBACK")
        self._last_frame, self._last_time = frame.frame_id, frame.observed_at_s
        if self.status in ("COMPLETED", "CANCELLED"):
            return self.snapshot(frame)
        if (not frame.current(frame.observed_at_s) or frame.paused or safety_override
                or (distinct and dt > self.max_observation_gap_s)):
            self._suspended, self._dispatched = True, False
            self.status = "SUSPENDED"
            self.reason_code = ("SAFETY_OVERRIDE" if safety_override else
                                "PAUSED" if frame.paused else "OBSERVATION_UNUSABLE_OR_GAPPED")
            return self.snapshot(frame)
        if self._suspended:
            self._suspended = False
            self._revise(frame, "REVALIDATE_AFTER_SUSPENSION")
            dt = 0.0
        allowed = isinstance(capabilities, Capabilities) and capabilities.allows(frame, self.required)
        if not allowed:
            if self._dispatched:
                self._suspended = True
            self._dispatched = False
            self.status, self.reason_code = "BLOCKED", "CAPABILITY_OR_CONTRACT_UNAVAILABLE"
            return self.snapshot(frame)
        if self.status == "BLOCKED":
            if self.reason_code != "CAPABILITY_OR_CONTRACT_UNAVAILABLE":
                return self.snapshot(frame)
            self._revise(frame, "CAPABILITY_AVAILABLE")
            dt = 0.0
        if distinct and self._dispatched:
            self._elapsed += dt
            self._no_progress += dt
        if self.status == "RETRY_WAIT":
            if self._elapsed >= self.retry_delay_s:
                self._revise(frame, "RETRY_DISPATCH")
            else:
                return self.snapshot(frame)
        self._dispatched = True
        request = self.snapshot(frame)
        if feedback is not None:
            usable, reason, fresh = self.validator.check(feedback, request, frame)
            self.reason_code = reason
            if usable and fresh and distinct:
                self.last_feedback, self.feedback_distinct = feedback, True
                if feedback.status == "REJECTED":
                    if feedback.retryable and self._total_retries < self.max_retries:
                        self.attempt += 1
                        self._total_retries += 1
                        self.status, self.reason_code = "RETRY_WAIT", feedback.reason_code or "PATH_REJECTED_RETRY"
                        self._elapsed = 0.0
                    else:
                        return self.block(frame, feedback.reason_code or "RETRY_LIMIT_OR_NONRETRYABLE")
                elif feedback.status == "FAILED":
                    return self.block(frame, feedback.reason_code or "EXECUTION_FAILED")
                else:
                    if feedback.progress > self._progress:
                        self._progress = feedback.progress
                        self._no_progress = 0.0
                    if (feedback.producer in ("control", "runtime")
                            and feedback.actual_progress_m > self._actual_progress_m + 0.01):
                        self._actual_progress_m = feedback.actual_progress_m
                        self._no_progress = 0.0
                    if (feedback.producer in ("control", "runtime")
                            and feedback.goal_pose_arrived and feedback.actual_standstill_confirmed
                            and feedback.standstill_duration_s > self._hold_progress):
                        self._hold_progress = feedback.standstill_duration_s
                        self._no_progress = 0.0
                    if feedback.producer == "planning" and feedback.status == "PLANNED":
                        if self.status == "REQUESTED":
                            self.status = "ACCEPTED"
                            self._no_progress = 0.0
                    elif feedback.producer in ("control", "runtime"):
                        self.status = "EXECUTING"
                self._event(frame, self.reason_code)
        if self.status == "REQUESTED" and self._elapsed >= self.acknowledgement_timeout_s:
            return self.block(frame, "PLANNING_ACK_TIMEOUT")
        if self.status in ("ACCEPTED", "EXECUTING") and self._no_progress >= self.progress_timeout_s:
            return self.block(frame, "EXECUTION_PROGRESS_TIMEOUT")
        return self.snapshot(frame)

    def snapshot(self, frame):
        if self.intent_id is None:
            return None
        allowed = self._dispatched and self.status in ("REQUESTED", "ACCEPTED", "EXECUTING")
        request = BehaviorRequest(frame, self.intent_id, self.maneuver, self.stage,
                                  self.revision, self.source_frame_id, self.goal,
                                  dispatch_allowed=allowed, status=self.status,
                                  reason_code=self.reason_code, attempt=self._total_retries,
                                  issued_at_s=self._issued_at_s)
        request.context = self.context
        return request
