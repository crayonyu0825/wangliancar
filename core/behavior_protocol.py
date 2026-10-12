"""Shared SDK-independent behavior identities, goals and feedback.

All times are seconds in an explicitly shared process-monotonic clock. This is
the shared single-process clock; RuntimeRequest adds explicit candidate authority.
"""

import hashlib
import json
import math

CONTRACT_VERSION = "r05-r09-decision-proposal-v1"
STATUSES = ("PLANNED", "EXECUTING", "ARRIVED", "DWELLING", "COMPLETED", "REJECTED", "FAILED")
PRODUCERS = ("planning", "control", "runtime")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


class TaskContext(object):
    __slots__ = ("case_id", "task_id", "scene_id", "session_id")

    def __init__(self, case_id, task_id, scene_id, session_id):
        require(all(isinstance(v, str) for v in (case_id, task_id, session_id))
                and bool(session_id) and type(scene_id) is int, "invalid task context")
        self.case_id, self.task_id = case_id, task_id
        self.scene_id, self.session_id = scene_id, session_id

    def key(self):
        return (self.case_id, self.task_id, self.scene_id, self.session_id)

    def token(self):
        return hashlib.sha256(json.dumps(self.key(), ensure_ascii=True).encode("utf-8")).hexdigest()[:16]

    def to_dict(self):
        return dict(case_id=self.case_id, task_id=self.task_id, scene_id=self.scene_id,
                    session_id=self.session_id)


class BehaviorFrame(object):
    __slots__ = ("context", "frame_id", "observed_at_s", "valid_until_s", "clock_id",
                 "usable", "paused", "ego_x", "ego_y", "ego_heading", "ego_speed")

    def __init__(self, context, frame_id, observed_at_s, valid_until_s,
                 ego_x=0.0, ego_y=0.0, ego_heading=0.0, ego_speed=0.0,
                 usable=False, paused=False, clock_id="process_monotonic"):
        require(isinstance(context, TaskContext) and type(frame_id) is int and frame_id >= 0,
                "invalid behavior frame identity")
        require(all(finite(v) for v in (observed_at_s, valid_until_s, ego_x, ego_y,
                                        ego_heading, ego_speed)) and ego_speed >= 0,
                "invalid behavior frame numbers")
        require(type(usable) is bool and type(paused) is bool and isinstance(clock_id, str)
                and bool(clock_id), "invalid behavior quality")
        self.context, self.frame_id = context, frame_id
        self.observed_at_s, self.valid_until_s = observed_at_s, valid_until_s
        self.clock_id, self.usable, self.paused = clock_id, usable, paused
        self.ego_x, self.ego_y = ego_x, ego_y
        self.ego_heading, self.ego_speed = ego_heading, ego_speed

    def current(self, now):
        return self.usable and finite(now) and self.observed_at_s <= now < self.valid_until_s


class LightIntent(object):
    __slots__ = ("left_signal", "right_signal", "hazard_signal")

    def __init__(self, left_signal=False, right_signal=False, hazard_signal=False):
        require(all(type(v) is bool for v in (left_signal, right_signal, hazard_signal)),
                "light intentions must be bool")
        require(not (left_signal and right_signal), "use hazard rather than both indicators")
        self.left_signal, self.right_signal, self.hazard_signal = left_signal, right_signal, hazard_signal

    def to_dict(self):
        return dict(left_signal=self.left_signal, right_signal=self.right_signal,
                    hazard_signal=self.hazard_signal)


class GoalPose(object):
    __slots__ = ("x", "y", "body_heading_rad")

    def __init__(self, x, y, body_heading_rad):
        require(all(finite(v) for v in (x, y, body_heading_rad)), "invalid goal pose")
        self.x, self.y, self.body_heading_rad = x, y, body_heading_rad

    def to_dict(self):
        return dict(x=self.x, y=self.y, body_heading_rad=self.body_heading_rad,
                    reference_point="ego_rear_axle")


class BehaviorGoal(object):
    __slots__ = ("target_lane_id", "parking_space_id", "goal_pose", "motion_direction",
                 "speed_cap_mps", "stop_distance_m", "precision_stop",
                 "minimum_standstill_duration_s", "parking_brake_at_stop", "light_intent", "stop_obligation_id")

    def __init__(self, target_lane_id=None, parking_space_id=None, goal_pose=None,
                 motion_direction=1, speed_cap_mps=0.0, stop_distance_m=-1.0,
                 precision_stop=False, minimum_standstill_duration_s=0.0,
                 parking_brake_at_stop=False, light_intent=None, stop_obligation_id=None):
        require(target_lane_id is None or isinstance(target_lane_id, str), "invalid target lane")
        require(parking_space_id is None or type(parking_space_id) is int, "invalid parking ID")
        require(goal_pose is None or isinstance(goal_pose, GoalPose), "invalid goal pose type")
        require(type(motion_direction) is int and motion_direction in (-1, 1), "invalid direction")
        require(finite(speed_cap_mps) and speed_cap_mps >= 0 and finite(stop_distance_m)
                and (stop_distance_m == -1 or stop_distance_m >= 0), "invalid goal speed/distance")
        require(finite(minimum_standstill_duration_s) and minimum_standstill_duration_s >= 0,
                "invalid minimum standstill duration")
        require(type(precision_stop) is bool and type(parking_brake_at_stop) is bool,
                "invalid stop flags")
        self.target_lane_id, self.parking_space_id, self.goal_pose = target_lane_id, parking_space_id, goal_pose
        self.motion_direction, self.speed_cap_mps = motion_direction, speed_cap_mps
        self.stop_distance_m, self.precision_stop = stop_distance_m, precision_stop
        self.minimum_standstill_duration_s = minimum_standstill_duration_s
        self.parking_brake_at_stop = parking_brake_at_stop
        require(stop_obligation_id is None or isinstance(stop_obligation_id, str), "invalid stop obligation ID")
        self.stop_obligation_id = stop_obligation_id
        self.light_intent = light_intent if light_intent is not None else LightIntent()
        require(isinstance(self.light_intent, LightIntent), "invalid lights type")

    def to_dict(self):
        return dict(target_lane_id=self.target_lane_id, parking_space_id=self.parking_space_id,
                    goal_pose=None if self.goal_pose is None else self.goal_pose.to_dict(),
                    motion_direction=self.motion_direction, speed_cap_mps=self.speed_cap_mps,
                    stop_distance_m=self.stop_distance_m, precision_stop=self.precision_stop,
                    minimum_standstill_duration_s=self.minimum_standstill_duration_s,
                    parking_brake_at_stop=self.parking_brake_at_stop,
                    light_intent=self.light_intent.to_dict(), stop_obligation_id=self.stop_obligation_id)


class BehaviorRequest(object):
    __slots__ = ("context", "intent_id", "maneuver", "stage", "revision", "source_frame_id",
                 "produced_frame_id", "produced_at_s", "issued_at_s", "valid_until_s", "clock_id", "goal",
                 "dispatch_allowed", "status", "reason_code", "attempt")

    def __init__(self, frame, intent_id, maneuver, stage, revision, source_frame_id,
                 goal, dispatch_allowed=False, status="PROPOSED", reason_code="CONTRACT_UNCONNECTED", attempt=0, issued_at_s=None):
        require(isinstance(goal, BehaviorGoal) and type(revision) is int and revision >= 1,
                "invalid behavior request")
        self.context, self.intent_id, self.maneuver, self.stage = frame.context, intent_id, maneuver, stage
        self.revision, self.source_frame_id = revision, source_frame_id
        self.produced_frame_id, self.produced_at_s = frame.frame_id, frame.observed_at_s
        self.issued_at_s = frame.observed_at_s if issued_at_s is None else issued_at_s
        self.valid_until_s, self.clock_id = frame.valid_until_s, frame.clock_id
        self.goal, self.dispatch_allowed = goal, dispatch_allowed
        self.status, self.reason_code, self.attempt = status, reason_code, attempt

    def to_dict(self):
        result = dict(contract_version=CONTRACT_VERSION, interface_status="proposal_not_runtime_connected",
                      task_context=self.context.to_dict(), intent_id=self.intent_id,
                      maneuver=self.maneuver, stage=self.stage, revision=self.revision,
                      source_frame_id=self.source_frame_id, produced_frame_id=self.produced_frame_id,
                      produced_at_s=self.produced_at_s, issued_at_s=self.issued_at_s, valid_until_s=self.valid_until_s,
                      clock_id=self.clock_id, dispatch_allowed=self.dispatch_allowed,
                      status=self.status, reason_code=self.reason_code, attempt=self.attempt)
        result.update(self.goal.to_dict())
        return result


class Capabilities(object):
    __slots__ = ("context", "actions", "produced_at_s", "valid_until_s", "clock_id", "usable", "version")

    def __init__(self, context, actions=(), produced_at_s=0.0, valid_until_s=0.0,
                 usable=False, clock_id="process_monotonic", version=CONTRACT_VERSION):
        require(isinstance(context, TaskContext) and type(usable) is bool,
                "invalid capabilities")
        require(all(isinstance(v, str) for v in actions)
                and finite(produced_at_s) and finite(valid_until_s), "invalid capability values")
        self.context, self.actions = context, frozenset(actions)
        self.produced_at_s, self.valid_until_s = produced_at_s, valid_until_s
        self.clock_id, self.usable, self.version = clock_id, usable, version

    def allows(self, frame, required):
        return (self.usable and self.version == CONTRACT_VERSION
                and self.context.key() == frame.context.key() and self.clock_id == frame.clock_id
                and self.produced_at_s <= frame.observed_at_s < self.valid_until_s
                and set(required) <= self.actions)


class ExecutionFeedback(object):
    __slots__ = ("context", "intent_id", "stage", "revision", "producer", "source_frame_id",
                 "produced_frame_id", "produced_at_s", "valid_until_s", "clock_id", "usable",
                 "status", "reason_code", "retryable", "progress", "actual_standstill_confirmed",
                 "goal_pose_arrived", "standstill_duration_s", "hold_completed", "lights_confirmed",
                 "motion_direction", "actual_pose", "actual_lane_id", "lights_duration_s", "candidate_id",
                 "lights_evidence_level", "evidence_level", "control_limited", "actual_progress_m", "actual_speed_mps")

    def __init__(self, context, intent_id, stage, revision, producer, source_frame_id,
                 produced_frame_id, produced_at_s, valid_until_s, status,
                 usable=False, reason_code="", retryable=False, progress=0.0,
                 actual_standstill_confirmed=False, goal_pose_arrived=False,
                 standstill_duration_s=0.0, hold_completed=False, lights_confirmed=False,
                 motion_direction=1, actual_pose=None, clock_id="process_monotonic",
                 actual_lane_id=None, lights_duration_s=0.0, candidate_id=None,
                 lights_evidence_level="unknown", evidence_level="unknown", control_limited=False, actual_progress_m=0.0, actual_speed_mps=0.0):
        self.context, self.intent_id, self.stage, self.revision = context, intent_id, stage, revision
        self.producer, self.source_frame_id, self.produced_frame_id = producer, source_frame_id, produced_frame_id
        self.produced_at_s, self.valid_until_s, self.clock_id = produced_at_s, valid_until_s, clock_id
        self.usable, self.status, self.reason_code, self.retryable = usable, status, reason_code, retryable
        self.progress = progress
        self.actual_standstill_confirmed, self.goal_pose_arrived = actual_standstill_confirmed, goal_pose_arrived
        self.standstill_duration_s, self.hold_completed = standstill_duration_s, hold_completed
        self.lights_confirmed, self.motion_direction, self.actual_pose = lights_confirmed, motion_direction, actual_pose
        self.actual_lane_id, self.lights_duration_s = actual_lane_id, lights_duration_s
        self.candidate_id = candidate_id
        self.lights_evidence_level, self.evidence_level = lights_evidence_level, evidence_level
        self.control_limited, self.actual_progress_m = control_limited, actual_progress_m
        self.actual_speed_mps = actual_speed_mps


class FeedbackValidator(object):
    def __init__(self, max_age_s=0.5):
        require(finite(max_age_s) and max_age_s > 0, "invalid feedback max age")
        self.max_age_s = max_age_s
        self._last = {}

    def reset(self):
        self._last.clear()

    def check(self, feedback, request, frame, record=True):
        if not isinstance(feedback, ExecutionFeedback):
            return False, "FEEDBACK_TYPE", False
        if not isinstance(feedback.context, TaskContext) or feedback.context.key() != frame.context.key():
            return False, "FEEDBACK_CONTEXT", False
        if (feedback.intent_id != request.intent_id or feedback.stage != request.stage
                or type(feedback.revision) is not int or feedback.revision != request.revision):
            return False, "FEEDBACK_INTENT_STAGE_REVISION", False
        producer_statuses = {"planning": ("PLANNED", "REJECTED", "FAILED"),
                             "control": ("EXECUTING", "ARRIVED", "DWELLING", "COMPLETED", "REJECTED", "FAILED"),
                             "runtime": ("EXECUTING", "ARRIVED", "DWELLING", "COMPLETED", "REJECTED", "FAILED")}
        if (feedback.producer not in PRODUCERS or feedback.status not in STATUSES
                or feedback.status not in producer_statuses.get(feedback.producer, ())):
            return False, "FEEDBACK_PRODUCER_STATUS", False
        if (type(feedback.source_frame_id) is not int or type(feedback.produced_frame_id) is not int
                or feedback.source_frame_id < request.source_frame_id
                or feedback.produced_frame_id < feedback.source_frame_id
                or feedback.produced_frame_id > frame.frame_id):
            return False, "FEEDBACK_FRAME", False
        if (feedback.clock_id != frame.clock_id or not finite(feedback.produced_at_s)
                or not finite(feedback.valid_until_s)
                or feedback.produced_at_s > frame.observed_at_s
                or feedback.produced_at_s < request.issued_at_s
                or frame.observed_at_s >= feedback.valid_until_s
                or frame.observed_at_s - feedback.produced_at_s > self.max_age_s):
            return False, "FEEDBACK_TIME", False
        flags = (feedback.usable, feedback.retryable, feedback.actual_standstill_confirmed,
                 feedback.goal_pose_arrived, feedback.hold_completed, feedback.lights_confirmed)
        if (any(type(v) is not bool for v in flags) or not feedback.usable
                or not finite(feedback.progress) or not 0 <= feedback.progress <= 1
                or not finite(feedback.standstill_duration_s) or feedback.standstill_duration_s < 0
                or type(feedback.motion_direction) is not int or feedback.motion_direction not in (-1, 1)
                or (feedback.actual_pose is not None and (not isinstance(feedback.actual_pose, GoalPose)
                    or not all(finite(v) for v in (feedback.actual_pose.x, feedback.actual_pose.y,
                                                   feedback.actual_pose.body_heading_rad))))
                or (feedback.actual_lane_id is not None and not isinstance(feedback.actual_lane_id, str))
                or not finite(feedback.lights_duration_s) or feedback.lights_duration_s < 0
                or not finite(feedback.actual_progress_m) or feedback.actual_progress_m < 0
                or not finite(feedback.actual_speed_mps) or feedback.actual_speed_mps < 0
                or type(feedback.control_limited) is not bool
                or not isinstance(feedback.reason_code, str)
                or (feedback.candidate_id is not None and not isinstance(feedback.candidate_id, str))):
            return False, "FEEDBACK_VALUES", False
        key = (feedback.producer, feedback.intent_id, feedback.stage, feedback.revision)
        sequence = (feedback.produced_frame_id, feedback.produced_at_s)
        previous = self._last.get(key)
        if previous is not None and sequence < previous:
            return False, "FEEDBACK_OUT_OF_ORDER", False
        distinct = previous is None or feedback.produced_frame_id > previous[0]
        if distinct and record:
            self._last[key] = sequence
            if len(self._last) > 128:
                self._last = {key: sequence}
        return True, "FEEDBACK_VALID" if distinct else "FEEDBACK_DUPLICATE", distinct


PROTOCOL_VERSION = "joint-behavior-runtime-v1"


class CandidateReference(object):
    __slots__ = ("candidate_id", "intent_id", "stage", "revision", "source_frame_id",
                 "issued_at_s", "valid_until_s")

    def __init__(self, candidate_id, request, valid_until_s):
        require(isinstance(candidate_id, str) and bool(candidate_id) and finite(valid_until_s),
                "invalid candidate reference")
        self.candidate_id, self.intent_id = candidate_id, request.intent_id
        self.stage, self.revision = request.stage, request.revision
        self.source_frame_id, self.issued_at_s = request.source_frame_id, request.issued_at_s
        self.valid_until_s = valid_until_s

    def key(self):
        return (self.intent_id, self.stage, self.revision, self.source_frame_id, self.issued_at_s)

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}


class RuntimeRequest(BehaviorRequest):
    __slots__ = ("protocol_version", "authorization", "candidate_reference", "origin_lane_id",
                 "constraint_ids", "target_id", "goal_region", "follow_clearance_m", "static_clearance_m")

    def __init__(self, frame, intent_id, maneuver, stage, revision, source_frame_id,
                 goal, authorization="EVALUATE_ONLY", candidate_reference=None,
                 origin_lane_id="", constraint_ids=(), target_id=None, goal_region=None,
                 issued_at_s=None, follow_clearance_m=10.5, static_clearance_m=0.5):
        require(finite(follow_clearance_m) and follow_clearance_m >= 0
                and finite(static_clearance_m) and static_clearance_m >= 0,"invalid behavior clearance")
        require(authorization in ("PREPARE", "EVALUATE_ONLY", "EXECUTE_SELECTED", "HOLD"),
                "invalid behavior authorization")
        require(candidate_reference is None or isinstance(candidate_reference, CandidateReference),
                "invalid adopted candidate")
        require(authorization != "EXECUTE_SELECTED" or candidate_reference is not None,
                "execution needs an evaluated candidate")
        require(isinstance(origin_lane_id, str) and all(isinstance(v, str) for v in constraint_ids),
                "invalid behavior constraint identity")
        require(target_id is None or type(target_id) is int and target_id >= 0, "invalid target identity")
        super(RuntimeRequest, self).__init__(frame, intent_id, maneuver, stage, revision,
            source_frame_id, goal, dispatch_allowed=True, status="REQUESTED", issued_at_s=issued_at_s)
        self.protocol_version, self.authorization = PROTOCOL_VERSION, authorization
        self.candidate_reference, self.origin_lane_id = candidate_reference, origin_lane_id
        self.constraint_ids, self.target_id, self.goal_region = tuple(constraint_ids), target_id, goal_region
        self.follow_clearance_m, self.static_clearance_m = follow_clearance_m, static_clearance_m

    def identity(self):
        return (self.intent_id, self.stage, self.revision, self.source_frame_id, self.issued_at_s)

    def to_dict(self):
        result = super(RuntimeRequest, self).to_dict()
        result.update(contract_version=PROTOCOL_VERSION,protocol_version=self.protocol_version, interface_status="runtime_connected",
                      authorization=self.authorization, origin_lane_id=self.origin_lane_id,
                      candidate_reference=None if self.candidate_reference is None else self.candidate_reference.to_dict(),
                      constraint_ids=list(self.constraint_ids), target_id=self.target_id, goal_region=self.goal_region,
                      follow_clearance_m=self.follow_clearance_m,static_clearance_m=self.static_clearance_m)
        return result


class PlanningFeedback(ExecutionFeedback):
    __slots__ = ("candidate_reference", "clearance_lower_bound_m", "computation_s", "goal_pose",
                 "segment_directions", "segment_end_poses", "bottleneck")

    def __init__(self, request, frame, status, reason_code="", candidate_reference=None,
                 clearance_lower_bound_m=None, computation_s=0.0, goal_pose=None,
                 segment_directions=(), segment_end_poses=(), retryable=False):
        super(PlanningFeedback, self).__init__(frame.context, request.intent_id, request.stage,
            request.revision, "planning", request.source_frame_id, frame.frame_id,
            frame.observed_at_s, frame.valid_until_s, status, usable=True,
            reason_code=reason_code, retryable=retryable,
            candidate_id=None if candidate_reference is None else candidate_reference.candidate_id)
        self.candidate_reference, self.clearance_lower_bound_m = candidate_reference, clearance_lower_bound_m
        self.computation_s, self.goal_pose = computation_s, goal_pose
        self.segment_directions, self.segment_end_poses = tuple(segment_directions), tuple(segment_end_poses)
        self.bottleneck, self.evidence_level = reason_code, "model_geometry_and_motion_verified"


def behavior_frame(perception, now_s):
    context = TaskContext(perception.case_id, perception.task_id, perception.scene_id, perception.session_id)
    return BehaviorFrame(context, perception.frame_id, now_s, perception.valid_until,
                         perception.ego.x, perception.ego.y, perception.ego.heading,
                         perception.ego.speed, usable=perception.valid)


class RuntimeCapabilities(Capabilities):
    __slots__ = ("vehicle_geometry", "protocol_version")

    def __init__(self, base, vehicle_geometry):
        require(isinstance(base, Capabilities) and isinstance(vehicle_geometry, dict)
                and all(finite(v) and v > 0 for v in vehicle_geometry.values()), "invalid runtime geometry")
        super(RuntimeCapabilities, self).__init__(base.context, base.actions, base.produced_at_s,
            base.valid_until_s, base.usable, base.clock_id, base.version)
        self.vehicle_geometry, self.protocol_version = dict(vehicle_geometry), PROTOCOL_VERSION
        self.version=PROTOCOL_VERSION

    def allows(self,frame,required):
        return (self.usable and self.version==PROTOCOL_VERSION and self.protocol_version==PROTOCOL_VERSION
                and self.context.key()==frame.context.key() and self.clock_id==frame.clock_id
                and self.produced_at_s<=frame.observed_at_s<self.valid_until_s
                and frame.current(frame.observed_at_s) and not frame.paused
                and set(required)<=self.actions)


def validate_runtime_request(request, frame):
    require(isinstance(request, RuntimeRequest) and request.protocol_version == PROTOCOL_VERSION,
            'behavior protocol version mismatch')
    require(request.context.key() == frame.context.key() and request.clock_id == frame.clock_id,
            'behavior context or clock mismatch')
    require(type(request.revision) is int and request.revision >= 1
            and type(request.source_frame_id) is int and 0 <= request.source_frame_id <= frame.frame_id
            and request.produced_frame_id == frame.frame_id
            and finite(request.issued_at_s) and finite(request.produced_at_s)
            and request.issued_at_s <= request.produced_at_s <= frame.observed_at_s
            and frame.observed_at_s < request.valid_until_s <= frame.valid_until_s,
            'behavior frame or time mismatch')
    require(request.authorization in ('PREPARE','EVALUATE_ONLY','EXECUTE_SELECTED','HOLD')
            and request.dispatch_allowed is True, 'behavior authority invalid')
    goal = request.goal
    require(isinstance(goal, BehaviorGoal) and finite(goal.speed_cap_mps) and goal.speed_cap_mps >= 0
            and finite(goal.stop_distance_m) and (goal.stop_distance_m == -1 or goal.stop_distance_m >= 0)
            and finite(goal.minimum_standstill_duration_s) and goal.minimum_standstill_duration_s >= 0
            and type(goal.motion_direction) is int and goal.motion_direction in (-1,1), 'behavior goal invalid')
    require(goal.goal_pose is None or (isinstance(goal.goal_pose, GoalPose)
            and all(finite(v) for v in (goal.goal_pose.x,goal.goal_pose.y,goal.goal_pose.body_heading_rad))),
            'behavior goal pose invalid')
    if request.authorization == 'EXECUTE_SELECTED':
        reference=request.candidate_reference
        require(isinstance(reference,CandidateReference) and reference.intent_id == request.intent_id
                and type(reference.revision) is int and 0 < reference.revision < request.revision
                and reference.source_frame_id <= request.source_frame_id and finite(reference.issued_at_s)
                and reference.issued_at_s <= request.issued_at_s and finite(reference.valid_until_s),
                'behavior candidate adoption invalid')
    return request
