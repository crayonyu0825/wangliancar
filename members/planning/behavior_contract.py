"""Common P01 request boundary for explicit component/runtime consumers.

Consume the *flat* BehaviorRequest.to_dict() proposal at decision PR20 fc42515.
Capabilities is an explicit dictionary of that proposal's attributes, with its
context represented as a TaskContext.to_dict() dictionary. No decision-private
types, diagnostic snapshots or public protocol extensions are used here.
KEEP_LANE/FOLLOW/STOP are a private trial subset of existing public modes, not
PR20's real new behaviors. Explicit runtime_stop and runtime_lane_change opt-ins
reuse the identity/time gates with their own goal/stage/capability checks.

READY_FOR_COMPONENT means only that this contract passed the checks below. It
is not planning acceptance, a feasible trajectory, execution feedback or proof
of arrival. The caller owns active identity/revision and current frame quality;
this stateless boundary does not duplicate the decision session lifecycle.
"""

import copy
import math


CONTRACT_VERSION = "r05-r09-decision-proposal-v1"
COMPONENT_ACTIONS = ("KEEP_LANE", "FOLLOW", "STOP")
LANE_CHANGE_ACTIONS = ("LANE_CHANGE", "AVOID", "OVERTAKE", "MERGE")
DRIVING_ACTIONS = ('INTERSECTION','CROSSWALK','JUNCTION_ENTRY','EMERGENCY')
_CONTEXT_KEYS = frozenset(("case_id", "task_id", "scene_id", "session_id"))
_IDENTITY_KEYS = frozenset(("intent_id", "stage", "revision", "source_frame_id", "issued_at_s"))
_CAPABILITY_KEYS = frozenset(("context", "actions", "produced_at_s", "valid_until_s",
                              "clock_id", "usable", "version"))
_LIGHT_KEYS = frozenset(("left_signal", "right_signal", "hazard_signal"))
_POSE_KEYS = frozenset(("x", "y", "body_heading_rad", "reference_point"))
_REQUEST_KEYS = frozenset((
    "contract_version", "interface_status", "task_context", "intent_id", "maneuver", "stage",
    "revision", "source_frame_id", "produced_frame_id", "produced_at_s", "issued_at_s",
    "valid_until_s", "clock_id", "dispatch_allowed", "status", "reason_code", "attempt",
    "target_lane_id", "parking_space_id", "goal_pose", "motion_direction", "speed_cap_mps",
    "stop_distance_m", "precision_stop", "minimum_standstill_duration_s",
    "parking_brake_at_stop", "light_intent", "stop_obligation_id"))


class _Rejected(Exception):
    def __init__(self, reason_code, retryable=False, missing=()):
        self.reason_code, self.retryable, self.missing = reason_code, retryable, missing


def _require(condition, reason_code, retryable=False, missing=()):
    if not condition:
        raise _Rejected(reason_code, retryable, missing)


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _text(value):
    return isinstance(value, str) and bool(value)


def _keys(value, expected, reason):
    _require(isinstance(value, dict) and set(value) == expected, reason)


def _context(value, reason):
    _keys(value, _CONTEXT_KEYS, reason)
    _require(all(isinstance(value[key], str) for key in ("case_id", "task_id", "session_id"))
             and bool(value["session_id"]) and type(value["scene_id"]) is int, reason)


def _strings(value, reason):
    _require(isinstance(value, (list, tuple, set, frozenset))
             and all(_text(item) for item in value), reason)
    return set(value)


PARKING_STAGES=('APPROACH','POSITION','REVERSE_ENTRY','ALIGN','PARKED_DWELL','EXIT_PREPARE','EXIT')


def _goal(request, current_lane_id, runtime_stop=False, runtime_lane_change=False,runtime_parking=False,
          runtime_driving=False):
    source=request.get('source_lane_id')
    _require(source is None or _text(source),'SOURCE_LANE_INVALID')
    _require(runtime_lane_change or source is None,'UNSUPPORTED_SOURCE_LANE')
    lane, parking, pose = (request[key] for key in ("target_lane_id", "parking_space_id", "goal_pose"))
    _require(lane is None or _text(lane), "TARGET_LANE_INVALID")
    _require(parking is None or type(parking) is int, "PARKING_TARGET_INVALID")
    if pose is not None:
        _keys(pose, _POSE_KEYS, "GOAL_POSE_INVALID")
        _require(all(_finite(pose[key]) for key in ("x", "y", "body_heading_rad"))
                 and pose["reference_point"] == "ego_rear_axle", "GOAL_POSE_INVALID")
    _require(type(request["motion_direction"]) is int
             and request["motion_direction"] in (-1, 1), "DIRECTION_INVALID")
    speed, distance, dwell = (request[key] for key in (
        "speed_cap_mps", "stop_distance_m", "minimum_standstill_duration_s"))
    _require(_finite(speed) and speed >= 0 and _finite(distance)
             and (distance == -1 or distance >= 0) and _finite(dwell) and dwell >= 0,
             "GOAL_NUMBERS_INVALID")
    _require(type(request["precision_stop"]) is bool
             and type(request["parking_brake_at_stop"]) is bool, "STOP_FLAGS_INVALID")
    obligation = request["stop_obligation_id"]
    _require(obligation is None or _text(obligation), "STOP_OBLIGATION_INVALID")
    lights = request["light_intent"]
    _keys(lights, _LIGHT_KEYS, "LIGHT_INTENT_INVALID")
    _require(all(type(value) is bool for value in lights.values())
             and not (lights["left_signal"] and lights["right_signal"]), "LIGHT_INTENT_INVALID")
    if runtime_parking:
        stage=request['stage']
        _require(request['maneuver']=='PARK' and stage in PARKING_STAGES,'PARKING_STAGE_NOT_PLANNABLE')
        _require(source is None and lane is None and parking is not None and parking>=0 and pose is not None,
                 'PARKING_FIXED_TARGET_REQUIRED')
        reverse=stage=='REVERSE_ENTRY'
        dwelling=stage=='PARKED_DWELL' or stage in ('REVERSE_ENTRY','ALIGN') and dwell>0
        _require(stage in ('ALIGN','PARKED_DWELL','EXIT_PREPARE') or
                 request['motion_direction']==(-1 if reverse else 1),'PARKING_STAGE_DIRECTION_MISMATCH')
        _require(speed>0 and distance==-1 and request['precision_stop'] and not any(lights.values()),
                 'PARKING_GOAL_EXTENSIONS_UNSUPPORTED')
        _require((dwelling and dwell>=10 and request['parking_brake_at_stop']
                  and obligation=='parking:'+str(parking)) or
                 (not dwelling and dwell==0 and not request['parking_brake_at_stop'] and obligation is None),
                 'PARKING_DWELL_OBLIGATION_INVALID')
        _require(_text(current_lane_id),'CURRENT_LANE_UNKNOWN',True)
        return
    if runtime_driving:
        _require(source is None and parking is None and request['motion_direction']==1
                 and _text(current_lane_id) and (lane is None or lane==current_lane_id),
                 'DRIVING_GOAL_SCOPE_INVALID')
        _require(dwell==0 and not request['parking_brake_at_stop'] and obligation is None,
                 'DRIVING_STOP_OBLIGATION_CONFLICT')
        if request['maneuver']=='EMERGENCY':
            _require(request['stage']=='SELECT' or request['stage'] in
                     ('EXECUTE_BRAKE','EXECUTE_LEFT','EXECUTE_RIGHT','EXECUTE_BRAKE_LEFT','EXECUTE_BRAKE_RIGHT',
                      'BRAKE_FALLBACK'),
                     'EMERGENCY_STAGE_INVALID')
            _require(lights['hazard_signal'] and not lights['left_signal'] and not lights['right_signal'],
                     'EMERGENCY_HAZARD_REQUIRED')
            _require(request['stage']=='SELECT' or pose is not None and speed>0
                     and _text(request.get('candidate_id')), 'EMERGENCY_CANDIDATE_AUTHORITY_REQUIRED')
        else:
            _require(pose is None and request.get('candidate_id') is None,
                     'PASSAGE_UNSUPPORTED_FIXED_GOAL_OR_CANDIDATE')
            _require(request['stage'] in ('WAIT_ENTRY','ENTER','EXIT') and speed>0
                     and not any(lights.values()), 'PASSAGE_STAGE_INVALID')
            _require(request['stage']!='WAIT_ENTRY' or distance>=0 and request['precision_stop'],
                     'PASSAGE_WAIT_STOP_REQUIRED')
        return
    stop_required = (request["maneuver"] == "STOP" or request["precision_stop"]
                     or dwell > 0 or request["parking_brake_at_stop"] or obligation is not None)
    _require(not stop_required or distance >= 0, "STOP_DISTANCE_UNKNOWN", True)
    _require(not (request["precision_stop"] and distance > 0 and speed == 0),
             "PRECISION_APPROACH_SPEED_UNAVAILABLE", True)
    _require(request["motion_direction"] == 1, "UNSUPPORTED_DIRECTION")
    _require(_text(current_lane_id), "CURRENT_LANE_UNKNOWN", True)
    if runtime_lane_change:
        _require(_text(source) and _text(lane) and source!=lane
                 and current_lane_id in (source,lane), 'LANE_CHANGE_LANE_PAIR_INVALID')
        _require(pose is not None and parking is None,'LANE_CHANGE_FIXED_POSE_REQUIRED')
        _require(speed>0 and distance==-1 and not request['precision_stop']
                 and dwell==0 and not request['parking_brake_at_stop'] and obligation is None,
                 'LANE_CHANGE_GOAL_EXTENSIONS_UNSUPPORTED')
        _require(lights['left_signal']!=lights['right_signal'] and not lights['hazard_signal'],
                 'LANE_CHANGE_INDICATOR_REQUIRED')
        _require(request['stage'] in ('PREPARE','WAIT_GAP','REQUEST_PATH','EXECUTE','SETTLE'),
                 'LANE_CHANGE_STAGE_NOT_PLANNABLE')
        return
    _require(lane is None or lane == current_lane_id, "UNSUPPORTED_TARGET_LANE")
    if runtime_stop and request['maneuver'] == 'CONTINUOUS_TASK':
        _require(parking is None and request['stage'] == 'FOLLOW_ROUTE' and distance >= 0
                 and dwell == 0 and obligation is None and request['precision_stop']
                 and not request['parking_brake_at_stop'] and not any(lights.values()),
                 'UNSUPPORTED_RUNTIME_TASK')
        return
    _require(parking is None and pose is None, "UNSUPPORTED_POSE_OR_PARKING_TARGET")
    if runtime_stop:
        blocked = request['maneuver'] == 'BLOCKED_ROAD'
        if blocked:
            _require(request['stage'] == 'STOP_AND_WARN' and obligation is None and dwell == 0
                     and request['precision_stop'] and not request['parking_brake_at_stop']
                     and lights['hazard_signal'] and not lights['left_signal'] and not lights['right_signal'],
                     'UNSUPPORTED_RUNTIME_BLOCKAGE')
            return
        _require(request["maneuver"] in ("SIGNAL_STOP", "RULE_STOP") and request["stage"]=="STOP_AND_DWELL"
                 and obligation is not None and dwell > (0 if request['maneuver'] == 'RULE_STOP' else 5)
                 and request["precision_stop"]
                 and not request["parking_brake_at_stop"] and not any(lights.values()),
                 "UNSUPPORTED_RUNTIME_STOP")
    else:
        _require(obligation is None, "UNSUPPORTED_STOP_OBLIGATION")
        _require(dwell == 0 and not request["parking_brake_at_stop"] and not any(lights.values()),
                 "UNSUPPORTED_STOP_EXTENSIONS")


def assess_behavior_request(request, context, frame_id, now_s, valid_until_s,
                            clock_id, active_identity, capabilities=None,
                            supported_actions=(), current_lane_id="",
                            frame_usable=False, frame_paused=True, runtime_stop=False,
                            runtime_lane_change=False,runtime_parking=False,runtime_driving=False):
    """Validate one proposal for an explicitly enabled forward component trial.

    ``now_s`` is the actual check time in ``clock_id``; the current frame's
    deadline and quality must come from the caller's trusted input. The caller
    must supply that clock's current time, including elapsed processing time;
    this function cannot independently establish the clock or its true time.
    ``active_identity`` has intent_id/stage/revision/source_frame_id/issued_at_s.
    Its source/issued values describe creation of the *revision*, so they may
    precede this frame; produced_frame_id must identify this refreshed request.
    Capabilities uses context/actions/produced_at_s/valid_until_s/clock_id/
    usable/version. The PR20 capability name PATH is required without a stop;
    PATH_STOP is required for a known stop distance. supported_actions is a
    separate local mode opt-in, not an additional capability vocabulary.
    Neither this function nor its defaults advertise production capabilities.
    runtime_lane_change enables only source-bound, fixed-goal P03 requests at
      PREPARE/WAIT_GAP stage ordinary lane paths with lamp commands only;
      REQUEST_PATH/EXECUTE/SETTLE use PATH/LANE_CHANGE/LIGHTS/FEEDBACK. It does
    not validate physical model sources or generate any candidate itself.

    Supports only explicitly opted-in KEEP_LANE/FOLLOW/STOP, current lane,
    forward direction, no pose/parking target or dwell/brake/light extensions.
    STOP/precision/obligation need a known nonnegative remaining stop distance.
    Returns a detached request only on success. Rejections never substitute a
    safe stop and never acknowledge the original maneuver. No feedback is made.
    """
    result = dict(eligible=False, status="REJECTED", reason_code="", retryable=False,
                  request=None, missing_capabilities=[], production_connected=False)
    try:
        _context(context, "CURRENT_CONTEXT_INVALID")
        _require(type(frame_id) is int and frame_id >= 0 and _finite(now_s)
                 and _finite(valid_until_s) and now_s < valid_until_s and _text(clock_id),
                 "CURRENT_FRAME_INVALID")
        _require(type(frame_usable) is bool and frame_usable
                 and type(frame_paused) is bool and not frame_paused, "CURRENT_FRAME_UNUSABLE", True)
        _keys(active_identity, _IDENTITY_KEYS, "ACTIVE_IDENTITY_INVALID")
        _require(_text(active_identity["intent_id"]) and _text(active_identity["stage"])
                 and type(active_identity["revision"]) is int and active_identity["revision"] >= 1
                 and type(active_identity["source_frame_id"]) is int
                 and 0 <= active_identity["source_frame_id"] <= frame_id
                 and _finite(active_identity["issued_at_s"])
                 and active_identity["issued_at_s"] <= now_s, "ACTIVE_IDENTITY_INVALID")
        _require(isinstance(request,dict) and _REQUEST_KEYS<=set(request)
                 <=_REQUEST_KEYS|{'source_lane_id','candidate_id'},'REQUEST_SCHEMA')
        _require(request.get('candidate_id') is None or _text(request['candidate_id']),
                 'REQUEST_CANDIDATE_ID_INVALID')
        _require(all(type(v) is bool for v in (runtime_stop,runtime_lane_change,runtime_parking,runtime_driving))
                 and sum((runtime_stop,runtime_lane_change,runtime_parking,runtime_driving))<=1
                 and request["contract_version"] == CONTRACT_VERSION
                 and request["interface_status"] == ("runtime_connected" if runtime_stop or runtime_lane_change or runtime_parking or runtime_driving
                                                       else "proposal_not_runtime_connected"), "REQUEST_VERSION")
        _context(request["task_context"], "REQUEST_CONTEXT_INVALID")
        _require(request["task_context"] == context, "REQUEST_CONTEXT_MISMATCH")
        _require(type(request["revision"]) is int and request["revision"] >= 1, "REQUEST_REVISION_INVALID")
        _require(request["revision"] == active_identity["revision"], "REQUEST_REVISION_MISMATCH")
        _require(all(request[key] == active_identity[key] for key in ("intent_id", "stage"))
                 and _text(request["maneuver"]), "REQUEST_IDENTITY_MISMATCH")
        _require(type(request["source_frame_id"]) is int and type(request["produced_frame_id"]) is int
                 and 0 <= request["source_frame_id"] <= request["produced_frame_id"] == frame_id
                 and request["source_frame_id"] == active_identity["source_frame_id"], "REQUEST_FRAME")
        _require(request["clock_id"] == clock_id
                 and all(_finite(request[key]) for key in ("issued_at_s", "produced_at_s", "valid_until_s"))
                 and request["issued_at_s"] == active_identity["issued_at_s"]
                 and request["issued_at_s"] <= request["produced_at_s"] <= now_s
                 and now_s < request["valid_until_s"] <= valid_until_s, "REQUEST_TIME", True)
        _require(type(request["dispatch_allowed"]) is bool and request["dispatch_allowed"]
                 and request["status"] in ("REQUESTED", "ACCEPTED", "EXECUTING"), "REQUEST_NOT_DISPATCHABLE")
        _require(isinstance(request["reason_code"], str) and type(request["attempt"]) is int
                 and request["attempt"] >= 0, "REQUEST_VALUES")
        enabled = _strings(supported_actions, "LOCAL_ACTIONS_INVALID")
        implemented = (DRIVING_ACTIONS if runtime_driving else ('PARK',) if runtime_parking else LANE_CHANGE_ACTIONS if runtime_lane_change
                       else ("SIGNAL_STOP", "RULE_STOP", "BLOCKED_ROAD", "CONTINUOUS_TASK") if runtime_stop else COMPONENT_ACTIONS)
        _require(enabled <= set(implemented), "LOCAL_ACTIONS_UNSUPPORTED")
        action = request["maneuver"]
        _require(action in implemented, "UNSUPPORTED_MANEUVER")
        _require(action in enabled, "LOCAL_ACTION_DISABLED")
        _goal(request, current_lane_id, runtime_stop,runtime_lane_change,runtime_parking,runtime_driving)
        _require(capabilities is not None, "CAPABILITIES_UNAVAILABLE", True)
        _keys(capabilities, _CAPABILITY_KEYS, "CAPABILITY_SCHEMA")
        _context(capabilities["context"], "CAPABILITY_CONTEXT_INVALID")
        _require(capabilities["context"] == context and capabilities["version"] == CONTRACT_VERSION
                 and capabilities["clock_id"] == clock_id, "CAPABILITY_IDENTITY")
        _require(type(capabilities["usable"]) is bool and capabilities["usable"]
                 and _finite(capabilities["produced_at_s"]) and _finite(capabilities["valid_until_s"])
                 and capabilities["produced_at_s"] <= now_s < capabilities["valid_until_s"],
                 "CAPABILITY_NOT_CURRENT", True)
        actions = _strings(capabilities["actions"], "CAPABILITY_ACTIONS_INVALID")
        required = "PATH_STOP" if request["stop_distance_m"] >= 0 else "PATH"
        needs = (('EMERGENCY_CANDIDATES','PATH','FEEDBACK') if runtime_driving and action=='EMERGENCY' else
                 ('INTERSECTION',required,'FEEDBACK') if runtime_driving else
                 ('PARK','PATH','FEEDBACK')+ (('REVERSE','DWELL') if request['stage'] in
                 ('REVERSE_ENTRY','ALIGN','PARKED_DWELL','EXIT_PREPARE') else ()) if runtime_parking else
                 ('LANE_CHANGE','PATH','LIGHTS','FEEDBACK') if runtime_lane_change else
                 (required,'TASK_ROUTE','FEEDBACK') if runtime_stop and action=='CONTINUOUS_TASK' else
                 (required,'LIGHTS','FEEDBACK') if runtime_stop and action=='BLOCKED_ROAD' else
                 (required,'DWELL','FEEDBACK') if runtime_stop else (required,))
        missing = sorted(set(needs) - actions)
        _require(not missing, "CAPABILITY_ACTIONS_MISSING", True, missing)
        result.update(eligible=True, status="READY_FOR_COMPONENT", reason_code="CONTRACT_CHECKED",
                      request=copy.deepcopy(request),production_connected=runtime_stop or runtime_lane_change or runtime_parking or runtime_driving)
    except _Rejected as error:
        result.update(reason_code=error.reason_code, retryable=error.retryable,
                      missing_capabilities=list(error.missing))
    return result
