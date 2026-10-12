"""Captain-owned transport and previous-frame feedback; no member internals."""
import copy
import math
import time
import uuid

from core.behavior_contract import (CONTRACT_VERSION, TaskContext, Capabilities,
                                    ExecutionFeedback, GoalPose, finite, require)
from core.maneuver_progress import ProgressTracker

CHANNEL_VERSION = "behavior-channel-v1"


def context_from_dict(value):
    require(isinstance(value,dict) and set(value)==set(TaskContext.__slots__), "invalid channel context")
    return TaskContext(**value)


def capabilities_dict(value):
    result = {key:getattr(value,key) for key in Capabilities.__slots__}
    result["context"],result["actions"] = value.context.to_dict(),sorted(value.actions)
    return result


def feedback_dict(value):
    result = {key:getattr(value,key) for key in ExecutionFeedback.__slots__}
    result["context"] = value.context.to_dict()
    result["actual_pose"] = None if value.actual_pose is None else value.actual_pose.to_dict()
    return result


def feedback_from_dict(value):
    optional = {'lights_command_duration_s', 'lights_evidence_level', 'candidate_options',
                'planning_metrics', 'evidence_level', 'command_send_status', 'gear_confirmed',
                'actual_brake_stop_confirmed'}
    require(isinstance(value,dict) and set(ExecutionFeedback.__slots__) - optional <= set(value)
            and set(value) <= set(ExecutionFeedback.__slots__), "invalid channel feedback")
    data = copy.deepcopy(value)
    data["context"] = context_from_dict(data["context"])
    pose = data["actual_pose"]
    if pose is not None:
        require(isinstance(pose,dict) and set(pose)=={"x","y","body_heading_rad","reference_point"}
                and pose["reference_point"]=="ego_rear_axle", "invalid actual reference")
        data["actual_pose"] = GoalPose(pose["x"],pose["y"],pose["body_heading_rad"])
    return ExecutionFeedback(**data)


def read_channel(perception, now_s):
    """Validate current transport before the existing session checks feedback."""
    channel = perception.behavior_channel
    require(isinstance(channel,dict) and set(channel)=={
        "contract_version","context","frame_id","produced_at_s","valid_until_s","capabilities","feedbacks"},
        "behavior channel unavailable")
    require(channel["contract_version"]==CHANNEL_VERSION and channel["frame_id"]==perception.frame_id
            and finite(channel["produced_at_s"]) and finite(channel["valid_until_s"])
            and channel["produced_at_s"]<=now_s<channel["valid_until_s"]<=perception.valid_until,
            "behavior channel expired or mismatched")
    context = context_from_dict(channel["context"])
    require(context.key()[:3]==(perception.case_id,perception.task_id,perception.scene_id), "behavior context mismatch")
    cap = copy.deepcopy(channel["capabilities"])
    require(isinstance(cap,dict) and set(cap)==set(Capabilities.__slots__)
            and cap["context"]==channel["context"] and isinstance(cap["actions"],list), "invalid channel capability")
    cap["context"] = context
    capabilities = Capabilities(**cap)
    require(isinstance(channel["feedbacks"],list) and len(channel["feedbacks"])<=4, "invalid channel feedback list")
    feedbacks = [feedback_from_dict(v) for v in channel["feedbacks"]]
    return context,capabilities,feedbacks


class BehaviorTransport(object):
    """One-process handoff. Capabilities reflect implemented consumers only.

    Planning acceptance and observed execution are separate records. Reusing
    a previous frame never refreshes its observation/deadline. A send receipt
    can invalidate execution, but can never mark arrival or completion.
    """
    def __init__(self, actions=(), clock=None):
        self.actions,self.clock = tuple(actions),clock or time.monotonic
        self.scope,self.context,self.frame,self.feedbacks = None,None,None,[]
        self._progress = {}
        self._motion_progress = {}
        self._path_progress = ProgressTracker()
        self._path_bounds,self._path_goals = {},{}

    def reset_session(self):
        """Captain-only reset after establishing a new SDK case/session.

        A member request or a backwards GPS frame cannot authorize this reset.
        """
        self.scope,self.context,self.frame = None,None,None
        self.feedbacks,self._progress,self._motion_progress = [],{},{}
        self._path_progress.reset()
        self._path_bounds,self._path_goals = {},{}

    def prepare(self,p):
        now = self.clock()
        gps = p.source_status.get("gps",{}) if isinstance(p.source_status,dict) else {}
        if (p.valid is not True or p.ego.valid is not True or type(p.frame_id) is not int or p.frame_id<0
                or not isinstance(gps,dict) or gps.get("usable") is not True
                or not finite(p.valid_until) or now>=p.valid_until):
            # An untrusted new task label must not discard an accepted duty.
            p.behavior_channel = {}
            self.feedbacks = []
            return
        scope = (p.case_id,p.task_id,p.scene_id)
        if self.context is not None and (scope!=self.scope or
                self.frame is not None and p.frame_id<self.frame):
            p.behavior_channel = {}
            return
        if self.context is None:
            self.scope,self.context = scope,TaskContext(*(scope+("runtime:"+uuid.uuid4().hex,)))
            self.feedbacks,self._progress = [],{}
            self._motion_progress = {}
            self._path_progress.reset()
            self._path_bounds,self._path_goals = {},{}
        self.frame = p.frame_id
        caps = Capabilities(self.context,self.actions,now,p.valid_until,usable=bool(self.actions))
        p.behavior_channel = dict(contract_version=CHANNEL_VERSION,context=self.context.to_dict(),
            frame_id=p.frame_id,produced_at_s=now,valid_until_s=p.valid_until,
            capabilities=capabilities_dict(caps),feedbacks=copy.deepcopy(self.feedbacks))

    def observe(self,p,decision,trajectory,control,safety_active=False,receipt=None):
        now = self.clock()
        records = []
        request = decision.behavior_request
        if not isinstance(request,dict) or self.context is None:
            self.feedbacks = []
            return
        if request.get("task_context")!=self.context.to_dict() or request.get("produced_frame_id")!=p.frame_id:
            self.feedbacks = []
            return
        if (not all(isinstance(request.get(k),str) and request[k] for k in ("intent_id","stage"))
                or type(request.get("revision")) is not int or request["revision"]<1
                or not finite(request.get("stop_distance_m")) or not finite(request.get("valid_until_s"))):
            self.feedbacks = []
            return
        if isinstance(trajectory.behavior_feedback,dict):
            records.append(copy.deepcopy(trajectory.behavior_feedback))
        identity = {key:request[key] for key in ("intent_id","stage","revision")}
        usable = bool(p.valid and control.valid and trajectory.valid and
                      now<min(p.valid_until,control.valid_until,trajectory.valid_until,request['valid_until_s']))
        if usable:
            observation = control.execution_observation
            actual = isinstance(observation,dict) and observation.get("identity")==identity
            metrics=getattr(trajectory,'planning_metrics',{})
            bounded=isinstance(metrics,dict) and 'progress_bounds' in metrics
            gps=p.source_status.get('gps',{}) if isinstance(p.source_status,dict) else {}
            current_gps=bool(p.ego.valid is True and p.ego.frame_id==p.frame_id and p.ego.timestamp==p.timestamp
                and isinstance(gps,dict) and gps.get('usable') is True
                and gps.get('quality') not in ('stale','invalid','regressed','unsynchronized','timing_unknown'))
            gps_actual=bool(bounded and trajectory.behavior_identity==identity and current_gps)
            failed = safety_active or (isinstance(receipt,dict) and receipt.get("attempted") is True
                                      and receipt.get("ok") is not True)
            if actual or gps_actual or failed:
                key = (request["intent_id"],request["revision"])
                initial = self._progress.setdefault(key,max(0.,request["stop_distance_m"]))
                progress = max(0.,min(1.,1.-max(0.,request["stop_distance_m"])/initial)) if initial>0 else 0.
                evidence='legacy_gps_controller_observed'
                progress_metrics={}
                progress_reason='GPS_CONTROLLER_OBSERVED'
                progress_failed=False
                moving_path=bool(isinstance(trajectory.points,(list,tuple)) and len(trajectory.points)>1
                    and any(math.hypot(b.x-a.x,b.y-a.y)>1e-9 for a,b in zip(trajectory.points,trajectory.points[1:])))
                if bounded and not gps_actual:
                    failed=True; progress=0.; progress_reason='PROGRESS_GPS_UNAVAILABLE'
                    progress_failed=True
                    evidence='gps_path_progress_rejected'
                elif bounded and (actual or gps_actual) and moving_path:
                    try:
                        identifier=getattr(trajectory,'candidate_id',None)
                        bounds=metrics['progress_bounds']
                        require(isinstance(identifier,str) and bool(identifier) and isinstance(bounds,dict)
                                and set(bounds)=={'max_speed_mps','max_acceleration_mps2','position_tolerance_m'}
                                and all(finite(v) and v>0 for v in bounds.values()),'PROGRESS_BOUNDS_REQUIRED')
                        expected=request.get('candidate_id')
                        require(expected is None or expected==identifier,'PROGRESS_CANDIDATE_ID_MISMATCH')
                        # Freeze the original path and physical bounds. Later
                        # replans cannot shorten its length or enlarge its
                        # reachable prefix to manufacture task completion.
                        new=identifier not in self._path_bounds
                        require(new or self._path_goals[identifier]==request.get('goal_pose'),'PROGRESS_ORIGINAL_GOAL_CHANGED')
                        frozen=copy.deepcopy(bounds) if new else self._path_bounds[identifier]
                        require(finite(p.ego.speed) and 0<=p.ego.speed<=frozen['max_speed_mps'],
                                'PROGRESS_ACTUAL_SPEED_OUTSIDE_LIMITS')
                        environment=getattr(p,'maneuver_environment',None)
                        observed=(environment.observed_at_s if getattr(environment,'valid',False) is True
                                  else gps.get('observed_at_s',now))
                        require(finite(observed) and 0<=observed<=now,'PROGRESS_FUTURE_OBSERVATION')
                        tracked=self._path_progress.observe(identifier,trajectory.points,
                            (p.ego.x,p.ego.y,p.ego.heading),p.ego.speed,p.frame_id,observed,**frozen)
                        if new:
                            self._path_bounds[identifier]=frozen
                            self._path_goals[identifier]=copy.deepcopy(request.get('goal_pose'))
                        progress=tracked['progress']
                        progress_metrics={'actual_path_progress':tracked,'progress_bounds':copy.deepcopy(frozen)}
                        evidence='gps_reachable_path_progress'; progress_reason=tracked['reason_code']
                        active=set(self._path_progress.entries)
                        for old in list(self._path_bounds):
                            if old not in active:
                                self._path_bounds.pop(old,None); self._path_goals.pop(old,None)
                    except (ValueError,TypeError,AttributeError,KeyError) as error:
                        failed=True; progress=0.; progress_reason=str(error)
                        progress_failed=True
                        evidence='gps_path_progress_rejected'
                elif bounded and not moving_path:
                    progress=0.; evidence='gps_controller_observed_stationary'
                elif (actual and ((request.get('maneuver') in ('LANE_CHANGE','AVOID','OVERTAKE','MERGE')
                        and request['stage'] in ('EXECUTE','SETTLE')) or
                        (request.get('maneuver')=='PARK' and request['stage'] in
                         ('APPROACH','POSITION','REVERSE_ENTRY','ALIGN','PARKED_DWELL','EXIT_PREPARE','EXIT')))):
                    pose=request.get('goal_pose')
                    if (isinstance(pose,dict) and pose.get('reference_point')=='ego_rear_axle'
                            and all(finite(pose.get(k)) for k in ('x','y','body_heading_rad'))):
                        target=tuple(pose[k] for k in ('x','y','body_heading_rad'))
                        remaining=math.hypot(p.ego.x-target[0],p.ego.y-target[1])
                        previous=self._motion_progress.setdefault(key,(target,remaining,0.))
                        if previous[0]!=target:
                            failed=True
                        else:
                            # Actual GPS distance reduction, not path time or
                            # planner acceptance. Even progress=1 cannot mark
                            # arrival, settled lane, lights or completed dwell.
                            progress=max(previous[2],max(0.,min(1.,1.-remaining/previous[1]))) if previous[1]>0 else 0.
                            self._motion_progress[key]=(target,previous[1],progress)
                        if len(self._motion_progress)>128:
                            self._motion_progress={key:self._motion_progress[key]}
                arrived = bool(actual and observation.get("goal_pose_arrived") is True)
                elapsed = observation.get("standstill_duration_s",0.) if actual else 0.
                completed = bool(actual and observation.get("hold_completed") is True)
                identifier=getattr(trajectory,'candidate_id',None)
                brake_stop=bool(not failed and actual and current_gps and
                    request.get('maneuver')=='EMERGENCY' and request['stage'] in
                    ('EXECUTE_BRAKE','EXECUTE_LEFT','EXECUTE_RIGHT','EXECUTE_BRAKE_LEFT','EXECUTE_BRAKE_RIGHT','BRAKE_FALLBACK')
                    and trajectory.behavior_identity==identity and isinstance(identifier,str) and identifier
                    and request.get('candidate_id')==identifier==observation.get('candidate_id')
                    and isinstance(metrics,dict) and metrics.get('emergency_outcome')=='verified_safe_brake_stop'
                    and metrics.get('original_goal_pose')==request.get('goal_pose')
                    and observation.get('actual_brake_stop_confirmed') is True
                    and finite(p.ego.speed) and 0<=p.ego.speed<=1e-9
                    and trajectory.stop_required is True and trajectory.target_speed==0
                    and not trajectory.emergency_stop and isinstance(trajectory.points,(tuple,list))
                    and 2<=len(trajectory.points)<=2000 and all(v.speed==0
                        and math.hypot(v.x-p.ego.x,v.y-p.ego.y)<=1e-9
                        and abs(math.atan2(math.sin(v.heading-p.ego.heading),math.cos(v.heading-p.ego.heading)))<=1e-9
                        for v in trajectory.points)
                    and control.throttle==0 and finite(control.brake) and control.brake>0 and control.hazard_signal is True
                    and isinstance(receipt,dict) and receipt.get('attempted') is True and receipt.get('ok') is True)
                if brake_stop:
                    # Safe early braking is a measured outcome of this command,
                    # never arrival at the unchanged original path endpoint.
                    arrived=completed=False;elapsed=0.;progress=0.
                    evidence='verified_actual_brake_stop'
                    progress_reason='safe_brake_stop_observed_not_goal'
                    progress_metrics=dict(emergency_outcome='verified_safe_brake_stop',
                        actual_stop_pose=[p.ego.x,p.ego.y,p.ego.heading],original_goal_pose=copy.deepcopy(request.get('goal_pose')))
                status = ("FAILED" if failed else "COMPLETED" if completed else "DWELLING" if arrived else "EXECUTING")
                reply = ExecutionFeedback(self.context,request["intent_id"],request["stage"],request["revision"],
                    "runtime",p.frame_id,p.frame_id,now,min(p.valid_until,control.valid_until),status,
                    usable=True,reason_code=progress_reason if progress_failed else
                        "EXECUTION_INTERRUPTED" if failed else progress_reason,
                    progress=progress,actual_standstill_confirmed=bool((arrived and p.ego.speed<=.05 or brake_stop) and not failed),
                    actual_brake_stop_confirmed=brake_stop,
                    goal_pose_arrived=arrived and not failed,standstill_duration_s=elapsed if not failed else 0.,
                    hold_completed=completed and not failed,motion_direction=trajectory.motion_direction,
                    actual_pose=GoalPose(p.ego.x,p.ego.y,p.ego.heading),actual_lane_id=p.lane.lane_id,
                    candidate_id=getattr(trajectory,'candidate_id',None),
                    command_send_status=('unknown' if not isinstance(receipt,dict) else
                        'not_attempted' if receipt.get('attempted') is not True else
                        'sent' if receipt.get('ok') is True else 'failed'),
                    evidence_level=evidence,planning_metrics=progress_metrics)
                records.append(feedback_dict(reply))
                if len(self._progress)>128:
                    self._progress = {key:initial}
        self.feedbacks = records
