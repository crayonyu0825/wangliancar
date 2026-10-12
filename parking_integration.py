"""Captain-owned paired parking inputs and independent current-path monitor.

Providers assert an explicit task/model; this adapter creates no calibration,
mission, visibility or permission. Default runtime construction installs none.
"""
import copy
import math
import time

from core.behavior_channel import read_channel,capabilities_dict
from core.behavior_contract import BehaviorFrame,require,finite
from core.geometry import normalize_angle,opposes_direction
from core.interfaces import DecisionTarget,Trajectory,ControlOut
from core.safety_supervisor import SafetyAssessment
from core.validation import validate_output
from core.maneuver_facts import ManeuverFacts
from core.execution_monitor import ExecutionMonitor
from members.decision.behaviors.parking_runtime import ParkingDecisionInputs
from members.planning.parking_behavior import ParkingPlanningInputs
from members.planning.maneuver_inputs import read_inputs
from members.planning.behavior_contract import assess_behavior_request
from members.planning.candidate_validation import ValidationBudget
from members.planning.parking_scene import validate_parking_path


PARKING_ACTIONS=('PARK','PATH','REVERSE','DWELL','FEEDBACK')


def _planning_copy(value):
    result=copy.copy(value)
    for name in ('context','mission','tolerances','vehicle','limits','search','budget','prediction_envelopes',
                 'execution_envelope','dynamic_search','speed_error_mps'):
        setattr(result,name,copy.deepcopy(getattr(value,name)))
    return result


def _mission_key(value):
    m=value.snapshot(); e=m.evidence
    return (m.mission_id,m.map_digest,m.space_id,m.road_lane_ids,m.access_edge_index,m.body_heading_rad,
            tuple((v.x,v.y,v.body_heading_rad) for v in (m.approach_pose,m.position_pose,m.exit_goal)),
            e.context.key(),e.frame_id,e.observed_at_s,e.valid_until_s,e.usable,e.coverage_verified,
            e.source,e.clock_id,e.source_kind)


class ParkingIntegrationInputs(object):
    """One original per-frame pair and a separately supplied monitor budget."""
    def __init__(self,decision,planning,monitor_budget):
        require(isinstance(decision,ParkingDecisionInputs) and isinstance(planning,ParkingPlanningInputs)
                and isinstance(monitor_budget,ValidationBudget),'PARKING_INPUT_PAIR_UNAVAILABLE')
        self.decision,self.planning=decision.snapshot(),_planning_copy(planning)
        self.monitor_budget=copy.deepcopy(monitor_budget)

    def snapshot(self):
        return ParkingIntegrationInputs(self.decision,self.planning,self.monitor_budget)


class ParkingInputBridge(object):
    def __init__(self,provider,clock=None):
        require(callable(provider),'PARKING_PAIR_PROVIDER_INVALID')
        self.provider,self.clock=provider,clock or time.monotonic
        self.settings=None
        self.key,self.value,self.error=None,None,None
        self.reason='NOT_PREPARED'
        self.execution_monitor=ExecutionMonitor()
        self.execution_tracking=None
        self._execution_key=None
        self._execution_model=None

    @staticmethod
    def _key(p,context):
        return (id(p),context.key(),p.frame_id,p.timestamp)

    def _observe_execution(self,p,t,model):
        self._execution_model=model
        if model.execution_envelope is None and not self.execution_monitor.history:
            self.execution_tracking=dict(active=False,status='NOMINAL_ONLY',reason_code='EXECUTION_ENVELOPE_NOT_CONFIGURED')
            return self.execution_tracking
        self.execution_monitor.max_speed_mps=model.limits.max_speed_mps
        self.execution_monitor.speed_error_mps=model.speed_error_mps
        gps=p.source_status.get('gps',{}) if isinstance(p.source_status,dict) else {}
        environment=getattr(p,'maneuver_environment',None)
        actual=(p.case_id,p.task_id,p.scene_id,p.frame_id,p.timestamp,p.ego.x,p.ego.y,p.ego.heading,
                p.ego.speed,p.ego.vx,p.ego.vy,p.valid,p.valid_until,p.ego.valid,p.ego.frame_id,p.ego.timestamp,
                tuple((key,copy.deepcopy(gps.get(key))) for key in
                      ('usable','quality','frame_id','timestamp','observed_at_s')),
                tuple(getattr(environment,key,None) for key in ('valid','frame_id','timestamp','observed_at_s','valid_until_s')),
                None if model.execution_envelope is None else tuple(sorted(vars(model.execution_envelope).items())),
                model.limits.max_speed_mps,model.speed_error_mps)
        if actual!=self._execution_key:
            self.execution_tracking=self.execution_monitor.observe(p,t,model.execution_envelope,self.clock())
            self._execution_key=actual
        return self.execution_tracking

    def assess_execution(self,p,trajectory):
        """Observe sent history every frame, including the stage after PARK.

        Retaining model values only checks the prior sent occupancy bounds;
        it does not create fresh parking task/model evidence for a new send.
        """
        try:
            context,unused,unused=read_channel(p,self.clock())
            pair=self._read(p,context)
            model=pair.planning if pair is not None else self._execution_model
            if model is None:
                return SafetyAssessment()
            report=self._observe_execution(p,trajectory,model)
            return SafetyAssessment('controlled_stop','PARKING_EXECUTION:'+report['reason_code']) if report['active'] else SafetyAssessment()
        except (ValueError,TypeError,AttributeError,KeyError) as error:
            return SafetyAssessment('controlled_stop','PARKING_EXECUTION_INPUT:'+str(error)) if self.execution_monitor.history else SafetyAssessment()

    def record_sent(self,p,trajectory,receipt,safety_active=False):
        """Only the captain's successful inactive-safety send arms a timeline."""
        if safety_active or not isinstance(receipt,dict) or receipt.get('attempted') is not True or receipt.get('ok') is not True:
            return dict(active=False,status='NOT_ARMED',reason_code='PARKING_SEND_NOT_CONFIRMED_OR_SAFETY_ACTIVE')
        try:
            context,unused,unused=read_channel(p,self.clock()); pair=self._read(p,context)
            require(pair is not None,'PARKING_PAIR_WITHDRAWN')
            model=pair.planning
            if model.execution_envelope is None:
                require(not self.execution_monitor.history,'PARKING_EXECUTION_BOUNDS_WITHDRAWN')
                return dict(active=False,status='NOMINAL_ONLY',reason_code='EXECUTION_ENVELOPE_NOT_CONFIGURED')
            self.execution_monitor.max_speed_mps=model.limits.max_speed_mps
            self.execution_monitor.speed_error_mps=model.speed_error_mps
            return self.execution_monitor.record_sent(p,trajectory,model.execution_envelope,self.clock(),receipt)
        except (ValueError,TypeError,AttributeError,KeyError) as error:
            return dict(active=True,status='BLOCKED',reason_code='PARKING_RECORD_SENT:'+str(error))

    def prepare(self,p):
        """Read once per original frame; repeated GPS never refreshes evidence."""
        try:
            context,unused_caps,unused_replies=read_channel(p,self.clock())
            key=self._key(p,context)
            if key==self.key: return
            self.key,self.value,self.error=key,None,None
            pair=self.provider(p,context)
            if pair is None:
                self.reason='NO_PARKING_TASK'; return
            require(isinstance(pair,ParkingIntegrationInputs),'PARKING_INPUT_PAIR_UNAVAILABLE')
            pair=pair.snapshot()
            self._validate(pair,p,context)
            self.value,self.reason=pair,'CURRENT_PAIRED_PARKING_INPUTS'
        except Exception as error:
            self.value,self.error=None,str(error)
            self.reason='PARKING_PAIR:'+str(error)

    def _validate(self,pair,p,context):
        d=pair.decision.snapshot(); m=read_inputs(lambda *args:pair.planning,p,context,
                                                self.clock,'PARKING_PAIR',ParkingPlanningInputs)
        now=self.clock(); frame=BehaviorFrame(context,p.frame_id,now,p.valid_until,
            p.ego.x,p.ego.y,p.ego.heading,p.ego.speed,usable=p.valid is True)
        e=d.model_evidence
        require(e.frame_id==p.frame_id and e.observed_at_s==m.produced_at_s
                and e.valid_until_s==m.valid_until_s
                and e.verified(frame,allowed_source_kinds=('task','verified_fusion')),
                'PARKING_PAIR_MODEL_EVIDENCE_MISMATCH')
        require(d.model_id==m.model_id and _mission_key(d.mission)==_mission_key(m.mission)
                and d.mission.evidence.verified(frame,allowed_source_kinds=('task','verified_fusion')),
                'PARKING_PAIR_TASK_OR_MODEL_MISMATCH')
        extents=tuple(getattr(m.vehicle,k) for k in ('front_offset_m','rear_offset_m','half_width_m'))
        require(d.extents==extents and d.tolerances==m.tolerances
                and finite(m.limits.max_speed_mps) and d.speed_cap<=m.limits.max_speed_mps,
                'PARKING_PAIR_GEOMETRY_OR_LIMIT_MISMATCH')
        require(self.settings is not None,'PARKING_RUNTIME_SETTINGS_UNAVAILABLE')
        for name,actual in tuple(zip(('front_offset_m','rear_offset_m','half_width_m'),extents))+(
                ('wheelbase_m',m.vehicle.wheelbase_m),('front_steer_max_rad',m.limits.max_front_steer_rad)):
            configured=getattr(self.settings,name,None)
            require(finite(configured) and finite(actual) and abs(configured-actual)<=1e-9,
                    'PARKING_RUNTIME_CONTROL_GEOMETRY_MISMATCH')
        b=pair.monitor_budget
        require(type(b.max_checks) is int and b.max_checks>0 and type(b.max_depth) is int
                and 0<=b.max_depth<=40 and finite(b.min_interval_s) and b.min_interval_s>0
                and finite(b.deadline_monotonic_s) and now<b.deadline_monotonic_s,
                'PARKING_MONITOR_BUDGET_UNAVAILABLE')
        require(self.clock()>=now and self.clock()<min(e.valid_until_s,d.mission.evidence.valid_until_s),
                'PARKING_PAIR_EXPIRED_DURING_READ')

    def _read(self,p,context):
        require(self.key==self._key(p,context),'PARKING_PAIR_NOT_PREPARED_FOR_FRAME')
        require(self.error is None,self.reason)
        if self.value is None: return None
        pair=self.value.snapshot(); self._validate(pair,p,context)
        return pair

    def decision_inputs(self,p,context):
        pair=self._read(p,context)
        return pair.decision if pair is not None else None

    def planning_inputs(self,p,context):
        pair=self._read(p,context)
        require(pair is not None,'PARKING_PAIR_WITHDRAWN')
        return pair.planning

    @staticmethod
    def _control(control,trajectory):
        require(control is not None,'PARKING_MONITOR_CONTROL_UNAVAILABLE')
        validated=validate_output(copy.copy(control),ControlOut,trajectory)
        require(all(getattr(validated,k)==getattr(control,k) for k in ('throttle','brake','steering')),
                'PARKING_MONITOR_CONTROL_NOT_NORMALIZED')

    def assess(self,p,decision,trajectory,control):
        """Recheck original task, actual direction and full nominal path now.

        The public P02 checker is reused with a separate budget and freshly
        read sources. This is an additional veto, never execution feedback.
        """
        mode='controlled_stop'
        actual=(p.case_id,p.task_id,p.scene_id,p.frame_id,p.timestamp,p.ego.x,p.ego.y,p.ego.heading,
                p.ego.speed,p.ego.vx,p.ego.vy)
        try:
            validate_output(decision,DecisionTarget,p)
            validate_output(trajectory,Trajectory,decision)
            self._control(control,trajectory)
            now=self.clock(); context,caps,unused_replies=read_channel(p,now)
            pair=self._read(p,context); require(pair is not None,'PARKING_PAIR_WITHDRAWN')
            assessment=assess_behavior_request(decision.behavior_request,context.to_dict(),p.frame_id,
                now,min(p.valid_until,decision.valid_until),'process_monotonic',
                decision.behavior_active_identity,capabilities=capabilities_dict(caps),
                supported_actions=('PARK',),current_lane_id=p.lane.lane_id,
                frame_usable=p.valid,frame_paused=False,runtime_parking=True)
            require(assessment['eligible'],assessment['reason_code'])
            r=assessment['request']; d,m=pair.decision,pair.planning
            tracking=self.assess_execution(p,trajectory)
            require(not tracking.active,tracking.reason)
            facts=ManeuverFacts(p,self.clock); bay=facts.parking(m.mission.space_id)
            original=m.mission.stage_pose(r['stage'],bay,*d.extents[:2]).to_dict()
            segmented=r['stage']=='ALIGN' and isinstance(r.get('candidate_id'),str) and bool(r['candidate_id'])
            require(facts.map_digest()==m.mission.map_digest and r['parking_space_id']==m.mission.space_id
                    and (r['goal_pose']==original or segmented),
                    'PARKING_MONITOR_ORIGINAL_TASK_MISMATCH')
            if segmented:
                require(trajectory.candidate_id==r['candidate_id']
                    and isinstance(trajectory.behavior_feedback,dict)
                    and trajectory.behavior_feedback.get('candidate_id')==r['candidate_id']
                    and trajectory.planning_metrics.get('parking_segment') is True
                    and trajectory.planning_metrics.get('original_parking_goal')==original,
                    'PARKING_MONITOR_SEGMENT_IDENTITY_MISMATCH')
            require(trajectory.motion_direction==r['motion_direction']
                    and all(finite(v) for v in (p.ego.vx,p.ego.vy,p.ego.heading,p.ego.speed))
                    and not opposes_direction(p.ego.vx,p.ego.vy,p.ego.heading,p.ego.speed,
                                              r['motion_direction']),
                    'PARKING_MONITOR_ACTUAL_DIRECTION_MISMATCH')
            first=trajectory.points[0]
            stopped_hold=(trajectory.stop_required and trajectory.stop_distance==0
                and p.ego.speed<=d.tolerances[2] and math.hypot(p.ego.vx,p.ego.vy)<=d.tolerances[2]
                and control.throttle==0 and control.brake>0
                and all(v.speed==0 and v.x==first.x and v.y==first.y and v.heading==first.heading
                        for v in trajectory.points))
            require(first.relative_time==0 and abs(first.x-p.ego.x)<=1e-9 and abs(first.y-p.ego.y)<=1e-9
                    and abs(normalize_angle(first.heading-p.ego.heading))<=1e-9
                    and (stopped_hold or abs(first.speed-p.ego.speed)<=1e-9),
                    'PARKING_MONITOR_ACTUAL_START_MISMATCH')
            identity={k:r[k] for k in ('intent_id','stage','revision')}
            proposal=trajectory.planning_metrics.get('parking_segment_proposal') is True
            preparing=trajectory.planning_metrics.get('parking_stopped_steering_preparation') is True or proposal
            if preparing:
                require(trajectory.behavior_identity is None and stopped_hold,'PARKING_MONITOR_PREPARATION_MOVES')
                if proposal:
                    require(isinstance(trajectory.behavior_feedback,dict)
                        and trajectory.behavior_feedback.get('status')=='REJECTED'
                        and trajectory.behavior_feedback.get('candidate_options'),
                        'PARKING_MONITOR_UNBOUND_SEGMENT_PROPOSAL')
            else:
                require(trajectory.behavior_identity==identity,'PARKING_MONITOR_IDENTITY_MISMATCH')
                goal=r['goal_pose']; end=trajectory.points[-1]
                require(math.hypot(end.x-goal['x'],end.y-goal['y'])<=d.tolerances[0]
                        and abs(normalize_angle(end.heading-goal['body_heading_rad']))<=d.tolerances[1]
                        and end.speed==0,'PARKING_MONITOR_ORIGINAL_ARRIVAL_REGION_MISMATCH')
            cap=min(d.speed_cap,r['speed_cap_mps'],decision.target_speed)
            require(all(v.speed<=max(p.ego.speed,cap)+1e-9 for v in trajectory.points),
                    'PARKING_MONITOR_SPEED_LIMIT_MISMATCH')
            report=validate_parking_path(p,m.mission,r['stage'],trajectory.points,trajectory.motion_direction,
                m.vehicle,m.limits,pair.monitor_budget,m.prediction_envelopes,self.clock,
                intermediate_segment=segmented and r['goal_pose']!=original,
                execution_envelope=m.execution_envelope)
            if report['status']=='unsafe': mode='emergency_stop'
            require(report['status']=='safe','PARKING_MONITOR_PATH:'+report['reason_code'])
            self._read(p,context)
            self._control(control,trajectory)
            require(actual==(p.case_id,p.task_id,p.scene_id,p.frame_id,p.timestamp,p.ego.x,p.ego.y,p.ego.heading,
                p.ego.speed,p.ego.vx,p.ego.vy),'PARKING_MONITOR_ACTUAL_STATE_CHANGED')
            require(self.clock()>=now and self.clock()<report['source_valid_until_s'],
                    'PARKING_MONITOR_SOURCE_EXPIRED_DURING_CHECK')
            self.reason='CURRENT_PARKING_PATH_RECHECKED'
            return SafetyAssessment()
        except Exception as error:
            self.reason='PARKING_MONITOR:'+str(error)
            return SafetyAssessment(mode,self.reason)
