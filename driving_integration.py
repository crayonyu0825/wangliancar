"""Captain paired passage/emergency inputs and an independent output veto."""
import copy
import math
import time

from core.behavior_channel import read_channel,capabilities_dict
from core.behavior_contract import BehaviorFrame,finite,require
from core.driving_mission import DrivingDecisionInputs
from core.execution_monitor import ExecutionMonitor
from core.geometry import normalize_angle,opposes_direction
from core.interfaces import DecisionTarget,Trajectory,ControlOut
from core.safety_supervisor import SafetyAssessment
from core.validation import validate_output
from core.maneuver_facts import ManeuverFacts
from core.passage_facts import read_passage_facts
from members.planning.behavior_contract import assess_behavior_request,DRIVING_ACTIONS as REQUEST_ACTIONS
from members.planning.driving_behavior import DrivingPlanningInputs,candidate_id
from members.planning.maneuver_inputs import read_inputs
from members.planning.maneuver_scene import read_maneuver_scene
from members.planning.route_scene import read_route_scene
from members.planning.candidate_validation import ValidationBudget,ExecutionEnvelope,validate_candidate
from members.planning.lane_change_generator import _source_predictions
from members.planning.passage_planner import body_occupancy_intervals,_window_check,_Work,_body


DRIVING_ACTIONS=('INTERSECTION','PATH','PATH_STOP','EMERGENCY_CANDIDATES','FEEDBACK')


def _model_copy(value):
    result=copy.copy(value)
    for name in vars(value):
        if name!='geometry_cache':setattr(result,name,copy.deepcopy(getattr(value,name)))
    return result


class DrivingIntegrationInputs(object):
    def __init__(self,decision,planning,monitor_budget):
        require(isinstance(decision,DrivingDecisionInputs) and isinstance(planning,DrivingPlanningInputs)
                and isinstance(monitor_budget,ValidationBudget),'DRIVING_PAIR_INVALID')
        self.decision,self.planning=decision.snapshot(),_model_copy(planning)
        self.monitor_budget=copy.deepcopy(monitor_budget)

    def snapshot(self):
        return DrivingIntegrationInputs(self.decision,self.planning,self.monitor_budget)


class DrivingInputBridge(object):
    def __init__(self,provider,clock=None):
        require(callable(provider),'DRIVING_PAIR_PROVIDER_INVALID')
        self.provider,self.clock=provider,clock or time.monotonic
        self.settings=None;self.key=None;self.value=None;self.reason='NOT_PREPARED'
        self.monitor=ExecutionMonitor();self.execution_tracking={};self.execution_key=None
        self.send_error=None

    @staticmethod
    def _key(p,context):return (id(p),context.key(),p.frame_id,p.timestamp)

    def _validate(self,pair,p,context):
        m=read_inputs(lambda *args:pair.planning,p,context,self.clock,'DRIVING_PAIR',DrivingPlanningInputs)
        d=pair.decision;mission=d.mission;now=self.clock()
        frame=BehaviorFrame(context,p.frame_id,now,p.valid_until,p.ego.x,p.ego.y,p.ego.heading,p.ego.speed,usable=p.valid)
        require(mission.evidence.frame_id==p.frame_id and mission.model_evidence.frame_id==p.frame_id
                and mission.model_evidence.observed_at_s==m.produced_at_s
                and mission.model_evidence.valid_until_s==m.valid_until_s
                and mission.evidence.verified(frame,allowed_source_kinds=('task','verified_fusion'))
                and mission.model_evidence.verified(frame,allowed_source_kinds=('task','verified_fusion')),
                'DRIVING_PAIR_SOURCE_MISMATCH')
        require(d.model_id==m.model_id and d.vehicle_extents==tuple(getattr(m.vehicle,n) for n in
                ('front_offset_m','rear_offset_m','half_width_m'))
                and d.max_deceleration_mps2==m.limits.max_deceleration_mps2
                and mission.mission_id==m.mission.mission_id and mission.map_digest==m.mission.map_digest
                and mission.signature()==m.mission.signature(),'DRIVING_PAIR_MISSION_OR_MODEL_MISMATCH')
        require(isinstance(m.execution_envelope,ExecutionEnvelope) and finite(m.speed_error_mps)
                and m.speed_error_mps>=0,'DRIVING_EXECUTION_BOUNDS_UNAVAILABLE')
        # Publish the same validated execution bounds to decision completion.
        # A rear axle lane change alone must not discard the tail's corridor.
        if d.execution_position_error_m is None and d.execution_heading_error_rad is None:
            d.execution_position_error_m=m.execution_envelope.position_error_m
            d.execution_heading_error_rad=m.execution_envelope.heading_error_rad
        require(d.execution_position_error_m==m.execution_envelope.position_error_m
                and d.execution_heading_error_rad==m.execution_envelope.heading_error_rad,
                'DRIVING_DECISION_EXECUTION_BOUNDS_MISMATCH')
        require(self.settings is not None,'DRIVING_RUNTIME_SETTINGS_UNAVAILABLE')
        for name,actual in tuple(zip(('front_offset_m','rear_offset_m','half_width_m'),d.vehicle_extents))+(
                ('wheelbase_m',m.vehicle.wheelbase_m),('front_steer_max_rad',m.limits.max_front_steer_rad)):
            value=getattr(self.settings,name,None)
            require(finite(value) and abs(value-actual)<=1e-9,'DRIVING_CONTROL_GEOMETRY_MISMATCH')
        b=pair.monitor_budget
        require(type(b.max_checks) is int and b.max_checks>0 and finite(b.deadline_monotonic_s)
                and now<b.deadline_monotonic_s,'DRIVING_MONITOR_BUDGET_UNAVAILABLE')
        facts=ManeuverFacts(p,self.clock)
        require(facts.map_digest()==mission.map_digest,'DRIVING_SOURCE_MAP_MISMATCH')
        facts.objects();facts.check_current(True)

    def prepare(self,p):
        try:
            context,unused,unused2=read_channel(p,self.clock());key=self._key(p,context)
            if key==self.key:return
            self.key,self.value=key,None
            pair=self.provider(p,context)
            if pair is None:self.reason='NO_DRIVING_TASK';return
            require(isinstance(pair,DrivingIntegrationInputs),'DRIVING_PAIR_INVALID')
            pair=pair.snapshot();self._validate(pair,p,context)
            self.value=pair;self.reason='CURRENT_PAIRED_DRIVING_INPUTS'
        except Exception as error:
            self.value=None;self.reason='DRIVING_PAIR:'+str(error)

    def _read(self,p,context):
        require(self.key==self._key(p,context),'DRIVING_PAIR_NOT_PREPARED')
        require(self.value is not None,self.reason)
        pair=self.value.snapshot();self._validate(pair,p,context);return pair

    def decision_inputs(self,p,context):
        if self.key==self._key(p,context) and self.reason=='NO_DRIVING_TASK':return None
        return self._read(p,context).decision

    def planning_inputs(self,p,context):return self._read(p,context).planning

    def assess_execution(self,p,t):
        key=(id(p),p.frame_id,p.timestamp,p.ego.x,p.ego.y,p.ego.heading,p.ego.speed)
        if key!=self.execution_key:
            self.execution_key=key
            try:
                context,unused,unused2=read_channel(p,self.clock());pair=self._read(p,context)
                m=pair.planning
                self.monitor.max_speed_mps=m.limits.max_speed_mps;self.monitor.speed_error_mps=m.speed_error_mps
                self.execution_tracking=self.monitor.observe(p,t,m.execution_envelope,self.clock())
            except Exception as error:
                self.execution_tracking=dict(brake_required=bool(self.monitor.history),reason_code=str(error))
        if self.send_error or self.execution_tracking.get('brake_required'):
            return SafetyAssessment('controlled_stop','DRIVING_EXECUTION:'+str(self.send_error or
                self.execution_tracking.get('reason_code')))
        return SafetyAssessment()

    def record_sent(self,p,t,receipt,safety_active=False):
        if safety_active:return
        try:
            context,unused,unused2=read_channel(p,self.clock());m=self._read(p,context).planning
            self.monitor.max_speed_mps=m.limits.max_speed_mps;self.monitor.speed_error_mps=m.speed_error_mps
            result=self.monitor.record_sent(p,t,m.execution_envelope,self.clock(),receipt)
            self.send_error=result.get('reason_code') if result.get('brake_required') else None
        except Exception as error:self.send_error=str(error)

    def assess(self,p,d,t,c):
        """Recheck delivered points, never replace them with another generated path."""
        mode='controlled_stop'
        original=(p.frame_id,p.timestamp,p.ego.x,p.ego.y,p.ego.heading,p.ego.speed)
        try:
            validate_output(d,DecisionTarget,p);validate_output(t,Trajectory,d)
            normalized=validate_output(copy.deepcopy(c),ControlOut,t)
            require(all(getattr(normalized,k)==getattr(c,k) for k in ('throttle','brake','steering')),
                    'DRIVING_CONTROL_NOT_NORMALIZED')
            context,caps,unused=read_channel(p,self.clock());pair=self._read(p,context);m=pair.planning
            request=getattr(d,'behavior_request',None)
            if not isinstance(request,dict) or request.get('maneuver')=='SIGNAL_STOP':
                return self._ordinary_assessment(p,d,t,c,context,pair)
            a=assess_behavior_request(d.behavior_request,context.to_dict(),p.frame_id,self.clock(),
                min(p.valid_until,d.valid_until),'process_monotonic',d.behavior_active_identity,
                capabilities=capabilities_dict(caps),supported_actions=REQUEST_ACTIONS,current_lane_id=p.lane.lane_id,
                frame_usable=p.valid,frame_paused=False,runtime_driving=True)
            require(a['eligible'],a['reason_code']);r=a['request']
            if r['maneuver']=='EMERGENCY' and r['stage']=='SELECT':
                require(t.emergency_stop and c.throttle==0 and c.brake>0 and c.hazard_signal,
                        'EMERGENCY_FIRST_CYCLE_BRAKE_REQUIRED')
                self.reason='CURRENT_IMMEDIATE_BRAKE_COMMAND_VERIFIED'
                return SafetyAssessment()
            execution=self.assess_execution(p,t)
            require(not execution.active,execution.reason)
            if r['maneuver']=='EMERGENCY' and t.emergency_stop:
                reply=t.behavior_feedback
                rejected=(isinstance(reply,dict) and reply.get('status')=='REJECTED'
                    and reply.get('producer')=='planning' and reply.get('candidate_id')==r.get('candidate_id')
                    and all(reply.get(k)==r[k] for k in ('intent_id','stage','revision'))
                    and reply.get('planning_metrics',{}).get('fallback_brake_required') is True)
                require((r['stage']=='BRAKE_FALLBACK' or rejected) and c.throttle==0 and c.brake>0
                    and c.hazard_signal,'EMERGENCY_BRAKE_FALLBACK_AUTHORITY_REQUIRED')
                # observe checked actual GPS and the old sent trajectory. The
                # current zero-speed points authorize a brake command only.
                self.reason='CURRENT_BRAKE_FALLBACK_COMMAND_VERIFIED_NO_MOTION_ACCEPTANCE'
                return SafetyAssessment()
            first=t.points[0]
            require(first.relative_time==0 and abs(first.x-p.ego.x)<=1e-9 and abs(first.y-p.ego.y)<=1e-9
                    and abs(normalize_angle(first.heading-p.ego.heading))<=1e-9
                    and abs(first.speed-p.ego.speed)<=1e-9 and t.motion_direction==1
                    and not opposes_direction(p.ego.vx,p.ego.vy,p.ego.heading,p.ego.speed),'DRIVING_ACTUAL_START_MISMATCH')
            identity={k:r[k] for k in ('intent_id','stage','revision')}
            require(t.behavior_identity==identity and isinstance(t.candidate_id,str) and t.candidate_id
                    and isinstance(t.behavior_feedback,dict) and t.behavior_feedback.get('candidate_id')==t.candidate_id,
                    'DRIVING_CANDIDATE_IDENTITY_MISMATCH')
            spent=0
            if r['maneuver']=='EMERGENCY':
                require(r.get('candidate_id')==t.candidate_id and c.throttle==0 and c.brake>0 and c.hazard_signal,
                        'EMERGENCY_ACCEPTED_BRAKING_CANDIDATE_REQUIRED')
                target=r['goal_pose'];end=t.points[-1]
                if r['stage']=='BRAKE_FALLBACK':
                    actions=[action for action in ('BRAKE','LEFT','RIGHT','BRAKE_LEFT','BRAKE_RIGHT')
                        if t.candidate_id==candidate_id(context,r['intent_id'],(action,
                            tuple(target[k] for k in ('x','y','body_heading_rad')),m.model_id,m.mission.mission_id))]
                    require(len(actions)==1,'EMERGENCY_FALLBACK_ORIGINAL_ACTION_UNKNOWN')
                    action=actions[0]
                else:action=r['stage'][8:]
                require(t.candidate_id==candidate_id(context,r['intent_id'],(action,
                    tuple(target[k] for k in ('x','y','body_heading_rad')),m.model_id,m.mission.mission_id)),
                    'EMERGENCY_CANDIDATE_BINDING_MISMATCH')
                brake_hold=t.planning_metrics.get('emergency_outcome')=='verified_safe_brake_stop'
                if brake_hold:
                    require(p.ego.speed<=1e-9 and not t.precision_stop and t.stop_required
                        and t.planning_metrics.get('original_goal_pose')==target
                        and all(v.speed==0 and math.hypot(v.x-p.ego.x,v.y-p.ego.y)<=1e-9
                            and abs(normalize_angle(v.heading-p.ego.heading))<=1e-9 for v in t.points),
                        'EMERGENCY_SAFE_STOP_NOT_MEASURED')
                else:
                    require(math.hypot(end.x-target['x'],end.y-target['y'])<=m.limits.distance_tolerance_m
                            and abs(normalize_angle(end.heading-target['body_heading_rad']))<=m.limits.heading_tolerance_rad,
                            'EMERGENCY_FROZEN_GOAL_MISMATCH')
                if action=='BRAKE':ids=(p.lane.lane_id,)
                else:
                    option=next((v for v in m.mission.emergency_goals if v['action']==action and
                        (v['goal'].x,v['goal'].y,v['goal'].body_heading_rad)==
                        (target['x'],target['y'],target['body_heading_rad'])),None)
                    require(option is not None,'EMERGENCY_TASK_AUTHORITY_MISMATCH');ids=option['lane_ids']
                scene=read_maneuver_scene(p,ids,self.clock,m.geometry_cache,p.lane.lane_id if len(ids)==2 else None)
                corridor=scene['legal_crossing_corridor'] if len(ids)==2 else scene['corridor']
                require(t.stop_required and (t.precision_stop or brake_hold) and t.points[-1].speed==0,
                        'EMERGENCY_TERMINAL_STOP_REQUIRED')
            else:
                scene=read_passage_facts(p,m.mission.passage_id,m.vehicle,self.clock);corridor=scene['corridor']
                shape=[(v.x,v.y,v.heading) for v in t.points];work=_Work(pair.monitor_budget)
                unused_stations,occupied=body_occupancy_intervals(shape,m.vehicle,scene['conflict_polygons'],
                    m.execution_envelope,.25,work)
                if r['stage']=='WAIT_ENTRY':
                    require(not occupied and t.stop_required and t.points[-1].speed==0,
                            'PASSAGE_WAIT_ENTERS_CONFLICT_REGION')
                else:
                    windows=[(max(0.,a-self.clock()),b-self.clock()) for a,b in
                             m.mission.allowed_windows_abs_s if b>self.clock()]
                    require(m.mission.right_of_way!='UNKNOWN' and scene['junction']['exit_occupancy']=='clear',
                            'PASSAGE_CURRENT_ADMISSION_AUTHORITY_UNKNOWN')
                    if not occupied:
                        from core.region_geometry import rectangle
                        actual=rectangle(p.ego.x,p.ego.y,p.ego.heading,m.vehicle.front_offset_m,
                            m.vehicle.rear_offset_m,m.vehicle.half_width_m)
                        require(r['stage']=='EXIT' and scene['exit_region'].body_clearance(
                            actual,work.step,pair.monitor_budget.max_checks-work.count)>1e-8,
                            'PASSAGE_EMPTY_OCCUPANCY_NOT_ACTUAL_EXIT')
                    _window_check(t.points,occupied,m.execution_envelope,windows,work)
                    require(scene['exit_region'].body_clearance(_body(shape[-1],m.vehicle,m.execution_envelope,
                        m.mission.exit_margin_m),work.step,pair.monitor_budget.max_checks-work.count)>1e-8,
                        'PASSAGE_EXIT_BODY_OUTSIDE')
                spent=work.count
            objects=_source_predictions(scene['objects'],m.prediction_envelopes,self.clock())
            for region,targets in ((corridor,objects),(scene['coverage'],[])):
                b=ValidationBudget(pair.monitor_budget.max_checks-spent,pair.monitor_budget.max_depth,
                    pair.monitor_budget.min_interval_s,min(pair.monitor_budget.deadline_monotonic_s,
                    time.monotonic()+scene['source_valid_until_s']-self.clock()))
                report=validate_candidate(t.points,1,m.vehicle,m.limits,region,targets,b,
                    execution_envelope=m.execution_envelope);spent+=report['checks']
                if report['status']=='unsafe':mode='emergency_stop'
                require(report['status']=='safe','DRIVING_MONITOR_PATH:'+report['reason_code'])
            require(original==(p.frame_id,p.timestamp,p.ego.x,p.ego.y,p.ego.heading,p.ego.speed),
                    'DRIVING_SOURCE_CHANGED_DURING_RECHECK')
            self._read(p,context);self.reason='CURRENT_DRIVING_PATH_RECHECKED'
            return SafetyAssessment()
        except Exception as error:
            self.reason='DRIVING_MONITOR:'+str(error);return SafetyAssessment(mode,self.reason)

    def _ordinary_assessment(self,p,d,t,c,context,pair):
        m=pair.planning
        execution=self.assess_execution(p,t)
        require(not execution.active,execution.reason)
        require(t.emergency_stop or t.planning_metrics.get('ordinary_dynamic_speed') is True
                or all(v.speed==0 and math.hypot(v.x-p.ego.x,v.y-p.ego.y)<=1e-9 for v in t.points),
                'ORDINARY_P05_MODEL_NOT_CONSUMED')
        scene=read_route_scene(p,self.clock,m.geometry_cache)
        spent=0;objects=_source_predictions(scene['objects'],m.prediction_envelopes,self.clock())
        for region,targets in ((scene['corridor'],objects),(scene['coverage'],[])):
            b=ValidationBudget(pair.monitor_budget.max_checks-spent,pair.monitor_budget.max_depth,
                pair.monitor_budget.min_interval_s,min(pair.monitor_budget.deadline_monotonic_s,
                time.monotonic()+scene['source_valid_until_s']-self.clock()))
            report=validate_candidate(t.points,1,m.vehicle,m.limits,region,targets,b,
                execution_envelope=m.execution_envelope);spent+=report['checks']
            require(report['status']=='safe','ORDINARY_MONITOR_PATH:'+report['reason_code'])
        self._read(p,context);self.reason='CURRENT_ORDINARY_DYNAMIC_PATH_RECHECKED'
        return SafetyAssessment()
