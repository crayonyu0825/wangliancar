"""Formal parking task dispatch behind the fixed decision entry, SDK-free."""
import copy

from core.behavior_contract import BehaviorFrame,TaskContext,finite,require
from core.behavior_evidence import Evidence
from core.behavior_channel import read_channel
from core.interfaces import DecisionMode
from core.parking_mission import ParkingMission
from members.decision import protocol
from members.decision.behaviors.parking import ParkingPolicy
from members.decision.behaviors.parking_environment import parking_facts


class ParkingDecisionInputs(object):
    """Explicit current model assertions; neither a calibration nor a task selector.

    The captain supplies the same geometry/tolerances to planning and control.
    Model evidence describes this input snapshot, not a physical proof created
    by the constructor. Object error bounds must be provided explicitly.
    """
    def __init__(self,model_evidence,mission,model_id,vehicle_extents,
                 position_uncertainty_m,velocity_uncertainty_mps,geometry_max_checks,
                 tolerances,speed_cap_mps,dwell_s=10.):
        require(isinstance(model_evidence,Evidence) and isinstance(mission,ParkingMission)
                and isinstance(model_id,str) and bool(model_id),'PARKING_DECISION_MODEL_UNAVAILABLE')
        require(isinstance(vehicle_extents,(tuple,list)) and len(vehicle_extents)==3
                and all(finite(v) and v>0 for v in vehicle_extents),'PARKING_DECISION_EXTENTS_UNAVAILABLE')
        require(all(finite(v) and v>=0 for v in (position_uncertainty_m,velocity_uncertainty_mps))
                and type(geometry_max_checks) is int and geometry_max_checks>0,
                'PARKING_DECISION_ERROR_OR_BUDGET_UNAVAILABLE')
        require(isinstance(tolerances,(tuple,list)) and len(tolerances)==3
                and all(finite(v) and v>0 for v in tolerances)
                and finite(speed_cap_mps) and speed_cap_mps>0 and finite(dwell_s) and dwell_s>=10.,
                'PARKING_DECISION_LIMITS_UNAVAILABLE')
        e=model_evidence
        self.model_evidence=Evidence(TaskContext(*e.context.key()),e.frame_id,e.observed_at_s,
            e.valid_until_s,e.usable,e.coverage_verified,e.source,e.clock_id,e.source_kind)
        self.mission,self.model_id=mission.snapshot(),model_id
        self.extents,self.tolerances=tuple(vehicle_extents),tuple(tolerances)
        self.position_error,self.velocity_error=position_uncertainty_m,velocity_uncertainty_mps
        self.max_checks,self.speed_cap,self.dwell_s=geometry_max_checks,speed_cap_mps,dwell_s

    def snapshot(self):
        return ParkingDecisionInputs(self.model_evidence,self.mission,self.model_id,self.extents,
            self.position_error,self.velocity_error,self.max_checks,self.tolerances,self.speed_cap,self.dwell_s)

    def binding(self):
        e=self.model_evidence
        return (self.model_id,self.extents,self.tolerances,self.position_error,self.velocity_error,
                self.max_checks,self.speed_cap,self.dwell_s,e.source_kind,e.source,e.clock_id)


class ParkingRuntime(object):
    def __init__(self,provider,session_factory,clock):
        require(provider is None or callable(provider),'PARKING_DECISION_PROVIDER_INVALID')
        self.provider,self.session_factory,self.clock=provider,session_factory,clock
        self.policy,self.binding,self.mission_id=None,None,None
        self.prepared,self.last_frame=None,None
        self.claimed,self.reason=False,'NOT_CONFIGURED'

    def pending(self):
        return bool(self.policy and self.policy.session.intent_id
                    and self.policy.session.status not in ('COMPLETED','CANCELLED'))

    def suspend(self):
        if self.pending() and self.last_frame is not None:
            frame=copy.copy(self.last_frame); frame.usable=False
            self.policy.session.tick(frame,safety_override=True)

    def prepare(self,p,now):
        self.prepared,self.claimed=None,False
        if self.provider is None and not self.pending(): return False
        try:
            context,caps,replies=read_channel(p,now)
            frame=BehaviorFrame(context,p.frame_id,now,p.valid_until,p.ego.x,p.ego.y,
                                p.ego.heading,p.ego.speed,usable=True)
            self.last_frame=frame
            value=self.provider(p,context) if self.provider is not None else None
            if value is None and not self.pending():
                self.reason='NO_PARKING_TASK'; return False
            self.claimed=True
            require(isinstance(value,ParkingDecisionInputs),'PARKING_DECISION_INPUTS_UNAVAILABLE')
            value=value.snapshot(); model=value.model_evidence
            require(model.frame_id==p.frame_id and model.valid_until_s<=p.valid_until
                    and model.verified(frame,allowed_source_kinds=('task','verified_fusion')),
                    'PARKING_DECISION_MODEL_EXPIRED_OR_MISMATCHED')
            require(value.mission.evidence.verified(frame,allowed_source_kinds=('task','verified_fusion')),
                    'PARKING_TASK_MISSION_UNVERIFIED')
            if (self.policy is not None and self.policy.session.status=='COMPLETED'
                    and self.mission_id==value.mission.mission_id):
                self.claimed=False; self.reason='PARKING_TASK_ALREADY_COMPLETED'; return False
            binding=value.binding()
            if self.policy is not None and self.policy.session.status in ('COMPLETED','CANCELLED'):
                require(value.mission.mission_id!=self.mission_id,'PARKING_TASK_RESTART_REQUIRES_NEW_ID')
                self.policy=None
            if self.policy is not None and self.policy.session.intent_id is None:
                self.policy=None
            if self.policy is None:
                self.policy=ParkingPolicy(*value.extents,dwell_s=value.dwell_s,
                    position_tolerance_m=value.tolerances[0],heading_tolerance_rad=value.tolerances[1],
                    standstill_speed_mps=value.tolerances[2],speed_cap_mps=value.speed_cap,
                    session=self.session_factory())
                self.binding,self.mission_id=binding,value.mission.mission_id
            require(binding==self.binding and self.mission_id==value.mission.mission_id,
                    'PARKING_ACTIVE_MODEL_OR_TASK_CHANGED')
            facts=parking_facts(p,frame,value.mission,value.extents,value.position_error,
                                value.velocity_error,value.max_checks,self.clock)
            deadline=min(facts.evidence.valid_until_s,model.valid_until_s,p.valid_until)
            facts.evidence.valid_until_s=deadline
            facts.free_area.valid_until_s=min(facts.free_area.valid_until_s,deadline)
            require(frame.current(self.clock()) and self.clock()<deadline,'PARKING_SOURCE_EXPIRED_DURING_READ')
            self.prepared=(frame,caps,replies,value,facts,deadline)
            self.reason='FORMAL_PARKING_TASK_READY'
        except Exception as error:
            self.claimed=True; self.reason='PARKING_INPUT:'+str(error)
            self.suspend()
        return self.claimed

    def provisional(self,p,output):
        if self.prepared is None:
            self.hold(p,output); return
        output.valid=True; output.mode=DecisionMode.KEEP_LANE
        output.target_lane_id=p.lane.lane_id
        output.target_speed=self.prepared[3].speed_cap if self.prepared else 0.
        output.stop_distance=-1.; output.precision_stop=False; output.reason=self.reason

    def hold(self,p,output):
        threshold=self.policy.standstill_speed_mps if self.policy else 0.
        output.mode=DecisionMode.EMERGENCY_BRAKE if p.ego.speed>threshold else DecisionMode.STOP
        output.target_speed,output.stop_distance=0.,0.
        output.precision_stop=True; output.reason=self.reason; output.valid=True
        output.behavior_request,output.behavior_active_identity=None,None

    def apply(self,p,output,priority=False):
        if not self.claimed: return
        limit=protocol.speed_limit(p)
        if priority or self.prepared is None or limit==0:
            self.suspend()
            if not priority:
                if limit==0: self.reason='PARKING_SPEED_LIMIT_ZERO'
                self.hold(p,output)
            else:
                self.reason='PARKING_WAITING_FOR_SAFETY_OR_SIGNAL'
                if output.behavior_request is None and output.mode not in (DecisionMode.STOP,DecisionMode.EMERGENCY_BRAKE):
                    self.hold(p,output)
            return
        frame,caps,replies,value,facts,deadline=self.prepared
        require(frame.current(self.clock()) and self.clock()<deadline,'PARKING_SOURCE_EXPIRED_DURING_DISPATCH')
        current=[r for r in replies if r.intent_id==self.policy.session.intent_id
                 and r.revision==self.policy.session.revision]
        suggestions=[r for r in current if r.producer=='planning' and r.candidate_options]
        self.policy.offer_candidate_options(frame,max(suggestions,key=lambda r:r.produced_at_s) if suggestions else None)
        reply=max(current,key=lambda r:(r.produced_at_s,r.producer!='planning')) if current else None
        result=self.policy.evaluate(frame,facts,caps,reply,intent_key=value.mission.mission_id)
        self.reason=result.reason
        request=result.request
        if request is None or not request.dispatch_allowed or request.status in ('COMPLETED','CANCELLED'):
            self.hold(p,output); return
        require(frame.current(self.clock()) and self.clock()<deadline,'PARKING_SOURCE_EXPIRED_DURING_DISPATCH')
        output.valid_until=min(output.valid_until,deadline,request.valid_until_s)
        output.mode=DecisionMode.KEEP_LANE
        output.target_speed=min(value.speed_cap,request.goal.speed_cap_mps,
                                limit if limit is not None else value.speed_cap)
        output.stop_distance=-1.; output.precision_stop=False
        output.reason='PARKING:'+request.stage+':'+result.reason
        output.behavior_request=request.to_dict('runtime_connected')
        output.behavior_request['valid_until_s']=output.valid_until
        output.behavior_active_identity={k:output.behavior_request[k] for k in
            ('intent_id','stage','revision','source_frame_id','issued_at_s')}

    def snapshot(self):
        session=self.policy.session if self.policy else None
        return dict(configured=self.provider is not None,claimed=self.claimed,reason=self.reason,
                    mission_id=self.mission_id,stage=session.stage if session else None,
                    status=session.status if session else None)
