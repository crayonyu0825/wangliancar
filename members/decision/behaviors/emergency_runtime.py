"""Production immediate protection, explicit emergency choice and new recovery."""
import time
from core.behavior_protocol import RuntimeRequest,BehaviorGoal,PlanningFeedback,GoalPose,behavior_frame
from core.behavior_environment import vehicle_footprint
from core.emergency_protocol import EmergencyTask
from core.geometry import project_polyline,normalize_angle
from members.decision.behaviors.session import BehaviorSession


class EmergencyRuntime(object):
    def __init__(self):self.reset()
    def reset(self):
        self.session=BehaviorSession(progress_timeout_s=10)
        self.context=None;self.phase='IDLE';self.reason='NO_ACTIVE_DANGER';self.danger_id=None
        self.reference=None;self.selected=None;self.return_lane=None;self.last_frame=None;self.clear=0;self.stopped=0

    def active(self):return self.phase not in ('IDLE','COMPLETE')

    def update(self,p,out):
        env=p.behavior_environment;task=env.emergency_task if env is not None else None
        if not isinstance(task,EmergencyTask):
            if self.active():out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1;self.reason='EMERGENCY_INPUT_LOST';return True
            return False
        f=behavior_frame(p,time.monotonic());dims=getattr(p.behavior_capabilities,'vehicle_geometry',None)
        if self.context!=f.context.key():
            if self.context is not None and self.context[-1]==f.context.session_id and self.active():
                out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1;self.reason='EMERGENCY_CONTEXT_UNTRUSTED';return True
            self.reset();self.context=f.context.key()
        if not task.dangerous and not self.active():return False
        # Immediate protection precedes candidate evaluation and every failure.
        legacy_emergency=out.mode=='EMERGENCY_BRAKE'
        out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1;out.precision_stop=False
        if not (dims is not None and task.verified(f,p.observation_mode) and env.current(f,p.observation_mode)
                and env.verified_task(f,p.observation_mode)):
            self.reason='EMERGENCY_PERMISSION_COVERAGE_OR_ROUTE_UNKNOWN';return True
        distinct=self.last_frame is None or f.frame_id>self.last_frame
        if self.last_frame is not None and f.frame_id<self.last_frame:
            self.reference=None;self.clear=0;self.last_frame=f.frame_id
            if self.session.intent_id:self.session.advance(f,'SELECT',BehaviorGoal(),force_revision=True)
            self.phase='SELECT';self.reason='EMERGENCY_ROLLBACK_REVALIDATE';return True
        if distinct:self.last_frame=f.frame_id
        if self.session.intent_id is None or task.dangerous and task.danger_id!=self.danger_id:
            if self.session.intent_id is None:
                self.session.start(f,'emergency:'+task.danger_id,'EMERGENCY','SELECT',BehaviorGoal(),('EMERGENCY_CANDIDATES','PATH','FEEDBACK'))
            else:
                if self.session.status=='BLOCKED':self.session.renew_after_verified_change(f,'new-danger:'+task.danger_id)
                self.session.advance(f,'SELECT',BehaviorGoal(),force_revision=True)
            self.danger_id=task.danger_id;self.reference=None;self.selected=None;self.phase='SELECT'
            self.return_lane=task.return_lane_id or env.return_lane_id or p.lane.lane_id
        if distinct:
            self.clear=self.clear+1 if not task.dangerous and not legacy_emergency else 0
            self.stopped=self.stopped+1 if p.ego.speed<=0.05 else 0
        if self.phase in ('SELECT','EXECUTE') and self.clear>=2 and self.stopped>=2:
            body=vehicle_footprint(GoalPose(p.ego.x,p.ego.y,p.ego.heading),dims['front_offset_m'],dims['rear_offset_m'],dims['half_width_m'])
            lane=env.lane(self.return_lane)
            if lane is not None and env.coverage.contains(body):
                if self.session.status=='BLOCKED':
                    self.session.renew_after_verified_change(f,'danger-cleared:'+self.danger_id)
                self.reference=None;self.phase='RECOVER_EVAL'
                self.session.advance(f,'RECOVER_EVAL',BehaviorGoal(target_lane_id=self.return_lane,speed_cap_mps=1.0),
                    required=('LANE_CHANGE','PATH','LIGHTS','FEEDBACK'),reason='OBSERVED_ACTUAL_STOP_AND_COVERED_CLEAR_NEW_RECOVERY')
        replies=[]
        for value in p.behavior_feedback:
            good,_,_=self.session.validator.check(value,self.session.snapshot(f),f,record=False)
            if good:replies.append(value)
        reply=next((v for v in replies if v.status in ('FAILED','REJECTED')),None)
        if self.phase=='SELECT':
            feasible=[v for v in replies if isinstance(v,PlanningFeedback) and v.status=='PLANNED'
                and v.candidate_reference is not None and v.risk_cost is not None and v.rule_cost is not None]
            if feasible:
                self.selected=min(feasible,key=lambda v:(v.risk_cost,-v.clearance_lower_bound_m,v.rule_cost,v.candidate_id))
                reply=self.selected
        else:
            reply=reply or next((v for v in replies if (v.producer=='planning')==(self.phase=='RECOVER_EVAL')),None)
        request=self.session.tick(f,p.behavior_capabilities,reply)
        if not request.dispatch_allowed:self.reason='EMERGENCY_BRAKE_CANDIDATES_OR_PROGRESS_UNAVAILABLE';return True
        feedback=self.session.last_feedback
        if self.phase=='SELECT' and self.selected is not None:
            self.reference=self.selected.candidate_reference
            self.session.advance(f,'EXECUTE_'+self.selected.candidate_action,BehaviorGoal(goal_pose=self.selected.goal_pose))
            self.phase='EXECUTE'
        if self.phase=='RECOVER_EVAL' and isinstance(feedback,PlanningFeedback) and feedback.status=='PLANNED':
            self.reference=feedback.candidate_reference;self.phase='RECOVER_EXEC'
            self.session.advance(f,'RECOVER_EXEC',BehaviorGoal(target_lane_id=self.return_lane,speed_cap_mps=1.0))
        recovery=self.phase.startswith('RECOVER')
        if recovery and (task.dangerous or legacy_emergency):
            self.reference=None;self.phase='SELECT';self.session.advance(f,'SELECT',BehaviorGoal(),
                required=('EMERGENCY_CANDIDATES','PATH','FEEDBACK'),force_revision=True);recovery=False
        if self.phase=='RECOVER_EXEC':
            body=vehicle_footprint(GoalPose(p.ego.x,p.ego.y,p.ego.heading),dims['front_offset_m'],dims['rear_offset_m'],dims['half_width_m'])
            lane=env.lane(self.return_lane);projection=project_polyline(lane.center_line,p.ego.x,p.ego.y) if lane else None
            if feedback is not None and feedback.producer in ('control','runtime') and lane and lane.contains(body) and projection and projection['distance']<0.15 and abs(normalize_angle(p.ego.heading-projection['heading']))<0.1:
                self.session.finish(f,True,'ACTUAL_ROUTE_RECOVERY_COMPLETED');self.phase='COMPLETE';self.reason='ACTUAL_ROUTE_RECOVERY_COMPLETED'
                out.mode='KEEP_LANE';out.target_speed=1.0;out.stop_distance=-1;return True
        if recovery:out.mode='KEEP_LANE';out.target_speed=1.0;out.stop_distance=-1
        snapshot=self.session.snapshot(f)
        evaluation=self.phase in ('SELECT','RECOVER_EVAL')
        out.behavior_request=RuntimeRequest(f,snapshot.intent_id,'LANE_CHANGE' if recovery else 'EMERGENCY',self.session.stage,
            self.session.revision,self.session.source_frame_id,self.session.goal,'EVALUATE_ONLY' if evaluation else 'EXECUTE_SELECTED',
            self.reference,p.lane.lane_id,issued_at_s=snapshot.issued_at_s)
        self.reason=self.phase
        return True
