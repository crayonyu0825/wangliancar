"""Observed unresolved obstruction: actual stop, hazard warning and clear gate."""
import time
from core.behavior_protocol import RuntimeRequest,BehaviorGoal,GoalPose,LightIntent,behavior_frame
from core.geometry import project_polyline
from core.polyline_position import point_at_distance
from core.traffic_quality import signal_stop_requirement
from members.decision.behaviors.stop_policies import BlockedRoadPolicy,BlockingObservation


class BlockedRuntime(object):
    def __init__(self):
        self.policy=BlockedRoadPolicy();self.context=None;self.goal=None;self.identifier=None;self.reason='NO_BLOCKAGE'

    def update(self,p,out,constraints):
        env=p.behavior_environment
        if env is None:return
        frame=behavior_frame(p,time.monotonic())
        if not env.current(frame,p.observation_mode) or not env.verified_task(frame,p.observation_mode):return
        if self.context!=frame.context.key():
            if self.context is not None and self.context[-1]==frame.context.session_id and self.policy.session.intent_id:
                out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1;out.precision_stop=False;return
            self.__init__();self.context=frame.context.key()
        if out.behavior_request is not None or signal_stop_requirement(p) is not None:return
        blocked=constraints.blockage_candidates if constraints is not None else ()
        observation=None
        if blocked:
            identifier,distance=min(blocked,key=lambda v:(v[1],v[0]))
            reference=p.lane.forward_reference if p.lane.forward_reference_valid else p.lane.center_line
            projection=project_polyline(reference,p.ego.x,p.ego.y)
            if projection is None:return
            if self.identifier!=identifier:
                self.goal=GoalPose(*point_at_distance(reference,projection['s']+distance));self.identifier=identifier
            observation=BlockingObservation('target:'+str(identifier),distance,True,speed_cap_mps=out.target_speed)
        elif self.policy.session.intent_id:
            # Missing targets clear the obligation only under current explicit
            # complete coverage; retained danger constraints still take priority.
            observation=BlockingObservation('prior',0,False,verified_clear=not p.targets)
        if observation is None:return
        feedback=None
        if self.policy.session.intent_id:
            for value in p.behavior_feedback:
                valid,_,_=self.policy.session.validator.check(value,self.policy.session.snapshot(frame),frame,record=False)
                if valid and value.producer in ('control','runtime'):feedback=value;break
        result=self.policy.evaluate(frame,observation,p.behavior_capabilities,feedback,safety_override=out.mode=='EMERGENCY_BRAKE')
        self.reason=result.reason;request=result.request
        if request is None or request.status in ('COMPLETED','CANCELLED'):return
        if not request.dispatch_allowed:return
        goal=BehaviorGoal(stop_distance_m=request.goal.stop_distance_m,speed_cap_mps=out.target_speed,
            goal_pose=self.goal,precision_stop=True,light_intent=LightIntent(hazard_signal=True))
        out.behavior_request=RuntimeRequest(frame,request.intent_id,'BLOCKED_ROAD',request.stage,request.revision,
            request.source_frame_id,goal,'HOLD',origin_lane_id=p.lane.lane_id,issued_at_s=request.issued_at_s)
