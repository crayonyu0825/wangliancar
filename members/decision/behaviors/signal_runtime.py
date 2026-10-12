"""Formal stop obligation: control owns the continuous standstill clock."""
import time

from core.behavior_protocol import RuntimeRequest,GoalPose,behavior_frame
from core.geometry import project_polyline
from core.polyline_position import point_at_distance
from members.decision.behaviors.stop_policies import SignalDwellPolicy,SignalStopObservation


class SignalRuntime(object):
    def __init__(self):self.policy=None;self.context=None;self.goal=None;self.stop_id=None

    def update(self,p,out,observation):
        env=p.behavior_environment
        if env is None or not env.stop_rule_id or env.rule_stop_duration_s<=0:return
        f=behavior_frame(p,time.monotonic())
        if not env.verified_task(f,p.observation_mode):return
        if self.context!=f.context.key():
            if self.context is not None and self.context[-1]==f.context.session_id and self.policy is not None:
                out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1;return
            duration=env.rule_stop_duration_s+(0.001 if env.rule_stop_strict else 0)
            self.policy=SignalDwellPolicy(duration);self.context=f.context.key();self.goal=None;self.stop_id=None
        if observation is not None and observation.usable and observation.stop_distance_m is not None:
            reference=p.lane.forward_reference if p.lane.forward_reference_valid else p.lane.center_line
            current=project_polyline(reference,p.ego.x,p.ego.y)
            if current is None:return
            if self.goal is None or observation.stop_id!=self.stop_id:
                value=point_at_distance(reference,current['s']+observation.stop_distance_m)
                self.goal=GoalPose(*value);self.stop_id=observation.stop_id
            observation=SignalStopObservation(observation.stop_id,observation.stop_distance_m,observation.red_required,
                usable=observation.usable,speed_cap_mps=max(out.target_speed,0.5),goal_pose=self.goal)
        feedback=None
        if self.policy.session.intent_id:
            req=self.policy.session.snapshot(f)
            for value in p.behavior_feedback:
                valid,_,_=self.policy.session.validator.check(value,req,f,record=False)
                if valid and value.producer in ('control','runtime'):feedback=value;break
        result=self.policy.evaluate(f,observation,p.behavior_capabilities,feedback,
                                    dwell_required=True,safety_override=out.mode=='EMERGENCY_BRAKE')
        request=result.request
        if request is None or request.status in ('COMPLETED','CANCELLED'):return
        if out.mode=='EMERGENCY_BRAKE':return
        if request.goal.stop_distance_m<0:return
        out.stop_distance=request.goal.stop_distance_m
        out.precision_stop=True
        if out.stop_distance<=0.1:out.mode='STOP';out.target_speed=0.0
        elif out.target_speed<=0:out.target_speed=0.5
        if request.dispatch_allowed:
            out.behavior_request=RuntimeRequest(f,request.intent_id,'SIGNAL_STOP',request.stage,request.revision,
                request.source_frame_id,request.goal,'HOLD',None,
                origin_lane_id=p.lane.lane_id,issued_at_s=request.issued_at_s)
