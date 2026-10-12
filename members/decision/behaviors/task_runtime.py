"""Formal D09 progress, temporary-action anchors and actual destination stop."""
import copy
import math
import time
from core.behavior_protocol import RuntimeRequest,behavior_frame
from core.behavior_environment import vehicle_footprint
from core.task_protocol import TaskRouteFacts
from core.geometry import project_polyline,projection_within_polyline
from core.road_region import RoadRegion
from members.decision.behaviors.task_progress import TaskProgressPolicy


class TaskRuntime(object):
    def __init__(self):
        self.policy=TaskProgressPolicy();self.phase='IDLE';self.reason='TASK_PROGRESS_NOT_SUPPLIED'
        self.context=None;self.anchor=None;self.completed_obligations=set()

    def update(self,p,out,temporary=False,completed_obligations=()):
        env=p.behavior_environment;facts=env.task_progress if env is not None else None
        if not isinstance(facts,TaskRouteFacts):
            if self.context is not None:
                out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1;out.precision_stop=False
                self.reason='FORMAL_TASK_PROGRESS_LOST'
            return
        frame=behavior_frame(p,time.monotonic())
        if self.context is not None and frame.context.key()!=self.context and frame.context.session_id==self.context[-1]:
            out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1;out.precision_stop=False
            self.reason='TASK_CONTEXT_CHANGED_WITHOUT_NEW_SESSION';return
        if self.context!=frame.context.key():
            self.policy=TaskProgressPolicy();self.context=frame.context.key();self.completed_obligations=set();self.anchor=None
        self.completed_obligations.update(completed_obligations)
        if temporary and self.anchor is None:self.anchor=(facts.route_id,facts.segment_id,facts.progress_s_m)
        feedback=None
        if self.policy.session.intent_id is not None:
            req=self.policy.session.snapshot(frame)
            for value in p.behavior_feedback:
                valid,_,_=self.policy.session.validator.check(value,req,frame,record=False)
                if valid and value.producer in ('control','runtime'):feedback=value;break
        dims=getattr(p.behavior_capabilities,'vehicle_geometry',None)
        obligations_complete=set(facts.required_obligations)<=self.completed_obligations
        if feedback is not None and feedback.goal_pose_arrived:
            feedback=copy.deepcopy(feedback)
            body=vehicle_footprint(feedback.actual_pose,dims['front_offset_m'],dims['rear_offset_m'],dims['half_width_m']) if dims and feedback.actual_pose else None
            if (not obligations_complete or body is None or facts.destination_region is None
                    or RoadRegion(facts.destination_region).body_gap(body)<=0):feedback.goal_pose_arrived=False
        result=self.policy.evaluate(frame,facts,p.behavior_capabilities,feedback,
            safety_override=out.mode=='EMERGENCY_BRAKE',temporary_maneuver_active=temporary)
        self.phase,self.reason=result.phase,result.reason
        if temporary or out.mode=='EMERGENCY_BRAKE':return
        request=result.request
        if result.hold_required and result.phase!='COMPLETE':
            out.mode='STOP' if p.ego.speed<=0.05 else 'EMERGENCY_BRAKE';out.target_speed=0
            out.stop_distance=0 if p.ego.speed<=0.05 else -1;out.precision_stop=False;return
        if request is None:return
        if (facts.destination_goal is None or facts.reference_end_reason!='task_destination'
                or facts.segment_id!=facts.segments[-1][0]):
            out.target_speed=min(out.target_speed,facts.speed_cap_mps)
            if facts.reference_end_reason in ('window_end','map_end','branch_ambiguous','map_query_failed'):
                segment=next(v for v in facts.segments if v[0]==facts.segment_id)
                if dims is None:
                    out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1;out.precision_stop=False
                    self.reason='FINITE_TASK_REFERENCE_BODY_EXTENT_UNKNOWN';return
                bound=max(0,segment[3]-facts.progress_s_m-dims['front_offset_m']-0.3)
                out.stop_distance=bound if out.stop_distance<0 else min(bound,out.stop_distance)
                out.precision_stop=True
                if bound<=0.1:out.mode='STOP';out.target_speed=0;out.stop_distance=0
            snapshot=self.policy.session.snapshot(frame)
            out.behavior_request=RuntimeRequest(frame,snapshot.intent_id,'CONTINUOUS_TASK',snapshot.stage,snapshot.revision,
                snapshot.source_frame_id,snapshot.goal,'HOLD',origin_lane_id=p.lane.lane_id,issued_at_s=snapshot.issued_at_s)
            self.anchor=None;return
        reference=p.lane.forward_reference if p.lane.forward_reference_valid else p.lane.center_line
        here=project_polyline(reference,p.ego.x,p.ego.y);goal=project_polyline(reference,facts.destination_goal.x,facts.destination_goal.y)
        if (here is None or goal is None or not projection_within_polyline(goal,len(reference))
                or goal['distance']>0.2 or goal['s']<here['s']-0.15):
            out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1;out.precision_stop=False
            self.reason='DESTINATION_OUTSIDE_SELECTED_FINITE_REFERENCE';return
        distance=max(0,goal['s']-here['s'])
        out.target_speed=min(out.target_speed,facts.speed_cap_mps) if out.target_speed>0 else facts.speed_cap_mps
        out.stop_distance=distance if out.stop_distance<0 else min(out.stop_distance,distance)
        out.precision_stop=True
        if result.phase=='COMPLETE' or distance<0.1:out.mode='STOP';out.target_speed=0;out.stop_distance=0
        snapshot=self.policy.session.snapshot(frame)
        out.behavior_request=RuntimeRequest(frame,snapshot.intent_id,'CONTINUOUS_TASK',snapshot.stage,snapshot.revision,
            snapshot.source_frame_id,snapshot.goal,'HOLD',origin_lane_id=p.lane.lane_id,issued_at_s=snapshot.issued_at_s)
        self.anchor=None
