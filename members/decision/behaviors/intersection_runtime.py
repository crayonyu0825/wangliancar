"""Production D05-D07: bounded occupancy, entry wait and verified admission."""
import math
import time
from core.behavior_protocol import RuntimeRequest,BehaviorGoal,PlanningFeedback,GoalPose,behavior_frame
from core.behavior_environment import vehicle_footprint
from core.intersection_protocol import IntersectionTask,body_stations,observed_objects
from core.geometry import project_polyline
from core.traffic_quality import signal_stop_requirement
from members.decision.behaviors.session import BehaviorSession


class IntersectionRuntime(object):
    def __init__(self):self.reset()
    def reset(self):
        self.context=None;self.task_key=None;self.session=BehaviorSession(progress_timeout_s=10)
        self.phase='IDLE';self.reason='NO_JUNCTION_TASK';self.reference=None
        self.last_frame=None;self.clear_frames=0;self.exit_frames=0
        self.retained={};self.missing_clear={};self.generation=0;self.was_usable=False;self.condition_at=None;self.authorized_at=None;self.moved_at=None

    def active(self):return self.phase not in ('IDLE','EXITED')

    def _wait(self,p,out,frame,task,geometry,reason,inside=False):
        self.reason=reason
        if inside or out.mode=='EMERGENCY_BRAKE':
            out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1;out.behavior_request=None
            return True
        goal=task.stop_goal(geometry);projection=project_polyline(task.route,p.ego.x,p.ego.y)
        distance=max(0,task.entry_s_m-geometry['front_offset_m']-0.3-projection['s'])
        out.target_speed=min(out.target_speed,task.speed_cap_mps) if out.target_speed>0 else task.speed_cap_mps
        out.stop_distance=distance if out.stop_distance<0 else min(distance,out.stop_distance)
        out.mode='KEEP_LANE' if p.ego.speed>0.05 or distance>0.15 else 'STOP';out.precision_stop=True
        out.behavior_request=RuntimeRequest(frame,self.session.intent_id,'JUNCTION_ENTRY',self.session.stage,
            self.session.revision,self.session.source_frame_id,
            BehaviorGoal(goal_pose=goal,stop_distance_m=out.stop_distance,speed_cap_mps=out.target_speed,precision_stop=True),
            'HOLD',origin_lane_id=p.lane.lane_id,issued_at_s=self.session.snapshot(frame).issued_at_s)
        return True

    def update(self,p,out):
        env=p.behavior_environment;task=env.intersection_task if env is not None else None
        if not isinstance(task,IntersectionTask):
            if self.active():out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1;self.reason='JUNCTION_INPUT_LOST';return True
            return False
        frame=behavior_frame(p,time.monotonic());geometry=getattr(p.behavior_capabilities,'vehicle_geometry',None)
        if self.context!=frame.context.key():
            if self.context is not None and self.context[-1]==frame.context.session_id and self.active():
                out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1;self.reason='JUNCTION_CONTEXT_UNTRUSTED';return True
            self.reset();self.context=frame.context.key()
        if self.phase=='EXITED':return False
        if geometry is None or not task.verified(frame,p.observation_mode) or not env.current(frame,p.observation_mode):
            out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1;self.reason='JUNCTION_OBSERVATION_UNKNOWN';return True
        body=vehicle_footprint(GoalPose(frame.ego_x,frame.ego_y,frame.ego_heading),geometry['front_offset_m'],geometry['rear_offset_m'],geometry['half_width_m'])
        lo,hi=body_stations(task.route,body);inside=hi>task.entry_s_m+0.05
        distinct=self.last_frame is None or frame.frame_id>self.last_frame
        if self.last_frame is not None and frame.frame_id<self.last_frame:
            self.clear_frames=0;self.reason='JUNCTION_FRAME_ROLLBACK_REVALIDATE';self.last_frame=frame.frame_id
            return self._wait(p,out,frame,task,geometry,self.reason,inside)
        if distinct:self.last_frame=frame.frame_id
        key=(task.identity,task.route,task.entry_s_m,task.exit_s_m)
        if self.task_key is not None and key!=self.task_key:
            return self._wait(p,out,frame,task,geometry,'JUNCTION_GEOMETRY_CHANGED_REPLAN',inside)
        self.task_key=key
        if self.session.intent_id is None:
            self.session.start(frame,'junction:'+task.identity,'JUNCTION_ENTRY','WAIT_ENTRY',BehaviorGoal(),('INTERSECTION','PATH','FEEDBACK'))
            self.phase='WAIT_ENTRY'
        objects=observed_objects(p);seen=set(o.identifier for o in objects)
        for obj in objects:self.retained[obj.identifier]=(obj,frame.observed_at_s);self.missing_clear[obj.identifier]=0
        missing=False
        for identifier in list(self.retained):
            if identifier in seen:continue
            obj,when=self.retained[identifier]
            dt=max(0,frame.observed_at_s-when)
            reachable=obj.sweep(0,min(dt+0.5,env.coverage.prediction_horizon_s),0.3+dt*env.coverage.velocity_uncertainty_mps)
            if not all(env.coverage.contains(obj.footprint(t,0.3+dt*env.coverage.velocity_uncertainty_mps)) for t in (0,min(dt,60))):
                self.missing_clear[identifier]=0;missing=True;continue
            if dt<0.5:
                self.missing_clear[identifier]=0;missing=True;continue
            if distinct:self.missing_clear[identifier]+=1
            if self.missing_clear[identifier]>=2:del self.retained[identifier];del self.missing_clear[identifier]
            else:missing=True
        if lo>task.exit_s_m+0.05 and task.exit_region.body_gap(body)>0:
            if distinct:self.exit_frames+=1
            if self.exit_frames>=2:
                self.session.finish(frame,True,'WHOLE_BODY_ACTUALLY_EXITED');self.phase='EXITED';self.reason='WHOLE_BODY_ACTUALLY_EXITED'
                return False
        else:self.exit_frames=0
        usable=(env.coverage.timing_verified(frame,p.observation_mode) and task.exit_has_space(p,geometry)
                and bool(task.open_windows(p,frame)) and not missing)
        if not inside and signal_stop_requirement(p) is not None:usable=False
        if distinct:
            if usable and not self.was_usable:self.generation+=1
            self.was_usable=usable
            self.clear_frames=self.clear_frames+1 if usable else 0
        if usable and self.condition_at is None:self.condition_at=frame.observed_at_s
        if not usable:
            self.condition_at=None;self.reference=None
            if self.phase!='WAIT_ENTRY':self.session.advance(frame,'WAIT_ENTRY',BehaviorGoal(),force_revision=True)
            self.phase='INSIDE' if inside else 'WAIT_ENTRY'
            return self._wait(p,out,frame,task,geometry,'OCCUPANCY_RULE_WINDOW_OR_EXIT_UNVERIFIED',inside)
        if out.mode=='EMERGENCY_BRAKE':return True
        if not inside and self.clear_frames<2:
            return self._wait(p,out,frame,task,geometry,'FRESH_CLEAR_CONFIRMING',False)
        reply=None
        for value in sorted(p.behavior_feedback,key=lambda v:0 if v.status in ('REJECTED','FAILED') else 1 if (v.producer=='planning')==(self.phase=='EVALUATE') else 2):
            valid,_,_=self.session.validator.check(value,self.session.snapshot(frame),frame,record=False)
            if valid:reply=value;break
        if self.phase=='EXECUTE' and isinstance(reply,PlanningFeedback) and reply.status=='REJECTED':
            self.reference=None;self.session.advance(frame,'WAIT_ENTRY',BehaviorGoal(),force_revision=True);self.phase='WAIT_ENTRY'
            return self._wait(p,out,frame,task,geometry,'CURRENT_PASSAGE_REJECTED',inside)
        goal=BehaviorGoal(speed_cap_mps=min(task.speed_cap_mps,out.target_speed) if out.target_speed>0 else task.speed_cap_mps)
        if self.session.status=='BLOCKED' and self.clear_frames>=2 and self.generation>1:
            if self.session.renew_after_verified_change(frame,'covered-clear-generation:'+str(self.generation)):
                self.session.advance(frame,'EVALUATE',goal,force_revision=True);self.phase='EVALUATE'
        if self.phase in ('WAIT_ENTRY','INSIDE') and self.reference is None:
            self.session.advance(frame,'EVALUATE',goal);self.phase='EVALUATE'
        request=self.session.tick(frame,p.behavior_capabilities,reply)
        if request is None or not request.dispatch_allowed:
            return self._wait(p,out,frame,task,geometry,'JUNCTION_PLANNER_RETRY_OR_PROGRESS_LIMIT',inside)
        feedback=self.session.last_feedback
        if self.phase=='EVALUATE' and isinstance(feedback,PlanningFeedback) and feedback.status=='PLANNED':
            self.reference=feedback.candidate_reference;self.session.advance(frame,'EXECUTE',goal);self.phase='EXECUTE'
            self.authorized_at=frame.observed_at_s
        if self.phase=='EXECUTE':
            if p.ego.speed>0.05 and self.moved_at is None:self.moved_at=frame.observed_at_s
            if isinstance(feedback,PlanningFeedback) and feedback.status=='REJECTED':
                self.reference=None;self.session.advance(frame,'WAIT_ENTRY',BehaviorGoal(),force_revision=True);self.phase='WAIT_ENTRY'
                return self._wait(p,out,frame,task,geometry,'CURRENT_PASSAGE_REJECTED',inside)
            out.mode='KEEP_LANE';out.target_speed=goal.speed_cap_mps;out.stop_distance=-1;out.precision_stop=False
        else:self._wait(p,out,frame,task,geometry,'EVALUATION_NOT_AUTHORITY',inside)
        snapshot=self.session.snapshot(frame)
        out.behavior_request=RuntimeRequest(frame,snapshot.intent_id,'JUNCTION_ENTRY',self.session.stage,self.session.revision,
            self.session.source_frame_id,self.session.goal,'EXECUTE_SELECTED' if self.phase=='EXECUTE' else 'EVALUATE_ONLY',
            self.reference,p.lane.lane_id,issued_at_s=snapshot.issued_at_s)
        self.reason=self.phase
        return True
