"""Actual parking selection and planner-generated segment authority."""
import math
import time

from core.behavior_protocol import (RuntimeRequest,BehaviorGoal,PlanningFeedback,GoalPose,behavior_frame)
from core.parking_protocol import ParkingTask
from core.behavior_environment import vehicle_footprint
from members.decision.behaviors.session import BehaviorSession


class ParkingRuntime(object):
    def __init__(self):self.reset()
    def reset(self):
        self.session=BehaviorSession(progress_timeout_s=30.0)
        self.context=None;self.phase='IDLE';self.bay=None;self.goal=None;self.reference=None
        self.directions=();self.ends=();self.index=0;self.task=None;self.last_frame=None;self.needs_replan=False;self.resume_exit=False;self.emergency_epoch=0;self.reason='NO_PARKING_TASK'

    def active(self):return self.phase not in ('IDLE','COMPLETE')

    def suspend_for_emergency(self,p):
        if not self.active():return
        if not self.needs_replan:
            self.resume_exit='EXIT' in self.phase;self.emergency_epoch+=1
        self.needs_replan=True
        self.session.tick(behavior_frame(p,time.monotonic()),p.behavior_capabilities,safety_override=True)

    def _goal(self,pose,direction=1,dwell=0):
        return BehaviorGoal(parking_space_id=self.bay.identity,goal_pose=pose,motion_direction=direction,
            speed_cap_mps=0.7,precision_stop=True,minimum_standstill_duration_s=dwell,
            parking_brake_at_stop=dwell>0,stop_obligation_id='parking:'+self.task.identity if dwell else None)

    def update(self,p,out):
        env=p.behavior_environment
        task=env.parking_task if env is not None else None
        if not isinstance(task,ParkingTask):return False
        f=behavior_frame(p,time.monotonic())
        if self.context!=f.context.key():
            if self.context is not None and self.context[-1]==f.context.session_id and self.active():
                out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1
                self.reason='PARKING_CONTEXT_CHANGED_WITHOUT_SESSION_RESET';return True
            self.reset();self.context=f.context.key()
        if self.last_frame is not None and f.frame_id<self.last_frame and self.active() and self.bay is not None:
            exiting='EXIT' in self.phase
            self.reference=None;self.directions=();self.ends=();self.index=0
            self.phase='EVALUATE_EXIT' if exiting else 'EVALUATE_PARK'
            self.session.advance(f,self.phase,self._goal(task.exit_goal if exiting else self.goal,1 if exiting else -1),
                force_revision=True,reason='PARKING_FRAME_ROLLBACK_NEW_CANDIDATE_REQUIRED')
        self.last_frame=f.frame_id
        if self.phase=='COMPLETE':
            out.valid=True;out.mode='STOP';out.target_speed=0.0;out.stop_distance=0.0
            out.behavior_request=RuntimeRequest(f,self.session.intent_id,'PARK','COMPLETE',self.session.revision,
                self.session.source_frame_id,self._goal(self.ends[-1],self.directions[-1]),'HOLD',
                origin_lane_id=p.lane.lane_id,issued_at_s=f.observed_at_s)
            return True
        if self.task is not None and (self.task.identity!=task.identity or self.task.minimum_dwell_s!=task.minimum_dwell_s
                or self.task.exit_goal.to_dict()!=task.exit_goal.to_dict()):
            out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1
            self.reason='PARKING_TASK_CHANGED_REVALIDATION_REQUIRED';return True
        self.task=task
        dims=getattr(p.behavior_capabilities,'vehicle_geometry',None)
        usable=(dims is not None and task.verified(f,p.observation_mode) and env.current(f,p.observation_mode))
        if not usable:
            out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1
            self.reason='PARKING_OBSERVATION_OR_FREE_SPACE_UNKNOWN'
            return True
        if self.needs_replan and self.bay is not None:
            if self.session.status=='BLOCKED':self.session.renew_after_verified_change(f,'emergency-epoch:'+str(self.emergency_epoch))
            self.reference=None;self.directions=();self.ends=();self.index=0
            self.phase='EVALUATE_EXIT' if self.resume_exit else 'EVALUATE_PARK'
            self.session.advance(f,self.phase,self._goal(task.exit_goal if self.resume_exit else self.goal,1 if self.resume_exit else -1),
                force_revision=True,reason='TEMPORARY_EMERGENCY_ENDED_NEW_PARKING_PATH_REQUIRED')
            self.needs_replan=False
        if self.bay is None:
            choices=[]
            for bay in task.bays:
                if bay.occupancy!='empty' or not bay.evidence.verified(f):continue
                goal=bay.rear_axle_goal(dims)
                body=vehicle_footprint(goal,dims['front_offset_m'],dims['rear_offset_m'],dims['half_width_m'])
                if bay.region.body_gap(body)>0.03 and task.body_clear(goal,dims):
                    choices.append((math.hypot(goal.x-p.ego.x,goal.y-p.ego.y),bay.identity,bay,goal))
            if not choices:
                out.mode='STOP';out.target_speed=0;out.stop_distance=0
                self.reason='NO_OBSERVED_FITTING_EMPTY_BAY';return True
            _,_,self.bay,self.goal=min(choices,key=lambda v:v[:2])
            self.session.start(f,'parking:'+task.identity,'PARK','EVALUATE_PARK',self._goal(self.goal,-1),('PARK','PATH','FEEDBACK'))
            self.phase='EVALUATE_PARK'
        current=next((b for b in task.bays if b.identity==self.bay.identity),None)
        if (current is None or not current.evidence.verified(f)
                or current.occupancy not in ('empty','self')
                or current.occupancy=='self' and (task.ego_target_id is None or not current.region.intersects(
                    vehicle_footprint(GoalPose(p.ego.x,p.ego.y,p.ego.heading),dims['front_offset_m'],dims['rear_offset_m'],dims['half_width_m'])))
                or current.region.points!=self.bay.region.points
                or current.body_heading_rad!=self.bay.body_heading_rad):
            self.session.block(f,'PARKING_OCCUPANCY_UNKNOWN_OR_OTHER')
        reply=None
        values=sorted(p.behavior_feedback,key=lambda v:(
            0 if getattr(v,'status',None) in ('REJECTED','FAILED') else
            1 if (getattr(v,'producer',None)=='planning')==self.phase.startswith('EVALUATE') else 2))
        for value in values:
            valid,_,_=self.session.validator.check(value,self.session.snapshot(f),f,record=False)
            if valid:reply=value;break
        request=self.session.tick(f,p.behavior_capabilities,reply)
        if request is None or not request.dispatch_allowed:
            out.mode='EMERGENCY_BRAKE';out.target_speed=0;out.stop_distance=-1
            return True
        reply=self.session.last_feedback
        if self.phase.startswith('EVALUATE'):
            if isinstance(reply,PlanningFeedback) and reply.status=='PLANNED' and reply.segment_end_poses:
                self.reference=reply.candidate_reference
                self.directions,self.ends=reply.segment_directions,reply.segment_end_poses
                self.index=0
                self.phase='EXECUTE_EXIT' if self.phase=='EVALUATE_EXIT' else 'EXECUTE_PARK'
                duration=task.minimum_dwell_s if self.phase=='EXECUTE_PARK' and len(self.ends)==1 else 0
                self.session.advance(f,self.phase+':0',self._goal(self.ends[0],self.directions[0],duration))
        elif self.phase.startswith('EXECUTE'):
            actual=(reply is not None and reply.producer in ('control','runtime') and reply.actual_pose is not None
                    and reply.actual_standstill_confirmed and reply.goal_pose_arrived
                    and reply.candidate_id==self.reference.candidate_id)
            if actual:
                endpoint=self.ends[self.index]
                actual=math.hypot(reply.actual_pose.x-endpoint.x,reply.actual_pose.y-endpoint.y)<=0.15
            if actual and self.index+1<len(self.ends):
                self.index+=1
                duration=task.minimum_dwell_s if self.phase=='EXECUTE_PARK' and self.index==len(self.ends)-1 else 0
                self.session.advance(f,self.phase+':'+str(self.index),self._goal(self.ends[self.index],self.directions[self.index],duration))
            elif actual and self.phase=='EXECUTE_PARK':
                body=vehicle_footprint(reply.actual_pose,dims['front_offset_m'],dims['rear_offset_m'],dims['half_width_m'])
                if (current.region.body_gap(body)>0 and reply.hold_completed
                        and reply.standstill_duration_s>=task.minimum_dwell_s):
                    self.phase='EVALUATE_EXIT';self.reference=None
                    self.session.advance(f,'EVALUATE_EXIT',self._goal(task.exit_goal,1),reason='ACTUAL_PARK_DWELL_COMPLETED')
            elif actual and self.phase=='EXECUTE_EXIT':
                self.session.finish(f,True,'PARK_AND_TASK_EXIT_ACTUALLY_COMPLETED');self.phase='COMPLETE'
                return self.update(p,out)
        snapshot=self.session.snapshot(f)
        authorization='EVALUATE_ONLY' if self.phase.startswith('EVALUATE') else 'EXECUTE_SELECTED'
        out.mode='KEEP_LANE';out.target_speed=0.7;out.target_lane_id=p.lane.lane_id
        out.stop_distance=-1;out.precision_stop=False;out.valid=True
        out.behavior_request=RuntimeRequest(f,snapshot.intent_id,'PARK',self.session.stage,self.session.revision,
            self.session.source_frame_id,self.session.goal,authorization,self.reference,
            p.lane.lane_id,issued_at_s=snapshot.issued_at_s,segment_index=self.index if authorization=='EXECUTE_SELECTED' else None)
        self.reason=self.phase
        return True
