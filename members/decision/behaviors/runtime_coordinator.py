"""Production opportunities and one stable evaluation/execution authority.

No trajectory or actuator generation. Existing constraints remain available to
planning; an overtake changes the acting corridor and speed intention only.
"""
import math
import time

from core.behavior_protocol import (RuntimeRequest, PlanningFeedback, BehaviorGoal,
    LightIntent, GoalPose, behavior_frame, FeedbackValidator)
from core.behavior_environment import BehaviorEnvironment, MotionObject, vehicle_footprint
from core.geometry import project_polyline, normalize_angle
from core.target_semantics import mapped_traffic_light
from members.decision.behaviors.lane_change import LaneCandidate, LaneChangePolicy
from members.decision.behaviors.session import BehaviorSession
from members.decision import speed_policy, protocol


class RuntimeCoordinator(object):
    def __init__(self, settings):
        self.settings=settings
        self.reset()

    def reset(self):
        self.session=BehaviorSession(progress_timeout_s=self.settings.trajectory_progress_timeout_s)
        self.phase='IDLE'; self.purpose='OVERTAKE'; self._context=None; self._slow_id=None; self._slow_elapsed=0.0
        self._last_frame=None; self._last_time=None; self._reference=None
        self.origin_lane_id=None; self.selected_lane_id=None; self.target_id=None
        self._settled=0; self.events=[]; self._temporary_safety=False; self.reason='NO_ACTIVE_BEHAVIOR'

    def suspend_for_emergency(self,p):
        if self.phase in ('IDLE','COMPLETE'):return
        frame=behavior_frame(p,time.monotonic())
        if self.session.context is None or self.session.context.key()!=frame.context.key():return
        self._temporary_safety=True
        self.session.tick(frame,p.behavior_capabilities,safety_override=True)

    def _phase(self,frame,phase,reason):
        self.phase,self.reason=phase,reason
        self.events.append(dict(frame_id=frame.frame_id,phase=phase,reason=reason,
                                intent_id=self.session.intent_id,revision=self.session.revision))
        del self.events[:-128]

    def _safe(self,p,frame,target_lane_id):
        env=p.behavior_environment
        lane=env.lane(target_lane_id)
        if lane is None:
            return False
        origin=env.lane(self.origin_lane_id or p.lane.lane_id)
        kinds=('synthetic',) if p.observation_mode=='model' else ('map','verified_fusion')
        if (origin is None or not lane.evidence.verified(frame,allowed_source_kinds=kinds)
                or not origin.evidence.verified(frame,allowed_source_kinds=kinds)):
            return False
        proj=project_polyline(origin.center_line,p.ego.x,p.ego.y)
        if proj is None or not lane.legal(frame,proj['s'],proj['s']+max(22.,p.ego.speed*5.)):
            return False
        width=math.hypot(lane.left_boundary[0][0]-lane.right_boundary[0][0],
                         lane.left_boundary[0][1]-lane.right_boundary[0][1])
        side='left' if lane.side=='left' else 'right'
        objects=[]
        for t in p.targets:
            if mapped_traffic_light(t,p):
                continue
            if t.valid is not True or min(t.length,t.width)<=0:
                return False
            objects.append(MotionObject(t.id,t.x,t.y,t.vx,t.vy,t.length,t.width,
                                        t.heading if t.heading is not None else p.ego.heading))
        body=p.behavior_capabilities
        dims=getattr(body,'vehicle_geometry',None)
        if not isinstance(dims,dict):
            return False
        policy=LaneChangePolicy(dims['front_offset_m'],dims['rear_offset_m'],dims['half_width_m'])
        candidate=LaneCandidate(lane.lane_id,side,lane.center_line,width,lane.marking,True,
            lane.same_direction,lane.evidence,True,objects,lane.left_boundary,lane.right_boundary)
        return policy.opportunity(frame,candidate)[0]

    def _goal(self,p,lane_id,lights=True):
        env=p.behavior_environment
        lane=env.lane(lane_id)
        origin=env.lane(p.lane.lane_id)
        side=lane.side
        if origin is not None:
            proj=project_polyline(origin.center_line,p.ego.x,p.ego.y)
            other=project_polyline(lane.center_line,p.ego.x,p.ego.y)
            if proj is not None and other is not None:
                dx,dy=other['point'][0]-proj['point'][0],other['point'][1]-proj['point'][1]
                side='left' if -math.sin(proj['heading'])*dx+math.cos(proj['heading'])*dy>0 else 'right'
        return BehaviorGoal(target_lane_id=lane_id,speed_cap_mps=speed_policy.desired_speed(p,self.settings),
            light_intent=LightIntent(left_signal=lights and side=='left',right_signal=lights and side=='right'))

    def update(self,p,output,constraints,history,now=None):
        if not p.session_id or not isinstance(p.behavior_environment,BehaviorEnvironment):
            if self.phase not in ('IDLE','COMPLETE'):
                self.reason='ACTIVE_BEHAVIOR_ENVIRONMENT_UNAVAILABLE'
                output.mode='EMERGENCY_BRAKE'; output.target_speed=0.0; output.stop_distance=-1.0
            return
        frame=behavior_frame(p,time.monotonic() if now is None else now)
        if self._context!=frame.context.key():
            if (self._context is not None and self._context[-1] == frame.context.session_id
                    and self.phase not in ('IDLE','COMPLETE')):
                self.reason='CONTEXT_CHANGED_WITHOUT_SESSION_RESET'
                output.mode='EMERGENCY_BRAKE'; output.target_speed=0.0; output.stop_distance=-1.0
                return
            self.reset(); self._context=frame.context.key()
        if self._last_frame is not None and frame.frame_id<self._last_frame:
            self._last_frame,self._last_time=frame.frame_id,frame.observed_at_s
            self._slow_elapsed,self._settled=0.0,0
            self.session.tick(frame,p.behavior_capabilities,safety_override=True)
            self.reason='SAME_SESSION_FRAME_ROLLBACK'
            if self.phase not in ('IDLE','COMPLETE'):
                output.mode='EMERGENCY_BRAKE'; output.target_speed=0.0; output.stop_distance=-1.0
            return
        distinct=self._last_frame is None or frame.frame_id>self._last_frame
        dt=0.0 if self._last_time is None else frame.observed_at_s-self._last_time
        if distinct:
            self._last_frame,self._last_time=frame.frame_id,frame.observed_at_s
        env=p.behavior_environment
        active=self.phase not in ('IDLE','COMPLETE')
        if (not output.valid or output.mode=='EMERGENCY_BRAKE' or not env.current(frame,p.observation_mode)
                or not env.verified_task(frame,p.observation_mode) or not protocol.targets_usable(p) or constraints is None or constraints.first('PROTECT') is not None):
            self._slow_elapsed=0.0
            self.reason='SAFETY_OR_OBSERVATION_SUSPENSION'
            if active:
                self.session.tick(frame,p.behavior_capabilities,safety_override=True)
                output.mode='EMERGENCY_BRAKE'; output.target_speed=0.0; output.stop_distance=-1.0
            return
        if any(c.source!='target' for c in constraints.items if c.kind in ('STOP','EMERGENCY')):
            self._slow_elapsed=0.0
            if active:
                self.session.tick(frame,p.behavior_capabilities,safety_override=True)
                if self.phase.startswith('EXECUTE') or self.phase=='PASS':
                    output.mode='EMERGENCY_BRAKE'; output.target_speed=0.0; output.stop_distance=-1.0
            return
        if self.phase in ('IDLE','COMPLETE'):
            if not env.verified_task(frame,p.observation_mode):
                self.reason='TASK_ROUTE_COMPATIBILITY_UNKNOWN'
                return
            follow=constraints.first('FOLLOW')
            lead=next((t for t in p.targets if follow is not None and t.id==follow.identifier),None)
            free=speed_policy.desired_speed(p,self.settings)
            desired=env.desired_lane_id
            blockage=constraints.blockage_candidates
            if desired is not None and desired!=p.lane.lane_id:
                purpose,identifier='MERGE',None
            elif blockage:
                identifier=min(blockage,key=lambda v:(v[1],v[0]))[0]
                lead=next((t for t in p.targets if t.id==identifier),None)
                purpose='AVOID'
            elif (lead is not None and math.hypot(lead.vx,lead.vy)+self.settings.slow_lead_min_advantage_mps<free
                  and output.stop_distance<0):
                purpose,identifier='OVERTAKE',lead.id
            else:
                self._slow_elapsed=0.0;self._slow_id=None
                return
            if purpose=='OVERTAKE':
                if self._slow_id!=lead.id or dt<0 or dt>0.5:
                    self._slow_id=lead.id;self._slow_elapsed=0.0
                elif distinct:self._slow_elapsed+=dt
                if self._slow_elapsed<self.settings.slow_lead_confirmation_s:return
            candidates=[v for v in env.lanes if v.lane_id!=p.lane.lane_id and self._safe(p,frame,v.lane_id)
                        and (purpose!='MERGE' or v.lane_id==desired)]
            if not candidates:
                self.reason='NO_LEGAL_SAFE_NEIGHBOR_WITH_COVERAGE'
                return
            self.origin_lane_id=p.lane.lane_id
            self.selected_lane_id=min(candidates,key=lambda v:(v.lane_id!=desired,v.lane_id)).lane_id
            self.target_id,self.purpose=identifier,purpose
            key=purpose.lower()+':'+str(identifier if identifier is not None else desired)
            self.session.start(frame,key,purpose,'PREPARE_OUT',self._goal(p,self.selected_lane_id),
                ('LANE_CHANGE','PATH','LIGHTS','FEEDBACK'))
            self._phase(frame,'PREPARE_OUT','VERIFIED_'+purpose+'_OPPORTUNITY')
        lane_id=self.selected_lane_id if 'RETURN' not in self.phase else self.origin_lane_id
        if env.lane(lane_id) is None:
            self.session.tick(frame,p.behavior_capabilities,safety_override=True)
            output.mode='EMERGENCY_BRAKE'; output.target_speed=0.0; output.stop_distance=-1.0
            return
        dims=getattr(p.behavior_capabilities,'vehicle_geometry',None)
        source_id=self.selected_lane_id if 'RETURN' in self.phase else self.origin_lane_id
        source_lane=env.lane(source_id)
        body=vehicle_footprint(GoalPose(p.ego.x,p.ego.y,p.ego.heading),dims['front_offset_m'],dims['rear_offset_m'],dims['half_width_m']) if dims else None
        before_cross=source_lane is not None and body is not None and source_lane.contains(body)
        safe=self._safe(p,frame,lane_id)
        lead=next((t for t in p.targets if t.id==self.target_id),None)
        purpose_exists=(self.purpose!='OVERTAKE' or lead is not None and math.hypot(lead.vx,lead.vy)+self.settings.slow_lead_min_advantage_mps
                        <speed_policy.desired_speed(p,self.settings))
        if ('RETURN' not in self.phase and self.phase!='PASS' and before_cross
                and (not safe or not purpose_exists)):
            self.session.finish(frame,False,'OPPORTUNITY_WITHDRAWN_BEFORE_CROSSING')
            self._phase(frame,'IDLE','OPPORTUNITY_WITHDRAWN_BEFORE_CROSSING')
            self._slow_elapsed=0.0
            return
        if self.phase.startswith('EXECUTE') and not safe:
            self.session.block(frame,'EXECUTION_GAP_CHANGED_RECOVERY_REQUIRED')
            self.reason='EXECUTION_GAP_CHANGED_RECOVERY_REQUIRED'
            output.mode='EMERGENCY_BRAKE';output.target_speed=0.0;output.stop_distance=-1.0
            return
        if (self.session.status=='BLOCKED' and self.session.reason_code=='EXECUTION_GAP_CHANGED_RECOVERY_REQUIRED'
                and safe and distinct and self.session.retry_after_evidence(frame,'FRESH_GAP_RECOVERY_REEVALUATE')):
            phase='EVALUATE_RETURN' if 'RETURN' in self.phase else 'EVALUATE_OUT'
            self._reference=None
            self.session.advance(frame,phase,self._goal(p,lane_id))
            self._phase(frame,phase,'FRESH_GAP_RECOVERY_REEVALUATE')
        if self._temporary_safety:
            phase='PREPARE_RETURN' if 'RETURN' in self.phase else 'PREPARE_OUT'
            self._reference=None;self._settled=0
            self.session.advance(frame,phase,self._goal(p,lane_id,lights=True),force_revision=True,
                reason='TEMPORARY_EMERGENCY_ENDED_NEW_LANE_CANDIDATE_REQUIRED')
            self._phase(frame,phase,'TEMPORARY_EMERGENCY_ENDED_NEW_LANE_CANDIDATE_REQUIRED')
            self._temporary_safety=False
        goal=self._goal(p,lane_id,lights=self.phase!='PASS')
        self.session.start(frame,self.session.key,self.purpose,self.session.stage,goal,
                           ('LANE_CHANGE','PATH','LIGHTS','FEEDBACK'))
        reply=None
        replies=sorted(p.behavior_feedback, key=lambda v: (
            0 if getattr(v,'status',None) in ('REJECTED','FAILED') else
            1 if (getattr(v,'producer',None)=='planning') == self.phase.startswith('EVALUATE') else 2))
        for value in replies:
            valid,_,_=self.session.validator.check(value,self.session.snapshot(frame),frame,record=False)
            if valid:
                reply=value
                break
        request=self.session.tick(frame,p.behavior_capabilities,reply)
        if request is None or not request.dispatch_allowed:
            self.reason=request.reason_code if request is not None else 'NO_REQUEST'
            if self.phase.startswith('EXECUTE') or self.phase=='PASS':
                output.mode='EMERGENCY_BRAKE'; output.target_speed=0.0; output.stop_distance=-1.0
            return
        reply=self.session.last_feedback
        if self.phase.startswith('PREPARE'):
            if (reply is not None and reply.producer in ('control','runtime') and reply.lights_confirmed
                    and reply.lights_duration_s>=self.settings.indicator_lead_s and self._safe(p,frame,lane_id)):
                phase='EVALUATE_RETURN' if 'RETURN' in self.phase else 'EVALUATE_OUT'
                self.session.advance(frame,phase,goal)
                self._phase(frame,phase,'INDICATOR_AND_GAP_CONFIRMED')
        elif self.phase.startswith('EVALUATE'):
            if (isinstance(reply,PlanningFeedback) and reply.status=='PLANNED'
                    and reply.candidate_reference is not None and self._safe(p,frame,lane_id)):
                self._reference=reply.candidate_reference
                phase='EXECUTE_RETURN' if 'RETURN' in self.phase else 'EXECUTE_OUT'
                self.session.advance(frame,phase,goal)
                self._phase(frame,phase,'MATCHING_CANDIDATE_ADOPTED')
        elif self.phase.startswith('EXECUTE'):
            dims=getattr(p.behavior_capabilities,'vehicle_geometry',None)
            pose=GoalPose(p.ego.x,p.ego.y,p.ego.heading)
            lane=env.lane(lane_id)
            projection=project_polyline(lane.center_line,pose.x,pose.y)
            inside=(dims is not None and lane.contains(vehicle_footprint(pose,dims['front_offset_m'],dims['rear_offset_m'],dims['half_width_m'])))
            executed=(reply is not None and reply.producer in ('control','runtime')
                      and reply.candidate_id==self._reference.candidate_id)
            if distinct and executed and inside and projection is not None and abs(normalize_angle(p.ego.heading-projection['heading']))<0.1:
                self._settled+=1
            else:
                self._settled=0
            if self._settled>=self.settings.maneuver_settle_frames:
                if 'RETURN' in self.phase:
                    self.session.finish(frame,True,'OVERTAKE_AND_TASK_ROUTE_RETURN_CONFIRMED')
                    self._phase(frame,'COMPLETE','WHOLE_BODY_SETTLED_ON_TASK_ROUTE')
                    return
                if self.purpose=='MERGE':
                    self.session.finish(frame,True,'TASK_REQUIRED_LANE_ACTUALLY_REACHED')
                    self._phase(frame,'COMPLETE','TASK_REQUIRED_LANE_ACTUALLY_REACHED')
                    return
                self._phase(frame,'PASS','WHOLE_BODY_SETTLED_IN_PASSING_CORRIDOR')
        elif self.phase=='PASS':
            lead=next((t for t in p.targets if t.id==self.target_id),None)
            origin=env.lane(self.origin_lane_id)
            a=project_polyline(origin.center_line,p.ego.x,p.ego.y) if origin else None
            b=project_polyline(origin.center_line,lead.x,lead.y) if origin and lead else None
            dims=getattr(p.behavior_capabilities,'vehicle_geometry',None)
            passed=(a is not None and b is not None and dims is not None
                    and a['s']-dims['rear_offset_m']>b['s']+lead.length/2+(self.settings.min_gap if self.purpose=='OVERTAKE' else self.settings.obstacle_stop_margin))
            if passed:
                if env.return_lane_id==self.origin_lane_id:
                    self._settled=0
                    self.session.advance(frame,'PREPARE_RETURN',self._goal(p,self.origin_lane_id))
                    self._phase(frame,'PREPARE_RETURN','WHOLE_BODY_PASSED_LEAD_RETURN_REQUIRED')
                else:
                    self.session.finish(frame,True,'WHOLE_BODY_OVERTAKE_CONFIRMED')
                    self._phase(frame,'COMPLETE','WHOLE_BODY_PASSED_LEAD')
                    return
        stage=self.session.stage
        authorization=('PREPARE' if self.phase.startswith('PREPARE') else
                       'EVALUATE_ONLY' if self.phase.startswith('EVALUATE') else 'EXECUTE_SELECTED')
        snapshot=self.session.snapshot(frame)
        output.behavior_request=RuntimeRequest(frame,snapshot.intent_id,self.purpose,stage,
            snapshot.revision,snapshot.source_frame_id,self.session.goal,authorization,self._reference if authorization=='EXECUTE_SELECTED' else None,
            self.origin_lane_id if 'RETURN' not in self.phase else self.selected_lane_id,
            [c.source+':'+str(c.identifier) for c in constraints.items],self.target_id,
            issued_at_s=snapshot.issued_at_s,follow_clearance_m=self.settings.min_gap,
            static_clearance_m=self.settings.obstacle_stop_margin)
