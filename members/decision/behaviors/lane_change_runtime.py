"""Source-bound overtake, static avoidance and task merge at the fixed entry.

The original task/model arrives through a captain-installed provider. No scene
number, SDK steering value, diagnostic snapshot or inferred rear visibility
can authorize a maneuver. LaneChangePolicy owns each crossing lifecycle.
"""
import copy
import math

from core.behavior_channel import read_channel
from core.behavior_contract import BehaviorFrame, GoalPose, finite, require
from core.behavior_evidence import Evidence
from core.geometry import project_polyline, normalize_angle, projection_within_polyline
from core.interfaces import DecisionMode
from core.maneuver_facts import ManeuverFacts
from core.traffic_quality import signal_stop_requirement
from members.decision import speed_policy
from members.decision.behaviors.environment import lane_candidates
from members.decision.behaviors.lane_change import LaneChangePolicy


class LaneChangeDecisionInputs(object):
    """Explicit task/model assertion and provisional decision limits.

    return_lane_id names the original task corridor; a neighbor's presence does
    not establish that task. allow_command_lights is an explicit trial policy:
    it accepts sent-command evidence, never claims the lamps were observed.
    MERGE requires a named target and no return; AVOID still needs an actual
    static target constraint. These task fields do not create Sensor facts.
    """
    def __init__(self, model_evidence, route_evidence, model_id, mission_id,
                 return_lane_id, vehicle_extents, speed_cap_mps,
                 confirmation_s=2., min_advantage_mps=1.5, transition_distance_m=25.,
                 min_front_gap_m=10.5, min_rear_gap_m=10.5, headway_s=1.2,
                 maneuver_time_s=4., acceleration_uncertainty_mps2=2.,
                 indicator_lead_s=1., settle_frames=2, allow_command_lights=False,
                 maneuver='OVERTAKE', target_lane_id=None, return_required=True):
        require(isinstance(model_evidence, Evidence) and isinstance(route_evidence, Evidence),
                'LANE_CHANGE_MODEL_OR_TASK_EVIDENCE_UNAVAILABLE')
        require(all(isinstance(v, str) and v for v in (model_id, mission_id, return_lane_id)),
                'LANE_CHANGE_TASK_IDENTITY_UNAVAILABLE')
        require(isinstance(vehicle_extents, (tuple, list)) and len(vehicle_extents)==3
                and all(finite(v) and v>0 for v in vehicle_extents), 'LANE_CHANGE_EXTENTS_UNAVAILABLE')
        values=(speed_cap_mps, confirmation_s, min_advantage_mps, transition_distance_m,
                min_front_gap_m, min_rear_gap_m, headway_s, maneuver_time_s,
                acceleration_uncertainty_mps2, indicator_lead_s)
        require(all(finite(v) and v>0 for v in values) and type(settle_frames) is int
                and settle_frames>0 and type(allow_command_lights) is bool,
                'LANE_CHANGE_DECISION_LIMITS_UNAVAILABLE')
        require(maneuver in ('OVERTAKE','AVOID','MERGE') and type(return_required) is bool
                and (target_lane_id is None or isinstance(target_lane_id,str) and target_lane_id)
                and (maneuver!='MERGE' or target_lane_id is not None and not return_required),
                'LANE_CHANGE_TASK_ACTION_OR_TARGET_INVALID')
        self.model_evidence,self.route_evidence=copy.deepcopy((model_evidence,route_evidence))
        self.model_id,self.mission_id,self.return_lane_id=model_id,mission_id,return_lane_id
        self.extents,self.speed_cap=tuple(vehicle_extents),speed_cap_mps
        self.confirmation,self.advantage,self.distance=confirmation_s,min_advantage_mps,transition_distance_m
        self.front_gap,self.rear_gap,self.headway=min_front_gap_m,min_rear_gap_m,headway_s
        self.duration,self.uncertainty=maneuver_time_s,acceleration_uncertainty_mps2
        self.indicator_lead,self.settle_frames=indicator_lead_s,settle_frames
        self.allow_command_lights=allow_command_lights
        self.maneuver,self.target_lane_id,self.return_required=maneuver,target_lane_id,return_required

    def snapshot(self):
        return copy.deepcopy(self)

    def binding(self):
        return (self.model_id,self.mission_id,self.return_lane_id,self.extents,self.speed_cap,
                self.confirmation,self.advantage,self.distance,self.front_gap,self.rear_gap,
                self.headway,self.duration,self.uncertainty,self.indicator_lead,self.settle_frames,
                self.allow_command_lights,self.maneuver,self.target_lane_id,self.return_required,
                self.model_evidence.source_kind,self.model_evidence.source,
                self.route_evidence.source_kind,self.route_evidence.source)


def _pose_at(line, station):
    require(finite(station) and station>=0, 'LANE_CHANGE_GOAL_STATION_INVALID')
    for a,b in zip(line,line[1:]):
        length=math.hypot(b[0]-a[0],b[1]-a[1])
        if length<=1e-9:
            continue
        if station<=length:
            ratio=station/length
            return GoalPose(a[0]+ratio*(b[0]-a[0]),a[1]+ratio*(b[1]-a[1]),
                            math.atan2(b[1]-a[1],b[0]-a[0]))
        station-=length
    raise ValueError('LANE_CHANGE_FIXED_GOAL_OUTSIDE_OBSERVED_GEOMETRY')


class LaneChangeRuntime(object):
    def __init__(self, provider, session_factory, clock, settings):
        require(provider is None or callable(provider),'LANE_CHANGE_PROVIDER_INVALID')
        self.provider,self.session_factory,self.clock,self.settings=provider,session_factory,clock,settings
        self.session=session_factory()
        self.policy,self.context,self.binding=None,None,None
        self.phase,self.reason='IDLE','NOT_CONFIGURED'
        self.origin_lane,self.target_lane,self.lead_id,self.goal=None,None,None,None
        self.lead_source=None
        self.slow_id,self.slow_elapsed,self.last_frame,self.last_time=None,0.,None,None
        self.generation=0
        self.last_observation=None

    def pending(self):
        return self.phase not in ('IDLE','COMPLETE')

    def suspend(self):
        if self.policy is not None and self.policy.session.intent_id is not None and self.last_observation is not None:
            frame=copy.copy(self.last_observation); frame.usable=False
            self.policy.session.tick(frame,safety_override=True)

    def hold(self,p,output,reason):
        self.reason=reason
        output.mode=DecisionMode.EMERGENCY_BRAKE if p.ego.speed>self.settings.standstill_speed else DecisionMode.STOP
        output.target_speed,output.stop_distance=0.,0.
        output.precision_stop=True
        output.behavior_request,output.behavior_active_identity=None,None
        output.reason=reason

    def observed_avoidance_handoff(self,p):
        """Retire one legacy hazard latch after a source-bound route handoff.

        This is not arrival or clearance of the whole car. The current P02/P05
        checks still own the crossing path and every observed object. Missing
        observations, label-only switches and new tasks cannot use this gate.
        """
        try:
            require(self.phase=='OUT' and self.policy is not None and self.binding is not None,
                    'AVOID_HANDOFF_NOT_ACTIVE')
            session=self.policy.session; selected=self.policy.selected
            require(session.maneuver=='AVOID' and session.stage in ('EXECUTE','SETTLE')
                    and session.status in ('ACCEPTED','EXECUTING')
                    and isinstance(session.goal.candidate_id,str) and session.goal.candidate_id
                    and selected is not None and selected.source_lane_id==self.origin_lane
                    and selected.lane_id==self.target_lane and p.lane.lane_id==self.target_lane,
                    'AVOID_HANDOFF_NOT_AUTHORIZED')
            now=self.clock(); context,caps,unused=read_channel(p,now)
            frame=BehaviorFrame(context,p.frame_id,now,p.valid_until,p.ego.x,p.ego.y,
                                p.ego.heading,p.ego.speed,usable=p.valid)
            require(context.key()==self.context==session.context.key()
                    and caps.allows(frame,session.required) and session.snapshot(frame).dispatch_allowed,
                    'AVOID_HANDOFF_CONTEXT_OR_CAPABILITY_CHANGED')
            value=self.provider(p,context)
            require(isinstance(value,LaneChangeDecisionInputs) and value.binding()==self.binding
                    and value.maneuver=='AVOID' and value.return_lane_id==self.origin_lane,
                    'AVOID_HANDOFF_TASK_OR_MODEL_CHANGED')
            for evidence in (value.model_evidence,value.route_evidence):
                require(evidence.frame_id==p.frame_id and evidence.valid_until_s<=p.valid_until
                        and evidence.verified(frame,allowed_source_kinds=('task','verified_fusion')),
                        'AVOID_HANDOFF_SOURCE_EXPIRED')
            previous=self.last_observation
            dt=0. if previous is None else now-previous.observed_at_s
            require(previous is not None and p.frame_id>previous.frame_id
                    and 0<dt<=self.settings.behavior_observation_gap_s
                    and math.hypot(p.ego.x-previous.ego_x,p.ego.y-previous.ego_y)
                        <=max(value.speed_cap,previous.ego_speed,p.ego.speed)*dt+1e-6,
                    'AVOID_HANDOFF_POSE_DISCONTINUITY')
            facts=ManeuverFacts(p,self.clock)
            require(p.target_source==self.lead_source
                    and facts.map_digest()==selected.map_digest,'AVOID_HANDOFF_SENSOR_OR_MAP_CHANGED')
            objects=facts.objects()
            retained=next((v for v in objects if v['id']==self.lead_id),None)
            require(retained is not None and math.hypot(retained['vx'],retained['vy'])
                    <=self.settings.static_speed_threshold,'AVOID_HANDOFF_HAZARD_UNOBSERVED')
            target=facts.road(self.target_lane)
            projection=project_polyline(target['center_line'],p.ego.x,p.ego.y)
            require(projection_within_polyline(projection,len(target['center_line']))
                    and p.lane.lane_width_valid and finite(p.lane.lane_width)
                    and projection['distance']<p.lane.lane_width/2,
                    'AVOID_HANDOFF_LABEL_WITHOUT_TARGET_POSE')
            facts.check_current(True)
            require(self.clock()<min(p.valid_until,value.model_evidence.valid_until_s,
                    value.route_evidence.valid_until_s),'AVOID_HANDOFF_SOURCE_EXPIRED')
            return True
        except Exception:
            return False

    def _policy(self,value,reuse=True):
        return LaneChangePolicy(*value.extents,min_front_gap_m=value.front_gap,
            min_rear_gap_m=value.rear_gap,time_headway_s=value.headway,
            maneuver_time_s=value.duration,acceleration_uncertainty_mps2=value.uncertainty,
            indicator_lead_s=value.indicator_lead,settle_frames=value.settle_frames,
            session=self.session if reuse else self.session_factory(),allow_command_lights=value.allow_command_lights)

    def _start(self,p,frame,value,target_lane,phase):
        candidates,unused=lane_candidates(p,frame,self.clock)
        probe=self._policy(value,False)
        usable=[c for c in candidates if (target_lane is None or c.lane_id==target_lane)
                and probe.opportunity(frame,c)[0]]
        if not usable:
            self.reason='NO_LEGAL_NEIGHBOR_WITH_FRONT_SIDE_REAR_COVERAGE'
            return False
        selected=min(usable,key=lambda c:c.lane_id)
        at=project_polyline(selected.center_line,p.ego.x,p.ego.y)
        require(at is not None,'LANE_CHANGE_TARGET_PROJECTION_UNAVAILABLE')
        self.goal=_pose_at(selected.center_line,at['s']+value.distance)
        self.policy,self.target_lane=self._policy(value),selected.lane_id
        self.generation+=1
        self.phase=phase
        return True

    def apply(self,p,output,constraints,priority=False):
        if self.provider is None:
            return
        was_active=self.pending()
        try:
            now=self.clock()
            context,caps,replies=read_channel(p,now)
            frame=BehaviorFrame(context,p.frame_id,now,p.valid_until,p.ego.x,p.ego.y,
                                p.ego.heading,p.ego.speed,usable=p.valid)
            self.last_observation=frame
            if self.context is not None and self.context!=context.key():
                require(not was_active,'LANE_CHANGE_ACTIVE_CONTEXT_CHANGED')
                self.phase,self.binding='IDLE',None
            self.context=context.key()
            value=self.provider(p,context)
            if value is None:
                require(not was_active,'LANE_CHANGE_ACTIVE_SOURCE_WITHDRAWN')
                self.reason='NO_LANE_CHANGE_TASK'; return
            require(isinstance(value,LaneChangeDecisionInputs),'LANE_CHANGE_DECISION_INPUTS_UNAVAILABLE')
            value=value.snapshot()
            for evidence in (value.model_evidence,value.route_evidence):
                require(evidence.frame_id==p.frame_id and evidence.valid_until_s<=p.valid_until
                        and evidence.verified(frame,allowed_source_kinds=('task','verified_fusion')),
                        'LANE_CHANGE_MODEL_OR_TASK_EXPIRED')
            require(self.binding is None or self.binding==value.binding(),'LANE_CHANGE_ACTIVE_TASK_OR_MODEL_CHANGED')
            if self.last_frame is not None:
                require(p.frame_id>=self.last_frame and now>=self.last_time,'LANE_CHANGE_FRAME_OR_CLOCK_ROLLBACK')
            distinct=self.last_frame is None or p.frame_id>self.last_frame
            dt=0. if self.last_time is None else now-self.last_time
            if distinct:
                self.last_frame,self.last_time=p.frame_id,now
            if (priority or signal_stop_requirement(p) is not None or not output.valid
                    or output.mode==DecisionMode.EMERGENCY_BRAKE or constraints is None
                    or constraints.first('PROTECT') is not None
                    or any(c.kind in ('STOP','EMERGENCY') and c.source not in ('target','target_history')
                           for c in constraints.items)):
                self.slow_elapsed=0.
                if was_active:
                    if self.policy is not None:
                        self.policy.evaluate(frame,self.policy.session.key,value.maneuver,[],caps,safety_override=True)
                    if not priority:
                        self.hold(p,output,value.maneuver+'_SAFETY_OR_RULE_SUSPENDED')
                return
            facts=ManeuverFacts(p,self.clock)
            objects=facts.objects()
            if self.phase in ('IDLE','COMPLETE'):
                # A one-way task remains completed; a fresh task requires a
                # captain reset/provider identity rather than another crossing.
                if self.phase=='COMPLETE' and not value.return_required:
                    return
                require(value.return_lane_id==p.lane.lane_id,
                        'LANE_CHANGE_ORIGINAL_TASK_CORRIDOR_MISMATCH')
                lead=None
                if value.maneuver=='OVERTAKE':
                    follow=constraints.first('FOLLOW')
                    lead=next((v for v in objects if follow is not None and v['id']==follow.identifier),None)
                    free=min(value.speed_cap,speed_policy.desired_speed(p,self.settings))
                    if lead is None or math.hypot(lead['vx'],lead['vy'])+value.advantage>=free:
                        self.slow_id,self.slow_elapsed=None,0.; return
                elif value.maneuver=='AVOID':
                    # The legacy decision's structured target constraint must
                    # refer to an object present in this formal Sensor packet.
                    # A reason string or disappeared target is not authority.
                    blocking=sorted((c for c in constraints.items
                        if c.kind in ('STOP','BLOCKED_ROAD','FOLLOW')
                        and c.source in ('target','target_history') and c.valid_until>now),
                        key=lambda c:c.key())
                    lead=next((v for c in blocking for v in objects if v['id']==c.identifier
                        and math.hypot(v['vx'],v['vy'])<=self.settings.static_speed_threshold),None)
                    if lead is None:
                        self.slow_id,self.slow_elapsed=None,0.
                        self.reason='STATIC_BLOCKING_TARGET_UNOBSERVED'; return
                # MERGE has explicit task authority and a target; it does not
                # pretend that an absent lead is a slow-car observation.
                if lead is not None:
                    if self.slow_id!=lead['id'] or dt>self.settings.behavior_observation_gap_s:
                        self.slow_id,self.slow_elapsed=lead['id'],0.
                    elif distinct:
                        self.slow_elapsed+=dt
                    if self.slow_elapsed<value.confirmation:
                        self.reason=('STATIC_BLOCKAGE_CONFIRMING' if value.maneuver=='AVOID'
                                     else 'SLOW_LEAD_CONFIRMING'); return
                if not self._start(p,frame,value,value.target_lane_id,'OUT'):
                    return
                self.binding=value.binding()
                self.origin_lane,self.lead_id=p.lane.lane_id,lead['id'] if lead is not None else None
                self.lead_source=p.target_source if lead is not None else None
            elif self.phase=='PASS':
                require(p.lane.lane_id==self.target_lane,'OVERTAKE_PASS_CORRIDOR_CHANGED')
                lead=next((v for v in objects if v['id']==self.lead_id),None)
                require(lead is not None,'OVERTAKE_LEAD_IDENTITY_UNOBSERVED')
                road=facts.road(self.origin_lane,require_coverage=False)
                a=project_polyline(road['center_line'],p.ego.x,p.ego.y)
                age=now-facts.env.dynamic_observed_at_s
                b=project_polyline(road['center_line'],lead['x']+lead['vx']*age,lead['y']+lead['vy']*age)
                require(a is not None and b is not None,'OVERTAKE_PASS_PROJECTION_UNAVAILABLE')
                extent=.5*math.hypot(lead['length'],lead['width'])
                angle=None if lead['heading'] is None else normalize_angle(lead['heading']-b['heading'])
                if angle is not None:
                    extent=.5*(lead['length']*abs(math.cos(angle))+lead['width']*abs(math.sin(angle)))
                passed=a['s']-value.extents[1]>b['s']+extent+value.front_gap
                if not passed:
                    self.reason='PASSING_ACTUAL_LEAD_BODY'; return
                if not value.return_required:
                    self.phase='COMPLETE'; self.reason=value.maneuver+'_ACTUAL_TARGET_PASSED'
                    return
                if not self._start(p,frame,value,self.origin_lane,'RETURN'):
                    self.reason='RETURN_GAP_WAIT_IN_VERIFIED_PASS_LANE'; return
            candidates,unused=lane_candidates(p,frame,self.clock,active_candidate=self.policy.selected)
            cap=min(value.speed_cap,speed_policy.desired_speed(p,self.settings))
            # Retain the lead speed until the whole ego body clears the source
            # lane. A future high-speed sample must not compress today's gap.
            if self.phase=='OUT' and value.maneuver=='OVERTAKE':
                selected=next((c for c in candidates if c.lane_id==self.target_lane),None)
                require(selected is not None,'OVERTAKE_SELECTED_CORRIDOR_UNAVAILABLE')
                if not self.policy._body_inside(GoalPose(p.ego.x,p.ego.y,p.ego.heading),selected,False):
                    lead=next((v for v in objects if v['id']==self.lead_id),None)
                    require(lead is not None,'OVERTAKE_LEAD_IDENTITY_UNOBSERVED')
                    cap=min(cap,max(0.,math.cos(p.ego.heading)*lead['vx']+math.sin(p.ego.heading)*lead['vy']))
            matched=[r for r in replies if r.intent_id==self.policy.session.intent_id
                     and r.revision==self.policy.session.revision and r.stage==self.policy.session.stage]
            prefer_planning=self.policy.session.stage=='REQUEST_PATH'
            matched.sort(key=lambda r:(r.status not in ('REJECTED','FAILED'),
                (r.producer=='planning')!=prefer_planning,-r.produced_at_s))
            result=self.policy.evaluate(frame,value.mission_id+':'+str(self.generation),value.maneuver,
                candidates,caps,matched[0] if matched else None,preferred_lane_id=self.target_lane,
                speed_cap_mps=cap,goal_pose=self.goal)
            request=result.request
            self.reason=result.reason
            if request is not None and request.status=='COMPLETED':
                if self.phase=='OUT' and value.maneuver!='MERGE':
                    self.phase='PASS'; self.reason='WHOLE_BODY_IN_PASSING_CORRIDOR'
                else:
                    self.phase='COMPLETE'; self.reason=value.maneuver+'_TASK_COMPLETED'
                return
            require(request is not None and request.dispatch_allowed and not result.hold_required,
                    value.maneuver+':'+result.reason)
            facts.check_current(True)
            deadline=min(p.valid_until,value.model_evidence.valid_until_s,value.route_evidence.valid_until_s,
                         request.valid_until_s,facts.env.dynamic_valid_until,facts.env.valid_until)
            require(self.clock()<deadline,'LANE_CHANGE_SOURCE_EXPIRED_DURING_DISPATCH')
            output.mode=DecisionMode.KEEP_LANE
            output.target_speed=cap
            output.target_lane_id=(p.lane.lane_id if request.stage in ('PREPARE','WAIT_GAP') else self.target_lane)
            if request.stage in ('EXECUTE','SETTLE'):
                output.stop_distance=-1.; output.precision_stop=False
            output.valid_until=min(output.valid_until,deadline)
            output.behavior_request=request.to_dict('runtime_connected')
            output.behavior_request['valid_until_s']=output.valid_until
            output.behavior_active_identity={k:output.behavior_request[k] for k in
                ('intent_id','stage','revision','source_frame_id','issued_at_s')}
            output.reason=value.maneuver+':'+self.phase+':'+request.stage+':'+result.reason
        except Exception as error:
            self.reason='LANE_CHANGE_INPUT:'+str(error)
            if was_active or self.pending():
                self.hold(p,output,self.reason)

    def snapshot(self):
        session=self.policy.session if self.policy else None
        return dict(configured=self.provider is not None,phase=self.phase,reason=self.reason,
                    origin_lane_id=self.origin_lane,target_lane_id=self.target_lane,lead_id=self.lead_id,
                    stage=session.stage if session else None,status=session.status if session else None)
