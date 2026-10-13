"""Formal passage/emergency arbitration; preserves independent stop duties."""
import copy
import math

from core.behavior_channel import read_channel
from core.behavior_contract import BehaviorFrame, BehaviorGoal, GoalPose, LightIntent, require, goal_from_dict
from core.driving_mission import DrivingDecisionInputs
from core.maneuver_facts import ManeuverFacts
from core.passage_facts import read_passage_facts
from core.region_geometry import rectangle, intersects, convex_cells
from core.traffic_quality import signal_stop_requirement
from members.decision.behaviors.emergency import EmergencyPolicy, EmergencyCandidate


class DrivingRuntime(object):
    def __init__(self,provider,session_factory,clock,settings):
        self.provider,self.factory,self.clock,self.settings=provider,session_factory,clock,settings
        self.passage=session_factory()
        self.emergency=EmergencyPolicy(session=session_factory())
        self._latest={}; self._mission=None; self._completed=set()
        self._passage_allowed = None
        self._passage_epoch = 0

    def pending(self):
        return any(s.intent_id and s.status not in ('COMPLETED','CANCELLED')
                   for s in (self.passage,self.emergency.session))

    def suspend(self):
        # No reset or time advancement; the next trustworthy frame revalidates.
        self._latest=dict(reason_code='DRIVING_OBSERVATION_SUSPENDED')

    def snapshot(self):
        return copy.deepcopy(self._latest)

    @staticmethod
    def _reply(feedbacks,session):
        matches=[v for v in feedbacks if v.intent_id==session.intent_id and v.stage==session.stage
                 and v.revision==session.revision]
        return max(matches,key=lambda v:(v.produced_at_s,v.producer!='planning')) if matches else None

    @staticmethod
    def _apply(output,request):
        output.behavior_request=request.to_dict('runtime_connected')
        output.behavior_active_identity={k:output.behavior_request[k] for k in
            ('intent_id','stage','revision','source_frame_id','issued_at_s')}
        output.target_lane_id=request.goal.target_lane_id or output.target_lane_id
        output.target_speed=request.goal.speed_cap_mps
        output.stop_distance=request.goal.stop_distance_m
        output.precision_stop=request.goal.precision_stop
        output.reason=request.maneuver+':'+request.reason_code

    def _fail(self,p,output,reason):
        self._latest=dict(reason_code=reason)
        if self.pending() or self.provider is not None:
            output.mode='EMERGENCY_BRAKE' if p.ego.speed>self.settings.standstill_speed else 'STOP'
            output.target_speed=0.;output.stop_distance=0.
            output.behavior_request=output.behavior_active_identity=None
            output.reason='DRIVING_PROTECTION:'+reason

    def apply(self,p,output,priority=False):
        if self.provider is None:
            return
        try:
            now=self.clock()
            context,caps,replies=read_channel(p,now)
            inputs=self.provider(p,context)
            if inputs is None:
                require(not self.pending(),'DRIVING_TASK_WITHDRAWN')
                return
            require(isinstance(inputs,DrivingDecisionInputs),'DRIVING_DECISION_INPUTS_REQUIRED')
            inputs=inputs.snapshot(); mission=inputs.mission
            frame=BehaviorFrame(context,p.frame_id,now,p.valid_until,p.ego.x,p.ego.y,p.ego.heading,p.ego.speed,usable=p.valid)
            require(mission.evidence.verified(frame,allowed_source_kinds=('task','verified_fusion'))
                    and mission.model_evidence.verified(frame,allowed_source_kinds=('task','verified_fusion')),
                    'DRIVING_TASK_OR_MODEL_UNVERIFIED')
            facts=ManeuverFacts(p,self.clock)
            require(facts.map_digest()==mission.map_digest,'DRIVING_TASK_MAP_MISMATCH')
            objects=facts.objects()
            require(self._mission in (None,(context.key(),mission.mission_id,mission.map_digest)),
                    'DRIVING_ACTIVE_TASK_CHANGED')
            self._mission=(context.key(),mission.mission_id,mission.map_digest)
            dangerous=output.mode=='EMERGENCY_BRAKE'
            if dangerous or self.emergency.session.intent_id and self.emergency.session.status!='COMPLETED':
                candidates=[]
                planning=[v for v in replies if v.producer=='planning'
                    and v.intent_id==self.emergency.session.intent_id and v.revision==self.emergency.session.revision]
                for feedback in planning:
                    for option in feedback.candidate_options:
                        goal=goal_from_dict(option['goal'])
                        wrapped=copy.deepcopy(feedback);wrapped.candidate_id=option['candidate_id']
                        candidates.append(EmergencyCandidate(option['candidate_id'],option['action'],wrapped,
                            option['feasible'],option['risk_cost'],option['rule_cost'],goal,
                            side_coverage_verified=option['feasible'],crossing_allowed=option['feasible']))
                danger='objects:'+','.join(str(v['id']) for v in objects)
                result=self.emergency.evaluate(frame,mission.model_evidence,danger,dangerous,candidates,caps,
                    self._reply(replies,self.emergency.session),verified_task_route_available=p.lane.valid)
                request=result.request
                if request is not None and request.dispatch_allowed:
                    if request.stage=='SELECT':
                        # Visible hazards during first-cycle immediate braking;
                        # candidate evaluation cannot request ordinary throttle.
                        self.emergency.session.goal=BehaviorGoal(light_intent=LightIntent(hazard_signal=True))
                        request=self.emergency.session.snapshot(frame)
                    self._apply(output,request)
                    output.mode='EMERGENCY_BRAKE'
                elif result.phase!='COMPLETE' and result.immediate_brake_required:
                    output.mode='EMERGENCY_BRAKE';output.target_speed=0.;output.stop_distance=0.
                    output.behavior_request=output.behavior_active_identity=None
                    if request is not None:
                        # Retain an undispatched emergency intent so immediate
                        # braking keeps its explicit hazards. The planner still
                        # rejects the unavailable motion authorization.
                        output.behavior_request=request.to_dict('runtime_connected')
                        output.behavior_active_identity={k:output.behavior_request[k] for k in
                            ('intent_id','stage','revision','source_frame_id','issued_at_s')}
                    output.reason='EMERGENCY_ACTIVE_REQUEST_UNAVAILABLE'
                self._latest=dict(phase=result.phase,reason_code=result.reason,
                    request=None if request is None else request.to_dict('runtime_connected'))
                return
            if priority or output.behavior_request is not None or mission.passage_id is None:
                return
            vehicle=type('BodyExtents',(),dict(zip(('front_offset_m','rear_offset_m','half_width_m'),inputs.vehicle_extents)))()
            scene=read_passage_facts(p,mission.passage_id,vehicle,self.clock)
            key=(context.key(),mission.mission_id,mission.passage_id)
            if key in self._completed:
                return
            windows=[(max(0.,a-now),b-now) for a,b in mission.allowed_windows_abs_s if b>now]
            allowed=mission.right_of_way!='UNKNOWN' and bool(windows) and signal_stop_requirement(p) is None
            # Current exit occupancy remains a separate admission veto. Every
            # predicted target, including pedestrians, is also checked by P05.
            allowed=allowed and scene['junction'].get('exit_occupancy')=='clear'
            cells = [cell for polygon in scene['conflict_polygons'] for cell in convex_cells(polygon)]
            age = now-facts.env.dynamic_observed_at_s
            for obj in objects:
                radius = .5*math.hypot(obj['length'], obj['width'])
                heading = obj['heading']
                body = rectangle(obj['x']+obj['vx']*age, obj['y']+obj['vy']*age,
                    heading if heading is not None else 0.,
                    obj['length']/2 if heading is not None else radius,
                    obj['length']/2 if heading is not None else radius,
                    obj['width']/2 if heading is not None else radius)
                if any(intersects(body, cell) for cell in cells):
                    allowed = False
                    break
            facts.check_current(True)
            recovered = self._passage_allowed is False and allowed
            self._passage_allowed = allowed
            if recovered and self.passage.status == 'BLOCKED':
                self._passage_epoch += 1
                self.passage.renew_after_verified_change(frame,
                    'passage-covered-occupied-to-clear:' + str(self._passage_epoch))
            inside=scene['already_inside']
            if inside and not allowed:
                if self.passage.intent_id is not None:
                    self.passage.tick(frame, caps, safety_override=True)
                output.mode, output.target_speed, output.stop_distance = 'EMERGENCY_BRAKE', 0., 0.
                output.behavior_request = output.behavior_active_identity = None
                self._latest = dict(phase='INSIDE_PROTECTION', reason_code='PASSAGE_PERMISSION_OR_SPACE_WITHDRAWN')
                return
            stage='EXIT' if inside or self.passage.stage=='EXIT' else 'ENTER' if allowed else 'WAIT_ENTRY'
            front=inputs.vehicle_extents[0]
            stop=max(0.,scene['junction'].get('entry_distance_m',0.)-front-mission.entry_margin_m)
            goal=BehaviorGoal(speed_cap_mps=mission.speed_cap_mps,
                stop_distance_m=stop if stage=='WAIT_ENTRY' else -1.,precision_stop=stage=='WAIT_ENTRY')
            if self.passage.intent_id is None:
                self.passage.start(frame,mission.mission_id,scene.get('kind','INTERSECTION'),stage,goal,('INTERSECTION','PATH','FEEDBACK'))
            elif self.passage.stage!=stage:
                self.passage.advance(frame,stage,goal,reason='CURRENT_PASSAGE_AUTHORITY_REVALIDATED')
            elif self.passage.status=='BLOCKED' and allowed:
                self.passage.retry_after_evidence(frame,'PASSAGE_SOURCE_RECOVERED')
                self.passage.start(frame,mission.mission_id,scene.get('kind','INTERSECTION'),stage,goal,('INTERSECTION','PATH','FEEDBACK'))
            else:
                self.passage.start(frame,mission.mission_id,scene.get('kind','INTERSECTION'),stage,goal,('INTERSECTION','PATH','FEEDBACK'))
            request=self.passage.tick(frame,caps,self._reply(replies,self.passage))
            actual=self.passage.last_feedback
            if (actual is not None and actual.producer in ('control','runtime') and actual.actual_pose is not None
                    and actual.usable and actual.status in ('EXECUTING','ARRIVED','DWELLING','COMPLETED')
                    and request.dispatch_allowed and self.passage.status in ('ACCEPTED','EXECUTING')):
                body=rectangle(p.ego.x,p.ego.y,p.ego.heading,*inputs.vehicle_extents)
                end=scene['exit_region'].body_clearance(body,lambda:None,100000)
                require(inputs.execution_position_error_m is not None
                        and inputs.execution_heading_error_rad is not None,
                        'PASSAGE_COMPLETION_EXECUTION_BOUNDS_UNKNOWN')
                radius=math.hypot(max(inputs.vehicle_extents[:2]),inputs.vehicle_extents[2])
                error=inputs.execution_position_error_m+2*radius*math.sin(inputs.execution_heading_error_rad/2)
                observed=actual.actual_pose
                observed_body=rectangle(observed.x,observed.y,observed.body_heading_rad,*inputs.vehicle_extents)
                observed_end=scene['exit_region'].body_clearance(observed_body,lambda:None,100000)
                # Actual whole-car exit, never a planned endpoint/rear axle alone.
                touches=any(intersects(body,cell) for region in scene['conflict_polygons'] for cell in convex_cells(region))
                observed_touches=any(intersects(observed_body,cell) for region in scene['conflict_polygons']
                    for cell in convex_cells(region))
                if (end>mission.exit_margin_m+error and observed_end>mission.exit_margin_m+error
                        and not touches and not observed_touches and self.passage.stage in ('ENTER','EXIT')):
                    self.passage.finish(frame,True,'WHOLE_CAR_ACTUALLY_EXITED')
                    self._completed.add(key)
                    if len(self._completed)>128:self._completed={key}
                    self._latest=dict(phase='COMPLETE',reason_code='WHOLE_CAR_ACTUALLY_EXITED')
                    return
            if request.dispatch_allowed:
                self._apply(output,request)
                output.mode='KEEP_LANE'
            else:
                self._fail(p,output,'PASSAGE_REQUEST_NOT_DISPATCHABLE')
            self._latest=dict(phase=stage,reason_code=request.reason_code,request=request.to_dict('runtime_connected'))
        except (ValueError,TypeError,KeyError,AttributeError,IndexError,OverflowError) as error:
            self._fail(p,output,str(error))
