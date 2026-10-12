"""P03 fixed-entry consumer with an explicitly supplied frame/model snapshot.

The captain supplies a provider after checking its physical sources. This
module cannot certify calibration, steering units or prediction error bounds.
No provider or runtime capability is installed by default. REQUEST_PATH only
acknowledges a nominal path; EXECUTE/SETTLE independently revalidate it.
"""
import copy
import time
import hashlib

from core.behavior_channel import read_channel,capabilities_dict,feedback_dict
from core.behavior_contract import TaskContext,ExecutionFeedback,require,finite
from core.interfaces import Trajectory,DecisionTarget,DecisionMode
from core.validation import validate_output
from core.traffic_quality import signal_stop_requirement
from core.maneuver_facts import ManeuverFacts
from core.geometry import project_polyline,projection_within_polyline,normalize_angle
import math
from members.planning.behavior_contract import assess_behavior_request,LANE_CHANGE_ACTIONS
from members.planning.lane_change_generator import plan_lane_change_candidate
from members.planning.lane_change_generator import _source_predictions,_Failure
from members.planning.lane_planner import build_trajectory
from members.planning.maneuver_scene import read_maneuver_scene
from members.planning.candidate_validation import validate_candidate,ValidationBudget
from members.planning.maneuver_inputs import ManeuverPlanningInputs,read_inputs


class LaneChangePlanningInputs(ManeuverPlanningInputs):
    """Caller-owned model assertion and current curvature, never inferred.

    All times are process_monotonic. Curvature is signed body curvature in
    inverse metres; positive turns left. The caller must establish its actual
    observation, reference point and error model, including at standstill.
    model_id names that caller's model revision, not proof of calibration.
    Existing VehicleGeometry/MotionLimits/Search/Budget/PredictionEnvelope
    values are reused and independently checked by the existing generator.
    """


class LaneChangeBehaviorPlanner(object):
    def __init__(self,inputs_provider=None,clock=None):
        require(inputs_provider is None or callable(inputs_provider),'invalid lane-change input provider')
        self.provider,self.clock=inputs_provider,clock or time.monotonic
        self.context,self.bindings,self.revisions=None,{},{}
        self.prepared,self.candidate_ids=set(),{}

    @staticmethod
    def _revision(request):
        return tuple(request[k] for k in ('revision','stage','source_frame_id','issued_at_s'))

    @staticmethod
    def _binding(request,model,digest):
        pose=request['goal_pose']; lights=request['light_intent']
        return (request['maneuver'],request['source_lane_id'],request['target_lane_id'],
                tuple(pose[k] for k in ('x','y','body_heading_rad','reference_point')),
                tuple(sorted(lights.items())),digest,model.model_id,
                tuple(sorted(vars(model.vehicle).items())),tuple(sorted(vars(model.limits).items())),
                None if model.execution_envelope is None else tuple(getattr(model.execution_envelope,k)
                    for k in ('position_error_m','heading_error_rad','early_time_s','late_time_s')),
                model.speed_error_mps)

    def _inputs(self,p,context):
        return read_inputs(self.provider,p,context,self.clock,'LANE_CHANGE',LaneChangePlanningInputs)

    @staticmethod
    def _reply(context,request,p,now,deadline,status,reason,candidate_id=None,metrics=None):
        return feedback_dict(ExecutionFeedback(context,request['intent_id'],request['stage'],request['revision'],
            'planning',p.frame_id,p.frame_id,now,deadline,status,usable=now<deadline,
            reason_code=reason,retryable=status=='REJECTED',candidate_id=candidate_id,
            planning_metrics=metrics or {},evidence_level='model_geometry_and_execution_bounds'))

    def _staging(self,p,decision,settings,model,checks,deadline,speed_cap):
        # Stay in the actual current lane until the decision consumes PLANNED.
        # The existing ordinary planner supplies the path, then the same P02
        # body/object/visibility model independently checks it. A straddling
        # recovery cannot silently snap back into either lane through staging.
        if model.geometry_cache is not None:
            require(model.geometry_cache.max_entries>=6,'LANE_CHANGE_STAGING_CACHE_NEEDS_SIX_ENTRIES')
        scene=read_maneuver_scene(p,(p.lane.lane_id,),self.clock,model.geometry_cache)
        deadline=min(deadline,scene['source_valid_until_s'])
        compiled=copy.deepcopy(decision)
        compiled.behavior_request=compiled.behavior_active_identity=None
        compiled.target_lane_id=p.lane.lane_id
        compiled.target_speed=speed_cap
        # The full lane reference can extend outside this observed local road.
        # Stop the staging path while its front still fits the current region;
        # P02 separately checks every body corner and intermediate motion.
        road=scene['roads'][0]; center=road['center_line']
        projection=project_polyline(center,p.ego.x,p.ego.y)
        require(projection_within_polyline(projection,len(center)),
                'LANE_CHANGE_STAGING_SOURCE_PROJECTION_UNAVAILABLE')
        length=sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(center,center[1:]))
        error=0.
        if model.execution_envelope is not None:
            radius=math.hypot(max(model.vehicle.front_offset_m,model.vehicle.rear_offset_m),
                              model.vehicle.half_width_m)
            error=(model.execution_envelope.position_error_m
                   +2*radius*math.sin(model.execution_envelope.heading_error_rad/2))
        boundary_stop=max(0.,length-projection['s']-model.vehicle.front_offset_m
                          -max(settings.traffic_stop_margin,model.limits.distance_tolerance_m)-error)
        compiled.stop_distance=(min(compiled.stop_distance,boundary_stop)
                                if compiled.stop_distance>=0 else boundary_stop)
        output=build_trajectory(p,compiled,settings)
        require(output.valid and not output.emergency_stop,'LANE_CHANGE_STAGING_PATH_UNAVAILABLE')
        a,b=output.points[:2]
        # The ordinary reference planner treats sub-micrometre residuals as
        # aligned. Preserve this measured pose at the maneuver boundary;
        # the steering-rate and P02 checks below still validate the new segment.
        if (math.hypot(a.x-p.ego.x,a.y-p.ego.y)<=1e-6
                and abs(normalize_angle(a.heading-p.ego.heading))<=1e-6):
            a.x,a.y,a.heading=p.ego.x,p.ego.y,p.ego.heading
        require(math.hypot(a.x-p.ego.x,a.y-p.ego.y)<=1e-9
                and abs(normalize_angle(a.heading-p.ego.heading))<=1e-9
                and abs(a.speed-p.ego.speed)<=1e-9,'LANE_CHANGE_STAGING_START_MOTION_CHANGED')
        distance=math.hypot(b.x-a.x,b.y-a.y)
        if distance>1e-9:
            first_steer=math.atan(model.vehicle.wheelbase_m*normalize_angle(b.heading-a.heading)/distance)
            current_steer=math.atan(model.vehicle.wheelbase_m*model.initial_curvature_m_inv)
            midpoint=.5*(b.relative_time-a.relative_time)
            require(midpoint>0 and abs(first_steer-current_steer)/midpoint
                    <=model.limits.max_steer_rate_rad_s+1e-9,'LANE_CHANGE_STAGING_INITIAL_STEERING_RATE_LIMIT')
        objects=_source_predictions(scene['objects'],model.prediction_envelopes,self.clock())
        for region,predictions in ((scene['corridor'],objects),(scene['coverage'],[])):
            remaining=model.budget.max_checks-checks
            require(remaining>0,'BUDGET_EXHAUSTED')
            budget=ValidationBudget(remaining,model.budget.max_depth,model.budget.min_interval_s,
                min(model.budget.deadline_monotonic_s,time.monotonic()+deadline-self.clock()))
            report=validate_candidate(output.points,1,model.vehicle,model.limits,region,predictions,budget,
                                      execution_envelope=model.execution_envelope)
            checks+=report['checks']
            require(report['status']=='safe','LANE_CHANGE_STAGING_'+report['reason_code'])
        require(self.clock()<deadline,'LANE_CHANGE_STAGING_SOURCE_EXPIRED')
        output.valid_until=deadline
        return output,deadline

    def plan(self,p,decision,settings):
        output=Trajectory().bind(decision)
        request=decision.behavior_request
        context=None
        deadline=min(p.valid_until,decision.valid_until)
        try:
            validate_output(decision,DecisionTarget,p)
            now=self.clock()
            context,caps,unused=read_channel(p,now)
            assessment=assess_behavior_request(request,context.to_dict(),p.frame_id,now,
                min(p.valid_until,decision.valid_until),'process_monotonic',decision.behavior_active_identity,
                capabilities=capabilities_dict(caps),supported_actions=LANE_CHANGE_ACTIONS,
                current_lane_id=p.lane.lane_id,frame_usable=p.valid,frame_paused=False,runtime_lane_change=True)
            require(assessment['eligible'],assessment['reason_code'])
            request=assessment['request']
            deadline=min(deadline,request['valid_until_s'],caps.valid_until_s)
            preparing=request['stage'] in ('PREPARE','WAIT_GAP')
            require(decision.mode in (DecisionMode.KEEP_LANE,DecisionMode.FOLLOW)
                    and (preparing or request['stage']=='REQUEST_PATH'
                         or (decision.stop_distance==-1 and not decision.precision_stop))
                    and decision.target_speed>0,'LANE_CHANGE_OVERRIDDEN_BY_STOP')
            require(decision.target_lane_id==(p.lane.lane_id if preparing else request['target_lane_id']),
                    'LANE_CHANGE_DECISION_TARGET_MISMATCH')
            require(signal_stop_requirement(p) is None,'LANE_CHANGE_SIGNAL_STOP_REQUIRED')
            if self.context!=context.key():
                self.context,self.bindings,self.revisions=context.key(),{},{}
                self.prepared,self.candidate_ids=set(),{}
            intent=request['intent_id']; revision=self._revision(request)
            require(intent in self.prepared or request['stage'] in ('PREPARE','WAIT_GAP','REQUEST_PATH'),
                    'LANE_CHANGE_REQUEST_PATH_REQUIRED')
            adopted=request.get('candidate_id')
            require(adopted is None or adopted==self.candidate_ids.get(intent),
                    'LANE_CHANGE_ADOPTED_CANDIDATE_MISMATCH')
            previous=self.revisions.get(intent)
            require(previous is None or (revision[0]>previous[0] and revision[2]>=previous[2]
                    and revision[3]>=previous[3]) or revision==previous,
                    'REQUEST_REVISION_REGRESSED_OR_CHANGED')
            require(intent in self.revisions or len(self.revisions)<128,'LANE_CHANGE_SESSION_CAPACITY')
            self.revisions[intent]=revision
            model=self._inputs(p,context)
            deadline=min(deadline,model.valid_until_s)
            facts=ManeuverFacts(p,self.clock)
            digest=facts.map_digest()
            binding=self._binding(request,model,digest)
            previous_binding=self.bindings.get(intent)
            require(previous_binding is None or binding==previous_binding,'LANE_CHANGE_ACTIVE_BINDING_CHANGED')
            pose=request['goal_pose']
            model.budget.deadline_monotonic_s=min(model.budget.deadline_monotonic_s,
                time.monotonic()+deadline-self.clock())
            if preparing:
                output,deadline=self._staging(p,decision,settings,model,0,deadline,
                    min(decision.target_speed,request['speed_cap_mps']))
                facts.check_current(True)
                self.bindings[intent]=binding
                output.left_signal=request['light_intent']['left_signal']
                output.right_signal=request['light_intent']['right_signal']
                output.reason='LANE_CHANGE_INDICATOR_PREPARATION_ONLY'
                # No candidate acceptance or execution identity before evaluation.
                return validate_output(output,Trajectory,decision)
            result=plan_lane_change_candidate(p,request['target_lane_id'],model.initial_curvature_m_inv,
                min(decision.target_speed,request['speed_cap_mps']),model.vehicle,model.limits,
                model.search,model.budget,model.prediction_envelopes,self.clock,model.geometry_cache,
                source_lane_id=request['source_lane_id'],
                fixed_goal_pose=tuple(pose[k] for k in ('x','y','body_heading_rad')),
                allow_target_lane_continuation=previous_binding is not None
                    and request['stage'] in ('EXECUTE','SETTLE'),
                execution_envelope=model.execution_envelope,dynamic_search=model.dynamic_search)
            require(result['status']=='safe',result['reason_code'])
            if result['path_scope']=='source_bound_crossing':
                side=result['crossing_side']
                require(request['light_intent']['left_signal']==(side=='left')
                        and request['light_intent']['right_signal']==(side=='right'),
                        'LANE_CHANGE_INDICATOR_SIDE_MISMATCH')
            require(self.clock()<deadline,'LANE_CHANGE_SOURCE_EXPIRED_DURING_DISPATCH')
            validate_output(decision,DecisionTarget,p)
            require(result['frame_id']==p.frame_id and result['map_digest']==digest,
                    'LANE_CHANGE_BINDING_CHANGED_DURING_DISPATCH')
            deadline=min(deadline,result['source_valid_until_s'])
            now=self.clock()
            require(now<deadline,'LANE_CHANGE_SOURCE_EXPIRED_DURING_DISPATCH')
            # Geometry in shared settings, if supplied, must describe this car.
            for name,actual in (('front_offset_m',model.vehicle.front_offset_m),
                ('rear_offset_m',model.vehicle.rear_offset_m),('half_width_m',model.vehicle.half_width_m),
                ('wheelbase_m',model.vehicle.wheelbase_m),('front_steer_max_rad',model.limits.max_front_steer_rad)):
                supplied=getattr(settings,name)
                require(supplied is None or abs(supplied-actual)<=1e-9,'LANE_CHANGE_CONTROL_GEOMETRY_MISMATCH')
            candidate_id=self.candidate_ids.setdefault(intent,'lane:'+hashlib.sha256(
                repr((context.key(),intent,binding)).encode('utf-8')).hexdigest()[:24])
            metrics=dict(checks=result['checks'],source_valid_until_s=deadline,
                         execution_robust=result.get('execution_robust',False),
                         path_scope=result.get('path_scope'),attempts=len(result.get('attempts',())))
            if (model.execution_envelope is not None
                    and model.execution_envelope.position_error_m>0):
                metrics['progress_bounds']=dict(max_speed_mps=model.limits.max_speed_mps,
                    max_acceleration_mps2=model.limits.max_acceleration_mps2,
                    position_tolerance_m=model.execution_envelope.position_error_m)
            if request['stage']=='REQUEST_PATH':
                output,deadline=self._staging(p,decision,settings,model,result['checks'],deadline,
                    min(decision.target_speed,request['speed_cap_mps']))
                facts.check_current(True)
                validate_output(decision,DecisionTarget,p)
                self.bindings[intent]=binding
                self.prepared.add(intent)
                output.candidate_id=candidate_id
                output.planning_metrics=metrics
                output.left_signal=request['light_intent']['left_signal']
                output.right_signal=request['light_intent']['right_signal']
                output.behavior_feedback=self._reply(context,request,p,self.clock(),deadline,'PLANNED',
                                                     'NOMINAL_LANE_CHANGE_PATH_READY',candidate_id,metrics)
                output.reason='LANE_CHANGE_PATH_PREPARED_WAIT_EXECUTE'
                return validate_output(output,Trajectory,decision)
            self.bindings[intent]=binding
            output.valid_until=deadline
            output.behavior_feedback=self._reply(context,request,p,now,deadline,'PLANNED',
                                                 'NOMINAL_LANE_CHANGE_PATH_READY',candidate_id,metrics)
            output.candidate_id=candidate_id
            output.planning_metrics=metrics
            output.points=result['points']
            output.target_speed=min(decision.target_speed,request['speed_cap_mps'])
            output.target_lane_id=request['target_lane_id']
            output.left_signal=request['light_intent']['left_signal']
            output.right_signal=request['light_intent']['right_signal']
            output.behavior_identity={key:request[key] for key in ('intent_id','stage','revision')}
            output.reason=('CURRENT_TARGET_NOMINAL_CONTINUATION_PATH'
                if result['path_scope']=='current_target_lane' else 'SOURCE_BOUND_NOMINAL_LANE_CHANGE_PATH')
            output.valid=True
            return validate_output(output,Trajectory,decision)
        except (ValueError,TypeError,AttributeError,KeyError,OverflowError,_Failure) as error:
            output=Trajectory().bind(decision)
            output.reason='BEHAVIOR_REQUEST:'+str(error)
            output.errors.append(output.reason)
            if context is not None and isinstance(request,dict):
                try:
                    output.behavior_feedback=self._reply(context,request,p,self.clock(),deadline,'REJECTED',str(error))
                except (ValueError,TypeError,KeyError):
                    pass
            return output
