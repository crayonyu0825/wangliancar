"""Fixed-entry P06/P08 consumer: source authority and generated motion stay separate."""
import copy
import hashlib
import math
import time

from core.behavior_channel import read_channel, capabilities_dict, feedback_dict
from core.behavior_contract import ExecutionFeedback, BehaviorGoal, GoalPose, LightIntent, require
from core.driving_mission import DrivingMission
from core.interfaces import Trajectory, TrajectoryPoint, DecisionTarget, DecisionMode
from core.validation import validate_output
from core.passage_facts import read_passage_facts
from core.maneuver_facts import ManeuverFacts
from core.geometry import project_polyline, normalize_angle
from members.planning.maneuver_inputs import ManeuverPlanningInputs, read_inputs
from members.planning.maneuver_scene import read_maneuver_scene
from members.planning.route_scene import read_route_scene
from members.planning.behavior_contract import assess_behavior_request, DRIVING_ACTIONS
from members.planning.lane_planner import build_trajectory
from members.planning.lane_change_generator import _source_predictions,_Failure
from members.planning.passage_planner import PassageConstraints, plan_passage
from members.planning.emergency_planner import generate_emergency_candidates
from members.planning.dynamic_speed import solve_dynamic_speed
from members.planning.candidate_validation import ValidationBudget


class DrivingPlanningInputs(ManeuverPlanningInputs):
    def __init__(self,*args,**kwargs):
        mission=kwargs.pop('mission',None)
        require(isinstance(mission,DrivingMission),'DRIVING_MISSION_REQUIRED')
        super(DrivingPlanningInputs,self).__init__(*args,**kwargs)
        self.mission=mission.snapshot()


def passage_reference(p,scene,vehicle,margin):
    """Only the captain's selected actual lane samples may extend the route."""
    q=copy.deepcopy(p)
    if scene.get('kind')=='CROSSWALK':
        # The selected actual current-road reference is sufficient; a crossing
        # polygon never grants permission to take a different road.
        q.lane.forward_reference=list(scene['road']['center_line'])
        q.lane.forward_reference_valid=True
        q.lane.forward_lane_ids=[p.lane.lane_id]
        q.lane.forward_lane_spans=[]
        q.lane.forward_reference_status='map_end'
        return q
    regions=scene['junction']['corridor_regions']
    ids=[p.lane.lane_id]+[v['lane_id'] for v in regions]
    require(p.lane.lane_id in ids,'PASSAGE_CURRENT_LANE_UNAVAILABLE')
    if p.lane.lane_id in [v['lane_id'] for v in regions]:
        start=next(i for i,v in enumerate(regions) if v['lane_id']==p.lane.lane_id)
        chosen=regions[start:]
        reference=list(p.lane.center_line)
        chosen=chosen[1:]
    else:
        reference=list(p.lane.center_line);chosen=regions
    require(reference,'PASSAGE_CURRENT_REFERENCE_UNAVAILABLE')
    for region in chosen:
        line=region['center_line']
        require(math.hypot(reference[-1][0]-line[0][0],reference[-1][1]-line[0][1])<=.3,
                'PASSAGE_ROUTE_REFERENCE_JOIN_INVALID')
        reference.extend(line[1:])
    q.lane.forward_reference=reference
    q.lane.forward_reference_valid=True
    q.lane.forward_lane_ids=list(dict.fromkeys(ids))
    q.lane.forward_lane_spans=[]
    q.lane.forward_reference_status='map_end'
    # The planner's stop at map end already reserves the front offset. P02
    # independently rejects endpoints whose complete inflated body will not fit.
    return q


def candidate_id(context,intent,binding):
    return 'driving:'+hashlib.sha256(repr((context.key(),intent,binding)).encode('utf-8')).hexdigest()[:24]


def _report_metrics(report):
    """Publish this attempt's measured work, including nested speed search."""
    if not report:return {}
    dynamic=report.get('dynamic_report',{})
    reason=report.get('reason_code','')
    result=dict(status=report.get('status','invalid'),reason_code=reason,
        nodes=report.get('nodes',dynamic.get('nodes',0)),
        budget_exhausted=bool(report.get('budget_exhausted',False) or 'BUDGET' in reason
            or dynamic.get('budget_exhausted',False) or 'BUDGET' in dynamic.get('reason_code','')))
    for field in ('checks','elapsed_s'):
        if field in report:result[field]=report[field]
    return result


def _rejected_metrics(report,error):
    result=_report_metrics(report)
    if 'status' in result:result['search_status']=result['status']
    result['status']='inconclusive' if 'BUDGET' in str(error) else 'rejected'
    result['reason_code']=str(error)
    result['budget_exhausted']=result.get('budget_exhausted',False) or 'BUDGET' in str(error)
    return result


class DrivingBehaviorPlanner(object):
    def __init__(self,inputs_provider=None,clock=None):
        self.provider,self.clock=inputs_provider,clock or time.monotonic
        self.context=None;self.bindings={};self.first_frames={};self.options={}
        self.last_report={}

    def ordinary(self,out,p,d,settings):
        """P05 retiming preserves the ordinary planner's geometry and every cap."""
        self.last_report={}
        if self.provider is None or not out.valid or out.emergency_stop:return out
        try:
            context,unused,unused2=read_channel(p,self.clock())
            supplied=self.provider(p,context)
            if supplied is None:return out
            m=read_inputs(lambda *args:supplied,p,context,self.clock,'DRIVING',DrivingPlanningInputs)
            require(m.execution_envelope is not None,'DRIVING_EXECUTION_MODEL_UNAVAILABLE')
            scene=read_route_scene(p,self.clock,m.geometry_cache)
            deadline=min(out.valid_until,m.valid_until_s,scene['source_valid_until_s'])
            if not out.emergency_stop and any(v.speed>0 for v in out.points):
                center=scene['reference']
                at=project_polyline(center,p.ego.x,p.ego.y)
                length=sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(center,center[1:]))
                radius=math.hypot(max(m.vehicle.front_offset_m,m.vehicle.rear_offset_m),m.vehicle.half_width_m)
                error=m.execution_envelope.position_error_m+2*radius*math.sin(m.execution_envelope.heading_error_rad/2)
                bound=max(0.,length-at['s']-m.vehicle.front_offset_m-error-settings.traffic_stop_margin-.1)
                compiled=copy.deepcopy(d);compiled.behavior_request=compiled.behavior_active_identity=None
                compiled.stop_distance=min(out.stop_distance,bound) if out.stop_required and out.stop_distance>=0 else bound
                adjusted=build_trajectory(p,compiled,settings)
                require(adjusted.valid and not adjusted.emergency_stop,'ORDINARY_BODY_STOP_INFEASIBLE')
                # Keep original behavior duty/lamp/identity metadata; geometry
                # and its caps are regenerated only for the stricter body bound.
                out.points=adjusted.points;out.stop_required=adjusted.stop_required
                out.stop_distance=adjusted.stop_distance
            if all(math.hypot(v.x-p.ego.x,v.y-p.ego.y)<1e-9 for v in out.points):
                from members.planning.candidate_validation import validate_candidate
                targets=_source_predictions(scene['objects'],m.prediction_envelopes,self.clock())
                total=dict(status='safe',reason_code='ORDINARY_HOLD_CERTIFIED',checks=0,nodes=0,elapsed_s=0.)
                hold_deadline=min(m.budget.deadline_monotonic_s,time.monotonic()+deadline-self.clock())
                for region,objects in ((scene['corridor'],targets),(scene['coverage'],[])):
                    remaining=m.budget.max_checks-total['checks']
                    if remaining<=0:
                        total.update(status='inconclusive',reason_code='ORDINARY_HOLD_BUDGET_EXHAUSTED')
                        self.last_report=dict(total)
                        require(False,total['reason_code'])
                    hold_budget=ValidationBudget(remaining,m.budget.max_depth,m.budget.min_interval_s,hold_deadline)
                    report=validate_candidate(out.points,1,m.vehicle,m.limits,region,objects,hold_budget,
                        execution_envelope=m.execution_envelope)
                    total['checks']+=report['checks'];total['elapsed_s']+=report['elapsed_s']
                    if report['status']!='safe':total.update(status=report['status'],reason_code=report['reason_code'])
                    self.last_report=dict(total)
                    require(report['status']=='safe','ORDINARY_HOLD:'+report['reason_code'])
                out.planning_metrics=_report_metrics(total)
                out.planning_metrics['execution_robust']=True
                out.valid_until=deadline
                require(self.clock()<deadline,'ORDINARY_SOURCE_EXPIRED_DURING_HOLD')
                return out
            first=out.points[0]
            # Tiny reference projection arithmetic must not change measured
            # initial velocity or steering. Larger mismatches stay unavailable.
            require(math.hypot(first.x-p.ego.x,first.y-p.ego.y)<=1e-6
                    and abs(normalize_angle(first.heading-p.ego.heading))<=1e-6
                    and abs(first.speed-p.ego.speed)<=1e-9,'ORDINARY_INITIAL_MOTION_CHANGED')
            first.x,first.y,first.heading=p.ego.x,p.ego.y,p.ego.heading
            shape=[(v.x,v.y,v.heading) for v in out.points]
            station=0.;caps=[]
            for i,v in enumerate(out.points):
                if i:station+=math.hypot(v.x-out.points[i-1].x,v.y-out.points[i-1].y)
                caps.append((station,v.speed))
            m.budget.deadline_monotonic_s=min(m.budget.deadline_monotonic_s,time.monotonic()+deadline-self.clock())
            # An ordinary output may declare a stop beyond its visible horizon.
            # Only its actual terminal zero speed is executable within this path.
            stopping=out.points[-1].speed==0.
            result=solve_dynamic_speed(shape,p.ego.speed,min(d.target_speed or p.ego.speed,m.limits.max_speed_mps),
                m.vehicle,m.limits,scene['corridor'],scene['coverage'],
                _source_predictions(scene['objects'],m.prediction_envelopes,self.clock()),m.budget,
                m.execution_envelope,search=m.dynamic_search,stop_required=stopping,
                terminal_speed_mps=0. if stopping else None,spatial_speed_limits=caps)
            self.last_report={k:v for k,v in result.items() if k!='points'}
            require(result['status']=='safe',result['reason_code'])
            out.points=result['points'];out.valid_until=deadline
            out.candidate_id=candidate_id(context,'ordinary:'+str(p.frame_id),(m.model_id,scene['map_digest']))
            out.planning_metrics=_report_metrics(result)
            out.planning_metrics.update(execution_robust=result['execution_robust'],ordinary_dynamic_speed=True,
                retained_spatial_speed_limits=caps)
            require(self.clock()<deadline,'ORDINARY_SOURCE_EXPIRED_DURING_SOLVE')
            return validate_output(out,Trajectory,d)
        except (ValueError,TypeError,KeyError,AttributeError,IndexError,OverflowError,_Failure) as error:
            rejected=Trajectory().bind(d);rejected.reason='ORDINARY_P05:'+str(error)
            rejected.planning_metrics=_rejected_metrics(self.last_report,error)
            rejected.errors.append(rejected.reason);return rejected

    @staticmethod
    def _brake(p,decision):
        out=Trajectory().bind(decision)
        out.points=[TrajectoryPoint(p.ego.x,p.ego.y,0.,p.ego.heading,0.),
                    TrajectoryPoint(p.ego.x,p.ego.y,0.,p.ego.heading,.1)]
        out.valid=True;out.emergency_stop=True;out.stop_required=True;out.stop_distance=0.
        out.target_speed=0.;out.hazard_signal=True
        out.reason='EMERGENCY_IMMEDIATE_BRAKE_BEFORE_SEARCH'
        out.planning_metrics=dict(status='brake_command_only',reason_code=out.reason,
            checks=0,nodes=0,elapsed_s=0.,budget_exhausted=False)
        return out

    def _reply(self,context,r,p,deadline,status,reason,cid=None,options=(),metrics=None):
        return feedback_dict(ExecutionFeedback(context,r['intent_id'],r['stage'],r['revision'],
            'planning',p.frame_id,p.frame_id,self.clock(),deadline,status,usable=self.clock()<deadline,
            reason_code=reason,retryable=status=='REJECTED',candidate_id=cid,
            candidate_options=list(options),planning_metrics=metrics or {},
            evidence_level='model_geometry_and_execution_bounds'))

    def _emergency(self,p,d,r,m,context,deadline):
        intent=r['intent_id'];first=self.first_frames.setdefault(intent,p.frame_id)
        if r['stage']=='SELECT' and p.frame_id-first<m.mission.immediate_brake_cycles:
            # No candidate generation, even before the first receipt. A bounded
            # search starts only on a distinct later source frame.
            out=self._brake(p,d)
            return out
        facts=ManeuverFacts(p,self.clock)
        current=read_maneuver_scene(p,(p.lane.lane_id,),self.clock,m.geometry_cache)
        deadlines=[deadline,current['source_valid_until_s']]
        stop=p.ego.speed**2/(2*m.limits.max_deceleration_mps2)+max(.5,m.limits.distance_tolerance_m)
        origin=project_polyline(current['roads'][0]['center_line'],p.ego.x,p.ego.y)
        from members.planning.passage_planner import pose_at,path_stations
        center=[tuple(v[:2])+(math.atan2(b[1]-a[1],b[0]-a[0]),) for v,a,b in
                zip(current['roads'][0]['center_line'],current['roads'][0]['center_line'],
                    current['roads'][0]['center_line'][1:]+current['roads'][0]['center_line'][-1:])]
        if len(center)>1:center[-1]=center[-1][:2]+center[-2][2:]
        stations=path_stations(center)
        require(origin['s']+stop<stations[-1]-m.vehicle.front_offset_m,'EMERGENCY_BRAKE_ROOM_UNKNOWN')
        goal=pose_at(center,stations,origin['s']+stop)
        regions=[dict(action='BRAKE',goal=goal,corridor=current['corridor'],coverage=current['coverage'],
                      goal_region=current['corridor'])]
        for option in m.mission.emergency_goals:
            scene=read_maneuver_scene(p,option['lane_ids'],self.clock,m.geometry_cache,
                p.lane.lane_id if len(option['lane_ids'])==2 else None)
            require(p.lane.lane_id in option['lane_ids'],'EMERGENCY_AUTHORIZED_LANE_SET_INVALID')
            corridor=scene['legal_crossing_corridor'] if len(option['lane_ids'])==2 else scene['corridor']
            require(corridor is not None,'EMERGENCY_BOUNDARY_AUTHORITY_UNKNOWN')
            g=option['goal'];deadlines.append(scene['source_valid_until_s'])
            regions.append(dict(action=option['action'],goal=(g.x,g.y,g.body_heading_rad),
                corridor=corridor,coverage=scene['coverage'],goal_region=scene['corridor']))
        deadline=min(deadlines)
        m.budget.deadline_monotonic_s=min(m.budget.deadline_monotonic_s,time.monotonic()+deadline-self.clock())
        objects=_source_predictions(facts.objects(),m.prediction_envelopes,self.clock())
        if r['stage']!='SELECT':
            require(r.get('candidate_id') in self.options,'EMERGENCY_CANDIDATE_NOT_ACCEPTED')
            old=self.options[r['candidate_id']]
            require((r['stage']=='BRAKE_FALLBACK' or old['action']==r['stage'][8:])
                and old['goal']==tuple(r['goal_pose'][k]
                for k in ('x','y','body_heading_rad')),'EMERGENCY_ORIGINAL_CANDIDATE_CHANGED')
            # Retain the originally authorized endpoint; only start motion and
            # measured objects refresh. BRAKE's stopping goal is not renewed.
            # Multiple side goals may share an action. The selected endpoint
            # binds its own current source region. BRAKE's computed endpoint
            # changes with measured speed, so its original goal is retained
            # separately against the refreshed current-road authority.
            authorized=next((v for v in regions if v['action']==old['action']
                and (old['action']=='BRAKE' or v['goal']==old['goal'])),None)
            require(authorized is not None,'EMERGENCY_ORIGINAL_REGION_AUTHORITY_UNAVAILABLE')
            authorized=dict(authorized,goal=old['goal']);regions=[authorized]
            if p.ego.speed<=1e-9:
                from members.planning.candidate_validation import validate_candidate
                out=Trajectory().bind(d);out.points=[TrajectoryPoint(p.ego.x,p.ego.y,0.,p.ego.heading,0.),
                    TrajectoryPoint(p.ego.x,p.ego.y,0.,p.ego.heading,.1)]
                total=dict(status='safe',reason_code='EMERGENCY_ACTUAL_HOLD_CERTIFIED',checks=0,nodes=0,elapsed_s=0.)
                for region,targets in ((authorized['corridor'],objects),(authorized['coverage'],[])):
                    remaining=m.budget.max_checks-total['checks']
                    if remaining<=0:
                        total.update(status='inconclusive',reason_code='EMERGENCY_ACTUAL_HOLD_BUDGET_EXHAUSTED')
                        self.last_report=dict(total)
                        require(False,total['reason_code'])
                    hold_budget=ValidationBudget(remaining,m.budget.max_depth,m.budget.min_interval_s,
                        m.budget.deadline_monotonic_s)
                    checked=validate_candidate(out.points,1,m.vehicle,m.limits,region,targets,hold_budget,
                        execution_envelope=m.execution_envelope)
                    total['checks']+=checked['checks'];total['elapsed_s']+=checked['elapsed_s']
                    if checked['status']!='safe':total.update(status=checked['status'],reason_code=checked['reason_code'])
                    self.last_report=dict(total)
                    require(checked['status']=='safe','EMERGENCY_ACTUAL_STOP:'+checked['reason_code'])
                out.valid=True;out.stop_required=True;out.precision_stop=False;out.stop_distance=0.
                out.hazard_signal=True;out.target_speed=0.;out.candidate_id=r['candidate_id']
                out.behavior_identity={k:r[k] for k in ('intent_id','stage','revision')};out.valid_until=deadline
                out.planning_metrics=_report_metrics(total)
                out.planning_metrics.update(emergency_outcome='verified_safe_brake_stop',
                    original_goal_pose=dict(r['goal_pose']),actual_stop_pose=(p.ego.x,p.ego.y,p.ego.heading))
                out.reason='EMERGENCY_VERIFIED_SAFE_HOLD_NOT_ORIGINAL_GOAL'
                out.behavior_feedback=self._reply(context,r,p,deadline,'PLANNED',out.reason,
                    out.candidate_id,metrics=out.planning_metrics)
                return out
            if r['stage']=='BRAKE_FALLBACK':
                out=self._brake(p,d);out.candidate_id=r['candidate_id']
                out.reason='EMERGENCY_AUTHORIZED_BRAKE_COMMAND_NO_MOTION_PROMISE'
                out.planning_metrics.update(fallback_brake_required=True,original_goal_pose=dict(r['goal_pose']))
                return out
        result=generate_emergency_candidates((p.ego.x,p.ego.y,p.ego.heading),p.ego.speed,
            m.initial_curvature_m_inv,min(max(.001,p.ego.speed-.01),m.mission.speed_cap_mps,m.limits.max_speed_mps),m.vehicle,
            m.limits,regions,objects,m.search,m.budget,m.execution_envelope,m.dynamic_search)
        self.last_report={k:v for k,v in result.items() if k!='candidates'}
        require(result['status']=='safe',result['reason_code'])
        metrics=_report_metrics(result)
        metrics.update(execution_robust=True,
            progress_bounds=dict(max_speed_mps=m.limits.max_speed_mps,
                max_acceleration_mps2=m.limits.max_acceleration_mps2,
                position_tolerance_m=m.execution_envelope.position_error_m))
        if r['stage']=='SELECT':
            cards=[]
            for value in result['candidates']:
                cid=candidate_id(context,intent,(value['action'],value['goal'],m.model_id,m.mission.mission_id))
                if cid not in self.options and len(self.options)>=1024:
                    self.options.pop(next(iter(self.options)))
                self.options[cid]=dict(action=value['action'],goal=value['goal'])
                goal=BehaviorGoal(goal_pose=GoalPose(*value['goal']),motion_direction=1,
                    speed_cap_mps=m.mission.speed_cap_mps,precision_stop=True,
                    stop_distance_m=value['stop_distance'],light_intent=LightIntent(hazard_signal=True),candidate_id=cid)
                cards.append(dict(candidate_id=cid,action=value['action'],goal=goal.to_dict(),feasible=True,
                    risk_cost=value['risk_cost'],rule_cost=value['rule_cost'],validation_reason='P02_P05_EXECUTION_BOUNDS_VALIDATED',
                    source_valid_until_s=deadline,segment_id=cid,segment_count=1))
            out=self._brake(p,d)
            out.planning_metrics=metrics
            out.behavior_feedback=self._reply(context,r,p,deadline,'PLANNED','AUTHORIZED_EMERGENCY_OPTIONS_READY',
                                             options=cards,metrics=metrics)
            return out
        value=result['candidates'][0]
        out=Trajectory().bind(d);out.points=value['points'];out.valid=True
        out.stop_distance=value['stop_distance'];out.stop_required=out.precision_stop=True
        out.target_speed=0.;out.hazard_signal=True
        out.candidate_id=r['candidate_id'];out.planning_metrics=metrics
        out.reason='ACCEPTED_EMERGENCY_BRAKE_AND_STEER_PATH'
        out.behavior_identity={k:r[k] for k in ('intent_id','stage','revision')}
        out.valid_until=deadline
        out.behavior_feedback=self._reply(context,r,p,deadline,'PLANNED',out.reason,out.candidate_id,metrics=metrics)
        return out

    def plan(self,p,d,settings):
        self.last_report={}
        out=Trajectory().bind(d);context=None;r=d.behavior_request
        deadline=min(p.valid_until,d.valid_until)
        try:
            validate_output(d,DecisionTarget,p)
            context,caps,unused=read_channel(p,self.clock())
            assessment=assess_behavior_request(r,context.to_dict(),p.frame_id,self.clock(),deadline,
                'process_monotonic',d.behavior_active_identity,capabilities=capabilities_dict(caps),
                supported_actions=DRIVING_ACTIONS,current_lane_id=p.lane.lane_id,
                frame_usable=p.valid,frame_paused=False,runtime_driving=True)
            require(assessment['eligible'],assessment['reason_code']);r=assessment['request']
            m=read_inputs(self.provider,p,context,self.clock,'DRIVING',DrivingPlanningInputs)
            require(m.execution_envelope is not None,'DRIVING_EXECUTION_MODEL_UNAVAILABLE')
            deadline=min(deadline,m.valid_until_s,r['valid_until_s'],caps.valid_until_s)
            facts=ManeuverFacts(p,self.clock)
            require(facts.map_digest()==m.mission.map_digest,'DRIVING_TASK_MAP_MISMATCH')
            if self.context!=context.key():
                self.context=context.key();self.bindings={};self.first_frames={};self.options={}
            binding=(r['maneuver'],m.model_id,m.mission.map_digest,m.mission.mission_id,
                     tuple(sorted(vars(m.vehicle).items())),tuple(sorted(vars(m.limits).items())),
                     tuple(getattr(m.execution_envelope,k) for k in
                         ('position_error_m','heading_error_rad','early_time_s','late_time_s')),
                     m.speed_error_mps)
            require(r['intent_id'] not in self.bindings or self.bindings[r['intent_id']]==binding,
                    'DRIVING_ACTIVE_MODEL_OR_MISSION_CHANGED')
            require(r['intent_id'] in self.bindings or len(self.bindings)<128,'DRIVING_SESSION_CAPACITY')
            self.bindings[r['intent_id']]=binding
            if r['maneuver']=='EMERGENCY':
                out=self._emergency(p,d,r,m,context,deadline)
            else:
                require(d.mode in (DecisionMode.KEEP_LANE,DecisionMode.FOLLOW),'PASSAGE_PRIORITY_STOP')
                scene=read_passage_facts(p,m.mission.passage_id,m.vehicle,self.clock)
                deadline=min(deadline,scene['source_valid_until_s'])
                q=passage_reference(p,scene,m.vehicle,m.mission.exit_margin_m)
                compiled=copy.deepcopy(d);compiled.behavior_request=compiled.behavior_active_identity=None
                compiled.target_lane_id=p.lane.lane_id
                reference=q.lane.forward_reference
                origin=project_polyline(reference,p.ego.x,p.ego.y)
                length=sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(reference,reference[1:]))
                radius=math.hypot(max(m.vehicle.front_offset_m,m.vehicle.rear_offset_m),m.vehicle.half_width_m)
                error=m.execution_envelope.position_error_m+2*radius*math.sin(m.execution_envelope.heading_error_rad/2)
                end_stop=max(0.,length-origin['s']-m.vehicle.front_offset_m-m.mission.exit_margin_m-error-.1)
                compiled.stop_distance=min(compiled.stop_distance,end_stop) if compiled.stop_distance>=0 else end_stop
                if r['stage']=='WAIT_ENTRY':
                    entry_stop=max(0.,scene['junction']['entry_distance_m']-m.vehicle.front_offset_m
                        -m.mission.entry_margin_m-error-.3)
                    compiled.stop_distance=min(compiled.stop_distance,entry_stop)
                # Preserve all legacy map/curvature/obstacle/signal restrictions.
                raw=build_trajectory(q,compiled,settings)
                require(raw.valid and not raw.emergency_stop,'PASSAGE_REFERENCE_OR_STOP_INFEASIBLE')
                shape=[(v.x,v.y,v.heading) for v in raw.points]
                station=0.;caps_by_s=[]
                for i,v in enumerate(raw.points):
                    if i:station+=math.hypot(v.x-raw.points[i-1].x,v.y-raw.points[i-1].y)
                    caps_by_s.append((station,v.speed))
                windows=[(max(0.,a-self.clock()),b-self.clock()) for a,b in
                         m.mission.allowed_windows_abs_s if b>self.clock()]
                constraints=PassageConstraints(m.mission.passage_id,scene['conflict_polygons'],
                    scene['exit_region'],windows,m.mission.entry_margin_m,m.mission.exit_margin_m,
                    right_of_way=m.mission.right_of_way!='UNKNOWN' and scene['junction']['exit_occupancy']=='clear',
                    waiting=r['stage']=='WAIT_ENTRY',already_inside=scene['already_inside'],
                    exiting=r['stage']=='EXIT')
                m.budget.deadline_monotonic_s=min(m.budget.deadline_monotonic_s,
                    time.monotonic()+deadline-self.clock())
                result=plan_passage(shape,p.ego.speed,min(d.target_speed,r['speed_cap_mps']),m.vehicle,m.limits,
                    scene['corridor'],scene['coverage'],_source_predictions(scene['objects'],m.prediction_envelopes,self.clock()),
                    m.execution_envelope,constraints,m.budget,m.dynamic_search,caps_by_s)
                self.last_report={k:v for k,v in result.items() if k!='points'}
                require(result['status']=='safe',result['reason_code'])
                out=Trajectory().bind(d);out.points=result['points'];out.valid=True;out.valid_until=deadline
                out.stop_required=result['stop_required'];out.stop_distance=result['stop_distance']
                out.precision_stop=out.stop_required;out.target_speed=d.target_speed;out.target_lane_id=p.lane.lane_id
                out.candidate_id=candidate_id(context,r['intent_id'],binding+(m.mission.passage_id,
                    'wait' if r['stage']=='WAIT_ENTRY' else 'passage'))
                out.planning_metrics=_report_metrics(result)
                out.planning_metrics.update(execution_robust=True,
                    body_occupancy_windows_s=result.get('body_occupancy_windows_s',[]),
                    progress_bounds=dict(max_speed_mps=m.limits.max_speed_mps,
                        max_acceleration_mps2=m.limits.max_acceleration_mps2,
                        position_tolerance_m=m.execution_envelope.position_error_m))
                out.behavior_identity={k:r[k] for k in ('intent_id','stage','revision')}
                out.reason=result['reason_code']
                out.behavior_feedback=self._reply(context,r,p,deadline,'PLANNED',out.reason,
                                                 out.candidate_id,metrics=out.planning_metrics)
            require(self.clock()<out.valid_until,'DRIVING_SOURCE_EXPIRED_DURING_DISPATCH')
            return validate_output(out,Trajectory,d)
        except (ValueError,TypeError,KeyError,AttributeError,IndexError,OverflowError,_Failure) as error:
            out=Trajectory().bind(d);out.reason='BEHAVIOR_REQUEST:'+str(error);out.errors.append(out.reason)
            if isinstance(r,dict) and r.get('maneuver')=='EMERGENCY':out=self._brake(p,d)
            out.planning_metrics=_rejected_metrics(self.last_report,error)
            out.planning_metrics.update(fallback_brake_required=isinstance(r,dict) and r.get('maneuver')=='EMERGENCY',
                primary_motion_rejection=str(error))
            if context is not None:
                try:out.behavior_feedback=self._reply(context,r,p,deadline,'REJECTED',str(error),
                    cid=r.get('candidate_id'),metrics=out.planning_metrics)
                except (ValueError,TypeError,KeyError):pass
            return out
