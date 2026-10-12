"""P04 fixed-entry consumer of the original task/stage and explicit model.

No SDK, task selection, calibration or capability provider is installed here.
Every stage rechecks formal sources; actual feedback alone advances stages.
"""
import copy
import hashlib
import math
import time
from core.behavior_channel import read_channel,capabilities_dict,feedback_dict
from core.behavior_contract import BehaviorFrame,ExecutionFeedback,BehaviorGoal,GoalPose,require,finite
from core.interfaces import Trajectory,TrajectoryPoint,DecisionTarget,DecisionMode
from core.validation import validate_output
from core.geometry import normalize_angle,project_polyline,opposes_direction
from core.maneuver_facts import ManeuverFacts
from core.parking_mission import ParkingMission
from core.traffic_quality import signal_stop_requirement
from members.planning.behavior_contract import assess_behavior_request
from members.planning.maneuver_inputs import ManeuverPlanningInputs,read_inputs
from members.planning.parking_scene import plan_parking_candidate,read_parking_scene
from members.planning.lane_change_generator import _Work,_source_predictions,_Failure,_timed,_motion_parameters
from members.planning.candidate_validation import ValidationBudget
from members.planning.parking_generator import _search_parameters,_timed_parking


class ParkingPlanningInputs(ManeuverPlanningInputs):
    def __init__(self,mission,tolerances,*args,**kwargs):
        super(ParkingPlanningInputs,self).__init__(*args,**kwargs)
        self.mission=copy.deepcopy(mission)
        self.tolerances=tuple(tolerances)


class ParkingBehaviorPlanner(object):
    transitions={'APPROACH':('POSITION',),'POSITION':('REVERSE_ENTRY',),
        'REVERSE_ENTRY':('ALIGN','PARKED_DWELL'),'ALIGN':('PARKED_DWELL',),
        'PARKED_DWELL':('EXIT_PREPARE',),'EXIT_PREPARE':('EXIT',),'EXIT':()}

    def __init__(self,inputs_provider=None,clock=None):
        require(inputs_provider is None or callable(inputs_provider),'invalid parking input provider')
        self.provider,self.clock=inputs_provider,clock or time.monotonic
        self.context,self.records,self.segment_plans=None,{},{}

    @staticmethod
    def _near(x,y,yaw,pose,tolerances):
        return (math.hypot(x-pose.x,y-pose.y)<=tolerances[0]
                and abs(normalize_angle(yaw-pose.body_heading_rad))<=tolerances[1])

    @staticmethod
    def _reply(context,request,p,now,deadline,status,reason,candidate_id=None,options=None,metrics=None):
        return feedback_dict(ExecutionFeedback(context,request['intent_id'],request['stage'],request['revision'],
            'planning',p.frame_id,p.frame_id,now,deadline,status,usable=now<deadline,
            reason_code=reason,retryable=status=='REJECTED',candidate_id=candidate_id,
            candidate_options=options,planning_metrics=metrics))

    def _transition(self,p,request,previous,feedbacks,context):
        stage=request['stage']
        if previous is None:
            require(stage=='APPROACH','PARKING_APPROACH_REQUIRED')
            return
        old=previous['revision']; revision=(request['revision'],stage,request['source_frame_id'],request['issued_at_s'])
        require(revision==old or (revision[0]>old[0] and revision[2]>=old[2] and revision[3]>=old[3]),
                'REQUEST_REVISION_REGRESSED_OR_CHANGED')
        changed_goal=(request['goal_pose']!=previous['pose'].to_dict()
                      or request['motion_direction']!=previous['direction']
                      or (previous.get('segment_candidate') and
                          request.get('candidate_id')!=previous.get('candidate_id')))
        if stage==old[1] and not changed_goal: return
        if previous.get('segment_proposal'):
            require(stage=='ALIGN' and p.ego.speed<=previous['tolerances'][2]
                    and self._near(p.ego.x,p.ego.y,p.ego.heading,previous['proposal_start_pose'],previous['tolerances']),
                    'PARKING_INITIAL_SEGMENT_REQUIRES_ORIGINAL_STOPPED_POSITION')
            return
        require(previous['accepted'],'PARKING_PRIOR_STAGE_PATH_NOT_ACCEPTED')
        require(stage in self.transitions[old[1]] or (old[1]==stage=='ALIGN' and changed_goal),
                'PARKING_STAGE_SEQUENCE_CHANGED')
        now=self.clock(); pose=previous['pose']; tolerances=previous['tolerances']
        alignment=old[1]=='REVERSE_ENTRY' and stage=='ALIGN' and not previous.get('segment_candidate')
        require(p.ego.speed<=tolerances[2] and (alignment or self._near(p.ego.x,p.ego.y,p.ego.heading,pose,tolerances)),
                'PARKING_PRIOR_GOAL_NOT_ACTUALLY_STOPPED')
        for f in feedbacks:
            if (f.context.key()==context.key() and f.intent_id==request['intent_id']
                    and f.stage==old[1] and f.revision==old[0] and f.producer in ('control','runtime')
                    and f.usable and f.clock_id=='process_monotonic'
                    and old[2]<=f.source_frame_id<=f.produced_frame_id<=p.frame_id
                    and f.produced_at_s<=now< f.valid_until_s and now-f.produced_at_s<=.5
                    and f.actual_pose is not None and f.actual_standstill_confirmed
                    and f.motion_direction==previous['direction']
                    and (previous.get('candidate_id') is None or f.candidate_id==previous['candidate_id'])
                    and f.status not in ('FAILED','REJECTED','PLANNED')
                    and self._near(p.ego.x,p.ego.y,p.ego.heading,f.actual_pose,tolerances)
                    and ((alignment and f.progress>=1) or (f.goal_pose_arrived and
                        self._near(f.actual_pose.x,f.actual_pose.y,f.actual_pose.body_heading_rad,pose,tolerances)))
                    and (old[1]!='PARKED_DWELL' or (f.hold_completed
                        and f.standstill_duration_s>=previous['dwell']))):
                return
        raise ValueError('PARKING_MATCHED_ACTUAL_STAGE_FEEDBACK_REQUIRED')

    @staticmethod
    def _option(intent,index,sequence,mission,cap,deadline):
        segment=sequence[index]; endpoint=segment['goal']
        goal=BehaviorGoal(parking_space_id=mission.space_id,goal_pose=GoalPose(*endpoint),
                          motion_direction=segment['direction'],speed_cap_mps=cap,precision_stop=True,
                          candidate_id=segment['candidate_id'])
        return dict(candidate_id=segment['candidate_id'],action='PARK',goal=goal.to_dict(),feasible=True,
            risk_cost=segment['risk_cost'],rule_cost=segment['rule_cost'],validation_reason='P02_SEGMENT_CERTIFIED',
            source_valid_until_s=deadline,segment_id=segment['candidate_id'],segment_count=len(sequence))

    def _propose_segments(self,p,mission,model,request,result,binding,deadline):
        require(p.ego.speed<=model.tolerances[2] and request['stage'] in ('REVERSE_ENTRY','ALIGN'),
                'PARKING_MULTISEGMENT_INITIAL_STOP_REQUIRED')
        sequence=[]
        raw=result['maneuver_segments']
        require(1<=len(raw)<=128,'PARKING_SEGMENT_COUNT_INVALID')
        for index,segment in enumerate(raw):
            end=segment['points'][-1]
            token=repr((request['intent_id'],p.frame_id,index,end.x,end.y,end.heading,segment['motion_direction']))
            identifier='parking:'+hashlib.sha256(token.encode('utf-8')).hexdigest()[:24]
            gap=segment['access_report'].get('clearance_lower_bound_m')
            require(finite(gap) and gap>0,'PARKING_SEGMENT_CLEARANCE_UNPROVED')
            sequence.append(dict(goal=(end.x,end.y,end.heading),direction=segment['motion_direction'],
                candidate_id=identifier,risk_cost=1./gap,rule_cost=segment['stop_distance']))
        plan=dict(binding=binding,sequence=tuple(sequence),authorized_index=-1,offered_index=0,
                  offered_until=deadline,original_goal=mission.stage_pose('PARKED_DWELL',
                    ManeuverFacts(p,self.clock).parking(mission.space_id),model.vehicle.front_offset_m,
                    model.vehicle.rear_offset_m).to_dict())
        self.segment_plans[request['intent_id']]=plan
        return self._option(request['intent_id'],0,sequence,mission,request['speed_cap_mps'],deadline)

    def _authorized_segment(self,request,binding,now):
        plan=self.segment_plans.get(request.get('intent_id'))
        require(plan is not None and plan['binding']==binding,'PARKING_SEGMENT_PLAN_UNAVAILABLE')
        matching=[i for i,s in enumerate(plan['sequence']) if GoalPose(*s['goal']).to_dict()==request['goal_pose']
                  and s['direction']==request['motion_direction']
                  and s['candidate_id']==request.get('candidate_id')]
        index=plan['authorized_index']
        if index not in matching:
            index=plan['offered_index']
            require(index in matching and index==plan['authorized_index']+1 and now<plan['offered_until'],
                    'PARKING_SEGMENT_NOT_CURRENTLY_OFFERED')
        require(request.get('candidate_id')==plan['sequence'][index]['candidate_id'],
                'PARKING_SEGMENT_CANDIDATE_ID_MISMATCH')
        return plan,index,GoalPose(*plan['sequence'][index]['goal'])

    def _intermediate(self,request):
        plan=self.segment_plans.get(request.get('intent_id'))
        return bool(request.get('candidate_id') and plan is not None
                    and request['goal_pose']!=plan['original_goal'])

    def _offer_next(self,p,mission,model,request,plan,index,deadline):
        if index+1>=len(plan['sequence']): return []
        current=GoalPose(*plan['sequence'][index]['goal'])
        if p.ego.speed>model.tolerances[2] or not self._near(p.ego.x,p.ego.y,p.ego.heading,current,model.tolerances):
            return []
        next_segment=plan['sequence'][index+1]
        search=copy.deepcopy(model.search)
        if search.lattice_search is not None:
            search.lattice_search.permitted_directions=(next_segment['direction'],)
            search.lattice_search.max_direction_changes=0
        remaining=model.budget.max_checks
        work=ValidationBudget(remaining,model.budget.max_depth,model.budget.min_interval_s,
                             model.budget.deadline_monotonic_s)
        result=plan_parking_candidate(p,mission.space_id,'ALIGN',next_segment['goal'],next_segment['direction'],
            model.initial_curvature_m_inv,request['speed_cap_mps'],model.vehicle,model.limits,search,work,
            model.prediction_envelopes,mission.road_lane_ids,mission.access_edge_index,self.clock,
            model.geometry_cache,intermediate_segment=index+1<len(plan['sequence'])-1)
        model.budget.max_checks-=result.get('checks',0)
        if result['status']!='safe': return []
        fresh=min(deadline,result['source_valid_until_s'])
        require(self.clock()<fresh,'PARKING_NEXT_SEGMENT_OFFER_EXPIRED')
        plan['offered_index'],plan['offered_until']=index+1,fresh
        return [self._option(request['intent_id'],index+1,plan['sequence'],mission,request['speed_cap_mps'],fresh)]

    def _stationary(self,p,mission,model,in_bay):
        scene=read_parking_scene(p,mission.space_id,mission.road_lane_ids,bool(in_bay),
                                mission.access_edge_index,self.clock,model.geometry_cache)
        model.budget.deadline_monotonic_s=min(model.budget.deadline_monotonic_s,
            time.monotonic()+scene['source_valid_until_s']-self.clock())
        points=[TrajectoryPoint(p.ego.x,p.ego.y,0.,p.ego.heading,0.),
                TrajectoryPoint(p.ego.x,p.ego.y,0.,p.ego.heading,1.)]
        work=_Work(model.budget)
        predictions=_source_predictions(scene['objects'],model.prediction_envelopes,self.clock())
        checks=[(scene['access_corridor'],predictions),(scene['coverage'],[])]
        if in_bay is not None: checks.append((scene['goal_corridor'],[]))
        for region,objects in checks:
            result=work.validate(points,model.vehicle,model.limits,region,objects,-1 if in_bay else 1,
                                 model.execution_envelope)
            require(result['status']=='safe',result['reason_code'])
        return dict(status='safe',points=points,stop_distance=0.,frame_id=p.frame_id,
                    source_valid_until_s=scene['source_valid_until_s'],map_digest=scene['map_digest'],
                    checks=work.count,execution_robust=model.execution_envelope is not None)

    def _moving(self,p,mission,model,request,pose,cap,previous):
        # Keep the original rear-axle position. A stop accepts the explicitly
        # supplied heading tolerance, rather than demanding an exact tangent
        # which can require an impossible tiny turn near the target. Each of
        # at most eight nominal endpoints still needs the complete P04 proof.
        checks=0
        if (previous is not None and previous['revision'][1]==request['stage'] and previous['accepted']
                and len(previous['reference'])>=3):
            retained=self._remaining(p,mission,model,request,cap,previous['reference'])
            checks=retained.get('checks',0)
            if retained['status']=='safe' or retained.get('reason_code')!='NO_CERTIFIED_PARKING_REMAINDER':
                return retained
            # A tiny exact-point reconnect can be impossible despite a real
            # stoppable path into the task's original arrival region. Capture
            # uses measured curvature, never a fabricated aligned pose.
            capture=self._capture(p,mission,model,request,pose,cap,checks)
            checks+=capture.get('checks',0)
            if capture['status']=='safe':
                capture['checks']=checks
                return capture
            if capture.get('reason_code')=='BUDGET_EXHAUSTED': return capture
        headings=[pose.body_heading_rad]
        if previous is not None and previous['revision'][1]==request['stage'] and previous['accepted']:
            chosen=previous['endpoint_heading']
            if abs(normalize_angle(chosen-pose.body_heading_rad))<=model.tolerances[1]:
                headings=[chosen] if chosen==pose.body_heading_rad else [chosen,pose.body_heading_rad]
        chord=math.atan2(pose.y-p.ego.y,pose.x-p.ego.x)
        if request['motion_direction']==-1: chord=normalize_angle(chord+math.pi)
        for heading in (p.ego.heading,chord,pose.body_heading_rad-model.tolerances[1]*.5,
                         pose.body_heading_rad+model.tolerances[1]*.5,
                         pose.body_heading_rad-model.tolerances[1]*(1.-1e-12),
                         pose.body_heading_rad+model.tolerances[1]*(1.-1e-12)):
            if (abs(normalize_angle(heading-pose.body_heading_rad))<=model.tolerances[1]
                    and all(abs(normalize_angle(heading-v))>1e-9 for v in headings)):
                headings.append(heading)
        # Bay entry must retain the task's nominal terminal tangent. Spending
        # the complete heading tolerance in a candidate leaves no tracking
        # margin and can strand the actual car outside the original goal.
        if request['stage'] in ('REVERSE_ENTRY','ALIGN'):
            headings=[pose.body_heading_rad]
        for heading in headings:
            if checks>=model.budget.max_checks:
                return dict(status='inconclusive',reason_code='BUDGET_EXHAUSTED',checks=checks,points=[])
            work=ValidationBudget(model.budget.max_checks-checks,model.budget.max_depth,
                model.budget.min_interval_s,model.budget.deadline_monotonic_s)
            result=plan_parking_candidate(p,mission.space_id,request['stage'],(pose.x,pose.y,heading),
                request['motion_direction'],model.initial_curvature_m_inv,cap,model.vehicle,model.limits,
                model.search,work,model.prediction_envelopes,mission.road_lane_ids,
                mission.access_edge_index,self.clock,model.geometry_cache,
                intermediate_segment=self._intermediate(request))
            checks+=result.get('checks',0)
            if result['status']=='safe':
                result['checks']=checks
                return result
            if result.get('reason_code')!='NO_CERTIFIED_PARKING_SEGMENT' or checks>=model.budget.max_checks:
                break
        result['checks']=checks
        return result

    def _capture(self,p,mission,model,request,pose,cap,used):
        report=dict(status='unsafe',reason_code='PARKING_ARRIVAL_REGION_UNREACHABLE',points=[],checks=0)
        curvature=model.initial_curvature_m_inv; direction=request['motion_direction']
        dx,dy=pose.x-p.ego.x,pose.y-p.ego.y
        if abs(curvature)<1e-9:
            distance=direction*(dx*math.cos(p.ego.heading)+dy*math.sin(p.ego.heading))
        else:
            cx=p.ego.x-math.sin(p.ego.heading)/curvature
            cy=p.ego.y+math.cos(p.ego.heading)/curvature
            heading=math.atan2(curvature*(pose.x-cx),-curvature*(pose.y-cy))
            distance=normalize_angle(heading-p.ego.heading)/(direction*curvature)
        # Only a local approach, never a full circle/loop or a distant shortcut.
        if not 1e-9<distance<=model.vehicle.front_offset_m+model.vehicle.rear_offset_m: return report
        def at(s):
            heading=p.ego.heading+direction*curvature*s
            if abs(curvature)<1e-9:
                return (p.ego.x+direction*s*math.cos(p.ego.heading),
                        p.ego.y+direction*s*math.sin(p.ego.heading),p.ego.heading)
            return (p.ego.x+(math.sin(heading)-math.sin(p.ego.heading))/curvature,
                    p.ego.y+(math.cos(p.ego.heading)-math.cos(heading))/curvature,normalize_angle(heading))
        end=at(distance)
        if not self._near(end[0],end[1],end[2],pose,model.tolerances): return report
        work=None
        try:
            require(used<model.budget.max_checks,'BUDGET_EXHAUSTED')
            scene=read_parking_scene(p,mission.space_id,mission.road_lane_ids,
                None if self._intermediate(request) else request['stage'] in ('REVERSE_ENTRY','ALIGN'),
                mission.access_edge_index,self.clock,model.geometry_cache)
            remaining=ValidationBudget(model.budget.max_checks-used,model.budget.max_depth,
                model.budget.min_interval_s,min(model.budget.deadline_monotonic_s,
                    time.monotonic()+scene['source_valid_until_s']-self.clock()))
            work=_Work(remaining)
            count=max(2,int(math.ceil(distance/model.search.spacing_m)))
            require(count+1<=model.search.max_points,'CANDIDATE_POINT_LIMIT')
            shape=[]
            for i in range(count+1): work.step(); shape.append(at(distance*i/count))
            shape[0]=(p.ego.x,p.ego.y,p.ego.heading)
            lengths=[math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(shape,shape[1:])]
            predictions=_source_predictions(scene['objects'],model.prediction_envelopes,self.clock())
            points,timing=_timed_parking(shape,lengths,p.ego.speed,cap,cap,curvature,model.vehicle,model.limits,
                          model.search,work,direction,scene['access_corridor'],scene['coverage'],predictions)
            for region,objects,path in ((scene['access_corridor'],predictions,points),
                (scene['coverage'],[],points),(scene['goal_corridor'],[],
                    [TrajectoryPoint(end[0],end[1],0.,end[2],t) for t in (0.,1.)])):
                result=work.validate(path,model.vehicle,model.limits,region,objects,direction,model.execution_envelope)
                if result['status']!='safe':
                    report['reason_code']=result['reason_code']; return report
            report.update(status='safe',reason_code='PARKING_ORIGINAL_ARRIVAL_REGION_CAPTURE',
                points=points,stop_distance=sum(lengths),source_valid_until_s=scene['source_valid_until_s'],
                map_digest=scene['map_digest'],frame_id=p.frame_id,execution_robust=timing['execution_robust'])
        except (_Failure,ValueError) as error:
            report['reason_code']=error.code if isinstance(error,_Failure) else str(error)
        finally: report['checks']=work.count if work else 0
        return report

    def _remaining(self,p,mission,model,request,cap,reference):
        """Keep the selected segment, bridge actual GPS, re-time and re-prove.

        Old times, speeds, visibility and object checks are never reused.
        Four spatial anchor choices use the existing search resolution; P02
        checks the entire actual-to-reference bridge and remaining path.
        """
        in_bay=None if self._intermediate(request) else request['stage'] in ('REVERSE_ENTRY','ALIGN')
        scene=read_parking_scene(p,mission.space_id,mission.road_lane_ids,in_bay,
                                mission.access_edge_index,self.clock,model.geometry_cache)
        model.budget.deadline_monotonic_s=min(model.budget.deadline_monotonic_s,
            time.monotonic()+scene['source_valid_until_s']-self.clock())
        work=_Work(model.budget)
        report=dict(status='unsafe',reason_code='NO_CERTIFIED_PARKING_REMAINDER',points=[],checks=0)
        try:
            _motion_parameters((p.ego.x,p.ego.y,p.ego.heading),p.ego.speed,model.initial_curvature_m_inv,
                               cap,model.vehicle,model.limits,model.budget,allow_initial_overspeed=True)
            projection=project_polyline(reference,p.ego.x,p.ego.y)
            along=[0.]
            for a,b in zip(reference,reference[1:]):
                work.step(); along.append(along[-1]+math.hypot(b[0]-a[0],b[1]-a[1]))
            predictions=_source_predictions(scene['objects'],model.prediction_envelopes,self.clock())
            tried=set()
            for scale in (1.,2.,4.,8.):
                anchor=projection['index']+1
                while anchor<len(reference)-1 and along[anchor]<projection['s']+scale*model.search.spacing_m:
                    work.step(); anchor+=1
                if anchor in tried: continue
                tried.add(anchor)
                shape=[(p.ego.x,p.ego.y,p.ego.heading)]+list(reference[anchor:])
                if len(shape)==2:
                    # Preserve a launch station in a short retained tail.
                    # Two points (tiny measured speed -> zero) otherwise
                    # encode braking only and strand the real controller
                    # outside the unchanged arrival region. This refines the
                    # same chord; the current speed and P02 proof stay intact.
                    a,b=shape
                    shape.insert(1,((a[0]+b[0])*.5,(a[1]+b[1])*.5,
                        normalize_angle(a[2]+.5*normalize_angle(b[2]-a[2]))))
                if len(shape)>model.search.max_points: continue
                lengths=[math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(shape,shape[1:])]
                if not lengths or min(lengths)<=1e-9: continue
                try:
                    points,timing=_timed_parking(shape,lengths,p.ego.speed,cap,cap,model.initial_curvature_m_inv,
                        model.vehicle,model.limits,model.search,work,request['motion_direction'],
                        scene['access_corridor'],scene['coverage'],predictions)
                    legal=work.validate(points,model.vehicle,model.limits,scene['access_corridor'],
                                        predictions,request['motion_direction'],model.execution_envelope)
                    if legal['status']!='safe':
                        require(legal['reason_code']!='BUDGET_EXHAUSTED','BUDGET_EXHAUSTED')
                        continue
                    visible=work.validate(points,model.vehicle,model.limits,scene['coverage'],[],request['motion_direction'],
                                          model.execution_envelope)
                    if visible['status']!='safe':
                        require(visible['reason_code']!='BUDGET_EXHAUSTED','BUDGET_EXHAUSTED')
                        continue
                    end=points[-1]
                    probe=[TrajectoryPoint(end.x,end.y,0.,end.heading,at) for at in (0.,1.)]
                    target=work.validate(probe,model.vehicle,model.limits,scene['goal_corridor'],[],request['motion_direction'],
                                         model.execution_envelope)
                    if target['status']!='safe':
                        require(target['reason_code']!='BUDGET_EXHAUSTED','BUDGET_EXHAUSTED')
                        continue
                    report.update(status='safe',reason_code='PARKING_REMAINDER_READY',points=points,
                        stop_distance=sum(lengths),source_valid_until_s=scene['source_valid_until_s'],
                        map_digest=scene['map_digest'],frame_id=p.frame_id,retained_reference=True,
                        execution_robust=timing['execution_robust'])
                    break
                except _Failure as error:
                    if error.code=='BUDGET_EXHAUSTED': raise
        except _Failure as error:
            report.update(status=error.status,reason_code=error.code)
        finally: report['checks']=work.count
        return report

    def plan(self,p,decision,settings):
        output=Trajectory().bind(decision); context=None
        request=decision.behavior_request; deadline=min(p.valid_until,decision.valid_until)
        try:
            validate_output(decision,DecisionTarget,p)
            now=self.clock(); context,caps,feedbacks=read_channel(p,now)
            assessment=assess_behavior_request(request,context.to_dict(),p.frame_id,now,deadline,
                'process_monotonic',decision.behavior_active_identity,capabilities=capabilities_dict(caps),
                supported_actions=('PARK',),current_lane_id=p.lane.lane_id,
                frame_usable=p.valid,frame_paused=False,runtime_parking=True)
            require(assessment['eligible'],assessment['reason_code']); request=assessment['request']
            stage=request['stage']; hold=stage in ('PARKED_DWELL','EXIT_PREPARE')
            require(decision.mode!=DecisionMode.EMERGENCY_BRAKE and
                ((hold and decision.mode in (DecisionMode.STOP,DecisionMode.KEEP_LANE)) or
                 (not hold and decision.mode in (DecisionMode.KEEP_LANE,DecisionMode.FOLLOW)
                    and decision.target_speed>0 and decision.stop_distance==-1 and not decision.precision_stop)),
                'PARKING_OVERRIDDEN_BY_STOP')
            require(signal_stop_requirement(p) is None,'PARKING_SIGNAL_STOP_REQUIRED')
            model=read_inputs(self.provider,p,context,self.clock,'PARKING',ParkingPlanningInputs)
            model.search.execution_envelope=copy.deepcopy(model.execution_envelope)
            model.search.dynamic_search=copy.deepcopy(model.dynamic_search)
            _search_parameters(model.search)
            _motion_parameters((p.ego.x,p.ego.y,p.ego.heading),p.ego.speed,model.initial_curvature_m_inv,
                request['speed_cap_mps'],model.vehicle,model.limits,model.budget,allow_initial_overspeed=True)
            require(all(finite(v) for v in (p.ego.vx,p.ego.vy))
                and not opposes_direction(p.ego.vx,p.ego.vy,p.ego.heading,p.ego.speed,request['motion_direction']),
                'ACTUAL_MOTION_OPPOSES_PARKING_STAGE')
            require(isinstance(model.tolerances,tuple) and len(model.tolerances)==3
                    and all(finite(v) and v>0 for v in model.tolerances),'PARKING_ARRIVAL_MODEL_UNAVAILABLE')
            require(isinstance(model.mission,ParkingMission),'PARKING_TASK_MISSION_UNAVAILABLE')
            mission=copy.deepcopy(model.mission).snapshot()
            frame=BehaviorFrame(context,p.frame_id,self.clock(),p.valid_until,p.ego.x,p.ego.y,p.ego.heading,
                                p.ego.speed,usable=p.valid)
            require(mission.evidence.verified(frame,allowed_source_kinds=('task','verified_fusion')),
                    'PARKING_TASK_MISSION_UNVERIFIED')
            facts=ManeuverFacts(p,self.clock); digest=facts.map_digest()
            require(mission.map_digest==digest,'PARKING_MISSION_MAP_MISMATCH')
            bay=facts.parking(mission.space_id)
            original_pose=mission.stage_pose(stage,bay,model.vehicle.front_offset_m,model.vehicle.rear_offset_m)
            require(request['parking_space_id']==mission.space_id,'PARKING_ORIGINAL_STAGE_GOAL_MISMATCH')
            for name,actual in (('front_offset_m',model.vehicle.front_offset_m),
                ('rear_offset_m',model.vehicle.rear_offset_m),('half_width_m',model.vehicle.half_width_m),
                ('wheelbase_m',model.vehicle.wheelbase_m),('front_steer_max_rad',model.limits.max_front_steer_rad)):
                supplied=getattr(settings,name)
                require(supplied is None or abs(supplied-actual)<=1e-9,'PARKING_CONTROL_GEOMETRY_MISMATCH')
            binding=(mission.binding(bay),model.model_id,tuple(sorted(vars(model.vehicle).items())),
                     tuple(sorted(vars(model.limits).items())),model.tolerances,
                     None if model.execution_envelope is None else tuple(sorted(vars(model.execution_envelope).items())),
                     model.speed_error_mps)
            if self.context!=context.key(): self.context,self.records,self.segment_plans=context.key(),{},{}
            intent=request['intent_id']; previous=self.records.get(intent)
            require(previous is None or previous['binding']==binding,'PARKING_ACTIVE_BINDING_CHANGED')
            require(previous is not None or len(self.records)<128,'PARKING_SESSION_CAPACITY')
            sequence,index,candidate_id=None,None,None
            if request.get('candidate_id') is not None:
                require(stage=='ALIGN','PARKING_SEGMENT_REQUIRES_ALIGNMENT_STAGE')
                sequence,index,pose=self._authorized_segment(request,binding,now)
                candidate_id=sequence['sequence'][index]['candidate_id']
            else:
                pose=original_pose
                require(request['goal_pose']==pose.to_dict(),'PARKING_ORIGINAL_STAGE_GOAL_MISMATCH')
            self._transition(p,request,previous,feedbacks,context)
            if sequence is not None: sequence['authorized_index']=index
            deadline=min(deadline,request['valid_until_s'],caps.valid_until_s,model.valid_until_s,
                         mission.evidence.valid_until_s)
            model.budget.deadline_monotonic_s=min(model.budget.deadline_monotonic_s,
                time.monotonic()+deadline-self.clock())
            stopped=p.ego.speed<=model.tolerances[2] and self._near(p.ego.x,p.ego.y,p.ego.heading,pose,model.tolerances)
            require(not hold or stopped,'PARKING_HOLD_REQUIRES_ACTUAL_PARKED_POSE')
            if stopped:
                result=self._stationary(p,mission,model,None if self._intermediate(request)
                    else stage not in ('APPROACH','POSITION','EXIT'))
            else:
                result=self._moving(p,mission,model,request,pose,min(decision.target_speed,request['speed_cap_mps']),previous)
            if result.get('reason_code')=='PARKING_SEGMENT_AUTHORIZATION_REQUIRED':
                deadline=min(deadline,result['source_valid_until_s'])
                option=self._propose_segments(p,mission,model,request,result,binding,deadline)
                remaining=model.budget.max_checks-result.get('checks',0)
                require(remaining>0,'BUDGET_EXHAUSTED')
                model.budget.max_checks=remaining
                hold_result=self._stationary(p,mission,model,None)
                deadline=min(deadline,hold_result['source_valid_until_s'])
                require(self.clock()<deadline,'PARKING_SEGMENT_PROPOSAL_EXPIRED')
                option['source_valid_until_s']=deadline
                self.segment_plans[intent]['offered_until']=deadline
                output.reason='PARKING_SEGMENT_AUTHORIZATION_REQUIRED'
                output.valid_until=deadline; output.points=hold_result['points']; output.target_speed=0.
                output.motion_direction=request['motion_direction']; output.precision_stop=True
                output.stop_required=True; output.stop_distance=0.; output.valid=True
                output.planning_metrics={'parking_segment_proposal':True}
                output.behavior_feedback=self._reply(context,request,p,self.clock(),deadline,'REJECTED',
                    output.reason,options=[option],metrics={'parking_segment_proposal':True})
                self.records[intent]=dict(binding=binding,revision=(request['revision'],stage,
                    request['source_frame_id'],request['issued_at_s']),pose=pose,dwell=0.,
                    tolerances=model.tolerances,direction=request['motion_direction'],accepted=False,
                    segment_proposal=True,proposal_start_pose=GoalPose(p.ego.x,p.ego.y,p.ego.heading))
                validate_output(output,Trajectory,decision)
                return output
            preparing=(result['status']!='safe' and result.get('reason_code')=='NO_CERTIFIED_PARKING_SEGMENT'
                and p.ego.speed<=model.tolerances[2] and model.initial_curvature_m_inv!=0.)
            if preparing:
                remaining=model.budget.max_checks-result.get('checks',0)
                require(remaining>0,'BUDGET_EXHAUSTED')
                model.budget.max_checks=remaining
                result=self._stationary(p,mission,model,None)
            require(result['status']=='safe',result.get('reason_code','PARKING_PATH_UNAVAILABLE'))
            facts.check_current(True)
            require(facts.map_digest()==digest and mission.binding(facts.parking(mission.space_id))==binding[0],
                    'PARKING_BINDING_CHANGED_DURING_DISPATCH')
            validate_output(decision,DecisionTarget,p)
            deadline=min(deadline,result['source_valid_until_s']); now=self.clock()
            require(now<deadline,'PARKING_SOURCE_EXPIRED_DURING_DISPATCH')
            if preparing:
                # A real zero-speed/body/view/object checked hold lets the
                # existing controller centre its steering. Never acknowledge
                # the unproved parking path or attach an arrival/dwell duty.
                output.valid_until=deadline; output.points=result['points']
                output.motion_direction=request['motion_direction']; output.precision_stop=True
                output.stop_required=True; output.stop_distance=0.; output.valid=True
                output.reason='PARKING_STOPPED_STEERING_PREPARATION'
                output.planning_metrics={'parking_stopped_steering_preparation':True}
                validate_output(output,Trajectory,decision)
                self.records[intent]=dict(binding=binding,revision=(request['revision'],stage,
                    request['source_frame_id'],request['issued_at_s']),pose=pose,dwell=0.,
                    tolerances=model.tolerances,direction=request['motion_direction'],accepted=False)
                return output
            output.valid_until=deadline; output.points=result['points']
            output.motion_direction=request['motion_direction']; output.precision_stop=True
            output.target_speed=0. if stopped else min(decision.target_speed,request['speed_cap_mps'])
            output.stop_required=True; output.stop_distance=result['stop_distance']
            output.hold_duration_s=request['minimum_standstill_duration_s']
            output.parking_brake_at_stop=request['parking_brake_at_stop']
            output.stop_obligation_id=request['stop_obligation_id'] or ''
            output.behavior_identity={key:request[key] for key in ('intent_id','stage','revision')}
            model.budget.max_checks-=result.get('checks',0)
            options=self._offer_next(p,mission,model,request,sequence,index,deadline) if sequence is not None else []
            if candidate_id is None:
                # The same authorized stage keeps one ID and original goal
                # across replans. A new GPS frame is not a new progress task.
                token=repr((intent,request['revision'],stage,request['goal_pose'],request['motion_direction']))
                candidate_id='parking:'+hashlib.sha256(token.encode('utf-8')).hexdigest()[:24]
            output.candidate_id=candidate_id
            output.planning_metrics=dict(parking_segment=sequence is not None,
                segment_index=index,segment_count=len(sequence['sequence']) if sequence else 1,
                original_parking_goal=original_pose.to_dict(),checks=result.get('checks',0),
                execution_robust=bool(result.get('execution_robust',False)))
            if model.execution_envelope is not None and model.execution_envelope.position_error_m>0:
                output.planning_metrics['progress_bounds']=dict(max_speed_mps=model.limits.max_speed_mps,
                    max_acceleration_mps2=model.limits.max_acceleration_mps2,
                    position_tolerance_m=model.execution_envelope.position_error_m)
            output.behavior_feedback=self._reply(context,request,p,now,deadline,'PLANNED','PARKING_STAGE_PATH_READY',
                candidate_id=candidate_id,options=options,metrics=output.planning_metrics)
            output.reason='SOURCE_BOUND_PARKING_STAGE'; output.valid=True
            validate_output(output,Trajectory,decision)
            self.records[intent]=dict(binding=binding,revision=(request['revision'],stage,request['source_frame_id'],
                request['issued_at_s']),pose=pose,dwell=request['minimum_standstill_duration_s'],
                tolerances=model.tolerances,direction=request['motion_direction'],accepted=True,
                candidate_id=candidate_id,segment_candidate=sequence is not None,
                endpoint_heading=output.points[-1].heading,reference=(previous['reference']
                    if result.get('retained_reference') else tuple((v.x,v.y,v.heading) for v in output.points)))
            return output
        except (_Failure,ValueError,TypeError,AttributeError,KeyError,OverflowError) as error:
            output=Trajectory().bind(decision); output.reason='BEHAVIOR_REQUEST:'+str(error)
            output.errors.append(output.reason)
            if context is not None and isinstance(request,dict):
                try: output.behavior_feedback=self._reply(context,request,p,self.clock(),deadline,'REJECTED',str(error))
                except (ValueError,TypeError,KeyError): pass
            return output
