"""Production parking search/adoption, current direction segment and dwell."""
import copy
import bisect
import math
import time

from core.behavior_protocol import CandidateReference,PlanningFeedback,GoalPose,behavior_frame,validate_runtime_request
from core.behavior_environment import MotionObject,vehicle_footprint
from core.interfaces import Trajectory,TrajectoryPoint
from core.geometry import normalize_angle,project_polyline
from members.planning.candidate_validation import (VehicleGeometry,MotionLimits,ObstaclePrediction,ValidationBudget,validate_candidate)
from members.planning.parking_search import search_parking
from members.planning.behavior_planner import stations,at


class ParkingPlanner(object):
    def __init__(self):self.reset()
    def reset(self):
        self.context=None;self.counter=0;self.candidates={};self.adopted={}

    def _objects(self,p,task,settings):
        result=[]
        for t in p.targets:
            if t.id==task.ego_target_id:
                center=(p.ego.x+(settings.front_offset_m-settings.rear_offset_m)/2*math.cos(p.ego.heading),
                        p.ego.y+(settings.front_offset_m-settings.rear_offset_m)/2*math.sin(p.ego.heading))
                if math.hypot(t.x-center[0],t.y-center[1])>0.5:
                    raise ValueError('ego target identity does not match the observed body')
                continue
            length,width=t.length,t.width
            if t.heading is None:length=width=math.hypot(length,width)
            result.append(MotionObject(t.id,t.x,t.y,t.vx,t.vy,length,width,t.heading or 0.0))
        return result

    def _profile(self,p,poses,direction,settings,speed=0.7):
        cleaned=[]
        for pose in poses:
            if not cleaned or math.hypot(pose.x-cleaned[-1].x,pose.y-cleaned[-1].y)>1e-6:cleaned.append(pose)
        if len(cleaned)<2:
            q=cleaned[0]
            return [TrajectoryPoint(q.x,q.y,0,q.body_heading_rad,0),TrajectoryPoint(q.x,q.y,0,q.body_heading_rad,0.1)],0.0
        lengths=[0.0]
        for a,b in zip(cleaned,cleaned[1:]):lengths.append(lengths[-1]+math.hypot(b.x-a.x,b.y-a.y))
        caps=[max(speed,math.sqrt(max(0.0,p.ego.speed**2-2*settings.deceleration*s))) for s in lengths];caps[-1]=0
        for i in range(len(caps)-2,-1,-1):caps[i]=min(caps[i],math.sqrt(caps[i+1]**2+2*settings.deceleration*(lengths[i+1]-lengths[i])))
        values=[p.ego.speed]
        for i in range(1,len(caps)):values.append(min(caps[i],math.sqrt(values[-1]**2+2*settings.acceleration*(lengths[i]-lengths[i-1]))))
        result=[];elapsed=0
        for i,q in enumerate(cleaned):
            if i:
                denominator=values[i]+values[i-1]
                if denominator<=1e-8:raise ValueError('parking zero-speed segment cannot advance')
                elapsed+=2*(lengths[i]-lengths[i-1])/denominator
            result.append(TrajectoryPoint(q.x,q.y,values[i],q.body_heading_rad,elapsed))
        return result,lengths[-1]

    def _validate(self,p,request,settings,points,objects):
        coverage=p.behavior_environment.coverage;task=p.behavior_environment.parking_task
        frame=behavior_frame(p,time.monotonic())
        if not coverage.timing_verified(frame,p.observation_mode):raise ValueError('parking execution timing unavailable')
        vehicle=VehicleGeometry(settings.front_offset_m,settings.rear_offset_m,settings.half_width_m,settings.wheelbase_m)
        limits=MotionLimits(max(0.8,p.ego.speed),settings.acceleration,settings.deceleration,settings.lateral_acceleration,
                            settings.front_steer_max_rad,settings.behavior_steer_rate_rad_s,0.2,0.05)
        predictions=[ObstaclePrediction(str(t.identifier),t.x,t.y,t.length+0.2,t.width+0.2,t.heading,t.vx,t.vy,
            coverage.prediction_horizon_s,coverage.position_uncertainty_m,coverage.velocity_uncertainty_mps,
            coverage.time_early_error_s,coverage.time_late_error_s) for t in objects]
        reports=[]
        for region in task.free_regions:
            report=validate_candidate(points,request.goal.motion_direction,vehicle,limits,region,predictions,
                ValidationBudget(30000,14,0.002,min(p.valid_until,time.monotonic()+settings.behavior_validation_budget_ms/1000)))
            reports.append(report)
            if report['status']=='safe':return report
        raise ValueError('parking current segment rejected:'+reports[0]['reason_code']+':'+reports[0]['constraint'])

    def plan(self,p,decision,settings,fallback):
        request=decision.behavior_request;frame=behavior_frame(p,time.monotonic())
        if self.context!=frame.context.key():self.reset();self.context=frame.context.key()
        try:
            from core.behavior_protocol import validate_generator_geometry
            validate_generator_geometry(p,settings,frame,('PARK','PATH','FEEDBACK'))
            validate_runtime_request(request,frame)
            task=p.behavior_environment.parking_task
            if task is None or not task.verified(frame,p.observation_mode) or not p.behavior_environment.current(frame,p.observation_mode):
                raise ValueError('parking environment or task unknown')
            objects=self._objects(p,task,settings)
            if request.authorization=='HOLD':
                goal=request.goal.goal_pose
                if (p.ego.speed>0.05 or math.hypot(goal.x-p.ego.x,goal.y-p.ego.y)>0.15
                        or abs(normalize_angle(goal.body_heading_rad-p.ego.heading))>0.15):
                    raise ValueError('parking completion hold requires actual goal and stop')
                points=[TrajectoryPoint(p.ego.x,p.ego.y,0,p.ego.heading,0),
                        TrajectoryPoint(p.ego.x,p.ego.y,0,p.ego.heading,0.1)]
                self._validate(p,request,settings,points,objects)
                fallback=Trajectory().bind(decision);fallback.valid=True;fallback.points=points
                fallback.behavior_request=request;fallback.target_speed=0;fallback.stop_required=True
                fallback.stop_distance=0;fallback.goal_pose=goal;fallback.precision_stop=True
                fallback.motion_direction=request.goal.motion_direction
                return fallback
            if request.authorization=='EVALUATE_ONLY':
                goal=request.goal.goal_pose
                vehicle=VehicleGeometry(settings.front_offset_m,settings.rear_offset_m,settings.half_width_m,settings.wheelbase_m)
                segments,report=search_parking(GoalPose(p.ego.x,p.ego.y,p.ego.heading),goal,task.free_regions,objects,vehicle,
                    settings.front_steer_max_rad*0.75,budget_s=0.04,final_direction=request.goal.motion_direction)
                if not segments:raise ValueError(report['reason'])
                self.counter+=1;ref=CandidateReference('parking:'+str(self.counter),request,frame.valid_until_s)
                self.candidates[ref.candidate_id]=(copy.deepcopy(ref),segments,request.goal.parking_space_id)
                while len(self.candidates)>16:self.candidates.pop(next(iter(self.candidates)))
                fallback.behavior_request=request
                fallback.planning_feedback=[PlanningFeedback(request,frame,'PLANNED',candidate_reference=ref,
                    computation_s=report['elapsed_s'],goal_pose=goal,
                    segment_directions=[s.direction for s in segments],segment_end_poses=[s.end_pose for s in segments])]
                return fallback
            ref=request.candidate_reference;stored=self.candidates.get(ref.candidate_id)
            key=(frame.context.key(),request.intent_id,ref.candidate_id)
            index=request.segment_index
            if key not in self.adopted:
                if stored is None or stored[0].key()!=ref.key() or time.monotonic()>=min(stored[0].valid_until_s,ref.valid_until_s):
                    raise ValueError('parking adoption absent expired or mismatched')
                self.adopted[key]=0
            if stored is None or stored[2]!=request.goal.parking_space_id or type(index) is not int or not 0<=index<len(stored[1]):
                raise ValueError('parking segment identity invalid')
            previous=self.adopted[key]
            if index<previous or index>previous+1:raise ValueError('parking segment order invalid')
            if index==previous+1:
                endpoint=stored[1][previous].end_pose
                if p.ego.speed>0.05 or math.hypot(p.ego.x-endpoint.x,p.ego.y-endpoint.y)>0.15:
                    raise ValueError('parking direction change before actual stop')
                self.adopted[key]=index
            segment=stored[1][index];goal=segment.end_pose
            if (request.goal.motion_direction!=segment.direction or request.goal.goal_pose is None
                    or math.hypot(request.goal.goal_pose.x-goal.x,request.goal.goal_pose.y-goal.y)>1e-6
                    or abs(normalize_angle(request.goal.goal_pose.body_heading_rad-goal.body_heading_rad))>1e-6):
                raise ValueError('parking segment authority does not match candidate')
            raw=[(q.x,q.y) for q in segment.poses]
            projection=project_polyline(raw,p.ego.x,p.ego.y)
            if projection is None or projection['distance']>0.6:raise ValueError('parking execution left candidate')
            arc=stations(raw);start=projection['s'];remaining=arc[-1]-start
            def original_pose(value):
                value=max(0.0,min(arc[-1],value))
                k=max(0,min(len(raw)-2,bisect.bisect_right(arc,value)-1))
                u=(value-arc[k])/max(1e-8,arc[k+1]-arc[k])
                first,last=segment.poses[k],segment.poses[k+1]
                heading=first.body_heading_rad+u*normalize_angle(last.body_heading_rad-first.body_heading_rad)
                return (first.x+u*(last.x-first.x),first.y+u*(last.y-first.y),
                        heading+(math.pi if segment.direction<0 else 0))
            origin=original_pose(start)
            travel_heading=p.ego.heading+(math.pi if segment.direction<0 else 0)
            error=(p.ego.x-origin[0],p.ego.y-origin[1])
            derivative=(math.cos(travel_heading)-math.cos(origin[2]),math.sin(travel_heading)-math.sin(origin[2]))
            blend=min(3.0,remaining);count=max(2,int(math.ceil(remaining/0.1)))
            xy=[]
            for i in range(count+1):
                distance=remaining*i/count;base=original_pose(start+distance)
                u=min(1.0,distance/max(1e-8,blend));position=1-10*u**3+15*u**4-6*u**5
                tangent=u-6*u**3+8*u**4-3*u**5
                xy.append((base[0]+error[0]*position+derivative[0]*blend*tangent,
                           base[1]+error[1]*position+derivative[1]*blend*tangent))
            poses=[]
            for i,point in enumerate(xy):
                a,b=xy[max(0,i-1)],xy[min(len(xy)-1,i+1)]
                heading=math.atan2(b[1]-a[1],b[0]-a[0])-(math.pi if segment.direction<0 else 0)
                poses.append(GoalPose(point[0],point[1],normalize_angle(heading)))
            poses[0]=GoalPose(p.ego.x,p.ego.y,p.ego.heading);poses[-1]=goal
            to_goal=(goal.x-p.ego.x,goal.y-p.ego.y)
            close=math.hypot(*to_goal)
            heading_error=abs(normalize_angle(goal.body_heading_rad-p.ego.heading))
            if close<2.0 and heading_error<0.1:
                ux,uy=math.cos(travel_heading),math.sin(travel_heading)
                along=to_goal[0]*ux+to_goal[1]*uy
                lateral=abs(-to_goal[0]*uy+to_goal[1]*ux)
                if lateral<0.08 and along>=0:
                    count=max(2,int(math.ceil(along/0.1)))
                    poses=[GoalPose(p.ego.x+ux*along*j/count,p.ego.y+uy*along*j/count,p.ego.heading)
                           for j in range(count+1)]
            if close<0.1 and heading_error<0.1:
                poses=[GoalPose(p.ego.x,p.ego.y,p.ego.heading),GoalPose(p.ego.x,p.ego.y,p.ego.heading)]
            points,distance=self._profile(p,poses,segment.direction,settings)
            report=self._validate(p,request,settings,points,objects)
            out=Trajectory().bind(decision);out.valid=True;out.points=points
            out.target_speed=0.0 if distance<0.1 else 0.7
            out.motion_direction=segment.direction;out.stop_required=True;out.stop_distance=distance
            out.precision_stop=True;out.goal_pose=goal;out.behavior_request=request;out.execution_path_id=ref.candidate_id
            out.hold_duration_s=request.goal.minimum_standstill_duration_s
            out.parking_brake_at_stop=request.goal.parking_brake_at_stop;out.stop_obligation_id=request.goal.stop_obligation_id
            out.planning_feedback=[PlanningFeedback(request,frame,'PLANNED',candidate_reference=ref,
                clearance_lower_bound_m=report['clearance_lower_bound_m'],computation_s=report['elapsed_s'])]
            return out
        except (ValueError,TypeError,AttributeError,IndexError,OverflowError) as exc:
            fallback=Trajectory().bind(decision);fallback.valid=True;fallback.target_speed=0
            fallback.emergency_stop=True;fallback.stop_required=True;fallback.stop_distance=0
            fallback.points=[TrajectoryPoint(p.ego.x,p.ego.y,0,p.ego.heading,0),TrajectoryPoint(p.ego.x,p.ego.y,0,p.ego.heading,0.1)]
            fallback.behavior_request=request
            fallback.planning_feedback=[PlanningFeedback(request,frame,'REJECTED',str(exc),retryable=True)]
            return fallback
