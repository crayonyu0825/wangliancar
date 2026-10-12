"""Bounded braking and quintic turn/brake candidates; explicit adoption only."""
import copy
import math
import time
from core.behavior_protocol import CandidateReference,PlanningFeedback,GoalPose,behavior_frame,validate_runtime_request
from core.interfaces import Trajectory,TrajectoryPoint
from core.emergency_protocol import EmergencyTask
from core.trajectory_validation import VehicleGeometry,MotionLimits,ObstaclePrediction,ValidationBudget,validate_candidate
from core.intersection_protocol import observed_objects
from core.behavior_environment import vehicle_footprint
from core.geometry import normalize_angle
from members.planning.behavior_planner import stations


class EmergencyPlanner(object):
    def __init__(self):self.reset()
    def reset(self):self.counter=0;self.candidates={};self.adopted={};self.context=None

    def _points(self,p,offset,settings,goal=None):
        if p.ego.speed<=0.05:
            return [TrajectoryPoint(p.ego.x,p.ego.y,0,p.ego.heading,0),TrajectoryPoint(p.ego.x,p.ego.y,0,p.ego.heading,0.1)]
        # Geometry spans a conservative braking distance. Revalidation starts
        # at the actual pose/speed on each fresh cycle, including early stops.
        extent=max(3.0,p.ego.speed**2/(2*settings.deceleration)+3.0,
            (60*abs(offset)*settings.wheelbase_m*p.ego.speed/settings.behavior_steer_rate_rad_s)**(1.0/3.0)*1.05)
        if goal is None:
            goal=GoalPose(p.ego.x+extent*math.cos(p.ego.heading)-offset*math.sin(p.ego.heading),
                          p.ego.y+extent*math.sin(p.ego.heading)+offset*math.cos(p.ego.heading),p.ego.heading)
        goalx,goaly=goal.x,goal.y
        ex,ey=goalx-p.ego.x,goaly-p.ego.y
        longitudinal=ex*math.cos(p.ego.heading)+ey*math.sin(p.ego.heading)
        lateral=-ex*math.sin(p.ego.heading)+ey*math.cos(p.ego.heading)
        if longitudinal<=1:raise ValueError('emergency candidate exhausted behind actual pose')
        count=max(2,int(math.ceil(longitudinal/0.1)));xy=[]
        for i in range(count+1):
            u=i/count;blend=10*u**3-15*u**4+6*u**5
            xy.append((p.ego.x+longitudinal*u*math.cos(p.ego.heading)-lateral*blend*math.sin(p.ego.heading),
                       p.ego.y+longitudinal*u*math.sin(p.ego.heading)+lateral*blend*math.cos(p.ego.heading)))
        arc=stations(xy);elapsed=0;result=[];values=[p.ego.speed*math.sqrt(max(0,1-s/arc[-1])) for s in arc]
        for i,(x,y) in enumerate(xy):
            a,b=xy[max(0,i-1)],xy[min(count,i+1)]
            heading=math.atan2(b[1]-a[1],b[0]-a[0]) if i else p.ego.heading
            if i:elapsed+=2*(arc[i]-arc[i-1])/(values[i]+values[i-1])
            result.append(TrajectoryPoint(x,y,values[i],heading,elapsed))
        return result

    def _validate(self,p,request,settings,points):
        task=p.behavior_environment.emergency_task;frame=behavior_frame(p,time.monotonic());coverage=p.behavior_environment.coverage
        if not isinstance(task,EmergencyTask) or not task.verified(frame,p.observation_mode) or not coverage.timing_verified(frame,p.observation_mode):
            raise ValueError('emergency permission or timing unknown')
        vehicle=VehicleGeometry(settings.front_offset_m,settings.rear_offset_m,settings.half_width_m,settings.wheelbase_m)
        limits=MotionLimits(max(0.1,p.ego.speed)+0.1,settings.acceleration,settings.deceleration,settings.lateral_acceleration,
            settings.front_steer_max_rad,settings.behavior_steer_rate_rad_s,0.2,0.05)
        obstacles=[ObstaclePrediction(str(o.identifier),o.x,o.y,o.length+0.2,o.width+0.2,o.heading,o.vx,o.vy,
            coverage.prediction_horizon_s,coverage.position_uncertainty_m,coverage.velocity_uncertainty_mps,
            coverage.time_early_error_s,coverage.time_late_error_s) for o in observed_objects(p)]
        if not all(coverage.contains(vehicle_footprint(GoalPose(q.x,q.y,q.heading),settings.front_offset_m,
            settings.rear_offset_m,settings.half_width_m)) for q in points):raise ValueError('emergency path outside coverage')
        reports=[]
        for region in task.allowed_regions:
            report=validate_candidate(points,1,vehicle,limits,region,obstacles,
                ValidationBudget(20000,12,0.002,min(p.valid_until,time.monotonic()+0.015)))
            reports.append(report)
            if report['status']=='safe':return report
        raise ValueError('emergency path '+reports[0]['reason_code']+':'+reports[0]['constraint'])

    def plan(self,p,decision,settings,fallback):
        request=decision.behavior_request;frame=behavior_frame(p,time.monotonic());task=p.behavior_environment.emergency_task
        if self.context!=frame.context.key():self.reset();self.context=frame.context.key()
        try:
            from core.behavior_protocol import validate_generator_geometry
            validate_generator_geometry(p,settings,frame,('EMERGENCY_CANDIDATES','PATH','FEEDBACK'))
            validate_runtime_request(request,frame)
            if p.ego.vx*math.cos(p.ego.heading)+p.ego.vy*math.sin(p.ego.heading)<-0.05:
                raise ValueError('reverse emergency keeps immediate brake; forward turn candidate unavailable')
            if not task.verified(frame,p.observation_mode):raise ValueError('emergency observation unavailable')
            if request.authorization=='EVALUATE_ONLY':
                feedback=[];deadline=time.perf_counter()+settings.behavior_validation_budget_ms/1000
                for offset in sorted(task.lateral_offsets_m,key=lambda v:(abs(v),v)):
                    if time.perf_counter()>=deadline:break
                    try:
                        points=self._points(p,offset,settings);report=self._validate(p,request,settings,points)
                        self.counter+=1;ref=CandidateReference('emergency:'+str(self.counter),request,frame.valid_until_s)
                        self.candidates[ref.candidate_id]=(copy.deepcopy(ref),task.danger_id,offset,GoalPose(points[-1].x,points[-1].y,points[-1].heading))
                        action='BRAKE' if offset==0 else 'BRAKE_LEFT' if offset>0 else 'BRAKE_RIGHT'
                        value=PlanningFeedback(request,frame,'PLANNED',candidate_reference=ref,
                            clearance_lower_bound_m=report['clearance_lower_bound_m'],computation_s=report['elapsed_s'],
                            goal_pose=GoalPose(points[-1].x,points[-1].y,points[-1].heading),candidate_action=action,
                            risk_cost=0.0,rule_cost=abs(offset))
                        feedback.append(value)
                    except (ValueError,TypeError) as exc:feedback.append(PlanningFeedback(request,frame,'REJECTED',str(exc)))
                while len(self.candidates)>16:self.candidates.pop(next(iter(self.candidates)))
                fallback.behavior_request=request;fallback.planning_feedback=feedback or [PlanningFeedback(request,frame,'REJECTED','EMERGENCY_BUDGET_EXHAUSTED')]
                return fallback
            ref=request.candidate_reference;stored=self.candidates.get(ref.candidate_id)
            key=(frame.context.key(),request.intent_id,ref.candidate_id)
            if (stored is None or stored[0].key()!=ref.key() or stored[1]!=task.danger_id or stored[2] not in task.lateral_offsets_m
                    or key not in self.adopted and time.monotonic()>=min(ref.valid_until_s,stored[0].valid_until_s)):
                raise ValueError('emergency adoption or danger identity invalid')
            # Keep the selected action, while rebuilding a verified brake path
            # from observed position and speed. No old throttle is restored.
            points=self._points(p,stored[2],settings,stored[3]);report=self._validate(p,request,settings,points);self.adopted[key]=True
            out=Trajectory().bind(decision);out.valid=True;out.points=points;out.target_speed=0
            out.emergency_stop=True;out.emergency_maneuver_authorized=True
            out.stop_required=True;out.stop_distance=stations([(q.x,q.y) for q in points])[-1]
            out.behavior_request=request;out.execution_path_id=ref.candidate_id;out.goal_pose=request.goal.goal_pose
            out.planning_feedback=[PlanningFeedback(request,frame,'PLANNED',candidate_reference=ref,
                clearance_lower_bound_m=report['clearance_lower_bound_m'],computation_s=report['elapsed_s'])]
            return out
        except (ValueError,TypeError,AttributeError,IndexError,OverflowError) as exc:
            fallback.behavior_request=request;fallback.planning_feedback=[PlanningFeedback(request,frame,'REJECTED',str(exc))]
            return fallback
