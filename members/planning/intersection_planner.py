"""Finite selected-route candidate with whole-body passage time validation."""
import copy
import math
import time
from core.behavior_protocol import (GoalPose,CandidateReference,PlanningFeedback,behavior_frame,validate_runtime_request)
from core.behavior_environment import vehicle_footprint
from core.intersection_protocol import body_stations,route_length,observed_objects
from core.interfaces import Trajectory,DecisionTarget,TrajectoryPoint
from core.geometry import project_polyline,normalize_angle
from core.trajectory_validation import (VehicleGeometry,MotionLimits,ObstaclePrediction,ValidationBudget,validate_candidate)
from members.planning.behavior_planner import BehaviorPlanner,stations,at
from members.planning.lane_planner import build_trajectory


class IntersectionPlanner(object):
    def __init__(self):self.reset()
    def reset(self):
        self.context=None;self.counter=0;self.candidates={};self.adopted={}
        self.profile=BehaviorPlanner()

    def _waiting(self,p,decision,settings):
        request=decision.behavior_request;task=p.behavior_environment.intersection_task
        stop=task.entry_s_m-settings.front_offset_m-0.3
        points=self._candidate(p,request,task,settings,end_s=stop)
        self._geometry_report(p,request,task,settings,points)
        out=Trajectory().bind(decision);out.valid=True;out.points=points
        out.target_speed=0 if decision.stop_distance<0.15 else request.goal.speed_cap_mps
        out.stop_required=True;out.stop_distance=decision.stop_distance;out.precision_stop=True
        return out

    def _candidate(self,p,request,task,settings,end_s=None):
        projection=project_polyline(task.route,p.ego.x,p.ego.y)
        if projection is None or projection['distance']>0.5:raise ValueError('intersection route not reachable from actual pose')
        arc=stations(task.route);start=projection['s']
        extent=(arc[-1]-settings.front_offset_m-0.5 if end_s is None else end_s)-start
        if end_s is not None and extent<0.12 and p.ego.speed<0.05:
            return [TrajectoryPoint(p.ego.x,p.ego.y,0,p.ego.heading,0),TrajectoryPoint(p.ego.x,p.ego.y,0,p.ego.heading,0.1)]
        if extent<=0:raise ValueError('intersection finite route exhausted')
        base=at(task.route,arc,start);length=min(8.0,extent)
        error=(p.ego.x-base[0],p.ego.y-base[1])
        tangent=(math.cos(p.ego.heading)-math.cos(base[2]),math.sin(p.ego.heading)-math.sin(base[2]))
        raw=[]
        count=max(2,int(math.ceil(extent/0.5)))
        for i in range(count+1):
            distance=extent*i/count;point=at(task.route,arc,start+distance)
            u=min(1,distance/length);position=1-10*u**3+15*u**4-6*u**5;derivative=u-6*u**3+8*u**4-3*u**5
            raw.append((point[0]+error[0]*position+tangent[0]*length*derivative,
                        point[1]+error[1]*position+tangent[1]*length*derivative))
        return self.profile._profile(p,request,settings,raw,None)

    def _geometry_report(self,p,request,task,settings,points):
        frame=behavior_frame(p,time.monotonic());env=p.behavior_environment;coverage=env.coverage
        if not (task.verified(frame,p.observation_mode) and env.current(frame,p.observation_mode)
                and coverage.timing_verified(frame,p.observation_mode)):
            raise ValueError('junction permission coverage or execution timing unknown')
        vehicle=VehicleGeometry(settings.front_offset_m,settings.rear_offset_m,settings.half_width_m,settings.wheelbase_m)
        limits=MotionLimits(max(p.ego.speed,request.goal.speed_cap_mps)+0.1,settings.acceleration,settings.deceleration,
            settings.lateral_acceleration,settings.front_steer_max_rad,settings.behavior_steer_rate_rad_s,0.2,0.05)
        obstacles=[ObstaclePrediction(str(t.identifier),t.x,t.y,t.length+0.4,t.width+0.4,t.heading,t.vx,t.vy,
            coverage.prediction_horizon_s,coverage.position_uncertainty_m,coverage.velocity_uncertainty_mps,
            coverage.time_early_error_s,coverage.time_late_error_s) for t in observed_objects(p)]
        report=validate_candidate(points,1,vehicle,limits,task.drivable_region,obstacles,
            ValidationBudget(30000,14,0.002,min(p.valid_until,time.monotonic()+settings.behavior_validation_budget_ms/1000)))
        if report['status']!='safe':raise ValueError('junction path '+report['reason_code']+':'+report['constraint'])
        return report

    def _validate(self,p,request,task,settings,points):
        report=self._geometry_report(p,request,task,settings,points)
        frame=behavior_frame(p,time.monotonic());coverage=p.behavior_environment.coverage
        geom=dict(front_offset_m=settings.front_offset_m,rear_offset_m=settings.rear_offset_m,half_width_m=settings.half_width_m)
        if not task.exit_has_space(p,geom):raise ValueError('exit cannot contain whole vehicle and business gap')
        front=[];rear=[]
        for q in points:
            body=vehicle_footprint(GoalPose(q.x,q.y,q.heading),settings.front_offset_m,settings.rear_offset_m,settings.half_width_m)
            if not coverage.contains(body):raise ValueError('junction candidate outside observed coverage')
            lo,hi=body_stations(task.route,body);front.append((hi,q.relative_time));rear.append((lo,q.relative_time))
        def crossing(values,bound):
            if values[0][0]>=bound:return 0.0
            for a,b in zip(values,values[1:]):
                if a[0]<=bound<=b[0] and b[0]>a[0]:return a[1]+(b[1]-a[1])*(bound-a[0])/(b[0]-a[0])
            raise ValueError('whole body exit beyond candidate')
        interval=(frame.observed_at_s+max(0,crossing(front,task.entry_s_m)-coverage.time_early_error_s),
                  frame.observed_at_s+crossing(rear,task.exit_s_m)+coverage.time_late_error_s)
        if not any(a<=interval[0] and interval[1]<=b for a,b in task.open_windows(p,frame)):
            raise ValueError('whole body execution interval outside open junction window')
        return report,interval

    def plan(self,p,decision,settings,fallback):
        request=decision.behavior_request;frame=behavior_frame(p,time.monotonic())
        if self.context!=frame.context.key():self.reset();self.context=frame.context.key()
        try:
            from core.behavior_protocol import validate_generator_geometry
            validate_generator_geometry(p,settings,frame,('INTERSECTION','PATH','FEEDBACK'))
            validate_runtime_request(request,frame);task=p.behavior_environment.intersection_task
            if request.authorization=='HOLD':
                out=self._waiting(p,decision,settings);out.behavior_request=request;out.goal_pose=request.goal.goal_pose
                return out
            if request.authorization=='EVALUATE_ONLY':fallback=self._waiting(p,decision,settings)
            points=self._candidate(p,request,task,settings)
            report,interval=self._validate(p,request,task,settings,points)
            signature=(task.identity,task.route,task.entry_s_m,task.exit_s_m,tuple(task.drivable_region.points))
            if request.authorization=='EVALUATE_ONLY':
                self.counter+=1;ref=CandidateReference('junction:'+str(self.counter),request,frame.valid_until_s)
                self.candidates[ref.candidate_id]=(copy.deepcopy(ref),signature)
                while len(self.candidates)>16:self.candidates.pop(next(iter(self.candidates)))
                fallback.behavior_request=request
                fallback.planning_feedback=[PlanningFeedback(request,frame,'PLANNED',candidate_reference=ref,
                    clearance_lower_bound_m=report['clearance_lower_bound_m'],computation_s=report['elapsed_s'],occupancy_window_s=interval)]
                return fallback
            ref=request.candidate_reference;key=(frame.context.key(),request.intent_id,ref.candidate_id)
            stored=self.candidates.get(ref.candidate_id)
            if (stored is None or stored[0].key()!=ref.key() or stored[1]!=signature
                    or key not in self.adopted and time.monotonic()>=min(ref.valid_until_s,stored[0].valid_until_s)):
                raise ValueError('junction adoption expired absent or mismatched')
            self.adopted[key]=True
            out=Trajectory().bind(decision);out.valid=True;out.points=points
            out.target_speed=request.goal.speed_cap_mps;out.target_lane_id=decision.target_lane_id
            out.stop_required=True;out.stop_distance=stations([(q.x,q.y) for q in points])[-1]
            out.behavior_request=request;out.execution_path_id=ref.candidate_id
            out.planning_feedback=[PlanningFeedback(request,frame,'PLANNED',candidate_reference=ref,
                clearance_lower_bound_m=report['clearance_lower_bound_m'],computation_s=report['elapsed_s'],occupancy_window_s=interval)]
            return out
        except (ValueError,TypeError,AttributeError,IndexError,OverflowError) as exc:
            protected=copy.deepcopy(decision);protected.mode='EMERGENCY_BRAKE';protected.target_speed=0;protected.stop_distance=-1;protected.precision_stop=False
            fallback=build_trajectory(p,protected,settings);fallback.bind(decision)
            fallback.behavior_request=request
            fallback.planning_feedback=[PlanningFeedback(request,frame,'REJECTED',str(exc),retryable=True)]
            return fallback
