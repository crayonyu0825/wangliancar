"""Additional, independent checks for explicitly authorized parking paths.

This guard only adds refusal conditions. Existing source, collision, lane,
traffic, expiry and control checks remain in force in the supervisor.
"""
import time
import math
from core.behavior_protocol import RuntimeRequest,behavior_frame,validate_runtime_request
from core.validation import current
from core.trajectory_validation import (VehicleGeometry,MotionLimits,ObstaclePrediction,ValidationBudget,validate_candidate)


def parking_path_is_safe(p,decision,trajectory,settings):
    request=trajectory.behavior_request
    if (not isinstance(request,RuntimeRequest) or request.maneuver!='PARK'
            or request.authorization!='EXECUTE_SELECTED'):
        return True
    if (settings is None or trajectory.execution_path_id!=request.candidate_reference.candidate_id
            or not current(decision,p) or not current(trajectory,decision)):
        return False
    try:
        frame=behavior_frame(p,time.monotonic());validate_runtime_request(request,frame)
        env=p.behavior_environment;task=env.parking_task
        if not task.verified(frame,p.observation_mode) or not env.current(frame,p.observation_mode):return False
        coverage=env.coverage
        if not coverage.timing_verified(frame,p.observation_mode):return False
        vehicle=VehicleGeometry(settings.front_offset_m,settings.rear_offset_m,settings.half_width_m,settings.wheelbase_m)
        limits=MotionLimits(max(0.8,p.ego.speed),settings.acceleration,settings.deceleration,
            settings.lateral_acceleration,settings.front_steer_max_rad,settings.behavior_steer_rate_rad_s,0.2,0.05)
        objects=[]
        for t in p.targets:
            if t.id==task.ego_target_id:
                offset=(settings.front_offset_m-settings.rear_offset_m)/2
                if math.hypot(t.x-p.ego.x-offset*math.cos(p.ego.heading),
                              t.y-p.ego.y-offset*math.sin(p.ego.heading))>0.5:return False
                continue
            objects.append(ObstaclePrediction(str(t.id),t.x,t.y,t.length+0.2,t.width+0.2,t.heading,t.vx,t.vy,
                coverage.prediction_horizon_s,coverage.position_uncertainty_m,coverage.velocity_uncertainty_mps,
                coverage.time_early_error_s,coverage.time_late_error_s))
        return any(validate_candidate(trajectory.points,trajectory.motion_direction,vehicle,limits,region,objects,
            ValidationBudget(30000,14,0.002,min(p.valid_until,time.monotonic()+0.025)))['status']=='safe' for region in task.free_regions)
    except (AttributeError,ValueError,TypeError,IndexError,OverflowError):return False


def intersection_path_is_safe(p,decision,trajectory,settings):
    request=trajectory.behavior_request
    if (not isinstance(request,RuntimeRequest) or request.maneuver not in ('INTERSECTION','CROSSWALK','JUNCTION_ENTRY')
            or request.authorization!='EXECUTE_SELECTED'):return True
    try:
        frame=behavior_frame(p,time.monotonic());validate_runtime_request(request,frame)
        if settings is None or not current(trajectory,decision) or not current(decision,p):return False
        if (decision.behavior_request.identity()!=request.identity()
                or trajectory.execution_path_id!=request.candidate_reference.candidate_id):return False
        task=p.behavior_environment.intersection_task;coverage=p.behavior_environment.coverage
        if not task.verified(frame,p.observation_mode) or not coverage.timing_verified(frame,p.observation_mode):return False
        from core.intersection_protocol import observed_objects,body_stations
        from core.behavior_environment import vehicle_footprint
        from core.behavior_protocol import GoalPose
        geometry=dict(front_offset_m=settings.front_offset_m,rear_offset_m=settings.rear_offset_m,half_width_m=settings.half_width_m)
        if not task.exit_has_space(p,geometry):return False
        vehicle=VehicleGeometry(settings.front_offset_m,settings.rear_offset_m,settings.half_width_m,settings.wheelbase_m)
        limits=MotionLimits(max(request.goal.speed_cap_mps,p.ego.speed)+0.1,settings.acceleration,settings.deceleration,
            settings.lateral_acceleration,settings.front_steer_max_rad,settings.behavior_steer_rate_rad_s,0.2,0.05)
        predictions=[ObstaclePrediction(str(o.identifier),o.x,o.y,o.length+0.4,o.width+0.4,o.heading,o.vx,o.vy,
            coverage.prediction_horizon_s,coverage.position_uncertainty_m,coverage.velocity_uncertainty_mps,
            coverage.time_early_error_s,coverage.time_late_error_s) for o in observed_objects(p)]
        report=validate_candidate(trajectory.points,1,vehicle,limits,task.drivable_region,predictions,
            ValidationBudget(30000,14,0.002,min(p.valid_until,time.monotonic()+0.025)))
        if report['status']!='safe':return False
        values=[]
        for q in trajectory.points:
            body=vehicle_footprint(GoalPose(q.x,q.y,q.heading),settings.front_offset_m,settings.rear_offset_m,settings.half_width_m)
            if not coverage.contains(body):return False
            values.append((body_stations(task.route,body),q.relative_time))
        def crossing(bound,index):
            if values[0][0][index]>=bound:return 0.0
            for a,b in zip(values,values[1:]):
                lo,hi=a[0][index],b[0][index]
                if lo<=bound<=hi and hi>lo:return a[1]+(b[1]-a[1])*(bound-lo)/(hi-lo)
            raise ValueError('junction whole body exit absent')
        lo=frame.observed_at_s+max(0,crossing(task.entry_s_m,1)-coverage.time_early_error_s)
        hi=frame.observed_at_s+crossing(task.exit_s_m,0)+coverage.time_late_error_s
        return any(a<=lo and hi<=b for a,b in task.open_windows(p,frame))
    except (AttributeError,ValueError,TypeError,IndexError,OverflowError):return False


def emergency_path_is_safe(p,decision,trajectory,settings):
    """Extra positive authority check; never changes any braking decision."""
    try:
        request=trajectory.behavior_request
        if (settings is None or not isinstance(request,RuntimeRequest) or request.maneuver!='EMERGENCY'
                or request.authorization!='EXECUTE_SELECTED' or not trajectory.emergency_stop
                or trajectory.emergency_maneuver_authorized is not True or trajectory.target_speed!=0
                or trajectory.motion_direction!=1 or not current(decision,p) or not current(trajectory,decision)
                or decision.behavior_request.identity()!=request.identity()
                or trajectory.execution_path_id!=request.candidate_reference.candidate_id):return False
        frame=behavior_frame(p,time.monotonic());validate_runtime_request(request,frame)
        if (not p.targets_valid or not p.target_source.startswith('sensor:')
                or p.source_status.get('targets',{}).get('usable') is not True):return False
        if p.ego.vx*math.cos(p.ego.heading)+p.ego.vy*math.sin(p.ego.heading)<-0.05:return False
        task=p.behavior_environment.emergency_task;coverage=p.behavior_environment.coverage
        if not task.verified(frame,p.observation_mode) or not coverage.timing_verified(frame,p.observation_mode):return False
        from core.intersection_protocol import observed_objects
        from core.behavior_environment import vehicle_footprint
        from core.behavior_protocol import GoalPose
        vehicle=VehicleGeometry(settings.front_offset_m,settings.rear_offset_m,settings.half_width_m,settings.wheelbase_m)
        limits=MotionLimits(max(0.1,p.ego.speed)+0.1,settings.acceleration,settings.deceleration,settings.lateral_acceleration,
            settings.front_steer_max_rad,settings.behavior_steer_rate_rad_s,0.2,0.05)
        if trajectory.points[-1].speed!=0 or any(b.speed>a.speed+1e-8 for a,b in zip(trajectory.points,trajectory.points[1:])):return False
        predictions=[ObstaclePrediction(str(o.identifier),o.x,o.y,o.length+0.2,o.width+0.2,o.heading,o.vx,o.vy,
            coverage.prediction_horizon_s,coverage.position_uncertainty_m,coverage.velocity_uncertainty_mps,
            coverage.time_early_error_s,coverage.time_late_error_s) for o in observed_objects(p)]
        if not all(coverage.contains(vehicle_footprint(GoalPose(q.x,q.y,q.heading),settings.front_offset_m,
            settings.rear_offset_m,settings.half_width_m)) for q in trajectory.points):return False
        return any(validate_candidate(trajectory.points,1,vehicle,limits,region,predictions,
            ValidationBudget(20000,12,0.002,min(p.valid_until,time.monotonic()+0.025)))['status']=='safe' for region in task.allowed_regions)
    except (AttributeError,ValueError,TypeError,IndexError,OverflowError):return False
