"""P04 single-direction goal segments, reusing P03 connectors and P02 sweeps.

Goal is rear-axle XY/body yaw. Direction is explicit, speeds stay nonnegative.
Each segment ends stopped; actual standstill, gear authorization, 10-second
dwell and stage transitions remain with the existing decision/control chain.
"""
import math
from core.interfaces import TrajectoryPoint
from core.validation import number
from core.geometry import normalize_angle
from members.planning.lane_change_generator import (
    _Failure, _Work, _require, _motion_parameters, _shape, _timed,
)


class ParkingSearch(object):
    def __init__(self,tangent_scales,speed_scales,spacing_m,max_points,end_tangent_scales=None,
                 terminal_straight_m=(0.,),lattice_search=None,execution_envelope=None,dynamic_search=None):
        self.tangent_scales,self.speed_scales=tangent_scales,speed_scales
        self.spacing_m,self.max_points=spacing_m,max_points
        self.end_tangent_scales=tangent_scales if end_tangent_scales is None else end_tangent_scales
        self.terminal_straight_m=terminal_straight_m
        self.lattice_search=lattice_search
        self.execution_envelope,self.dynamic_search=execution_envelope,dynamic_search


def _search_parameters(search):
    _require(isinstance(search,ParkingSearch)
             and all(isinstance(v,(tuple,list)) and 1<=len(v)<=8 for v in
                     (search.tangent_scales,search.end_tangent_scales,search.speed_scales,search.terminal_straight_m))
             and all(number(v) and v>0 for v in search.tangent_scales)
             and all(number(v) and v>0 for v in search.end_tangent_scales)
             and all(number(v) and v>=0 for v in search.terminal_straight_m)
             and len(search.tangent_scales)*len(search.end_tangent_scales)*len(search.speed_scales)*len(search.terminal_straight_m)<=64
             and all(number(v) and 0<v<=1 for v in search.speed_scales)
             and number(search.spacing_m) and search.spacing_m>0
             and type(search.max_points) is int and 3<=search.max_points<=2000,'PARKING_SEARCH_INVALID')
    if search.lattice_search is not None:
        from members.planning.parking_lattice import validate_lattice_search
        validate_lattice_search(search.lattice_search)


def _parking_shape(start,goal,curvature,tangent,end_tangent,straight,search,work,direction):
    # The connector ends tangent to a final straight, with zero curvature.
    # A straight of zero retains the original one-connector candidate family.
    join=(goal[0]-direction*straight*math.cos(goal[2]),
          goal[1]-direction*straight*math.sin(goal[2]),goal[2])
    shape,lengths=_shape(start,join,curvature,tangent,search,work,direction,end_tangent)
    if straight:
        count=max(1,int(math.ceil(straight/search.spacing_m)))
        if len(shape)+count>search.max_points:
            raise _Failure('CANDIDATE_POINT_LIMIT','inconclusive')
        for i in range(1,count+1):
            work.step()
            ratio=float(i)/count
            shape.append((join[0]+ratio*(goal[0]-join[0]),
                          join[1]+ratio*(goal[1]-join[1]),goal[2]))
        shape[-1]=tuple(goal)
        lengths=[math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(shape,shape[1:])]
    return shape,lengths


def _timed_parking(shape,lengths,initial,desired,cap,curvature,vehicle,limits,search,work,
                   direction,access,coverage,obstacles):
    """Use explicit execution bounds when supplied, otherwise nominal timing."""
    if search.execution_envelope is None:
        return (_timed(shape,lengths,initial,desired,cap,curvature,vehicle,limits,work,direction,
                       True,allow_initial_overspeed=True),dict(execution_robust=False))
    from members.planning.dynamic_speed import solve_dynamic_speed
    from members.planning.candidate_validation import ValidationBudget
    work.step()
    remaining=ValidationBudget(work.budget.max_checks-work.count,work.budget.max_depth,
        work.budget.min_interval_s,work.budget.deadline_monotonic_s)
    result=solve_dynamic_speed(shape,initial,desired,vehicle,limits,access,coverage,obstacles,
        remaining,search.execution_envelope,search=search.dynamic_search,motion_direction=direction,
        stop_required=True,terminal_speed_mps=0.)
    work.count+=result.get('checks',0); work.step()
    if result['status']!='safe': raise _Failure(result['reason_code'],result['status'])
    points=result['points']
    ds=math.hypot(points[1].x-points[0].x,points[1].y-points[0].y)
    dt=points[1].relative_time-points[0].relative_time
    first=math.atan(vehicle.wheelbase_m*normalize_angle(points[1].heading-points[0].heading)/(direction*ds))
    rate=abs(first-math.atan(vehicle.wheelbase_m*curvature))/(dt*.5)
    _require(rate<=limits.max_steer_rate_rad_s+1e-9,'INITIAL_STEERING_RATE_LIMIT')
    return points,result


def generate_parking_segment(start_pose,goal_pose,initial_speed_mps,initial_curvature_m_inv,
                             motion_direction,speed_cap_mps,vehicle,limits,
                             access_corridor,coverage,goal_corridor,obstacles,search,budget):
    """Generate a forward approach/exit or reverse-entry segment ending at zero.

    Explicit lattice_search adds a bounded bicycle-pose fallback after the
    finite connector family. It stays within this stage's authorized direction.
    An explicit source-bound access corridor is supplied by the caller; plain
    free-space geometry alone never grants road/bay crossing permission.
    """
    report=dict(status='invalid',reason_code='INVALID_INPUT',points=[],attempts=[],checks=0,
                authority='nominal_candidate_only',motion_direction=motion_direction)
    work=None
    try:
        _motion_parameters(start_pose,initial_speed_mps,initial_curvature_m_inv,
                           speed_cap_mps,vehicle,limits,budget,allow_initial_overspeed=True)
        _require(type(motion_direction) is int and motion_direction in (-1,1),'PARKING_DIRECTION_INVALID')
        _require(isinstance(goal_pose,(tuple,list)) and len(goal_pose)==3
                 and all(number(v) for v in goal_pose),'PARKING_REAR_AXLE_GOAL_UNAVAILABLE')
        _search_parameters(search)
        from members.planning.crossing_corridor import CrossingCorridor
        _require(isinstance(access_corridor,CrossingCorridor),'PARKING_ACCESS_CORRIDOR_REQUIRED')
        _require(coverage is not None and goal_corridor is not None,'PARKING_COVERAGE_OR_GOAL_REGION_MISSING')
        work=_Work(budget); work.step()
        if (tuple(start_pose[:2])==tuple(goal_pose[:2])
                and abs(normalize_angle(start_pose[2]-goal_pose[2]))<=1e-9):
            _require(initial_speed_mps==0.,'MOVING_START_CANNOT_BECOME_STATIONARY_GOAL')
            points=[TrajectoryPoint(start_pose[0],start_pose[1],0.,start_pose[2],0.),
                    TrajectoryPoint(goal_pose[0],goal_pose[1],0.,goal_pose[2],1.)]
            options=[(None,None,None,None,points,0.)]
        else:
            options=[(t,e,s,l,None,None) for l in search.terminal_straight_m
                     for t in search.tangent_scales for e in search.end_tangent_scales for s in search.speed_scales]
        for tangent,end_tangent,speed,straight,stationary,distance in options:
            work.step()
            attempt=dict(tangent_scale=tangent,end_tangent_scale=end_tangent,speed_scale=speed,
                         terminal_straight_m=straight)
            try:
                if stationary is None:
                    shape,lengths=_parking_shape(start_pose,goal_pose,initial_curvature_m_inv,tangent,
                        end_tangent,straight,search,work,motion_direction)
                    points,timing=_timed_parking(shape,lengths,initial_speed_mps,speed_cap_mps*speed,speed_cap_mps,
                        initial_curvature_m_inv,vehicle,limits,search,work,motion_direction,
                        access_corridor,coverage,obstacles)
                    distance=sum(lengths)
                else: points=stationary; timing=dict(execution_robust=search.execution_envelope is not None)
                legal=work.validate(points,vehicle,limits,access_corridor,obstacles,motion_direction,
                                    execution_envelope=search.execution_envelope)
                attempt['access_report']=legal
                result=legal
                if legal['status']=='safe':
                    visible=work.validate(points,vehicle,limits,coverage,[],motion_direction,
                                          execution_envelope=search.execution_envelope)
                    attempt['visibility_report']=visible; result=visible
                    if visible['status']=='safe':
                        end=points[-1]
                        probe=[TrajectoryPoint(end.x,end.y,0.,end.heading,0.),
                               TrajectoryPoint(end.x,end.y,0.,end.heading,1.)]
                        target=work.validate(probe,vehicle,limits,goal_corridor,[],motion_direction,
                                             execution_envelope=search.execution_envelope)
                        attempt['goal_pose_report']=target; result=target
                        if target['status']=='safe':
                            work.step()
                            report.update(status='safe',reason_code='NOMINAL_PARKING_SEGMENT_READY',points=points,
                                stop_required=True,precision_stop=True,stop_distance=distance,
                                target_speed=min(speed_cap_mps,limits.max_speed_mps),goal_pose=tuple(goal_pose),
                                access_report=legal,visibility_report=visible,goal_pose_report=target,
                                execution_robust=bool(timing.get('execution_robust')),timing_report=timing)
                            report['attempts'].append(attempt)
                            return report
                attempt.update(status=result['status'],reason_code=result['reason_code'])
                if result['reason_code'] in ('BUDGET_EXHAUSTED','REGION_PREPARATION_LIMIT'):
                    report['attempts'].append(attempt)
                    report.update(status='inconclusive',reason_code=result['reason_code'])
                    return report
            except _Failure as error:
                attempt.update(status=error.status,reason_code=error.code)
                if error.code=='BUDGET_EXHAUSTED':
                    report['attempts'].append(attempt); raise
            report['attempts'].append(attempt)
        if search.lattice_search is not None:
            from members.planning.parking_lattice import _search_with_work
            fallback=_search_with_work(start_pose,goal_pose,initial_speed_mps,initial_curvature_m_inv,
                motion_direction,speed_cap_mps,vehicle,limits,access_corridor,coverage,goal_corridor,
                obstacles,search,search.lattice_search,work)
            report['lattice_report']=fallback
            if fallback['status']=='safe':
                if (len(fallback['segments'])!=1
                        or fallback['segments'][0]['motion_direction']!=motion_direction):
                    report.update(status='inconclusive',reason_code='PARKING_SEGMENT_AUTHORIZATION_REQUIRED',
                        maneuver_segments=fallback['segments'],points=[],needs_direction_authorization=True)
                    return report
                segment=fallback['segments'][0]
                report.update(status='safe',reason_code='NOMINAL_PARKING_LATTICE_SEGMENT_READY',
                    points=segment['points'],stop_required=True,precision_stop=True,
                    stop_distance=segment['stop_distance'],target_speed=min(speed_cap_mps,limits.max_speed_mps),
                    goal_pose=tuple(goal_pose),access_report=segment['access_report'],
                    visibility_report=segment['visibility_report'],goal_pose_report=fallback['goal_pose_report'],
                    execution_robust=fallback.get('execution_robust',False))
                return report
            report.update(status=fallback['status'],reason_code=fallback['reason_code'])
            return report
        report.update(status='inconclusive' if any(v['status']=='inconclusive' for v in report['attempts']) else 'unsafe',
                      reason_code='NO_CERTIFIED_PARKING_SEGMENT')
    except _Failure as error:
        report.update(status=error.status,reason_code=error.code,points=[])
    except (ValueError,TypeError,AttributeError,OverflowError,ZeroDivisionError) as error:
        report.update(status='invalid',reason_code='INVALID_INPUT',details=str(error),points=[])
    finally:
        report['checks']=work.count if work else 0
    return report
