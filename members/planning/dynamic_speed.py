"""Bounded multi-object progress/speed/time search for generated geometry.

The search uses actual initial speed, explicit motion limits and caller-supplied
execution error bounds. Every edge and published profile is independently swept
by candidate_validation; a time-layer endpoint is never a collision proof.
Inputs are planning-local model snapshots, not new public SimOne contracts.
"""
import bisect
import math
import time

from core.geometry import normalize_angle
from core.interfaces import TrajectoryPoint
from core.validation import number
from members.planning.candidate_validation import (
    ExecutionEnvelope, MotionLimits, ObstaclePrediction, ValidationBudget,
    VehicleGeometry, validate_candidate, validate_execution_envelope,
)


EPS = 1e-9


class DynamicSearch(object):
    """Finite engineering search settings, not measured runtime capacity."""
    def __init__(self, dt_s=0.5, acceleration_samples=None, beam_width=24,
                 max_steps=80, max_nodes=4000, reserve_s=0.01,
                 speed_resolution_mps=0.1, progress_resolution_m=0.25,
                 max_jerk_mps3=None):
        self.dt_s, self.acceleration_samples = dt_s, acceleration_samples
        self.beam_width, self.max_steps, self.max_nodes = beam_width, max_steps, max_nodes
        self.reserve_s = reserve_s
        self.speed_resolution_mps = speed_resolution_mps
        self.progress_resolution_m = progress_resolution_m
        self.max_jerk_mps3 = max_jerk_mps3


class _Failure(Exception):
    def __init__(self, reason, status='invalid'):
        self.reason, self.status = reason, status


def _require(condition, reason):
    if not condition:
        raise _Failure(reason)


class _Work(object):
    def __init__(self, budget, search):
        self.budget, self.search = budget, search
        self.checks, self.nodes = 0, 0
        self.deadline = budget.deadline_monotonic_s-search.reserve_s

    def step(self, node=False):
        self.checks += 1
        if node:
            self.nodes += 1
        if (self.checks > self.budget.max_checks or self.nodes > self.search.max_nodes
                or time.monotonic() >= self.deadline):
            raise _Failure('BUDGET_EXHAUSTED', 'inconclusive')

    def validate(self, points, direction, vehicle, limits, region, obstacles, envelope):
        self.step()
        remaining = self.budget.max_checks-self.checks
        if remaining <= 0:
            raise _Failure('BUDGET_EXHAUSTED', 'inconclusive')
        work = ValidationBudget(remaining, self.budget.max_depth,
                                self.budget.min_interval_s, self.deadline)
        report = validate_candidate(points, direction, vehicle, limits, region,
                                    obstacles, work, envelope)
        self.checks += report['checks']
        self.step()
        if report['reason_code'] == 'BUDGET_EXHAUSTED':
            raise _Failure('BUDGET_EXHAUSTED', 'inconclusive')
        return report


def _pose(shape, stations, progress):
    index = min(len(shape)-2, max(0, bisect.bisect_right(stations, progress)-1))
    a, b = shape[index:index+2]
    ratio = max(0., min(1., (progress-stations[index])/(stations[index+1]-stations[index])))
    return (a[0]+ratio*(b[0]-a[0]), a[1]+ratio*(b[1]-a[1]),
            a[2]+ratio*normalize_angle(b[2]-a[2]))


def _duration(distance, speed, acceleration):
    if distance <= EPS:
        return 0.
    if abs(acceleration) < EPS:
        return distance/speed if speed > EPS else None
    squared = speed*speed+2*acceleration*distance
    if squared < -EPS:
        return None
    # Stable positive root of .5*a*t^2+v*t-distance=0.
    terminal = math.sqrt(max(0., squared))
    denominator = speed+terminal
    return 2*distance/denominator if denominator > EPS else None


def _edge(shape, stations, s0, s1, speed, acceleration, dt):
    cuts = [s0]+[value for value in stations if s0+EPS < value < s1-EPS]+[s1]
    result = []
    for progress in cuts:
        at = _duration(progress-s0, speed, acceleration)
        if progress == s1:
            at = dt
        if at is None:
            raise _Failure('KINEMATIC_EDGE_INVALID')
        pose = _pose(shape, stations, progress)
        result.append(TrajectoryPoint(pose[0], pose[1], max(0., speed+acceleration*at),
                                      pose[2], at))
    return result


def _predictions_at(obstacles, offset):
    return [ObstaclePrediction(
        item.identity, item.x+item.vx*offset, item.y+item.vy*offset,
        item.length, item.width, item.heading, item.vx, item.vy,
        item.valid_until_s-offset,
        item.position_uncertainty_m+item.velocity_uncertainty_mps*offset,
        item.velocity_uncertainty_mps) for item in obstacles]


def _windows_ok(s0, s1, speed, acceleration, dt, start_time, windows, envelope, vehicle,
                progress_error_included=False):
    radius=math.hypot(max(vehicle.front_offset_m,vehicle.rear_offset_m),vehicle.half_width_m)
    progress_error=(0. if progress_error_included else
                    envelope.position_error_m+2*radius*math.sin(envelope.heading_error_rad*.5))
    for lo, hi, opened, closed in windows:
        # The supplied progress interval denotes whole-body occupancy. Rear
        # axle position error adds a conservative progress uncertainty here.
        lo -= progress_error
        hi += progress_error
        if s1 < lo-EPS or s0 > hi+EPS:
            continue
        entered = max(s0, lo)-s0
        departed = min(s1, hi)-s0
        ta = _duration(entered, speed, acceleration)
        tb = _duration(departed, speed, acceleration)
        if ta is None or tb is None:
            return False
        if (max(0.,start_time+ta-envelope.early_time_s) < opened-EPS
                or start_time+tb+envelope.late_time_s > closed+EPS):
            return False
    return True


def _join(state):
    edges = []
    while state['parent'] is not None:
        edges.append((state['parent']['t'], state['edge']))
        state = state['parent']
    result = []
    for offset, edge in reversed(edges):
        for index, point in enumerate(edge):
            if result and index == 0:
                continue
            result.append(TrajectoryPoint(point.x, point.y, point.speed, point.heading,
                                          offset+point.relative_time))
    return result


def _jerk_ok(points, bound, initial_acceleration):
    if bound is None:
        return True
    previous, previous_time = initial_acceleration, 0.
    for a,b in zip(points,points[1:]):
        dt=b.relative_time-a.relative_time
        acceleration=(b.speed-a.speed)/dt
        midpoint=.5*(a.relative_time+b.relative_time)
        if abs(acceleration-previous)/(midpoint-previous_time)>bound+EPS:
            return False
        previous,previous_time=acceleration,midpoint
    return True


def _spatial_cap(station, values, cap):
    """Piecewise linear speed squared; matches constant-acceleration edges."""
    if not values:
        return cap
    if station<=values[0][0]:
        return min(cap,values[0][1])
    for a,b in zip(values,values[1:]):
        if station<=b[0]:
            ratio=(station-a[0])/(b[0]-a[0])
            return min(cap,math.sqrt(max(0.,a[1]**2+ratio*(b[1]**2-a[1]**2))))
    return min(cap,values[-1][1])


def _spatial_cuts(s0,s1,values,cap,initial,deceleration):
    cuts=[s0]+[v[0] for v in values if s0<v[0]<s1]+[s1]
    # Split where a varying supplied cap meets the global cap.
    extra=[]
    for lo,hi in zip(cuts,cuts[1:]):
        raw_a=_spatial_cap(lo,values,float('inf'))**2
        raw_b=_spatial_cap(hi,values,float('inf'))**2
        if abs(raw_b-raw_a)>EPS:
            ratio=(cap*cap-raw_a)/(raw_b-raw_a)
            if EPS<ratio<1-EPS:
                extra.append(lo+ratio*(hi-lo))
    cuts=sorted(cuts+extra)
    initial_cap=_spatial_cap(0.,values,cap)
    recovering=initial>initial_cap+EPS
    if recovering:
        zero=initial*initial/(2*deceleration)
        if s0<zero<s1:
            cuts=sorted(cuts+[zero])
        extra=[]
        for lo,hi in zip(cuts,cuts[1:]):
            # The max of the limit and reachable braking floor has a kink.
            # Endpoint-only checks across that kink would miss overspeed.
            ca,cb=_spatial_cap(lo,values,cap)**2,_spatial_cap(hi,values,cap)**2
            fa=max(0.,initial**2-2*deceleration*lo)
            fb=max(0.,initial**2-2*deceleration*hi)
            denominator=(cb-ca)-(fb-fa)
            if abs(denominator)>EPS:
                ratio=(fa-ca)/denominator
                if EPS<ratio<1-EPS:
                    extra.append(lo+ratio*(hi-lo))
        cuts=sorted(cuts+extra)
    return cuts


def _spatial_ok(s0,s1,v0,acceleration,values,cap,initial,deceleration):
    if not values:
        return True  # Preserve the legacy global cap's reachable braking rule.
    cuts=_spatial_cuts(s0,s1,values,cap,initial,deceleration)
    recovering=initial>_spatial_cap(0.,values,cap)+EPS
    for station in cuts:
        squared=max(0.,v0*v0+2*acceleration*(station-s0))
        allowed=_spatial_cap(station,values,cap)
        if recovering:
            # An observed initial overspeed is disclosed and reduced by the
            # maximum reachable braking profile. Never rewrite the measure.
            allowed=max(allowed,math.sqrt(max(0.,initial**2-2*deceleration*station)))
        if squared>allowed*allowed+EPS:
            return False
    return True


def _static_proposal(shape,stations,length,initial,cap,vehicle,limits,direction,terminal,work,spatial=None,
                     allow_acceleration=True):
    """Cheap feasible-physics seed; still checked against every timed target."""
    positions = sorted(set([s for s in stations if s < length-EPS]+
                           [s for s,v in (spatial or []) if 0<s<length-EPS]+[length]))
    if spatial:
        positions=sorted(set(positions+_spatial_cuts(0.,length,spatial,
                         min(cap,limits.max_speed_mps),initial,limits.max_deceleration_mps2)))
    poses = [_pose(shape,stations,s) for s in positions]
    caps = [_spatial_cap(s,spatial,min(cap,limits.max_speed_mps)) for s in positions]
    lengths = [b-a for a,b in zip(positions,positions[1:])]
    for index,(a,b) in enumerate(zip(poses,poses[1:])):
        work.step()
        curvature = abs(normalize_angle(b[2]-a[2])/lengths[index])
        if abs(math.atan(vehicle.wheelbase_m*curvature)) > limits.max_front_steer_rad+EPS:
            return None
        bound = math.sqrt(limits.max_lateral_acceleration_mps2/curvature) if curvature>EPS else limits.max_speed_mps
        for i in (index,index+1):
            caps[i] = min(caps[i],bound)
    for index in range(1,len(poses)-1):
        work.step()
        a,b,c = poses[index-1:index+2]
        dx,dy,ex,ey = b[0]-a[0],b[1]-a[1],c[0]-b[0],c[1]-b[1]
        chord = math.hypot(c[0]-a[0],c[1]-a[1])
        if chord<=EPS:
            return None
        curvature = abs(2*(dx*ey-dy*ex)/(lengths[index-1]*lengths[index]*chord))
        if abs(math.atan(vehicle.wheelbase_m*curvature)) > limits.max_front_steer_rad+EPS:
            return None
        bound = math.sqrt(limits.max_lateral_acceleration_mps2/curvature) if curvature>EPS else limits.max_speed_mps
        for i in (index-1,index,index+1):
            caps[i] = min(caps[i],bound)
    if initial>_spatial_cap(0.,spatial,cap):
        for index,s in enumerate(positions):
            caps[index] = max(caps[index],math.sqrt(max(0.,initial**2-2*limits.max_deceleration_mps2*s)))
    if terminal is not None:
        caps[-1] = min(caps[-1],terminal)
    for index in range(len(caps)-2,-1,-1):
        work.step()
        caps[index] = min(caps[index],math.sqrt(caps[index+1]**2+2*limits.max_deceleration_mps2*lengths[index]))
    if initial>caps[0]+EPS:
        return None
    speeds=[initial]
    for index,ds in enumerate(lengths):
        work.step()
        acceleration=limits.max_acceleration_mps2 if allow_acceleration else 0.
        speeds.append(min(caps[index+1],math.sqrt(speeds[-1]**2+2*acceleration*ds)))
    if terminal is not None and abs(speeds[-1]-terminal)>1e-6:
        return None
    points=[]; elapsed=0.
    for index,pose in enumerate(poses):
        if index:
            if speeds[index-1]+speeds[index]<=EPS:
                return None
            elapsed+=2*lengths[index-1]/(speeds[index-1]+speeds[index])
        points.append(TrajectoryPoint(pose[0],pose[1],speeds[index],pose[2],elapsed))
    return points,positions


def solve_dynamic_speed(shape, initial_speed_mps, speed_cap_mps, vehicle, limits,
                        corridor, coverage, obstacles, budget, execution_envelope,
                        search=None, motion_direction=1, stop_required=False,
                        terminal_speed_mps=None, stop_distance_m=-1.,
                        progress_time_windows=None, initial_acceleration_mps2=0.,
                        window_progress_error_included=False,spatial_speed_limits=None,
                        allow_acceleration=True):
    """Search finite (progress,speed,previous acceleration) time layers.

    A success covers the requested geometry/stop bound; a partial braking path
    is not reported as goal completion. Interior waiting is excluded because
    today's controller executes spatial speed rather than timed wait release.
    Explicit execution envelopes cover allowed timing/pose errors for collision
    and admission claims, but do not invent measured calibration evidence.
    """
    started = time.monotonic()
    report = dict(status='invalid', reason_code='INVALID_INPUT', points=[], nodes=0,
                  checks=0, elapsed_s=0., execution_robust=False, goal_reached=False,
                  rejected_edges={}, authority='model_candidate_only',
                  assumptions=['constant-velocity target prediction with finite bounds',
                               'caller-supplied execution error envelope',
                               'no interior timed waiting; no actuator calibration inferred'])
    work = None
    try:
        search = search or DynamicSearch()
        _require(isinstance(search, DynamicSearch), 'DYNAMIC_SEARCH_INVALID')
        _require(isinstance(execution_envelope, ExecutionEnvelope), 'EXECUTION_ENVELOPE_UNAVAILABLE')
        try:
            validate_execution_envelope(execution_envelope)
        except Exception:
            raise _Failure('EXECUTION_ENVELOPE_INVALID')
        _require(isinstance(vehicle, VehicleGeometry) and isinstance(limits, MotionLimits),
                 'MOTION_MODEL_UNAVAILABLE')
        _require(isinstance(budget, ValidationBudget) and type(budget.max_checks) is int
                 and budget.max_checks > 0 and type(budget.max_depth) is int
                 and 0 <= budget.max_depth <= 40 and number(budget.min_interval_s)
                 and budget.min_interval_s > 0 and number(budget.deadline_monotonic_s),
                 'VALIDATION_BUDGET_INVALID')
        _require(type(motion_direction) is int and motion_direction in (-1, 1), 'DIRECTION_INVALID')
        _require(type(stop_required) is bool, 'STOP_CONTRACT_INVALID')
        _require(type(allow_acceleration) is bool,'ACCELERATION_AUTHORITY_INVALID')
        report['allow_acceleration']=allow_acceleration
        _require(type(window_progress_error_included) is bool,'ADMISSION_WINDOWS_INVALID')
        for value in (search.dt_s, search.speed_resolution_mps, search.progress_resolution_m):
            _require(number(value) and value > 0, 'DYNAMIC_SEARCH_INVALID')
        _require(number(search.reserve_s) and search.reserve_s >= 0, 'DYNAMIC_SEARCH_INVALID')
        for value, maximum in ((search.beam_width,256),(search.max_steps,1000),(search.max_nodes,100000)):
            _require(type(value) is int and 0 < value <= maximum, 'DYNAMIC_SEARCH_INVALID')
        if search.max_jerk_mps3 is not None:
            _require(number(search.max_jerk_mps3) and search.max_jerk_mps3 > 0, 'DYNAMIC_SEARCH_INVALID')
        _require(number(initial_speed_mps) and initial_speed_mps >= 0
                 and number(speed_cap_mps) and speed_cap_mps > 0
                 and number(initial_acceleration_mps2), 'INITIAL_MOTION_INVALID')
        _require(initial_speed_mps <= limits.max_speed_mps+EPS, 'INITIAL_SPEED_OUTSIDE_LIMITS')
        _require(number(stop_distance_m) and (stop_distance_m == -1 or stop_distance_m >= 0),
                 'STOP_DISTANCE_INVALID')
        _require(stop_distance_m < 0 or stop_required, 'STOP_CONTRACT_INVALID')
        _require(terminal_speed_mps is None or number(terminal_speed_mps)
                 and 0 <= terminal_speed_mps <= speed_cap_mps, 'TERMINAL_SPEED_INVALID')
        _require(isinstance(shape,(tuple,list)) and 2 <= len(shape) <= 10000, 'PATH_GEOMETRY_INVALID')
        shape = [tuple(value) for value in shape]
        _require(all(len(value)==3 and all(number(x) for x in value) for value in shape),
                 'PATH_GEOMETRY_INVALID')
        work = _Work(budget, search); work.step()
        stations = [0.]
        for a,b in zip(shape,shape[1:]):
            work.step()
            distance = math.hypot(b[0]-a[0],b[1]-a[1])
            _require(distance > EPS, 'PATH_GEOMETRY_DEGENERATE')
            stations.append(stations[-1]+distance)
        if stop_distance_m > stations[-1]+EPS:
            raise _Failure('PATH_COVERAGE_INSUFFICIENT','inconclusive')
        length = stations[-1] if stop_distance_m < 0 else min(stations[-1],stop_distance_m)
        _require(length > EPS, 'ZERO_LENGTH_MOTION_REQUEST')
        spatial=[] if spatial_speed_limits is None else spatial_speed_limits
        _require(isinstance(spatial,(list,tuple)) and len(spatial)<=10000 and all(
            isinstance(v,(list,tuple)) and len(v)==2 and all(number(x) for x in v)
            and 0<=v[0]<=stations[-1]+EPS and v[1]>=0 for v in spatial)
            and all(a[0]<b[0] for a,b in zip(spatial,spatial[1:])), 'SPATIAL_SPEED_LIMITS_INVALID')
        report['initial_overspeed_mps']=max(0.,initial_speed_mps-
            _spatial_cap(0.,spatial,min(speed_cap_mps,limits.max_speed_mps)))
        report['spatial_speed_limit_model']='piecewise_linear_speed_squared'
        if stop_required:
            _require(terminal_speed_mps is None or terminal_speed_mps == 0.,'STOP_CONTRACT_INVALID')
            terminal_speed_mps = 0.
            if initial_speed_mps**2 > 2*limits.max_deceleration_mps2*length+EPS:
                raise _Failure('STOP_DISTANCE_INFEASIBLE','unsafe')
        _require(coverage is not None, 'VISIBILITY_REGION_REQUIRED')
        _require(isinstance(obstacles,(list,tuple)) and all(isinstance(v,ObstaclePrediction) for v in obstacles),
                 'PREDICTIONS_INVALID')
        _require(all(number(v.vx) and number(v.vy) and number(v.valid_until_s)
                     and v.valid_until_s > 0 for v in obstacles), 'PREDICTIONS_INVALID')
        windows = [] if progress_time_windows is None else progress_time_windows
        _require(isinstance(windows,(list,tuple)) and all(
            isinstance(v,(list,tuple)) and len(v)==4 and all(number(x) for x in v)
            and 0 <= v[0] <= v[1] and 0 <= v[2] < v[3] for v in windows), 'ADMISSION_WINDOWS_INVALID')
        samples = (search.acceleration_samples if search.acceleration_samples is not None else
                   (-limits.max_deceleration_mps2, -limits.max_deceleration_mps2*.5,
                    0., limits.max_acceleration_mps2*.5, limits.max_acceleration_mps2))
        _require(isinstance(samples,(list,tuple)) and 1 <= len(samples) <= 31
                 and all(number(v) and -limits.max_deceleration_mps2 <= v <= limits.max_acceleration_mps2
                         for v in samples), 'ACCELERATION_SAMPLES_INVALID')
        proposal = _static_proposal(shape,stations,length,initial_speed_mps,speed_cap_mps,
                                    vehicle,limits,motion_direction,terminal_speed_mps,work,spatial,allow_acceleration)
        if proposal is not None:
            work.step(node=True)
            points,positions = proposal
            spatial_ok=all(_spatial_ok(sa,sb,a.speed,(b.speed-a.speed)/(b.relative_time-a.relative_time),
                          spatial,min(speed_cap_mps,limits.max_speed_mps),initial_speed_mps,
                          limits.max_deceleration_mps2)
                          for sa,sb,a,b in zip(positions,positions[1:],points,points[1:]))
            windows_ok = all(_windows_ok(sa,sb,a.speed,(b.speed-a.speed)/(b.relative_time-a.relative_time),
                                       b.relative_time-a.relative_time,a.relative_time,windows,execution_envelope,vehicle,
                                       window_progress_error_included)
                             for sa,sb,a,b in zip(positions,positions[1:],points,points[1:]))
            legal = work.validate(points,motion_direction,vehicle,limits,corridor,obstacles,execution_envelope)
            jerk_ok=_jerk_ok(points,search.max_jerk_mps3,initial_acceleration_mps2)
            if legal['status']=='safe' and windows_ok and jerk_ok and spatial_ok:
                visible = work.validate(points,motion_direction,vehicle,limits,coverage,[],execution_envelope)
                if visible['status']=='safe':
                    report.update(status='safe',reason_code='DYNAMIC_PROFILE_CERTIFIED',points=points,
                                  execution_robust=True,goal_reached=True,path_distance_m=length,
                                  stop_required=stop_required,stop_distance=length if stop_required else -1.,
                                  target_speed=speed_cap_mps,motion_direction=motion_direction,
                                  legal_report=legal,visibility_report=visible,
                                  planning_horizon_s=points[-1].relative_time,layers=0)
                    return report
            code = (legal['reason_code'] if legal['status']!='safe' else
                    'SPATIAL_SPEED_LIMIT' if not spatial_ok else 'ADMISSION_WINDOW')
            report['rejected_edges'][code] = report['rejected_edges'].get(code,0)+1
        beam = [dict(s=0.,v=initial_speed_mps,a=initial_acceleration_mps2,t=0.,parent=None,edge=None,cost=0.)]
        for step in range(search.max_steps):
            next_layer = {}
            goals = []
            for state in beam:
                options = list(samples)
                remaining = length-state['s']
                if state['v'] > EPS and terminal_speed_mps is not None:
                    exact = (terminal_speed_mps**2-state['v']**2)/(2*remaining)
                    if -limits.max_deceleration_mps2 <= exact <= limits.max_acceleration_mps2:
                        options.append(exact)
                for acceleration in options:
                    work.step(node=True)
                    if not allow_acceleration and acceleration>EPS:
                        continue
                    dt = search.dt_s
                    if acceleration < 0:
                        dt = min(dt, state['v']/-acceleration)
                    if dt <= EPS:
                        continue
                    distance = state['v']*dt+.5*acceleration*dt*dt
                    if distance <= EPS:
                        continue
                    if distance >= remaining-EPS:
                        dt = _duration(remaining,state['v'],acceleration)
                        if dt is None or dt <= EPS:
                            continue
                        distance = remaining
                    speed = max(0.,state['v']+acceleration*dt)
                    # Above an authorized cap, preserve measured starting speed
                    # and admit only braking toward the cap; never clamp it.
                    cap = min(speed_cap_mps,limits.max_speed_mps)
                    if speed > max(cap,state['v'])+EPS or state['v'] > cap+EPS and acceleration >= 0:
                        continue
                    if speed > limits.max_speed_mps+EPS:
                        continue
                    if search.max_jerk_mps3 is not None and abs(acceleration-state['a'])/dt > search.max_jerk_mps3+EPS:
                        continue
                    progress = state['s']+distance
                    if not _spatial_ok(state['s'],progress,state['v'],acceleration,spatial,
                                       min(speed_cap_mps,limits.max_speed_mps),initial_speed_mps,
                                       limits.max_deceleration_mps2):
                        report['rejected_edges']['SPATIAL_SPEED_LIMIT']=report['rejected_edges'].get('SPATIAL_SPEED_LIMIT',0)+1
                        continue
                    at_goal = progress >= length-EPS
                    if at_goal and terminal_speed_mps is not None and abs(speed-terminal_speed_mps)>1e-6:
                        continue
                    if not at_goal and speed <= EPS:
                        continue  # No unsupported timed release at an interior stop.
                    if stop_required and speed**2 > 2*limits.max_deceleration_mps2*(length-progress)+EPS:
                        continue
                    if not _windows_ok(state['s'],progress,state['v'],acceleration,dt,
                                       state['t'],windows,execution_envelope,vehicle,window_progress_error_included):
                        report['rejected_edges']['ADMISSION_WINDOW'] = report['rejected_edges'].get('ADMISSION_WINDOW',0)+1
                        continue
                    edge = _edge(shape,stations,state['s'],progress,state['v'],acceleration,dt)
                    shifted = _predictions_at(obstacles,state['t'])
                    legal = work.validate(edge,motion_direction,vehicle,limits,corridor,shifted,execution_envelope)
                    if legal['status'] == 'safe':
                        visible = work.validate(edge,motion_direction,vehicle,limits,coverage,[],execution_envelope)
                        result = visible
                    else:
                        result = legal
                    if result['status'] != 'safe':
                        code = result['reason_code']
                        report['rejected_edges'][code] = report['rejected_edges'].get(code,0)+1
                        continue
                    cost = state['cost']+dt*(speed_cap_mps-speed)**2*.1+abs(acceleration-state['a'])*.02
                    new = dict(s=progress,v=speed,a=acceleration,t=state['t']+dt,parent=state,edge=edge,cost=cost)
                    if at_goal:
                        goals.append(new)
                    else:
                        key = (round(progress/search.progress_resolution_m),round(speed/search.speed_resolution_mps),
                               round(acceleration/max(.1,limits.max_acceleration_mps2*.25)))
                        old = next_layer.get(key)
                        if old is None or cost < old['cost']:
                            next_layer[key] = new
            for goal in sorted(goals,key=lambda value:(value['t'],value['cost'])):
                points = _join(goal)
                if not _jerk_ok(points,search.max_jerk_mps3,initial_acceleration_mps2):
                    continue
                legal = work.validate(points,motion_direction,vehicle,limits,corridor,obstacles,execution_envelope)
                if legal['status'] == 'safe':
                    visible = work.validate(points,motion_direction,vehicle,limits,coverage,[],execution_envelope)
                    if visible['status'] == 'safe':
                        report.update(status='safe',reason_code='DYNAMIC_PROFILE_CERTIFIED',points=points,
                                      execution_robust=True,goal_reached=True,path_distance_m=length,
                                      stop_required=stop_required,stop_distance=length if stop_required else -1.,
                                      target_speed=speed_cap_mps,motion_direction=motion_direction,
                                      legal_report=legal,visibility_report=visible,
                                      planning_horizon_s=goal['t'],layers=step+1)
                        return report
                else:
                    code = legal['reason_code']
                    report['rejected_edges'][code] = report['rejected_edges'].get(code,0)+1
            if not next_layer:
                break
            beam = sorted(next_layer.values(),key=lambda value:(-value['s'],value['cost']))[:search.beam_width]
        raise _Failure('NO_CERTIFIED_DYNAMIC_PROFILE','inconclusive')
    except _Failure as error:
        report.update(status=error.status,reason_code=error.reason)
    except (ValueError,TypeError,AttributeError,ZeroDivisionError,OverflowError) as error:
        report.update(status='invalid',reason_code='INVALID_INPUT',details=str(error))
    finally:
        report['elapsed_s'] = time.monotonic()-started
        if work is not None:
            report.update(checks=work.checks,nodes=work.nodes,
                          reserved_time_s=work.search.reserve_s,deadline_monotonic_s=work.deadline)
    return report
