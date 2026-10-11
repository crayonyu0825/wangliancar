"""Production candidate generation, explicit adoption and current revalidation.

The first implementation is a bounded quintic lane transition. All obstacles
remain in the candidate validator; follow clearance applies in overlapping
corridors, rather than an old centre-line stop distance on the new path.
"""
import math
import time
import bisect
import copy

from core.interfaces import Trajectory, TrajectoryPoint
from core.validation import current
from core.geometry import project_polyline, normalize_angle
from core.behavior_protocol import (RuntimeRequest, CandidateReference, PlanningFeedback,
    Capabilities, RuntimeCapabilities, behavior_frame, GoalPose, PROTOCOL_VERSION, validate_runtime_request)
from core.behavior_environment import BehaviorEnvironment, vehicle_footprint
from core.traffic_quality import signal_stop_requirement
from core.target_semantics import mapped_traffic_light
from members.planning.candidate_validation import (VehicleGeometry, MotionLimits,
    ObstaclePrediction, ValidationBudget, validate_candidate)


def geometry(settings):
    values = (settings.front_offset_m, settings.rear_offset_m, settings.half_width_m,
              settings.wheelbase_m, settings.front_steer_max_rad)
    return all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values)


def capabilities(perception, settings):
    if not perception.session_id or perception.valid is not True or perception.ego.valid is not True or not geometry(settings):
        return None
    f = behavior_frame(perception, time.monotonic())
    result = Capabilities(f.context, ("LANE_CHANGE", "PATH", "LIGHTS", "FEEDBACK"),
                          f.observed_at_s, f.valid_until_s, usable=f.usable)
    return result


def stations(points):
    result = [0.0]
    for a, b in zip(points, points[1:]):
        result.append(result[-1] + math.hypot(b[0]-a[0], b[1]-a[1]))
    return result


def at(points, arc, value):
    if not -1e-8 <= value <= arc[-1]+1e-8:
        raise ValueError("reference coverage exhausted")
    value=max(0.0,min(arc[-1],value))
    i = max(0, min(len(points)-2, bisect.bisect_right(arc, value)-1))
    length = arc[i+1]-arc[i]
    if length <= 1e-8:
        raise ValueError("degenerate lane")
    u = (value-arc[i])/length
    return (points[i][0]+u*(points[i+1][0]-points[i][0]),
            points[i][1]+u*(points[i+1][1]-points[i][1]),
            math.atan2(points[i+1][1]-points[i][1], points[i+1][0]-points[i][0]))


class BehaviorPlanner(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._candidates = {}
        self._executions = {}
        self._counter = 0
        self._context = None

    def _generate_lane(self, p, request, settings):
        env = p.behavior_environment
        origin, target = env.lane(request.origin_lane_id), env.lane(request.goal.target_lane_id)
        kinds=('synthetic',) if p.observation_mode=='model' else ('map','verified_fusion')
        frame=behavior_frame(p,time.monotonic())
        if (origin is None or target is None or not origin.evidence.verified(frame,allowed_source_kinds=kinds)
                or not target.evidence.verified(frame,allowed_source_kinds=kinds)):
            raise ValueError("required lane geometry unavailable")
        proj = project_polyline(origin.center_line, p.ego.x, p.ego.y)
        if proj is None:
            raise ValueError("origin projection unavailable")
        arc = stations(origin.center_line)
        start_s = proj['s']
        length = max(settings.behavior_lane_length_min_m, p.ego.speed*settings.behavior_transition_time_s)
        if start_s+length+12.0 > arc[-1]:
            raise ValueError("lane maneuver length unavailable")
        if not target.legal(behavior_frame(p, time.monotonic()), start_s, start_s+length):
            raise ValueError("crossing permission outside verified range")
        base = at(origin.center_line, arc, start_s)
        offset = -(p.ego.x-base[0])*math.sin(base[2])+(p.ego.y-base[1])*math.cos(base[2])
        raw = []
        for i in range(int(math.ceil((length+12.0)/settings.behavior_point_spacing_m))+1):
            distance = min(length+12.0, i*settings.behavior_point_spacing_m)
            x, y, heading = at(origin.center_line, arc, start_s+distance)
            other = project_polyline(target.center_line, x, y)
            if other is None or not 0 <= other['raw_ratio'] <= 1:
                raise ValueError("target lane coverage unavailable")
            u = min(1.0, distance/length)
            blend = 10*u**3-15*u**4+6*u**5
            delta = -(other['point'][0]-x)*math.sin(heading)+(other['point'][1]-y)*math.cos(heading)
            lateral = offset*(1-blend)+delta*blend
            raw.append((x-math.sin(heading)*lateral, y+math.cos(heading)*lateral))
        raw[0] = (p.ego.x, p.ego.y)
        return raw,start_s+length

    def _current_speed_cap(self,p,request,settings):
        target=p.behavior_environment.lane(request.goal.target_lane_id)
        origin=p.behavior_environment.lane(request.origin_lane_id)
        lead=next((t for t in p.targets if t.id==request.target_id),None)
        body=vehicle_footprint(GoalPose(p.ego.x,p.ego.y,p.ego.heading),
            settings.front_offset_m,settings.rear_offset_m,settings.half_width_m)
        a=project_polyline(origin.center_line,p.ego.x,p.ego.y) if origin else None
        b=project_polyline(origin.center_line,lead.x,lead.y) if origin and lead else None
        cap=request.goal.speed_cap_mps
        if target is not None and not target.contains(body) and a is not None and b is not None and b['s']>a['s']:
            speed=math.cos(b['heading'])*lead.vx+math.sin(b['heading'])*lead.vy
            cap=min(cap,max(0.05,speed))
        return cap

    def _profile(self, p, request, settings, raw, fallback):
        arc = stations(raw)
        headings = []
        for i, point in enumerate(raw):
            a, b = (raw[max(0,i-1)], raw[min(len(raw)-1,i+1)])
            headings.append(math.atan2(b[1]-a[1], b[0]-a[0]))
        headings[0] = p.ego.heading
        current_cap=self._current_speed_cap(p,request,settings)
        caps = [max(current_cap,math.sqrt(max(0.0,p.ego.speed**2-2*settings.deceleration*s))) for s in arc]
        for i in range(1,len(raw)-1):
            curve = abs(normalize_angle(headings[i+1]-headings[i-1])) / max(1e-6,arc[i+1]-arc[i-1])
            if curve > 1e-8:
                caps[i] = min(caps[i], math.sqrt(settings.lateral_acceleration/curve))
        # Keep the current follow cap while the body still shares the followed
        # object's strip. Once geometrically separated, allow overtake speed.
        lead = next((t for t in p.targets if t.id == request.target_id), None)
        if lead is not None:
            for i, point in enumerate(raw):
                lateral = abs(-(lead.x-point[0])*math.sin(headings[i])+(lead.y-point[1])*math.cos(headings[i]))
                if lateral < settings.half_width_m+lead.width/2+0.2:
                    lead_speed=max(0.0,math.cos(headings[i])*lead.vx+math.sin(headings[i])*lead.vy)
                    braking_envelope=math.sqrt(max(0.0,p.ego.speed**2-2*settings.deceleration*arc[i]))
                    caps[i] = min(caps[i], max(0.05,lead_speed,braking_envelope))
        # A known end is always reachable with zero terminal speed.
        caps[-1] = 0.0
        for i in range(len(caps)-2,-1,-1):
            caps[i] = min(caps[i],math.sqrt(caps[i+1]**2+2*settings.deceleration*(arc[i+1]-arc[i])))
        if p.ego.speed > caps[0]+0.1:
            raise ValueError("current speed incompatible with candidate")
        speeds = [p.ego.speed]
        for i in range(1,len(raw)):
            speeds.append(min(caps[i],math.sqrt(speeds[-1]**2+2*settings.acceleration*(arc[i]-arc[i-1]))))
        points, elapsed = [], 0.0
        for i, point in enumerate(raw):
            if i:
                if speeds[i]+speeds[i-1] <= 1e-8:
                    raise ValueError("zero-speed segment cannot advance")
                elapsed += 2*(arc[i]-arc[i-1])/(speeds[i]+speeds[i-1])
            points.append(TrajectoryPoint(point[0],point[1],speeds[i],headings[i],elapsed))
        return points

    def _validate(self, p, request, settings, points):
        env, coverage = p.behavior_environment, p.behavior_environment.coverage
        vehicle = VehicleGeometry(settings.front_offset_m,settings.rear_offset_m,settings.half_width_m,settings.wheelbase_m)
        limits = MotionLimits(max(request.goal.speed_cap_mps,p.ego.speed)+0.1,
            settings.acceleration,settings.deceleration,settings.lateral_acceleration,
            settings.front_steer_max_rad,settings.behavior_steer_rate_rad_s,0.2,0.05)
        obstacles=[]
        for t in p.targets:
            if mapped_traffic_light(t,p):
                continue
            if t.valid is not True or min(t.length,t.width) <= 0:
                raise ValueError("target dimensions invalid")
            # Business net distance remains longitudinal in overlapping strips.
            clearance = request.follow_clearance_m if t.id == request.target_id else request.static_clearance_m
            obstacles.append(ObstaclePrediction(str(t.id),t.x,t.y,t.length+2*clearance,t.width,
                t.heading,t.vx,t.vy,coverage.prediction_horizon_s,
                coverage.position_uncertainty_m,coverage.velocity_uncertainty_mps))
        if not all(coverage.contains(vehicle_footprint(GoalPose(q.x,q.y,q.heading),
                 settings.front_offset_m,settings.rear_offset_m,settings.half_width_m)) for q in points):
            raise ValueError("candidate leaves observed coverage")
        reports=[]
        for region in env.road_regions:
            report=validate_candidate(points,1,vehicle,limits,region,obstacles,
                ValidationBudget(30000,14,0.002,min(p.valid_until,time.monotonic()+settings.behavior_validation_budget_ms/1000.0)))
            reports.append(report)
            if report['status']=='safe':
                return report
        raise ValueError("candidate rejected: "+reports[0]['reason_code']+":"+reports[0]['constraint']+":"+str(reports[0]['details']))

    def plan(self, p, decision, settings, fallback):
        request=decision.behavior_request
        if not isinstance(request, RuntimeRequest):
            return fallback
        now=time.monotonic()
        frame=behavior_frame(p,now)
        if self._context != frame.context.key():
            self.reset(); self._context=frame.context.key()
        fallback.behavior_request=request
        lights=request.goal.light_intent
        fallback.left_signal,fallback.right_signal,fallback.hazard_signal=lights.left_signal,lights.right_signal,lights.hazard_signal
        if request.authorization in ('PREPARE','HOLD'):
            return fallback
        try:
            env=p.behavior_environment
            validate_runtime_request(request,frame)
            if not current(decision,p):
                raise ValueError('decision input invalid or expired')
            if request.maneuver not in ('OVERTAKE','LANE_CHANGE','AVOID','MERGE'):
                raise ValueError('behavior generator unsupported')
            caps=p.behavior_capabilities
            if (not isinstance(caps,RuntimeCapabilities) or caps.protocol_version!=PROTOCOL_VERSION
                    or not caps.allows(frame,('LANE_CHANGE','PATH','LIGHTS','FEEDBACK'))
                    or any(not math.isclose(caps.vehicle_geometry[name],getattr(settings,name),rel_tol=1e-9,abs_tol=1e-9)
                           for name in ('front_offset_m','rear_offset_m','half_width_m','wheelbase_m','front_steer_max_rad'))):
                raise ValueError('generator capabilities or shared body unavailable')
            if request.goal.motion_direction != 1:
                raise ValueError('lane generator does not support reverse authority')
            if (request.protocol_version != PROTOCOL_VERSION or not frame.current(now)
                    or request.context.key()!=frame.context.key() or request.produced_frame_id!=p.frame_id
                    or request.valid_until_s>p.valid_until or not request.dispatch_allowed
                    or not isinstance(env,BehaviorEnvironment) or not env.current(frame,p.observation_mode)):
                raise ValueError('request or environment unusable')
            if signal_stop_requirement(p) is not None:
                raise ValueError('traffic stop still applies')
            if request.authorization=='EVALUATE_ONLY':
                raw,cross_end_s=self._generate_lane(p,request,settings)
                points=self._profile(p,request,settings,raw,fallback)
                report=self._validate(p,request,settings,points)
                self._counter+=1
                reference=CandidateReference('lane:'+str(self._counter),request,frame.valid_until_s)
                self._candidates[reference.candidate_id]=(copy.deepcopy(reference),raw,request.goal.target_lane_id,request.origin_lane_id,cross_end_s)
                while len(self._candidates)>32:
                    self._candidates.pop(next(iter(self._candidates)))
                fallback.planning_feedback=[PlanningFeedback(request,frame,'PLANNED',candidate_reference=reference,
                    clearance_lower_bound_m=report['clearance_lower_bound_m'],computation_s=report['elapsed_s'])]
                return fallback
            reference=request.candidate_reference
            stored=self._candidates.get(reference.candidate_id)
            execution_key=(request.context.key(),request.intent_id,request.stage)
            adopted=self._executions.get(execution_key)
            if adopted is None:
                if (stored is None or stored[0].key()!=reference.key() or now>=min(reference.valid_until_s,stored[0].valid_until_s)
                        or stored[2]!=request.goal.target_lane_id):
                    raise ValueError('adopted candidate absent expired or mismatched')
                adopted=stored
                self._executions[execution_key]=adopted
                while len(self._executions)>64:
                    self._executions.pop(next(iter(self._executions)))
            if (adopted[0].candidate_id!=reference.candidate_id or adopted[0].key()!=reference.key()
                    or adopted[2]!=request.goal.target_lane_id or adopted[3]!=request.origin_lane_id):
                raise ValueError('execution candidate identity changed without evaluation')
            target=p.behavior_environment.lane(request.goal.target_lane_id)
            body=vehicle_footprint(GoalPose(p.ego.x,p.ego.y,p.ego.heading),
                settings.front_offset_m,settings.rear_offset_m,settings.half_width_m)
            kinds=('synthetic',) if p.observation_mode=='model' else ('map','verified_fusion')
            origin=env.lane(request.origin_lane_id)
            if (target is None or origin is None or not target.evidence.verified(frame,allowed_source_kinds=kinds)
                    or not origin.evidence.verified(frame,allowed_source_kinds=kinds) or not target.same_direction):
                raise ValueError('current corridor evidence unavailable')
            current_projection=project_polyline(origin.center_line,p.ego.x,p.ego.y)
            if (not target.contains(body) and (current_projection is None or not target.legal(frame,
                    current_projection['s'],max(current_projection['s'],adopted[4])))):
                raise ValueError('crossing authority withdrawn during execution')
            tail=project_polyline(adopted[1],p.ego.x,p.ego.y)
            if target is not None and target.contains(body) and tail is not None and len(adopted[1])-tail['index']<40:
                arc=stations(target.center_line)
                end=project_polyline(target.center_line,*adopted[1][-1])
                if end is not None:
                    extension=min(50.0,arc[-1]-end['s']-settings.front_offset_m-1.0)
                    if extension>1.0:
                        added=[at(target.center_line,arc,end['s']+j*0.5)[:2]
                               for j in range(1,int(extension/0.5)+1)]
                        adopted=(adopted[0],adopted[1]+added,adopted[2],adopted[3],adopted[4])
                        self._executions[execution_key]=adopted
            proj=project_polyline(adopted[1],p.ego.x,p.ego.y)
            if proj is None or proj['distance']>0.6:
                raise ValueError('execution left the verified candidate')
            source=adopted[1]
            arc=stations(source)
            start=proj['s']
            remaining=arc[-1]-start
            blend_length=min(8.0,remaining)
            base=at(source,arc,start)
            error=(p.ego.x-base[0],p.ego.y-base[1])
            derivative=(math.cos(p.ego.heading)-math.cos(base[2]),
                        math.sin(p.ego.heading)-math.sin(base[2]))
            raw=[]
            for i in range(int(math.ceil(remaining/0.5))+1):
                distance=min(remaining,i*0.5)
                point=at(source,arc,start+distance)
                u=min(1.0,distance/max(1e-6,blend_length))
                position_basis=1-10*u**3+15*u**4-6*u**5
                derivative_basis=u-6*u**3+8*u**4-3*u**5
                raw.append((point[0]+error[0]*position_basis+derivative[0]*blend_length*derivative_basis,
                            point[1]+error[1]*position_basis+derivative[1]*blend_length*derivative_basis))
            raw[0]=(p.ego.x,p.ego.y)
            if len(raw)<3:
                raise ValueError('execution candidate exhausted')
            points=self._profile(p,request,settings,raw,fallback)
            report=self._validate(p,request,settings,points)
            out=Trajectory().bind(decision)
            out.valid=True; out.points=points
            out.target_speed=self._current_speed_cap(p,request,settings)
            out.stop_required=True; out.stop_distance=stations(raw)[-1]
            out.target_lane_id=request.goal.target_lane_id
            out.behavior_request=request; out.execution_path_id=reference.candidate_id
            out.left_signal,out.right_signal,out.hazard_signal=lights.left_signal,lights.right_signal,lights.hazard_signal
            out.planning_feedback=[PlanningFeedback(request,frame,'PLANNED',candidate_reference=reference,
                clearance_lower_bound_m=report['clearance_lower_bound_m'],computation_s=report['elapsed_s'])]
            return out
        except (ValueError,TypeError,AttributeError,IndexError,OverflowError) as exc:
            fallback.planning_feedback=[PlanningFeedback(request,frame,'REJECTED',str(exc),retryable=True)]
            return fallback
