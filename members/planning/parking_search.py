"""Bounded bidirectional hybrid search from an actual rear-axle pose.

Only geometry is searched here. Full speed/time/clearance validation and
explicit segment authority remain with the production planner and decision.
"""
import heapq
import math
import time

from core.behavior_protocol import GoalPose
from core.behavior_environment import vehicle_footprint,intersects
from core.geometry import normalize_angle


class SearchBudgetExceeded(ValueError):
    pass


class ParkingSegment(object):
    def __init__(self,direction,poses):
        self.direction,self.poses=direction,tuple(poses)
        self.end_pose=poses[-1]


def _move(pose,direction,steering,distance,wheelbase):
    curvature=math.tan(steering)/wheelbase
    travelled=direction*distance
    heading=pose.body_heading_rad+curvature*travelled
    if abs(curvature)<1e-8:
        x=pose.x+travelled*math.cos(pose.body_heading_rad)
        y=pose.y+travelled*math.sin(pose.body_heading_rad)
    else:
        x=pose.x+(math.sin(heading)-math.sin(pose.body_heading_rad))/curvature
        y=pose.y-(math.cos(heading)-math.cos(pose.body_heading_rad))/curvature
    return GoalPose(x,y,normalize_angle(heading))


def _clear(pose,regions,objects,vehicle,deadline=None):
    if deadline is not None and time.perf_counter()>=deadline:raise SearchBudgetExceeded("parking search budget exhausted")
    body=vehicle_footprint(pose,vehicle.front_offset_m,vehicle.rear_offset_m,vehicle.half_width_m)
    return (any(region.body_gap(body)>0.03 for region in regions)
            and not any(intersects(body,obj.footprint(margin=0.1)) for obj in objects))


def _connector(start,goal,direction,regions,objects,vehicle,max_steer,max_distance=3.0,deadline=None):
    length=math.hypot(goal.x-start.x,goal.y-start.y)
    if length<1e-8:
        return [start,goal] if abs(normalize_angle(goal.body_heading_rad-start.body_heading_rad))<0.03 else None
    if length>max_distance:return None
    a=start.body_heading_rad+(math.pi if direction<0 else 0)
    b=goal.body_heading_rad+(math.pi if direction<0 else 0)
    tangent0=(math.cos(a)*length,math.sin(a)*length)
    tangent1=(math.cos(b)*length,math.sin(b)*length)
    result=[]
    count=max(2,int(math.ceil(length/0.1)))
    previous=None
    for i in range(count+1):
        u=i/count
        h00,h10,h01,h11=2*u**3-3*u*u+1,u**3-2*u*u+u,-2*u**3+3*u*u,u**3-u*u
        dx=(6*u*u-6*u)*start.x+(3*u*u-4*u+1)*tangent0[0]+(-6*u*u+6*u)*goal.x+(3*u*u-2*u)*tangent1[0]
        dy=(6*u*u-6*u)*start.y+(3*u*u-4*u+1)*tangent0[1]+(-6*u*u+6*u)*goal.y+(3*u*u-2*u)*tangent1[1]
        ddx=(12*u-6)*start.x+(6*u-4)*tangent0[0]+(-12*u+6)*goal.x+(6*u-2)*tangent1[0]
        ddy=(12*u-6)*start.y+(6*u-4)*tangent0[1]+(-12*u+6)*goal.y+(6*u-2)*tangent1[1]
        speed=math.hypot(dx,dy)
        if speed<1e-8:return None
        curvature=(dx*ddy-dy*ddx)/speed**3
        if abs(math.atan(vehicle.wheelbase_m*curvature))>max_steer:return None
        pose=GoalPose(h00*start.x+h10*tangent0[0]+h01*goal.x+h11*tangent1[0],
                      h00*start.y+h10*tangent0[1]+h01*goal.y+h11*tangent1[1],
                      normalize_angle(math.atan2(dy,dx)-(math.pi if direction<0 else 0)))
        if not _clear(pose,regions,objects,vehicle,deadline):return None
        if previous is not None and abs(normalize_angle(pose.body_heading_rad-previous.body_heading_rad))>0.15:return None
        result.append(pose);previous=pose
    result[-1]=goal
    return result



def _forward_csc(start,goal,regions,objects,vehicle,max_steer,deadline=None):
    """Rank analytic CSC geometry before the expensive swept-body checks."""
    radius=vehicle.wheelbase_m/math.tan(max_steer)
    candidates=[]
    def mod(value):
        result=value%(2*math.pi)
        return 0.0 if result>2*math.pi-1e-7 else result
    for first_sign in (-1,1):
        for last_sign in (-1,1):
            a=(start.x-first_sign*radius*math.sin(start.body_heading_rad),start.y+first_sign*radius*math.cos(start.body_heading_rad))
            b=(goal.x-last_sign*radius*math.sin(goal.body_heading_rad),goal.y+last_sign*radius*math.cos(goal.body_heading_rad))
            dx,dy=b[0]-a[0],b[1]-a[1];distance=math.hypot(dx,dy)
            if distance<1e-8:continue
            heading=math.atan2(dy,dx)
            if first_sign!=last_sign:
                if distance<2*radius:continue
                heading+=math.asin(2*first_sign*radius/distance)
            first_angle=mod(first_sign*(heading-start.body_heading_rad))
            last_angle=mod(last_sign*(goal.body_heading_rad-heading))
            straight=distance*math.cos(math.atan2(dy,dx)-heading)
            if straight<0:continue
            primitives=((first_sign*max_steer,first_angle*radius),(0.0,straight),(last_sign*max_steer,last_angle*radius))
            candidates.append((first_angle*radius+straight+last_angle*radius,primitives))
    for length,primitives in sorted(candidates,key=lambda value:value[0]):
        legs=[];pose=start;safe=True
        for steering,extent in primitives:
            if extent<1e-7:continue
            count=max(2,int(math.ceil(extent/0.1)))
            points=[pose]+[_move(pose,1,steering,extent*i/count,vehicle.wheelbase_m) for i in range(1,count+1)]
            if not all(_clear(p,regions,objects,vehicle,deadline) for p in points):safe=False;break
            legs.append(ParkingSegment(1,points));pose=points[-1]
        if (safe and legs and math.hypot(pose.x-goal.x,pose.y-goal.y)<0.01
                and abs(normalize_angle(pose.body_heading_rad-goal.body_heading_rad))<0.01):
            return length,legs
    return None


def _analytic_seed(start,goal,regions,objects,vehicle,max_steer,final_direction,deadline=None):
    bearing=math.atan2(goal.y-start.y,goal.x-start.x)
    for direction in (1,-1):
        if final_direction is not None and direction!=final_direction:continue
        travel=start.body_heading_rad+(math.pi if direction<0 else 0)
        if (abs(normalize_angle(bearing-travel))<0.1
                and abs(normalize_angle(goal.body_heading_rad-start.body_heading_rad))<0.1):
            direct=_connector(start,goal,direction,regions,objects,vehicle,max_steer,max_distance=50.0,deadline=deadline)
            if direct is not None:return [ParkingSegment(direction,direct)]
    if final_direction in (None,1):
        candidate=_forward_csc(start,goal,regions,objects,vehicle,max_steer,deadline)
        if candidate is not None:return candidate[1]
    distance=math.hypot(start.x-goal.x,start.y-goal.y)
    bearing=math.atan2(goal.y-start.y,goal.x-start.x)
    if (final_direction in (None,-1) and abs(normalize_angle(bearing-start.body_heading_rad-math.pi))<0.01
            and abs(normalize_angle(goal.body_heading_rad-start.body_heading_rad))<0.01):
        count=max(2,int(math.ceil(distance/0.1)))
        tail=[_move(start,-1,0.0,distance*i/count,vehicle.wheelbase_m) for i in range(count+1)]
        if all(_clear(p,regions,objects,vehicle,deadline) for p in tail):
            tail[-1]=goal
            return [ParkingSegment(-1,tail)]
    for distance in (3.0,4.0,6.0,8.0):
        approach=GoalPose(goal.x+distance*math.cos(goal.body_heading_rad),
                          goal.y+distance*math.sin(goal.body_heading_rad),goal.body_heading_rad)
        candidate=_forward_csc(start,approach,regions,objects,vehicle,max_steer,deadline)
        if candidate is None:continue
        tail=[_move(approach,-1,0.0,distance*i/max(2,int(distance/0.1)),vehicle.wheelbase_m)
              for i in range(max(2,int(distance/0.1))+1)]
        if all(_clear(p,regions,objects,vehicle,deadline) for p in tail):
            tail[-1]=goal
            return candidate[1]+[ParkingSegment(-1,tail)]
    return None

def _search_parking(start,goal,regions,objects,vehicle,max_steer,max_nodes=14000,
                   budget_s=0.5,step_m=0.5,position_grid_m=0.25,heading_grid_rad=math.pi/36,final_direction=None,deadline=None):
    if not _clear(start,regions,objects,vehicle,deadline) or not _clear(goal,regions,objects,vehicle,deadline):
        return [],dict(status='REJECTED',reason='START_OR_GOAL_OBSTRUCTED',expanded=0)
    began=time.perf_counter()
    if math.hypot(start.x-goal.x,start.y-goal.y)<=0.15 and abs(normalize_angle(start.body_heading_rad-goal.body_heading_rad))<=0.15:
        direction=final_direction if final_direction is not None else 1
        return [ParkingSegment(direction,[start,start])],dict(status='PLANNED',reason='ACTUAL_POSE_ALREADY_IN_GOAL_TOLERANCE',
            expanded=0,elapsed_s=time.perf_counter()-began)
    analytic=_analytic_seed(start,goal,regions,objects,vehicle,max_steer,final_direction,deadline)
    if analytic is not None:
        return analytic,dict(status='PLANNED',reason='BOUNDED_ANALYTIC_PRIMITIVES',expanded=0,elapsed_s=time.perf_counter()-began)
    def key(p,d):return (round(p.x/position_grid_m),round(p.y/position_grid_m),round(normalize_angle(p.body_heading_rad)/heading_grid_rad),d)
    radius=vehicle.wheelbase_m/math.tan(max_steer)
    def heuristic(p):return math.hypot(p.x-goal.x,p.y-goal.y)+radius*abs(normalize_angle(p.body_heading_rad-goal.body_heading_rad))*0.6
    nodes=[(start,1,0.0,None,[],0.0)]
    queue=[(heuristic(start),0)]
    best={key(start,1):0.0}
    expanded=0;found=None;connector=None
    while queue and expanded<max_nodes and time.perf_counter()-began<budget_s:
        _,index=heapq.heappop(queue)
        pose,direction,cost,parent,primitive,steering=nodes[index]
        expanded+=1
        for finish_direction in (direction,-direction):
            if final_direction is not None and finish_direction!=final_direction:continue
            tail=_connector(pose,goal,finish_direction,regions,objects,vehicle,max_steer,deadline=deadline)
            if tail is not None:
                found=index;connector=(finish_direction,tail);break
        if found is not None:break
        for requested in (direction,-direction):
            for angle in (-max_steer,-max_steer/2,0.0,max_steer/2,max_steer):
                poses=[_move(pose,requested,angle,step_m*j/5,vehicle.wheelbase_m) for j in range(1,6)]
                if not all(_clear(p,regions,objects,vehicle,deadline) for p in poses):continue
                endpoint=poses[-1]
                next_cost=cost+step_m*(1.15 if requested<0 else 1.0)+(2.0 if requested!=direction else 0)+0.1*abs(angle-steering)
                identity=key(endpoint,requested)
                if next_cost>=best.get(identity,float('inf')):continue
                best[identity]=next_cost
                nodes.append((endpoint,requested,next_cost,index,poses,angle))
                child=len(nodes)-1
                heapq.heappush(queue,(next_cost+heuristic(endpoint),child))
    report=dict(expanded=expanded,elapsed_s=time.perf_counter()-began,status='PLANNED' if found is not None else 'REJECTED',
                reason='GEOMETRY_FOUND' if found is not None else 'SEARCH_BUDGET_OR_NO_PATH')
    if found is None:return [],report
    legs=[];index=found
    while nodes[index][3] is not None:
        legs.append((nodes[index][1],nodes[index][4]));index=nodes[index][3]
    legs.reverse();legs.append(connector)
    segments=[];current=[start];direction=legs[0][0]
    for requested,poses in legs:
        if requested!=direction:
            segments.append(ParkingSegment(direction,current));current=[current[-1]];direction=requested
        current.extend(poses if poses[0] is not current[-1] else poses[1:])
    segments.append(ParkingSegment(direction,current))
    return segments,report


def search_parking(start,goal,regions,objects,vehicle,max_steer,max_nodes=14000,
                   budget_s=0.5,step_m=0.5,position_grid_m=0.25,heading_grid_rad=math.pi/36,final_direction=None):
    began=time.perf_counter()
    try:
        return _search_parking(start,goal,regions,objects,vehicle,max_steer,max_nodes,budget_s,step_m,
                               position_grid_m,heading_grid_rad,final_direction,began+budget_s)
    except SearchBudgetExceeded:
        return [],dict(status='REJECTED',reason='SEARCH_BUDGET_EXHAUSTED',elapsed_s=time.perf_counter()-began,expanded=0)
