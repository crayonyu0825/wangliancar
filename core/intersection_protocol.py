"""Formal route-relative intersection boundaries and time permissions.

Stations belong to the supplied finite, selected route. No nearest-lane ID,
scene number or signal colour supplies missing permission or exit coverage.
"""
import math
from core.behavior_protocol import finite,require,GoalPose
from core.behavior_environment import Evidence,MotionObject,hull,vehicle_footprint
from core.geometry import project_polyline,projection_within_polyline
from core.polyline_position import point_at_distance
from core.road_region import RoadRegion


def route_length(path):
    return sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(path,path[1:]))


def body_stations(path,body):
    values=[project_polyline(path,*p) for p in body]
    require(all(v is not None and projection_within_polyline(v,len(path)) for v in values),
            'body outside finite intersection route')
    return min(v['s'] for v in values),max(v['s'] for v in values)


def observed_objects(p):
    values=[]
    from core.target_semantics import mapped_traffic_light
    for t in p.targets:
        if mapped_traffic_light(t,p):continue
        require(t.valid is True and min(t.length,t.width)>0,'intersection target geometry unknown')
        length,width=t.length,t.width
        if t.heading is None:length=width=math.hypot(length,width)
        values.append(MotionObject(t.id,t.x,t.y,t.vx,t.vy,length,width,t.heading or 0.0))
    return values


class IntersectionTask(object):
    def __init__(self,identity,evidence,route,entry_s_m,exit_s_m,conflict_regions,
                 drivable_region,exit_region,permission_windows_s,right_of_way='UNKNOWN',
                 signal_permits=False,speed_cap_mps=2.0,exit_gap_m=4.0):
        require(isinstance(identity,str) and bool(identity) and isinstance(evidence,Evidence),'invalid junction identity')
        require(isinstance(route,(list,tuple)) and 2<=len(route)<=512,'selected intersection route missing')
        require(all(len(p)==2 and all(finite(v) for v in p) for p in route),'invalid intersection route')
        require(all(finite(v) for v in (entry_s_m,exit_s_m,speed_cap_mps,exit_gap_m))
                and 0<entry_s_m<exit_s_m<route_length(route) and speed_cap_mps>0 and exit_gap_m>=0,'invalid intersection bounds')
        require(right_of_way in ('EGO_PRIORITY','YIELD','UNKNOWN') and type(signal_permits) is bool,'unknown intersection rule')
        require(all(len(w)==2 and all(finite(v) for v in w) and w[0]<w[1] for w in permission_windows_s),'invalid permission windows')
        self.identity,self.evidence=identity,evidence
        self.route=tuple(tuple(p) for p in route)
        self.entry_s_m,self.exit_s_m=entry_s_m,exit_s_m
        self.conflict_regions=tuple(RoadRegion(p) for p in conflict_regions)
        require(bool(self.conflict_regions),'conflict area unknown')
        self.drivable_region,self.exit_region=RoadRegion(drivable_region),RoadRegion(exit_region)
        self.permission_windows_s=tuple(tuple(w) for w in permission_windows_s)
        self.right_of_way,self.signal_permits=right_of_way,signal_permits
        self.speed_cap_mps,self.exit_gap_m=speed_cap_mps,exit_gap_m

    def verified(self,frame,mode):
        kinds=('synthetic',) if mode=='model' else ('task','verified_fusion')
        return self.evidence.verified(frame,allowed_source_kinds=kinds)

    def stop_goal(self,geometry):
        station=self.entry_s_m-geometry['front_offset_m']-0.3
        return GoalPose(*point_at_distance(self.route,station))

    def exit_has_space(self,p,geometry):
        """Consecutive rear-axle placement range that fits the complete body."""
        required=geometry['front_offset_m']+geometry['rear_offset_m']+self.exit_gap_m
        horizon=self.exit_s_m+geometry['rear_offset_m']+required
        if horizon+geometry['front_offset_m']>=route_length(self.route):return False
        objects=observed_objects(p);coverage=p.behavior_environment.coverage
        for i in range(int(math.ceil(required/0.25))+1):
            s=self.exit_s_m+geometry['rear_offset_m']+0.3+min(required,i*0.25)
            pose=GoalPose(*point_at_distance(self.route,s))
            body=vehicle_footprint(pose,geometry['front_offset_m'],geometry['rear_offset_m'],geometry['half_width_m'])
            if self.exit_region.body_gap(body)<=0.05 or not coverage.contains(body):return False
            if any(self.exit_region.intersects(o.footprint(margin=0.2))
                   and RoadRegion(body).intersects(o.footprint(margin=0.2)) for o in objects):return False
        return True

    def open_windows(self,p,frame):
        coverage=p.behavior_environment.coverage
        if not self.verified(frame,p.observation_mode) or not self.signal_permits or self.right_of_way=='UNKNOWN':return ()
        end=frame.observed_at_s+coverage.prediction_horizon_s-coverage.time_late_error_s
        windows=[(max(frame.observed_at_s,a),min(end,b)) for a,b in self.permission_windows_s
                 if max(frame.observed_at_s,a)<min(end,b)]
        for obj in observed_objects(p):
            for i in range(int(math.ceil(coverage.prediction_horizon_s/0.25))):
                lo=i*0.25;hi=min(coverage.prediction_horizon_s,lo+0.25)
                early=max(0.0,lo-coverage.time_early_error_s)
                late=min(coverage.prediction_horizon_s,hi+coverage.time_late_error_s)
                margin=0.3+coverage.position_uncertainty_m+late*coverage.velocity_uncertainty_mps
                swept=obj.sweep(early,late,margin)
                if not any(region.intersects(swept) for region in self.conflict_regions):continue
                blocked=(frame.observed_at_s+max(0,lo-0.5),frame.observed_at_s+hi+0.5)
                remaining=[]
                for a,b in windows:
                    if blocked[1]<=a or blocked[0]>=b:remaining.append((a,b));continue
                    if a<blocked[0]:remaining.append((a,blocked[0]))
                    if blocked[1]<b:remaining.append((blocked[1],b))
                windows=remaining
        return tuple(windows)
