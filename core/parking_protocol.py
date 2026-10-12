"""Shared parking observations and authorized segment identities."""
import math
from core.behavior_protocol import GoalPose,finite,require
from core.behavior_environment import Evidence,vehicle_footprint
from core.road_region import RoadRegion


class ParkingBay(object):
    def __init__(self,identity,boundary,body_heading_rad,occupancy,evidence):
        require(type(identity) is int and identity>=0 and finite(body_heading_rad),'invalid parking bay')
        require(occupancy in ('empty','self','other','unknown') and isinstance(evidence,Evidence),'invalid parking occupancy')
        self.identity,self.region=identity,RoadRegion(boundary)
        self.body_heading_rad,self.occupancy,self.evidence=body_heading_rad,occupancy,evidence

    def rear_axle_goal(self,geometry):
        x=sum(p[0] for p in self.region.points)/len(self.region.points)
        y=sum(p[1] for p in self.region.points)/len(self.region.points)
        offset=(geometry['front_offset_m']-geometry['rear_offset_m'])/2
        return GoalPose(x-offset*math.cos(self.body_heading_rad),y-offset*math.sin(self.body_heading_rad),self.body_heading_rad)


class ParkingTask(object):
    def __init__(self,identity,evidence,bays,free_regions,exit_goal,
                 minimum_dwell_s=10.0,ego_target_id=None):
        require(isinstance(identity,str) and bool(identity) and isinstance(evidence,Evidence),'invalid parking task')
        require(all(isinstance(b,ParkingBay) for b in bays) and isinstance(exit_goal,GoalPose),'parking task facts missing')
        require(len(set(b.identity for b in bays))==len(bays),'ambiguous repeated parking identity')
        require(finite(minimum_dwell_s) and minimum_dwell_s>=10.0,'parking dwell must be at least ten seconds')
        require(ego_target_id is None or type(ego_target_id) is int and ego_target_id>=0,'invalid observed ego identity')
        self.identity,self.evidence,self.bays=identity,evidence,tuple(bays)
        self.free_regions=tuple(RoadRegion(p) for p in free_regions)
        self.exit_goal,self.minimum_dwell_s,self.ego_target_id=exit_goal,minimum_dwell_s,ego_target_id

    def verified(self,frame,mode):
        kinds=('synthetic',) if mode=='model' else ('verified_fusion','task')
        return bool(self.free_regions) and self.evidence.verified(frame,allowed_source_kinds=kinds)

    def body_clear(self,pose,geometry):
        body=vehicle_footprint(pose,geometry['front_offset_m'],geometry['rear_offset_m'],geometry['half_width_m'])
        return any(region.body_gap(body)>0.03 for region in self.free_regions)
