"""Explicit caller-verified parking task shared by decision and planning."""
import copy
import math
from core.behavior_contract import TaskContext,GoalPose,require,finite
from core.behavior_evidence import Evidence


def rear_axle_bay_goal(boundary,heading,front,rear):
    offset=(front-rear)/2.
    return GoalPose(sum(p[0] for p in boundary)/len(boundary)-offset*math.cos(heading),
                    sum(p[1] for p in boundary)/len(boundary)-offset*math.sin(heading),heading)


class ParkingMission(object):
    """Explicit verified-task snapshot, not inferred from bay/case numbers."""
    def __init__(self,evidence,mission_id,map_digest,space_id,road_lane_ids,access_edge_index,
                 body_heading_rad,approach_pose,position_pose,exit_goal):
        require(isinstance(evidence,Evidence) and isinstance(mission_id,str) and bool(mission_id)
                and isinstance(map_digest,str) and type(space_id) is int and space_id>=0,
                'PARKING_MISSION_IDENTITY_INVALID')
        require(isinstance(road_lane_ids,(tuple,list)) and 1<=len(road_lane_ids)<=3
                and all(isinstance(v,str) and v for v in road_lane_ids)
                and len(set(road_lane_ids))==len(road_lane_ids), 'PARKING_MISSION_ROAD_SET_INVALID')
        require(type(access_edge_index) is int and 0<=access_edge_index<=3
                and finite(body_heading_rad) and all(isinstance(v,GoalPose) for v in
                    (approach_pose,position_pose,exit_goal)), 'PARKING_MISSION_POSES_OR_ACCESS_UNAVAILABLE')
        context=TaskContext(*evidence.context.key())
        self.evidence=Evidence(context,evidence.frame_id,evidence.observed_at_s,evidence.valid_until_s,
            evidence.usable,evidence.coverage_verified,evidence.source,evidence.clock_id,evidence.source_kind)
        self.mission_id,self.map_digest=mission_id,map_digest
        self.space_id,self.road_lane_ids,self.access_edge_index=space_id,tuple(road_lane_ids),access_edge_index
        self.body_heading_rad=body_heading_rad
        self.approach_pose,self.position_pose,self.exit_goal=tuple(
            GoalPose(p.x,p.y,p.body_heading_rad) for p in (approach_pose,position_pose,exit_goal))

    def binding(self,bay):
        poses=tuple((v.x,v.y,v.body_heading_rad) for v in (self.approach_pose,self.position_pose,self.exit_goal))
        return (self.mission_id,self.map_digest,self.space_id,self.road_lane_ids,self.access_edge_index,
                self.body_heading_rad,poses,tuple(bay['boundary_knots']),tuple(bay['entrance_edge']),
                self.evidence.source_kind,self.evidence.source,self.evidence.clock_id)

    def snapshot(self):
        return ParkingMission(self.evidence,self.mission_id,self.map_digest,self.space_id,
            self.road_lane_ids,self.access_edge_index,self.body_heading_rad,
            self.approach_pose,self.position_pose,self.exit_goal)

    def stage_pose(self,stage,bay,front,rear):
        if stage=='APPROACH': return copy.deepcopy(self.approach_pose)
        if stage=='POSITION': return copy.deepcopy(self.position_pose)
        if stage=='EXIT': return copy.deepcopy(self.exit_goal)
        require(stage in ('REVERSE_ENTRY','ALIGN','PARKED_DWELL','EXIT_PREPARE'),
                'PARKING_STAGE_UNSUPPORTED')
        return rear_axle_bay_goal(bay['boundary'],self.body_heading_rad,front,rear)
