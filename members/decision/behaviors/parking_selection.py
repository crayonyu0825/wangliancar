"""Select one observed fitting bay; keep main's ParkingMission and planner."""
import math

from core.behavior_contract import GoalPose, finite, require
from core.geometry import project_polyline, projection_within_polyline
from core.maneuver_facts import ManeuverFacts
from core.parking_mission import ParkingMission, rear_axle_bay_goal
from core.corridor_region import CorridorRegion
from core.region_geometry import rectangle


def _pose_at(line, station):
    require(finite(station) and station >= 0, 'PARKING_APPROACH_STATION_INVALID')
    for a, b in zip(line, line[1:]):
        length = math.hypot(b[0]-a[0], b[1]-a[1])
        if length <= 1e-9:
            continue
        if station <= length:
            u = station/length
            return GoalPose(a[0]+u*(b[0]-a[0]), a[1]+u*(b[1]-a[1]),
                            math.atan2(b[1]-a[1], b[0]-a[0]))
        station -= length
    raise ValueError('PARKING_APPROACH_OUTSIDE_OBSERVED_ROAD')


class ParkingMissionSelector(object):
    def __init__(self, mission_id, road_lane_ids, candidates, exit_goal, clock, max_checks=100000):
        require(isinstance(candidates, (tuple, list)) and 1 <= len(candidates) <= 64,
                'PARKING_CANDIDATES_REQUIRED')
        require(type(max_checks) is int and max_checks > 0, 'PARKING_SELECTION_BUDGET_INVALID')
        self.mission_id, self.road_lane_ids = mission_id, tuple(road_lane_ids)
        self.candidates, self.exit_goal, self.clock = tuple(dict(v) for v in candidates), exit_goal, clock
        self.max_checks, self.context, self.selected = max_checks, None, None

    def select(self, p, evidence, model, map_digest):
        context = evidence.context.key()
        if self.context is not None and context != self.context:
            require(context[-1] != self.context[-1], 'PARKING_SELECTION_CONTEXT_CHANGED_WITHOUT_RESET')
            self.selected = None
        self.context = context
        facts = ManeuverFacts(p, self.clock, self.max_checks)
        require(facts.map_digest() == map_digest, 'PARKING_SELECTION_MAP_CHANGED')
        if self.selected is None:
            require(p.lane.lane_id in self.road_lane_ids, 'PARKING_APPROACH_ROAD_UNAUTHORIZED')
            road = facts.road(p.lane.lane_id)
            line = road['center_line']
            here = project_polyline(line, p.ego.x, p.ego.y)
            require(projection_within_polyline(here, len(line)), 'PARKING_APPROACH_PROJECTION_UNKNOWN')
            vehicle = model.vehicle
            extents = (vehicle.front_offset_m, vehicle.rear_offset_m, vehicle.half_width_m)
            checks = [0]
            def work():
                checks[0] += 1
                require(checks[0] <= self.max_checks, 'PARKING_SELECTION_BUDGET_EXHAUSTED')
                facts.check_current(True)
            choices = []
            for spec in self.candidates:
                try:
                    work()
                    bay = facts.parking(spec['space_id'])
                    require(bay['occupancy'] == 'empty', 'PARKING_BAY_NOT_OBSERVED_EMPTY')
                    goal = rear_axle_bay_goal(bay['boundary'], spec['body_heading_rad'], *extents[:2])
                    body = rectangle(goal.x, goal.y, goal.body_heading_rad, *extents)
                    require(CorridorRegion((bay['boundary'],)).body_clearance(body, work,
                            self.max_checks) > 0, 'PARKING_WHOLE_BODY_DOES_NOT_FIT')
                    station = project_polyline(line, goal.x, goal.y)
                    require(projection_within_polyline(station, len(line)), 'PARKING_BAY_OUTSIDE_SELECTED_ROAD')
                    margin = max(.3, model.execution_envelope.position_error_m)
                    braking = (p.ego.speed**2/(2*model.limits.max_deceleration_mps2)
                               + p.ego.speed*model.execution_envelope.late_time_s + margin)
                    approach = _pose_at(line, max(here['s']+braking,
                                                   station['s']+vehicle.front_offset_m+margin))
                    approach_body = rectangle(approach.x, approach.y, approach.body_heading_rad, *extents)
                    require(CorridorRegion((road['polygon'],)).body_clearance(approach_body,
                            work, self.max_checks) > 0,
                            'PARKING_APPROACH_WHOLE_BODY_OUTSIDE_ROAD')
                    choices.append((math.hypot(goal.x-p.ego.x, goal.y-p.ego.y), spec['space_id'], spec, approach))
                except (ValueError, TypeError, KeyError, OverflowError):
                    facts.check_current(True)
            work()
            require(choices, 'NO_OBSERVED_FITTING_EMPTY_BAY')
            unused, unused, spec, approach = min(choices, key=lambda v: v[:2])
            self.selected = (dict(spec), approach)
        spec, approach = self.selected
        facts.check_current(True)
        return ParkingMission(evidence, self.mission_id, map_digest, spec['space_id'],
            self.road_lane_ids, spec['access_edge_index'], spec['body_heading_rad'],
            approach, approach, self.exit_goal)
