"""Explicit captain/task authority for passage and emergency maneuvers.

These SDK-independent inputs accompany the existing per-frame model provider.
They do not derive right of way from an empty detection list.
"""
import copy
import math
import re

from core.behavior_contract import GoalPose, finite, require
from core.behavior_evidence import Evidence


class DrivingMission(object):
    def __init__(self, mission_id, map_digest, evidence, model_evidence,
                 passage_id=None, allowed_windows_abs_s=(), right_of_way='UNKNOWN',
                 speed_cap_mps=3., entry_margin_m=.3, exit_margin_m=.3,
                 emergency_goals=(), immediate_brake_cycles=1):
        require(isinstance(mission_id, str) and bool(mission_id)
                and isinstance(map_digest, str) and re.fullmatch('[0-9a-f]{32}', map_digest) is not None,
                'DRIVING_MISSION_IDENTITY_INVALID')
        require(isinstance(evidence, Evidence) and isinstance(model_evidence, Evidence),
                'DRIVING_MISSION_EVIDENCE_REQUIRED')
        require(passage_id is None or isinstance(passage_id, str) and bool(passage_id), 'PASSAGE_ID_INVALID')
        require(right_of_way in ('UNKNOWN', 'YIELD', 'EGO_PRIORITY'), 'PASSAGE_AUTHORITY_INVALID')
        require(all(finite(v) and v >= 0 for v in (speed_cap_mps, entry_margin_m, exit_margin_m))
                and speed_cap_mps > 0, 'DRIVING_MISSION_LIMITS_INVALID')
        require(isinstance(allowed_windows_abs_s, (list, tuple)) and len(allowed_windows_abs_s) <= 64,
                'PASSAGE_WINDOWS_INVALID')
        previous = -1.
        for window in allowed_windows_abs_s:
            require(isinstance(window, (list, tuple)) and len(window) == 2
                    and all(finite(v) and v >= 0 for v in window)
                    and previous <= window[0] < window[1], 'PASSAGE_WINDOWS_INVALID')
            previous = window[1]
        require(isinstance(emergency_goals, (list, tuple)) and len(emergency_goals) <= 4,
                'EMERGENCY_GOAL_LIMIT')
        for option in emergency_goals:
            require(isinstance(option, dict) and set(option) == {'action', 'goal', 'lane_ids'}
                    and option['action'] in ('LEFT', 'RIGHT', 'BRAKE_LEFT', 'BRAKE_RIGHT')
                    and isinstance(option['goal'], GoalPose)
                    and isinstance(option['lane_ids'], (list, tuple)) and 1 <= len(option['lane_ids']) <= 2
                    and all(isinstance(v, str) and v for v in option['lane_ids']), 'EMERGENCY_GOAL_INVALID')
        require(type(immediate_brake_cycles) is int and 1 <= immediate_brake_cycles <= 5,
                'EMERGENCY_FIRST_RESPONSE_INVALID')
        self.mission_id, self.map_digest = mission_id, map_digest
        self.evidence, self.model_evidence = copy.deepcopy((evidence, model_evidence))
        self.passage_id, self.right_of_way = passage_id, right_of_way
        self.allowed_windows_abs_s = tuple(tuple(v) for v in allowed_windows_abs_s)
        self.speed_cap_mps, self.entry_margin_m, self.exit_margin_m = speed_cap_mps, entry_margin_m, exit_margin_m
        self.emergency_goals = copy.deepcopy(tuple(emergency_goals))
        self.immediate_brake_cycles = immediate_brake_cycles

    def snapshot(self):
        return copy.deepcopy(self)

    def signature(self):
        """Stable authority; refreshed observations cannot change the mission."""
        goals=tuple((v['action'],v['goal'].x,v['goal'].y,v['goal'].body_heading_rad,
                     tuple(v['lane_ids'])) for v in self.emergency_goals)
        return (self.mission_id,self.map_digest,self.passage_id,self.allowed_windows_abs_s,
                self.right_of_way,self.speed_cap_mps,self.entry_margin_m,self.exit_margin_m,
                goals,self.immediate_brake_cycles)


class DrivingDecisionInputs(object):
    def __init__(self, mission, model_id, vehicle_extents, max_deceleration_mps2,
                 execution_position_error_m=None, execution_heading_error_rad=None):
        require(isinstance(mission,DrivingMission) and isinstance(model_id,str) and bool(model_id)
                and isinstance(vehicle_extents,(tuple,list)) and len(vehicle_extents)==3
                and all(finite(v) and v>0 for v in vehicle_extents)
                and finite(max_deceleration_mps2) and max_deceleration_mps2>0,'DRIVING_DECISION_INPUTS_INVALID')
        self.mission,self.model_id=mission.snapshot(),model_id
        self.vehicle_extents=tuple(vehicle_extents)
        self.max_deceleration_mps2=max_deceleration_mps2
        require((execution_position_error_m is None and execution_heading_error_rad is None)
                or (finite(execution_position_error_m) and execution_position_error_m>=0
                    and finite(execution_heading_error_rad) and 0<=execution_heading_error_rad<=math.pi),
                'DRIVING_DECISION_EXECUTION_BOUNDS_INVALID')
        self.execution_position_error_m=execution_position_error_m
        self.execution_heading_error_rad=execution_heading_error_rad

    def snapshot(self):
        return copy.deepcopy(self)
