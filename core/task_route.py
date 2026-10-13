"""Selected task route facts using main's existing evidence and behavior contract."""
import copy
from core.behavior_contract import GoalPose, finite, require
from core.behavior_evidence import Evidence
from core.region_geometry import simple_outline


class TaskRouteFacts(object):
    def __init__(self, evidence, route_id, segments, segment_id, progress_s_m,
                 reference_end_reason, destination_region=None, destination_goal=None,
                 speed_cap_mps=3.0, selected_branch_verified=True, required_obligations=()):
        require(isinstance(evidence, Evidence) and isinstance(route_id, str) and bool(route_id), "invalid task route")
        require(isinstance(segments, (tuple, list)) and bool(segments), "task segments unavailable")
        normalized = []
        for segment in segments:
            require(isinstance(segment, (tuple, list)) and len(segment) == 4,
                    "segment requires identity,lane,start,end")
            identity, lane, start, end = segment
            require(isinstance(identity, str) and isinstance(lane, str) and finite(start) and finite(end)
                    and 0 <= start < end, "invalid task segment")
            require(not normalized or start >= normalized[-1][3], "task segment overlap")
            require(all(identity != s[0] for s in normalized), "ambiguous repeated segment identity")
            normalized.append((identity, lane, start, end))
        require(isinstance(segment_id, str) and finite(progress_s_m) and progress_s_m >= 0,
                "invalid task progress")
        require(reference_end_reason in ("task_destination", "window_end", "map_end", "branch_ambiguous", "map_query_failed"),
                "invalid reference end meaning")
        require(destination_goal is None or isinstance(destination_goal, GoalPose), "invalid task destination")
        require(finite(speed_cap_mps) and speed_cap_mps >= 0 and type(selected_branch_verified) is bool, "invalid task cap")
        self.evidence, self.route_id, self.segments, self.segment_id = evidence, route_id, tuple(normalized), segment_id
        self.progress_s_m, self.reference_end_reason = progress_s_m, reference_end_reason
        self.destination_region = simple_outline(destination_region) if destination_region is not None else None
        self.destination_goal, self.speed_cap_mps = destination_goal, speed_cap_mps
        self.selected_branch_verified = selected_branch_verified
        require(isinstance(required_obligations, (tuple, list)) and len(required_obligations) <= 128
                and all(isinstance(v, str) and v for v in required_obligations), 'invalid task obligations')
        self.required_obligations = tuple(required_obligations)

    def binding(self):
        goal = None if self.destination_goal is None else (self.destination_goal.x,
            self.destination_goal.y, self.destination_goal.body_heading_rad)
        return (self.route_id, self.segments, goal,
                None if self.destination_region is None else tuple(self.destination_region), self.required_obligations)


class TaskDecisionInputs(object):
    def __init__(self, facts, vehicle_extents, map_digest, stop_rule=None):
        require(isinstance(facts, TaskRouteFacts) and isinstance(vehicle_extents, (tuple, list))
                and len(vehicle_extents) == 3 and all(finite(v) and v > 0 for v in vehicle_extents)
                and isinstance(map_digest, str) and len(map_digest) == 32, 'TASK_INPUTS_INVALID')
        self.facts = copy.deepcopy(facts)
        self.vehicle_extents = tuple(vehicle_extents)
        self.map_digest = map_digest
        require(stop_rule is None or isinstance(stop_rule, dict) and set(stop_rule) ==
                {'id', 'minimum_duration_s', 'strict'} and isinstance(stop_rule['id'], str)
                and bool(stop_rule['id']) and finite(stop_rule['minimum_duration_s'])
                and stop_rule['minimum_duration_s'] > 0 and type(stop_rule['strict']) is bool,
                'TASK_STOP_RULE_INVALID')
        self.stop_rule = copy.deepcopy(stop_rule)

    def snapshot(self):
        return copy.deepcopy(self)
