"""Shared verified selected-route progress; SDK independent."""
from core.behavior_protocol import GoalPose,finite,require
from core.behavior_environment import Evidence
from core.road_region import RoadRegion


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
        self.destination_region = RoadRegion(destination_region).points if destination_region is not None else None
        self.destination_goal, self.speed_cap_mps = destination_goal, speed_cap_mps
        self.selected_branch_verified = selected_branch_verified
        require(all(isinstance(v,str) and bool(v) for v in required_obligations),"invalid task obligations")
        self.required_obligations=tuple(required_obligations)
