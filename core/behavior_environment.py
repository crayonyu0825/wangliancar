"""Typed R06/R08/R09 observation proposals and SDK-free geometry checks."""

import math
from core.behavior_protocol import TaskContext, GoalPose, finite, require
from core.road_region import RoadRegion


class Evidence(object):
    def __init__(self, context, frame_id, observed_at_s, valid_until_s, usable=False,
                 coverage_verified=False, source="unknown", clock_id="process_monotonic",
                 source_kind="unknown"):
        require(isinstance(context, TaskContext) and type(frame_id) is int and frame_id >= 0,
                "invalid observation identity")
        require(finite(observed_at_s) and finite(valid_until_s) and type(usable) is bool
                and type(coverage_verified) is bool and isinstance(source, str), "invalid observation evidence")
        self.context, self.frame_id, self.observed_at_s, self.valid_until_s = context, frame_id, observed_at_s, valid_until_s
        require(source_kind in ("unknown", "sensor", "map", "task", "planning", "verified_fusion",
                                "synthetic", "diagnostic_ground_truth"), "invalid observation source kind")
        self.usable, self.coverage_verified, self.source, self.clock_id = usable, coverage_verified, source, clock_id
        self.source_kind = source_kind

    def verified(self, frame, max_age_s=0.5,
                 allowed_source_kinds=("sensor", "map", "task", "planning", "verified_fusion", "synthetic")):
        return (self.source_kind in allowed_source_kinds and self.usable and self.coverage_verified and self.source != "unknown"
                and self.context.key() == frame.context.key() and self.clock_id == frame.clock_id
                and self.frame_id <= frame.frame_id
                and self.observed_at_s <= frame.observed_at_s < self.valid_until_s
                and frame.observed_at_s - self.observed_at_s <= max_age_s
                and frame.current(frame.observed_at_s) and not frame.paused)


def polygon(raw):
    require(isinstance(raw, (tuple, list)) and 3 <= len(raw) <= 128, "polygon coverage missing")
    points = []
    for item in raw:
        require(isinstance(item, (tuple, list)) and len(item) == 2
                and all(finite(v) for v in item), "invalid polygon coordinate")
        points.append((float(item[0]), float(item[1])))
    area = sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(points, points[1:] + points[:1]))
    require(abs(area) > 1e-8, "degenerate polygon")
    signs = []
    for i in range(len(points)):
        a, b, c = points[i - 1], points[i], points[(i + 1) % len(points)]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-8:
            signs.append(cross > 0)
    require(bool(signs) and all(s == signs[0] for s in signs), "nonconvex or ambiguous polygon")
    return points


def inside(point, region):
    signs = []
    for a, b in zip(region, region[1:] + region[:1]):
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        if abs(cross) > 1e-8:
            signs.append(cross > 0)
    return not signs or all(s == signs[0] for s in signs)


def intersects(left, right):
    for shape in (left, right):
        for a, b in zip(shape, shape[1:] + shape[:1]):
            nx, ny = -(b[1] - a[1]), b[0] - a[0]
            l = [nx * p[0] + ny * p[1] for p in left]
            r = [nx * p[0] + ny * p[1] for p in right]
            if max(l) < min(r) - 1e-8 or max(r) < min(l) - 1e-8:
                return False
    return True


def hull(points):
    ordered = sorted(set(points))
    require(len(ordered) >= 3, "insufficient swept footprint")
    def cross(a, b, c):
        return (b[0]-a[0]) * (c[1]-a[1]) - (b[1]-a[1]) * (c[0]-a[0])
    lower, upper = [], []
    for p in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def vehicle_footprint(pose, front, rear, half_width):
    require(isinstance(pose, GoalPose) and all(finite(v) and v > 0 for v in (front, rear, half_width)),
            "vehicle footprint unavailable")
    c, s = math.cos(pose.body_heading_rad), math.sin(pose.body_heading_rad)
    return [(pose.x + x*c - y*s, pose.y + x*s + y*c)
            for x, y in ((front, half_width), (front, -half_width),
                         (-rear, -half_width), (-rear, half_width))]


def corridor_contains(body, left_boundary, right_boundary):
    """Whole convex body in an upstream-verified simple lane outline.

    Ray membership supports curved/concave corridors. Boundary crossings also
    reject a body spanning a concavity even when all four corners are inside.
    This checks settlement/space only, never maneuver-path feasibility.
    """
    outline = list(left_boundary) + list(reversed(right_boundary))
    if len(outline) < 4:
        return False

    def cross(a, b, c):
        return (b[0]-a[0]) * (c[1]-a[1]) - (b[1]-a[1]) * (c[0]-a[0])

    def member(point):
        contained = False
        for a, b in zip(outline, outline[1:] + outline[:1]):
            if (abs(cross(a, b, point)) <= 1e-8
                    and min(a[0], b[0])-1e-8 <= point[0] <= max(a[0], b[0])+1e-8
                    and min(a[1], b[1])-1e-8 <= point[1] <= max(a[1], b[1])+1e-8):
                return True
            if (a[1] > point[1]) != (b[1] > point[1]):
                x = a[0] + (point[1]-a[1]) * (b[0]-a[0]) / (b[1]-a[1])
                if x > point[0]:
                    contained = not contained
        return contained

    if not all(member(point) for point in body):
        return False
    for a, b in zip(body, body[1:] + body[:1]):
        for c, d in zip(outline, outline[1:] + outline[:1]):
            if cross(a, b, c) * cross(a, b, d) < -1e-8 and cross(c, d, a) * cross(c, d, b) < -1e-8:
                return False
    # A corridor vertex strictly inside the body marks a cutout which may touch
    # the body's edge without a proper crossing.
    for point in outline:
        signs = [cross(a, b, point) for a, b in zip(body, body[1:] + body[:1])]
        if all(v > 1e-8 for v in signs) or all(v < -1e-8 for v in signs):
            return False
    return True


class MotionObject(object):
    def __init__(self, identifier, x, y, vx, vy, length, width, heading=0.0):
        require(type(identifier) is int and identifier >= 0
                and all(finite(v) for v in (x, y, vx, vy, length, width, heading))
                and min(length, width) > 0, "invalid motion object")
        self.identifier, self.x, self.y, self.vx, self.vy = identifier, x, y, vx, vy
        self.length, self.width, self.heading = length, width, heading

    def footprint(self, time_s=0.0, margin=0.0):
        require(finite(time_s) and time_s >= 0 and finite(margin) and margin >= 0, "invalid prediction range")
        return vehicle_footprint(GoalPose(self.x+self.vx*time_s, self.y+self.vy*time_s, self.heading),
                                 self.length/2+margin, self.length/2+margin, self.width/2+margin)

    def sweep(self, start_s, end_s, margin=0.0):
        return hull(self.footprint(start_s, margin) + self.footprint(end_s, margin))


def occupancy_windows(objects, region, horizon_s=3.0, step_s=0.25, uncertainty_m=0.3):
    require(finite(horizon_s) and 0 < horizon_s <= 20 and finite(step_s) and 0 < step_s <= horizon_s,
            "invalid occupancy horizon")
    windows = []
    for obj in objects:
        require(isinstance(obj, MotionObject), "invalid occupancy object")
        index = 0
        while index * step_s < horizon_s:
            start = index * step_s
            end = min(horizon_s, start + step_s)
            if intersects(obj.sweep(start, end, uncertainty_m), region):
                windows.append((obj.identifier, start, end))
            index += 1
    return windows


class LaneCorridor(object):
    """World XY map lane with explicitly bounded crossing permission."""
    def __init__(self, lane_id, center_line, left_boundary, right_boundary, evidence,
                 side="current", same_direction=False, marking="UNKNOWN", crossing_range_m=None):
        require(isinstance(lane_id, str) and bool(lane_id) and isinstance(evidence, Evidence), "invalid lane identity")
        def points(raw):
            require(isinstance(raw, (tuple, list)) and 2 <= len(raw) <= 20000, "lane geometry missing")
            result = [tuple(p[:2]) for p in raw]
            require(all(len(p) == 2 and all(finite(v) for v in p) for p in result), "lane geometry invalid")
            return result
        self.lane_id, self.center_line = lane_id, points(center_line)
        self.left_boundary, self.right_boundary = points(left_boundary), points(right_boundary)
        require(side in ("current", "left", "right") and type(same_direction) is bool, "invalid lane relationship")
        require(crossing_range_m is None or (len(crossing_range_m) == 2 and all(finite(v) for v in crossing_range_m)
                and 0 <= crossing_range_m[0] < crossing_range_m[1]), "invalid crossable range")
        self.evidence, self.side, self.same_direction = evidence, side, same_direction
        self.marking, self.crossing_range_m = marking, crossing_range_m

    def legal(self, frame, required_start_m, required_end_m):
        return (self.evidence.verified(frame, allowed_source_kinds=("map", "verified_fusion", "synthetic"))
                and self.same_direction and self.marking in ("DASHED", "BROKEN")
                and self.crossing_range_m is not None
                and self.crossing_range_m[0] <= required_start_m <= required_end_m <= self.crossing_range_m[1])

    def contains(self, body):
        return corridor_contains(body, self.left_boundary, self.right_boundary)


class ObservationCoverage(object):
    def __init__(self, evidence, regions=(), scope=(), proof_id="", proof_kind="unknown",
                 position_uncertainty_m=None, velocity_uncertainty_mps=None, prediction_horizon_s=None,
                 time_early_error_s=None,time_late_error_s=None,timing_evidence=None):
        require(isinstance(evidence, Evidence), "coverage evidence missing")
        self.evidence, self.regions = evidence, tuple(polygon(p) for p in regions)
        self.scope, self.proof_id, self.proof_kind = frozenset(scope), proof_id, proof_kind
        for v in (position_uncertainty_m, velocity_uncertainty_mps, prediction_horizon_s):
            require(v is None or finite(v) and v >= 0, "invalid prediction uncertainty")
        self.position_uncertainty_m, self.velocity_uncertainty_mps = position_uncertainty_m, velocity_uncertainty_mps
        self.prediction_horizon_s = prediction_horizon_s
        require(all(v is None or finite(v) and v>=0 for v in (time_early_error_s,time_late_error_s)),
                "invalid execution timing bounds")
        require(timing_evidence is None or isinstance(timing_evidence,Evidence),"invalid timing evidence")
        self.time_early_error_s,self.time_late_error_s=time_early_error_s,time_late_error_s
        self.timing_evidence=timing_evidence

    def timing_verified(self,frame,mode):
        kinds=("synthetic",) if mode=="model" else ("verified_fusion","planning")
        return (self.time_early_error_s is not None and self.time_late_error_s is not None
                and self.timing_evidence is not None
                and self.timing_evidence.verified(frame,allowed_source_kinds=kinds))

    def verified(self, frame, mode="production", required=("front", "side", "rear")):
        source_kinds = ("sensor", "verified_fusion") if mode == "production" else ("synthetic",)
        proof = "sensor_fov_occlusion_validated" if mode == "production" else "model_complete_world"
        return (mode in ("production", "model") and self.evidence.verified(frame, allowed_source_kinds=source_kinds)
                and bool(self.proof_id) and self.proof_kind == proof and bool(self.regions)
                and set(required) <= self.scope
                and self.position_uncertainty_m is not None and self.velocity_uncertainty_mps is not None
                and self.prediction_horizon_s is not None and self.prediction_horizon_s > 0)

    def contains(self, body):
        return any(all(inside(p, region) for p in body) for region in self.regions)


class BehaviorEnvironment(object):
    def __init__(self, lanes=(), coverage=None, road_regions=(), return_lane_id=None,
                 task_route_id=None, destination=None, rule_stop_duration_s=0.0,
                 rule_stop_strict=False, stop_rule_id=None, task_evidence=None, desired_lane_id=None, parking_task=None, intersection_task=None, emergency_task=None, task_progress=None):
        require(all(isinstance(lane, LaneCorridor) for lane in lanes), "invalid behavior lanes")
        require(coverage is None or isinstance(coverage, ObservationCoverage), "invalid behavior coverage")
        require(all(v is None or isinstance(v, str) for v in (return_lane_id, task_route_id, stop_rule_id)), "invalid task identity")
        require(destination is None or isinstance(destination, GoalPose), "invalid task destination")
        require(finite(rule_stop_duration_s) and rule_stop_duration_s >= 0 and type(rule_stop_strict) is bool, "invalid stop policy")
        self.lanes, self.coverage = tuple(lanes), coverage
        self.road_regions = tuple(RoadRegion(p) for p in road_regions)
        self.return_lane_id, self.task_route_id, self.destination = return_lane_id, task_route_id, destination
        self.rule_stop_duration_s, self.rule_stop_strict, self.stop_rule_id = rule_stop_duration_s, rule_stop_strict, stop_rule_id
        require(task_evidence is None or isinstance(task_evidence,Evidence),"invalid task evidence")
        self.task_evidence=task_evidence
        require(desired_lane_id is None or isinstance(desired_lane_id,str) and bool(desired_lane_id),"invalid required task lane")
        self.desired_lane_id=desired_lane_id
        self.parking_task=parking_task
        self.intersection_task=intersection_task
        self.emergency_task=emergency_task
        self.task_progress=task_progress

    def verified_task(self,frame,mode):
        kinds=("synthetic",) if mode=="model" else ("task","verified_fusion")
        return (bool(self.task_route_id) and self.task_evidence is not None
                and self.task_evidence.verified(frame,allowed_source_kinds=kinds))

    def lane(self, lane_id):
        return next((v for v in self.lanes if v.lane_id == lane_id), None)

    def current(self, frame, mode):
        return (self.coverage is not None and self.coverage.verified(frame, mode)
                and bool(self.road_regions))
