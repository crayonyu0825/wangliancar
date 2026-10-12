"""Typed R06/R08/R09 observation proposals and SDK-free geometry checks."""

import math
from members.decision.behaviors.contract import TaskContext, GoalPose, finite, require


from core.behavior_evidence import Evidence


# Shared geometry preserves the existing decision entry points.
from core.region_geometry import convex_polygon as polygon, inside, intersects, hull


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
                and all(finite(v) for v in (x, y, vx, vy, length, width))
                and (heading is None or finite(heading))
                and min(length, width) > 0, "invalid motion object")
        self.identifier, self.x, self.y, self.vx, self.vy = identifier, x, y, vx, vy
        self.length, self.width, self.heading = length, width, heading

    def footprint(self, time_s=0.0, margin=0.0):
        require(finite(time_s) and time_s >= 0 and finite(margin) and margin >= 0, "invalid prediction range")
        if self.heading is None:
            # An axis-aligned square encloses the bounding circle for every
            # possible body orientation; zero is its envelope orientation.
            radius = .5 * math.hypot(self.length, self.width) + margin
            return vehicle_footprint(GoalPose(self.x+self.vx*time_s, self.y+self.vy*time_s, 0.),
                                     radius, radius, radius)
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
