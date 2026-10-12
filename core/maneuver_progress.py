"""Bounded actual-pose progress, with no time-based invented motion."""
import math
from collections import OrderedDict

from core.behavior_contract import finite, require
from core.geometry import normalize_angle


class ProgressTracker(object):
    def __init__(self, max_entries=128, max_gap_s=.5, projection_margin_m=.5,
                 ambiguity_m=.05, heading_tolerance_rad=.5):
        require(type(max_entries) is int and 1 <= max_entries <= 128
                and all(finite(v) and v > 0 for v in (max_gap_s, projection_margin_m,
                                                     ambiguity_m, heading_tolerance_rad)),
                'PROGRESS_SETTINGS_INVALID')
        self.max_entries, self.max_gap_s = max_entries, max_gap_s
        self.margin, self.ambiguity, self.heading_tolerance = projection_margin_m, ambiguity_m, heading_tolerance_rad
        self.entries = OrderedDict()

    def reset(self):
        self.entries.clear()

    def observe(self, candidate_id, points, pose, speed, frame_id, now_s,
                max_speed_mps, max_acceleration_mps2, position_tolerance_m=None):
        require(isinstance(candidate_id, str) and bool(candidate_id)
                and type(frame_id) is int and frame_id >= 0 and finite(now_s)
                and len(pose) == 3 and all(finite(v) for v in pose)
                and all(finite(v) and v >= 0 for v in (speed, max_speed_mps, max_acceleration_mps2)),
                'PROGRESS_OBSERVATION_INVALID')
        require(isinstance(points, (list, tuple)) and 2 <= len(points) <= 2000,
                'PROGRESS_PATH_REQUIRED')
        margin = self.margin if position_tolerance_m is None else position_tolerance_m
        require(finite(margin) and margin > 0, 'PROGRESS_POSITION_BOUND_REQUIRED')
        previous = self.entries.get(candidate_id)
        if previous is None:
            # A candidate's first published point must be the measured pose.
            a = points[0]
            require(math.hypot(a.x - pose[0], a.y - pose[1]) <= margin
                    and abs(normalize_angle(a.heading - pose[2])) <= self.heading_tolerance,
                    'PROGRESS_INITIAL_ANCHOR_MISMATCH')
            shape = tuple((p.x, p.y, p.heading) for p in points)
            stations = [0.]
            for a, b in zip(shape, shape[1:]):
                require(all(finite(v) for v in a + b), 'PROGRESS_PATH_INVALID')
                stations.append(stations[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
            require(stations[-1] > 0, 'PROGRESS_STATIONARY_PATH')
            value = (shape, tuple(stations), 0., tuple(pose), speed, frame_id, now_s)
            self.entries[candidate_id] = value
            self.entries.move_to_end(candidate_id)
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
            return dict(progress=0., station_m=0., remaining_m=stations[-1], distinct=True,
                        reason_code='ACTUAL_START_ANCHORED')
        shape, stations, station, last_pose, last_speed, last_frame, last_time = previous
        require(frame_id >= last_frame and now_s >= last_time, 'PROGRESS_OUT_OF_ORDER')
        if frame_id == last_frame:
            require(tuple(pose) == last_pose, 'PROGRESS_REPEATED_FRAME_CHANGED_POSE')
            return dict(progress=station / stations[-1], station_m=station,
                        remaining_m=stations[-1] - station, distinct=False, reason_code='PROGRESS_DUPLICATE')
        dt = now_s - last_time
        require(0 < dt <= self.max_gap_s, 'PROGRESS_OBSERVATION_GAP_REQUIRES_REVALIDATION')
        travel = min(max_speed_mps * dt, last_speed * dt + .5 * max_acceleration_mps2 * dt * dt)
        require(math.hypot(pose[0] - last_pose[0], pose[1] - last_pose[1]) <= travel + 2 * margin,
                'PROGRESS_ACTUAL_POSE_JUMP')
        low, high = max(0., station - margin), min(stations[-1], station + travel + 2 * margin)
        options = []
        for i, (a, b) in enumerate(zip(shape, shape[1:])):
            if stations[i] > high or stations[i + 1] < low:
                continue
            dx, dy = b[0] - a[0], b[1] - a[1]
            length2 = dx * dx + dy * dy
            if length2 <= 1e-18:
                continue
            u = max(0., min(1., ((pose[0] - a[0]) * dx + (pose[1] - a[1]) * dy) / length2))
            progress = stations[i] + u * math.sqrt(length2)
            yaw = normalize_angle(a[2] + u * normalize_angle(b[2] - a[2]))
            distance = math.hypot(pose[0] - a[0] - u * dx, pose[1] - a[1] - u * dy)
            if low <= progress <= high and distance <= margin and abs(normalize_angle(pose[2] - yaw)) <= self.heading_tolerance:
                options.append((distance, progress))
        require(options, 'PROGRESS_ACTUAL_POSE_OUTSIDE_REACHABLE_PREFIX')
        options.sort()
        best = options[0]
        require(not any(abs(other[0] - best[0]) <= self.ambiguity and abs(other[1] - best[1]) > 2 * margin
                        for other in options[1:]), 'PROGRESS_SELF_INTERSECTION_AMBIGUOUS')
        observed = max(station, best[1])
        self.entries[candidate_id] = (shape, stations, observed, tuple(pose), speed, frame_id, now_s)
        self.entries.move_to_end(candidate_id)
        return dict(progress=observed / stations[-1], station_m=observed,
                    remaining_m=stations[-1] - observed, distinct=True, reason_code='ACTUAL_REACHABLE_PROGRESS')
