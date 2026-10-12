"""Spatial path sampling in travel order (metres, speed magnitudes in m/s)."""

import math
from bisect import bisect_left

from core.geometry import normalize_angle


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class PreparedPath(object):
    def __init__(self, points, max_segment_m, curve_window_m=3.0):
        self.points = []
        self.arc = []
        self.headings = []
        for point in points:
            x, y, speed = point.x, point.y, point.speed
            if not all(_finite(value) for value in (x, y, speed, point.heading)) or speed < 0:
                raise ValueError("invalid path point")
            if self.points:
                previous = self.points[-1]
                distance = math.hypot(x - previous[0], y - previous[1])
                if distance <= 1e-6:
                    # A repeated position may carry a lower speed limit.
                    self.points[-1] = (previous[0], previous[1],
                                       min(previous[2], float(speed)))
                    self.headings[-1] = float(point.heading)
                    continue
                if distance > max_segment_m:
                    raise ValueError("path segment gap exceeds bound")
                self.arc.append(self.arc[-1] + distance)
            else:
                self.arc.append(0.0)
            self.points.append((float(x), float(y), float(speed)))
            self.headings.append(float(point.heading))
        if len(self.points) < 2 or self.arc[-1] <= 1e-6:
            raise ValueError("path has no forward geometry")
        self.curvatures = [self.curvature_at(s, curve_window_m)
                           for s in self.arc]
        # Only the initial nondecreasing speed prefix may be previewed for
        # propulsion. An internal stop or an earlier slowdown ends this
        # prefix permanently, even if the speed rises again farther ahead.
        self.launch_prefix_end = 0.0
        previous_speed = self.points[0][2]
        for point_s, point in zip(self.arc[1:], self.points[1:]):
            if point[2] <= 1e-6 or point[2] < previous_speed:
                break
            self.launch_prefix_end = point_s
            previous_speed = point[2]

    @property
    def length(self):
        return self.arc[-1]

    def project(self, x, y, heading, settings, end_tolerance_m=0.0,
                progress_hint=None, max_progress_delta=None):
        if (progress_hint is not None and
                (not _finite(progress_hint) or not _finite(max_progress_delta)
                 or max_progress_delta < 0)):
            raise ValueError("invalid path progress hint")
        candidates = []
        for index in range(len(self.points) - 1):
            ax, ay = self.points[index][:2]
            bx, by = self.points[index + 1][:2]
            dx, dy = bx - ax, by - ay
            segment = self.arc[index + 1] - self.arc[index]
            raw = ((x - ax) * dx + (y - ay) * dy) / (segment * segment)
            ratio = max(0.0, min(1.0, raw))
            px, py = ax + ratio * dx, ay + ratio * dy
            distance = math.hypot(x - px, y - py)
            progress = self.arc[index] + ratio * segment
            if progress_hint is not None:
                if (progress < progress_hint - settings.projection_progress_slack_m or
                        progress > progress_hint + max_progress_delta):
                    continue
            candidates.append((distance, progress, math.atan2(dy, dx), index, raw))
        if not candidates:
            raise ValueError("no projection within reachable path progress")
        candidates.sort(key=lambda item: item[0])
        best = candidates[0]
        if best[0] > settings.max_projection_error_m:
            raise ValueError("vehicle too far from planned path")
        if abs(normalize_angle(best[2] - heading)) > settings.max_heading_error_rad:
            raise ValueError("path direction opposes vehicle")
        if (best[3] == len(self.points) - 2 and best[4] > 1.0
                and best[0] > end_tolerance_m):
            raise ValueError("vehicle beyond path end")
        for other in candidates[1:]:
            if (other[0] <= best[0] + 0.15 and
                    abs(other[1] - best[1]) > 5.0):
                raise ValueError("ambiguous crossing-path projection")
        return best[1], best[0], normalize_angle(best[2] - heading)

    def sample(self, s):
        s = max(0.0, min(self.length, s))
        index = max(0, min(len(self.arc) - 2, bisect_left(self.arc, s) - 1))
        segment = self.arc[index + 1] - self.arc[index]
        ratio = (s - self.arc[index]) / segment
        first, second = self.points[index], self.points[index + 1]
        # The planner encodes constant acceleration on spatial segments:
        # v(s)^2 = v0^2 + 2*a*ds. Linear speed interpolation changes that
        # profile whenever equivalent intermediate points are inserted.
        # hypot avoids overflowing the square of an otherwise finite speed.
        speed = math.hypot(first[2]*math.sqrt(1.0-ratio),
                           second[2]*math.sqrt(ratio))
        speed = max(min(first[2], second[2]), min(max(first[2], second[2]), speed))
        return (first[0]+ratio*(second[0]-first[0]),
                first[1]+ratio*(second[1]-first[1]), speed)

    def heading_at(self, s):
        """Interpolate requested body yaw across +/-pi without a full turn."""
        s = max(0.0, min(self.length, s))
        index = max(0, min(len(self.arc) - 2, bisect_left(self.arc, s) - 1))
        ratio = (s - self.arc[index]) / (self.arc[index + 1] - self.arc[index])
        return normalize_angle(self.headings[index] + ratio *
                               normalize_angle(self.headings[index + 1] - self.headings[index]))

    def curvature_at(self, s, window_m=3.0):
        """Signed path curvature from three equally spaced arc samples."""
        left_s = max(0.0, s - window_m)
        right_s = min(self.length, s + window_m)
        if right_s - left_s <= 1e-6:
            return 0.0
        # Clipping at a rolling path's start/end must keep the three arc
        # samples equally spaced. A middle sample a few centimetres from
        # one endpoint overweights that map segment's tangent and invents
        # a curvature spike when the same road is replanned one frame later.
        middle_s = (left_s + right_s) / 2.0
        ax, ay = self.sample(left_s)[:2]
        bx, by = self.sample(middle_s)[:2]
        cx, cy = self.sample(right_s)[:2]
        ab = math.hypot(bx - ax, by - ay)
        bc = math.hypot(cx - bx, cy - by)
        ac = math.hypot(cx - ax, cy - ay)
        if min(ab, bc, ac) <= 1e-6:
            return 0.0
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        return 2.0 * cross / (ab * bc * ac)

    def curve_speed_limit(self, s, settings):
        """Slow before geometric turns using the same braking envelope as points."""
        limit = settings.max_track_speed_mps
        for turn_s, curvature in zip(self.arc, self.curvatures):
            if turn_s < s - settings.turn_exit_margin_m or abs(curvature) <= 1e-6:
                continue
            # A crawl-speed floor must not override a lateral-acceleration cap.
            turn_speed = math.sqrt(settings.max_lateral_accel_mps2 / abs(curvature))
            distance = max(0.0, turn_s - s)
            limit = min(limit, math.sqrt(turn_speed * turn_speed +
                                         2.0 * settings.preview_decel_mps2 * distance))
        return limit

    def speed_reference(self, s, trajectory, settings):
        """Point speed with future slowdowns and known path end as upper bounds."""
        local = self.sample(s)[2]
        rising_preview_end = s
        # Replanned paths start at measured speed; retained maneuver paths
        # keep their original zero-speed start. Continue bounded preview after
        # the first tiny movement so neither form becomes trapped in HOLD.
        if s < self.launch_prefix_end and trajectory.target_speed > 0:
            preview_s = min(self.launch_prefix_end, s + settings.launch_preview_m)
            preview_speed = self.sample(preview_s)[2]
            launch_limit = math.sqrt(local * local +
                                     2.0 * settings.launch_accel_mps2 *
                                     max(0.0, preview_s - s))
            local = max(local, min(preview_speed, launch_limit))
            rising_preview_end = preview_s
        precision = getattr(trajectory, "precision_stop", False)
        margin = settings.precision_end_margin_m if precision else settings.path_end_margin_m
        max_speed = (settings.max_reverse_speed_mps
                     if getattr(trajectory, "motion_direction", 1) == -1
                     else settings.max_track_speed_mps)
        remaining = max(0.0, self.length - s - margin)
        limit = min(local, max_speed,
                    math.sqrt(2.0 * settings.preview_decel_mps2 * remaining))
        if trajectory.stop_required:
            if not _finite(trajectory.stop_distance) or trajectory.stop_distance < 0:
                raise ValueError("stop distance missing")
            stop_remaining = max(0.0, trajectory.stop_distance - margin)
            limit = min(limit, math.sqrt(2.0 * settings.preview_decel_mps2 * stop_remaining))
            if precision:
                # Creep to a precise endpoint instead of using the general
                # half-metre path-coverage margin. Never raises a point limit.
                limit = min(limit, settings.precision_approach_gain_per_s *
                            min(remaining, stop_remaining))
        if not trajectory.stop_required or trajectory.target_speed > 0:
            limit = min(limit, trajectory.target_speed)
        for index, future_s in enumerate(self.arc):
            if future_s <= s:
                continue
            # A rising waypoint inside the accepted launch preview is not a
            # slowdown. Previewing it as a brake boundary cancels propulsion
            # whenever a replan retains a map vertex just centimetres ahead.
            if future_s < rising_preview_end:
                continue
            future_speed = self.points[index][2]
            distance = future_s - s
            limit = min(limit, math.sqrt(future_speed * future_speed +
                                         2.0 * settings.preview_decel_mps2 * distance))
        return max(0.0, limit), remaining

    def acceleration_reference(self, s, trajectory, settings, reference):
        """Spatial v^2 slope, bounded by the profile's acceleration limits.

        No SDK clock or trajectory time derivative is assumed. A measured-speed
        start point uses its first segment; a future constraint is previewed
        with a corresponding reduction in GPS-relative stopping distance.
        """
        distance = min(settings.acceleration_preview_m, self.length - s)
        if distance <= 1e-6 or reference <= settings.hold_speed_mps:
            return 0.0
        if s <= 1e-6:
            initial = min(self.points[0][2], reference)
            future = min(self.sample(s + distance)[2], reference)
        else:
            initial = reference
            # The stop distance is relative to the current ego position, so
            # subtract the virtual preview travel without mutating input.
            future, _ = self.speed_reference(s + distance, trajectory, settings)
            if trajectory.stop_required:
                margin = (settings.precision_end_margin_m
                          if getattr(trajectory, "precision_stop", False)
                          else settings.path_end_margin_m)
                stop = max(0.0, trajectory.stop_distance - distance - margin)
                future = min(future, math.sqrt(2.0 * settings.preview_decel_mps2 * stop))
        future = min(future, self.curve_speed_limit(s + distance, settings))
        acceleration = (future * future - initial * initial) / (2.0 * distance)
        return max(-settings.preview_decel_mps2,
                   min(settings.launch_accel_mps2, acceleration))
