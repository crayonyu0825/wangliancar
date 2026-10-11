"""Offline forward lane planner. Python 3.6; no SDK or third-party packages.

Sensor targets constrain the path only when vehicle dimensions are supplied.
Limits remain provisional planning settings, not measured vehicle capability.
See BASELINE.md for the output contract and integration gates.
"""

import math
import os

from core.geometry import (normalize_angle, project_polyline, opposes_direction,
                           DEFAULT_FORWARD_HEADING_ERROR_RAD)
from core.interfaces import DecisionMode, DecisionTarget, Perception, Trajectory, TrajectoryPoint
from core.validation import current, number, validate_output
from core.scene_requirements import requires_targets
from core.traffic_quality import signal_stop_requirement, signal_stop_bound
from members.planning.obstacle_guard import obstacle_stop
from members.planning.speed_constraints import upcoming_limits, speed_caps
from members.planning.reference_recovery import (smooth_sparse, lateral_recovery,
    check_path, capability, recovery_minimum, curvature)


EPS = 1e-6


class PlannerSettings(object):
    def __init__(self, spacing=1.0, horizon=60.0, acceleration=1.0,
                 deceleration=2.0, lateral_acceleration=1.5,
                 max_lateral_error=0.5, max_heading_error=DEFAULT_FORWARD_HEADING_ERROR_RAD,
                 max_corner_angle=0.35, hold_time=0.1,
                 front_offset_m=None, half_width_m=None,
                 obstacle_margin_m=4.0, lateral_margin_m=0.0,
                 motion_tolerance_mps=0.0, lateral_guard_time_s=3.0,
                 curve_recovery_time_s=0.2, rear_offset_m=None, wheelbase_m=None,
                 front_steer_max_rad=None, max_lateral_acceleration_mps2=None,
                 approach_deceleration_ratio=0.5, traffic_stop_margin=0.3,
                 projection_tolerance_m=2.5, route_ambiguity_m=2.0,
                 behavior_lane_length_min_m=22.0, behavior_transition_time_s=5.0,
                 behavior_validation_budget_ms=40.0, behavior_point_spacing_m=0.5,
                 behavior_steer_rate_rad_s=0.8):
        self.spacing = spacing
        self.horizon = horizon
        self.acceleration = acceleration
        self.deceleration = deceleration
        self.lateral_acceleration = lateral_acceleration
        self.max_lateral_error = max_lateral_error
        self.max_heading_error = max_heading_error
        self.max_corner_angle = max_corner_angle
        self.hold_time = hold_time
        self.front_offset_m = front_offset_m
        self.half_width_m = half_width_m
        self.obstacle_margin_m = obstacle_margin_m
        self.lateral_margin_m = lateral_margin_m
        self.motion_tolerance_mps = motion_tolerance_mps
        self.lateral_guard_time_s = lateral_guard_time_s
        self.curve_recovery_time_s = curve_recovery_time_s
        self.rear_offset_m = rear_offset_m
        self.wheelbase_m = wheelbase_m
        self.front_steer_max_rad = front_steer_max_rad
        self.max_lateral_acceleration_mps2 = max_lateral_acceleration_mps2
        self.approach_deceleration_ratio = approach_deceleration_ratio
        self.traffic_stop_margin = traffic_stop_margin
        self.projection_tolerance_m = projection_tolerance_m
        self.route_ambiguity_m = route_ambiguity_m
        self.behavior_lane_length_min_m = behavior_lane_length_min_m
        self.behavior_transition_time_s = behavior_transition_time_s
        self.behavior_validation_budget_ms = behavior_validation_budget_ms
        self.behavior_point_spacing_m = behavior_point_spacing_m
        self.behavior_steer_rate_rad_s = behavior_steer_rate_rad_s

    @classmethod
    def from_environment(cls, environ=None):
        environ = os.environ if environ is None else environ
        values = {}
        for name, field in (("NEVC_VEHICLE_FRONT_OFFSET_M", "front_offset_m"),
                            ("NEVC_VEHICLE_HALF_WIDTH_M", "half_width_m"),
                            ("NEVC_VEHICLE_REAR_OFFSET_M", "rear_offset_m"),
                            ("NEVC_VEHICLE_WHEELBASE_M", "wheelbase_m"),
                            ("NEVC_VEHICLE_FRONT_STEER_MAX_RAD", "front_steer_max_rad"),
                            ("NEVC_VEHICLE_MAX_LATERAL_ACCELERATION_MPS2", "max_lateral_acceleration_mps2"),
                            ("NEVC_PLANNING_HORIZON_M", "horizon"),
                            ("NEVC_PLANNING_APPROACH_DECELERATION_RATIO", "approach_deceleration_ratio"),
                            ("NEVC_DECISION_TRAFFIC_STOP_MARGIN", "traffic_stop_margin"),
                            ("NEVC_DECISION_PROJECTION_TOLERANCE_M", "projection_tolerance_m"),
                            ("NEVC_DECISION_ROUTE_AMBIGUITY_M", "route_ambiguity_m"),
                            ("NEVC_PLANNING_OBSTACLE_MARGIN_M", "obstacle_margin_m"),
                            ("NEVC_PLANNING_LATERAL_MARGIN_M", "lateral_margin_m"),
                            ("NEVC_PLANNING_MOTION_TOLERANCE_MPS", "motion_tolerance_mps"),
                            ("NEVC_PLANNING_LATERAL_GUARD_TIME_S", "lateral_guard_time_s"),
                            ("NEVC_PLANNING_DECELERATION_MPS2", "deceleration"),
                            ("NEVC_PLANNING_BEHAVIOR_LANE_LENGTH_MIN_M", "behavior_lane_length_min_m"),
                            ("NEVC_PLANNING_BEHAVIOR_TRANSITION_TIME_S", "behavior_transition_time_s"),
                            ("NEVC_PLANNING_BEHAVIOR_VALIDATION_BUDGET_MS", "behavior_validation_budget_ms"),
                            ("NEVC_PLANNING_BEHAVIOR_POINT_SPACING_M", "behavior_point_spacing_m"),
                            ("NEVC_PLANNING_BEHAVIOR_STEER_RATE_RAD_S", "behavior_steer_rate_rad_s")):
            if name in environ:
                try:
                    values[field] = float(environ[name])
                except ValueError:
                    raise ValueError("invalid planning override: " + name)
        return cls(**values)

    def validate(self):
        for name, value in vars(self).items():
            if name in ("front_offset_m", "half_width_m", "rear_offset_m", "wheelbase_m",
                        "front_steer_max_rad", "max_lateral_acceleration_mps2") and value is None:
                continue
            if (not number(value) or value < 0.0 or
                    (name not in ("lateral_margin_m", "motion_tolerance_mps", "traffic_stop_margin")
                     and value <= EPS)):
                raise ValueError("planner setting {0} must be finite and positive".format(name))
        if (self.front_offset_m is None) != (self.half_width_m is None):
            raise ValueError("vehicle front offset and half-width must be configured together")
        if self.horizon / self.spacing > 1998:
            raise ValueError("requested sample budget exceeds 2000 points")
        if self.max_heading_error >= math.pi / 2 or self.max_corner_angle >= math.pi / 2:
            raise ValueError("forward-only angle limits must be below pi/2")
        if self.curve_recovery_time_s > 0.5:
            raise ValueError('curve recovery must remain within 0.5 seconds')
        if self.front_steer_max_rad is not None and self.front_steer_max_rad >= math.pi/2:
            raise ValueError('configured steering capability must remain below pi/2')
        if self.approach_deceleration_ratio > 1:
            raise ValueError('approach deceleration ratio must not exceed hard deceleration')


class ReferenceGeometry(tuple):
    def __new__(cls, source, origin, recovered=False, smoothed=False):
        value=tuple.__new__(cls,(source,origin))
        value.recovered,value.smoothed=recovered,smoothed
        return value


def _clean_reference(raw):
    if not isinstance(raw, (list, tuple)) or len(raw) > 20000:
        raise ValueError("reference must be a bounded point sequence")
    points = []
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError("malformed reference point")
        x, y = point[:2]
        if not number(x) or not number(y):
            raise ValueError("nonfinite or nonnumeric reference coordinate")
        if not points or math.hypot(x - points[-1][0], y - points[-1][1]) > EPS:
            points.append((x, y))
    if len(points) < 2:
        raise ValueError("degenerate reference")
    return points


def _straight_reference(reference, distance):
    """Small recentering must not reinterpret sampled road turn as pose error.

    Curved tracking retains its original source geometry and curvature limits;
    this extra blend is for a measured offset on a verified straight section.
    EPS is the existing numerical geometry tolerance, not a scene threshold.
    """
    heading = None
    for (s, a), (_, b) in zip(reference, reference[1:]):
        if s >= distance-EPS:
            break
        segment = math.atan2(b[1]-a[1], b[0]-a[0])
        if heading is None:
            heading = segment
        elif abs(normalize_angle(segment-heading)) > EPS:
            return False
    return heading is not None


def _reference_ahead(lane, ego, settings):
    current_points = _clean_reference(lane.center_line)
    points = current_points
    if lane.forward_reference_valid is True:
        points = _clean_reference(lane.forward_reference)
        if (len(points) < len(current_points) or points[:len(current_points)] != current_points
                or not isinstance(lane.forward_lane_ids, list) or not lane.forward_lane_ids
                or lane.forward_lane_ids[0] != lane.lane_id):
            raise ValueError("forward reference does not preserve current lane geometry")
    # Anchor progress to the reported current lane. A later self-crossing or
    # nearby successor must not move the vehicle to a future route segment.
    projection = project_polyline(current_points, ego.x, ego.y)
    if projection is None:
        raise ValueError('vehicle projection unavailable')
    if projection["index"] == 0 and projection["raw_ratio"] < -EPS:
        raise ValueError("vehicle before known reference; extrapolation unsupported")
    original_projection=projection
    points,current_count,smoothed=smooth_sparse(points,len(current_points),lane,ego,settings)
    if smoothed:
        projection=project_polyline(points[:current_count],ego.x,ego.y)
    source_distances = [0.0]
    for a, b in zip(points, points[1:]):
        source_distances.append(source_distances[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    if not all(number(s) for s in source_distances):
        raise ValueError("reference length overflow")
    geometry = ReferenceGeometry(list(zip(source_distances, points)), projection["s"],smoothed=smoothed)
    if projection["index"] == len(points) - 2 and projection["raw_ratio"] >= 1.0:
        return [], 0.0, geometry
    path = [projection["point"]]
    for point in points[projection["index"] + 1:]:
        if math.hypot(point[0] - path[-1][0], point[1] - path[-1][1]) > EPS:
            path.append(point)
    if len(path) < 2:
        return [], 0.0, geometry
    heading = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
    if abs(normalize_angle(heading - ego.heading)) > settings.max_heading_error:
        raise ValueError("reference direction disagrees with vehicle heading")
    distances = [0.0]
    for a, b in zip(path, path[1:]):
        distances.append(distances[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    if not all(number(s) for s in distances):
        raise ValueError("reference length overflow")
    # An accepted small pose error is still a discontinuity if the trajectory
    # begins on the lane centre rather than at the measured rear axle. Low-speed
    # tracking then tries to remove the whole offset within its short carrot.
    # Use the same geometry-checked blend for ordinary recentering whenever
    # measured capability and sufficient current route are available.
    pose_error = original_projection['distance'] > EPS
    current_remaining = (sum(math.hypot(b[0]-a[0], b[1]-a[1])
                             for a,b in zip(current_points,current_points[1:]))
                         - original_projection['s'])
    blend_available = False
    if capability(settings):
        heading_error = normalize_angle(ego.heading-heading)
        # Boundaries describe only the current lane, not its predecessor or
        # successors. A rear axle can enter a new segment while its tail is
        # still in the preceding one. Do not make an optional comfort blend
        # demand unavailable corridor coverage at that artificial map seam.
        rear_reach = (settings.rear_offset_m*math.cos(heading_error) +
                      settings.half_width_m*abs(math.sin(heading_error)))
        minimum = recovery_minimum(original_projection['distance'], heading_error,
                                   ego.speed, settings)
        # The full body must also fit ahead throughout the shortest candidate;
        # a forward centreline alone does not provide successor boundaries.
        front_reach = math.hypot(settings.front_offset_m, settings.half_width_m)
        blend_available = (original_projection['s'] >= rear_reach+settings.lateral_margin_m+EPS
            and min(distances[-1], current_remaining, settings.horizon) >=
            minimum+front_reach+settings.lateral_margin_m+EPS)
    minor_blend = (pose_error and blend_available and
                   _straight_reference(list(zip(distances, path)),
                                       min(settings.horizon, current_remaining)))
    if original_projection['distance'] > settings.max_lateral_error or minor_blend:
        path,recovery_length=lateral_recovery(list(zip(distances,path)),lane,ego,settings)
        distances=[0.]
        for a,b in zip(path,path[1:]):
            distances.append(distances[-1]+math.hypot(b[0]-a[0],b[1]-a[1]))
        geometry=ReferenceGeometry(list(zip(distances,path)),0.,recovered=True,smoothed=smoothed)
    return list(zip(distances, path)), distances[-1], geometry


def _sample(reference, horizon, spacing, anchors=()):
    # A global distance grid avoids restarting spacing at each map segment.
    # Keep original vertices too: otherwise sampling can cut across a corner.
    positions = [0.0, horizon]
    positions.extend(i * spacing for i in range(1, int(horizon / spacing) + 1)
                     if i * spacing < horizon - EPS)
    positions.extend(s for s, _ in reference if EPS < s < horizon - EPS)
    positions.extend(s for s in anchors if EPS < s < horizon - EPS)
    if len(positions) == 2:
        positions.append(horizon * 0.5)  # Allows accelerate-then-stop on a short path.
    positions.sort()
    unique = []
    for s in positions:
        if not unique or s - unique[-1] > EPS:
            unique.append(s)
    unique[-1] = horizon
    if len(unique) > 2000:
        raise ValueError("reference sample budget exceeds 2000 points")
    samples, index = [], 0
    for s in unique:
        while index + 1 < len(reference) - 1 and reference[index + 1][0] < s:
            index += 1
        sa, a = reference[index]
        sb, b = reference[index + 1]
        ratio = (s - sa) / (sb - sa)
        samples.append((s, (a[0] + ratio * (b[0] - a[0]),
                            a[1] + ratio * (b[1] - a[1]))))
    return samples


def _curve_limits(samples, settings, geometry=None):
    headings = [math.atan2(b[1][1] - a[1][1], b[1][0] - a[1][0])
                for a, b in zip(samples, samples[1:])]
    for i in range(1, len(samples) - 1):
        turn = abs(normalize_angle(headings[i] - headings[i - 1]))
        if turn > settings.max_corner_angle:
            raise ValueError("sharp reference corner; smooth road geometry required")
    # Extra collinear samples and a sliding ego projection change adjacent
    # sample lengths without changing the road. Estimate turn/length only on
    # the full original map geometry, including vertices behind the ego.
    source, origin_s = geometry if geometry is not None else (samples, 0.0)
    source_headings = [math.atan2(b[1][1] - a[1][1], b[1][0] - a[1][0])
                       for a, b in zip(source, source[1:])]
    source_limits = [float("inf")] * len(source)
    for i in range(1, len(source) - 1):
        turn = abs(normalize_angle(source_headings[i] - source_headings[i - 1]))
        if turn > EPS:
            length = (source[i + 1][0] - source[i - 1][0]) * 0.5
            limit = math.sqrt(settings.lateral_acceleration * length / turn)
            # Apply the vertex limit on both sides, not just at a single point.
            for j in (i - 1, i, i + 1):
                source_limits[j] = min(source_limits[j], limit)
    limits, index = [], 0
    for s, _ in samples:
        position = origin_s + s
        while index + 1 < len(source) - 1 and source[index + 1][0] < position - EPS:
            index += 1
        left_s, right_s = source[index][0], source[index + 1][0]
        left, right = source_limits[index], source_limits[index + 1]
        if position <= left_s + EPS:
            limit = left
        elif position >= right_s - EPS:
            limit = right
        elif math.isfinite(left) and math.isfinite(right):
            ratio = (position - left_s) / (right_s - left_s)
            limit = math.sqrt(left ** 2 + ratio * (right ** 2 - left ** 2))
        else:
            # A finite vertex ahead is still present in samples; the backward
            # braking pass propagates its constraint across a straight edge.
            limit = float("inf")
        limits.append(limit)
    # Unwrap headings so crossing +/-pi does not create a 2*pi jump.
    unwrapped = [headings[0]]
    for heading in headings[1:]:
        unwrapped.append(unwrapped[-1] + normalize_angle(heading - unwrapped[-1]))
    unwrapped.append(unwrapped[-1])
    return unwrapped, limits


def _speed_profile(samples, curve_limits, initial, desired, stop, settings,
                   external_limits=None, curve_recovery=False, deceleration=None,
                   geometry_recovery=False, braking_constraint=None):
    caps = []
    external_limits = (external_limits if external_limits is not None
                       else [float("inf")] * len(samples))
    deceleration=settings.deceleration if deceleration is None else deceleration
    if geometry_recovery and math.isfinite(curve_limits[0]):
        recovery_distance=max(0.,(initial**2-curve_limits[0]**2)/(2*deceleration))
    else:
        recovery_distance = max(0.0, initial * settings.curve_recovery_time_s
                                - .5 * deceleration * settings.curve_recovery_time_s**2)
    if curve_recovery and not geometry_recovery and (initial > curve_limits[0] +
                           settings.deceleration * settings.curve_recovery_time_s):
        return None
    for (s, _), curve, external in zip(samples, curve_limits, external_limits):
        # When overspeeding, request bounded braking instead of jumping to target.
        demand = max(desired, math.sqrt(max(0.0, initial ** 2 - 2 * deceleration * s)))
        # Keep the hard stop boundary independent of comfort anticipation.
        # A stop inside the sampled horizon is previewed by the backward pass;
        # a remote known route end must not lower an otherwise steady curve cap.
        stop_cap = math.sqrt(max(0.0, 2 * settings.deceleration * (stop - s)))
        physical_cap=(curve*math.sqrt(settings.max_lateral_acceleration_mps2/settings.lateral_acceleration)
                      if settings.max_lateral_acceleration_mps2 is not None else float('inf'))
        if curve_recovery and s < recovery_distance - EPS and (
                not geometry_recovery or curve >= curve_limits[0]-EPS):
            # The measured speed already violates the provisional curve cap.
            # Brake at the configured rate for a short recovery window; never
            # relax a stop boundary or a published speed-limit constraint.
            curve = max(curve, math.sqrt(max(0.0, initial**2-2*deceleration*s)))
            demand=min(demand, math.sqrt(max(0.0, initial**2-2*deceleration*s)))
        caps.append(min(demand, stop_cap, curve, external,physical_cap))
        if braking_constraint is not None and s <= braking_constraint[0]+EPS:
            # Comfort preview has already become infeasible, but a hard-rate
            # solution still exists. Begin the necessary braking now instead
            # of accelerating up to the last hard envelope before braking.
            required = braking_constraint[1]
            caps[-1] = min(caps[-1], math.sqrt(max(0., initial**2-2*required*s)))
    for i in range(len(caps) - 2, -1, -1):
        ds = samples[i + 1][0] - samples[i][0]
        caps[i] = min(caps[i], math.sqrt(caps[i + 1] ** 2 + 2 * deceleration * ds))
    if initial > caps[0] + EPS:
        return None  # No profile can preserve v(0) under these provisional limits.
    speeds = [initial]
    for i in range(1, len(caps)):
        ds = samples[i][0] - samples[i - 1][0]
        speeds.append(min(caps[i], math.sqrt(speeds[-1] ** 2 + 2 * settings.acceleration * ds)))
    return speeds


def _required_braking(samples, curves, limits, stop, initial):
    constraints = [(s, cap) for (s, _), cap in zip(samples, curves)
                   if s > EPS and math.isfinite(cap)]
    constraints.extend((s, cap) for s, cap in limits if s > EPS)
    if stop > EPS:
        constraints.append((stop, 0.))
    required, distance = 0., 0.
    for s, cap in constraints:
        deceleration = (initial**2-cap**2)/(2*s)
        if deceleration > required:
            required, distance = deceleration, s
    return (distance, required) if required > EPS else None


def _steering_stop_distance(geometry, distance, settings):
    """Bound forward travel before curvature the configured vehicle cannot turn.

    Speed reduction cannot solve a steering-radius violation. Inspect the same
    source geometry used by curve limits, including support behind the ego,
    but only within the reachable profile. No predecessor boundary is assumed.
    """
    if settings.wheelbase_m is None or settings.front_steer_max_rad is None:
        return None  # Legacy inputs have no measured steering capability.
    source, origin = geometry
    points = [point for _, point in source]
    maximum = math.tan(settings.front_steer_max_rad)/settings.wheelbase_m
    for i in range(1, len(source)-1):
        if source[i+1][0] < origin-EPS:
            continue
        if source[i-1][0] > origin+distance+EPS:
            break
        if abs(curvature(points, i)) <= maximum+EPS:
            continue
        if settings.front_offset_m is None or settings.half_width_m is None:
            raise ValueError('steering-infeasible reference requires body extent for a safe stop')
        front_reach = math.hypot(settings.front_offset_m, settings.half_width_m)
        return max(0., source[i-1][0]-origin-front_reach-settings.lateral_margin_m)
    return None


def _hold(output, ego, emergency, reason, settings):
    # This is the repository's stop-request encoding, not a physical braking path.
    output.emergency_stop = emergency
    output.stop_required = True
    output.target_speed = 0.0
    output.points = [TrajectoryPoint(ego.x, ego.y, 0.0, ego.heading, 0.0),
                     TrajectoryPoint(ego.x, ego.y, 0.0, ego.heading, settings.hold_time)]
    output.valid = True
    output.reason = reason
    return output


def build_trajectory(perception, decision, settings=None):
    output = Trajectory()
    try:
        settings = settings if settings is not None else PlannerSettings.from_environment()
        settings.validate()
        if not isinstance(perception, Perception) or not isinstance(decision, DecisionTarget):
            raise ValueError("expected Perception and DecisionTarget")
        output.bind(decision)
        validate_output(decision, DecisionTarget, perception)
        ego = perception.ego
        if (not current(perception) or ego.valid is not True
                or not all(number(v) for v in (ego.x, ego.y, ego.heading, ego.speed, ego.vx, ego.vy))
                or ego.speed < 0 or type(ego.gear) is not int):
            raise ValueError("invalid or expired ego state")
        if decision.stop_distance < 0 and decision.stop_distance != -1:
            raise ValueError("unknown stop distance must be -1")
        output.target_lane_id = decision.target_lane_id or perception.lane.lane_id
        output.stop_distance = decision.stop_distance
        output.precision_stop = decision.precision_stop and decision.stop_distance >= 0
        output.stop_required = decision.mode in (DecisionMode.STOP, DecisionMode.EMERGENCY_BRAKE)
        if decision.mode == DecisionMode.EMERGENCY_BRAKE:
            _hold(output, ego, True, "emergency stop requested; controller must brake", settings)
        else:
            _moving_reference(output, perception, decision, settings)
        validate_output(output, Trajectory, decision)
        if not current(perception):
            raise ValueError("perception expired during planning")
    except (AttributeError, TypeError, ValueError, OverflowError, IndexError) as exc:
        output.valid = False
        output.points = []
        output.reason = str(exc)
        output.errors.append(str(exc))
    return output


def _moving_reference(output, perception, decision, settings):
    ego, lane = perception.ego, perception.lane
    # GPS gear is a raw gearbox position, not the driver's command enum.
    # Determine actual reverse motion from the signed longitudinal velocity.
    if (number(ego.vx) and number(ego.vy) and
            opposes_direction(ego.vx, ego.vy, ego.heading, ego.speed)):
        raise ValueError("reverse trajectory unsupported")
    if decision.mode == DecisionMode.STOP and decision.stop_distance < 0:
        _hold(output, ego, ego.speed > EPS,
              "stop point unknown; hold or brake without inventing a path", settings)
        return
    if decision.target_lane_id and decision.target_lane_id != lane.lane_id:
        raise ValueError("target lane geometry unavailable; lane change unsupported")
    if ego.speed <= EPS and (decision.mode == DecisionMode.STOP or decision.target_speed == 0):
        _hold(output, ego, False, "stationary hold", settings)
        return
    if lane.valid is not True:
        raise ValueError("map lane unavailable")
    signal_stop = signal_stop_requirement(perception)
    signal_limit = None
    if signal_stop is not None:
        if settings.front_offset_m is None:
            raise ValueError('vehicle front required for signal stopping')
        signal_limit = signal_stop_bound(perception, settings.front_offset_m,
                                        settings.traffic_stop_margin)
        if signal_limit is None:
            raise ValueError('applicable signal unavailable without a verified stop-line boundary')
        if (signal_stop.get('state_known') is not True
                and (decision.stop_distance < 0 or decision.stop_distance > signal_limit+1e-6)):
            raise ValueError('unresolved signal requires a bounded decision stop request')
    # Keep source and geometry gates before applying any motion envelope.
    status = perception.source_status.get("targets", {})
    optional_unavailable = (not requires_targets(perception.scene_id)
                            and (perception.targets_valid is not True
                                 or perception.target_source == "ground_truth"
                                 or (isinstance(status, dict)
                                     and status.get("sensor_presence") == "not_configured"
                                     and not perception.target_source.startswith("sensor:"))))
    if perception.targets_valid is not True and not optional_unavailable:
        raise ValueError("target observations unavailable; clear-road baseline requires valid observations")
    if not isinstance(perception.targets, list):
        raise ValueError("target observations must be a list")
    reference, remaining, geometry = _reference_ahead(lane, ego, settings)
    if remaining <= EPS:
        output.stop_distance = 0.0
        _hold(output, ego, ego.speed > EPS, "end of known reference", settings)
        return
    clearance = (decision.obstacle_clearance_m
                 if decision.obstacle_clearance_m >= 0 else None)
    motion_bound = decision.stop_distance
    if signal_limit is not None:
        motion_bound = min(motion_bound, signal_limit) if motion_bound >= 0 else signal_limit
    reachable = min(remaining, settings.horizon)
    if motion_bound >= 0:
        reachable = min(reachable, motion_bound)
    steering_limit = _steering_stop_distance(geometry, reachable, settings)
    if steering_limit is not None:
        motion_bound = min(motion_bound, steering_limit) if motion_bound >= 0 else steering_limit
    obstacle_limit = (obstacle_stop(perception, reference, settings, clearance,
                                   decision.obstacle_clearances_m,
                                   motion_stop_distance=motion_bound)
                      if perception.targets and not optional_unavailable else None)
    stop = remaining
    if signal_limit is not None:
        stop = min(stop, signal_limit)
    if steering_limit is not None:
        stop = min(stop, steering_limit)
    stopping = decision.mode == DecisionMode.STOP or decision.target_speed == 0
    if obstacle_limit is not None:
        stop = min(stop, obstacle_limit)
    if decision.stop_distance >= 0:
        stop = min(stop, decision.stop_distance)
    if decision.mode != DecisionMode.STOP and decision.target_speed == 0:
        stop = min(stop, ego.speed ** 2 / (2 * settings.deceleration))
    output.stop_distance = stop
    output.stop_required = (stopping or stop <= settings.horizon
                            or decision.stop_distance >= 0 or obstacle_limit is not None
                            or signal_limit is not None)
    if stop <= EPS:
        reason = ('steering curvature exceeds configured capability; immediate stop requested'
                  if steering_limit is not None else 'immediate stop requested')
        _hold(output, ego, ego.speed > EPS, reason, settings)
        return
    limits = upcoming_limits(perception, settings)
    horizon = min(settings.horizon, stop)
    recovery_distance = max(0.0, ego.speed * settings.curve_recovery_time_s
                            - .5 * settings.deceleration * settings.curve_recovery_time_s**2)
    samples = _sample(reference, horizon, settings.spacing,
                      anchors=[distance for distance, _ in limits])
    headings, curve_limits = _curve_limits(samples, settings, geometry)
    desired = ego.speed if decision.mode == DecisionMode.STOP else decision.target_speed
    comfort_deceleration=settings.deceleration*settings.approach_deceleration_ratio
    speeds = _speed_profile(samples, curve_limits, ego.speed, desired, stop, settings,
                            speed_caps(samples, limits, comfort_deceleration),
                            deceleration=comfort_deceleration)
    hard_approach=False
    if speeds is None:
        speeds=_speed_profile(samples,curve_limits,ego.speed,desired,stop,settings,
                              speed_caps(samples,limits,settings.deceleration),
                              braking_constraint=_required_braking(samples,curve_limits,
                                                                   limits,stop,ego.speed))
        hard_approach=speeds is not None
    recovering_curve = False
    geometry_recovery = False
    recovery_failure = ''
    if speeds is None:
        if math.isfinite(curve_limits[0]) and ego.speed>curve_limits[0]+EPS and capability(settings):
            recovery_distance=max(0.,(ego.speed**2-curve_limits[0]**2)/(2*settings.deceleration))
            try:
                check_path([point for _,point in samples],lane,settings,
                           check_until=recovery_distance,initial_heading=ego.heading)
                geometry_recovery=True
            except ValueError as exc:
                recovery_failure='; curve recovery unsupported: '+str(exc)
        samples = _sample(reference, horizon, settings.spacing,
                          anchors=[distance for distance, _ in limits] + [recovery_distance])
        headings, curve_limits = _curve_limits(samples, settings, geometry)
        speeds = (None if recovery_failure else _speed_profile(samples, curve_limits, ego.speed, desired, stop, settings,
                                speed_caps(samples, limits, settings.deceleration),
                                curve_recovery=True,geometry_recovery=geometry_recovery))
        recovering_curve = speeds is not None
    if speeds is None:
        _hold(output, ego, True,
              "braking, curve or speed-limit constraint infeasible; controller must brake"+recovery_failure,
              settings)
        return
    elapsed = 0.0
    if getattr(geometry,'recovered',False):
        headings[0]=ego.heading
    for i, ((s, (x, y)), heading, speed) in enumerate(zip(samples, headings, speeds)):
        if i:
            ds = s - samples[i - 1][0]
            if speeds[i - 1] + speed <= EPS:
                raise ValueError("zero speed cannot traverse a nonzero distance")
            elapsed += 2.0 * ds / (speeds[i - 1] + speed)
        output.points.append(TrajectoryPoint(x, y, speed, heading, elapsed))
    output.target_speed = 0.0 if stopping else decision.target_speed
    output.valid = True
    if obstacle_limit is not None:
        output.reason = "lane reference with conservative obstacle stop; moving target prediction unverified"
    elif perception.targets and not optional_unavailable:
        output.reason = "lane reference checked against current sensor target envelopes"
    elif optional_unavailable:
        output.reason = "lane reference with optional target observations unavailable; collision clearance unverified"
    else:
        output.reason = "clear-road lane reference; vehicle footprint and tracking not validated"
    if lane.forward_reference_valid is True:
        output.reason += "; forward reference: " + lane.forward_reference_status
    if steering_limit is not None:
        output.reason += '; stop before curvature exceeds configured steering capability'
    if limits:
        output.reason += "; upcoming speed limits applied: {0}".format(len(limits))
    if hard_approach:
        output.reason += '; comfort approach infeasible; required braking begins within configured hard limit'
    if getattr(geometry,'recovered',False):
        output.reason += '; vehicle/corridor-checked lateral recovery from measured pose'
    if getattr(geometry,'smoothed',False):
        output.reason += '; coherent sparse curve reconstructed and footprint/steering checked'
    if recovering_curve:
        if geometry_recovery:
            output.reason += '; minimum-risk curve-speed recovery: measured comfort-cap overspeed, monotonic configured braking'
            if settings.max_lateral_acceleration_mps2 is None:
                output.reason += '; physical lateral capability unavailable, grip assumption unvalidated'
        else:
            output.reason += '; bounded curve-speed recovery within {0:.2f}s; legacy geometry capability unavailable'.format(
                settings.curve_recovery_time_s)
