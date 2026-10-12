"""Independent candidate checks, not a public feedback or driving authority.

Rear-axle XY is interpolated with the segment's constant tangential acceleration;
body yaw follows shortest-angle interpolation in travelled distance. Continuous
clearance certificates cover that nominal model, including between input points.
Steering and lateral acceleration checks are discrete bicycle-model constraints,
not calibrated actuator or tracking guarantees. Dynamic obstacles explicitly use
constant world velocity and fixed yaw, with the supplied uncertainty envelope.
All parameters are caller-supplied; this module neither reads SDK nor guesses
missing geometry, coverage, motion, calibration or allowed road area.
"""

import math
import time

from core.geometry import normalize_angle
from core.validation import number
from members.planning.corridor_region import CorridorRegion,NumericResolution,PreparationLimit


EPS = 1e-9


class VehicleGeometry(object):
    def __init__(self, front_offset_m, rear_offset_m, half_width_m, wheelbase_m):
        self.front_offset_m = front_offset_m
        self.rear_offset_m = rear_offset_m
        self.half_width_m = half_width_m
        self.wheelbase_m = wheelbase_m


class MotionLimits(object):
    def __init__(self, max_speed_mps, max_acceleration_mps2, max_deceleration_mps2,
                 max_lateral_acceleration_mps2, max_front_steer_rad,
                 max_steer_rate_rad_s, heading_tolerance_rad, distance_tolerance_m):
        self.max_speed_mps = max_speed_mps
        self.max_acceleration_mps2 = max_acceleration_mps2
        self.max_deceleration_mps2 = max_deceleration_mps2
        self.max_lateral_acceleration_mps2 = max_lateral_acceleration_mps2
        self.max_front_steer_rad = max_front_steer_rad
        self.max_steer_rate_rad_s = max_steer_rate_rad_s
        self.heading_tolerance_rad = heading_tolerance_rad
        self.distance_tolerance_m = distance_tolerance_m


class ObstaclePrediction(object):
    """XY is body centre; horizon/uncertainty are relative to candidate t=0.

    heading=None uses an orientation-independent bounding circle. vx/vy=None
    is invalid, including for an allegedly static object: zero must be explicit.
    Uncertainty is a position-radius + velocity-radius*t Minkowski expansion.
    """
    def __init__(self, identity, x, y, length, width, heading, vx, vy,
                 valid_until_s, position_uncertainty_m, velocity_uncertainty_mps):
        self.identity, self.x, self.y = identity, x, y
        self.length, self.width, self.heading = length, width, heading
        self.vx, self.vy, self.valid_until_s = vx, vy, valid_until_s
        self.position_uncertainty_m = position_uncertainty_m
        self.velocity_uncertainty_mps = velocity_uncertainty_mps


class ValidationBudget(object):
    def __init__(self, max_checks, max_depth, min_interval_s, deadline_monotonic_s):
        self.max_checks, self.max_depth = max_checks, max_depth
        self.min_interval_s = min_interval_s
        self.deadline_monotonic_s = deadline_monotonic_s


class ExecutionEnvelope(object):
    """Caller-verified tracking bounds; presence is not platform calibration.

    At nominal progress/time t, actual rear-axle position can differ by a disk
    of position_error_m, body yaw by heading_error_rad, and execution time by
    [-early_time_s,+late_time_s]. These are occupancy bounds, not a claim that
    the controller follows absolute timestamps or that actuator errors vanish.
    """
    def __init__(self, position_error_m, heading_error_rad, early_time_s, late_time_s):
        self.position_error_m = position_error_m
        self.heading_error_rad = heading_error_rad
        self.early_time_s = early_time_s
        self.late_time_s = late_time_s


def validate_execution_envelope(envelope):
    if not isinstance(envelope, ExecutionEnvelope):
        _invalid('explicit ExecutionEnvelope required for execution-time claims')
    for name in ('position_error_m', 'heading_error_rad', 'early_time_s', 'late_time_s'):
        _positive(name, getattr(envelope, name), True)
    if envelope.heading_error_rad > math.pi:
        _invalid('execution heading bound must be at most pi')
    return envelope


class _Stop(Exception):
    def __init__(self, status, code, constraint, **details):
        self.status, self.code, self.constraint, self.details = status, code, constraint, details


def _invalid(message):
    raise _Stop("invalid", "INVALID_INPUT", "input", message=message)


def _positive(name, value, allow_zero=False):
    if not number(value) or (value < 0 if allow_zero else value <= 0):
        _invalid(name + " must be finite and " + ("nonnegative" if allow_zero else "positive"))


def _finite(values):
    if not all(number(value) for value in values):
        _invalid("nonnumeric or nonfinite geometry/motion")


class _Checks(object):
    def __init__(self, budget):
        self.budget, self.count = budget, 0

    def consume(self):
        self.count += 1
        if (self.count > self.budget.max_checks
                or time.monotonic() >= self.budget.deadline_monotonic_s):
            raise _Stop("inconclusive", "BUDGET_EXHAUSTED", "budget")


def _cross(a, b):
    return a[0]*b[1] - a[1]*b[0]


def _edges(polygon):
    return zip(polygon, polygon[1:] + polygon[:1])


def _corridor(raw, checks):
    if isinstance(raw,CorridorRegion):
        return raw.prepare(checks.consume,checks.budget.max_checks)
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        _invalid("explicit convex corridor polygon required")
    polygon = []
    for point in raw:
        checks.consume()
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            _invalid("corridor vertex must contain XY only")
        _finite(point)
        if not polygon or tuple(point) != polygon[-1]:
            polygon.append(tuple(point))
    if polygon and polygon[0] == polygon[-1]:
        polygon.pop()
    if len(polygon) < 3:
        _invalid("corridor polygon degenerate")
    origin = polygon[0]
    area = sum(_cross((a[0]-origin[0], a[1]-origin[1]),
                      (b[0]-origin[0], b[1]-origin[1])) for a, b in _edges(polygon))
    _finite((area,))
    if abs(area) <= EPS:
        _invalid("corridor polygon has no area")
    if area < 0:
        polygon.reverse()
    planes = []
    for a, b in _edges(polygon):
        length = math.hypot(b[0]-a[0], b[1]-a[1])
        _positive("corridor edge length", length)
        normal = (-(b[1]-a[1])/length, (b[0]-a[0])/length)
        for p in polygon:
            checks.consume()
            if (p[0]-a[0])*normal[0] + (p[1]-a[1])*normal[1] < -EPS:
                _invalid("corridor must be convex, ordered and non-self-intersecting")
        planes.append((a, normal))
    return planes


def _rectangle(x, y, heading, front, rear, half_width):
    c, s = math.cos(heading), math.sin(heading)
    result = [(x+u*c-v*s, y+u*s+v*c) for u, v in
              ((-rear, -half_width), (front, -half_width),
               (front, half_width), (-rear, half_width))]
    _finite([value for point in result for value in point])
    return result


def _polygon_gap(first, second):
    """SAT detects intersection; vertex/edge distance measures separated boxes."""
    gap = 0.0
    for polygon in (first, second):
        for a, b in _edges(polygon):
            length = math.hypot(b[0]-a[0], b[1]-a[1])
            _positive('distance edge length',length)
            nx, ny = -(b[1]-a[1])/length, (b[0]-a[0])/length
            # Translate before projecting to reduce cancellation in world XY.
            pa = [(p[0]-a[0])*nx + (p[1]-a[1])*ny for p in first]
            pb = [(p[0]-a[0])*nx + (p[1]-a[1])*ny for p in second]
            _finite(pa+pb)
            gap = max(gap, min(pb)-max(pa), min(pa)-max(pb))
    if gap <= 0:
        return 0.
    # A separating-axis gap alone is only a lower bound. Subtracting a disk
    # uncertainty from it could falsely report collision near a box corner.
    distance = float("inf")
    for vertices, boundary in ((first, second), (second, first)):
        for point in vertices:
            for a, b in _edges(boundary):
                dx, dy = b[0]-a[0], b[1]-a[1]
                px, py = point[0]-a[0], point[1]-a[1]
                length=math.hypot(dx,dy)
                _positive('distance edge length',length)
                ux,uy=dx/length,dy/length
                projection=px*ux+py*uy
                _finite((px,py,projection))
                along=max(0.,min(length,projection))
                distance = min(distance, math.hypot(px-along*ux,py-along*uy))
    return distance


def _circle_gap(polygon, x, y, radius):
    inside, distance = True, float("inf")
    for a, b in _edges(polygon):
        dx, dy = b[0]-a[0], b[1]-a[1]
        px, py = x-a[0], y-a[1]
        inside = inside and _cross((dx, dy), (px, py)) >= -EPS
        ratio = max(0., min(1., (px*dx+py*dy)/(dx*dx+dy*dy)))
        distance = min(distance, math.hypot(px-ratio*dx, py-ratio*dy))
    return -radius if inside else distance-radius


def _obstacle_gap(body, obstacle, at, execution=None):
    x, y = obstacle.x+obstacle.vx*at, obstacle.y+obstacle.vy*at
    uncertainty = obstacle.position_uncertainty_m + obstacle.velocity_uncertainty_mps*at
    if execution is not None:
        # Cover every allowed execution timestamp using a Minkowski disk.
        # Target uncertainty is evaluated at the latest allowed instant.
        uncertainty += (math.hypot(obstacle.vx, obstacle.vy)*
                        max(execution.early_time_s, execution.late_time_s)
                        + obstacle.velocity_uncertainty_mps*execution.late_time_s)
    _finite((x, y, uncertainty))
    if obstacle.heading is None:
        return _circle_gap(body, x, y, math.hypot(obstacle.length, obstacle.width)*.5 + uncertainty)
    shape = _rectangle(x, y, obstacle.heading, obstacle.length*.5,
                       obstacle.length*.5, obstacle.width*.5)
    return _polygon_gap(body, shape)-uncertainty


def _motion(points, direction, vehicle, limits, checks):
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        _invalid("at least two explicit TrajectoryPoints required")
    values = []
    for point in points:
        checks.consume()
        try:
            value = (point.x, point.y, point.heading, point.speed, point.relative_time)
        except AttributeError:
            _invalid("expected TrajectoryPoint attributes")
        _finite(value)
        if value[3] < 0 or value[4] < 0:
            _invalid("speed and relative time must be nonnegative")
        if value[3] > limits.max_speed_mps + EPS:
            raise _Stop("unsafe", "MOTION_LIMIT", "speed", speed=value[3])
        values.append(value)
    if values[0][4] != 0:
        _invalid("candidate must start at relative_time=0")
    segments, moving, previous = [], [], None
    for index, (a, b) in enumerate(zip(values, values[1:])):
        checks.consume()
        dt = b[4]-a[4]
        if dt <= 0:
            _invalid("relative time must strictly increase")
        ds = math.hypot(b[0]-a[0], b[1]-a[1])
        turn = normalize_angle(b[2]-a[2])
        distance = .5*(a[3]+b[3])*dt
        _finite((ds, turn, distance))
        if abs(ds-distance) > limits.distance_tolerance_m:
            _invalid("distance/speed/time inconsistent at segment {0}".format(index))
        if ds <= EPS:
            if a[3] > EPS or b[3] > EPS or abs(turn) > EPS:
                _invalid("stationary segment cannot translate or rotate")
            curvature = 0.
            scale = 1.
        else:
            if distance <= 0:
                _invalid("zero speeds cannot traverse distance")
            travel_heading = math.atan2(b[1]-a[1], b[0]-a[0])
            body_heading = a[2]+turn*.5+(math.pi if direction == -1 else 0.)
            if abs(normalize_angle(travel_heading-body_heading)) > limits.heading_tolerance_rad+EPS:
                raise _Stop("unsafe", "MOTION_LIMIT", "body_heading", segment=index)
            curvature = turn/(direction*ds)
            scale = ds/distance
        acceleration = (b[3]-a[3])/dt
        model_acceleration = acceleration*scale
        model_speed = max(a[3], b[3])*scale
        steer = math.atan(vehicle.wheelbase_m*curvature)
        lateral = model_speed**2*abs(curvature)
        _finite((model_acceleration, model_speed, steer, lateral))
        for condition, constraint, observed in (
                (model_speed > limits.max_speed_mps+EPS, "speed", model_speed),
                (model_acceleration > limits.max_acceleration_mps2+EPS, "acceleration", model_acceleration),
                (-model_acceleration > limits.max_deceleration_mps2+EPS, "deceleration", -model_acceleration),
                (abs(steer) > limits.max_front_steer_rad+EPS, "steering", abs(steer)),
                (lateral > limits.max_lateral_acceleration_mps2+EPS, "lateral_acceleration", lateral)):
            if condition:
                raise _Stop("unsafe", "MOTION_LIMIT", constraint, segment=index, observed=observed)
        midpoint_time = .5*(a[4]+b[4])
        if previous is not None:
            rate = abs(steer-previous[0])/(midpoint_time-previous[1])
            if rate > limits.max_steer_rate_rad_s+EPS:
                raise _Stop("unsafe", "MOTION_LIMIT", "steering_rate", segment=index, observed=rate)
        previous = steer, midpoint_time
        segment = (a, b, ds, turn, acceleration)
        segments.append(segment)
        if ds > EPS:
            moving.append((index, segment, model_speed))
    # Heading metadata cannot conceal the geometric turning demand of a path.
    # These circumcircle and steering-rate checks supplement the yaw-per-distance
    # model; both remain discrete constraints rather than a continuous actuator proof.
    # Pair the real moving legs across stationary samples, without removing their
    # time occupancy from segments. Arrival times retain the original rate clock.
    previous = None
    for incoming, outgoing in zip(moving, moving[1:]):
        checks.consume()
        index = incoming[0] + 1
        a, b = incoming[1][:2]
        c, d = outgoing[1][:2]
        before, after = (b[0]-a[0], b[1]-a[1]), (d[0]-c[0], d[1]-c[1])
        ab, bc = incoming[1][2], outgoing[1][2]
        ac = math.hypot(before[0]+after[0], before[1]+after[1])
        if ac <= EPS:
            raise _Stop("unsafe", "MOTION_LIMIT", "path_fold", point=index)
        geometric = 2*_cross(before, after)/(ab*bc*ac)
        steer = math.atan(vehicle.wheelbase_m*geometric/direction)
        speed = max(incoming[2], outgoing[2])
        lateral = speed**2*abs(geometric)
        _finite((geometric, steer, lateral))
        if abs(steer) > limits.max_front_steer_rad+EPS:
            raise _Stop("unsafe", "MOTION_LIMIT", "steering", point=index, observed=abs(steer))
        if lateral > limits.max_lateral_acceleration_mps2+EPS:
            raise _Stop("unsafe", "MOTION_LIMIT", "lateral_acceleration", point=index, observed=lateral)
        if previous is not None:
            rate = abs(steer-previous[0])/(b[4]-previous[1])
            if rate > limits.max_steer_rate_rad_s+EPS:
                raise _Stop("unsafe", "MOTION_LIMIT", "steering_rate", point=index, observed=rate)
        previous = steer, b[4]
    return segments, values[-1][4]


def _pose(segment, at):
    a, b, ds, turn, acceleration = segment
    elapsed, duration = at-a[4], b[4]-a[4]
    denominator = .5*(a[3]+b[3])*duration
    u = ((a[3]*elapsed+.5*acceleration*elapsed*elapsed)/denominator
         if ds > EPS else 0.)
    u = max(0., min(1., u))
    return (a[0]+u*(b[0]-a[0]), a[1]+u*(b[1]-a[1]), a[2]+u*turn), u


def _sweep(segments, vehicle, planes, obstacles, checks, execution=None):
    radius = math.hypot(max(vehicle.front_offset_m, vehicle.rear_offset_m), vehicle.half_width_m)
    error = (execution.position_error_m + 2*radius*math.sin(execution.heading_error_rad*.5)
             if execution is not None else 0.)
    minimum = float("inf")

    def gaps(segment, at):
        checks.consume()
        pose, u = _pose(segment, at)
        body = _rectangle(pose[0], pose[1], pose[2], vehicle.front_offset_m,
                          vehicle.rear_offset_m, vehicle.half_width_m)
        road = (planes.gap(body,pose[:2],checks.consume) if hasattr(planes,'gap') else
                min((p[0]-origin[0])*normal[0]+(p[1]-origin[1])*normal[1]
                    for origin, normal in planes for p in body))
        road -= error
        if road <= EPS:
            raise _Stop("unsafe", "CORRIDOR_COLLISION", "corridor", relative_time_s=at)
        result = [road]
        for obstacle in obstacles:
            checks.consume()
            gap = _obstacle_gap(body, obstacle, at, execution)-error
            if gap <= EPS:
                raise _Stop("unsafe", "OBSTACLE_COLLISION", "collision",
                            obstacle_id=obstacle.identity, relative_time_s=at)
            result.append(gap)
        _finite(result)
        return result, u

    def interval(segment, start, end, depth):
        nonlocal minimum
        midpoint = .5*(start+end)
        distances, um = gaps(segment, midpoint)
        _, ua = _pose(segment, start)
        _, ub = _pose(segment, end)
        # Rotation moves every body point by at most R*abs(delta_yaw).
        # u(midtime) need not be .5: acceleration changes interval occupancy.
        displacement = max(abs(um-ua), abs(ub-um))*(segment[2]+radius*abs(segment[3]))
        lower_bounds = [distances[0]-displacement]
        for obstacle, gap in zip(obstacles, distances[1:]):
            drift = math.hypot(obstacle.vx, obstacle.vy)*(end-start)*.5
            growth = obstacle.velocity_uncertainty_mps*(end-midpoint)
            lower_bounds.append(gap-displacement-drift-growth)
        _finite(lower_bounds)
        if min(lower_bounds) > EPS:
            minimum = min(minimum, min(lower_bounds))
            return
        if depth >= checks.budget.max_depth or end-start <= checks.budget.min_interval_s:
            raise _Stop("inconclusive", "RESOLUTION_LIMIT", "continuous_clearance",
                        start_relative_time_s=start, end_relative_time_s=end)
        interval(segment, start, midpoint, depth+1)
        interval(segment, midpoint, end, depth+1)

    for segment in segments:
        # Endpoints are included explicitly; a contact exactly at t=0/end must
        # not disappear when subdivision reaches its finite resolution bound.
        gaps(segment, segment[0][4])
        gaps(segment, segment[1][4])
        interval(segment, segment[0][4], segment[1][4], 0)
    return minimum


def validate_candidate(points, motion_direction, vehicle, limits, corridor, obstacles, budget,
                       execution_envelope=None):
    """Return a detached model-scoped report; never mutate/rebind a trajectory."""
    started, checks = time.monotonic(), None
    report = {"status": "invalid", "reason_code": "INVALID_INPUT", "constraint": "input",
              "details": {}, "checks": 0, "elapsed_s": 0., "clearance_lower_bound_m": None,
              "execution_robust": False,
              "assumptions": ["explicit convex verified corridor; observed-empty obstacles require caller evidence",
                  "constant tangential acceleration normalized within distance tolerance; scaled limits checked",
                  "shortest yaw by spatial progress; geometric curvature checked independently",
                  "constant-velocity fixed-yaw targets with supplied finite horizon/uncertainty",
                  "discrete bicycle/steering-rate limits; no actuator calibration or time-tracking guarantee",
                  "not public behavior feedback, live source validation, or permission to drive"]}
    try:
        if execution_envelope is not None:
            validate_execution_envelope(execution_envelope)
            report['assumptions'].append('caller-supplied position/yaw/time error envelope; no inferred calibration')
        if isinstance(corridor,CorridorRegion):
            report['assumptions'][0]='explicit simple-polygon union; gaps/holes retained; caller supplies source/coverage authority'
        if not isinstance(budget, ValidationBudget):
            _invalid("explicit ValidationBudget required")
        if (type(budget.max_checks) is not int or budget.max_checks <= 0
                or type(budget.max_depth) is not int or not 0 <= budget.max_depth <= 40):
            _invalid("budget needs positive max_checks and max_depth in [0,40]")
        _positive("min_interval_s", budget.min_interval_s)
        _positive("deadline_monotonic_s", budget.deadline_monotonic_s)
        checks = _Checks(budget)
        checks.consume()
        if type(motion_direction) is not int or motion_direction not in (-1, 1):
            _invalid("one explicit motion direction +1 or -1 required")
        if not isinstance(vehicle, VehicleGeometry) or not isinstance(limits, MotionLimits):
            _invalid("explicit VehicleGeometry and MotionLimits required")
        for name, value in vars(vehicle).items():
            _positive(name, value)
        if vehicle.front_offset_m < vehicle.wheelbase_m:
            _invalid("front offset cannot be shorter than wheelbase")
        for name, value in vars(limits).items():
            _positive(name, value, name in ("heading_tolerance_rad", "distance_tolerance_m"))
        if limits.max_front_steer_rad >= math.pi*.5 or limits.heading_tolerance_rad >= math.pi*.5:
            _invalid("steering and heading tolerance must be below pi/2")
        if not isinstance(obstacles, (list, tuple)):
            _invalid("explicit obstacle list required")
        identities = set()
        for obstacle in obstacles:
            checks.consume()
            if not isinstance(obstacle, ObstaclePrediction):
                _invalid("expected explicit ObstaclePrediction")
            if not isinstance(obstacle.identity, str) or not obstacle.identity or obstacle.identity in identities:
                _invalid("obstacle identities must be nonempty unique strings")
            identities.add(obstacle.identity)
            _finite((obstacle.x, obstacle.y, obstacle.vx, obstacle.vy))
            if obstacle.heading is not None:
                _finite((obstacle.heading,))
            for name in ("length", "width", "valid_until_s", "position_uncertainty_m", "velocity_uncertainty_mps"):
                _positive(name, getattr(obstacle, name), name.endswith("uncertainty_m") or name.endswith("uncertainty_mps"))
        planes = _corridor(corridor, checks)
        segments, horizon = _motion(points, motion_direction, vehicle, limits, checks)
        for obstacle in obstacles:
            required_horizon = horizon + (execution_envelope.late_time_s if execution_envelope else 0.)
            if required_horizon > obstacle.valid_until_s:
                raise _Stop("inconclusive", "PREDICTION_HORIZON", "prediction_horizon",
                            obstacle_id=obstacle.identity, required_horizon_s=required_horizon)
        clearance = _sweep(segments, vehicle, planes, obstacles, checks, execution_envelope)
        checks.consume()  # A final expensive check must not finish past deadline unnoticed.
        report.update(status="safe", reason_code="CERTIFIED", constraint="nominal_model",
                      execution_robust=execution_envelope is not None,
                      clearance_lower_bound_m=clearance,
                      details={"segment_count": len(segments), "validated_horizon_s": horizon})
        if isinstance(corridor,CorridorRegion):
            report['details'].update(region_count=len(planes.polygons),exterior_edge_count=planes.edge_count)
            if hasattr(planes,'barrier_count'):
                report['details']['source_perimeter_barrier_count']=planes.barrier_count
                report['assumptions'].append('complete nominal source perimeter except explicitly source-bound shared openings')
    except PreparationLimit as error:
        report.update(status='inconclusive',reason_code='REGION_PREPARATION_LIMIT',
                      constraint='region_topology',details={'message':str(error)})
    except NumericResolution as error:
        report.update(status='inconclusive',reason_code='REGION_NUMERIC_RESOLUTION',
                      constraint='region_topology',details={'message':str(error)})
    except _Stop as stop:
        report.update(status=stop.status, reason_code=stop.code, constraint=stop.constraint, details=stop.details)
        if (stop.code=='BUDGET_EXHAUSTED' and isinstance(corridor,CorridorRegion)
                and corridor.preparation_max_checks is not None and corridor._prepared is None):
            report['details'].update(preparation_complete=False,preparation_checks=corridor._job_steps)
    except (ValueError, TypeError, AttributeError, OverflowError, ZeroDivisionError) as exc:
        report.update(status="invalid", reason_code="INVALID_INPUT", constraint="input", details={"message": str(exc)})
    report["checks"] = checks.count if checks is not None else 0
    report["elapsed_s"] = time.monotonic()-started
    return report
