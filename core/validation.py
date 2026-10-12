"""Validate contracts at each member boundary before anything can be sent."""

import math
import time

from core.interfaces import DecisionMode, DecisionTarget, Trajectory, ControlOut


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def current(value, source=None):
    if (type(value.valid) is not bool or not value.valid
            or type(value.frame_id) is not int or value.frame_id < 0
            or type(value.timestamp) is not int or value.timestamp < 0
            or not number(value.valid_until) or time.monotonic() >= value.valid_until):
        return False
    return source is None or (source.valid and value.frame_id == source.frame_id
                              and value.timestamp == source.timestamp
                              and value.valid_until <= source.valid_until)


def validate_output(value, expected, source):
    if not isinstance(value, expected):
        raise ValueError("wrong output type")
    if not current(value, source):
        raise ValueError("invalid, expired or mismatched source frame")
    if expected is DecisionTarget:
        if value.behavior_request is not None:
            from core.behavior_protocol import behavior_frame, validate_runtime_request
            now=time.monotonic()
            validate_runtime_request(value.behavior_request,behavior_frame(source,now))
        if (value.mode not in (DecisionMode.KEEP_LANE, DecisionMode.FOLLOW,
                               DecisionMode.STOP, DecisionMode.EMERGENCY_BRAKE)
                or not number(value.target_speed) or value.target_speed < 0
                or not number(value.stop_distance) or value.stop_distance < -1
                or not number(value.obstacle_clearance_m)
                or (value.obstacle_clearance_m < 0 and value.obstacle_clearance_m != -1)
                or type(value.precision_stop) is not bool
                or (value.precision_stop and value.stop_distance < 0)
                or not isinstance(value.target_lane_id, str)):
            raise ValueError("invalid decision fields")
        if (not isinstance(value.obstacle_clearances_m, dict)
                or any(not isinstance(key, str) or not key.isdigit()
                       or key != str(int(key)) or not number(gap) or gap < 0
                       for key, gap in value.obstacle_clearances_m.items())):
            raise ValueError("invalid per-target clearance")
    elif expected is Trajectory:
        from core.behavior_protocol import RuntimeRequest, PlanningFeedback
        if value.behavior_request is not None and not isinstance(value.behavior_request,RuntimeRequest):
            raise ValueError("invalid trajectory behavior request")
        if (not isinstance(value.planning_feedback,list) or any(not isinstance(v,PlanningFeedback)
                for v in value.planning_feedback)):
            raise ValueError("invalid planning feedback")
        if (not number(value.target_speed) or value.target_speed < 0
                or len(value.points) < 2 or type(value.emergency_stop) is not bool
                or type(value.emergency_maneuver_authorized) is not bool
                or type(value.stop_required) is not bool
                or not number(value.stop_distance) or value.stop_distance < -1):
            raise ValueError("invalid trajectory fields")
        direction = getattr(value, "motion_direction", 1)
        if type(direction) is not int or direction not in (-1, 1):
            raise ValueError("trajectory direction must be +1 or -1")
        for name in ("precision_stop", "parking_brake_at_stop",
                     "left_signal", "right_signal", "hazard_signal"):
            if type(getattr(value, name, False)) is not bool:
                raise ValueError("trajectory intent fields must be bool")
        dwell = getattr(value, "hold_duration_s", 0.0)
        if not number(dwell) or dwell < 0:
            raise ValueError("invalid trajectory hold duration")
        if ((dwell > 0 or getattr(value, "parking_brake_at_stop", False)
             or getattr(value, "precision_stop", False))
                and (not value.stop_required or value.stop_distance < 0)):
            raise ValueError("parking hold requires an explicit stop")
        if (getattr(value, "left_signal", False) and
                getattr(value, "right_signal", False) and
                not getattr(value, "hazard_signal", False)):
            raise ValueError("opposing turn signals without hazard intent")
        previous = None
        for point in value.points:
            if (not all(number(getattr(point, key)) for key in
                        ("x", "y", "speed", "heading", "relative_time"))
                    or point.speed < 0 or point.relative_time < 0):
                raise ValueError("invalid trajectory point")
            if previous is not None:
                dt = point.relative_time - previous.relative_time
                ds = math.hypot(point.x - previous.x, point.y - previous.y)
                if dt <= 0 or (ds > 1e-6 and point.speed + previous.speed <= 0):
                    raise ValueError("invalid trajectory timing")
            previous = point
        authorized=value.emergency_maneuver_authorized
        if authorized:
            request=value.behavior_request
            if (not isinstance(request,RuntimeRequest) or request.maneuver!='EMERGENCY'
                    or request.authorization!='EXECUTE_SELECTED' or not value.emergency_stop
                    or value.execution_path_id!=request.candidate_reference.candidate_id
                    or value.target_speed!=0 or value.points[-1].speed!=0
                    or any(b.speed>a.speed+1e-8 for a,b in zip(value.points,value.points[1:]))):
                raise ValueError('invalid turn/brake authority or braking profile')
        if value.emergency_stop and (value.target_speed != 0 or not authorized and any(p.speed != 0 for p in value.points)):
            raise ValueError("emergency trajectory contains propulsion")
    elif expected is ControlOut:
        value.clamp()
        if getattr(source, "emergency_stop", False) and (value.throttle != 0 or value.brake <= 0):
            raise ValueError("controller ignored emergency stop")
        direction = getattr(source, "motion_direction", 1)
        if value.gear == 2 and direction != -1 and any(point.speed > 0 for point in source.points):
            raise ValueError("reverse gear with forward-only trajectory")
        if value.gear == 1 and direction == -1 and value.throttle > 0:
            raise ValueError("drive propulsion with reverse trajectory")
        if value.gear == 3 and any(point.speed > 0 for point in source.points):
            raise ValueError("parking gear with moving trajectory")
    return value
